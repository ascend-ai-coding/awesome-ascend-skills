# NDDMA 规则

> 多维数据搬运 `DataCopy<T, dim, config>`（GM→UB）：随路 broadcast / 跳读 / 转置 / Padding。

---

## 1 总览：NDDMA 能干什么

### 1.1 定位与能力清单

NDDMA 是 `DataCopy` 的多维模板重载：把最多 5 层嵌套循环（每维 = 元素数 + 源步长 + 目的步长 + 左右补值）压成一条硬件指令，在 **CopyIn（GM→UB）搬运阶段**完成跨步、重复、转置、补值——UB 侧得到的就是目标排布，无需额外重排。

| 能力 | 配置手段 | 典型场景 |
|------|---------|---------|
| 多维整块搬运 | 每维 loopSize + stride | ≤5 维 tensor 单指令搬入；子块截取 |
| 跨步跳读 | 某维 `loopSrcStride > 1` | Slice 尾轴 stride > 1 |
| 随路 broadcast | 某维 `loopSrcStride = 0`（源原地重读） | inShape=1 → outShape=N；广播在搬运阶段完成 |
| 转置 | src / dst stride 交叉 | 尾轴转置 |
| 左右补值 | `loopLpSize`/`loopRpSize` + 常数/最近值填充 | Pad 类算子随路填充 |

### 1.2 能力边界（负面清单）

- 数据通路**仅 GM → VECIN（UB）**，不支持 UB→GM 方向
- `dim ∈ [1, 5]`：一条指令最多描述 5 个维度，硬件固定，不随切分变化；超出 5 维拆 for 循环
- **全部参数单位 = 元素个数**（不是字节），禁止 `× sizeof(T)`
- 不支持负 stride（`loopSrcStride ∈ [0, 2^40)`，0 = 原地重读）

---

## 2 接口与参数结构

### 2.1 函数原型

```cpp
// NDDMA Cache 刷新，DataCopy 前调用
__aicore__ inline void NdDmaDci();

// 头文件：basic_api/kernel_operator_data_copy_intf.h
template <typename T, uint8_t dim, const NdDmaConfig& config = kDefaultNdDmaConfig>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst,
                                const GlobalTensor<T>& src,
                                const NdDmaParams<T, dim>& params);
```

- 模板参数：`T` 数据类型；`dim ∈ [1, 5]`；`config` 搬运配置，默认 `kDefaultNdDmaConfig`（取值见 §2.4）

### 2.2 NdDmaLoopInfo —— 每维搬运信息（五字段）

```cpp
template <uint8_t dim>            // dim ∈ [1, 5]
struct NdDmaLoopInfo {
    uint64_t loopSrcStride[dim] = {0};  // 源步距
    uint32_t loopDstStride[dim] = {0};  // 目的步距
    uint32_t loopSize[dim]       = {0}; // 迭代数（不含 padding）
    uint8_t  loopLpSize[dim]     = {0}; // 左 padding
    uint8_t  loopRpSize[dim]     = {0}; // 右 padding
};
```

| 字段 | 类型 | 单位 | 范围 | 语义 |
|------|------|------|------|------|
| `loopSrcStride` | uint64_t | 元素 | [0, 2^40) | 该维源侧步距（advance）：+1 迭代源地址前进量。**0=原地重读（broadcast）**，1=相邻，>1=跳读。不支持负值 |
| `loopDstStride` | uint32_t | 元素 | [0, 2^20) | 该维目的（UB）侧步距 |
| `loopSize` | uint32_t | 元素 | [0, 2^20) | 该维迭代数，不含 padding 元素；0 = 0 次迭代（该维不搬运） |
| `loopLpSize` | uint8_t | 元素 | [0, 255] | 该维左侧补值个数 |
| `loopRpSize` | uint8_t | 元素 | [0, 255] | 该维右侧补值个数 |

- 数组索引 [0, dim)：**索引 0 = 最内维**（shape 最右边的维），索引 dim-1 = 最外维
- stride 语义与 torch tensor 的 stride 同一概念：`loopSrcStride` = 源 tensor 的 stride，`loopDstStride` = 目的 tensor（UB 内布局）的 stride；broadcast 轴 `loopSrcStride` 直接填 0（torch expand 是逻辑视图，NDDMA 由硬件真按 stride=0 重读）
- 五字段数组带 `= {0}` 默认初始化，聚合初始化省略的字段取 0；**未给 `loopSize` 赋值的维度为 0 次迭代（静默不搬运）**，需按实际形状填写

### 2.3 NdDmaParams —— 搬运参数 + 填充值

```cpp
template <typename T, uint8_t dim>
struct NdDmaParams {
    NdDmaLoopInfo<dim> loopInfo;
    T constantValue;   // 常数填充值：存在 padding 且 isNearestValueMode=false 时生效；b64 时必须填 0
};
```

⚠️ `constantValue` 无默认值，需要显式赋值。

### 2.4 NdDmaConfig —— 搬运配置（模板参数，可选）

```cpp
struct NdDmaConfig {
    static constexpr uint16_t unsetPad = 0xffff;
    bool isNearestValueMode = false;   // true: padding 取该维最左/最右元素值；false: 填 constantValue
    uint16_t loopLpSize = unsetPad;    // 全维度统一左 padding
    uint16_t loopRpSize = unsetPad;    // 全维度统一右 padding
    bool ascOptimize = false;          // 预留参数，暂不支持
};
```

- `isNearestValueMode`：**b64 时必须为 false**
- `loopLpSize`/`loopRpSize` 取 `[0, 255]` 时覆盖所有维度（`NdDmaLoopInfo` 的同名每维值不生效）；取 `unsetPad`（默认）时按每维值生效

**默认值 `kDefaultNdDmaConfig`**（编译期常量，`config` 不传时使用）：

```cpp
constexpr NdDmaConfig kDefaultNdDmaConfig = { false, NdDmaConfig::unsetPad,
                                               NdDmaConfig::unsetPad, false };
```

逐字段：`isNearestValueMode=false`（常数填充）、`loopLpSize`/`loopRpSize=unsetPad`（不启用全局 padding，按每维值生效）、`ascOptimize=false`。无 padding 场景（broadcast / 跳读 / 转置）所需的正是这套值。

### 2.5 dtype 支持

b8/b16/b32/b64 全系：`bool`、`int8_t`、`uint8_t`、`fp8_e4m3fn_t`、`fp8_e5m2_t`、`fp8_e8m0_t`、`fp4x2_e2m1_t`、`fp4x2_e1m2_t`（fp8/fp4 走 b8 通路）、`int16_t`、`uint16_t`、`half`、`bfloat16_t`、`int32_t`、`uint32_t`、`float`、`complex32`、`int64_t`、`uint64_t`、`double`、`complex64`。源/目的 dtype 必须一致。

**b64（int64_t / uint64_t / double / complex64）**：`isNearestValueMode` 必须 false，`constantValue` 必须 0。

---

## 3 通用使用规则

### 3.1 NdDmaDci：DataCopy 前必调

`NdDmaDci()` 刷新 NDDMA Cache（32KB），必须在 DataCopy 之前调用，防止读到 Cache 中的陈旧数据。调用策略二选一：

- **无写后读依赖场景**（算子内部不存在写 GM/workspace 后，再 NDDMA 读）：`Init` 中调一次即可
- **有写后读依赖场景**（写 GM/workspace 后，再 NDDMA 读）：DataCopy 之前必须重新刷新 cache

### 3.2 最多 5 维

- 一条指令最多描述 **5 个维度**（dim0~dim4），硬件固定，不随切分变化
- 超出 5 维的轴用 **for 循环软驱动**：每轮循环计算该批轴的起始偏移，发一条 NDDMA；参数中 5 个维度填最内 5 层
- 实际维度不足 5 时声明 `dim` = 实际维度数即可；若采用通用写法模板参数固定为 5，剩余维度填 `loopSize=1 / srcStride=0`（迭代 1 次，无害）

### 3.3 参数必须反映实际形状

- `loopSize` 等参数若预填了占位值，**调用前必须按实际形状覆写**（如末块长度小于主块时沿用主块值 → 内存越界）

---

## 4 典型场景用法

通用填法：每个维度填该轴的源 stride（`loopSrcStride`）、目的 stride（`loopDstStride`）、长度（`loopSize`）。示例中 `ubLocal`/`xLocal` 为 UB 目的 tensor，`xGm_[inOff]` 为 GM 源 tensor 起始地址（元素偏移）。

### 4.1 随路 broadcast

- 判定：`inShape[ax]=1 && outShape[ax]>1`
- 填法：广播轴 `loopSrcStride=0`（源原地重读）、`loopSize=outShape[ax]`（目标长度）、`loopDstStride=该轴输出 stride`
- 搬入后 UB 内即广播展开排布，无需额外广播操作，不另占 UB buffer

```cpp
// srcShape=[1,128] → dstShape=[3,128]，外层 broadcast；dim[0]=最内维
NdDmaLoopInfo<2> loopInfo{
    {1, 0},      // loopSrcStride: 内层连续=1, 外层 broadcast=0（原地重读）
    {1, 128},    // loopDstStride: 内层=1, 外层跳 128（一份的长度）
    {128, 3},    // loopSize: 目标 [128, 3]（非源 [128, 1]）
    {0, 0},      // loopLpSize
    {0, 0}       // loopRpSize
};
NdDmaParams<T, 2> params{loopInfo, 0};
NdDmaDci();
DataCopy<T, 2>(ubLocal, xGm_[inOff], params);
```

broadcast 与 padding 可叠加（广播轴 stride=0 与其他维补值互不干扰）：

```cpp
// srcShape=[1,128] → dstShape=[3,128]，外层 broadcast，右补 4 个 0，pad 值 = 0（UB 内 shape=[3, 132]）
NdDmaLoopInfo<2> loopInfo{
    {1, 0},      // loopSrcStride: 内层连续=1, 外层 broadcast=0（原地重读）
    {1, 132},    // loopDstStride: 外层跳 128+4（数据 128 + 右 pad 4）
    {128, 3},    // loopSize: 目标 [128, 3]（不含 pad 元素）
    {0, 0},      // loopLpSize
    {4, 0}       // loopRpSize: 内层右侧补 4 个 0
};
NdDmaParams<T, 2> params{loopInfo, 0};   // constantValue = 0
NdDmaDci();
DataCopy<T, 2>(ubLocal, xGm_[inOff], params);
```

### 4.2 尾轴跳读（最内层 stride > 1）

- 判定：Slice 尾轴 stride > 1
- 填法：该维 `loopSrcStride = sliceStride × inStride[ax]`（尾轴 inStride=1 时即 sliceStride），`loopDstStride=1`，`loopSize=输出迭代数`

```cpp
// stride = 2: 读 c₀, 跳 c₁, 读 c₂, ...
NdDmaLoopInfo<1> loopInfo{{2}, {1}, {count}, {0}, {0}};
NdDmaParams<T, 1> params{loopInfo, 0};
NdDmaDci();
DataCopy<T, 1>(ubLocal, xGm_[inOff], params);
```

### 4.3 尾轴转置

- 判定：perm[N-1] ≠ N-1（尾轴参与转置，输入尾轴不连续读取）
- 填法：src/dst stride 交叉——沿输入列方向读（src 跳行宽），UB 按转置后布局写（dst 内层连续）

```cpp
// [H, W] → [W, H]: 沿 H 读一列（跳 W），迭代 W 列
NdDmaLoopInfo<2> loopInfo{
    {W, 1},      // loopSrcStride: 内层沿 H 方向跳 W, 外层步进 1 到下一列
    {1, H},      // loopDstStride: 内层逐元素写, 外层跳过 H 个
    {H, W},      // loopSize: 每列 H 个, W 列
    {0, 0}, {0, 0}
};
NdDmaParams<T, 2> params{loopInfo, 0};
NdDmaDci();
DataCopy<T, 2>(ubLocal, xGm_[inOff], params);
```

> 等价镜像填法：内层改为输入行连续读取（srcStride 内=1、外=行宽），dst 内层跨步（=行数）。两种排列结果相同。

### 4.4 Padding 补值

- 填法：每维 `loopLpSize`/`loopRpSize`（或 NdDmaConfig 全局值）；填充值 = `constantValue`（默认）或该维最左/最右元素值（`isNearestValueMode=true`）
- `loopSize` 不含 padding 元素；每维 padded 跨度 = Lp + size + Rp，外层 `loopDstStride` 按该跨度填

```cpp
// 常数填充：xGm [16,32]（16 行 × 32 列）→ xLocal [32,64]
// 左 15 / 右 17（内维），上 13 / 下 3（外维），padding 值 = 0
NdDmaLoopInfo<2> loopInfo{{1, 32}, {1, 64}, {32, 16}, {15, 13}, {17, 3}};
NdDmaParams<T, 2> params{loopInfo, 0};   // constantValue = 0
NdDmaDci();
DataCopy<T, 2>(xLocal, xGm, params);
```

```cpp
// 最近值填充：左右 padding 取当前维最左/最右元素的值
static constexpr NdDmaConfig dmaConfig = {true};   // isNearestValueMode = true
DataCopy<T, 2, dmaConfig>(xLocal, xGm, params);    // params 同上
```

---

## 5 约束汇总

| # | 约束 | 违规后果 |
|---|------|---------|
| 1 | stride / size 全部单位 = **元素个数**，禁止 `× sizeof(T)` | 地址错位 |
| 2 | `loopSrcStride ∈ [0, 2^40)`，**不支持负 stride**；0 = 原地重读 | — |
| 3 | `loopDstStride ∈ [0, 2^20)` | 超范围静默截断（无报错），地址错位 |
| 4 | `loopSize ∈ [0, 2^20)`，不含 padding，按实际搬运长度填；0 = 该维 0 次迭代（不搬运） | 静默丢数据 |
| 5 | `dim ∈ [1, 5]`；变化轴 > 5 → for 循环驱动多次 NDDMA | 编不过 / 越界 |
| 6 | `NdDmaDci()` 必须在 DataCopy 前（NDDMA Cache 32KB） | 多核读过期数据 |
| 7 | dtype b8/b16/b32/b64；b64 时 `isNearestValueMode=false` 且 `constantValue=0` | — |
| 8 | 单指令总地址跨度 ≤ 40 位（1TB）：每层循环跨度 = `(loopLpSize + loopSize + loopRpSize − 1) × 该层 stride`（src/dst 两侧同理），所有层加总 ≤ 2^40 | — |
| 9 | 各层 dstStride 为升序序列时，不同循环间地址空间不能交织或重叠 | 数据错乱 |
| 10 | 通路仅 GM → VECIN（UB），不支持 UB→GM | — |

---

## 6 常见错误（实战案例）

| 案例 | 症状 | 根因 | 修复 |
|------|------|------|------|
| 参数单位填成字节 | 地址错位、结果错乱 | stride/size 填了 `× sizeof(T)` 的字节数 | 全部按元素个数填 |
| 末块忘覆写 loopSize | 末块越界 / UB 溢出 | 按主块 ubFactor 搬运了尾块 | 调用前用 `min(ubFactor, 剩余)` 覆写 |
| padding 值随机 | padding 区域为随机数 | `constantValue` 未显式赋值 | 显式赋值 |
