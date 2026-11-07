# DataCopy / DataCopyPad 使用指南

GM ↔ UB、UB ↔ UB 数据搬运的完整指南（含 `DataCopyParams` gap 地址公式与参数单位速查）。多维切片见 [api-nd2nz-nz2nd-nddma.md](api-nd2nz-nz2nd-nddma.md)。

> **边界**：本文只讲 GM↔UB / UB↔UB 的**纯搬运**；Cube 通路（ND2NZ / LoadData / Fixpipe）见各自家族文档。全局接口选型与跨接口 stride 对照见 [api-quickref.md](api-quickref.md)。

---

## 目录

1. [选择规则](#选择规则)
2. [32 字节对齐要求](#32-字节对齐要求)
3. [连续 DataCopy 与 32B 截断](#连续-datacopy-与-32b-截断)
4. [DataCopyPad 参数详解](#datacopypad-参数详解)
5. [SetPadValue](#setpadvalue)
6. [使用场景示例](#使用场景示例)
7. [stride 参数详解](#stride-参数详解)
8. [DataCopyParams 与 gap 地址公式](#datacopyparams-与-gap-地址公式)
9. [二维 stride 搬运示例](#二维-stride-搬运示例)
10. [性能规则：64B pitch 与 2D 合并搬运](#性能规则64b-pitch-与-2d-合并搬运)
11. [逐 chunk 循环中的 DataCopy 快路径](#逐-chunk-循环中的-datacopy-快路径datacopypad-开销规避)
12. [UB 内搬运 DataCopy 与 Copy 的区别](#ub-内搬运-datacopy-与-copy-的区别)
13. [参数单位速查](#参数单位速查)
14. [常见错误与调试](#常见错误与调试)

---

## 选择规则

**原则：不确定是否 32 字节对齐时，优先使用 DataCopyPad**（已确认严格 32B 对齐且连续时，`DataCopy(..., count)` 更直接，见「连续 DataCopy 与 32B 截断」）

| 场景 | API | 原因 |
|-----|-----|------|
| **非对齐或不确定对齐** | `DataCopyPad` | 自动处理对齐/非对齐，避免边界 bug |
| **数据量严格 32 字节对齐** | `DataCopy` 或 `DataCopyPad` | 确定对齐时 DataCopy 可用，DataCopyPad 更安全 |

### ⛔️ 黑名单 API（禁止在生产代码中使用）

| API | 禁止原因 | 仅允许场景 |
|-----|---------|-----------|
| `GlobalTensor::SetValue(idx, val)` | 效率极低，单元素逐个写入 | **仅调试时使用** |
| `GlobalTensor::GetValue(idx)` | 效率极低，单元素逐个读取 | **仅调试时使用** |

```cpp
// ❌ 禁止：生产代码使用 SetValue/GetValue
for (uint32_t i = 0; i < size; i++) {
    xGm.SetValue(i, value);    // ⛔️ 效率极低
    T val = xGm.GetValue(i);   // ⛔️ 效率极低
}

// ✅ 正确：使用 DataCopyPad 批量搬运
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);

// ✅ 允许：调试时单点验证
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));  // 仅调试
```

**为什么优先 DataCopyPad？**

1. 自动处理非对齐，无需手动判断
2. CopyIn 和 CopyOut 都适用
3. Tiling 设计时可能产生非对齐的 tile 大小
4. 对齐场景下性能差异可忽略

### DataCopyPad 参数选择：统一使用 Ext 版本

`DataCopyPad` 同时接受 `DataCopyParams` 和 `DataCopyExtParams`，两者底层走**同一条硬件指令**，性能无差异。统一使用 Ext 版本：

| 参数 | 非 Ext 版本 | Ext 版本 | 推荐 |
|------|-----------|---------|------|
| 搬运参数 | `DataCopyParams`（uint16_t，blockLen 最大 65535） | `DataCopyExtParams`（uint32_t，blockLen 最大 2097151） | **Ext** |
| 填充参数 | `DataCopyPadParams`（paddingValue 为 uint64_t） | `DataCopyPadExtParams<T>`（paddingValue 为 T 类型） | **Ext** |

**理由**：
1. 参数范围更大，避免溢出风险
2. `DataCopyPadExtParams<T>` 填充值类型安全，编译期检查
3. 性能完全一致（底层同一条 MTE 指令）
4. 唯一代价是多写一个 `rsv=0` 字段

```cpp
// ❌ 不推荐：DataCopyParams 参数范围有限，paddingValue 类型不安全
AscendC::DataCopyParams copyParams{1, cols * sizeof(float), 0, 0};
AscendC::DataCopyPadParams padParams{true, 0, padElements, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);

// ✅ 推荐：统一使用 Ext 版本
AscendC::DataCopyExtParams copyParams{1, cols * sizeof(float), 0, 0, 0};
AscendC::DataCopyPadExtParams<float> padParams{true, 0, padElements, 0.0f};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

---

## 32 字节对齐要求

**DataCopy 要求 32 字节对齐**。不满足时的具体后果见下节「连续 DataCopy 与 32B 截断」——**静默向下截断**，不报错。

| 数据类型 | 对齐元素数 | 最小对齐字节数 |
|---------|-----------|--------------|
| half (2 bytes) | 16 | 32 |
| float (4 bytes) | 8 | 32 |
| int32_t (4 bytes) | 8 | 32 |
| fp8 (1 byte) | 32 | 32 |

---

## 连续 DataCopy 与 32B 截断

```cpp
// GM -> UB
DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count);
// UB -> GM
DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count);
```

`count` 是**元素数量**，不是 byte 数、也不是 32B block 数：

```cpp
LocalTensor<half> xLocal;
GlobalTensor<half> xGm;
DataCopy(xLocal, xGm, 128);   // 128 half × 2B = 256B = 8 个 DataBlock
```

### ⚠️ 最关键约束：非 32B 整数倍时向下截断（不报错）

`count * sizeof(T)` 必须是 32B 的整数倍。**不满足时不会搬全部数据，而是向下截断到完整 32B**：

```cpp
DataCopy<half>(dst, src, 17);   // 17 half = 34B → 只搬 16 half = 32B
```

头文件实现说明：`In NPU mode, no error is reported. The value is rounded down by 32B.` 这是**静默丢数据**——不中断执行，只表现为尾部数据不对，比报错更难查。（NPU 模式下 `ASCENDC_ASSERT` 仍会打印 `KERNEL_ERROR` 日志，但**不会终止执行**；因此不能靠「运行没报错」判断搬运完整。）

地址对齐要求：UB 侧地址必须 32B 对齐，GM 侧至少按元素类型对齐。

> 因此：**连续 + 长度 32B 对齐时优先 `DataCopy(..., count)`；不确定对齐一律 `DataCopyPad`。**

**来源**：`asc/include/basic_api/kernel_operator_data_copy_intf.h`（`DataCopy(count)` 无 arch 守卫，DAV_2201 与 DAV_3510 均可用）；截断语义见 `impl/basic_api/kernel_operator_data_copy_intf_impl.h`。

---

## DataCopyPad 参数详解

### isPad 参数

| isPad | 含义 |
|-------|------|
| `false` | 框架自动填充，用户不指定填充值 |
| `true` | 用用户指定的 `paddingValue` 填充 |

### blockLen 非对齐时的填充行为

**GM → UB（CopyIn）**：

| 条件 | isPad | dummy 填充值 |
|-----|-------|-------------|
| leftPadding=0, rightPadding=0 | false | **第一个元素值** |
| leftPadding=0, rightPadding=0 | true | paddingValue |
| leftPadding≠0 或 rightPadding≠0 | false | 随机值 |
| leftPadding≠0 或 rightPadding≠0 | true | paddingValue |

**UB → GM（CopyOut）**：
- 框架自动处理非对齐
- 搬到 GM 时自动丢弃 dummy

### UB 端起始地址 32B 对齐（易踩坑）

`DataCopyPad(GM, UB, ...)` 与 `DataCopyPad(UB, GM, ...)` 的 **UB 端起始地址必须 32 字节对齐**（blockLen 可以非 32B 对齐，但起始地址不能）。

按行索引访问 UB buffer 时，行偏移字节数必须是 32 的倍数：

```cpp
// ❌ cols * sizeof(elem) 不是 32 倍数时，row * cols 偏移可能落非对齐地址
// 例如 fp8 + cols=4 时每行 4 字节，只有 row ∈ {0, 8, 16,...} 满足 32B 对齐
DataCopyPad(gmOut[off], ubBuf[row * cols], copyParams);
```

修复方式：引入 strided staging buffer，把不规则行宽数据重排到每行 32B 对齐的连续区：

```cpp
// ✅ 用 strided buf 重排，保证每行 UB src 起址 32B 对齐
auto stridedBuf = strideBuf_.Get<elem_T>();
for (int row = 0; row < mEff; ++row) {
    for (int j = 0; j < cols; ++j) {
        stridedBuf.SetValue(row * 32 + j, ubBuf.GetValue(row * cols + j));
    }
}
DataCopyPad(gmOut[off], stridedBuf[row * 32], copyParams);  // src 每行 32B 对齐
```

**错误码症状**：
- `AIV error 80: The UB address accessed by the VEC instruction is not aligned`
- 连锁触发 `AIC error: timeout or trap error. subErrType: 0x4`

### blockCount 参数限制

`DataCopyPad` 的 `blockCount` 字段最大值 4095，超出需分批搬运。Host 侧 Tiling 计算必须 clip：

```cpp
constexpr uint32_t MAX_BLOCK_COUNT = 4095;
tileRows = std::max(1u, std::min(tileRows, MAX_BLOCK_COUNT));
```

### DataCopyExtParams 字段与单位（GM → UB）

```cpp
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    int64_t  srcStride;   // DAV_3510（950）
    int64_t  dstStride;
#else
    uint32_t srcStride;   // DAV_2201（A2/A3）
    uint32_t dstStride;
#endif
    uint32_t rsv;         // 保留字段，默认 0
};
```

| 参数 | 类型（2201 / 3510） | GM → UB 单位 |
|---|---|---|
| `blockCount` | `uint16_t` | 次 |
| `blockLen` | `uint32_t` | **Byte** |
| `srcStride` | `uint32_t` / `int64_t` | **Byte** |
| `dstStride` | `uint32_t` / `int64_t` | **32B DataBlock** |
| `rsv` | `uint32_t` | 保留字段，默认 0 |

这与普通 `DataCopyParams` 最大的区别：普通 `blockLen` 以 32B 为单位，`DataCopyPad` 的 `blockLen` 以 **1 byte** 为单位——因为它必须能表达 `47B`、`59 × sizeof(float)`、`126B` 等非对齐长度。

### 跨代字段宽度差异

`DataCopyExtParams` 的 `srcStride` / `dstStride` 在 **DAV_3510 上是 `int64_t`**，在 **A2/A3（DAV_2201）上是 `uint32_t`**。字段宽度变化会影响聚合初始化与溢出行为，跨代移植时需确认。

### DataCopyPadExtParams：padding 单位是 element

```cpp
template <class T>
struct DataCopyPadExtParams {
    bool isPad;
    uint8_t leftPadding;
    uint8_t rightPadding;
    T paddingValue;
};
```

| 参数 | 含义 |
|---|---|
| `isPad` | 是否明确指定 pad value |
| `leftPadding` | 左侧补几个 **元素** |
| `rightPadding` | 右侧补几个 **元素** |
| `paddingValue` | 填充值 |

> ⚠️ `leftPadding` / `rightPadding` 单位 = **element，不是 byte**。例如 `T = half`、`blockLen = 20 * sizeof(half)`（40B）、`rightPadding = 12`，表示 `20 half 原数据 + 12 half padding = 32 half = 64B`。
> 【推断】官方另有「left/right padding 各自占用字节数不超过 32B」的约束，本机头文件中未见对应注释，使用时以官方文档为准。

### CopyOut 方向没有 DataCopyPadExtParams

`DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams)` **只有 3 个参数**——UB → GM 方向不存在 `DataCopyPadExtParams` 重载。框架按 `blockLen` 截断，自动丢弃 UB 尾部 dummy，无需也不能指定填充值。

---

## SetPadValue

```cpp
SetPadValue(value);
```

可提前设置 DMA padding 寄存器：

```cpp
SetPadValue((half)0);

DataCopyPadExtParams<half> pad;
pad.isPad = false;
DataCopyPad(...);
```

但 `SetPadValue` 的语义不如 `pad.isPad = true; pad.paddingValue = ...` 明确。**若 padding 会参与后续计算，应显式走后者**，不要依赖 unspecified / dummy 值：

```cpp
pad.isPad = true;
pad.paddingValue = 0;
```

`SetPadValue` 在 DAV_2201 与 DAV_3510 均可用。

---

## 使用场景示例

### 场景1：非对齐 CopyIn，不关心填充值

```cpp
// cols=5 (FP32)，blockLen=20字节，非对齐
// 后续计算只处理 cols 个元素，dummy 被忽略
AscendC::DataCopyExtParams copyParams{1, cols * sizeof(float), 0, 0, 0};
AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);

// 后续计算只处理 cols 个元素
AscendC::ReduceMax(tmpReduce, xLocal, tmpReduce, cols, false);
```

### 场景2：非对齐 CopyIn，指定填充值

```cpp
uint32_t padElements = paddedCols - cols;
AscendC::DataCopyPadExtParams<float> padParams{true, 0, padElements, 0.0f};
AscendC::DataCopyExtParams copyParams{1, cols * sizeof(float), 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

### 场景3：非对齐 CopyOut

```cpp
// CopyOut 自动处理非对齐，搬到 GM 时丢弃 dummy
AscendC::DataCopyExtParams copyParams{1, cols * sizeof(float), 0, 0, 0};
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 完整示例：多行批量搬运

```cpp
__aicore__ inline void CopyInBatch(uint32_t startLocalRow, uint32_t rowsThisTile)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = rowsThisTile;
    copyParams.blockLen = cols * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    
    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = paddedColsT - cols;
    padParams.paddingValue = 0;
    
    AscendC::DataCopyPad(xLocal, xGm[startLocalRow * cols], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}
```

### 场景4：逐行独立处理模式（Softmax/LayerNorm 推荐）

**适用场景**：Softmax / LayerNorm 等逐行独立计算的算子，不需要跨行 Reduce。

**核心要点**：blockCount 模式 + UB 对齐存储

```cpp
// ========== Tiling 参数 ==========
uint32_t rLength = 13;                           // 有效数据个数
uint32_t rLengthAlign = (rLength + 7) / 8 * 8;   // 对齐到 8 元素（FP32 下 32 字节）

// ========== 数据搬运 ==========
AscendC::DataCopyExtParams copyInParams{
    static_cast<uint16_t>(rows),             // blockCount: 行数
    static_cast<uint32_t>(rLength * sizeof(T)), // blockLen: 有效数据长度（非对齐！）
    0, 0, 0};                                // stride + rsv: 连续存储
AscendC::DataCopyPadExtParams<T> padInParams{false, 0, 0, static_cast<T>(0)};
AscendC::DataCopyPad(xLocal, xGm[offset], copyInParams, padInParams);

inQueueX.EnQue(xLocal);
auto xIn = inQueueX.DeQue<T>();

// ========== 逐行处理 ==========
for (uint32_t row = 0; row < rows; row++) {
    // 关键：UB 偏移用 rLengthAlign，不是 rLength！
    uint32_t rowOffset = row * rLengthAlign;
    
    // Reduce API 只传 rLength（有效数据个数）
    AscendC::ReduceMax<T>(rowTmp, xIn[rowOffset], reduceTmp, 
        static_cast<int32_t>(rLength), false);
    // ... Sub, Exp, ReduceSum, Div
}

// ========== 写回 GM ==========
AscendC::DataCopyExtParams copyOutParams{
    static_cast<uint16_t>(rows), static_cast<uint32_t>(rLength * sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(yGm[offset], yOut, copyOutParams);
```

**关键对照表**：

| 参数位置 | 用 rLength | 用 rLengthAlign |
|---------|-----------|-----------------|
| DataCopyPad blockLen | ✓ | ✗ |
| Reduce API count | ✓ | ✗ |
| Sub/Exp/Div count | ✓ | ✗ |
| UB rowOffset | ✗ | ✓ |
| Buffer 大小 | ✗ | ✓ |

**UB 数据布局示意**：

```
GM（连续存储）:  [row0: 13元素][row1: 13元素][row2: 13元素]...
                         ↓ DataCopyPad blockCount 模式
UB（对齐存储）:  [row0: 13+3=16][row1: 13+3=16][row2: 13+3=16]...
                         ↑
                  每行 padding 到 8 元素对齐
```

---

## stride 参数详解

**stride 参数单位取决于操作数位置**：

| 操作数位置 | stride 单位 | 说明 |
|-----------|------------|------|
| GlobalTensor (GM) | **字节** | 相邻数据块的字节间隔 |
| LocalTensor (UB) | **dataBlock (32字节)** | 相邻数据块的 32字节块间隔 |

**stride 含义**：相邻数据块之间的间隔（前一块尾部到后一块头部的距离）

> **记忆规则**：`DataCopyPad` 中 **GM 一侧的 stride 是 byte；UB 一侧的 stride 是 32B DataBlock**（GM→UB 与 UB→GM 两个方向都适用）。而 `DataCopyParams` 的 `srcGap`/`dstGap` 是**两侧都按 32B**，语义为 tail → next head（见「DataCopyParams 与 gap 地址公式」）。

### UB → GM 多行搬运（CopyOut）

```cpp
// UB 中每行: [cols 有效数据][padElements padding]
// 相邻行间隔 = paddedColsT - cols 个元素
// stride 单位取决于操作数逻辑位置：VECIN/VECOUT 侧为 32 字节块，GM 侧为字节
AscendC::DataCopyExtParams copyParams;
copyParams.blockCount = rowsThisTile;
copyParams.blockLen = cols * sizeof(T);
copyParams.srcStride = (paddedColsT - cols) * sizeof(T) / 32;  // UB stride 单位: 32字节
copyParams.dstStride = 0;  // GM stride 单位: 字节
copyParams.rsv = 0;

AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 常见错误

```cpp
// ❌ 错误：srcStride 理解为行长度
copyParams.srcStride = paddedColsT * sizeof(T) / 32;  // 这会导致输出错位

// ✅ 正确：srcStride 是间隔
copyParams.srcStride = (paddedColsT - cols) * sizeof(T) / 32;
```

---

## DataCopyParams 与 gap 地址公式

普通 GM↔UB `DataCopy` 做高性能 stride 搬运最重要的一组参数：

```cpp
DataCopy(dst, src, DataCopyParams params);

struct DataCopyParams {
    uint16_t blockCount;
    uint16_t blockLen;
    uint16_t srcGap;   // 旧名 srcStride（deprecated）
    uint16_t dstGap;   // 旧名 dstStride（deprecated）
};
```

头文件中 `srcGap` / `srcStride` 以 `union` 并存，语义相同——**`Gap` 与 `Stride` 在这里是同义词，不是两种单位**。

| 参数 | 单位 | 含义 |
|---|---:|---|
| `blockCount` | block 个数 | 一共搬多少段（≤ 4095） |
| `blockLen` | 32B | 每一段实际搬多少 |
| `srcGap` | 32B | 两段**源**数据之间跳过多少 |
| `dstGap` | 32B | 两段**目的**数据之间跳过多少 |

### gap 是「上一段尾 → 下一段头」，不是 pitch

以 `blockCount = 3`、`blockLen = 2`、`srcGap = 1`、`dstGap = 0` 为例（每个 DataBlock = 32B）：

- 源：`[64B 有效][32B 跳过][64B 有效][32B 跳过][64B 有效]`
- 目的：`[64B 有效][64B 有效][64B 有效]`

地址公式：

```text
src_i = src_base + i × (blockLen + srcGap) × 32
dst_i = dst_base + i × (blockLen + dstGap) × 32
```

> ⚠️ `srcGap` / `dstGap` 是「上一段**尾** → 下一段**头**的空隙」，**不是常规数组意义上的 pitch**。只有 `gap = 0` 才意味着连续。
> 这与 `LoadData2D` 的 `srcStride`（**头 → 头**，`= 1` 表示连续）语义相反，不要跨接口套用——见 [api-loaddata.md](api-loaddata.md)。

---

## 二维 stride 搬运示例

GM `shape = [16, 8192 float]`，每行 `8192 × 4B = 32KB`，但每行只取前 `512 float = 2048B`，搬 16 行并在 UB 紧密排列：

```cpp
constexpr uint32_t H = 16;
constexpr uint32_t valid = 512;
constexpr uint32_t pitch = 8192;

DataCopyParams p;
p.blockCount = H;
p.blockLen   = valid * sizeof(float) / 32;              // 512*4/32 = 64
p.srcGap     = (pitch - valid) * sizeof(float) / 32;    // (8192-512)*4/32 = 960
p.dstGap     = 0;

DataCopy(xLocal, xGm, p);
```

含义：GM 中每行先搬 2048B、再跳过 30720B；经 DMA 后 UB 中紧密排列为 `[2048B][2048B]…`。

这通常明显优于拆成 16 次小 `DataCopy`：

```cpp
// ❌ 能用一次 DataCopyParams 描述二维块搬运时，不要拆成大量小 DataCopy
for (int i = 0; i < 16; ++i) {
    DataCopy(...);
}
```

---

## 性能规则：64B pitch 与 2D 合并搬运

多行搬运场景（如批量归约把 GM 的分片数据搬入 UB）的两条性能规则：

1. **行 pitch 对齐到 64B（而非最小 32B）**：UB 侧行宽 padding 到 64B 的整数倍，GM 侧目标地址同样按 64B 对齐——DMA 突发粒度对齐后 GM 访存/MTE2/MTE3 带宽利用率显著提升。UB 预算必须按 **pitch 后**元素数核算（不是按有效列数）
2. **非连续行用一次 2D DataCopyPad（blockCount + srcStride/dstStride），禁止逐行多次 1D 调用**：每次 DataCopyPad 调用都有固定下发开销，逐行调用会把搬运变成同步点密集的串行序列；2D 模式一次调用完成全部行，且便于与双缓冲/事件流水配合
3. **累加/归约循环内 blockCount 必须 > 1**：归约/累加循环搬运多行数据时，`blockCount` 必须等于本批行数（多行一次搬运）。`blockCount=1` 逐行搬运 = **性能反模式**——flag/同步次数 = 行数 × 段数 × 来源数 爆炸（生产踩坑：归约逐行搬运曾是某多行归约实现的性能缺陷根因之一）

> ⚠️ **硬件隐式上限（DAV_3510 实测）**：2D DataCopyPad 当 `srcStride > 0` 或 `dstStride > 0`（行间跳步）时，`blockCount` 存在 **~29-32 行隐式上限**，超出行**静默丢弃为零**（无报错、无异常，输出呈周期性零值分布，period=blockCount）。防御三法（可并用）：① host 侧钳制单批行数 ≤ 32（strided 场景）；② strided 场景退化 1D 逐行 DataCopyPad（blockCount=1 无 stride，绕开限制——非 strided 场景仍保留 2D 批量）；③ 用例必须覆盖 strided 边界场景（隐式上限类缺陷只在边界用例下暴露，常规用例无法发现）。
>
> **证据状态**：本条为 DAV_3510 上的实测结论，本机 CANN 头文件与官方文档均**无对应说明**（未引证，平台/CANN 构建号待补）。已据此设计规避方案的调用方见 `ascendc-mc2-best-practice`（`references/foundations/apace/fundamentals/fusion.md` 的归约搬入/输出条款、`references/foundations/apace/review-checklist.md` R17 门禁），**修改本条前需同步核对那两处**。

```cpp
// ✅ 2D 合并搬运：tileM 行一次完成，行 pitch 64B 对齐
// GM 侧：行间隔 dstStride/srcStride（字节）；UB 侧：paddedCols = round_up(cols*sizeof(T), 64)/sizeof(T)
copyParams.blockCount = tileM;
copyParams.blockLen   = cols * sizeof(T);
copyParams.srcStride  = gmRowPitchBytes - cols * sizeof(T);        // GM 行间隔（字节）
copyParams.dstStride  = (paddedCols - cols) * sizeof(T) / 32;      // UB 行间隔（32B 块）
```

> 连带纪律：批量（多行）处理后**一批数据一次 SetFlag/WaitFlag**（MTE2/V/MTE3 FIFO 特性），不要逐行配对事件——flag 次数是搬运-计算流水的主要隐藏开销。

## 逐 chunk 循环中的 DataCopy 快路径（DataCopyPad 开销规避）

逐块循环处理的算子（scatter/归约/前缀类）中，全量使用 `DataCopyPad` 会把 pad 逻辑
固定开销逐 chunk 放大（实测整类算子 geomean 差 ~32%）：

- **全对齐 chunk 用 `DataCopy`，仅尾部残差用 `DataCopyPad`**：保持 `CHUNK` 为 32 的
  倍数、`offset` 为 `CHUNK` 倍数 → 源/目的地址天然 32B 对齐，主体循环走无 pad 开销
  的快路径；仅最后一chunk 任意长度走 `DataCopyPad`
- **`Copy`（UB→UB，count 模式）仅 `__NPU_ARCH__ == 3510` 可用**：该重载在头文件中被
  `#if (__NPU_ARCH__ == 3510)` 包住，Atlas A2/A3 不可用
- **`DataCopy(GM↔UB, count)` 没有 arch 守卫**：DAV_2201 与 DAV_3510 均可用，A2 上使用它
  **不需要**任何额外的 `SOC_VERSION` 配置

```cpp
if (nRaw == CHUNK) {   // 全 chunk：天然 32B 对齐
    DataCopy(local_, gm_[offset], CHUNK);
} else {               // 仅尾部残差
    DataCopyPad(local_, gm_[offset], cp, pp);
}
```

---

## UB 内搬运 DataCopy 与 Copy 的区别

当前 API 同时存在：

```cpp
DataCopy(UB, UB, ...)   // UB 内搬运
Copy(...)               // UB 内 Vector 型搬运
```

官方 API 表把它们明确分成两类，可以粗略理解（【推断】模型类比，非 API 事实）：

| 接口 | 模型 |
|---|---|
| `DataCopy` | 更接近 memory-transfer / DataBlock 模型 |
| `Copy` | 更接近 Vector 数据访问模型，支持 mask、repeat、block stride 等 |

**不要因为都能完成 `UB A → UB B`，就认为底层机制和参数模型相同**——它们的 stride 单位、mask 语义与平台可用性都不同（`Copy` 的 count 模式仅 DAV_3510 可用，见本文「逐 chunk 循环中的 DataCopy 快路径」）。

---

## 参数单位速查

```cpp
// ==================================================
// 1. 普通连续 DataCopy
// ==================================================
DataCopy(dst, src, count);
// count: elements
// count * sizeof(T) 必须是 32B 的整数倍（否则向下截断，见「连续 DataCopy 与 32B 截断」）


// ==================================================
// 2. 普通 block / stride DataCopy
// ==================================================
DataCopyParams p;
p.blockCount = ...;  // blocks（≤ 4095）
p.blockLen   = ...;  // 32B
p.srcGap     = ...;  // 32B, tail -> next head
p.dstGap     = ...;  // 32B, tail -> next head

DataCopy(dst, src, p);


// ==================================================
// 3. GM -> UB 非对齐
// ==================================================
DataCopyExtParams p {
    blockCount,      // 次
    blockLenBytes,   // Byte
    srcGapBytes,     // GM 侧：Byte
    dstGap32B,       // UB 侧：32B DataBlock
    0                // rsv：保留字段，默认 0
};
DataCopyPadExtParams<T> pad {
    isPad,
    leftPaddingElements,   // element
    rightPaddingElements,  // element
    paddingValue
};
DataCopyPad(dstLocal, srcGM, p, pad);


// ==================================================
// 4. UB -> GM 非对齐
// ==================================================
DataCopyExtParams p {
    blockCount,      // 次
    blockLenBytes,   // Byte
    srcGap32B,       // UB 侧：32B DataBlock
    dstGapBytes,     // GM 侧：Byte
    0                // rsv：保留字段，默认 0
};
DataCopyPad(dstGM, srcLocal, p);   // 无 DataCopyPadExtParams
```

> 上面骨架是「可复制粘贴」形态，单位注释即最终口径；规则的推导与记忆法见本文「stride 参数详解」与「DataCopyParams 与 gap 地址公式」。

其余接口族（ND2NZ / NZ2ND / NDDMA / LoadData / Fixpipe）的单位见 [api-nd2nz-nz2nd-nddma.md](api-nd2nz-nz2nd-nddma.md)、[api-loaddata.md](api-loaddata.md)、[api-cross-gen-fixpipe.md](api-cross-gen-fixpipe.md) 与 [api-quickref.md](api-quickref.md) 的跨接口速查表。

---

## 常见错误与调试

### 错误1：CopyIn/CopyOut 非对齐数据用 DataCopy

```cpp
// ❌ 错误
AscendC::DataCopy(xLocal, xGm, 4);  // cols=4 (16 bytes)，数据错误

// ✅ 正确
AscendC::DataCopyExtParams copyParams{1, 4 * sizeof(float), 0, 0, 0};
AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

### 错误2：CopyIn 用 DataCopyPad，CopyOut 用 DataCopy

```cpp
// ❌ 错误：CopyIn 和 CopyOut 都需要处理非对齐
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
AscendC::DataCopy(yGm, yLocal, 4);  // 输出错误

// ✅ 正确：两边都用 DataCopyPad（统一 Ext 版本）
AscendC::DataCopyExtParams copyParams{1, cols * sizeof(float), 0, 0, 0};
AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 调试步骤

遇到数据错误时：

1. **分别验证 CopyIn 和 CopyOut**
   - 用 "CopyIn → CopyOut" 测试搬运是否正确
2. **检查数据量是否 32 字节对齐**
3. **非对齐场景：CopyIn 和 CopyOut 都用 DataCopyPad**

### 实战案例：SoftmaxV5

**问题**：FP32 cols=4,5,6,7 时结果错误，cols=8 正常

**根因**：
1. CopyIn 用 DataCopyPad 但 isPad=false（填充随机值）
2. CopyOut 用 DataCopy 处理非对齐输出

**解决**：CopyIn 和 CopyOut 都用 DataCopyPad
