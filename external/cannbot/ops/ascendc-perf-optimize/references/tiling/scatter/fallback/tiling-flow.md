# Scatter 族 — 通用 Tiling 流程

> 代际限定：本文 RegBase / Reg / MicroAPI / VF 相关实现仅适用于 arch35（A5 / DAV_3510）；通用 Tiling 建模不受此限制。

> Scatter 类算子的**六路径决策树**与线程数 / block 数 / workspace 推导。
>
> **重要**：本族的 Tiling 不切 UB。写地址由运行期 `indices` 决定，UB tile 化不成立；
> 真正要决定的是 **① 并行划分方式**（线程 = 一条纤维 还是 一个更新元素）与 **② 冲突消解方式**（免原子 / 硬件原子 / 键仲裁）。
> 选错是数量级差距——实测同一用例 **12602us ↔ 3767us**（`multiply` 被错误地排除在元素级原子路径外）。
>
> 阈值与吞吐数字均为 **Ascend950PR 实测**（该实现评测 72.46 分 / 20-20 通过 / Scatter 算子榜第 1），推断性结论已显式标注。

---

## 1. 核心概念

| 术语 | 含义 |
|------|------|
| **纤维（fiber）** | 固定所有**非 dim 坐标**、只让 dim 维坐标变化扫出的那条线。`dim=0` 的二维张量里就是"一列" |
| **d** | 归一后的 scatter 轴：`d = dim < 0 ? dim + rank : dim` |
| **outer** | `d` 之前所有维乘积（按 `indices` 的 shape 取） |
| **inner** | `d` 之后所有维乘积（一条纤维内沿 dim 的步长） |
| **numFibers** | `outer × inner` —— **纤维路径的并行宽度上限** |
| **sIdx / sUpd / sVar** | `indices` / `updates` / `data` 在 `d` 维的长度 |
| **epTotal** | `outer × sIdx × inner` = `indices` 元素数 —— **元素级路径的并行宽度** |
| **esize** | 值的字节宽（2 / 4 / 8） |
| **extentsMatch** | 三张量**非 dim 维范围是否全部相等**。决定走快路还是通用路径 |
| **starved** | `numFibers < coreNum × 1024` —— 纤维路径铺不满核 |

### 1.1 合轴公式（Step 0）

```
给定: data.shape, indices.shape, updates.shape, dim

d      = dim < 0 ? dim + rank : dim
inner  = prod(indices.shape[d+1 : rank])
outer  = prod(indices.shape[0 : d])
numFibers = outer * inner
sIdx   = indices.shape[d];  sUpd = updates.shape[d];  sVar = data.shape[d]
epTotal   = outer * sIdx * inner          // == indices.numel()
numelData = data.numel()
esize     = sizeof(dtype)

extentsMatch = ∀i:  indices.size(i) == updates.size(i)
                 && (i == d || indices.size(i) == data.size(i))
```

> **合轴只用 `indices` 的 shape**，不是 `data` 的。torch 语义只要求 `index.size(k) ≤ self.size(k)`，
> 非 dim 维 `indices`/`updates` 可以比 `data` 小；用 `data.shape` 合轴会算出错误的 `inner`。

### 1.2 核心不变量（决策树的全部依据）

更新元素的落点**只把第 `d` 维坐标换成 index 值，其余坐标原样保留** ⇒ 两个更新撞同一输出位置，
当且仅当非 dim 坐标全同且 index 值相同 ⇒ **冲突严格局限在一条纤维内**。由此得到两条可直接变现的结论：

1. **按纤维分线程 ⇒ 输出集互不相交 ⇒ 完全免原子**；且线程内沿 dim 递增 ⇒ `reduce=none` 天然"后写覆盖"。
2. **元素级并行时 `atomic_max` 按 dim 下标裁决出的胜者唯一**。因为
   `dst = o·sVar·inner + ix·inner + c`（`0≤c<inner`、`0≤ix<sVar`）是**混合进制表示**，`(o,ix,c) ↔ dst` 一一对应
   ⇒ 同一 `dst` 只可能来自同一条纤维，纤维内 `j` 互不相同 ⇒ 比 `j` 永远分得出胜负。

> **后写覆盖是硬要求**：判据 MARE 取最大相对误差，重复索引选错一个胜者就整例挂。
> CANN 内置的 `REDU_NONE` 是裸写、不提供这个保证——`none` 那三条仲裁路径（②③⑤）付的开销就是买这个保证。

---

## 2. 决策树（Step 1–3）

```
Step 0 — extentsMatch?
  ├─ 否 → ⑥ 通用逐维 stride 路径（正确性优先：索引一律 int64，不收窄）
  └─ 是 ↓

Step 1 — starved = (numFibers < coreNum × 1024)?
  ├─ 否（铺得满核）→ ① 纤维独占            ★ 免原子、免 workspace，到此为止
  └─ 是 ↓

Step 2 — 按 reduce 分叉
  ├─ add / amin / amax / multiply（结果与顺序无关）→ ④ 直接原子
  │     · multiply 无硬件原子乘 → CAS 循环（fp32 / int32 / int64）
  │     · fp16/bf16 的 add / multiply → 叠加 fp32 累加器（见 §3.5）
  │     · fp16/bf16 的 amin / amax → 直接用原生半精度原子，**不要**绕 workspace
  └─ none（必须保证后写覆盖）↓

Step 3 — 按键宽与纤维数选仲裁方式
  ├─ esize==4 && jBits+idxBits ≤ 31 && numFibers ≥ 1024 → ⑤ warp 段末
  ├─ esize==2 && sIdx+1 ≤ 65535                        → ② 打包键单趟
  └─ 其余（4/8 字节值、或键装不下）                       → ③ 两趟胜者
```

### 2.1 判据的位宽计算

```cpp
auto bitWidth = [](int64_t n) { int32_t b = 0; for (int64_t v = n - 1; v > 0; v >>= 1) ++b; return b; };

const int32_t jBits   = bitWidth(sIdx);      // ⑤ 用：index 在键内的起始位
const int32_t idxBits = bitWidth(sVar);
// 预留 0xffffffff 作无效 lane 哨兵 ⇒ 合法键最多用低 31 位
const bool warpKeyFits32 = (jBits > 0) && (jBits + idxBits <= 31);

// ② 用：2 字节值 + (j+1) 挤进高 16 位
const bool key32 = (esize == 2) && (sIdx + 1 <= 65535);

// ③ 用：胜者数组记的是线性下标 i+1，必须落在 uint32 内
const bool winnerLinearFits32 = epTotal < (int64_t{1} << 32) - 1;
```

### 2.2 完整判据（Host 侧，四条快路都必须带 `extentsMatch`）

```cpp
const bool needScatter = (numFibers > 0 && sIdx > 0);
const bool starved     = (numFibers < coreNum * 1024);

useGeneric   = !extentsMatch && needScatter;

useWarpLast  = (red == NONE) && needScatter && extentsMatch && starved
               && warpKeyFits32 && esize == 4 && numFibers >= 1024;
usePackArb   = (red == NONE) && needScatter && extentsMatch && starved
               && key32 && !useWarpLast;
useWinPass   = (red == NONE) && needScatter && extentsMatch && starved
               && !key32 && !useWarpLast && winnerLinearFits32;
useAtomicEP  = (red == ADD || red == AMIN || red == AMAX || red == MUL)
               && needScatter && extentsMatch && starved;
// 以上全 false ⇒ ① 纤维独占
```

> ⚠️ **漏掉 `extentsMatch` 是一个已踩过的真实缺陷**：`usePackArb` 少了它，"2 字节 + none + starved" 时快路与通用路径**同时成立**，
> `y` 被重绑到 workspace 视图而通用分支按连续下标写，**输出以 2 倍元素步长整体错位**。
> 公开用例往往一条通用路径都不触发 ⇒ **标准集满分也照样挂隐藏集**。加任何新快路，第一件事是确认它排除了 `!extentsMatch`。

### 2.3 路径速查

| 路径 | 散射趟数 | 顺序流量 | workspace | 原子宽度 |
|---|---|---|---|---|
| ① 纤维独占 | 1 | clone（1× numel） | 无 | 无 |
| ② 打包键 | 1 | init + unpack（2× numel） | `int32[numel]` | 32 位 |
| ③ 两趟胜者 | 2 | clone + fill-zero（2× numel） | `int32[numel]` | 32 位 |
| ④ 直接原子 | 1 | clone（1× numel）；半精度累加类 +2 趟 cast | 半精度累加类 `float[numel]` | 32 位 |
| ⑤ warp 段末 | 1 | clone（1× numel） | **无** | **无** |
| ⑥ 通用 | 同 ③/④ | 同 ③/④ | 同 ③/④ | 同 ③/④ |

⑤ 在成本上全面占优，但适用面最窄（四条条件全中才用）。

---

## 3. 各路径的 Tiling 推导

### 3.1 ① 纤维独占：线程数是唯一调优点

并行宽度被 `numFibers` **死死卡住**，固定 512 线程/block 会让纤维数小的 shape 只拿到 1~2 个核（实测 `numFibers ≈ 1024` 时加速比 **0.02×**）。

> **策略：在能铺满核的前提下取最大的线程数；铺不满就取最小的。**
> 总线程数不变，但把访存发射摊到尽可能多的核上。

```cpp
constexpr int32_t CANDIDATES[] = {1024, 512, 128, 32, 8};
int32_t chosen = 8;                                  // 兜底取最小
for (int32_t t : CANDIDATES) {
    if ((numFibers + t - 1) / t >= coreNum) { chosen = t; break; }   // 能铺满核的最大线程数
}
int64_t blocks = min(coreNum, (numFibers + chosen - 1) / chosen);
```

`LAUNCH_BOUND` / `dim3` 要求**编译期常量** ⇒ 线程数只能在这 5 个候选值里选，核内按 `switch (threads)` 展开。

### 3.2 ②③④ 元素级并行：固定 1024 线程，按元素数算 block

```cpp
int64_t calc_scatter_blocks(int64_t count) {       // count = epTotal 或 numelData
    int64_t need = (count + 1024 - 1) / 1024;
    return min(coreNum, max<int64_t>(1, need));
}
```

各路径的 `count` 取值：

| 用途 | count |
|---|---|
| ②的 arb pass、③的 win pass、④的原子 pass | `epTotal` |
| ②的 init/unpack、③的 fill-zero、④半精度的 cast | `numelData` |
| ③的 apply pass | 见 §3.4（按密度二选一） |

**每 block 线程数按 pass 定，不是一律 1024**（实测值，非推导）：

| pass | 线程/block | 备注 |
|---|---|---|
| ② arb（打包键 `atomic_max`） | 256 | |
| ③ win（裁决 `atomic_max`） | 256 | |
| ④ `add` | **256** | 实测优于 1024，推测是原子冲突下的重试放大，**换硬件/CANN 版本应重测** |
| ④ `amin`/`amax`/`multiply` | 1024 | |
| ③ apply、fill-zero、⑤、⑥ | 1024 | |

### 3.3 ⑤ warp 段末：按纤维数算 block

一条 warp（32 lane）独占一条纤维，每 block 1024 线程 = 32 个 warp：

```cpp
int64_t calc_scatter_warp_blocks(int64_t numFibers) {
    return min(coreNum, max<int64_t>(1, numFibers));
}
```

每 warp 的排序轮数 = `ceil(sIdx / 32)`，每轮 15 步 bitonic。**这个乘积是⑤的固定成本**，也是纤维数门槛的由来（§5）。

### 3.4 ③ 两趟胜者：apply pass 按密度二选一

```cpp
const bool denseWinResolve = (epTotal * 4 >= numelData);   // 更新足够密
```

| | 遍历对象 | count | blocks | 省掉什么 |
|---|---|---|---|---|
| **密** | 按 `dst` 扫全部输出位置 | `numelData` | `calc_scatter_blocks(numelData)` | 坐标分解（`winner != 0` 就写） |
| **疏** | 按更新元素扫 | `epTotal` | `calc_scatter_blocks(epTotal)` | 扫过大量空位置 |

实现上用 `outer == 0` 作为"密"的标记传入 kernel。

> 趟 1 记的是**线性下标 `i+1` 而不是 `j+1`**：对固定 `(o,c,dst)`，`i` 随 `j` 严格递增，同样能裁决最后写；
> 而且趟 2 可以直接 `updates[winner-1]` 取值，**不必重算 `dst`、也不必再读 `indices`**。
> （⑥ 通用路径例外：`updates` 有自己的 stride，`i` 不能当 `updates` 下标，所以那里记 `j+1`。）

### 3.5 fp16 / bf16 的 fp32 累加器（叠加在 ④ 与 ①/⑥ 之上）

```cpp
const bool accumulating = (red == ADD || red == MUL);
const bool useF32Acc    = isHalfLike && needScatter && accumulating;   // amin/amax 不需要
```

命中时额外分配 `float[numelData]` 并插两趟 SIMD cast（`data → f32`、`f32 → y`）。

| reduce | 半精度走法 | 原因 |
|---|---|---|
| `add` / `multiply` | **必须** fp32 累加器 | `asc_atomic_add(__gm__ half*)` **存在**（在 `simt_api/asc_fp16.h`，**不在** `device_atomic_functions.h`），但**每次操作都舍回 fp16** —— 换用后 MARE 与软件版**逐位重现** `0.020576`，精度过不了 |
| `amin` / `amax` | 直接用原生半精度原子 | 不产生舍入；绕 workspace 只是白付两趟 cast（实测那两趟能占某用例约一半时间） |

`multiply` 的半精度还有一层硬约束：**half/bf16 没有标量 CAS**（只有 `half2` / `bfloat16x2_t`），
所以 `if constexpr (!IsHalfLike<ACC_T>())` 挡掉实例化，半精度乘只能走 fp32 累加器。

### 3.6 workspace 预算汇总

| 路径 | workspace | 尺寸 | 用途 |
|---|---|---|---|
| ② | `int32[numelData]` | `4 × numelData` | 打包键数组（init 装 `data`，arb 仲裁，unpack 取低 16 位） |
| ③ / ⑥(`none`) | `int32[numelData]` | `4 × numelData` | 胜者数组（须预置 0 表示"无人写"） |
| ④/①/⑥ + 半精度累加 | `float[numelData]` | `4 × numelData` | fp32 累加器 |
| ⑥ | `int64[26]` | 208 B | meta：`[0]=rank [1]=dim [2..9]=idxShape [10..17]=dataStride [18..25]=updStride` |
| ①⑤ | — | 0 | 无需 workspace |

> ②③ 的 workspace 按 `numelData`（输出规模）而不是 `epTotal`（更新规模）——胜者/键是**按输出位置**存的。

### 3.7 阶段一 SIMD clone / cast 的 tiling

`y = data` 的整块 clone 与半精度 cast 是**规则顺序流量，必须走 SIMD 的 MTE**（用 SIMT 标量循环实测慢一个数量级：某用例 **1565us → 89us**）。这部分是常规的两级切分：

```cpp
// clone：(numBlocks, blockLength, tileSize)
numBlocks   = min(coreNum, ceil(totalLength / 1024));        // MIN_ELEMS_PER_CORE = 1024
blockLength = ceil(totalLength / numBlocks);
tileSize    = floor(ubSize / PIPELINE_DEPTH(2) / BUFFER_NUM(2) / 512) * 512;   // 512B 对齐
```

两个工程简化：

- **按定宽整型实例化，不按 dtype**：`uint16_t`（2 字节）/ `uint32_t`（4 字节），**int64 按 2× 元素数走 4 字节版本** ⇒ 5 种 dtype 只需两份实例
- cast pass 用固定 `TILE = 16384` 元素，每核连续分片
- `tileSize` **必须 512B 对齐**，否则 `DataCopyPad` 起址/长度不对齐

---

## 4. 多核切分与地址宽度

### 4.1 block 数上限一律是 `coreNum`

所有路径的 block 数都 `min(coreNum, ...)`，靠 grid-stride 循环覆盖余量。不做"超发 block 让调度器排队"的尝试。

### 4.2 索引运算宽度（`narrowIdx`）

```cpp
const int64_t maxLinear = max({numelData, epTotal, outer * sUpd * inner});
const bool narrowIdx = (maxLinear < (int64_t{1} << 31));    // 可用 int32 做索引算术
```

> ⚠️ **实测零收益（+0.04 分）**。向量单元 lane 是 32 位的，int64 算术要拆多条指令，理论上该有收益——
> 但本族瓶颈是**访存延迟**不是标量流水，所以省不出时间。
> 保留该字段是为了对寄存器压力敏感的 VF 复用，**不要指望它提性能**。
> 同理：CANN 内置做了而这里没做的 magic-number 快速除法（`GetUintDivMagicAndShift` + `Simt::UintDiv`）、
> 按 rank 特化 kernel（`SimtComputeDim1~8`）同属标量优化，**预期同样无用**。

### 4.3 坐标分解的一个免费优化

元素级路径把线性下标 `i` 分解成 `(o, j, c)` 时，`outer == 1` 可省掉一次除法：

```cpp
if (outer == 1) { o = 0; j = i / inner;      c = i - j * inner; }
else            { o = i / (sIdx * inner);
                  rem = i - o * (sIdx * inner);
                  j = rem / inner;           c = rem - j * inner; }
```

大量真实 shape 是 `dim=0` 的二维/三维张量，这条特判值得保留。

---

## 5. 实测阈值的来源（改阈值前先读这节）

### 5.1 硬性规律

> 本节数字与最佳实践货架的 [scatter_design.md §5](../../../../../ascendc-performance-best-practices/references/scatter/scatter_design.md) **同源**，改动请同步两处。


| 规律 | 实测数据 | 对 Tiling 的含义 |
|---|---|---|
| **散射吞吐 ∝ 总线程数** | 每 block 8 / 32 / 128 / 512 线程 → 0.5 / 1.7 / 3.3 / 10~27 G updates/s | 访存**延迟**受限，不是带宽受限（顺序流式 pass 能跑 ~600 GB/s）。优先堆总线程数 |
| **32 位原子 ≈ 3× 64 位原子** | 11~18 G/s vs 4.7~5.0 G/s | 宁可拆两趟 32 位（③），别用一趟 64 位键；顺序流量还从 24B/元素降到 12B/元素 |
| **随机散射写放大是数量级的** | 同 kernel、同指令数、同读模式，只换 index 分布：random 2805us / rowwise 583us（**4.81×**）/ constant 562us | 写地址 `y[ix*inner+c]`，一个 warp 内 32 线程 `c` 连续、`ix` 各异 → 落 32 条不同 cacheline 每条只用 2 字节。**换路径救不了**（②③写同一个地址式子） |

### 5.2 `starved` 门槛 = `coreNum × 1024`

元素级并行**买的是并行宽度、付的是原子/仲裁开销**。已经铺满核的 shape 切过去只是多付一趟仲裁，
所以门槛就取"纤维路径能否喂满 `coreNum × 每 block 最大线程数`"。

### 5.3 ⑤ 的纤维数门槛 1024 **不能往下调**

放宽到 256 后，某 `int64 none` 用例（**511 纤维、`sIdx=2048`**）从③的 133.67us 劣化到⑤的 **295.94us**：

- 511 条纤维只能铺 `511 × 32 = 16352` 线程；③按元素级铺满核（约 105 万个更新）
- 每 warp 要跑 `2048/32 = 64` chunk × 15 步 bitonic = **960 步排序**
- 省掉的 fill-zero（8.51us）+ workspace 往返远不足以补偿

### 5.4 ⑤ 放宽到 8 字节值：安全但无收益证据

改动本身安全（排序全基于 uint32 键，值类型只出现在最后一句赋值），但实测零收益（噪声内），
且 §5.3 那条 int64 用例的数据反而指向 warp 对 int64 不利。**要启用先补"int64 + 大纤维数"的实测。**

### 5.5 `multiply` 必须留在 ④，不要因为"CAS 可能活锁"排除它

曾在 `useAtomicEP` 上多挂 `&& red != MUL`，让 multiply 一律退回①，
结果某 fp32 multiply 用例（16.7M 元素）**从 3767us 劣化到 12602us**。
随机索引下同一输出位置平均只被更新 ~1 次，竞争很轻，CAS 循环几乎总是一次成功。

反过来也成立：**冲突密集时 CAS 会退化**，那时①才对。**判据应该是冲突度，不是"是不是 multiply"**——
当前实现用的是 `starved`（间接相关），如果遇到"更新数远大于输出位置数"的形状，这里值得补一条真正的冲突度判据。

### 5.6 判据速查（算 `useWarpLast` 前先看这张表）

| numFibers | jBits+idxBits | 结论 |
|---|---|---|
| 2048 | 21 ✓ | 走 ⑤ |
| 511 | 24 ✓ | 卡纤维数门槛 → ③（**已验证这样更快**） |
| 1 | 32 ✗ | 单纤维且键装不下 31 位 → ③ |

---

## 6. bound 类型怎么标（本族与通用口径的偏差）

按带宽口径标 `MEM BOUND` 会误导后续策略：**散射是访存延迟受限**，顺序流量能跑到 ~600 GB/s，
而散射侧只有几 G updates/s，远未打满带宽。建议标注为 **MEM BOUND（latency，非 bandwidth）**
并同时给出"总线程数"这一实际制约量，否则会被引向"减少搬运量"这类无效方向。

---

## 7. 已知的 Tiling 侧空白

| 空白 | 现状 | 备注 |
|---|---|---|
| ⑥ 通用路径完全未调优 | 索引一律 int64、每元素逐维 `%` + `/`、固定 1024 线程 | 可按 rank 编译期特化消掉运行期循环。生产/隐藏用例走这条的比例可能远高于公开用例 |
| 按 `ix` 分块改善写局部性 | 未实现 | dim 维切 N 块、每轮只处理落在本块的更新，写局部性改善 N 倍，代价是 indices 扫 N 遍的读放大。只对 "`sVar` 小 & `inner` 大" 划算，**需要新的选路判据**，属算法重构不是调参 |
| 冲突度判据 | 用 `starved` 间接近似 | 见 §5.5 |
| 逐用例挑最优线程数 | 未做 | 1024 净收益更好但不均匀（个别用例退化 8~24%）；按用例挑最优收益很小**且没有正当的形状判据** |
