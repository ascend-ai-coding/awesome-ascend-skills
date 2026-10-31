# 实现①：纤维独占（零原子，天然后写覆盖）

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> 总览与选型见 [scatter_design.md](scatter_design.md)。本文只讲①。
> 一句话：**利用"冲突只发生在纤维内"这条不变量，把线程绑到纤维上，冲突就自动消失了。**

## 1. 适用条件

| 条件 | 值 |
|---|---|
| 并行宽度 | `numFibers ≥ coreNum × 1024`（即 `!starved`）——铺得满核 |
| reduce | 全部（`none/add/multiply/amin/amax`） |
| 张量形状 | 三张量非 dim 维范围全等（`extentsMatch`） |
| 相对其它路径的优势 | **不用原子、不用 workspace、不用额外 pass**，散射只走一趟普通存储 |

**只要铺得满核就用它，不要考虑别的。** 元素级并行（②③④）对满核 shape 来说只是多付一趟仲裁。
`multiply` 是个例外：它在元素级只能用 CAS 循环，规模大时 CAS 反而更快（实测 12602us → 3767us，见
[element_parallel_design.md](element_parallel_design.md#5-multiply用-cas-循环补原子乘)），所以
`multiply` 也参与 `starved` 判定，不要无条件留在①。

## 2. 并行划分

```
线程 f  ←→  第 f 条纤维（非 dim 坐标固定）
f -> (o, c)：  o = f / inner,  c = f - o * inner
线程内沿 dim 递增遍历 j = 0 .. sIdx-1
```

三个基址一次算好，循环内只做一次乘加：

```cpp
const COMP_T baseIdx = o * cSIdx * cInner + c;   // indices
const COMP_T baseUpd = baseIdx;                  // 快路径约定 updates.shape == indices.shape
const COMP_T baseVar = o * cSVar * cInner + c;   // 输出 y
```

**为什么这样绑线程号**：`tid` 落在最内坐标 `c` 上 ⇒ 相邻线程读 `indices[baseIdx + j*inner]`、
`updates[...]` 时地址连续 ⇒ **读侧是完全合并访问**。不规则只体现在写侧。
反过来把 `tid` 绑到 `j` 上就全毁了。

## 3. 两条正确性保证（都是免费的）

1. **免原子**：线程 `f` 的写集合是 `{baseVar + ix*inner | ix ∈ [0,sVar)}`，只依赖 `(o,c)`。
   不同线程 `(o,c)` 不同 ⇒ 写集合不相交 ⇒ **无论 index 怎么重复都不会跨线程冲突**。
2. **后写覆盖**：线程内 `j` 严格递增、串行执行，`reduce=none` 时后面的 `y[dst] = v` 自然覆盖前面的，
   与 CPU golden 的遍历顺序逐位一致。**不需要任何仲裁**。

## 4. 核心循环

完整可拷贝版本见 [code/scatter_simt_common.h](code/scatter_simt_common.h) 的 `ScatterFiberVF`。要点：

```cpp
template <typename ACC_T, typename UPD_T, typename IDX, typename COMP_T, int32_t RED, uint32_t THREADS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void ScatterFiberVF(
    __gm__ ACC_T* y, __gm__ IDX* indices, __gm__ UPD_T* updates,
    int64_t numFibers, int64_t inner, int64_t sIdx, int64_t sUpd, int64_t sVar)
{
    const COMP_T tid    = static_cast<COMP_T>(blockIdx.x) * blockDim.x + threadIdx.x;
    const COMP_T stride = static_cast<COMP_T>(gridDim.x) * blockDim.x;
    /* ... cInner / cSIdx / cSVar 收窄 ... */

    for (COMP_T f = tid; f < cNumFibers; f += stride) {
        const COMP_T o = f / cInner;
        const COMP_T c = f - o * cInner;
        const COMP_T baseIdx = o * cSIdx * cInner + c;
        const COMP_T baseVar = o * cSVar * cInner + c;

        for (COMP_T j = 0; j < cSIdx; ++j) {
            COMP_T ix = static_cast<COMP_T>(indices[baseIdx + j * cInner]);
            if (ix < 0) ix += cSVar;                 // torch 容忍负索引
            if (ix < 0 || ix >= cSVar) continue;     // 非法索引跳过，不越界写
            const COMP_T dst = baseVar + ix * cInner;
            const ACC_T  v   = ScatterConvert<ACC_T, UPD_T>(updates[baseIdx + j * cInner]);
            if constexpr (RED == RED_NONE) y[dst] = v;              // 后写覆盖，普通存储
            else                           y[dst] = ScatterCombine<RED, ACC_T>(y[dst], v);
        }
    }
}
```

三个模板参数各自的作用：

| 参数 | 作用 |
|---|---|
| `RED` | reduce 语义提到编译期，消除 VF 内的运行期分支（见 [simt/optimization-guide.md](../simt/optimization-guide.md) Rule 001） |
| `COMP_T` | 索引运算宽度。所有线性下标都 `< 2^31` 时用 `int32_t`。**实测本族零收益**，但保留（对寄存器压力大的 VF 仍有意义） |
| `ACC_T` / `UPD_T` | 输出/累加器类型与 updates 类型分离。二者相同即普通路径；fp16/bf16 的累积型 reduce 走 `ACC_T=float` |

## 5. 线程数与 block 数（真正的调优点）

**并行宽度被 `numFibers` 卡死**，固定 512 线程/block 会让纤维数小的 shape 只拿到 1~2 个核
（实测 `numFibers ≈ 1024` 时加速比 0.02×）。策略：

> **在能铺满核的前提下取最大的线程数；铺不满就取最小的。**
> 总线程数不变，但访存发射摊到更多核上。

```cpp
constexpr int32_t CANDIDATES[] = {1024, 512, 128, 32, 8};
int32_t chosen = 8;
for (int32_t t : CANDIDATES) {
    if ((numFibers + t - 1) / t >= coreNum) { chosen = t; break; }   // 能铺满核的最大线程数
}
int64_t blocks = std::min(coreNum, (numFibers + chosen - 1) / chosen);
```

`LAUNCH_BOUND` 与 `dim3` 都要求**编译期常量**，所以核内要按候选值展开一个 switch：

```cpp
switch (threads) {
    case 1024: SCATTER_RED_SWITCH(1024, CTYPE); break;
    case 512:  SCATTER_RED_SWITCH(512,  CTYPE); break;
    case 128:  SCATTER_RED_SWITCH(128,  CTYPE); break;
    case 32:   SCATTER_RED_SWITCH(32,   CTYPE); break;
    default:   SCATTER_RED_SWITCH(8,    CTYPE); break;
}
```

> **别把 reduce 也做成模板实参传到 host**：`dtype × idx 宽度 × reduce × 线程数` 会爆出几十份 kernel 二进制。
> 正确做法是 reduce 作为**运行期标量**在核内 switch 分发到 `if constexpr` 特化的 VF —— 二进制数量可控，
> VF 内部仍然无分支。

## 6. 阶段一：`y = data` 的整块拷贝要用 SIMD

散射只写被 index 命中的位置，其余位置必须保留 `data` 的原值，所以要先整块 clone。
**这一步是规则的顺序流量，必须走 SIMD 的 MTE 搬运**，不要用 SIMT 标量循环。

```cpp
template <typename U>
__global__ __aicore__ __vector__ void scatter_copy_kernel(GM_ADDR src, GM_ADDR dst, int64_t totalLength,
                                                          int64_t blockLength, uint32_t tileSize)
{
    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> queue;
    pipe.InitBuffer(queue, 2, tileSize);
    /* 每核连续分片 + DataCopyPad 双缓冲搬运，见 code/scatter_host_routing.cpp 的 tiling */
}
```

两个工程简化：

- **按定宽整型实例化，不按 dtype**：`U = uint16_t`（2 字节）或 `uint32_t`（4 字节），
  int64 按 2× 元素数走 4 字节版本。5 种 dtype 只需两份实例。
- **tile 大小 512B 对齐**：`tileSize = (ubSize / depth / bufNum / 512) * 512`，避免 `DataCopyPad` 起址/长度不对齐。

## 7. 已知短板

| 短板 | 表现 | 出路 |
|---|---|---|
| 纤维数少时严重饥饿 | 769 条纤维就只有 769 个线程 | 交给②③④⑤（`starved` 判据） |
| 随机 index 的写放大 | 一个 warp 内 32 线程 `c` 连续、`ix` 各异 → 32 条不同 cacheline，每条只用 2 字节。实测这一项就占 **4.81×** | **换路径救不了**（②③写同一个地址式子）。唯一出路是按 `ix` 分块：dim 维切 N 块，每轮只处理落在本块的更新，写局部性改善 N 倍，代价是 indices 扫 N 遍的读放大。只对 "`sVar` 小 & `inner` 大" 划算，需新的选路判据 —— **属算法重构，不是调参** |
