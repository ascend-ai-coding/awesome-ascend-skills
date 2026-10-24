# 实现⑤：warp 段末写回（单趟、确定性、免 workspace）

> 总览与选型见 [scatter_design.md](scatter_design.md)。
> 一句话：**一条 warp 独占一条纤维，用 warp 内 bitonic 排序把同 index 的更新聚到一起，只让段末那个 lane 写。**
> 这是 `reduce=none` 里唯一**既单趟、又不用原子、又不用 workspace** 的实现。

## 1. 适用条件（四条全中才用）

```cpp
const bool useWarpLast = (red == SCATTER_RED_NONE) && needScatter && extentsMatch && starved
                      && warpKeyFits32          // jBits + idxBits ≤ 31
                      && esize == 4             // 4 字节值
                      && numFibers >= 1024;     // 纤维数够多
```

| 条件 | 为什么 |
|---|---|
| `jBits + idxBits ≤ 31` | `(index, j)` 要压进一个 32 位键，且要留 `0xffffffff` 作无效 lane 哨兵 |
| `esize == 4` | 2 字节值走②更快（打包键吞吐更高）；8 字节**实测无收益证据**，见下方 §5 |
| `numFibers ≥ 1024` | **这条门槛是实测定的，别往下调**（§5） |

## 2. 原理

```
warp（32 lane）独占纤维 f，每轮并行读连续 32 个 j
      ↓
每 lane 把自己的 (index, j) 压成 32 位键： key = (safeIx << jBits) | j
      ↓
warp 内 15 级 bitonic sort（asc_shfl_xor 交换）→ 键按 index 升序，同 index 内按 j 升序
      ↓
用 asc_shfl_down 看右邻居：右邻 index 不同（或自己是 lane 31）⇒ 我是本 index 的段末
      ↓
段末 lane 写回 y[baseVar + sortedIx*inner] = updates[baseUpd + winnerJ*inner]
```

**为什么段末就是胜者**：键的高位是 `index`、低位是 `j`，升序排完后同一 `index` 的一段里 `j` 递增，
段末即该轮内 `j` 最大者 = 后写覆盖。

**跨轮怎么办**：各轮按 `j` 递增执行（`chunk += 32`），轮间做一次可见性同步，所以后轮天然覆盖前轮 ——
**严格等价于 CPU 的后写覆盖**，不需要任何全局仲裁。

## 3. 关键代码

完整版见 [code/scatter_warp_last.h](code/scatter_warp_last.h)。三处要点：

### 3.1 无效 lane 用哨兵而不是分支

```cpp
uint32_t safeIx = 0xffffffffu;                       // 越界 / 超出 sIdx 的 lane
if (activeFiber && originalJ < sIdx) {
    int64_t ix = static_cast<int64_t>(indices[baseIdx + originalJ * inner]);
    if (ix < 0) ix += sVar;
    if (ix >= 0 && ix < sVar) safeIx = static_cast<uint32_t>(ix);
}
uint32_t key = (safeIx != 0xffffffffu) ? ((safeIx << jBits) | uint32_t(originalJ)) : 0xffffffffu;
```

无效 lane 的键是最大值，排序后全被推到高位端，写回时 `sortedIx < sVar` 一判就滤掉。
**warp 内所有 lane 走同一条指令路径**，没有 divergence。

### 3.2 15 级 bitonic 网络（完全展开）

```cpp
#define SCATTER_BITONIC_STEP(MASK, WIDTH)                                   \
    do {                                                                     \
        const uint32_t other = asc_shfl_xor(key, MASK, 32);                 \
        const bool ascending  = (lane & WIDTH) == 0;                         \
        const bool lowPartner = (lane & MASK)  == 0;                         \
        const bool wantMin    = lowPartner == ascending;                     \
        if ((wantMin && key > other) || (!wantMin && key < other)) key = other; \
    } while (0)

SCATTER_BITONIC_STEP(1,2);
SCATTER_BITONIC_STEP(2,4);  SCATTER_BITONIC_STEP(1,4);
SCATTER_BITONIC_STEP(4,8);  SCATTER_BITONIC_STEP(2,8);  SCATTER_BITONIC_STEP(1,8);
SCATTER_BITONIC_STEP(8,16); SCATTER_BITONIC_STEP(4,16); SCATTER_BITONIC_STEP(2,16); SCATTER_BITONIC_STEP(1,16);
SCATTER_BITONIC_STEP(16,32);SCATTER_BITONIC_STEP(8,32); SCATTER_BITONIC_STEP(4,32); SCATTER_BITONIC_STEP(2,32); SCATTER_BITONIC_STEP(1,32);
```

`log₂32 = 5` 个阶段，第 k 阶段 k 步，共 `1+2+3+4+5 = 15` 步。**必须展开成直线代码**——
`MASK`/`WIDTH` 是编译期常量才能让 `asc_shfl_xor` 生成最短序列。

### 3.3 段末判定与轮间同步

```cpp
const uint32_t sortedIx = key >> jBits;
const uint32_t nextIx   = asc_shfl_down(key, 1, 32) >> jBits;
const bool groupEnd = (lane == 31) || (sortedIx != nextIx);
if (activeFiber && groupEnd && sortedIx < uint32_t(sVar)) {
    const uint32_t winnerJ = key & ((uint32_t{1} << jBits) - 1);
    y[baseVar + int64_t(sortedIx) * inner] = updates[baseUpd + int64_t(winnerJ) * inner];
}
asc_threadfence_block();     // ← 只需 block 级
```

> **同步粒度**：同一个输出位置在相邻 chunk 中可能由不同 lane 写，所以轮间要同步；
> 但 warp 内本就锁步执行，**只需保证本 warp 的 GM 写可见**（`asc_threadfence_block`），
> 不需要让 1024 个线程互等。不同 warp 处理不同纤维，天然无交叉。

## 4. 相比其它 `none` 路径的账

| | ⑤ warp 段末 | ② 打包键 | ③ 两趟胜者 |
|---|---|---|---|
| 散射趟数 | **1** | 1 | 2 |
| 原子 | **无** | 32 位 | 32 位 |
| workspace | **无** | `int32[numel]` | `int32[numel]` |
| 额外顺序 pass | 无 | init + unpack | fill-zero |
| 每更新的额外指令 | **15 步排序 + 2 次 shfl** | 1 次 atomic | 1 次 atomic + 1 次比对 |

省下的是内存流量，付出的是排序指令。**所以纤维数必须够多**：纤维少时 warp 并行度上不去，
排序的固定开销直接暴露。

## 5. 两条实测边界（都别再重新推导）

### 门槛 1024 不能往下调

放宽到 256 后，某 `int64 none` 用例（**511 纤维、`sIdx=2048`**）从③的 133.67us 劣化到⑤的 **295.94us**：

- 511 条纤维只能铺 `511 × 32 = 16352` 线程；③按元素级并行铺满核（约 105 万个更新）
- 每 warp 要跑 `2048/32 = 64` 个 chunk × 15 步 bitonic = **960 步排序**
- 省掉的 fill-zero（8.51us）+ workspace 往返远不足以补偿

### 放宽到 8 字节值：安全但无收益证据

改动本身是安全的——`ScatterWarpLastVF` 的 `T` 只出现在最后一句赋值，**排序全基于 uint32 键**，与值宽无关。
但实测标准集零收益（噪声内），且上面那条 int64 用例的数据反而指向 warp 对 int64 不利。
**要启用先补"int64 + 大纤维数"的实测**，不要凭"看起来能支持"就放开。

### 判据速查

| numFibers | jBits+idxBits | 结论 |
|---|---|---|
| 2048 | 21 ✓ | 走⑤ |
| 511 | 24 ✓ | 卡纤维数门槛 → ③（**已验证这样更快**） |
| 1 | 32 ✗ | 单纤维且键装不下 31 位 → ③ |
