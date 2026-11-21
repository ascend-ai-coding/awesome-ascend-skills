# reduction 范式 binary-base Tiling 切分策略

> UB 切分、多核切分、Workspace 分配的完整算法

依赖输入：[reduction-template-overview.md](../reduction-template-overview.md)（§1.1 术语表）、[reduction-binary-base-dag-buffers.md](reduction-binary-base-dag-buffers.md)、[reduction-tiling-preprocess.md](../reduction-tiling-preprocess.md)

> 所有 reduce 算子（sum/mean/max/min/prod/any/all）统一使用二分缓存树，UB 切分三步算法、多核切分、UB 预算不等式、Buffer 大小公式完全通用。

## 0 速查

通用的 tiling 切分都是分三个阶段，每段解决一个核心问题。reduction范式可以归纳如下:

| 步骤 | 核心问题 | 产出 | 对应章节 |
|------|---------|------|---------|
| ① 找 UB 切分轴 | A/R 各选哪个轴切入 UB | aSplitIdx / rSplitIdx | §1.1-1.2 Step 1-2 |
| ② 定切分因子 | 每个切分轴切多大 | aUbFactor / rUbFactor | §1.2 Step 1-2 |
| ③ 多核切分 | 外层循环怎么分给多核 | aLoopCntTotal / aSplitChunkCnt / 大小核 | §2 |


| 概念 | 一句话 |
|------|--------|
| **UB 双切分** | A、R 各选一个切分轴（`aSplitIdx`/`rSplitIdx`），**互相独立** |
| **轴归属** | 切分点**左侧**轴 + 切分轴 `.o` → **循环**；切分轴 `.i` + **右侧**轴 → **驻 UB** |
| **三步走** | ① 先定 A 切分 → ② 再定 R 切分 → ③ 若所有 R 全驻 UB，再调整 A 切分 |
| **UB 预算** | `(P_pre+P_pre_ext) × preBufSize + P_post × postBufSize ≤ ubAvailable` |
| **ubAvailable** | `ubSize − 16KB`（先扣固定 cacheBuf） |
| **整轴 = 切多 chunk 的特例** | 整轴即 `factor = axisSize`，`.o=1`；切多 chunk 即 `factor < axisSize` |

## 1 UB 切分

> **前置依赖**：
> - 合轴后的 axisShape / axisNum（A/R 严格交替、A 起头），来自 [reduction-tiling-preprocess.md](../reduction-tiling-preprocess.md) §2
> - isTailR / isTailA（axisNum 偶数→tailR，奇数→tailA）
> - buffer 份数 P_pre / P_pre_ext / P_post，来自 [dag-buffers.md](reduction-binary-base-dag-buffers.md) §1.5
> - 平台信息（coreNum, ubSize, blockSize, cacheLineSize）
> - 空 tensor 已在预处理阶段短路，本节只处理非空 tensor
> - UB buffer 跨 dtype 复用：搬入 buffer（D_T）和计算 buffer（fp32）生命周期不重叠时可复用同一物理 buffer，所有 UB buffer 统一按 `maxDtypeSize = max(sizeof(D_T), sizeof(float))` 分配容量，来自 [dag-buffers.md](reduction-binary-base-dag-buffers.md) §1.1.1

### 1.1 切分原理

UB 双切分（A 轴与 R 轴各有 UB 切分点）：

- **R 必须切**：UB 内必须做 reduce，必须把 R 搬到 UB
- **A 必须切**：tail-A 下 R 在 GM 非连续排布，A 切入 UB 才能保证搬运连续性
- **统一 AR/RA 用同一套双切分框架**

(1) 选 A 切分轴 `aSplitIdx`：假设切分因子为 a.i，令 a.o=ceil(a, a.i)，则 `a.o 和左侧（外层）的所有A轴` 做 for 循环；`a.i 和右侧（内层）的所有A轴` **全部驻 UB**

(2) 选 R 切分轴 `rSplitIdx`：假设切分因子为 r.i，令 r.o=ceil(r, r.i)，则 `r.o 和左侧（外层）的所有R轴` 做 for 循环；`r.i 和右侧（内层）的所有R轴` **全部驻 UB**

(3) 两个切分点**互相独立**，可以一个偏外、一个偏内

(4) **切分轴整轴与轴切分无区别**，整轴即 a.i = a 或 r.i = r，只是切分因子不同罢了

**总体顺序**：(1) 先定 A 切分 → (2) 再定 R 切分 → (3) 若所有 R 全驻 UB，再调整 A 切分。

### 1.2 切分公式

#### Step 1：定 `aSplitIdx` 与 `aUbFactor`（`ComputeAUbFactor()`）

**原理**：

采用联合爬坡策略——A 轴与 R 轴一起从最内轴向外爬坡，目标上界为 `cachelineTmp`（一个 cache line 的元素数）。爬坡停止后，按停止轴的类型（A/R）决定切分点与 `aUbFactor`，再单独统计 `innerAProdAlign`。

> **R 轴在爬坡中的角色**：联合爬坡中 R 轴参与 `product` 累乘（用于估算 UB 内 A+R 总元素量是否超 `cachelineTmp`），但 R 轴本身不进 `innerAProdAlign`——`innerAProdAlign` 只统计 A 轴。

**(1) 计算 `cachelineTmp`（UB 内目标元素数上界）**：

```cpp
// cacheLineSize 走 GetCacheLineSize() 接口（不写死）：常见 arch35 为 256B
//   → fp32 → 64 个元素，fp16/bf16 → 128 个元素
const int64_t bsElem = blockSize / sizeof(D_T); //  ⛔ 是 sizeof(D_T)，按实际数据类型计算
cachelineTmp = FloorAlign(cacheLineSize / sizeof(D_T), bsElem);   // block 对齐
```

**(2) 联合爬坡**：

从最内轴（`axisNum-1`）向外逐根吸入 `product`。尾轴无论 A 还是 R 一律 `CeilAlign` 到 block 对齐（burst 尾轴方向），非尾轴按原值累乘。终止于 `product × axisSize > cachelineTmp`（用 `>` 而非 `>=`：tail-A 下当 CeilAlign(LastA) == cachelineTmp 时仍吸入 LastA，保证 innerAProdAlign 含 CeilAlign(LastA) 因子，从而使 aUnit 一定对齐；若改 `>=` 会停在 LastA，aUbFactor=min(cachelineTmp, axisShape[LastA]) 可能取非对齐原值，aUnit 非对齐）：

```cpp
product = 1;
int32_t idx = axisNum - 1;
while (idx >= 0) {
    int64_t axisSize = (idx == axisNum - 1)
        ? CeilAlign(axisShape[idx], bsElem)   // 尾轴 burst 方向需 block 对齐
        : axisShape[idx];
    if (product * axisSize > cachelineTmp) {
        break;
    }
    product *= axisSize;
    idx--;
}
```

**(3) 切分点判定与 `aUbFactor`**（`idx` 是爬坡停止时的轴下标）：

| `idx` 的情况 | `aSplitIdx` | `aUbFactor` | 说明 |
|--------------|-------------|-------------|------|
| `idx < 0`（所有轴全驻） | 0 | `axisShape[0]` | 所有轴装入仍未超 `cachelineTmp`，无轴可切 |
| `idx` 偶数（A 轴） | `idx` | `min(cachelineTmp / product, axisShape[idx])` | 直接切该 A 轴，右侧所有轴驻 UB |
| `idx` 奇数（R 轴） | `idx − 1` | 1 | R 轴不能作为 A 切分点，左移到其左侧第一根 A（A 起头+严格交替，R 左侧必为 A），每 chunk 取 1 |

```cpp
if (idx < 0) {
    aSplitIdx = 0;
    aUbFactor = axisShape[0];
} else if (IsAAxis(idx)) {
    aSplitIdx = idx;
    aUbFactor = cachelineTmp / product;        // 向下取整
    aUbFactor = min(aUbFactor, axisShape[aSplitIdx]);
} else {
    aSplitIdx = idx - 1;
    aUbFactor = 1;
}
```

**(4) 计算 `innerAProdAlign`**：

`innerAProdAlign` 只统计 `aSplitIdx` 右侧的 A 轴（不含切分轴本身），tail-A 时最内 A 需 CeilAlign：

```cpp
innerAProdAlign = 1;
for (int32_t k = aSplitIdx + 2; k < axisNum; k += 2) {  // A 起头+严格交替，偶数下标=A
    if (k == axisNum - 1 && isTailA) {
        innerAProdAlign *= CeilAlign(axisShape[k], bsElem);   // 尾轴 A burst 方向 block 对齐
    } else {
        innerAProdAlign *= axisShape[k];
    }
}
```

**定义 `aUnit`**：`aUnit = aUbFactor × innerAProdAlign`（UB 内 A 维元素数, tail-A 时是包含了 pad 后的元素数）。tail-A 下 aUnit 一定 block 对齐；tail-R 下 aUnit 可能非 block 对齐

**约束**：
- `aUbFactor` 不保证 block 对齐
- 不需要 `aUbFactorAlign`：tail-A 下且 aSplitIdx==LastA 时 aUbFactor 可以保证已对齐; tail-A 下但 aSplitIdx!=LastA 时, 不需要 aUbFactorAlign 参数，innerAProdAlign 可保证对齐; tail-R 下更不需要 aUbFactorAlign 参数。三种情况下直接使用 aUbFactor 。

> ⛔ **dtype 参数适用边界（强制）**：范式有两个 dtype 派生参数，适用场景不同，**禁止混用**
>
> | 参数 | 定义 | 适用场景 | 理由 |
> |------|------|----------|------|
> | `bsElem = blockSize / sizeof(D_T)` | block 的 D_T 元素数 | **元素数取整** | 搬运/对齐针对输入数据，burst 按 D_T 计 |
> | `maxDtypeSize = max(sizeof(D_T), sizeof(float))` | 每元素最大字节宽 | **字节容量** | buffer 跨 dtype 复用，按最宽分配 |

#### Step 2：定 `rSplitIdx` 与 `rUbFactor`（`ComputeRUbFactor()`）

**原理**：

分四步：
1. **先算 `postBufSize`**：post 阶段单份 buffer 字节数，`CeilAlign(aUnit × maxDtypeSize, blockSize)`。post buffer 是 1D 的，必须 blockSize 对齐
2. **UB 预算反解 `r_i_max`**：`ubAvailable` 先扣固定 16KB cacheBuf，再减 post 阶段独占字节（按 `postBufSize` 计），剩余除以"每 R 元素的 pre 阶段字节开销"得 `r_i_max`。每增加 1 个 R 元素，pre 阶段增加 `(P_pre + P_pre_ext) × aUnit × maxDtypeSize` 字节（pre buffer 是 2D 布局：tail-R 下每行多 maxDtypeSize、共 aUnit 行；tail-A 下多 1 行、每行 aUnit × maxDtypeSize；两种场景统一）。buffer 份数（`P_pre + P_pre_ext` / `P_post`）从 [dag-buffers.md](reduction-binary-base-dag-buffers.md) §1.5 传入
3. **R 端 Climb**：从最内 R 轴向外逐根吸入，累乘进 `innerRProdAlign`，直到超预算
4. **定 `rUbFactor` 与 `rUbFactorAlign`**：tail-R + rSplitIdx==LastR 时（burst 尾轴方向），切多 chunk 需 FloorAlign，整轴需 CeilAlign 并校验不超预算；其他情况不需对齐，rUbFactorAlign=rUbFactor

**2a：先算 `postBufSize`，再 UB 预算反解 `r.i_max`**：

```
maxDtypeSize  = max(sizeof(D_T), sizeof(float))
aUnit         = aUbFactor × innerAProdAlign
postBufSize   = CeilAlign(aUnit × maxDtypeSize, blockSize)        // post buffer 1D，block 对齐
aOnlyBytes    = P_post × postBufSize                               // post 阶段不含 R 维
bytesPerRElem = (P_pre + P_pre_ext) × aUnit × maxDtypeSize         // 每增加 1 个 R 元素，pre 阶段增加的字节
r_i_max       = (ubAvailable − aOnlyBytes) / bytesPerRElem          // 向下取整
```

其中 `ubAvailable = ubSize − cacheBufUbSize`（先扣固定 16KB）。

> **⚠ `aOnlyBytes` 用 `postBufSize`，`bytesPerRElem` 用 `aUnit × maxDtypeSize`**：post buffer 是 1D 的，必须 block 对齐（CeilAlign）；pre buffer 是 2D 的，burst 尾轴方向已由 rUbFactorAlign/innerAProdAlign 保证对齐，非尾轴方向不对齐，每 R 元素的开销按 `aUnit × maxDtypeSize` 算即可，无需额外 CeilAlign。tail-A 下 aUnit 一定对齐（见 §1.2 Step 1），`aUnit × maxDtypeSize` 自然 block 对齐；tail-R 下 aUnit 是行数（不对齐），每增加 1 个 R 元素 → aUnit 行 × maxDtypeSize。

**2b：R 端 Climb**：

从最内 R 轴向外逐根吸入 `innerRProdAlign`。tailR 时尾轴 R 先 CeilAlign 到 block 对齐（burst 尾轴方向），再与 r_i_max 比较；非尾轴按原值。终止于 `axisSize × innerRProdAlign > r_i_max`：

```cpp
innerRProdAlign = 1;
rSplitIdx = LastRAxisIdx();                              // 最大奇数下标
while (rSplitIdx > 1) {
    int64_t axisSize = (rSplitIdx == LastRAxisIdx() && isTailR)
        ? CeilAlign(axisShape[rSplitIdx], blockSize / sizeof(D_T))   // 尾轴 R 先对齐再比较
        : axisShape[rSplitIdx];
    if (axisSize * innerRProdAlign > r_i_max) {
        break;                // 吸入后会超预算则停
    }
    innerRProdAlign *= axisSize;
    rSplitIdx -= 2;  // 向外移动一根 R 轴（跳过中间的 A 轴）
}
```

**2c：`rUbFactor` 与 `rUbFactorAlign`**：

```cpp
rUbFactor = r_i_max / innerRProdAlign;        // 向下取整
rUbFactor = min(rUbFactor, axisShape[rSplitIdx]);

bool isBurstTailR = (isTailR && rSplitIdx == LastRAxisIdx());

if (isBurstTailR) {
    if (rUbFactor < axisShape[rSplitIdx]) {
        // 切多 chunk：burst 尾轴方向需 block 对齐
        rUbFactor = FloorAlign(rUbFactor, blockSize / sizeof(D_T));
        if (rUbFactor == 0) {
            return TILING_FAIL;   // 连一个 block 都装不下
        }
        rUbFactorAlign = rUbFactor;
    } else {
        // 整轴：CeilAlign 后校验不超预算
        rUbFactorAlign = CeilAlign(rUbFactor, blockSize / sizeof(D_T));
        if (rUbFactorAlign * innerRProdAlign > r_i_max) {
            // 放大后超预算，退回切多 chunk
            rUbFactor = FloorAlign(rUbFactor, blockSize / sizeof(D_T));
            if (rUbFactor == 0) {
                return TILING_FAIL;
            }
            rUbFactorAlign = rUbFactor;
        }
    }
} else {
    // 非 burst 尾轴方向，不需要对齐
    rUbFactorAlign = rUbFactor;
}
```

#### Step 3：所有 R 全驻后扩 A（`ExpandAIfRFullyLoaded()`）

**原理**：

若 `rUbFactor == axisShape[rSplitIdx]` 且 rSplitIdx 已在最外 R（所有 R 全驻 UB），回头扩 A 以填满 UB——用新的 `aUnitMax` 作为爬坡目标上界，从最内 A 轴向外爬坡，切分点可能向外移动，吸入更多 A 轴进 `innerAProdAlign`，从而增大 `aUnit` 回收剩余空间。

分三步：
1. **解 `aUnitMax`**：将 R 端固定占用代回 UB 预算不等式，对 aUnitMax 提公因式，解出新的预算上限 `aUnitMax`。tail-A 时 FloorAlign 到 block 对齐，tail-R 时不 FloorAlign。UB 预算公式：
   `(P_pre+P_pre_ext) × aUnitMax × rUbFactorAlign × innerRProdAlign × maxDtypeSize + P_post × aUnitMax × maxDtypeSize ≤ ubAvailable`
2. **钳制 cacheBuf**：所有 reduce 算子 R 全驻时 cacheCount=1，aUnitMax 不能超过 cacheBuf 容量（16KB/4B=4096 lane），否则二分树溢出，必须显式钳制
3. **重做 A 端爬坡**：用 `aUnitMax` 作为目标上界，从最内 A 轴向外爬坡（只爬 A 轴，R 已全驻不参与），切分点可能外移


```cpp
if (rUbFactor == axisShape[rSplitIdx] && 没有更外 R 轴):
    // R 端总占用已固定，UB 剩余空间全给 A
    // postBufSize = aUnitMax × maxDtypeSize
    //   （tail-A 下 aUnitMax 已 FloorAlign，无需 CeilAlign；tail-R 下 aUnit 是行数不需对齐，
    //     postBufSize 仍按 CeilAlign 算但预算公式中用 aUnitMax × maxDtypeSize 近似）
    // UB 预算: (P_pre+P_pre_ext) × aUnitMax × rPaddedElems × maxDtypeSize + P_post × aUnitMax × maxDtypeSize
    //        ≤ ubAvailable
    //        = aUnitMax × maxDtypeSize × ((P_pre+P_pre_ext) × rPaddedElems + P_post) ≤ ubAvailable
    // aUnitMax ≤ ubAvailable / (maxDtypeSize × ((P_pre+P_pre_ext) × rPaddedElems + P_post))
    int64_t rPaddedElems = rUbFactorAlign * innerRProdAlign;
    aUnitMax = ubAvailable / (maxDtypeSize * ((P_pre + P_pre_ext) * rPaddedElems + P_post));
    aUnitMax = min(aUnitMax, ∏ 所有 A 轴 size);   // 不超过所有 A 轴总乘积
    // ★ cacheBuf 硬约束：所有 R 全驻即 cacheCount=1，
    //   aUnitMax ≤ cacheBufUbSize / sizeof(float)。不钳制会 cacheBuf 溢出
    aUnitMax = min(aUnitMax, cacheBufUbSize / sizeof(float));
    if (isTailA) {
        // ⚠ 必须在 min(∏A) 钳制之后 FloorAlign：∏A 可能非对齐，aUnitMax 就可能非对齐
        //   爬坡停在尾轴（aSplitIdx==LastA）时
        //   aUbFactor = min(aUnitMax, LastA) 会取到非对齐值 → UB 行 stride 非 32B 对齐
        aUnitMax = FloorAlign(aUnitMax, blockSize / sizeof(D_T));   // tail-A 下 aUnit 是 burst 尾轴方向，需 block 对齐
    }

    // aUnitMax 可能 ≤ 当前 aUnit（R 全驻后 UB 剩余空间不足），此时扩 A 无效，
    // 下方 if 守卫会跳过，保持 Step 1 的 aUbFactor 不变
    if (aUnitMax > 当前 aUbFactor × innerAProdAlign):
        // 重做 A 端爬坡，切分点可能外移
        // 只爬 A 轴（R 已全驻，不参与 product）
        product = 1;
        int32_t idx = LastAAxisIdx();
        while (idx >= 0) {
            int64_t axisSize = (idx == axisNum - 1 && isTailA)
                ? CeilAlign(axisShape[idx], bsElem)   // tail-A 尾轴 burst 方向需 block 对齐
                : axisShape[idx];
            if (product * axisSize > aUnitMax) {
                break;
            }
            product *= axisSize;
            idx -= 2;  // 向外移动一根 A 轴（跳过中间的 R 轴）
        }
        // 切分点判定
        if (idx < 0) {
            aSplitIdx = 0;
            aUbFactor = axisShape[0];
        } else {
            aSplitIdx = idx;
            aUbFactor = aUnitMax / product;     // 向下取整
            aUbFactor = min(aUbFactor, axisShape[aSplitIdx]);
        }
        // 重算 innerAProdAlign（同 Step 1 (4)）
        innerAProdAlign = 1;
        for (int32_t k = aSplitIdx + 2; k < axisNum; k += 2) {
            if (k == axisNum - 1 && isTailA) {
                innerAProdAlign *= CeilAlign(axisShape[k], bsElem);
            } else {
                innerAProdAlign *= axisShape[k];
            }
        }
```

> **与 Step 1 的区别**：Step 3 用 `aUnitMax`（所有 R 全驻后反解的 A 预算上限）替换 `cachelineTmp` 作为爬坡目标上界，只爬 A 轴（R 已全驻不参与）。若 `aUnitMax > 当前 aUnit`，爬坡能吸入更多 A 轴，切分点可能外移。

> **扩 A 后 UB 重算**：扩 A 后 `aUnit` 增大，`postBufSize` 和 `preBufSize` 需重算。Step 2 算出的 rUbFactor 不变（R 已全驻），UB 预算仍满足。

### 1.3 特殊场景

> 不涉及

## 2 多核切分

### 2.1 切分原理

外层 A loop 全部 fuse 成一根线性计数，多核按此瓜分。优先 A 切分以保证单核内完成 reduce，避免二次开销；A 并行度不足时借 R 轴补并行度（Group 模板，见 [reduction-binary-group-tiling.md](../reduction-template-binary-group/reduction-binary-group-tiling.md)）。

记 A 切分轴为 Ai，切分因子为 `aUbFactor`，则 `aSplitChunkCnt = CeilDiv(Ai, aUbFactor)` 为切分后的 chunk 数，`Ai.o = aSplitChunkCnt`。切分轴左侧的所有外层 A 轴（A0, A1, ..., A_{i-1}）整根参与外层循环，与 Ai.o 一起 fuse 成一根线性计数：`aLoopCntTotal = A0 × A1 × ... × Ai.o`。多核按 `aLoopCntTotal` 瓜分，每核处理若干个 aLoop。

### 2.2 切分公式

**aLoopCntTotal 计算**（`ComputeFusedALoopSplit()`）：

```cpp
int64_t aSplitChunkCnt = CeilDiv(axisShape[aSplitIdx], aUbFactor);
int64_t outerAProd     = 1;
for (int k = 0; k < aSplitIdx; k += 2) {    // A 起头+严格交替，偶数下标=A
    outerAProd *= axisShape[k];
}
int64_t aLoopCntTotal  = outerAProd × aSplitChunkCnt;
```

线性编码（row-major，aSplitIdx chunk 在最内）：

```
aLoopIdx = ((((i0 × A1 + i1) × A2 + i2) × ...) × A_{m-1} + i_{m-1}) × aSplitChunkCnt + aSplitChunkIdx
其中 i_k 为外层 A 轴 A_k 的轴下标（k = 0, 2, ..., aSplitIdx−2）
解码顺序相反：先 % aSplitChunkCnt 取 chunk，再逐轴 % /，与 UnravelALoop 一致
```

**大小核均衡分核**：

```cpp
int64_t aSmallCoreLoopCnt = aLoopCntTotal / coreNum;
int64_t aBigCoreCnt       = aLoopCntTotal % coreNum;
int64_t aBigCoreLoopCnt   = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0);
int32_t usedCoreNum       = (aSmallCoreLoopCnt > 0) ? coreNum : static_cast<int32_t>(aBigCoreCnt);
```

- 前 `aBigCoreCnt` 核：处理 `aBigCoreLoopCnt` 个 aLoop
- 其余核：处理 `aSmallCoreLoopCnt` 个 aLoop
- blockIdx ≥ usedCoreNum：idle
- 最大负载差 ≤ 1

**Kernel 端映射**：

```cpp
// blockIdx → [aLoopStart, aLoopEnd)
if (blockIdx < aBigCoreCnt) {
    aLoopStart = blockIdx * aBigCoreLoopCnt;
    aLoopEnd   = aLoopStart + aBigCoreLoopCnt;
} else {
    aLoopStart = aBigCoreCnt * aBigCoreLoopCnt + (blockIdx - aBigCoreCnt) * aSmallCoreLoopCnt;
    aLoopEnd   = aLoopStart + aSmallCoreLoopCnt;
}
```

**rLoopCntTotal 计算**（`ComputeRLoopCnt()`）：

```cpp
outerR_total = ∏(R axis idx < rSplitIdx) axisShape[i]
             × CeilDiv(axisShape[rSplitIdx], rUbFactor);
rLoopCntTotal = outerR_total;
```

**约束**：
- 严禁 `SetBlockDim(0)`，空 tensor 场景若使用核数为 0，也需要 `SetBlockDim(1)`, kernel 侧通过 `usedCoreNum=0` 短路处理
- 设置的 BlockDim 严禁超过实际的物理核数

### 2.3 特殊场景

无。

## 3 切分策略微调

### 3.1 微调方式

> 本章节是出于性能考虑，为了平衡 ub 切分与多核切分做的优化调整。

reduction 范式 UB 切分优先吃满 UB 一次性算完，多核并行度不足时由 Group 模板触发判定接手（见 [reduction-binary-group-tiling.md](../reduction-template-binary-group/reduction-binary-group-tiling.md) §2）。


## 4 整体切分伪码

```cpp
GenericReduceTiling(context):
    // ─── 0) 平台信息 ───
    GetPlatformInfo()                         // coreNum, ubSize, blockSize, cacheLineSize
    // ⚠ 获取后必须校验取值合法性：
    //   指针判空 + coreNum==0 / ubSize==0 / blockSize==0 / cacheLineSize==0 逐一拦截
    //   （OP_CHECK_IF ... return ge::GRAPH_FAILED），任一命中即失败——
    //   coreNum/ubSize 后续作除数（多核均分、UB 预算反解），零值未拦截即除零
    GetShapeAttrsInfo()                       // 原始 shape, dtype；按算子公式解析 reduce 轴参数，构建 A/R 类型表

    // ─── ★ 空 tensor 短路（必须在合轴四步之前）───
    if (∃ A 轴 axisShape == 0 || (∃ R 轴 axisShape == 0 && ∀A>0)):
        return HandleEmptyTensor();           // 见 reduction-empty-tiling.md

    // ─── 以下为 BASE / GROUP 模板（isEmptyTensor=0）───
    // pattern 预处理（四步顺序不可换）
    DropSizeOneAxes();
    FuseAxis();
    PadLeadingOneA();
    PadRIfPureA();

    if (axisNum < 2 || axisNum > MAX_PATTERN_RANK) {
        return FAIL;
    }

    isTailR = (axisNum % 2 == 0);
    isTailA = !isTailR;

    // 1) UB 切分：定 aUbFactor → 反解 rUbFactor → 所有 R 全驻后扩 A
    ComputeAUbFactor();             // Step1: 定 aSplitIdx + aUbFactor + innerAProdAlign
    ComputeRUbFactor();             // Step2: 反解 rSplitIdx + rUbFactor + innerRProdAlign
    ExpandAIfRFullyLoaded();        // Step3: 所有 R 全驻 → 用剩余 UB 反解更大的 aUbFactor

    // 2) 多核切分（fused aLoop 分核）
    ComputeFusedALoopSplit();       // aLoopCntTotal + 大小核均衡

    // 3) 算 rLoopCntTotal
    ComputeRLoopCnt();              // rLoopCntTotal = outerRProd × CeilDiv(axisShape[rSplitIdx], rUbFactor)

    // 4) Group 模板判定（条件性）
    if (ShouldUseGroup(aLoopCntTotal, rLoopCntTotal)):
        ComputeGroupSplit();        // A×R 2D 分核 → usedCoreNum, rGroupCnt
        SetScheduleMode(1);         // SyncAll 要求

    // 5) 算 UB sizes、填 tilingdata、tilingkey、blockDim、workspace
    ComputeUbSizes();               // preBufSize / postBufSize / cacheBufUbSize
    FillAndLogTilingData();         // 填充tilingdata，并使用 OP_LOGI 打印tilingdata
    SetTilingKey({isGroup, isEmptyTensor});
    SetBlockDim(usedCoreNum);
    SetWorkspaceSize(wsSize);       // wsSize = GetLibApiWorkSpaceSize()；group 时额外加 usrWorkspaceBytes
```

## 5 切分后处理

### 5.1 UB 大小计算

搬入 buffer（D_T）和计算 buffer（fp32）生命周期不重叠时可复用同一物理 buffer，为容纳最大 dtype，统一按 maxDtypeSize 分配。元素个数按 D_T 计算（搬运、对齐、切分逻辑不变），UB 实际占用按元素个数 × maxDtypeSize。详见 [dag-buffers.md](reduction-binary-base-dag-buffers.md) §1.1.1。

```
maxDtypeSize   = max(sizeof(D_T), sizeof(float))
aUnit          = aUbFactor × innerAProdAlign
postBufSize    = CeilAlign(aUnit × maxDtypeSize, blockSize)        // post buffer 单份大小，1D，block 对齐
preBufSize     = aUnit × rUbFactorAlign × innerRProdAlign × maxDtypeSize  // pre buffer 2D，[行 × 列] dense
cacheBufUbSize = 16 × 1024
```

> **cacheBuf 容量结论**：`cacheCount × CeilAlign(aUnit, 8) × sizeof(float) ≤ cacheBufUbSize` 恒成立（`cacheCount = CalLog2(FindNearestPower2(rLoopCntTotal)) + 1`）——fp32/All Reduce 恒满足（`aUnit ≤ cachelineTmp=64` → 最多 64 层）；b16（fp16/bf16）需 `rLoopCntTotal ≥ 2^32` 才违反，实际不可达，不设校验。R 全驻（`cacheCount=1`）由 §1.2 Step 3 的 `aUnitMax ≤ cacheBufUbSize / sizeof(float)` 钳制。

> **postBufSize vs preBufSize 的对齐差异**：post buffer 是 1D 的，必须 `CeilAlign` 到 block；pre buffer 是 2D 的，burst 尾轴方向已由 rUbFactorAlign/innerAProdAlign 保证对齐，整体按 `aUnit × rUbFactorAlign × innerRProdAlign × maxDtypeSize` 计算。tail-A 下 aUnit 一定对齐；tail-R 下 rUbFactorAlign × innerRProdAlign 一定对齐。


**UB 预算不等式**：

```
(P_pre + P_pre_ext) × preBufSize + P_post × postBufSize ≤ ubAvailable
```

其中 `ubAvailable = ubSize − cacheBufUbSize`（先扣固定 16KB）。`P_pre` / `P_pre_ext`（固定 1）/ `P_post` 由 [dag-buffers.md](reduction-binary-base-dag-buffers.md) §1.5 Buffer 产出汇总传入。

### 5.2 Workspace 分配

```cpp
size_t* ws = context->GetWorkspaceSizes(1);
OP_CHECK_NULL_WITH_CONTEXT(context, ws);
size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
ws[0] = sysWorkspaceSize;
```

**约束**:
- 必须设置，即使不需要使用 workspace，也需要显式设置
- 如果要使用 workspace，总大小为 `sysWorkspaceSize + 实际使用`（ascendc 要求）
- 如果涉及 workspace 使用，必须使用 INFO 级别日志打印分配大小

**Group 模板**：`ws[0] = sysWorkspaceSize + usrWorkspaceBytes`，其中 `usrWorkspaceBytes = rGroupCnt × aTotal × sizeof(fp32)`（见 [reduction-binary-group-tiling.md](../reduction-template-binary-group/reduction-binary-group-tiling.md) §5.2）。

### 5.3 TilingKey 设置

```cpp
SetTilingKey({isGroup, isEmptyTensor});
```

**约束**：必须使用 INFO 级别日志打印 tilingkey 各模板参数。

### 5.4 TilingData 设置

填充 TilingData 全字段 + OP_LOGI 全量打印。完整字段定义见 [reduction-template-overview.md](../reduction-template-overview.md) §4.1。

**约束**：
- TilingData 每个字段必须 `OP_LOGI` 打印
- `OP_LOGE` 必须含具体变量值（不能只写描述）

**Tiling 工程规范**：
- Attr 按索引访问，禁止按名：`attrs->GetAttrPointer<float>(0)`
- 整数类型严格匹配：`GetDimNum()` 返回 `size_t`，不能直接与 `int64_t` 比较，需 `static_cast`
- 禁止死代码：`-Werror=unused-variable`，声明但未使用的变量导致编译错误

### 5.5 ScheduleMode 设置

**约束**:
- SyncAll() 场景必须设置 `SetScheduleMode(1)`

Base 模板不调用 SyncAll，**不设置** ScheduleMode。Group 模板必须设置（见 [reduction-binary-group-tiling.md](../reduction-template-binary-group/reduction-binary-group-tiling.md) §5.5）。

## 6 切分策略产出校验

> 枚举 AR / ARAR pattern 在 Base 模板下的 UB 切分轴与多核切分轴。

### 6.1 AR pattern（2 轴，tail-R）

`A0 R0`，尾轴是 R0。单 A 轴 + 单 R 轴，切分轴唯一：

| # | aSplitIdx | rSplitIdx | blkSplit | UB内的轴 |
|---|-----------|-----------|------|----------|
| 1 | A0 | R0 | A0.o | A0.i × R0.i |

### 6.2 ARAR pattern（4 轴，tail-R）

`A0 R0 A1 R1`，尾轴是 R1。A 轴可选 A0 或 A1，R 轴可选 R0 或 R1：

| # | aSplitIdx | rSplitIdx | blkSplit | UB内的轴 |
|---|-----------|-----------|------|----------|
| 1 | A0 | R1 | A0.o | A0.i × A1 × R1.i |
| 2 | A0 | R0 | A0.o | A0.i × A1 × R0.i × R1 |
| 3 | A1 | R1 | A0 * A1.o | A1.i × R1.i |
| 4 | A1 | R0 | A0 * A1.o | A1.i × R0.i × R1 |
