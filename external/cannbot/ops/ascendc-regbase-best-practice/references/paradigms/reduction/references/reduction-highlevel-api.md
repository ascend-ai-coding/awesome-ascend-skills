# Reduce 高层 API（ReduceXxx 系列）

> ReduceSum / ReduceMax / ReduceMin / ReduceProd / ReduceAny / ReduceAll 的 API 原型、硬约束、Pattern 选择、srcShape 计算、isReuseSource 与 buffer 份数。

## 1 API 原型

```cpp
// 重载 1：不传 sharedTmpBuffer（API 内部 PopStackBuffer）
template <class T, class pattern, bool isReuseSource = false>
void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
               const uint32_t srcShape[], bool srcInnerPad);

// 重载 2：显式传 sharedTmpBuffer
template <class T, class pattern, bool isReuseSource = false>
void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
               const LocalTensor<uint8_t>& sharedTmpBuffer,
               const uint32_t srcShape[], bool srcInnerPad);

// ReduceMax / ReduceMin / ReduceProd / ReduceAny / ReduceAll 同族同签名
```

## 2 硬约束

| 约束 | 规则 | 违反后果 |
|------|------|---------|
| **src/dst 地址不可重叠** | `src` 和 `dst` 必须指向 UB 不同段 | 结果不可预测 |
| **src 起始地址 32B 对齐** | ReduceSum src 起始地址需 32B 对齐 | UB 非对齐访问运行时错 |
| **sharedTmpBuffer 与 src/dst 不可重叠** | sharedTmpBuffer 必须与 src/dst 指向 UB 不同段 | 内部 scratch 污染 src/dst |
| **不处理累加溢出** | b16 输入必须先 Cast 到 fp32 再做 reduce | 大数值/长 R 累加溢出 |
| **srcInnerPad** | 必须传 `true`（已按 32B 对齐填充） | tail-R 下读错 padding lane |
| **isReuseSource** | `true`：API 把 src 当内部 scratch，调用后 **src 内容失效**。结果始终写入 dst（src/dst 不可重叠）。`false`：src 内容保持不变，Reduce 后可继续读 | 误读已破坏的 src |

## 3 Pattern 选择

| tail 类型 | AscendC Pattern | srcShape | 含义 |
|---------|----------------|----------|------|
| tail-R | `Pattern::Reduce::AR` | `{aBundle, rBundle}` | 沿内层 R reduce，保留外层 A |
| tail-A | `Pattern::Reduce::RA` | `{rBundle, aBundle}` | 沿外层 R reduce，保留内层 A |

其中 `aBundle = aUbFactor × innerAProdAlign`，`rBundle = rUbFactorAlign × innerRProdAlign`。

## 4 srcShape 计算

srcShape 是 `uint32_t[2]`，按 **padded 值**计算（含 CeilAlign 因子），与 `srcInnerPad=true` 配合：

- **tail-R**：`srcShape = {aUbFactor × innerAProdAlign, rUbFactorAlign × innerRProdAlign}`
- **tail-A**：`srcShape = {rUbFactorAlign × innerRProdAlign, aUbFactor × innerAProdAlign}`

## 5 isReuseSource 与 buffer 份数

**本范式统一使用重载 2（带 sharedTmpBuffer）**，`preReduceResultTail` 兼作 sharedTmpBuffer。preReduceResultTail 本就为 Phase A 主尾配对分配，Reduce 时已空闲，复用不增加 UB 开销。

isReuseSource 的选择根据算子特征判定：

- **isReuseSource=true**：Reduce 后不依赖 src（后续计算不读 src），API 把 src 当内部 scratch（调用后 src 内容失效）
- **isReuseSource=false**：Reduce 后依赖 src（后续计算需读 src），src 内容保持不变

无论 isReuseSource 取何值，buffer 份数固定为 **3 份**：

| buffer | 用途 |
|---|---|
| `preReduceResult` | Reduce 的 src |
| `preReduceResultTail` | 兼作 sharedTmpBuffer（Phase A 配对后空闲） |
| `cacheBuf` | Reduce 的 dst |

## 6 ReduceSum 内部累加方式

> 本节为内部实现分析，以 arch35 为例，仅供理解 API 行为。范式将 ReduceXxx 视为黑盒，调用方式不受本节内容影响。

ReduceSum 内部采用 **多路并行二叉树折叠**（Multi-way Parallel Binary Tree Folding），非顺序累加。以 AR pattern 为例（fp32，`vlSize = GetVecLen() / sizeof(T)`）。

### 6.1 关键参数

> B64 = 64 位宽类型（如 fp64/int64）；非 B64 = 32 位及以下。每轮加载 16 个寄存器（B64 为 8 个），做 4 级（B64 为 3 级）并行二叉树合并。

| 参数 | 含义 | 计算 |
|------|------|------|
| `mainR` | ≤dimR 的最大 2^n | `CalculateMainR(dimR, isAR, vlSize)` |
| `tailR` | dimR 的非 2 幂尾数 | `dimR - mainR`（isReuseSource=false 且 tailR=0 时强制拆半） |
| `base` | VL 块数 | `mainR / vlSize` |
| `folds` | 总折叠层数 | `log2(base)` |
| `avgFolds` | 每轮主折叠层数 | 非 B64 = 4（16 寄存器 → 1），B64 = 3（8 → 1） |
| `mainTimes` | 主折叠轮数 | `folds / avgFolds` |
| `tailFolds` | 尾折叠层数 | `folds % avgFolds` → 选 foldZero/One/Two/Three 之一 |

### 6.2 四步算法

**Step 0 — 尾块合并**（dimR → mainR）：将 tailR 个尾元素加到 mainR 对应位置，剩余 mainR − tailR 个元素原样拷贝到 tmpBuf（isReuseSource=true 则原地写 src）。若 dimR 恰好是 2 的幂（tailR=0），isReuseSource=false 时强制拆半 `mainR /= 2, tailR = mainR`——等价于二叉树第一层。结果：mainR 个元素。

**Step 1 — 主折叠**（`mainTimes` 轮）：每轮加载 16 个寄存器（B64 为 8 个），做 4 级（B64 为 3 级）并行二叉树合并，数据量缩减 16 倍（B64 为 8 倍）：

```
非 B64（16→1, 4 级）:
  L1: 8 adds  (16→8)   vreg[i] += vreg[i+8]
  L2: 4 adds  (8→4)    vreg[i] += vreg[i+4]
  L3: 2 adds  (4→2)    vreg[i] += vreg[i+2]
  L4: 1 add   (2→1)    vreg[0] += vreg[1]

B64（8→1, 3 级）:
  L1: 4 adds  (8→4)
  L2: 2 adds  (4→2)
  L3: 1 add   (2→1)
```

**Step 2 — 尾折叠**（根据 `tailFolds` 选一分支）：

| tailFolds | 分支 | 寄存器 | 树级数 | 操作 |
|-----------|------|--------|--------|------|
| 0 | foldZero | 1 | 0 | 直接 Step 3 |
| 1 | foldOne | 2 | 1 | 1 add → Step 3 |
| 2 | foldTwo | 4 | 2 | 2 级树 → Step 3 |
| 3 | foldThree | 8 | 3 | 3 级树 → Step 3 |

> bf16/int8 在 Step 2 中先 Cast 到 fp32/half 再做硬件 Reduce。

**Step 3 — 硬件 Reduce 收尾**：对最后 1 个寄存器（vlSize 个元素）调用 `Reg::ReduceSum` 硬件指令，两两配对相加，归约为 1 个标量写入 dst。

### 6.3 示例

**示例 1**：AR, fp32, dimA=1, dimR=4096, isReuseSource=false（vlSize=64）

| 步骤 | 操作 | 元素数 | VL 块数 |
|------|------|--------|---------|
| Step 0 | tailR=0 → 拆半 Add(2048+2048) | 4096→2048 | 32 |
| Step 1 ×1 | 16-way 4 级树 | 2048→128 | 32→2 |
| Step 2 | foldOne, 2 regs 1 add | 128→64 | 2→1 |
| Step 3 | Reg::ReduceSum | 64→1 | — |

**示例 2**：AR, fp32, dimA=1, dimR=131072, isReuseSource=false（vlSize=64）

| 步骤 | 操作 | 元素数 | VL 块数 |
|------|------|--------|---------|
| Step 0 | tailR=0 → 拆半 Add(65536+65536) | 131072→65536 | 1024 |
| Step 1 ×1 | 16-way 4 级树 | 65536→4096 | 1024→64 |
| Step 1 ×2 | 16-way 4 级树 | 4096→256 | 64→4 |
| Step 2 | foldTwo, 4 regs 2 级树 | 256→64 | 4→1 |
| Step 3 | Reg::ReduceSum | 64→1 | — |

### 6.4 与范式二分缓存树的关系

ReduceSum 内部二叉树处理 **单个 rChunk 内** 的 R 维归约（rUbFactorAlign 个元素 → 1 个结果）。范式二分缓存树处理 **跨 rChunk** 的累加（多个 rChunk 的 reduce 结果 → cacheBuf 二分合并）。两者构成两级树：

- **API 级树**（本节）：rChunk 内，多路并行二叉树 + 硬件 Reduce 指令
- **范式级树**（[reduction-binary-base-kernel-template.md](reduction-template-binary-base/reduction-binary-base-kernel-template.md) §1.1）：rChunk 间，cacheBuf 二分缓存树

范式无需关心 API 内部实现——传入 srcShape，API 返回 dst。API 内部已用树形累加保证单次 reduce 的精度，范式二分缓存树进一步保证跨 chunk 累加精度。
