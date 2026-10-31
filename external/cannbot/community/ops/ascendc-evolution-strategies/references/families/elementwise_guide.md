# Elementwise 族策略选型指南

> 算子族：`elementwise`
> 适用算子：`Add`, `Sub`, `Mul`, `Div`, `Sin`, `Cos`, `Exp`, `Log`, `Sqrt`, `Relu`, `Sigmoid`, `Tanh`, `Foreach*` 等
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.2

## 选型决策树

```
是否计算密集（Vector busy 主导）？
├── 是 (vec_ratio > 50%)
│   ├── 简单运算 (Add/Mul/Relu) → 基础组合（P1 + P5 + P10）
│   ├── 多步链式 (Mul → Add → Cast) → UB 融合 (P69 + P84)
│   ├── 复杂运算 (Exp/Log/Sqrt) → P67 Vector Counter 模式
│   └── 大量 cast (fp16 ↔ fp32) → D1 Mixed Precision + D2 Template Kernel
└── 否（搬运密集，rare）→ P10 向量化拷贝 + P26 stride 搬运

是否数据非对齐？
├── 是 (tail block / 非 32B) → P7 32B Alignment + DataCopyPad
└── 否 → 默认 DataCopy
```

## 族特定策略表（7 张）

| 策略 | Tier | Playbook | 触发场景 | 核心手段 |
|---|---|---|---|---|
| **P2** Adaptive Tiling | L0 | — | 各种 shape，自适应分块 | host 根据 N 计算最优 tileSize / blockDim |
| **P67** Vector Counter 模式 | L1 | ✅ | 循环有条件分支 / 复杂控制流 | `SetVectorMask(counter_mode)` 在 vector 指令内置 mask |
| **P69** UB 融合连续 Vector | L1 | — | 多步算子链 (Mul-Add-Cast) | 避免每步 GM 往返，UB 内累积 |
| **P84** Vector Compute Efficiency | L2 | — | vector busy 已接近极限 | 指令组合、双发射、avoid bank conflict |
| **D2** Template Kernel | L0 | — | foreach / 跨多 dtype 同算子 | C++ template + TilingKey 编译期 dispatch |
| **A1** FP32 中间计算 | L0 | — | fp16 输入 + 精度敏感运算 | 中间结果保 FP32，输出再 cast 回 fp16 |
| **A4** SetFlag/WaitFlag 事件同步 | L0 | — | 多 vector 链有数据依赖 | event 替代 barrier，保数据精度 |

## 推荐组合

| 组合 | 适用 | 说明 |
|---|---|---|
| **P1 + P5 + P10** | 基础 elementwise（Add/Mul/Relu） | 标准双缓冲 + 同步 + 向量化（覆盖 80% 场景） |
| **P1 + P5 + P10 + P69** | 多步链式 (e.g. foreach_addcmul) | 加 UB 融合避免 GM 往返 |
| **D2 + D1** | foreach 多 dtype | template + 混合精度 |
| **P67 + P84** | 复杂控制流 + vector 极限 | Counter mode + 指令级优化 |

## 叠加正交性

| 优化 | P1 | P5 | P10 | P67 | P69 |
|---|---|---|---|---|---|
| **P1** | — | 协同 | 协同 | 正交 | 正交 |
| **P5** | 协同 | — | 协同 | 正交 | 正交 |
| **P10** | 协同 | 协同 | — | 正交 | 协同 |
| **P67** | 正交 | 正交 | 正交 | — | 协同 |
| **P69** | 正交 | 正交 | 协同 | 协同 | — |

## 性能参考（cannbot ops-profiling 数据）

| 算子类型 | 主导流水 | 预期 ratio | 异常信号 |
|---|---|---|---|
| Add/Mul/Relu | VEC | vec_ratio 50-80% | MTE2 ratio > VEC ratio |
| Sin/Cos/Sigmoid | VEC | vec_ratio 60-85% | 大量 cast 指令 |
| 大量 cast 链 | VEC | vec_fp32 + vec_fp16 占比高 | 应用 D1 |

## Playbook 状态

- ✅ P1, P5, P10, P67 已有 Playbook
- ❌ P69, P84, D2, A1, A4 待 Phase C 补

---

## Ascend C Elementwise Tiling 系统化设计

> 来源：插件本地 [`tiling-design/elewise/patterns.md`](../tiling-design/elewise/patterns.md)
>
> 本章节沉淀 Elementwise 族在 Ascend C 上的 Tiling 设计方法论。

### 1. 场景判定流程

```
给定: N 个输入 shape + M 个输出 shape
Step 1 — Shape 判定:
  所有输入输出 shape 完全相同？
    ├─ YES → EleWise，展平为 dim0，1D 线性处理 → Step 2
    └─ NO  → Broadcast → 参考 broadcast_guide.md

Step 2 — dtype × 运算 判定:
  运算为 Add/Sub / 以加减为主的累加链路 AND dtype ∈ {FP16, BF16} AND spec 未声明"输入同量级"？
    ├─ YES → 升精度分支：Cast→FP32 计算→Cast，需额外 K 份 FP32 中间 Buffer
    └─ NO  → 原 dtype 直接计算分支
```

### 2. 通用规则

- **多核对齐**：元素数对齐到 512 的倍数（`ELEM_ALIGN_FACTOR`），每个核至少 4KB 数据。
- **UB 对齐**：按 256B 对齐，确保 Vector 指令效率。
- **区分首/尾 block**：尾 block 数据量可能小于首 block，循环次数和 tail 大小不同。

### 3. UB 切分

从最内轴向外累乘，找到第一个放不下的轴作为 `ubSplitAxis`：

```python
curProduct = 1
ubSplitAxis = 0
allFit = true
for i = shapeLen-1 downto 0:
    curProduct *= dims[i]
    if curProduct > maxElemNum:
        ubSplitAxis = i
        curProduct /= dims[i]
        allFit = false
        break

if allFit:
    curProduct /= dims[0]

if shapeLen == 1:
    ubFormer = maxElemNum
else:
    ubFormer = maxElemNum / curProduct

ubOuter = ceil(dims[ubSplitAxis] / ubFormer)
ubTail  = dims[ubSplitAxis] - (ubOuter-1) * ubFormer
```

其中：

```python
maxElemNum = (ubSize - extraSize) * 8 / (bufferNum * maxDtypeBits)
maxElemNum = floor_align(maxElemNum, 256 * 8 / minDtypeBits)
```

### 4. 多核切分

```python
fusedProduct = ubOuter × (ubSplitAxis 之前所有轴乘积)
blockFormer  = ceil(fusedProduct / coreNum)
blockNum     = ceil(fusedProduct / blockFormer)
blockTail    = fusedProduct - (blockNum - 1) * blockFormer
```

核利用率不足时，循环缩小 `maxElemNum`（每次减 `CACHE_LINE`），重算 `ubFormer/ubOuter/fusedProduct`，直到能喂满更多核。

### 5. 设计检查清单

- [ ] 输入输出 shape 是否完全相同？否则应走 Broadcast 路径。
- [ ] FP16/BF16 加减链路是否考虑升精度分支？
- [ ] 元素数是否对齐到 512（多核）和 256B（UB）？
- [ ] 是否区分首 block 与 tail block？
- [ ] `maxElemNum` 是否按 `bufferNum × maxDtypeBits` 计算并向下对齐？
- [ ] 核利用率不足时是否尝试缩小 `maxElemNum` 重新切分？
