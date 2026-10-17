# AdamApplyOneAssign 详细设计

## 1. 逻辑存活节点

### 1.1 核心原则

**存活节点** = 同一时刻 UB 中必须同时驻留的 tensor buffer 数量。

**逻辑存活节点** = 纯计算图结构决定的数学下界——不涉及硬件参数、不依赖 API 能力、不考虑实现策略。通过调度顺序优化后，计算图中不可避免的最小峰值存活节点数。P ≥ L 恒成立，物理不可能低于逻辑。

### 1.2 物理意义

逻辑存活节点是 Tiling 分析的起点。它回答一个问题：在不考虑任何硬件限制的前提下，这个算子的计算图拓扑结构本身，至少需要几个 buffer 同时在线？

方法：
1. 画出完整计算图（含 broadcast），来自接口分析
2. 识别分叉结构——同一 tensor 被多处引用，产生需要跨步保留的中间量
3. 按最优调度顺序（先集中消费分叉源，用完即放）逐步追踪
4. 每步记录存活节点 = 上下文中间量 + 本轮输入 + 本轮输出
5. 峰值即为最小逻辑存活节点

意义：为后续物理存活节点分析提供不可突破的下界。`perBufferMax = UB / L`，这是任何实现策略都无法超越的内存上限。

### 1.3 计算图

```
Path 1 (→ output0):
  S1:  squareResult = vmul(dataInput0, dataInput0)
  S2:  mul3Result  = vmul(squareResult, dataInputMul3)         [broadcast]
  S3:  mul2Result  = vmul(dataInput1, dataInputMul2)         [broadcast]
  S4:  output0 = vadd(mul2Result, mul3Result)          [broadcast]
  S5:  sqrtResult   = vsqrt(output0)
  S6:  add2Result  = vadd(sqrtResult, dataInputAdd2)           [broadcast]

Path 2 (→ output1):
  S7:  mul0Result  = vmul(dataInput2, dataInputMul)         [broadcast]
  S8:  mul1Result  = vmul(dataInput0, dataInputMul1)      [broadcast]
  S9:  output1 = vadd(mul0Result, mul1Result)          [broadcast]

Merge (→ output2):
  S10a: output1 = vmuls(output1, 1)           [恒等, in-place]
  S10b: truedivResult = vdiv(output1, add2Result)        [broadcast]
  S11: mul4Result   = vmul(truedivResult, input4)        [broadcast]
  S12: output2 = vsub(dataInput3, mul4Result)        [broadcast]
```

### 1.4 分叉点

`dataInput0` 同时被 S1 (Path1) 和 S8 (Path2) 引用，是唯一的分叉源。

### 1.5 最优调度

策略：先集中消费分叉源 `dataInput0`（S1→S8 背靠背），S8 完成后立即释放。剩余操作按数据依赖排布，尽早释放不再需要的中间量。

```
Step 1: 载 dataInput0
        S1: squareResult = vmul(dataInput0, dataInput0)
        存活: [dataInput0, squareResult] = 2

Step 2: 载 dataInputMul1
        S8: mul1Result = vmul(dataInput0, dataInputMul1)
        存活: [dataInput0, squareResult, dataInputMul1, mul1Result] = 4
        释放 dataInput0, dataInputMul1
        存活: [squareResult, mul1Result] = 2

Step 3: 载 dataInput2, dataInputMul
        S7: mul0Result = vmul(dataInput2, dataInputMul)
        存活: [squareResult, mul1Result, dataInput2, dataInputMul, mul0Result] = 5  ← 峰值
        释放 dataInput2, dataInputMul
        存活: [squareResult, mul1Result, mul0Result] = 3

Step 4: S9: output1 = vadd(mul0Result, mul1Result)
        存活: [squareResult, mul0Result, mul1Result, output1] = 4
        释放 mul0Result, mul1Result
        存活: [squareResult, output1] = 2

Step 5: 载 dataInputMul3
        S2: mul3Result = vmul(squareResult, dataInputMul3)
        存活: [squareResult, output1, dataInputMul3, mul3Result] = 4
        释放 squareResult, dataInputMul3
        存活: [output1, mul3Result] = 2

Step 6: 载 dataInput1, dataInputMul2
        S3: mul2Result = vmul(dataInput1, dataInputMul2)
        存活: [output1, mul3Result, dataInput1, dataInputMul2, mul2Result] = 5  ← 峰值
        释放 dataInput1, dataInputMul2
        存活: [output1, mul3Result, mul2Result] = 3

Step 7: S4: output0 = vadd(mul2Result, mul3Result)
        存活: [output1, mul2Result, mul3Result, output0] = 4
        释放 mul2Result, mul3Result
        存活: [output1, output0] = 2

Step 8: S5: sqrtResult = vsqrt(output0)
        存活: [output1, output0, sqrtResult] = 3

Step 9: 载 dataInputAdd2
        S6: add2Result = vadd(sqrtResult, dataInputAdd2)
        存活: [output1, output0, sqrtResult, dataInputAdd2, add2Result] = 5  ← 峰值
        释放 sqrtResult, dataInputAdd2
        存活: [output1, output0, add2Result] = 3

Step 10: S10a: output1 = vmuls(output1, 1)    [in-place]
         S10b: truedivResult = vdiv(output1, add2Result)
         存活: [output1, output0, add2Result, truedivResult] = 4
         释放 add2Result
         存活: [output1, output0, truedivResult] = 3

Step 11: 载 input4
         S11: mul4Result = vmul(truedivResult, input4)
         存活: [output1, output0, truedivResult, input4, mul4Result] = 5  ← 峰值
         释放 truedivResult, input4
         存活: [output1, output0, mul4Result] = 3

Step 12: 载 dataInput3
         S12: output2 = vsub(dataInput3, mul4Result)
         存活: [output1, output0, dataInput3, mul4Result, output2] = 5  ← 峰值
         释放 dataInput3, mul4Result
         存活: [output0, output1, output2] = 3
```

### 1.6 结论

| 指标 | 值 |
|------|-----|
| **逻辑存活节点 L** | **5** |
| 峰值出现步骤 | Step 3, 6, 9, 11, 12 |
| 瓶颈 | `dataInput0` 分叉产生 `squareResult` 和 `mul1Result` 两个跨步上下文量 |
| 下界不可降原因 | 2 上下文 + 2 输入 + 1 输出 = 5，分叉结构决定了上下文无法合并 |

### 1.7 对本算子的意义

- perBuffer理论上限 = 256KB / 5 ≈ 51KB > 48KB，逻辑下界本身已可接受
- 该计算图上任何tiling方案的存活节点 **≥ 5**
- 不可降：继续划分任务时，每个step都最少需要 5 个节点
- 简化瓶颈：如果找到减少分叉的方法，可进一步降低逻辑存活节点

---

## 2. 物理存活节点

### 2.1 核心原则

**物理存活节点 = f(计算流, API 能力, 算法实现)**。三要素缺一不可：

- **计算流**：规则 3 产出的完整计算图（含 broadcast），不可简化
- **API 能力**：芯片/框架能做什么——NDDMA 随路 broadcast、MulAddDst 融合、RegBase 寄存器域
- **算法实现**：你选择怎么做——broadcast 提前还是延迟、哪些操作融合、用不用 RegBase

同一张计算图，不同实现策略 → 不同物理节点数。物理存活节点不由计算图唯一确定。

**计数域**：UB 上同时驻留的 buffer 数。寄存器不计入（UB→Reg→UB 过路中转）。

**P ≥ L 恒成立**。逻辑和物理在同一公平基线上——两端都不假设输入输出 buffer 复用。差异只能来自 API 硬件能力。

### 2.2 物理意义

逻辑存活节点回答"至少几个"，物理存活节点回答"实际几个"。两者可能一致（P = L），也可能有差距（P > L）。

三要素如何在每一步中起作用：
1. 从完整计算图中取一步（broadcast + vXXX）
2. 标每个 tensor 的物理数据通路：从哪来（GM 搬入 / UB 驻留 / 上一步产出），broadcast 放在哪步做（随 DataCopy 还是独立），是否有融合指令可用，到哪去（下游哪步用到 / 释放）
3. 物理节点 = 此时刻 UB 需同时驻留的不同 buffer 数
4. 穷尽算法选择（broadcast 时机、融合配对），取最小的物理节点

**公平基线**：逻辑分析不假设复用，物理分析同样不假设。只有硬件融合指令（MulAddDst）自带的复用计入 API 能力。

**RegBase 边界**：硬件不支持同一 buffer 同一循环内 in-place read-modify-write。RegBase 对单次二元操作仍需 src0 + src1 + dst 三块独立 UB buffer，价值在性能（减少 UB 读写次数），不在降峰值节点。

### 2.3 可用 API 能力

| API | 能力 | 对物理节点的意义 | 来源 |
|-----|------|-----------------|------|
| **NDDMA 随路 broadcast** | DataCopy(GM→UB) 时一并 broadcast | 不另占 UB buffer，broadcast 被 DataCopy 消化 | ascendc.md |
| **MulAddDst** | `dst = src0×src1 + dst`，vmul+vadd 合为 1 条，dst 就地覆写 | 省 1 个 UB 中间 buffer | ascendc.md |
| **RegBase (Reg::Mul/Add/Sub)** | UB→Reg→UB，中间值留在寄存器 | 将部分 UB 临时 buffer 转移至寄存器，压低 UB 物理节点 | regbase.md |
| **Reg::Sqrt / Reg::Div** | 寄存器级 Sqrt/Div | **未确认**，需查 reg_vector.h | regbase.md |

### 2.4 融合机会（MulAddDst）

```
S7: mul0Result = vmul(dataInput2, dataInputMul)
S9: output1      = vadd(mul0Result, mul1Result)
    → MulAddDst(mul1Result, dataInput2, dataInputMul) → output1

S3: mul2Result = vmul(dataInput1, dataInputMul2)
S4: output0      = vadd(mul2Result, mul3Result)
    → MulAddDst(mul3Result, dataInput1, dataInputMul2) → output0
```

条件：中间 vmuls 结果无其他引用，dst(累加器) 被覆写后下游不再需要原值。

### 2.5 物理节点分析

NDDMA broadcast + MulAddDst，逐步追踪：

```
Step 1:
  DC: dataInput0 (GM→UB, NDDMA broadcast) → dataInput0'
  C:  vmul(dataInput0', dataInput0') → squareResult
  UB: [dataInput0', squareResult] = 2

Step 2:
  DC: dataInputMul1 (GM→UB, NDDMA broadcast) → dataInputMul1'
  C:  vmul(dataInput0', dataInputMul1') → mul1Result
  峰值: [squareResult, dataInput0', dataInputMul1', mul1Result] = 4
  释放 dataInput0', dataInputMul1'
  UB: [squareResult, mul1Result] = 2

Step 3:  ← S7+S9 MulAddDst
  DC: dataInput2 (GM→UB, NDDMA broadcast) → dataInput2'
  DC: dataInputMul (GM→UB, NDDMA broadcast) → dataInputMul'
  C:  MulAddDst(mul1Result, dataInput2', dataInputMul') → output1
      (硬件融合: dst=mul1Result 就地覆写为 output1)
  峰值: [squareResult, mul1Result, dataInput2', dataInputMul'] = 4
  释放 dataInput2', dataInputMul'
  UB: [squareResult, output1] = 2

Step 4:
  DC: dataInputMul3 (GM→UB, NDDMA broadcast) → dataInputMul3'
  C:  vmul(squareResult, dataInputMul3') → mul3Result
  峰值: [output1, squareResult, dataInputMul3', mul3Result] = 4
  释放 squareResult, dataInputMul3'
  UB: [output1, mul3Result] = 2

Step 5:  ← S3+S4 MulAddDst
  DC: dataInput1 (GM→UB, NDDMA broadcast) → dataInput1'
  DC: dataInputMul2 (GM→UB, NDDMA broadcast) → dataInputMul2'
  C:  MulAddDst(mul3Result, dataInput1', dataInputMul2') → output0
      (硬件融合: dst=mul3Result 就地覆写为 output0)
  峰值: [output1, mul3Result, dataInput1', dataInputMul2'] = 4
  释放 dataInput1', dataInputMul2'
  UB: [output1, output0] = 2

Step 6:
  C: vsqrt(output0) → sqrtResult
  UB: [output1, output0, sqrtResult] = 3

Step 7:  ← 峰值 ★
  DC: dataInputAdd2 (NDDMA broadcast) → dataInputAdd2'
  C:  vadd(sqrtResult, dataInputAdd2') → add2Result
      src0: sqrtResult      (读)
      src1: dataInputAdd2' (读)
      dst:  add2Result     (写, 独立 buffer)
  峰值: [output1, output0, sqrtResult, dataInputAdd2', add2Result]
       = 2(持久输出) + 3(vadd src0+src1+dst) = 5
  释放 sqrtResult, dataInputAdd2'
  UB: [output1, output0, add2Result] = 3

Step 8:
  C: vdiv(output1, add2Result) → truedivResult
  峰值: [output1, output0, add2Result, truedivResult] = 4
  释放 add2Result
  UB: [output1, output0, truedivResult] = 3

Step 9:  ← 峰值 ★
  DC: input4 (NDDMA broadcast) → input4'
  C:  vmul(truedivResult, input4') → mul4Result
      src0: truedivResult (读)
      src1: input4'        (读)
      dst:  mul4Result   (写, 独立 buffer)
  峰值: [output1, output0, truedivResult, input4', mul4Result]
       = 2(持久输出) + 3(vmul src0+src1+dst) = 5
  释放 truedivResult, input4'
  UB: [output1, output0, mul4Result] = 3

Step 10:  ← 峰值 ★
  DC: dataInput3 (NDDMA broadcast) → dataInput3'
  C:  vsub(dataInput3', mul4Result) → output2
      src0: dataInput3'  (读)
      src1: mul4Result  (读)
      dst:  output2       (写, 独立 buffer)
  峰值: [output1, output0, dataInput3', mul4Result, output2]
       = 2(持久输出) + 3(vsub src0+src1+dst) = 5
  释放 dataInput3', mul4Result
  UB: [output0, output1, output2] = 3  ← 三输出就位
```

### 2.6 结论

| 指标 | 值 |
|------|-----|
| 逻辑下界 L | 5 |
| **物理存活节点 P** | **5** |
| P vs L | P = L，打到下界 |
| 峰值步骤 | Step 7, 9, 10 |
| 依赖 API | NDDMA broadcast + MulAddDst — **全部已确认** |
| 瓶颈 | output0+output1 持久化 = 2 上下文，后续二元操作 src0+src1+dst = 3，合 5 |

物理没有低于逻辑——逻辑数 tensor，物理数 UB buffer，两者都不假设输入输出复用。MulAddDst 的复用是硬件能力（不属于软件调度假设），RegBase 无法做到同等的 in-place 覆写。

### 2.7 对本算子的意义

- P = 5，perBuffer = 256KB / 5 ≈ 51KB > 48KB ✓
- tile 长度上限 = 51KB / sizeof(dtype)
  - FP16: ~26K elements/tile
  - FP32: ~13K elements/tile
- 物理已打到逻辑下界，此算子 Tiling 分析完成

---

## 3. 计算流伪码

### 3.1 核心原则

计算流伪码是 §2 物理存活节点分析的序列化产物。§2 选定了三要素组合（NDDMA broadcast + MulAddDst），伪码就是把这把选择的每一步写成 `DataCopy → 计算 → 释放` 的完整序列。每步标注 UB 驻留数，验证全程 ≤ P。

伪码不是实现——它是物理分析到 kernel 编码之间的桥梁。后续所有设计（计算流验证、输入预处理、UB 切分）都以这份伪码为唯一参照。

### 3.2 物理意义

§2 的物理节点分析是"每一步 peak 多少"的抽象论证。伪码把它落实到每一行的具体操作：哪个 buffer 在哪个时刻被创建、被消费、被释放。它回答三个问题：

- §2 的 P=5 能否兑现为一个具体的步骤序列？
- 每步的 UB 驻留是否始终 ≤ P？
- 融合（MulAddDst）和 broadcast（NDDMA 随路）是否正确嵌入？

### 3.3 计算流伪码

```
输入:  dataInput0..4, dataInputMul, dataInputMul1..3, dataInputAdd2
输出:  output0, output1, output2
约束:  UB 最多同时驻留 5 块 buffer

// ─── Step 1: 载入 dataInput0，算 square ───
DataCopy(dataInput0,  GM→UB, NDDMA_broadcast)   → bufInput0
AscendC::Mul(bufSquare, bufInput0, bufInput0)           → bufSquare
// UB: [bufInput0, bufSquare] = 2

// ─── Step 2: 载入 mul1_x，算 mul1 ───
DataCopy(dataInputMul1, GM→UB, NDDMA_broadcast) → bufMul1X
AscendC::Mul(bufMul1, bufInput0, bufMul1X)             → bufMul1
// 释放 bufInput0, bufMul1X
// UB: [bufSquare, bufMul1] = 2

// ─── Step 3: 载入 dataM + mul0_x，MulAddDst 融合 S7+S9 → output1 ───
DataCopy(dataInput2,    GM→UB, NDDMA_broadcast) → bufDataM
DataCopy(dataInputMul, GM→UB, NDDMA_broadcast) → bufMul0X
MulAddDst(bufMul1, bufDataM, bufMul0X)      → output1
// dst=output1 就地覆写 bufMul1
// 释放 bufDataM, bufMul0X
// UB: [bufSquare, output1] = 2

// ─── Step 4: 载入 mul3_x，算 mul3 ───
DataCopy(dataInputMul3, GM→UB, NDDMA_broadcast) → bufMul3X
AscendC::Mul(bufMul3, bufSquare, bufMul3X)             → bufMul3
// 释放 bufSquare, bufMul3X
// UB: [output1, bufMul3] = 2

// ─── Step 5: 载入 dataV + mul2_x，MulAddDst 融合 S3+S4 → output0 ───
DataCopy(dataInput1,     GM→UB, NDDMA_broadcast) → bufDataV
DataCopy(dataInputMul2, GM→UB, NDDMA_broadcast) → bufMul2X
MulAddDst(bufMul3, bufDataV, bufMul2X)       → output0
// dst=output0 就地覆写 bufMul3
// 释放 bufDataV, bufMul2X
// UB: [output1, output0] = 2

// ─── Step 6: sqrt ───
AscendC::Sqrt(bufSqrt, output0)                           → bufSqrt
// UB: [output1, output0, bufSqrt] = 3

// ─── Step 7: 载入 add2_y，算 add2 ───  ← 峰值 5
DataCopy(dataInputAdd2, GM→UB, NDDMA_broadcast) → bufAdd2Y
AscendC::Add(bufAdd2, bufSqrt, bufAdd2Y)               → bufAdd2
// UB: [output1, output0, bufSqrt, bufAdd2Y, bufAdd2] = 5
// 释放 bufSqrt, bufAdd2Y
// UB: [output1, output0, bufAdd2] = 3

// ─── Step 8: truediv ───
AscendC::Div(bufTruediv, output1, bufAdd2)               → bufTruediv
// 释放 bufAdd2
// UB: [output1, output0, bufTruediv] = 3

// ─── Step 9: 载入 input4，算 mul4 ───  ← 峰值 5
DataCopy(dataInput4, GM→UB, NDDMA_broadcast)     → bufInput4
AscendC::Mul(bufMul4, bufTruediv, bufInput4)            → bufMul4
// UB: [output1, output0, bufTruediv, bufInput4, bufMul4] = 5
// 释放 bufTruediv, bufInput4
// UB: [output1, output0, bufMul4] = 3

// ─── Step 10: 载入 dataVar，算 sub → output2 ───  ← 峰值 5
DataCopy(dataInput3, GM→UB, NDDMA_broadcast)     → bufDataVar
Sub(output2, bufDataVar, bufMul4)              → output2
// UB: [output1, output0, bufDataVar, bufMul4, output2] = 5
// 释放 bufDataVar, bufMul4
// UB: [output0, output1, output2] = 3

// ─── 搬出 ───
DataCopy(output0, UB→GM)
DataCopy(output1, UB→GM)
DataCopy(output2, UB→GM)
```

### 3.4 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| broadcast 策略 | 全部 DataCopy 时随路 NDDMA broadcast | 不另占 UB buffer |
| 融合策略 | S7+S9 → MulAddDst, S3+S4 → MulAddDst | 各省 1 中间 buffer |
| S10a 恒等乘 vmuls(output1,1) | 省略 | x×1.0 数学恒等（IEEE754 下 NaN/±Inf/-0 均不变），原图中是 broadcast 物化的历史产物；kernel 直接以 Div 消费 output1，省每 tile 一次全量向量乘 |
| output0/output1 持久化 | 保留至搬出，不提前释放 | 下游 truediv 和最终搬出都需要 |
| RegBase | 不使用 | 该算子二元操作 src 本就在 UB，RegBase 不降节点 |

### 3.5 结论

10 步序列全程 UB 驻留 ≤ 5，P=5 可兑现。NDDMA broadcast 消化了所有独立 broadcast 操作，MulAddDst 消化了 S7+S9 和 S3+S4 的中间 buffer。

### 3.6 对本算子的意义

计算流伪码是后续所有设计阶段的唯一参照：§4 验证语义等价，§5.4 kernel 循环骨架中"执行计算流"段直接嵌入这 10 步的 DataCopy → 计算序列。

---

## 4. 物理计算流验证

### 4.1 核心原则

物理计算流含融合（MulAddDst）、重排等优化，可能与原始算子的计算语义产生偏差。必须在进入编码前，用 numpy 验证优化版和原始版等价。

验证方法：**golden**（原始语义）+ **opt**（§3 优化伪码），全整数随机输入 100 次，`np.array_equal` 判等，不通过不进编码。

golden 的来源取决于算子类型：
- **有竞品（TF/Torch）**：以竞品公式为基准，直接形成 numpy golden 或直接跑竞品代码
- **无竞品、A2/A3 算子**：基于 A2/A3 源码的计算流，严格逐行翻译为 numpy golden

opt 始终按 §3 计算流伪码（含所有融合和重排）实现。

### 4.2 物理意义

融合和重排可能改变浮点累加顺序。全整数输入可规避累加顺序带来的数值差异，但仅对**单次二元操作**有效。

**Reduce 类操作的额外风险**：Reduce（如 sum、mean）对累加顺序敏感——不同的归约顺序在浮点下产生不同结果，即便整数验证通过也无法保证浮点等价。若算子含 Reduce，需单独提示用户评估累加顺序偏差是否可接受。

### 4.3 验证文件

| 文件 | 内容 |
|------|------|
| `adam_apply_one_assign_golden.py` | 源码逐行翻译，标注行号，不掺解读 |
| `adam_apply_one_assign_opt.py` | 按 §3 计算流伪码实现 |
| `compare_golden_opt.py` | 对比脚本 |

覆盖 6 种 shape 组合：同形 / 1D 广播 / 标量 / 混合 / 16×32 / 3D。

例：S7+S9 MulAddDst 融合的验证

```
golden (源码逐行):
  S7: mul0Result = dataInput2 * dataInputMul      // shape broadcast
  S9: output1      = mul0Result + mul1Result        // shape broadcast

opt (MulAddDst 融合):
  MulAddDst(dst=mul1Result, src0=dataInput2, src1=dataInputMul) → output1
```

全整数下 `np.array_equal(goldenOutput1, optOutput1) → True`，融合不改变整数语义。

### 4.4 结论

100 次随机整数输入，`np.array_equal` 三输出全部 True。**优化计算流与原始源码语义等价。**

### 4.5 对本算子的意义

MulAddDst 融合和调度重排已确认安全，§3 伪码可正式作为 kernel 实现的唯一参照，进入 §5 Tiling 详细设计。

---

## 5. Tiling 详细设计

### 5.1 输入预处理

#### 5.1.1 核心原则

输入预处理是几乎每个算子都要做的通用步骤。目标：将不同 shape 的入参和出参归一化，拉到同一个参照系 `maximumBroShape` 下。

预处理产出三类 shape：
- `maximumBroShape` — 坐标系，所有输入/输出的 broadcast 上界
- `normalInputShapes` — 归一化后各输入 shape，与 maximumBroShape 同 rank
- `normalOutputShapes` — 归一化后各输出 shape，与 maximumBroShape 同 rank 且必须稠密（每维等于该维 broadcast 最大值）。广播输出（某维为 1）经 stride=0 映射后多核 tile 落到同一 GM 地址构成写-写数据竞争，且本算子为就地 assign 语义、广播输出无定义，Host 侧直接拒绝

基本操作：
- **补 1**：低 rank 入参/输出在 shape 最前面补 1，拉齐 rank
- **去 1**：所有入参/输出都为 1 的轴 squeeze 掉，消除废维度
- **合轴**：将多个连续轴合并为一个轴，消除维度差异（如 (16,8) → (128,)）

预处理只操作 shape 元数据，不碰实际 tensor。在 host 侧一次性完成，device 侧只按统一坐标执行。

**参照系的确定**：由计算流伪码决定。不同算子类型推导参照系的方法不同——例如 broadcast 型算子取 broadcastMax(所有入参和输出)。

#### 5.1.2 物理意义

不同入参 rank 不同、废维不同、broadcast 关系不同。输出必须稠密——所有输出 shape 均等于 maximumBroShape（广播输出会造成多核写-写数据竞争，Host 侧拒绝）。预处理的产物（maximumBroShape + normalInputShapes + normalOutputShapes）就是这张映射表，device 侧按统一坐标迭代，通过各自 shape 映射回实际数据。

#### 5.1.3 去 1 补 1

##### 5.1.3.1 物理意义

入参和出参 rank 可能不同，且对齐后可能出现所有参数都为 1 的废维度。

补 1 约束：**只能在 shape 最前面补 1**——低 rank 入参/输出在 shape 最前面补 1 拉齐到 maxRank，不允许在中间或末尾维度补 1。

归一化后需验证 broadcast 兼容性：对每个维度，若某入参在该维的大小不为 1 且与其他入参同维不为 1 的大小不同，则输出报错日志；同时校验输出稠密性——该维存在非 1 值时输出不允许为 1，否则报错日志。

```
补 1 — 低 rank 在 shape 最前面补 1，拉齐到 maxRank（含入参+出参）
去 1 — 对齐后，所有入参+出参在某轴都是 1 → squeeze 掉
归一 — 去 1 后若 maximumBroShape 为空（全部均为标量），填 (1,)
校验 — 归一化后逐维检查：输入同维非 1 大小必须一致；输出必须稠密（该维存在非 1 值时输出不得为 1），违者输出报错日志

产出三样：
  maximumBroShape    = broadcastMax(所有入参+出参)，后续 UB/多核切分的坐标系
  normalInputShapes  = 每入参归一化后 shape，与坐标系同 rank
  normalOutputShapes = 每出参归一化后 shape，与坐标系同 rank 且稠密（= maximumBroShape）
```

例 1（正常广播，补 1 个 1）：

```
入参 A: shape = (4, 8)
入参 B: shape = (1, 8)
入参 C: shape = (8,)       // 在最前面补一个 1 → (1,8)

补 1 → A=(4,8), B=(1,8), C=(1,8)    // C 最前面补 1
去 1 → dim0: A=4, B=1, C=1 → 不全为 1，保留
       dim1: A=8, B=8, C=8 → 全为 8≠1，保留
maximumBroShape = (4, 8)
normal_A = (4, 8)   // 非 broadcast
normal_B = (1, 8)   // 轴 0 broadcast
normal_C = (1, 8)   // 轴 0 broadcast
校验 → dim0: 4,1,1 → ref=4 ✓
       dim1: 8,8,8 → ref=8 ✓
```

例 2（正常广播，补多个 1）：

```
入参 A: shape = (2, 4, 8)
入参 B: shape = (8,)       // 在最前面补两个 1 → (1,1,8)

补 1 → A=(2,4,8), B=(1,1,8)    // B 最前面补 1,1
去 1 → dim0: A=2, B=1 → 不全为 1，保留
       dim1: A=4, B=1 → 不全为 1，保留
       dim2: A=8, B=8 → 全为 8≠1，保留
maximumBroShape = (2, 4, 8)
normal_A = (2, 4, 8)   // 非 broadcast
normal_B = (1, 1, 8)   // 轴 0、轴 1 broadcast
校验 → dim0: 2,1 → ref=2 ✓
       dim1: 4,1 → ref=4 ✓
       dim2: 8,8 → ref=8 ✓
```

例 3（broadcast 不兼容报错）：

```
入参 A: shape = (4, 8)
入参 B: shape = (4, 3)     // 同 rank，无需补 1

补 1 → A=(4,8), B=(4,3)     // 同 rank，无需补 1
去 1 → dim0: 4,4 → 不全为 1，保留
       dim1: 8,3 → 不全为 1，保留
maximumBroShape = (4, 8)
normal_A = (4, 8)
normal_B = (4, 3)
校验 → dim0: 4,4 → ref=4 ✓
       dim1: 8,3 → 两者均非 1 且 8≠3 → 报错！
→ OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "input size and ref size", "dim 1 input[1] size 3 and 8", "broadcast incompatible: input sizes must be equal or 1")
→ return false
```

例 4（标量）：

```
入参 A: shape = ()    // 标量
入参 B: shape = ()    // 标量

补 1 → A=(), B=()     // rank=0，无轴可补
去 1 → maximumBroShape = ()
归一 → maximumBroShape = (1,)
normal_A = (1,)
normal_B = (1,)
校验 → dim0: 1,1 → 全为 1 ✓
```

##### 5.1.3.2 函数模块

```
函数名：PadAndSqueeze
文件归属：Tiling
作用：入参+出参归一化 — 最前面补 1 拉 rank、去 1 消废维、标量归一
in：   inputShapes            所有入参原始 shape
in：   outputShapes           所有出参原始 shape
out：  maximumBroShape       坐标系 → TilingData
out：  normalInputShapes     归一化后各入参 shape → TilingData
out：  normalOutputShapes    归一化后各出参 shape → TilingData
```

```
函数名：CheckBroadcastShape
文件归属：Tiling
作用：broadcast 兼容校验 — 归一化后逐维检查输入同维非 1 大小是否一致、输出是否稠密，违者报错
in：   paddedIn     补 1 后各入参 shape
in：   paddedOut    补 1 后各出参 shape
in：   maxRank      最大 rank
ret：  bool          true=兼容，false=不兼容
校验： 输入同维非 1 大小必须一致；输出必须稠密（该维存在非 1 值时输出不得为 1）。违者 OP_LOGE_FOR_INVALID_VALUES_WITH_REASON 报错并返回 false
```

```cpp
bool CheckBroadcastShape(
    const std::vector<std::vector<int64_t>>& paddedIn,
    const std::vector<std::vector<int64_t>>& paddedOut,
    int64_t maxRank)
{
    for (int64_t d = 0; d < maxRank; d++) {
        int64_t ref = -1;
        for (size_t i = 0; i < paddedIn.size(); i++) {
            if (paddedIn[i][d] != 1) {
                if (ref == -1) ref = paddedIn[i][d];
                else if (paddedIn[i][d] != ref) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "input size and ref size",
                        (std::string("dim ") + std::to_string(d) + " input[" + std::to_string(i) +
                         "] size " + std::to_string(paddedIn[i][d]) + " and " + std::to_string(ref)).c_str(),
                        "broadcast incompatible: input sizes must be equal or 1");
                    return false;
                }
            }
        }
        for (size_t i = 0; i < paddedOut.size(); i++) {
            if (paddedOut[i][d] != 1) {
                if (ref == -1) ref = paddedOut[i][d];
                else if (paddedOut[i][d] != ref) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "output size and ref size",
                        (std::string("dim ") + std::to_string(d) + " output[" + std::to_string(i) +
                         "] size " + std::to_string(paddedOut[i][d]) + " and " + std::to_string(ref)).c_str(),
                        "broadcast incompatible: output sizes must be equal or 1");
                    return false;
                }
            }
        }
        // 输出稠密性校验: 该维存在非 1 维值 (ref != -1) 时, 输出不允许为广播维 (1)。
        // 广播输出维 stride=0, 多核 tile 映射到同一 GM 地址并发写, 构成写-写数据竞争,
        // 且本算子为就地 assign 语义, 广播输出无定义, Host 侧直接拒绝。
        if (ref != -1) {
            for (size_t i = 0; i < paddedOut.size(); i++) {
                if (paddedOut[i][d] == 1) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "output size and ref size",
                        (std::string("dim ") + std::to_string(d) + " output[" + std::to_string(i) +
                         "] size 1 and " + std::to_string(ref)).c_str(),
                        "broadcast output is not supported: output must be dense (equal to broadcast max dim)");
                    return false;
                }
            }
        }
    }
    return true;
}

bool PadAndSqueeze(
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<std::vector<int64_t>>& outputShapes,
    std::vector<int64_t>&                    maximumBroShape,
    std::vector<std::vector<int64_t>>&       normalInputShapes,
    std::vector<std::vector<int64_t>>&       normalOutputShapes
) {
    int64_t numInputs  = (int64_t)inputShapes.size();
    int64_t numOutputs = (int64_t)outputShapes.size();

    // 合并所有 shape 找 maxRank
    int64_t maxRank = 0;
    for (auto& s : inputShapes)  maxRank = std::max(maxRank, (int64_t)s.size());
    for (auto& s : outputShapes) maxRank = std::max(maxRank, (int64_t)s.size());

    // 补 1 — 低 rank 在 shape 最前面补 1
    auto padShape = [&](const std::vector<int64_t>& s) {
        std::vector<int64_t> p;
        int64_t pad = maxRank - (int64_t)s.size();
        p.assign(pad, 1);
        p.insert(p.end(), s.begin(), s.end());
        return p;
    };
    std::vector<std::vector<int64_t>> paddedIn(numInputs), paddedOut(numOutputs);
    for (int64_t i = 0; i < numInputs;  i++) paddedIn[i]  = padShape(inputShapes[i]);
    for (int64_t i = 0; i < numOutputs; i++) paddedOut[i] = padShape(outputShapes[i]);

    // 去 1 — 所有入参+出参都为 1 的轴 squeeze，其余轴取 max
    maximumBroShape.clear();
    normalInputShapes.assign(numInputs, std::vector<int64_t>());
    normalOutputShapes.assign(numOutputs, std::vector<int64_t>());
    for (int64_t d = 0; d < maxRank; d++) {
        bool allOne = true;
        int64_t maxDim = 0;
        for (int64_t i = 0; i < numInputs;  i++) { if (paddedIn[i][d]  != 1) allOne = false; maxDim = std::max(maxDim, paddedIn[i][d]); }
        for (int64_t i = 0; i < numOutputs; i++) { if (paddedOut[i][d] != 1) allOne = false; maxDim = std::max(maxDim, paddedOut[i][d]); }
        if (!allOne) {
            maximumBroShape.push_back(maxDim);
            for (int64_t i = 0; i < numInputs;  i++) normalInputShapes[i].push_back(paddedIn[i][d]);
            for (int64_t i = 0; i < numOutputs; i++) normalOutputShapes[i].push_back(paddedOut[i][d]);
        }
    }

    // 归一：全标量 → (1,)
    if (maximumBroShape.empty()) {
        maximumBroShape.push_back(1);
        for (int64_t i = 0; i < numInputs;  i++) normalInputShapes[i].push_back(1);
        for (int64_t i = 0; i < numOutputs; i++) normalOutputShapes[i].push_back(1);
    }

    return true;
}
```

##### 5.1.3.3 验证文件

| 文件 | 内容 |
|------|------|
| `verify/test_pad_and_squeeze.cpp` | 去 1 补 1 自验证程序 |

覆盖：100 次随机 shape 组合（2~10 入参，rank 1~5，dim 1~16），不变量全部满足。✅

例（补 1 个 1）：
```
输入: A=(4,8), B=(1,8), C=(8,)

补 1 → C 最前面补 1 → A=(4,8), B=(1,8), C=(1,8)
去 1 → dim0: A=4, B=1, C=1 → 不全为1保留
       dim1: A=8, B=8, C=8 → 全8≠1保留

expected: maximumBroShape = (4,8)
          normal_A = (4,8), normal_B = (1,8), normal_C = (1,8)
actual:   maximumBroShape = (4,8) ✓
          normal_A = (4,8), normal_B = (1,8), normal_C = (1,8) ✓
校验 → dim0: 4,1,1 → ref=4 ✓; dim1: 8,8,8 → ref=8 ✓
```

例（补多个 1）：
```
输入: A=(2,4,8), B=(8,)

补 1 → B 最前面补 1,1 → A=(2,4,8), B=(1,1,8)
去 1 → dim0: A=2, B=1 → 不全为1保留
       dim1: A=4, B=1 → 不全为1保留
       dim2: A=8, B=8 → 全8≠1保留

expected: maximumBroShape = (2,4,8)
          normal_A = (2,4,8), normal_B = (1,1,8)
actual:   maximumBroShape = (2,4,8) ✓
          normal_A = (2,4,8), normal_B = (1,1,8) ✓
校验 → dim0: 2,1 → ref=2 ✓; dim1: 4,1 → ref=4 ✓; dim2: 8,8 → ref=8 ✓
```

例（broadcast 不兼容报错）：
```
输入: A=(4,8), B=(4,3)

补 1 → A=(4,8), B=(4,3)     // 同 rank，无需补 1
去 1 → dim0: 4,4 → 不全为1保留; dim1: 8,3 → 不全为1保留
校验 → dim1: A=8, B=3 均非 1 且 8≠3
→ OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("CheckBroadcastShape", "input size and ref size", "dim 1 input[1] size 3 and 8", "broadcast incompatible: input sizes must be equal or 1")
→ return false ✓
```

##### 5.1.3.4 结论

补 1 仅在 shape 最前面。去 1 消废维后，maximumBroShape 统一到 ≥1 rank，无废维度，标量已归一为 (1,)。normalInputShapes 和 normalOutputShapes 均与坐标系同 rank，输入 dim=1 标识 broadcast 轴，输出必须稠密（无 dim=1 轴）。归一化完成后由 CheckBroadcastShape 逐维校验 broadcast 兼容性（输入同维非 1 大小不一致、或输出存在广播维则报错）。

##### 5.1.3.5 对本算子的意义

maximumBroShape + normalInputShapes + normalOutputShapes 交付给 5.2 做 UB 切分坐标系，交付给 5.5 做各入参/出参的 GM 偏移和搬运量计算。broadcast 兼容校验在 Tiling 侧提前拦截非法 shape 组合，避免 kernel 侧运行时异常。

---

### 5.2 UB 切分

#### 5.2.1 核心原则

UB 切分解决"有效数据远超单 buffer 容量"的问题。以 maximumBroShape 为坐标系，选一根轴切分——内侧全带进 UB，外侧循环迭代。切分因子由物理存活节点 P 框定的 buffer 上限 + UB 最大利用率原则共同决定。

切分对象 = maximumBroShape（所有入参 broadcast 的最大坐标系）。目前仅含单切分——从最内轴向外扫描，算子用这一种即可实现功能全覆盖 + 部分场景性能最优。后续学习过程中会陆续引入新的切分算法。

#### 5.2.2 物理意义

UB 切分轴选择，三处依据汇到一个决策：

1. **切分对象 = maximumBroShape**。基于它切分兼容所有小 shape 场景（部分场景非最优性能，后续性能模板可替换）。
2. **单切分模型**。被切分轴 a = ubOuter × ubFactor。ubFactor 及所有内侧轴为「UB 内轴」；ubOuter 至最外侧轴为「UB 外轴」，是 UB 外的 for 循环。
3. **单 buffer 上限 = UB/P**。来自 §2 物理存活节点，ubFactor × inner ≤ perBufferElems。

#### 5.2.3 UB 单切分

##### 5.2.3.1 物理意义

选轴方法：从最内轴向最外轴扫描，累积内侧轴连乘 inner。当 dK × inner > perBufferElems 时，该轴无法全量随内侧轴放入 buffer——在此切分。

```
inner = 1
for k from rank-1 down to 0:
    if dK × inner > perBufElems:    → split at k,  ubFactor = perBufElems / inner
    inner ×= dK
```

边界：
- d_{n-1} > perBufElems → 在最末轴切分，ubFactor = perBufElems
- 全量 ≤ perBufElems → 走到 k=0 也不超，ubSplitIdx=0, ubFactor = d_0, ubOuter = 1
- 选轴永不失败——至少有一根可切轴

##### 5.2.3.2 函数模块

```
数据结构：SplitResult
文件归属：Tiling → TilingData
字段：
  ubSplitIdx        选中的切分轴
  ubFactor         内切因子
  ubOuter         外切因子（外循环轮数）
  ubTail    尾轮内切因子（dK % ubFactor，整除时为 ubFactor）
```

```cpp
struct SplitResult {
    int64_t ubSplitIdx;
    int64_t ubFactor;
    int64_t ubOuter;
    int64_t ubTail;
};
```

```
函数名：FindSplitAxis
文件归属：Tiling
作用：UB 单切分选轴 — 从最内轴向外扫描，dK × inner > perBufElems 时在 k 切分
in：   maximumBroShape     有效 shape（坐标系）
in：   dtypeSize    sizeof(dtype)
in：   ubPerCore   UB = 256 × 1024
in：   physNodes    物理存活节点数 P
out：  out           SplitResult → TilingData
```

```cpp
// Ops::Base::CeilDiv 需在 tiling 侧 #include "util/math_util.h"（位于 op_common/op_host 下）
void FindSplitAxis(
    const std::vector<int64_t>& maximumBroShape,
    int64_t                     dtypeSize,
    int64_t                     ubPerCore,
    int64_t                     physNodes,
    SplitResult &               out
) {
    int64_t perBufBytes = (ubPerCore / physNodes) & UB_ALIGN_MASK;  // 32B 对齐 (TBuf 硬件要求)
    int64_t perBufElems = perBufBytes / dtypeSize;
    int64_t rank = (int64_t)maximumBroShape.size();

    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (maximumBroShape[k] * inner > perBufElems) {
            out.ubFactor = perBufElems / inner;
            out.ubOuter = Ops::Base::CeilDiv(maximumBroShape[k], out.ubFactor);
            int64_t rem = maximumBroShape[k] % out.ubFactor;
            out.ubTail = (rem == 0) ? out.ubFactor : rem;
            out.ubSplitIdx = k;
            return;
        }
        if (k == 0) {
            out.ubSplitIdx = 0;
            out.ubFactor = maximumBroShape[0];
            out.ubOuter = 1;
            out.ubTail = maximumBroShape[0];
            return;
        }
        inner *= maximumBroShape[k];
    }
}
```

##### 5.2.3.3 验证文件

| 文件 | 内容 |
|------|------|
| `verify/test_find_split_axis.cpp` | FindSplitAxis 自验证程序 |

覆盖：8 固定 case + 100 次随机 shape（rank 1~5，dim 1~200000，FP16/FP32，P=3/5/7），不变量全部满足。✅

例：
```
输入：maximumBroShape=(200000, 200000), dtype=FP16, P=5
perBufElems = 256K/5 / 2 = 26214

k=1: d_1=200000 > 26214 → split at ubSplitIdx=1
     ubFactor = 26214/1 = 26214
     ubOuter = ceil(200000/26214) = 8
     ubTail = 200000 % 26214 = 16552

expected: ubSplitIdx=1, ubFactor=26214, ubOuter=8, ubTail=16552 ✓
```

##### 5.2.3.4 结论

选轴永不失败，ubFactor × inner ≤ perBufElems 恒成立。

##### 5.2.3.5 对本算子的意义

SplitResult 交付给 5.3 做多核切分、5.4 做 kernel 循环骨架。

---

### 5.3 多核切分

#### 5.3.1 核心原则

多核切分的轴天然在 UB 外轴上选择——UB 内轴是一个 tile 的原子单位，不可跨核拆分。切分顺序是先 UB 再 多核，两刀独立。核间无数据交互，纯 data-parallel，各自 UB 独立，遵守相同的物理存活节点约束。

#### 5.3.2 物理意义

三个要素：

```
要素 1 — UB 内外轴（FindSplitAxis 产出）

    maximumBroShape = (d_0, ... , d_{k-1},   dK,   d_{k+1}, ... , d_{n-1})
                                             ↑ 切分轴
    UB 外轴:  d_0 × ... × d_{k-1} × ubOuter     (ubOuter 含主/尾两段)
    UB 内轴:           ubFactor        × d_{k+1} × ... × d_{n-1}

    多核切分的候选轴天然落在 UB 外轴集合里——UB 内轴是一个 tile 的原子单位，
    不能被拆到多个核上。

要素 2 — 先切 UB，再切多核

    第一刀（UB 切分）：沿轴 k 切，产生 UB 主块（ubFactor）和 UB 尾块（ubTail）。
    第二刀（多核切分）：在 UB 外轴的迭代空间上按核分配。多核切分不关心 ubOuter 的
    主/尾边界——它对 UB 外轴的空间做划分，产生多核主块（正常核）和多核尾块（尾核）。

    两刀先后独立，第二刀不区分第一刀的内部结构。因此一个核的迭代区间内可能同时
    覆盖 UB 主块和 UB 尾块——产生四种组合：

    ┌────────────┬─────────────────────┬──────────────────────┐
    │            │  UB 主块 (ubFactor)       │  UB 尾块 (ubTail)    │
    ├────────────┼─────────────────────┼──────────────────────┤
    │ 多核主块    │ 主-主                │ 主-尾                 │
    │ (正常核)    │ ubFactor × inner 全量搬运  │ ubTail × inner 搬运  │
    ├────────────┼─────────────────────┼──────────────────────┤
    │ 多核尾块    │ 尾-主                │ 尾-尾                 │
    │ (尾核)      │ ubFactor × inner 全量搬运  │ ubTail × inner 搬运  │
    └────────────┴─────────────────────┴──────────────────────┘

    四种块的区别：搬运尺寸不同（ubFactor vs ubTail）× 迭代次数不同（主核 vs 尾核）。
```

#### 5.3.3 函数模块

```
数据结构：MultiCoreResult
文件归属：Tiling → TilingData
字段：
  usedCoreNum   实际使用的核数（≤ maxCores, ≤ totalTiles）
  totalTiles UB 外轴迭代空间的总 tile 数
  mainTiles  每核主块 tile 数 = CeilDiv(totalTiles, usedCoreNum)
  mainCoreNum  主核数量（前 mainCoreNum 核各处理 mainTiles 块，其余核各处理 mainTiles-1 块）
```

```cpp
struct MultiCoreResult {
    int64_t usedCoreNum;
    int64_t totalTiles;
    int64_t mainTiles;
    int64_t mainCoreNum;
};
```

```
函数名：MultiCoreSplit
文件归属：Tiling
作用：UB 外轴迭代空间按核均分 — 前 mainCoreNum 核各处理 mainTiles 块，其余核各处理 mainTiles-1 块
in：   maximumBroShape   有效 shape
in：   ubSplit    FindSplitAxis 输出
in：   maxCores   可用总核数（如 56）
out：  out         MultiCoreResult → TilingData
```

```cpp
bool MultiCoreSplit(
    const std::vector<int64_t>& maximumBroShape,
    const SplitResult &         ubSplit,
    int64_t                     maxCores,
    MultiCoreResult &           out
) {
    int64_t k = ubSplit.ubSplitIdx;
    int64_t outerProd = 1;
    for (int64_t j = 0; j < k; j++) {
        outerProd *= maximumBroShape[j];
    }
    out.totalTiles = outerProd * ubSplit.ubOuter;

    out.usedCoreNum = (out.totalTiles < maxCores) ? out.totalTiles : maxCores;
    out.mainTiles = Ops::Base::CeilDiv(out.totalTiles, out.usedCoreNum);
    out.mainCoreNum = out.totalTiles - (out.mainTiles - 1) * out.usedCoreNum;

    return true;
}
```

#### 5.3.4 验证文件

| 文件 | 内容 |
|------|------|
| `verify/test_multi_core_split.cpp` | MultiCoreSplit 自验证程序 |

覆盖：8 固定 case + 100 次随机（rank 2~5，dim 1~1000，maxCores 1~56），不变量全部满足。✅

例：
```
输入：maximumBroShape=(100, 200, 300), ubSplit.ubSplitIdx=1, ubSplit.ubOuter=8, maxCores=56
outerProd = 100, totalTiles = 100 × 8 = 800

usedCoreNum = min(800, 56) = 56
mainTiles = ceil(800 / 56) = 15
mainCoreNum = 800 - (15-1) × 56 = 16

expected: usedCoreNum=56, totalTiles=800, mainTiles=15, mainCoreNum=16 ✓
  验证：15 × 16 + 14 × 40 = 240 + 560 = 800 = totalTiles ✓
```

#### 5.3.5 结论

totalTiles > 0，usedCoreNum ≤ maxCores 且每核 tile 数差 ≤ 1，核间负载均衡。

#### 5.3.6 对本算子的意义

MultiCoreResult 交付给 5.4 做 kernel 循环骨架（GetCoreRange）。

---

### 5.4 Kernel 循环骨架

#### 5.4.1 核心原则

§5.2 切出了 UB 内/外轴，§5.3 把 UB 外轴空间按核均分。kernel 侧需要回答两个问题：

1. **UB 外循环怎么跑** — 每个核分别跑哪些 tile？
2. **每轮用主块还是尾块** — 这轮用 `ubFactor` 还是 `ubTail`？

答案来自两个函数：`GetCoreRange` 给出纵向区间，`GetUBSplitRange` 给出横向段长，两者正交产生四种块，不需预分类。

#### 5.4.2 物理意义

**UB 外循环 = flatten 迭代**

UB 外轴空间打平为一维，`ubOuter` 是最内维：

```
flatten 空间 = (d_0 × d_1 × ... × d_{k-1}) × ubOuter
                                              ↑ ubOuter 是最内循环
```

`MultiCoreResult` 已算出每核 tile 区间。`GetCoreRange(coreId)` 算出 `[start, end)`。

**主/尾块判定 = `flat % ubOuter`**

不需要坐标映射。`ubOuter` 是 flatten 空间的最内维，直接取模：

```
ubBlockIdx = flat % ubOuter
  → ubBlockIdx == ubOuter - 1 → UB 尾块（ubTail）
  → 其余              → UB 主块（ubFactor）
```

**四种块是两套判定的自然结果**

纵向 `GetCoreRange`（多核主/尾）× 横向 `GetUBSplitRange`（UB 主/尾），产生四种组合。kernel 不需要预分类，迭代到具体 flat 时两套判定正交生效即可。

#### 5.4.3 函数模块

##### GetCoreRange

```
函数名：GetCoreRange
文件归属：Kernel
作用：每核 flatten 迭代范围，前 mainCoreNum 核各处理 mainTiles 块，其余核各处理 mainTiles-1 块
in：   coreId      当前核编号
in：   usedCoreNum    总核数
in：   totalTiles  总 tile 数
in：   mainTiles   每核主块 tile 数
in：   mainCoreNum   主核数
out：  start, end   该核 [start, end) tile 区间
```

```cpp
inline void GetCoreRange(
    int64_t  coreId,
    int64_t  usedCoreNum,
    int64_t  totalTiles,
    int64_t  mainTiles,
    int64_t  mainCoreNum,
    int64_t& start,
    int64_t& end
) {
    if (coreId < mainCoreNum) {
        start = coreId * mainTiles;
        end   = start + mainTiles;
    } else {
        start = mainCoreNum * mainTiles
              + (coreId - mainCoreNum) * (mainTiles - 1);
        end   = start + mainTiles - 1;
    }
}
```

##### GetUBSplitRange

```
函数名：GetUBSplitRange
文件归属：Kernel
作用：本轮切分轴段长，ubOuter 最后一段用 ubTail，其余用 ubFactor
in：   ubBlockIdx    当前 flat 在 ubOuter 上的段号
in：   ubOuter        ubOuter 总段数
in：   ubFactor        切分轴主段长
in：   ubTail   切分轴尾段长
ret：  切分轴段长（非元素总数）
```

```cpp
inline int64_t GetUBSplitRange(
    int64_t ubBlockIdx,
    int64_t ubOuter,
    int64_t ubFactor,
    int64_t ubTail
) {
    return (ubBlockIdx == ubOuter - 1) ? ubTail : ubFactor;
}
```

##### 主循环

```
// 入参：SplitResult + MultiCoreResult

int64_t start, end;
GetCoreRange(coreId, usedCoreNum, totalTiles, mainTiles, mainCoreNum, start, end);

for (int64_t flat = start; flat < end; flat++) {
    int64_t ubBlockIdx  = flat % ubOuter;
    int64_t ubBlockLength  = GetUBSplitRange(ubBlockIdx, ubOuter, ubFactor, ubTail);  // 切分轴段长

    // DataCopy GM→UB (偏移计算见 §5.5)
    // 执行计算流
    // DataCopy UB→GM
}
```

#### 5.4.4 验证文件

| 文件 | 内容 |
|------|------|
| `verify/test_tile_iter_control.cpp` | GetCoreRange + GetUBSplitRange 自验证程序 |

覆盖：8 固定 case + 100 次随机 shape（rank 2~5，dim 1~20，FP16/FP32，P=3/5/7，maxCores 1~56），不变量全部满足。✅

例：
```
输入：maximumBroShape=(4, 7), ubSplitIdx=1, ubOuter=3, ubFactor=3, ubTail=1
      totalTiles=4×3=12, mainTiles=ceil(12/5)=3, mainCoreNum=12-(3-1)×5=2, usedCoreNum=5, coreId=0

GetCoreRange(0, 5, 12, 3, 2):
  coreId=0 < mainCoreNum=2 → start=0×3=0, end=3
  该核跑 flat ∈ [0, 3)

flat=0: ubBlockIdx=0, GetUBSplitRange(0, 3, 3, 1)=3  (主段, 搬 3×inner)
flat=1: ubBlockIdx=1, GetUBSplitRange(1, 3, 3, 1)=3  (主段, 搬 3×inner)
flat=2: ubBlockIdx=2, GetUBSplitRange(2, 3, 3, 1)=1  (尾段, 搬 1×inner, ubTail≠ubFactor)
  → 核 0：2 个主段 + 1 个尾段 = 3 tiles ✓
```

#### 5.4.5 结论

GetCoreRange + GetUBSplitRange 两函数覆盖 kernel 循环全部跑法，四种块无需预分类。

#### 5.4.6 对本算子的意义

主循环伪码直接嵌入 kernel 实现，§5.5 在此基础上补完地址偏移。

---

### 5.5 地址偏移计算

#### 5.5.1 核心原则

kernel 循环在 output shape 坐标系上迭代（`flat`、`ubBlockLength`）。DataCopy 要的是每个 tensor 在自己 shape 空间里的 GM 始址和搬运量。对输出——输出稠密（shape=maximumBroShape），需要 stride 预计算完成坐标系到 GM 偏移的线性映射。对输入——坐标系不同，需要从 output shape 坐标映射回各入参自身 shape：broadcast 轴 dim=1 锁 0，其余轴照搬。映射逻辑在 Tiling 侧预计算为 stride（broadcast 轴设 0），kernel 侧只剩乘加。

#### 5.5.2 物理意义

地址计算分两路：

- **输出**：输出稠密，normalOutputShape=maximumBroShape。effCoord 通过预计算 stride 映射为 GM 偏移。搬运量按输出自身 shape 计算。
- **输入**：每个入参有自己的 normalInputShape，需要从 maximumBroShape 坐标映射回入参实际坐标——broadcast 轴 dim=1 锁 0，stride 预计算（broadcast 轴设 0）。搬运量按入参自身 shape 计算，切分轴 broadcast 则只搬 1 份。

#### 5.5.3 函数模块

##### FlatToEffectiveCoord

```
函数名：FlatToEffectiveCoord
文件归属：Kernel
作用：flat → 多维坐标
in：   flat          当前 flat tile 索引
in：   maxBroShape 坐标系指针 (int64_t[RANK])
in：   rank          维度数
in：   ubSplitIdx    切分轴 k
in：   ubFactor           切分轴主段长
in：   ubOuter           ubOuter 段数
out：  effCoord     输出坐标 (int64_t[RANK])
```

```cpp
bool FlatToEffectiveCoord(
    int64_t flat, const int64_t* maxBroShape, int64_t rank,
    int64_t ubSplitIdx, int64_t ubFactor, int64_t ubOuter,
    int64_t* effCoord)
{
    for (int64_t d = 0; d < rank; d++) effCoord[d] = 0;
    int64_t ubBlockIdx = flat % ubOuter;
    int64_t outer   = flat / ubOuter;
    for (int64_t d = ubSplitIdx - 1; d >= 0; d--) {
        effCoord[d] = outer % maxBroShape[d];
        outer /= maxBroShape[d];
    }
    effCoord[ubSplitIdx] = ubBlockIdx * ubFactor;
    return true;
}
```

例：maximumBroShape=(4,6,5)，切 k=1，ubFactor=2，ubOuter=3

```
flatten 空间 = d_0 × ubOuter = 4 × 3 = 12 个 tile

flat=0: ubBlockIdx=0, outer=0 → coord=(0, 0, 0)
flat=1: ubBlockIdx=1, outer=0 → coord=(0, 2, 0)
flat=2: ubBlockIdx=2, outer=0 → coord=(0, 4, 0)
flat=3: ubBlockIdx=0, outer=1 → coord=(1, 0, 0)
flat=4: ubBlockIdx=1, outer=1 → coord=(1, 2, 0)
...
flat=11: ubBlockIdx=2, outer=3 → coord=(3, 4, 0)

outer 用 radix 展开还原轴 0 坐标：outer=1 → coord[0]=1%4=1
ubBlockIdx 乘 ubFactor 得轴 1 偏移：ubBlockIdx=1 → coord[1]=1×2=2
内侧轴(轴 2)填 0：coord[2]=0
```

##### 输入地址 — GM 偏移

`effCoord` → 入参 GM 元素偏移。broadcast 轴 dim=1 → 坐标锁 0。stride 在 Tiling 侧预计算（broadcast 轴设 0），kernel 侧纯乘加。GlobalTensor::operator[] 取元素索引，非字节。

```
函数名：PrecomputeStrides
文件归属：Tiling
作用：预计算某个入参/输出的 stride，broadcast 轴设 0（输入与输出规则相同，同一函数复用）
in：   normalShape  该入参/输出归一化后 shape
out：  strides        broadcast 轴 = 0，其余 = Π_{j>d} normalShape[j] → TilingData
```

```cpp
bool PrecomputeStrides(
    const std::vector<int64_t>& normalShape,
    std::vector<int64_t>&       strides
) {
    int64_t rank = (int64_t)normalShape.size();
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (normalShape[d] == 1) {
            strides[d] = 0;
        } else {
            int64_t prod = 1;
            for (int64_t j = d + 1; j < rank; j++) {
                prod *= normalShape[j];
            }
            strides[d] = prod;
        }
    }
    return true;
}
```

例：normalInputShape = (4, 1, 8)，rank=3

```
d=2: dim=8≠1 → stride[2] = Π_{j>2} = 1        (最内侧，右侧无轴)
d=1: dim=1 → stride[1] = 0                    (broadcast 轴)
d=0: dim=4≠1 → stride[0] = Π_{j>0} = 1×8 = 8

strides = [8, 0, 1]

对 effCoord=(2, 0, 5):
  offset = 2×8 + 0×0 + 5×1 = 21 elements  (轴1的0×0自动归零，不受effCoord[1]影响)
```

```
函数名：CalcOffset
文件归属：Kernel
作用：有效坐标 + stride → GM 元素偏移（GlobalTensor::operator[] 取元素索引，非字节）
      输入与输出规则相同，同一函数复用
in：   effCoord    坐标数组 (int64_t[RANK])
in：   strides      stride 数组 (int64_t[RANK])
in：   rank         维度数
ret：  GM 元素偏移
```

```cpp
int64_t CalcOffset(
    const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++) offset += effCoord[d] * strides[d];
    return offset;  // 元素个数
}
```

##### 输入地址 — 搬运元素数

`ubBlockLength` 是 maximumBroShape 切分轴上的段长。入参在该轴可能只有 1 个元素（broadcast），实际只需搬 1 份，由 NDDMA 展开。内侧轴同理按入参自身 shape。

```
切分轴搬运量 = (normalInputShape[k] == 1) ? 1 : ubBlockLength
内侧轴搬运量 = Π_{d>k} normalInputShape[d]
搬运总数     = 切分轴搬运量 × 内侧轴搬运量
```

```
函数名：CalcTransferCount
文件归属：Kernel
作用：本轮搬运元素数（broadcast 轴只搬 1 份；输入与输出规则相同，同一函数复用）
in：   normalShape  入参/输出归一化 shape (int64_t[RANK])
in：   rank          维度数
in：   ubSplitIdx    切分轴 k
in：   ubBlockLength       本轮切分轴段长
ret：  搬运元素数
```

```cpp
int64_t CalcTransferCount(
    const int64_t* normalShape, int64_t rank, int64_t ubSplitIdx, int64_t ubBlockLength)
{
    int64_t splitElems = (normalShape[ubSplitIdx] == 1) ? 1 : ubBlockLength;
    int64_t innerElems = 1;
    for (int64_t d = ubSplitIdx + 1; d < rank; d++) innerElems *= normalShape[d];
    return splitElems * innerElems;
}
```

##### 输出地址 — 搬运元素数

输出搬运量的逻辑和输入一路对称：输出稠密（shape=maximumBroShape），切分轴与内侧轴照常参与搬运量计算。

```
切分轴搬运量 = (normalOutputShape[k] == 1) ? 1 : ubBlockLength
内侧轴搬运量 = Π_{d>k} normalOutputShape[d]
搬运总数     = 切分轴搬运量 × 内侧轴搬运量
```

```
函数名：CalcOutputTransferCount → 已并入 CalcTransferCount
文件归属：Kernel
作用：输出本轮搬运元素数——与输入同一规则，复用 CalcTransferCount，不再单独定义
```

##### 输出地址 — GM 偏移

输出稠密，shape=maximumBroShape（广播输出在 Host 侧已被 CheckBroadcastShape 拒绝）。输出 stride 即坐标系自身的线性映射，直接复用 `PrecomputeStrides`，不再单独定义。GM 偏移同理复用 `CalcOffset`。

```
函数名：CalcOffset（输出侧复用）
```

##### 主循环

```cpp
// strides 已在 Tiling 侧预计算好

int64_t start, end;
GetCoreRange(coreId, usedCoreNum, totalTiles, mainTiles, mainCoreNum, start, end);

for (int64_t flat = start; flat < end; flat++) {
    int64_t ubBlockIdx  = flat % ubOuter;
    int64_t ubBlockLength  = GetUBSplitRange(ubBlockIdx, ubOuter, ubFactor, ubTail);

    std::vector<int64_t> effCoord;
    FlatToEffectiveCoord(flat, maximumBroShape, ubSplitIdx, ubFactor, ubOuter, effCoord);

    // 入参搬入
    // DataCopy(CalcOffset(effCoord, stridesInput0),
    //          GM→UB, CalcTransferCount(normalInput0, ubSplitIdx, ubBlockLength), ...)
    // ...

    // 执行计算流（见 §3）

    // 出参搬出
    // DataCopy(CalcOffset(effCoord, stridesOutput),
    //          UB→GM, CalcTransferCount(normalOutput, ubSplitIdx, ubBlockLength), ...)
}
```

#### 5.5.4 验证文件

| 文件 | 内容 |
|------|------|
| `verify/test_address_offset.cpp` | FlatToEffectiveCoord + 输入/输出 GM 偏移 + CalcTransferCount 自验证程序 |

覆盖：FlatToEffectiveCoord 6 hand-computed + 100 随机；Strides/Offset 100 随机；Input/Output TransferCount 各 2 hand-computed + 100 随机；broadcast zeroing + output dim=1 stride。6 组全部 PASS。✅

#### 5.5.5 结论

8 个函数覆盖地址计算全流程：flat → 坐标（公共）→ GM 偏移 + 搬运量（输入/输出两路）。输入 stride 预计算消除 kernel 侧 broadcast 分支。

#### 5.5.6 对本算子的意义

地址计算是 kernel 循环的最后一块拼图。与 5.1~5.4 的产出合在一起，kernel 实现的所有参数（坐标、偏移、搬运量、循环区间）全部就位。

---

### 5.6 TilingData 统合

#### 5.6.1 核心原则

§5.6 的目标是产出一套可直接合入 CANN 算子仓的 Tiling 代码。本节以 ops-math（CANN 数学类算子仓）为目标仓库，wiki 读完本节后应能独立完成：生产 Tiling 文件及目录 → ops-math 本地 UT 验证通过 → 提交 PR。

**目标仓库**：`cann/ops-math`，目录范式 `math/{算子名}/`

**关键技术**（两项）：

1. **双档 TilingData** — 所有场景都走 8 维 `maxBroShape[8]` 太臃肿。以 rank 为分界：rank≤4 用 `AdamTilingData<4>`（120 int64），rank≥5 用 `AdamTilingData<8>`（228 int64）。两个 struct 大小 → 两个 kernel 实例 → TilingData 体积随实际场景压缩。

2. **Tiling 模板编程** — 取代传统的 `TILING_KEY_IS` 写法。CANN 标准体系：`ASCENDC_TPL_ARGS_DECL` 声明模板参数 RANK ∈ {4,8}，Host 侧 `GET_TPL_TILING_KEY(RANK=mapped)` 生成 key，Kernel 侧 `template<int32_t RANK>` + `if constexpr` 编译期分支——不再有 `TILING_KEY_IS(1)`、`TILING_KEY_IS(2)` 这种看不出含义的魔术数字（参考 `Tiling模板编程` 文档）。`adam_apply_one_assign_struct.h` 是此项技术在 Adam 算子上的落地。

**交付物**：

| 文件 | 位置 | 内容 |
|------|------|------|
| `CMakeLists.txt` | `math/{op}/` | `add_all_modules_sources(OPTYPE ... COMPUTE_UNIT ascend950 TILING_DIR arch35)` 注册算子到 cmake 体系 |
| `adam_apply_one_assign_struct.h` | `op_kernel/arch35/` | Tiling 模板参数：`ASCENDC_TPL_ARGS_DECL` 声明 RANK ∈ {4,8}，`ASCENDC_TPL_SEL` 声明 2 组合法组合 |
| `adam_apply_one_assign_tiling_struct.h` | `op_kernel/arch35/` | `AdamTilingData<RANK>` C++ POD 模板，`GetTilingData<AdamTilingData<R>>()` 读写，Kernel 侧 `GET_TILING_DATA_WITH_STRUCT` 消费 |
| `adam_apply_one_assign_arch35.h` | `op_host/arch35/` | 函数声明 + `AdamTiling` 类声明（对接 CANN 框架） |
| `adam_apply_one_assign_arch35.cpp` | `op_host/arch35/` | CANN 主线：`GetShapeInfo`（PadAndSqueeze 一次）+ `DoTilingAndSet<R>`（模板填充）+ `GET_TPL_TILING_KEY` + `TilingFunc` + `TilingPrepare`（空实现）+ `IMPL_OP_OPTILING` 注册 |
| `test_adam_tiling.cpp` | `tests/ut/op_host/arch35/` | UT，使用 ops-math 框架 `TilingContextPara` + `ExecuteTestCase` |

**CANN 框架对接**：
```
IMPL_OP_OPTILING(OpName).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingPrepare)
```
- `TilingFunc(gert::TilingContext*)`：运行期入口。从 context 取 shape/dtype → PadAndSqueeze 得实际 rank → 映射为 RANK=4 或 8 → `GetTilingData<AdamTilingData<R>>()` 写入 → `GET_TPL_TILING_KEY(RANK=R)` 生成 tiling key → `SetTilingKey` → `SetBlockDim` → 设 workspace
- `TilingPrepare(gert::TilingParseContext*)`：编译期入口，空实现（CompileInfo 为空结构）
- 平台参数（`coreNum`/`ubSize`）：`TilingFunc` 内通过 `GetPlatformInfo()` + `PlatformAscendC` 现场获取，不经 CompileInfo 中转

**验证标准**：在 ops-math 仓根目录执行 `bash build.sh -u --ops={op} --soc=ascend950`，编译 + UT 全部 PASS，UT 用例覆盖小 shape 无切分 + 大 shape 切分多核。

**最终动作**：代码合入 `examples/{算子}/code/`（同时可直接 copy 到 ops-math 仓的 `math/{算子}/`），提交 PR 到 `cann/ops-math`。

#### 5.6.2 物理意义

TilingData 是 §4（计算流验证）和 §5.1–§5.5（详细设计）全部决策的物化。下半句同样重要：**这些决策最终以 CANN 框架规定的物理形式落盘为代码文件和 UT。**

**1. 文件层的物理意义 — `arch35/` 子目录**

不同芯片 UB 大小、核数不同。`op_host/arch35/` 是 ascend950 的专属 Tiling 代码目录。`CMakeLists.txt` 通过 `SUPPORT_COMPUTE_UNIT`/`SUPPORT_TILING_DIR` 这组 KV 对做"芯片→代码目录"的映射。新芯片来了，加一对 KV、新建一个 `archXX/` 目录即可，不会影响已有芯片的逻辑。

**2. 函数层的物理意义 — GetShapeInfo / DoTilingAndSet<R> / RunTiling 链**

CANN 路径的主线不是某个"纯逻辑"函数一次搞定——因为运行期 PadAndSqueeze 之后才知道 rank，而 C++ 无法根据运行期 rank 返回不同大小的 struct。真正的做法是单次归一化、映射、模板填充：

`GetShapeInfo`：从 context 取原始 shape/dtype，调 `PadAndSqueeze` 一次，归一化 shape 和 rank 全存为成员变量。这是链的起点——后续所有步骤都基于这次产出。

`RunTiling`：将实际 rank（1~8）映射为 4 或 8，调用 `DoTilingAndSet<4>()` 或 `<8>()`。映射是物理必要的——只有两个 kernel 实例，不是八个。

`DoTilingAndSet<R>()`：模板方法，编译期 R 确定数组大小。直接用 `GetShapeInfo` 存好的归一化 shape 做 FindSplitAxis/MultiCoreSplit/stride/打包，写 `GetTilingData<AdamTilingData<R>>()`、`SetBlockDim`、`GET_TPL_TILING_KEY(RANK=R)`。

`平台参数`：`TilingPrepare` 为空实现（CompileInfo 为空结构，仅保留注册挂载点）——平台常数在 `DoTilingAndSet<R>` 内直接 `GetPlatformInfo()` + `PlatformAscendC` 取 `coreNum`/`ubSize`（零值守卫），不经 CompileInfo 中转。若后续需要 binary 复用的跨次缓存字段，挂到 CompileInfo 并在 TilingParse 中解析。

**3. TilingData 的物理意义 — POD 约束的根因**

`GetTilingData<T>()` 返回的指针指向 host→device 共享内存。不是 C++ 语法限制，是物理传输限制：这块内存要跨进程/跨地址空间传递，指针和虚表在目标侧无意义。Kernel 侧 `GET_TILING_DATA` 拿到的是同一块数据的只读视图。

**4. UT 的物理意义 — TilingContextPara = 无芯片的 context**

ops-math 的 Tiling UT 统一用 `TilingContextPara` 构造一个 fake context——填入 shape、dtype 等参数，就能调用 `TilingFunc` 走完整 Tiling 流程。`ExecuteTestCase` 验证返回值、tilingKey、workspace。不需要真实芯片和 CANN runtime，直接 g++ 编译 + 本地运行。

#### 5.6.3 工程目录结构

算子上库 ops-math 仓的标准目录范式如下（对标 `math/add/` 结构）。wiki 产出的代码文件应直接可放入此目录树，通过 `bash build.sh -u --ops=adam_apply_one_assign --soc=ascend950` 编译并运行 UT：

```
math/adam_apply_one_assign/
├── CMakeLists.txt                                # 算子注册到 cmake 体系
├── op_kernel/
│   └── arch35/
│       ├── adam_apply_one_assign_struct.h                         # TilingKey 模板参数（RANK ∈ {4,8}）
│       └── adam_apply_one_assign_tiling_struct.h                  # AdamTilingData<RANK> 模板
├── op_host/
│   └── arch35/
│       ├── adam_apply_one_assign_arch35.h                  # 函数声明 + AdamTiling 类
│       └── adam_apply_one_assign_arch35.cpp                # 全量实现 + CANN 注册
└── tests/
    └── ut/
        └── op_host/
            └── arch35/
                └── test_adam_tiling.cpp          # Tiling UT
```

六份文件的职责：

| 文件 | 谁读 | 内容 |
|------|------|------|
| `CMakeLists.txt` | cmake | `add_all_modules_sources(OPTYPE adam_apply_one_assign ...)` 注册算子 |
| `adam_apply_one_assign_struct.h` | Tiling + Kernel | `ASCENDC_TPL_ARGS_DECL` 定义 RANK 模板参数（4 和 8），`ASCENDC_TPL_SEL` 声明 2 组合法组合 |
| `adam_apply_one_assign_tiling_struct.h` | Tiling + Kernel | `AdamTilingData<RANK>` C++ POD 模板，`RANK=4`（120 int64）或 `RANK=8`（228 int64），Kernel 侧 `GET_TILING_DATA_WITH_STRUCT` 消费 |
| `adam_apply_one_assign_arch35.h` | Tiling 实现 | 函数模块声明 + `AdamTiling` 类声明（`GetShapeInfo` / `DoTilingAndSet<R>` / `RunTiling`） |
| `adam_apply_one_assign_arch35.cpp` | Tiling 实现 | CANN 主线：`GetShapeInfo`（归一化一次）+ `DoTilingAndSet<R>`（模板填充）+ `GET_TPL_TILING_KEY` + `TilingFunc` + `TilingPrepare`（空实现）+ `IMPL_OP_OPTILING` 注册 |
| `test_adam_tiling.cpp` | UT | `TilingContextPara` + `ExecuteTestCase` 验证 Tiling 全流程 |

以下逐文件说明关键代码。

#### 5.6.4 CMakeLists.txt

`add_all_modules_sources` 是 ops-math 统一入口。`SUPPORT_COMPUTE_UNIT`/`SUPPORT_TILING_DIR` 这组 KV 对建立"芯片→代码目录"映射——新芯片加一对 KV、新建对应 `archXX/` 目录即可：

```cmake
set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_all_modules_sources(OPTYPE adam_apply_one_assign ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP FALSE)
```

`DISABLE_IN_OPP FALSE` 确保本地 UT 编译时 Tiling 源文件被纳入。`ACLNNTYPE aclnn_exclude` 表示本算子暂无 aclnn API。

#### 5.6.5 op_kernel/arch35/ — TilingKey 模板参数 + TilingData

`op_kernel/arch35/` 下两个头文件，分别定义 TilingKey 模板参数和 TilingData 结构体。

##### adam_apply_one_assign_struct.h — TilingKey 模板参数

按 CANN Tiling 模板编程规范，用 `ASCENDC_TPL_ARGS_DECL` 将 RANK 声明为模板参数。只设两个值——`RANK_4`（覆盖实际 rank=1~4）和 `RANK_8`（覆盖实际 rank=5~8）。Host 侧按实际 rank 映射后调用 `GET_TPL_TILING_KEY(RANK=mapped)`。Kernel 侧 `template<int32_t RANK>` `if constexpr` 编译期分支，**完全不用 `TILING_KEY_IS`**。2 个 tiling key → 2 个 kernel 模板实例。

```cpp
// math/adam_apply_one_assign/op_kernel/arch35/adam_apply_one_assign_struct.h
#include "ascendc/host_api/tiling/template_argument.h"

#define ADAM_RANK_4 4
#define ADAM_RANK_8 8

ASCENDC_TPL_ARGS_DECL(AdamApplyOneAssign,
    ASCENDC_TPL_UINT_DECL(RANK, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, ADAM_RANK_4, ADAM_RANK_8)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ADAM_RANK_4)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ADAM_RANK_8))
);
```

##### adam_apply_one_assign_tiling_struct.h — TilingData

POD 约束的物理根因：`GetTilingData<T>()` 返回的指针指向 host→device 共享内存。Tiling 运行期先调 `PadAndSqueeze` 归一化得到实际 rank（1~8），再映射为模板参数 RANK=4 或 8——此时才确定选 `AdamTilingData<4>`（120 int64）还是 `AdamTilingData<8>`（228 int64），以及 `GET_TPL_TILING_KEY(RANK=4)` 还是 `RANK=8`。归一化是第一环，切分、分核、stride、选 struct、选 key 全依赖它的产出。

```
数据结构：AdamTilingData<RANK>
字段：
  split         SplitResult        (§5.2) UB 切分结果
  multicore     MultiCoreResult    (§5.3) 多核切分结果
  rank          int64_t            (§5.1) 实际 rank（1~8），Host 侧写入供 dump 对齐，Kernel 不读取（用 RANK 模板参数）
  perBufBytes int64_t            (§2) UB/P 向下对齐 32B，Kernel 用此初始化 TBuf
  maxBroShape int64_t[RANK]     (§5.1) 坐标系
  numInputs    int64_t                    实际入参数
  numOutputs   int64_t                    实际出参数
  inputShapes  int64_t[10][RANK] (§5.1) 每入参归一化 shape
  inputStrides int64_t[10][RANK] (§5.5) 每入参 stride
  outputShapes int64_t[3][RANK]  (§5.1) 每出参归一化 shape
  outputStrides int64_t[3][RANK] (§5.5) 每出参 stride
```

```cpp
// math/adam_apply_one_assign/op_kernel/arch35/adam_apply_one_assign_tiling_struct.h
#ifndef ADAM_APPLY_ONE_ASSIGN_TILING_STRUCT_H_
#define ADAM_APPLY_ONE_ASSIGN_TILING_STRUCT_H_
#include <cstdint>

// 防御性初值（NSDMI）：计数/块数类字段默认 1（单块、单核退化配置），索引默认 0，
// 避免 Host 侧异常路径（如空 shape 导致 FindSplitAxis 未写入）时 Kernel 读到未定义值
struct SplitResult {
    int64_t ubSplitIdx = 0;
    int64_t ubFactor    = 1;
    int64_t ubOuter     = 1;
    int64_t ubTail      = 1;
};

struct MultiCoreResult {
    int64_t usedCoreNum = 1;
    int64_t totalTiles  = 1;
    int64_t mainTiles   = 1;
    int64_t mainCoreNum = 1;
};

template<int64_t RANK>
struct AdamTilingData {
    SplitResult     split;
    MultiCoreResult multicore;
    int64_t         rank            = 1;  // 实际 rank (1~8)，Host 侧写入供 dump 对齐，Kernel 不读取（用 RANK 模板参数）
    int64_t         perBufBytes     = 0;  // UB / P，Kernel 用此初始化 TBuf
    int64_t         maxBroShape[RANK]                  = {0};
    int64_t         numInputs                          = 0;
    int64_t         numOutputs                         = 0;
    int64_t         inputShapes [10][RANK]             = {0};
    int64_t         inputStrides[10][RANK]             = {0};
    int64_t         outputShapes[3][RANK]              = {0};
    int64_t         outputStrides[3][RANK]             = {0};
};
// 实际使用：AdamTilingData<4> 或 AdamTilingData<8>
```

| RANK | 覆盖实际 rank | `maxBroShape` | TilingData 大小 |
|-------|-------------|-----------------|-----------------|
| 4 | 1~4 | 4 | 120 int64 |
| 8 | 5~8 | 8 | 228 int64 |

rank=1~4 的场景用 `AdamTilingData<4>`，相较 228 int64 节省约 48% 空间。rank≤4 的 dim 填不满 4 的用 0 补齐。

#### 5.6.6 op_host/arch35/ — Tiling 实现 + CANN 注册

`.h` 声明函数模块和 `AdamTiling` 类，`.cpp` 包含全量实现。CANN 主线是单次归一化→映射→模板填充，`PadAndSqueeze` 只跑一次。

**CANN 主线** — `GetShapeInfo` 归一化 → `RunTiling` 映射 → `DoTilingAndSet<R>` 填充：

`GetShapeInfo()` 从 context 取原始 shape/dtype，调 `PadAndSqueeze` 一次，产出的归一化 shape 和 rank 全存为成员变量。`RunTiling()` 将实际 rank 映射为模板参数 RANK=4 或 8，调用 `DoTilingAndSet<R>()`——该模板方法直接用存好的归一化 shape 做 FindSplitAxis、MultiCoreSplit、stride 计算、打包，写 `GetTilingData<AdamTilingData<R>>()`、`SetBlockDim`、`GET_TPL_TILING_KEY(RANK=R)`、workspace。PadAndSqueeze 不跑第二次。

Tiling 的核心逻辑由 `PadAndSqueeze`、`CheckBroadcastShape`、`FindSplitAxis`、`MultiCoreSplit`、`PrecomputeStrides` 这些无状态函数承载——它们与 CANN 框架零耦合，直接在 `DoTilingAndSet<R>` 中被调用。

**关键 1 — 平台参数直接从系统获取**。本算子无跨次编译缓存信息：`AdamCompileInfo` 为空结构，`TilingPrepareForAdam` 为空实现（恒返回成功），仅保留注册挂载点。平台常数（AIV 核数、UB 大小）在 Tiling 阶段通过 `ctx_->GetPlatformInfo()` + `platform_ascendc::PlatformAscendC` 现场获取，任一为 0 视为环境异常直接失败。后续若做 binary 复用，缓存字段挂 CompileInfo 并在 TilingParse 中解析。

**关键 2 — TilingKey 模板化 + 双档 TilingData**。`PadAndSqueeze` 产出实际 rank（1~8）后，映射为 4 或 8，`DoTilingAndSet<R>()` 内 `GetTilingData<AdamTilingData<R>>()` 选对应大小的 struct，`GET_TPL_TILING_KEY(RANK=R)` 生成 key。Kernel 侧 `template<int32_t RANK>` + `if constexpr` 编译期分支，**完全不用 `TILING_KEY_IS`**：

```cpp
template<int32_t RANK>
__global__ __aicore__ void adam_apply_one_assign(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR input3, GM_ADDR input4,
    GM_ADDR mul0_x, GM_ADDR mul1_x, GM_ADDR mul2_x, GM_ADDR mul3_x, GM_ADDR add2_y,
    GM_ADDR out0, GM_ADDR out1, GM_ADDR out2,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (RANK == 4) {
        GET_TILING_DATA_WITH_STRUCT(AdamTilingData<4>, td, tiling);
    } else {
        GET_TILING_DATA_WITH_STRUCT(AdamTilingData<8>, td, tiling);
    }
    // td.rank → 实际 rank；for (int32_t d = 0; d < RANK; d++) — 编译期常量
}
```

2 个 tiling key → 2 种 TilingData 大小 → 2 个 kernel 模板实例。

```
数据结构：AdamCompileInfo
文件归属：op_host/arch35/adam_apply_one_assign_arch35.h → TilingParse
字段：无（空结构，仅作 TilingParse 注册的挂载类型；
      平台参数 coreNum/ubSize 在 Tiling 阶段直接从系统 GetPlatformInfo 获取，不经此缓存。
      后续若做 binary 复用，缓存字段挂此结构并在 TilingParse 中解析）
```

```
类名：AdamTiling
文件归属：op_host/arch35/adam_apply_one_assign_arch35.h
作用：CANN 主线——单次归一化→映射→模板填充
方法：
  RunTiling()           串联 GetShapeInfo → 映射 rank → DoTilingAndSet<R>
  GetShapeInfo()        从 context 取 shape/dtype，PadAndSqueeze 一次，存归一化结果和 rank
  DoTilingAndSet<R>()   模板方法，用存好的归一化 shape 做切分/分核/stride/打包，写 TilingData+BlockDim+Key
```

```cpp
// op_host/arch35/adam_apply_one_assign_arch35.cpp
#include "math/adam_apply_one_assign/op_kernel/arch35/adam_apply_one_assign_struct.h"

// GetShapeInfo() — PadAndSqueeze 一次，存归一化结果和 rank_
ge::graphStatus AdamTiling::GetShapeInfo() {
    for (size_t i = 0; i < ctx_->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        auto shape = ctx_->GetInputShape(i);
        const gert::Shape& s = shape->GetStorageShape();
        const size_t rank = s.GetDimNum();
        std::vector<int64_t> dims;
        for (size_t d = 0; d < rank; ++d) dims.push_back(s.GetDim(d));
        rawInputShapes_.push_back(dims);
    }
    for (size_t i = 0; i < ctx_->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        auto shape = ctx_->GetOutputShape(i);
        const gert::Shape& s = shape->GetStorageShape();
        const size_t rank = s.GetDimNum();
        std::vector<int64_t> dims;
        for (size_t d = 0; d < rank; ++d) dims.push_back(s.GetDim(d));
        rawOutputShapes_.push_back(dims);
    }
    auto inputDesc = ctx_->GetInputDesc(0);
    ge::DataType dtype = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT};
    OP_CHECK_IF(supportedDtypes.count(dtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ctx_->GetNodeName(), "input0",
            ge::TypeUtils::DataTypeToSerialString(dtype).c_str(),
            "only DT_FLOAT16 and DT_FLOAT are supported"),
        return GRAPH_FAILED);
    dtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(dtype));

    PadAndSqueeze(rawInputShapes_, rawOutputShapes_,
                  maxBroShape_, normalInputShapes_, normalOutputShapes_);
    rank_ = (int64_t)maxBroShape_.size();

    OP_CHECK_IF(
        !CheckBroadcastShape(normalInputShapes_, normalOutputShapes_, rank_),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "input/output",
            "incompatible", "check broadcast shape failed, shapes must be broadcast-compatible"),
        return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

// RunTiling() — 串联 GetShapeInfo → 映射 → 填充
ge::graphStatus AdamTiling::RunTiling() {
    GetShapeInfo();                           // 1. 归一化，得 rank_
    int64_t mapped = (rank_ <= 4) ? 4 : 8;    // 2. 映射为 4 或 8
    if (mapped == 4) {                         // 3. 模板填充 + SetTilingKey
        auto ret = DoTilingAndSet<4>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(ADAM_RANK_4));
        return ret;
    } else {
        auto ret = DoTilingAndSet<8>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(ADAM_RANK_8));
        return ret;
    }
}

// DoTilingAndSet<R>() — 模板填充
template<int64_t R>
ge::graphStatus AdamTiling::DoTilingAndSet() {
    auto* td = ctx_->GetTilingData<AdamTilingData<R>>();

    // 平台参数 tiling 时直接从系统获取（不经 CompileInfo 缓存），任一为 0 视为环境异常
    fe::PlatFormInfos *platformInfo = ctx_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t ubPerCore = static_cast<int64_t>(ubSize);

    FindSplitAxis(maxBroShape_, dtypeSize_, ubPerCore, PHYS_NODES, td->split);
    MultiCoreSplit(maxBroShape_, td->split, coreNum, td->multicore);
    int64_t ni = normalInputShapes_.size(), no = normalOutputShapes_.size();
    std::vector<std::vector<int64_t>> ist(ni), ost(no);
    for (int64_t i = 0; i < ni; i++) PrecomputeStrides(normalInputShapes_[i], ist[i]);
    for (int64_t i = 0; i < no; i++) PrecomputeStrides(normalOutputShapes_[i], ost[i]);

    td->rank = rank_;
    for (int64_t d = 0; d < rank_; d++) td->maxBroShape[d] = maxBroShape_[d];
    for (int64_t d = rank_; d < R; d++) td->maxBroShape[d] = 0;

    td->numInputs = ni; td->numOutputs = no;
    for (int64_t i = 0; i < ni; i++)
        for (int64_t d = 0; d < rank_; d++) {
            td->inputShapes[i][d]  = normalInputShapes_[i][d];
            td->inputStrides[i][d] = ist[i][d];
        }
    for (int64_t i = 0; i < no; i++)
        for (int64_t d = 0; d < rank_; d++) {
            td->outputShapes[i][d]  = normalOutputShapes_[i][d];
            td->outputStrides[i][d] = ost[i][d];
        }

    ctx_->SetBlockDim(td->multicore.usedCoreNum);
    return GRAPH_SUCCESS;
}

// 注册：CompileInfo 为空结构，TilingParse 为空实现（平台参数在 Tiling 时直取）
IMPL_OP_OPTILING(AdamApplyOneAssign)
    .Tiling(TilingFuncAdam)
    .TilingParse<AdamCompileInfo>(TilingPrepareForAdam);
```

`AdamTiling` 的完整实现见 `example/examples-adam-code/op_host/arch35/adam_apply_one_assign_tiling_arch35.cpp`，§5.1–§5.5 的函数模块（`PadAndSqueeze`、`CheckBroadcastShape`、`FindSplitAxis`、`MultiCoreSplit`、`PrecomputeStrides` 等）均在同一个 `.cpp` 文件中。

#### 5.6.7 Tiling UT

Tiling UT 用 ops-math 标准框架 `TilingContextPara` + `ExecuteTestCase` 验证 §5.1–§5.6 的全部 Tiling 逻辑。每个用例列出完整 10入3出的 shape、预期 TilingData 各字段的取值、以及 golden 字符串。标杆由 Python 脚本 `code/tests/gen_golden.py` 预计算，UT 逐 `int64_t` 精确对比。

以下 10 个用例均固定 FP32、P=5、ub=256KB、maxCores=56。

##### 1. case1 — 主测 PadAndSqueeze（混合 rank + 标量 + broadcast）

**输入（10 个）：**

| # | shape | 说明 |
|---|-------|------|
| 0 | (4, 6, 3, 5) | 4D 全量，坐标基准 |
| 1 | (6, 3, 5) | 3D → 补 1 为 (1, 6, 3, 5)，broadcast 轴 0 |
| 2 | (3, 5) | 2D → 补 1 为 (1, 1, 3, 5)，broadcast 轴 0,1 |
| 3 | (5,) | 1D → 补 1 为 (1, 1, 1, 5)，broadcast 轴 0,1,2 |
| 4 | (4, 1, 1, 5) | 4D，轴 1,2 为 1，broadcast 轴 1,2 |
| 5 | (4, 6, 1, 1) | 4D，轴 2,3 为 1，broadcast 轴 2,3 |
| 6 | () | 标量 → 归一为 (1, 1, 1, 1) |
| 7 | (4, 1, 1, 1) | 4D，仅轴 0 dim≠1 |
| 8 | (1, 1, 1, 5) | 4D，仅轴 3 dim≠1，broadcast 轴 0,1,2 |
| 9 | (4, 6, 3, 5) | 4D 全量 |

**输出（3 个）：** `(4, 6, 3, 5)` ×3

**预期结果：**

| 字段 | 值 | 说明 |
|------|-----|------|
| maxBroShape | (4, 6, 3, 5), rank=4 | 各轴取 max，无废维 |
| input1 归一化 shape | (1, 6, 3, 5) | 3D→4D，轴 0 补 1 |
| input3 归一化 shape | (1, 1, 1, 5) | 1D→4D |
| input6 归一化 shape | (1, 1, 1, 1) | 标量→4D 全 1 |
| input1 stride | (0, 15, 5, 1) | 轴 0 broadcast → stride=0 |
| input3 stride | (0, 0, 0, 1) | 轴 0,1,2 broadcast |
| input6 stride | (0, 0, 0, 0) | 全 1 → stride 全 0 |
| SplitResult | ubSplitIdx=0, ubFactor=4, ubOuter=1, ubTail=4 | 全量 360≪13107，不切 |
| MultiCoreResult | usedCoreNum=1, totalTiles=1, mainTiles=1, mainCoreNum=1 | 单核 |

##### 2. case2 — 主测 PadAndSqueeze（去 1）

**输入（10 个）：** 全部 `(1, 4, 6)`
**输出（3 个）：** 全部 `(1, 4, 6)`

**预期结果：** 轴 0 全为 1 → squeeze 掉。maxBroShape=(4, 6), rank=2。SplitResult: ubSplitIdx=0, ubFactor=4, ubOuter=1。单核。

##### 3. case3 — 主测 PadAndSqueeze（全标量）

**输入（10 个）：** 全部 `()`
**输出（3 个）：** 全部 `()`

**预期结果：** maxBroShape=(1), rank=1。SplitResult: ubFactor=1, ubOuter=1。单核。

##### 4. case4 — 主测 FindSplitAxis（不整切）

**输入（10 个）+ 输出（3 个）：** 全部 `(1, 1, 203, 200)`

**预期结果：**

| 字段 | 值 | 说明 |
|------|-----|------|
| maxBroShape | (203, 200), rank=2 | 轴 0,1 全为 1 → squeeze |
| perBufElems | 13107 | 256K/5/4 |
| SplitResult | ubSplitIdx=0, ubFactor=65, ubOuter=4, ubTail=8 | d=1: 200≤13107→inner=200; d=0: 203×200=40600>13107→切分 |
| MultiCoreResult | usedCoreNum=4, totalTiles=4, mainTiles=1, mainCoreNum=4 | outerProd=1, ubOuter=4 |

##### 5. case5 — 主测 FindSplitAxis（全量入 UB 不切）

**输入（10 个）：** `(3, 16, 32)` ×7 + `(1, 16, 32)` ×3
**输出（3 个）：** `(3, 16, 32)` ×3

**预期结果：** total=1536≪13107。ubSplitIdx=0, ubFactor=3, ubOuter=1。单核。

##### 6. case6 — 主测 MultiCoreSplit（多核均分）

**输入（10 个）+ 输出（3 个）：** 全部 `(7, 13, 203, 200)`

**预期结果：**

| 字段 | 值 | 说明 |
|------|-----|------|
| SplitResult | ubSplitIdx=2, ubFactor=65, ubOuter=4, ubTail=8 | case4 同逻辑，轴 2 切分 |
| outerProd | 7×13=91 | 轴 0,1 乘积 |
| MultiCoreResult | usedCoreNum=56, totalTiles=364, mainTiles=7, mainCoreNum=28 | 91×4=364, ceil(364/56)=7, 364-(7-1)×56=28 |

##### 7. case7 — 主测 MultiCoreSplit（单核）

**输入（10 个）+ 输出（3 个）：** 全部 `(1, 1, 100, 100)`

**预期结果：** ubSplitIdx=0, ubOuter=1, totalTiles=1, usedCoreNum=1。

##### 8. case8 — 主测 Stride（broadcast 三种模式）

**输入（10 个）：**

| # | shape | broadcast 模式 |
|---|-------|---------------|
| 0 | (4, 6, 195, 200) | 无 broad |
| 1 | (4, 1, 195, 200) | 单轴 broad（轴 1） |
| 2 | (1, 1, 1, 200) | 全 broad（仅末轴） |
| 3~9 | 同 input0 | 无 broad |

**输出（3 个）：** `(4, 6, 195, 200)` ×3

**预期结果：**

| 入参 | stride | 说明 |
|------|--------|------|
| input0 | (234000, 39000, 200, 1) | 全非 0 |
| input1 | (39000, 0, 200, 1) | 轴 1 broadcast → stride[1]=0 |
| input2 | (0, 0, 0, 1) | 仅末轴 dim≠1 → stride=(0,0,0,1) |

##### 9. case9 — 主测 Stride（无 broadcast）

**输入（10 个）+ 输出（3 个）：** 全部 `(4, 6, 195, 200)`

**预期结果：** 所有 stride 全非 0，值 = (234000, 39000, 200, 1)。

##### 10. case10 — 贯通（10入3出全开）

**输入（10 个）：**

| # | shape | 说明 |
|---|-------|------|
| 0 | (7, 13, 203, 200) | 4D 坐标基准 |
| 1 | (13, 203, 200) | 3D → 补 1，broadcast 轴 0 |
| 2 | (203, 200) | 2D → 补 1，broadcast 轴 0,1 |
| 3 | (200,) | 1D → 补 1，broadcast 轴 0,1,2 |
| 4 | (7, 1, 203, 200) | 4D，broadcast 轴 1 |
| 5 | (7, 13, 1, 200) | 4D，broadcast 轴 2 |
| 6 | (7, 1, 1, 200) | 4D，broadcast 轴 1,2 |
| 7 | (1, 1, 1, 200) | 4D，broadcast 轴 0,1,2 |
| 8 | (7, 13, 203, 200) | 4D 全量 |
| 9 | (7, 13, 203, 200) | 4D 全量 |

**输出（3 个）：** `(7, 13, 203, 200)` ×3

**预期结果：** maxBroShape=(7,13,203,200), rank=4。SplitResult: ubSplitIdx=2, ubFactor=65, ubOuter=4, ubTail=8。MultiCoreResult: usedCoreNum=56, totalTiles=364, mainTiles=7, mainCoreNum=28。input1 stride=(0, 40600, 200, 1)；input3 stride=(0, 0, 0, 1)；input5 stride=(2600, 200, 0, 1)。

##### 验证

在 ops-math 仓根目录执行：

```bash
bash build.sh -u --ops=adam_apply_one_assign --soc=ascend950
```

预期 10/10 PASS。每条用例走完整 CANN 路径，`expectTilingData` 由 `code/tests/gen_golden.py` 预计算，`ExecuteTestCase` 逐 `int64_t` 精确对比。后续修改 Tiling 逻辑后重跑此命令即可拦截回归。

---

## 6. Kernel 详细设计

### 6.1 核心原则

**1. Kernel 文件划分**

| 文件 | 意义 |
|------|------|
| `{算子}_apt.cpp` | Kernel 对外入口。`template<int32_t RANK>` 编译期确定数组大小，`GET_TILING_DATA_WITH_STRUCT` 取 TilingData，实例化 Kernel 类，调 `Init()` → `Process()`。 |
| `{算子}_kernel.h` | Kernel 类。`Init()` 绑定 GM 地址 + 初始化 TBuf；`Process()` 按计算流伪码交替搬运与计算。私有方法 `CopyInBrc` / `CopyOutOne` 分别处理单路 GM↔UB 搬运。 |

**2. 约束**

1. 申请内存使用 `TBuf`
2. 申请多少内存依赖存活节点分析（P 来自 §2，`perBufBytes = UB / P`，须 32B 对齐）。内存资源使用 TBuf（详见 `knowledge/coding/Tbuf.md`）
3. 业务编码遵循 §3 计算流伪码
4. API 使用 AscendC 基础 API / RegBase API，不使用 ATVOSS
5. 持有注释判定依据为全流程生命表（全局视角），不可局部推理。格式：`持有=N: name(buffer)→未来消费者`（无→=当前计算步消费），计算进行中增标 `执行中持有=N ← 峰值`

**3. 推演方法论**

Kernel 设计不是一步写出完整代码，是按以下顺序逐层推进、每层验证通过后再进下一层：

```
§3 计算流伪码                     ← 唯一参照源
    ↓
6.4.2 无同步版                    ← 先把业务写对, 持有法则验证 P 不超
    ↓  (持有注释暴露 RAW/WAR)
6.4.3 同步版                      ← 从持有注释推导同步点, 精准插入 SetFlag/WaitFlag
    ↓  (抽函数, 看 API 契约)
6.4.4 函数模块                    ← NDDMA 接口契约, VF 融合, 基础版→优化版推演
    ↓  (取各节最优版本)
6.4.5 kernel.h 完全体             ← 收束, 前面小毛病在此统一修掉
```

每层只解决一个问题。遇到未知 API（NDDMA、RegBase、SetFlag）先学官方文档落知识文件，再回设计文档应用。

**4. 代码生成流程**

1. 先生成 `apt.cpp`，再生成 `_kernel.h`
2. `_kernel.h` 第一版不带同步，全局分析 buffer 依赖后生成带同步版本

### 6.2 文件布局

```
math/adam_apply_one_assign/op_kernel/
├── adam_apply_one_assign.cpp            # Kernel 入口
└── arch35/
    └── adam_apply_one_assign_kernel.h     # AdamKernel<T, RANK> 类
```

### 6.3 Kernel 入口 — {算子}_apt.cpp

Kernel 入口的核心职责是"消费 Tiling 决定的 Key"。Key 的数量和取值由 Tiling 阶段确定，Kernel 入口用模板化方案对接。

以本算子为例：Tiling（§5.6.5）通过 `ASCENDC_TPL_ARGS_DECL` 声明了 2 个 Key——RANK=4（覆盖 rank 1~4）和 RANK=8（覆盖 rank 5~8）。Host 侧 `DoTilingAndSet<4>` / `DoTilingAndSet<8>` 各填一份不同大小的 `AdamTilingData<4>`（120 int64）和 `<8>`（228 int64）。Kernel 入口用 `template<int32_t RANK>` + `if constexpr (RANK == 4/8)` 做编译期分支，`GET_TILING_DATA_WITH_STRUCT` 取对应 struct，实例化 `AdamKernel<T,RANK>`。

模板化的好处：RANK 是编译期常量后，`int64_t[RANK]` 数组、`MultiCopyLoopInfo<RANK>`、NDDMA dim 全部编译期展开，Device 侧不需要 VLA 和 `std::vector`。

```cpp
#include "arch35/adam_apply_one_assign_kernel.h"
#include "arch35/adam_apply_one_assign_tiling_struct.h"

template<int32_t RANK>
__global__ __aicore__ void adam_apply_one_assign(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR input3, GM_ADDR input4,
    GM_ADDR mul0_x, GM_ADDR mul1_x, GM_ADDR mul2_x, GM_ADDR mul3_x, GM_ADDR add2_y,
    GM_ADDR out0, GM_ADDR out1, GM_ADDR out2,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (RANK == 4) {
        GET_TILING_DATA_WITH_STRUCT(AdamTilingData<4>, td, tiling);
        AdamKernel<half, 4> kernel;
        GM_ADDR ins[10]  = {input0, input1, input2, input3, input4,
                             mul0_x, mul1_x, mul2_x, mul3_x, add2_y};
        GM_ADDR outs[3]  = {out0, out1, out2};
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(AdamTilingData<8>, td, tiling);
        AdamKernel<half, 8> kernel;
        GM_ADDR ins[10]  = {input0, input1, input2, input3, input4,
                             mul0_x, mul1_x, mul2_x, mul3_x, add2_y};
        GM_ADDR outs[3]  = {out0, out1, out2};
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
```

### 6.4 Kernel 类 — {算子}_kernel.h

#### 成员变量

5 个 TBuf buffer 就是 5 个物理 UB buffer。每个 buffer 不绑定死角色——本轮迭代中，同一个 buffer 可以先拿来做 CopyIn，释放后做中间结果，再释放后做 CopyOut。这种动态复用是 TBuf 相比 TQue 的核心优势。

| 成员 | 类型 | 数量 | 物理意义 |
|------|------|------|---------|
| `td_` | `const AdamTilingData<RANK>*` | 1 | 只读 TilingData 指针 |
| `pipe_` | `TPipe` | 1 | 流水线管理器 |
| `gmIn_[i]` | `GlobalTensor<T>` | 10 | 入参 GM 视图 |
| `gmOut_[i]` | `GlobalTensor<T>` | 3 | 出参 GM 视图 |
| `buf_[i]` | `TBuf<TPosition::VECCALC>` | 5 | **P=5 个万能 UB buffer buffer** |

#### 方法

| 方法 | 访问 | 职责 |
|------|------|------|
| `Init` | public | 绑定 10+3 个 GlobalTensor，初始化 5 个 TBuf |
| `Process` | public | 主循环：按 §3 计算流逐步 Copy→Compute，遵从生命表 |
| `CopyInBrc` | private | 搬单个入参到指定 TBuf buffer |
| `CopyOutOne` | private | 从指定 TBuf buffer 搬单个出参到 GM |

#### 类声明

```cpp
template <typename T, int64_t RANK>
class AdamKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR inputs[10], GM_ADDR outputs[3],
        const AdamTilingData<RANK>* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBrc(
        const int64_t* coord, int32_t inputIdx, int32_t buffer, int64_t count);
    __aicore__ inline void CopyOutOne(
        const int64_t* coord, int32_t outputIdx, int32_t buffer, int64_t count);

    TPipe pipe_;
    const AdamTilingData<RANK>* td_;
    GlobalTensor<T> gmIn_[10];
    GlobalTensor<T> gmOut_[3];
    TBuf<TPosition::VECCALC> buf_[5];
};
```

以下逐方法展开实现。

#### 6.4.1 Init — 建立流水线

绑定 GM 地址，初始化 5 个 TBuf 为最大 tile 大小。

```cpp
__aicore__ inline void Init(
    GM_ADDR inputs[10], GM_ADDR outputs[3],
    const AdamTilingData<RANK>* td)
{
    td_ = td;
    for (int32_t i = 0; i < 10; i++)
        gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
    for (int32_t i = 0; i < 3; i++)
        gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);

    for (int32_t i = 0; i < 5; i++)
        pipe_.InitBuffer(buf_[i], td_->perBufBytes);
}
```

#### 6.4.2 Process — 主循环（无同步版）

**本节目标**：
1. **实现** — 基于 §3 计算流伪码实现 Kernel 主循环
2. **标注** — 基于持有法则，每个执行动作标注执行前/中/后 buffer 持有列表（详见 `knowledge/coding/hold-law.md`）
3. **验证** — 计算流、内存持有、物理存活节点 P=5、§3 伪码 四者一致。不一致不通过

同步后置，本版本不包含。

```cpp
__aicore__ inline void Process() {
    int64_t start, end;
    GetCoreRange(GetBlockIdx(), td_->multicore.usedCoreNum,
                 td_->multicore.totalTiles, td_->multicore.mainTiles,
                 td_->multicore.mainCoreNum, start, end);

    constexpr int32_t UB0 = 0, UB1 = 1, UB2 = 2, UB3 = 3, UB4 = 4;
    constexpr int32_t IN0 = 0, IN1 = 1, IN2 = 2, IN3 = 3, IN4 = 4;
    constexpr int32_t MUL = 5, MUL1 = 6, MUL2 = 7, MUL3 = 8, ADD2 = 9;
    constexpr int32_t OUT0 = 0, OUT1 = 1, OUT2 = 2;

    int64_t coord[8] = {};
    for (int64_t flat = start; flat < end; flat++) {
        int64_t ubBlockLength = GetUBSplitRange(flat % td_->split.ubOuter, td_->split.ubOuter,
                                          td_->split.ubFactor, td_->split.ubTail);
        int64_t count = ubBlockLength;
        FlatToEffectiveCoord(flat, td_->maxBroShape, RANK,
                             td_->split.ubSplitIdx, td_->split.ubFactor, td_->split.ubOuter, coord);

        // ═══ S1: in0 → square ═══
        // 执行前: 持有=[]
        // 执行中: 持有=[UB0]
        // 执行后: 持有=[UB0]
        CopyInBrc(coord, IN0, UB0);
        // 执行前: 持有=[UB0]
        // 执行中: 持有=[UB0, UB2]
        // 执行后: 持有=[UB0, UB2]
        AscendC::Mul(buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), buf_[UB0].Get<T>(), count);

        // ═══ S2: in0·mul1_x → mul1 ═══
        // 执行前: 持有=[UB0, UB2]
        // 执行中: 持有=[UB0, UB1, UB2]
        // 执行后: 持有=[UB0, UB1, UB2]
        CopyInBrc(coord, MUL1, UB1);
        // 执行前: 持有=[UB0, UB1, UB2]
        // 执行中: 持有=[UB0, UB1, UB2, UB3]
        // 执行后: 持有=[UB2, UB3]
        AscendC::Mul(buf_[UB3].Get<T>(), buf_[UB0].Get<T>(), buf_[UB1].Get<T>(), count);

        // ═══ S3: MulAddDst(mul1, dataM, mul0_x) → output1 ═══
        // 执行前: 持有=[UB2, UB3]
        // 执行中: 持有=[UB0, UB2, UB3]
        // 执行后: 持有=[UB0, UB2, UB3]
        CopyInBrc(coord, IN2, UB0);
        // 执行前: 持有=[UB0, UB2, UB3]
        // 执行中: 持有=[UB0, UB1, UB2, UB3]
        // 执行后: 持有=[UB0, UB1, UB2, UB3]
        CopyInBrc(coord, MUL, UB1);
        // 执行前: 持有=[UB0, UB1, UB2, UB3]
        // 执行中: 持有=[UB0, UB1, UB2, UB3]
        // 执行后: 持有=[UB2, UB3]
        asc_vf_call<MulAddVF<T>>(buf_[UB3], buf_[UB1], buf_[UB0], count);

        // ═══ S4: square·mul3_x → mul3 ═══
        // 执行前: 持有=[UB2, UB3]
        // 执行中: 持有=[UB2, UB3, UB4]
        // 执行后: 持有=[UB2, UB3, UB4]
        CopyInBrc(coord, MUL3, UB4);
        // 执行前: 持有=[UB2, UB3, UB4]
        // 执行中: 持有=[UB2, UB3, UB4]
        // 执行后: 持有=[UB3, UB4]
        AscendC::Mul(buf_[UB4].Get<T>(), buf_[UB2].Get<T>(), buf_[UB4].Get<T>(), count);

        // ═══ S5: MulAddDst(mul3, dataV, mul2_x) → output0 ═══
        // 执行前: 持有=[UB3, UB4]
        // 执行中: 持有=[UB0, UB3, UB4]
        // 执行后: 持有=[UB0, UB3, UB4]
        CopyInBrc(coord, IN1, UB0);
        // 执行前: 持有=[UB0, UB3, UB4]
        // 执行中: 持有=[UB0, UB2, UB3, UB4]
        // 执行后: 持有=[UB0, UB2, UB3, UB4]
        CopyInBrc(coord, MUL2, UB2);
        // 执行前: 持有=[UB0, UB2, UB3, UB4]
        // 执行中: 持有=[UB0, UB2, UB3, UB4]
        // 执行后: 持有=[UB3, UB4]
        asc_vf_call<MulAddVF<T>>(buf_[UB4], buf_[UB0], buf_[UB2], count);

        // ═══ S6: sqrt(output0) → sqrt ═══
        // 执行前: 持有=[UB3, UB4]
        // 执行中: 持有=[UB0, UB3, UB4]
        // 执行后: 持有=[UB0, UB3, UB4]
        AscendC::Sqrt(buf_[UB0].Get<T>(), buf_[UB4].Get<T>(), count);

        // ═══ S7: sqrt + add2_y → add2 ═══
        // 执行前: 持有=[UB0, UB3, UB4]
        // 执行中: 持有=[UB0, UB2, UB3, UB4]
        // 执行后: 持有=[UB0, UB2, UB3, UB4]
        CopyInBrc(coord, ADD2, UB2);
        // 执行前: 持有=[UB0, UB2, UB3, UB4]
        // 执行中: 持有=[UB0, UB1, UB2, UB3, UB4]
        // 执行后: 持有=[UB1, UB3, UB4]
        AscendC::Add(buf_[UB1].Get<T>(), buf_[UB0].Get<T>(), buf_[UB2].Get<T>(), count);

        // ═══ S8: output1 / add2 → truediv ═══
        // 执行前: 持有=[UB1, UB3, UB4]
        // 执行中: 持有=[UB1, UB2, UB3, UB4]
        // 执行后: 持有=[UB2, UB3, UB4]
        AscendC::Div(buf_[UB2].Get<T>(), buf_[UB3].Get<T>(), buf_[UB1].Get<T>(), count);

        // ═══ S9: truediv·input4 → mul4 ═══
        // 执行前: 持有=[UB2, UB3, UB4]
        // 执行中: 持有=[UB0, UB2, UB3, UB4]
        // 执行后: 持有=[UB0, UB2, UB3, UB4]
        CopyInBrc(coord, IN4, UB0);
        // 执行前: 持有=[UB0, UB2, UB3, UB4]
        // 执行中: 持有=[UB0, UB1, UB2, UB3, UB4]
        // 执行后: 持有=[UB1, UB3, UB4]
        AscendC::Mul(buf_[UB1].Get<T>(), buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), count);

        // ═══ S10: dataVar − mul4 → output2 ═══
        // 执行前: 持有=[UB1, UB3, UB4]
        // 执行中: 持有=[UB0, UB1, UB3, UB4]
        // 执行后: 持有=[UB0, UB1, UB3, UB4]
        CopyInBrc(coord, IN3, UB0);
        // 执行前: 持有=[UB0, UB1, UB3, UB4]
        // 执行中: 持有=[UB0, UB1, UB2, UB3, UB4]
        // 执行后: 持有=[UB2, UB3, UB4]
        AscendC::Sub(buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), buf_[UB1].Get<T>(), count);

        // ═══ CopyOut ═══
        // 执行前: 持有=[UB2, UB3, UB4]
        // 执行中: 持有=[UB2, UB3, UB4]
        // 执行后: 持有=[UB2, UB3]
        CopyOutOne(coord, OUT0, UB4);
        // 执行前: 持有=[UB2, UB3]
        // 执行中: 持有=[UB2, UB3]
        // 执行后: 持有=[UB2]
        CopyOutOne(coord, OUT1, UB3);
        // 执行前: 持有=[UB2]
        // 执行中: 持有=[UB2]
        // 执行后: 持有=[]
        CopyOutOne(coord, OUT2, UB2);
    }
}
```

**UB 生命周期验证**（逐列 = 该步最后一个 Compute 执行后持有，与代码注释逐行对照）：

| UB | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|----|----|----|----|----|----|----|----|----|----|-----|
| UB0 | in0→S2 | — | — | — | — | sqrt→S7 | — | — | — | — |
| UB1 | — | — | — | — | — | — | add2→S8 | — | mul4→S10 | — |
| UB2 | square→S4 | square→S4 | square→S4 | — | — | — | — | truediv→S9 | — | output2→Out |
| UB3 | — | mul1→S3 | output1→S8 | output1→S8 | output1→S8 | output1→S8 | output1→S8 | output1→Out | output1→Out | output1→Out |
| UB4 | — | — | — | mul3→S5 | output0→Out | output0→Out | output0→Out | output0→Out | output0→Out | output0→Out |
| 持有 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | 3 | 3 |
| 执行中峰值 | 2 | 4 | 4 | 3 | 4 | 3 | **5** | 4 | **5** | **5** |

验证：持有行 ≤3 ≤ P=5 ✓；峰值 S7/S9/S10 达到 5，与 §2 一致 ✓；释放时机与 §3 一致 ✓。

#### 6.4.3 Process — 主循环（同步版）

**本节目标**：在 6.4.2 无同步版基础上，基于持有法则暴露的 RAW/WAR 依赖，插入 SetFlag/WaitFlag（详见 `knowledge/coding/sync.md`）。

**同步点推导**（从 6.4.2 持有注释逐行分析）：

| 事件 | 依赖 | 次数 | 触发条件（持有注释直接可见） |
|------|------|------|---------------------------|
| MTE2_V | RAW | 8 | CopyIn 后执行后含某 buffer → 下一 Compute 执行中含该 buffer（S1/S2/S3/S4+S5/S7/S9/S10） |
| V_MTE2 | WAR | 4 | V 执行后持有某 buffer → 后续 CopyIn 要覆盖该 buffer（S2→S3 覆盖 UB0,UB1 / S4→S5 覆盖 UB2 / S8→S9 覆盖 UB0 / S9→S10 覆盖 UB0） |
| V_MTE3 | RAW | 1 | S10 执行后持有 [UB2,UB3,UB4] → CopyOut 的 MTE3 要读这些 buffer |
| MTE3_MTE2 | WAR 跨迭代 | 1 | 上轮 CopyOut 读 UB2 → 下轮 S5 CopyIn 写 UB2 |

```cpp
__aicore__ inline void Process() {
    event_t evMTE2toV    = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t evVtoMTE2    = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t evVtoMTE3    = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t evMTE3toMTE2 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    int64_t start, end;
    GetCoreRange(GetBlockIdx(), td_->multicore.usedCoreNum,
                 td_->multicore.totalTiles, td_->multicore.mainTiles,
                 td_->multicore.mainCoreNum, start, end);

    constexpr int32_t UB0 = 0, UB1 = 1, UB2 = 2, UB3 = 3, UB4 = 4;
    constexpr int32_t IN0 = 0, IN1 = 1, IN2 = 2, IN3 = 3, IN4 = 4;
    constexpr int32_t MUL = 5, MUL1 = 6, MUL2 = 7, MUL3 = 8, ADD2 = 9;
    constexpr int32_t OUT0 = 0, OUT1 = 1, OUT2 = 2;

    int64_t coord[8] = {};
    for (int64_t flat = start; flat < end; flat++) {
        int64_t ubBlockLength = GetUBSplitRange(flat % td_->split.ubOuter, td_->split.ubOuter,
                                          td_->split.ubFactor, td_->split.ubTail);
        int64_t count = ubBlockLength;
        FlatToEffectiveCoord(flat, td_->maxBroShape, RANK,
                             td_->split.ubSplitIdx, td_->split.ubFactor, td_->split.ubOuter, coord);

        // MTE3→MTE2: 等上一轮 CopyOut 读完 UB (首轮跳过)
        if (flat != start) WaitFlag<HardEvent::MTE3_MTE2>(evMTE3toMTE2);

        // ═══ S1: in0 → square ═══
        // 执行前: 持有=[]
        // 执行中: 持有=[UB0]
        // 执行后: 持有=[UB0]
        CopyInBrc(coord, IN0, UB0);
        // MTE2→V: MTE2 写完 UB0, V 可以读
        SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
        // 执行前: 持有=[UB0]
        // 执行中: 持有=[UB0, UB2]
        // 执行后: 持有=[UB0, UB2]
        AscendC::Mul(buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), buf_[UB0].Get<T>(), count);

        // ═══ S2: in0·mul1_x → mul1 ═══
        // 执行前: 持有=[UB0, UB2]
        // 执行中: 持有=[UB0, UB1, UB2]
        // 执行后: 持有=[UB0, UB1, UB2]
        CopyInBrc(coord, MUL1, UB1);
        // MTE2→V
        SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
        // 执行前: 持有=[UB0, UB1, UB2]
        // 执行中: 持有=[UB0, UB1, UB2, UB3]
        // 执行后: 持有=[UB2, UB3]
        AscendC::Mul(buf_[UB3].Get<T>(), buf_[UB0].Get<T>(), buf_[UB1].Get<T>(), count);
        // V→MTE2: V 读完 UB0,UB1, MTE2 可覆盖 (S3 将写 UB0,UB1)
        SetFlag<HardEvent::V_MTE2>(evVtoMTE2); WaitFlag<HardEvent::V_MTE2>(evVtoMTE2);

        // ═══ S3: MulAddDst(mul1, dataM, mul0_x) → output1 ═══
        // 执行前: 持有=[UB2, UB3]
        // 执行中: 持有=[UB0, UB2, UB3]
        // 执行后: 持有=[UB0, UB2, UB3]
        CopyInBrc(coord, IN2, UB0);
        // 执行前: 持有=[UB0, UB2, UB3]
        // 执行中: 持有=[UB0, UB1, UB2, UB3]
        // 执行后: 持有=[UB0, UB1, UB2, UB3]
        CopyInBrc(coord, MUL, UB1);
        // MTE2→V
        SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
        // 执行前: 持有=[UB0, UB1, UB2, UB3]
        // 执行中: 持有=[UB0, UB1, UB2, UB3]
        // 执行后: 持有=[UB2, UB3]
        asc_vf_call<MulAddVF<T>>(buf_[UB3], buf_[UB1], buf_[UB0], count);

        // ═══ S4: square·mul3_x → mul3 ═══
        // 执行前: 持有=[UB2, UB3]
        // 执行中: 持有=[UB2, UB3, UB4]
        // 执行后: 持有=[UB2, UB3, UB4]
        CopyInBrc(coord, MUL3, UB4);
        // MTE2→V
        SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
        // 执行前: 持有=[UB2, UB3, UB4]
        // 执行中: 持有=[UB2, UB3, UB4]
        // 执行后: 持有=[UB3, UB4]
        AscendC::Mul(buf_[UB4].Get<T>(), buf_[UB2].Get<T>(), buf_[UB4].Get<T>(), count);
        // V→MTE2: V 读完 UB2, MTE2 可覆盖 (S5 将写 UB2)
        SetFlag<HardEvent::V_MTE2>(evVtoMTE2); WaitFlag<HardEvent::V_MTE2>(evVtoMTE2);

        // ═══ S5: MulAddDst(mul3, dataV, mul2_x) → output0 ═══
        // 执行前: 持有=[UB3, UB4]
        // 执行中: 持有=[UB0, UB3, UB4]
        // 执行后: 持有=[UB0, UB3, UB4]
        CopyInBrc(coord, IN1, UB0);
        // 执行前: 持有=[UB0, UB3, UB4]
        // 执行中: 持有=[UB0, UB2, UB3, UB4]
        // 执行后: 持有=[UB0, UB2, UB3, UB4]
        CopyInBrc(coord, MUL2, UB2);
        // MTE2→V
        SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
        // 执行前: 持有=[UB0, UB2, UB3, UB4]
        // 执行中: 持有=[UB0, UB2, UB3, UB4]
        // 执行后: 持有=[UB3, UB4]
        asc_vf_call<MulAddVF<T>>(buf_[UB4], buf_[UB0], buf_[UB2], count);

        // ═══ S6: sqrt(output0) → sqrt ═══
        // 执行前: 持有=[UB3, UB4]
        // 执行中: 持有=[UB0, UB3, UB4]
        // 执行后: 持有=[UB0, UB3, UB4]
        AscendC::Sqrt(buf_[UB0].Get<T>(), buf_[UB4].Get<T>(), count);

        // ═══ S7: sqrt + add2_y → add2 ═══
        // 执行前: 持有=[UB0, UB3, UB4]
        // 执行中: 持有=[UB0, UB2, UB3, UB4]
        // 执行后: 持有=[UB0, UB2, UB3, UB4]
        CopyInBrc(coord, ADD2, UB2);
        // MTE2→V
        SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
        // 执行前: 持有=[UB0, UB2, UB3, UB4]
        // 执行中: 持有=[UB0, UB1, UB2, UB3, UB4]
        // 执行后: 持有=[UB1, UB3, UB4]
        AscendC::Add(buf_[UB1].Get<T>(), buf_[UB0].Get<T>(), buf_[UB2].Get<T>(), count);

        // ═══ S8: output1 / add2 → truediv ═══
        // 执行前: 持有=[UB1, UB3, UB4]
        // 执行中: 持有=[UB1, UB2, UB3, UB4]
        // 执行后: 持有=[UB2, UB3, UB4]
        AscendC::Div(buf_[UB2].Get<T>(), buf_[UB3].Get<T>(), buf_[UB1].Get<T>(), count);
        // V→MTE2: V 读完 UB0 (S7 Add), MTE2 可覆盖 (S9 将写 UB0)
        SetFlag<HardEvent::V_MTE2>(evVtoMTE2); WaitFlag<HardEvent::V_MTE2>(evVtoMTE2);

        // ═══ S9: truediv·input4 → mul4 ═══
        // 执行前: 持有=[UB2, UB3, UB4]
        // 执行中: 持有=[UB0, UB2, UB3, UB4]
        // 执行后: 持有=[UB0, UB2, UB3, UB4]
        CopyInBrc(coord, IN4, UB0);
        // MTE2→V
        SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
        // 执行前: 持有=[UB0, UB2, UB3, UB4]
        // 执行中: 持有=[UB0, UB1, UB2, UB3, UB4]
        // 执行后: 持有=[UB1, UB3, UB4]
        AscendC::Mul(buf_[UB1].Get<T>(), buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), count);
        // V→MTE2: V 读完 UB0, MTE2 可覆盖 (S10 将写 UB0)
        SetFlag<HardEvent::V_MTE2>(evVtoMTE2); WaitFlag<HardEvent::V_MTE2>(evVtoMTE2);

        // ═══ S10: dataVar − mul4 → output2 ═══
        // 执行前: 持有=[UB1, UB3, UB4]
        // 执行中: 持有=[UB0, UB1, UB3, UB4]
        // 执行后: 持有=[UB0, UB1, UB3, UB4]
        CopyInBrc(coord, IN3, UB0);
        // MTE2→V
        SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
        // 执行前: 持有=[UB0, UB1, UB3, UB4]
        // 执行中: 持有=[UB0, UB1, UB2, UB3, UB4]
        // 执行后: 持有=[UB2, UB3, UB4]
        AscendC::Sub(buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), buf_[UB1].Get<T>(), count);

        // V→MTE3: V 写完 UB2,UB3,UB4 → MTE3 可搬
        SetFlag<HardEvent::V_MTE3>(evVtoMTE3); WaitFlag<HardEvent::V_MTE3>(evVtoMTE3);

        // ═══ CopyOut ═══
        // 执行前: 持有=[UB2, UB3, UB4]
        // 执行中: 持有=[UB2, UB3, UB4]
        // 执行后: 持有=[UB2, UB3]
        CopyOutOne(coord, OUT0, UB4);
        // 执行前: 持有=[UB2, UB3]
        // 执行中: 持有=[UB2, UB3]
        // 执行后: 持有=[UB2]
        CopyOutOne(coord, OUT1, UB3);
        // 执行前: 持有=[UB2]
        // 执行中: 持有=[UB2]
        // 执行后: 持有=[]
        CopyOutOne(coord, OUT2, UB2);

        // MTE3→MTE2: MTE3 读完, 通知下轮 MTE2 (末轮跳过)
        if (flat != end - 1) SetFlag<HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    }
}
```

#### 6.4.4 函数模块

**本节目标**：将 §3 计算流伪码中的 DataCopy 映射为 AscendC API。核心是搞清楚 NDDMA 的接口契约，把 broadcast 做对。这不是 Adam 特例，是所有 Vector 算子搬入搬出的通用模式。详见 `knowledge/coding/nddma-programming.md`。

三条关键知识：
- **loopSize = 目标 shape**（broadcast 后大小），非源 shape；srcStride=0 控制播轴
- **NDDMA dim[0]=最内维**，Tiling d=RANK-1=最内维，`nd = RANK-1-d` 翻转
- **inputStrides = GM 物理 stride**（PrecomputeStrides 已算好，播轴=0），直接填 loopSrcStride

##### CopyInBrc — 基础版

NDDMA 随路 broadcast 搬入。每 tile 调用时构造 loopInfo，只有 `ubBlockLength` 和受其影响的 `inner`/`dstStride` 是动态的。

```cpp
__aicore__ inline void CopyInBrc(
    const int64_t* coord, int32_t inputIdx, int32_t buffer, int64_t ubBlockLength)
{
    int64_t k = td_->split.ubSplitIdx;
    int64_t off = CalcOffset(coord, td_->inputStrides[inputIdx], RANK);
    const int64_t* dstShape = td_->maxBroShape;

    if constexpr (RANK <= MAX_NDDMA_DIMS) {
        AscendC::MultiCopyLoopInfo<RANK> loopInfo;
        int64_t inner = 1;
        for (int64_t d = RANK - 1; d >= 0; d--) {
            int64_t nd = RANK - 1 - d;
            loopInfo.loopSize[nd] = (d == k) ? ubBlockLength : dstShape[d];
            loopInfo.loopSrcStride[nd] = td_->inputStrides[inputIdx][d];
            loopInfo.loopDstStride[nd] = inner;
            inner *= loopInfo.loopSize[nd];
        }
        AscendC::MultiCopyParams<T, RANK> params = { loopInfo, 0 };
        static constexpr AscendC::NdDmaConfig cfg = { false };
        AscendC::DataCopy<T, RANK, cfg>(buf_[buffer].Get<T>(), gmIn_[inputIdx][off], params);

    } else {
        constexpr int64_t ND = MAX_NDDMA_DIMS, outerRank = RANK - ND;
        AscendC::MultiCopyLoopInfo<ND> loopInfo;
        int64_t inner = 1;
        for (int64_t d = RANK - 1; d >= outerRank; d--) {
            int64_t nd = RANK - 1 - d;
            loopInfo.loopSize[nd] = (d == k) ? ubBlockLength : dstShape[d];
            loopInfo.loopSrcStride[nd] = td_->inputStrides[inputIdx][d];
            loopInfo.loopDstStride[nd] = inner;
            inner *= loopInfo.loopSize[nd];
        }
        AscendC::MultiCopyParams<T, ND> params = { loopInfo, 0 };
        static constexpr AscendC::NdDmaConfig cfg = { false };
        LocalTensor<T> buf = buf_[buffer].Get<T>();

        int64_t outerIters = 1;
        for (int64_t d = 0; d < outerRank; d++)
            outerIters *= (d == k) ? ubBlockLength : dstShape[d];

        int64_t elemBase = off / sizeof(T);
        for (int64_t oi = 0; oi < outerIters; oi++) {
            int64_t elemAdj = 0, tmp = oi;
            for (int64_t d = outerRank - 1; d >= 0; d--) {
                int64_t sz = (d == k) ? ubBlockLength : dstShape[d];
                elemAdj += (tmp % sz) * td_->inputStrides[inputIdx][d];
                tmp /= sz;
            }
            AscendC::DataCopy<T, ND, cfg>(
                buf[oi * inner], gmIn_[inputIdx][elemBase + elemAdj], params);
        }
    }
}
```

##### CopyInBrc — 优化版

基础版中 loopSrcStride、非切分轴 loopSize、loopLpSize/loopRpSize 不随 tile 变化。Init 时预计算 10 路输入的 `MultiCopyParams` 存入成员数组，运行时直接复用——仅切分轴的 `loopSize[k]` 及依赖它的 `inner`/`dstStride` 需要按 `ubBlockLength` 调整。

成员变量新增：

```cpp
// RANK≤5 时用 MultiCopyParams<T,RANK>, RANK>5 时用 MultiCopyParams<T,MAX_NDDMA_DIMS>
static constexpr int64_t MAX_NDDMA_DIMS = 5;
static constexpr int64_t ND = (RANK <= MAX_NDDMA_DIMS) ? RANK : MAX_NDDMA_DIMS;
AscendC::MultiCopyParams<T, ND> nddmaParams_[MAX_INPUT_SLOTS];  // 每输入一路, Init 预填
int64_t nddmaOuterIters_[MAX_INPUT_SLOTS];                      // RANK>5 时外维迭代总数
```

Init 中预计算（与基础版构造逻辑相同，`ubBlockLength` 位置先填占位值 `0`）：

```cpp
for (int32_t inp = 0; inp < 10; inp++) {
    int64_t inner = 1;
    for (int64_t d = RANK - 1; d >= RANK - ND; d--) {
        int64_t nd = RANK - 1 - d;
        nddmaParams_[inp].loopInfo.loopSize[nd] = (d == k) ? 0 /* 占位 */ : dstShape[d];
        nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->inputStrides[inp][d];
        nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
        inner *= (d == k) ? td_->split.ubFactor : dstShape[d];  // 按满 tile 算 inner
    }
    // RANK>5 时算外维迭代数
    nddmaOuterIters_[inp] = 1;
    for (int64_t d = 0; d < RANK - ND; d++)
        nddmaOuterIters_[inp] *= (d == k) ? td_->split.ubFactor : dstShape[d];
}
```

```cpp
__aicore__ inline void CopyInBrcFast(
    const int64_t* coord, int32_t inputIdx, int32_t buffer, int64_t ubBlockLength)
{
    int64_t k = td_->split.ubSplitIdx;
    int64_t off = CalcOffset(coord, td_->inputStrides[inputIdx], RANK);
    const int64_t* dstShape = td_->maxBroShape;

    // 复用预计算参数，仅更新受 ubBlockLength 影响的字段
    auto params = nddmaParams_[inputIdx];
    int64_t baseInner = params.loopInfo.loopDstStride[0];
    int64_t kNd = RANK - 1 - k;
    int64_t inner = 1;
    for (int64_t nd = 0; nd < ND; nd++) {
        if (nd == kNd) {
            params.loopInfo.loopSize[nd] = ubBlockLength;
        }
        params.loopInfo.loopDstStride[nd] = inner;
        inner *= params.loopInfo.loopSize[nd];
    }

    static constexpr AscendC::NdDmaConfig cfg = { false };

    if constexpr (RANK <= MAX_NDDMA_DIMS) {
        AscendC::DataCopy<T, ND, cfg>(buf_[buffer].Get<T>(), gmIn_[inputIdx][off], params);
    } else {
        LocalTensor<T> buf = buf_[buffer].Get<T>();
        int64_t elemBase = off / sizeof(T);
        for (int64_t oi = 0; oi < nddmaOuterIters_[inputIdx]; oi++) {
            int64_t elemAdj = 0, tmp = oi;
            for (int64_t d = RANK - ND - 1; d >= 0; d--) {
                int64_t sz = (d == k) ? ubBlockLength : dstShape[d];
                elemAdj += (tmp % sz) * td_->inputStrides[inputIdx][d];
                tmp /= sz;
            }
            AscendC::DataCopy<T, ND, cfg>(
                buf[oi * inner], gmIn_[inputIdx][elemBase + elemAdj], params);
        }
    }
}
```

##### MulAddVF

`Reg::MulAddDst` 的 VF 包装。单指令 `dst = src0 * src1 + dst`，替代 Mul + Add 两步。

```cpp
template <typename T>
__simd_vf__ inline void MulAddVF(
    __ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::Reg::LoadAlign(dstReg, dstAddr, aReg);
        AscendC::Reg::MulAddDst(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}
```

##### CopyOutOne

UB→GM 单路搬出。无 broadcast，普通 DataCopy。

```cpp
// coord     — 当前 tile 的 multi-index 坐标, +outputStrides → GM 地址
// outputIdx — 哪个输出 (0..2), 索引 td_->outputShapes/outputStrides/gmOut_
// buffer      — 源 TBuf (UB0..UB4)
// ubBlockLength   — 切分轴本轮段长, 决定搬运元素数
__aicore__ inline void CopyOutOne(
    const int64_t* coord, int32_t outputIdx, int32_t buffer, int64_t ubBlockLength)
{
    int64_t off = CalcOffset(coord, td_->outputStrides[outputIdx], RANK);
    int64_t cnt = CalcTransferCount(td_->outputShapes[outputIdx], RANK,
                                          td_->split.ubSplitIdx, ubBlockLength);
    LocalTensor<T> buf = buf_[buffer].Get<T>();
    DataCopy(gmOut_[outputIdx][off], buf, cnt);
}
```

- NDDMA 不支持 UB→GM，直接普通 DataCopy
- output 无 broadcast 需求，shape 已是 broadcast 后形态

#### 6.4.5 kernel.h 完全体

取各节最优版本，统一修掉前面的小毛病，产出完整可用的 `adam_apply_one_assign_kernel.h`。

**取法**：Init = 6.4.1 + 6.4.4 优化版，Process = 6.4.3 同步版 + 6.4.4 函数模块，私有方法 = 6.4.4 优化版，MulAddVF = 见 6.4.4。

```cpp
template <typename T, int64_t RANK>
class AdamKernel {
    static constexpr int64_t MAX_NDDMA_DIMS = 5;
    static constexpr int64_t ND = (RANK <= MAX_NDDMA_DIMS) ? RANK : MAX_NDDMA_DIMS;
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(T);

    TPipe pipe_;
    const AdamTilingData<RANK>* td_;
    GlobalTensor<T> gmIn_[10];
    GlobalTensor<T> gmOut_[3];
    TBuf<TPosition::VECCALC> buf_[5];
    AscendC::MultiCopyParams<T, ND> nddmaParams_[10];
    int64_t nddmaOuterIters_[10];

public:
    __aicore__ inline void Init(GM_ADDR inputs[10], GM_ADDR outputs[3],
                                const AdamTilingData<RANK>* td)
    {
        td_ = td;
        for (int32_t i = 0; i < 10; i++)
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        for (int32_t i = 0; i < 3; i++)
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        for (int32_t i = 0; i < 5; i++)
            pipe_.InitBuffer(buf_[i], td_->perBufBytes);

        // 预计算 10 路 NDDMA 参数 (静态部分, ubBlockLength 运行时更新)
        const int64_t* dstShape = td_->maxBroShape;
        int64_t k = td_->split.ubSplitIdx;
        for (int32_t inp = 0; inp < 10; inp++) {
            int64_t inner = 1;
            for (int64_t d = RANK - 1; d >= RANK - ND; d--) {
                int64_t nd = RANK - 1 - d;
                nddmaParams_[inp].loopInfo.loopSize[nd]      = (d == k) ? 0 : dstShape[d];
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->inputStrides[inp][d];
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                inner *= (d == k) ? td_->split.ubFactor : dstShape[d];
            }
            nddmaOuterIters_[inp] = 1;
            for (int64_t d = 0; d < RANK - ND; d++)
                nddmaOuterIters_[inp] *= (d == k) ? td_->split.ubFactor : dstShape[d];
        }
    }

    __aicore__ inline void Process()
    {
        event_t evMTE2toV    = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t evVtoMTE2    = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t evVtoMTE3    = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t evMTE3toMTE2 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

        int64_t start, end;
        GetCoreRange(GetBlockIdx(), td_->multicore.usedCoreNum,
                     td_->multicore.totalTiles, td_->multicore.mainTiles,
                     td_->multicore.mainCoreNum, start, end);

        constexpr int32_t UB0 = 0, UB1 = 1, UB2 = 2, UB3 = 3, UB4 = 4;
        constexpr int32_t IN0 = 0, IN1 = 1, IN2 = 2, IN3 = 3, IN4 = 4;
        constexpr int32_t MUL = 5, MUL1 = 6, MUL2 = 7, MUL3 = 8, ADD2 = 9;
        constexpr int32_t OUT0 = 0, OUT1 = 1, OUT2 = 2;

        int64_t innerCount = 1;
        for (int64_t d = td_->split.ubSplitIdx + 1; d < RANK; d++)
            innerCount *= td_->maxBroShape[d];

        int64_t coord[8] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t ubBlockLength = GetUBSplitRange(flat % td_->split.ubOuter, td_->split.ubOuter,
                                              td_->split.ubFactor, td_->split.ubTail);
            int64_t count = ubBlockLength * innerCount;
            FlatToEffectiveCoord(flat, td_->maxBroShape, RANK,
                                 td_->split.ubSplitIdx, td_->split.ubFactor, td_->split.ubOuter, coord);

            if (flat != start) WaitFlag<HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            // ═══ S1: in0 → square ═══
            // 执行前: 持有=[]  执行中: 持有=[UB0]  执行后: 持有=[UB0]
            CopyInBrc(coord, IN0, UB0, ubBlockLength);
            SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
            // 执行前: 持有=[UB0]  执行中: 持有=[UB0,UB2]  执行后: 持有=[UB0,UB2]
            AscendC::Mul(buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), buf_[UB0].Get<T>(), count);

            // ═══ S2: in0·mul1_x → mul1 ═══
            // 执行前: 持有=[UB0,UB2]  执行中: 持有=[UB0,UB1,UB2]  执行后: 持有=[UB0,UB1,UB2]
            CopyInBrc(coord, MUL1, UB1, ubBlockLength);
            SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
            // 执行前: 持有=[UB0,UB1,UB2]  执行中: 持有=[UB0,UB1,UB2,UB3]  执行后: 持有=[UB2,UB3]
            AscendC::Mul(buf_[UB3].Get<T>(), buf_[UB0].Get<T>(), buf_[UB1].Get<T>(), count);
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2); WaitFlag<HardEvent::V_MTE2>(evVtoMTE2);

            // ═══ S3: MulAddDst → output1 ═══
            // 执行前: 持有=[UB2,UB3]  执行中: 持有=[UB0,UB2,UB3]  执行后: 持有=[UB0,UB2,UB3]
            CopyInBrc(coord, IN2, UB0, ubBlockLength);
            // 执行前: 持有=[UB0,UB2,UB3]  执行中: 持有=[UB0,UB1,UB2,UB3]  执行后: 持有=[UB0,UB1,UB2,UB3]
            CopyInBrc(coord, MUL, UB1, ubBlockLength);
            SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
            // 执行前: 持有=[UB0,UB1,UB2,UB3]  执行中: 持有=[UB0,UB1,UB2,UB3]  执行后: 持有=[UB2,UB3]
            uint16_t repeatTimesS3 = AscendC::CeilDivision(count, VL);
            asc_vf_call<MulAddVF<T>>((__ubuf__ T*)buf_[UB3].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[UB1].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[UB0].Get<T>().GetPhyAddr(),
                                     count, VL, repeatTimesS3);

            // ═══ S4: square·mul3_x → mul3 ═══
            // 执行前: 持有=[UB2,UB3]  执行中: 持有=[UB2,UB3,UB4]  执行后: 持有=[UB2,UB3,UB4]
            CopyInBrc(coord, MUL3, UB4, ubBlockLength);
            SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
            // 执行前: 持有=[UB2,UB3,UB4]  执行中: 持有=[UB2,UB3,UB4]  执行后: 持有=[UB3,UB4]
            AscendC::Mul(buf_[UB4].Get<T>(), buf_[UB2].Get<T>(), buf_[UB4].Get<T>(), count);
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2); WaitFlag<HardEvent::V_MTE2>(evVtoMTE2);

            // ═══ S5: MulAddDst → output0 ═══
            // 执行前: 持有=[UB3,UB4]  执行中: 持有=[UB0,UB3,UB4]  执行后: 持有=[UB0,UB3,UB4]
            CopyInBrc(coord, IN1, UB0, ubBlockLength);
            // 执行前: 持有=[UB0,UB3,UB4]  执行中: 持有=[UB0,UB2,UB3,UB4]  执行后: 持有=[UB0,UB2,UB3,UB4]
            CopyInBrc(coord, MUL2, UB2, ubBlockLength);
            SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
            // 执行前: 持有=[UB0,UB2,UB3,UB4]  执行中: 持有=[UB0,UB2,UB3,UB4]  执行后: 持有=[UB3,UB4]
            uint16_t repeatTimesS5 = AscendC::CeilDivision(count, VL);
            asc_vf_call<MulAddVF<T>>((__ubuf__ T*)buf_[UB4].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[UB0].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[UB2].Get<T>().GetPhyAddr(),
                                     count, VL, repeatTimesS5);

            // ═══ S6: sqrt(output0) → sqrt ═══
            // 执行前: 持有=[UB3,UB4]  执行中: 持有=[UB0,UB3,UB4]  执行后: 持有=[UB0,UB3,UB4]
            AscendC::Sqrt(buf_[UB0].Get<T>(), buf_[UB4].Get<T>(), count);

            // ═══ S7: sqrt + add2_y → add2 ═══
            // 执行前: 持有=[UB0,UB3,UB4]  执行中: 持有=[UB0,UB2,UB3,UB4]  执行后: 持有=[UB0,UB2,UB3,UB4]
            CopyInBrc(coord, ADD2, UB2, ubBlockLength);
            SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
            // 执行前: 持有=[UB0,UB2,UB3,UB4]  执行中: 持有=[UB0,UB1,UB2,UB3,UB4]  执行后: 持有=[UB1,UB3,UB4]
            AscendC::Add(buf_[UB1].Get<T>(), buf_[UB0].Get<T>(), buf_[UB2].Get<T>(), count);

            // ═══ S8: output1 / add2 → truediv ═══
            // 执行前: 持有=[UB1,UB3,UB4]  执行中: 持有=[UB1,UB2,UB3,UB4]  执行后: 持有=[UB2,UB3,UB4]
            AscendC::Div(buf_[UB2].Get<T>(), buf_[UB3].Get<T>(), buf_[UB1].Get<T>(), count);
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2); WaitFlag<HardEvent::V_MTE2>(evVtoMTE2);

            // ═══ S9: truediv·input4 → mul4 ═══
            // 执行前: 持有=[UB2,UB3,UB4]  执行中: 持有=[UB0,UB2,UB3,UB4]  执行后: 持有=[UB0,UB2,UB3,UB4]
            CopyInBrc(coord, IN4, UB0, ubBlockLength);
            SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
            // 执行前: 持有=[UB0,UB2,UB3,UB4]  执行中: 持有=[UB0,UB1,UB2,UB3,UB4]  执行后: 持有=[UB1,UB3,UB4]
            AscendC::Mul(buf_[UB1].Get<T>(), buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), count);
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2); WaitFlag<HardEvent::V_MTE2>(evVtoMTE2);

            // ═══ S10: dataVar − mul4 → output2 ═══
            // 执行前: 持有=[UB1,UB3,UB4]  执行中: 持有=[UB0,UB1,UB3,UB4]  执行后: 持有=[UB0,UB1,UB3,UB4]
            CopyInBrc(coord, IN3, UB0, ubBlockLength);
            SetFlag<HardEvent::MTE2_V>(evMTE2toV); WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
            // 执行前: 持有=[UB0,UB1,UB3,UB4]  执行中: 持有=[UB0,UB1,UB2,UB3,UB4]  执行后: 持有=[UB2,UB3,UB4]
            AscendC::Sub(buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), buf_[UB1].Get<T>(), count);

            SetFlag<HardEvent::V_MTE3>(evVtoMTE3); WaitFlag<HardEvent::V_MTE3>(evVtoMTE3);

            // ═══ CopyOut ═══
            // 执行前: 持有=[UB2,UB3,UB4]  执行后: 持有=[UB2,UB3]
            CopyOutOne(coord, OUT0, UB4, ubBlockLength);
            // 执行前: 持有=[UB2,UB3]  执行后: 持有=[UB2]
            CopyOutOne(coord, OUT1, UB3, ubBlockLength);
            // 执行前: 持有=[UB2]  执行后: 持有=[]
            CopyOutOne(coord, OUT2, UB2, ubBlockLength);

            if (flat != end - 1) SetFlag<HardEvent::MTE3_MTE2>(evMTE3toMTE2);
        }
    }

private:
    __aicore__ inline void CopyInBrc(
        const int64_t* coord, int32_t inputIdx, int32_t buffer, int64_t ubBlockLength)
    {
        int64_t k = td_->split.ubSplitIdx;
        int64_t off = CalcOffset(coord, td_->inputStrides[inputIdx], RANK);
        const int64_t* dstShape = td_->maxBroShape;

        auto params = nddmaParams_[inputIdx];
        int64_t kNd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == kNd) params.loopInfo.loopSize[nd] = ubBlockLength;
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = { false };

        if constexpr (RANK <= MAX_NDDMA_DIMS) {
            AscendC::DataCopy<T, ND, cfg>(
                buf_[buffer].Get<T>(), gmIn_[inputIdx][off], params);
        } else {
            LocalTensor<T> buf = buf_[buffer].Get<T>();
            int64_t elemBase = off / sizeof(T);
            for (int64_t oi = 0; oi < nddmaOuterIters_[inputIdx]; oi++) {
                int64_t elemAdj = 0, tmp = oi;
                for (int64_t d = RANK - ND - 1; d >= 0; d--) {
                    int64_t sz = (d == k) ? ubBlockLength : dstShape[d];
                    elemAdj += (tmp % sz) * td_->inputStrides[inputIdx][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<T, ND, cfg>(
                    buf[oi * inner], gmIn_[inputIdx][elemBase + elemAdj], params);
            }
        }
    }

    __aicore__ inline void CopyOutOne(
        const int64_t* coord, int32_t outputIdx, int32_t buffer, int64_t ubBlockLength)
    {
        int64_t off = CalcOffset(coord, td_->outputStrides[outputIdx], RANK);
        int64_t cnt = CalcTransferCount(td_->outputShapes[outputIdx], RANK,
                                              td_->split.ubSplitIdx, ubBlockLength);
        DataCopy(gmOut_[outputIdx][off], buf_[buffer].Get<T>(), cnt);
    }
};

// MulAddVF 定义见 §6.4.4
```
