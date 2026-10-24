# broadcast 范式 standard 计算流伪码

> 计算流伪码是物理存活节点（P）分析的序列化产物，物理分析到 kernel 编码之间的桥梁。

依赖输入：broadcast-standard-dag-buffers.md（P trace 结论）、broadcast-standard-kernel-template.md（§1.2 IR 描述、§8 产出校验）

## 1 定义与定位

### 1.1 什么是计算流伪码

计算流伪码是 [broadcast-standard-dag-buffers.md](broadcast-standard-dag-buffers.md) 物理存活节点分析的序列化产物。P trace 选定了三要素组合（API 能力 + 融合策略 + buffer 分配），伪码把这一选择写成 `DataCopy → 计算 → 释放` 的完整序列。每步标注 UB 驻留数，验证全程 ≤ P。

伪码不是实现——它是物理分析到 kernel 编码之间的桥梁。后续所有设计（计算流验证、输入预处理、UB 切分、kernel 编码）都以这份伪码为唯一参照。

### 1.2 与 kernel IR 描述的关系

[broadcast-standard-kernel-template.md](broadcast-standard-kernel-template.md) §1.2 的 kernel IR 描述是高层概述（`out[i] = f(x, y)`），说明计算语义。计算流伪码是详细序列化展开，包含：
- 每步的 buffer 分配/释放
- UB 驻留计数
- 融合和 broadcast 的具体嵌入

IR 描述回答"算什么"，计算流伪码回答"怎么算、用哪些 buffer、何时释放"。

### 1.3 与 kernel 产出校验的关系

[broadcast-standard-kernel-template.md](broadcast-standard-kernel-template.md) §8 的 golden for 循环是验证基准，纯计算逻辑。计算流伪码是验证的输入——§8 的 golden 应与计算流伪码语义等价。

### 1.4 物理意义

P trace 的物理节点分析是"每一步 peak 多少"的抽象论证。伪码把它落实到每一行的具体操作：哪个 buffer 在哪个时刻被创建、被消费、被释放。它回答三个问题：
- P 能否兑现为一个具体的步骤序列？
- 每步的 UB 驻留是否始终 ≤ P？
- 融合（如 MulAddDst）和 broadcast（NDDMA 随路）是否正确嵌入？

## 2 何时需要

agent 自行判断是否执行：

| 条件 | 是否执行 |
|------|---------|
| 链长 ≤3、单输出、无 Cast+VF 混合 | 跳过，P trace 已充分表达 |
| P trace 已充分表达计算步骤 | 跳过 |
| 链长 >5、含分支/并行路径、多输出 | 建议执行 |

## 3 规范格式

### 3.1 头部声明

```
输入:  <输入变量列表>
输出:  <输出变量列表>
约束:  UB 最多同时驻留 P 块 buffer
```

### 3.2 步骤格式

每步包含：
1. **注释标题**：`// ─── Step N: <操作描述> ───`
2. **操作行**：`<操作>(<参数>) → <bufXxx>`，DataCopy 标注搬运方向和 broadcast 策略
3. **UB 驻留计数**：`// UB: [<buf1>, <buf2>, ...] = N`
4. **释放注释**（如有）：`// 释放 <bufXxx>, <bufYyy>`
5. **释放后驻留**（如有释放）：`// UB: [<buf1>, ...] = N`
6. **峰值标注**（如达到 P）：在 Step 标题行标注 `← 峰值 P`

### 3.3 dtype 分路径

每种 dtype 路径分别写一份计算流伪码。不同 dtype 路径可能因 Cast 插入而步骤数不同——Cast 不参与 VF 融合，作为链的断点，可能增加中间 buffer 和步骤。

### 3.4 搬出段

所有计算完成后，统一搬出所有输出：
```
// ─── 搬出 ───
DataCopy(output0, UB→GM)
DataCopy(output1, UB→GM)
```

多输出时，MTE3 顺序执行，CopyOut 之间不需额外同步。输出 buffer 在 CopyOut 前不可被覆写。

## 4 关键设计决策表

| 决策 | 选择 | 理由 |
|------|------|------|
| broadcast 策略 | NDDMA 随路 / 独立 broadcast | <理由> |
| 融合策略 | MulAddDst / 无融合 | <理由> |
| output 持久化 | 保留至搬出 / 提前释放 | <理由> |
| RegBase | 使用 / 不使用 | <理由> |

## 5 结论

N 步序列全程 UB 驻留 ≤ P，P=<值> 可兑现。<融合策略> 消化了 <具体中间 buffer>，<broadcast 策略> 消化了 <具体 broadcast 操作>。

## 6 完整示例

### 6.1 简化示例（二元算子，P=3）

```pseudocode
输入:  x, y
输出:  out
约束:  UB 最多同时驻留 3 块 buffer

// ─── Step 1: 载入 x ───
DataCopy(x, GM→UB, NDDMA_broadcast)   → bufX
// UB: [bufX] = 1

// ─── Step 2: 载入 y，算 mul ───  ← 峰值 3
DataCopy(y, GM→UB, NDDMA_broadcast)   → bufY
AscendC::Mul(bufOut, bufX, bufY)       → bufOut
// UB: [bufX, bufY, bufOut] = 3
// 释放 bufX, bufY
// UB: [bufOut] = 1

// ─── 搬出 ───
DataCopy(out, UB→GM)
```

关键设计决策：

| 决策 | 选择 | 理由 |
|------|------|------|
| broadcast 策略 | NDDMA 随路 | 不另占 UB buffer |
| 融合策略 | 无融合 | 单步二元操作，无相邻 Vector 可融合 |
| output 持久化 | 立即搬出 | 无下游消费 |
| RegBase | 不使用 | 二元操作 src 本就在 UB，RegBase 不降节点 |

结论：2 步序列全程 UB 驻留 ≤ 3，P=3 可兑现。

### 6.2 复杂示例（AdamApplyOneAssign算子，P=5，含 MulAddDst 融合）

```pseudocode
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
Div(bufTruediv, output1, bufAdd2)               → bufTruediv
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

关键设计决策：

| 决策 | 选择 | 理由 |
|------|------|------|
| broadcast 策略 | 全部 DataCopy 时随路 NDDMA broadcast | 不另占 UB buffer |
| 融合策略 | S7+S9 → MulAddDst, S3+S4 → MulAddDst | 各省 1 中间 buffer |
| S10a 恒等乘 vmuls(output1,1) | 省略 | x×1.0 数学恒等（IEEE754 下 NaN/±Inf/-0 均不变），原图中是 broadcast 物化的历史产物；kernel 直接以 Div 消费 output1，省每 tile 一次全量向量乘 |
| output0/output1 持久化 | 保留至搬出，不提前释放 | 下游 truediv 和最终搬出都需要 |
| RegBase | 不使用 | 该算子二元操作 src 本就在 UB，RegBase 不降节点 |

结论：10 步序列全程 UB 驻留 ≤ 5，P=5 可兑现。NDDMA broadcast 消化了所有独立 broadcast 操作，MulAddDst 消化了 S7+S9 和 S3+S4 的中间 buffer。
