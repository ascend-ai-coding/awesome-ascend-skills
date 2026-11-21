# AdamApplyOneAssign 接口分析

## 1. 来源

- [x] TensorFlow — 无
- [x] PyTorch — 无
- [x] 昇腾独有

## 2. 入参

| # | 形参名 | 对外参数名 | 代码说明的角色 | dtype | 准入 |
|---|--------|-----------|--------------|-------|------|
| 0 | `dataInput0` | `input0` | input of square and mul1 | float16, float32 | REQUIRED_INPUT |
| 1 | `dataInput1` | `input1` | input of mul2 | float16, float32 | REQUIRED_INPUT |
| 2 | `dataInput2` | `input2` | input of mul0 | float16, float32 | REQUIRED_INPUT |
| 3 | `dataInput3` | `input3` | input of sub | float16, float32 | REQUIRED_INPUT |
| 4 | `dataInput4` | `input4` | input of mul4 | float16, float32 | REQUIRED_INPUT |
| 5 | `dataInputMul` | `mul0_x` | input of mul0 | float16, float32 | REQUIRED_INPUT |
| 6 | `dataInputMul1` | `mul1_x` | input of mul1 | float16, float32 | REQUIRED_INPUT |
| 7 | `dataInputMul2` | `mul2_x` | input of mul2 | float16, float32 | REQUIRED_INPUT |
| 8 | `dataInputMul3` | `mul3_x` | input of mul3 | float16, float32 | REQUIRED_INPUT |
| 9 | `dataInputAdd2` | `add2_y` | 代码未说明（docstring 写 "input of mul3"，但实际参与 add2 运算） | float16, float32 | REQUIRED_INPUT |

host侧 `data_dict`（line 229-231）建立了语义名到索引的映射：
- `dataGrad`(0) = input0
- `dataV`(1) = input1
- `dataM`(2) = input2
- `dataVar`(3) = input3

## 3. 出参

| 输出 | 代码说明的角色 | dtype |
|------|--------------|-------|
| `output0` | output of add1 | float16, float32 |
| `output1` | output of add0 | float16, float32 |
| `output2` | output of sub | float16, float32 |

## 4. 计算

```
Step 1  — square:   vmul(dataGrad, dataGrad)          → squareResult
Step 2  — mul3:    vmul(squareResult, mul3_x)         → mul3Result
Step 3  — mul2:    vmul(dataV, mul2_x)                → mul2Result
Step 4  — add1:    vadd(mul2Result, mul3Result)    → output0
Step 5  — sqrt:     vsqrt(output0)                      → sqrtResult
Step 6  — add2:    vadd(sqrtResult, add2_y)           → add2Result
Step 7  — mul0:    vmul(dataM, mul0_x)                → mul0Result
Step 8  — mul1:    vmul(dataGrad, mul1_x)             → mul1Result
Step 9  — add0:    vadd(mul0Result, mul1Result)    → output1
Step 10 — truediv:  vmuls(output1, const(1))            → output1  (恒等)
                    vdiv(output1, add2Result)         → truedivResult
Step 11 — mul4:    vmul(truedivResult, dataInput4)   → mul4Result
Step 12 — sub:      vsub(dataVar, mul4Result)        → output2
```

## 5. Broadcast 链路

```
Step 2:  broadcast(squareResult, mul3_x)
         shape → max(shape(dataGrad), shape(mul3_x))

Step 3:  broadcast(dataV, mul2_x)
         shape → max(shape(dataV), shape(mul2_x))

Step 4:  broadcast(mul3Result, mul2Result)
         shape → max(shape(mul3), shape(mul2))
               = max(dataGrad, mul3_x, dataV, mul2_x)

Step 6:  broadcast(add2_y, sqrtResult)
         shape → max(shape(add2_y), shape(sqrtResult))
               = max(add2_y, output0 的 shape)

Step 7:  broadcast(dataM, mul0_x)
         shape → max(shape(dataM), shape(mul0_x))

Step 8:  broadcast(dataGrad, mul1_x)
         shape → max(shape(dataGrad), shape(mul1_x))

Step 9:  broadcast(mul0Result, mul1Result)
         shape → max(shape(mul0), shape(mul1))
               = max(dataM, mul0_x, dataGrad, mul1_x)

Step 10: broadcast(add2Result, output1)
         shape → max(shape(add2), shape(output1))

Step 11: broadcast(truedivResult, dataInput4)
         shape → max(shape(truediv), shape(input4))

Step 12: broadcast(mul4Result, dataVar)
         shape → max(shape(mul4), shape(dataVar))
```

```
           output0 链路                       output1 链路
   ┌─────────────────────────┐     ┌─────────────────────────┐
   │                         │     │                         │
   │  dataGrad → square     │     │  dataM ─→ mul0       │
   │      │          │       │     │      ↑        │        │
   │      │     [S2] ⊕ mul3_x│     │  mul0_x [S7]  │        │
   │      │          │       │     │               │        │
   │      │       mul3 ─────│─┐   │            mul0 ────┐  │
   │      │                  │ │   │               │       │  │
   │  dataV ───→ mul2      │ │   │  dataGrad → mul1   │  │
   │      ↑         │        │ │   │               ↑       │  │
   │  mul2_x   [S3] ⊕        │ │   │          [S8] ⊕ mul1_x│  │
   │                │        │ │   │               │       │  │
   │            mul2 ───────│─┘   │            mul1 ────┘  │
   │                │        │     │               │         │
   │           [S4] ⊕ ───────│── ─ │ ─ ─ ─ ─ ─[S9] ⊕        │
   │                │        │     │               │         │
   │             add1       │     │            add0        │
   │                │        │     │               │         │
   │              sqrt       │     │           output1       │
   │                │        │     │               │         │
   │         [S6] ⊕ add2_y   │     │               │         │
   │                │        │     │               │         │
   │             add2 ──────│─────│──────────[S10] ⊕        │
   └─────────────────────────┘     └───────────────│─────────┘
                                                   │
                                              truediv ──→ mul4 ──→ sub ←── dataVar
                                                       [S11] ⊕ input4  [S12] ⊕
```

[S2]–[S12] 标注对应 Step 2–12 的 broadcast 调用位置。两条链路在 truediv 汇聚，最终 shape 由所有 broadcast 节点联合决定。

## 6. Shape 约束

- **显式 shape check**: 无
- **隐式约束**: 每次二元操作前由 `shape_broadcast()` 调 `shape_util.broadcast_shapes()` 做 broadcast 兼容性校验，不兼容时由 `param_name_input1`/`param_name_input2` 报错（line 62-63）
- **动态 shape**: 支持。host 侧 `classify(dynamic_inputs, ELEWISE_WITH_BROADCAST)` + `shape_util.variable_shape()` 处理动态 shape（line 251, 256）

## 7. 未说明项

- `add2_y` 的 docstring（line 103）写 "the input tensor of mul3"，但代码实际将其用于 add2 操作（line 139-141），docstring 与代码不一致，代码未说明
- Step 10 中 `vmuls(output1, const(1))`（line 158）对 output1 乘以标量 1，数值恒等，代码未说明其目的
- host 函数 docstring 写 "For bert fuse"（line 186），除此注释外无更多上下文说明该算子与 bert 的具体关系

## 8. 参考文档

- op_desc: 无
- 设计文档：无
- 其他: 无
- **推断边界**: 无文档 → 代码说什么就是什么，不引入外部知识
