# Layout 支持范围与不支持边界专题

本文是本场景的 Layout/变体边界专题。Step 3 用它设计支持域和拒绝域；DESIGN 冻结后，Step 3 用它编译 PLAN 的边界验证。本文不提供固定 Layout recipe。

## 1. 输入与输出合同

输入：`matmul_base_analysis`、Investigation 中已闭合的 Layout Pattern 事实、当前 Prologue/BlockMmad witness。输出：

```text
supported_layout_contract
unsupported_boundary_contract
layout_validation_additions
```

## 2. 当前支持范围

DESIGN 必须基于 Investigation 确认的源码事实冻结支持范围。当前参考支持域（以 Asset 为结构起点，实际由 Investigation 确认）：

| 维度 | 支持 | Layout Pattern | 说明 |
|---|---|---|---|
| A Layout | ND | `NDExtLayoutPtn` | transA=false |
| A Layout | DN | `DNExtLayoutPtn` | transA=true |
| A Layout | NZ | `NZLayoutPtn` | 分形变体 |
| B Layout | ND | `NDExtLayoutPtn` | transB=false |
| B Layout | DN | `DNExtLayoutPtn` | transB=true |
| C Layout | ND（行主序） | `NDExtLayoutPtn` | 输出 |
| transB | true / false | — | 影响分形轴（N 或 K）和 AIV 切分方向 |
| hasOffset | true / false | — | 影响 VF 分支和 UB 布局 |
| hasBias | true / false | — | 影响 BT 路径和 L1 空间 |

## 3. 当前不支持边界

以下变体当前不支持，DESIGN 必须记录不支持原因和恢复路径：

| 变体 | 支持状态 | 条件/原因 | 恢复路径 |
|---|---|---|---|
| int8 weight | 已验证（产线+skill） | 默认支持 | — |
| fp8_e4m3fn / fp8_e5m2 weight | 已验证 | Cast 链（fp8→float→bf16/half）经编译+精度验证通过 | 见 [VF 反量化链路专题](prologue-vf-dequant-design.md) §3.1 |
| pergroup 量化 | 不支持 | 当前只支持 perchannel（scale/offset 按 N 维广播） | 需扩展 Prologue 循环结构和 Tiling |
| StreamK / Full-load | 不支持 | Tiling 仅提供 SWAT 流式路径 | 需新增 Scheduler 和 Tiling Engine |
| Weight-Quant + Vector Epilogue 后融合 | 不支持 | Epilogue 固定为 void（直写 GM） | 需在 Kernel 中扩展 Epilogue 模板参数 |

## 4. 互斥性声明

本场景与 `elementwise-broadcast-epilogue-fusion` 场景互斥：

| 对比维度 | 本场景（weight-dequant-prologue） | elementwise-broadcast-epilogue |
|---|---|---|
| 数据流方向 | V+C（AIV 先反量化 → AIC 后 MMAD） | C+V（AIC 先 MMAD → AIV 后处理） |
| Vector 位置 | MMAD 之前（Prologue） | MMAD 之后（Epilogue） |
| 自定义层 | BlockMmad(替换) + Kernel(替换) + Prologue(新增) | Epilogue(新增) |
| 同时命中 | 不可能：同一需求不能既是"先 Vector 后 MatMul"又是"先 MatMul 后 Vector" | 同左 |

如果需求同时包含 V+C prologue 和 C+V epilogue，则两个场景都不唯一命中，Step 3 输出 `unsupported`。

## 5. 与纯 Quantized MatMul 的区分

本场景不属纯 Quantized MatMul（A8W8/MX）范畴。区分依据：

| 对比维度 | 纯 Quantized MatMul | 本场景（weight-dequant-prologue） |
|---|---|---|
| 量化路径 | Blaze 原生 scale 路径（A8W8/MX） | AIV 侧反量化 prologue |
| B 输入 | Blaze 支持的低比特格式 | int8 权重，需 device 侧反量化 |
| Blaze 覆盖 | `Blaze::Gemm` 原生支持 | Blaze 无此入口（native_gap） |
| 路线 | blaze_native | blaze_custom |

## 6. 验证门禁

- 支持范围内的每种 Layout/transpose/hasOffset/hasBias 组合均有验证用例；
- 不支持边界有明确的拒绝条件和错误信息；
- transB=true 和 transB=false 的分形轴对齐约束分别验证；
- NZ A Layout 的分形变体路径闭合；
- 与 elementwise 场景的互斥性通过跨场景用例验证；
- 官方源码区与 Blaze Asset 原文件保持零改动。

具体 Layout Pattern、dtype 和 shape 组合来自 DESIGN/PLAN 和当前 Investigation，不由本文固定。
