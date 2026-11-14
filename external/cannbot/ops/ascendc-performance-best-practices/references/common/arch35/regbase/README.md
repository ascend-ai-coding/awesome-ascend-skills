# Reg VF 范例库

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> 本 catalog 的实现范围是 DAV_3510 / A5。先按 [代际范围](../../architecture-scope.md) 筛选；A2/A3 不触发下文的强制 VF 检索，readiness 不代表跨代兼容。


本目录按**计算模式**组织 VF 算法条目和本地代码。`catalog.json` 记录匹配提示、本地代码路径、构建契约和接入形式；每个条目说明“为什么这样写”和“怎么使用”。它和 `ascendc-regbase-best-practice` 都是实施专家可按需查询的手册：本目录提供本地算法范例，后者提供开源算子检索以及出问题后的 API/精度知识。

实施顺序：编码前同时查本目录 catalog/代码和 `ascendc-regbase-best-practice` 的开源算子表，命中任意一边即可作为参考，但必须实际打开代码；然后实现并编译。首次编译/集成失败后才读 [CANN / Reg VF 编译问题 FAQ](cann_version_compat_faq.md)；API 或精度问题按需进入 `ascendc-regbase-best-practice` 的 API/pitfalls 索引，不需要其 design/paradigm 部分。

## 先看 readiness

| readiness | 能做什么 | 不能做什么 |
|---|---|---|
| `integration_skeleton` | 复用已收录的 VF 函数，补齐目标工程上下文 | 必须在目标工程完成编译、精度和性能验证 |
| `annotated_excerpt` | 离线读取本地真实代码摘录和逐段注解，作为目标工程适配骨架 | 摘录不是独立类；省略上下文必须由目标工程补齐并重验 |
| `reference_only` | 做算法和 shape/layout 选型 | 在完成目标工程最小编译复核前不能称为实现候选 |

## 怎么找到条目

先用源码语义、公式和 shape/layout 匹配 `catalog.json` 的 `match_hints`，再读条目说明；不要只搜现有 API 名。特别是：

- 看到 `rope`、`rotary`、`rotary position embedding`，或 `x*cos + rotate(x)*sin`、前后半交换乘加，即使基线没有 `Reg`/`MicroAPI`，也必须把 `paired-segment-rotate` 列为候选，完整读取其说明和代码后再按布局排除。
- 量化先按数据依赖路由：MX/shared-exponent/block-scale/FP4-FP8 Pack 优先匹配 `shared-exponent-pack`；同一 UB 行先求 amax/min-max scale、再做动态量化匹配 `two-pass-amax-cast`。仅有量化 dtype、外部静态 scale、dequant/requant 或量化 MatMul 名称时不得硬套；须记录两个条目的选中/排除理由。
- 向量计算链（elementwise、activation 或其他向量运算，不限算子类型）已 UB 驻留且 VEC 耗时主导时，把 `register-resident-vector-chain` 列为候选：整链寄存器化、消除每个向量 API 的 UB 读写与链内 staging。其他条目未命中不排除该路径，这是深度 VEC bound 的正常分析路径而非例外。
- 命中候选不等于允许复制。仍须比较 `preconditions`、`readiness`、源/目标构建契约，制定目标工程的编译、精度和性能验证方案。

## 计算模式

| 计算模式 | 范例 ID | readiness | 入口 |
|---|---|---|---|
| 短归约轴转置后沿长伴生轴向量化 | `reduce-small-axis-transpose` | `integration_skeleton` | [说明](reduce_small_axis_transpose.md) |
| 单行或多行全量驻留 UB，稳定归约后逐元素输出 | `reduce-row-full-load` | `integration_skeleton` | [说明](reduce_row_full_load.md) |
| 跨 UB chunk 维护 running max/sum | `reduce-online-max-sum` | `integration_skeleton` | [说明](reduce_online_max_sum.md) |
| RoPE/rotary 或两个对齐分段以不同 offset load，乘加留在寄存器 | `paired-segment-rotate` | `integration_skeleton` | [说明](paired_segment_rotate.md) |
| amax/min-max 与 Cast/Pack 两遍扫描同一 UB 数据 | `two-pass-amax-cast` | `annotated_excerpt` | [说明](two_pass_amax_cast.md) |
| shared exponent、reciprocal、低比特 Pack 寄存器流水 | `shared-exponent-pack` | `annotated_excerpt` | [说明](shared_exponent_pack.md) |
| 任意向量计算链整链寄存器驻留：load 一次、链内接力、store 一次 | `register-resident-vector-chain` | `integration_skeleton` | [说明](register_resident_vector_chain.md) |

目录结构、讲解章节、代码摘要和构建探针检查：

```bash
python3 ops/ascendc-performance-best-practices/scripts/validate_vf_examples.py
```

代码与说明用于完成选型和目标工程适配；条目未覆盖的分支应明确记录为范例缺口。

## 实施顺序

1. 同时查本目录 catalog/代码和独立 RegBase skill 的开源算子表，命中任意一边后实际打开代码并记录可/不可复用点；两边均无时记录 `no_matching_reference_implementation`。
2. 以 shape、布局、UB 容量和精度语义选择模式，不按算子名硬套。
3. 读取 catalog 的 `match_hints`、`readiness`、本地代码路径、`example_build_contract`，再完整读取命中代码与说明。
4. 运行 `scripts/vf_preflight.py`，比较范例/目标构建契约；实施契约 `expectations.vf/casttrait` 为 true 时最终调用分别增加 `--expect-vf`/`--expect-cast`，禁止删除实现后以 `not_applicable` 通过。不同或未知时先做单 dtype、单 VF 最小编译探针；涉及 `CastTrait` 时使用 `scripts/run_reg_cast_probe.py` 的固定正例，不临时手写。
5. 新代码使用 `AscendC::Reg::`；不修改 SDK，不手工补命名空间别名，不凭记忆拼 API。
6. 首次编译/集成失败后读 FAQ，保存完整证据、只处理首错并做同环境反例/最小探针；API 或精度问题改查独立 RegBase skill 对应索引。
7. 在目标工程分别完成编译、精度、性能验证；结论须有对应的目标环境证据。

外层 `TPipe`、`TQue`、`LocalTensor` 和 GM↔UB `DataCopy` 不等于 VF 退化；判断标准是热计算链、mask/tail、Cast/Pack 是否仍在寄存器路径并通过验证。
