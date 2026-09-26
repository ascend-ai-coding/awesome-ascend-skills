# Elementwise/Broadcast Epilogue 精度诊断

本文是本场景的验证方法专题。Step 3 用它设计验证合同并编译 PLAN；Step 4 以 PLAN 初始验证基线为起点，按实际失败证据读取本文并执行诊断。本文不选择路线、不维护 SplitM 公式，也不把固定 case 数写成通用支持矩阵。

## 1. 基线合同

每个项目在 DESIGN/PLAN 中冻结：

```text
logical_inputs_and_seed
physical_conversion_and_packing
formula_and_dtype_order
shape/layout/broadcast/slot/split partitions
actual blockShape and Tiling
CPU golden
atol/rtol/nonfinite gate
required negative/positive comparisons
repeat policy
```

阈值由需求和数值合同决定；必须全元素通过且拒绝非有限值。若当前 FP32 链额外要求零误差，写入该项目合同，不外推到其他 dtype/公式。构建、设备结果和 `device_verified` 只绑定当前 Investigation、目标 Blaze 组装方案和验证记录。

### 1.1 公式的数值稳定性

代数等价不等于设备 dtype 下的数值等价。诸如 tanh/GELU 的指数形式可能在
大正值上先产生 `inf`，随后出现 `inf/inf -> NaN`；该 NaN 还可能让 absmax
漏掉真正的最大值，最终同时污染 `yScale` 和 `y`。遇到此类首错时，先保留用户
Golden 和验收公式，使用大正值、大负值、零附近及非有限值策略的定向输入比较
`workspace -> yScale -> y`。只有在用户合同允许、dtype 顺序和特殊值行为已证明
一致时，才能替换为数值稳定的等价实现；不能用设备 PASS 反向授权改写 Golden，
也不能用放宽容差掩盖产生 NaN 的中间结果。稳定实现的阈值、常数和公式仍属于
目标 DESIGN/PLAN，不写成通用 Skill 常量。

## 2. 五种隔离模式

所有模式使用同一逻辑输入、物理转换、Tiling 记录和 Golden，只改变 DESIGN 声明的隔离变量：

| 模式 | 隔离目标 | 必须记录 |
|---|---|---|
| C-direct-GM | MatMul 主体与最终 GM 输出 | 真实 GM 分支/等价路径、C Golden、MatMul 证据 |
| C-through-fusion | L0C2UB、slot、C-ready、identity writeback | adapter、slot、同步、C 输出 |
| V-zero-C | 额外 operand、公式、mapping、mask/tail、写回 | 零 C 生成/注入方式、输入地址和 V Golden |
| V-known-C | 非零 C、adapter、offset、同步 | known-C 产生方式、真实 wait/release、V Golden |
| Full | 完整 MatMul + elementwise/broadcast Epilogue | 全公式、reuse、final drain、重复回归 |

未激活的条件性模式必须在 DESIGN 说明 N/A 依据。C-direct 不证明 L0C2UB/Vector；单独通过模式不替代 Full。

## 3. 最早失败域

| 最早失败 | 责任域 | 优先核对 |
|---|---|---|
| C-direct-GM | MatMul 主体 | 输入/layout、Tiling、Mmad、归并、GM output |
| C-direct 通过且 C-through 失败 | MatMul->Vector 交接 | L0C2UB、物理 pitch、slot、GetTensor 单位、C-ready pipe |
| 前两项通过且 V-zero/V-known 失败 | Vector | operand mapping、mask、公式顺序、写回、dtype |
| 隔离模式通过且 Full 失败 | 组合生命周期 | RAW、offset、slot reuse、同步、final drain |

V-zero 通过而 V-known 失败时，优先比较非零 C 的物理 layout、formula order、mask 和原地依赖。结论必须附首错位置和对应 source/device evidence。

## 4. 单变量负向/正向

对每个候选根因建立：

```text
baseline_id
changed_variable
negative_reproduction
positive_recovery
affected_boundary_cases
full_repeat_result
```

只移除一个 bridge、offset、trait、mask、slot wait 或 staging 变量；不能把多个改动一起作为根因。只有负向稳定失败、正向稳定恢复、相关边界和清理后 Full 都通过，才能将结论写成 `device_verified`；否则为 `unverified` 或保护性调整。

## 5. 数据与 Dump

优先使用 row-encoded、column-encoded、one-hot 或其他能区分 M/N/offset 的数据。全常量输入不能证明地址正确。V-known-C 可以由真实 MatMul 产生已知非零 C，也可以在 DESIGN 授权的诊断点注入，但必须保留真实 wait/release/final drain，不得跳过 Kernel 生命周期。

Dump/诊断点只放在已确认 producer 完成之后，记录 tile、slot、sub、stage、row/column、元素单位和物理边界。Dump 可能改变时序；首错确认后必须移除 Dump 并复现，不能把 Dump 版本作为修复证据。

CPU Golden 由 PLAN 指定的 host/Python 逻辑生成；C++ Launcher 只执行设备 Kernel、同步和写结果，不现场计算 Golden 或改变阈值。

## 6. 清理与交付门禁

交付前必须删除 PLAN 和 `execution_record` 中记录的 identity/zero/known-C 注入、故障开关、诊断 Params、Dump 和临时输出；清理后重新构建，并执行 DESIGN 声明范围内的五模式、边界、multi-tile、slot reuse、SplitM（如适用）和 Full 重复回归。

最终记录：实际输入/shape/dtype/layout、Tiling/blockShape、slot/sub、错误统计、非有限值、source/device evidence、支持/未验证范围和清理状态。只有清理后 required checkpoints 全部通过，PLAN 才能标记交付完成。
