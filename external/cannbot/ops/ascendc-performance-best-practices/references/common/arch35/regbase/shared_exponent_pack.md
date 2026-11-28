# `shared-exponent-pack`

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

## 解决的问题

把 max exponent、shared exponent、reciprocal scale、量化 Cast、双半部 Pack/Interleave 和 Store 组织为一个寄存器流水，避免用通用数学链和 UB 临时结果替代位级算法。

## 为什么这样写

- shared exponent 来自输入浮点位级表示；用掩码、移位和 Pack 可保留目标格式的精确编码，通用 `log2/pow/div` 不等价。
- reciprocal 通过指数变换生成，但必须依次修正 Inf、特殊指数和 0；删除任一 `Select` 都会改变特殊值结果。
- FP32 与 B16 的 scale Pack/存储布局不同，`LOWEST/HIGHEST/Or` 的组合由 lhs/rhs 物理布局决定。
- `ComputeMaxExponents → ComputeScalesAndSharedExp → QuantizeValues` 共用寄存器结构和常量；拆成多个 UB pass 会丢掉这条流水及其布局关系。

## 怎么使用

- 本地注解代码：[`code/shared_exponent_pack_vf.h`](code/shared_exponent_pack_vf.h)
- 重点函数：`ComputeMaxExponents`、`ComputeScalesAndSharedExp`、`ComputeReciprocalScale`、`QuantizeValues`，以及按 dtype 分派的 `ProcessQuantization*`
- 源构建契约：`unverified`；本地摘录不是独立类，当前无最终编译命令

1. 先确认目标格式、`calcMode`、输入/输出 dtype、lhs/rhs 和 interleave 布局与摘录相同。
2. 离线读取本地摘录，以顶层调用顺序为骨架，从目标工程补齐寄存器结构、常量、LoadData 和 `ProcessQuantization*`；不要重新推导位掩码。
3. 把 RNE bias、特殊值修正、Pack 高低半和 Store 地址作为一个精度单元迁移。
4. 用 0、次正规数、边界指数、Inf/NaN、奇偶 tie、lhs/rhs 交错和尾块做位级 golden 验证，再测性能。

## 前置条件

- Arch35 / DAV_3510。
- `calcMode`、输入/输出 dtype、是否可交错、是否有 RHS 与目标布局完全匹配。
- exponent/mantissa 位掩码、特殊值修正、Pack 高低半部和 StoreDist 必须整体迁移并做位级精度验证。

这是 `annotated_excerpt`：关键真实指令链已随库提供，不要求访问网络；但它不是完整类。省略的寄存器结构、常量和 dtype 分支必须从目标工程的同构实现补齐，不能根据几条 `Reg::And/ShiftRights/Pack` 重新发明公式。
