# GLU 精度诊断

本页只处理 GLU/SwiGLU 已闭合正式执行路径后的精度异常。先证明
`测试模式 -> Host 分支 -> launch symbol -> Kernel 实例 -> Block/Epilogue`
的实际执行路径，并按
[`ops-precision-standard`](../../../ops-precision-standard/SKILL.md)使用项目
锁定的指标、阈值和特殊值规则。本页不修改 Golden，也不为所有 GLU 实现默认
启用精度修正。

## 0. 精度补偿硬门禁

Full 输出出现 mismatch 只能证明最终症状，不能直接证明累加、L0C2UB、Vector
公式或写回是根因。任何 `RefineNearZeroFp16`、近零 Swish 近似或其他精度补偿
进入实现前，必须先完成以下证据闭合：

- 证明实际测试路径到达目标 Kernel、Block 和 Epilogue；
- 使用同一输入、shape、Tiling 和 Golden 完成 C-direct（或等价的 C 分层证据）
  与 V-known 对照；
- 保存最大误差点的 raw C、act、gate、Golden 量级、绝对/相对误差和 K 顺序；
- 分别判断 act 近零、gate 近零、布局/同步和 activation 公式是否为最早失败边界。

如果诊断入口或分阶段 Golden 不可用，状态必须保持 `blocked` 或 `unverified`，
不得通过调大阈值、修改 Golden 或只依据 Full 结果制造 PASS。只有证明普通
MMAD/C 路径和 Vector 公式不是最早失败边界后，才可以进入本页的局部补偿合同；
补偿门限仍由具体算子的 dtype、layout、Golden 顺序和失败证据确定，不得成为
Skill 默认常量。

### 0.1 指标与 probe 的独立门禁

在生成输入和解释失败前，先冻结项目精度标准中的指标定义、非有限值分类、相对误差
分母规则和输出 dtype。验收门禁与诊断统计必须分开：近零值会放大单点的最大相对
误差，因此一个 max-relative outlier 不能单独证明实现错误，也不能单独授权放宽
门禁。MERE、MARE 或其他正式指标的含义必须来自项目合同/精度标准，不能在失败后
临时发明分母 floor、mask 或阈值。

用于定位的有限值范围和命中率只是 probe 覆盖，不是算子支持域。不能因为某个有界
probe 通过就静默收窄输入合同，也不能因为高幅值 probe 失败就改写 Golden。遇到
Full gate 失败时，先用完全相同的 raw 输入、shape、Tiling 和 Golden 重跑
C-direct/V-known，比较 raw C、act、gate、绝对误差和相对误差，再决定是否讨论
公式、布局、同步或局部补偿。

这些门禁针对实际出现过的“高幅值随机数据在近零点触发 max-relative 失败”“把
有限 probe 范围误写成支持域”以及“尚未完成 C 分层就调大 refinement”的原始
问题；具体分母、数值范围和阈值仍由目标算子记录保存。

## 1. 先定位最早失败边界

使用同一组输入、shape、Tiling 和分阶段 Golden，依次运行 C-direct、
C-through-L0C2UB、V-zero-C、V-known-C 和 Full：

- C-direct 失败：检查 A/B 搬运、layout、MMAD、累加顺序、归并和 C 输出。
- C-direct 通过而 C-through 失败：检查 L0C/UB 物理布局、SplitM、C/V pipe、
  UB 地址、pitch 和生命周期。
- C-through 通过而 V-known 失败：检查 activation 公式、逐操作 dtype、API
  近似、mask/tail 和写回。
- 分层模式通过而 Full 失败：检查真实 offset、RAW、slot reuse 和 final drain。

近零相对误差不能仅凭一个 seed 归因。保存最大误差点的 raw C、act、gate、
绝对误差、Golden 量级和 K 顺序对照，证明累加路径或 Vector 公式是最早失败
边界。

## 2. FP16 累加近零修正

参考 GLU `GemmUniversal` specialization 通过
`MatmulDualBranchGlu::REFINE_NEAR_ZERO_FP16` 选择的静态近零修正策略只适用于
以下全部成立的合同：

- A/B 是连续、非转置 FP16，累加和 SwiGLU 输出是 FP32；
- Golden 按 K 从小到大执行逐项 FP32 累加；
- C-direct 已证明普通 MMAD 结果在绝对误差上正确，但 act 或 gate 接近零时，
  累加次序差异违反锁定的严格相对误差门禁；
- V-known 已证明 SwiGLU 公式、局部布局和写回不是最早失败边界。

参考 Kernel 将该边界编码为编译期合同：开启 refinement 且 `LayoutA` 或
`LayoutB` 为 DN 时直接 `static_assert`；关闭 refinement 后，普通 MMAD 仍可使用
连续 ND/DN A/B。NZ/ZN 不属于当前 Grouped GLU 参考资产的已支持范围。

开启后，Host 通过 `Params.refineAbsThreshold` 传入当前算子证据确定的正门限；
`0` 表示禁用。Host 拒绝非有限值和负值。AIV 在 C-ready 后检查本地 act 和
gate；任一路满足门限时，Kernel 按 Golden 的 K 顺序从原始 FP16 x/weight
重新累加该位置的两路值，并覆盖同一 UB tile。

该策略属于 Kernel，因为它需要 group/tile offset、原始 x/weight 和当前 UB
tile。首个消费者变为 Scalar 时，C-ready 必须由 `PIPE_S` 等待；Scalar 写 UB
后使用 `S_V` 建立对 Vector Epilogue 的可见性。该策略不增加第二个 Kernel、
GM workspace 或 L0C2GM 正式路径。

## 3. Swish 近零公式修正

只有 V-known 证明普通 FP32 `Exp/Div` 仅在 act 近零区间违反锁定门禁时，才可
通过 `Params.nearZeroSwishThreshold` 为静态精度 entry 启用局部
`act/2 + act^2/4`；`0` 表示禁用。它与 MMAD 重算门限是两个独立合同，必须
分别取证，不能复制某个算子的数值作为 Skill 默认值。

## 4. 验证和交付

启用任一静态策略时必须记录：

- 门限来源、适用 dtype/layout/公式和静态 entry；
- 门限前、等于、门限后，以及 act 近零和 gate 近零；
- NaN/Inf、多个随机 seed、命中率和重复运行；
- raw C-direct、V-known、正式 Full 和 task time；
- 清理诊断代码后的 clean build 与回归。

逐点 GM 读取是有边界的精度优先策略，不是默认数据平面。没有同公式、同精度
的 matched baseline 时只报告测量值，不声明性能达标。

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 只有 Full mismatch 就调大门限或开启修正 | 最终症状不能证明最早失败边界，且可能掩盖布局/同步错误 | 先完成 C-direct/V-known 证据；入口缺失时保持 `blocked/unverified` |
| 只检查 act 近零 | gate 近零同样可能放大相对误差 | 两路都按同一合同检查 |
| 修改 Golden、放宽阈值或扩大门限制造 PASS | 改变了用户验收合同并扩大额外读取 | 保持合同，验证门限边界和性能代价 |
| 把重算放进 Epilogue | Epilogue 不拥有原始输入和全局 tile 视图 | 由 Kernel 在 C-ready 后、Epilogue 前执行 |
