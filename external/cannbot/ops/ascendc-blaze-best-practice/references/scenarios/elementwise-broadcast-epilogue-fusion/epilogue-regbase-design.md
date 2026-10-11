# RegBase Epilogue层设计与实现

本文是本场景 RegBase 路线专题。Step 3 用它比较当前 elementwise/broadcast 公式的 RegBase 可行性，DESIGN 选定后由 Step 3 编译 PLAN；Step 4 只执行 PLAN。本文不固定公式、VF 宽度、guard、adapter、dtype、tile 或 slot。

## 1. 入口与依赖

必须先加载 [`/ascendc-regbase-best-practice/SKILL.md`](../../../../ascendc-regbase-best-practice/SKILL.md)，读取其 Reg API、限制、pitfalls 和真实 RegBase 参考实现。若实际使用 DataCopyPad、HardEvent 等普通 AscendC API，同时加载 [`/ascendc-api-best-practices/SKILL.md`](../../../../ascendc-api-best-practices/SKILL.md)。依赖 Skill 根入口不能由叶子链接替代。

DESIGN 至少记录：

```text
regbase_evidence_refs
formula_dag
operand_distribution_and_broadcast_mapping
vf_register_contract
adapter_and_params_contract
staging_and_slot_budget
active_mask_and_tail_contract
event_and_pipeline_contract
output_and_support_boundary
```

## 2. 适用性审查

逐节点验证：

1. 每个 operand 的 GM/UB 物理映射和 Reg/VF load API；
2. broadcast index、row/column pitch、有效列和尾部 mask；
3. 注册器数量、临时值、stage rows 和 slot 容量；
4. 整寄存器读取是否在 active mask 前发生，以及 guard 是否必要；
5. dtype、cast、round、saturate 和 store 顺序；
6. event、buffer reuse、首消费者和 final drain。

Mask 只限制明确允许的 arithmetic/store。若 API 在 mask 前整向量读取，容量 guard、alignment 和合法地址必须单独证明，不能以 active mask 代替。

## 3. Adapter、寄存器和地址

当前 concrete Kernel 的 `Init`、`GetTensor(slot)`、tile operator、Params 字段和返回单位必须从 Investigation 调用点抄录；不同 Kernel 重新恢复，不得从示例 Asset推断。

每个 slot 单独预算：

```text
C range
operand staging ranges
intermediate/register lifetime
output range
optional load guard
slot reuse dependency
```

每路 operand 独立计算 dtype alignment、row pitch、GM offset、stage offset 和有效 `curM/curN`。若 SplitM 启用，UB C 的 sub 起点和 GM full-tensor 的全局 sub 行偏移分别记录；禁止对 UB C 重复增加 sub offset。

## 4. VF 计算合同

以当前公式为准建立伪代码/字段表，不固定某种操作链：

```text
for each valid local row:
    load each operand according to its mapping
    derive active mask from effective columns
    execute formula DAG in DESIGN order
    store only valid output elements
```

记录 VF API、数据类型、mask 更新、寄存器临时值、中间结果和输出写回证据。计算逻辑不能依赖 padding；所有加载和存储范围必须在当前 slot/GM 合同内。

## 5. 本地流水与事件

按当前 API 和 Kernel pipe 设计：

1. 预发射每路 operand 的 GM->UB 或 Reg load 依赖；
2. 等待对应 buffer/slot 可读或可覆盖；
3. 执行 VF 公式；
4. 释放输入、中间和输出依赖；
5. 完成输出写回和 final drain。

任何 MTE/V/Event bridge 必须引用当前控制流的 source/device evidence，并用“只移除该 bridge”的负向对照证明必要性；不得把其他路线的同步常量或性能判断带入。

## 6. Asset 规则与输出

[RegBase 示例 Asset](../../../assets/blaze_custom/epilogue/epilogue_fusion_regbase.h) 只提供结构骨架，不能声明公式、active mask、guard、adapter、dtype、slot、ratio 或 ABI。只有 DESIGN 授权时才在 PLAN 中复制到项目目标并重新适配。

RegBase 评估输出为：

```text
route_evidence
api_and_reference_implementation_constraints
formula/vf/mask_mapping
resource_and_event_budget
known_broadcast_and_tail_limits
selection_status: 首选 | 备选
blocking_or_disposition
```

若资源、API 或同步不兼容，记录独立处置并回场景设计；Step 4不能自动切换 MemBase。设备结果推翻前提时回 Step 3，需要新事实先回 Step 2。
