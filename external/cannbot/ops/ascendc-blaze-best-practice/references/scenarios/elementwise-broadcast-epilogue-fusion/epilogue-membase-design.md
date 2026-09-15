# MemBase Epilogue层设计与实现

本文是本场景 MemBase 路线专题。Step 3 用它比较当前公式的 MemBase 可行性，DESIGN 选定 MemBase 后由 Step 3 编译 PLAN；Step 4 只执行 PLAN。本文不把某个公式、adapter、dtype、tile 或 Asset 声明为通用能力。

## 1. 入口与依赖

必须先加载 [`/ascendc-api-best-practices/SKILL.md`](../../../../ascendc-api-best-practices/SKILL.md)，再按其路由读取当前公式实际需要的 DataCopy、Buffer、pipeline、arithmetic、mask/tail 和真实参考实现资料。直接阅读叶子不能替代根入口。当前 CANN 的实际头文件、目标 Kernel 调用点和 Investigation source refs 优先于旧专题示例。

DESIGN 至少写出：

```text
membase_evidence_refs
formula_dag
operand_distribution_and_broadcast_mapping
adapter_and_params_contract
staging_and_slot_budget
api_and_pipeline_contract
dtype_alignment_mask_tail
output_and_sync_contract
support_boundary
```

## 2. 适用性审查

按当前公式逐节点判断：

1. 每个 operand 是否能以已确认的 LocalTensor/GM/UB 访问表达；
2. broadcast 轴、stride、offset 和有效行列是否可由当前 API 闭合；
3. 中间结果是否需要写回、重读或额外 staging；
4. 每个 dtype 的 Copy/compute/store alignment 和转换是否有证据；
5. tail、mask、空 tile、slot reuse 和 event 生命周期是否不超过资源合同；
6. 公式操作顺序与 CPU Golden 是否一致。

“操作数量少”只能作为比较线索。API、资源、同步或公式映射不闭合时，MemBase 为 blocking/淘汰，不能用历史单操作结果替代。

## 3. Adapter 与地址合同

如果当前 concrete Kernel 提供 `Init`、`GetTensor(slot)` 或 tile operator，逐项引用真实签名和单位；否则设计一个明确的场景 adapter 合同并记录其 ABI owner。禁止从 Asset 或相似 Kernel继承接口。

每次调用必须明确：

- 当前 slot 的 C 起点、元素/字节单位和独立容量；
- 每路 operand 的逻辑到物理 broadcast 映射、GM row/column offset 和 stride；
- `curM/curN`、stage rows、有效 bytes、padding 和 mask；
- 输出目的地、final 时机、错误/拒绝条件和 Kernel release。

不同 dtype 分别计算 alignment 和 row pitch。当前 AIV 若从本地 slot 起点读取 C，不得再次叠加 sub offset；GM full-tensor operands/output 是否增加 sub offset 必须由 DESIGN 证据明确。

## 4. Staging 与流水

DESIGN/PLAN 需要按 slot 预算列出 C、每路 operand、中间值和输出的 byte ranges，以及预取、覆盖前等待、compute、写回和 final drain。常见顺序仅作为待验证方法：

```text
producer ready
-> GM/UB prefetch
-> dependency wait
-> MemBase arithmetic
-> output writeback
-> buffer release
```

每个 event/pipe 必须绑定当前 Kernel 首消费者和 source/device evidence。新增 bridge 必须有只移除 bridge 的负向版本和清理后正向 Full 回归；不能从另一个 Kernel 泛化。

## 5. Asset 规则

[MemBase 示例 Asset](../../../assets/blaze_custom/epilogue/epilogue_fusion_membase.h) 只是可选结构起点，保留原路径和原内容。它不提供正式 adapter、公式、dtype、slot、ratio、同步或支持边界。只有 DESIGN 授权时才由 PLAN 显式复制到项目目标文件；复制后按当前 API、公式、ABI 和验证合同重新适配。

## 6. 输出与回退

MemBase 评估输出为：

```text
route_evidence
api_constraints
formula_mapping
resource_and_sync_budget
known_tail_and_broadcast_limits
selection_status: 首选 | 备选
blocking_or_disposition
```

若与 RegBase 比较后不适用，记录独立处置，不让 Step 4自动切换。若实现或设备证据推翻本合同，回 Step 3；需要新的 Blaze 源码/API 事实先回 Step 2。
