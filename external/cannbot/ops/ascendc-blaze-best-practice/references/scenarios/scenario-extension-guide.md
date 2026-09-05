# Blaze 定制扩展场景接入指导

本文只用于新增或维护 `/ascendc-blaze-best-practice` 自身的定制扩展场景，不属于普通算子任务的 Step 1--4 阅读路径。运行普通算子任务时不要读取本文。

## 1. 最小接入单元

新增场景只提交一个高内聚单元：

```text
references/scenarios/<scenario-id>/
  <scenario-id>-design.md
  <scenario-id>-development.md
  <optional focused references>
references/scenarios/index.md 中一个注册项
```

当前只支持唯一命中一个定制场景，不支持多个场景组合。除非能证明通用合同无法表达，新增场景不得修改 Step 2、Step 3 或 Step 4 的流程骨架；与现有场景的准入条件可能同时满足时必须收窄边界，否则让 Step 3 输出多命中 `unsupported`。

## 2. 先定义需求语义边界

1. 使用稳定、唯一、kebab-case 的 `scenario_id`，不包含版本、芯片快照或 Blaze 组装方案名称。
2. 定义支持范围，且在同一范围中明确排除的语义；只引用 Step 3 需求合同中可观察的语义，不引用 Blaze 源码文件名、Asset、候选优劣或设备结果。
3. 定义全部准入条件；每项都必须能由精确需求合同判断，并且不能与现有场景同时满足。
4. 证明它不是 Basic、Batch、Grouped、Quantized、MX、layout、transpose 或 shape 等纯 MatMul 维度；详细不支持边界和用户恢复路径写入场景设计指导。

## 3. 编写场景设计指导

场景设计指导必须声明：

```text
scope and exclusions
required Step 3 contracts
required Blaze source facts from the Investigation report
whether matmul_base_analysis or another common analysis is consumed
how each prerequisite is checked
one generic Step 2 supplement-question format
dependency Skill loading gates
design decisions and evidence boundaries
consumed/preserved/replaced/added contracts
interface, Kernel, resource, data, synchronization and validation outputs
customization authorization
unsupported boundary and return targets
```

Step 2 不读取注册表或任何场景文档。场景不能通过修改 Step 2、要求固定场景 bundle 或令 Step 2 感知场景 ID 来获得调查事实。发现前提缺失时，设计指导只允许提出一次无场景名的补充问题：问题须描述需求语义、待确认的 Blaze 源码关系、影响 requirement IDs 和有限源码读取前沿。

场景是否消费 Step 3 的 Blaze 官方库分析由自己定义。若不消费 `matmul_base_analysis`，不得把它列为必需输入。设计指导只能在 Step 3 的 DESIGN 冻结前读取，不能把开发动作塞入 DESIGN，也不能把选择留给 Step 4。

## 4. 编写场景开发指导

开发指导是 Step 3 在 DESIGN 冻结后编译项目 PLAN 的方法输入，必须提供：

```text
plan inputs and implementation_route=blaze_custom condition
required readings and dependency loading rules
ordered action rules and prerequisites
target file rules and implementation constraints
build/Tiling/Launcher/data/Golden wiring rules
validation checkpoints and deliverables
cleanup and rollback
```

开发指导只定义可实例化的方法和条件分支，不写某个项目的固定 Blaze 组装方案、dtype、shape、slot、ratio、文件清单或验证次数。Step 3 必须把激活要求展开到 PLAN 的 `scenario_guidance_compliance`、`reading_manifest`、`ordered_actions`、checkpoint 和交付件；Step 4 不直接解释开发指导。

## 5. 注册合同

在 [场景索引](index.md) 增加唯一一行：

1. 填写稳定且唯一的场景 ID、支持范围、全部准入条件、设计指导链接和开发指导链接。
2. 支持范围必须同时说明支持语义和明确排除的语义；准入条件只使用 Step 3 需求合同可判断的信息。
3. 依赖 Skill、调查前提、合同依赖、定制范围、不支持边界、维护规则和验证要求只写入场景设计/开发指导，不写入索引。
4. 不新增 JSON、marker、schema、场景 bundle、预匹配、支持状态或路线状态字段，也不维护第二份目录表。

## 6. 接入验证

至少验证：

- 索引行的场景 ID 唯一，支持范围和准入条件非空，设计/开发链接均可访问；
- 正例唯一命中，范围外语义不命中；与每个已启用场景的重叠用例产生明确 `unsupported`；
- Step 2 不读取注册表、设计指导、开发指导或依赖 Skill；
- Step 3 仅在官方 Blaze 方案存在 `native_gaps` 时读取注册表，并按 `design -> DESIGN -> development -> PLAN` 顺序加载；
- 场景源码前提缺失时只产生一次无场景名补充调查，不生成可执行 PLAN；
- Asset、官方源码和未注册 custom 层未被授权；
- Markdown 链接、Skill 校验和 `git diff --check` 通过。
