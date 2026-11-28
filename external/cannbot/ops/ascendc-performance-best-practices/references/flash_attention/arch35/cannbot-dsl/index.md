# FA 族 CANNBot-DSL 实现知识（arch35）

本目录沉淀 **FA 族的 CANNBot-DSL 实现知识**（arch35 / A5），分两层组织；跨算子族的公共部分（五段链方法论、DSL 编程模型要点、证据纪律、建库规则）见 [CANNBot-DSL 公共实现知识](../../../common/arch35/cannbotdsl/index.md)，本目录不重复。[SparseFlashAttention（SFA）](assemble/index.md) 是目前唯一端到端验证过的完整实例（seed 逐位一致、37 步消融、多次盲测复现），在两层中作为贯穿示例引用——同构计算链（MLA-absorb、稀疏 FA、paged 解码）按"替换到同类算子"路径拓展，异构链按方法链逐层重建，SFA 只作示例对照，其结论不随之迁移。

## 第 1 层：optimize/ —— DSL FA 算子怎么优化（用 SFA 举例）

讲优化手段本身：数学语义建模 → 流水建模（AIC+AIV 延迟线）→ 容量方程（四层法），每篇方法在前、SFA 实例在后。入口：[optimize/index.md](optimize/index.md)。

| 步 | 手段 | 进入信号 | 文档 |
|---|---|---|---|
| 1. 数学语义建模 | 把"算什么、保留什么语义"写成可检验约束 | 新变体/新计算链建模起点 | [1-math-semantics](optimize/1-math-semantics.md) |
| 2. 流水建模 | 双核独立延迟线流水 | CUBE busy < AIC 窗口（出洞）或 AIV 断流；一侧 SCALAR 贴满但 CUBE <97% | [2-pipeline-modeling](optimize/2-pipeline-modeling.md) |
| 3. 容量方程 | 四层法台账求解（贴顶即最优） | 需定 tile/环深/滞后，或调参前缺容量证明 | [3-capacity-equations](optimize/3-capacity-equations.md) |

## 第 2 层：assemble/ —— 用积木与胶水组装高性能 FA

讲怎么把手段落成代码：积木（modules/，M0-M4 按可复用功能命名的整类投放件）+ 胶水（glue-idioms 表达层规则 / glue-fragments 拼接件），含组装配方、模块台账、使用规则与替换路径。入口：[assemble/index.md](assemble/index.md)。

| 材料 | 内容 | 文件 |
|---|---|---|
| 积木（第 4 层） | 跨核通道/MM/softmax/主循环/host 装配等整类模块 | [modules/](assemble/modules/m0-ring-channel-gather-primitives.md)（台账见 [assemble/index.md](assemble/index.md)） |
| 胶水（第 5 层） | 表达层合法写法与校验器规则（实施前必读） | [glue-idioms](assemble/glue-idioms.md) |
| 胶水（第 5 层） | 逐字拼接件（拼接点/顺序/易错点） | [glue-fragments](assemble/glue-fragments.md) |

## 替换到同类算子（通用路径）

[optimize](optimize/index.md)：数学语义证同构 → 流水建模重导滞后/环深 → 容量方程重解 → [assemble](assemble/index.md)：M0 原样保留、M4 重推、M1/M2/M3 逐模块替换并按消融纪律验收；异构计算链不复用 SFA 结论，按方法链重建。
