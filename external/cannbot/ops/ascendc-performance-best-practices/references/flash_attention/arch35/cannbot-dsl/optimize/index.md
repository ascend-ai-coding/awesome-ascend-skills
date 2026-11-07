# FA 优化层：DSL FA 算子怎么优化（方法 + SFA 实例）

本目录是最佳实践库第 1-3 层的 FA 落地：讲**怎么优化一个 DSL 写的 FA 算子**——三步方法链按序推导，SparseFlashAttention（SFA）作贯穿实例。用积木与胶水把手段落成代码见 [组装层](../assemble/index.md)；跨族方法论见 [CANNBot-DSL 公共实现知识](../../../../common/arch35/cannbotdsl/index.md)。

| 步 | 回答的问题 | 文档 |
|---|---|---|
| 1. **数学语义建模** | 算子算什么，哪些数值语义必须保留 | [1-math-semantics](1-math-semantics.md) |
| 2. **流水建模** | 数据在 Cube/Vector 两侧如何流动、节拍如何编排（AIC+AIV 延迟线手段） | [2-pipeline-modeling](2-pipeline-modeling.md) |
| 3. **容量方程** | tile 形状、缓冲深度、滞后如何从台账解出（四层法手段） | [3-capacity-equations](3-capacity-equations.md) |

通用路径：[数学语义](1-math-semantics.md)钉语义与不变量 → [流水建模](2-pipeline-modeling.md)导阶段/滞后/环深 → [容量方程](3-capacity-equations.md)解 tile 与贴顶证明 → [组装层](../assemble/index.md)落成代码并按消融纪律验收。
