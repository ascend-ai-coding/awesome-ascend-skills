# References 说明

本目录为 ascendc skill 的细化文档，与 `../SKILL.md` 配合使用。**参考示例以 CANN 官方文档为准**，各文档内设有「官方文档参考」小节；Agent 应先读 SKILL.md 中的「References 索引」与「何时查阅」，再按任务打开对应文件。

| 文件 | 主题 |
|------|------|
| 01-type-format.md | op_host 的 DataType/Format/UnknownShapeFormat 约定与 JSON 映射 |
| 02-kernel-guide.md | Kernel 开发（GlobalTensor/TQue、CopyIn/Compute/CopyOut）、Matmul/Cube 模板、GMM 转置 |
| 03-op-host-examples.md | FFN/GMM/MoE 的 Input/Output/Attr 示例代码 |
| 04-op-kernel-skeletons.md | FFN/GMM/MoE 的 op_kernel 命名空间与主类骨架 |
| 05-json-types-flow.md | JSON + graph/types.h 驱动 op_host/infershape/tiling 对齐流程 |
| 06-tiling.md | Tiling 两种实现方式（标准 C++ vs 宏定义）及写回 context 差异 |
| 07-aclnn-template.md | aclnn 示例通用模板与生成步骤 |
| 08-genop.md | genop 命令、生成结构、定制与常见问题 |

文档间通过相对路径互相引用（如 `[01-type-format](01-type-format.md)`），阅读时按需跳转即可。
