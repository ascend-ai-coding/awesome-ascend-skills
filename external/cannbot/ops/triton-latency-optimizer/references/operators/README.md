# 算子特定优化经验

本目录用于统一存放两类文档：
1. **特定算子类型**的优化经验文档（如 SwiGLU、AdaIN、归一化、纯拷贝型算子），描述某类算子特有的代码模式、判断逻辑和优化技巧。
2. **通用辅助文档**（如通用优化洞察、验证与调试工作流），这些文档不对应某个具体优化点，但在算子探索和调优过程中提供通用指导。

它们与上层 `references/` 中通用的优化点文档（如 tiling、autotune、scalar_to_vector 等）形成互补。

## 目录

### 算子类型特定文档

| 文件 | 对应算子类型 | 核心优化点 |
|------|-------------|-----------|
| [adain.md](adain.md) | Adaptive Instance Normalization 2D Backward | reduce + apply 双 kernel、自适应 BLOCK_S、FP32 累加 |
| [continuous-copy-aggregation.md](continuous-copy-aggregation.md) | Split / Chunk / Slice / Pad / Unbind 等纯拷贝型算子 | 连续拷贝聚合、减少 kernel 启动 |
| [dimension-merge-large-block.md](dimension-merge-large-block.md) | BatchNorm / LayerNorm / GroupNorm / InstanceNorm / RMSNorm / Softmax | 维度合并、大 BLOCK 累加、提高 mask 覆盖率 |
| [permute-layout-transform.md](permute-layout-transform.md) | Permute / Transpose / reshape-as-copy 等布局变换算子 | 模式特化、连续维度合并、view 短路 |

### 通用辅助文档

| 文件 | 用途 |
|------|------|
| [general-insights.md](general-insights.md) | Triton-Ascend 通用优化洞察与跨算子经验 |
| [workflow-and-debugging.md](workflow-and-debugging.md) | 验证、指标对比与调试工作流 |

**使用方式**：在 `SKILL.md` 优化点索引中，若该优化点命中的是某类算子的特有问题，则引用本目录下对应的算子类型文档；通用辅助文档可在需要时按需加载。
