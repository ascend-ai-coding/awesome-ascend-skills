---
name: ai-for-science-ai4s-main
description: AI for Science 昇腾 NPU 总入口 Skill，用于在用户只给出 AI for Science 需求、模型名、TensorFlow/Keras 项目或性能采集诉求时，判断应该进入 Profiling 采集、模型迁移或 TF 框架三条路线，并分流到对应子 skill。
keywords:
  - ai-for-science
  - ai4s
  - profiling
  - migration
  - tensorflow
  - pytorch
  - ascend
---

# AI for Science 总入口 Skill

本 Skill 只负责路线判断和子 skill 分流，不展开具体迁移或调优细节。
当用户只给出一个宽泛的 AI for Science 需求、模型名、TensorFlow/Keras 项目，或只说“帮我迁到昇腾/采集 profiling”时，先从这里判断进入哪个子 skill。

## 三条主路线

| 方向 | 进入条件 | 推荐子 Skill |
|------|------|------|
| Profiling 采集 | 代码已经能训练或推理，只需要采集 trace、分析热点算子、调用栈、内存或瓶颈 | [ai4s-profiling](../ai4s-profiling/SKILL.md) |
| 模型迁移 | 已知模型名，或要把 AI4S 模型从 GPU/CUDA 迁移到昇腾 NPU | [ai4s-basic](../models/ai4s-basic/SKILL.md) 或模型专属 skill |
| TF 框架 | 原项目是 TensorFlow/Keras，需要决定保留 TensorFlow 还是改写到 PyTorch | [ascend-tf-community](../tf-framework/ascend-tf-community/SKILL.md) / [tf-to-pytorch](../tf-framework/tf-to-pytorch/SKILL.md) |

## 模型分流表

| 模型或任务 | 进入的 Skill | 说明 |
|------|------|------|
| Boltz2 | [boltz2](../models/boltz2/SKILL.md) | 蛋白结构预测与端到端推理复现 |
| BoltzGen | [boltzgen](../models/boltzgen/SKILL.md) | 生成式蛋白设计与逆折叠 |
| DeepFRI，保留 TensorFlow | [deepfri-tf-npu](../models/deepfri-tf-npu/SKILL.md) | 保留 TF 运行时和原始实现 |
| DeepFRI，迁移到 PyTorch | [deepfri](../models/deepfri/SKILL.md) | 做 TF 到 PyTorch 改写与权重转换 |
| DiffSBDD | [diffsbdd](../models/diffsbdd/SKILL.md) | 结构化药物设计与扩散推理 |
| GENERator | [generator](../models/generator/SKILL.md) | DNA 序列生成模型迁移 |
| OligoFormer | [oligoformer](../models/oligoformer/SKILL.md) | siRNA 效能预测与 RNA-FM 依赖适配 |
| ProteinBERT | [proteinbert](../models/proteinbert/SKILL.md) | 蛋白语言模型权重转换、embedding 与微调 |
| 未沉淀的新模型 | [ai4s-basic](../models/ai4s-basic/SKILL.md) | 先走通用迁移流程，再沉淀模型专属 skill |

## 决策规则

1. 如果用户已经能跑，只是想采集 profiling 或定位性能问题，直接进入 [ai4s-profiling](../ai4s-profiling/SKILL.md)。
2. 如果用户明确要保留 TensorFlow/Keras 原始实现，进入 [ascend-tf-community](../tf-framework/ascend-tf-community/SKILL.md)。
3. 如果用户明确要迁移到 PyTorch，或后续要接入 `torch_npu` 生态、统一训练推理流程，进入 [tf-to-pytorch](../tf-framework/tf-to-pytorch/SKILL.md)。
4. 如果用户已经给出具体模型名，优先进入对应模型 skill；只有在没有模型专属 skill 时，才进入 [ai4s-basic](../models/ai4s-basic/SKILL.md)。
5. 本 Skill 完成分流后，就在对应子 skill 中继续执行环境、适配、验证和参考资料读取，不在这里重复展开。
