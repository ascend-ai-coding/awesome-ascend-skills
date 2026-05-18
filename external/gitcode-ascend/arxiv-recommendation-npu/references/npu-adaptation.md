# NPU 适配指南

> 完整的 NPU 适配流程请参考 **npu-model-migration skill**。
> 本文是快速入门指引。

## 完整迁移流程（六阶段）

参考 `./npu-model-migration/SKILL.md`：

| 阶段 | 内容 |
|------|------|
| 阶段 1 | 目标分析（环境、代码、难度评估） |
| 阶段 1.5 | 快速尝试（transfer_to_npu） |
| 阶段 2 | 方案设计（识别修改文件、确定策略） |
| 阶段 3 | 代码迁移（设备适配、依赖修复、API 替换） |
| 阶段 4 | NPU 验证（克隆代码、安装依赖、运行测试） |
| 阶段 5 | 调试与迭代（分析报错、定位根因、修复验证） |

## 快速尝试 (阶段 1.5)

大多数 PyTorch 模型可直接跑通，优先尝试：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## 调用 npu-model-migration-skill

arXiv 论文适配后，查看 `daily_report.md` 或 `migration_task.json`，然后：

```bash
cd {model_dir}
# 使用 npu-model-migration-skill 进行迁移
```

## 相关文档

- [npu-model-migration-skill/SKILL.md](../npu-model-migration/SKILL.md)
- [npu-model-migration/references/npu-api-mapping.md](../npu-model-migration/references/npu-api-mapping.md)
- [npu-model-migration/references/common-issues.md](../npu-model-migration/references/common-issues.md)