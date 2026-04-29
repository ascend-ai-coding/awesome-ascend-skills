---
name: external-mindstudio-msmodelslim-model-adapt
description: 为 msModelSlim 创建基础 Transformers 模型适配器（Model Adapter）。 包含创建适配器、实现必需接口及四步验证流程。
  适用：Decoder-only LLM、理解类 VLM（仅 LLM/text 部分）。 不适用：多模态生成模型（图像/视频/语音生成）、Encoder-only、非
  Transformers 架构。
original-name: model-adapt
synced-from: https://github.com/kali20gakki/mindstudio-skills
synced-date: '2026-04-29'
synced-commit: f6695da3cfa864a100db5a94f594b638aeff6389
license: UNKNOWN
---

# msModelSlim 基础模型适配 Skill

本 Skill 指导如何为新模型创建基础适配器，使其跑通 W8A8/W4A16 量化流程。

## 适用范围

- **支持**：Decoder-only LLM、理解类 VLM（只处理文本/LLM 主干）
- **不支持**：多模态生成（如 Stable Diffusion/Flux/Wan）、Encoder-only、非 Transformers

## 核心工作流

### 1. 准备工作
- **下载模型**：建议使用 `modelscope download` 下载非权重文件。
  - 示例：`modelscope download --model <org>/<model> --local_dir ./models/<name> --exclude '*.safetensors'`
- **分析模型**：阅读 `config.json` 与 `modeling_*.py`，确认结构与实现。
  - 详见：[模型结构分析指南](references/model_analysis.md)

### 2. 创建适配器
- **使用模板**：
  - LLM: `assets/model_adapter_template.py`
  - VLM: `assets/vlm_model_adapter_template.py`
- **实现接口**：实现 `handle_dataset`, `init_model`, `generate_model_visit`, `generate_model_forward`, `enable_kv_cache`。
- **关键原则**：
  - `visit` 与 `forward` 必须严格一致。
  - MoE 模型建议 unpack 为纯线性层。
  - 详见：[适配器实现指南](references/implementation_guide.md)

### 3. 注册与安装
- 在 `config/config.ini` 注册模型与入口，并执行 `bash install.sh`。
- 详见：[适配器注册指南](references/registration_guide.md)

### 4. 验证适配器 (必需)
- 必须执行四步验证：生成测试模型 -> 全回退量化 -> 校验全回退模型与浮点权重严格一致且可完整加载/保存 -> 验证实际量化流程正常（含描述文件规则校验）。
- 详见：[适配器验证指南](references/verification_guide.md)

## 常用脚本

脚本位于 `scripts/` 目录下：

- `scripts/step1_generate_test_model.py`
- `scripts/step2_run_quantization.py`
- `scripts/step3_verify_weights.py`
- `scripts/step4_verify_quant_description.py`

## 参考资料

- [模型结构分析指南](references/model_analysis.md)
- [适配器实现指南](references/implementation_guide.md)
- [适配器注册指南](references/registration_guide.md)
- [适配器验证指南](references/verification_guide.md)
- [接口检查清单](references/interface_checklist.md)
- [核心工作流](references/core_workflow.md)
