# 模型结构分析指南

## 1. 确认模型结构来源

- 读取 `config.json` 的 `model_type`、`architectures`、`auto_map`
- 确定结构来自模型仓内 `modeling_*.py` 还是 transformers 官方实现

## 2. 定位并阅读模型实现（必须）

- **自定义实现**：如果 `auto_map` 指向自定义实现（如 `modeling_xxx.XXXForCausalLM`），优先阅读模型目录中的 `modeling_*.py`
- **官方实现**：如果使用 transformers 官方实现，通常在：
  - 源码路径：`transformers/src/transformers/models/<model_type>/modeling_<model_type>.py`
  - 导入路径：`transformers.models.<model_type>.modeling_<model_type>`
- **重点阅读**：
  - DecoderLayer 定义
  - attention/MLP 命名
  - `forward` 入参与返回值
- **MoE 模型额外检查**：
  - `experts` 权重是否为 3D packed 结构（如 `experts.gate_up_proj` / `experts.down_proj`）
