# Atlas NPU Model Reference
# 昇腾 NPU 型号参考

## 常见昇腾 NPU 卡型号

| 型号 | 类型 | 说明 |
|------|------|------|
| Atlas 800-3000 | 训练型 AI 服务器 | 高性能训练 |
| Atlas 800-2000 | 推理型 AI 服务器 | 高性能推理 |
| Atlas 300I Pro | 推理卡 | 推理场景 |
| Atlas 300V Pro | 训练卡 | 训练场景 |
| Atlas 500 Pro | 边缘计算 | 边缘部署 |

## NPU 命名规范

量化后的模型命名格式：`{original_name}-{quant_type}-{features}`

示例：
- `qwen2-7b-w8a8` (Qwen2-7B, W8A8量化)
- `qwen3-32b-w8a8s-QuaRot` (Qwen3-32B, W8A8稀疏量化, QuaRot特征)
- `deepseek-v3-w4a8c8` (DeepSeek-V3, W4A8C8量化)
