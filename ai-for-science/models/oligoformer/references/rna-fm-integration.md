# OligoFormer RNA-FM Integration Notes

## 依赖关系

- OligoFormer 运行依赖 RNA-FM 提供表征能力。
- Shell 脚本要显式 source CANN 环境，并使用目标 conda 环境里的 Python。

## 设备配置

- 入口脚本启用 `transfer_to_npu`。
- 训练和测试脚本同时设置 `CUDA_VISIBLE_DEVICES` 与 `ASCEND_RT_VISIBLE_DEVICES`。
- 先验证单卡推理，再考虑训练路径。

## 输出检查

- `result/` 目录内有排序结果文件。
- 预测结果列非空。
- 如果脚本调用 RNA-FM 失败，优先检查 shell 脚本路径和环境变量。
