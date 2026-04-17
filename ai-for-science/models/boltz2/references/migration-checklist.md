# Boltz2 Migration Checklist

## 迁移前确认

- 权重和数据文件已放到 `~/.boltz/`。
- `torch` 与 `torch_npu` 版本和 CANN 匹配。
- 目标机器至少能完成单卡推理。

## 代码适配重点

- 入口文件启用 `torch_npu` 与 `transfer_to_npu`。
- 处理 `get_device_properties(...).major` 这类 CUDA 专属能力判断。
- 为 Lightning 补充 NPU accelerator 注册。
- 将 CUDA only kernels 切换到 NPU 等价实现或 fallback。

## 验证建议

- 先做最小示例 YAML 推理，再放大到真实输入。
- 记录输出目录、日志关键字和推理耗时。
- 如果推理失败，优先检查权重缺失、设备映射和 Lightning 设备注册。

## 常见问题

- 权重文件名不匹配：先核对 `~/.boltz/` 的实际文件名。
- NPU capability 判断失败：优先回看模型 `setup()` 中的硬编码分支。
- 多卡或 worker 问题：先降到 `--devices 1 --num_workers 0` 验证单卡路径。
