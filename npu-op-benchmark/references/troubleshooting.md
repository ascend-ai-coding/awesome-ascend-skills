# Troubleshooting

如果出现以下任一情况：

- `torch_npu` 无法导入
- `torch.npu.is_available()` 为 `False`
- 算子执行报 NPU / CANN 环境错误

处理方式统一为：

- 返回 `torch` / `torch_npu` / `CANN` 信息
- 返回实际执行的 `demo.py` 内容
- 返回输入输出 `tensor shape`
- 返回关键报错
- 停止测试，要求使用者提供新的可用环境
