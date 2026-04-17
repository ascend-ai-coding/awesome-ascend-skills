# BoltzGen Runtime Notes

## 资源准备

- 预训练权重尽量提前放入 `~/.cache/`，减少运行时下载依赖。
- `mols.zip` 或等价数据包要保证结构正确，否则运行时会在数据准备阶段失败。

## 代码适配重点

- 两个入口文件都要启用 `transfer_to_npu`。
- 对 `get_device_capability` 返回空值的情况做兼容处理。
- 对 CUDA kernel 路径补 NPU 等价实现或纯 PyTorch fallback。

## 运行建议

- 先跑最小 `budget` 和 `num_designs` 验证路径可通。
- 如果出现下载失败，先切国内镜像或改为本地缓存。
- 如果出现 kernel 相关错误，先确认是否真的走到 NPU fallback。

## 输出检查

- 确认输出目录中生成设计结果文件。
- 核对日志里是否出现 NPU 设备、权重加载完成和设计完成提示。
