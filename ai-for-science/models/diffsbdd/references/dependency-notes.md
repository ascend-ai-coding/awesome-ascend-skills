# DiffSBDD Dependency Notes

## 关键依赖

- `torch_scatter` 必须源码编译。
- `numpy` 必须固定在和 CANN 兼容的版本。
- `rdkit`、`openbabel` 安装后要重新检查 `numpy` 是否被覆盖。

## 适配建议

- 所有入口统一注入 `transfer_to_npu`。
- 保留 `analysis/` 相关目录的 `__init__.py`，避免运行时导入失败。
- 如果只想先验证迁移正确性，优先使用最小推理命令而不是训练路径。

## 验证输出

- 生成的 SDF 文件非空。
- 日志中看不到 CUDA only 的报错。
- 如果输出分布与基线相比有偏差，优先核对 fp64 自动降级行为。
