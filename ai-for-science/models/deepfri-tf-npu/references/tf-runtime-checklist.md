# DeepFRI TF Native Runtime Checklist

## 框架侧检查

- TensorFlow 版本固定为和 tfplugin 匹配的版本。
- aarch64 场景确认是否完成了源码编译和 ABI 对齐。
- `npu_device` 可以被成功导入。

## 运行前检查

- `set_env.sh` 已 source。
- 运行脚本中已经加入 `npu_device.open().as_default()`。
- 模型文件使用的是预期版本，尤其区分 CPU 版和 GPU 版权重。

## 精度检查

- 先跑 CPU，再跑 NPU，用统一输入对比输出。
- 对于混合精度或 HF32 导致的差异，重点关注排名和相对趋势，而不是逐点完全相等。

## 常见失败原因

- TF wheel 与 tfplugin ABI 不匹配。
- `predict.py` 没有加 `--npu` 路径或初始化逻辑。
- 用户误用了不兼容的模型包版本。
