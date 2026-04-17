# AI4S Migration Validation Checklist

本参考文档补充通用迁移 skill 的“完成定义”。

## 环境完成标准

- `npu-smi info` 可正常显示设备。
- `source set_env.sh` 后 Python 能成功 `import torch_npu`。
- 基础 tensor 在 `npu:0` 上可运行。
- 关键依赖版本已固定，尤其是 `numpy`、`torch`、`torch_npu`。

## 代码完成标准

- 所有入口脚本都完成了 NPU 初始化或 `transfer_to_npu` 注入。
- 明确记录了无法自动迁移的 CUDA API。
- 第三方 CUDA 专属库已替换、降级或禁用。
- 分布式场景已经确认 `hccl`、DDP 和设备可见性配置。

## 结果完成标准

- 至少有一条端到端命令可以成功执行。
- 有输出文件或日志标记可作为成功依据。
- 如果存在基线结果，已经做过 CPU、GPU 或原框架对比。
- 明确说明当前阶段是“能跑通”还是“已经对齐精度”。

## 建议沉淀物

- 环境版本表。
- 关键 patch 列表。
- 最小复现命令。
- 常见错误及处理方法。
