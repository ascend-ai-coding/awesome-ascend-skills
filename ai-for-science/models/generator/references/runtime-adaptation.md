# GENERator Runtime Adaptation

## 模型加载

- `from_pretrained` 使用 `torch_dtype=`，不要使用旧的 `dtype=` 参数。
- 如需镜像源，提前设置 `HF_ENDPOINT`。
- 如果模型较大，先确认本地缓存和磁盘空间。

## 多进程适配

- 子进程函数内部要重新导入 `torch_npu`。
- 子进程内显式设置 `torch.npu.set_device()`。
- 多卡时优先先跑单卡，验证主进程路径无误后再扩展。

## 结果检查

- 结果 parquet 文件非空。
- 日志中出现任务完成和结果保存提示。
- 如果结果条目数明显异常，先看 shard 切分和子进程设备绑定。
