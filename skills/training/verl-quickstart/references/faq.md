# 常见问题（参考）

- `IMAGE_TAG`：用户/运维指定优先；未指定用默认（910B/A3）或去 Quay tags 页自选。
- HF 拉取失败：先设 `export HF_ENDPOINT="https://hf-mirror.com"` 再试。
- OOM：先降 `ppo_micro_batch_size_per_gpu` 到 1，再降 `gpu_memory_utilization`，再缩短 max length。
- 数据路径错误：确认 `$DATA_DIR/train.parquet` 与 `$DATA_DIR/test.parquet` 存在。
- 模型目录错误：`MODEL_PATH` 必须指向带具体模型名的目录，而不是系列父目录。
- 日志查看：以 `tee` 输出的 `$LOG` 为准，`tail -f` 持续观察。
