# 数据预处理（参考）

```bash
export WORKSPACE="$(pwd)"
cd "$WORKSPACE"

export HF_ENDPOINT="${HF_ENDPOINT:-}"
# 外网拉取失败时可启用：
# export HF_ENDPOINT="https://hf-mirror.com"

export DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
ls -la "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet"
```

产物：`$DATA_DIR/train.parquet`、`$DATA_DIR/test.parquet`。
