# checkpoint 合并（参考）

```bash
# 例如：checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_step_<N>/actor
export CKPT_ACTOR_DIR="<path-to-checkpoints-actor-dir>"
export HF_OUT_DIR="${HF_OUT_DIR:-$CKPT_ACTOR_DIR/huggingface}"

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$CKPT_ACTOR_DIR" \
  --target_dir "$HF_OUT_DIR"
```
