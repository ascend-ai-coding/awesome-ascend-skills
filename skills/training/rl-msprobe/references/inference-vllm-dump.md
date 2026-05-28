# vLLM 推理侧 msprobe 采集

verl 使用 vLLM-Ascend 作为 rollout 后端时，推理 dump **无需改 verl 源码**。vLLM-Ascend 已内置 `PrecisionDebugger` 逻辑，通过 `additional_config.dump_config_path` 传入 msprobe 配置即可。

## 强约束

- **必须 eager 模式**：`actor_rollout_ref.rollout.enforce_eager=True`（不开图 / 不 capture cuda graph）
- `config.json` **必须设置 `dump_path`**（与训练侧 profiler 不同）
- 训推一致性场景还需 `dispatch_logger` 等额外 patch，见 consistency 模式

## 步骤 1：准备 generate config.json

```json
{
  "task": "statistics",
  "dump_path": "/path/to/dump/generate",
  "rank": [],
  "step": [0],
  "level": "mix",
  "async_dump": false,
  "extra_info": true,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

模板：`assets/config_generate.json.example`

要点：

- `dump_path`：推理 dump 输出根目录（按 rank/pid/step 子目录展开）
- `step: [0]`：通常只采第一步 forward；可按需调整
- `level: "mix"`：**默认推荐**（L0 模块级 + L1 API 级混合采集）
- **rank 数**：产物中 `rank_M` 数量 = vLLM `tensor_model_parallel_size`（非训练 world_size）。训推一致时 FSDP 须 **TP = rollout 卡数**；Megatron 须 **沿用原脚本 train/gen 并行**（可用最小卡数采集）。见 [shared-prerequisites.md](shared-prerequisites.md#训推并行切分对齐consistency-必检)。

## 步骤 2：训练脚本追加参数

复制原脚本为 `run_infer_dump_<timestamp>.sh`，追加：

```bash
export TORCHDYNAMO_DISABLE=1   # 推荐，避免 dynamo 干扰

python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.enforce_eager=True \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dump_config_path="/path/to/generate_config.json" \
  ... # 原脚本其余参数
```

Hydra 字典写法（与 msprobe 异步文档一致）：

```bash
'+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config={dump_config_path:"/path/to/generate_config.json"}'
```

两种写法等价，优先使用用户脚本中已有的风格。

训推一致性额外参数：

```bash
export PROMPTS_ONLY=1
trainer.val_before_train=False \
trainer.balance_batch=False \
```

## 步骤 3：验证产物

vLLM-Ascend 在 `enforce_eager=True` 且 `dump_config_path` 有效时，每次 `execute_model` 产生 dump：

```text
{dump_path}/{pid}/step_N/rank_M/dump.json
```

训推一致性场景还会有 `dispatch_log.jsonl`（需 vLLM-Ascend `dispatch_logger` patch）。

## 校验清单

- [ ] `bash scripts/ensure-msprobe-env.sh` 通过
- [ ] `actor_rollout_ref.rollout.name=vllm`
- [ ] `actor_rollout_ref.rollout.enforce_eager=True`
- [ ] `dump_config_path` 指向存在的 json
- [ ] json 中 `dump_path` 目录可写
- [ ] msprobe 可 import
- [ ] **未**开启 cudagraph（`enforce_eager=False` 会导致 dump 失败）

## 故障排查

| 症状 | 原因 |
|------|------|
| `Dumping/debugging only works in eager mode` | 未设 `enforce_eager=True` |
| 无 dump 目录 | `dump_config_path` 路径错误或 vLLM-Ascend 版本不支持 |
| dump 含 decode step | 训推一致仅支持 prefill；检查 PROMPTS_ONLY 与 step 过滤 |
| 推理 dump 仅 2 个 rank、训练 4 个 rank | FSDP：`gen_tp` 小于 rollout 卡数；consistency 须 `tensor_model_parallel_size=卡数` |
| Megatron 训推 rank 无法对应 | 训练 TP/PP/EP 与 rollout 切分不一致；逐项对齐并 `balance_batch=False` |

## 参考

- msprobe 异步训推文档：`msprobe/docs/zh/dump/verl_async_consistency_preprocess_dump.md`
- vLLM-Ascend `model_runner_v1.py` 内置 debugger 逻辑
