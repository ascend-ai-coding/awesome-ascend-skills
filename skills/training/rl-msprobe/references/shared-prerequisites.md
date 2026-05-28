# 训推一致性 Dump 公共前置（训练脚本 / 环境）

本文档汇总 FSDP 与 Megatron **共用**的前置约束。训练侧 engine worker 采集 patch 见各后端 reference。

## 环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| `DUMP_ON` | `1` | 开启训练侧 engine debugger |
| `PROMPTS_ONLY` | `1` | 训练仅保留 prompt（prefill 对齐） |
| `TORCHDYNAMO_DISABLE` | `1` | FSDP 推荐；避免 dynamo 干扰 |
| `DUMP_PHASE` | `log_prob` / `update_actor` / `all` | 控制 engine 内采集阶段 |

## 训推并行切分对齐（consistency 必检）

## msprobe 环境

采集前在 verl 使用的 Python 环境中确认 msprobe 可用：

```bash
bash scripts/ensure-msprobe-env.sh
# PyPI: pip install mindstudio-probe
# 校验: python3 -c "from msprobe.pytorch import PrecisionDebugger; print('OK')"
```

安装失败时由用户自行安装后再继续；勿在未 import 成功时启动 dump。

---

训推一致比对要求**两侧 dump 在每张卡上的 tensor 切分一致**，否则 `rank_M` 无法一一对应。推理侧 msprobe 的 rank 数 = **vLLM 进程组大小**（通常等于 rollout 的 TP 组大小），与训练侧 world_size **无必然相等**；须通过下面规则主动对齐。

### FSDP 后端

FSDP 训练按 **FSDP rank（≈ 训练占用卡数）** 分片，无 TP；推理按 **vLLM TP 进程组** dump。

| 约束 | 说明 |
|------|------|
| **推理 TP = rollout 占用卡数** | `actor_rollout_ref.rollout.tensor_model_parallel_size` 须等于该 rollout replica 实际使用的 NPU/GPU 数 |
| **consistency 推荐** | 单 replica、不开推理 DP：`gen_tp = trainer.n_gpus_per_node`（如 4 卡 → `gen_tp=4`） |
| **反例** | 4 卡节点设 `gen_tp=2` → 推理 dump 仅 `rank0/rank1`，训练 FSDP 仍有 4 rank，**无法按卡比对** |

```bash
# 4 卡 FSDP + vLLM async，训推一致采集推荐
trainer.n_gpus_per_node=4
actor_rollout_ref.rollout.tensor_model_parallel_size=4   # gen_tp 与 rollout 卡数一致
actor_rollout_ref.rollout.data_parallel_size=1           # consistency 阶段勿开推理 DP
```

校验：推理产物 `{generate_dump_path}/step0/` 下 `rank_*` 数量应等于 `tensor_model_parallel_size`。

### Megatron 后端

Megatron 训练与 vLLM 推理均按 TP/PP/DP/EP 切分。**训推一致采集要求 actor/ref 与 rollout 在 TP/PP/EP 上逐项相同**，否则两侧 dump 的 layer 分片与 rank 无法一一对应。

| 维度 | 训练侧（actor + ref 须相同） | 推理侧（rollout） | 对齐规则 |
|------|------------------------------|-------------------|----------|
| TP | `actor.megatron.tensor_model_parallel_size` | `rollout.tensor_model_parallel_size` | **必须相等** |
| PP | `actor.megatron.pipeline_model_parallel_size` | `rollout.pipeline_model_parallel_size` | **必须相等** |
| EP | `actor.megatron.expert_model_parallel_size` | `rollout.expert_parallel_size` | **必须相等** |
| DP | `world_size / TP / PP / CP` | `rollout.data_parallel_size` | 由 replica 占用卡数推导 |

> **常见陷阱**：生产训练脚本（如 `run_qwen2.5-3b-megatron.sh`）往往设 `train_pp=2`，但 `ROLLOUT_CONFIG` **未**设置 `rollout.pipeline_model_parallel_size`（Hydra 默认 **1**）。此时训练 PP=2、推理 PP=1，**权重同步可以跑通，但 dump 切分不一致，不能用于训推一致比对**。采集脚本必须显式覆盖：
>
> ```bash
> actor_rollout_ref.rollout.pipeline_model_parallel_size=${train_pp}
> ```

**verl vLLM 限制（当前版本）**：`rollout.pipeline_model_parallel_size > 1` 尚未实现。生产 `train_pp=2` 时，**采集须将 train/ref/rollout PP 统一为 1**（默认 `CONSIST_PP=1`），不能「训练 PP=2 + 推理 PP=1」。待 verl 支持 vLLM PP 后，再令 `CONSIST_PP=train_pp`。

```bash
# 采集：train/ref/rollout PP 均覆写为 1（与 gen_pp 一致）
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
actor_rollout_ref.rollout.pipeline_model_parallel_size=1

NPUS_PER_NODE=1 bash run_consistency_dump_megatron.sh
# → PP=1 全模型，actor rank0 ↔ generate rank0
```

**禁止**在 NPU 采集阶段无必要地改大 `gen_dp`（如 `gen_dp=4`）：多 DP vLLM replica 下 train→rollout IPC 权重同步可能触发 `rebuild_ipc` 失败。

另须 `trainer.balance_batch=False`（见下方 Hydra），避免 batch 重排导致卡间样本不对应。

## Batch 约束（必须满足）

1. **mini_batch_num = 1**

   `mini_batch_num = train_batch_size / train_ppo_mini_batch_size`

2. **gac（梯度累计步数）= 1**

   `gac = train_ppo_mini_batch_size * n_resp_per_prompt / train_ppo_micro_batch_size_per_gpu / DP`

   Megatron 的 `DP = world_size / TP / PP / CP`；FSDP 无 TP/PP 时 `DP = world_size / SP`

3. **FSDP 额外：`real_train_batch_size` 整除性**

   `real_train_batch_size = train_batch_size * n_resp_per_prompt` 必须能被 `minimal_bsz` 整除。
   非 Megatron 场景通常 `minimal_bsz = n_gpus`（如 4 卡需 `train_batch_size * n >= 4`）。

4. **Hybrid AgentLoop 额外：agent worker 整除**

   `train_batch_size`（或 chunk 前的 batch 长度）必须能被 `actor_rollout_ref.rollout.agent.num_workers` 整除。
   否则报错：`only support equal chunk. Got size of DataProto X and chunk Y`.

5. 关闭 `use_dynamic_bsz` 后须显式设置 log_prob micro batch：

```bash
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro}
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro}
```

6. **优先关闭 SP（Ulysses sequence parallel）**

   训推一致采集 **优先** 将 actor/ref 的 `ulysses_sequence_parallel_size` 设为 `1`，即使原脚本 `sp_size>1` 也须在采集脚本中 Hydra 覆盖。SP 开启时 DP/batch 对齐与 dump 关联更复杂。

   ```bash
   actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
   actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
   ```

   采集脚本可读取原脚本 `sp_size`，若 `>1` 则日志提示 `sp_size X -> 1`。SP 关闭后 `DP = world_size / SP = world_size`。

7. **推荐 batch 参数**（`train/mini/micro` 均为 1，`n = DP`，不考虑 gac）：

```bash
data.train_batch_size=1
actor_rollout_ref.actor.ppo_mini_batch_size=1
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.n=${DP}              # DP = world_size / SP，SP=1 时等于 n_gpus
actor_rollout_ref.rollout.agent.num_workers=1  # train_batch_size=1 时按 prompt 切分
```

### 4 卡 FSDP + vLLM async 参考值（SP 关闭，已验证）

```bash
train_batch_size=1  ppo_mini_batch_size=1  micro_batch_per_gpu=1
n=4               # DP = world_size，SP=1
agent.num_workers=1
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
actor_rollout_ref.rollout.tensor_model_parallel_size=4   # consistency：TP 与 rollout 卡数一致
actor_rollout_ref.rollout.data_parallel_size=1
# real_train_batch_size = 1 * 4 = 4；推理/训练 dump 均为 step0/rank0..rank3
```

## Hydra 参数

```bash
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=False \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
trainer.val_before_train=False \
trainer.balance_batch=False \
actor_rollout_ref.rollout.skip.enable=False \
algorithm.rollout_correction.rollout_is=null \
algorithm.rollout_correction.rollout_rs=null \
algorithm.rollout_correction.rollout_rs_threshold=null \
algorithm.rollout_correction.rollout_is_batch_normalize=False \
```

## msprobe 训练侧配置

- 路径：`MSPROBE_ACTOR_CONFIG`（默认 `/home/config_actor.json`）
- 模板见 `assets/config_actor.json.example`
- `dump_path` 目录需可写，训练完成后在此目录下生成 `step_N/rank_M/dump.json`

## 异步架构额外说明（推理侧）

完整训推一致性还需推理侧 dump，按 rollout 后端选择：

| 后端 | 配置方式 | 文档 |
|------|----------|------|
| **vLLM** | `++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dump_config_path=...` + `enforce_eager=True` | [inference-vllm-dump.md](inference-vllm-dump.md) |
| **SGLang** | `++actor_rollout_ref.rollout.engine_kwargs.sglang.msprobe_dump_config=...` 或 patch ModelRunner | [inference-sglang-dump.md](inference-sglang-dump.md) |

**必须** `actor_rollout_ref.rollout.enforce_eager=True`（不开图模式）。

本 skill 训练侧 engine patch 见 fsdp/megatron reference；推理侧见上述文档。
