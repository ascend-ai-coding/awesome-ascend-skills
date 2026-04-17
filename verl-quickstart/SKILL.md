---
name: verl-quickstart
description: Generates an executable, end-to-end VERL reinforcement learning quickstart runbook for Ascend/NPU (docker image, dataset preprocessing, model setup, main_ppo training, and examples/run_*.sh flow). Use when users mention verl, quickstart, PPO/GRPO, gsm8k, run_*.sh, Ascend, or NPU training.
keywords:
    - verl
    - rl
    - quickstart
    - ppo
    - grpo
    - gsm8k
    - ascend
    - npu
    - docker
    - 训练
---

# VERL RL Quickstart

最小可跑的 VERL 强化学习流程（Ascend/NPU 友好）。

## 强约束

- 镜像版本 `IMAGE_TAG`：
  - 用户/运维已指定：必须使用其值
  - 未指定默认：
    - 910B: `verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.1`
    - A3: `verl-8.5.0-a3-ubuntu22.04-py3.11-v0.7.1`
  - 也可在 Quay tags 自选：`https://quay.io/repository/ascend/verl?tab=tags&tag=latest`
- 修改或新建 `examples/**/run_*.sh` 后，除非用户明确说“只写不跑”，必须执行一次冒烟（至少 `trainer.total_epochs=1 trainer.test_freq=1`），并返回日志绝对路径与 `tail -f` 命令。
- 使用 **Megatron** 训练后端时，`use_flash_attn=True` 作为默认参数必须开启（至少 actor 侧开启；若 ref 侧未继承则显式补齐）。

## 最小输入变量

- `IMAGE_REPO=quay.io/ascend/verl`
- `IMAGE_TAG`（按上面规则）
- `ASCEND_HW`（可选：`a3|910b`，用于默认选 tag）
- `TRAIN_BACKEND`（可选：`fsdp|megatron`，**默认 `fsdp`**；根据用户诉求选择）
- `WORKSPACE=$(pwd)`、`EXAMPLES_DIR=$WORKSPACE/examples`
- `DATA_DIR=$HOME/data/gsm8k`
- `MODEL_PATH=<绝对路径，且末级目录为具体模型名>`
- `NGPUS=1`、`NNODES=1`、`LOG=verl_run.log`
- `SCRIPT_QUERY`（可选）、`HF_ENDPOINT`（可选）

## 执行步骤

0. Preflight: 检查是否有进程占用 NPU（必要时先释放）

```bash
# 列出占用 NPU 的进程（PID/进程名/显存）
/usr/local/sbin/npu-smi info

# 如果看到有进程占卡，先确认是否为你自己的训练/推理任务；
# 建议先“优雅退出”，不行再 kill：
kill <PID>
sleep 3
/usr/local/sbin/npu-smi info

# 仍未退出再强杀（谨慎使用）
kill -9 <PID>
```

1. Step 0: 环境/镜像  
   参考 `references/docker-ascend.md`
2. Step 1: 数据处理（生成 parquet）  
   参考 `references/data-preprocess.md`
3. Step 2: 模型准备（ModelScope + 路径规范）  
   参考 `references/model-download.md`
4. Step 3A: `main_ppo` 训练（按后端选择）  
   - **默认 FSDP**：适合多数场景，参考 `references/main-ppo-quickstart.md` 的 FSDP 示例  
   - **Megatron**：当用户明确要求或大模型/并行策略需要时选择，参考同文件的 Megatron 示例
     - 默认追加：`+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True`
     - 若 ref 侧未继承该配置，再追加：`++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True`（已存在该 key 时不要用单 `+`）。
5. Step 3B: examples 脚本训练与冒烟  
   参考 `references/examples-scripts.md`
6. Step 4: checkpoint 合并  
   参考 `references/checkpoint-merge.md`
7. FAQ / 排障  
   参考 `references/faq.md`
