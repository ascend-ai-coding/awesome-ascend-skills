# 训练 PrecisionDebugger 采集

verl 已通过统一 profiler 接口集成 msprobe，**无需修改 engine worker 源码**。参考 verl 文档：
`docs/ascend_tutorial/dev_guide/precision_analysis/precision_debugger_zh.md`

## 训练采集三阶段

训练侧 dump 按 **stage** 划分，PPO 场景下核心三阶段为：

| stage | 角色 | 说明 |
|-------|------|------|
| `actor_compute_log_prob` | actor | actor 前向 log prob 计算 |
| `ref_compute_log_prob` | ref | ref 前向 log prob 计算 |
| `actor_update` | actor | actor 策略更新（前向+反向） |

采集时通过 `global_profiler.global_tool_config.precision_debugger.stages` 指定要采哪些 stage；每个 stage 独立产生一份 dump，路径为 `{save_path}/step_{global_step}/{stage}/step0/rank0/dump.json`。

扩展 stage（需额外开启 critic / reward_model profiler）：

| stage | 角色 |
|-------|------|
| `compute_values` | critic |
| `critic_update` | critic |
| `compute_rm_score` | reward_model |

`rollout_generate` **被忽略**——推理 rollout 走 inference 模式。

## 适用场景

- PPO/GRPO 等 RL 训练各 stage 精度采集
- 对比两次训练 run 的同 stage dump
- 排查 `actor_update` / `actor_compute_log_prob` 等训练阶段数值问题

**不适用**：训推一致性 prefill 对齐比对 → 走 consistency 模式（engine patch）。

## 配置结构（两部分）

### 1. global_profiler：工具与 step 过滤

```yaml
global_profiler:
  tool: precision_debugger
  steps: [1, 2]                    # 唯一 step 过滤器
  save_path: outputs/profile
  global_tool_config:
    precision_debugger:
      _target_: verl.utils.profiler.config.PrecisionDebuggerToolConfig
      config_path: /path/to/config.json
      stages:
        - actor_compute_log_prob
        - ref_compute_log_prob
        - actor_update
      strict: False
```

要点：

- `global_profiler.steps` 是**唯一** step 控制项；`config.json` 里 `step` 留空 `[]`
- **不要在 config.json 里设 `dump_path`**，输出路径由 verl 控制
- 实际路径：`{save_path}/step_{global_step}/{stage}/step0/rank0/dump.json`

### 2. role profiler：按角色开启

```yaml
actor_rollout_ref:
  actor:
    profiler:
      enable: True
  ref:
    profiler:
      enable: True
critic:
  profiler:
    enable: True
```

仅采集 actor/ref 三阶段时，critic / reward_model 的 `profiler.enable` 保持 `False`。

## msprobe config.json 示例

**level 优先使用 `"mix"`**（L0+L1 混合粒度，训练/推理/训推一致场景通用默认）。

### statistics 模式（推荐先试）

```json
{
  "task": "statistics",
  "rank": [],
  "step": [],
  "level": "mix",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

### tensor 模式

```json
{
  "task": "tensor",
  "rank": [],
  "step": [],
  "level": "mix",
  "async_dump": false,
  "tensor": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

模板文件：`assets/config_training_statistics.json.example`、`assets/config_training_tensor.json.example`

## 最小 CLI 示例

复制用户训练脚本为 `run_profiler_<timestamp>.sh`，追加：

```bash
python3 -m verl.trainer.main_ppo \
  global_profiler.tool=precision_debugger \
  global_profiler.steps='[1,2]' \
  global_profiler.save_path=outputs/profile \
  +global_profiler.global_tool_config.precision_debugger.config_path=/path/to/config.json \
  +global_profiler.global_tool_config.precision_debugger.stages='[actor_compute_log_prob,ref_compute_log_prob,actor_update]' \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.ref.profiler.enable=True \
  ... # 原脚本其余参数
```

## 输出目录

```text
outputs/profile/
  step_1/
    actor_compute_log_prob/step0/rank0/dump.json
    actor_update/step0/rank0/dump.json
    ref_compute_log_prob/step0/rank0/dump.json
  step_2/
    actor_compute_log_prob/step0/rank0/dump.json
    ...
```

- 外层 `step_<global_step>`：verl 创建
- 内层 `step0/rank0`：msprobe 创建；每个 stage 独立 session

## 与 FSDP / Megatron 的关系

普通训练采集**不需要**按后端 patch `transformer_impl.py`。verl 在
`verl/utils/profiler/precision_debugger_profile.py` 中按 stage 解析模型：

- engine 路径：`actor.engine.module` / `ref.engine.module` / `critic.engine.module`
- legacy 路径：`actor_module_fsdp` / `actor_module` 等

FSDP 与 Megatron 共用同一套 profiler 配置；若出现 `PrecisionDebugger model not resolved`，说明该 worker/engine 路径未被 resolver 覆盖，需检查 verl 版本或上报 resolver 扩展。

## 开销提示

PrecisionDebugger 是**重量级**精度工具，非轻量 profiler：

- L0 profiled step 约为 baseline 的 3–4x
- L1 profiled step 约为 baseline 的 9–10x

建议先采 1–2 个 representative step，再按需缩小 stage 集合。

## 质量检查清单

- [ ] `bash scripts/ensure-msprobe-env.sh` 通过（或已手动 `pip install mindstudio-probe`）
- [ ] `global_profiler.tool=precision_debugger`
- [ ] `global_profiler.steps` 含目标 step
- [ ] `config_path` 已设置且 msprobe 可 import
- [ ] 目标 role 的 `profiler.enable=True`
- [ ] `config.json` **未**设置 `dump_path`
- [ ] 产物存在于 `{save_path}/step_<N>/<stage>/step0/rank0/dump.json`

## 下游分析

```bash
msprobe compare \
  --target-path /path/to/target_dump/dump.json \
  --golden-path /path/to/golden_dump/dump.json
```

可对比：同 stage 两次 run、同 run 不同 global step、多 rank（若在 config 中启用 rank 过滤）。

## 故障排查

| 症状 | 检查项 |
|------|--------|
| 无 dump 目录 | tool/steps/role enable/msprobe 安装 |
| `model not resolved` | engine 路径是否在 resolver 支持列表 |
| dump.json 异常增大 | verl 已在 stop 后 reset_status；检查 msprobe 版本 |
| 非 profile step 也很慢 | 确认 `global_profiler.steps` 范围够小 |
