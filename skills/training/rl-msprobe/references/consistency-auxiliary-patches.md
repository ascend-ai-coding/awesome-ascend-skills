# 训推一致性辅助 Patch（metrics / rollout_corr / request_id）

Legacy megatron 文档中除 engine/worker 主 patch 外，还有三处**辅助改动**。Engine worker 架构下仍需保留。

## 1. `verl/utils/debug/metrics.py`

PROMPTS_ONLY 模式下 `responses` 为空，跳过 rollout vs actor logprob diff 计算，避免报错或无意义指标。

在 `calculate_debug_metrics()` 开头增加：

```python
import os

if int(os.getenv("PROMPTS_ONLY", "0")):
    return { "training/rollout_probs_diff_valid": 0, ... }

if "responses" not in data.batch or data.batch["responses"].size(1) == 0:
    return { "training/rollout_probs_diff_valid": 0, ... }
```

完整 early-return 逻辑见 legacy megatron 文档 `verl/utils/debug/metrics.py` diff。

## 2. `verl/trainer/ppo/rollout_corr_helper.py`

PROMPTS_ONLY 模式下跳过 rollout correction（依赖 response_mask）：

```python
import os

def compute_rollout_correction_and_add_to_batch(...):
    if int(os.getenv("PROMPTS_ONLY", "0")):
        return batch, {}
    ...
```

## 3. request_id 注入（二选一）

### 方案 A：较新 verl — `verl/workers/rollout/llm_server.py`

```python
vllm_request_id = uuid4().hex
output = await server.generate.remote(request_id=vllm_request_id, ...)
output.extra_fields["request_id"] = vllm_request_id
```

### 方案 B：部分部署 — `verl/experimental/agent_loop/agent_loop.py`

在 `LLMServerClient.generate()` 中同样注入：

```python
vllm_request_id = uuid4().hex
output = await server.generate.remote(request_id=vllm_request_id, ...)
output.extra_fields["request_id"] = vllm_request_id
```

校验：`check-consistency-patch.py` 要求 **至少一处** 存在 `extra_fields["request_id"]`。

## 验证

```bash
python3 scripts/check-consistency-patch.py --verl-root "$VERL_ROOT" --backend fsdp --skip-vllm
```
