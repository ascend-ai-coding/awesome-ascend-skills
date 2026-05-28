# Megatron Engine Worker 采集 Patch

**目标文件**：`{VERL_ROOT}/verl/workers/engine/megatron/transformer_impl.py`  
**目标类**：`MegatronEngine`（基类方法）+ `MegatronEngineWithLMHead`（hook 点）

> **训推一致前置**：Megatron **沿用原训练脚本的 train/gen 并行**（与 rollout 配置一致），并设 `trainer.balance_batch=False`。采集可用最小卡数（不必 8 卡）。见 [shared-prerequisites.md](shared-prerequisites.md#训推并行切分对齐consistency-必检)。

Megatron 使用 `get_forward_backward_func()` PP schedule，**不能**像 FSDP 那样在 `forward_backward_batch` 里简单 for-loop 包裹。应在 `forward_step`（前向开始）与 `postprocess_micro_batch_func`（loss 计算后、backward 由 schedule 触发）挂载 hook。

## 1. `MegatronEngine.__init__` 末尾追加

```python
        self._debugger = None
        self._update_actor_logger_fp = None
        self._dump_should_record = False
        self._dump_micro_batch = None
```

## 2. 在 `MegatronEngine` 类中新增方法

```python
    def _ensure_debugger(self):
        if self._debugger is not None:
            return
        if self.engine_config.forward_only:
            return
        dump_flag = int(os.environ.get("DUMP_ON", 0))
        if not dump_flag:
            return
        from pathlib import Path

        from msprobe.pytorch import PrecisionDebugger, seed_all

        seed_all(mode=True)
        config_path = os.environ.get(
            "MSPROBE_ACTOR_CONFIG", "/home/config_actor.json")
        self._debugger = PrecisionDebugger(config_path=config_path)
        try:
            dump_path = self._debugger.config.dump_path
            log_dir = Path(dump_path) / str(os.getpid())
            log_dir.mkdir(parents=True, exist_ok=True)
            self._update_actor_logger_fp = open(
                log_dir / "update_actor_log.jsonl", "a")
        except Exception as e:
            logger.warning(f"Failed to initialize update_actor_logger: {e}")

    def _should_dump(self, forward_only: bool) -> bool:
        if self._debugger is None:
            return False
        dump_phase = os.environ.get("DUMP_PHASE", "log_prob")
        phase = "log_prob" if forward_only else "update_actor"
        return dump_phase == "all" or dump_phase == phase

    def _log_update_actor_step(self, micro_batch: TensorDict) -> None:
        if self._update_actor_logger_fp is None:
            return
        try:
            req_data = tu.get(micro_batch, key="request_id", default=None)
            if not req_data:
                request_ids = []
            elif isinstance(req_data, list):
                request_ids = [str(r) for r in req_data]
            else:
                request_ids = [str(req_data)]
        except Exception:
            request_ids = []

        import json
        import time

        record = {
            "source": "update_actor",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "pid": os.getpid(),
            "rank": torch.distributed.get_rank(),
            "step": self._debugger.service.current_iter,
            "request_ids": request_ids,
            "num_requests": len(request_ids),
        }
        self._update_actor_logger_fp.write(
            json.dumps(record, ensure_ascii=False) + "\n")
        self._update_actor_logger_fp.flush()

    def _finish_dump_step(self):
        if not self._dump_should_record or self._debugger is None:
            return
        self._debugger.stop()
        self._debugger.step()
        if self._dump_micro_batch is not None:
            self._log_update_actor_step(self._dump_micro_batch)
        self._dump_should_record = False
        self._dump_micro_batch = None
```

## 3. 修改 `MegatronEngine.forward_backward_batch` 开头

在 `prepare_micro_batches` 之前调用一次 lazy init：

```python
        self._ensure_debugger()
```

## 4. 修改 `MegatronEngineWithLMHead.forward_step`

在 `batch: TensorDict = next(batch_iter)` 之后、模型前向之前：

```python
        forward_only_flag = tu.get_non_tensor_data(
            batch, key="forward_only", default=False)
        # 若 forward_only 由外层 schedule 传入，也可从 closure 捕获；此处以 batch meta 为准
        if self._should_dump(forward_only=False):  # update_actor 路径
            from megatron.core.utils import unwrap_model
            self._debugger.start(model=unwrap_model(model))
            self._dump_should_record = True
            self._dump_micro_batch = batch
```

> 说明：`forward_only` 的实际值需与调用上下文一致。若 batch 中无该字段，可在 `forward_backward_batch` 里 `tu.assign_non_tensor(data, forward_only=forward_only)` 并在 `forward_step` 读取。

## 5. 修改 `MegatronEngineWithLMHead.postprocess_micro_batch_func`

在 `scaled_loss = loss * data["num_micro_batch"]` 之后：

**纯前向（log_prob，`forward_only=True`）**：直接 finish dump

```python
        if self._dump_should_record and forward_only:
            self._finish_dump_step()
```

**训练（update_actor，`forward_only=False`）**：在 backward 完成后 finish

```python
        if self._dump_should_record and not forward_only:
            def _dump_backward_hook(grad):
                self._finish_dump_step()
                return grad
            scaled_loss.register_hook(_dump_backward_hook)
```

## Megatron 特有注意

- **最小卡数采集**：缩小 `world_size` 时仍须 **train/rollout TP·PP·EP 对齐**；`train_pp=2` 时须 `rollout.pipeline_model_parallel_size=2`（生产脚本常缺此项）。训练 dump 仅在 PP last stage；推理 PP=2 时与 `generate` 同 stage 的 rank 比对
- `use_remove_padding=True` 必须开启
- Megatron 旧版 actor 层还有 `megatron_actor.py` PROMPTS_ONLY 改动；**engine 架构下不必移植 megatron_actor 大段 diff**，以 engine + trainer 裁剪 + 辅助 patch 为主（见 [consistency-engine-worker.md](consistency-engine-worker.md)）
- PP / VPP 场景下仅 last stage 写 log 即可；若多 stage 重复 dump，可通过 `mpu.is_pipeline_last_stage()` _gate `_finish_dump_step`
- MoE + router replay 开启时，先关闭或确认 dump 与 replay 不冲突

## 验证

```bash
python3 scripts/check-engine-patch.py --verl-root "$VERL_ROOT" --backend megatron
```
