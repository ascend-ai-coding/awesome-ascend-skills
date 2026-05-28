# FSDP Engine Worker 采集 Patch

**目标文件**：`{VERL_ROOT}/verl/workers/engine/fsdp/transformer_impl.py`  
**目标类**：`FSDPEngine`

> **训推一致前置**：FSDP 后端须令推理 `tensor_model_parallel_size` 与 rollout 占用卡数一致（如 4 卡 → `gen_tp=4`），否则推理 dump rank 数少于训练 rank，无法按卡比对。见 [shared-prerequisites.md](shared-prerequisites.md#训推并行切分对齐consistency-必检)。

## 1. `FSDPEngine.__init__` 末尾追加

```python
        self._debugger = None
        self._update_actor_logger_fp = None
```

## 2. 在 `FSDPEngine` 类中新增方法

```python
    def _ensure_debugger(self):
        """Lazy init debugger; skip ref engine (forward_only=True)."""
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
            "rank": self.rank,
            "step": self._debugger.service.current_iter,
            "request_ids": request_ids,
            "num_requests": len(request_ids),
        }
        self._update_actor_logger_fp.write(
            json.dumps(record, ensure_ascii=False) + "\n")
        self._update_actor_logger_fp.flush()
```

## 3. 修改 `FSDPEngine.forward_backward_batch`

在 `micro_batches, indices = prepare_micro_batches(...)` 之后、`for micro_batch in micro_batches:` 之前：

```python
        self._ensure_debugger()
        dump_phase = os.environ.get("DUMP_PHASE", "log_prob")
        phase = "log_prob" if forward_only else "update_actor"
        should_dump = dump_phase == "all" or dump_phase == phase
```

在循环内，包裹 `forward_step` + `backward`：

```python
        for micro_batch in micro_batches:
            if self._debugger is not None and should_dump:
                self._debugger.start(model=self.module)
            with ctx:
                loss, meta_info = self.forward_step(
                    micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    loss.backward()
            if self._debugger is not None and should_dump:
                self._debugger.stop()
                self._debugger.step()
                self._log_update_actor_step(micro_batch)
            output_lst.append(meta_info)
```

## FSDP 特有注意

- 必须 `actor_rollout_ref.actor.use_dynamic_bsz=False`
- 推荐 `export TORCHDYNAMO_DISABLE=1`
- `self.rank` 已在 `FSDPEngine.__init__` 中定义，直接用于日志
- PROMPTS_ONLY / metrics / rollout_corr 辅助 patch 见 [consistency-auxiliary-patches.md](consistency-auxiliary-patches.md)
- 完整训推一致流程见 [consistency-engine-worker.md](consistency-engine-worker.md)

## 验证

```bash
python3 scripts/check-engine-patch.py --verl-root "$VERL_ROOT" --backend fsdp
```
