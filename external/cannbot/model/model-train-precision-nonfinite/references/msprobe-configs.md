# E04 msProbe 配置模板

执行前必须核对现场 msProbe 版本、命令帮助和 Python API 签名。

## Statistics

```json
{
  "task": "statistics",
  "dump_path": "/path/to/dump/statistics",
  "rank": [],
  "step": [0],
  "level": "mix",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

## Targeted Tensor

```json
{
  "task": "tensor",
  "dump_path": "/path/to/dump/tensor",
  "rank": [0],
  "step": [0],
  "level": "mix",
  "async_dump": false,
  "tensor": {
    "scope": [],
    "list": ["Torch.softmax.0.forward"],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

## 注入骨架

以下代码只表示一类训练循环中的调用顺序，不代表所有框架的固定插入位置。必须先确认实际加载文件、计算 worker、Rank 映射和
目标 forward/backward/optimizer 生命周期；由 Plugin 编排时，遵循其“实际训练入口与框架接入点”共享规则。需要反向证据时，
`stop()`/`step()` 不得位于 backward 之前；所有参数仍以现场签名为准。

```python
from msprobe.pytorch import PrecisionDebugger, seed_all

seed_all(seed=1234, mode=True, rm_dropout=True)  # 先核对现场签名
debugger = PrecisionDebugger(config_path="config_statistics.json")

for batch in data_loader:
    debugger.start(model=model)
    loss = train_step(batch)
    loss.backward()
    optimizer.step()
    debugger.stop()
    debugger.step()
```

若现场版本支持 `task: "nan_check"`，可作为补充信号使用，但不得跳过版本探测，也不得用它替代有限放大、非有限转换边界和传播分析。使用前，level须配置为"L1"且设置环境变量：INF_NAN_MODE_FORCE_DISABLE=1。
