# E01/E02 msProbe 配置模板

以下是当前官方字段形态的起点，不是跨版本固定接口。执行前检查现场 `msprobe --help`、Python 签名和配置文档。

## Statistics

```json
{
  "task": "statistics",
  "dump_path": "/path/to/dump/statistics",
  "rank": [],
  "step": [],
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
    "list": ["Torch.matmul.0.forward"],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

## 注入骨架

以下代码只表示一类训练循环中的调用顺序，不代表所有框架的固定插入位置。必须先确认实际加载文件、计算 worker、Rank 映射和
目标 forward/backward/optimizer 生命周期；由 Plugin 编排时，遵循其“实际训练入口与框架接入点”共享规则。不要按框架名称、
截图或源码目录直接套用，也不要照搬现场签名不支持的 `model`、`rank_id` 或 `seed_all` 参数。

```python
from msprobe.pytorch import PrecisionDebugger, seed_all

# 必须在模型、数据和 Dropout 初始化前调用；先核对现场签名。
seed_all(seed=1234, mode=True, rm_dropout=True)
debugger = PrecisionDebugger(config_path="config_statistics.json")

for batch in data_loader:
    debugger.start(model=model)  # 现场版本若不接受 model，则按实际签名调整
    loss = train_step(batch)
    loss.backward()
    optimizer.step()
    debugger.stop()
    debugger.step()
```

## Custom API 可见性

若 `node_a → node_b` 区间疑似存在未采集的项目 custom API，先确认运行时实际导入模块，再用 `inspect.signature` 核对
`register_custom_api`、`restore_custom_api` 或现场等价接口；也可按现场安装目录实际存在的 `custom_wrap_ops.yaml` 方式处理。
不要假定某个融合算子已在支持列表内。target/golden 应使用相同注册集合、顺序和生命周期，补采完成后恢复临时注册或包装。

custom API 未注册只说明区间证据可能不完整，不能据此认定该 API 是根因。补采、扩大 level、mix 或 targeted tensor 均须提前批准。
