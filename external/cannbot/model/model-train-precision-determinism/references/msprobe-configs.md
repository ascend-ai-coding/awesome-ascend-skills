# E08 msProbe 配置模板

以下模板须经现场版本探测后再用。

## Statistics + 校验值

```json
{
  "task": "statistics",
  "dump_path": "/path/to/dump/run_a",
  "rank": [],
  "step": [0],
  "level": "mix",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "summary_mode": "md5"
  }
}
```

## Targeted Tensor + bench_path

```json
{
  "task": "tensor",
  "dump_path": "/path/to/dump/run_b_tensor",
  "rank": [],
  "step": [0],
  "level": "mix",
  "async_dump": false,
  "tensor": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "bench_path": "/path/to/dump/run_a",
    "summary_mode": "md5",
    "diff_nums": 1
  }
}
```

`bench_path` 必须指向由兼容版本、相同配置和 `summary_mode: "md5"` 生成的预置数据。若现场版本不支持该字段，
改为先 compare 定位，再用 `list`/`scope` 手工缩小 tensor dump。

## 注入骨架

以下代码只表示一类训练循环中的调用顺序，不代表所有框架的固定插入位置。必须先确认实际加载文件、计算 worker、Rank 映射和
目标 forward/backward/optimizer 生命周期；由 Plugin 编排时，遵循其“实际训练入口与框架接入点”共享规则。run A/run B 必须
命中相同进程角色和边界；不要照搬现场不支持的 `model`、`rank_id`、`seed_all` 或 `bench_path` 参数。

```python
from msprobe.pytorch import PrecisionDebugger, seed_all

seed_all(seed=1234, mode=True, rm_dropout=True)  # 先核对签名和副作用
debugger = PrecisionDebugger(config_path="config_md5.json")

for batch in data_loader:
    debugger.start(model=model)
    loss = train_step(batch)
    loss.backward()
    optimizer.step()
    debugger.stop()
    debugger.step()
```

## Custom API 可见性

若首个问题 API 的输入已不一致，应检查最后一致节点到该输入之间是否存在未采集的项目 custom API。先用运行时导入路径确认
模块对象，再以 `inspect.signature` 核对 `register_custom_api`、`restore_custom_api` 或现场版本提供的注册方式；不要假定网上示例
的参数形式可用。注册应在目标计算开始前完成，run A/run B 使用相同 API 集合、顺序和生命周期，并在补采后恢复临时注册或包装。

custom API 未注册只能说明可疑区间可能不完整，不能据此认定该 API 是根因。补采、扩大 level 或 targeted tensor 前仍需用户批准。
