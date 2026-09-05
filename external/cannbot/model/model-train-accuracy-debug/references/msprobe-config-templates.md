# msprobe 配置模板参考

## 公共参数说明

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `task` | string | dump 任务类型：`statistics` / `tensor` / `overflow_check` / `free_benchmark` |
| `dump_path` | string | dump 数据保存目录 |
| `rank` | list[int] | 采集的 rank 列表，`[]` 表示所有 rank |
| `step` | list[int] | 采集的 step 列表，`[]` 表示所有 step |
| `level` | string | dump 级别：`L0`（模块级）/ `L1`（算子级）/ `mix`（结构+统计） |
| `scope` | list[str] | 要采集的模块/算子名称列表，`[]` 表示全部 |
| `async_dump` | bool | 是否异步 dump，默认 `false` |

## 1. statistics 模式 — 轻量统计

仅采集张量统计量（均值、方差、最大值、最小值），适合初步筛查。

```json
{
    "task": "statistics",
    "dump_path": "/path/to/dump/statistics",
    "rank": [],
    "step": [0, 1, 2],
    "level": "L1",
    "seed": 1234,
    "statistics": {
        "scope": [],
        "list": []
    }
}
```

## 2. tensor 模式 — 完整张量 dump

采集完整张量数据，适合精确精度比对。

```json
{
    "task": "tensor",
    "dump_path": "/path/to/dump/tensor",
    "rank": [0],
    "step": [0],
    "level": "mix",
    "seed": 1234,
    "tensor": {
        "scope": [],
        "list": []
    }
}
```

> [!TIP]
>
> - 使用 `"level": "mix"` 同时获取网络结构（`construct.json`）和算子统计，便于后续可视化和比对。
> - `tensor` 模式数据量较大，建议限制 `rank` 和 `step` 范围。

## 3. overflow_check 模式 — 溢出检测

检测训练过程中的溢出（上溢/下溢），适合 NaN 场景深入分析。

```json
{
    "task": "overflow_check",
    "dump_path": "/path/to/dump/overflow",
    "rank": [],
    "step": [],
    "level": "L1",
    "overflow_check": {
        "scope": [],
        "list": []
    }
}
```

## 4. compare.json — 精度比对配置

用于 `msprobe compare` 命令的输入配置。

```json
{
    "npu_path": "/path/to/npu_dump/dump.json",
    "bench_path": "/path/to/bench_dump/dump.json",
    "stack_path": "/path/to/npu_dump/stack.json"
}
```

执行比对命令：

```bash
msprobe compare -i compare.json -o ./compare_output
```

比对结果文件：`compare_output/compare_result_{timestamp}.xlsx`。

## 5. 在训练脚本中注入 PrecisionDebugger

推荐在训练入口处注入（torchtitan-npu 为 `entry.py`，其他框架对应训练脚本入口）：

```python
from msprobe.pytorch import PrecisionDebugger

# 在模型初始化后、训练循环开始前
debugger = PrecisionDebugger(config_path="<config.json 路径>")

# 在训练循环中
for step in range(num_steps):
    debugger.start()    # 开启 dump
    # ... 训练逻辑 ...
    debugger.stop()     # 关闭 dump
    debugger.step()     # 推进 step 计数
```

> [!IMPORTANT]
>
> - `debugger.start()` 和 `debugger.stop()` 应包裹完整的 forward + backward + optimizer.step。
> - 确保基准环境与异常环境使用相同的 `seed`、相同的 `step` 和 `rank` 配置。
