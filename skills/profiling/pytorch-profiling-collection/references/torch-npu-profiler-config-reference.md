# torch_npu.profiler 配置参考

## Profiler Level 对比

| Level | 采集内容 | 数据量 | 推荐场景 |
|-------|---------|-------|---------|
| Level_none | 不采集 Level 层级控制的数据 | 极小 | 自定义打点场景 |
| Level0 | 上层应用数据、底层 NPU 数据、算子信息 | 大 | 深度算子分析 |
| Level1 | Level0 + CANN 层 AscendCL + AI Core 性能指标 + 通信算子 | 较大 | 常规通信+计算分析（推荐） |
| Level2 | Level1 + CANN 层 Runtime + AI CPU 数据 | 大 | 全量 CANN 层分析 |

## 三种采集方式对比

| 方式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 方式A（推荐） | 标准训练循环 | 自动识别循环并注入 step() | 需能识别出训练循环位置 |
| 方式B | 推理脚本/无循环 | 最简单，不需要 schedule | 没有 step 维度数据 |
| 方式C | 指定代码段 | 精准控制采集范围 | 需要用户指定起始位置 |

## Profiling 输出产物结构

```
profiling_result/
├── worker_0_20260101_120000/     # 采集 worker 目录
│   ├── trace_1.json              # Chrome trace（可拖入 chrome://tracing 查看）
│   ├── memory_1.json             # 内存数据
│   └── ...
├── worker_0_20260101_120000.pt.json  # 汇总 trace（给分析工具用）
└── profiler_metadata.json        # 元信息
```

## 常见配置项

| 配置项 | 类型 | 说明 |
|--------|------|------|
| activities | list | 采集的活动类型，可选 `torch_npu.profiler.ProfilerActivity.CPU` 和 `ProfilerActivity.NPU` |
| schedule.wait | int | 采集开始前等待的 step 数 |
| schedule.warmup | int | 预热 step 数（不采集） |
| schedule.active | int | 实际采集的 step 数 |
| schedule.repeat | int | 重复次数 |
| schedule.skip_first | int | 整个采集开始前一次性跳过的 step 数（动态 Shape 场景建议 ≥ 10） |
| schedule.skip_first_wait | int | 是否跳过第一次循环的 `wait` 阶段；1=跳过，0=不跳过（默认） |
| record_shapes | bool | 是否记录 tensor 维度信息 |
| profile_memory | bool | 是否采集内存数据 |
| with_stack | bool | 是否记录调用栈 |
| with_modules | bool | 是否记录模块层级 |
| with_flops | bool | 是否记录 FLOPs |
| experimental_config.aic_metrics | enum | AI Core 指标类型 |
| experimental_config.profiler_level | enum | 采集级别 Level0/Level1/Level2 |
| experimental_config.mstx | bool | 是否启用 mstx 打点 |
| experimental_config.l2_cache | bool | 是否采集 L2 Cache 数据 |
| experimental_config.op_attr | bool | 是否采集算子属性 |
| experimental_config.data_simplification | bool | 是否启用数据简化 |
| experimental_config.record_op_args | bool | 是否记录算子参数 |
| experimental_config.host_sys | list | Host 侧系统事件采集列表 |
| experimental_config.sys_io | bool | 是否采集系统 IO |
| experimental_config.sys_interconnection | bool | 是否采集系统互联 |

## schedule 参数 `skip_first` vs `skip_first_wait`

| 参数 | 何时生效 | 典型用途 |
|------|---------|---------|
| `skip_first` | 整个 schedule 启动**之前** | 跳过训练初期的抖动 / 动态 Shape 不稳定 step（建议 ≥ 10） |
| `skip_first_wait` | 第一次循环的 `wait` 阶段 | 想让第一次采集尽快开始，避免在 `skip_first` 之后还要等一整个 `wait` 周期 |

**示例场景**：`wait=20, skip_first=10`

- `skip_first_wait=0`（默认）：第一次预热前需等待 `skip_first + wait = 10 + 20 = 30` 步
- `skip_first_wait=1`：第一次预热前只需等待 `skip_first = 10` 步；后续循环间仍按 `wait=20` 等待

**配置公式**：`step 总数 >= skip_first + (wait + warmup + active) * repeat`

## 参考链接

- [昇腾 PyTorch Profiler 官方文档](https://www.hiascend.com/document/detail/zh/mindstudio/830/T&ITools/Profiling/atlasprofiling_16_0121.html)
- [mstx 打点指南](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu-mstx.md)
- [profiling-analysis 系列 Skill](../../profiling-analysis/SKILL.md)
