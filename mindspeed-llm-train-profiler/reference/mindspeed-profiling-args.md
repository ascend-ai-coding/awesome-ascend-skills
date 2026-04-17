# MindSpeed-LLM Profiling 参数参考

## Mcore 后端（pretrain_gpt.py CLI 参数）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--profile` | bool | false | 启用 Profiling |
| `--profile-step-start` | int | *（无）* | 开始采集的步骤（包含）。**必须 >= 1**，框架默认值 0 但运行时会被拒绝。 |
| `--profile-step-end` | int | -1 | 结束采集的步骤（不包含）。-1 表示采集到训练结束 |
| `--profile-ranks` | int list | [0] | 采集的卡号。-1 表示所有卡 |
| `--profile-level` | str | level0 | `level0` / `level1`（推荐）/ `level2` |
| `--profile-with-cpu` | bool | false | 采集 CPU 活动（数据加载、调度） |
| `--profile-with-memory` | bool | false | 采集 NPU 显存分配/释放事件 |
| `--profile-with-stack` | bool | false | 采集调用堆栈 |
| `--profile-record-shapes` | bool | false | 记录 tensor 形状 |
| `--profile-save-path` | str | ./profile | 输出目录 |
| `--profile-export-type` | str | text | `text` 或 `db` |

## FSDP2 后端（YAML 配置，training: 下）

入口脚本：`train_fsdp2.py <config.yaml>`。在 `training:` 下添加 Profiling 字段：

```yaml
training:
  profile: true
  profile_step_start: 2
  profile_step_end: 4
  profile_ranks: [0]
  profile_level: level1
  profile_with_cpu: true
  profile_save_path: ./profiling_output
```

完整配置模板参见 `MindSpeed-LLM/examples/fsdp2/` 目录。

## 采集级别

| 级别 | 采集内容 | 适用场景 |
|------|----------|----------|
| `level0` | 基础算子耗时 | 快速概览 |
| `level1` | + AICore 利用率、通信算子 | **推荐** |
| `level2` | + 缓存、显存详细计数 | 深度调试 |

## 输出目录结构

```
<主机名>_<PID>_<时间戳>_ascend_pt/
├── ASCEND_PROFILER_OUTPUT/    # 解析后的结果
├── PROF_<id>/
│   ├── device_0/data/         # NPU 数据
│   └── host/data/             # CPU 数据
└── logs/
```

使用 **MindStudio Insight** 可视化分析：https://gitcode.com/Ascend/msinsight
