# MindSpeed-MM Profiling 配置参考

## Megatron 引擎（tools.json）

配置文件路径通过 shell 脚本中的 `MM_TOOL` 或 YAML 中的 `MM_TOOL_PATH` 指定，默认 `./mindspeed_mm/tools/tools.json`。

### profile 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `profile.enable` | bool | false | 启用 Profiling |
| `profile.profile_type` | str | static | `static`（静态）/ `dynamic`（动态） |
| `profile.ranks` | int list | [-1] | 采集的卡号。-1 表示所有卡 |
| `profile.static_param.level` | str | level1 | `level0` / `level1`（推荐）/ `level2` |
| `profile.static_param.with_stack` | bool | false | 采集调用堆栈 |
| `profile.static_param.with_memory` | bool | false | 采集 NPU 显存分配/释放事件 |
| `profile.static_param.record_shapes` | bool | false | 记录 tensor 输入形状和类型 |
| `profile.static_param.with_cpu` | bool | true | 采集 CPU 活动 |
| `profile.static_param.save_path` | str | ./npu_profiling | 输出目录 |
| `profile.static_param.start_step` | int | 10 | 开始采集的步骤（包含） |
| `profile.static_param.end_step` | int | 12 | 结束采集的步骤（不包含） |
| `profile.static_param.data_simplification` | bool | false | 启用数据简化模式 |
| `profile.static_param.aic_metrics_type` | str | PipeUtilization | `PipeUtilization` / `ArithmeticUtilization` |
| `profile.static_param.analyse_flag` | bool | true | 是否在线解析 |

### 完整示例

```json
{
  "profile": {
    "enable": true,
    "profile_type": "static",
    "ranks": [0],
    "static_param": {
      "level": "level1",
      "with_stack": false,
      "with_memory": false,
      "record_shapes": false,
      "with_cpu": true,
      "save_path": "./npu_profiling",
      "start_step": 10,
      "end_step": 12,
      "data_simplification": false,
      "aic_metrics_type": "PipeUtilization",
      "analyse_flag": true
    }
  }
}
```

## 独立 FSDP2 引擎（YAML tools.profile）

直接在训练 YAML 配置中添加 `tools` 字段。

### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tools.profile.enable` | bool | false | 启用 Profiling |
| `tools.profile.profile_type` | str | static | `static`（静态）/ `dynamic`（动态） |
| `tools.profile.ranks` | int list | [0] | 采集的卡号。-1 表示所有卡 |
| `tools.profile.static_param.level` | str | level1 | `level0` / `level1`（推荐）/ `level2` |
| `tools.profile.static_param.with_stack` | bool | false | 采集调用堆栈 |
| `tools.profile.static_param.with_memory` | bool | false | 采集 NPU 显存分配/释放事件 |
| `tools.profile.static_param.record_shapes` | bool | false | 记录 tensor 输入形状和类型 |
| `tools.profile.static_param.with_cpu` | bool | false | 采集 CPU 活动 |
| `tools.profile.static_param.save_path` | str | ./profiling | 输出目录 |
| `tools.profile.static_param.start_step` | int | 10 | 开始采集的步骤（包含） |
| `tools.profile.static_param.end_step` | int | 11 | 结束采集的步骤（不包含） |
| `tools.profile.static_param.data_simplification` | bool | false | 启用数据简化模式 |
| `tools.profile.static_param.aic_metrics_type` | str | PipeUtilization | `PipeUtilization` / `ArithmeticUtilization` |
| `tools.profile.static_param.analyse_flag` | bool | true | 是否在线解析 |

### 完整示例

```yaml
tools:
  profile:
    enable: true
    profile_type: static
    ranks:
      - 0
    static_param:
      level: level1
      with_stack: false
      with_memory: false
      record_shapes: false
      with_cpu: true
      save_path: ./npu_profiling
      start_step: 10
      end_step: 12
      data_simplification: false
      aic_metrics_type: PipeUtilization
      analyse_flag: true
```

## 采集级别

| 级别 | 采集内容 | 适用场景 |
|------|----------|----------|
| `level0` | 基础算子耗时（最小开销） | 快速概览 |
| `level1` | + AICore 利用率、通信算子、详细算子信息 | **推荐** |
| `level2` | + 缓存、显存详细计数、更细粒度数据 | 深度调试 |

## 动态采集配置

动态模式下，训练运行时在 `dynamic_param.config_path` 目录下生成 `profiler_config.json`，修改该文件即可动态控制：

```json
{
  "start_step": 10,
  "end_step": 12
}
```

## 输出目录结构

```
<save_path>/
├── <hostname>_<pid>_<timestamp>_ascend_pt/
│   ├── ASCEND_PROFILER_OUTPUT/    # 解析后的结果（analyse_flag=true 时）
│   ├── PROF_<id>/
│   │   ├── device_0/data/         # NPU 数据
│   │   └── host/data/             # CPU 数据（with_cpu=true 时）
│   └── logs/
└── ...
```

使用 **MindStudio Insight** 可视化分析。

## 离线解析 CLI

```bash
# Megatron 引擎
python mindspeed_mm/tools/profiler.py --mm-tool ./mindspeed_mm/tools/tools.json --profiler-path <profiling 数据目录>

# 独立 FSDP2 引擎
python mindspeed_mm/fsdp/tools/profiler.py --profiler-path <profiling 数据目录>
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mm-tool` | path | ./mindspeed_mm/tools/tools.json | MM 工具配置文件路径 |
| `--profiler-path` | path | 从配置文件读取 | Profiler 数据目录 |
| `--max-process-number` | int | CPU 核心数/2 | 分析的最大进程数 |
| `--export-type` | str list | text | 导出类型：`text` / `db` |
