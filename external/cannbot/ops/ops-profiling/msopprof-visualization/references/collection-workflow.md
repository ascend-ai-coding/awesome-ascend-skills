# 采集工作流与命令规划

## 目标

用最少的 profiler replay 覆盖用户请求的模块，同时保证不同模块使用的数据语义一致。

## 能力优先

没有有效环境档案时，先执行并保存：

```bash
msprof op --help
```

只使用当前安装版本实际暴露的选项和指标名称。必须保留原始大小写，例如 `PCSampling` 与 `PcSampling`。

## 特性与采集块

| 报告能力 | 主要采集块 | 首选指标或产物 | 说明 |
|---|---|---|---|
| Details/Occupancy | `details` | `Occupancy` | 核与子核占用率 |
| Compute/Memory/Advice/Cache | `memory_detail` | `MemoryDetail` | 缺失时可校验 Roofline 产物后复用 |
| Roofline | `roofline` | `Roofline` | 同时可能提供标准 CSV 指标族 |
| Timeline | `timeline` | `PipeTimeline` | 独立 replay，不与普通计数器合并 |
| Source | `source` | `Source` | 精确构建树必须包含 `.debug_line` |
| Warp Stall | `warp_stall` | `PCSampling`/`PcSampling` | 需要 SIMT 证据或显式请求 |
| Instruction Timeline | `instruction_timeline` | `instrTimeLine` > `TimelineDetail,Default` > `TimelineDetail` | 按当前 CLI 能力选择 |
| Raw Data | `raw_data` | 标准 CSV | 优先 alias MemoryDetail/Roofline |
| On-Chip Memory | `onchip_memory` | `memory_info.json` | 不执行 msprof replay |
| KernelScale | `kernel_scale` | `KernelScale` | 目前主要作为定向采集数据 |

## 去重规则

1. 精确 Kernel 且 `launch-count=1` 时，成功的 BasicInfo canary 可作为 Discovery。
2. `Roofline` 已包含完整标准 CSV 时，不再执行独立 Default。
3. `MemoryDetail` 不可用时，只有 Roofline 产物通过计算、内存、缓存和建议合同后，才可 alias。
4. Raw Data 依次尝试复用 MemoryDetail、Roofline、Details。
5. 如果 `Default` 不存在，可用一个命令采集当前 CLI 暴露的标准计数器：

```text
ArithmeticUtilization,L2Cache,Memory,MemoryL0,MemoryUB,PipeUtilization,ResourceConflictRatio
```

6. `--independent-default` 只用于有意研究跨 replay 方差。
7. 不得复用超时、非零、partial、empty 或顶层状态非 `completed` 的结果。

## Source

- `-g` 必须加入实际 Kernel/可执行文件的构建命令；
- `Source` 是 msprof 指标，`-g` 不是 msprof 参数；
- 调试符号检查必须限制在所选可执行文件的所属构建树；
- sibling build 中的符号不能满足当前 Source 门禁；
- 可使用用户提供的 `--debug-rebuild-command` 条件重建，不得硬编码工程命令。

## 可执行文件与工作目录

- 显式 `--app` 优先级最高；
- 自动发现时，NPU 构建目录（如 `build_npu`）优先于普通 host build；
- 记录全部候选、评分、歧义和最终选择；
- `--app-cwd` 默认等于算子根目录，以确保相对输入文件可解析。

## 常用命令

完整采集：

```bash
python scripts/run_pipeline.py \
  --operator-path OP \
  --output OUT \
  --preset complete
```

定向采集 Source：

```bash
python scripts/run_pipeline.py \
  --operator-path OP \
  --output OUT \
  --mode collect \
  --feature source \
  --debug-rebuild-command '用户现有的带 -g 构建命令'
```

只重绘已有数据：

```bash
python scripts/run_pipeline.py \
  --mode visualize \
  --output OUT \
  --preset complete
```
