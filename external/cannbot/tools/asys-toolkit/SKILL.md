---
name: external-cannbot-tools-asys-toolkit
description: CANN 一键式故障信息收集工具 asys 使用指导。覆盖故障信息收集（collect）、业务复跑收集（launch）、软硬件与 Device
  状态查询（info）、健康检查（health）、综合检测与组件检测（diagnose）、trace/coredump/stackcore/coretrace/UB/AI
  Core Error 六类文件解析（analyze）、实时堆栈导出、环境配置（config）、性能数据采集（profiling）。触发场景：需要收集 NPU 故障维测信息、训练或推理进程卡死要导出堆栈、Segmentation
  fault 要解析 coredump、日志出现 aivec/aicore error exception、排查 Device 健康状态与故障码、做 HBM/CPU/AI
  Core 硬件检测、或用 asys 采集系统级性能数据时。
original-name: asys-toolkit
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# asys 故障维测工具

asys 是 CANN 提供的一键式故障信息收集工具,用于提高系统故障维测效率。

> **形态限制**:asys **仅支持在 Ascend EP 形态下使用**,RC 形态不支持。使用前先确认,避免在不支持的环境上白跑。

## 环境准备

前提是已在 CANN 运行环境上安装 Toolkit 软件包。以 CANN 运行用户登录环境后设置环境变量:

```bash
source ${INSTALL_DIR}/set_env.sh
```

`${INSTALL_DIR}` 替换为 CANN 软件安装路径。以 root 用户安装为例,默认路径为 `/usr/local/Ascend/cann`。

设置环境变量后可**直接输入 `asys` 命令**,无需进入工具所在目录。

```bash
# 定位 asys 并检查依赖是否就绪
bash scripts/find_asys.sh
```

---

## 症状路由

按现场症状选命令,不确定时从 `asys collect` 起步。

| 现场症状 | 命令 | 详细文档 |
|---------|------|---------|
| 需要打包一份完整维测信息交付分析 | `asys collect` | [collect-and-launch.md](references/collect-and-launch.md) |
| 问题可稳定复现,想连带复跑一起收集 | `asys launch --task=...` | [collect-and-launch.md](references/collect-and-launch.md) |
| 训练/推理进程卡住不退出 | `asys collect -r=stacktrace` | [collect-and-launch.md](references/collect-and-launch.md) |
| 日志或屏幕出现 `there is an aivec error exception` / `there is an aicore error exception` | `asys analyze -r=aicore_error` | [analyze-files.md](references/analyze-files.md) |
| 进程中断退出并报 `Segmentation fault` | `asys analyze -r=coredump` | [analyze-files.md](references/analyze-files.md) |
| 拿到 `stackcore_*.txt` / `coretrace.*` / trace `*.bin` / UB `*.bin` 要解析 | `asys analyze -r=stackcore\|coretrace\|trace\|ub` | [analyze-files.md](references/analyze-files.md) |
| 怀疑 Device 不健康、要看故障码 | `asys health` | [info-and-health.md](references/info-and-health.md) |
| 要看芯片型号、温度、功耗、HBM、AI Core 利用率 | `asys info -r=status` | [info-and-health.md](references/info-and-health.md) |
| 要确认 CANN 包版本、系统内核、硬件规格 | `asys info -r=software\|hardware` | [info-and-health.md](references/info-and-health.md) |
| 怀疑硬件故障(AI Core / HBM / CPU / AI Core STL / AI Vector) | `asys diagnose -r=...` | [diagnose-and-config.md](references/diagnose-and-config.md) |
| 压测异常退出后电压未恢复 | `asys config --restore --stress_detect` | [diagnose-and-config.md](references/diagnose-and-config.md) |
| 要采集 AI Core / 内存 / 带宽 / 功耗性能数据 | `asys profiling` | [profiling.md](references/profiling.md) |
| 复跑报错、堆栈导出超时 | — | [faq.md](references/faq.md) |

---

## 子命令总览

参数风格统一:`-r=` 指定模式,`--output=` 指定输出目录前缀。

```
asys
 ├── collect    收集故障信息（-r=stacktrace 时为实时堆栈导出）
 ├── launch     复跑业务 + 收集故障信息
 ├── info       软硬件与 Device 状态展示（-r=status|software|hardware）
 ├── health     Device 健康检查
 ├── diagnose   综合检测与组件检测
 │              （-r=stress_detect|hbm_detect|cpu_detect|aicore_stl_detect|component）
 ├── analyze    文件解析
 │              （-r=trace|coredump|stackcore|coretrace|ub|aicore_error）
 ├── config     环境配置获取与恢复（--get / --restore）
 └── profiling  性能数据采集
```

### 最常用的三条命令

```bash
# 1. 不复跑,直接收集环境上的软硬件信息与日志,并打包
asys collect --tar="True" --output=$HOME/dfx_info

# 2. 复跑业务并收集（默认开启算子编译文件、GE dump 图收集）
asys launch --task="sh ../run.sh" --tar="True" --output=$HOME/dfx_info

# 3. AI Core Error 故障解析
asys analyze -r=aicore_error --path=${HOME}/aic_err_info_timestamp
```

---

## 输出目录约定

`--output` 是**路径前缀**,不是最终目录:

| 子命令 | 最终目录 |
|--------|---------|
| `collect` / `launch` / `analyze` | `{output}/asys_output_timestamp` |
| `diagnose` | `{output}/diagnose_result_{time_stamp}.txt` |
| `profiling` | `{output}/asys_profiling_result_timestamp` |

不带 `--output` 时结果落在命令执行目录(`diagnose` 与 `component` 检测则只在终端显示、不落盘)。取值为空、为无效字符串、目录无写权限或创建失败时,asys **退出执行并报错**。

---

## 使用前必读的四条约束

1. **环境变量要与业务运行时一致**。自动收集依赖 `ASCEND_PROCESS_LOG_PATH`、`NPU_COLLECT_PATH`、`DUMP_GRAPH_PATH`、`ASCEND_WORK_PATH`、`ASCEND_CACHE_PATH`、`ASCEND_CUSTOM_OPP_PATH`,取值不一致时收集到的信息可能不准确。
2. **不要并行执行**。asys 涉及大量维测信息收集、占用内存,多进程并行可能导致执行出错或环境异常;同一卡住进程不支持并行导出堆栈;组件检测不支持并行执行。
3. **权限决定数据范围**。Device 侧固件日志、系统日志、黑匣子、stackcore、coretrace 需 root;`diagnose` 与 `config` 必须在物理机且 root 用户下执行。
4. **先清 trace 日志**。asys 会检索 trace 日志所在目录,文件过多会导致执行时间长。默认路径 `$HOME/ascend/atrace/`。

完整约束、产品支持矩阵、外部依赖(gdb / readelf / addr2line)见 [constraints.md](references/constraints.md)。

---

## 产品支持速查

并非所有功能全系列可用,在不支持的型号上执行会白跑:

| 功能 | Atlas A3 / A2 系列 | Atlas 200I/500 A2 推理、Atlas 推理/训练系列 |
|------|:---:|:---:|
| `collect` / `launch` / `info` / `health` | 支持 | 支持 |
| `analyze -r=trace\|coredump\|stackcore\|ub\|aicore_error` | 支持 | 支持 |
| `diagnose -r=component` | 支持 | 支持 |
| `diagnose`(综合检测) | 支持 | **不支持** |
| `config` / `profiling` | 支持 | **不支持** |
| `analyze -r=coretrace` | **不支持** | **不支持** |
| `diagnose -r=aicore_stl_detect` | **不支持** | **不支持** |

`coretrace` 解析与 AI Core STL 检测**仅支持 Ascend 950PR / Ascend 950DT**。

---

## 参考文档索引

| 文档 | 内容 |
|------|------|
| [collect-and-launch.md](references/collect-and-launch.md) | `collect`、`launch`、实时堆栈导出的参数、产物结构、asys.ini 配置 |
| [info-and-health.md](references/info-and-health.md) | `info` 三种信息类型、`health` 状态与故障级别映射 |
| [diagnose-and-config.md](references/diagnose-and-config.md) | 压力 / HBM / CPU / AI Core STL / 组件检测,电压获取与恢复 |
| [analyze-files.md](references/analyze-files.md) | trace / coredump / stackcore / coretrace / UB / aicore_error 六种解析 |
| [profiling.md](references/profiling.md) | `-r` 采集类型、`--aic_metrics` PMU 类型 |
| [constraints.md](references/constraints.md) | 使用约束、产品支持矩阵、外部依赖、环境变量清单 |
| [faq.md](references/faq.md) | 复跑报错、实时堆栈导出超时的定位方法 |

## 脚本工具

| 脚本 | 用途 |
|------|------|
| `scripts/find_asys.sh` | 定位 asys 可执行文件,检查 EP 形态、外部依赖与关键环境变量 |

## 与其他 Skill 的关系

**本 skill 是 asys 命令与参数的真源**,其他 skill 需要 asys 用法时引用本 skill,不要复制命令细节。

- `ascendc-env-check`:npu-smi 不可用时的设备查询回退,回退命令引用本 skill。
- `ascendc-crash-debug`:卡死/崩溃调试流程,堆栈与 coredump 解析步骤引用本 skill。
- `ops-profiling`:算子性能采集主路径是 msprof;`asys profiling` 是系统级周期采样,定位不同。
- `msnpureport-toolkit`:Device 侧 stackcore、coretrace、UB 文件的导出工具,是本 skill 解析功能的上游。

## 信息来源

命令、参数、约束与输出示例均来自 CANN oam-tools 开源仓 `docs/zh/asys`(19 个文档)。开源仓与商用版文档存在差异时以开源仓为准。文档中未给出的行为不做推断。本 skill 内容未在真实昇腾环境上实跑验证。
