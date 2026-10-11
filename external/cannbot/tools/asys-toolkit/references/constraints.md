# 使用约束、产品支持与依赖

---

## 六条使用约束

来自「asys 工具功能及约束」章节：

1. **不支持在 Ascend RC 形态下使用。** asys 仅支持 Ascend EP 形态。
2. **相同用户、相同时间段内，同机器同时作业时，收集到的数据会有交叉。**
3. **非 root 用户获取到的数据范围会受限**，具体限制见下方权限要求。
4. **集群、容器、虚拟机、云场景不支持**一键式工具收集故障信息。
5. asys 涉及大量维测信息的收集，因此涉及内存占用，**不建议多进程并行执行**，否则可能导致 asys 执行出错或环境异常。
6. asys 会检索 trace 日志所在的目录，**若 trace 日志文件过多，可能导致执行时间长**。trace 日志默认存放路径 `$HOME/ascend/atrace/`。

### 额外的并行限制

- 组件检测（`diagnose -r=component`）**不支持并行执行**。
- **不支持对同一个卡住进程并行导出堆栈信息**，否则可能执行命令失败。

### root 权限要求

| 场景 | 权限 |
|------|------|
| Device 侧固件日志（`device-*`） | **root** |
| Device 侧系统日志（message、device-os） | **root** |
| 黑匣子、stackcore 文件、coretrace 文件 | **root** |
| `asys diagnose` 全部检测模式 | **物理机 + root** |
| `asys config` | **物理机 + root** |

### 其他用户一致性要求

run 包安装日志的收集要求 **run 包安装用户与应用程序执行用户一致**。

---

## 产品支持矩阵

| 功能 | Atlas A3 训练/推理 | Atlas A2 训练/推理 | Atlas 200I/500 A2 推理 | Atlas 推理系列 | Atlas 训练系列 | Ascend 950PR / 950DT |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| `collect`（故障信息收集） | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `launch`（复跑+收集） | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `collect -r=stacktrace`（实时堆栈） | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `info`（软硬件/状态） | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `health`（健康检查） | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `analyze -r=trace` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `analyze -r=coredump` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `analyze -r=stackcore` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `analyze -r=ub` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `analyze -r=aicore_error` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `diagnose -r=component`（组件检测） | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `diagnose`（综合检测：stress / hbm / cpu） | 支持 | 支持 | **不支持** | **不支持** | **不支持** | 支持 |
| `config`（环境配置） | 支持 | 支持 | **不支持** | **不支持** | **不支持** | 支持 |
| `profiling`（性能采集） | 支持 | 支持 | **不支持** | **不支持** | **不支持** | 支持 |
| `diagnose -r=aicore_stl_detect` | **不支持** | **不支持** | **不支持** | **不支持** | **不支持** | **仅此支持** |
| `analyze -r=coretrace` | **不支持** | **不支持** | **不支持** | **不支持** | **不支持** | **仅此支持** |

两个功能限定单一平台，跨型号使用前务必确认：

- **`analyze -r=coretrace`**：仅 Ascend 950PR / Ascend 950DT 支持。
- **`diagnose -r=aicore_stl_detect`**：仅 Ascend 950PR / Ascend 950DT 支持。

---

## 外部依赖

asys 的部分能力依赖系统工具，缺失时对应功能不可用：

| 依赖 | 用途 | 安装 |
|------|------|------|
| `gdb` | `analyze -r=coredump` 解析 core 文件 | `apt-get install gdb` 或 `yum install gdb` |
| `readelf` | `analyze -r=stackcore` 获取文件信息 | Linux 系统自带，需确认已安装 |
| `addr2line` | `analyze -r=stackcore` / `-r=coretrace` 解析堆栈函数名和行号 | Linux 系统自带，需确认已安装 |
| 算子二进制包 `Ascend-cann-*-ops-*.run` | `diagnose -r=stress_detect` 执行算子 | 按 CANN 安装流程安装 |

`readelf` 和 `addr2line` 还要求**执行该脚本的用户有权限执行**。

### 上游工具

以下文件由 msnpureport 工具从 Device 侧导出，是 asys 解析功能的输入：

| 文件 | 用于 |
|------|------|
| Device 侧 stackcore 文件 | `analyze -r=stackcore` |
| `coretrace.*` | `analyze -r=coretrace` |
| UB 维测信息 `*.bin` | `analyze -r=ub` |

导出方法见《msnpureport 工具》的「导出 Device 侧系统类日志和其他维测信息 > 单次导出 Device 侧系统类日志和其他维测信息」章节。

---

## 环境变量清单

### 影响自动收集范围

`collect` 与 `analyze -r=aicore_error` 在未指定源目录时自动收集，行为受下列变量影响。**执行 asys 命令时取值需与业务运行时保持一致**，否则收集到的信息可能不准确：

| 变量 | 作用 |
|------|------|
| `ASCEND_PROCESS_LOG_PATH` | 进程日志路径 |
| `NPU_COLLECT_PATH` | 故障信息的保存路径。设置后系统在该目录下新建 `/extra-info/ops/`，写入 `op_compile_stats.log`；不设置则系统不生成该文件，asys 也不会收集算子编译过程信息 |
| `DUMP_GRAPH_PATH` | dump 图路径 |
| `ASCEND_WORK_PATH` | 工作路径 |
| `ASCEND_CACHE_PATH` | 缓存路径 |
| `ASCEND_CUSTOM_OPP_PATH` | 自定义算子包安装路径。设置时收集该目录下的 `config/*.json` |

`aicore_error` 场景下若这些环境变量都不存在，会从执行 asys 命令的当前目录下收集。

### 决定收集内容有无

| 变量 | 影响 |
|------|------|
| `ASCEND_OPP_PATH` | 算子库的安装路径。设置时 asys 按 `${ASCEND_OPP_PATH}/vendors/config.ini` 的 `load_priority` 字段收集 `${ASCEND_OPP_PATH}/vendors` 下的自定义算子配置信息；也决定是否收集 `${ASCEND_OPP_PATH}/debug_kernel` 下的调试版本二进制信息。未配置或配置不正确则默认不收集 |

### 被 asys launch 临时接管

`asys launch` 执行时自动开启、结束时自动关闭。执行前手动设置的会被覆盖不生效；复跑任务脚本中设置的会反过来覆盖 asys 的设置：

```
NPU_COLLECT_PATH
ASCEND_PROCESS_LOG_PATH
ASCEND_WORK_PATH
```

### 影响实时堆栈导出

| 变量 | 影响 |
|------|------|
| `ASCEND_COREDUMP_SIGNAL` | 导出实时堆栈需向指定进程发送**信号 35**。设为 `none` 时关闭 trace 处理的信号集，会**终止卡住进程**且无法导出堆栈。「打开信号集」指设为非 `none` 的其他值或未设置该变量 |

### asys.ini 对应的环境变量

`[launch]` 段各配置项与环境变量的对应关系见 [collect-and-launch.md](collect-and-launch.md) 的「可选配置 asys.ini」。

---

## 需另查的手册

asys 输出中的部分内容需要配套手册解读：

| 内容 | 手册 |
|------|------|
| 故障码详细描述 | 《黑匣子异常错误码信息列表》、《健康管理故障定义》（需取对应版本） |
| 环境变量详细配置说明 | 《环境变量参考》 |
| trace 日志与 Host 侧 stackcore 文件获取 | 《日志参考》「查看 trace 日志」 |
| Device 侧 stackcore / coretrace / UB 文件导出 | 《msnpureport 工具》 |
| profiling 结果文件字段含义 | 《性能调优工具》性能数据文件参考 |
| CANN 软件安装 | 《CANN 软件安装》 |
