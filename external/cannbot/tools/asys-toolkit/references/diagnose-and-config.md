# 综合检测、组件检测与环境配置

覆盖 `asys diagnose`（压力检测、HBM 检测、CPU 检测、AI Core STL 检测、组件检测）与 `asys config`（环境配置获取与恢复）。

> **硬性前提**：综合检测与环境配置相关命令**必须在物理机且 root 用户下执行**。

## 产品支持

| 功能 | Atlas A3 / A2 系列 | Atlas 200I/500 A2 推理、Atlas 推理/训练系列 | Ascend 950PR / 950DT |
|------|:---:|:---:|:---:|
| `diagnose -r=stress_detect` / `hbm_detect` / `cpu_detect` | 支持 | **不支持** | 支持 |
| `diagnose -r=aicore_stl_detect` | **不支持** | **不支持** | **仅此支持** |
| `diagnose -r=component` | 支持 | 支持 | 支持 |
| `config` | 支持 | **不支持** | 支持 |

---

## asys diagnose — 综合检测

包括压力检测、HBM 硬件检测、CPU 检测、AI Core STL 硬件检测。

### 命令格式

```bash
# AI Core 压力检测，可能需要时间较长
asys diagnose -r=stress_detect -d=deviceId --output=path

# HBM 检测
asys diagnose -r=hbm_detect -d=deviceId --timeout=num --output=path

# CPU 检测
asys diagnose -r=cpu_detect -d=deviceId --timeout=num --output=path

# AI Core STL 硬件检测
asys diagnose -r=aicore_stl_detect -d=deviceId --output=path
```

### 参数

| 参数 | 必选 | 说明 |
|------|:---:|------|
| `-r` | **是** | 检测模式：`stress_detect` / `hbm_detect` / `cpu_detect` / `aicore_stl_detect` |
| `-d` | 否 | 指定待检测的 deviceId。不设置时默认显示所有 device 的检测结果。`Pass` 表示正常，`Warn` 表示异常 |
| `--timeout` | 否 | 指定硬件检测时间，单位秒。不传默认 **600 秒**。**仅 HBM 检测、CPU 检测时生效**。HBM 取值范围 `[0, 604800]`，设为 `0` 时表示仅执行一轮 HBM 检测；CPU 取值范围 `[1, 604800]` |
| `--output` | 否 | 检测结果文件 `diagnose_result_{time_stamp}.txt` 的保存目录。**不带该参数时输出结果不落盘、仅在终端屏幕显示** |

---

### stress_detect — AI Core 压力检测

**前置依赖**：该功能涉及执行算子，环境中需**提前安装算子二进制包**（包名 `Ascend-cann-*-ops-*.run`）。

**电压风险（重要）**：AI Core 压力检测涉及对 device 侧部分电压调整。压力检测正常结束时可自行恢复；**部分压力检测异常退出时，存在电压不能自行恢复**，这时可根据 asys 环境配置功能手动恢复电压。建议在执行压力检测**前、后分别获取电压**，用于判断电压是否异常以及是否需要恢复。

**结果判读**：

| 结果 | 含义 | 处理 |
|------|------|------|
| `Pass` | 检测成功 | — |
| `Warn` | **检测失败** | 查 Host 侧 plog（默认 `$HOME/ascend/log/run\|debug/plog/plog-{pid}_*.log`），按关键字 `[ERROR] AML` 查看日志，再按错误码定位 |

错误码首位含义：

| 首位 | 含义 |
|:---:|------|
| `1` | 用例执行失败、任务下发失败等 |
| `2` | 精度比对失败 |
| `3` | 硬件问题 |

---

### hbm_detect — HBM 检测

**结果判读**：

| 结果 | 含义 | 处理 |
|------|------|------|
| `Pass` | 检测成功 | 若返回数值 > 0，该数值表示检测后**新增 ECC 错误的个数**，用于提前激发风险地址报错并隔离，保证后续业务正常运行 |
| `Warn` | 检测失败 | 查 plog，按 `[ERROR] AML` 过滤，再按错误码定位 |

错误码首位含义：

| 首位 | 含义 |
|:---:|------|
| `1` | 用例执行失败、任务下发失败等 |
| `4` | 硬件问题 |

数值显示形式：聚合形式如 `(0, 9, 0, 0)`，单卡形式如 `Pass(9)`。

---

### cpu_detect — CPU 检测

三态结果：

| 结果 | 含义 | 处理 |
|------|------|------|
| `Pass` | 检测成功 | — |
| `Warn` | 检测过程中任务调度出现问题 | 查 plog 详细信息定位，可先按 `[ERROR] AML` 筛选 |
| `Fail` | **检测出硬件故障** | 需联系技术支持 |

---

### aicore_stl_detect — AI Core STL 硬件检测

> **仅支持在 Ascend 950PR / Ascend 950DT 上运行。**

三态结果：

| 结果 | 含义 | 处理 |
|------|------|------|
| `Pass` | 检测成功 | — |
| `Warn` | 检测过程中任务调度出现问题，**可能是硬件故障或软件问题** | 查 plog 详细信息定位，可先按 `[ERROR] AML` 筛选 |
| `Fail` | **检测出硬件故障** | 需联系技术支持 |

---

## asys diagnose -r=component — 组件检测

**当前只支持 AI Vector 组件检测，不支持并行执行。**

### 命令格式

```bash
asys diagnose -r=component -d=deviceId --output=path
```

### 参数

| 参数 | 必选/可选 | 说明 |
|------|:---:|------|
| `-r` | 必选 | 设置为 `component`，表示组件检测 |
| `-d` | 可选 | 指定待检测的 deviceId。不设置时默认显示所有 device 的检测结果。`Pass` 表示正常，`Fail` 表示异常 |
| `--output` | 可选 | 检测结果文件 `diagnose_result_{time_stamp}.txt` 的保存目录。不带该参数时结果不落盘、仅终端显示 |

**结果判读**：若检测结果为 `Fail`，可查看 `debug_info.txt` 日志定位问题。

### 输出示例

```bash
# 不指定 device，四卡全部正常
asys diagnose -r=component
 +------------------------+------------------------+
 | Group of 4 Device      | Diagnostic Result      |
 +========================+========================+
 +--- Component ----------+------------------------+
 | AI Vector              | Pass - All             |
 +------------------------+------------------------+

# 不指定 device，部分 device 正常
asys diagnose -r=component
 +------------------------+------------------------+
 | Group of 4 Device      | Diagnostic Result      |
 +========================+========================+
 +--- Component ----------+------------------------+
 | AI Vector              | Pass, Fail, Pass, Fail |
 +------------------------+------------------------+

# 指定 device 0
asys diagnose -d=0 -r=component
 +------------------------+------------------------+
 | Device ID: 0           | Diagnostic Result      |
 +========================+========================+
 +--- Component ----------+------------------------+
 | AI Vector              | Pass                   |
 +------------------------+------------------------+
```

---

## 检测结果显示规则（五种检测通用）

- 不指定 device 但 device 只有一个时，仅显示这个 device 的状态。
- 显示所有 device 的检测结果时，若所有 device 状态一致，直接显示汇总标签，如 `Pass - All`、`Warn - All`（CPU 检测与 AI Core STL 检测另有 `Fail - All`，组件检测为 `Pass - All` / `Fail - All`）。
- 若存在 device 状态不一致的情况，则依次列出每个 device 的具体状态，例如四卡场景显示 `Pass, Warn, Warn, Warn`；CPU 检测与 AI Core STL 检测可出现 `Pass, Warn, Pass, Fail`；组件检测为 `Pass, Fail, Pass, Fail`。

输出表格中的分组：压力检测归在 `Performance` 分组下，HBM / CPU / AI Core STL 检测归在 `Hardware` 分组下，组件检测归在 `Component` 分组下。

### 综合检测输出示例

```bash
asys diagnose -r=hbm_detect --timeout=3000
 +------------------------+------------------------+
 | Group of 4 Device      | Diagnostic Result      |
 +========================+========================+
 +--- Hardware -----------+------------------------+
 | HBM Detect             | Pass - All             |
 |                        | (0, 9, 0, 0)           |
 +------------------------+------------------------+

asys diagnose -r=aicore_stl_detect
 +------------------------+------------------------+
 | Group of 4 Device      | Diagnostic Result      |
 +========================+========================+
 +--- Hardware -----------+------------------------+
 | AICore STL Detect      | Pass - All             |
 +------------------------+------------------------+

asys diagnose -d=0 -r=stress_detect
 +--------------------+------------------------+
 | Device ID: 0       | Diagnostic Result      |
 +====================+========================+
 +--- Performance ----+------------------------+
 | Stress Detect      | Pass                   |
 +--------------------+------------------------+
```

---

## asys config — 环境配置

获取或恢复指定配置。

### 命令格式

```bash
# 查询压测相关配置
asys config -d=deviceId --get --stress_detect

# 恢复压测相关配置
asys config -d=deviceId --restore --stress_detect
```

### 参数

| 参数 | 说明 |
|------|------|
| `-d` | 可选，指定待操作的 deviceId。不设置时默认获取或恢复 device 0 的配置 |
| `--get` | 获取指定配置 |
| `--restore` | 恢复指定配置 |
| `--stress_detect` | 表示压测相关配置。与 `--get` 配合用于获取压测相关配置，与 `--restore` 配合用于恢复压测相关配置 |

### 使用示例和输出

各产品型号的输出信息有所不同，请以实际输出信息为准。

```bash
# 获取压测配置
asys config -d=0 --get --stress_detect
+--------------------------------+---------------------------------+
| Device ID: 0                   | CURRENT CONFIGURATION           |
+================================+=================================+
| AI Core Voltage (MV)           | 850                             |
| Bus Voltage (MV)               | 850                             |
+--------------------------------+---------------------------------+

# 恢复压测配置
asys config -d=0 --restore --stress_detect
[ASYS] [INFO]: Configuration successfully restore, on device 0.
```

### 与压力检测配合的推荐流程

```bash
# 1. 检测前取一次电压做基线
asys config -d=0 --get --stress_detect

# 2. 执行压力检测
asys diagnose -r=stress_detect -d=0 --output=$HOME/dfx_info

# 3. 检测后再取一次，对比是否已恢复
asys config -d=0 --get --stress_detect

# 4. 若未恢复（尤其检测异常退出时），手动还原
asys config -d=0 --restore --stress_detect
```
