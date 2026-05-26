---
name: external-gitcode-ascend-fault_diagnose
description: Ascend 故障诊断工具，提供日志采集、清洗、诊断全流程。支持集群/单机/超节点故障诊断，当用户需要排查 NPU 训练推理故障或性能劣化问题时调用。
original-name: fault_diagnose
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# 故障诊断

## 功能概述

Ascend 故障诊断工具，用于检测和分析 NPU 相关故障，帮助用户快速定位和解决问题。

## 前置条件

1. **安装 ascend-fault-diagnose**：通过 `ascend-fd version` 命令检查是否已安装
2. **安装教程**：https://www.hiascend.com/document/detail/zh/mindcluster/730/faultdiag/faultdiagug/mindxdlFDUG008.html

## 使用说明

### 使用建议

1. **集群规格限制**：因 Linux 系统最大进程数限制（默认为 1024），集群规格建议 ≤128 台服务器（1024 卡）。若服务器数量超过此规格，需使用 `ulimit -n` 命令调整文件描述符上限
2. **避免管道命令**：使用诊断工具命令时，尽量不使用管道命令，可能会影响用户 IP 的获取和日志审计

### 支持的场景

1. **整机满卡训练及推理**：仅支持对整机满卡训练及推理任务提供故障诊断能力，若非满卡场景可能导致故障根因定位错误或失败
2. **IPv4 环境**：当前仅支持 IPv4，不支持 IPv6

### 系统时间说明

1. **同步服务器时间**：请同步各训练及推理服务器的系统时间，系统时间不一致可能导致分析结果不准确
2. **同步 Host 与 Device 时间**：请同步每个服务器上 Host 系统时间与 Device 的系统时间
3. **同步容器时间**：若使用容器执行任务，请同步宿主机与容器的时间

## 使用限制

### 日志版本配套表

| 日志文件 | 对应软件 | 软件版本 | 说明 |
|----------|----------|----------|------|
| CANN应用类日志 | CANN | 7.0.RC1及以上 | Host 侧和 Device 侧应用类日志 |
| PyTorch框架训练及推理日志 | PyTorch1.11.0框架适配插件 | 5.0.RC3及以上 | - |
| MindSpore框架训练日志 | MindSpore | 2.1.0及以上 | 部分故障类型包含版本说明 |
| TensorFlow框架训练日志 | TensorFlow | - | 仅支持用户自定义 TensorFlow 故障 |
| Host OS日志 | - | - | 支持 CentOS 7.6、Debian 10.0、EulerOS 2.10/2.12、CTyunOS 22.06 等，建议日志大小 ≤512MB |
| Device侧日志 | Ascend HDK | 23.0.RC3及以上 | - |
| MindCluster组件日志 | Device Plugin/NodeD等 | 6.0.RC3及以上 | - |
| MindIE组件日志 | MindIE Server/LLM等 | 6.0.0及以上 | - |
| AMCT组件日志 | AMCT | 7.0.RC1及以上 | 集成在 CANN 包中发布 |
| MindIE Pod控制台日志 | MindIE Pod | - | - |

## 诊断流程

```
日志采集 → 日志清洗 → 故障诊断
```

### 开始诊断前需提供的信息

在开始故障诊断前，用户需要提供以下信息：

1. **故障发生阶段**：训练及推理的阶段（前/中/后）
   - **前**：训练及推理开始前，采集 NPU 环境检查文件
   - **中**：训练及推理过程中，采集 NPU 状态监测、主机资源信息等
   - **后**：训练及推理结束后，采集完整日志用于诊断

2. **节点连接数据**：提供所有节点的连接信息，用于日志收集和清洗
   - 节点 IP 地址列表
   - SSH 连接信息（用户名、密码或密钥）
   - 节点角色（master/worker）

### 日志采集脚本

使用 `log_collector.py` 脚本自动采集日志：

**节点配置文件格式**（JSON）：
```json
[
    {"ip": "192.168.1.1", "user": "root", "password": "xxx", "name": "master"},
    {"ip": "192.168.1.2", "user": "root", "key_file": "/path/to/key", "name": "worker-1"}
]
```

**使用方法**：
```bash
# 训练前采集
python3 log_collector.py -s before -n nodes.json -o ./logs

# 训练中采集
python3 log_collector.py -s during -n nodes.json -o ./logs

# 训练后采集
python3 log_collector.py -s after -n nodes.json -o ./logs
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `-s, --stage` | 采集阶段：before/during/after |
| `-n, --nodes` | 节点配置文件路径 |
| `-o, --output` | 输出目录 |

## 一、日志采集

### 1.1 训练及推理前日志采集

**文件**：`npu_info_before.txt`  
**路径**：`采集目录/environment_check/`

**采集命令**：

| 工具 | 命令 | 说明 |
|------|------|------|
| hccn_tool | `-net_health -g` | 网络健康状态 |
| hccn_tool | `-link -g` | RoCE 链路状态 |
| hccn_tool | `-ip -g` | IP 及掩码 |
| hccn_tool | `-stat -g` | 收发报文统计 |
| hccn_tool | `-link_stat -g` | 历史 link 统计 |
| npu-smi | `info` | 设备基础信息 |
| npu-smi | `info -t ecc` | ECC 计数 |
| npu-smi | `info -t board` | 硬件信息 |
| npu-smi | `info -t usages` | 内存用量 |
| npu-smi | `info -c 0 -t health` | 芯片健康状态 |

### 1.2 训练及推理中日志采集

| 文件类型 | 命名约束 | 采集方式 |
|----------|----------|----------|
| NPU网口统计 | `npu_{id}_details.csv` | hccn_tool -stat，每15秒 |
| NPU状态监测 | `npu_smi_{id}_details.csv` | npu-smi，每15秒 |
| 主机资源信息 | `host_metrics_{core_num}.json` | top 命令 |
| MindIE Pod日志 | `{pod_name}.json` | kubectl logs |

### 1.3 训练及推理后日志采集

| 日志类型 | 命名约束 | 存放路径 |
|----------|----------|----------|
| NPU环境检查 | `npu_info_after.txt` | `environment_check/` |
| 用户训练日志 | `rank-*.log`、`worker-*.log` | 采集目录 |
| CANN应用日志 | `plog-*.log`、`device-*.log` | `process_log/` |
| 主机侧日志 | `messages`、`dmesg` | 采集目录 |
| Device侧日志 | `device-os_*.log` | `device_log/` |
| MindCluster日志 | `devicePlugin*.log` 等 | `dl_log/` |
| MindIE日志 | `mindie-*.log` | `mindie/log/debug/` |
| AMCT日志 | `amct_*.log` | `amct_log/` |

**主机侧日志采集命令**：

```bash
# 内核消息日志
dmesg -T | tail -n 100000 > 采集目录/dmesg

# 系统监测日志
cp /var/log/sysmonitor.log 采集目录/

# Device侧日志
msnpureport
```

**MindCluster日志采集命令**：

```bash
cp /var/log/mindx-dl/devicePlugin 采集目录/dl_log
cp /var/log/mindx-dl/noded 采集目录/dl_log
cp /var/log/ascend-docker-runtime 采集目录/dl_log
cp /var/log/mindx-dl/volcano-scheduler 采集目录/dl_log
cp /var/log/mindx-dl/volcano-controller 采集目录/dl_log
cp /var/log/mindx-dl/npu-exporter 采集目录/dl_log
```

### 1.4 采集目录结构

```
采集目录/
├── messages                    # 主机侧操作系统日志
├── dmesg                       # 主机侧内核消息日志
├── sysmonitor.log              # 主机侧系统监测日志
├── rank-*.txt                  # 训练控制台日志
├── process_log/                # CANN应用日志
├── device_log/                 # Device侧日志
├── dl_log/                     # MindCluster组件日志
├── mindie/                     # MindIE组件日志
├── amct_log/                   # AMCT组件日志
└── environment_check/          # NPU环境检查文件
    ├── npu_info_before.txt
    ├── npu_info_after.txt
    ├── npu_smi_*_details.csv
    ├── npu_*_details.csv
    └── host_metrics_*.json
```

## 二、日志清洗

### 2.1 环境配置

```bash
# 配置环境变量（普通用户）
export PATH=$PATH:/usr/local/python3.7.5/bin

# 验证配置
ascend-fd -h
```

### 2.2 执行清洗

```bash
# 创建输出目录
mkdir 清洗输出目录

# 执行清洗命令
ascend-fd parse -i 采集目录 -o 清洗输出目录 --performance
```

### 2.3 清洗输出文件

| 文件 | 说明 |
|------|------|
| `ascend-kg-parser.json` | 故障事件分析清洗结果（推理引擎输入） |
| `ascend-kg-analyzer.json` | 故障事件分析清洗结果 |
| `ascend-rc-parser.json` | 根因节点分析清洗结果 |
| `device_ip_info.json` | 设备 IP 信息 |
| `nic_clean.csv` | 网络拥塞清洗结果 |
| `nad_clean.csv` | 计算降频清洗结果 |
| `process_*.csv` | CPU 资源抢占清洗结果 |
| `plog-parser-*.log` | 根因节点分析清洗后日志 |

### 2.4 清洗业务流日志

通过 Python API 接口清洗业务流日志。

**导入接口**：
```python
from ascend_fd import parse_fault_type
```

**调用示例**：
```python
results, err_msg_list = parse_fault_type(input_log_list)
```

**输入参数 input_log_list 格式**：

```python
[
    {
        "log_domain": {
            "server": "10.1.1.1",
            "device": ["0", "1"]
        },
        "log_items": [
            {
                "item_type": "MindIE",
                "log_lines": [
                    "[ERROR] xxx",
                    "[ERROR] yyy"
                ]
            }
        ]
    }
]
```

**输入参数说明**：

| 字段 | 参数类型 | 必选 | 描述 |
|------|----------|------|------|
| log_domain | Dictionary | 是 | 日志域 |
| server | String | 是 | 服务器地址 |
| device | List | 是 | 发生过故障的全量卡信息 |
| log_items | List | 是 | 日志项 |
| item_type | String | 是 | 日志类型 |
| log_lines | List | 是 | 待解析的日志行 |

**输出参数 results 格式**：

```python
[
    {
        "error_type": "AISW_MindIE_MS_HttpServer_01",
        "fault_domain": "Software",
        "attribute": {
            "key_info": "",
            "component": "MindIE",
            "module": "MS",
            "cause": "Httpserver通信超时",
            "description": "等待时间超过设定的时延。",
            "suggestion": ["1. 请联系华为工程师处理；"]
        },
        "device_list": [
            {
                "server": "172.0.0.1",
                "device": ["0", "1", "2"]
            }
        ]
    }
]
```

**输出参数说明**：

| 字段 | 参数类型 | 描述 |
|------|----------|------|
| error_type | String | 故障码 |
| fault_domain | String | 故障领域 |
| attribute | Dictionary | 故障属性 |
| key_info | String | 关键日志 |
| component | String | 故障组件 |
| module | String | 故障模块 |
| cause | String | 故障原因 |
| description | String | 故障描述 |
| suggestion | List | 建议方案 |
| device_list | List | 发生该故障的设备列表 |

**错误信息 err_msg_list**：接口执行过程中产生的错误信息列表。

### 2.5 根因节点清洗及诊断

通过 Python API 接口清洗和诊断根因节点。

**导入接口**：
```python
from ascend_fd import parse_root_cluster
from ascend_fd import diag_root_cluster
```

**调用示例**：
```python
# 根因节点清洗
rc_parse_results, rc_parse_err_msg = parse_root_cluster(input_log_list)

# 根因节点诊断
results, err_msg_list = diag_root_cluster(rc_parse_results)
```

**输入参数 input_log_list 格式**：
```python
[
    {
        "log_domain": {
            "server": "10.1.1.1",
            "instance_id": "instance_name"
        },
        "log_items": [
            {
                "item_type": "plog",
                "pid": 3199,
                "device_id": 0,
                "rank_id": 0,
                "log_lines": ["[ERROR] xxx."]
            }
        ]
    }
]
```

**输入参数说明**：

| 字段 | 参数类型 | 必选 | 描述 |
|------|----------|------|------|
| log_domain | Dictionary | 是 | 日志域 |
| server | String | 是 | 服务器地址 |
| instance_id | String | 是 | 实例 ID |
| log_items | List | 是 | 日志项 |
| item_type | String | 是 | 日志类型 |
| pid | Int | 是 | 进程 ID |
| device_id | Int | 是 | 设备 ID |
| rank_id | Int | 是 | Rank ID |
| log_lines | List | 是 | 待解析的日志行 |

**输出参数 results 格式**：
```python
{
    'analyze_success': True,
    'fault_description': {
        'code': 102,
        'string': '所有有效节点的Plog都没有错误日志信息，无法定位根因节点。'
    },
    'root_cause_device': ['ALL Device'],
    'device_link': [],
    'remote_link': '',
    'first_error_device': '',
    'last_error_device': ''
}
```

**输出参数说明**：

| 字段 | 参数类型 | 描述 |
|------|----------|------|
| analyze_success | Bool | 诊断是否成功（True/False） |
| fault_description | Dictionary | 故障描述 |
| code | Int | 故障码 |
| string | String | 故障码描述 |
| root_cause_device | List | 根因设备信息 |
| device_link | List | 根因节点链 |
| remote_link | String | 卡间等待链 |
| first_error_device | String | 最早发生错误的 Device |
| last_error_device | String | 最晚发生错误的 Device |

**错误信息**：
- `rc_parse_err_msg`：清洗过程中产生的错误信息列表
- `err_msg_list`：诊断过程中产生的错误信息列表

### 2.6 故障事件清洗及诊断

通过 Python API 接口清洗和诊断故障事件。

**导入接口**：
```python
from ascend_fd import parse_knowledge_graph
from ascend_fd import diag_knowledge_graph
```

**调用示例**：
```python
# 故障事件清洗
kg_parse_results, kg_parse_err_msg = parse_knowledge_graph(input_log_list, custom_entity)

# 故障事件诊断
results, err_msg_list = diag_knowledge_graph(kg_parse_results)
```

**输入参数 input_log_list 格式**：
```python
[
    {
        "log_domain": {
            "server": "10.1.1.1"
        },
        "log_items": [
            {
                "item_type": "MindIE",
                "path": "/log/debug/mindie-ms_11_202411061400.log",
                "device_id": 0,
                "modification_time": "2025-08-21 23:50:59.999999",
                "component": "Controller",
                "log_lines": ["[ERROR] xxx."]
            }
        ]
    }
]
```

**输入参数说明**：

| 字段 | 参数类型 | 必选 | 描述 |
|------|----------|------|------|
| log_domain | Dictionary | 是 | 日志域 |
| server | String | 是 | 服务器地址 |
| log_items | List | 是 | 日志项 |
| item_type | String | 是 | 日志类型 |
| path | String | 否 | 日志文件路径（清洗 NPU 环境检查文件时必填） |
| device_id | Int | 否 | 设备卡号 |
| modification_time | String | 否 | 日志修改时间（作为故障发生时间） |
| component | String | 否 | 组件名称（支持 Coordinator 和 Controller） |
| log_lines | List | 是 | 待解析的日志行 |

**自定义故障实体 custom_entity 格式**：
```python
{
    "41001": {  # 故障码（自定义，不能与已有故障码相同）
        "attribute.class": "Software",
        "attribute.component": "AI Framework",
        "attribute.module": "Compiler",
        "attribute.cause_zh": "抽象类型合并失败",
        "attribute.description_zh": "对函数输出求梯度时，抽象类型不匹配，导致抽象类型合并失败。",
        "attribute.suggestion_zh": [
            "1. 检查求梯度的函数的输出类型与sens_param的类型是否相同，如果不相同，修改为相同类型；",
            "2. 自动求导报错Type Join Failed"
        ],
        "attribute.error_case": [
            "grad = ops.GradOperation(sens_param=True)",
            "# test_net输出类型为tuple(Tensor, Tensor)",
            "def test_net(a, b):",
            "    return a, b"
        ],
        "attribute.fixed_case": [
            "grad = ops.GradOperation(sens_param=True)",
            "# test_net输出类型为tuple(Tensor, Tensor)",
            "def test_net(a, b):",
            "    return a, b"
        ],
        "rule": [{"dst_code": "20106"}],
        "source_file": "TrainLog",
        "regex.in": ["Abstract type", "cannot join with"]
    }
}
```

**custom_entity 参数说明**：

| 字段 | 参数类型 | 必选 | 描述 |
|------|----------|------|------|
| attribute.class | String | 是 | 故障类别 |
| attribute.component | String | 是 | 故障组件 |
| attribute.module | String | 是 | 故障模块 |
| attribute.cause_zh | String | 是 | 故障原因 |
| attribute.description_zh | String | 是 | 故障描述 |
| attribute.suggestion_zh | List | 是 | 建议方案列表 |
| attribute.error_case | List | 否 | 错误示例代码 |
| attribute.fixed_case | List | 否 | 修复示例代码 |
| rule | List | 是 | 故障传播规则 |
| source_file | String | 是 | 日志来源文件类型 |
| regex.in | List | 是 | 匹配正则表达式列表 |

**输出参数 results 格式**：
```python
[
    {
        'analyze_success': True,
        'version_info': {},
        'note': '',
        'fault': [{
            'code': 'NORMAL_OR_UNSUPPORTED',
            'component': '',
            'module': '',
            'cause_zh': '故障事件分析模块无结果',
            'description_zh': '可能为正常训练作业，无故障发生。',
            'suggestion_zh': '1. 若存在问题无法解决，请联系华为工程师处理',
            'class': '',
            'fault_source': ['1.1.1.1 device-Unknown'],
            'fault_chains': []
        }]
    }
]
```

**输出参数说明**：

| 字段 | 参数类型 | 描述 |
|------|----------|------|
| analyze_success | Bool | 诊断是否成功（True/False） |
| version_info | Dictionary | 版本信息 |
| note | String | 备注 |
| fault | List | 故障事件列表 |
| code | String | 故障码 |
| component | String | 故障组件 |
| module | String | 故障模块 |
| cause_zh | String | 故障原因 |
| description_zh | String | 故障描述 |
| suggestion_zh | String | 建议方案 |
| class | String | 故障类别 |
| fault_source | List | 故障来源 |
| fault_chains | List | 故障传播链 |

**错误信息**：
- `kg_parse_err_msg`：清洗过程中产生的错误信息列表
- `err_msg_list`：诊断过程中产生的错误信息列表

## 三、故障诊断

### 3.1 集群故障诊断

```bash
# 创建诊断结果输出目录
mkdir 诊断结果输出目录

# 执行诊断命令（默认仅诊断根因节点、故障事件两个模块）
ascend-fd diag -i 诊断输入目录 -o 诊断结果输出目录

# 执行完整诊断（包含性能劣化检测模块）
ascend-fd diag -i 诊断输入目录 -o 诊断结果输出目录 --performance
```

### 3.2 单机故障诊断

单机诊断默认返回故障事件模块的对应数据，适用于单节点故障排查。

```bash
# 创建单机诊断结果输出目录
mkdir 单机诊断结果输出目录

# 执行单机诊断命令
ascend-fd single-diag -i 采集目录 -o 单机诊断结果输出目录
```

**单机诊断输出目录**：
```
单机诊断结果输出目录/
└── fault_diag_result/
    └── diag_report.json    # 诊断结果
```

**单机诊断说明**：
- 单机诊断会扫描节点中所有有效日志的故障事件
- 若回显出现故障事件分析，表示当前故障可能导致训练或推理任务异常退出
- 执行出错时，可通过 `diag_report.json` 文件查看所有异常信息

### 3.3 超节点故障诊断

超节点故障诊断提供三种场景：
- 超节点拓扑信息非手动关联场景
- 超节点拓扑信息手动关联场景
- 缺失 Host 日志场景

#### 3.3.1 非手动关联场景

要求 BMC、Host、LCNE 三类日志同时存在，不可缺失某一类日志。

```bash
# 创建超节点诊断结果输出目录
mkdir 超节点诊断结果输出目录

# 执行超节点诊断命令
ascend-fd diag -i 诊断输入目录 -o 诊断结果输出目录 -s super_pod
```

**清洗结果目录结构**：
```
超节点清洗结果输出目录/
├── bmc/
│   ├── bmc_xxx.xx.xx.xx4_1/
│   │   ├── ascend-kg-analyzer.json
│   │   ├── ascend-kg-parser.json
│   │   └── server-info.json
│   └── bmc_xxx.xx.xx.xx5_1/
│       └── ...
├── host/
│   ├── log_collect_node-29-121_20250616/
│   │   ├── ascend-kg-analyzer.json
│   │   ├── ascend-kg-parser.json
│   │   ├── ascend-rc-parser.json
│   │   └── server-info.json
│   └── log_collect_node-29-124_20250616/
│       └── ...
└── lcne/
    ├── xxx.xx.xx.xx6/
    │   └── ...
    └── xxx.xx.xx.xx7/
        └── ...
```

**诊断结果输出**：
```
fault_diag_result/
├── diag_report.json    # 诊断结果
└── topo_info.json      # 超节点拓扑信息
```

#### 3.3.2 手动关联场景

清洗时需手动关联 BMC、Host、LCNE 三类日志。

**清洗命令示例**：
```bash
ascend-fd parse \
  --host_log parse_input/host/xxx.xx.xx.131/host_log/ \
  --mindie_log parse_input/host/xxx.xx.xx.131/mindie/ \
  --process_log parse_input/host/xxx.xx.xx.131/process_log/ \
  --bmc_log parse_input/bmc/worker-104 \
  --lcne_log parse_input/lcne/worker-204 \
  -o 清洗结果输出目录/worker-1
```

**清洗结果目录结构**：
```
超节点清洗结果输出目录/
├── worker-1/
│   ├── ascend-kg-analyzer.json
│   ├── ascend-kg-parser.json
│   ├── ascend-rc-parser.json
│   ├── device_ip_info.json
│   └── server-info.json
├── worker-2/
│   └── ...
└── worker-5/
    └── ...
```

**诊断命令**：
```bash
ascend-fd diag -i 诊断输入目录 -o 诊断结果输出目录
```

#### 3.3.3 超节点特有故障类型

| 故障名称 | 描述 | 建议方案 |
|----------|------|----------|
| 转发引擎模块功能失效 | LANSWITCH 芯片不稳定 | 联系华为工程师处理 |
| 转发芯片端口降到1/2 lane故障 | 转发芯片端口 L1<-->CPU 故障 | 联系华为工程师处理 |
| 转发芯片端口down故障 | 转发芯片端口down故障 L1<-->CPU | 联系华为工程师处理 |
| 转发引擎局部功能失效 | 转发芯片配置错误告警 | 联系华为工程师处理 |
| 转发引擎整体功能失效 | 转发芯片内部致命故障 | 联系华为工程师处理 |
| MindIE建链失败 | MindIE 实例建链失败 | 排查发生建链失败的节点 |

#### 3.3.4 缺失Host日志场景

缺失 Host 日志时，需要将 BMC、LCNE 清洗结果存放至同一目录下。

**清洗结果目录结构**（仅有 LCNE 日志）：
```
超节点清洗结果输出目录/lcne/
├── worker-200/
│   ├── ascend-kg-analyzer.json
│   ├── ascend-kg-parser.json
│   └── server-info.json
├── worker-201/
│   └── ...
├── worker-202/
│   └── ...
├── worker-203/
│   └── ...
└── worker-204/
    └── ...
```

**诊断命令**：
```bash
ascend-fd diag -i 诊断输入目录/lcne -o 诊断结果输出目录
```

**诊断结果输出**：
```
fault_diag_result/
└── diag_report.json    # 诊断结果
```

**说明**：
- 缺失 Host 日志时，根因节点分析无法定位根因节点（显示 `Unknown Device`）
- 现象描述会提示"未查找到有效的Plog文件"
- 故障事件分析仍可检测 LCNE/BMC 相关故障

#### 3.3.5 注意事项

- 若缺失 LCNE 或 BMC 的某一类日志，请使用手动关联场景
- 若缺失 Host 日志，将清洗输出结果存放至同一文件夹下进行诊断
- 日志级别配置较低时，会存在日志刷屏冲刷关键日志无法诊断的情况

### 3.4 诊断报告解读

诊断报告包含以下模块：

| 模块 | 说明 |
|------|------|
| 版本信息 | Fault-Diag 版本号 |
| 根因节点分析 | 定位故障发生的设备节点 |
| 故障事件分析 | 识别具体故障类型和建议方案 |
| 设备资源分析 | 分析设备资源使用情况（需 --performance 参数） |
| 网络拥塞分析 | 分析节点间的网络状态（需 --performance 参数） |

#### 关键参数说明

| 模块 | 参数 | 说明 |
|------|------|------|
| 根因节点分析 | 根因节点 | 根因设备所在的 Device |
| | 现象描述 | 根因节点分析的现象描述 |
| | 首错节点 | 任务中最早发生错误的 Device |
| | 尾错节点 | 任务中最晚发生错误的 Device |
| | 根因节点链 | 重传超次时故障节点的传播关系 |
| | 卡间等待链 | 发生 Socket/Notify 超时故障节点的传播关系 |
| 故障事件分析 | 状态码 | 故障码或 NORMAL_OR_UNSUPPORTED |
| | 故障名称 | 具体的故障名称 |
| | 故障分类 | 故障的类别及所在的组件和模块 |
| | 故障设备 | 发生故障的设备 |
| | 故障描述 | 针对该故障的详细描述 |
| | 建议方案 | 针对该故障的处理建议 |
| | 关键日志 | 该故障对应的故障日志 |
| | 关键传播链 | 该故障引发关系中最长的一条链路 |
| 设备资源分析 | 状态码 | 故障码或 NODE_DIAGNOSIS_NORMAL |
| | 故障设备 | 发生故障所在的节点名称 |
| | 故障进程 | 发生故障的进程 PID |
| | 故障区间 | 故障发生的时间区间和概率 |
| | 故障名称 | 具体的故障名称 |
| 网络拥塞分析 | 状态码 | 故障码或 NET_DIAGNOSIS_NORMAL |
| | 故障设备 | 发生故障所在的节点名称 |
| | 故障节点 | 发生故障的 Device 列表 |
| | 故障名称 | 具体的故障名称 |

### 3.5 常见故障及建议方案

| 故障名称 | 描述 | 建议方案 |
|----------|------|----------|
| Link Down: NPU端闪断错误 | NPU网口发生Link Down闪断错误，且闪断时间超过30s | 联系物理网络运维同事，收集交换机日志；排查硬件问题 |
| RDMA通信重传超次 | RDMA通信发生重传超次 | 检查网络配置、网络带宽 |
| BackendConfig配置参数校验失败 | MindIE 配置参数不合法 | 检查配置文件参数 |
| 计算降频 | NPU 频率异常降低 | 检查散热和电源状态 |
| CPU抢占（部分进程抢占） | 部分训练进程发生 CPU 资源抢占 | 检查进程资源使用情况 |
| 链路拥塞异常 | 部分通信链路发生冲突拥塞 | 检查交换机路由策略 |

### 3.6 诊断结果输出目录

```
诊断结果输出目录/
└── fault_diag_result/
    ├── diag_report.json           # 诊断结果
    └── diag_report_{实例名}.json   # 多实例推理诊断结果
```

### 3.7 注意事项

1. **磁盘空间**：清洗命令指定的输出目录磁盘空间需大于 5G
2. **敏感信息**：清洗时会读取用户采集的日志文件，请确认目录中无敏感信息
3. **单机清洗**：待清洗目录仅包含单台训练设备的原始日志
4. **参数选择**：
   - 不指定 `--performance`：仅执行根因节点分析与故障事件分析
   - 指定 `--performance`：执行所有诊断模块（含设备资源分析、网络拥塞分析）
5. **诊断场景**：
   - 出现根因节点分析和故障事件分析 → 训练任务异常退出
   - 未诊断出根因节点且故障事件分析无结果 → 性能劣化问题（不会导致训练异常退出）
6. **日志级别**：日志级别配置较低时，会存在日志刷屏冲刷关键日志无法诊断的情况，涉及环境变量：`ASCEND_GLOBAL_EVENT_ENABLE`、`HCCL_ENTRY_LOG_ENABLE`、`ASCEND_GLOBAL_LOG_LEVEL`、`ASCEND_MODULE_LOG_LEVEL`
7. **MindIE 多实例**：暂不支持发生过（有/无）冗余恢复的 MindIE 多实例推理集群故障诊断

## 参考文档

- [Ascend 故障诊断工具用户指南](https://www.hiascend.com/document/detail/zh/mindcluster/730/faultdiag/faultdiagug/mindxdlFDUG008.html)
