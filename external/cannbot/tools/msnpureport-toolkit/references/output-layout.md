# 导出目录结构说明

导出内容存储在**执行命令时的当前目录**（或 `-o` 指定路径）下、以时间戳命名的文件夹中。

日志路径中的 `id` 和 `pid` 分别代表 Device ID 和进程 ID，日志文件名中的 `*` 表示该日志文件创建时的时间戳。

## 单次导出目录结构

```sh
├── hisi_logs
│   └── device-id   //对应设备的异常信息
│       ├── 时间戳  //以时间戳命名的文件夹，记录黑匣子日志、黑匣子的设备事件信息和黑匣子存储空间中的历史维测信息等
│       │   ├── bin
│       │   │   └── vmcore //vmcore文件，仅在Device OS心跳丢失时产生
│       ├── ub      //Unified Bus统一总线的维测信息，当前该目录软链接ub_info/mami，后续将废弃
│       │    ...
│       └── history.log  //记录各文件夹生成的异常场景信息
│   └── device_info.txt  //记录设备信息
├── slog
│   └── dev-os-id
│       ├── debug
│       │   ├── device-id
│       │   │   └── device-id_*.log  //Device侧非Control CPU上的系统类日志
│       │   ├── device-app-pid
│       │   │   └── device-app-pid.log  //Device侧应用进程产生的调试日志，仅在回传到Host失败时生成
│       │   └── device-os
│       │       └── device-os_*.log  //Device侧系统进程产生的调试日志
│       ├── run
│       │   ├── device-app-pid
│       │   │   └── device-app-pid.log  //Device侧应用进程产生的运行日志，仅在回传到Host失败时生成
│       │   ├── device-os
│       │   │   └── device-os_*.log  //Device侧系统进程产生的运行日志
│       │   └── event
│       │       └── event_*.log  //Device侧系统进程产生的EVENT日志
│       ├── security
│       │   ├── device-app-pid
│       │   │   └── device-app-pid.log  //Device侧应用进程产生的安全日志，仅在回传到Host失败时生成
│       │   └── device-os
│       │       └── device-os_*.log  //Device侧系统进程产生的安全日志
│       └── slogd  //日志工具自身产生的维测日志
│   └── host
│       └── host_kernel.log  //Host驱动内核日志
├── stackcore   //进程coredump时的调用栈信息
│   └── dev-os-id
│       │    ... //文件名因产品型号而异（stackcore.slogd.xx 或 coretrace.slogd.xx），可用asys工具解析
├── message  //Device侧OS系统日志
└── system_info
│   └── dev-os-id
│       │      ├── event_sched  //事件调度模块的维测信息
│       │      ├── hal  //驱动proc日志
│       │      ├── start_dmesg  //驱动proc日志
│       │      └── tee  //tee模块日志
│       │      └── ...
├── ub_info //ub领域的维测信息，可用asys工具解析
│   └── dev-os-{id}
│       ├── ubctl //ubctl相关信息
│       ├── ...
│       └── mami  //mami领域的维测信息
│           ├── ubnl_dfx_statistic.bin
│           ├── ubnl_dfx_ssu_schedule.bin
│           ├── ubnl_dfx_config_item.bin
│           ├── ubmem_daw.bin
│           ├── ubtpl_acl.bin
│           ├── sl_to_vl.bin
│           └── ...
├── ao_info //AO领域的维测信息
│   └── dev-os-{id}
│       ├── ao_self //AO区日志
│       ├── ao_cnt //AO计数
│       ├── ao_uart //串口录音
```

## 连续导出目录结构

```sh
├── hisi_logs
│   └── device-id
│       ├── 时间戳  //以时间戳命名的文件夹，记录黑匣子日志、黑匣子的设备事件信息和黑匣子存储空间中的历史维测信息等
│       └── history.log  //记录各文件夹生成的异常场景信息，如Device心跳丢失等
├── slog
│   └── dev-os-id
│       ├── debug
│       │   ├── device-id
│       │   │   └── device-id_*.log  //Device侧非Control CPU上的系统类日志
│       │   ├── device-app-pid
│       │   │   └── device-app-pid.log  //Device侧应用进程产生的调试日志，仅在回传到Host失败时生成
│       │   └── device-os
│       │       └── device-os_*.log  //Device侧系统进程产生的调试日志
│       ├── run
│       │   ├── device-app-pid
│       │   │   └── device-app-pid.log
│       │   ├── device-os
│       │   │   └── device-os_*.log
│       │   └── event
│       │       └── event_*.log
│       ├── security
│       │   ├── device-app-pid
│       │   │   └── device-app-pid.log
│       │   └── device-os
│       │       └── device-os_*.log
│       └── slogd  //日志工具自身产生的维测日志
├── stackcore   //进程coredump时的调用栈信息
│   └── dev-os-id
│       │    ... //文件名因产品型号而异，可用asys工具解析
└── system_info
    └── device-id
    │   └── fault_event_{故障码}_*  //Proc文件和Device侧OS系统日志，Device故障触发时生成
    │      ├── event_sched // 事件调度模块的维测信息
    │      ├── hal // 驱动proc日志
    │      ├── message // Device侧OS系统日志
    │      └── ...
```

> 连续导出的结构中没有 `slog/host/`、`message`、`ub_info/`、`ao_info/`，Device 侧 OS 系统日志位于 `system_info/device-id/fault_event_*/message`。

## Device ID 语义

| 产品 | 目录中 Device ID 含义 |
|------|---------------------|
| Ascend 950PR/Ascend 950DT | 逻辑 ID |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 物理 ID |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 物理 ID |
| Atlas 训练系列产品 | 物理 ID |
| Atlas 推理系列产品 | 物理 ID |
| Atlas 200I/500 A2 推理产品 | 物理 ID |

## 关键文件定位指引

| 想查什么 | 看哪里 |
|---------|-------|
| AI 处理器整体运行情况 | `slog/{dev-os-id}/run/device-os/` 与 `debug/device-os/` |
| 低功耗、Task Scheduler、ISP 组件情况 | `slog/{dev-os-id}/debug/{device-id}/device-id_*.log` |
| Device 异常场景概览（如心跳丢失） | `hisi_logs/{device-id}/history.log` |
| 进程崩溃调用栈 | `stackcore/{dev-os-id}/`（需 asys 解析） |
| Device OS 心跳丢失后的内核转储 | `hisi_logs/{device-id}/{时间戳}/bin/vmcore` |
| Host 侧驱动内核日志 | `slog/host/host_kernel.log` |
| Device 故障触发的 proc/OS 日志 | `system_info/{device-id}/fault_event_{故障码}_*/` |
| 设备信息 | `hisi_logs/device_info.txt` |

> `stackcore/`、`ub_info/` 下文件需使用 asys 工具解析，详见《故障处理》中「UB 文件解析」章节：<https://hiascend.com/document/redirect/CannCommunityasys>
