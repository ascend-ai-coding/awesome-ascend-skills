> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_system_cycle.md

# asc_get_system_cycle

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| <cann-filter npu_type="950">Ascend 950PR/Ascend 950DT | √    </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = (cycle数/50)us。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_system_cycle()
```

## 参数说明

无

## 返回值说明

返回系统Cycle数。

## 流水类型

PIPE_S

## 约束说明

该接口是PIPE_S流水，若需要测试其他流水的指令时间，需要在调用该接口前插入对应流水的同步。

## 调用示例

```cpp
asc_sync_notify(PIPE_V, PIPE_S, 0);
int64_t system_cycle_before = asc_get_system_cycle(); // 计算操作指令前的cycle数
......// 进行PIPE_V流水的计算操作
asc_sync_notify(PIPE_V, PIPE_S, 0);
int64_t system_cycle_after = asc_get_system_cycle(); // 计算操作指令后的cycle数
int64_t total_cycle = system_cycle_after - system_cycle_before; // 执行Add指令所用的cycle数
```