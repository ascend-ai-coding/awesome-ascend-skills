> **原始文档路径**: asc-devkit/docs/api/context/c_api/sync/asc_sync_data_barrier.md

# asc_sync_data_barrier

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √ </cann-filter>|

## 功能说明

用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可以通过参数控制）执行结束。

## 函数原型

```cpp
__aicore__ inline void asc_sync_data_barrier(mem_dsb_t arg)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :---  | :--- | :--- |
| arg | 输入 | 表示需要等待的内存位置，类型为mem_dsb_t，可取值为： <br>&bull; DSB_ALL：等待所有内存访问指令。 <br>&bull; DSB_DDR：等待GM访问指令。<br>&bull; DSB_UB：等待UB访问指令。<br>&bull; DSB_SEQ：预留参数，暂未启动，为后续的功能扩展做保留。|


## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

mem_dsb_t类型的入参必须直接传递枚举值字面量。

## 调用示例

```cpp
asc_sync_data_barrier(mem_dsb_t::DSB_ALL);
```