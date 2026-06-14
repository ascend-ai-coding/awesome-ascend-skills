> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_program_counter.md

# asc_get_program_counter

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_program_counter()
```

## 参数说明

无

## 返回值说明

返回int64_t类型的程序计数器指针。

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

```cpp
asc_get_program_counter();
```