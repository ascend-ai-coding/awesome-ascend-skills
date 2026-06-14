> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_core_id.md

# asc_get_core_id

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √ </cann-filter>|

## 功能说明

获取当前核的编号。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_core_id()
```

## 参数说明

无

## 返回值说明

返回当前核的编号。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无