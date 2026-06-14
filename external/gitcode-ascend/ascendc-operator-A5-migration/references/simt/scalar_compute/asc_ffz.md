> **原始文档路径**: asc-devkit/docs/api/context/c_api/scalar_compute/asc_ffz.md

# asc_ffz

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

获取一个uint64_t类型数字的二进制表示中从最低有效位开始的第一个0出现的位置，如果没找到则返回-1。

## 函数原型

```c++
__aicore__ inline int64_t asc_ffz(uint64_t value)
```

## 参数说明

|参数名|输入/输出|描述|
| :------ | :--- | :------------ |
|value   |输入   |输入数据，数据类型是uint64_t。|

## 返回值说明

value中第一个0出现的位置。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
uint64_t value = 33;
// 输出数据count_first_zero为1
int64_t count_first_zero = asc_ffz(value);
```
