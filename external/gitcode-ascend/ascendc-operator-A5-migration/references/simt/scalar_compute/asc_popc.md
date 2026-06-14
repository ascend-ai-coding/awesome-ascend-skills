> **原始文档路径**: asc-devkit/docs/api/context/c_api/scalar_compute/asc_popc.md

# asc_popc

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |
| <cann-filter npu_type="950"> Ascend 950PR/Ascend 950DT | √ </cann-filter>|

## 功能说明

获取一个uint64_t类型数字的二进制中1的个数。

## 函数原型

```c++
__aicore__ inline int64_t asc_popc(uint64_t value)
```

## 参数说明

|参数名|输入/输出|描述|
| :------ | :--- | :------------ |
|value   |输入   |被统计的二进制数字。

## 返回值说明

value的二进制中1的个数。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
uint64_t scalar = 33;
// 输出数据count_one为2
int64_t count_one = asc_popc(scalar);
```