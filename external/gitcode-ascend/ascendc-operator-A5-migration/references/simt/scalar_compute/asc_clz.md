> **原始文档路径**: asc-devkit/docs/api/context/c_api/scalar_compute/asc_clz.md

# asc_clz

## 产品支持情况

|产品   | 是否支持 |
| :------------|:----:|
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

计算一个uint64_t类型整数在二进制表示下的前导零个数，即从二进制最高位开始，到第一个出现二进制1为止，中间连续的0的数量。

## 函数原型

```c++
__aicore__ inline int64_t asc_clz(uint64_t value_in)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| value_in     | 输入     | 待统计的数字。|

## 返回值说明

反馈value_in的前导0的个数。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
uint64_t value_in = 0x0fffffffffffffff;
int64_t ans = asc_clz(value_in); //返回ans = 4
```
