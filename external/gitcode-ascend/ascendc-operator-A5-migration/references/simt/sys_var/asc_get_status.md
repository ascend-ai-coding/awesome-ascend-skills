> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_status.md

# asc_get_status

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

获取状态信息。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_status()
```

## 参数说明

无

## 返回值说明

状态信息。各bit含义如下：
| bit范围    | 含义 |
| ----------- |:----|
| 5 | 浮点运算溢出。SIMD指令int16_t和int32_t算术运算溢出也会上报到该位。 |
| 6 | 浮点运算下溢（结果浮点数小于非规格化数能表示的最小值，则结果为0）。 |
| 7 | 将任意浮点数转换为无符号整数时，输入为负数。 |
| 8 | 从L0C到UB的数据搬运过程中发生溢出（float->half、int32_t->half）。 |
| 9 | 从L0C到UB的数据搬运过程中发生下溢（float->half）。 |
| 10 | CUBE累加运算溢出（可能是float、half、int32_t）。 |
| 11 | CUBE累加运算下溢（可能是float、half）。 |
| 13 | 标量指令输入为NaN/INF。 |
| 14 | 向量指令输入为NaN/INF。 |
| 15 | CUBE指令输入为NaN/INF。 |
| 61 | 数据搬运类指令指令输入为NaN/INF |
| 其它bit位 | 保留位。 |

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
int64_t status = asc_get_status();
printf("status is %x", status);// 需用%x将其打印成十六进制的数
```