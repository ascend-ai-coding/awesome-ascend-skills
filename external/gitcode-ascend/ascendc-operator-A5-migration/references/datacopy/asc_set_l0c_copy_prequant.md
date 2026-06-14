> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_set_l0c_copy_prequant.md

# asc_set_l0c_copy_prequant

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| <cann-filter npu_tpye="950">Ascend 950PR/Ascend 950DT | √    </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

数据搬运过程中进行随路量化时，通过调用该接口设置量化流程中的标量量化参数。

## 函数原型

```c++
__aicore__ inline void asc_set_l0c_copy_prequant(uint64_t config)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| config     | 输入     | 量化过程中使用到的标量量化参数。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
uint64_t config = 2;
asc_set_l0c_copy_prequant(config);
```