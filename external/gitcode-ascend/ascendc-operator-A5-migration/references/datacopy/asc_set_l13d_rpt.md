> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_set_l13d_rpt.md

# asc_set_l13d_rpt

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

用于设置Load3Dv2接口的repeat参数。

## 函数原型

```c++
__aicore__ inline void asc_set_l13d_rpt(asc_load3d_v2_config& config)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输入 | 用于设置Load3Dv2接口的repeat参数，详细说明请参考[asc_load3d_v2_config.md](../struct/asc_load3d_v2_config.md)。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 需配合[asc_load3d_v2_config.md](../struct/asc_load3d_v2_config.md)使用，用于设置Load3Dv2接口的repeat参数。

## 调用示例

```c++
asc_load3d_v2_config config;
asc_set_l13d_rpt(config); // 设置Load3D的repeat相关参数为默认值rpt_stride：0，rpt_time：1，rpt_mode：0
```