> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_set_l3d_rpt_b.md

# asc_set_l3d_rpt_b

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |

## 功能说明

用于设置接口[asc_copy_l12l0a](../cube_datamove/asc_copy_l12l0a.md)、[asc_copy_l12l0b](../cube_datamove/asc_copy_l12l0b.md)的2D格式搬运的repeat参数。

## 函数原型

```cpp
__aicore__ inline void asc_set_l3d_rpt_b(uint64_t config)
```

## 参数说明

表1 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输入 | 用于设置接口asc_copy_l12l0a、asc_copy_l12l0b的2D格式搬运repeat参数。比特位说明参考表2。 |

表2 常用重复控制寄存器比特位说明
|L3D_RPT_B比特位    |功能|
| :-------     | :---- |
| L3D_RPT_B[15:0]    | 表示重复步长    | 
| L3D_RPT_B[23:16]   | 表示在M或K方向的重复次数，默认值为1。      | 
| L3D_RPT_B[24]      | 表示重复模式 <br> - 1'b0: 在M方向重复。 <br> - 1'b1: 在K方向重复。     | 
| L3D_RPT_B[47:32]   | 表示输出矩阵在K方向上的步长，以分形为单位。     | 
| L3D_RPT_B[63:48]   | 表示输出矩阵在M方向上的起始位置，以分形为单位。      | 

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 需要配合接口[asc_copy_l12l0a](../cube_datamove/asc_copy_l12l0a.md)、[asc_copy_l12l0b](../cube_datamove/asc_copy_l12l0b.md)使用。

## 调用示例

```cpp
uint64_t config = 0;
asc_set_l3d_rpt_b(config);
```
