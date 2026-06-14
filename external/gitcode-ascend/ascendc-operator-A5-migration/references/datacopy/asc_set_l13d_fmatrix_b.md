> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_set_l13d_fmatrix_b.md

# asc_set_l13d_fmatrix_b

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

设置Feature map属性描述，用于在调用[asc_copy_l12l0a](./asc_copy_l12l0a.md)/[asc_copy_l12l0b](./asc_copy_l12l0b.md)的3D格式搬运接口时配置填充数值。
仅当asc_copy_l12l0a/asc_copy_l12l0b接口的f_matrix_ctrl参数指示从右矩阵获取FeatureMap的属性时使用本接口设置Feature map属性描述，否则使用asc_set_l13d_fmatrix接口。

## 函数原型

```c++
__aicore__ inline void asc_set_l13d_fmatrix_b(asc_l13d_fmatrix_config& config)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输入 | 用于设置asc_copy_l12l0a/asc_copy_l12l0b的3D格式搬运接口的Feature map属性参数，详细说明请参考[asc_l13d_fmatrix_config](../struct/asc_l13d_fmatrix_config.md)。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
asc_l13d_fmatrix_config config;
asc_set_l13d_fmatrix_b(config);
```