> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_datamove/asc_set_ndim_pad_value.md
# asc_set_ndim_pad_value

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

当[asc_ndim_copy_gm2ub](./asc_ndim_copy_gm2ub.md)接口的padding_mode参数为True时，该值将设置为Padding的填充固定值。fp8_e8m0_t，fp8_e5m2_t，fp8_e4m3fn_t类型的数据需转换成int8_t类型后再调用本接口。

## 函数原型

```c++
__aicore__ inline void asc_set_ndim_pad_value(int8_t pad_value)
__aicore__ inline void asc_set_ndim_pad_value(uint8_t pad_value)
__aicore__ inline void asc_set_ndim_pad_value(int16_t pad_value)
__aicore__ inline void asc_set_ndim_pad_value(uint16_t pad_value)
__aicore__ inline void asc_set_ndim_pad_value(half pad_value)
__aicore__ inline void asc_set_ndim_pad_value(bfloat16_t pad_value)
__aicore__ inline void asc_set_ndim_pad_value(int32_t pad_value)
__aicore__ inline void asc_set_ndim_pad_value(uint32_t pad_value)
__aicore__ inline void asc_set_ndim_pad_value(float pad_value)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| pad_value | 输入 | 用于设置asc_ndim_copy_gm2ub接口Padding的填充固定值。数据类型需要与asc_ndim_copy_gm2ub接口中dst和src的数据类型保持一致。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

需配合[asc_ndim_copy_gm2ub](./asc_ndim_copy_gm2ub.md)使用。

## 调用示例

```c++
half pad_value = 2.0;
asc_set_ndim_pad_value(pad_value); 
```