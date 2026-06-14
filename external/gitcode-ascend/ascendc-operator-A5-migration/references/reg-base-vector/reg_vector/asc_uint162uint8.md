> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_uint162uint8.md

# asc_uint162uint8

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

将uint16_t类型转化为uint8_t类型，无舍入模式。

溢出模式：
- 非饱和模式：结果被截断至目标格式的数据宽度，且保留最低有效位（LSB）。
- 饱和模式： 结果被限制在目标格式的可表示范围内。

由于源操作数与目的操作数类型位宽比为2:1，写入数据时需要将一个VL大小的数据分为两部分，根据不同接口选取索引0或者索引1。

## 函数原型

```cpp
// 非饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_uint162uint8(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask)
// 饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_uint162uint8_sat(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask)
// 非饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_uint162uint8_v2(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask)
// 饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_uint162uint8_sat_v2(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask)
```

## 参数说明

| 参数名    | 输入/输出 | 描述                |
| :------ | :----- | :----------------- |
| dst    | 输出    | 目的操作数（矢量数据寄存器）。            |
| src    | 输入    | 源操作数（矢量数据寄存器）。             |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 非饱和模式：输入数据超出目标数据范围时，会截断为目标数据格式，例如输入uint16_t值为256，输出uint8_t值为0。
- 饱和模式：输入数据超出目标数据范围时，返回输出类型的对应最值。
- 使能饱和模式和非饱和模式生效时，需配置ctrl寄存器，ctrl寄存器的详细说明请参见[asc_set_ctrl.md](../../sys_var/asc_set_ctrl.md)。

## 调用示例

```cpp
vector_uint16_t src;
vector_uint8_t dst;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_uint162uint8(dst, src, mask);
```
