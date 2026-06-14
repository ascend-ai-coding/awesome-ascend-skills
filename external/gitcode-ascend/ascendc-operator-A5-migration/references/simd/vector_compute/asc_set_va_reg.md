> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_set_va_reg.md
# asc_set_va_reg

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| Ascend 950PR/Ascend 950DT |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

用于设置transpose的16个DataBlock地址，将操作数地址序列与地址寄存器关联，接口要求前8个和后8个地址序列与地址寄存器分别关联。

## 函数原型

```cpp
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr, __ubuf__ int8_t** src_array)
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr, __ubuf__ uint8_t** src_array)
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr, __ubuf__ int16_t** src_array)
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr, __ubuf__ uint16_t** src_array)
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr, __ubuf__ half** src_array)
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr, __ubuf__ int32_t** src_array)
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr, __ubuf__ uint32_t** src_array)
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr, __ubuf__ float** src_array)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|addr|输出|地址寄存器，类型为ub_addr8_t，可取值为：<br>&bull; VA0<br>&bull; VA1<br>&bull; VA2<br>&bull; VA3<br>&bull; VA4<br>&bull; VA5<br>&bull; VA6<br>&bull; VA7<br>数字代表寄存器顺序，每个地址寄存器只能关联8个地址，使用方法请参考调用示例|
|src_array|输入|操作数地址序列。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址对齐约束请参考[通用地址对齐约束](../general_instruction.md#通用地址对齐约束)。

## 调用示例

```cpp
constexpr uint64_t total_length = 256;    // total_length指参与计算的数据长度
__ubuf__ half src[total_length];
__ubuf__ half dst[total_length];
const uint32_t STEP = 16;
// src_list和dst_list是16个DataBlock地址的数组
__ubuf__ half* src_list[16] = {
    (__ubuf__ half*)(src + 0 * STEP),
    (__ubuf__ half*)(src + 1 * STEP),
    (__ubuf__ half*)(src + 2 * STEP),
    (__ubuf__ half*)(src + 3 * STEP),
    (__ubuf__ half*)(src + 4 * STEP),
    (__ubuf__ half*)(src + 5 * STEP),
    (__ubuf__ half*)(src + 6 * STEP),
    (__ubuf__ half*)(src + 7 * STEP),
    (__ubuf__ half*)(src + 8 * STEP),
    (__ubuf__ half*)(src + 9 * STEP),
    (__ubuf__ half*)(src + 10 * STEP),
    (__ubuf__ half*)(src + 11 * STEP),
    (__ubuf__ half*)(src + 12 * STEP),
    (__ubuf__ half*)(src + 13 * STEP),
    (__ubuf__ half*)(src + 14 * STEP),
    (__ubuf__ half*)(src + 15 * STEP)
};
__ubuf__ half* dst_list[16] = {
    (__ubuf__ half*)(dst + 0 * STEP),
    (__ubuf__ half*)(dst + 1 * STEP),
    (__ubuf__ half*)(dst + 2 * STEP),
    (__ubuf__ half*)(dst + 3 * STEP),
    (__ubuf__ half*)(dst + 4 * STEP),
    (__ubuf__ half*)(dst + 5 * STEP),
    (__ubuf__ half*)(dst + 6 * STEP),
    (__ubuf__ half*)(dst + 7 * STEP),
    (__ubuf__ half*)(dst + 8 * STEP),
    (__ubuf__ half*)(dst + 9 * STEP),
    (__ubuf__ half*)(dst + 10 * STEP),
    (__ubuf__ half*)(dst + 11 * STEP),
    (__ubuf__ half*)(dst + 12 * STEP),
    (__ubuf__ half*)(dst + 13 * STEP),
    (__ubuf__ half*)(dst + 14 * STEP),
    (__ubuf__ half*)(dst + 15 * STEP)
};
const int32_t VA_REG_ARRAY_LEN = 8;

// 接口要求前8个和后8个地址序列与地址寄存器分别关联
asc_set_va_reg(VA0, dst_list);
asc_set_va_reg(VA1, dst_list + VA_REG_ARRAY_LEN);
asc_set_va_reg(VA2, src_list);
asc_set_va_reg(VA3, src_list + VA_REG_ARRAY_LEN);
```