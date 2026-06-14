> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l12l0a_mx.md

# asc_copy_l12l0a_mx

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

将Mx scale矩阵从L1 Buffer搬运到L0A Buffer。

## 函数原型

```cpp
__aicore__ inline void asc_copy_l12l0a_mx(uint64_t dst, __cbuf__ fp8_e8m0_t* src, uint16_t x_start_pos, uint16_t y_start_pos, uint8_t x_step, uint8_t y_step, uint16_t src_stride, uint16_t dst_stride)
__aicore__ inline void asc_copy_l12l0a_mx_sync(uint64_t dst, __cbuf__ fp8_e8m0_t* src, uint16_t x_start_pos, uint16_t y_start_pos, uint8_t x_step, uint8_t y_step, uint16_t src_stride, uint16_t dst_stride)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst       | 输出 | 目的操作数的地址。 |
| src | 输入 | 源操作数（Mx scale矩阵）的起始地址，数据类型为fp8_e8m0_t。 |
| x_start_pos | 输入 | 源操作数X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。 |
| y_start_pos | 输入 | 源操作数Y轴方向的起始位置，即K维度方向，单位为32B。 |
| x_step | 输入 | 源操作数X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：[0, 255]。 |
| y_step | 输入 | 源操作数Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：[0, 255]。|
| src_stride | 输入 | 源操作数X轴方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。 |
| dst_stride | 输入 | 目的操作数X轴方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。 |

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- scale矩阵的分形固定为(16, 2, 16)，对应L0A Buffer的分形为(16, 32, 16)，占L0A Buffer地址的1 / 16，需要按照这种地址约束存放地址。
- dst（L0A Buffer的地址）的数据类型支持fp4x2_e2m1_t、fp4x2_e1m2_t、fp8_e5m2_t、fp8_e4m3fn_t，src的数据类型支持fp8_e8m0_t。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
__cbuf__ fp8_e8m0_t src[1024];
__ca__ fp8_e5m2_t dst[1024];
uint16_t x_start_pos = 0;
uint16_t y_start_pos = 0;
uint8_t x_step = 0;
uint8_t y_step = 0;
uint16_t src_stride = 8;
uint16_t dst_stride = 8;
uint64_t mx_dst_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst)) / 16;
asc_copy_l12l0a_mx(mx_dst_addr, src, x_start_pos, y_start_pos, x_step, y_step, src_stride, dst_stride);
```