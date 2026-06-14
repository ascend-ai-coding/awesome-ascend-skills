> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_store/asc_storealign/asc_storealign_v2.md

# asc_storealign

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

reg计算数据搬运接口，适用于从矢量数据寄存器或掩码寄存器连续对齐搬出到UB的场景，并支持多种搬出模式，接口通过地址寄存器传入偏移，用户可以选择更新偏移或者更新目的操作数的地址。

- 从矢量数据寄存器连续对齐搬出到UB的场景
  - NORM搬出模式：正常模式，搬运VL数据，支持数据类型为b8、b16、b32。
  - FIRST搬出模式：忽略mask，向dst_align32b中搬运src第一个元素，支持数据类型为b8、b16、b32。
  - PACK搬出模式：压缩模式，根据mask将src中有效元素的低半部分bit数据连续存储于dst_align32b中，支持数据类型为b16、b32、b64。
  - PACKV2搬出模式：压缩模式，根据mask将src中有效元素的低8bits数据连续存储于dst_align32b中，支持数据类型为b32。
  - INTLV搬出模式：双搬出模式，忽略mask，将src0，src1中的元素交错存储于dst中，存储长度为VL*2。
- 从掩码寄存器连续对齐搬出到UB的场景
  - NORM搬出模式：正常模式，搬运VL数据，支持数据类型为u8、u16、u32。
  - PACK搬出模式：压缩模式，将src中元素的低半部分bit数据连续存储于dst中，支持数据类型为u8、u16、u32。

## 函数原型

- 从矢量数据寄存器连续对齐搬出到UB的场景
  ```cpp
  // NORM搬出模式
  __simd_callee__ inline void asc_storealign(__ubuf__  int8_t* dst_align32b, vector_int8_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  uint8_t* dst_align32b, vector_uint8_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  int16_t* dst_align32b, vector_int16_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  int32_t* dst_align32b, vector_int32_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  half* dst_align32b, vector_half src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  float* dst_align32b, vector_float src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  hifloat8_t* dst_align32b, vector_hifloat8_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign(__ubuf__  int4b_t* dst_align32b, vector_int4x2_t src, iter_reg offset, vector_bool mask)

  // FIRST搬出模式
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  int8_t* dst_align32b, vector_int8_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  uint8_t* dst_align32b, vector_uint8_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  int16_t* dst_align32b, vector_int16_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  int32_t* dst_align32b, vector_int32_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  half* dst_align32b, vector_half src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  float* dst_align32b, vector_float src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  hifloat8_t* dst_align32b, vector_hifloat8_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, iter_reg offset)
  __simd_callee__ inline void asc_storealign_1st(__ubuf__  int4b_t* dst_align32b, vector_int4x2_t src, iter_reg offset)

  // PACK搬出模式
  __simd_callee__ inline void asc_storealign_pack(__ubuf__  int16_t* dst_align32b, vector_int16_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__  int32_t* dst_align32b, vector_int32_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__  int64_t* dst_align32b, vector_int64_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__  uint64_t* dst_align32b, vector_uint64_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__  half* dst_align32b, vector_half src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__  float* dst_align32b, vector_float src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src, iter_reg offset, vector_bool mask)
  
  // PACKV2搬出模式
  __simd_callee__ inline void asc_storealign_pack_v2(__ubuf__  int32_t* dst_align32b, vector_int32_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack_v2(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src, iter_reg offset, vector_bool mask)
  __simd_callee__ inline void asc_storealign_pack_v2(__ubuf__  float* dst_align32b, vector_float src, iter_reg offset, vector_bool mask)
  
  // INTLV搬出模式
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  half* dst_align32b, vector_half src0, vector_half src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, iter_reg offset)
  __simd_callee__ inline void asc_storealign_intlv(__ubuf__  int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1, iter_reg offset)
  ```
- 从掩码寄存器连续对齐搬出到UB的场景
  ```cpp
  // NORM搬出模式
  __simd_callee__ inline void asc_storealign(__ubuf__ uint32_t* dst, vector_bool src, iter_reg offset)
  __simd_callee__ inline void asc_storealign(__ubuf__ uint16_t* dst, vector_bool src, iter_reg offset)
  __simd_callee__ inline void asc_storealign(__ubuf__ uint8_t* dst, vector_bool src, iter_reg offset)
  
  // PACK搬出模式
  __simd_callee__ inline void asc_storealign_pack(__ubuf__ uint32_t* dst,  vector_bool src,  iter_reg offset)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__ uint16_t* dst,  vector_bool src,  iter_reg offset)
  __simd_callee__ inline void asc_storealign_pack(__ubuf__ uint8_t* dst,  vector_bool src,  iter_reg offset)
  ```

## 参数说明

- 从矢量数据寄存器连续对齐搬出到UB的场景

  | 参数名  | 输入/输出 | 描述 |
  | :----- | :------- | :------- |
  | dst_align32b | 输出 | 目的操作数（矢量）的起始地址。 |
  | src | 输入 | 源操作数（矢量数据寄存器）。 |
  | src0 | 输入 | 源操作数0（矢量数据寄存器）。 |
  | src1 | 输入 | 源操作数1（矢量数据寄存器）。 |
  | mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |
  | offset | 输入 | 地址寄存器，存储地址偏移量。 |

- 从掩码寄存器连续对齐搬出到UB的场景

  | 参数名  | 输入/输出 | 描述 |
  | :----- | :------- | :------- |
  | dst | 输出 | 目的操作数（矢量）的起始地址。 |
  | src | 输入 | 源操作数（掩码寄存器）。 |
  | offset | 输入 | 地址寄存器，存储地址偏移量。 |

矢量数据寄存器、掩码寄存器和地址寄存器的详细说明请参见[reg数据类型定义.md](../../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

- 从矢量数据寄存器连续对齐搬出到UB的场景
  ```cpp
  __ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(0);
  vector_bool mask = asc_create_mask_b16(PAT_ALL);
  uint32_t offset = 128;
  iter_reg addr_reg = asc_create_iter_reg_b16(offset);
  asc_storealign(dst, src, addr_reg, mask)
  ```
- 从掩码寄存器连续对齐搬出到UB的场景
  ```cpp
  __ubuf__ uint32_t* dst = (__ubuf__ uint32_t*)asc_get_phy_buf_addr(0);
  vector_bool src;
  uint32_t offset = 0;
  iter_reg addr_reg = asc_create_iter_reg_b16(offset);
  asc_storealign(dst, src, addr_reg);
  ```
