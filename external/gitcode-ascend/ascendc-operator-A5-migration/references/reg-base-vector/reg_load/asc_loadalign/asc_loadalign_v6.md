> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_loadalign/asc_loadalign_v6.md


# asc_loadalign

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √    |

### 功能说明

Reg矢量计算数据搬运接口，适用于从UB搬入MaskReg。

### 函数原型

- 普通搬运

    ```cpp
    __simd_callee__ inline void asc_loadalign(vector_bool& dst, __ubuf__ uint32_t* src)
    __simd_callee__ inline void asc_loadalign(vector_bool& dst, __ubuf__ uint32_t* src, int32_t offset)
    __simd_callee__ inline void asc_loadalign(vector_bool& dst, __ubuf__ uint32_t* src, iter_reg offset)
    ```

- 上采样搬运

    ```cpp
    __simd_callee__ inline void asc_loadalign_upsample(vector_bool& dst, __ubuf__ uint32_t* src)
    __simd_callee__ inline void asc_loadalign_upsample(vector_bool& dst, __ubuf__ uint32_t* src, int32_t offset)
    __simd_callee__ inline void asc_loadalign_upsample(vector_bool& dst, __ubuf__ uint32_t* src, iter_reg offset)
    ```

- 下采样搬运

    ```cpp
    __simd_callee__ inline void asc_loadalign_downsample(vector_bool& dst, __ubuf__ uint32_t* src)
    __simd_callee__ inline void asc_loadalign_downsample(vector_bool& dst, __ubuf__ uint32_t* src, int32_t offset)
    __simd_callee__ inline void asc_loadalign_downsample(vector_bool& dst, __ubuf__ uint32_t* src, iter_reg offset)
    ```

### 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| offset | 输入 | 当输入为reg_iter类型时，用户通过地址寄存器传入偏移；当输入为int32_t类型时，用户直接以数值的方式传入偏移。 |

### 返回值说明

无

### 流水类型

PIPE_V

### 约束说明

- offset在缺省时默认偏移值offset=0。
- 用户可以通过自增基地址或者偏移的方式使能。

### 调用示例

```cpp
constexpr uint64_t total_length = 256;
vector_bool dst = asc_create_mask_b16(PAT_ALL);
__ubuf__ uint32_t src[total_length];
iter_reg offset = asc_create_iter_reg_b32(64);
asc_loadalign(dst, src, offset);
```