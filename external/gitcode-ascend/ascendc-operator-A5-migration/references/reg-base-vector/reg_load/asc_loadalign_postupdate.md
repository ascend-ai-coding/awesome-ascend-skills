> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_loadalign_postupdate.md


# asc_loadalign_postupdate

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √    |

### 功能说明

Reg矢量计算数据搬运接口，适用于从UB搬入MaskReg。使能post mode，每调用一次接口更新目的操作数在UB上的地址。

### 函数原型

- 普通搬运

    ```cpp
    __simd_callee__ inline void asc_loadalign_postupdate(vector_bool& dst, __ubuf__ uint32_t*& src, int32_t offset)
    ```

- 上采样搬运

    ```cpp
    __simd_callee__ inline void asc_loadalign_upsample_postupdate(vector_bool& dst, __ubuf__ uint32_t*& src, int32_t offset)
    ```

- 下采样搬运

    ```cpp
    __simd_callee__ inline void asc_loadalign_downsample_postupdate(vector_bool& dst, __ubuf__ uint32_t*& src, int32_t offset)
    ```

### 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| offset | 输入 | 数据搬运的偏移量。 |

### 返回值说明

无

### 流水类型

PIPE_V

### 约束说明

无

### 调用示例

```cpp
//数据总量为256
constexpr uint64_t total_length = 256;
vector_bool dst = asc_create_mask_b16(PAT_ALL);
__ubuf__ uint32_t src[total_length];
//数据搬运的偏移量为64
int32_t offset = 64;
asc_loadalign_postupdate(dst, src, offset);
```