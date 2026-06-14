> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l12gm.md

# asc_copy_l12gm

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |

## 功能说明

将数据从L1 Buffer搬运到Global Memory。

## 函数原型

- 常规搬运

    ```c++
    __aicore__ inline void asc_copy_l12gm(__gm__ void* dst, __cbuf__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride)
    ```
- 同步搬运

    ```c++
    __aicore__ inline void asc_copy_l12gm_sync(__gm__ void* dst, __cbuf__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride)
    ```

## 参数说明


| 参数名 | 输入/输出 | 描述                                                              |
| :--- |:------|:----------------------------------------------------------------|
| dst | 输出    | 目的操作数的起始地址。|
| src | 输入    | 源操作数的起始地址。|
| n_burst | 输入    | 待搬运的连续传输数据块个数。取值范围：[1, 4095]。 |
| len_burst | 输入    | 待搬运的每个连续传输数据块的长度，单位为32个字节。取值范围：[1, 65535]。 |
| src_stride | 输入    | 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位为32个字节。 |
| dst_stride | 输入    | 相邻迭代间，目标操作数前一个分形起始地址与后一个分形起始地址的间隔，单位为32个字节。 |

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- src的起始地址要求按照对应数据类型所占字节数对齐。
- dst的起始地址要求32字节对齐。
- 如果需要执行多条asc_copy_l12gm指令，且asc_copy_l12gm指令的目的地址存在重叠，需要插入同步指令，保证多个asc_copy_l12gm指令的串行化，防止出现异常数据。

## 调用示例

```cpp
//待搬运的连续传输数据块为1
constexpr uint16_t n_burst = 1;
//待搬运的每个连续传输数据块的长度为32个字节
constexpr uint16_t len_burst = 1;
//相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔为0
constexpr uint16_t src_stride = 0;
//相邻迭代间，目标操作数前一个分形与后一个分形起始地址的间隔为32个字节
constexpr uint16_t dst_stride = 1;
__cbuf__ half src[256];
__gm__ half dst[256];
asc_copy_l12gm(dst, src, n_burst, len_burst, src_stride, dst_stride);
```