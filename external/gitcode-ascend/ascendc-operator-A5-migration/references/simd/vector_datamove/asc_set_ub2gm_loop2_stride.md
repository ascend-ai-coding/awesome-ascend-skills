> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_datamove/asc_set_ub2gm_loop2_stride.md
# asc_set_ub2gm_loop2_stride

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

将数据从Unified Buffer (UB) 搬运到 Global Memory (GM)时，通过调用该接口设置外层循环中源操作数在相邻迭代间的数据块间隔，以及目的操作数在相邻迭代间的数据块间隔。

以源操作数搬运场景为例，如下图所示。

![源操作数搬运场景示例](../figures/源操作数搬运场景示例.png)

## 函数原型

```cpp
__aicore__ inline void asc_set_ub2gm_loop2_stride(uint64_t loop2_src_stride, uint64_t loop2_dst_stride)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| loop2_src_stride | 输入 | 用于设置外层循环中相邻迭代源操作数的数据块间的间隔，单位为Byte，取值范围为[0, 2^21)，并且loop2_src_stride必须32B对齐。 |
| loop2_dst_stride | 输入 | 用于设置外层循环中相邻迭代目的操作数的数据块间的间隔，单位为Byte，取值范围为[0, 2^40)。 |

## 返回值说明

无

## 约束说明

每次设置循环相关参数后，需要进行寄存器的复位（循环次数设置为1），否则会影响下一次搬运的使用。

## 流水类型

PIPE_V

## 调用示例

```cpp
uint32_t loop1_size = 2;
uint32_t loop2_size =2;
uint64_t loop1_src_stride = 96;
uint64_t loop1_dst_stride = 128;
uint32_t loop2_src_stride = 196;
uint32_t loop2_dst_stride = 288;
asc_set_ub2gm_loop_size(loop1_size, loop2_size);
asc_set_ub2gm_loop1_stride(loop1_src_stride, loop1_dst_stride);
asc_set_ub2gm_loop2_stride(loop2_src_stride, loop2_dst_stride);
asc_copy_ub2gm_align(dst, src, 2, 48 * sizeof(int8_t), 0, 0, 0);
asc_set_ub2gm_loop_size(1, 1);
```