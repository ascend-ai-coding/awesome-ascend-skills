> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_datamove/asc_set_gm2ub_loop1_stride.md

# asc_set_gm2ub_loop1_stride

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

将数据从Global Memory (GM) 搬运到 Unified Buffer (UB)时，通过调用该接口设置内层循环中相邻迭代数据块间的间隔。

以源操作数搬运场景为例，如下图所示。

![源操作数搬运场景示例](../figures/源操作数搬运场景示例.png)

## 函数原型

```cpp
__aicore__ inline void asc_set_gm2ub_loop1_stride(uint64_t loop1_src_stride, uint64_t loop1_dst_stride)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| loop1_src_stride     | 输入     | 内层循环中相邻迭代源操作数的数据块间的间隔，单位为Byte，取值范围为[0,2^40]。|
| loop1_dst_stride     | 输入     | 内层循环中相邻迭代目标操作数的数据块间的间隔，单位为Byte，取值范围为[0,2^21]，且必须32B对齐。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
asc_set_gm2ub_loop_size(2, 2);
asc_set_gm2ub_loop1_stride(96, 128);
asc_set_gm2ub_loop2_stride(192, 288);
asc_copy_gm2ub_align(dst, src, 2, 48 * sizeof(int8_t), 0, 0, false, 0, 0, 0);
asc_set_gm2ub_loop_size(1, 1);
```
