> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_set_gm2l1_pad.md

# asc_set_gm2l1_pad

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

将数据从Global Memory (GM) 非对齐搬运到L1 Buffer时，通过调用该接口设置连续搬运数据块左右两侧需要填补的数据值。

## 函数原型

```cpp
__aicore__ inline void asc_set_gm2l1_pad(uint32_t pad_val)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| pad_val     | 输入     | 左右两侧需要填补的数据值。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
asc_set_gm2l1_loop_size(loop1_size, loop2_size);
asc_set_gm2l1_loop1_stride(loop1_src_stride, loop1_dst_stride);
asc_set_gm2l1_loop2_stride(loop2_src_stride, loop2_dst_stride);
// 源操作数非对齐，需要填补数据
asc_set_gm2l1_pad(0);
asc_copy_gm2l1_align(dst, src, size);
asc_set_gm2l1_loop_size(1, 1);
```
