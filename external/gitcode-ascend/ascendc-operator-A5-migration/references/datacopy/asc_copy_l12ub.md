> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l12ub.md

# asc_copy_l12ub

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :----------------------- | :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    x    |

## 功能说明

将数据从L1 Buffer搬运到Unified Buffer中。

## 函数原型

- 高维切分搬运
    ```cpp
    __aicore__ inline void asc_copy_l12ub(__ubuf__ void * dst_addr, __cbuf__ void * src_addr, bool sub_blockid, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    ```

- 同步搬运
    ```c++
    __aicore__ inline void asc_copy_l12ub_sync(__ubuf__ void * dst_addr, __cbuf__ void * src_addr, bool sub_blockid, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst_addr | 输出 | 目的操作数起始地址。 |
| src_addr | 输入 | 源操作数起始地址。 |
| sub_blockid | 输入 | 使用的子块ID。 |
| n_burst | 输入 | 待搬运的连续传输数据块个数。 |
| len_burst | 输入 | 待搬运的每个连续传输数据块的长度，单位为DataBlock（32字节）。 |
| src_gap | 输入 | 源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。 |
| dst_gap | 输入 | 目的操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。 |

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- dst_addr、src_addr的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- 由于此接口实现了Cube核与Vector核间的数据搬运，所以如果核函数调用了该接口，则不能使用__cube__或__vector__的执行空间修饰符，否则会导致执行失败。

## 调用示例

```cpp
constexpr uint16_t n_burst = 1;
constexpr uint16_t len_burst = 1;
constexpr uint16_t src_gap = 0;
constexpr uint16_t dst_gap = 1;
__cbuf__ void src[256];
__ubuf__ void dst[256];
asc_copy_l12ub(dst, src, false, n_burst, len_burst, src_gap, dst_gap);
```
