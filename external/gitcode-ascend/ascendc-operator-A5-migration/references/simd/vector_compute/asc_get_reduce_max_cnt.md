> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_get_reduce_max_cnt.md
# asc_get_reduce_max_cnt

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

此接口用于获取执行[asc_repeat_reduce_max](./asc_repeat_reduce_max.md)操作后的最大值，以及第一个最大值时的索引。

## 函数原型

```cpp
__aicore__ inline void asc_get_reduce_max_cnt(half& val, uint32_t& index)
__aicore__ inline void asc_get_reduce_max_cnt(float& val, uint32_t& index)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| val | 输出 | 最大值。 |
| index | 输出 | 第一个最大值时的索引。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 需和[asc_repeat_reduce_max](./asc_repeat_reduce_max.md)操作配合使用。
- 需通过同步操作确保[asc_repeat_reduce_max](./asc_repeat_reduce_max.md)执行完成后再调用本接口获取结果。

## 调用示例

```cpp
// 示例计算256个half的最大值和索引，分2次迭代完成，计算结果为2组value_index
constexpr uint32_t total_length = 256;
__ubuf__ half src[total_length];
constexpr uint32_t dst_length = 4;
__ubuf__ half dst[dst_length];

asc_repeat_reduce_max_value_index(dst, src, total_length);
asc_sync();

half maxValue;
uint32_t maxIndex;
asc_get_reduce_max_cnt(maxValue, maxIndex);
```

结果示例：

```
输入数据src：[1, 2, 3, ..., 128, 129, ..., 256]
maxValue: 256.0
maxIndex: 255
```