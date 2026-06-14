> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_get_vms4_sr.md
# asc_get_vms4_sr

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

此接口用于获取执行[asc_mrgsort4](./asc_mrgsort4.md)操作后的4个队列中，每个队列已经理过的Region Proposal个数。使用时，需要将asc_mrgsort4中的if_exhausted_suspension参数配置为true，该配置模式下某条队列耗尽后，MrgSort4指令停止。

## 函数原型

```cpp
__aicore__ inline void asc_get_vms4_sr(uint16_t sorted_num[ASC_C_API_MRGSORT_ELEMENT_LEN])
```

## 参数说明


|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
| sorted_num | 输出 | 每个队列已经理过的Region Proposal个数。队列数量ASC_C_API_MRGSORT_ELEMENT_LEN值为4。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 需和[asc_mrgsort4](./asc_mrgsort4.md)操作配合使用，且需asc_mrgsort4中的if_exhausted_suspension参数配置为true。
- 需通过同步操作确保[asc_mrgsort4](./asc_mrgsort4.md)执行完成后再调用本接口获取结果。

## 调用示例

```cpp
// 示例对4个已排序好的Region Proposal队列进行归并排序，每个队列包含32个元素。
constexpr uint16_t element_length = 32;
constexpr uint16_t array_length = element_length * 2;
// 每个元素包含float32的score和4字节的index，占8字节，相当于2个sizeof(float)。
__ubuf__ float src_workspace[array_length * 4];

__ubuf__ float* src[ASC_C_API_MRGSORT_ELEMENT_LEN];
__ubuf__ float dst[array_length * 4];

uint8_t repeat = 1;
uint8_t valid_bit = 0b1111;
bool if_exhausted_suspension = true;
src[0] = &src_workspace[0];
src[1] = &src_workspace[1 * array_length];
src[2] = &src_workspace[2 * array_length];
src[3] = &src_workspace[3 * array_length];

asc_mrgsort4(dst, src, repeat, element_length, element_length, element_length, element_length, if_exhausted_suspension, valid_bit);
asc_sync();

uint16_t sorted_num[ASC_C_API_MRGSORT_ELEMENT_LEN];
asc_get_vms4_sr(sorted_num);
```