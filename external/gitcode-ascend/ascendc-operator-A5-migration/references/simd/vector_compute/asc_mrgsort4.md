> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_mrgsort4.md
# asc_mrgsort4

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √ </cann-filter>|

## 功能说明

将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序。

asc_mrgsort4处理的数据一般是经过asc_bitsort指令处理后的数据，也就是asc_bitsort指令的输出，队列的结构Region Proposal如下所示：
- 数据类型为float，每个结构占据8 Bytes。

![ ](../figures/mrgsort4_01.png)

- 数据类型为half，每个结构也占据8 Bytes，中间有2 Bytes保留。

![ ](../figures/mrgsort4_02.png)

## 函数原型

- 计算

    ```cpp
    __aicore__ inline void asc_mrgsort4(__ubuf__ half* dst, __ubuf__ half* src[ASC_C_API_MRGSORT_ELEMENT_LEN], uint8_t repeat, uint16_t element_length_0, uint16_t element_length_1, uint16_t element_length_2, uint16_t element_length_3, bool if_exhausted_suspension, uint8_t valid_bit)
    __aicore__ inline void asc_mrgsort4(__ubuf__ float* dst, __ubuf__ float* src[ASC_C_API_MRGSORT_ELEMENT_LEN], uint8_t repeat, uint16_t element_length_0, uint16_t element_length_1, uint16_t element_length_2, uint16_t element_length_3, bool if_exhausted_suspension, uint8_t valid_bit)
    ```

    <cann-filter npu_type = "950">

    ```cpp
    // 如下原型仅支持Ascend 950PR/Ascend 950DT
    __aicore__ inline void asc_mrgsort4(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeat, uint16_t element_length_0, uint16_t element_length_1, uint16_t element_length_2, uint16_t element_length_3, bool if_exhausted_suspension, uint8_t valid_bit)
    __aicore__ inline void asc_mrgsort4(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t element_length_0, uint16_t element_length_1, uint16_t element_length_2, uint16_t element_length_3, bool if_exhausted_suspension, uint8_t valid_bit)
    ```

    </cann-filter>

- 同步计算

    ```cpp
    __aicore__ inline void asc_mrgsort4_sync(__ubuf__ half* dst, __ubuf__ half* src[ASC_C_API_MRGSORT_ELEMENT_LEN], uint8_t repeat, uint16_t element_length_0, uint16_t element_length_1, uint16_t element_length_2, uint16_t element_length_3, bool if_exhausted_suspension, uint8_t valid_bit)
    __aicore__ inline void asc_mrgsort4_sync(__ubuf__ float* dst, __ubuf__ float* src[ASC_C_API_MRGSORT_ELEMENT_LEN], uint8_t repeat, uint16_t element_length_0, uint16_t element_length_1, uint16_t element_length_2, uint16_t element_length_3, bool if_exhausted_suspension, uint8_t valid_bit)
    ```

    <cann-filter npu_type = "950">

    ```cpp
    // 如下原型仅支持Ascend 950PR/Ascend 950DT
    __aicore__ inline void asc_mrgsort4_sync(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeat, uint16_t element_length_0, uint16_t element_length_1, uint16_t element_length_2, uint16_t element_length_3, bool if_exhausted_suspension, uint8_t valid_bit)
    __aicore__ inline void asc_mrgsort4_sync(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t element_length_0, uint16_t element_length_1, uint16_t element_length_2, uint16_t element_length_3, bool if_exhausted_suspension, uint8_t valid_bit)
    ```
    
    </cann-filter>

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址的数组，待合并的Region Proposal队列。队列数量ASC_C_API_MRGSORT_ELEMENT_LEN值为4。|
| repeat | 输入 | 迭代次数，每一次源操作数和目的操作数计算并跳过四个队列总长度。 |
| element_length_0 | 输入 | 源队列0的长度。 |
| element_length_1 | 输入 | 源队列1的长度。 |
| element_length_2 | 输入 | 源队列2的长度。 |
| element_length_3 | 输入 | 源队列3的长度。 |
| if_exhausted_suspension | 输入 | 某条队列耗尽后，指令是否需要停止。 |
| valid_bit | 输入 | 有效队列个数。有效取值如下：<br>3：前两条队列有效。<br>7：前三条队列有效。<br>15：四条队列全部有效。  |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 当 repeat=1 时，表示进行一次迭代，可自由设置 element_length_..., if_exhausted_suspension, valid_bit参数。
- 当 repeat>1 时，表示进行多次迭代，必须满足以下条件：包含四条源队列，四个源队列的长度一致且在内存中连续存储，valid_bit的值为15，if_exhausted_suspension的值为false。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
// 示例对8个已排序好的Region Proposal队列进行归并排序，每个队列包含32个元素。
// 第一步，每次排序4个32元素队列src，共repeat两次，得到两个排序好的128元素的新队列tmp。
// 第二步，将两个排序好的128元素新队列tmp再进行归并排序，得到最终的256元素队列dst。

constexpr uint16_t element_length = 32;
// 每个元素包含float32的score和4字节的index，占8字节，相当于2个sizeof(float)。
constexpr uint16_t array_length = element_length * 2;
__ubuf__ float src_workspace[array_length * 8];
__ubuf__ float tmp_workspace[array_length * 8];
__ubuf__ float dst_workspace[array_length * 8];

__ubuf__ float* src[ASC_C_API_MRGSORT_ELEMENT_LEN];
__ubuf__ float* dst;
uint8_t valid_bit;

// 每个迭代可以处理4个队列，共2个迭代。
src[0] = &src_workspace[0];
src[1] = &src_workspace[1 * array_length];
src[2] = &src_workspace[2 * array_length];
src[3] = &src_workspace[3 * array_length];
dst = tmp_workspace;
valid_bit = 15;
asc_mrgsort4(dst, src, 2, element_length, element_length, element_length, element_length, false, valid_bit);

// 迭代1次，处理2个队列。
src[0] = &tmp_workspace[0];
src[1] = &tmp_workspace[1 * 4 * array_length];
dst = dst_workspace;
valid_bit = 3;
asc_mrgsort4(dst, src, 1, 4 * element_length, 4 * element_length, 0, 0, false, valid_bit);
```