> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_bitsort.md
# asc_bitsort

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √ </cann-filter>|

## 功能说明

位图排序函数，一次迭代可以完成32个元素的排序，数据按如下描述结构进行保存：

Score和Index分别存储在src0和src1中，按Score进行排序（Score大的元素排前面），排序后的Score与其对应的Index一起以（Score，Index）的结构存储在dst中。

## 函数原型

- 连续计算
  ```cpp
  __aicore__ inline void asc_bitsort(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ uint32_t* src1, int32_t repeat)
  __aicore__ inline void asc_bitsort(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ uint32_t* src1, int32_t repeat)
  ```

- 同步计算
  ```cpp
  __aicore__ inline void asc_bitsort_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ uint32_t* src1, int32_t repeat)
  __aicore__ inline void asc_bitsort_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ uint32_t* src1, int32_t repeat)
  ```

## 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| dst       | 输出    | 目的操作数（矢量）的起始地址。 |
| src0      | 输入    | 源操作数（矢量）的起始地址。 |
| src1      | 输入    | 源操作数（矢量）的起始地址。 |
| repeat    | 输入    | 重复迭代次数，每次迭代完成32个元素的排序，下次迭代src0和src1个跳过32个元素，dst跳过256 Byte空间。取值范围：[0, 255]。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
__ubuf__ half dst[256];
__ubuf__ half src0[256];
__ubuf__ uint32_t src1[256];
asc_bitsort(dst, src0, src1, 8);
```
