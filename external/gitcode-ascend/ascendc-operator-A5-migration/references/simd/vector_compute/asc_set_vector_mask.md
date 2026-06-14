> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_set_vector_mask.md
# asc_set_vector_mask

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

某些矢量计算接口需要提前设置Mask，提示哪些数据需要参与计算。
该API用于设置Mask，使用前需要先调用[asc_set_mask_count](asc_set_mask_count.md)和[asc_set_mask_norm](asc_set_mask_norm.md)设置Mask模式。在不同的模式下Mask的含义如下：
- Normal模式用于高维切分计算类API。该模式下，Mask参数用来控制单次迭代内参与计算的元素。可以按位控制哪些元素参与计算：bit位的值为1表示参与计算，0表示不参与。分为mask1(高位Mask)和mask0（低位Mask）。参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask0，mask1∈[0, 2^64-1]，并且不同时为0；当操作数为32位时，mask1为0，mask0∈(0, 2^64-1]；当操作数为64位时，mask1为0，mask0∈(0, 2^32-1]。
- Counter模式用于前n个数据计算类API。该模式下，Mask参数用于控制整个计算过程中参与计算的元素个数。前n个数据计算类接口中已经实现了Normal模式和Counter模式的转换，用户不需要进行自行设置。

## 函数原型

```cpp
__aicore__ inline void asc_set_vector_mask(uint64_t mask1, uint64_t mask0)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|mask1|输入|Normal模式：高位mask值。<br>Counter模式：需要置0，本入参不生效。|
|mask0|输入|Normal模式：低位mask值。<br>Counter模式：整个矢量计算过程中，参与计算的元素个数。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

需先调用[asc_set_mask_count](asc_set_mask_count.md)或[asc_set_mask_norm](asc_set_mask_norm.md)设置Mask模式。

## 调用示例

请参考[asc_set_mask_count](asc_set_mask_count.md#调用示例)或[asc_set_mask_norm](asc_set_mask_norm.md#调用示例)。