> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_set_mask_count.md
# asc_set_mask_count

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

设置Mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入数据计算数量，实际迭代次数由Vector计算单元自动推断。在前n个数据计算类的接口内部会自动设置该模式，不需要自行设置。

## 函数原型

```cpp
__aicore__ inline void asc_set_mask_count()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

该API需和[asc_set_vector_mask](asc_set_vector_mask.md)配合使用。

## 调用示例

```cpp
asc_set_mask_count();
asc_set_vector_mask(0, static_cast<uint64_t>(64)); // 设置前64个元素参与计算
... // 计算操作
asc_set_mask_norm();    // 恢复为Normal模式。此步骤可根据需求，按需设置。
```