# （beta）torch_npu.npu_softmax_cross_entropy_with_logits

> [!NOTICE]
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

- API功能：计算softmax的交叉熵损失。

- 计算公式：

$$
     loss = -\sum_{i=1}^{N}y_i * log(softmax(x_i))
$$
其中，$x_i$对应输入的`features`，$y_i$对应输入的`labels`，$N$表示输入特征的长度。

## 函数原型

```python
torch_npu.npu_softmax_cross_entropy_with_logits(features, labels) -> Tensor
```

## 参数说明

- **features** (`Tensor`)：必选参数，输入特征，大小为`[1, batch_size * num_classes]`的矩阵。对应公式中的$x_i$。数据格式支持$ND$，支持非连续的Tensor。输入最大支持2维。支持空Tensor。数据类型支持float、`torch.float16`、`torch.bfloat16`。

  <!-- npu="910,310p" id5 -->
  - <term>Atlas训练系列产品</term>、<term>Atlas推理系列产品</term>：数据类型不支持`torch.bfloat16`。
  <!-- end id5 -->

- **labels** (`Tensor`)：必选参数，输入标签，shape和数据类型与`features`保持一致。对应公式中的$y_i$。数据格式支持$ND$，支持非连续的Tensor。输入最大支持2维。支持空Tensor。数据类型支持float、`torch.float16`、`torch.bfloat16`。

  <!-- npu="910,310p" id6 -->
  - <term>Atlas训练系列产品</term>、<term>Atlas推理系列产品</term>：数据类型不支持`torch.bfloat16`。
  <!-- end id6 -->

## 返回值说明

`Tensor`

对应公式中$loss$，表示softmax和cross entropy的交叉熵损失计算结果。

## 调用示例

```python
>>> import torch, torch_npu
>>> batch_size = 4
>>> num_classes = 12
>>> features = torch.rand(1, batch_size * num_classes).npu()
>>> labels = torch.rand(1, batch_size * num_classes).npu()
>>> output = torch_npu.npu_softmax_cross_entropy_with_logits(features, labels)
>>> print(output)
tensor([97.9450], device='npu:0')
```
