# （beta）torch_npu.npu_fused_attention_score

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议使用torch_npu.npu_fusion_attention。

## 产品支持情况

<!-- npu="310p" id1 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910" id2 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id2 -->

## 功能说明

实现“Transformer attention score”的融合计算逻辑，主要将matmul、transpose、add、softmax、dropout、batchmatmul、permute等计算进行了融合。

## 函数原型

```python
torch_npu.npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose=False, key_transpose=False, bmm_score_transpose_a=False, bmm_score_transpose_b=False, value_transpose=False, dx_transpose=False) -> Tensor
```

## 参数说明

- **query_layer**（`Tensor`）：必选参数，仅支持`torch.float16`。不考虑转置参数的情况下，形状为`[batch_size, num_heads, q_seq_len, head_dim]`。
- **key_layer**（`Tensor`）：必选参数，仅支持`torch.float16`。不考虑转置参数的情况下，形状为`[batch_size, num_heads, k_seq_len, head_dim]`。
- **value_layer**（`Tensor`）：必选参数，仅支持`torch.float16`。不考虑转置参数的情况下，形状为`[batch_size, num_heads, v_seq_len, head_dim]`。
- **attention_mask**（`Tensor`）：必选参数，仅支持`torch.float16`。不考虑转置参数的情况下，形状为`[batch_size, num_heads, q_seq_len, k_seq_len]`。
- **scale**（`Scalar`）：必选参数，缩放系数，浮点数标量。
- **keep_prob**（`float`）：必选参数，不做dropout的概率，0-1之间，浮点数。
- **query_transpose**（`bool`）：可选参数，query是否做转置，默认值为False。
- **key_transpose**（`bool`）：可选参数，key是否做转置，默认值为False。
- **bmm_score_transpose_a**（`bool`）：可选参数，bmm计算中左矩阵是否做转置，默认值为False。
- **bmm_score_transpose_b**（`bool`）：可选参数，bmm计算中右矩阵是否做转置，默认值为False。
- **value_transpose**（`bool`）：可选参数，value是否做转置，默认值为False。
- **dx_transpose**（`bool`）：可选参数，反向计算时dx是否做转置，默认值为False。

## 返回值说明

`Tensor`

注意力计算后的输出张量，数据类型为`torch.float16`。在默认参数下，输出张量的形状与query_layer相同。

## 约束说明

输入tensor的格式编号必须均为29，数据类型为`torch.float16`。query_layer、key_layer以及value_layer的num_heads维度必须相同，query_layer和key_layer的head_dim维度必须相同，用于计算注意力分数。value_layer的head_dim可以不同，决定了输出特征维度。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> query_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> key_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> value_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> attention_mask = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 512).npu(), 29).half()
>>> scale = 0.125
>>> keep_prob = 0.5
>>> context_layer = torch_npu.npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob)
>>> print(context_layer)
tensor([[0.5010, 0.4709, 0.4841,  ..., 0.4321, 0.4448, 0.4834],
        [0.5107, 0.5049, 0.5239,  ..., 0.4436, 0.4375, 0.4651],
        [0.5308, 0.4944, 0.5005,  ..., 0.5010, 0.5103, 0.5303],
        ...,
        [0.5142, 0.5068, 0.5176,  ..., 0.5498, 0.4868, 0.4805],
        [0.4941, 0.4731, 0.4863,  ..., 0.5161, 0.5239, 0.5190],
        [0.5459, 0.5107, 0.5415,  ..., 0.4641, 0.4688, 0.4531]],
       device='npu:0', dtype=torch.float16)
```
