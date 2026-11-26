# （beta）torch\_npu.npu\_bert\_apply\_adam

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

针对BERT模型，获取Adam优化器的计算结果。

## 函数原型

```python
torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size=None, adam_mode=0, *, out=(var, m, v)) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **lr** (`Scalar`)：必选参数，数据类型与`var`相同。
- **beta1** (`Scalar`)：必选参数，数据类型与`var`相同。
- **beta2** (`Scalar`)：必选参数，数据类型与`var`相同。
- **epsilon** (`Scalar`)：必选参数，数据类型与`var`相同。
- **grad** (`Tensor`)：必选参数，数据类型和shape与`var`相同。
- **max\_grad\_norm** (`Scalar`)：必选参数，数据类型与`var`相同。
- **global\_grad\_norm** (`Scalar`)：必选参数，数据类型与`var`相同。
- **weight\_decay**(`Scalar`)：必选参数，数据类型与`var`相同。
- **step\_size** (`Scalar`)：可选参数，默认值为None，数据类型与`var`相同。
- **adam\_mode** (`int`)：可选参数，默认值为0，选择adam模式。0表示“adam”，1表示“mbert\_adam”。
- **out** (`tuple[Tensor, Tensor, Tensor]`)：必选关键字参数，包含`var`、`m`、`v`三个输出张量的元组。其中`var`为`torch.float16`或`torch.float32`类型张量；`m`、`v`的数据类型和shape与`var`相同。

## 返回值说明

`tuple[Tensor, Tensor, Tensor]`

返回包含`var`、`m`、`v`三个Tensor的元组，分别为`out`中传入的`var`、`m`、`v`输出张量。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> var_in = torch.rand(321538).uniform_(-32., 21.).npu()
>>> m_in = torch.zeros(321538).npu()
>>> v_in = torch.zeros(321538).npu()
>>> grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
>>> max_grad_norm = -1.
>>> beta1 = 0.9
>>> beta2 = 0.99
>>> weight_decay = 0.
>>> lr = 0.
>>> epsilon = 1e-06
>>> global_grad_norm = 0.
>>> var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out=(var_in, m_in, v_in))
>>> print(var_out)
tensor([ 14.7733, -30.1218,  -1.3647,  ..., -16.6840,   7.1518,   8.4872],
      device='npu:0')
```
