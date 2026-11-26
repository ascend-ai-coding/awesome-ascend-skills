# torch_npu.npu_sim_exponential_

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：根据参数`lambd`生成指数分布随机数，并原地填充至输入张量`input`。
- 计算公式：
    $$f(x) = -1/λ * ln(1-u), u ~ Uniform(0, 1]$$

## 函数原型

```python
torch_npu.npu_sim_exponential_(input, lambd=1, *, generator=None) -> Tensor
```

## 参数说明

- **input**(`Tensor`)：必选参数，源数据张量，公式中的$f(x)$。要求为连续的Tensor，数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`，数据格式支持$ND$，shape支持0~8维。

- **lambd**(`double`)：可选参数，指数分布的参数，公式中的$λ$，可配置为任意正实数，默认值为1。

- **generator**(`Generator`)：可选参数，用于生成seed和offset，供aclnnSimThreadExponential算子使用，默认为None。

## 返回值说明

`Tensor`

表示公式中的$f(x)$，即原地更新后的`input`张量。

## 调用示例

```python
>>> import torch
>>> import torch_npu

>>> shape = [100, 400]
>>> gen = torch.Generator(device="npu")
>>> gen.manual_seed(0)
>>> input = torch.zeros(shape, dtype=torch.float32).npu()
>>> torch_npu.npu_sim_exponential_(input, lambd=1, generator=gen)

```
