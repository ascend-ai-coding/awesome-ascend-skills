# torch_npu.npu_gelu

> [!NOTICE]  
> 在NPU环境下使用PyTorch原生接口gelu时，approximate参数不起作用且默认为tanh，如果需要将approximate的默认值设定为None，或者需要区分不同的approximate方法，则需使用此接口。

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
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

- API功能：计算高斯误差线性单元（GELU）激活函数。
- 计算公式：

    Gaussian Error Linear Unit(GELU)的表达式为：
    $$
    \text{GELU}(x) = x \cdot \Phi(x)
    $$

    $\Phi(x)$是Gaussian Distribution的CDF(Cumulative Distribution Function)，表达式为：
    $$
    \Phi(x) = P(X \leq x) = \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \right]
    $$
    当`approximate="none"`时，使用上述公式；若采用`approximate="tanh"`，则将 $\text{erf}(\cdot)$ 替换为 $\tanh(\cdot)$ 的近似形式。

## 函数原型

```python
torch_npu.npu_gelu(self, *, approximate='none') -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，对应公式中的$x$，待进行`npu_gelu`计算的入参，数据格式支持$ND$，支持非连续的Tensor，支持空Tensor。输入最大支持8维。

    <!-- npu="910,310p" id5 -->
    - <term>Atlas训练系列产品</term>、<term>Atlas推理系列产品</term>：数据类型支持`torch.float16`、`torch.float32`。
    <!-- end id5 -->
    <!-- npu="950,910b" id6 -->
    - <term>Atlas A2系列产品</term>、<term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。
    <!-- end id6 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **approximate** (`str`)：可选参数，字符串类型，计算使用的激活函数模式，可配置为`none`或者`tanh`。其中`none`代表使用erf模式，`tanh`代表使用tanh模式。

## 返回值说明

`Tensor`

数据类型必须和`self`一样，数据格式支持$ND$，shape必须和`self`一样，支持非连续的Tensor。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式。
- `self`输入不能为None。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> input_tensor = torch.randn(100, 200)
    >>> output_tensor = torch_npu.npu_gelu(input_tensor.npu(), approximate='tanh')
    >>>
    >>> print(output_tensor)
    tensor([[ 0.5795, -0.0274, -0.1477,  ...,  0.2422, -0.0843, -0.1154],
            [-0.0385,  0.8736,  0.1809,  ..., -0.0676,  0.2404,  0.4038],
            [ 0.0438,  0.0205, -0.1536,  ...,  0.2910,  1.1553,  0.3319],
            ...,
            [-0.1698, -0.0031,  0.5120,  ..., -0.1390, -0.0082,  0.6286],
            [ 0.1980,  0.0535, -0.1685,  ..., -0.1528, -0.1484,  1.0703],
            [-0.1351,  1.5851, -0.0222,  ..., -0.0230,  1.4319, -0.1700]],
        device='npu:0')
    >>> print(output_tensor.shape)
    torch.Size([100, 200])
    >>> print(output_tensor.dtype)
    torch.float32
    ```

- 图模式调用

    ```python
    # 入图方式
    import os
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    torch_npu.npu.set_compile_mode(jit_compile=True)
    
    class Net(torch.nn.Module):
    
        def __init__(self):
            super().__init__()
        def forward(self, self_0, approximate):
            out = torch_npu.npu_gelu(self_0, approximate=approximate)
            return out
    
    x = torch.randn(100, 10, 20).npu()
    model = Net()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    npu_out = model(x, approximate="none")    
    print(npu_out.shape, npu_out.dtype)

    # 执行上述代码的输出类似如下
    torch.Size([100, 10, 20]) torch.float32
    ```
