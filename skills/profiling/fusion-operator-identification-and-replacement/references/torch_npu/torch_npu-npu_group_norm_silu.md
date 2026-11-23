# torch_npu.npu_group_norm_silu

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

## 功能说明

- API功能：对输入张量`input`依次执行GroupNorm和SiLU激活，返回三个张量：out（SiLU激活后的输出）、meanOut（归一化均值）、rstdOut（归一化标准差的倒数）。
- 计算公式：
    - GroupNorm：$x$为输入`input`，公式中$\gamma$和$\beta$分别对应可选参数`weight`和`bias`，当`weight`、`bias`均为空时不进行仿射变换（等价于$\gamma = 1$、$\beta = 0$），$E[x] = \bar{x}$代表$x$的均值，$ Var[x]=\frac{1}{n}\sum_{i=1}^{n} (x_i - E[x])^2 $ 代表$x$的方差，则
    $$
    \begin{cases}
    \text{groupnormOut} = \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    \text{meanOut}  = E[x] \\
    \text{rstdOut}  = \frac{1}{\sqrt{Var[x] + eps}}
    \end{cases}
    $$
    - Silu：
    $$
    \text{out} = \frac{\text{groupnormOut}}{1 + e^{-\text{groupnormOut}}}
    $$

    其中，$n$表示单个样本的单个group中参与归一化的元素数量，均值和方差按group分别计算。

## 函数原型

```python
torch_npu.npu_group_norm_silu(input, weight, bias, group, eps=0.00001) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，源数据张量，维度需要为2~8维且第1维度能被`group`整除。数据格式支持$ND$，支持非连续的Tensor。

    <!-- npu="310p" id4 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`、`torch.float32`。
    <!-- end id4 -->
    <!-- npu="910b" id5 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id6 -->

- **weight** (`Tensor`)：可选参数，缩放张量，维度为1且元素数量需与输入`input`的第1维度保持相同，数据格式支持$ND$，支持非连续的Tensor。

    <!-- npu="310p" id7 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`、`torch.float32`。
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id9 -->

- **bias** (`Tensor`)：可选参数，偏移张量，维度为1且元素数量需与输入`input`的第1维度保持相同，数据格式支持$ND$，支持非连续的Tensor。

    <!-- npu="310p" id10 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`、`torch.float32`。
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id12 -->

- **group** (`int`)：必选参数，表示将输入`input`的第1维度分为group组，group需大于0。数据类型支持`torch.int64`。

- **eps** (`float`)：可选参数，为保持数值稳定性而加到分母上的值，若保持精度，则eps需大于0。默认值为0.00001。数据类型支持`torch.float32`。

## 返回值说明

- **out** (`Tensor`)：数据类型和shape与`input`相同，支持$ND$，支持非连续的Tensor。

    <!-- npu="310p" id13 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`、`torch.float32`。
    <!-- end id13 -->
    <!-- npu="910b" id14 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id15 -->

- **meanOut** (`Tensor`)：数据类型与`input`相同，shape为\(N, group\)，其中N为`input`第0维度值。数据格式支持$ND$，支持非连续的Tensor。

    <!-- npu="310p" id16 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`、`torch.float32`。
    <!-- end id16 -->
    <!-- npu="910b" id17 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id17 -->
    <!-- npu="950" id18 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id18 -->

- **rstdOut** (`Tensor`)：数据类型与`input`相同，shape为\(N, group\)，其中N为`input`第0维度值。数据格式支持$ND$，支持非连续的Tensor。

    <!-- npu="310p" id19 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`、`torch.float32`。
    <!-- end id19 -->
    <!-- npu="910b" id20 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id20 -->
    <!-- npu="950" id21 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`。
    <!-- end id21 -->

## 约束说明

- 该接口支持推理、训练场景下使用。
- `input`、`weight`、`bias`、`out`、`meanOut`、`rstdOut`数据类型必须在支持的范围之内。
- `out`、`meanOut`、`rstdOut`的数据类型与`input`相同；`weight`、`bias`与`input`可以不同。
- `weight`与`bias`的数据类型必须保持一致，且数据类型的精度不能低于`input`的数据类型。
- `weight`与`bias`的维度需为1且元素数量需与输入`input`的第1维度保持相同。
- `input`维度需大于一维且小于等于八维，且`input`第1维度能被`group`整除。
- `input`任意维都需大于0。
- `out`的shape与`input`相同。
- `meanOut`与`rstdOut`的shape为\(N, group\)，其中N为`input`第0维度值。
- `eps`需大于0。
- `group`需大于0。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import numpy as np
    import torch_npu
        
    dtype = np.float32
    shape_x = [24,320,48,48]
    num_groups = 32
    shape_c = [320]
    eps = 0.00001
        
    input_npu=torch.randn(shape_x,dtype=torch.float32).npu()
    weight_npu=torch.randn(shape_c,dtype=torch.float32).npu()
    bias_npu=torch.randn(shape_c,dtype=torch.float32).npu()
    out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
        
        
    input_npu=torch.randn(shape_x,dtype=torch.bfloat16).npu()
    weight_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
    bias_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
    out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
        
    input_npu=torch.randn(shape_x,dtype=torch.float16).npu()
    weight_npu=torch.randn(shape_c,dtype=torch.float16).npu()
    bias_npu=torch.randn(shape_c,dtype=torch.float16).npu()
    out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
    ```

<!-- npu="950" id22 -->
- 图模式调用：仅适用于<term>Ascend 950PR&950DT系列产品</term>。

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, weight, bias, group, eps):
            return torch_npu.npu_group_norm_silu(input, weight, bias, group, eps)

    x = torch.randn(10, 1024, 4, 8, dtype=torch.float32).npu()
    weight = torch.randn(1024, dtype=torch.float32).npu()
    bias = torch.randn(1024, dtype=torch.float32).npu()
    model = Net().npu()
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    out, mean, rstd = model(x, weight, bias, group=4, eps=0.0001)
    ```
<!-- end id22 -->
