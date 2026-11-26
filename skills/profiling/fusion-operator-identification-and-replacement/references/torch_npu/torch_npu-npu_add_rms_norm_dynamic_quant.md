# torch_npu.npu_add_rms_norm_dynamic_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- **API功能**：

    RMSNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicQuant算子则是为输入张量进行对称动态量化的算子。AddRmsNormDynamicQuant算子将RMSNorm前的Add算子和RMSNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。

- **计算公式**：

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RMSNorm}(x)=\frac{x}{\operatorname{RMS}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  input1 =\begin{cases}
    y\cdot smoothScale1Optional & \ \ smoothScale1Optional \\
    y & !\ smoothScale1Optional
    \end{cases}
  $$

  $$
  input2 =\begin{cases}
    y\cdot smoothScale2Optional & \ \ smoothScale2Optional \\
    y & !\ smoothScale2Optional
    \end{cases}
  $$

  $$
  scale1Out=\begin{cases}
    row\_max(abs(input1))/127 & outputMask[0]=True\ ||\ !outputMask \\
    无效输出 & outputMask[0]=False
    \end{cases}
  $$

  $$
  y1Out=\begin{cases}
    round(input1/scale1Out) & outputMask[0]=True\ ||\ !outputMask \\
    无效输出 & outputMask[0]=False
    \end{cases}
  $$

$$
  scale2Out=\begin{cases}
    row\_max(abs(input2))/127 & outputMask[1]=True\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ smoothScale2Optional) \\
    无效输出 & outputMask[1]=False\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ !smoothScale2Optional)
    \end{cases}
$$

$$
  y2Out=\begin{cases}
    round(input2/scale2Out) & outputMask[1]=True\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ smoothScale2Optional)\\
    无效输出 & outputMask[1]=False\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ !smoothScale2Optional)
    \end{cases}
$$

  公式中的row\_max代表每行求最大值。

## 函数原型

```python
torch_npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, *, smooth_scale1=None, smooth_scale2=None, beta=None, epsilon=1e-6, output_mask=None, y_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **x1**(`Tensor`)：必选参数，表示用于Add计算的第一个输入，对应公式中的$x1$。数据格式支持$ND$，不支持空Tensor，支持非连续Tensor。数据类型支持`torch.float16`、`torch.bfloat16`。支持2-8维张量。
- **x2**(`Tensor`)：必选参数，表示用于Add计算的第二个输入，对应公式中的$x2$。数据格式支持$ND$，不支持空Tensor，支持非连续Tensor。数据类型支持`torch.float16`、`torch.bfloat16`。shape与`x1`保持一致。
- **gamma**(`Tensor`)：必选参数，表示RMSNorm的缩放因子（权重），对应公式中的$gamma$。数据格式支持$ND$，不支持空Tensor，支持非连续Tensor。数据类型与`x1`保持一致。shape为一维，元素数量与`x1`最后一维大小一致。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **smooth_scale1**(`Tensor`)：可选参数，第一路量化的smooth缩放因子，对应公式中的$smoothScale1Optional$。数据类型与`x1`保持一致。shape为一维，元素数量与`x1`最后一维大小一致。默认值为`None`，为`None`时量化分支不进行smooth操作。
- **smooth_scale2**(`Tensor`)：可选参数，第二路量化的smooth缩放因子，对应公式中的$smoothScale2Optional$。数据类型与`x1`保持一致。shape为一维，元素数量与`x1`最后一维大小一致。默认值为`None`，为`None`时量化分支不进行smooth操作。
- **beta**(`Tensor`)：可选参数，RMSNorm的偏置项，对应公式中的$beta$。数据类型与`x1`保持一致。shape为一维，元素数量与`x1`最后一维大小一致。默认值为`None`，为`None`时不添加偏置。
- **epsilon**(`float`)：可选参数，表示添加到分母中的值，以确保数值稳定，对应公式中的$epsilon$。默认值为`1e-6`。
- **output_mask**(`bool[2]`)：可选参数，长度为2的布尔数组，用于控制是否计算两路量化输出，对应公式中的$outputMask$。output_mask[0]控制第一路量化输出（y1, scale1），output_mask[1]控制第二路量化输出（y2, scale2）。
- **y_dtype**(`int`)：可选参数，y1和y2的量化输出数据类型。支持None或`torch.int8`、`torch.quint4x2`或`torch_npu.int4`、`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`，`torch.quint4x2`或`torch_npu.int4`场景下`x1`最后一维必须能被8整除。默认值为None。

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：支持None或`torch.int8`、`torch.quint4x2`或`torch_npu.int4`。
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持None或`torch.int8`、`torch.quint4x2`或`torch_npu.int4`、`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。
  <!-- end id5 -->

## 返回值说明

- **y1**(`Tensor`)：第一路动态量化后的输出Tensor，对应公式中的$y1Out$。当`output_mask`为None或output_mask[0]为True时有效输出，数据类型支持`torch.int8`、`torch_npu.int4`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。数据类型由参数`y_dtype`指定，如果参数`y_dtype`表示`torch_npu.int4`，那么输出的dtype是`torch.int32`，shape的最后一维是输入`x1`最后一维的1/8，其他维度和输入`x1`一致；如果参数`y_dtype`表示`torch_npu.hifloat8`，那么输出的dtype是`torch.uint8`，shape与输入`x1`一致；如果参数`y_dtype`不为`torch_npu.int4`或`torch_npu.hifloat8`，那么输出dtype与参数`y_dtype`表示的数据类型一致，shape与输入`x1`一致。当output_mask[0]为False时则不输出，返回空Tensor。

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.int8`、`torch_npu.int4`。
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch_npu.int4`、`torch.hifloat8`、`torch.float_e5m2`、`torch.float8_e4m3fn`。
  <!-- end id7 -->

- **y2**(`Tensor`)：第二路动态量化后的输出Tensor，对应公式中的$y2Out$。当`output_mask`为None并且`smooth_scale2`不为None，或者output_mask[1]为True时有效输出，数据类型和shape与`y1`的有效输出一致。当output_mask[1]为False时则不输出，返回空Tensor。

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.int8`、`torch.int4`。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch_npu.int4`、`torch_npu.hifloat8`、`torch.float_e5m2`、`torch.float8_e4m3fn`。
  <!-- end id9 -->

- **x_out**(`Tensor`)：Add计算的结果，对应公式中的$x$。数据类型和shape与输入`x1`保持一致。
- **scale1**(`Tensor`)：第一路动态量化的缩放系数，对应公式中的$scale1Out$。有效输出的条件与`y1`一致，数据类型为`torch.float32`，shape为`x1`的shape剔除最后一维；当output_mask[0]为False时则不输出，返回空Tensor。
- **scale2**(`Tensor`)：第二路动态量化的缩放系数，对应公式中的$scale2Out$。有效输出的条件与`y2`一致，数据类型为`torch.float32`，shape为`x1`的shape剔除最后一维；当output_mask[1]为False时则不输出，返回空Tensor。

## 约束说明

- 当`output_mask`不为空时，参数`smooth_scale1`有值时，则`output_mask[0]`必须为True。参数`smooth_scale2`有值时，则`output_mask[1]`必须为True。
- 当`output_mask`不为空时，`output_mask[0]`与`output_mask[1]`不能同时为False。
- 当`output_mask`为空时，不能只有`smooth_scale2`存在。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    x1 = torch.randn(2, 3, 32, dtype=torch.float16, device='npu')
    x2 = torch.randn(2, 3, 32, dtype=torch.float16, device='npu')
    gamma = torch.ones(32, dtype=torch.float16, device='npu')
    beta = torch.zeros(32, dtype=torch.float16, device='npu')
    smooth_scale1 = torch.ones(32, dtype=torch.float16, device='npu')
    smooth_scale2 = torch.ones(32, dtype=torch.float16, device='npu')
    epsilon = 1e-6

    y1, y2, x_out, scale1, scale2 = torch_npu.npu_add_rms_norm_dynamic_quant(
        x1, x2, gamma,
        smooth_scale1=smooth_scale1,
        smooth_scale2=smooth_scale2,
        beta=beta,
        epsilon=epsilon,
        output_mask=[True, True],
    )

    print("y1:", y1)
    print("y1.shape:", y1.shape)
    print("y1.dtype:", y1.dtype)
    print("y2:", y2)
    print("y2.shape:", y2.shape)
    print("y2.dtype:", y2.dtype)
    print("x_out:", x_out)
    print("x_out.shape:", x_out.shape)
    print("x_out.dtype:", x_out.dtype)
    print("scale1:", scale1)
    print("scale1.shape:", scale1.shape)
    print("scale2:", scale2)
    print("scale2.shape:", scale2.shape)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, gamma, smooth1, smooth2, beta, epsilon, output_mask):
            return torch_npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, smooth_scale1=smooth1, smooth_scale2=smooth2, beta=beta, epsilon=epsilon, output_mask=output_mask)

    x1 = torch.randn(2, 3, 32, dtype=torch.float16, device='npu')
    x2 = torch.randn(2, 3, 32, dtype=torch.float16, device='npu')
    gamma = torch.ones(32, dtype=torch.float16, device='npu')
    beta = torch.zeros(32, dtype=torch.float16, device='npu')
    smooth_scale1 = torch.ones(32, dtype=torch.float16, device='npu')
    smooth_scale2 = torch.ones(32, dtype=torch.float16, device='npu')
    epsilon = 1e-6
    output_mask = [True, True]

    model = NetModel()
    model.to('npu')
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
    y1, y2, x_out, scale1, scale2 = model(x1, x2, gamma, smooth_scale1, smooth_scale2, beta, epsilon, output_mask)
    y1 = y1.cpu()
    y2 = y2.cpu()
    x_out = x_out.cpu()
    scale1 = scale1.cpu()
    scale2 = scale2.cpu()
    print("y1:", y1)
    print("y1.shape:", y1.shape)
    print("y1.dtype:", y1.dtype)
    print("y2:", y2)
    print("y2.shape:", y2.shape)
    print("y2.dtype:", y2.dtype)
    print("x_out:", x_out)
    print("x_out.shape:", x_out.shape)
    print("x_out.dtype:", x_out.dtype)
    print("scale1:", scale1)
    print("scale1.shape:", scale1.shape)
    print("scale2:", scale2)
    print("scale2.shape:", scale2.shape)
    ```
