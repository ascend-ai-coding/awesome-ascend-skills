# torch_npu.npu_add_rms_norm_cast

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：RmsNorm算子是大模型常用的归一化操作，AddRmsNormCast算子将AddRmsNorm后的Cast算子融合起来，减少搬入搬出操作。

- **计算公式**：

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  y1=\operatorname{RMSNorm}(x_i)=\frac{x_i}{\operatorname{RMS}(\mathbf{x})} gamma_i, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  y2=cast(y1)
  $$

## 函数原型

```python
torch_npu.npu_add_rms_norm_cast(x1, x2, gamma, epsilon=1e-06) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **x1**（`Tensor`）：必选参数，表示需要归一化的原始数据输入，公式中的$x1$。数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。数据类型支持`torch.float16`、`torch.bfloat16`。shape支持1-8维。
- **x2**（`Tensor`）：必选参数，表示需要归一化的原始数据输入，公式中的$x2$。数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。shape和数据类型需要与`x1`一致。
- **gamma**（`Tensor`）：必选参数，表示标准化过程中的权重张量，公式中的$gamma$。数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。数据类型需要与`x1`一致。shape支持1-8维，shape与`x1`需要norm的维度一致。
- **epsilon**（`float`）：可选参数，公式中的输入$epsilon$，用于防止除0错误，数据类型为`torch.double`。建议传较小的正数，默认值为1e-6。

## 返回值说明

- **y1**（`Tensor`）：表示归一化后经过类型转换的输出Tensor，公式中的$y1$。数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。数据类型支持`torch.float32`，shape与输入`x1`一致。
- **y2**（`Tensor`）：表示归一化后的输出Tensor，公式中的$y2$。数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。shape和数据类型需要与计算输入`x1`保持一致。
- **rstd**（`Tensor`）：表示x的标准差，对应公式中的$RMS(x)$。数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。数据类型支持`torch.float32`，维度数与`x1`保持一致，不需要norm的维度与`x1`对应维度保持一致，需要norm的维度为1。当输入`x1`为空Tensor时，`rstd`也必须为空Tensor。`rstd`shape与`x1`shape，`gamma`shape关系举例：
  - 若`x1`shape:(2,3,4,8)，`gamma`shape:(8)，`rstd`shape:(2,3,4,1)；
  - 若`x1`shape:(2,3,4,8)，`gamma`shape:(4,8)，`rstd`shape:(2,3,1,1)。
- **x**（`Tensor`）：表示`x1`和`x2`的和，公式中的$x$。数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。shape和数据类型需要与计算输入`x1`一致。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 维度的边界说明：
  参数`x1`、`x2`、`gamma`、`y1`、`y2`、 `rstd`、 `x`的shape中每一维大小都不大于`torch.int32`的最大值2147483647。
- 边界值场景说明：
  - 当输入是Inf时， 输出为Inf。
  - 当输入是NaN，输出为NaN。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    input_x1 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
    input_x2 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
    input_gamma = torch.randn([64], dtype=torch.float16).npu()

    y1, y2, rstd, x = torch_npu.npu_add_rms_norm_cast(input_x1, input_x2, input_gamma)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    class EagerMode(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_x1, input_x2, input_gamma):
            y1, y2, rstd, x = torch_npu.npu_add_rms_norm_cast(input_x1, input_x2, input_gamma)
            return y1, y2, rstd, x

    def dynamo_mode_api(input_x1, input_x2, input_gamma):
        model = EagerMode().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        output = model(input_x1, input_x2, input_gamma)
        return output
    input_x1 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
    input_x2 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
    input_gamma = torch.randn([64], dtype=torch.float16).npu()
    outputs = dynamo_mode_api(input_x1, input_x2, input_gamma)
    ```
