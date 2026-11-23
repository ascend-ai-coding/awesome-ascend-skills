# torch\_npu.npu\_gelu\_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：对张量进行GELU（Gaussian Error Linear Unit，高斯误差线性单元）激活操作，再对结果进行静态/动态量化。

- 计算公式：

  - **先计算Gelu得到geluOut**

     - approximate = "tanh"

       $$
       geluOut = 0.5 * self * (1 + Tanh(\sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
       $$

     - approximate = "none"

       $$
       geluOut = 0.5 * self * [1 + erf(self/\sqrt{2})]
       $$

  - **再对geluOut进行量化操作**

     - quant_mode = "static"

       $$
       y = round\_to\_dstType(geluOut * input\_scale + input\_offset, round\_mode)
       $$

     - quant_mode = "dynamic"

       $$
       geluOut = geluOut * input\_scale \\
       Max = max(abs(geluOut)) \\
       out\_scale = Max / maxValue \\
       y = round\_to\_dstType(geluOut / out\_scale, round\_mode)
       $$

  - maxValue：对应数据类型的最大值。

    | 数据类型 | maxValue |
    | :---: | :---: |
    | `torch.int8` | 127 |
    | `torch.float8_e4m3fn` | 448 |
    | `torch.float8_e5m2` | 57344 |
    | `torch_npu.hifloat8` | 32768 |

## 函数原型

```python
torch_npu.npu_gelu_quant(self, *, input_scale=None, input_offset=None, approximate="none", quant_mode="dynamic", dst_type=None, round_mode="rint") -> (Tensor, Tensor)
```

## 参数说明

- **self** (`Tensor`)：必选参数，待进行Gelu激活并量化的输入张量，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式支持$ND$，支持非连续的Tensor。不支持空Tensor。
  - 当`quant_mode`为`"dynamic"`时，shape支持2-8维度。
  - 当`quant_mode`为`"static"`时，shape支持1-8维度。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **input_scale** (`Tensor`)：可选参数，表示输入的量化尺度（静态场景）或平滑缩放尺度（动态场景）。数据类型需与`self`保持一致或采用精度更高的数据类型（例如`self`为`torch.bfloat16`，`input_scale`为`torch.float32`），shape仅支持1维，大小只能是`self`的尾轴维度大小或1，支持非连续的Tensor，数据格式支持$ND$。当`quant_mode`为`"static"`时为必选参数且不能为`None`，为`"dynamic"`时为可选输入。
- **input_offset** (`Tensor`)：可选参数，表示静态量化场景下量化输入的偏置。数据类型和shape需与`input_scale`保持一致，shape仅支持1维。当`quant_mode`为`"dynamic"`且`input_scale`为`None`时，`input_offset`也需要为`None`。
- **approximate** (`str`)：可选参数，Gelu激活函数的模式，支持取值`"none"`、`"tanh"`，分别对应erf和tanh模式，默认值为`"none"`。
- **quant_mode** (`str`)：可选参数，量化的模式，支持取值`"static"`（静态量化）、`"dynamic"`（动态量化），默认值为`"dynamic"`。
- **dst_type** (`int`)：可选参数，指定量化后输出y的数据类型。数据类型支持`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`，默认值为`None`，表示`torch.int8`。
- **round_mode** (`str`)：可选参数，数据转换的模式，默认值为`"rint"`。
  - 当`dst_type`为`torch.int8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`时，仅支持取值`"rint"`。
  - 当`dst_type`为`torch_npu.hifloat8`时，支持取值`"round"`、`"hybrid"`。

## 返回值说明

- **y** (`Tensor`)：Gelu激活后量化输出的结果，数据类型由`dst_type`指定，shape与输入`self`一致。当`dst_type`为`torch_npu.hifloat8`时，`y`的数据类型为`torch.uint8`（实际承载`torch_npu.hifloat8`类型）。
- **out_scale** (`Tensor`)：动态量化时计算出的量化尺度，数据类型为`torch.float32`，shape为`self`的shape剔除最后一维。当`quant_mode`为`"static"`时，`out_scale`为空Tensor。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 当`dst_type`为`torch_npu.hifloat8`时，必须传入`round_mode`为`"round"`或`"hybrid"`，不支持默认值`"rint"`。

## 调用示例

- 单算子模式调用

    ```python
  import torch
  import torch_npu

  x = torch.randn((32,128), dtype=torch.float32).npu()
  y, scale = torch_npu.npu_gelu_quant(x)
  ```

- 图模式调用

    ```python
  import torch
  import torch_npu
  import torchair
  import numpy as np

  class NetModel(torch.nn.Module):
      def __init__(self):
          super().__init__()
      def forward(self, x, input_scale=None, input_offset=None, approximate='none', quant_mode='dynamic', dst_type=None, round_mode="rint"):
          return torch_npu.npu_gelu_quant(x, input_scale=input_scale, input_offset=input_offset, approximate=approximate, quant_mode=quant_mode, dst_type=dst_type, round_mode=round_mode)
  # generate inputs
  x = torch.randn((32,128), dtype=torch.float16).npu()
  model = NetModel()
  config = torchair.CompilerConfig()
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  y, scale = model(x)
  ```
