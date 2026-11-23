# torch\_npu.npu_dequant\_swiglu\_quant

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

- API功能：对张量`x`做dequant反量化+swiglu激活+quant量化操作，同时支持分组。
- 计算公式：
    - group分组（目前仅支持count模式）：

        对输入`x`进行分组计算，`group_index`表示每个group分组的Tokens数，每组使用不同的量化scale（如`weight_scale`、`activation_scale`、`quant_scale`）。

        举例说明：假设x.shape=\[128, 2H\]，group\_index=\[2, 1, 3\]，表示有3个group，对应的scale维度为\[3, 2H\]。每个group数据使用不同的scale分别做dequant反量化+swiglu激活+quant量化操作。

        - group0=x\[0:2, :\]，scale0=scale\[0, :\]
        - group1=x\[2:3, :\]，scale1=scale\[1, :\]
        - group2=x\[3:6, :\]，scale2=scale\[2, :\]

    - dequant反量化：分别进行权重量化的反量化、激活量化的反量化。
        $$
        x=x*weight\_scale\\
        x=x*activation\_scale
        $$

    - swiglu激活：对反量化后的`x`做swiglu（通过attr activate\_left控制左右激活）。

        - 当swiglu_mode=0（标准Swiglu），以左激活为例：
            $$
            swiglu(x)=swish(x[:,0:H])*x[:,H:2H]
            $$
            其中：
            $$
            swish(z)=z*sigmoid(z)
            $$

        - 当swiglu_mode=1（变种Swiglu），对输入`x`按奇偶交错拆分：
            $$
            x\_{glu}=clamp(x\_{even},\ max=clamp\_limit)\\
            x\_{linear}=clamp(x\_{odd},-clamp\_limit,clamp\_limit)\\
            swiglu(x)=swish(x\_{glu},\alpha)*(x\_{linear}+bias)
            $$
            其中：
            $$
            swish(z,\alpha)=z*sigmoid(\alpha*z)
            $$
        - 当swiglu_mode=2（标准Swiglu的拆分方式+变种Swiglu的计算方式），以左激活为例：
            $$
            x\_{glu}=clamp(x[:,0:H],\ max=clamp\_limit)\\
            x\_{linear}=clamp(x[:,H:2H],-clamp\_limit,clamp\_limit)\\
            swiglu(x)=swish(x\_{glu},\alpha)*(x\_{linear}+bias)
            $$
            其中：
            $$
            swish(z,\alpha)=z*sigmoid(\alpha*z)
            $$
        - 当swiglu_mode=3（标准Swiglu的拆分方式+变种Swiglu的计算方式，clamp在swish之后），以左激活为例：
            $$
            x\_{glu}=clamp(swish(x[:,0:H]),\ max=clamp\_limit)\\
            x\_{linear}=clamp(x[:,H:2H],-clamp\_limit,clamp\_limit)\\
            swiglu(x)=x\_{glu}*x\_{linear}
            $$
            其中：
            $$
            swish(z)=z*sigmoid(z)
            $$

    - quant量化：
        1. （可选）先进行smooth量化。
            $$
            out=out*quant\_scale
            $$

        2. 动态/静态量化操作：对激活后的结果进行量化，以动态量化为例。
            $$
            out,scale=dynamicquant(out)
            $$

## 函数原型

```python
torch_npu.npu_dequant_swiglu_quant(x, *, weight_scale=None, activation_scale=None, bias=None, quant_scale=None, quant_offset=None, group_index=None, activate_left=False, quant_mode=0, dst_type=None, round_mode=None, activate_dim=None, swiglu_mode=0, clamp_limit=7.0, glu_alpha=1.702, glu_bias=1.0) -> (Tensor, Tensor)
```

## 参数说明

> [!NOTE]
> Tensor中shape使用的变量说明：
>
>- TokensNum：表示传输的Tokens数，取值≥0。
>- H：表示嵌入向量的长度，取值\>0。
>- groupNum：表示group\_index输入的长度，取值\>0。

- **x** (`Tensor`)：必选参数，表示目标张量。要求为2维张量，shape为\[TokensNum, 2H\]，尾轴为偶数。数据类型支持`torch.int32`、`torch.float16`、`torch.bfloat16`，数据格式为$ND$。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **weight\_scale** (`Tensor`)：可选参数，表示权重量化对应的反量化系数。要求为2维张量，shape为\[groupNum, 2H\]，数据类型支持`torch.float32`，数据格式为$ND$。当`x`为`torch.int32`时，建议提供weight_scale以进行反量化。
- **activation\_scale** (`Tensor`)：可选参数，表示pertoken激活量化对应的反量化系数。shape为\[TokensNum, 1\]，最后一维为1，其余与x保持一致。数据类型支持`torch.float32`，数据格式为$ND$。当`x`为`torch.int32`时，要求该参数非None，表示需要做反量化。
- **bias** (`Tensor`)：可选参数，表示`x`的偏置变量。数据类型支持`torch.int32`，数据格式为$ND$。当group_index非None时，bias必须为None。
- **quant\_scale** (`Tensor`)：可选参数，表示smooth量化系数。要求为2维张量，shape为\[groupNum, H\]，数据类型支持`torch.float32`、`torch.float16`和`torch.bfloat16`，数据格式为$ND$。
  > **注意：**静态量化下，quant\_scale仅支持`torch.float32`类型。
- **quant\_offset** (`Tensor`)：可选参数，表示量化中的偏移项。数据类型支持`torch.float32`、`torch.float16`和`torch.bfloat16`，数据格式为$ND$。`group_index`场景下（非None），该参数不生效，需为None。
- **group\_index** (`Tensor`)：可选参数，当前只支持count模式，表示该模式下指定分组的Tokens数（要求非负整数）。要求为1维张量，数据类型支持`torch.int64`，数据格式$ND$。
- **activate\_left** (`bool`)：可选参数，用于控制对输入沿最后一维等分后的左半部分还是右半部分做 swish 激活，仅在 swiglu_mode=0/2/3 时生效，默认值为 False。
    - 取True时，out=swish\(split\[x, -1, 2\]\[0\]\)\*split\[x, -1, 2\]\[1\]
    - 取False时，out=swish\(split\[x, -1, 2\]\[1\]\)\*split\[x, -1, 2\]\[0\]

- **quant\_mode** (`int`)：可选参数，表示量化类型，默认值为0。0表示静态量化，1表示动态量化。
- **dst\_type**(`int`)：可选参数，输出`out`的数据类型。`1`表示`torch.int8`，`290`表示`torch_npu.hifloat8`，`291`表示`torch.float8_e5m2`，`292`表示`torch.float8_e4m3fn`，`296`表示`torch_npu.float4_e2m1fn_x2`，`297`表示`torch_npu.float4_e1m2fn_x2`。默认值为`None`，表示输出为`torch.int8`。当输出为`torch_npu.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`时，`out`的最后一维为对应量化结果的一半（两个元素打包为一个字节存储）。
- **round\_mode**(`int`)：可选参数，输出`out`的舍入模式。取值范围为$[0,4]$：`0`表示`"rint"`，`1`表示`"round"`，`2`表示`"floor"`，`3`表示`"ceil"`，`4`表示`"trunc"`。当输出数据类型为`torch.int8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`时仅支持`0`；为`torch_npu.hifloat8`时仅支持`1`。默认值为`0`。
- **activate\_dim**(`int`)：可选参数，进行swish计算时选择的切分轴。取值范围为$[-x.dim(),\ x.dim()-1]$，负数表示从末尾开始计数。当`activate_dim`为非尾轴时，`group_index`必须为`None`，且`quant_mode`仅支持静态量化（`0`）。默认值为`-1`，表示尾轴。
- **swiglu\_mode**（`int`）：可选参数，swiglu计算模式，默认值为0。0表示传统swiglu，1表示变种swiglu（支持clamp、alpha、bias），2表示传统swiglu的拆分方式+变种swiglu的计算方式，3表示传统swiglu的拆分方式+变种swiglu的计算方式（clamp在swish之后，不支持alpha、bias）。
- **clamp\_limit**（`float`）：可选参数，swiglu输入门限，默认7.0。
- **glu\_alpha**（`float`）：可选参数，glu激活函数系数，默认1.702。
- **glu\_bias**（`float`）：可选参数，swiglu计算中的偏差，默认1.0。

## 返回值说明

- **out** (`Tensor`)：表示量化后的输出tensor。要求是2D的Tensor，shape=\[TokensNum, H\]，数据类型由`dst_type`决定，默认为`torch.int8`，数据格式为$ND$。当`dst_type`为`torch_npu.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`类型时，尾轴为H的一半。
- **scale** (`Tensor`)：表示量化的scale参数。要求是1D的Tensor，shape=\[TokensNum\]，数据类型支持`torch.float32`，数据格式为$ND$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式。
- `group_index`场景下（非None）约束说明：
    - `group_index`只支持count模式，需要网络保证`group_index`输入的求和不超过`x`的TokensNum维度，否则会出现越界访问。
    - H轴有维度大小限制：H≤10496同时64对齐场景；规格不满足场景会进行校验。
    - 输出`out`和`scale`超过`group_index`总和的部分未进行清理处理，该部分内存为垃圾数据，可能会存在inf/nan异常值，网络使用的时候需要注意影响。
- 当x为`torch.int32`时，建议提供weight_scale以进行反量化。
- 当x为`torch.float16`或`torch.bfloat16`时，weight_scale可选（通常为None，但允许传入），activation_scale、bias必须为None。
- x的最后一维长度必须为偶数。
- 当激活维度不是x的最后一维时，group_index必须为None。
- 当`group_index`非None，且为动态量化（即`quant_mode`为1）时，bias、quant_offset不生效。
- clamp_limit、glu_alpha、glu_bias仅在swiglu_mode=1/2时生效，swiglu_mode=3时仅clamp_limit生效。

## 调用示例

- 单算子模式调用

    ```python
    import os
    import shutil
    import unittest

    import torch
    import torch_npu
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SupportedDevices

    class TestNPUDequantSwigluQuant(TestCase):
        def test_npu_dequant_swiglu_quant(self, device="npu"):
            tokens_num = 4608
            hidden_size = 2048
            x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.int32)
            weight_scale = torch.randn((1, hidden_size), dtype=torch.float32)
            activation_scale = torch.randn((tokens_num, 1), dtype=torch.float32)
            quant_scale = torch.randn((1, hidden_size // 2), dtype=torch.float32)
            group_index = torch.tensor([tokens_num], dtype=torch.int64)
            bias = None
            out, scale = torch_npu.npu_dequant_swiglu_quant(
                x.npu(),
                weight_scale=weight_scale.npu(),
                activation_scale=activation_scale.npu(),
                bias=None,
                quant_scale=quant_scale.npu(),
                quant_offset=None,
                group_index=group_index.npu(),
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=7.0,
                glu_alpha=1.702,
                glu_bias=1.0
            )

    if __name__ == "__main__":
        run_tests()

    ```

- 图模式调用

    ```python
    import os
    import shutil
    import unittest

    import torch
    import torch_npu
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SupportedDevices
    from torchair.configs.compiler_config import CompilerConfig
    import torchair as tng

    class Model(torch.nn.Module):
        def forward(
            self,
            x, weight_scale, activation_scale, bias,
            quant_scale, quant_offset, group_index,
            activate_left, quant_mode, swiglu_mode, clamp_limit, glu_alpha, glu_bias
        ):
            return torch_npu.npu_dequant_swiglu_quant(
                x,
                weight_scale=weight_scale,
                activation_scale=activation_scale,
                bias=bias,
                quant_scale=quant_scale,
                quant_offset=quant_offset,
                group_index=group_index,
                activate_left=activate_left,
                quant_mode=quant_mode,
                swiglu_mode=swiglu_mode,
                clamp_limit=clamp_limit,
                glu_alpha=glu_alpha,
                glu_bias=glu_bias
            )

    class TestNPUDequantSwigluQuant(TestCase):
        def test_npu_dequant_swiglu_quant(self, device="npu"):
            tokens_num = 4608
            hidden_size = 2048
            x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.int32)
            weight_scale = torch.randn((1, hidden_size), dtype=torch.float32)
            activation_scale = torch.randn((tokens_num, 1), dtype=torch.float32)
            quant_scale = torch.randn((1, hidden_size // 2), dtype=torch.float32)
            group_index = torch.tensor([tokens_num], dtype=torch.int64)
            bias = None
            quant_offset = None

            compiler_config = CompilerConfig()
            npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
            npu_mode = 1
            model = Model().npu()

            if npu_mode == 1:
                model = torch.compile(model, backend=npu_backend, dynamic=False)
            else:
                model = torch.compile(model, backend=npu_backend, dynamic=True)

            out, scale = model(
                x.npu(),
                weight_scale.npu(),
                activation_scale.npu(),
                bias,
                quant_scale.npu(),
                quant_offset,
                group_index.npu(),
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=7.0,
                glu_alpha=1.702,
                glu_bias=1.0
            )

    if __name__ == "__main__":
        run_tests()

    ```
