# torch_npu.npu_dynamic_quant_asymmetric

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

- API功能：

    对输入的张量进行动态非对称量化。支持pertoken、pertensor、perchannel和MoE（Mixture of Experts，混合专家模型）场景。

- 计算公式：
    
    rowMax、rowMin代表取最大值、最小值的模式：在`pertoken`模式下表示按行取最大值、最小值（此处的“行”对应`input`最后一个维度的数据，即一个token），在`pertensor`模式下表示求整个张量的最大值、最小值，在`perchannel`模式下表示按列取最大值、最小值。DST_MAX、DST_MIN分别对应量化后dtype的最大值和最小值，公式如下：

    $$
    \text{scale} = \frac{\text{rowMax}(\mathbf{x}) - \text{rowMin}(\mathbf{x})}{DST\_MAX - DST\_MIN}\\
    \text{offset} = DST\_MAX - \frac{\text{rowMax}(\mathbf{x})}{\text{scale}}\\
    y = \text{round}(\frac{\mathbf{x}}{\text{scale}} + \text{offset})
    $$

    - MoE和非MoE场景下，若使用smooth quant，非MoE场景下会引入smooth_scales输入，其形状与x最后一个维度大小一致，在进行量化前，会先令x乘以smooth_scales，再按上述公式进行量化。MoE场景下会同时引入smooth_scales和group_index，此时smooth_scales中包含多组smooth向量，按group_index中的数值作用到x的不同行上。具体地，假如x包含m个token，smooth_scales有n行，smooth_scales[0]会作用到x[0:group_index[0]]上，smooth_scales[i]会作用到x[group_index[i-1]: group_index[i]]上，i=1,2, ...,n-1。

## 函数原型

```python
torch_npu.npu_dynamic_quant_asymmetric(input, *, smooth_scales=None, group_index=None, dst_type=None, quant_mode="pertoken", dst_type_max=0.0) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，需要进行量化的源数据张量，数据类型支持`torch.float16`、`torch.bfloat16`，数据格式支持$ND$，支持非连续的Tensor。输入`input`的维度必须大于1。进行`torch_npu.int4`量化时，要求`input`形状的最后一维是8的整数倍。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **smooth_scales** (`Tensor`)：可选参数，用于提供缩放系数(scales)的张量，数据类型支持`torch.float16`、`torch.bfloat16`，数据格式支持$ND$，支持非连续的Tensor。
    - 在非MoE场景shape必须是1维，和`input`的最后一维相等。
    - 在MoE场景shape是2维[E, H]。其中E是专家数，取值范围在[1, 1024]且与group_index的第一维相同；H是x的最后一维。
    - 单算子模式下`smooth_scales`的dtype必须和`input`保持一致，图模式下可以不一致。
- **group_index** (`Tensor`)：可选参数，用于对`smooth_scales`进行分组的下标张量（代表`input`的行数索引），仅在MoE场景下生效。数据类型支持`torch.int32`，数据格式支持$ND$，支持非连续的Tensor。`group_index`的shape为[E,]，E的取值范围在[1, 1024]且与smooth_scales第一维相同。Tensor的取值必须递增且范围为[1, S]，最后一个值必须等于S（S代表输入`input`的行数，是`input`的shape除最后一维度外的乘积）。
- **dst_type** (`int`)：可选参数，指定量化输出的类型，传None时当作`torch.int8`处理。

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int8`、`torch.quint4x2`。
    <!-- end id4 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch.quint4x2`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。
    <!-- end id6 -->

- **quant_mode** (`str`)：可选参数，指定量化模式，默认值为`"pertoken"`。如果`group_index`不为`None`，仅支持取值`"pertoken"`。

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数，默认按`"pertoken"`处理。
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持取值`"pertoken"`（按token粒度）、`"perchannel"`（按通道）、`"pertensor"`（整个张量共用一个scale）。
    <!-- end id8 -->

- **dst_type_max** (`float`)：可选参数，指定目标数据类型的最大表示值，默认值为0.0。

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数。
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT系列产品</term>：仅在`dst_type`为`torch_npu.hifloat8`时生效，支持取值0.0~32768.0；取值为0.0时使用目标精度能表示的最大值，取值为非0.0时使用传入值作为目标数据类型的最大值。
    <!-- end id10 -->

## 返回值说明

- **y** (`Tensor`)：量化后的输出，数据类型由`dst_type`指定。当`dst_type`是`torch.quint4x2`时，`y`的数据类型为`torch.int32`，形状最后一维为`input`最后一维除以8，其余维度与`input`一致，每个`torch.int32`元素包含8个`torch_npu.int4`结果。其他场景下`y`形状与输入`input`一致，数据类型由`dst_type`指定。
    <!-- npu="950" id5 -->
    - 在<term>Ascend 950PR&950DT系列产品</term>上，当`dst_type`为`torch_npu.hifloat8`时，`y`的数据类型为`torch.uint8`（实际承载`torch_npu.hifloat8`类型）。
    <!-- end id5 -->
- **scale** (`Tensor`)：非对称动态量化过程中计算出的缩放系数，数据类型为`torch.float32`。当`quant_mode`为`"pertoken"`时，shape为`input`的形状剔除最后一维；当`quant_mode`为`"perchannel"`时，shape为`input`的形状剔除倒数第二维，最后一维保持与`input`一致；当`quant_mode`为`"pertensor"`时，shape为`(1,)`。
- **offset** (`Tensor`)：非对称动态量化过程中计算出的偏移系数，数据类型为`torch.float32`，shape和`scale`一致。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- 使用可选参数`smooth_scales`、`group_index`、`dst_type`时，必须使用关键字传参。

## 调用示例

- 单算子模式调用
    - 只有一个输入`input`，进行`torch.int8`量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x)
        print(y, scale, offset)
        ```

    - 只有一个输入`input`，进行`torch_npu.int4`量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, dst_type=torch.quint4x2)
        print(y, scale, offset)
        ```

    - 使用`smooth_scales`输入，非MoE场景（不使用`group_index`），进行`torch.int8`量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        smooth_scales = torch.rand((8,), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales)
        print(y, scale, offset)
        ```

    - 使用`smooth_scales`输入，MoE场景（使用`group_index`），进行`torch.int8`量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        smooth_scales = torch.rand((2, 8), dtype=torch.half).npu()
        group_index = torch.Tensor([1, 3]).to(torch.int32).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales, group_index=group_index)
        print(y, scale, offset)
        ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    device=torch.device(f'npu:0')
    
    torch_npu.npu.set_device(device)
    
    class DynamicQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, input_tensor, smooth_scales=None, group_index=None, dst_type=None):
            out, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(input_tensor, smooth_scales=smooth_scales, group_index=group_index, dst_type=dst_type)
            return out, scale, offset
    
    x = torch.randn((2, 4, 6),device='npu',dtype=torch.float16).npu()
    smooth_scales = torch.randn((6),device='npu',dtype=torch.float16).npu()
    dynamic_quant_model = DynamicQuantModel().npu()
    dynamic_quant_model = torch.compile(dynamic_quant_model, backend=npu_backend, dynamic=True)
    out, scale, offset = dynamic_quant_model(x, smooth_scales=smooth_scales)
    print(out)
    print(scale)
    print(offset)
    ```
