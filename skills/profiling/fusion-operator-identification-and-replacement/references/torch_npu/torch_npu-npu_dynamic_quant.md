# torch_npu.npu_dynamic_quant

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

- **API功能**：对输入的张量进行动态量化，支持pertoken、pertensor、perchannel和MoE（Mixture of Experts，混合专家模型）场景。

    如果是MoE（Mixture of Experts，混合专家模型）场景，会引入`group_index`，`smooth_scales`中包含多组smooth向量，按`group_index`中的数值作用到`input`的不同行上。具体地，假如`input`包含m个token，`smooth_scales`有n行，`smooth_scales[0]`会作用到`input[0:group_index[0]]`上，`smooth_scales[i]`会作用到`input[group_index[i-1]: group_index[i]]`上，`i=1, 2, ..., n-1`。

    通过`quant_mode`可指定量化模式，支持`pertoken`（默认）、`perchannel`、`pertensor`三种模式。其中`pertoken`表示按token粒度量化，`perchannel`表示按通道量化，`pertensor`表示对整个张量使用同一个scale量化。
    
    <!-- npu="950" id11 -->
    `perchannel`和`pertensor`模式仅在<term>Ascend 950PR&950DT系列产品</term>上支持，详见参数说明。

    <!-- end id11 -->
- 计算公式：
    - 若`smooth_scales`不存在：
    $$
    \text{scale} = \frac{\text{rowMax}(\text{abs}(\mathbf{x}))}{DTYPE\_MAX} \\
    y = \text{round}\left(\frac{\mathbf{x}}{\text{scale}}\right)
    $$

    - 若`smooth_scales`存在：
    $$
    \text{scale} = \frac{\text{rowMax}(\text{abs}(\mathbf{x} * smooth\_scales))}{DTYPE\_MAX}  \\
    y = \text{round}\left(\frac{\mathbf{x} * smooth\_scales}{\text{scale}}\right)
    $$

    rowMax表示求最大值的模式：在`pertoken`模式下表示求一行的最大值，在`pertensor`模式下表示求整个张量的最大值，在`perchannel`模式下表示求一列的最大值。DTYPE_MAX表示常量，是y输出对应的数据类型的最大值。当`dst_type_max`不为0时，使用`dst_type_max`的值作为DTYPE_MAX。

## 函数原型

```python
torch_npu.npu_dynamic_quant(input, *, smooth_scales=None, group_index=None, dst_type=None, quant_mode="pertoken", dst_type_max=0.0) ->(Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，需要进行量化的源数据张量，数据类型支持`torch.float16`、`torch.bfloat16`，数据格式支持$ND$，支持非连续的Tensor。输入`input`的维度必须大于1。进行`torch_npu.int4`量化时，要求`input`形状的最后一维是8的整数倍。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **smooth_scales** (`Tensor`)：可选参数，用于对`input`进行缩放的张量，数据类型支持`torch.float16`、`torch.bfloat16`，数据格式支持$ND$，支持非连续的Tensor。shape见约束说明。
- **group_index** (`Tensor`)：可选参数，对`smooth_scales`进行分组的下标，仅在MoE场景下生效。数据类型支持`torch.int32`，数据格式支持$ND$，支持非连续的Tensor。`group_index`为1维Tensor，元素数量与`smooth_scales`的第一维一致；`group_index`不为`None`时，`smooth_scales`必须不为`None`。
- **dst_type** (`int`)：可选参数，指定量化输出的类型，传`None`时当作`torch.int8`处理。
    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持取值`torch.int8`、`torch.quint4x2`。
    <!-- end id4 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持取值`torch.int8`、`torch.quint4x2`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。
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

- **y** (`Tensor`)：量化后的输出Tensor，数据类型由`dst_type`指定。当`dst_type`是`torch.quint4x2`时，`y`的数据类型为`torch.int32`，形状最后一维为`input`最后一维除以8，其余维度与`input`一致，每个`torch.int32`元素包含8个`torch_npu.int4`结果。其他场景下`y`形状与输入`input`一致，数据类型由`dst_type`指定。
    <!-- npu="950" id5 -->
    - 在<term>Ascend 950PR&950DT系列产品</term>上，当`dst_type`为`torch_npu.hifloat8`时，`y`的数据类型为`torch.uint8`（实际承载`torch_npu.hifloat8`类型）。
    <!-- end id5 -->
- **scale** (`Tensor`)：对称动态量化过程中计算出的缩放系数，数据类型为`torch.float32`。
  - 当`quant_mode`为`"pertoken"`时，形状为`input`的形状剔除最后一维。
  - 当`quant_mode`为`"perchannel"`时，形状为`input`的形状剔除倒数第二维，最后一维保持与`input`一致。
  - 当`quant_mode`为`"pertensor"`时，形状为`(1,)`。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- 该接口支持MoE场景。

- 使用`smooth_scales`时：
    - 在`pertoken`/`pertensor`模式且不使用`group_index`时，`smooth_scales`必须是一维Tensor，元素数量与`input`的最后一维大小一致。
    - 在`perchannel`模式时，`smooth_scales`必须是一维Tensor，元素数量与`input`的倒数第二维大小一致。
    - 若使用`group_index`，`smooth_scales`必须是二维Tensor，第一维（专家数）取值范围为`[1, 1024]`，第二维元素数量与`input`的最后一维大小一致，`group_index`必须是一维数组，元素数量与`smooth_scales`第一维一致。`group_index`中的元素必须是单调递增的，其最后一个元素的值，应等于`input`的元素数量除以`input`的最后一个维度。
    - 单算子模式下`smooth_scales`的数据类型必须与`input`保持一致，图模式下可以不一致。

## 调用示例

- 单算子模式调用
    - 只有一个输入`input`

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 3), dtype=torch.float16).to("npu")
        >>> print(x)
        tensor([[0.7261, 0.3726, 0.9126],
                [0.9023, 0.9990, 0.1279],
                [0.8628, 0.6240, 0.9028]], device='npu:0', dtype=torch.float16)
        >>>
        >>> output, scale = torch_npu.npu_dynamic_quant(x)
        >>> print(output)
        tensor([[101,  52, 127],
                [115, 127,  16],
                [121,  88, 127]], device='npu:0', dtype=torch.int8)
        >>> print(scale)
        tensor([0.0072, 0.0079, 0.0071], device='npu:0')
        ```

    - 使用`smooth_scales`输入

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 3), dtype=torch.float16).to("npu")
        >>> print(x)
        tensor([[0.6680, 0.9492, 0.0845],
                [0.1924, 0.5278, 0.1484],
                [0.6631, 0.9497, 0.0957]], device='npu:0', dtype=torch.float16)
        >>>
        >>> smooth_scales = torch.rand((3,), dtype=torch.float16).to("npu")
        >>> print(smooth_scales)
        tensor([0.8042, 0.0884, 0.8901], device='npu:0', dtype=torch.float16)
        >>>
        >>> output, scale = torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales)
        >>> print(output)
        tensor([[127,  20,  18],
                [127,  38, 108],
                [127,  20,  20]], device='npu:0', dtype=torch.int8)
        >>> print(scale)
        tensor([0.0042, 0.0012, 0.0042], device='npu:0')
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

    device = torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    
    class DynamicQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, input_tensor, smooth_scales=None, group_index=None, dst_type=None):
            out, scale = torch_npu.npu_dynamic_quant(input_tensor, smooth_scales=smooth_scales, group_index=group_index,dst_type=dst_type)
            return out, scale

    x = torch.randn((2, 4, 6), device='npu', dtype=torch.float16).npu()
    smooth_scales = torch.randn((6), device='npu', dtype=torch.float16).npu()
    dynamic_quant_model = DynamicQuantModel().npu()
    dynamic_quant_model = torch.compile(dynamic_quant_model, backend=npu_backend, dynamic=True)
    out, scale = dynamic_quant_model(x, smooth_scales=smooth_scales)
    print(out)
    print(scale)
    
    # 执行上述代码的输出类似如下
    tensor([[[-116,  127,   14, -105,   12,  -44],
            [   7, -127,  -49,  -27,   -4,   -7],
            [ -49,   18,  127,   39,   14,   13],
            [  12,  -47,  127,   73,   28,    1]],
    
            [[  62,  127,  -61,  -15,   -9,   -8],
            [ 127,  -74,  -66,  117,   27,   27],
            [   3,   65,   29,  127,  -27,   20],
            [  -4, -127,   13,  -40,  -21,  -11]]], device='npu:0',
        dtype=torch.int8)
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now,
    dtype cast replace with float. (function operator())
    tensor([[0.0080, 0.0422, 0.0219, 0.0132],
            [0.0176, 0.0069, 0.0093, 0.0368]], device='npu:0')
    ```
