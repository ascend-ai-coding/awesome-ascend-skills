# torch_npu.npu_quantize

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
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

- API功能：对输入的张量进行量化处理。
- 计算公式：
    - 若`div_mode`为`True`：
        $$
        result=(self/scales)+zero\_points
        $$

    - 若`div_mode`为`False`：

        $$
        result=(self*scales)+zero\_points
        $$

## 函数原型

```python
torch_npu.npu_quantize(self, scales, zero_points, dtype, axis=1, div_mode=True) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，需要进行量化的源数据张量，对应公式中的`self`。数据格式支持$ND$，支持空Tensor，支持非连续的Tensor。`div_mode`为`False`且`dtype`为`torch.quint4x2`时，最后一维需要能被8整除。

    <!-- npu="310p" id5 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float`、`torch.float16`。
    <!-- end id5 -->
    <!-- npu="A3,910b" id6 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float`、`torch.float16`、`torch.bfloat16`。`div_mode`为`False`时，且当数据类型为`torch.float`时，数据格式支持$NZ$。
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float`、`torch.float16`、`torch.bfloat16`。
    <!-- end id7 -->

- **scales** (`Tensor`)：必选参数，对`self`进行缩放的张量，对应公式中的`scales`。支持空Tensor，支持非连续的Tensor。
    - `div_mode`为`True`时：

        <!-- npu="310p" id8 -->
        - <term>Atlas推理系列产品</term>：数据类型支持`torch.float`。
        <!-- end id8 -->
        <!-- npu="A3,910b" id9 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float`、`torch.bfloat16`。
        <!-- end id9 -->

    - `div_mode`为`False`时，数据格式支持$ND$。支持1维或多维（1维时，对应轴的大小需要与`self`中第`axis`维相等或等于1；多维时，`scales`的shape需要与`self`的shape维度相等，除`axis`指定的维度，其他维度为1，`axis`指定的维度必须和`self`对应的维度相等或等于1）。数据类型、数据格式需要和`self`的数据类型和数据格式一致。

        <!-- npu="310p" id10 -->
        - <term>Atlas推理系列产品</term>：数据类型支持`torch.float`、`torch.float16`。
        <!-- end id10 -->
        <!-- npu="A3,910b" id11 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float`、`torch.float16`、`torch.bfloat16`。当数据格式为$NZ$时，`scales`的所有元素值为1。
        <!-- end id11 -->

- **zero_points** (`Tensor`)：必选参数，允许为None，对`self`进行偏移的张量，对应公式中的`zero_points`。支持空Tensor，支持非连续的Tensor。
    - `div_mode`为`True`时：

        <!-- npu="310p" id12 -->
        - <term>Atlas推理系列产品</term>：数据类型支持`torch.int8`、`torch.uint8`、`torch.int32`。
        <!-- end id12 -->
        <!-- npu="A3,910b" id13 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int8`、`torch.uint8`、`torch.int32`、`torch.bfloat16`。
        <!-- end id13 -->

    - `div_mode`为`False`时，数据格式支持$ND$。支持1维或多维（1维时，对应轴的大小需要与`self`中第`axis`维相等或等于1；多维时，`scales`的shape需要与`self`维度相等，除`axis`指定的维度，其他维度为1，`axis`指定的维度必须和`self`对应的维度相等）。`zero_points`的shape和dtype需要和`scales`一致。

        <!-- npu="310p" id14 -->
        - <term>Atlas推理系列产品</term>：数据类型支持`torch.float`、`torch.float16`。
        <!-- end id14 -->
        <!-- npu="A3,910b" id15 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float`、`torch.float16`、`torch.bfloat16`。当`self`的数据格式为$NZ$时，值为空。
        <!-- end id15 -->

- **dtype** (`int`)：必选参数，指定输出参数的类型。
    - `div_mode`为`True`时：

        <!-- npu="310p" id16 -->
        - <term>Atlas推理系列产品</term>：类型支持`torch.qint8`、`torch.quint8`、`torch.int32`。
        <!-- end id16 -->
        <!-- npu="A3,910b" id17 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：类型支持`torch.qint8`、`torch.quint8`、`torch.int32`。
        <!-- end id17 -->
        <!-- npu="950" id18 -->
        - <term>Ascend 950PR&950DT系列产品</term>：类型支持`torch.qint8`、`torch.quint8`、`torch.int32`、`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`。
        <!-- end id18 -->

    - `div_mode`为`False`时：

        <!-- npu="310p" id19 -->
        - <term>Atlas推理系列产品</term>：类型支持`torch.qint8`、`torch.int8`。
        <!-- end id19 -->
        <!-- npu="A3,910b" id20 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：类型支持`torch.qint8`、`torch.int8`、`torch.quint4x2`。当`dtype`为`torch.quint4x2`时，输出tensor类型为`torch.int32`，由8个`torch_npu.int4`拼接。
        <!-- end id20 -->
        <!-- npu="950" id21 -->
        - <term>Ascend 950PR&950DT系列产品</term>：类型支持`torch.qint8`、`torch.int8`、`torch.quint4x2`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`。当`dtype`为`torch.quint4x2`时，输出tensor类型为`torch.int32`，由8个`torch_npu.int4`拼接。
        <!-- end id21 -->

- **axis** (`int`)：可选参数，量化的element-wise轴，其他的轴做broadcast，默认值为`1`。

    - `div_mode`为`True`时，支持范围为小于输入`self`的维度数且大于等于`self`维度数的负值。当输入的`scales`和`zero_points`的size均为1时，该参数实际不使用。

    - `div_mode` 为`False`时，`axis`取值范围是[-2, +∞)且指定的轴不能超过输入`self`的维度数。如果`axis`为-2，代表量化的element-wise轴是输入`self`的倒数第二根轴；如果`axis`大于-2，量化的element-wise轴是输入的最后一根轴（即`axis`>-2时，都指定-1轴）。

- **div_mode** (`bool`)：可选参数，表示计算`scales`模式，对应公式中的`div_mode`。当`div_mode`为`True`时，表示用除法计算`scales`；`div_mode`为`False`时，表示用乘法计算`scales`，默认值为`True`。

## 返回值说明

`Tensor`

对应公式中的`result`。数据类型由参数`dtype`指定，如果参数`dtype`为`torch.quint4x2`，输出的`dtype`是`torch.int32`，shape的最后一维是`self`的shape最后一维的1/8，shape其他维度和`self`的shape其他维度保持一致；如果参数`dtype`不为`torch.quint4x2`时，shape与输入`self`的shape保持一致。输出的数据格式与输入`self`的数据格式保持一致，且当数据格式为$NZ$时，数据类型仅支持`torch.int32`。支持空Tensor，支持非连续的Tensor。

<!-- npu="950" id22 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - 单算子模式：当`dtype`为`torch_npu.hifloat8`时，`result`的数据类型为`torch.uint8`（实际承载`torch_npu.hifloat8`类型）。
  - 图模式：当`dtype`为`torch_npu.hifloat8`时，`result`的数据类型为`torch.bits8`（实际承载`torch_npu.hifloat8`类型）。
<!-- end id22 -->

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- `self`数据格式为$NZ$时，`self`输入shape支持3维，形如\(e，k，n\)，k为256的倍数，n为8的倍数，`scales`输入shape支持1维或3维，`zero_points`输入为None，`dtype`为`torch.quint4x2`。
- `div_mode`为`False`时：

    <!-- npu="910b" id23 -->
    - 支持<term>Atlas A2系列产品</term>。
    <!-- end id23 -->
    <!-- npu="310p" id24 -->
    - 当`dtype`为`torch.quint4x2`或者`axis`为-2时，不支持<term>Atlas推理系列产品</term>。
    <!-- end id24 -->

## 调用示例

- 单算子模式调用

    <!-- npu="910b" id25 -->
    - <term>Atlas A2系列产品</term>

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.randn(1, 1, 12).bfloat16().npu()
        >>> scale = torch.tensor([0.1] * 12).bfloat16().npu()
        >>> out = torch_npu.npu_quantize(x, scale, None, torch.qint8, -1, False)
        >>> print(x)
        tensor([[[ 0.9609,  1.3281, -0.6172,  0.5469, -1.1797, -1.1719, -0.7422,
                0.9727, -0.9062, -0.0815, -0.8047,  1.0703]]], device='npu:0',
            dtype=torch.bfloat16)
        >>> print(out)
        tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], device='npu:0',
            dtype=torch.int8)
        ```
    <!-- end id25 -->
    <!-- npu="310p" id26 -->
    - <term>Atlas推理系列产品</term>

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.randn((2, 3, 12), dtype=torch.float).npu()
        >>> scale = torch.tensor(([3] * 12),dtype=torch.float).npu()
        >>> out = torch_npu.npu_quantize(x, scale, None, torch.qint8, -1, False)
        >>> print(x)
        tensor([[[-7.7834e-01, -1.0473e+00, -1.1155e+00,  1.2233e+00, -1.2271e+00,
                -2.5612e+00, -1.8274e-01,  2.8293e+00,  1.9029e-01, -1.9333e+00,
                -4.9270e-01, -1.0650e+00],
                [-8.9416e-01,  3.1869e-02, -5.8144e-01, -4.9477e-01,  9.7323e-02,
                -3.8681e-01,  2.1969e-03, -6.3244e-01,  7.1591e-01, -1.8587e-01,
                -1.3381e+00, -2.6253e-01],
                [ 1.8462e-02,  1.2397e-01, -9.0656e-01, -9.9280e-01, -4.4235e-02,
                1.0623e+00, -9.8437e-02,  1.2941e+00,  1.0805e+00, -1.7269e-01,
                -9.9205e-02, -6.1429e-01]],

                [[ 1.3678e+00, -2.7348e-01, -4.1354e-01, -9.4638e-01,  4.2792e-01,
                8.0462e-01,  9.3584e-01,  6.3704e-01,  1.1269e+00, -1.5329e+00,
                5.8572e-01, -1.3966e+00],
                [ 3.5882e-01,  8.7029e-01, -1.3176e+00,  1.1601e+00, -3.6984e-01,
                7.3642e-01, -1.0755e+00,  6.6557e-01,  3.1149e+00, -6.8776e-01,
                -1.0913e+00,  4.4962e-01],
                [-1.2505e+00,  1.5474e+00, -7.4332e-02, -1.6657e+00,  1.3275e+00,
                5.8914e-02,  8.4287e-01, -1.7109e+00,  1.8256e-01,  3.2937e-01,
                2.4875e+00,  1.3921e+00]]], device='npu:0')
        >>> print(out)
        tensor([[[-2, -3, -3,  4, -4, -8, -1,  8,  1, -6, -1, -3],
                [-3,  0, -2, -1,  0, -1,  0, -2,  2, -1, -4, -1],
                [ 0,  0, -3, -3,  0,  3,  0,  4,  3, -1,  0, -2]],

                [[ 4, -1, -1, -3,  1,  2,  3,  2,  3, -5,  2, -4],
                [ 1,  3, -4,  3, -1,  2, -3,  2,  9, -2, -3,  1],
                [-4,  5,  0, -5,  4,  0,  3, -5,  1,  1,  7,  4]]], device='npu:0',
            dtype=torch.int8)
        ```
    <!-- end id26 -->

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    x = torch.randn((2, 3, 12), dtype=torch.float16).npu()
    scale = torch.tensor(([3] * 12), dtype=torch.float16).npu()
    axis = 1
    div_mode = False


    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, x, scale, zero_points, dst_type, div_mode):
            return torch_npu.npu_quantize(x, scale, zero_points=zero_points, dtype=dst_type, div_mode=div_mode)


    model = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    config.debug.graph_dump.type = 'pbtxt'
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
    output_data = model(x, scale, None, dst_type=torch.qint8, div_mode=div_mode)
    print("shape of x:", x.shape)
    print("shape of output_data:", output_data.shape)
    print("x:", x)
    print("output_data:", output_data)



    # 执行上述代码的输出类似如下
    shape of x: torch.Size([2, 3, 12])
    shape of output_data: torch.Size([2, 3, 12])
    x: tensor([[[ 0.2600,  0.6782,  0.5024,  0.9492,  0.6089,  0.7461,  1.5332,
            -0.2123,  0.6558, -0.8354, -0.5366, -0.6821],
            [-0.2522,  0.2415, -0.0269, -0.1497,  0.2256, -0.5239,  0.7363,
            -0.2468,  1.6064,  1.4170, -0.2213,  1.5947],
            [-0.6328,  0.8105,  0.2532,  1.0684, -1.2119, -0.6865,  0.7451,
            -0.8120,  0.6401, -2.1270, -0.9482, -1.1973]],

            [[-1.7461, -1.1758, -0.5352,  1.5938,  1.8945, -2.2500, -0.5073,
            -0.8164,  0.8267, -0.4377,  1.2490,  0.2415],
            [ 0.8062, -1.0498, -0.8345,  1.1465, -0.7349,  0.1317,  0.2280,
            -0.8145,  0.2673,  1.4756, -1.6768,  1.1572],
            [-0.3147, -0.4446, -1.0508,  0.8325,  1.4590,  0.2096, -0.9961,
            0.6089, -0.2460,  1.1543,  0.9277,  0.1079]]], device='npu:0',
        dtype=torch.float16)
        
    output_data: tensor([[[ 1,  2,  2,  3,  2,  2,  5, -1,  2, -3, -2, -2],
            [-1,  1,  0,  0,  1, -2,  2, -1,  5,  4, -1,  5],
            [-2,  2,  1,  3, -4, -2,  2, -2,  2, -6, -3, -4]],

            [[-5, -4, -2,  5,  6, -7, -2, -2,  2, -1,  4,  1],
            [ 2, -3, -3,  3, -2,  0,  1, -2,  1,  4, -5,  3],
            [-1, -1, -3,  2,  4,  1, -3,  2, -1,  3,  3,  0]]], device='npu:0',
        dtype=torch.int8)

    ```
