# torch_npu.npu_dynamic_block_quant

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

- API功能：对输入张量，通过给定的`row_block_size`和`col_block_size`将输入划分成多个数据块，以数据块为基本粒度进行量化。在每个块中，先计算出当前块对应的量化参数`scale`，并根据`scale`对输入进行量化。输出最终的量化结果，以及每个块的量化参数`scale`。

- 计算公式：
  $$
  input\_max = block\_reduce\_max(abs(x))
  $$

  $$
  scale =
  \begin{cases}
  min(input\_max / FP8\_MAX, 1 / min\_scale), & dst\_type \text{ 指定为 FP8} \\
  min(input\_max / HiF8\_MAX, 1 / min\_scale), & dst\_type \text{ 指定为 HiF8} \\
  min(input\_max / `torch.int8`\_MAX, 1 / min\_scale), & dst\_type \text{ 指定为 `torch.int8`}
  \end{cases}
  $$

  $$
  y = cast\_to\_[FP8/HiF8/`torch.int8`](x / scale)
  $$

  其中$block\_reduce\_max$代表求每个`block`中的最大值。`FP8_MAX`、`HiF8_MAX`、`INT8_MAX`分别表示FP8、HiF8、`torch.int8`目标量化类型可表示的最大正数值，由`dst_type`决定。当`dst_type_max`不为0时，使用`dst_type_max`的值作为目标类型的最大值。
  FP8、HiF8、`torch.int8`均为8-bit低精度量化目标类型，分别表示FP8浮点、HiF8浮点和`torch.int8`整数格式。当前支持的最大值如下：

  | 目标类型 | 最大正数值 |
  | --- | --- |
  | FP8 | FP8_MAX = 448 |
  | HiF8 | HiF8_MAX = 32768 |
  | `torch.int8` | INT8_MAX = 127 |

## 函数原型

```python
torch_npu.npu_dynamic_block_quant(x, *, min_scale=0.0, round_mode="rint", dst_type=1, row_block_size=1, col_block_size=128, dst_type_max=0.0) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，数据类型支持`torch.float16`、`torch.bfloat16`，支持非连续的Tensor，数据格式支持$ND$。当前shape支持2维和3维。不支持空Tensor。
- **min_scale** (`float`)：可选参数，参与`scale`计算的最小`scale`值。当前支持取值大于等于0。
- **round_mode** (`str`)：可选参数，指定类型转换到输出的转换方式，默认值为`"rint"`。

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持取值`"rint"`。
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
      - 当`dst_type`为`torch.int8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`时，仅支持取值`"rint"`。
      - 当`dst_type`为`torch_npu.hifloat8`时，仅支持取值`"round"`。
    <!-- end id5 -->

- **dst_type** (`int`)：可选参数，指定输出`y`的数据类型，默认值为`torch.int8`。

    <!-- npu="A3,910b" id6 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int8`。
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。
    <!-- end id7 -->

- **row_block_size** (`int`)：可选参数，指定单个量化的数据块的行大小。

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持取值1。
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持取值1、128、256、512。
    <!-- end id9 -->

- **col_block_size** (`int`)：可选参数，指定单个量化的数据块的列大小。

    <!-- npu="A3,910b" id10 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持取值1或128。
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持取值64、128、192、256。
    <!-- end id11 -->

- **dst_type_max** (`float`)：可选参数，指定目标数据类型的最大值，默认值为0.0。取值为0.0时表示使用数据类型原始的最大值。

    <!-- npu="A3,910b" id12 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数。
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR&950DT系列产品</term>：仅在`dst_type`为`torch_npu.hifloat8`时生效，支持取值0.0、15.0、56.0、224.0、32768.0。
    <!-- end id13 -->

## 返回值说明

- **y** (`Tensor`)：量化结果，数据类型由参数`dst_type`指定，shape与输入`x`一致。

    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
        - 单算子模式/静态图模式：当`dst_type`为`torch_npu.hifloat8`时，`y`输出的数据类型为`torch.uint8`（实际承载的是`torch_npu.hifloat8`类型）。
        - 动态图模式：当`dst_type`为`torch_npu.hifloat8`时，`y`输出的数据类型为`torch.bits8`（实际承载的是`torch_npu.hifloat8`类型）。
    <!-- end id14 -->

- **scale** (`Tensor`)：量化时使用的量化参数，数据类型为`torch.float32`。如果输入`x`的shape为`[M, N]`，`scale`的shape为`[ceil(M/row_block_size), ceil(N/col_block_size)]`；如果输入`x`的shape为`[B, M, N]`，`scale`的shape为`[B, ceil(M/row_block_size), ceil(N/col_block_size)]`。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式。

## 调用示例

- 单算子模式调用

    ```python
  >>> import torch
  >>> import torch_npu
  
  >>> x = torch.rand(3, 4).to("npu").to(torch.float16)
  >>> min_scale = 0
  >>> dst_type = 1
  >>> row_block_size = 1
  >>> col_block_size = 128
  
  >>> y, scale = torch_npu.npu_dynamic_block_quant(x, min_scale=min_scale, dst_type=dst_type, row_block_size=row_block_size, col_block_size=col_block_size)
  >>> print(y)
  tensor([[ 92,  65,  15, 127],
          [100, 127, 116,  64],
          [ 95,  15,  87, 127]], device='npu:0', dtype=torch.int8)
  >>> print(scale)
  tensor([[0.0063],
          [0.0076],
          [0.0073]], device='npu:0')
  ```

<!-- npu="950" id15 -->
- 图模式调用：仅适用于<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import torch
    import torch_npu
    import torchair

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, min_scale=0.0, round_mode="rint", dst_type=torch.float8_e5m2, row_block_size=1, col_block_size=128):
            return torch_npu.npu_dynamic_block_quant(x, min_scale=min_scale, round_mode=round_mode, dst_type=dst_type, row_block_size=row_block_size, col_block_size=col_block_size)

    x = torch.zeros((1, 128)).to(torch.float16).to('npu')
    model = Model()
    model = model.to('npu')
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    y, scale = model(x, 0.0, "rint", torch.float8_e5m2, 1, 128)
    ```
<!-- end id15 -->
