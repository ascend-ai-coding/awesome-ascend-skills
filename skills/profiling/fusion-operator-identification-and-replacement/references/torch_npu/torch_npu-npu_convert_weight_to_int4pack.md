# torch_npu.npu_convert_weight_to_int4pack

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

将`torch.int32`类型的输入Tensor打包为`torch_npu.int4`存放，每8个`torch_npu.int4`数据通过一个`torch.int32`数据承载，并进行交叠排放。

<!-- npu="950" id11 -->
<term>Ascend 950PR&950DT系列产品</term>：除了上述能力，还支持将`torch.float32`类型输入Tensor打包为`torch_npu.float4_e2m1fn_x2`存放，每8个`torch_npu.float4_e2m1fn_x2`数据通过一个`torch.float32`数据承载，并进行交叠排放。
<!-- end id11 -->

## 函数原型

```python
torch_npu.npu_convert_weight_to_int4pack(weight, inner_k_tiles=0) -> Tensor
```

## 参数说明

- **weight**(`Tensor`)：必选参数，待处理的张量。数据格式支持$ND$、$FRACTAL\_NZ$，使用接口torch_npu.npu_format_cast可以将数据格式转换为$FRACTAL\_NZ$，不支持非连续的Tensor。

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int32`，要求`weight`中元素取值在`torch_npu.int4`的表示范围内，即$[-8, 7]$。维度支持2维，shape支持$(k, n)$、$(n, k)$，最后一维度需要8个元素对齐。
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int32`、`torch.float32`。当输入的数据类型为`torch.int32`时，要求`weight`中元素取值在`torch_npu.int4`的表示范围内，即$[-8, 7]$。当输入数据类型为`torch.float32`时，要求`weight`中元素取值在`torch_npu.float4_e2m1fn_x2`表示的范围内，即$[-6.0, 6.0]$。维度支持2维或3维，shape支持$(k, n)$、$(n, k)$、$(g, k, n)$、$(g, n, k)$，最后一维度需要8个元素对齐。

    > **说明：**
    > - PyTorch 2.7及之前版本：表示在`torch_npu.float4_e2m1fn_x2`范围内使用的是Python中`ml_dtypes`库提供的`float4_e2m1fn`类型，`ml_dtypes`版本要求不小于0.5.0，具体使用参见[调用示例](#调用示例)。
    > - PyTorch 2.8版本开始：可直接使用`torch.float4_e2m1fn_x2`。
  <!-- end id5 -->

- **inner\_k\_tiles**(`int`)：可选参数，用于指定内部打包格式中多少个K-tiles被打包在一起，默认值为`0`。**预留参数，暂未使用**。

## 返回值说明

**out**(`Tensor`)：代表打包后的输出张量，shape为$(k, n/8)$、$(n, k/8)$或$(g, k, n/8)$，数据格式和`weight`一致。

<!-- npu="A3,910b" id6 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int32`。
<!-- end id6 -->
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int32`（对应`weight`为`torch.int32`）、`torch.float32`（对应`weight`为`torch.float32`）。
<!-- end id7 -->

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。

## 调用示例

- 单算子模式调用

  - 输入为`torch.int32`场景

    ```python
    import torch
    import torch_npu

    m = 128
    k = 64
    n = 128
    trans_weight = False

    cpu_x = torch.randn((m, k), dtype=torch.float16)
    if trans_weight:
        cpu_weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32)
        cpu_antiquantscale = torch.randn((n, 1), dtype=torch.float16)
        cpu_antiquantoffset = torch.randn((n, 1), dtype=torch.float16)
    else:
        cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32)
        cpu_antiquantscale = torch.randn((1, n), dtype=torch.float16)
        cpu_antiquantoffset = torch.randn((1, n), dtype=torch.float16)

    weight_int4 = torch_npu.npu_convert_weight_to_int4pack(cpu_weight.npu())

    if trans_weight:
        cpu_weight = cpu_weight.transpose(-1, -2)
        weight_int4 = weight_int4.transpose(-1, -2)
        cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)
        cpu_antiquantoffset = cpu_antiquantoffset.transpose(-1, -2)

    npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), weight_int4, cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())
    ```

  <!-- npu="950" id8 -->
  - 输入为`torch.float32`场景：该示例仅<term>Ascend 950PR&950DT系列产品</term>支持。

    ```python
    import torch
    import torch_npu
    import numpy as np
    # ml_dtypes版本要求不小于0.5.0
    from ml_dtypes import float4_e2m1fn

    m = 128
    k = 64
    n = 128
    E2M1_MIN, E2M1_MAX = -6, 6
    antiquant_group_size = 32
    trans_weight = False

    cpu_x = torch.randn((m, k), dtype=torch.float16)
    if trans_weight:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((n, k))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (n, k // antiquant_group_size), dtype=torch.uint8) # 使用uint8类型承载scale数据
    else:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((k, n))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (k // antiquant_group_size, n), dtype=torch.uint8) # 使用uint8类型承载scale数据

    weight_packed = torch_npu.npu_convert_weight_to_int4pack(cpu_weight.npu())

    if trans_weight:
        weight_packed = weight_packed.transpose(-1, -2)
        cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)

    npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), weight_packed, cpu_antiquantscale.npu(), None, None, None, None, antiquant_group_size)
    ```
  <!-- end id8 -->

- 图模式调用

  - $ND$格式，输入类型为`torch.int32`

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    m = 16
    k = 64
    n = 128
    trans_weight = False
    is_weight_nz = False

    cpu_x = torch.randn((m, k),dtype=torch.float16)
    if trans_weight:
        cpu_weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32)
        cpu_antiquantscale = torch.ones((n, 1),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((n, 1),dtype=torch.float16)
    else:
        cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32)
        cpu_antiquantscale = torch.ones((1, n),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((1, n),dtype=torch.float16)

    npu_weight = cpu_weight.npu()

    # 判断是否进行ND格式转NZ格式
    if is_weight_nz:
        npu_weight = torch_npu.npu_format_cast(npu_weight, 29)
    # int32 to int4pack
    weight_int4pack = torch_npu.npu_convert_weight_to_int4pack(npu_weight)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
            if trans_weight:
                weight  = weight.transpose(-1, -2)
                antiquant_scale = antiquant_scale.transpose(-1, -2)
                antiquant_offset = antiquant_offset.transpose(-1, -2)
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size)

    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)

    npu_out = model(cpu_x.npu(), weight_int4pack, cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
    ```

  - $FRACTAL\_NZ$格式，输入类型为`torch.int32`

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    m = 16
    k = 64
    n = 64
    trans_weight = False
    is_weight_nz = True
    cpu_x = torch.randn((m, k),dtype=torch.float16)

    if trans_weight:
        cpu_weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32)
        cpu_antiquantscale = torch.ones((n, 1),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((n, 1),dtype=torch.float16)
    else:
        cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32)
        cpu_antiquantscale = torch.ones((1, n),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((1, n),dtype=torch.float16)
    npu_weight = cpu_weight.npu()

    # 判断是否进行ND格式转NZ格式
    if is_weight_nz:
        # nd to fractal_nz
            npu_weight = torch_npu.npu_format_cast(npu_weight.npu(), 29, customize_dtype=cpu_x.dtype)

    # int32 to int4pack
    weight_int4pack = torch_npu.npu_convert_weight_to_int4pack(npu_weight)
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
            if trans_weight:
                weight  = weight.transpose(-1, -2)
                antiquant_scale = antiquant_scale.transpose(-1, -2)
                antiquant_offset = antiquant_offset.transpose(-1, -2)
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size)
    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)
    npu_out = model(cpu_x.npu(), weight_int4pack, cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
    ```

  <!-- npu="950" id9 -->
  - $ND$格式，输入类型为`torch.float32`，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    import numpy as np
    from ml_dtypes import float4_e2m1fn

    m = 128
    k = 64
    n = 128
    E2M1_MIN, E2M1_MAX = -6, 6
    antiquant_group_size = 32
    trans_weight = False
    is_weight_nz = False

    cpu_x = torch.randn((m, k),dtype=torch.float16)
    if trans_weight:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((n, k))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (n, k // antiquant_group_size), dtype=torch.uint8) # 使用uint8类型承载scale数据
    else:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((k, n))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (k // antiquant_group_size, n), dtype=torch.uint8) # 使用uint8类型承载scale数据
    npu_weight = cpu_weight.npu()

    # 判断是否进行ND格式转NZ格式
    if is_weight_nz:
        npu_weight = torch_npu.npu_format_cast(npu_weight, 29, customize_dtype=torch.float16)
    weight_packed = torch_npu.npu_convert_weight_to_int4pack(npu_weight)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
            if trans_weight:
                weight  = weight.transpose(-1, -2)
                antiquant_scale = antiquant_scale.transpose(-1, -2)
                antiquant_offset = antiquant_offset.transpose(-1, -2)
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size)

    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)
    npu_out = model(cpu_x.npu(), weight_packed, cpu_antiquantscale.npu(), None, None, None, None, antiquant_group_size)
    ```
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - $FRACTAL\_NZ$格式，输入类型为`torch.float32`，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    import numpy as np
    from ml_dtypes import float4_e2m1fn

    m = 128
    k = 64
    n = 128
    E2M1_MIN, E2M1_MAX = -6, 6
    antiquant_group_size = 32
    trans_weight = False
    is_weight_nz = True

    cpu_x = torch.randn((m, k),dtype=torch.float16)
    if trans_weight:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((n, k))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (n, k // antiquant_group_size), dtype=torch.uint8) # 使用uint8类型承载scale数据
    else:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((k, n))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (k // antiquant_group_size, n), dtype=torch.uint8) # 使用uint8类型承载scale数据
    npu_weight = cpu_weight.npu()

    # 判断是否进行ND格式转NZ格式
    if is_weight_nz:
        npu_weight = torch_npu.npu_format_cast(npu_weight, 29, customize_dtype=torch.float16)
    weight_packed = torch_npu.npu_convert_weight_to_int4pack(npu_weight)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
            if trans_weight:
                weight  = weight.transpose(-1, -2)
                antiquant_scale = antiquant_scale.transpose(-1, -2)
                antiquant_offset = antiquant_offset.transpose(-1, -2)
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size)

    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)
    npu_out = model(cpu_x.npu(), weight_packed, cpu_antiquantscale.npu(), None, None, None, None, antiquant_group_size)
    ```
  <!-- end id10 -->
