# torch_npu.npu_rotate_quant

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

API功能：`npu_rotate_quant`是一种融合旋转（Rotate）和量化（Quant）的计算方法。该方法适用于需要对输入数据进行旋转变换后进行量化的场景，融合算子在底层能够对部分过程并行，达到性能优化的效果。

## 函数原型

```python
torch_npu.npu_rotate_quant(x, rotation, *, alpha=None, dst_dtype=None, axis=-1, round_mode="rint", scale_alg=0, dst_type_max=0.0, transpose_y=False) -> (Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选输入，输入tensor。数据类型支持`torch.bfloat16`和`torch.float16`，数据格式支持ND，支持非连续的Tensor。
- **rotation**（`Tensor`）：必选输入，旋转矩阵tensor。shape支持2维，数据类型支持`torch.bfloat16`和`torch.float16`，数据格式支持ND，支持非连续的tensor。
- **alpha**（`Tensor`）：可选输入，旋转角度缩放因子，数据类型为1维`Tensor`，默认值为None。
- **dst_dtype**（`int`）：可选输入，指定量化输出的类型，传None时当做`torch.int8`处理。支持的量化输出类型包括：`torch.int8`、`torch.quint4x2`、`torch_npu.float4_e2m1fn_x2`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。
- **axis**（`int`）：可选输入，指定量化输出的轴，默认值为-1。
- **round_mode**（`str`）：可选输入，指定取整模式，支持"rint"、"round"、"floor"，默认值为"rint"。
- **scale_alg**（`int`）：可选输入，指定scale算法，默认值为0。
- **dst_type_max**（`float`）：可选输入，指定量化输出的最大值，默认值为0.0。
- **transpose_y**（`bool`）：可选输入，指定输出是否转置，默认值为False，当前版本仅支持False。

## 返回值说明

- **y**（`Tensor`）：输出的量化结果，数据类型根据`dst_dtype`决定。数据格式支持ND，int场景中支持非连续的Tensor。
- **scale**（`Tensor`）：输出的量化因子，数据类型根据`dst_dtype`决定。当`dst_dtype`为MX类型（`torch_npu.float4_e2m1fn_x2`/`torch.float8_e5m2`/`torch.float8_e4m3fn`）时，`scale`为MX格式量化因子（`torch.uint8`表示`torch_npu.float8_e8m0fnu`）；其他类型时，`scale`为pertoken量化因子（`torch.float32`）。

## 约束说明

- 该接口支持推理和训练场景下使用。
- 该接口仅在Atlas A2系列产品和Atlas A3系列产品上支持图模式。
- 对于输入数据的shape，存在以下约束：
    - 当前在Atlas A2系列产品和Atlas A3系列产品上，`rotation`的shape仅支持`[K, K]`一种形式，`K`的取值范围为[16, 1024]；在Ascend 950PR&950DT系列产品平台上，`rotation`的shape支持`[K, K]`和`[block_num, K, K]`两种形式，`K`仅支持32、64、128三种取值, 其中`block_num = N/K`，`N`为输入`x`最后一维的大小。
    - 输入`x`的最后一维需要能被`K`整除。在Atlas A2系列产品和Atlas A3系列产品上，`x`的shape仅支持2维，最后一维的大小需要在128~16000之间且必须同时可被8整除；在Ascend 950PR&950DT系列产品平台上，`x`的shape支持1～7维，且当`dst_dtype`为`torch_npu.float4_e2m1fn_x2`时，输入`x`的最后一维必须同时可被2整除。
- 对于输入数据的取值范围，存在以下约束：   
    - `dst_dtype`在Atlas A2系列产品和Atlas A3系列产品上仅支持`torch.int8`和`torch.quint4x2`；在Ascend 950PR&950DT系列产品平台上仅支持`torch_npu.float4_e2m1fn_x2`、`torch.float8_e5m2`和`torch.float8_e4m3fn`。
    - `alpha`在Atlas A2系列产品和Atlas A3系列产品上仅支持传入None，无实际功能；在Ascend 950PR&950DT系列产品平台上取值范围为(0.0, 1.0)，传入None或不在有效取值范围内时不做处理。
    - `axis`在Atlas A2系列产品和Atlas A3系列产品上仅支持-1或1；在Ascend 950PR&950DT系列产品平台上仅支持-1或D-1，D为输入`x`的维度数。
    - 在Ascend 950PR&950DT系列产品平台上，当`dst_dtype`为`torch.float8_e5m2`或`torch.float8_e4m3fn`时，`round_mode`仅支持"rint"。
    - `scale_alg`在Atlas A2系列产品和Atlas A3系列产品上仅支持0；在Ascend 950PR&950DT系列产品平台上支持取值0、1、2，其中当`dst_dtype`为`torch.float8_e5m2`或`torch.float8_e4m3fn`时仅支持0和1，当`dst_dtype`为`torch_npu.float4_e2m1fn_x2`时仅支持0和2。
    - `dst_type_max`在Atlas A2系列产品和Atlas A3系列产品上仅支持0.0；在Ascend 950PR&950DT系列产品平台上，当`scale_alg`为2时取值范围为[6.0, 12.0]，其余场景仅支持0.0。
- `x`和`rotation`的数据类型必须一致。

## 调用示例

- 单算子模式调用

    ```python
    import numpy as np
    import torch
    import torch_npu

    def gen_input_data(M, N, K):
        x = torch.randn(M, N, dtype=torch.bfloat16)
        rotation = torch.randn(K, K, dtype=torch.bfloat16)
        return x, rotation

    M = 512
    N = 1024
    K = 128
    x, rotation = gen_input_data(M, N, K)
    # int8 quantization, only supported on Atlas A2/A3
    output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=torch.int8)
    # mxfp4 quantization, only supported on Ascend 950PR&950DT系列产品
    output0_npu, output1_npu = torch_npu.npu_rotate_quant(
        x.npu(), rotation.npu(), dst_dtype=torch_npu.float4_e2m1fn_x2, axis=-1, round_mode="rint"
    )
    ```

<!-- npu="A3,910b" id4 -->
- 图模式调用（仅在Atlas A2系列产品和Atlas A3系列产品上支持）：

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, rotation):
            output = torch_npu.npu_rotate_quant(x, rotation, dst_dtype=torch.int8)
            return output

    def gen_input_data(M, N, K):
        x = torch.randn(M, N, dtype=torch.bfloat16)
        rotation = torch.randn(K, K, dtype=torch.bfloat16)
        return x, rotation

    M = 512
    N = 1024
    K = 1024
    x, rotation = gen_input_data(M, N, K)

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), rotation.npu())
    ```
<!-- end id4 -->
