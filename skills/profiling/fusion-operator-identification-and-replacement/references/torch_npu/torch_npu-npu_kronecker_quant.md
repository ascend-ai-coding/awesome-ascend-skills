# torch_npu.npu_kronecker_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：推理场景下，对输入张量依次进行两次矩阵乘法，并对矩阵乘结果量化至`torch_npu.int4`类型或`torch_npu.float4_e2m1fn_x2`类型。

  pertoken量化支持`torch_npu.int4`输出类型，并以8个一组打包成`torch.int32`类型输出，同时也输出`torch.float32`的量化缩放系数。

  pergroup量化支持输出`torch_npu.float4_e2m1fn_x2`量化结果和`torch_npu.float8_e8m0fnu`量化系数，组内的元素共享一个量化参数。

- **pertoken计算公式**：

    1. 输入`x`右乘`kroneckerP2`：

        $$
        x' = x @ kroneckerP2
        $$

    2. `kroneckerP1`左乘`x'`：

        $$
        x'' = kroneckerP1 @ x'
        $$

    3. 沿着$x''$的0维计算最大绝对值并除以$(7 / clipRatio)$，以计算需量化为`torch_npu.int4`格式的量化缩放系数`quantScale`：

        $$
        quantScale = \frac{[\max(\operatorname{abs}(x''[0, :, :])),\ \max(\operatorname{abs}(x''[1, :, :])),\ \ldots,\ \max(\operatorname{abs}(x''[K, :, :]))]}{7 / clipRatio}
        $$

    4. 计算输出的`out`：

        $$
        out = x'' / quantScale
        $$

- **pergroup计算公式**：

    1. 输入`x`右乘`kroneckerP2`：

        $$
        x' = x @ kroneckerP2
        $$

    2. `kroneckerP1`左乘`x'`：

        $$
        x'' = kroneckerP1 @ x'
        $$

    3. $x''$进行pergroup量化需转换shape，记为`x2`。形如[K,M,N]转换成[K,M*N]。沿着$x''$的第二个维度进行pergroup量化。一个group中包含元素对应的指数$e_0, e_1, \ldots, e_{31}$。计算$emax$：

        $$
        emax = \max(e_0, e_1, \ldots, e_{31})
        $$

    4. 计算reduceMaxValue和sharedExp：

        $$
        reduceMaxValue = \log_2(\operatorname{reduceMax}(x2)),\ groupsize = 32
        $$

        $$
        sharedExp = reduceMaxValue - emax
        $$

    5. 计算quantScale：

        $$
        quantScale = 2^{sharedExp}
        $$

    6. 每blocksize共享一个quantscale并计算out：

        $$
        out = x2 / quantScale
        $$

## 函数原型

```python
torch_npu.npu_kronecker_quant(x, kronecker_p1, kronecker_p2, clip_ratio=None, dst_dtype=None, dst_type_max=None) -> (Tensor, Tensor)
```

## 参数说明

- **x**(`Tensor`)：必选参数，需要做量化的源数据张量，对应公式中的`x`。数据类型支持`torch.float16`、`torch.bfloat16`，shape为(K, M, N)，数据格式支持$ND$。
- **kronecker_p1**(`Tensor`)：必选参数，`x`的左乘矩阵，对应公式中的`kroneckerP1`。数据类型与`x`一致，shape为(M, M)，数据格式支持$ND$。
- **kronecker_p2**(`Tensor`)：必选参数，`x`的右乘矩阵，对应公式中的`kroneckerP2`。数据类型与`x`一致，shape为(N, N)（`dst_dtype`为`torch_npu.float4_e2m1fn_x2`情况时，shape支持为(0,0)），数据格式支持$ND$。
- **clip_ratio**(`float`)：**可选参数**，量化的裁剪比例，对应公式中的`clipRatio`。数据类型支持`float`，取值范围为(0, 1.0]，默认值为None表示按1.0比例裁剪。
- **dst_dtype**(`ScalarType`)：**可选参数**，量化的目标输出类型。如果是`torch_npu.float4_e2m1fn_x2`类型输出时该值为`torch_npu.float4_e2m1fn_x2`。`torch.int`类型输出时该项不用填写。
- **dst_type_max**(`float`)：**可选参数**，表示量化数据目标的最大值。数据类型支持`float`，取值范围为0.0、6.0-12.0，取值为0.0代表不使用该参数；取值为6.0-12.0代表目标数据类型的最大值。仅支持在`torch_npu.float4_e2m1fn_x2`数据类型时设置该值。

## 返回值说明

- **out**(`Tensor`)：量化后的输出，对应公式中的`out`。数据格式支持$ND$。
  - pertoken量化方式：数据类型支持`torch.int32`，shape为(K, M, N / 8)。
  - pergroup量化方式：数据类型支持`torch_npu.float4_e2m1`，shape为(K, M*N)。

- **quant_scale**(`Tensor`)：量化缩放系数，对应公式中的`quantScale`。数据格式支持$ND$。
  - pertoken量化方式：数据类型支持`torch.float32`，shape为(K,)。
  - pergroup量化方式：数据类型支持`torch_npu.float8_e8m0fnu`，shape为(K, ceilDiv(M*N, 64), 2)。比如输入的shape是(16, 128, 64)，`out`的shape是(16, 8192)，`quant_scale`的shape是(16, 128, 2)。

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口仅支持单算子模式和TorchAir图模式调用。
- 参数说明里Shape使用的变量说明：
  - K：输入张量`x`的首维大小，取值范围为[1, 262144]。
  - M：输入张量`x`的中间维大小，取值范围为[1, 256]。
  - N：输入张量`x`的最后一维大小，取值范围为[1, 256]，N为偶数。如果`out`输出类型是`torch.int32`，则N必须为8的整数倍。

## 调用示例

- 单算子模式调用

  - 输出为`torch_npu.int4`类型

    ```python
    import torch
    import torch_npu

    K = 16
    M = 64
    N = 64
    x = torch.randn(K, M, N).half().npu()
    p1 = torch.randn(M, M).half().npu()
    p2 = torch.randn(N, N).half().npu()
    clip_ratio = 1.0
    out, quant_scale = torch_npu.npu_kronecker_quant(x, p1, p2, clip_ratio)
    print(out)
    print(quant_scale)
    ```

  - 输出为`torch_npu.float4_e2m1fn_x2`类型

    ```python
    import torch
    import torch_npu
    K = 16
    M = 64
    N = 64
    x = torch.randn(K, M, N).half().npu()
    p1 = torch.randn(M, M).half().npu()
    p2 = torch.randn(N, N).half().npu()
    clip_ratio = 1.0
    dst_type_max = 0.0
    dst_dtype = torch_npu.float4_e2m1fn_x2
    out, quant_scale = torch_npu.npu_kronecker_quant(x, p1, p2, clip_ratio, dst_dtype, dst_type_max)
    print(out.cpu())
    print(quant_scale.cpu())
    ```

- 图模式调用

  - 输出为`torch_npu.int4`类型

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, p1, p2, clip_ratio):
                return torch_npu.npu_kronecker_quant(x, p1, p2, clip_ratio)

    K = 16
    M = 64
    N = 64
    x = torch.randn(K, M, N).half().npu()
    p1 = torch.randn(M, M).half().npu()
    p2 = torch.randn(N, N).half().npu()
    clip_ratio = 1.0

    module = torch.compile(Module().npu(), backend=npu_backend)
    out, quant_scale = module(x, p1, p2, clip_ratio)
    print(out)
    print(quant_scale)
    ```

  - 输出为`torch_npu.float4_e2m1fn_x2`类型

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, p1, p2, clip_ratio):
                return torch_npu.npu_kronecker_quant(x, p1, p2, clip_ratio, torch_npu.float4_e2m1fn_x2, dst_type_max)

    K = 16
    M = 64
    N = 64
    x = torch.randn(K, M, N).half().npu()
    p1 = torch.randn(M, M).half().npu()
    p2 = torch.randn(N, N).half().npu()
    clip_ratio = 1.0
    dst_type_max = 0.0

    module = torch.compile(Module().npu(), backend=npu_backend)
    out, quant_scale = module(x, p1, p2, clip_ratio)
    print(out.cpu())
    print(quant_scale.cpu())
    ```
