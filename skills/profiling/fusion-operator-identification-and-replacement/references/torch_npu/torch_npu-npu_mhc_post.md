# torch\_npu.npu\_mhc\_post

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：对mHC\(Manifold-Constrained Hyper-Connections\)架构中第$l$层输出$h_l^{out}$进行Post Mapping，对第$l$层的输入$x_l$进行Res Mapping，然后对二者进行残差连接，得到第$(l+1)$层的输入$x_{l+1}$。当`h_res`传入`None`时，跳过Res Mapping，直接进行残差相加。
- 计算公式：

    当`h_res`提供时：

    $$ x_{l+1} = (H_l^{res}) \times x_l + h_l^{out} \otimes H_l^{post} $$

    当`h_res`缺省（传入`None`）时：

    $$ x_{l+1} = x_l + h_l^{out} \otimes H_l^{post} $$

## 函数原型

```python
torch_npu.npu_mhc_post(x, h_res, h_out, h_post) -> Tensor
```

## 参数说明

- **x**（`Tensor`）：必选参数，待计算的张量，表示网络中mHC层的输入数据，数据类型支持`torch.bfloat16`、`torch.float16`，shape为\(B, S, n, D\)或\(T, n, D\)，数据格式支持ND，支持非连续Tensor，支持空Tensor。
- **h\_res**（`Tensor`）：可选参数，默认值为`None`，mHC的hRes变换矩阵，是做完Sinkhorn变换后的双随机矩阵，数据类型支持`torch.float32`，shape为\(B, S, n, n\)或\(T, n, n\)，数据格式支持ND，支持非连续Tensor，支持空Tensor。传入`None`时跳过Res Mapping，计算公式退化为$x_{l+1} = x_l + h_l^{out} \otimes H_l^{post}$。
- **h\_out**（`Tensor`）：必选参数，Atten/MLP层的输出，数据类型与x相同，shape为\(B, S, D\)或\(T, D\)，数据格式支持ND，支持非连续Tensor，支持空Tensor。
- **h\_post**（`Tensor`）：必选参数，mHC的hPost变换矩阵，数据类型支持`torch.float32`，shape为\(B, S, n\)或\(T, n\)，数据格式支持ND，支持非连续Tensor，支持空Tensor。

## 返回值说明

**y**（`Tensor`）：必选输出，网络中mHC层的输出数据，作为下一层的输入，数据类型与`x`相同，shape为\(B, S, n, D\)或\(T, n, D\)，数据格式支持ND。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。

## 调用示例

- 单算子模式调用

    ```python
    import torch 
    import torch_npu
    import numpy as np
    print(torch.npu.is_available())
    # 检查 NPU 可用性
    assert torch.npu.is_available(), "NPU not available"
    print("get npu number.")
    num_npus = torch.npu.device_count()
    print("Number of NPUs:", num_npus)
    x_shape = [1,4,512]
    h_res_shape = [1,4,4]
    h_out_shape = [1,512]
    h_post_shape = [1,4]
    x = torch.rand(x_shape, dtype=torch.float16)
    h_res = torch.rand(h_res_shape, dtype=torch.float32)
    h_out = torch.rand(h_out_shape, dtype=torch.float16)
    h_post = torch.rand(h_post_shape, dtype=torch.float32)
    x_npu = x.npu()
    h_res_npu = h_res.npu()
    h_out_npu = h_out.npu()
    h_post_npu = h_post.npu()
    y_npu = torch_npu.npu_mhc_post(x_npu, h_res_npu, h_out_npu, h_post_npu)
    # h_res缺省时:
    # y_npu = torch_npu.npu_mhc_post(x_npu, None, h_out_npu, h_post_npu)
    ```

- TorchAir图模式调用（aclgraph）

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    from torchair.core.utils import logger
    import os
    import logging
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    device=torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    class MhcPostModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, h_res, h_out, h_post):
            y = torch.ops.npu.npu_mhc_post(x, h_res, h_out, h_post)
            return y
    x_shape = [1,1,4,512]
    h_res_shape = [1,1,4,4]
    h_out_shape = [1,1,512]
    h_post_shape = [1,1,4]
    x = torch.rand(x_shape, device='npu', dtype=torch.float16)
    h_res = torch.rand(h_res_shape, device='npu', dtype=torch.float32)
    h_out = torch.rand(h_out_shape, device='npu', dtype=torch.float16)
    h_post = torch.rand(h_post_shape, device='npu', dtype=torch.float32)
    mhc_post_model = MhcPostModel().npu()
    mhc_post_model = torch.compile(mhc_post_model, backend=npu_backend)
    y = mhc_post_model(x, h_res, h_out, h_post)
    ```
