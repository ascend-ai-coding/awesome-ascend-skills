# torch\_npu.npu\_mhc\_sinkhorn

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：对mHC架构中的H′<sub>res</sub>矩阵（即网络中mHC层的输入数据）执行Sinkhorn迭代归一化变换，最终得到双随机矩阵H<sub>res</sub>，支持输出迭代过程中的中间归一化结果（norm\_out）和求和结果（sum\_out），用于反向梯度计算。
- 计算公式：
    1. 在第一次迭代即初始化时：

        $$
        \begin{aligned}
        \operatorname{norm\_out}[0] &= \operatorname{softmax}(x, \dim=-1) + \varepsilon\\
        \operatorname{sum\_out}[1] &= \sum_{\dim=-2,keepdim=True} \operatorname{norm\_out}[0] + \varepsilon\\
        \operatorname{norm\_out}[1] &= \frac{\operatorname{norm\_out}[0]}{\operatorname{sum\_out}[1]}
        \end{aligned}
        $$

    2. 第i次迭代\(i=1,2,..., \(num\_iters-1\)\)：

        $$
        \begin{aligned}
        \operatorname{sum\_out}[2i] &= \sum_{\dim=-1,keepdim=True} \operatorname{norm\_out}[2i-1] + \varepsilon\\
        \operatorname{norm\_out}[2i] &= \frac{\operatorname{norm\_out}[2i-1]}{\operatorname{sum\_out}[2i]}\\
        \operatorname{sum\_out}[2i+1] &= \sum_{\dim=-2,keepdim=True} \operatorname{norm\_out}[2i] + \varepsilon\\
        \operatorname{norm\_out}[2i+1] &= \frac{\operatorname{norm\_out}[2i]}{\operatorname{sum\_out}[2i+1]}
        \end{aligned}
        $$

    3. 最终输出:

        $$
        \operatorname{output} = \operatorname{norm\_out}[2 \times num\_iters - 1]
        $$

## 函数原型

```python
torch_npu.npu_mhc_sinkhorn(x, eps=1e-6, num_iters=20, out_flag=0) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选参数，待计算的张量，表示网络中mHC层的输入数据，数据类型支持`torch.float32`，shape为\[B, S, n, n\]或\[T, n, n\]，数据格式支持ND，支持非连续Tensor，不支持空Tensor。shape的n仅支持4、6、8三个值。
- **eps**（float）：可选参数，归一化防除零的参数，数据类型支持`torch.float32`，默认值为1e-6。
- **num\_iters**（`int`）：可选参数，指代迭代次数，默认值20，数据范围\[1,100\]。
- **out\_flag**（`int`）：可选参数，决定是否输出中间结果。支持取值0（默认）和1，当值为0时，仅输出最终变换结果；当值为1时，输出y+normOut+sumOut。

## 返回值说明

- **y**（`Tensor`）：必选输出，MhcSinkhorn变换最终结果，数据类型支持`torch.float32`，shape为\[B, S, n, n\]或者\[T, n, n\]，数据格式支持ND，支持非连续Tensor。要求维度与输入`x`一致。shape的n仅支持4、6、8三个值。
- **norm\_out**（`Tensor`）：可选输出，迭代过程中的归一化中间结果，数据类型支持`torch.float32`，shape为\[2numIters, n, n, B, S\]或者\[2numIters, n, n, T\]，数据格式支持ND，支持非连续Tensor。当且仅当`out_flag`为1时有效。在训练及推理prefill时，B\*S支持\[512, 65536\], \[1, 512\]，在推理场景下，B\*S支持\[1, 512\]。

- **sumOut**（`Tensor`）：可选输出，迭代过程中的求和中间结果，数据类型支持`torch.float32`，shape为\[2numIters, n, B, S\]或者\[2numIters, n, T\]，数据格式支持ND，支持非连续Tensor。当且仅当`out_flag`为1时有效。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    device=torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    
    x_shape = [1, 128, 4 , 4]
    x = torch.rand(x_shape, dtype=torch.float32).clamp(min=1e-4)
    x_npu = x.npu()
    eps = 1e-6
    num_iters = 20
    out_flag = 0
    y = torch_npu.npu_mhc_sinkhorn(x_npu, eps=eps, num_iters=num_iters, out_flag=out_flag)
    ```

- TorchAir图模式调用（aclgraph）

    ```python
    import torch
    import torch_npu
    import torchair as tng
    config = tng.CompilerConfig()
    config.mode="reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    device=torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    
    class MhcSinkhornModel(torch.nn.Module):
        def __init__(self):
                super().__init__()
        def forward(self, x, eps, num_iters, out_flag):
                y = torch_npu.npu_mhc_sinkhorn(x, eps=eps, num_iters=num_iters, out_flag=out_flag)
                return y
    
    x_shape = [1, 128, 4 , 4]  
    x = torch.rand(x_shape, device="npu", dtype=torch.float32)  
    eps = 1e-6
    num_iters = 20
    out_flag = 0
    mhc_sinkhorn_model = MhcSinkhornModel().npu()
    mhc_sinkhorn_model = torch.compile(mhc_sinkhorn_model, backend=npu_backend)
    y = mhc_sinkhorn_model(x, eps=eps, num_iters=num_iters, out_flag=out_flag)
    ```
