# torch_npu.npu_all_to_all_matmul

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- **API功能**：完成AlltoAll通信、Permute（保证通信后地址连续）和Matmul计算的融合，先通信后计算。

- **计算公式**：

    假设输入x1的shape为\(BS, H\)，x2的shape为\(H\*rankSize, N\)，rankSize为NPU卡数。

    $$
    commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
    permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
    output = permutedOut @ x2 + bias \\
    $$

## 函数原型

```python
torch_npu.npu_all_to_all_matmul(x1, x2, hcom, world_size, bias=None, all2all_axes=None, all2all_out_flag=True, comm_mode=None) -> (Tensor, Tensor)
```

## 参数说明

- **x1**（`Tensor`）：**必选参数**，表示融合算子的左矩阵输入，对应公式中的$x1$。该输入进行AlltoAll通信与Permute操作后，结果作为Matmul计算的左矩阵输入。数据类型支持`torch.bfloat16`、`torch.float16`，维度只能为2D，shape为\(BS, H\)，数据格式支持$ND$，不支持非连续Tensor，支持第一维度为0的空Tensor。
- **x2**（`Tensor`）：**必选参数**，表示融合算子的右矩阵输入，也是Matmul计算的右矩阵，对应公式中的$x2$。数据类型与`x1`一致，维度只能为2D，shape为\(H\*rankSize, N\)，数据格式支持$ND$，支持转置非连续Tensor。
- **hcom**（`str`）：**必选参数**，Host侧标识列组的字符串，即通信域名称，通过`get_hccl_comm_name`接口获取。
- **world_size**（`int`）：**必选参数**，通信域内的rank总数，对应公式中的$rankSize$，支持范围\[2, 4, 8, 16\]，CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。
- **bias**（`Tensor`）：**可选参数**，矩阵乘运算后累加的偏置，对应公式中的$bias$。数据类型由输入`x1`和`x2`决定，当`x1`和`x2`为`torch.float16`时，`bias`的数据类型为`torch.float16`和`torch.float32`；当`x1`和`x2`为`torch.bfloat16`时，`bias`的数据类型为`torch.bfloat16`和`torch.float32`。维度只能为1D，shape为\(N\)，数据格式支持$ND$。
- **all2all_axes**（`List[int]`）：**可选参数**，AlltoAll和Permute数据交换的方向，支持为空或者\[-2, -1\]，表示将输入`x1`由\(BS, H\)转为\(BS/rankSize, rankSize\*H\)。
- **all2all_out_flag**（`bool`）：**可选参数**，表示是否输出AlltoAll和Permute后的结果，默认为`True`。
- **comm_mode**（`str`）：**可选参数**，表示通信引擎模式，默认值为`None`。

    <!-- npu="A3" id3 -->
    - <term>Atlas A3系列产品</term>：取值支持`None`、`ai_cpu`。传入`None`时，使用`ai_cpu`通信。
    <!-- end id3 -->
    <!-- npu="950" id4 -->
    - <term>Ascend 950PR&950DT系列产品</term>：取值支持`None`、`ai_cpu`、`ccu`。传入`None`时，使用`ai_cpu`通信。
    <!-- end id4 -->

## 返回值说明

- **y**（`Tensor`）：表示最终的计算结果，公式中的$output$，数据类型与计算输入`x1`或`x2`类型一致，维度为2D，shape为\(BS/rankSize, N\)，数据格式支持$ND$，不支持非连续的Tensor。
- **all2all_out**（`Tensor`）：表示AlltoAll和Permute后的结果，公式中的$permutedOut$。当`all2all_out_flag`为`True`时输出实际Tensor，数据类型与计算输入`x1`的类型一致，维度为2D，shape为\(BS/rankSize, H\*rankSize\)，数据格式支持$ND$，不支持非连续的Tensor；当`all2all_out_flag`为`False`时输出`None`。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式。

<!-- npu="950" id5 -->
- 该接口在<term>Ascend 950PR&950DT系列产品</term>产品上支持图模式调用。
<!-- end id5 -->
- 除`x1`以外的输入参数均不支持空Tensor。
- 通信域名称`hcom`不支持传入空字符串，长度取值范围为\[1, 127\]。
- 参数里Shape使用的变量如下：
    - BS：输入左矩阵的第一维度大小，表示输入序列sequence的条数，取值范围\[0, 2147483647\]，必须整除rankSize。
    - H：输入左矩阵的第二维度大小，表示隐藏层维度，取值范围受NPU卡数限制，H\*rankSize取值范围为\[2, 65535\]。
    - H\*rankSize：输入右矩阵第一维度大小，表示输入左矩阵经过AlltoAll通信后隐藏层维度大小，取值范围为\[2, 65535\]。
    - N：输入右矩阵第二维度大小，表示输出序列sequence的长度，取值范围为\[1, 2147483647\]。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np

    def run_all_to_all_matmul(rank, world_size, master_ip, master_port, x1_shape, x2_shape):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        x1_tensor = torch.randn(x1_shape, dtype=torch.float16).npu()
        x2_tensor = torch.randn(x2_shape, dtype=torch.float16).npu()
        output, all2all_out = torch_npu.npu_all_to_all_matmul(
            x1_tensor,
            x2_tensor,
            hcom_info,
            world_size
        )
        print("output: ", output)
        print("all2all_out: ", all2all_out)


    if __name__ == "__main__":
        worksize = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [1024, 256]
        x2_shape =  [512, 3072]

        mp.spawn(
            run_all_to_all_matmul,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape),
            nprocs=worksize,
        )
    ```

<!-- npu="950" id6 -->
- 图模式调用：仅适用于<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    config = CompilerConfig()
    config.debug.graph_dump.type = 'pbtxt'
    # npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_backend = tng.get_npu_backend(compiler_config=None)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, hcom, world_size):
            return torch_npu.npu_all_to_all_matmul(x1, x2, hcom, world_size)

    def run_all_to_all_matmul_graph(rank, world_size, master_ip, master_port, x1_shape, x2_shape):
        # get hcom_info
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        # generate input tensor
        x1_tensor = torch.randn(x1_shape, dtype=torch.float16).npu()
        x2_tensor = torch.randn(x2_shape, dtype=torch.float16).npu()
        cpu_model = Model()
        # dynamic = False -> static graph; dynamic = True -> dynamic graph
        model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
        # model = torch.compile(cpu_model, backend=npu_backend, dynamic=False)
        output = model(
            x1_tensor,
            x2_tensor,
            hcom_info,
            world_size
        )
        print("print output: ", output)
    if __name__ == "__main__":
        world_size = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [88, 128]
        x2_shape = [256, 256]
        mp.spawn(
            run_all_to_all_matmul_graph,
            args=(world_size, master_ip, master_port, x1_shape, x2_shape),
            nprocs=world_size,
        )
    ```
<!-- end id6 -->
