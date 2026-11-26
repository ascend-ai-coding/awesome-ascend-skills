# torch_npu.npu_alltoallv_gmm

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- **API功能**：MoE（Mixture of Experts，混合专家模型）网络中，完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，先通信后计算。

- **计算公式**：
    - 路由专家：
        $$
        ata\_out = AlltoAllv(gmm\_x)
        $$

        $$
        permute\_out = Permute(ata\_out)
        $$

        $$
        gmm\_y = permute\_out \times gmm\_weight
        $$

        - ata\_out是gmm\_x进行AlltoAllv通信的输出结果，后续用于Permute计算。
        - permute\_out是ata\_out进行Permute计算的输出结果，作为路由专家进行GroupedMatMul计算的左矩阵。
        - gmm\_weight指路由专家进行GroupedMatMul计算的右矩阵。
        - gmm\_y指路由专家进行GroupedMatMul计算的输出。

    - 共享专家：
        $$
        mm\_y = mm\_x \times mm\_weight
        $$

        - mm\_x指共享专家MatMul计算的左矩阵。
        - mm\_weight指共享专家MatMul计算的右矩阵。
        - mm\_y指共享专家MatMul计算的输出。

## 函数原型

```python
torch_npu.npu_alltoallv_gmm(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, trans_gmm_weight=False, trans_mm_weight=False, permute_out_flag=False, comm_mode=None) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **gmm\_x**（`Tensor`）：**必选参数**，该输入进行AlltoAllv通信与Permute操作后结果作为GroupedMatMul计算的左矩阵。数据类型支持`torch.float16`、`torch.bfloat16`，支持2维，shape为\(BSK, H1\)，数据格式支持$ND$。
- **gmm\_weight**（`Tensor`）：**必选参数**，GroupedMatMul计算的右矩阵。数据类型与`gmm_x`保持一致，支持3维，shape为\(e, H1, N1\)，数据格式支持$ND$。
- **hcom**（`str`）：**必选参数**，专家并行的通信域名，字符串长度要求\(0, 128\)。
- **ep\_world\_size**（`int`）：**必选参数**，EP通信域size，CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。

    <!-- npu="A3" id3 -->
    - <term>Atlas A3系列产品</term>：取值支持8、16、32、64、128。
    <!-- end id3 -->
    <!-- npu="950" id4 -->
    - <term>Ascend 950DT系列产品</term>：取值支持2、4、8、16、32、64。
    <!-- end id4 -->

- **send\_counts**（`List[int]`）：**必选参数**，表示发送给其他卡的token数，数据类型支持int，取值大小为e\*`ep_world_size`，最大为256。输入类型需为list。
- **recv\_counts**（`List[int]`）：**必选参数**，表示接收其他卡的token数，数据类型支持int，取值大小为e\*`ep_world_size`，最大为256。输入类型需为list。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **send\_counts\_tensor**（`Tensor`）：**可选参数**，数据类型支持`torch.int`，shape为\(e\*ep\_world\_size,\)，数据格式支持$ND$。**当前版本暂不支持**，使用默认值即可。
- **recv\_counts\_tensor**（`Tensor`）：**可选参数**，数据类型支持`torch.int`，shape为\(e\*ep\_world\_size,\)，数据格式支持$ND$。**当前版本暂不支持**，使用默认值即可。
- **mm\_x**（`Tensor`）：**可选参数**，共享专家MatMul计算中的左矩阵。当需要融合共享专家矩阵计算时，该参数必选，数据类型支持`torch.float16`、`torch.bfloat16`，支持2维，shape为\(BS, H2\)。
- **mm\_weight**（`Tensor`）：**可选参数**，共享专家MatMul计算中的右矩阵。当需要融合共享专家矩阵计算时，该参数必选，数据类型与`mm_x`保持一致，支持2维，shape为\(H2, N2\)。
- **trans\_gmm\_weight**（`bool`）：**可选参数**，GroupedMatMul的右矩阵是否需要转置，`true`表示需要转置，`false`表示不转置（默认值）。
- **trans\_mm\_weight**（`bool`）：**可选参数**，共享专家MatMul的右矩阵是否需要转置，`true`表示需要转置，`false`表示不转置（默认值）。
- **permute\_out\_flag**（`bool`）：**可选参数**，Permute结果是否需要输出，`true`表明需要输出，`false`表明不需要输出（默认值）。
- **comm\_mode**（`str`）：**可选参数**，表示通信引擎模式，默认值为`None`。

    <!-- npu="A3" id5 -->
    - <term>Atlas A3系列产品</term>：取值支持`None`、`ai_cpu`。传入任意值都将使用AI CPU通信。
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950DT系列产品</term>：取值支持`None`、`ai_cpu`、`ccu`。当为`None`时，使用AI CPU通信。
    <!-- end id6 -->

## 返回值说明

- **gmm\_y**（`Tensor`）：计算输出，表示最终的计算结果，数据类型与输入`gmm_x`保持一致，支持2维，shape为\(A, N1\)。
- **mm\_y**（`Tensor`）：计算输出，共享专家MatMul的输出，数据类型与`mm_x`保持一致，支持2维，shape为\(BS, N2\)。仅当传入`mm_x`与`mm_weight`才输出。
- **permute\_out**（`Tensor`）：计算输出，Permute之后的输出，数据类型与`gmm_x`保持一致。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- **单卡通信量要求**：

  <!-- npu="A3" id7 -->
  - <term>Atlas A3系列产品</term>：单卡通信量大于等于2MB。
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950DT系列产品</term>：单卡通信量无约束。
  <!-- end id8 -->

- 输入参数Tensor中shape使用的变量说明：
  - BSK：本卡发送的token数（BS\*K=BSK），是send\_counts参数累加之和，取值范围\(0, 52428800\)。
  - H1：表示路由专家hidden size隐藏层大小，取值范围\(0, 65536\)。
  - H2：表示共享专家hidden size隐藏层大小，取值范围\(0, 12288\]。
  - e：表示单卡上专家个数，e<=32，e\*ep\_world\_size最大支持256。
  - N1：表示路由专家的head\_num，取值范围\(0, 65536\)。
  - N2：表示共享专家的head\_num，取值范围\(0, 65536\)。
  - BS：表示batch sequence size。
  - K：表示选取topK个专家，K的范围\[2, 8\]。
  - A：本卡收到的token数，是recv\_counts参数累加之和。
  - EP通信域内所有卡上的A累加和等于所有卡上的BSK累加和。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def run_npu_alltoallv_gmm(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)

        input = torch.randn(gmm_x, dtype=dtype).npu()
        weight = torch.randn(gmm_w, dtype=dtype).npu()

        print(torch_npu.npu_alltoallv_gmm(gmm_x =input,
                                                gmm_weight = weight,
                                                hcom= hcom_info,
                                                ep_world_size = ep_world_size,
                                                send_counts = list(send_counts),
                                                recv_counts = list(recv_counts),
                                                send_counts_tensor = None, #send_counts_tensor,
                                                recv_counts_tensor = None, #recv_counts_tensor,
                                                mm_x = None,
                                                mm_weight = None,
                                                trans_gmm_weight = False,
                                                trans_mm_weight  = False,
                                                permute_out_flag  = False))

    if __name__ == "__main__":
        epWorkSize = 8
        e = 4
        master_ip = '127.0.0.1'
        master_port = '50001'
        BS = 512
        K = 8
        gmm_x_shape = [BS*K, 2048]
        gmm_weight_shape = [e, 2048, 2048]
        send_counts = [128] * (e * epWorkSize)
        recv_counts = [128] * (e * epWorkSize)
        dtype = torch.float16
        mp.spawn(run_npu_alltoallv_gmm, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torchair

    class ALLTOALLV_GMM_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,gmm_x, gmm_weight,
                    hcom, ep_world_size,
                    send_counts, recv_counts, send_counts_tensor, recv_counts_tensor,
                    mm_x, mm_weight,
                    trans_gmm_weight, trans_mm_weight, permute_out_flag):
            return torch_npu.npu_alltoallv_gmm(gmm_x =gmm_x,
                                                gmm_weight = gmm_weight,
                                                hcom= hcom,
                                                ep_world_size = ep_world_size,
                                                send_counts = list(send_counts),
                                                recv_counts = list(recv_counts),
                                                send_counts_tensor = None, #send_counts_tensor,
                                                recv_counts_tensor = None, #recv_counts_tensor,
                                                mm_x = mm_x,
                                                mm_weight = mm_weight,
                                                trans_gmm_weight = trans_gmm_weight,
                                                trans_mm_weight  = trans_mm_weight,
                                                permute_out_flag  = permute_out_flag)

    def run_npu_alltoallv_gmm(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)

        input = torch.randn(gmm_x, dtype=dtype).npu()
        weight = torch.randn(gmm_w, dtype=dtype).npu()

        model = ALLTOALLV_GMM_GRAPH_Model()
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        # 静态图：dynamic=False；动态图：dynamic=True
        model = torch.compile(ALLTOALLV_GMM_GRAPH_Model(), backend=npu_backend, dynamic=False)
        print(model(gmm_x=input,
                        gmm_weight=weight,
                        send_counts_tensor=None,
                        recv_counts_tensor=None,
                        mm_x=None,
                        mm_weight=None,
                        hcom=hcom_info,
                        ep_world_size=ep_world_size,
                        send_counts=send_counts,
                        recv_counts=recv_counts,
                        trans_gmm_weight=False,
                        trans_mm_weight=False,
                        permute_out_flag=True))

    if __name__ == "__main__":
        epWorkSize = 8
        e = 4
        master_ip = '127.0.0.1'
        master_port = '50001'
        BS = 512
        K = 8
        gmm_x_shape = [BS*K, 2048]
        gmm_weight_shape = [e, 2048, 2048]
        send_counts = [128] * (e * epWorkSize)
        recv_counts = [128] * (e * epWorkSize)
        dtype = torch.float16
        mp.spawn(run_npu_alltoallv_gmm, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
    ```

## 参考资源

| 术语 | 定义 |
| --- | --- |
| **MoE**（Mixture of Experts，混合专家模型） | 一种神经网络架构，将前馈层替换为多个“专家”子网络，通过门控机制为每个输入token动态选择部分专家进行计算，从而在扩大模型容量的同时控制计算成本。 |
| **路由专家**（Routing Expert） | MoE 中由门控机制动态选择的专家。每个token经TopK路由后，仅激活少数路由专家进行计算。 |
| **共享专家**（Shared Expert） | MoE 中所有token都会经过的固定专家，不参与路由选择，用于捕获通用知识，与路由专家并行计算。 |
| **token** | 模型处理的基本数据单元。在NLP中通常对应一个单词或子词；在MoE上下文中，每个token经门控后被分配到若干专家进行计算。 |
| **TopK 路由** | 门控机制为每个输入token选择得分最高的K个专家进行处理，K的取值范围通常为 \[2, 8\]。 |
| **EP**（Expert Parallelism，专家并行） | 将不同专家分布在不同计算卡（rank）上的并行策略。每张卡只存储部分专家的权重，通过跨卡通信将token发送到持有目标专家的卡上进行计算。 |
| **EP 通信域** | 参与专家并行的所有计算卡组成的通信组。`ep_world_size` 表示该通信域内的卡数。 |
| **AlltoAllv** | 集合通信操作，每张卡向其他每张卡发送不同数量的数据，同时接收来自其他每张卡的数据。与标准AlltoAll的区别在于各卡间发送/接收的数据量可以不等。 |
| **Permute** | 数据重排操作。在MoE通信后，将token按专家维度重新排列，使同一专家的token在内存中连续存放，便于后续批量计算。 |
| **GroupedMatMul**（GMM） | 分组矩阵乘法。对多组矩阵对分别执行矩阵乘法，各组数据的形状可以不同，适用于MoE中不同专家处理不同数量token的场景。 |
| **MatMul** | 标准矩阵乘法。 |
| **hidden size** | 隐藏层维度大小，即特征向量的长度。路由专家与共享专家可分别具有不同的hidden size（H1、H2）。 |
| **head\_num** | 注意力头数量。在本算子中用于描述专家输出维度（N1、N2）。 |
| **batch sequence size（BS）** | 批次序列长度，即输入序列中的token数量。 |
| **专家个数（e）** | 单张计算卡上部署的专家数量。全部卡上的专家总数为 `e * ep_world_size`。 |
