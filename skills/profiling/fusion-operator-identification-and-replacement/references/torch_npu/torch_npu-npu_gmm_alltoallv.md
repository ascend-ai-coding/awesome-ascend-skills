# torch_npu.npu_gmm_alltoallv

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- **API功能**：MoE网络中，完成路由专家GroupedMatMul、AlltoAllv融合并实现与共享专家MatMul并行融合，先计算后通信。

- **计算公式**：
    - 路由专家：
        $$
        gmm\_y = gmm\_x \times gmm\_weight
        $$

        $$
        unpermute\_out = Unpermute(gmm\_y)
        $$

        $$
        y = AlltoAllv(unpermute\_out)
        $$

        - gmm\_x指路由专家GroupedMatMul计算的左矩阵。
        - gmm\_weight指路由专家GroupedMatMul计算的右矩阵。
        - gmm\_y指路由专家进行GroupedMatMul计算的输出，后续用于Unpermute计算。
        - unpermute\_out是gmm\_y进行Unpermute计算的输出结果，作为AlltoAllv通信的输入。
        - y指对unpermute\_out进行AlltoAllv通信输出。

    - 共享专家：
        $$
        mm\_y = mm\_x \times mm\_weight
        $$

        - mm\_x指共享专家MatMul计算的左矩阵。
        - mm\_weight指共享专家MatMul计算的右矩阵。
        - mm\_y指共享专家MatMul计算的输出。

## 函数原型

```python
torch_npu.npu_gmm_alltoallv(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, trans_gmm_weight=False, trans_mm_weight=False, comm_mode=None) -> (Tensor, Tensor)
```

## 参数说明

- **gmm\_x**（`Tensor`）：**必选参数**，GroupedMatMul计算的左矩阵。数据类型支持`torch.float16`、`torch.bfloat16`，支持2维，shape为\(A, H1\)，数据格式支持$ND$。
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
- **mm\_x**（`Tensor`）：**可选参数**，共享专家MatMul计算中的左矩阵。当需要融合共享专家矩阵计算时，该参数必选，支持2维，shape为\(BS, H2\)。
- **mm\_weight**（`Tensor`）：**可选参数**，共享专家MatMul计算中的右矩阵。当需要融合共享专家矩阵计算时，该参数必选，支持2维，shape为\(H2, N2\)。
- **trans\_gmm\_weight**（`bool`）：**可选参数**，GroupedMatMul的右矩阵是否需要转置，`true`表示需要转置，`false`表示不转置。
- **trans\_mm\_weight**（`bool`）：**可选参数**，共享专家MatMul的右矩阵是否需要转置，`true`表示需要转置，`false`表示不转置。
- **comm\_mode**（`str`）：**可选参数**，表示通信引擎模式，默认值为`None`。

    <!-- npu="A3" id5 -->
    - <term>Atlas A3系列产品</term>：取值支持`None`、`ai_cpu`。传入任意值都将使用AI CPU通信。
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950DT系列产品</term>：取值支持`None`、`ai_cpu`、`ccu`。当为`None`时，使用AI CPU通信。
    <!-- end id6 -->

## 返回值说明

- **y**（`Tensor`）：表示最终计算结果，数据类型与输入`gmm_x`保持一致，支持2维，shape为\(BSK, N1\)。
- **mm\_y**（`Tensor`）：共享专家MatMul的输出，数据类型与`mm_x`保持一致，支持2维，shape为\(BS, N2\)。仅当传入`mm_x`与`mm_weight`才输出。

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
  - BSK：本卡接收的token数（BS\*K=BSK），是recv\_counts参数累加之和，取值范围\(0, 52428800\)。
  - H1：表示路由专家hidden size隐藏层大小，取值范围\(0, 65536\)。
  - H2：表示共享专家hidden size隐藏层大小，取值范围\(0, 12288\]。
  - e：表示单卡上专家个数，e<=32，e\*ep\_world\_size最大支持256。
  - N1：表示路由专家的head\_num，取值范围\(0, 65536\)。
  - N2：表示共享专家的head\_num，取值范围\(0, 65536\)。
  - BS：batch sequence size。
  - K：表示选取top\_k个专家，K的范围\[2, 8\]。
  - A：本卡发送的token数，是send\_counts参数累加之和。
  - EP通信域内所有卡上的A参数的累加和等于所有卡上的BSK参数的累加和。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def run_npu_gmm_alltoallv(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
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

        print(torch_npu.npu_gmm_alltoallv(gmm_x =input,
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
                                                trans_mm_weight  = False))

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

        mp.spawn(run_npu_gmm_alltoallv, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torchair

    class GMM_ALLTOALLV_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,gmm_x, gmm_weight,
                    hcom, ep_world_size,
                    send_counts, recv_counts, send_counts_tensor, recv_counts_tensor,
                    mm_x, mm_weight,
                    trans_gmm_weight, trans_mm_weight):
            return torch_npu.npu_gmm_alltoallv(gmm_x =gmm_x,
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
                                                trans_mm_weight  = trans_mm_weight)

    def run_npu_gmm_alltoallv(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
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

        model = GMM_ALLTOALLV_GRAPH_Model()
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        # 静态图：dynamic=False；动态图：dynamic=True
        model = torch.compile(GMM_ALLTOALLV_GRAPH_Model(), backend=npu_backend, dynamic=False)
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
                        trans_mm_weight=False))

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

        mp.spawn(run_npu_gmm_alltoallv, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
    ```
