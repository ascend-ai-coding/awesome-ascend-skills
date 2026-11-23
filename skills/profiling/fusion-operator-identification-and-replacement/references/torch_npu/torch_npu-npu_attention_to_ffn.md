# torch\_npu.npu\_attention\_to\_ffn

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->

## 功能说明

将Attention节点上数据发往FFN节点。

## 函数原型

```python
torch_npu.npu_attention_to_ffn(x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, group, world_size, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num, *, scales=None, active_mask=None, quant_mode=0, sync_flag=0, ffn_start_rank_id=0) -> ()
```

## 参数说明

- **x** (`Tensor`)：必选参数，表示计算使用的token数据，需根据`expert_ids`和`expert_rank_table`来发送给其他卡。要求为3维张量，shape为\(X, BS, H\)，表示有X个microBatch，每个microBatch里有BS个token，数据类型支持`torch.bfloat16`、`torch.float16`，数据格式为$ND$，支持非连续的Tensor。
- **session\_id** (`Tensor`)：必选参数，Attention域本卡ID，要求为1维张量，shape为\(X, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。
- **micro\_batch\_id** (`Tensor`)：必选参数，当前microBatch组的ID，要求为1维张量，shape为\(X, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。
- **layer\_id** (`Tensor`)：必选参数，模型层数ID，要求为1维张量，shape为\(X, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。
- **expert\_ids** (`Tensor`)：必选参数，每个micro batch组中每个token的topK个专家索引，决定每个token要发给哪些专家。要求为3维张量，shape为\(X, BS, K\)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。张量里value取值范围为\[0, moe\_expert\_num\)，且同一行（同一个token）中的K个value不能重复。
- **expert\_rank\_table** (`Tensor`)：必选参数，每个micro batch组中专家ID到FFN卡专家部署的映射表，外部需保证值正确。要求为3维张量，shape为\(L, shared\_expert\_num + moe\_expert\_num, M\)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。
- **group** (`str`)：必选参数，通信域名称，专家并行的通信域。字符串长度范围为\[1,128\)。
- **world\_size**(`int`)：必选参数，通信域size。取值支持\[2, 768\]。
- **ffn\_token\_info\_table\_shape** (`List[int]`)：必选参数，表示FFN卡上token信息表格shape大小，长度为3，包括Attention节点的数量、microBatchSize的大小以及每个token对应的相关发送状态信息shape的大小。
- **ffn\_token\_data\_shape** (`List[int]`)：必选参数，表示FFN卡上token数据表格shape大小，长度为5，包括Attention节点的数量、microBatchSize的大小、batchSize的大小、每个token需发送的专家数量（包括共享专家）、单个token的长度。
- **attn\_token\_info\_table\_shape** (`List[int]`)：必选参数，表示Attention卡上token信息表格shape大小，长度为3，包括microBatchSize的大小、batchSize的大小、每个token需发送的专家数量（包括共享专家）。
- **moe\_expert\_num** (`int`)：必选参数，MoE专家数量，取值范围\[1, 1024\]。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **scales** (`Tensor`)：可选参数，表示每个专家的权重，非量化场景不传，动态量化场景可传可不传。若传值要求为3维张量，shape为\(L, shared\_expert\_num + moe\_expert\_num, H\)，数据类型支持`torch.float`，数据格式为$ND$，不支持非连续的Tensor。当`quant_mode`为2，`scales`可不为None；当`quant_mode`为0，`scales`必须为None。
- **active\_mask** (`Tensor`)：可选参数，表示token是否参与通信。要求是一个2维张量，shape为\(X, BS\)。数据类型支持`torch.bool`，数据格式要求为$ND$，支持非连续的Tensor。参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true} 为非法输入。默认所有token都会参与通信。
- **quant\_mode** (`int`)：可选参数，表示量化模式。支持取值：0表示非量化（默认），2表示动态量化。
- **sync\_flag** (`int`)：可选参数，表示同步、异步。支持取值：0表示同步（默认），1表示异步。
- **ffn\_start\_rank\_id** (`int`)：可选参数，FFN域起始ID。取值范围\[0, world\_size\)，默认为0。

## 返回值说明

- 无

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持静态图模式，分离系列算子必须配套使用。
- 调用接口过程中使用的`group`、`world_size`、`moe_expert_num`参数取值所有卡需保持一致，且网络中不同层也需保持一致。

<!-- npu="A3" id2 -->
- Atlas A3系列产品：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
<!-- end id2 -->
- 参数里Shape使用的变量如下：
    - X：表示micro batch sequence size，即token组数，当前版本仅支持 X = 1。

    - H：表示hidden size隐藏层大小，取值为\[1024, 8192\]。

    - BS：表示batch sequence size，即本卡最终输出的token数量，取值范围为0 < BS ≤ 512。

    - K：表示选取topK个专家，取值范围为0 < K ≤ 16，同时满足0 < K ≤ moe\_expert\_num。

    - L：表示模型层数，当前版本仅支持 L = 1。

    - shared_expert_num：表示共享专家数量（一个共享专家可以复制部署到多个FFN节点上），取值范围为\[0, 4\]。

    - M：表示每个专家对应的FFN卡数量或相关映射维度，需根据实际部署配置确定。

- HCCL通信域缓存区大小：调用本接口前需检查通信域缓存区大小取值是否合理，单位MB，不配置时默认为200MB。

## 调用示例

- 单算子模式调用

    ```python
    import os
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time

    # 控制模式
    quant_mode = 2  # 2为动态量化
    sync_flag = 1   # 1为异步
    is_mask = True  # 是否剪枝
    is_attn2ffn_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 4
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    micro_batch_num = 1
    X = 1
    L = 1
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 4
    hs = h + 128
    random_seed = 0
    shared_expert_num = 1  # 共享专家数
    rank_num_per_shared_expert = 1
    shared_ffn_rank_num = shared_expert_num * rank_num_per_shared_expert
    moe_expert_per_rank = 2  # 各FFN卡上的MoE专家数
    moe_ffn_rank_num = 2  # MoE FFN卡数
    moe_expert_num = moe_ffn_rank_num * moe_expert_per_rank
    ffn_worker_num = moe_ffn_rank_num + shared_ffn_rank_num
    attention_worker_num = world_size - ffn_worker_num
    expert_num_per_token = k + shared_expert_num
    is_quant = (quant_mode > 0)
    ffn_token_info_table_shape = [attention_worker_num, micro_batch_num, 2 + bs * expert_num_per_token]
    ffn_token_data_shape = [attention_worker_num, micro_batch_num, bs, expert_num_per_token, hs if is_quant else h]
    attn_token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info
    

    def set_windows(rank, comm_group, hcomm_info):
        if rank >= ffn_worker_num:
            # 当前 rank 属于后半部分，需与前 ffn_rank_num 个 rank 通信
            target_ranks = list(range(ffn_worker_num))
        else:
            # 当前 rank 属于前半部分，需与后面 rank 通信
            target_ranks = list(range(ffn_worker_num, world_size))
        window_size = 120 * 1024 * 200
        comm_group._get_backend(torch.device('npu'))._window_register_and_exchange(window_size, target_ranks)


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + dev_num * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        rank_list = list(range(world_size))
        comm_group = dist.new_group(backend="hccl", ranks=rank_list)
        hcomm_info = get_hcomm_info(rank, comm_group)
        set_windows(rank, comm_group, hcomm_info)

        # 创建输入tensor
        x = torch.randn(X, bs, h, dtype=input_dtype).npu()
        session_id = torch.tensor([rank - ffn_worker_num], dtype=torch.int32).npu()
        micro_batch_id = torch.tensor([0], dtype=torch.int32).npu()
        layer_id = torch.tensor([0], dtype=torch.int32).npu()
        expert_ids = torch.tensor([[[0, 1, 2, 3],
                                [1, 2, 3, 0],
                                [2, 3, 0, 1],
                                [3, 0, 1, 2],
                                [0, 2, 1, 3],
                                [1, 3, 2, 0],
                                [2, 0, 3, 1],
                                [3, 1, 0, 2]]], dtype=torch.int32).npu()
        expert_rank_table = torch.tensor([[[1, 0, 0],
                                        [1, 0, 1],
                                        [1, 1, 2],
                                        [1, 1, 3],
                                        [1, 2, 4]]], dtype=torch.int32).npu()

        scales_shape = (L, shared_expert_num + moe_expert_num, h)
        if is_attn2ffn_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        
        if is_mask:
            active_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool).npu()
        else:
            active_mask = None

        if rank >= ffn_worker_num:
            torch_npu.npu_attention_to_ffn(
                x=x,
                session_id = session_id,
                micro_batch_id = micro_batch_id,
                layer_id = layer_id,
                expert_ids=expert_ids,
                expert_rank_table = expert_rank_table,
                group=hcomm_info,
                world_size=world_size,
                moe_expert_num=moe_expert_num,
                ffn_token_info_table_shape = ffn_token_info_table_shape,
                ffn_token_data_shape = ffn_token_data_shape,
                attn_token_info_table_shape = attn_token_info_table_shape,
                scales=scales,
                active_mask=active_mask,
                quant_mode=quant_mode,
                sync_flag=sync_flag)

            print(f'rank {rank} npu finished! \n')
        
        if rank < ffn_worker_num:
            time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"shared_expert_num={shared_expert_num}", flush=True)
        print(f"moe_expert_num={moe_expert_num}")

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        for p in p_list:
            if p.exitcode != 0:
                raise RuntimeError(f"subprocess {p.pid} failed with exitcode {p.exitcode}")

        print("run npu success.")
    ```

- 图模式调用

    ```python
    # 仅支持静态图
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time


    # 控制模式
    quant_mode = 2  # 2为动态量化
    sync_flag = 1   # 1为异步
    is_mask = True  # 是否剪枝
    is_attn2ffn_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 4
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    micro_batch_num = 1
    X = 1
    L = 1
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 4
    hs = h + 128
    random_seed = 0
    shared_expert_num = 1  # 共享专家数
    rank_num_per_shared_expert = 1
    shared_ffn_rank_num = shared_expert_num * rank_num_per_shared_expert
    moe_expert_per_rank = 2  # 各FFN卡上的MoE专家数
    moe_ffn_rank_num = 2  # MoE FFN卡数
    moe_expert_num = moe_ffn_rank_num * moe_expert_per_rank
    ffn_worker_num = moe_ffn_rank_num + shared_ffn_rank_num
    attention_worker_num = world_size - ffn_worker_num
    expert_num_per_token = k + shared_expert_num
    is_quant = (quant_mode > 0)
    ffn_token_info_table_shape = [attention_worker_num, micro_batch_num, 2 + bs * expert_num_per_token]
    ffn_token_data_shape = [attention_worker_num, micro_batch_num, bs, expert_num_per_token, hs if is_quant else h]
    attn_token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]


    class ATTENTION_TO_FFN_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, session_id, micro_batch_id, layer_id, expert_ids,
                expert_rank_table, hcomm_info, world_size, moe_expert_num,
                ffn_token_info_table_shape, ffn_token_data_shape,
                attn_token_info_table_shape, scales, active_mask, quant_mode,
                sync_flag):
            output = torch_npu.npu_attention_to_ffn(
                x=x,
                session_id = session_id,
                micro_batch_id = micro_batch_id,
                layer_id = layer_id,
                expert_ids=expert_ids,
                expert_rank_table = expert_rank_table,
                group=hcomm_info,
                world_size=world_size,
                moe_expert_num=moe_expert_num,
                ffn_token_info_table_shape = ffn_token_info_table_shape,
                ffn_token_data_shape = ffn_token_data_shape,
                attn_token_info_table_shape = attn_token_info_table_shape,
                scales=scales,
                active_mask=active_mask,
                quant_mode=quant_mode,
                sync_flag=sync_flag)
            
            return output


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info
    

    def set_windows(rank, comm_group, hcomm_info):
        if rank >= ffn_worker_num:
            # 当前 rank 属于后半部分，需与前 ffn_rank_num 个 rank 通信
            target_ranks = list(range(ffn_worker_num))
        else:
            # 当前 rank 属于前半部分，需与后面 rank 通信
            target_ranks = list(range(ffn_worker_num, world_size))
        window_size = 120 * 1024 * 200
        comm_group._get_backend(torch.device('npu'))._window_register_and_exchange(window_size, target_ranks)


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + dev_num * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        rank_list = list(range(world_size))
        comm_group = dist.new_group(backend="hccl", ranks=rank_list)
        hcomm_info = get_hcomm_info(rank, comm_group)
        set_windows(rank, comm_group, hcomm_info)

        # 创建输入tensor
        x = torch.randn(X, bs, h, dtype=input_dtype).npu()
        session_id = torch.tensor([rank - ffn_worker_num], dtype=torch.int32).npu()
        micro_batch_id = torch.tensor([0], dtype=torch.int32).npu()
        layer_id = torch.tensor([0], dtype=torch.int32).npu()
        expert_ids = torch.tensor([[[0, 1, 2, 3],
                                [1, 2, 3, 0],
                                [2, 3, 0, 1],
                                [3, 0, 1, 2],
                                [0, 2, 1, 3],
                                [1, 3, 2, 0],
                                [2, 0, 3, 1],
                                [3, 1, 0, 2]]], dtype=torch.int32).npu()
        expert_rank_table = torch.tensor([[[1, 0, 0],
                                        [1, 0, 1],
                                        [1, 1, 2],
                                        [1, 1, 3],
                                        [1, 2, 4]]], dtype=torch.int32).npu()

        scales_shape = (L, shared_expert_num + moe_expert_num, h)
        if is_attn2ffn_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        
        if is_mask:
            active_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool).npu()
        else:
            active_mask = None

        if rank >= ffn_worker_num:
            model = ATTENTION_TO_FFN_GRAPH_Model()
            model = model.npu()
            npu_backend = torchair.get_npu_backend()
            model = torch.compile(model, backend=npu_backend, dynamic=False)
            _ = model.forward(
                x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table,
                hcomm_info, world_size, moe_expert_num, ffn_token_info_table_shape, 
                ffn_token_data_shape, attn_token_info_table_shape, scales, active_mask, quant_mode, sync_flag
            )

            print(f'rank {rank} npu finished! \n')

        if rank < ffn_worker_num:
            time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"shared_expert_num={shared_expert_num}", flush=True)
        print(f"moe_expert_num={moe_expert_num}")

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        for p in p_list:
            if p.exitcode != 0:
                raise RuntimeError(f"subprocess {p.pid} failed with exitcode {p.exitcode}")

        print("run npu success.")
    ```
