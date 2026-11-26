# torch\_npu.npu\_ffn\_to\_attention<a name="ZH-CN_TOPIC_0000002343094193"></a>

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->

## 功能说明<a name="zh-cn_topic_0000002203575833_section14441124184110"></a>

  将FFN节点上的数据发往Attention节点。

## 函数原型<a name="zh-cn_topic_0000002203575833_section45077510411"></a>

```python
torch_npu.npu_ffn_to_attention(x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, group, world_size,token_info_table_shape, token_data_shape, *, attn_rank_table=None) -> ()
```

## 参数说明<a name="zh-cn_topic_0000002203575833_section112637109429"></a>

- **x** (`Tensor`)：必选参数，表示计算使用的token数据，需根据`session_ids`来发送给其他卡。要求为2维张量，shape为\(Y, H\)，表示有Y个token，数据类型支持`torch.bfloat16`、`torch.float16`，数据格式为$ND$，支持非连续的Tensor。
- **session\_ids** (`Tensor`)：必选参数，每个token的Attention Worker节点索引，决定每个token要发给哪些Attention Worker节点。要求为1维张量，shape为\(Y, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。张量里value取值范围为\[0, attnRankNum-1]。
- **micro\_batch\_ids** (`Tensor`)：必选参数，表示每个token的microBatch索引，要求为1维张量，shape为\(Y, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。张量里value取值范围为\[0, MicroBatchNum-1]。
- **token\_ids** (`Tensor`)：必选参数，表示每个token在microBatch中的token索引，要求为1维张量，shape为\(Y, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。张量里value取值范围为\[0, BS-1]。
- **expert\_offsets** (`Tensor`)：必选参数，表示每个token在tokenInfoTableShape中PerTokenExpertNum的索引，要求为1维张量，shape为\(Y, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。张量里value取值范围为\[0, ExpertNumPerToken-1]。
- **actual\_token\_num** (`Tensor`)：必选参数，表示本卡发送的token总数，要求为1维张量，shape为\(1, \)，数据类型支持`torch.int64`，数据格式为$ND$，支持非连续的Tensor。张量里value取值为[0, Y]。
- **group** (`str`)：必选参数，通信域名称，专家并行的通信域。字符串长度范围为\[1,128\)。
- **world\_size**(`torch.int64`)：必选参数，通信域size。取值支持\[2, 768\]。
- **token\_info\_table\_shape**(`List[int]`)：必选参数，Token信息列表大小。包含microBatch的大小（MicroBatchNum）、BatchSize的大小（BS）、以及每个Token对应的Expert数量（ExpertNumPerToken）。
- **token\_data\_shape**(`List[int]`)：必选参数，Token数据列表大小。包含microBatch的大小（MicroBatchNum）、BatchSize的大小（BS）、每个Token对应的Expert数量(ExpertNumPerToken)、以及token和scale长度(HS)。
- **attn\_rank\_table** (`Tensor`)：可选参数，映射每一个Attention Worker对应的卡Id，要求为1维张量，shape为\(Y, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。张量里value取值范围为\[0, attnRankNum-1]。

## 约束说明<a name="zh-cn_topic_0000002203575833_section12345537164214"></a>

- 调用接口过程中使用的`group`、`world_size`、`token_info_table_shape`、`token_data_shape`参数及`HCCL_BUFFSIZE`参数取值所有卡需保持一致，网络中不同层也需保持一致。

<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
<!-- end id2 -->
- 参数说明里shape格式说明：
    - Y：表示本卡需要分发的最大token数量。

    - BS：表示各Attention节点上的发送token数，取值范围为0 < `BS` ≤ 512。

    - H：表示hidden size隐藏层大小，取值范围为1152 ≤ `H` ≤ 8320。

    - HS：表示hidden与scale 隐藏层大小，取值范围为1024 ≤ `HS` ≤ 8192。

    - MicroBatchNum：表示microBatch的大小，目前仅支持MicroBatchNum = 1。

    - ExpertNumPerToken：表示每个Token对应的发送的Expert数量，`ExpertNumPerToken` = `K` + `sharedExpertNum`。

    - K：表示选取topK个专家，取值范围为0 < `K` ≤ 16。

    - ffnRankNum：表示选取ffnRankNum个卡作为FFNWorker，取值范围为0 < `ffnRankNum` < `world_size`。

    - attnRankNum：表示选取attnRankNum个卡作为AttnWorker，取值范围为0 < `attnRankNum` < `world_size`。

    - sharedExpertNum：表示共享专家数量（一个共享专家可以复制部署到多个ffnRank卡上），取值范围为0 ≤ `sharedExpertNum` ≤ 4。

- 通信域使用约束：
    - FFNToAttention算子的通信域中不允许有其他算子。

## 调用示例<a name="zh-cn_topic_0000002203575833_section14459801435"></a>

- 单算子模式调用

    ```python
    import os
    import math
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time

    # 控制模式
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    micro_batch_num = 1
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    scale = 128
    hs = h + scale 
    k = 4
    random_seed = 0
    shared_expert_num = 1  # 共享专家数
    rank_num_per_shared_expert = 1
    shared_ffn_rank_num = shared_expert_num * rank_num_per_shared_expert
    moe_expert_per_rank = 2  # 各ffn卡moe专家数
    moe_ffn_rank_num = 4    #FFN卡数
    moe_expert_num = moe_ffn_rank_num * moe_expert_per_rank
    ffn_worker_num = moe_ffn_rank_num + shared_ffn_rank_num
    attention_worker_num = world_size - ffn_worker_num
    expert_num_per_token = k + shared_expert_num
    token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]
    token_data_table_shape = [micro_batch_num, bs, expert_num_per_token, hs]
    Y = int(math.ceil(micro_batch_num * bs * attention_worker_num * expert_num_per_token / ffn_worker_num))


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def ffn2attn_get_kwargs(
        x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num,
        attn_rank_table, group, world_size, token_info_table_shape, token_data_shape
    ):
        x = x.to(input_dtype).npu()
        session_ids = session_ids.to(torch.int32).npu()
        micro_batch_ids = micro_batch_ids.to(torch.int32).npu()
        token_ids = token_ids.to(torch.int32).npu()
        expert_offsets = expert_offsets.to(torch.int32).npu()
        actual_token_num = actual_token_num.to(torch.int64).npu()
        attn_rank_table = attn_rank_table.to(torch.int32).npu()
        
        return {
            'x':x,
            'session_ids':session_ids,
            'micro_batch_ids':micro_batch_ids,
            'token_ids':token_ids,
            'expert_offsets':expert_offsets,
            'actual_token_num':actual_token_num,
            'attn_rank_table':attn_rank_table,
            'group':group,
            'world_size':world_size,
            'token_info_table_shape':token_info_table_shape,
            'token_data_shape':token_data_shape
        }

    def set_windows(rank, comm_group, hcomm_info):
        if attention_worker_num <= rank < world_size:
            target_ranks = list(range(attention_worker_num))
        else:
            target_ranks = list(range(attention_worker_num, world_size))
        window_size = 1024 * 1024 * 200
        comm_group._get_backend(torch.device('npu'))._window_register_and_exchange(window_size,target_ranks)

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        rank_list = list(range(world_size))
        comm_group = dist.new_group(backend="hccl", ranks=rank_list)
        hcomm_info = get_hcomm_info(rank, comm_group)
        set_windows(rank, comm_group, hcomm_info)

        # 创建输入tensor
        x = torch.randn((Y, h), dtype=input_dtype, device='npu')
        base_ids = torch.arange(Y, dtype=torch.int32, device='npu')
        session_ids = torch.fmod(base_ids, attention_worker_num)
        micro_batch_ids = torch.fmod(base_ids, micro_batch_num)
        token_ids = torch.fmod(base_ids, bs)
        expert_offsets = torch.fmod(base_ids, expert_num_per_token)
        token_num = torch.tensor([Y], dtype=torch.int64, device='npu')
        attn_rank_table = torch.arange(attention_worker_num,dtype = torch.int32)

        ffn2attn_kwargs = ffn2attn_get_kwargs(
            x=x,
            session_ids = session_ids,
            micro_batch_ids = micro_batch_ids,
            token_ids = token_ids,
            expert_offsets = expert_offsets,
            actual_token_num = token_num,
            attn_rank_table = attn_rank_table,
            group=hcomm_info,
            world_size=world_size,
            token_info_table_shape = token_info_table_shape,
            token_data_shape = token_data_table_shape
        )
        if rank >= attention_worker_num:
            torch_npu.npu_ffn_to_attention(**ffn2attn_kwargs)

        else:
            time.sleep(10)
        print(f'rank {rank} npu finished! \n')


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"Y={Y}")
        print(f"k={k}")
        print(f"ffn_worker_num={ffn_worker_num}")
        print(f"attention_worker_num={attention_worker_num}")

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```

- 图模式调用

    ```python
    # 仅支持静态图
    import os
    import math
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torchair
    import time

    # 控制模式
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    micro_batch_num = 1
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    scale = 128
    hs = h + scale 
    k = 4
    random_seed = 0
    shared_expert_num = 1  # 共享专家数
    rank_num_per_shared_expert = 1
    shared_ffn_rank_num = shared_expert_num * rank_num_per_shared_expert
    moe_expert_per_rank = 2  # 各ffn卡moe专家数
    moe_ffn_rank_num = 4    #FFN卡数
    moe_expert_num = moe_ffn_rank_num * moe_expert_per_rank
    ffn_worker_num = moe_ffn_rank_num + shared_ffn_rank_num
    attention_worker_num = world_size - ffn_worker_num
    expert_num_per_token = k + shared_expert_num
    token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]
    token_data_table_shape = [micro_batch_num, bs, expert_num_per_token, hs]
    Y = int(math.ceil(micro_batch_num * bs * attention_worker_num * expert_num_per_token / ffn_worker_num))


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    class FFN_TO_ATTENTION_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
                self, x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table, group, world_size, token_info_table_shape, 
                token_data_shape
            ):
            output = torch_npu.npu_ffn_to_attention(
                x=x,
                session_ids=session_ids,
                micro_batch_ids=micro_batch_ids,
                token_ids=token_ids,
                expert_offsets=expert_offsets,
                actual_token_num=actual_token_num,
                attn_rank_table=attn_rank_table,
                group=group,
                world_size=world_size,
                token_info_table_shape=token_info_table_shape,
                token_data_shape=token_data_shape
            )
            return output


    def set_windows(rank, comm_group, hcomm_info):
        if attention_worker_num <= rank < world_size:
            target_ranks = list(range(attention_worker_num))
        else:
            target_ranks = list(range(attention_worker_num, world_size))
        window_size = 1024 * 1024 * 200
        comm_group._get_backend(torch.device('npu'))._window_register_and_exchange(window_size,target_ranks)

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        rank_list = list(range(world_size))
        comm_group = dist.new_group(backend="hccl", ranks=rank_list)
        hcomm_info = get_hcomm_info(rank, comm_group)
        set_windows(rank, comm_group, hcomm_info)

        # 创建输入tensor
        x = torch.randn((Y, h), dtype=input_dtype, device='npu')
        base_ids = torch.arange(Y, dtype=torch.int32, device='npu')
        session_ids = torch.fmod(base_ids, attention_worker_num)
        micro_batch_ids = torch.fmod(base_ids, micro_batch_num)
        token_ids = torch.fmod(base_ids, bs)
        expert_offsets = torch.fmod(base_ids, expert_num_per_token)
        token_num = torch.tensor([Y], dtype=torch.int64, device='npu')
        attn_rank_table = torch.arange(attention_worker_num,dtype = torch.int32)


        if rank >= attention_worker_num:
            model = FFN_TO_ATTENTION_GRAPH_Model()
            model = model.npu()
            npu_backend = torchair.get_npu_backend()
            print('传统入图模式，静态shape+在线编译场景')
            model = torch.compile(model, backend = npu_backend, dynamic = False)
            _ = model.forward(
                x, session_ids, micro_batch_ids, token_ids, expert_offsets, token_num,
                attn_rank_table, hcomm_info, world_size, token_info_table_shape, 
                token_data_table_shape)

        else:
            time.sleep(10)
        print(f'rank {rank} npu finished! \n')


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"Y={Y}")
        print(f"k={k}")
        print(f"ffn_worker_num={ffn_worker_num}")
        print(f"attention_worker_num={attention_worker_num}")

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```
