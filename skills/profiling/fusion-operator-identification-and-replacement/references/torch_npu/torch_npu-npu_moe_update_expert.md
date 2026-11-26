# torch_npu.npu_moe_update_expert

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- **API功能**：本API支持负载均衡和专家剪枝功能。经过映射后的专家表和mask可传入MoE层进行数据分发和处理。

- **计算公式**：
    - 负载均衡：完成冗余专家部署场景下每个token的topK个专家逻辑卡号到物理卡号的映射。计算方法如下所示：

        代码中符号说明：`F`表示`eplb_table`的列数；`ceil(a, b)`表示向上取整除法，即$\lceil a/b \rceil$。

        负载均衡对于`expert_ids`的第i行，即第i个token的topK个专家索引：

        ```python
        new_expert_id = eplb_table[table_offset + 1]
        expert_id = expert_ids[i]
        table_offset = expert_id * F
        place_num = eplb_table[table_offset]
        if (eplb_table[table_offset] == 1):
            new_expert_id = eplb_table[table_offset + 1]
        else:
            if (balance_mode == 0):
                mode_value = ceil(world_size, place_num)
                place_idx = local_rank_id / mode_value + 1
            else:
                place_idx = i % place_num
        new_expert_id = eplb_table[table_offset + place_idx]
        ```

    - 专家剪枝：支持根据阈值对token发送的topK个专家进行剪枝。计算方法如下所示：

        将shape为$(BS,)$的`active_mask`进行broadcast成为shape为$(BS,K)$的active_mask_tensor，其中`active_mask`中值为False的token会直接被剪枝。对于active_mask_tensor为True的`expert_scales`的元素，满足条件也将被剪枝。

        ```python
        active_mask_tensor = broadcast(active_mask, (BS, K))
        for i in range(BS):
            expert_scales_vec[:] = sum(expert_scales[i, :]) * pruning_threshold[:]
            balanced_active_mask[i, :] = (expert_scales_vec[i, :] < expert_scales[:]) & active_mask_tensor[i, :]
        ```

## 函数原型

```python
torch_npu.npu_moe_update_expert(expert_ids, eplb_table, *, expert_scales=None, pruning_threshold=None, active_mask=None, local_rank_id=-1, world_size=-1, balance_mode=0) -> (Tensor, Tensor)
```

## 参数说明

- **expert\_ids**（`Tensor`）：**必选参数**，表示每个token的topK个专家索引，shape为\(BS, K\)。数据类型支持`torch.int32`、`torch.int64`，数据格式为$ND$，支持非连续的Tensor。
- **eplb\_table**（`Tensor`）：**必选参数**，表示逻辑专家到物理专家的映射表，外部调用者需保证输入Tensor的值正确：每行第一列为行号对应逻辑专家部署的实例数count，值需大于等于1，每行\[1, count\]列为对应实例的卡号，取值范围\[0, moe\_expert\_num\)，shape为\(log\_expert\_num, F\)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。其中：
    - `log_expert_num`：表示逻辑专家数量，即`eplb_table`的行数，每个逻辑专家对应映射表中的一行，取值范围\(0, 1024\)。
    - `F`：表示输入映射表的列数，取值范围\[2, `world_size`+1\]，第一列为各行号对应逻辑专家部署的实例个数（值>0），后F-1列为该逻辑专家部署的物理卡号。
    - `moe_expert_num`：表示物理专家总数，即所有逻辑专家部署的副本个数之和（等于`eplb_table`第一列count之和），取值范围\(0, 1024\]。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **expert\_scales**（`Tensor`）：**可选参数**，每个token的topK个专家的scale权重，用户需保证scale在token内部按照降序排列，可选择传入有效数据或不传，该参数传入有效数据时，`pruning_threshold`也需要传入有效数据。shape为\(BS, K\)。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式为$ND$，支持非连续的Tensor。
- **pruning\_threshold**（`Tensor`）：**可选参数**，专家scale权重的最小阈值，当某个token对应的某个topK专家scale小于阈值时，该token将对该专家进行剪枝，即token不发送至该专家处理，可选择传入有效数据或None。该参数传入有效数据时，`expert_scales`也需要传入有效数据。shape为\(K, \)或\(1, K\)。数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
- **active\_mask**（`Tensor`）：**可选参数**，表示token是否参与通信，可选择传入有效数据或None。传入有效数据时，`expert_scales`、`pruning_threshold`也必须传入有效数据，参数为true表示对应的token参与通信，true必须排到false之前，例：\{true, false, true\}为非法输入；传入None时表示所有token都会参与通信。shape为\(BS, \)。数据类型支持`torch.bool`，数据格式为$ND$，支持非连续的Tensor。
- **local\_rank\_id**（`int`）：**可选参数**，本卡ID。数据类型支持`torch.int64`，当`balance_mode`设置为0时，本属性取值范围为\[0, `world_size`\)。
- **world\_size**（`int`）：**可选参数**，通信域size。数据类型支持`torch.int64`，当`balance_mode`设置为0时，本属性取值范围为\[2, 768\]。
- **balance\_mode**（`int`）：**可选参数**，均衡规则。数据类型支持`torch.int64`，取值支持0和1，0表示用`local_rank_id`进行负载均衡，1表示使用`token_id`进行负载均衡。当本属性取值为0时，`local_rank_id`和`world_size`必须传入有效值。

## 返回值说明

- **balanced\_expert\_ids**（`Tensor`）：映射后每个token的topK个专家所在物理卡的卡号，shape为\(BS, K\)，数据类型、数据格式与`expert_ids`保持一致。
- **balanced\_active\_mask**（`Tensor`）：剪枝后的`active_mask`，当`expert_scales`、`pruning_threshold`传入有效数据时该输出有效。shape为\(BS, K\)，数据类型支持`torch.bool`，数据格式为$ND$，支持非连续的Tensor。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。

<!-- npu="A3" id3 -->
- <term>Atlas A3系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
<!-- end id3 -->
- 参数说明里shape格式说明：
    - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为0<BS≤512。
    - `K`：表示选取topK个专家，取值范围为0<K≤16，同时满足0 < K ≤ log\_expert\_num。
    - `log_expert_num`：表示逻辑专家数量，即`eplb_table`的行数，每个逻辑专家对应映射表中的一行，取值范围\(0, 1024\)。
    - `moe_expert_num`：表示物理专家数，取值范围\(0, 1024\]。
    - `F`：表示输入映射表`eplb_table`的列数，取值范围为\[2, `world_size` + 1\]，第一列为各行号对应逻辑专家部署的实例个数（值>0），后F-1列为该逻辑专家部署的物理卡号。
    - 每个专家部署副本个数值（即`eplb_table`第一列的count），最小为1，最大为`world_size`。
    - 所有专家部署的副本个数和（即`eplb_table`第一列count之和）需小于等于1024，且整除`world_size`。

- 该接口必须与`torch_npu.npu_moe_distribute_dispatch`或`torch_npu.npu_moe_distribute_dispatch_v2`接口配合使用。
- 调用接口过程中使用的`world_size`、`moe_expert_num`参数取值所有卡须保持一致，网络中不同层也需保持一致，本接口中参数和`torch_npu.npu_moe_distribute_dispatch`或`torch_npu.npu_moe_distribute_dispatch_v2`有如下对应关系：

    **表 1** 参数对应关系

    | torch_npu.npu_moe_update_expert | torch_npu.npu_moe_distribute_dispatch/torch_npu.npu_moe_distribute_dispatch_v2 |
    | --- | --- |
    | `local_rank_id` | `ep_rank_id` |
    | `world_size` | `ep_world_size` |
    | `eplb_table`第一列的count之和 | `moe_expert_num` |
    | `BS` | `BS` |
    | `K` | `K` |

## 调用示例

- 单算子模式调用

    ```python
    # 调用示例单算子模式调用
    import os
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp

    # 控制模式
    dtype = torch.int32
    local_moe_expert_num = 4
    is_pruning = 1
    balance_mode = 1
    quant_mode = 2                       # 2为动态量化
    is_dispatch_scales = True            # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16         # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 2
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    shared_expert_rank_num = 0                      # 共享专家数
    BS = 8                                       # token数量
    h = 7168                                     # 每个token的长度
    K = 2
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    moe_expert_num = moe_rank_num * local_moe_expert_num
    globalBS = BS * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    phy_ep_size = (ep_world_size - shared_expert_rank_num) * local_moe_expert_num
    log_ep_size = phy_ep_size // local_moe_expert_num
    F = random.randint(2, world_size + 1)

    def get_new_group(rank):
        for i in range(tp_world_size):
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
            tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_t = tp_group
                print(f"rank:{rank} tp_ranks:{tp_ranks}")
        return ep_group_t, tp_group_t

    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        # tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # 创建输入tensor
        x = torch.randn(BS, h, dtype=input_dtype).npu()
        eplb_table = np.zeros((log_ep_size, F - 1))
        count_column = np.random.randint(1, F, size=(log_ep_size, 1))
        all_ranks = np.arange(phy_ep_size)
        for i in range(log_ep_size):
            np.random.shuffle(all_ranks)
            for j in range(count_column[i][0]):
                eplb_table[i][j] = all_ranks[j]
        eplb_table = np.hstack((count_column, eplb_table))
        expert_ids = np.random.randint(low=0, high=log_ep_size, size=(BS, K))
        expert_scales = torch.randn(BS, K, dtype=torch.float32).npu()
        pruning_threshold = None
        active_mask = None
        if is_pruning:
            expert_scales_temp = -np.sort(-np.random.uniform(low=0, high=0.25, size=(BS, K)), axis=1)
            pruning_threshold_temp = np.random.uniform(low=0, high=0.15, size=(1, K))
            num_true = np.random.randint(0, BS + 1)
            active_mask_temp = np.concatenate([np.ones(num_true, dtype=bool), np.zeros(BS - num_true, dtype=bool)])
            expert_scales = torch.from_numpy(expert_scales_temp).to(torch.float32).npu()
            pruning_threshold = torch.from_numpy(pruning_threshold_temp).to(torch.float32).npu()
            active_mask = torch.from_numpy(active_mask_temp).to(torch.bool).npu()
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        eplb_table_tensor = torch.from_numpy(eplb_table).to(torch.int32)
        expert_ids_tensor = torch.from_numpy(expert_ids).to(dtype)

        npu_balanced_expert_ids, npu_balanced_active_mask = torch_npu.npu_moe_update_expert(
            expert_ids=expert_ids_tensor.npu(),
            eplb_table=eplb_table_tensor.npu(),
            expert_scales=expert_scales if is_pruning else None,
            pruning_threshold=pruning_threshold if is_pruning else None,
            active_mask=active_mask if is_pruning else None,
            local_rank_id=rank,
            world_size=world_size,
            balance_mode=balance_mode)

        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=npu_balanced_expert_ids,
            x_active_mask=npu_balanced_active_mask,
            expert_scales=expert_scales,
            group_ep=ep_hcomm_info,
            # group_tp=tp_hcomm_info,
            ep_world_size=ep_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=rank // tp_world_size,
            # tp_rank_id=rank % tp_world_size,
            expert_shard_type=0,
            shared_expert_rank_num=shared_expert_rank_num,
            moe_expert_num=phy_ep_size,
            scales=scales,
            quant_mode=quant_mode,
            global_bs=globalBS)
        if is_quant:
            expand_x = expand_x.to(input_dtype)
        x = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x,
                                                 expert_ids=npu_balanced_expert_ids,
                                                 x_active_mask=npu_balanced_active_mask,
                                                 assist_info_for_combine=assist_info_for_combine,
                                                 ep_send_counts=ep_recv_counts,
                                                 tp_send_counts=tp_recv_counts,
                                                 expert_scales=expert_scales,
                                                 group_ep=ep_hcomm_info,
                                                #  group_tp=tp_hcomm_info,
                                                 ep_world_size=ep_world_size,
                                                 tp_world_size=tp_world_size,
                                                 ep_rank_id=rank // tp_world_size,
                                                #  tp_rank_id=rank % tp_world_size,
                                                 expert_shard_type=0,
                                                 shared_expert_rank_num=shared_expert_rank_num,
                                                 moe_expert_num=moe_expert_num,
                                                 global_bs=globalBS)
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    if __name__ == "__main__":
        print(f"BS={BS}")
        print(f"global_bs={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"moe_expert_num={moe_expert_num}")
        print(f"K={K}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)

        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num不能大于ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size必须是shared_expert_rank_num的整数倍")
            exit(0)

        if moe_expert_num % moe_rank_num != 0:
            print("moe_expert_num必须是moe_rank_num的整数倍")
            exit(0)

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
    # 图模式调用
    # 修改graph_type支持静态图、动态图
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # 控制模式
    quant_mode = 2
    is_dispatch_scales = True
    input_dtype = torch.bfloat16
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 2
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)
    shared_expert_rank_num = 0
    moe_expert_num = 14
    bs = 8
    h = 7168
    k = 2
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = moe_expert_num // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    is_pruning = 1
    balance_mode = 1

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, expert_ids, group_ep, ep_world_size, tp_world_size, ep_rank_id,
                    expert_shard_type, shared_expert_rank_num, moe_expert_num, scales, quant_mode, global_bs, expert_scales, eplb_table, pruning_threshold, active_mask, balance_mode):
            balanced_expert_ids, balanced_active_mask = torch_npu.npu_moe_update_expert(expert_ids=expert_ids,
                                                                                        eplb_table=eplb_table,
                                                                                        expert_scales=expert_scales if is_pruning else None,
                                                                                        pruning_threshold=pruning_threshold,
                                                                                        active_mask=active_mask,
                                                                                        local_rank_id=ep_rank_id,
                                                                                        world_size=ep_world_size,
                                                                                        balance_mode=balance_mode)
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(x=x,
                                                    expert_ids=balanced_expert_ids,
                                                    group_ep=group_ep,
                                                    ep_world_size=ep_world_size,
                                                    tp_world_size=tp_world_size,
                                                    ep_rank_id=ep_rank_id,
                                                    expert_shard_type=expert_shard_type,
                                                    shared_expert_rank_num=shared_expert_rank_num,
                                                    moe_expert_num=moe_expert_num,
                                                    scales=scales,
                                                    quant_mode=quant_mode,
                                                    global_bs=global_bs)
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x_npu,
                                                                        expert_ids=expert_ids,
                                                                        assist_info_for_combine=assist_info_for_combine_npu,
                                                                        ep_send_counts=ep_recv_counts_npu,
                                                                        tp_send_counts=tp_recv_counts_npu,
                                                                        expert_scales=expert_scales,
                                                                        group_ep=group_ep,
                                                                        ep_world_size=ep_world_size,
                                                                        tp_world_size=tp_world_size,
                                                                        ep_rank_id=ep_rank_id,
                                                                        expert_shard_type=expert_shard_type,
                                                                        shared_expert_rank_num=shared_expert_rank_num,
                                                                        moe_expert_num=moe_expert_num,
                                                                        global_bs=global_bs)
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]

    def gen_unique_topk_array(low, high, bs, k):
        array = []
        for i in range(bs):
            top_idx = list(np.arange(low, high, dtype=np.int32))
            random.shuffle(top_idx)
            array.append(top_idx[0:k])
        return np.array(array)

    def get_new_group(rank):
        for i in range(tp_world_size):
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
            tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_t = tp_group
                print(f"rank:{rank} tp_ranks:{tp_ranks}")
        return ep_group_t, tp_group_t

    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        # tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moe_expert_num, bs, k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()
        eplb_table = np.array([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13]]).astype(np.int32)
        eplb_table = torch.from_numpy(eplb_table).npu()
        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()

        pruning_threshold = None
        active_mask = None
        if is_pruning:
            expert_scales_temp = -np.sort(-np.random.uniform(low=0, high=0.25, size=(bs, k)), axis=1)
            pruning_threshold_temp = np.random.uniform(low=0, high=0.15, size=(1, k))
            num_true = np.random.randint(0, bs + 1)
            active_mask_temp = np.concatenate([np.ones(num_true, dtype=bool), np.zeros(bs - num_true, dtype=bool)])
            expert_scales_tensor = torch.from_numpy(expert_scales_temp).to(torch.float32)
            pruning_threshold_tensor = torch.from_numpy(pruning_threshold_temp).to(torch.float32)
            active_mask_tensor = torch.from_numpy(active_mask_temp).to(torch.bool)
            expert_scales = expert_scales_tensor.npu()
            pruning_threshold = pruning_threshold_tensor.npu()
            active_mask = active_mask_tensor.npu()

        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(x, expert_ids, ep_hcomm_info, ep_world_size, tp_world_size,
                            rank // tp_world_size, 0, shared_expert_rank_num, moe_expert_num, scales,
                            quant_mode, globalBS, expert_scales, eplb_table, pruning_threshold, active_mask, balance_mode)
        torch.npu.synchronize()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"global_bs={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"moe_expert_num={moe_expert_num}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)

        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num不能大于ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size必须是shared_expert_rank_num的整数倍")
            exit(0)

        if moe_expert_num % moe_rank_num != 0:
            print("moe_expert_num必须是moe_rank_num的整数倍")
            exit(0)

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        print("run npu success")
    ```
