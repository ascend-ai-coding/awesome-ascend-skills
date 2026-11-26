# torch_npu.npu_moe_distribute_combine_add_rms_norm

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- **API功能**：完成moe\_distribute\_combine+add+rms\_norm融合。需与[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)配套使用，相当于按npu\_moe\_distribute\_dispatch\_v2算子收集数据的路径原路返回后对数据进行add\_rms\_norm操作。与[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的参数配对关系见[参数说明](#参数说明)和[约束说明](#约束说明)。

> [!NOTE]
> 若未按配对关系使用，可能导致路由恢复错误或通信结果错误。

- **计算公式**：
    - 数据整合功能：

        $$
        rs\_out = ReduceScatterV(expand\_x)
        $$

        $$
        ata\_out = AlltoAllv(rs\_out)
        $$

        $$
        combine\_out = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)
        $$

        $$
        x = combine\_out + residual\_x
        $$

        $$
        y = \frac{x}{RMS(x)} * gamma, \quad \text{where } RMS(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H}x_i^2 + norm\_eps}
        $$

    - 特殊专家场景：
        - zero\_expert\_num ≠ 0：通过传入大于0的zero\_expert\_num参数开启。

            $$
            Moe(ori\_x)=0
            $$

        - copy\_expert\_num ≠ 0：通过传入大于0的copy\_expert\_num参数开启，且需传入有效的ori\_x参数。

            $$
            Moe(ori\_x)=ori\_x
            $$

        - const\_expert\_num ≠ 0：通过传入大于0的const\_expert\_num参数开启，且需传入有效的ori\_x、const\_expert\_alpha\_1、const\_expert\_alpha\_2、const\_expert\_v参数。

            $$
            Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v
            $$

## 函数原型

```python
torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, residual_x, gamma, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0, comm_alg="", norm_eps=1e-06, zero_expert_num=0, copy_expert_num=0, const_expert_num=0) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **expand\_x**（`Tensor`）：**必选参数**，根据`expert_ids`进行扩展过的token特征，要求为2维张量，shape为\(A, H\)，数据格式为$ND$，支持非连续的Tensor。数据类型支持`torch.bfloat16`。
- **expert\_ids**（`Tensor`）：**必选参数**，每个token的topK个专家索引，要求为2维张量，shape为\(BS, K\)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的`expert_ids`输入，张量里value取值范围为\[0, moe\_expert\_num\)，且同一行中的K个value不能重复。
- **expand\_idx**（`Tensor`）：**必选参数**，表示给同一专家发送的token个数，要求是1维张量，shape为\(A \* 128, \)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的`assist_info_for_combine`输出。
- **ep\_send\_counts**（`Tensor`）：**必选参数**，表示本卡每个专家发给EP（Expert Parallelism）域每个卡的数据量，要求是1维张量，shape为\(ep\_world\_size\*local\_expert\_num, \)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的`ep_recv_counts`输出。
- **expert\_scales**（`Tensor`）：**必选参数**，表示每个token的topK个专家的权重，要求是2维张量，shape为\(BS, K\)，其中共享专家不需要乘权重系数，直接相加即可。数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
- **residual\_x**（`Tensor`）：**必选参数**，表示处理后的token需要add的参数，要求是3维张量，shape为\(BS, 1, H\)。数据类型支持`torch.bfloat16`，数据格式为$ND$，支持非连续的Tensor。
- **gamma**（`Tensor`）：**必选参数**，表示rms\_norm的权重，要求是1维张量，shape为\(H, \)。数据类型支持`torch.bfloat16`，数据格式为$ND$，支持非连续的Tensor。
- **group\_ep**（`str`）：**必选参数**，EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)。
- **ep\_world\_size**（`int`）：**必选参数**，EP通信域size，取值范围为\[2, 768\]。
- **ep\_rank\_id**（`int`）：**必选参数**，EP通信域本卡ID，取值范围\[0, ep\_world\_size\)，同一个EP通信域中各卡的`ep_rank_id`不重复。
- **moe\_expert\_num**（`int`）：**必选参数**，MoE专家数量，取值范围\[1, 1024\]，并且满足moe\_expert\_num\%\(ep\_world\_size-shared\_expert\_rank\_num\)=0。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **tp\_send\_counts**（`Tensor`）：**可选参数**，表示本卡每个专家发给TP（Tensor Parallelism）通信域每个卡的数据量。预留参数，当前版本不支持，传默认值None即可。
- **x\_active\_mask**（`Tensor`）：**可选参数**，表示token是否参与通信，默认所有token参与通信，要求是一个1维或2维张量。当输入为1维时，shape为\(BS, \)；当输入为2维时，shape为\(BS, K\)。数据类型支持`torch.bool`，数据格式为$ND$，支持非连续的Tensor。当输入为1维时，参数为true表示对应的token参与通信，true必须排到false之前，例：\{true, false, true\}为非法输入；当输入为2维时，参数为true表示当前token对应的`expert_ids`参与通信，若当前token对应的K个`torch.bool`值全为false，表示当前token不会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。
- **activation\_scale**（`Tensor`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **weight\_scale**（`Tensor`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **group\_list**（`Tensor`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **expand\_scales**（`Tensor`）：**可选参数**，对应[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的`expand_scales`输出。暂不支持该参数，使用默认值即可。
- **shared\_expert\_x**（`Tensor`）：**可选参数**，数据类型需与`expand_x`保持一致。仅在共享专家卡数量`shared_expert_rank_num`为0的场景下使用，表示共享专家token，在combine时需要加上。shape为\(BS, H\)。
- **elastic\_info**（`Tensor`）：**可选参数**，预留参数，当前版本不支持，传默认值None即可。
- **ori\_x**（`Tensor`）：**可选参数**，表示未经过FFN的token数据，在开启copy\_expert或const\_expert的场景下需要本输入数据。可选择传入有效数据或填None，当`copy_expert_num`不为0或`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求是一个2维张量，shape为\(BS, H\)，数据类型需跟`expand_x`保持一致；数据格式为$ND$，支持非连续的Tensor。
- **const\_expert\_alpha\_1**（`Tensor`）：**可选参数**，表示const\_expert场景的计算系数。可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求是一个2维张量，shape为\(const\_expert\_num, H\)，数据类型需跟`expand_x`保持一致；数据格式为$ND$，支持非连续的Tensor。
- **const\_expert\_alpha\_2**（`Tensor`）：**可选参数**，表示const\_expert场景的计算系数。可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求是一个2维张量，shape为\(const\_expert\_num, H\)，数据类型需跟`expand_x`保持一致；数据格式为$ND$，支持非连续的Tensor。
- **const\_expert\_v**（`Tensor`）：**可选参数**，表示const\_expert场景的计算系数。可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求是一个2维张量，shape为\(const\_expert\_num, H\)，数据类型需跟`expand_x`保持一致；数据格式为$ND$，支持非连续的Tensor。
- **group\_tp**（`str`）：**可选参数**，TP通信域名称，数据并行的通信域。预留参数，当前版本不支持，传默认值""即可。
- **tp\_world\_size**（`int`）：**可选参数**，TP通信域size。预留参数，当前版本不支持，传默认值0即可。
- **tp\_rank\_id**（`int`）：**可选参数**，TP通信域本卡ID。预留参数，当前版本不支持，传默认值0即可。
- **expert\_shard\_type**（`int`）：**可选参数**，表示共享专家卡排布类型。当前仅支持0，表示共享专家卡排在MoE专家卡前面。
- **shared\_expert\_num**（`int`）：**可选参数**，表示共享专家数量，一个共享专家可以复制部署到多个卡上。**预留参数，暂未使用，仅支持默认值0。**
- **shared\_expert\_rank\_num**（`int`）：**可选参数**，表示共享专家卡数量。**预留参数，暂未使用，仅支持默认值0。**
- **global\_bs**（`int`）：**可选参数**，表示EP域全局的batch size。当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size，其中`max_bs`表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。
- **out\_dtype**（`int`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **comm\_quant\_mode**（`int`）：**可选参数**，表示通信量化类型。**预留参数，暂未使用，使用默认值即可。**
- **group\_list\_type**（`int`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **comm\_alg**（`str`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **norm\_eps**（`float`）：**可选参数**，用于防止add\_rms\_norm除0错误，默认值为1e-6。
- **zero\_expert\_num**（`int`）：**可选参数**，表示零专家数量，默认值为0，取值范围为\[0, MAX\_INT32\)，其中MAX\_INT32的值为2147483647，合法专家ID范围为\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num\]。
- **copy\_expert\_num**（`int`）：**可选参数**，表示拷贝专家数量，默认值为0，取值范围为\[0, MAX\_INT32\)，其中MAX\_INT32的值为2147483647，合法专家ID范围为\[moe\_expert\_num+zero\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num\]。
- **const\_expert\_num**（`int`）：**可选参数**，表示常量专家数量，默认值为0，取值范围为\[0, MAX\_INT32\)，其中MAX\_INT32的值为2147483647，合法专家ID范围为\[moe\_expert\_num+zero\_expert\_num+copy\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num+const\_expert\_num\]。

## 返回值说明

- **y**（`Tensor`）：表示combine处理后的token进行add\_rms\_norm计算后的结果，要求是3维张量，shape为\(BS, 1, H\)，数据类型与输入`residual_x`保持一致，数据格式为$ND$，不支持非连续的Tensor。
- **rstd\_out**（`Tensor`）：表示add\_rms\_norm的输出结果，要求是3维张量，shape为\(BS, 1, 1\)，数据类型支持`torch.float`，数据格式为$ND$，不支持非连续的Tensor。
- **x**（`Tensor`）：表示combine处理后的token进行add计算后的结果，要求是3维张量，shape为\(BS, 1, H\)，数据类型与输入`residual_x`保持一致，数据格式为$ND$，不支持非连续的Tensor。

## 约束说明

- 该接口支持推理场景下使用。
<!-- npu="950" id9 -->
- 通信方式约束：
    - <term>Ascend 950DT系列产品</term>：仅支持UB Memory通信。
<!-- end id9 -->

- 该接口支持单算子模式和静态图模式。
- 调用接口过程中使用的`expert_ids`、`x_active_mask`、`elastic_info`、`group_ep`、`ep_world_size`、`moe_expert_num`、`group_tp`、`tp_world_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`global_bs`、`comm_alg`、`zero_expert_num`、`copy_expert_num`、`const_expert_num`参数、HCCL\_BUFFSIZE取值，所有卡需保持一致，网络中不同层也需保持一致，且和[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)对应参数也保持一致。
- moe\_expert\_num+zero\_expert\_num+copy\_expert\_num+const\_expert\_num < MAX\_INT32，其中MAX\_INT32的值为2147483647。

<!-- npu="A3" id10 -->
- <term>Atlas A3系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
<!-- end id10 -->
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足A=global\_bs\*shared\_expert\_num/shared\_expert\_rank\_num。
        - 对于MoE专家，当`global_bs`为0时，要满足A\>=BS\*ep\_world\_size\*min\(local\_expert\_num, K\)；当`global_bs`非0时，要满足A\>=global\_bs\*min\(local\_expert\_num, K\)。

    - H：表示hidden size隐藏层大小，取值范围\[1024, 8192\]。
    - BS：表示待发送的token数量，取值范围为0<BS≤512。
    - K：表示选取topK个专家，取值范围为0<K≤16，同时满足0 < K ≤ moe\_expert\_num+zero\_expert\_num+copy\_expert\_num+const\_expert\_num。
    - local\_expert\_num：表示本卡专家数量。
        - 对于共享专家卡，local\_expert\_num=1。
        - 对于MoE专家卡，local\_expert\_num=moe\_expert\_num/\(ep\_world\_size-shared\_expert\_rank\_num\)。

- HCCL通信域缓存区大小：

    调用本接口前需检查HCCL\_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。该场景通信域缓存区大小支持通过环境变量HCCL\_BUFFSIZE配置，也支持通过hccl\_buffer\_size配置（参考[《PyTorch训练模型迁移调优》](https://hiascend.com/document/redirect/canncommercial-ptmigr)中“性能调优\>性能调优方法\>通信优化\>优化方法\>hccl\_buffer\_size”章节）。

    - ep通信域内：设置大小要求\>=2且满足\>=2\*\(local\_expert\_num\*max\_bs\*ep\_world\_size\*Align512\(Align32\(2\*H\)+64\)+\(K+shared\_expert\_num\)\*max\_bs\*Align512\(2\*H\)\)，local\_expert\_num表示需使用MoE专家卡的本卡专家数。
    - 其中Align512(x) = ((x+512-1)/512)\*512，Align32(x) = ((x+32-1)/32)\*32。

- 通信域使用约束：
    - 一个模型中的`npu_moe_distribute_dispatch_v2`和`npu_moe_distribute_combine_add_rms_norm`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。
    <!-- npu="A3" id13 -->
    - <term>Atlas A3系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。
    <!-- end id13 -->

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

    # 控制模式
    quant_mode = 2                       # 2为动态量化
    is_dispatch_scales = True            # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16         # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    sharedExpertRankNum = 0                      # 共享专家数
    moeExpertNum = 32                            # moe专家数
    bs = 8                                       # token数量
    h = 7168                                     # 每个token的长度
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - sharedExpertRankNum
    local_moe_expert_num = moeExpertNum // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (sharedExpertRankNum > 0)
    is_quant = (quant_mode > 0)

    def gen_unique_topk_array(low, high, bs, k):
        array = []
        for i in range(bs):
            top_idx = list(np.arange(low, high, dtype=np.int32))
            random.shuffle(top_idx)
            array.append(top_idx[0:k])
        return np.array(array)

    def get_new_group(rank):
        for i in range(tp_world_size):
            # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
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
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moeExpertNum, bs, k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moeExpertNum, h) if sharedExpertRankNum else (moeExpertNum, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch(
            x=x,
            expert_ids=expert_ids,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_world_size=ep_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            expert_shard_type=0,
            shared_expert_rank_num=sharedExpertRankNum,
            moe_expert_num=moeExpertNum,
            scales=scales,
            quant_mode=quant_mode,
            global_bs=globalBS)
        if is_quant:
            expand_x = expand_x.to(input_dtype)

        bs_local = expert_ids.shape[0]
        torch.manual_seed(42)
        residual_x = torch.rand((bs_local, h), dtype=torch.bfloat16).npu()
        torch.manual_seed(random_seed)
        gamma = torch.ones(h).to(input_dtype).npu()
        norm_eps = 1e-6
        y, rstd_out, x = torch_npu.npu_moe_distribute_combine_add_rms_norm(
            expand_x=expand_x,
            residual_x=residual_x,
            gamma=gamma,
            norm_eps=norm_eps,
            expert_ids=expert_ids,
            expand_idx=expand_idx,
            ep_send_counts=ep_recv_counts,
            expert_scales=expert_scales,
            group_ep=ep_hcomm_info,
            ep_world_size=ep_world_size,
            ep_rank_id=rank // tp_world_size,
            expert_shard_type=0,
            shared_expert_num=0,
            shared_expert_rank_num=sharedExpertRankNum,
            moe_expert_num=moeExpertNum,
            global_bs=globalBS,
        )
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"global_bs={globalBS}")
        print(f"shared_expert_rank_num={sharedExpertRankNum}")
        print(f"moe_expert_num={moeExpertNum}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)

        if sharedExpertRankNum > ep_world_size:
            print("sharedExpertRankNum 不能大于 ep_world_size")
            exit(0)

        if sharedExpertRankNum > 0 and ep_world_size % sharedExpertRankNum != 0:
            print("ep_world_size 必须是 sharedExpertRankNum的整数倍")
            exit(0)

        if moeExpertNum % moe_rank_num != 0:
            print("moeExpertNum 必须是 moe_rank_num 的整数倍")
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

    # 控制模式
    quant_mode = 2                         # 2为动态量化
    is_dispatch_scales = True              # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16           # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    sharedExpertRankNum = 0                      # 共享专家数
    moeExpertNum = 32                            # moe专家数
    bs = 8                                       # token数量
    h = 7168                                     # 每个token的长度
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - sharedExpertRankNum
    local_moe_expert_num = moeExpertNum // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (sharedExpertRankNum > 0)
    is_quant = (quant_mode > 0)

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, expert_ids, group_ep, group_tp, ep_world_size, tp_world_size,
                    ep_rank_id, tp_rank_id, expert_shard_type, shared_expert_rank_num, moe_expert_num,
                    scales, quant_mode, global_bs, expert_scales, residual_x, gamma, norm_eps):
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch(x=x,
                                                                        expert_ids=expert_ids,
                                                                        group_ep=group_ep,
                                                                        group_tp=group_tp,
                                                                        ep_world_size=ep_world_size,
                                                                        tp_world_size=tp_world_size,
                                                                        ep_rank_id=ep_rank_id,
                                                                        tp_rank_id=tp_rank_id,
                                                                        expert_shard_type=expert_shard_type,
                                                                        shared_expert_rank_num=shared_expert_rank_num,
                                                                        moe_expert_num=moe_expert_num,
                                                                        scales=scales,
                                                                        quant_mode=quant_mode,
                                                                        global_bs=global_bs)

            expand_x_npu, _, expand_idx_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            y, rstd_out, x = torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x=expand_x_npu,
                                                                               expert_ids=expert_ids,
                                                                               expand_idx=expand_idx_npu,
                                                                               ep_send_counts=ep_recv_counts_npu,
                                                                               expert_scales=expert_scales,
                                                                               group_ep=group_ep,
                                                                               ep_world_size=ep_world_size,
                                                                               ep_rank_id=ep_rank_id,
                                                                               expert_shard_type=expert_shard_type,
                                                                               shared_expert_rank_num=shared_expert_rank_num,
                                                                               moe_expert_num=moe_expert_num,
                                                                               global_bs=global_bs,
                                                                               shared_expert_num=0,
                                                                               residual_x=residual_x,
                                                                               gamma=gamma,
                                                                               norm_eps=norm_eps)

            return [y, rstd_out, x]

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
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moeExpertNum, bs, k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moeExpertNum, h) if sharedExpertRankNum else (moeExpertNum, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        bs_local = expert_ids.shape[0]
        torch.manual_seed(42)
        residual_x = torch.rand((bs_local, 1, h), dtype=torch.bfloat16).npu()
        torch.manual_seed(random_seed)
        gamma = torch.ones(h).to(input_dtype).npu()
        norm_eps = 1e-6

        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
                               rank // tp_world_size, rank % tp_world_size, 0, sharedExpertRankNum, moeExpertNum, scales,
                               quant_mode, globalBS, expert_scales, residual_x, gamma, norm_eps)
        torch.npu.synchronize()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"global_bs={globalBS}")
        print(f"shared_expert_rank_num={sharedExpertRankNum}")
        print(f"moe_expert_num={moeExpertNum}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)

        if sharedExpertRankNum > ep_world_size:
            print("sharedExpertRankNum 不能大于 ep_world_size")
            exit(0)

        if sharedExpertRankNum > 0 and ep_world_size % sharedExpertRankNum != 0:
            print("ep_world_size 必须是 sharedExpertRankNum的整数倍")
            exit(0)

        if moeExpertNum % moe_rank_num != 0:
            print("moeExpertNum 必须是 moe_rank_num 的整数倍")
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
