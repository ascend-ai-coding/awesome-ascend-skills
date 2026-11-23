# torch_npu.npu_moe_distribute_combine

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- **API功能**：先进行reduce\_scatterv通信，再进行alltoallv通信，最后将接收的数据整合（乘权重再相加）。需与[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)配套使用，相当于按npu\_moe\_distribute\_dispatch算子收集数据的路径原路返回。

- **计算公式**：

    $$
    rs\_out = ReduceScatterV(expand\_x)
    $$

    $$
    ata\_out = AlltoAllv(rs\_out)
    $$

    $$
    x = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)
    $$

## 函数原型

```python
torch_npu.npu_moe_distribute_combine(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, shared_expert_x=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0) -> Tensor
```

## 参数说明

- **expand\_x**（`Tensor`）：**必选参数**，根据`expert_ids`进行扩展过的token特征，要求为2维张量，shape为\(max\(tp\_world\_size, 1\) \* A, H\)，数据格式为$ND$，支持非连续的Tensor。数据类型支持`torch.bfloat16`、`torch.float16`。
    <!-- npu="910b" id53 -->
    - Atlas A2系列产品不支持共享专家场景。
    <!-- end id53 -->

- **expert\_ids**（`Tensor`）：**必选参数**，每个token的topK个专家索引，要求为2维张量，shape为\(BS, K\)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`expert_ids`输入，张量里value取值范围为\[0, moe\_expert\_num\)，且同一行中的K个value不能重复。
- **expand\_idx**（`Tensor`）：**必选参数**，表示给同一专家发送的token个数，要求是1维张量，shape为\(BS \* K, \)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`expand_idx`输出。
- **ep\_send\_counts**（`Tensor`）：**必选参数**，表示本卡每个专家发给EP（Expert Parallelism）域每个卡的token数（token数以前缀和的形式表示），要求是1维张量。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`ep_recv_counts`输出。

    <!-- npu="910b" id4 -->
    - <term>Atlas A2系列产品</term>：要求shape为\(moe\_expert\_num+2\*global\_bs\*K\*server\_num, \)，前`moe_expert_num`个数表示在EP通信域内，该卡上每个专家收到来自其他各卡的token数（以前缀和的形式表示），2\*global\_bs\*K\*server\_num用于存储机间和机内通信前，combine可提前做reduce操作的token个数和通信区偏移量，`global_bs`传入0时此处按照bs\*ep\_world\_size计算。
    <!-- end id4 -->
    <!-- npu="950,A3" id5 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：要求shape为\(ep\_world\_size\*max\(tp\_world\_size, 1\)\*local\_expert\_num, \)。
    <!-- end id5 -->

- **expert\_scales**（`Tensor`）：**必选参数**，表示每个token的topK个专家的权重，要求是2维张量，shape为\(BS, K\)，其中共享专家不需要乘权重系数，直接相加即可。数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
- **group\_ep**（`str`）：**必选参数**，EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)，不能和`group_tp`相同。
- **ep\_world\_size**（`int`）：**必选参数**，EP通信域size。

    <!-- npu="910b" id6 -->
    - <term>Atlas A2系列产品</term>：取值支持16、32、64。
    <!-- end id6 -->
    <!-- npu="A3" id7 -->
    - <term>Atlas A3系列产品</term>：取值支持8、16、32、64、128、144、256、288。
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950DT系列产品</term>：取值支持2、4、8、16、32、64、128、144、256、288。
    <!-- end id8 -->

- **ep\_rank\_id**（`int`）：**必选参数**，EP通信域本卡ID，取值范围\[0, ep\_world\_size\)，同一个EP通信域中各卡的`ep_rank_id`不重复。
- **moe\_expert\_num**（`int`）：**必选参数**，MoE专家数量，取值范围\[1, 512\]，并且满足以下条件：moe\_expert\_num\%\(ep\_world\_size - shared\_expert\_rank\_num\)\=0。

    <!-- npu="910b" id9 -->
    - <term>Atlas A2系列产品</term>：还需满足moe\_expert\_num/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。
    <!-- end id9 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **tp\_send\_counts**（`Tensor`）：**可选参数**，表示本卡每个专家发给TP（Tensor Parallelism）通信域每个卡的数据量。对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`tp_recv_counts`输出。

    <!-- npu="950,910b" id10 -->
    - <term>Atlas A2系列产品</term>、<term>Ascend 950DT系列产品</term>：不支持TP通信域，使用默认输入。
    <!-- end id10 -->
    <!-- npu="A3" id11 -->
    - <term>Atlas A3系列产品</term>：支持TP通信域，要求是一个1维张量，shape为\(tp\_world\_size, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id11 -->

- **x\_active\_mask**（`Tensor`）：**可选参数**，表示输出x中的token是否有效，false表示无效，不接收。

    <!-- npu="950,910b" id12 -->
    - <term>Atlas A2系列产品</term>、<term>Ascend 950DT系列产品</term>：**预留参数，暂未使用，使用默认值即可。**
    <!-- end id12 -->
    <!-- npu="A3" id13 -->
    - <term>Atlas A3系列产品</term>：要求是一个1维张量，shape为\(BS, \)，数据类型支持`torch.bool`，数据格式为$ND$，支持非连续的Tensor。当每张卡的BS数量不一致时，所有token必须全部有效。
    <!-- end id13 -->

- **activation\_scale**（`Tensor`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **weight\_scale**（`Tensor`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **group\_list**（`Tensor`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **expand\_scales**（`Tensor`）：**可选参数**，对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`expand_scales`输出。

    <!-- npu="910b" id14 -->
    - <term>Atlas A2系列产品</term>：必选参数，要求是1维张量，shape为\(A, \)，数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id14 -->
    <!-- npu="950,A3" id15 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id15 -->

- **shared\_expert\_x**（`Tensor`）：**可选参数**，数据类型需与`expand_x`保持一致。仅在共享专家卡数量`shared_expert_rank_num`为0的场景下使用，表示共享专家token，在combine后需要做add的值。

    <!-- npu="950,910b" id16 -->
    - <term>Atlas A2系列产品</term>、<term>Ascend 950DT系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id16 -->
    <!-- npu="A3" id17 -->
    - <term>Atlas A3系列产品</term>：要求是一个2维或3维的张量，当张量为2维时，shape为\(BS, H\)；当张量为3维时，前两维的乘积需等于BS，第三维需等于H。
    <!-- end id17 -->

- **group\_tp**（`str`）：**可选参数**，TP通信域名称，数据并行的通信域。有TP域通信才需要传参。

    <!-- npu="950,910b" id18 -->
    - <term>Atlas A2系列产品</term>、<term>Ascend 950DT系列产品</term>：不支持TP域通信，使用默认值即可。
    <!-- end id18 -->
    <!-- npu="A3" id19 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，字符串长度范围为\[1, 128\)，不能和`group_ep`相同。
    <!-- end id19 -->

- **tp\_world\_size**（`int`）：**可选参数**，TP通信域size。有TP域通信才需要传参。

    <!-- npu="910b" id20 -->
    - <term>Atlas A2系列产品</term>：不支持TP域通信，使用默认值0即可。
    <!-- end id20 -->
    <!-- npu="A3" id21 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，取值范围\[0, 2\]，0和1表示无TP域通信，2表示有TP域通信。
    <!-- end id21 -->
    <!-- npu="950" id22 -->
    - <term>Ascend 950DT系列产品</term>：当前版本不支持，传1即可。
    <!-- end id22 -->

- **tp\_rank\_id**（`int`）：**可选参数**，TP通信域本卡ID。有TP域通信才需要传参，不支持TP域通信传默认值0即可。

    <!-- npu="950,910b" id23 -->
    - <term>Atlas A2系列产品</term>、<term>Ascend 950DT系列产品</term>：不支持TP域通信。
    <!-- end id23 -->
    <!-- npu="A3" id24 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，取值范围\[0, 1\]，同一个TP通信域中各卡的`tp_rank_id`不重复。无TP域通信时，传0即可。
    <!-- end id24 -->

- **expert\_shard\_type**（`int`）：**可选参数**，表示共享专家卡排布类型。默认值为0。

    <!-- npu="910b" id25 -->
    - <term>Atlas A2系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id25 -->
    <!-- npu="950,A3" id26 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：当前仅支持0，表示共享专家卡排在MoE专家卡前面。
    <!-- end id26 -->

- **shared\_expert\_num**（`int`）：**可选参数**，表示共享专家数量，一个共享专家可以复制部署到多个卡上。默认值为1。

    <!-- npu="910b" id27 -->
    - <term>Atlas A2系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id27 -->
    <!-- npu="A3" id28 -->
    - <term>Atlas A3系列产品</term>：取值范围\[0, 4\]，0表示无共享专家。
    <!-- end id28 -->
    <!-- npu="950" id29 -->
    - <term>Ascend 950DT系列产品</term>：当前版本仅支持1。
    <!-- end id29 -->

- **shared\_expert\_rank\_num**（`int`）：**可选参数**，表示共享专家卡数量。

    <!-- npu="910b" id30 -->
    - <term>Atlas A2系列产品</term>：不支持共享专家，传0即可。
    <!-- end id30 -->
    <!-- npu="950,A3" id31 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, ep\_world\_size-1\)。取0表示无共享专家，不取0需满足ep\_world\_size\%shared\_expert\_rank\_num=0。
    <!-- end id31 -->

- **global\_bs**（`int`）：**可选参数**，表示EP域全局的batch size。

    <!-- npu="910b" id32 -->
    - <term>Atlas A2系列产品</term>：当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size或者256\*ep\_world\_size，其中max\_bs表示单rank BS最大值，建议按max\_bs\*ep\_world\_size传入；若固定按256\*ep\_world\_size传入，在后续版本BS大于256的场景下会无法支持；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。
    <!-- end id32 -->
    <!-- npu="950,A3" id33 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size，其中max\_bs表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。
    <!-- end id33 -->

- **out\_dtype**（`int`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**
- **comm\_quant\_mode**（`int`）：**可选参数**，表示通信量化类型。

    <!-- npu="910b" id34 -->
    - <term>Atlas A2系列产品</term>：支持取0和2。0表示通信时不量化，2表示通信时进行`torch.int8`量化。仅当`HCCL_INTRA_PCIE_ENABLE`=1且`HCCL_INTRA_ROCE_ENABLE`=0且驱动版本不低于25.0.RC1.1时才支持取2。
    <!-- end id34 -->
    <!-- npu="950,A3" id35 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：支持取0和2。0表示通信时不量化，2表示通信时进行`torch.int8`量化。当且仅当`tp_world_size`不等于2时，可以开启`torch.int8`量化。
    <!-- end id35 -->

- **group\_list\_type**（`int`）：**可选参数**，**预留参数，暂未使用，使用默认值即可。**

## 返回值说明

`Tensor`

表示处理后的token，要求是2维张量，shape为\(BS, H\)，数据类型与输入`expand_x`保持一致，数据格式为$ND$，不支持非连续的Tensor。

## 约束说明

- 该接口支持推理场景下使用。
<!-- npu="950" id36 -->
- 通信方式约束：

    - <term>Ascend 950DT系列产品</term>：仅支持UB Memory通信。
<!-- end id36 -->

- 该接口支持单算子模式和静态图模式，并且`npu_moe_distribute_dispatch`和`npu_moe_distribute_combine`必须配套使用。
- 在不同产品型号、不同通信算法或不同版本中，`npu_moe_distribute_dispatch`的Tensor输出`expand_idx`、`ep_recv_counts`、`tp_recv_counts`、`expand_scales`中的元素值可能不同，使用时直接将上述Tensor传给`npu_moe_distribute_combine`对应参数即可，模型其他业务逻辑不应对其存在依赖。
- 调用接口过程中使用的`group_ep`、`ep_world_size`、`moe_expert_num`、`group_tp`、`tp_world_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`global_bs`参数取值所有卡需保持一致，`group_ep`、`ep_world_size`、`group_tp`、`tp_world_size`、`expert_shard_type`、`global_bs`网络中不同层也需保持一致，且和[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)对应参数也保持一致。

<!-- npu="A3" id37 -->
- <term>Atlas A3系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
<!-- end id37 -->
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足A=global\_bs\*shared\_expert\_num/shared\_expert\_rank\_num。
        - 对于MoE专家，当`global_bs`为0时，要满足A\>=BS\*ep\_world\_size\*min\(local\_expert\_num, K\)；当`global_bs`非0时，要满足A\>=global\_bs\*min\(local\_expert\_num, K\)。

    - H：表示hidden size隐藏层大小。

        <!-- npu="910b" id38 -->
        - <term>Atlas A2系列产品</term>：取值范围\(0, 7168\]，且保证是32的整数倍。
        <!-- end id38 -->
        <!-- npu="A3" id39 -->
        - <term>Atlas A3系列产品</term>：仅支持7168。
        <!-- end id39 -->
        <!-- npu="950" id40 -->
        - <term>Ascend 950DT系列产品</term>：仅支持7168。
        <!-- end id40 -->

    - BS：表示待发送的token数量。

        <!-- npu="910b" id41 -->
        - <term>Atlas A2系列产品</term>：取值范围为0<BS≤256。
        <!-- end id41 -->
        <!-- npu="950,A3" id42 -->
        - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围为0<BS≤512。
        <!-- end id42 -->

    - K：表示选取topK个专家，需满足0<K≤moe\_expert\_num。

        <!-- npu="910b" id43 -->
        - <term>Atlas A2系列产品</term>：保证取值范围为0<K≤16。
        <!-- end id43 -->
        <!-- npu="A3" id44 -->
        - <term>Atlas A3系列产品</term>：保证取值范围为0<K≤8。
        <!-- end id44 -->
        <!-- npu="950" id45 -->
        - <term>Ascend 950DT系列产品</term>：取值范围为0<K≤8且小于moe\_expert\_num。
        <!-- end id45 -->

    - server\_num：表示服务器的节点数，取值只支持2、4、8。

        <!-- npu="910b" id46 -->
        - <term>Atlas A2系列产品</term>：仅该场景的shape使用了该变量。
        <!-- end id46 -->

    - local\_expert\_num：表示本卡专家数量。
        - 对于共享专家卡，local\_expert\_num=1。
        - 对于MoE专家卡，local\_expert\_num=moe\_expert\_num/\(ep\_world\_size-shared\_expert\_rank\_num\)，当local\_expert\_num\>1时，不支持TP域通信。

- HCCL通信域缓存区大小：

    调用本接口前需检查HCCL\_BUFFSIZE环境变量取值是否合理。

    > [!NOTE]
    > CANN环境变量HCCL\_BUFFSIZE：表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。

    <!-- npu="910b" id47 -->
    - <term>Atlas A2系列产品</term>：要求\>=2\*\(BS\*ep\_world\_size\*min\(local\_expert\_num, K\)\*H\*sizeof\(`torch.uint16`\)+2MB\)。
    <!-- end id47 -->
    <!-- npu="A3" id48 -->
    - <term>Atlas A3系列产品</term>：要求\>=2且满足\>=2\*\(local\_expert\_num\*max\_bs\*ep\_world\_size\*Align512\(Align32\(2\*H\)+64\)+\(K+shared\_expert\_num\)\*max\_bs\*Align512\(2\*H\)\)，local\_expert\_num需使用MoE专家卡的本卡专家数。
    <!-- end id48 -->
    <!-- npu="950" id49 -->
    - <term>Ascend 950DT系列产品</term>：要求\>=2且满足1024^2\*\(HCCL\_BUFFSIZE-2\)/2\>=BS\*2\*\(H+128\)\*\(ep\_world\_size\*local\_expert\_num+K+1\)，local\_expert\_num需使用MoE专家卡的本卡专家数。
    <!-- end id49 -->

<!-- npu="910b" id50 -->
- <term>Atlas A2系列产品</term>：配置环境变量HCCL\_INTRA\_PCIE\_ENABLE=1和HCCL\_INTRA\_ROCE\_ENABLE=0可以减少跨机通信数据量，提升算子性能。此时要求HCCL\_BUFFSIZE\>=moe\_expert\_num\*BS\*\(H\*sizeof\(dtypeX\)+4\*\(\(K+7\)/8\*8\)\*sizeof\(`torch.uint32`\)\)+4MB+100MB。并且，对于入参`moe_expert_num`，只要求moe\_expert\_num\%\(ep\_world\_size - shared\_expert\_rank\_num\)\=0，不要求moe\_expert\_num/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。
<!-- end id50 -->
- 通信域使用约束：
    - 一个模型中的`npu_moe_distribute_dispatch`和`npu_moe_distribute_combine`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的`npu_moe_distribute_dispatch`和`npu_moe_distribute_combine`算子仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

    <!-- npu="A3" id51 -->
    - <term>Atlas A3系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。
    <!-- end id51 -->

<!-- npu="910b" id52 -->
- 组网约束：

    - <term>Atlas A2系列产品</term>：多机场景仅支持交换机组网，不支持双机直连组网。
<!-- end id52 -->

- 公式中的“/”表示整除。

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
    sharedExpertRankNum = 2                      # 共享专家数
    moeExpertNum = 14                            # moe专家数
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
        x = torch_npu.npu_moe_distribute_combine(expand_x=expand_x,
                                                 expert_ids=expert_ids,
                                                 expand_idx=expand_idx,
                                                 ep_send_counts=ep_recv_counts,
                                                 tp_send_counts=tp_recv_counts,
                                                 expert_scales=expert_scales,
                                                 group_ep=ep_hcomm_info,
                                                 group_tp=tp_hcomm_info,
                                                 ep_world_size=ep_world_size,
                                                 tp_world_size=tp_world_size,
                                                 ep_rank_id=rank // tp_world_size,
                                                 tp_rank_id=rank % tp_world_size,
                                                 expert_shard_type=0,
                                                 shared_expert_rank_num=sharedExpertRankNum,
                                                 moe_expert_num=moeExpertNum,
                                                 global_bs=globalBS)
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
    sharedExpertRankNum = 2                      # 共享专家数
    moeExpertNum = 14                            # moe专家数
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
                    scales, quant_mode, global_bs, expert_scales):
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
            output_combine_npu = torch_npu.npu_moe_distribute_combine(expand_x=expand_x_npu,
                                                                      expert_ids=expert_ids,
                                                                      expand_idx=expand_idx_npu,
                                                                      ep_send_counts=ep_recv_counts_npu,
                                                                      tp_send_counts=tp_recv_counts_npu,
                                                                      expert_scales=expert_scales,
                                                                      group_ep=group_ep,
                                                                      group_tp=group_tp,
                                                                      ep_world_size=ep_world_size,
                                                                      tp_world_size=tp_world_size,
                                                                      ep_rank_id=ep_rank_id,
                                                                      tp_rank_id=tp_rank_id,
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

        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
                               rank // tp_world_size, rank % tp_world_size, 0, sharedExpertRankNum, moeExpertNum, scales,
                               quant_mode, globalBS, expert_scales)
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
