# torch_npu.npu_moe_distribute_combine_v2

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

- **API功能**：需与[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)配套使用，相当于按npu\_moe\_distribute\_dispatch\_v2算子收集数据的路径原路返回。
    - 支持数据整合功能，先进行reduce\_scatterv通信，再进行alltoallv通信，最后将接收的数据整合（乘权重再相加）。
    - 支持特殊专家场景（零专家、拷贝专家、常量专家），配合dispatch\_v2的`assist_info_for_combine`实现更高效的全卡同步。

- **计算公式**：
    - 数据整合功能：

        $$
        rs\_out = ReduceScatterV(expand\_x)
        $$

        $$
        ata\_out = AlltoAllv(rs\_out)
        $$

        $$
        x = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)
        $$

    - 特殊专家场景：

        零专家场景，即`zero_expert_num`不为0：

        $$
        Moe(ori\_x)=0
        $$

        拷贝专家场景，即`copy_expert_num`不为0：

        $$
        Moe(ori\_x)=ori\_x
        $$

        常量专家场景，即`const_expert_num`不为0：

        $$
        Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v
        $$

## 函数原型

```python
torch_npu.npu_moe_distribute_combine_v2(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, performance_info=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, comm_quant_mode=0, comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0) -> Tensor
```

## 参数说明

- **expand\_x**（`Tensor`）：**必选参数**，根据`expert_ids`进行扩展过的token特征，要求为2维张量，shape为\(max\(tp\_world\_size, 1\) \* A, H\)，数据格式为$ND$，支持非连续的Tensor。数据类型支持`torch.bfloat16`、`torch.float16`。
    <!-- npu="910b" id69 -->
    - Atlas A2系列产品不支持共享专家场景。
    <!-- end id69 -->

- **expert\_ids**（`Tensor`）：**必选参数**，每个token的topK个专家索引，要求为2维张量，shape为\(BS, K\)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的`expert_ids`输入，张量里value取值范围为\[0, moe\_expert\_num\)，且同一行中的K个value不能重复。
- **assist\_info\_for\_combine**（`Tensor`）：**必选参数**，表示给同一专家发送的token个数，要求为1维张量，shape为\(A \* 128, \)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的`assist_info_for_combine`输出。
- **ep\_send\_counts**（`Tensor`）：**必选参数**，表示本卡每个专家发给EP（Expert Parallelism）域每个卡的token数（token数以前缀和的形式表示），要求为1维张量。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的`ep_recv_counts`输出。

    <!-- npu="910b" id4 -->
    - <term>Atlas A2系列产品</term>：要求shape为\(moe\_expert\_num+2\*global\_bs\*K\*server\_num, \)，前`moe_expert_num`个数表示在EP通信域内，该卡上每个专家收到来自其他各卡的token数（以前缀和的形式表示），2\*global\_bs\*K\*server\_num用于存储机间和机内通信前，combine可提前做reduce操作的token个数和通信区偏移量，`global_bs`传入0时此处按照bs\*ep\_world\_size计算。
    <!-- end id4 -->
    <!-- npu="950,A3" id5 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：要求shape为\(ep\_world\_size\*max\(tp\_world\_size, 1\)\*local\_expert\_num, \)。
    <!-- end id5 -->

- **expert\_scales**（`Tensor`）：**必选参数**，表示每个token的topK个专家的权重。

    <!-- npu="A3,910b" id6 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：要求为2维张量，shape为\(BS, K\)，其中共享专家不需要乘权重系数，直接相加即可。数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - <term>Ascend 950DT系列产品</term>：支持传入空Tensor或2维张量，传入非空Tensor时shape为\(BS, K\)，其中共享专家不需要乘权重系数，直接相加即可。数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id7 -->

- **group\_ep**（`str`）：**必选参数**，EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)。

    <!-- npu="A3" id8 -->
    - <term>Atlas A3系列产品</term>：取值不能和`group_tp`相同。
    <!-- end id8 -->

- **ep\_world\_size**（`int`）：**必选参数**，EP通信域size。

    <!-- npu="910b" id9 -->
    - <term>Atlas A2系列产品</term>：`ep_world_size`的取值范围如下所示。
        - `comm_alg`设置为"fullmesh"时，`ep_world_size`取值范围为16、32、64、128、256。
        - `comm_alg`设置为"hierarchy"时，`ep_world_size`取值范围为16、32、64。
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3系列产品</term>：取值支持\[2, 768\]。`comm_alg`设置为"hierarchy"时，取值范围为\[16, 256\]，且为16的整数倍。
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950DT系列产品</term>：取值支持\[2, 1024\]。
    <!-- end id11 -->

- **ep\_rank\_id**（`int`）：**必选参数**，EP通信域本卡ID，取值范围\[0, ep\_world\_size\)，同一个EP通信域中各卡的`ep_rank_id`不重复。
- **moe\_expert\_num**（`int`）：**必选参数**，MoE专家数量，取值范围\[1, 1024\]，并且满足以下条件：moe\_expert\_num\%\(ep\_world\_size - shared\_expert\_rank\_num\)\=0。

    <!-- npu="A3" id12 -->
    - <term>Atlas A3系列产品</term>：`comm_alg`设置为"hierarchy"时，取值范围为\(0, 512\]。
    <!-- end id12 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **tp\_send\_counts**（`Tensor`）：**可选参数**，表示本卡每个专家发给TP（Tensor Parallelism）通信域每个卡的数据量。对应[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的`tp_recv_counts`输出。

    <!-- npu="950,910b" id13 -->
    - <term>Atlas A2系列产品</term>、<term>Ascend 950DT系列产品</term>：不支持TP通信域，使用默认输入。
    <!-- end id13 -->
    <!-- npu="A3" id14 -->
    - <term>Atlas A3系列产品</term>：支持TP通信域，要求为1维张量，shape为\(tp\_world\_size, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id14 -->

- **x\_active\_mask**（`Tensor`）：**可选参数**，表示token是否参与通信。

    <!-- npu="910b" id15 -->
    - <term>Atlas A2系列产品</term>：
        - `comm_alg`设置为"fullmesh"时，要求为一个1维张量，shape为\(BS, \)。数据类型支持`torch.bool`，数据格式为$ND$，支持非连续的Tensor。参数为true表示对应的token参与通信，true必须排到false之前，例：\{true, false, true\}为非法输入。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。
        - `comm_alg`设置为"hierarchy"时，当前版本不支持，使用默认值即可。
    <!-- end id15 -->
    <!-- npu="950,A3" id16 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：要求为一个1维或2维张量。当输入为1维时，shape为\(BS, \)；当输入为2维时，shape为\(BS, K\)。数据类型支持`torch.bool`，数据格式为$ND$，支持非连续的Tensor。当输入为1维时，参数为true表示对应的token参与通信，true必须排到false之前，例：\{true, false, true\}为非法输入；当输入为2维时，参数为true表示当前token对应的`expert_ids`参与通信，若当前token对应的K个`torch.bool`值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。
    <!-- end id16 -->

- **expand\_scales**（`Tensor`）：**可选参数**，对应[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)的`expand_scales`输出。

    <!-- npu="910b" id17 -->
    - <term>Atlas A2系列产品</term>：必选参数，要求为1维张量，shape为\(A, \)，数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id17 -->
    <!-- npu="950,A3" id18 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id18 -->

- **shared\_expert\_x**（`Tensor`）：**可选参数**，数据类型需与`expand_x`保持一致。仅在共享专家卡数量`shared_expert_rank_num`为0的场景下使用，表示共享专家token，在combine后需要做add的值。

    <!-- npu="910b" id19 -->
    - <term>Atlas A2系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id19 -->
    <!-- npu="950,A3" id20 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：要求为一个2维或3维的张量，当张量为2维时，shape为\(BS, H\)；当张量为3维时，前两维的乘积需等于BS，第三维需等于H。
    <!-- end id20 -->

- **elastic\_info**（`Tensor`）：**可选参数**，预留参数，当前版本不支持，传默认值None即可。
- **ori\_x**（`Tensor`）：**可选参数**，表示未经过FFN的token数据，在`copy_expert_num`不为0或`const_expert_num`不为0的场景下需要本输入数据。

    <!-- npu="910b" id21 -->
    - <term>Atlas A2系列产品</term>：预留参数，当前版本不支持，传None即可。
    <!-- end id21 -->
    <!-- npu="950,A3" id22 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：可选择传入有效数据或填None，当`copy_expert_num`不为0或`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求是一个2维张量，shape为\(BS, H\)，数据类型需跟`expand_x`保持一致；数据格式为$ND$，支持非连续的Tensor。
    <!-- end id22 -->

- **const\_expert\_alpha\_1**（`Tensor`）：**可选参数**，在`const_expert_num`不为0的场景下需要输入的计算系数。

    <!-- npu="910b" id23 -->
    - <term>Atlas A2系列产品</term>：预留参数，当前版本不支持，传None即可。
    <!-- end id23 -->
    <!-- npu="950,A3" id24 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求是一个2维张量，shape为\(const\_expert\_num, H\)，数据类型需跟`expand_x`保持一致；数据格式为$ND$，支持非连续的Tensor。
    <!-- end id24 -->

- **const\_expert\_alpha\_2**（`Tensor`）：**可选参数**，在`const_expert_num`不为0的场景下需要输入的计算系数。

    <!-- npu="910b" id25 -->
    - <term>Atlas A2系列产品</term>：预留参数，当前版本不支持，传None即可。
    <!-- end id25 -->
    <!-- npu="950,A3" id26 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求是一个2维张量，shape为\(const\_expert\_num, H\)，数据类型需跟`expand_x`保持一致；数据格式为$ND$，支持非连续的Tensor。
    <!-- end id26 -->

- **const\_expert\_v**（`Tensor`）：**可选参数**，在`const_expert_num`不为0的场景下需要输入的计算系数。

    <!-- npu="910b" id27 -->
    - <term>Atlas A2系列产品</term>：预留参数，当前版本不支持，传None即可。
    <!-- end id27 -->
    <!-- npu="950,A3" id28 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求是一个2维张量，shape为\(const\_expert\_num, H\)，数据类型需跟`expand_x`保持一致；数据格式为$ND$，支持非连续的Tensor。
    <!-- end id28 -->

- **performance\_info**（`Tensor`）：**可选参数**，表示本卡等待各卡数据的通信时间，单位为us（微秒）。单次算子调用各卡通信耗时会累加到该Tensor上，算子内部不进行自动清零，因此每次启用此Tensor开始记录耗时前需对Tensor清零。当传入None时表示不开启记录通信耗时功能；当传入有效数据时，要求是一个1维张量，shape为\(ep\_world\_size, \)，数据类型支持`torch.int64`，数据格式为$ND$，支持非连续的Tensor。
- **group\_tp**（`str`）：**可选参数**，TP通信域名称，数据并行的通信域。有TP域通信才需要传参，若无TP域通信，使用默认值""即可。

    <!-- npu="910b" id29 -->
    - <term>Atlas A2系列产品</term>：eager模式使用默认值即可，图模式传入与`group_ep`相同。
    <!-- end id29 -->
    <!-- npu="A3" id30 -->
    - <term>Atlas A3系列产品</term>：字符串长度范围为\[1, 128\)，不能和`group_ep`相同。
    <!-- end id30 -->
    <!-- npu="950" id31 -->
    - <term>Ascend 950DT系列产品</term>：不支持TP域通信，使用默认输入。
    <!-- end id31 -->

- **tp\_world\_size**（`int`）：**可选参数**，TP通信域size。有TP域通信才需要传参。

    <!-- npu="910b" id32 -->
    - <term>Atlas A2系列产品</term>：不支持TP域通信，使用默认值0即可。
    <!-- end id32 -->
    <!-- npu="A3" id33 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，取值范围\[0, 2\]，0和1表示无TP域通信，2表示有TP域通信。
    <!-- end id33 -->
    <!-- npu="950" id34 -->
    - <term>Ascend 950DT系列产品</term>：不支持TP域通信，使用默认值0即可。
    <!-- end id34 -->

- **tp\_rank\_id**（`int`）：**可选参数**，TP通信域本卡ID。有TP域通信才需要传参。

    <!-- npu="910b" id35 -->
    - <term>Atlas A2系列产品</term>：不支持TP域通信，使用默认值0即可。
    <!-- end id35 -->
    <!-- npu="A3" id36 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，取值范围\[0, 1\]，同一个TP通信域中各卡的`tp_rank_id`不重复。无TP域通信时，传0即可。
    <!-- end id36 -->
    <!-- npu="950" id37 -->
    - <term>Ascend 950DT系列产品</term>：不支持TP域通信，使用默认值0即可。
    <!-- end id37 -->

- **expert\_shard\_type**（`int`）：**可选参数**，表示共享专家卡排布类型。

    <!-- npu="910b" id38 -->
    - <term>Atlas A2系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id38 -->
    <!-- npu="950,A3" id39 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：当前仅支持0，表示共享专家卡排在MoE专家卡前面。
    <!-- end id39 -->

- **shared\_expert\_num**（`int`）：**可选参数**，表示共享专家数量，一个共享专家可以复制部署到多个卡上。

    <!-- npu="910b" id40 -->
    - <term>Atlas A2系列产品</term>：暂不支持该参数，使用默认值1即可。
    <!-- end id40 -->
    <!-- npu="950,A3" id41 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, 4\]，0表示无共享专家，默认值为1。
    <!-- end id41 -->

- **shared\_expert\_rank\_num**（`int`）：**可选参数**，表示共享专家卡数量。

    <!-- npu="910b" id42 -->
    - <term>Atlas A2系列产品</term>：不支持共享专家，使用默认值即可。
    <!-- end id42 -->
    <!-- npu="950,A3" id43 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, ep\_world\_size\)。取0表示无共享专家，不取0需满足shared\_expert\_rank\_num\%shared\_expert\_num=0。
    <!-- end id43 -->

- **global\_bs**（`int`）：**可选参数**，表示EP域全局的batch size。当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size，其中`max_bs`表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。
- **comm\_quant\_mode**（`int`）：**可选参数**，表示通信量化类型。

    <!-- npu="910b" id44 -->
    - <term>Atlas A2系列产品</term>：支持取0和2。0表示通信时不量化，2表示通信时进行`torch.int8`量化。仅当`comm_alg`配置为"hierarchy"或`HCCL_INTRA_PCIE_ENABLE`=1且`HCCL_INTRA_ROCE_ENABLE`=0且驱动版本不低于25.0.RC1.1时才支持取2。
    <!-- end id44 -->
    <!-- npu="A3" id45 -->
    - <term>Atlas A3系列产品</term>：支持取0和2。0表示通信时不量化，2表示通信时进行`torch.int8`量化。当且仅当`tp_world_size`不等于2时，可以开启`torch.int8`量化。
    <!-- end id45 -->
    <!-- npu="950" id46 -->
    - <term>Ascend 950DT系列产品</term>：取值范围0、2、3或4。0表示不量化，2表示`torch.int8`量化，3表示mxfp量化输出`torch.float8_e5m2`类型，4表示mxfp量化输出`torch.float8_e4m3fn`类型。
    <!-- end id46 -->

- **comm\_alg**（`str`）：**可选参数**，表示通信亲和内存布局算法。

    <!-- npu="910b" id47 -->
    - <term>Atlas A2系列产品</term>：当前版本支持""、"fullmesh"、"hierarchy"三种输入方式。推荐配置"hierarchy"并搭配25.0.RC1.1及以上版本驱动使用。
        - "": 配置HCCL\_INTRA\_PCIE\_ENABLE=1和HCCL\_INTRA\_ROCE\_ENABLE=0时，调用"hierarchy"算法，否则调用"fullmesh"算法。不推荐使用该方式。
        - "fullmesh"：Token数据直接通过RDMA方式发回目标卡。
        - "hierarchy"：Token数据经过机内、跨机两次发送，先在server内将同一个token数据汇总求和，再跨机发送，以减少跨机数据量。
    <!-- end id47 -->
    <!-- npu="950,A3" id48 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id48 -->

- **zero\_expert\_num**（`int`）：**可选参数**，表示零专家的数量。

    <!-- npu="910b" id49 -->
    - <term>Atlas A2系列产品</term>：当前版本不支持，传0即可。
    <!-- end id49 -->
    <!-- npu="950,A3" id50 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, MAX\_INT32\)，MAX\_INT32 = 2^31 - 1，合法的零专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num\)。
    <!-- end id50 -->

- **copy\_expert\_num**（`int`）：**可选参数**，表示拷贝专家的数量。

    <!-- npu="910b" id51 -->
    - <term>Atlas A2系列产品</term>：当前版本不支持，传0即可。
    <!-- end id51 -->
    <!-- npu="950,A3" id52 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, MAX\_INT32\)，MAX\_INT32 = 2^31 - 1，合法的拷贝专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num\)。
    <!-- end id52 -->

- **const\_expert\_num**（`int`）：**可选参数**，表示常量专家的数量。

    <!-- npu="910b" id53 -->
    - <term>Atlas A2系列产品</term>：当前版本不支持，传0即可。
    <!-- end id53 -->
    <!-- npu="950,A3" id54 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, MAX\_INT32\)，MAX\_INT32 = 2^31 - 1，合法的常量专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num+const\_expert\_num\)。
    <!-- end id54 -->

## 返回值说明

`Tensor`

表示处理后的token，要求为2维张量，shape为\(BS, H\)，数据类型支持`torch.bfloat16`、`torch.float16`，与输入`expand_x`保持一致，数据格式为$ND$，不支持非连续的Tensor。

## 约束说明

- 该接口支持推理场景下使用。
- 通信方式约束：

    <!-- npu="950" id55 -->
    - <term>Ascend 950DT系列产品</term>：仅支持UB Memory通信。
    <!-- end id55 -->

- 该接口支持单算子模式和静态图模式调用，`npu_moe_distribute_dispatch_v2`和`npu_moe_distribute_combine_v2`必须配套使用。
- 在不同产品型号、不同通信算法或不同版本中，`npu_moe_distribute_dispatch_v2`的Tensor输出`assist_info_for_combine`、`ep_recv_counts`、`tp_recv_counts`、`expand_scales`中的元素值可能不同，使用时直接将上述Tensor传给`npu_moe_distribute_combine_v2`对应参数即可，模型其他业务逻辑不应对其存在依赖。
- 调用接口过程中使用的`group_ep`、`ep_world_size`、`moe_expert_num`、`group_tp`、`tp_world_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`global_bs`参数取值所有卡需保持一致，`group_ep`、`ep_world_size`、`group_tp`、`tp_world_size`、`expert_shard_type`、`global_bs`网络中不同层也需保持一致，且和[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)对应参数也保持一致。

<!-- npu="A3" id56 -->
- <term>Atlas A3系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
<!-- end id56 -->
- moe\_expert\_num + zero\_expert\_num + copy\_expert\_num + const\_expert\_num < MAX\_INT32。
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足A=BS\*ep\_world\_size\*shared\_expert\_num/shared\_expert\_rank\_num。
        - 对于MoE专家，当`global_bs`为0时，要满足A\>=BS\*ep\_world\_size\*min\(local\_expert\_num, K\)；当`global_bs`不为0时，要满足A\>=global\_bs\*min\(local\_expert\_num, K\)。

    - H：表示hidden size隐藏层大小。

        <!-- npu="910b" id57 -->
        - <term>Atlas A2系列产品</term>：`H`的取值范围如下所示。
            - `comm_alg`设置为"fullmesh"时，`H`的取值范围\(0, 7168\]，且保证是32的整数倍。
            - `comm_alg`设置为"hierarchy"且驱动版本不低于25.0.RC1.1时，`H`的取值范围\(0, 10 \* 1024\]，且保证是32的整数倍。
        <!-- end id57 -->
        <!-- npu="950,A3" id58 -->
        - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[1024, 8192\]。
        <!-- end id58 -->

    - BS：表示待发送的token数量。

        <!-- npu="910b" id59 -->
        - <term>Atlas A2系列产品</term>：取值范围为0<BS≤256。
        <!-- end id59 -->
        <!-- npu="950,A3" id60 -->
        - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围为0<BS≤512。
        <!-- end id60 -->

    - K：表示选取topK个专家，取值范围为0<K≤16，同时满足0 < K ≤ moe\_expert\_num + zero\_expert\_num + copy\_expert\_num + const\_expert\_num。
    - server\_num：表示服务器的节点数，取值只支持2、4、8。

        <!-- npu="910b" id61 -->
        - <term>Atlas A2系列产品</term>：仅该场景的shape使用了该变量。
        <!-- end id61 -->

    - local\_expert\_num：表示本卡专家数量。
        - 对于共享专家卡，local\_expert\_num=1。
        - 对于MoE专家卡，local\_expert\_num=moe\_expert\_num/\(ep\_world\_size-shared\_expert\_rank\_num\)，当local\_expert\_num\>1时，不支持TP域通信。

        <!-- npu="950,A3" id62 -->
        - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：应满足0 < local\_expert\_num \* ep\_world\_size ≤ 2048。
        <!-- end id62 -->

- HCCL通信域缓存区大小：

    调用本接口前需检查通信域缓存区大小取值是否合理，单位MB，不配置时默认为200MB。

    <!-- npu="910b" id63 -->
    - <term>Atlas A2系列产品</term>：该场景支持通过环境变量HCCL\_BUFFSIZE配置。
        - `comm_alg`配置为"": 仅在此配置下HCCL\_INTRA\_PCIE\_ENABLE和HCCL\_INTRA\_ROCE\_ENABLE生效，依照HCCL\_INTRA\_PCIE\_ENABLE和HCCL\_INTRA\_ROCE\_ENABLE配置选择"fullmesh"或"hierarchy"公式。
        - `comm_alg`配置为"fullmesh"：设置大小要求\>=2\*\(BS\*ep\_world\_size\*min\(local\_expert\_num, K\)\*H\*sizeof\(`torch.uint16`\)+2MB\)。
        - `comm_alg`配置为"hierarchy"：设置大小要求=moe\_expert\_num\*BS\*\(H\*sizeof\(dtype\_x\)+4\*\(\(K+7\)/8\*8\)\*sizeof\(`torch.uint32`\)\)+4MB+100MB，不要求moe\_expert\_num/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。
    <!-- end id63 -->
    <!-- npu="A3" id64 -->
    - <term>Atlas A3系列产品</term>：该场景不仅支持通过环境变量HCCL\_BUFFSIZE配置，还支持通过hccl\_buffer\_size配置（参考[《PyTorch训练模型迁移调优》](https://hiascend.com/document/redirect/canncommercial-ptmigr)中“性能调优\>性能调优方法\>通信优化\>优化方法\>hccl\_buffer\_size”章节）。
        - ep通信域内：设置大小要求\>=2且满足\>=2\*\(local\_expert\_num\*max\_bs\*ep\_world\_size\*Align512\(Align32\(2\*H\)+64\)+\(K+shared\_expert\_num\)\*max\_bs\*Align512\(2\*H\)\)。
        - tp通信域内：设置大小要求\>=(A \* Align512(Align32(H \* 2) + 44) + A \* Align512(H \* 2)) \* 2。
        - 其中480Align512(x) = ((x+480-1)/480)\*512，Align512(x) = ((x+512-1)/512)\*512，Align32(x) = ((x+32-1)/32)\*32。
    <!-- end id64 -->
    <!-- npu="950" id65 -->
    - <term>Ascend 950DT系列产品</term>：要求\>=2且满足\>=2\*\(local\_expert\_num\*max\_bs\*ep\_world\_size\*Align512\(Align32\(2\*H\)+64\)+\(K+shared\_expert\_num\)\*max\_bs\*Align512\(2\*H\)\) + 1024 \* 1024 \* 10。其中480Align512(x) = ((x+480-1)/480)\*512，Align512(x) = ((x+512-1)/512)\*512，Align32(x) = ((x+32-1)/32)\*32。
    <!-- end id65 -->

<!-- npu="910b" id66 -->
- <term>Atlas A2系列产品</term>：HCCL\_INTRA\_PCIE\_ENABLE和HCCL\_INTRA\_ROCE\_ENABLE环境变量不再推荐使用，建议`comm_alg`配置"hierarchy"。
<!-- end id66 -->
- 通信域使用约束：
    - 一个模型中的`npu_moe_distribute_dispatch_v2`和`npu_moe_distribute_combine_v2`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的`npu_moe_distribute_dispatch_v2`和`npu_moe_distribute_combine_v2`算子仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

    <!-- npu="A3" id67 -->
    - <term>Atlas A3系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。
    <!-- end id67 -->

<!-- npu="910b" id68 -->
- 组网约束：

    - <term>Atlas A2系列产品</term>：多机场景仅支持交换机组网，不支持双机直连组网。
<!-- end id68 -->

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
    quant_mode = 2  # 2为动态量化
    is_dispatch_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    shared_expert_rank_num = 0  # 共享专家数
    moe_expert_num = 32  # moe专家数
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = moe_expert_num // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 1

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v

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

    def warm_up_dispatch(rank, group_ep, group_tp):
        x_warm_up = torch.empty(size=[1, h], dtype=input_dtype).uniform_(-1024, 1024).to(input_dtype).npu()
        expert_ids_warm_up = torch.arange(0, k, dtype=torch.int32).unsqueeze(0).npu()
        dispatch_kwargs_before = get_dispatch_kwargs_warmup(
            x_warm_up=x_warm_up,
            expert_ids_warm_up=expert_ids_warm_up,
            group_ep=group_ep,
            group_tp=group_tp,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
        )
        (
            expand_x, dynamic_scales, expand_idx,
            expert_token_nums, ep_recv_counts, tp_recv_counts, _
        ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs_before)
        return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts

    def get_dispatch_kwargs_warmup(
        x_warm_up, expert_ids_warm_up, group_ep, group_tp, ep_rank_id, tp_rank_id,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        expert_ids_warm_up = expert_ids_warm_up.to(torch.int32).npu()
        return {
            'x': x_warm_up,
            'expert_ids': expert_ids_warm_up,
            'x_active_mask': None,
            'group_ep': group_ep,
            'group_tp': group_tp,
            'ep_rank_id': ep_rank_id,
            'tp_rank_id': tp_rank_id,
            'ep_world_size': ep_world_size,
            'tp_world_size': tp_world_size,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'moe_expert_num': moe_expert_num,
            'scales': None,
            'quant_mode': 2,
            'global_bs': 16,
        }

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)
        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = torch.tensor([[5, 7, 17, 4, 2, 6, 11, 16],
                                [10, 12, 13, 15, 19, 4, 18, 1],
                                [19, 33, 1, 17, 9, 5, 0, 32],
                                [19, 11, 17, 0, 10, 5, 7, 9],
                                [10, 16, 11, 17, 33, 8, 9, 3],
                                [12, 19, 5, 7, 1, 3, 18, 16],
                                [11, 9, 13, 16, 12, 33, 17, 14],
                                [16, 4, 9, 5, 0, 10, 11, 17]], dtype=torch.int32).npu()
        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
        const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
        const_expert_v = gen_const_expert_v().npu()
        out = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)

        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_world_size=ep_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            expert_shard_type=0,
            shared_expert_rank_num=shared_expert_rank_num,
            moe_expert_num=moe_expert_num,
            scales=scales,
            quant_mode=quant_mode,
            global_bs=globalBS,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num)

        if is_quant:
            expand_x = expand_x.to(input_dtype)

        x = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x,
                                                expert_ids=expert_ids,
                                                assist_info_for_combine=assist_info_for_combine,
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
                                                shared_expert_rank_num=shared_expert_rank_num,
                                                moe_expert_num=moe_expert_num,
                                                global_bs=globalBS,
                                                ori_x=x,
                                                const_expert_alpha_1=const_expert_alpha_1,
                                                const_expert_alpha_2=const_expert_alpha_2,
                                                const_expert_v=const_expert_v,
                                                zero_expert_num=zero_expert_num,
                                                copy_expert_num=copy_expert_num,
                                                const_expert_num=const_expert_num)
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
            print("shared_expert_rank_num 不能大于 ep_world_size")
            exit(0)
        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size 必须是 shared_expert_rank_num的整数倍")
            exit(0)
        if moe_expert_num % moe_rank_num != 0:
            print("moe_expert_num 必须是 moe_rank_num 的整数倍")
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
    import time

    # 控制模式
    quant_mode = 2  # 2为动态量化
    is_dispatch_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    shared_expert_rank_num = 0  # 共享专家数
    moe_expert_num = 32  # moe专家数
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = moe_expert_num // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 1

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, expert_ids, group_ep, group_tp, ep_world_size, tp_world_size,
                    ep_rank_id, tp_rank_id, expert_shard_type, shared_expert_rank_num, moe_expert_num,
                    scales, quant_mode, global_bs, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
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
                global_bs=global_bs,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )

            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)

            output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(
                expand_x=expand_x_npu,
                expert_ids=expert_ids,
                assist_info_for_combine=assist_info_for_combine_npu,
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
                global_bs=global_bs,
                elastic_info=elastic_info,
                ori_x=x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v

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

    def warm_up_dispatch(rank, group_ep, group_tp):
        x_warm_up = torch.empty(size=[1, h], dtype=input_dtype).uniform_(-1024, 1024).to(input_dtype).npu()
        expert_ids_warm_up = torch.arange(0, k, dtype=torch.int32).unsqueeze(0).npu()

        dispatch_kwargs_before = get_dispatch_kwargs_warmup(
            x_warm_up=x_warm_up,
            expert_ids_warm_up=expert_ids_warm_up,
            group_ep=group_ep,
            group_tp=group_tp,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
        )

        (
            expand_x, dynamic_scales, expand_idx,
            expert_token_nums, ep_recv_counts, tp_recv_counts, _
        ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs_before)
        return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts

    def get_dispatch_kwargs_warmup(
        x_warm_up, expert_ids_warm_up, group_ep, group_tp, ep_rank_id, tp_rank_id,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        expert_ids_warm_up = expert_ids_warm_up.to(torch.int32).npu()

        return {
            'x': x_warm_up,
            'expert_ids': expert_ids_warm_up,
            'x_active_mask': None,
            'group_ep': group_ep,
            'group_tp': group_tp,
            'ep_rank_id': ep_rank_id,
            'tp_rank_id': tp_rank_id,
            'ep_world_size': ep_world_size,
            'tp_world_size': tp_world_size,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'moe_expert_num': moe_expert_num,
            'scales': None,
            'quant_mode': 2,
            'global_bs': 16,
        }

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = torch.tensor([
            [0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]
        ], dtype=torch.int32).npu()
        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        elastic_info = None
        const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
        const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
        const_expert_v = gen_const_expert_v().npu()
        out = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(
            x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
            rank // tp_world_size, rank % tp_world_size, 0, shared_expert_rank_num, moe_expert_num, scales,
            quant_mode, globalBS, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
            zero_expert_num, copy_expert_num, const_expert_num
        )
        torch.npu.synchronize()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

        time.sleep(10)

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
            print("shared_expert_rank_num 不能大于 ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size 必须是 shared_expert_rank_num的整数倍")
            exit(0)

        if moe_expert_num % moe_rank_num != 0:
            print("moe_expert_num 必须是 moe_rank_num 的整数倍")
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
