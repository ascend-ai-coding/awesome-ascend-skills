# torch_npu.npu_moe_distribute_dispatch

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

- **API功能**：需与[torch\_npu.npu\_moe\_distribute\_combine](torch_npu-npu_moe_distribute_combine.md)配套使用，完成MoE的并行部署下的token dispatch与combine。对token数据先进行量化（可选），再进行EP（Expert Parallelism）域的alltoallv通信，再进行TP（Tensor Parallelism）域的allgatherv通信（可选）。

- **计算公式**：`x`表示输入`x`，`scales`表示输入`scales`，`quant_mode`表示输入`quant_mode`。
    - 如果`quant_mode`不为2（非动态量化）：

        $$
        quant\_out =
        \begin{cases}
        x, & quant\_mode=0 \\
        CastToInt8(CastToFp32(x) \times scales), & quant\_mode \ne 0
        \end{cases}
        $$

        $$
        alltoall\_x\_out = alltoallv(quant\_out)
        $$

        $$
        expand\_x =
        \begin{cases}
        allgatherv(alltoall\_x\_out), & 有TP通信域 \\
        alltoall\_x\_out, & 无TP通信域
        \end{cases}
        $$

    - 如果`quant_mode`为2（动态量化）：

        $$
        x\_fp32 = CastToFp32(x) \times scales
        $$

        $$
        dynamic\_scales\_value = 127.0 / Max(Abs(x\_fp32))
        $$

        $$
        quant\_out = CastToInt8(x\_fp32 \times dynamic\_scales\_value)
        $$

        $$
        alltoall\_x\_out = alltoallv(quant\_out)
        $$

        $$
        alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales)
        $$

        $$
        expand\_x =
        \begin{cases}
        allgatherv(alltoall\_x\_out), & 有TP通信域 \\
        alltoall\_x\_out, & 无TP通信域
        \end{cases}
        $$

        $$
        dynamic\_scales =
        \begin{cases}
        allgatherv(alltoall\_dynamic\_scales\_out), & 有TP通信域 \\
        alltoall\_dynamic\_scales\_out, & 无TP通信域
        \end{cases}
        $$

## 函数原型

```python
torch_npu.npu_moe_distribute_dispatch(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, scales=None, x_active_mask=None, expert_scales=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：**必选参数**，表示计算使用的token数据，需根据`expert_ids`来发送给其他卡。要求为2维张量，shape为\(BS, H\)，表示有BS个token。数据格式为$ND$，支持非连续的Tensor。数据类型支持`torch.float16`、`torch.bfloat16`。
- **expert\_ids**（`Tensor`）：**必选参数**，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为\(BS, K\)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_combine](torch_npu-npu_moe_distribute_combine.md)的`expert_ids`输入，张量里value取值范围为\[0, moe\_expert\_num\)，且同一行中的K个value不能重复。
- **group\_ep**（`str`）：**必选参数**，EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)，不能和`group_tp`相同。
- **ep\_world\_size**（`int`）：**必选参数**，EP通信域size。

    <!-- npu="910b" id4 -->
    - <term>Atlas A2系列产品</term>：取值支持16、32、64。
    <!-- end id4 -->
    <!-- npu="A3" id5 -->
    - <term>Atlas A3系列产品</term>：取值支持8、16、32、64、128、144、256、288。
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950DT系列产品</term>：取值支持2、4、8、16、32、64、128、144、256、288。
    <!-- end id6 -->

- **ep\_rank\_id**（`int`）：**必选参数**，EP通信域本卡ID，取值范围\[0, ep\_world\_size\)，同一个EP通信域中各卡的`ep_rank_id`不重复。
- **moe\_expert\_num**（`int`）：**必选参数**，MoE专家数量，取值范围\[1, 512\]，并且满足条件moe\_expert\_num\%\(ep\_world\_size - shared\_expert\_rank\_num\)=0。

    <!-- npu="910b" id7 -->
    - <term>Atlas A2系列产品</term>：还需满足moe\_expert\_num/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。
    <!-- end id7 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **scales**（`Tensor`）：**可选参数**，表示每个专家的权重，非量化场景不传，动态量化场景可传可不传。若传值要求为2维张量，如果有共享专家，shape为\(shared\_expert\_num+moe\_expert\_num, H\)，如果没有共享专家，shape为\(moe\_expert\_num, H\)。数据格式为$ND$，不支持非连续的Tensor。

    <!-- npu="910b" id8 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`。当HCCL\_INTRA\_PCIE\_ENABLE为1且HCCL\_INTRA\_ROCE\_ENABLE为0时，要求传默认值。
    <!-- end id8 -->
    <!-- npu="950,A3" id9 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：数据类型支持`torch.float32`。当`quant_mode`取值为0时，不用传参；当`quant_mode`取值为2时，可传入有效数据或传空，传入有效数据时，其shape为\(H, \)、\(shared\_expert\_num+moe\_expert\_num, H\)、\(moe\_expert\_num, H\)。
    <!-- end id9 -->

- **x\_active\_mask**（`Tensor`）：**可选参数**，表示x中的token是否有效，false表示无效。

    <!-- npu="910b" id10 -->
    - <term>Atlas A2系列产品</term>：预留参数，当前版本不支持，使用默认值即可。
    <!-- end id10 -->
    <!-- npu="A3" id11 -->
    - <term>Atlas A3系列产品</term>：要求是一个1维张量，shape为\(BS, \)，数据类型支持`torch.bool`，数据格式为$ND$，支持非连续的Tensor。默认值为所有token全部有效；当每张卡的BS数量不一致时，所有token必须全部有效。
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950DT系列产品</term>：预留参数，当前版本不支持，传默认值即可。
    <!-- end id12 -->

- **expert\_scales**（`Tensor`）：**可选参数**，表示每个token的topK个专家权重。

    <!-- npu="910b" id13 -->
    - <term>Atlas A2系列产品</term>：要求为2维张量，shape为\(BS, K\)，数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id13 -->
    <!-- npu="950,A3" id14 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：预留参数，当前版本不支持，传默认值即可。
    <!-- end id14 -->

- **group\_tp**（`str`）：**可选参数**，TP通信域名称，数据并行的通信域。若有TP域通信需要传参，若无TP域通信，使用默认值即可。

    <!-- npu="910b" id15 -->
    - <term>Atlas A2系列产品</term>：不支持TP域通信，使用默认值即可。
    <!-- end id15 -->
    <!-- npu="A3" id16 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，字符串长度范围为\[1, 128\)，不能和`group_ep`相同。
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950DT系列产品</term>：预留参数，当前版本不支持，传默认值即可。
    <!-- end id17 -->

- **tp\_world\_size**（`int`）：**可选参数**，TP通信域size。有TP域通信才需要传参。

    <!-- npu="910b" id18 -->
    - <term>Atlas A2系列产品</term>：不支持TP域通信，使用默认值0即可。
    <!-- end id18 -->
    <!-- npu="A3" id19 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，取值范围\[0, 2\]，0和1表示无TP域通信，2表示有TP域通信。
    <!-- end id19 -->
    <!-- npu="950" id20 -->
    - <term>Ascend 950DT系列产品</term>：当前版本仅支持传1。
    <!-- end id20 -->

- **tp\_rank\_id**（`int`）：**可选参数**，TP通信域本卡ID。有TP域通信才需要传参。

    <!-- npu="910b" id21 -->
    - <term>Atlas A2系列产品</term>：不支持TP域通信，使用默认值即可。
    <!-- end id21 -->
    <!-- npu="A3" id22 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，取值范围\[0, 1\]，默认为0，同一个TP通信域中各卡的`tp_rank_id`不重复。无TP域通信时，传0即可。
    <!-- end id22 -->
    <!-- npu="950" id23 -->
    - <term>Ascend 950DT系列产品</term>：不支持TP域通信，传0即可。
    <!-- end id23 -->

- **expert\_shard\_type**（`int`）：**可选参数**，表示共享专家卡排布类型。

    <!-- npu="910b" id24 -->
    - <term>Atlas A2系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id24 -->
    <!-- npu="950,A3" id25 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：当前仅支持0，表示共享专家卡排在MoE专家卡前面。
    <!-- end id25 -->

- **shared\_expert\_num**（`int`）：**可选参数**，表示共享专家数量，一个共享专家可以复制部署到多个卡上。

    <!-- npu="910b" id26 -->
    - <term>Atlas A2系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id26 -->
    <!-- npu="A3" id27 -->
    - <term>Atlas A3系列产品</term>：取值范围\[0, 4\]，0表示无共享专家，默认值为1。
    <!-- end id27 -->
    <!-- npu="950" id28 -->
    - <term>Ascend 950DT系列产品</term>：当前版本仅支持传1，1表示一个共享专家。
    <!-- end id28 -->

- **shared\_expert\_rank\_num**（`int`）：**可选参数**，表示共享专家卡数量。

    <!-- npu="910b" id29 -->
    - <term>Atlas A2系列产品</term>：不支持共享专家，传0即可。
    <!-- end id29 -->
    <!-- npu="950,A3" id30 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, ep\_world\_size\)。取0表示无共享专家，不取0需满足ep\_world\_size\%shared\_expert\_rank\_num=0。
    <!-- end id30 -->

- **quant\_mode**（`int`）：**可选参数**，表示量化模式。支持取值：0表示非量化（默认），2表示动态量化。当`quant_mode`为2，`dynamic_scales`不为None；当`quant_mode`为0，`dynamic_scales`为None。
- **global\_bs**（`int`）：**可选参数**，表示EP域全局的batch size。

    <!-- npu="910b" id31 -->
    - <term>Atlas A2系列产品</term>：当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size或者256\*ep\_world\_size，其中max\_bs表示单rank BS最大值，建议按max\_bs\*ep\_world\_size传入；若固定按256\*ep\_world\_size传入，在后续版本BS大于256的场景下会无法支持；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。
    <!-- end id31 -->
    <!-- npu="950,A3" id32 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size，其中max\_bs表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。
    <!-- end id32 -->

- **expert\_token\_nums\_type**（`int`）：**可选参数**，表示输出`expert_token_nums`的值类型，取值范围\[0, 1\]，0表示每个专家收到token数量的前缀和，1表示每个专家收到的token数量（默认）。

## 返回值说明

- **expand\_x**（`Tensor`）：表示本卡收到的token数据，要求为2维张量，shape为\(max\(tp\_world\_size, 1\) \* A, H\)，A表示在EP通信域可能收到的最大token数。数据格式为$ND$，支持非连续的Tensor。量化时类型为`torch.int8`，非量化时与`x`数据类型保持一致。

    <!-- npu="950,A3,910b" id33 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`、`torch.int8`。
    <!-- end id33 -->

- **dynamic\_scales**（`Tensor`）：表示计算得到的动态量化参数。当`quant_mode`非0时才有该输出，要求为1维张量，shape为\(A, \)。数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
- **expand\_idx**（`Tensor`）：表示给同一专家发送的token个数，要求是1维张量，shape为\(BS \* K, \)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_combine](torch_npu-npu_moe_distribute_combine.md)的`expand_idx`输入。
- **expert\_token\_nums**（`Tensor`）：表示本卡每个专家实际收到的token数量，要求为1维张量，shape为\(local\_expert\_num, \)。数据类型支持`torch.int64`，数据格式为$ND$，支持非连续的Tensor。
- **ep\_recv\_counts**（`Tensor`）：表示EP通信域各卡收到的token数（token数以前缀和的形式表示），要求为1维张量。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_combine](torch_npu-npu_moe_distribute_combine.md)的`ep_send_counts`输入。

    <!-- npu="910b" id34 -->
    - <term>Atlas A2系列产品</term>：要求shape为\(moe\_expert\_num+2\*global\_bs\*K\*server\_num, \)，前`moe_expert_num`个数表示在EP通信域内，该卡上每个专家收到来自其他各卡的token数（以前缀和的形式表示），2\*global\_bs\*K\*server\_num用于存储机间和机内通信前，combine可提前做reduce操作的token个数和通信区偏移量，`global_bs`传入0时此处按照bs\*ep\_world\_size计算。
    <!-- end id34 -->
    <!-- npu="950,A3" id35 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：要求shape为\(ep\_world\_size\*max\(tp\_world\_size, 1\)\*local\_expert\_num, \)。
    <!-- end id35 -->

- **tp\_recv\_counts**（`Tensor`）：表示TP通信域各卡收到的token数量。对应[torch\_npu.npu\_moe\_distribute\_combine](torch_npu-npu_moe_distribute_combine.md)的`tp_send_counts`输入。

    <!-- npu="950,910b" id36 -->
    - <term>Atlas A2系列产品</term>、<term>Ascend 950DT系列产品</term>：不支持TP通信域，暂无该输出。
    <!-- end id36 -->
    <!-- npu="A3" id37 -->
    - <term>Atlas A3系列产品</term>：支持TP通信域，要求是一个1维张量，shape为\(tp\_world\_size, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id37 -->

- **expand\_scales**（`Tensor`）：表示`expert_scales`与`x`一起进行alltoallv之后的输出。

    <!-- npu="910b" id38 -->
    - <term>Atlas A2系列产品</term>：要求是1维张量，shape为\(A, \)，数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id38 -->
    <!-- npu="950,A3" id39 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：暂不支持该输出，返回None。
    <!-- end id39 -->

## 约束说明

- 该接口支持推理场景下使用。
<!-- npu="950" id40 -->
- 通信方式约束：
    - <term>Ascend 950DT系列产品</term>：仅支持UB Memory通信。
<!-- end id40 -->

- 该接口支持单算子模式和静态图模式，并且`npu_moe_distribute_dispatch`和`npu_moe_distribute_combine`必须配套使用。
- 在不同产品型号、不同通信算法或不同版本中，`npu_moe_distribute_dispatch`的Tensor输出`expand_idx`、`ep_recv_counts`、`tp_recv_counts`、`expand_scales`中的元素值可能不同，使用时直接将上述Tensor传给`npu_moe_distribute_combine`对应参数即可，模型其他业务逻辑不应对其存在依赖。
- 调用接口过程中使用的`group_ep`、`ep_world_size`、`moe_expert_num`、`group_tp`、`tp_world_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`global_bs`参数取值所有卡需保持一致，`group_ep`、`ep_world_size`、`group_tp`、`tp_world_size`、`expert_shard_type`、`global_bs`网络中不同层也需保持一致，且和[torch\_npu.npu\_moe\_distribute\_combine](torch_npu-npu_moe_distribute_combine.md)对应参数也保持一致。

<!-- npu="A3" id41 -->
- <term>Atlas A3系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
<!-- end id41 -->
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，取值范围如下：
        - 对于共享专家，当`global_bs`为0时，要满足A=BS\*shared\_expert\_num/shared\_expert\_rank\_num；当`global_bs`非0时，要满足A=global\_bs\*shared\_expert\_num/shared\_expert\_rank\_num。
        - 对于MoE专家，当`global_bs`为0时，要满足A\>=BS\*ep\_world\_size\*min\(local\_expert\_num, K\)；当`global_bs`非0时，要满足A\>=global\_bs\*min\(local\_expert\_num, K\)。

    - H：表示hidden size隐藏层大小。

        <!-- npu="910b" id42 -->
        - <term>Atlas A2系列产品</term>：取值范围\(0, 7168\]，且保证是32的整数倍。
        <!-- end id42 -->
        <!-- npu="A3" id43 -->
        - <term>Atlas A3系列产品</term>：取值为\[1024, 7168\]。
        <!-- end id43 -->
        <!-- npu="950" id44 -->
        - <term>Ascend 950DT系列产品</term>：取值为7168。
        <!-- end id44 -->

    - BS：表示待发送的token数量。

        <!-- npu="910b" id45 -->
        - <term>Atlas A2系列产品</term>：取值范围为0<BS≤256。
        <!-- end id45 -->
        <!-- npu="950,A3" id46 -->
        - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围为0<BS≤512。
        <!-- end id46 -->

    - K：表示选取topK个专家，需满足0<K≤moe\_expert\_num。

        <!-- npu="910b" id47 -->
        - <term>Atlas A2系列产品</term>：保证取值范围为0<K≤16。
        <!-- end id47 -->
        <!-- npu="A3" id48 -->
        - <term>Atlas A3系列产品</term>：保证取值范围为0<K≤8。
        <!-- end id48 -->
        <!-- npu="950" id49 -->
        - <term>Ascend 950DT系列产品</term>：取值范围为0<K≤8且小于moe\_expert\_num。
        <!-- end id49 -->

    - server\_num：表示服务器的节点数，单机场景可取1，多机场景取值只支持2、4、8。

        <!-- npu="910b" id50 -->
        - <term>Atlas A2系列产品</term>：仅该场景的shape使用了该变量。
        <!-- end id50 -->

    - local\_expert\_num：表示本卡专家数量。
        - 对于共享专家卡，local\_expert\_num=1。
        - 对于MoE专家卡，local\_expert\_num=moe\_expert\_num/\(ep\_world\_size-shared\_expert\_rank\_num\)，当local\_expert\_num\>1时，不支持TP域通信。

- HCCL通信域缓存区大小：

    调用本接口前需检查HCCL\_BUFFSIZE环境变量取值是否合理。

    > [!NOTE]
    > CANN环境变量HCCL\_BUFFSIZE：表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。

    <!-- npu="910b" id51 -->
    - <term>Atlas A2系列产品</term>：要求\>=2\*\(BS\*ep\_world\_size\*min\(local\_expert\_num, K\)\*H\*sizeof\(`torch.uint16`\)+2MB\)。
    <!-- end id51 -->
    <!-- npu="A3" id52 -->
    - <term>Atlas A3系列产品</term>：要求\>=2且满足\>=2\*\(local\_expert\_num\*max\_bs\*ep\_world\_size\*Align512\(Align32\(2\*H\)+64\)+\(K+shared\_expert\_num\)\*max\_bs\*Align512\(2\*H\)\)，local\_expert\_num需使用MoE专家卡的本卡专家数。
    <!-- end id52 -->
    <!-- npu="950" id53 -->
    - <term>Ascend 950DT系列产品</term>：要求\>=2且满足1024^2\*\(HCCL\_BUFFSIZE-2\)/2\>=BS\*2\*\(H+128\)\*\(ep\_world\_size\*local\_expert\_num+K+1\)，local\_expert\_num表示需使用MoE专家卡的本卡专家数。
    <!-- end id53 -->

<!-- npu="910b" id54 -->
- <term>Atlas A2系列产品</term>：配置环境变量HCCL\_INTRA\_PCIE\_ENABLE=1和HCCL\_INTRA\_ROCE\_ENABLE=0可以减少跨机通信数据量，提升算子性能。此时要求HCCL\_BUFFSIZE\>=moe\_expert\_num\*BS\*\(H\*sizeof\(dtypeX\)+4\*\(\(K+7\)/8\*8\)\*sizeof\(`torch.uint32`\)\)+4MB+100MB。并且，对于入参`moe_expert_num`，只要求moe\_expert\_num\%\(ep\_world\_size - shared\_expert\_rank\_num\)\=0，不要求moe\_expert\_num/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。
<!-- end id54 -->
- 通信域使用约束：
    - 一个模型中的`npu_moe_distribute_dispatch`和`npu_moe_distribute_combine`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的`npu_moe_distribute_dispatch`和`npu_moe_distribute_combine`算子仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

    <!-- npu="A3" id55 -->
    - <term>Atlas A3系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。
    <!-- end id55 -->

<!-- npu="910b" id56 -->
- 组网约束：
    - <term>Atlas A2系列产品</term>：多机场景仅支持交换机组网，不支持双机直连组网。
<!-- end id56 -->

- 版本配套约束：

    静态图模式下，从TorchNPU 7.1.0版本开始，框架会对静态图中最后一个节点输出结果做Meta推导与inferShape推导的结果强校验。当图中只有一个Dispatch算子，若CANN版本落后于TorchNPU版本，会出现Shape不匹配报错，建议用户升级CANN版本，详细的版本配套关系参见[《TorchNPU版本说明》](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/release_notes.md)中“相关产品版本配套说明”。

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
