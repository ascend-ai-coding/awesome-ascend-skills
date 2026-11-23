# torch_npu.npu_moe_distribute_dispatch_v2

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

- **API功能**：需与[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)或[torch\_npu.npu\_moe\_distribute\_combine\_add\_rms\_norm](torch_npu-npu_moe_distribute_combine_add_rms_norm.md)配套使用，完成MoE的并行部署下的token dispatch\_v2与combine\_v2。
    - 支持动态量化场景，对token数据先进行量化（可选），再进行EP（Expert Parallelism）域的alltoallv通信，再进行TP（Tensor Parallelism）域的allgatherv通信（可选）。
    - 支持特殊专家场景（零专家、拷贝专家、常量专家）。

- 相较于npu\_moe\_distribute\_dispatch接口，该接口变更如下：
    - npu\_moe\_distribute\_dispatch中shape为\(BS \* K, \)的返回值`expand_idx`替换为shape为\(A \* 128, \)的`assist_info_for_combine`，以包含更详细的token信息辅助torch\_npu.npu\_moe\_distribute\_combine\_v2高效地进行全卡同步。
    - 新增输入参数`comm_alg`，可用于代替HCCL\_INTRA\_PCIE\_ENABLE和HCCL\_INTRA\_ROCE\_ENABLE环境变量。

- **计算公式**：
    - 量化场景：
        - 如果`quant_mode`为0（非量化）：

            $$
            quant\_out = x
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

        - 如果`quant_mode`为1（静态量化）：

            $$
            quant\_out = Cast(CastToFp32(x) \times scales, dstType)
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
            dynamic\_scales\_value = dst\_type\_max / Max(Abs(x\_fp32))
            $$

            $$
            quant\_out = Cast(x\_fp32 \times dynamic\_scales\_value, dstType)
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

        - 如果`quant_mode`为3（pergroup量化）：

            $$
            x\_fp32 = CastToFp32(x) \times scales
            $$

            $$
            dynamic\_scales\_value = dst\_type\_max / Max(Abs(x\_fp32))
            $$

            $$
            quant\_out = Cast(x\_fp32 \times dynamic\_scales\_value, dstType)
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

        - 如果`quant_mode`为4（mxfp量化）：

            $$
            shared\_exp = floor(log_2(max(x))) - emax
            $$

            $$
            dynamic\_scales\_value = 2^{shared\_exp}
            $$

            $$
            quant\_out = Cast(x / dynamic\_scales\_value, dstType)
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

        - 如果`quant_mode`为5（mxfp8量化clip方法）：

            $$
            max\_abs = max(abs(x))
            $$

            $$
            max\_abs\_clamp = max(max\_abs, 10^{-4})
            $$

            $$
            shared\_exp = ceil(log_2(max\_abs\_clamp / fp8\_max))
            $$

            $$
            dynamic\_scales\_value = 2^{shared\_exp}
            $$

            $$
            quant\_out = CastToFp8(x / dynamic\_scales\_value)
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

        其中参数`ori_x`、`const_expert_alpha_1`、`const_expert_alpha_2`、`const_expert_v`参见[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)文档。

## 函数原型

```python
torch_npu.npu_moe_distribute_dispatch_v2(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, scales=None, x_active_mask=None, expert_scales=None, elastic_info=None, performance_info=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0, y_dtype=None, x_dtype=None, scales_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：**必选参数**，表示计算使用的token数据，需根据`expert_ids`来发送给其他卡。要求为2维张量，shape为\(BS, H\)，表示有BS个token。数据格式为$ND$，支持非连续的Tensor。数据类型支持`torch.bfloat16`、`torch.float16`。

    <!-- npu="910b" id4 -->
    - <term>Atlas A2系列产品</term>：额外支持`torch.int32`，每个`torch.int32`元素作为8个`torch_npu.int4`数据的打包载体，算子内部按位透传。
    <!-- end id4 -->

- **expert\_ids**（`Tensor`）：**必选参数**，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为\(BS, K\)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)的`expert_ids`输入，张量里value取值范围为\[0, moe\_expert\_num\)，且同一行中的K个value不能重复。
- **group\_ep**（`str`）：**必选参数**，EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)。取值不能和`group_tp`相同。
- **ep\_world\_size**（`int`）：**必选参数**，EP通信域size。

    <!-- npu="910b" id5 -->
    - <term>Atlas A2系列产品</term>：取值范围如下所示。
        - `comm_alg`设置为"fullmesh"时，`ep_world_size`取值范围为2、3、4、5、6、7、8、16、32、64、128、256。
        - `comm_alg`设置为"hierarchy"时，`ep_world_size`取值范围为16、32、64。
    <!-- end id5 -->
    <!-- npu="A3" id6 -->
    - <term>Atlas A3系列产品</term>：取值支持\[2, 768\]。`comm_alg`设置为"hierarchy"时，取值范围为\[16, 256\]，且为16的整数倍。
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - <term>Ascend 950DT系列产品</term>：取值支持\[2, 1024\]。
    <!-- end id7 -->

- **ep\_rank\_id**（`int`）：**必选参数**，EP通信域本卡ID，取值范围\[0, ep\_world\_size\)，同一个EP通信域中各卡的`ep_rank_id`不重复。
- **moe\_expert\_num**（`int`）：**必选参数**，MoE专家数量，取值范围\[1, 1024\]，并且满足条件moe\_expert\_num\%\(ep\_world\_size - shared\_expert\_rank\_num\)=0。

    <!-- npu="A3" id8 -->
    - <term>Atlas A3系列产品</term>：`comm_alg`设置为"hierarchy"时，取值范围为\[1, 512\]。
    <!-- end id8 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **scales**（`Tensor`）：**可选参数**，表示每个专家的权重。更多关于shape在不同产品或场景下的要求请参见[quant\_mode相关约束](#quant_mode相关约束)。
    - 当`x`的数据类型为`torch.int32`时必须传入，表示每个输入token对应的缩放参数。要求为1维张量，shape为\(BS, \)，数据类型支持`torch.float32`，数据格式为$ND$，不支持非连续的Tensor。该参数随对应token一起通信。

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：非量化场景不传，动态量化场景可传可不传。若传值要求为2维张量，如果有共享专家，shape为\(shared\_expert\_num+moe\_expert\_num, H\)，如果没有共享专家，shape为\(moe\_expert\_num, H\)。数据类型支持`torch.float`，数据格式为$ND$，不支持非连续的Tensor。
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950DT系列产品</term>：数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`。
        - 当`quant_mode`取值为0时，可传None或有效数据，传入有效数据时要求2维Tensor，且第一维为bs，第二维要求小于H。
        - 当`quant_mode`取值为1时，必须传入有效数据，其shape为\(1, \)、\(H, \)、\(shared\_expert\_num+moe\_expert\_num, H\)、\(moe\_expert\_num, H\)。
        - 当`quant_mode`取值为2或3时，可传入None或者有效数据，传入有效数据时，其shape为\(shared\_expert\_num+moe\_expert\_num, H\)、\(moe\_expert\_num, H\)。
        - 当`quant_mode`取值为4、5时，传默认值即可。
    <!-- end id10 -->

- **x\_active\_mask**（`Tensor`）：**可选参数**，表示token是否参与通信。

    <!-- npu="910b" id11 -->
    - <term>Atlas A2系列产品</term>：
        - `comm_alg`设置为"fullmesh"时，要求为一个1维张量，shape为\(BS, \)。数据类型支持`torch.bool`，数据格式为$ND$，支持非连续的Tensor。参数为true表示对应的token参与通信，true必须排到false之前，例：\{true, false, true\}为非法输入。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。
        - `comm_alg`设置为"hierarchy"时，当前版本不支持，使用默认值即可。
    <!-- end id11 -->
    <!-- npu="950,A3" id12 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：要求是一个1维或2维张量。当输入为1维时，shape为\(BS, \)；当输入为2维时，shape为\(BS, K\)。数据类型支持`torch.bool`，数据格式为$ND$，支持非连续的Tensor。当输入为1维时，参数为true表示对应的token参与通信，true必须排到false之前，例：\{true, false, true\}为非法输入；当输入为2维时，参数为true表示当前token对应的`expert_ids`参与通信，若当前token对应的K个`torch.bool`值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。
    <!-- end id12 -->

- **expert\_scales**（`Tensor`）：**可选参数**，表示每个token的topK个专家权重。

    <!-- npu="910b" id13 -->
    - <term>Atlas A2系列产品</term>：要求为2维张量，shape为\(BS, K\)，数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id13 -->
    <!-- npu="A3" id14 -->
    - <term>Atlas A3系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950DT系列产品</term>：支持传入None或2维张量，传入有效数据时shape为\(BS, K\)。数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id15 -->

- **elastic\_info**（`Tensor`）：**可选参数**，预留参数，当前版本不支持，传默认值None即可。
- **performance\_info**（`Tensor`）：**可选参数**，表示本卡等待各卡数据的通信时间，单位为us（微秒）。单次算子调用各卡通信耗时会累加到该Tensor上，算子内部不进行自动清零，因此每次启用此Tensor开始记录耗时前需对Tensor清零。当传入None时表示不开启记录通信耗时功能；当传入有效数据时，要求是一个1维张量，shape为\(ep\_world\_size, \)，数据类型支持`torch.int64`，数据格式为$ND$，支持非连续的Tensor。
- **group\_tp**（`str`）：**可选参数**，TP通信域名称，数据并行的通信域。若有TP域通信需要传参，若无TP域通信，使用默认值""即可。

    <!-- npu="910b" id16 -->
    - <term>Atlas A2系列产品</term>：eager模式使用默认值即可，图模式传入与`group_ep`相同。
    <!-- end id16 -->
    <!-- npu="A3" id17 -->
    - <term>Atlas A3系列产品</term>：字符串长度范围为\[1, 128\)，不能和`group_ep`相同。
    <!-- end id17 -->
    <!-- npu="950" id18 -->
    - <term>Ascend 950DT系列产品</term>：不支持TP域通信，使用默认输入。
    <!-- end id18 -->

- **tp\_world\_size**（`int`）：**可选参数**，TP通信域size。有TP域通信才需要传参。

    <!-- npu="910b" id19 -->
    - <term>Atlas A2系列产品</term>：不支持TP域通信，使用默认值0即可。
    <!-- end id19 -->
    <!-- npu="A3" id20 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，取值范围\[0, 2\]，0和1表示无TP域通信，2表示有TP域通信。
    <!-- end id20 -->
    <!-- npu="950" id21 -->
    - <term>Ascend 950DT系列产品</term>：不支持TP域通信，使用默认值0即可。
    <!-- end id21 -->

- **tp\_rank\_id**（`int`）：**可选参数**，TP通信域本卡ID。有TP域通信才需要传参。

    <!-- npu="910b" id22 -->
    - <term>Atlas A2系列产品</term>：不支持TP域通信，使用默认值即可。
    <!-- end id22 -->
    <!-- npu="A3" id23 -->
    - <term>Atlas A3系列产品</term>：当有TP域通信时，取值范围\[0, 1\]，默认为0，同一个TP通信域中各卡的`tp_rank_id`不重复。无TP域通信时，传0即可。
    <!-- end id23 -->
    <!-- npu="950" id24 -->
    - <term>Ascend 950DT系列产品</term>：不支持TP域通信，使用默认值0即可。
    <!-- end id24 -->

- **expert\_shard\_type**（`int`）：**可选参数**，表示共享专家卡排布类型。

    <!-- npu="910b" id25 -->
    - <term>Atlas A2系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id25 -->
    <!-- npu="950,A3" id26 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：当前仅支持0，表示共享专家卡排在MoE专家卡前面。
    <!-- end id26 -->

- **shared\_expert\_num**（`int`）：**可选参数**，表示共享专家数量，一个共享专家可以复制部署到多个卡上。

    <!-- npu="910b" id27 -->
    - <term>Atlas A2系列产品</term>：暂不支持该参数，使用默认值即可。
    <!-- end id27 -->
    <!-- npu="950,A3" id28 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, 4\]，0表示无共享专家，默认值为1。
    <!-- end id28 -->

- **shared\_expert\_rank\_num**（`int`）：**可选参数**，表示共享专家卡数量。

    <!-- npu="910b" id29 -->
    - <term>Atlas A2系列产品</term>：不支持共享专家，使用默认值即可。
    <!-- end id29 -->
    <!-- npu="950,A3" id30 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, ep\_world\_size\)。取0表示无共享专家，不取0需满足shared\_expert\_rank\_num\%shared\_expert\_num=0。
    <!-- end id30 -->

- **quant\_mode**（`int`）：**可选参数**，表示量化模式。

    <!-- npu="A3,910b" id31 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：0表示非量化（默认），2表示动态量化。当`quant_mode`为2，`dynamic_scales`不为None；当`quant_mode`为0，`dynamic_scales`为None。当`x`的数据类型为`torch.int32`时仅支持取值0，此时`dynamic_scales`不为None。
    <!-- end id31 -->
    <!-- npu="950" id32 -->
    - <term>Ascend 950DT系列产品</term>：0表示非量化（默认），1表示静态量化，2表示动态pertoken量化，3表示pergroup量化，4表示mxfp量化，5表示mxfp量化clip实现。当`quant_mode`为2、3、4、5，`dynamic_scales`不为None；当`quant_mode`为0或1，`dynamic_scales`为None。
    <!-- end id32 -->

- **global\_bs**（`int`）：**可选参数**，表示EP域全局的batch size。当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size，其中`max_bs`表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。
- **expert\_token\_nums\_type**（`int`）：**可选参数**，表示输出`expert_token_nums`的值类型，取值范围\[0, 1\]，0表示每个专家收到token数量的前缀和，1表示每个专家收到的token数量（默认）。
- **comm\_alg**（`str`）：**可选参数**，表示通信亲和内存布局算法。

    <!-- npu="910b" id33 -->
    - <term>Atlas A2系列产品</term>：当前版本支持""、"fullmesh"、"hierarchy"三种输入方式。推荐配置"hierarchy"并搭配25.0.RC1.1及以上版本驱动使用。
        - "": 配置HCCL\_INTRA\_PCIE\_ENABLE=1和HCCL\_INTRA\_ROCE\_ENABLE=0时，调用"hierarchy"算法，否则调用"fullmesh"算法。不推荐使用该方式。
        - "fullmesh"：Token数据直接通过RDMA方式发往topk个目标专家所在的卡。
        - "hierarchy"：Token数据经过跨机、机内两次发送，仅不同server同号卡之间使用RDMA通信，server内使用HCCS通信。
    <!-- end id33 -->
    <!-- npu="950,A3" id34 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：当前版本仅支持""、"fullmesh\_v1"、"fullmesh\_v2"三种输入方式。
        - ""：默认值，不开启fullmesh\_v2模板。
        - "fullmesh\_v1"：不开启fullmesh\_v2模板。
        - "fullmesh\_v2"：开启fullmesh\_v2模板，其中`comm_alg`仅在`tp_world_size`取值为1时生效，且不支持在各卡BS不一致、输入`x_active_mask`和特殊专家场景下开启。
    <!-- end id34 -->

- **zero\_expert\_num**（`int`）：**可选参数**，表示零专家的数量。

    <!-- npu="910b" id35 -->
    - <term>Atlas A2系列产品</term>：当前版本不支持，传0即可。
    <!-- end id35 -->
    <!-- npu="950,A3" id36 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, MAX\_INT32\)，MAX\_INT32 = 2^31 - 1，合法的零专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num\)。
    <!-- end id36 -->

- **copy\_expert\_num**（`int`）：**可选参数**，表示拷贝专家的数量。

    <!-- npu="910b" id37 -->
    - <term>Atlas A2系列产品</term>：当前版本不支持，传0即可。
    <!-- end id37 -->
    <!-- npu="950,A3" id38 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, MAX\_INT32\)，MAX\_INT32 = 2^31 - 1，合法的拷贝专家的ID值是\[moe\_expert\_num+zero\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num\)。
    <!-- end id38 -->

- **const\_expert\_num**（`int`）：**可选参数**，表示常量专家的数量。

    <!-- npu="910b" id39 -->
    - <term>Atlas A2系列产品</term>：当前版本不支持，传0即可。
    <!-- end id39 -->
    <!-- npu="950,A3" id40 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：取值范围\[0, MAX\_INT32\)，MAX\_INT32 = 2^31 - 1，合法的常量专家的ID值是\[moe\_expert\_num+zero\_expert\_num+copy\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num+const\_expert\_num\)。
    <!-- end id40 -->

- **y\_dtype**（`int`）：**可选参数**，表示输出的数据类型。

    <!-- npu="A3,910b" id41 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数。
    <!-- end id41 -->
    <!-- npu="950" id42 -->
    - <term>Ascend 950DT系列产品</term>：当`quant_mode`不为0时应当指定，不指定默认为`torch.int8`类型。
    <!-- end id42 -->

- **x\_dtype**（`int`）：**可选参数**，表示输入的Tensor `x`的数据类型，当其为PyTorch原生不支持的数据类型需通过该参数指定。
- **scales\_dtype**（`int`）：**可选参数**，表示输入的Tensor `scales`的数据类型，当其为PyTorch原生不支持的数据类型需通过该参数指定。

## 返回值说明

- **expand\_x**（`Tensor`）：表示本卡收到的token数据，要求为2维张量，shape为\(max\(tp\_world\_size, 1\) \* A, H\)，A表示在EP通信域可能收到的最大token数。数据格式为$ND$，支持非连续的Tensor。量化时类型为`torch.int8`，非量化时与`x`数据类型保持一致。

    <!-- npu="A3,910b" id43 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`、`torch.int8`。
    <!-- end id43 -->
    <!-- npu="910b" id44 -->
    - <term>Atlas A2系列产品</term>：非量化场景下还支持`torch.int32`。
    <!-- end id44 -->
    <!-- npu="950" id45 -->
    - <term>Ascend 950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`。
    <!-- end id45 -->

- **dynamic\_scales**（`Tensor`）：表示计算得到的动态量化参数，或随数据类型为`torch.int32`的token通信的缩放参数。当`quant_mode`不为0或1时才有该输出；当`x`的数据类型为`torch.int32`且`quant_mode`为0时有该输出，其中前实际接收token数量个元素有效。数据格式为$ND$，支持非连续的Tensor。

    <!-- npu="A3,910b" id46 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：要求为1维张量，shape为\(A, \)，数据类型支持`torch.float`。
    <!-- end id46 -->
    <!-- npu="950" id47 -->
    - <term>Ascend 950DT系列产品</term>：数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`，不同场景下的shape要求详见[quant\_mode相关约束](#quant_mode相关约束)部分：
        - `quant_mode`取值为0时，如果输入`x`类型为`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`/`torch_npu.hifloat8`且`expand_x`类型和输入的`x`类型保持一致，要求2维Tensor且第一维为A，否则不输出有效值。
        - `quant_mode`取值为2时，其shape为\(A, \)；`quant_mode`取值为3时，其shape为\(A, ceil\(H/128\)\)。
        - `quant_mode`取值为4、5时，其shape为\(A, ceil\(H/32\)\)且ceil\(H/32\) % 2 == 0。
    <!-- end id47 -->

- **assist\_info\_for\_combine**（`Tensor`）：表示给同一专家发送的token个数，要求是1维张量，shape为\(A \* 128, \)。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)的`assist_info_for_combine`输入。
- **expert\_token\_nums**（`Tensor`）：表示本卡每个专家实际收到的token数量，要求为1维张量，shape为\(local\_expert\_num, \)。数据类型支持`torch.int64`，数据格式为$ND$，支持非连续的Tensor。
- **ep\_recv\_counts**（`Tensor`）：表示EP通信域各卡收到的token数（token数以前缀和的形式表示），要求为1维张量。数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)的`ep_send_counts`输入。

    <!-- npu="910b" id48 -->
    - <term>Atlas A2系列产品</term>：要求shape为\(moe\_expert\_num+2\*global\_bs\*K\*server\_num, \)，前`moe_expert_num`个数表示在EP通信域内，该卡上每个专家收到来自其他各卡的token数（以前缀和的形式表示），2\*global\_bs\*K\*server\_num用于存储机间和机内通信前，combine可提前做reduce操作的token个数和通信区偏移量，`global_bs`传入0时此处按照bs\*ep\_world\_size计算。
    <!-- end id48 -->
    <!-- npu="950,A3" id49 -->
    - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：要求shape为\(ep\_world\_size\*max\(tp\_world\_size, 1\)\*local\_expert\_num, \)。
    <!-- end id49 -->

- **tp\_recv\_counts**（`Tensor`）：表示TP通信域各卡收到的token数量。对应[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)的`tp_send_counts`输入。

    <!-- npu="950,910b" id50 -->
    - <term>Atlas A2系列产品</term>、<term>Ascend 950DT系列产品</term>：不支持TP通信域，暂无该输出。
    <!-- end id50 -->
    <!-- npu="A3" id51 -->
    - <term>Atlas A3系列产品</term>：支持TP通信域，要求是1维张量，shape为\(tp\_world\_size, \)，数据类型支持`torch.int32`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id51 -->

- **expand\_scales**（`Tensor`）：表示`expert_scales`与`x`一起进行alltoallv之后的输出。

    <!-- npu="910b" id52 -->
    - <term>Atlas A2系列产品</term>：要求是1维张量，shape为\(A, \)，数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id52 -->
    <!-- npu="A3" id53 -->
    - <term>Atlas A3系列产品</term>：暂不支持该输出，返回None。
    <!-- end id53 -->
    <!-- npu="950" id54 -->
    - <term>Ascend 950DT系列产品</term>：仅`expert_scales`输入有效数据时有输出，要求是1维张量，shape为\(A, \)，数据类型支持`torch.float`，数据格式为$ND$，支持非连续的Tensor。
    <!-- end id54 -->

## 约束说明

- 该接口支持推理场景下使用。
<!-- npu="950" id55 -->
- 通信方式约束：
    - <term>Ascend 950DT系列产品</term>：仅支持UB Memory通信。
<!-- end id55 -->

- 该接口支持单算子模式和静态图模式调用，且必须与`npu_moe_distribute_combine_v2`或`npu_moe_distribute_combine_add_rms_norm`配套使用。
- 在不同产品型号、不同通信算法或不同版本中，`npu_moe_distribute_dispatch_v2`的Tensor输出`assist_info_for_combine`、`ep_recv_counts`、`tp_recv_counts`、`expand_scales`中的元素值可能不同，使用时直接将上述Tensor传给`npu_moe_distribute_combine_v2`对应参数即可，模型其他业务逻辑不应对其存在依赖。
- 调用接口过程中使用的`group_ep`、`ep_world_size`、`moe_expert_num`、`group_tp`、`tp_world_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`global_bs`参数取值所有卡需保持一致，`group_ep`、`ep_world_size`、`group_tp`、`tp_world_size`、`expert_shard_type`、`global_bs`网络中不同层也需保持一致，且和[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)对应参数也保持一致。

<!-- npu="A3" id56 -->
- <term>Atlas A3系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
<!-- end id56 -->
- moe\_expert\_num + zero\_expert\_num + copy\_expert\_num + const\_expert\_num < MAX\_INT32。
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足A=BS\*shared\_expert\_num/shared\_expert\_rank\_num。
        - 对于MoE专家，当`global_bs`为0时，要满足A\>=BS\*ep\_world\_size\*min\(local\_expert\_num, K\)；当`global_bs`不为0时，要满足A\>=global\_bs\*min\(local\_expert\_num, K\)。

    - H：表示hidden size隐藏层大小。

        <!-- npu="910b" id57 -->
        - <term>Atlas A2系列产品</term>：
            - 当`x`的数据类型为`torch.int32`时，仅支持`comm_alg`设置为"fullmesh"，`H`的取值范围\(0, 5120\]，且保证是32的整数倍。
            - `comm_alg`设置为"fullmesh"时（`x`的数据类型不为`torch.int32`），`H`的取值范围\(0, 7168\]，且保证是32的整数倍。
            - `comm_alg`设置为"hierarchy"且驱动版本不低于25.0.RC1.1时，`H`的取值范围\(0, 10 \* 1024\]，且保证是32的整数倍。
        <!-- end id57 -->
        <!-- npu="A3" id58 -->
        - <term>Atlas A3系列产品</term>：取值为\[1024, 8192\]。
        <!-- end id58 -->
        <!-- npu="950" id59 -->
        - <term>Ascend 950DT系列产品</term>：取值范围\[1024, 8192\]，且`expand_x`为`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`类型时要求为偶数。
        <!-- end id59 -->

    - BS：表示batch sequence size，即本卡最终输出的token数量。

        <!-- npu="910b" id60 -->
        - <term>Atlas A2系列产品</term>：取值范围为0<BS≤256。
        <!-- end id60 -->
        <!-- npu="A3" id61 -->
        - <term>Atlas A3系列产品</term>：取值范围为0<BS≤512。
        <!-- end id61 -->
        <!-- npu="950" id62 -->
        - <term>Ascend 950DT系列产品</term>：取值范围为0<BS≤512。
        <!-- end id62 -->

    - K：表示选取topK个专家，`comm_alg`取值""或"fullmesh\_v1"时取值范围为0<K≤16，`comm_alg`取值"fullmesh\_v2"时，0<K≤12。需同时满足0 < K ≤ moe\_expert\_num + zero\_expert\_num + copy\_expert\_num + const\_expert\_num。
    - server\_num：表示服务器的节点数，取值只支持2、4、8。

        <!-- npu="910b" id63 -->
        - <term>Atlas A2系列产品</term>：仅该场景的shape使用了该变量。
        <!-- end id63 -->

    - local\_expert\_num：表示本卡专家数量。
        - 对于共享专家卡，local\_expert\_num为1。
        - 对于MoE专家卡，local\_expert\_num=moe\_expert\_num/\(ep\_world\_size-shared\_expert\_rank\_num\)，当local\_expert\_num大于1时，不支持TP域通信。

        <!-- npu="950,A3" id64 -->
        - <term>Atlas A3系列产品</term>、<term>Ascend 950DT系列产品</term>：应满足0 < local\_expert\_num \* ep\_world\_size ≤ 2048。
        <!-- end id64 -->

- quant\_mode相关约束如下：
    <a id="quant_mode相关约束"></a>

    <!-- npu="A3,910b" id65 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>

        **表 1** quant\_mode约束说明

        | quant_mode | 量化模式 | expand_x参数 | scales参数 | dynamic_scales |
        | --- | --- | --- | --- | --- |
        | 0 | 非量化 | `torch.float16`/`torch.bfloat16` | None | 不输出有效值 |
        | 2 | 动态量化 | `torch.int8` | 支持None或非空。非空时shape为(sharedExpertNum+moeExpertNum, H)或者(moeExpertNum, H)，数据类型为`torch.float32`。 | shape为(A, )。数据类型为`torch.float32`。 |
    <!-- end id65 -->
    <!-- npu="950" id66 -->
    - <term>Ascend 950DT系列产品</term>

        **表 2** quant\_mode约束说明

        | quant_mode | 量化模式 | expand_x参数 | scales参数 | dynamic_scales |
        | --- | --- | --- | --- | --- |
        | 0 | 非量化 | `torch.float16`/`torch.bfloat16`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` | 如果expand_x类型为`torch.float16`/`torch.bfloat16`，则输入None；<br>如果expand_x类型为`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`，输入要求2维Tensor且第一维大小为bs，第二维大小小于H，数据类型为`torch_npu.float8_e8m0fnu`；<br>如果expand_x类型为`torch_npu.hifloat8`且和输入x一致，则输入要求2维Tensor且第一维大小为bs，数据类型为`torch.float32`；否则输入None。 | 如果expand_x类型为`torch.float16`/`torch.bfloat16`，则不输出有效值；<br>如果expand_x类型为`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`，则输出2维Tensor且第一维大小为A，第二维大小、数据类型和输入scales保持一致；<br>如果expand_x类型为`torch_npu.hifloat8`且和输入x一致，则输出2维Tensor且第一维大小为A，第二维大小、数据类型和输入scales保持一致，否则不输出有效值。 |
        | 1 | 静态量化 | `torch.int8` | 非空，数据类型为`torch.float32`。<br>输入scales代表量化系数则shape为(1, )；<br>输入scales代表每个专家共享的平滑权重则shape为(H, )；<br>输入scales代表融了专家共享的平滑权重的量化系数时，其shape在有共享专家卡情况下为(sharedExpertNum+moeExpertNum, H)，无共享专家卡情况下为(moeExpertNum, H)。 | 不输出有效值。 |
        | 1 | 静态量化 | `torch_npu.hifloat8` | 非空，shape为(1, )，数据类型为`torch.float32`。 | 不输出有效值。 |
        | 2 | pertoken动态量化 | `torch.int8`/`torch.float8_e4m3fn`/`torch.float8_e5m2` | 支持None或者非空。非空时，有共享专家卡情况下shape为(sharedExpertNum+moeExpertNum, H)，无共享专家卡情况下为(moeExpertNum, H)，数据类型为`torch.float32`。 | 输出有效值。shape为(A, )。数据类型为`torch.float32`。 |
        | 3 | pergroup动态量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | 支持None或者非空。非空时，有共享专家卡情况下shape为(sharedExpertNum+moeExpertNum, H)，无共享专家卡情况下为(moeExpertNum, H)，数据类型为`torch.float32`。 | 输出有效值。shape为(A, Ceil(H, 128))。数据类型为`torch.float32`。 |
        | 4 | mx量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` | None。 | 输出有效值。shape为(A, Ceil(H, 32))且ceil(H/32) % 2 == 0。数据类型为`torch_npu.float8_e8m0fnu`。 |
        | 5 | mxfp8量化clip实现 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None。 | 输出有效值。shape为(A, Ceil(H, 32))且ceil(H/32) % 2 == 0。数据类型为`torch_npu.float8_e8m0fnu`。 |
    <!-- end id66 -->

- HCCL通信域缓存区大小：

    调用本接口前需检查通信域缓存区大小取值是否合理，单位MB，不配置时默认为200MB。

    <!-- npu="910b" id67 -->
    - <term>Atlas A2系列产品</term>：该场景支持通过环境变量HCCL\_BUFFSIZE配置。
        - `comm_alg`配置为"": 仅在此配置下HCCL\_INTRA\_PCIE\_ENABLE和HCCL\_INTRA\_ROCE\_ENABLE生效，依照HCCL\_INTRA\_PCIE\_ENABLE和HCCL\_INTRA\_ROCE\_ENABLE配置选择"fullmesh"或"hierarchy"公式。
        - `comm_alg`配置为"fullmesh"且`x`的数据类型为`torch.int32`：设置大小要求\>=2\*\(BS\*ep\_world\_size\*min\(local\_expert\_num, K\)\*\(H\*sizeof\(`torch.int32`\)+32B\)+2MB\)。
        - `comm_alg`配置为"fullmesh"且`x`的数据类型不为`torch.int32`：设置大小要求\>=2\*\(BS\*ep\_world\_size\*min\(local\_expert\_num, K\)\*H\*sizeof\(`torch.uint16`\)+2MB\)。
        - `comm_alg`配置为"hierarchy"：设置大小要求=moe\_expert\_num\*BS\*\(H\*sizeof\(dtype\_x\)+4\*\(\(K+7\)/8\*8\)\*sizeof\(`torch.uint32`\)\)+4MB+100MB，不要求moe\_expert\_num/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。
    <!-- end id67 -->
    <!-- npu="A3" id68 -->
    - <term>Atlas A3系列产品</term>：该场景不仅支持通过环境变量HCCL\_BUFFSIZE配置，还支持通过hccl\_buffer\_size配置（参考[《PyTorch训练模型迁移调优》](https://hiascend.com/document/redirect/canncommercial-ptmigr)中“性能调优\>性能调优方法\>通信优化\>优化方法\>hccl\_buffer\_size”章节）。
        - ep通信域内，`comm_alg`配置为"fullmesh\_v1"或""：设置大小要求\>=2\*\(local\_expert\_num\*max\_bs\*ep\_world\_size\*Align512\(Align32\(2\*H\)+64\)+\(K+shared\_expert\_num\)\*max\_bs\*Align512\(2\*H\)\)。
        - ep通信域内，`comm_alg`配置为"fullmesh\_v2"：设置大小要求\>=2\*\(local\_expert\_num\*max\_bs\*ep\_world\_size\*480Align512\(Align32\(2\*H\)+64\)+\(K+shared\_expert\_num\)\*max\_bs\*Align512\(2\*H\)\)。
        - tp通信域内：设置大小要求\>=(A \* Align512(Align32(H \* 2) + 44) + A \* Align512(H \* 2)) \* 2。
        - 其中480Align512(x) = ((x+480-1)/480)\*512，Align512(x) = ((x+512-1)/512)\*512，Align32(x) = ((x+32-1)/32)\*32。
    <!-- end id68 -->
    <!-- npu="950" id69 -->
    - <term>Ascend 950DT系列产品</term>：取值范围如下所示。
        - ep通信域内，`comm_alg`配置为"fullmesh\_v1"或""：设置大小要求\>=2\*\(local\_expert\_num\*max\_bs\*ep\_world\_size\*Align512\(Align32\(2\*H\)+64\)+\(K+shared\_expert\_num\)\*max\_bs\*Align512\(2\*H\)\) + 1024 \* 1024 \* 10。
        - ep通信域内，`comm_alg`配置为"fullmesh\_v2"：设置大小要求\>=2\*\(local\_expert\_num\*max\_bs\*ep\_world\_size\*480Align512\(Align32\(2\*H\)+64\)+\(K+shared\_expert\_num\)\*max\_bs\*Align512\(2\*H\)\) + 1024 \* 1024 \* 10。
        - 其中480Align512(x) = ((x+480-1)/480)\*512，Align512(x) = ((x+512-1)/512)\*512，Align32(x) = ((x+32-1)/32)\*32。
    <!-- end id69 -->

<!-- npu="910b" id70 -->
- <term>Atlas A2系列产品</term>：HCCL\_INTRA\_PCIE\_ENABLE和HCCL\_INTRA\_ROCE\_ENABLE环境变量不再推荐使用，建议`comm_alg`配置"hierarchy"。
<!-- end id70 -->
- 通信域使用约束：
    - 一个模型中的`npu_moe_distribute_dispatch_v2`和`npu_moe_distribute_combine_v2`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的`npu_moe_distribute_dispatch_v2`和`npu_moe_distribute_combine_v2`算子仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

    <!-- npu="A3" id71 -->
    - <term>Atlas A3系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。
    <!-- end id71 -->

<!-- npu="910b" id72 -->
- 组网约束：
    - <term>Atlas A2系列产品</term>：多机场景仅支持交换机组网，不支持双机直连组网。
<!-- end id72 -->

- 版本配套约束：

    静态图模式下，从TorchNPU 8.0.0版本开始，TorchNPU框架会对静态图中最后一个节点输出结果做Meta推导与inferShape推导的结果强校验。当图中只有一个Dispatch\_v2算子，若CANN版本落后于TorchNPU版本，会出现Shape不匹配报错，建议用户升级CANN版本，详细的版本配套关系参见[《TorchNPU版本说明》](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/release_notes.md)中“相关产品版本配套说明”。

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
