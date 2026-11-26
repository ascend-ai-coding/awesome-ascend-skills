# torch_npu.npu_alltoallv_quant_gmm

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：实现路由专家AlltoAllv和GroupedMatMul的融合，先通信后计算。同时与共享专家MatMul计算并行融合。

- 计算公式：

    假设通信域中的总卡数为`epWorldSize`，每张卡上通信后路由专家个数为`e`，每张卡GroupedMatmul只负责本卡专家的计算。对于每张卡的计算公式如下：

    - 本卡共享专家Matmul计算

        $$
        mm\_y = mm\_x\_scale \times mm\_weight\_scale \times (mm\_x \mathbin{@} mm\_weight)
        $$

        - mm\_y指共享专家MatMul的输出。
        - mm\_x指共享专家Matmul的左矩阵输入。
        - mm\_x\_scale指共享专家左矩阵mm\_x的量化系数。
        - mm\_weight指共享专家Matmul的右矩阵输入。
        - mm\_weight\_scale指共享专家右矩阵mm\_weight的量化系数。

    - Alltoallv通信和permute

        $$
        \begin{aligned}
        &permute\_out = Permute(AlltoAllv(gmm\_x))
        \end{aligned}
        $$

        - permute\_out指Permute之后的输出。
        - gmm\_x指本卡通信前路由专家原始左矩阵的输入。

    - 本卡路由专家按专家维度GroupedMatmul计算

        $$
        gmm\_y = gmm\_x\_scale \times gmm\_weight\_scale \times (permute\_out \mathbin{@} gmm\_weight)
        $$

        - gmm\_weight指本卡路由专家GroupedMatmul的右矩阵输入。
        - gmm\_x\_scale指路由专家左矩阵gmm\_x的量化系数。
        - gmm\_weight\_scale指路由专家右矩阵gmm\_weight的量化系数。

## 函数原型

```python
torch_npu.npu_alltoallv_quant_gmm(gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, hcom, ep_world_size, send_counts, recv_counts, gmm_y_dtype, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, mm_x_scale=None, mm_weight_scale=None, gmm_x_quant_mode=None, gmm_weight_quant_mode=None, mm_x_quant_mode=None, mm_weight_quant_mode=None, permute_out_flag=False, group_size=None, gmm_x_dtype=None, gmm_weight_dtype=None, gmm_x_scale_dtype=None, gmm_weight_scale_dtype=None, mm_x_dtype=None, mm_weight_dtype=None, mm_x_scale_dtype=None, mm_weight_scale_dtype=None, mm_y_dtype=None, comm_mode=None) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **gmm\_x** (`Tensor`)：必选参数，表示本卡通信前路由专家原始左矩阵的输入。数据类型支持`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.float4_e2m1fn_x2`。支持2维，shape为$(BSK，H1)$，数据格式支持$ND$，其中数据类型为`torch_npu.float4_e2m1fn_x2`时内轴$H1$需要为偶数，$H1$不能为2，以保证8bits可以转换为2个`torch_npu.float4_e2m1fn_x2`。
- **gmm\_weight** (`Tensor`)：必选参数，表示本卡路由专家GroupedMatmul的右矩阵输入。数据类型支持`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.float4_e2m1fn_x2`。支持3维，shape为$(e，H1，N1)$，数据格式支持$ND$。mx量化场景下，当`gmm_x`为`torch_npu.float4_e2m1fn_x2`系列、`gmm_weight`为`torch_npu.float4_e2m1fn_x2`系列时，仅支持推理场景，此时输入`gmm_x`的$H1$需要为偶数，$H1$不能为2，且当`gmm_weight`不转置时内轴$N1$需为偶数，转置时内轴$H1$需要为偶数，$H1$不能为2，以保证8bits可以转换为2个`torch_npu.float4_e2m1fn_x2`。
- **gmm\_x\_scale** (`Tensor`)：必选参数，表示路由专家左矩阵`gmm_x`的量化系数。数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`。pertensor量化场景下支持1维，shape为$(1，)$。mx量化场景下支持3维，shape为$(BSK, ceil(H1/64), 2)$。数据格式支持$ND$。
- **gmm\_weight\_scale** (`Tensor`)：必选参数，表示路由专家右矩阵`gmm_weight`的量化系数。数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`。pertensor量化场景下支持1维，shape为$(1，)$。mx量化场景下支持4维，shape为$(e, ceil(H1/64), N1, 2)$。数据格式支持$ND$。
- **hcom** (`str`)：必选参数，Host侧标识列组的字符串，即通信域名称，通过get\_hccl\_comm\_name接口获取。
- **ep\_world\_size** (`int`)：必选参数，通信域内的rank总数。支持范围为2、4、8、16、32、64、128、256, CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。
- **send\_counts** (`List[int]`)：必选参数，长度为`e * ep_world_size`的整数列表，表示本卡发送给每个目标卡的token数。假设目标卡号为$i$（$0 \le i < ep\_world\_size$），发送专家号为$j$（$0 \le j < e$），`send_counts[i][j]`表示本卡发送给第$i$张卡第$j$个专家的token数。约束：长度必须等于`e * ep_world_size`，且元素均为非负整数。
- **recv\_counts** (`List[int]`)：必选参数，长度为`e * ep_world_size`的整数列表，表示本卡从每个目标卡接收的token数。假设目标卡号为$i$（$0 \le i < ep\_world\_size$），接收专家号为$j$（$0 \le j < e$），`recv_counts[i][j]`表示本卡接收第$i$张卡第$j$个专家的token数。约束：长度必须等于`e * ep_world_size`，且元素均为非负整数。
- **gmm\_y\_dtype** (`int`)：必选参数，表示路由专家GroupedMatmul计算输出张量`gmm_y`的数据类型。数据类型支持`torch.float16`、`torch.bfloat16`。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **send\_counts\_tensor** (`Tensor`)：可选参数，当前仅支持输入None。
- **recv\_counts\_tensor** (`Tensor`)：可选参数，当前仅支持输入None。
- **mm\_x** (`Tensor`)：可选参数，默认None。表示共享专家Matmul的左矩阵输入，仅在启用共享专家时输入。数据类型支持`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.float4_e2m1fn_x2`，且和`gmm_x`类型一致。支持2维，shape为$(BS，H2)$。数据格式支持$ND$，其中数据类型为`torch_npu.float4_e2m1fn_x2`时内轴$H2$需为偶数，$H2$不能为2，以保证8bits可以转换为2个`torch_npu.float4_e2m1fn_x2`。
- **mm\_weight** (`Tensor`)：可选参数，默认None。表示共享专家Matmul的右矩阵输入，仅在启用共享专家时输入。数据类型支持`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.float4_e2m1fn_x2`，且和`gmm_weight`类型一致。支持2维，shape为$(H2，N2)$。数据格式支持$ND$。mx量化场景下，当`mm_x`为`torch_npu.float4_e2m1fn_x2`系列、`mm_weight`为`torch_npu.float4_e2m1fn_x2`系列时，仅支持推理场景，此时输入`mm_x`的$H2$需要为偶数，$H2$不能为2，且当`mm_weight`不转置时内轴$N2$需为偶数，转置时内轴$H2$需要为偶数，$H2$不能为2，以保证8bits可以转换为2个`torch_npu.float4_e2m1fn_x2`。
- **mm\_x\_scale** (`Tensor`)：可选参数，默认None。表示共享专家左矩阵`mm_x`的量化系数。数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`。pertensor量化场景下支持1维，shape为$(1，)$。mx量化场景下支持3维，shape为$(BS, ceil(H2/64), 2)$。数据格式支持$ND$。
- **mm\_weight\_scale** (`Tensor`)：可选参数，默认None。表示共享专家右矩阵`mm_weight`的量化系数。数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`。pertensor量化场景下支持1维，shape为$(1，)$。mx量化场景下支持3维，shape为$(ceil(H2/64), N2, 2)$。数据格式支持$ND$。
- **gmm\_x\_quant\_mode** (`int`)：可选参数，表示路由专家左矩阵的量化模式。当前版本支持配置为1和6，分别表示pertensor量化和mx量化。
- **gmm\_weight\_quant\_mode** (`int`)：可选参数，表示路由专家右矩阵的量化模式。当前版本支持配置为1和6，分别表示pertensor量化和mx量化。
- **mm\_x\_quant\_mode** (`int`)：可选参数，表示共享专家左矩阵的量化模式。当前版本支持配置为1和6，分别表示pertensor量化和mx量化。
- **mm\_weight\_quant\_mode** (`int`)：可选参数，表示共享专家右矩阵的量化模式。当前版本支持配置为1和6，分别表示pertensor量化和mx量化。
- **permute\_out\_flag** (`bool`)：可选参数，默认False。是否返回通信后重排的路由专家矩阵（即`permute_out`）。若为True，则返回值中包含该张量。
- **group\_size** (`List[int]`)：可选参数，默认为\[0, 0, 0\]。表示量化中`gmm_x_scale`、`gmm_weight_scale`、`mm_x_scale`、`mm_weight_scale`输入的一个数在其所在的对应维度方向上可以用于该方向`gmm_x`、`gmm_weight`、`mm_x`、`mm_weight`输入的多少个数的量化，`group_sizes`为\[groupSizeM，groupSizeN，groupSizeK\]。groupSizeM、groupSizeN、groupSizeK分别表示一个量化系数在各个维度对应的数的个数。

    仅mx量化场景时，`group_sizes`取值有效，其他场景需传入\[0, 0, 0\]。设置原理参考约束说明。

- **gmm\_x\_dtype** (`int`)：可选参数，默认None。表示路由专家左矩阵`gmm_x`的实际数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`）需要指定该参数取值。
- **gmm\_weight\_dtype** (`int`)：可选参数，默认None。表示路由专家右矩阵`gmm_weight`的实际数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`）需要指定该参数取值。
- **gmm\_x\_scale\_dtype** (`int`)：可选参数，默认None。表示路由专家左矩阵量化系数`gmm_x_scale`的实际数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0fnu`）需要指定该参数取值。
- **gmm\_weight\_scale\_dtype** (`int`)：可选参数，默认None。表示路由专家右矩阵量化系数`gmm_weight_scale`的实际数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0fnu`）需要指定该参数取值。
- **mm\_x\_dtype** (`int`)：可选参数，表示共享专家左矩阵`mm_x`的数据类型。当存在共享专家计算时，对于PyTorch原生不支持的数据类型（如`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`）需要指定该参数取值。
- **mm\_weight\_dtype** (`int`)：可选参数，表示共享专家右矩阵`mm_weight`的数据类型。当存在共享专家计算时，对于PyTorch原生不支持的数据类型（如`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`）需要指定该参数取值。
- **mm\_x\_scale\_dtype** (`int`)：可选参数，表示共享专家左矩阵量化系数`mm_x_scale`的数据类型。当存在共享专家计算时，对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0fnu`）需要指定该参数取值。
- **mm\_weight\_scale\_dtype** (`int`)：可选参数，表示共享专家右矩阵量化系数`mm_weight_scale`的数据类型。当存在共享专家计算时，对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0fnu`）需要指定该参数取值。
- **mm\_y\_dtype** (`int`)：可选参数，表示共享专家输出张量`mm_y`的数据类型。当存在共享专家计算时，需要指定该参数取值。数据类型支持`torch.float16`、`torch.bfloat16`。
- **comm\_mode** (`str`)：可选参数，表示通信模式。默认值为None。支持`"ai_cpu"`、`"ccu"`和None。当为None时，默认走AI CPU通信。

## 返回值说明

- **gmm\_y** (`Tensor`)：表示路由专家GroupedMatmul计算的输出，数据类型为`gmm_y_dtype`指定的类型，支持`torch.float16`、`torch.bfloat16`。支持2维，shape为$(A，N1)$。数据格式支持$ND$。
- **mm\_y** (`Tensor`)：表示共享专家MatMul的输出，数据类型为`mm_y_dtype`指定的类型，支持`torch.float16`、`torch.bfloat16`，并和`gmm_y`数据类型保持一致。支持2维，shape为$(BS，N2)$。仅当传入`mm_x`与`mm_weight`才输出。数据格式支持$ND$。
- **permute\_out** (`Tensor`)：计算输出，Permute之后的输出，数据类型与`gmm_x`保持一致。支持2维，shape为$(A，H1)$。数据格式支持$ND$。

## 约束说明

- 该接口支持训练、推理场景下使用。
- **通信引擎约束**：支持CCU和AI CPU通信，通过`comm_mode`参数配置。当`comm_mode`为None时，走AI CPU通信。
- 该接口支持单算子模式调用和T-T量化场景的图模式调用。
- 参数说明中Shape涉及的变量说明：
    - BS表示batch sequence size。
    - K表示选取的topK专家个数。当存在共享专家计算时，K需要满足取值范围\[2，8\]。
    - BSK=sum(send\_counts)，表示本卡AlltoAllv通信中发送给其他卡的总token数，取值范围\(0，52428800\)。
    - H1表示本卡路由专家的hidden size，取值范围\(0，65536\)。
    - H2表示本卡共享专家的hidden size，取值范围\(0，12288\]。
    - N1表示路由专家输出维度，取值范围\(0，65536\)。
    - N2表示共享专家输出维度，取值范围\(0，65536\)。
    - e表示通信后每张卡上的专家数量，取值范围\(0，32\]。$e \times ep\_world\_size \le 256$。
    - A表示路由专家计算输出的总token数。$A=sum(recv\_counts)$。EP通信域内所有卡上的$A$累加和等于所有卡上的$BSK$累加和。
    - 第$i$张卡发送到第$j$张卡数据量为`send_counts[j]`与第$j$张卡接收数据量为`recv_counts[i]`必须相等。
- `gmm_x_quant_mode`、`gmm_weight_quant_mode`、`mm_x_quant_mode`、`mm_weight_quant_mode`值与量化模式关系如下：
    - 0：非量化
    - 1：pertensor
    - 2：perchannel
    - 3：pertoken
    - 4：pergroup
    - 5：perblock
    - 6：mx量化
    - 7：pertoken动态量化
    - 当前`gmm_x_quant_mode`、`gmm_weight_quant_mode`的组合仅支持\[1, 1\]和\[6, 6\]，分别表示T-T量化和mx量化。
    - 当前`mm_x_quant_mode`和`mm_weight_quant_mode`的组合仅支持\[1, 1\]和\[6, 6\]，分别表示T-T量化和mx量化，且量化组合需与`gmm_x_quant_mode`、`gmm_weight_quant_mode`组合保持一致。
- **group\_sizes**：
    - groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据输入`gmm_x_scale`、`gmm_weight_scale`、`mm_x_scale`、`mm_weight_scale`、`gmm_x`、`gmm_weight`、`mm_x`、`mm_weight`的shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。
    - 设置原理：如果groupSizeM=0，表示m方向量化分组值由接口推导，推导公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与`gmm_x`、`mm_x`中的m方向值一致，scaleM与`gmm_x_scale`、`mm_x_scale`中的m方向值一致；如果groupSizeK=0，表示k方向量化分组值由接口推导，推导公式为groupSizeK = k / scaleK（需保证k能被scaleK整除），其中k与`gmm_x`、`mm_x`中的k方向值一致，scaleK与`gmm_x_scale`、`mm_x_scale`中的k方向值一致；如果groupSizeN=0，表示n方向量化分组值由接口推导，推导公式为groupSizeN = n / scaleN（需保证n能被scaleN整除），其中n与`gmm_weight`、`mm_weight`中的n方向值一致，scaleN与`gmm_weight_scale`、`mm_weight_scale`中的n方向值一致。
    - 如果满足重新设置条件，当`gmm_x_scale`、`mm_x_scale`、`mm_weight_scale`输入都是3维，`gmm_weight_scale`是4维时，且数据类型都为`torch_npu.float8_e8m0fnu`时，\[groupSizeM, groupSizeN, groupSizeK\]取值组合会推导为\[1, 1, 32\]。
- 当存在某张卡的输出张量（`gmm_y`、`mm_y`、`permute_out`）均为空Tensor时，必须显式调用 `torch.distributed.barrier()`，确保这张卡进程同步等待其他卡完成通信与计算。若未添加同步，AlltoAllv通信将因进程不同步而阻塞。

- 各量化模式下输入输出数据类型详细约束如下表：

    **表 1** T-T量化数据类型约束

    | gmm_x | gmm_weight | gmm_x_scale | gmm_weight_scale | gmm_x_quant_mode/gmm_weight_quant_mode | gmm_y | mm_x | mm_weight | mm_x_scale | mm_weight_scale | mm_x_quant_mode/mm_weight_quant_mode | mm_y |
    |---------|--------|--------|--------|--------|--------|---------|--------|--------|--------|--------|--------|
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.float16` | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.float16` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.bfloat16` | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.bfloat16` |

    **表 2** mx量化数据类型约束

    | gmm_x | gmm_weight | gmm_x_scale | gmm_weight_scale | gmm_x_quant_mode/gmm_weight_quant_mode | gmm_y | mm_x | mm_weight | mm_x_scale | mm_weight_scale | mm_x_quant_mode/mm_weight_quant_mode | mm_y |
    |---------|--------|--------|--------|--------|--------|---------|--------|--------|--------|--------|--------|
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |

## 调用示例

- 单算子模式调用示例

    - T-T量化场景示例

        ```python
        import torch
        import torch_npu
        import torch.distributed as dist
        import torch.multiprocessing as mp
        import numpy as np

        def generate_counts(ep_world_size, e, total_tokens, seed=None):
            np.random.seed(seed if seed is not None else 42)
            per_rank_total = total_tokens
            base = per_rank_total // (ep_world_size * e)
            remainder = per_rank_total % (ep_world_size * e)
            send_counts = [base] * (ep_world_size * e)
            for i in range(remainder):
                send_counts[-1 - i] += 1
            recv_counts = send_counts.copy()
            return send_counts, recv_counts

        def run_npu_alltoallv_quant_gmm(rank, world_size, master_ip, master_port):
            torch_npu.npu.set_device(rank)
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            if torch.__version__ > "2.0.1":
                hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            else:
                hcom_info = default_pg.get_hccl_comm_name(rank)

            BS = 128
            K = 2
            e = 2
            H1, N1 = 256, 256
            H2, N2 = 256, 128
            total_tokens = BS * K
            send_counts, recv_counts = generate_counts(world_size, e, total_tokens, seed=rank)
            gmm_x = torch.randint(0, 255, (total_tokens, H1), dtype=torch.uint8).npu()
            gmm_weight = torch.randint(0, 255, (e, H1, N1), dtype=torch.uint8).npu()
            gmm_x_scale = torch.tensor([0.5], dtype=None).npu()
            gmm_weight_scale = torch.tensor([0.3], dtype=None).npu()
            mm_x = torch.randint(0, 255, (BS, H2), dtype=torch.uint8).npu()
            mm_weight = torch.randint(0, 255, (H2, N2), dtype=torch.uint8).npu()
            mm_x_scale = torch.tensor([0.4], dtype=None).npu()
            mm_weight_scale = torch.tensor([0.2], dtype=None).npu()
            quant_mode = 1
            out_dtype = torch.float16

            gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_quant_gmm(
                gmm_x=gmm_x,
                gmm_weight=gmm_weight,
                gmm_x_scale=gmm_x_scale,
                gmm_weight_scale=gmm_weight_scale,
                hcom=hcom_info,
                ep_world_size=world_size,
                send_counts=send_counts,
                recv_counts=recv_counts,
                gmm_y_dtype=out_dtype,
                mm_x=mm_x,
                mm_weight=mm_weight,
                mm_x_scale=mm_x_scale,
                mm_weight_scale=mm_weight_scale,
                gmm_x_quant_mode=quant_mode,
                gmm_weight_quant_mode=quant_mode,
                mm_x_quant_mode=quant_mode,
                mm_weight_quant_mode=quant_mode,
                permute_out_flag=True,
                gmm_x_dtype=torch_npu.hifloat8,
                gmm_weight_dtype=torch_npu.hifloat8,
                gmm_x_scale_dtype=None,
                gmm_weight_scale_dtype=None,
                mm_x_dtype=torch_npu.hifloat8,
                mm_weight_dtype=torch_npu.hifloat8,
                mm_x_scale_dtype=None,
                mm_weight_scale_dtype=None,
                mm_y_dtype=out_dtype,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                group_size=None
            )

        if __name__ == "__main__":
            world_size = 2
            master_ip = "127.0.0.1"
            master_port = "50001"
            mp.spawn(run_npu_alltoallv_quant_gmm, args=(world_size, master_ip, master_port), nprocs=world_size, join=True)
        ```

    - mx量化场景示例-mxfp8

        ```python
        import torch
        import torch_npu
        import torch.distributed as dist
        import torch.multiprocessing as mp
        import numpy as np
        import math

        def generate_counts(ep_world_size, e, total_tokens, seed=None):
            np.random.seed(seed if seed is not None else 42)
            per_rank_total = total_tokens
            base = per_rank_total // (ep_world_size * e)
            remainder = per_rank_total % (ep_world_size * e)
            send_counts = [base] * (ep_world_size * e)
            for i in range(remainder):
                send_counts[-1 - i] += 1
            recv_counts = send_counts.copy()
            return send_counts, recv_counts

        def run_npu_alltoallv_quant_gmm(rank, world_size, master_ip, master_port):
            torch_npu.npu.set_device(rank)
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            if torch.__version__ > "2.0.1":
                hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            else:
                hcom_info = default_pg.get_hccl_comm_name(rank)

            BS = 128
            K = 2
            e = 2
            H1, N1 = 256, 256
            H2, N2 = 256, 128
            total_tokens = BS * K
            send_counts, recv_counts = generate_counts(world_size, e, total_tokens, seed=rank)
            gmm_x = torch.ones(total_tokens, H1, dtype=torch.int8).to(torch.float8_e4m3fn).npu()
            gmm_weight = torch.ones(e, H1, N1, dtype=torch.int8).to(torch.float8_e5m2).npu()
            gmm_x_scale = torch.ones(total_tokens, math.ceil(H1 / 64), 2, dtype=torch.int8).npu()
            gmm_weight_scale = torch.ones(e, math.ceil(H1 / 64), N1, 2, dtype=torch.int8).npu()
            mm_x = torch.ones(BS, H2, dtype=torch.int8).to(torch.float8_e4m3fn).npu()
            mm_weight = torch.ones(H2, N2, dtype=torch.int8).to(torch.float8_e5m2).npu()
            mm_x_scale = torch.ones(BS, math.ceil(H2 / 64), 2, dtype=torch.int8).npu()
            mm_weight_scale = torch.ones(math.ceil(H2 / 64), N2, 2, dtype=torch.int8).npu()
            quant_mode = 6
            out_dtype = torch.float16

            gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_quant_gmm(
                gmm_x=gmm_x,
                gmm_weight=gmm_weight,
                gmm_x_scale=gmm_x_scale,
                gmm_weight_scale=gmm_weight_scale,
                hcom=hcom_info,
                ep_world_size=world_size,
                send_counts=send_counts,
                recv_counts=recv_counts,
                gmm_y_dtype=out_dtype,
                mm_x=mm_x,
                mm_weight=mm_weight,
                mm_x_scale=mm_x_scale,
                mm_weight_scale=mm_weight_scale,
                gmm_x_quant_mode=quant_mode,
                gmm_weight_quant_mode=quant_mode,
                mm_x_quant_mode=quant_mode,
                mm_weight_quant_mode=quant_mode,
                permute_out_flag=True,
                gmm_x_dtype=None,
                gmm_weight_dtype=None,
                gmm_x_scale_dtype=torch_npu.float8_e8m0fnu,
                gmm_weight_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_x_dtype=None,
                mm_weight_dtype=None,
                mm_x_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_weight_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_y_dtype=out_dtype,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                group_size=None
            )

        if __name__ == "__main__":
            world_size = 2
            master_ip = "127.0.0.1"
            master_port = "50001"
            mp.spawn(run_npu_alltoallv_quant_gmm, args=(world_size, master_ip, master_port), nprocs=world_size, join=True)
        ```

    - mx量化场景示例-mxfp4

        ```python
        import torch
        import torch_npu
        import torch.distributed as dist
        import torch.multiprocessing as mp
        import numpy as np
        import math

        def generate_counts(ep_world_size, e, total_tokens, seed=None):
            np.random.seed(seed if seed is not None else 42)
            per_rank_total = total_tokens
            base = per_rank_total // (ep_world_size * e)
            remainder = per_rank_total % (ep_world_size * e)
            send_counts = [base] * (ep_world_size * e)
            for i in range(remainder):
                send_counts[-1 - i] += 1
            recv_counts = send_counts.copy()
            return send_counts, recv_counts

        def run_npu_alltoallv_quant_gmm(rank, world_size, master_ip, master_port):
            torch_npu.npu.set_device(rank)
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            if torch.__version__ > "2.0.1":
                hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            else:
                hcom_info = default_pg.get_hccl_comm_name(rank)

            BS = 128
            K = 2
            e = 2
            H1, N1 = 256, 256
            H2, N2 = 256, 128
            total_tokens = BS * K
            send_counts, recv_counts = generate_counts(world_size, e, total_tokens, seed=rank)
            # 使用uint8代替两个mxfp4，此处uint8的内轴需要扩展2倍才是真实的mxfp4的内轴
            gmm_x = torch.ones(total_tokens, int(H1/2), dtype=torch.uint8).to(torch.float8_e4m3fn).npu()
            gmm_weight = torch.ones(e, H1, int(N1/2), dtype=torch.uint8).to(torch.float8_e5m2).npu()
            gmm_x_scale = torch.ones(total_tokens, math.ceil(H1 / 64), 2, dtype=torch.int8).npu()
            gmm_weight_scale = torch.ones(e, math.ceil(H1 / 64), N1, 2, dtype=torch.int8).npu()
            mm_x = torch.ones(BS, int(H2/2), dtype=torch.uint8).to(torch.float8_e4m3fn).npu()
            mm_weight = torch.ones(H2, int(N2/2), dtype=torch.uint8).to(torch.float8_e5m2).npu()
            mm_x_scale = torch.ones(BS, math.ceil(H2 / 64), 2, dtype=torch.uint8).npu()
            mm_weight_scale = torch.ones(math.ceil(H2 / 64), N2, 2, dtype=torch.int8).npu()
            quant_mode = 6
            out_dtype = torch.float16

            gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_quant_gmm(
                gmm_x=gmm_x,
                gmm_weight=gmm_weight,
                gmm_x_scale=gmm_x_scale,
                gmm_weight_scale=gmm_weight_scale,
                hcom=hcom_info,
                ep_world_size=world_size,
                send_counts=send_counts,
                recv_counts=recv_counts,
                gmm_y_dtype=out_dtype,
                mm_x=mm_x,
                mm_weight=mm_weight,
                mm_x_scale=mm_x_scale,
                mm_weight_scale=mm_weight_scale,
                gmm_x_quant_mode=quant_mode,
                gmm_weight_quant_mode=quant_mode,
                mm_x_quant_mode=quant_mode,
                mm_weight_quant_mode=quant_mode,
                permute_out_flag=True,
                gmm_x_dtype=torch_npu.float4_e2m1fn_x2,
                gmm_weight_dtype=torch_npu.float4_e2m1fn_x2,
                gmm_x_scale_dtype=torch_npu.float8_e8m0fnu,
                gmm_weight_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_x_dtype=torch_npu.float4_e2m1fn_x2,
                mm_weight_dtype=torch_npu.float4_e2m1fn_x2,
                mm_x_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_weight_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_y_dtype=out_dtype,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                group_size=None
            )

        if __name__ == "__main__":
            world_size = 2
            master_ip = "127.0.0.1"
            master_port = "50001"
            mp.spawn(run_npu_alltoallv_quant_gmm, args=(world_size, master_ip, master_port), nprocs=world_size, join=True)
        ```

- 图模式调用示例

    - T-T量化场景调用示例

        ```python
        import torch
        import torch_npu
        import torch.distributed as dist
        import torch.multiprocessing as mp
        import torchair
        import numpy as np
        from en_dtypes import hifloat8

        class ALLTOALLV_GMM_GRAPH_Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, _gmm_x, _gmm_weight, _gmm_x_scale, _gmm_weight_scale, _hcom, _ep_world_size,
                        _send_counts, _recv_counts, _gmm_y_dtype, _mm_y_dtype, _mm_x, _mm_weight, _mm_x_scale,
                        _mm_weight_scale, _permute_out_flag, _gmm_x_quant_mode, _gmm_weight_quant_mode,
                        _mm_x_quant_mode, _mm_weight_quant_mode, _gmm_x_dtype, _gmm_weight_dtype, _mm_x_dtype,
                        _mm_weight_dtype, _gmm_weight_scale_dtype=None, _gmm_x_scale_dtype=None,
                        _mm_x_scale_dtype=None, _mm_weight_scale_dtype=None, _trans_gmm_weight=False,
                        _trans_mm_weight=False):
                if _trans_gmm_weight:
                    _gmm_weight = torch.transpose(_gmm_weight, -2, -1)
                if _trans_mm_weight and _mm_weight is not None:
                    _mm_weight = _mm_weight.t()
                gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_quant_gmm(
                    gmm_x=_gmm_x,
                    gmm_weight=_gmm_weight,
                    gmm_x_scale=_gmm_x_scale,
                    gmm_weight_scale=_gmm_weight_scale,
                    hcom=_hcom,
                    ep_world_size=_ep_world_size,
                    send_counts=_send_counts,
                    recv_counts=_recv_counts,
                    gmm_y_dtype=_gmm_y_dtype,
                    mm_y_dtype=_mm_y_dtype,
                    mm_x=_mm_x,
                    mm_weight=_mm_weight,
                    mm_x_scale=_mm_x_scale,
                    mm_weight_scale=_mm_weight_scale,
                    permute_out_flag=_permute_out_flag,
                    gmm_x_quant_mode=_gmm_x_quant_mode,
                    gmm_weight_quant_mode=_gmm_weight_quant_mode,
                    mm_x_quant_mode=_mm_x_quant_mode,
                    mm_weight_quant_mode=_mm_weight_quant_mode,
                    gmm_x_dtype=_gmm_x_dtype,
                    gmm_weight_dtype=_gmm_weight_dtype,
                    mm_x_dtype=_mm_x_dtype,
                    mm_weight_dtype=_mm_weight_dtype
                )
                return gmm_y, mm_y, permute_out

        def run_npu_alltoallv_gmm(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype,
                                gmm_x_scale, gmm_w_scale):
            torch_npu.npu.set_device(rank)
            init_method = 'tcp://' + master_ip + ':' + master_port
            dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            if torch.__version__ > '2.0.1':
                hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            else:
                hcom_info = default_pg.get_hccl_comm_name(rank)

            input_arr = np.random.uniform(1, -1, gmm_x).astype(hifloat8)
            weight_arr = np.random.uniform(1, -1, gmm_w).astype(hifloat8)
            input = torch.from_numpy(input_arr.view(np.uint8)).npu()
            weight = torch.from_numpy(weight_arr.view(np.uint8)).npu()
            input_scale = torch.randn(gmm_x_scale, dtype=torch.float32).npu()
            weight_scale = torch.randn(gmm_w_scale, dtype=torch.float32).npu()

            model = ALLTOALLV_GMM_GRAPH_Model()
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            # 静态图：dynamic=False；动态图：dynamic=True
            model = torch.compile(ALLTOALLV_GMM_GRAPH_Model(), backend=npu_backend, dynamic=False)
            print(model(_gmm_x=input,
                        _gmm_weight=weight,
                        _gmm_x_scale=input_scale,
                        _gmm_weight_scale=weight_scale,
                        _hcom=hcom_info,
                        _ep_world_size=ep_world_size,
                        _send_counts=send_counts,
                        _recv_counts=recv_counts,
                        _gmm_y_dtype=torch.float16,
                        _mm_y_dtype=None,
                        _mm_x=None,
                        _mm_weight=None,
                        _mm_x_scale=None,
                        _mm_weight_scale=None,
                        _permute_out_flag=False,
                        _gmm_x_quant_mode=1,
                        _gmm_weight_quant_mode=1,
                        _mm_x_quant_mode=None,
                        _mm_weight_quant_mode=None,
                        _gmm_x_dtype=dtype,
                        _gmm_weight_dtype=dtype,
                        _mm_x_dtype=None,
                        _mm_weight_dtype=None,
                        _gmm_weight_scale_dtype=torch.float32,
                        _gmm_x_scale_dtype=torch.float32,
                        _mm_x_scale_dtype=None,
                        _mm_weight_scale_dtype=None,
                        _trans_gmm_weight=False,
                        _trans_mm_weight=False))

        if __name__ == "__main__":
            epWorldSize = 2
            e = 4
            master_ip = '127.0.0.1'
            master_port = '50001'
            BS = 512
            K = 8
            gmm_x_shape = [BS*K, 2048]
            gmm_weight_shape = [e, 2048, 2048]
            send_counts = [512] * (e * epWorldSize)
            recv_counts = [512] * (e * epWorldSize)
            scale_shape = [1]
            dtype = torch_npu.hifloat8
            mp.spawn(run_npu_alltoallv_gmm, args=(epWorldSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype, scale_shape, scale_shape), nprocs=epWorldSize)
        ```
