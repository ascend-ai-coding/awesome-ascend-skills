# torch_npu.npu_quant_gmm_alltoallv

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：实现路由专家GroupedMatmul和AlltoAllv的融合，先计算后通信，同时与共享专家MatMul计算并行融合。支持T-T量化模式（即pertensor-pertensor量化模式）和mx量化模式（即特殊的pergroup-pergroup量化模式）。

- **计算公式**：
    - 路由专家：

        $$
        \begin{aligned}
        &gmm\_y = (gmm\_x \times gmm\_x\_scale) \mathbin{@} (gmm\_weight \times gmm\_weight\_scale) \\
        &unpermute\_out = Unpermute(gmm\_y) \\
        &y = AlltoAllv(unpermute\_out)
        \end{aligned}
        $$

        - gmm\_x指路由专家GroupedMatMul计算的左矩阵。
        - gmm\_x\_scale指路由专家左矩阵的量化参数。
        - gmm\_weight指路由专家GroupedMatMul计算的右矩阵。
        - gmm\_weight\_scale指路由专家右矩阵的量化参数。
        - gmm\_y指路由专家进行GroupedMatMul计算的输出，后续用于Unpermute计算。
        - unpermute\_out是gmm\_y进行Unpermute计算的输出结果，作为AlltoAllv通信的输入。
        - y指对unpermute\_out进行AlltoAllv通信输出。

    - 共享专家：

        $$
        mm\_y = (mm\_x \times mm\_x\_scale) \mathbin{@} (mm\_weight \times mm\_weight\_scale)
        $$

        - mm\_x指共享专家MatMul计算的左矩阵。
        - mm\_x\_scale指共享专家左矩阵的量化参数。
        - mm\_weight指共享专家MatMul计算的右矩阵。
        - mm\_weight\_scale指共享专家右矩阵的量化参数。
        - mm\_y指共享专家MatMul计算的输出。

## 函数原型

```python
torch_npu.npu_quant_gmm_alltoallv(gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, hcom, ep_world_size, send_counts, recv_counts, gmm_y_dtype, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, mm_x_scale=None, mm_weight_scale=None, comm_quant_scale=None, gmm_x_quant_mode=None, gmm_weight_quant_mode=None, mm_x_quant_mode=None, mm_weight_quant_mode=None, comm_quant_mode=None, group_size=None, gmm_x_dtype=None, gmm_weight_dtype=None, gmm_x_scale_dtype=None, gmm_weight_scale_dtype=None, mm_x_dtype=None, mm_weight_dtype=None, mm_x_scale_dtype=None, mm_weight_scale_dtype=None, comm_quant_dtype=None, mm_y_dtype=None, comm_mode=None) -> (Tensor, Tensor)
```

## 参数说明

- **gmm\_x**（`Tensor`）：**必选参数**，表示GroupedMatmul计算的左矩阵Tensor。数据类型支持`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1fn_x2`。支持2维，shape为\(A, H1\)，数据格式支持$ND$，其中数据类型为float4时内轴H1需要为偶数，以保证8bits可以转换为2个float4。
- **gmm\_weight**（`Tensor`）：**必选参数**，GroupedMatmul的右矩阵。数据类型与`gmm_x`一致。支持3维，shape为\(e, H1, N1\)，数据格式支持$ND$，全量化场景下，当`gmm_x`、`gmm_weight`均为float4系列时，仅支持推理场景，此时输入`gmm_x`的H1需要为偶数，且当`gmm_weight`不转置时内轴N1需为偶数，转置时内轴H1需要为偶数，以保证8bits可以转换为2个float4。
- **gmm\_x\_scale**（`Tensor`）：**必选参数**，表示左矩阵的量化缩放系数，数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`。pertensor量化场景下支持1维，shape为\(1,\)。mx量化场景下支持3维，shape为\(A, ceil\(H1/64\), 2\)。数据格式为$ND$。
- **gmm\_weight\_scale**（`Tensor`）：**必选参数**，表示右矩阵的量化参数，数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`。pertensor量化场景下支持1维，shape为\(1,\)。mx量化场景下支持4维，shape为\(e, ceil\(H1/64\), N1, 2\)。数据格式为$ND$。
- **hcom**（`str`）：**必选参数**，表示专家并行（EP）的通信域名称，字符串长度需在\(0,128\)范围内。
- **ep\_world\_size**（`int`）：**必选参数**，表示EP通信域的size，取值支持2、4、8、16、32、64、128、256，CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。
- **send\_counts**（`List[int]`）：**必选参数**，表示发送给其他卡的token数列表，数据类型支持`torch.int64`，数组大小为e \* ep\_world\_size。
- **recv\_counts**（`List[int]`）：**必选参数**，表示接收其他卡的token数列表，数据类型支持`torch.int64`，数组大小为e \* ep\_world\_size。
- **gmm\_y\_dtype**（`int`）：**必选参数**，表示路由专家GroupedMatmul计算输出张量`gmm_y`的数据类型（例如`torch.float16`）。数据类型支持`torch.float16`、`torch.bfloat16`。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **send\_counts\_tensor**（`Tensor`）：**可选参数**，默认值为`None`，当前仅支持输入None。
- **recv\_counts\_tensor**（`Tensor`）：**可选参数**，默认值为`None`，当前仅支持输入None。
- **mm\_x**（`Tensor`）：**可选参数**，默认值为`None`，表示共享专家MatMul计算中的左矩阵，数据类型与`gmm_x`一致。支持2维，Shape为\(BS, H2\)，数据格式为$ND$，其中数据类型为float4时内轴H2需为偶数，以保证8bits可以转换为2个float4。
- **mm\_weight**（`Tensor`）：**可选参数**，默认值为`None`，表示共享专家MatMul计算中的右矩阵，数据类型与`gmm_x`一致。支持2维，shape为\(H2, N2\)。数据格式为$ND$。全量化场景下，当`mm_x`、`mm_weight`均为float4系列时，仅支持推理场景，此时输入`mm_x`的H2需要为偶数，且当`mm_weight`不转置时内轴N2需为偶数，转置时内轴H2需要为偶数，以保证8bits可以转换为2个float4。
- **mm\_x\_scale**（`Tensor`）：**可选参数**，默认值为`None`，表示共享专家MatMul左矩阵的量化参数，数据类型为`torch.float32`。pertensor量化场景下支持1维，shape为\(1,\)。mx量化场景下支持3维，shape为\(BS, ceil\(H2/64\), 2\)。数据格式为$ND$。
- **mm\_weight\_scale**（`Tensor`）：**可选参数**，默认值为`None`，表示共享专家MatMul右矩阵的量化参数，数据类型为`torch.float32`。pertensor量化场景下支持1维，shape为\(1,\)。mx量化场景下支持3维，shape为\(ceil\(H2/64\), N2, 2\)。数据格式为$ND$。
- **comm\_quant\_scale**（`Tensor`）：**可选参数**，默认值为`None`，表示低比特通信的量化参数，数据类型为`torch.float32`，维度为1维，当前暂不支持。
- **gmm\_x\_quant\_mode**（`int`）：**可选参数**，默认值为`None`，表示左矩阵量化模式，当前版本支持配置为1和6，分别表示pertensor量化和mx量化。
- **gmm\_weight\_quant\_mode**（`int`）：**可选参数**，默认值为`None`，表示右矩阵量化模式，当前版本支持配置为1和6，分别表示pertensor量化和mx量化。
- **mm\_x\_quant\_mode**（`int`）：**可选参数**，默认值为`None`，表示共享专家左矩阵的量化模式，当前版本支持配置为1和6，分别表示pertensor量化和mx量化。
- **mm\_weight\_quant\_mode**（`int`）：**可选参数**，默认值为`None`，表示共享专家右矩阵的量化模式，当前版本支持配置为1和6，分别表示pertensor量化和mx量化。
- **comm\_quant\_mode**（`int`）：**可选参数**，默认值为`None`，表示低比特通信的量化模式，当前仅支持配置为0。
- **group\_size**（`List[int]`）：**可选参数**，默认为\[0, 0, 0\]。表示量化中`gmm_x_scale`、`gmm_weight_scale`、`mm_x_scale`、`mm_weight_scale`输入的一个数在其所在的对应维度方向上可以用于该方向`gmm_x`、`gmm_weight`、`mm_x`、`mm_weight`输入的多少个数的量化，`group_size`为\[groupSizeM, groupSizeN, groupSizeK\]。groupSizeM、groupSizeN、groupSizeK分别表示一个量化系数在各个维度对应的数的个数。
    - 仅mx量化场景时，`group_size`取值有效，其他场景需传入\[0, 0, 0\]。

- **gmm\_x\_dtype**（`int`）：**可选参数**，默认值为`None`。表示路由专家左矩阵`gmm_x`的实际数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`）需要指定该参数取值。
- **gmm\_weight\_dtype**（`int`）：**可选参数**，默认值为`None`。表示路由专家右矩阵`gmm_weight`的实际数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`）需要指定该参数取值。
- **gmm\_x\_scale\_dtype**（`int`）：**可选参数**，默认值为`None`。表示路由专家左矩阵量化系数`gmm_x_scale`的实际数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0fnu`）需要指定该参数取值。
- **gmm\_weight\_scale\_dtype**（`int`）：**可选参数**，默认值为`None`。表示路由专家右矩阵量化系数`gmm_weight_scale`的实际数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0fnu`）需要指定该参数取值。
- **mm\_x\_dtype**（`int`）：**可选参数**，默认值为`None`，表示共享专家左矩阵`mm_x`的数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`）需要指定该参数取值。
- **mm\_weight\_dtype**（`int`）：**可选参数**，默认值为`None`，表示共享专家右矩阵`mm_weight`的数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`）需要指定该参数取值。
- **mm\_x\_scale\_dtype**（`int`）：**可选参数**，默认值为`None`，表示共享专家左矩阵量化系数`mm_x_scale`的数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0fnu`）需要指定该参数取值。
- **mm\_weight\_scale\_dtype**（`int`）：**可选参数**，默认值为`None`，表示共享专家右矩阵量化系数`mm_weight_scale`的数据类型。对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0fnu`）需要指定该参数取值。
- **comm\_quant\_dtype**（`int`）：**可选参数**，默认值为`None`，低比特通信量化后的数据类型，当前暂不支持。
- **mm\_y\_dtype**（`int`）：**可选参数**，默认值为`None`，表示共享专家输出张量`mm_y`的数据类型，数据类型支持`torch.float16`、`torch.bfloat16`。
- **comm\_mode**（`str`）：**可选参数**，表示通信引擎模式，默认值为`None`。取值支持`None`、`ai_cpu`和`ccu`，当为`None`时，使用AI CPU通信。

## 返回值说明

- **y**（`Tensor`）：表示路由专家GroupedMatmul的最终计算结果。数据类型为`gmm_y_dtype`指定的类型，支持2维，shape为\(BSK, N1\)。数据格式为$ND$。
- **mm\_y**（`Tensor`）：表示共享专家MatMul的输出，数据类型为`mm_y_dtype`指定的类型，支持2维，shape为\(BS, N2\)。仅当传入`mm_x`与`mm_weight`才输出。数据格式为$ND$。

## 约束说明

- 该接口支持训练、推理场景下使用。
<!-- npu="950" id2 -->
- 通信引擎约束：
    - <term>Ascend 950PR&950DT系列产品</term>：支持CCU通信。
<!-- end id2 -->

- 该接口支持单算子模式调用和T-T量化场景的图模式调用。
- 参数说明里shape使用的变量：
    - BSK：本卡接收的token数，是recvCounts参数累加之和，取值范围\(0, 52428800\)。
    - H1：表示路由专家hidden size隐藏层大小，取值范围\(0, 65536\)。
    - H2：表示共享专家hidden size隐藏层大小，取值范围\(0, 12288\]。
    - e：表示单卡上专家个数，e<=32，e \* epWorldSize最大支持256。
    - N1：表示路由专家的中间层维度，取值范围\(0, 65536\)。
    - N2：表示共享专家的中间层维度，取值范围\(0, 65536\)。
    - BS：batch sequence size。
    - K：表示选取TopK个专家，K的范围\[2,8\]。
    - A：本卡发送的token数，是sendCounts参数累加之和。
    - ep通信域内所有卡的A参数的累加和等于所有卡上的BSK参数的累加和。

- `gmm_x_quant_mode`、`gmm_weight_quant_mode`、`mm_x_quant_mode`、`mm_weight_quant_mode`值与量化模式关系如下：
    - 0：非量化
    - 1：pertensor
    - 2：perchannel
    - 3：pertoken
    - 4：pergroup
    - 5：perblock
    - 6：mx量化
    - 7：pertoken动态量化

- 当前gmm\_x\_quant\_mode、gmm\_weight\_quant\_mode的组合仅支持\[1, 1\]和\[6, 6\]，分别表示T-T量化和mx量化。
- 当前mm\_x\_quant\_mode和mm\_weight\_quant\_mode的组合仅支持\[1, 1\]和\[6, 6\]，分别表示T-T量化和mx量化，且量化组合需与gmm\_x\_quant\_mode、gmm\_weight\_quant\_mode组合保持一致。

- `group_size`：
    - groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据输入gmm\_x\_scale、gmm\_weight\_scale、mm\_x\_scale、mm\_weight\_scale、gmm\_x、gmm\_weight、mm\_x、mm\_weight的shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。
    - 设置原理：如果groupSizeM=0，表示m方向量化分组值由接口推导，推导公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与gmm\_x、mm\_x中的m方向值一致，scaleM与gmm\_x\_scale、mm\_x\_scale中的m方向值一致；如果groupSizeK=0，表示k方向量化分组值由接口推导，推导公式为groupSizeK = k / scaleK（需保证k能被scaleK整除），其中k与gmm\_x、mm\_x中的k方向值一致，scaleK与gmm\_x\_scale、mm\_x\_scale中的k方向值一致；如果groupSizeN=0，表示n方向量化分组值由接口推导，推导公式为groupSizeN = n / scaleN（需保证n能被scaleN整除），其中n与gmm\_weight、mm\_weight中的n方向值一致，scaleN与gmm\_weight\_scale、mm\_weight\_scale中的n方向值一致。
    - 如果满足重新设置条件，当gmm\_x\_scale、mm\_x\_scale、mm\_weight\_scale输入都是3维，gmm\_weight\_scale输入的是4维时，且数据类型都为`torch_npu.float8_e8m0fnu`时，\[groupSizeM, groupSizeN, groupSizeK\]取值组合会推导为\[1, 1, 32\]。
    - 当存在某张卡的输出张量（y、mm\_y）均为空Tensor时，必须显式调用torch.distributed.barrier\(\)，确保这张卡进程同步等待其他卡完成通信与计算。若未添加同步，AlltoAllv通信将因进程不同步而阻塞。

- 各量化模式下输入输出数据类型详细约束如下表：

    **表 1**  T-T量化数据类型约束

    | gmm_x | gmm_weight | gmm_x_scale | gmm_weight_scale | gmm_x_quant_mode/gmm_weight_quant_mode | gmm_y | mm_x | mm_weight | mm_x_scale | mm_weight_scale | mm_x_quant_mode/mm_weight_quant_mode | mm_y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.float16` | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.float16` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.bfloat16` | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.bfloat16` |

    **表 2**  mx量化数据类型约束

    | gmm_x | gmm_weight | gmm_x_scale | gmm_weight_scale | gmm_x_quant_mode/gmm_weight_quant_mode | gmm_y | mm_x | mm_weight | mm_x_scale | mm_weight_scale | mm_x_quant_mode/mm_weight_quant_mode | mm_y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |

## 调用示例

- 单算子模式调用示例

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def run_npu_quant_gmm_alltoallv(rank, world_size, master_ip, master_port):
        torch_npu.npu.set_device(rank)
        init_method = f"tcp://{master_ip}:{master_port}"
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)

        BS, K = 128, 2
        H1, N1 = 256, 256
        H2, N2 = 256, 128
        e = 2
        ep_world_size = world_size
        total_tokens = BS * K
        out_dtype = torch.float16
        gmm_x = torch.randint(0, 30, (total_tokens, H1), dtype=torch.uint8).npu()
        gmm_weight = torch.randint(0, 30, (e, H1, N1), dtype=torch.uint8).npu()
        gmm_x_scale = torch.tensor([0.5], dtype=torch.float32).npu()
        gmm_weight_scale = torch.tensor([0.3], dtype=torch.float32).npu()
        mm_x = torch.randint(0, 30, (BS, H2), dtype=torch.uint8).npu()
        mm_weight = torch.randint(0, 30, (H2, N2), dtype=torch.uint8).npu()
        mm_x_scale = torch.tensor([0.4], dtype=torch.float32).npu()
        mm_weight_scale = torch.tensor([0.2], dtype=torch.float32).npu()
        send_counts = [total_tokens // (e * ep_world_size)] * (e * ep_world_size)
        recv_counts = [total_tokens // (e * ep_world_size)] * (e * ep_world_size)

        y, mm_y = torch_npu.npu_quant_gmm_alltoallv(
            gmm_x=gmm_x,
            gmm_weight=gmm_weight,
            gmm_x_scale=gmm_x_scale,
            gmm_weight_scale=gmm_weight_scale,
            hcom=hcom_info,
            ep_world_size=ep_world_size,
            send_counts=send_counts,
            recv_counts=recv_counts,
            gmm_y_dtype=out_dtype,
            mm_x=mm_x,
            mm_weight=mm_weight,
            mm_x_scale=mm_x_scale,
            mm_weight_scale=mm_weight_scale,
            gmm_x_quant_mode=1,
            gmm_weight_quant_mode=1,
            mm_x_quant_mode=1,
            mm_weight_quant_mode=1,
            comm_quant_mode=0,
            gmm_x_dtype=torch_npu.hifloat8,
            gmm_weight_dtype=torch_npu.hifloat8,
            gmm_x_scale_dtype=torch.float32,
            gmm_weight_scale_dtype=torch.float32,
            mm_x_dtype=torch_npu.hifloat8,
            mm_weight_dtype=torch_npu.hifloat8,
            mm_x_scale_dtype=torch.float32,
            mm_weight_scale_dtype=torch.float32,
            mm_y_dtype=out_dtype,
        )

    if __name__ == "__main__":
        world_size = 2
        master_ip = '127.0.0.1'
        master_port = '50001'
        mp.spawn(run_npu_quant_gmm_alltoallv, args=(world_size, master_ip, master_port), nprocs=world_size, join=True)

    ```

- 图模式调用示例

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torchair
    import numpy as np
    from en_dtypes import hifloat8
    class GMM_ALLTOALLV_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, _gmm_x, _gmm_weight, _gmm_x_scale, _gmm_weight_scale, _hcom, _ep_world_size,
                    _send_counts, _recv_counts, _gmm_y_dtype, _mm_y_dtype, _mm_x, _mm_weight, _mm_x_scale,
                    _mm_weight_scale, _comm_quant_scale, _gmm_x_quant_mode, _gmm_weight_quant_mode,
                    _mm_x_quant_mode, _mm_weight_quant_mode, _comm_quant_mode, _gmm_x_dtype, _gmm_weight_dtype,
                    _mm_x_dtype, _mm_weight_dtype, _gmm_weight_scale_dtype=None, _gmm_x_scale_dtype=None,
                    _mm_x_scale_dtype=None, _mm_weight_scale_dtype=None, _trans_gmm_weight=False,
                    _trans_mm_weight=False):
            if _trans_gmm_weight:
                _gmm_weight = torch.transpose(_gmm_weight, -2, -1)
            if _trans_mm_weight and _mm_weight is not None:
                _mm_weight = _mm_weight.t()
            gmm_y, mm_y = torch_npu.npu_quant_gmm_alltoallv(
                _gmm_x,
                _gmm_weight,
                _gmm_x_scale,
                _gmm_weight_scale,
                _hcom,
                _ep_world_size,
                _send_counts,
                _recv_counts,
                _gmm_y_dtype,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                mm_x=_mm_x,
                mm_weight=_mm_weight,
                mm_x_scale=_mm_x_scale,
                mm_weight_scale=_mm_weight_scale,
                comm_quant_scale=_comm_quant_scale,
                gmm_x_quant_mode=_gmm_x_quant_mode,
                gmm_weight_quant_mode=_gmm_weight_quant_mode,
                mm_x_quant_mode=_mm_x_quant_mode,
                mm_weight_quant_mode=_mm_weight_quant_mode,
                comm_quant_mode=_comm_quant_mode,
                group_size=None,
                gmm_x_dtype=_gmm_x_dtype,
                gmm_weight_dtype=_gmm_weight_dtype,
                gmm_x_scale_dtype=_gmm_x_scale_dtype,
                gmm_weight_scale_dtype=_gmm_weight_scale_dtype,
                mm_x_dtype=_mm_x_dtype,
                mm_weight_dtype=_mm_weight_dtype,
                mm_x_scale_dtype=_mm_x_scale_dtype,
                mm_weight_scale_dtype=_mm_weight_scale_dtype,
                comm_quant_dtype=None,
                mm_y_dtype=_mm_y_dtype
            )
            return gmm_y, mm_y
    def run_npu_gmm_alltoallv(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype,
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
        model = GMM_ALLTOALLV_GRAPH_Model()
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        # 静态图：dynamic=False；动态图：dynamic=True
        model = torch.compile(GMM_ALLTOALLV_GRAPH_Model(), backend=npu_backend, dynamic=False)
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
                    _comm_quant_scale=None,
                    _gmm_x_quant_mode=1,
                    _gmm_weight_quant_mode=1,
                    _mm_x_quant_mode=1,
                    _mm_weight_quant_mode=1,
                    _comm_quant_mode=0,
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
        epWorkSize = 2
        e = 4
        master_ip = '127.0.0.1'
        master_port = '50001'
        BS = 512
        K = 8
        gmm_x_shape = [BS*K, 2048]
        gmm_weight_shape = [e, 2048, 2048]
        send_counts = [512] * (e * epWorkSize)
        recv_counts = [512] * (e * epWorkSize)
        dtype = torch_npu.hifloat8
        scale_shape = [1]
        mp.spawn(run_npu_gmm_alltoallv, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype, scale_shape, scale_shape), nprocs=epWorkSize)
    ```
