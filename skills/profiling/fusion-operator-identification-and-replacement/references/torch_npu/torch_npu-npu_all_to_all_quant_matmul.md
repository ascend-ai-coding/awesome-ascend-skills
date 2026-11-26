# torch_npu.npu_all_to_all_quant_matmul

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：完成量化的Matmul计算、Permute（保证通信后地址连续）和AlltoAll通信的融合，先通信后计算。支持K-C动态量化模式（即pertoken动态-perchannel量化模式）和mx量化模式（即特殊的pergroup-pergroup量化模式）。

- **计算公式**：

  假设输入`x1`的shape为\(BS, H\)，`x2`的shape为\(H\*rankSize, N\)，mx量化场景下x1Scale输入shape为\(BS, ceil\(H/64\), 2\)，rankSize为NPU卡数。

  - K-C动态量化模式：

    $$
    commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H))
    $$

    $$
    permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H)
    $$

    $$
    dynQuantX1, dynQuantX1Scale = dynamicQuant(permutedOut)
    $$

    $$
    output = (dynQuantX1@x2 + bias) * dynQuantX1Scale * x2Scale
    $$

  - mx量化模式：

    $$
    commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H))
    $$

    $$
    permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H)
    $$

    $$
    commScale = AlltoAll(x1Scale.view(rankSize, BS/rankSize, ceil(H/64), 2))
    $$

    $$
    permutedScale = commScale.permute(1, 0, 2, 3).view(BS/rankSize, ceil(H/64)*rankSize, 2)
    $$

    $$
    output = \sum_{0}^{\frac{k}{blockSize=32}} (permutedOut@x2*(permutedScale*x2Scale)) + bias
    $$

## 函数原型

```python
torch_npu.npu_all_to_all_quant_matmul(x1, x2, hcom, world_size, *, all2all_out_flag=True, bias=None, x1_scale=None, x2_scale=None, common_scale=None, x1_offset=None, x2_offset=None, x1_quant_mode=None, x2_quant_mode=None, common_quant_mode=None, group_sizes=None, all2all_axes=None, comm_quant_dtype=None, x1_quant_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None, output_scale_dtype=None, comm_scale_dtype=None, y_dtype=None, comm_mode=None) -> (Tensor, Tensor)
```

## 参数说明

- **x1**（`Tensor`）：**必选参数**，表示融合算子的左矩阵输入，对应公式中的x1。该输入进行AlltoAll通信和Permute操作后，结果作为Matmul计算的左矩阵输入。数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1fn_x2`，维度只能为2D，shape为\(BS, H\)，数据格式支持$ND$，**不支持非连续Tensor**。
- **x2**（`Tensor`）：**必选参数**，表示融合算子的右矩阵输入，也是Matmul计算的右矩阵，对应公式中的x2。数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1fn_x2`，维度只能为2D，shape为\(H\*rankSize, N\)，数据格式支持$ND$，支持转置非连续Tensor。
- **hcom**（`str`）：**必选参数**，Host侧标识列组的字符串，即通信域名称，通过`get_hccl_comm_name`接口获取。
- **world\_size**（`int`）：**必选参数**，通信域内的rank总数，对应公式中的rankSize，取值支持2、4、8、16，CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。
- <strong>*</strong>：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **all2all\_out\_flag**（`bool`）：**可选参数**，表示是否输出AlltoAll和Permute后的结果，默认为True。
- **bias**（`Tensor`）：**可选参数**，矩阵乘运算后累加的偏置，对应公式中的bias。数据类型支持`torch.float32`，维度只能为1D，shape为\(N\)，数据格式支持$ND$。
- **x1\_scale**（`Tensor`）：**可选参数**，表示左矩阵的量化系数，对应公式中的x1Scale。K-C动态量化场景下，`x1`的量化系数会自动计算，所以不支持输入实际的`x1_scale`，输入也不生效，传默认值即可。mx量化场景下，数据类型支持`torch_npu.float8_e8m0`，维度为3D，shape为\(BS, ceil\(H/64\), 2\)，数据格式支持$ND$。mx量化场景要求必须输入`x1_scale`，不允许为None。
- **x2\_scale**（`Tensor`）：**可选参数**，表示右矩阵的量化系数，对应公式中的x2Scale。K-C动态量化场景下，数据类型支持`torch.float32`，维度为1D，shape为\(N\)，数据格式支持$ND$。mx量化场景下，数据类型支持`torch_npu.float8_e8m0`，维度为3D，shape为\(BS, ceil\(H\*rankSize/64\), 2\)，数据格式支持$ND$。
- **common\_scale**（`Tensor`）：**可选参数**，表示低比特通信的量化系数。预留参数，当前版本传默认值即可。
- **x1\_offset**（`Tensor`）：**可选参数**，表示左矩阵的量化偏置。预留参数，当前版本传默认值即可。
- **x2\_offset**（`Tensor`）：**可选参数**，表示右矩阵的量化偏置。预留参数，当前版本传默认值即可。
- **x1\_quant\_mode**（`int`）：**可选参数**，表示左矩阵的量化方式。当前版本支持配置为6和7，分别表示mx量化和pertoken动态量化。
- **x2\_quant\_mode**（`int`）：**可选参数**，表示右矩阵的量化方式。当前版本支持配置为2和6，分别表示perchannel量化和mx量化。
- **common\_quant\_mode**（`int`）：**可选参数**，表示低比特通信的量化方式。预留参数，当前版本传默认值即可。
- **group\_sizes**（`List[int]`）：**可选参数**，用于Matmul计算三个方向上的量化分组大小。mx量化场景下需要配置，K-C量化场景下使用默认值即可，配置也不生效。
  - 非None时，仅支持三维列表，形如\[group\_m, group\_n, group\_k\]，分别表示在m、n、k维度上的量化分组情况。以group\_m为例，表示在m维度上group\_m个数对应一个量化参数。
  - 当\[group\_m, group\_n, group\_k\]中有1个或多个为0时，接口会根据`x1`、`x2`、`x1_scale`、`x2_scale`输入shape重新设置该值。计算原理：假设group\_m=0，表示m方向量化分组值由接口推断，推断公式为group\_m=m/scale\_m（保证m能被scale\_m整除），m与`x1` shape中的m一致，scale\_m与pertoken\_scale shape中的m一致，n和k方向同理。
  - **mx量化场景下需要配置该值为\[1, 1, 32\]**，或者保证在能够推导的情况下，确保推导后的值为\[1, 1, 32\]。

- **all2all\_axes**（`List[int]`）：**可选参数**，AlltoAll和Permute数据交换的方向，支持为空或者\[-2, -1\]，表示将输入`x1`由\(BS, H\)转为\(BS/rankSize, rankSize\*H\)。
- **comm\_quant\_dtype**（`int`）：**可选参数**，表示低比特通信的量化数据类型。预留参数，当前版本传默认值即可。
- **x1\_quant\_dtype**（`int`）：**可选参数**，表示量化Matmul左矩阵的量化类型，AlltoAll通信与Permute操作后结果，按照该参数配置量化后作为MatMul计算的左矩阵输入。该参数目前只在K-C动态量化场景下生效，支持取值为23（`torch.float8_e5m2`）或者24（`torch.float8_e4m3fn`）。mx量化场景下该参数配置不生效，使用默认值即可。
- **x1\_dtype**（`int`）：**可选参数**，表示输入的左矩阵的数据类型，对于PyTorch原生不支持的数据类型（如`torch_npu.float4_e2m1fn_x2`）需要通过本参数配置数据类型。
- **x2\_dtype**（`int`）：**可选参数**，表示输入的右矩阵的数据类型，对于PyTorch原生不支持的数据类型（如`torch_npu.float4_e2m1fn_x2`）需要通过本参数配置数据类型。
- **x1\_scale\_dtype**（`int`）：**可选参数**，表示输入左矩阵量化系数的数据类型，对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0`）需要指定该参数取值。
- **x2\_scale\_dtype**（`int`）：**可选参数**，表示输入右矩阵量化系数的数据类型，对于PyTorch原生不支持的数据类型（如`torch_npu.float8_e8m0`）需要指定该参数取值。
- **output\_scale\_dtype**（`int`）：**可选参数**，表示输出量化系数的数据类型。预留参数，当前版本传默认值即可。
- **comm\_scale\_dtype**（`int`）：**可选参数**，表示低比特通信量化系数的数据类型。预留参数，当前版本传默认值即可。
- **y\_dtype**（`int`）：**可选参数**，表示输出数据类型，支持取值为5（`torch.float16`）、6（`torch.float32`）或者15（`torch.bfloat16`）。
- **comm\_mode**（`str`）：**可选参数**，表示通信引擎模式，默认值为`None`。取值支持`None`、`ai_cpu`、`ccu`，传入`None`时使用`ai_cpu`通信。

## 返回值说明

- **y**（`Tensor`）：表示最终的计算结果，公式中的output，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，维度为2D，shape为\(BS/rankSize, N\)，数据格式支持$ND$，不支持非连续的Tensor。当`y_dtype`传入且不为undefined时，根据`y_dtype`决定数据类型；当`y_dtype`为默认值None或者为undefined时，数据类型默认为`torch.float32`，表示高精度。
- **all2all\_out**（`Tensor`）：表示AlltoAll和Permute后的结果，公式中的permutedOut。当`all2all_out_flag`为True时输出实际Tensor，数据类型与计算输入`x1`的类型一致，维度为2D，shape为\(BS/rankSize, H\*rankSize\)，数据格式支持$ND$，不支持非连续的Tensor；当`all2all_out_flag`为False时输出None。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输入与输出参数均不支持空Tensor。
- 通信域名称`hcom`不支持传入空字符串，长度取值范围为\[1, 127\]。
- 参数里Shape使用的变量如下：
  - BS：输入左矩阵的第一维度大小，表示输入序列sequence的条数，取值范围为\[2, 2147483647\]，必须整除rankSize。
  - H：输入左矩阵的第二维度大小，表示隐藏层维度，取值范围受NPU卡数限制，H\*rankSize取值范围为\[1, 65535\]。
  - H\*rankSize：输入右矩阵第一维度大小，表示输入左矩阵经过AlltoAll通信后隐藏层维度大小，取值范围为\[1, 65535\]。
  - N：输入右矩阵第二维度大小，表示输出序列sequence的长度，取值范围为\[1, 2147483647\]。
- `x1_quant_mode`、`x2_quant_mode`、`common_quant_mode`的枚举值与量化模式关系如下：
  - 0：不量化
  - 1：pertensor
  - 2：perchannel
  - 3：pertoken
  - 4：pergroup
  - 5：perblock
  - 6：mx量化
  - 7：pertoken动态量化

- 当前`x1_quant_mode`和`x2_quant_mode`的组合仅支持\[7, 2\]和\[6, 6\]，分别表示K-C动态量化和mx量化。
- mx量化场景下，`x2`必须转置（即需要使用.t\(\)方法），转置后的shape需满足\(H\*rankSize, N\)，且H必须整除64。
- 当前两种量化场景下`x2_scale`都必须传入，不允许为None。
- `x1_dtype`和`x2_dtype`在mx量化模式下，当数据类型为`torch_npu.float4_e2m1fn_x2`时，需要设置为296，其它场景传默认值即可。
- `x1_scale_dtype`和`x2_scale_dtype`在mx量化模式下，当数据类型为`torch_npu.float8_e8m0`时，需要设置为293，其它场景传默认值即可。
- 各量化模式输入输出数据类型详细约束如下表

    **表 1**  K-C动态量化数据类型约束

    | x1 | x2 | bias | x1_scale | x2_scale | output | x1_quant_mode | x2_quant_mode |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.float8_e4m3fn` | `torch.float32` | - | `torch.float32` | `torch.float16` | 7 | 2 |
    | `torch.float16` | `torch.float8_e4m3fn` | `torch.float32` | - | `torch.float32` | `torch.bfloat16` | 7 | 2 |
    | `torch.float16` | `torch.float8_e4m3fn` | `torch.float32` | - | `torch.float32` | `torch.float32` | 7 | 2 |
    | `torch.float16` | `torch.float8_e5m2` | `torch.float32` | - | `torch.float32` | `torch.float16` | 7 | 2 |
    | `torch.float16` | `torch.float8_e5m2` | `torch.float32` | - | `torch.float32` | `torch.bfloat16` | 7 | 2 |
    | `torch.float16` | `torch.float8_e5m2` | `torch.float32` | - | `torch.float32` | `torch.float32` | 7 | 2 |
    | `torch.bfloat16` | `torch.float8_e4m3fn` | `torch.float32` | - | `torch.float32` | `torch.float16` | 7 | 2 |
    | `torch.bfloat16` | `torch.float8_e4m3fn` | `torch.float32` | - | `torch.float32` | `torch.bfloat16` | 7 | 2 |
    | `torch.bfloat16` | `torch.float8_e4m3fn` | `torch.float32` | - | `torch.float32` | `torch.float32` | 7 | 2 |
    | `torch.bfloat16` | `torch.float8_e5m2` | `torch.float32` | - | `torch.float32` | `torch.float16` | 7 | 2 |
    | `torch.bfloat16` | `torch.float8_e5m2` | `torch.float32` | - | `torch.float32` | `torch.bfloat16` | 7 | 2 |
    | `torch.bfloat16` | `torch.float8_e5m2` | `torch.float32` | - | `torch.float32` | `torch.float32` | 7 | 2 |

    **表 2**  mx量化数据类型约束

    | x1 | x2 | bias | x1_scale | x2_scale | output | x1_quant_mode | x2_quant_mode |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float16` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.bfloat16` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float32` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float16` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.bfloat16` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float32` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float16` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.bfloat16` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float32` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float16` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.bfloat16` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float32` | 6 | 6 |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float16` | 6 | 6 |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.bfloat16` | 6 | 6 |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float32` | `torch_npu.float8_e8m0` | `torch_npu.float8_e8m0` | `torch.float32` | 6 | 6 |

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    from ml_dtypes import float8_e4m3fn, float8_e5m2

    def run_all_to_all_quant_matmul(rank, world_size, master_ip, master_port, x1_shape, x2_shape, bias_shape, x2_scale_shape):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        x1_tensor = torch.randn(x1_shape, dtype=torch.float16).npu()
        x2_np = np.random.uniform(-1, 1, x2_shape).astype(float8_e4m3fn)
        x2_tensor = torch.from_numpy(x2_np.astype(np.float32)).to(torch.float8_e4m3fn).npu()
        bias_tensor = torch.randn(bias_shape, dtype=torch.float32).npu()
        x2_scale_tensor = torch.randn(x2_scale_shape, dtype=torch.float32).npu()
        output, alltoallout = torch_npu.npu_all_to_all_quant_matmul(
            x1_tensor,
            x2_tensor,
            hcom_info,
            world_size,
            bias=bias_tensor,
            x2_scale=x2_scale_tensor)
        print("output: ", output)
        print("alltoallout: ", alltoallout)
    if __name__ == "__main__":
        worksize = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [1024, 256]
        x2_shape =  [512, 3072]
        bias_shape = [3072]
        x2_scale_shape = [3072]
        mp.spawn(
            run_all_to_all_quant_matmul,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape, bias_shape, x2_scale_shape),
            nprocs=worksize,
        )
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    import pdb
    from ml_dtypes import int4, bfloat16, float8_e4m3fn, float8_e5m2, float8_e8m0fnu, float4_e2m1fn
    config = CompilerConfig()
    config.debug.graph_dump.type = 'pbtxt'
    npu_backend = tng.get_npu_backend(compiler_config=None)
    DTYPE_MAP = {
            "fp16": np.float16,
            "bf16": bfloat16,
            "fp32": np.float32,
            "fp64": np.float64,
            "int64": np.int64,
            "uint64": np.int64,
            "int32": np.int32,
            "uint32": np.int32,
            "int16": np.int16,
            "uint16": np.int16,
            "int8": np.int8,
            "uint8": np.uint8,
            "int4": int4,
            "float8_e4m3fn": float8_e4m3fn,
            "float8_e5m2": float8_e5m2,
            "hif8": np.uint8,
            "fp8_e8m0": float8_e8m0fnu,
            "fp4_e2m1": float4_e2m1fn,
            "fp4_e1m2": float4_e2m1fn,
        }
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, hcom, world_size, bias, x2_scale):
            return torch_npu.npu_all_to_all_quant_matmul(x1=x1, x2=x2, hcom=hcom, world_size=world_size,
                                                         bias=bias, x1_scale=None, x2_scale=x2_scale,
                                                         common_scale=None,
                                                         x1_offset=None,
                                                         x2_offset=None,
                                                         x1_quant_mode=7,
                                                         x2_quant_mode=2,
                                                         x1_quant_dtype=23,
                                                         common_quant_mode=0,
                                                         group_sizes=[0,0,0],
                                                         all2all_axes=None,
                                                         comm_quant_dtype=28,
                                                         x1_dtype=None,
                                                         x2_dtype=None,
                                                         x1_scale_dtype=None,
                                                         x2_scale_dtype=None,
                                                         output_scale_dtype=None,
                                                         comm_scale_dtype=None,
                                                         y_dtype=6,
                                                         all2all_out_flag=True)
    def create_tensor(tensor_flag, shape=(), dtype="", transpose=False):
        device = torch.device("npu")
        if tensor_flag:
            array = np.random.uniform(-1, 1, shape).astype(DTYPE_MAP[dtype])
            if array.dtype in ["float8_e4m3fn", "float8_e5m2", "bfloat16"]:
                tensor = torch.reshape(torch.from_numpy(array.astype(np.float32)), shape).to(getattr(torch, str(array.dtype))).to(device)
            elif array.dtype == "float8_e8m0fnu":
                tensor = torch.reshape(torch.from_numpy(array.view(np.uint8)), shape).to(device)
            else:
                tensor = torch.reshape(torch.from_numpy(array), shape).to(device)
            if transpose:
                tensor = torch.transpose(tensor, 0, 1)
        else:
            tensor = None
        return tensor
    def run_all_to_all_quant_matmul_graph(rank, world_size, master_ip, master_port, x1_shape, x2_shape, bias_shape, x2_scale_shape):
        # get hcom_info
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        # generate input tensor
        x1_tensor = torch.randn(x1_shape, dtype=torch.float16).npu()
        x2_tensor = create_tensor(1, [512, 3072], "float8_e4m3fn")
        bias_tensor = torch.randn(bias_shape, dtype=torch.float32).npu()
        x2_scale_tensor = torch.randn(x2_scale_shape, dtype=torch.float32).npu()
        cpu_model = Model()
        model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
        output = model(
            x1=x1_tensor,
            x2=x2_tensor,
            hcom=hcom_info,
            world_size=world_size,
            bias = bias_tensor,
            x2_scale = x2_scale_tensor,
        )
        print("print output: ", output)
    if __name__ == "__main__":
        world_size = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [1024, 256]
        x2_shape = [512, 3072]
        bias_shape = [3072]
        x2_scale_shape = [3072]
        mp.spawn(
            run_all_to_all_quant_matmul_graph,
            args=(world_size, master_ip, master_port, x1_shape, x2_shape, bias_shape, x2_scale_shape),
            nprocs=world_size,
        )
    ```
