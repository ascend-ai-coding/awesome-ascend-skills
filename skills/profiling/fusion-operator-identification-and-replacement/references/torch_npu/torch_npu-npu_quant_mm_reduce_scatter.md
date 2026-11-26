# torch_npu.npu_quant_mm_reduce_scatter

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

- API功能：融合MatMul（矩阵乘）与ReduceScatter（归约散射）集合通信计算，对入参x1（self）、x2进行矩阵乘（可选加bias、反量化）计算后，按卡数对M轴切分并进行ReduceScatter通信，常用于MoE等分布式场景（MC2）。<br>
  该接口是`torch_npu.npu_mm_reduce_scatter_base`接口的功能扩展，在支持x1和x2输入类型为`torch.float16`/`torch.bfloat16`的基础上：
    <!-- npu="950" id4 -->
    - <term>Ascend 950DT系列产品</term>新增了对低精度数据类型`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`的支持（支持pertensor、perblock、mx量化方式，其中mx量化支持mxfp8和mxfp4场景）；
    <!-- end id4 -->
    <!-- npu="A3,910b" id23 -->
    - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>新增了对`torch.int8`的支持（支持pertoken/perchannel量化方式）。
    <!-- end id23 -->

- 计算公式：
  - 场景1：x1和x2数据类型为`torch.float16`/`torch.bfloat16`时，对入参x1、x2、bias进行matmul计算后，进行ReduceScatter通信。

    $$
    output=ReduceScatter(x1@x2 + bias_{optional})
    $$

  - 场景2：x1和x2数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`的pertensor场景，或x1和x2数据类型为`torch.int8`的perchannel、pertoken场景，且不输出amax时，入参x1、x2进行matmul计算和dequant计算后，进行ReduceScatter通信。

    $$
    output=ReduceScatter((x1Scale*x2Scale)*(x1@x2) + bias_{optional})
    $$

  - 场景3：x1和x2数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`的perblock场景，且不输出amax时，当x1的shape为(m, k)、x2的shape为(k, n)，x1Scale的shape为(ceildiv(m, 128), ceildiv(k, 128))、x2Scale的shape为(ceildiv(k, 128), ceildiv(n, 128))时，入参x1、x2进行matmul计算和dequant计算后，再进行ReduceScatter通信。

    $$
    output=ReduceScatter(\sum_{0}^{ceildiv} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
    $$

    其中

    $$
    ceildiv = \left \lceil \frac{k}{blockSize=128} \right \rceil
    $$

  - 场景4：x1和x2数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.float4_e2m1fn_x2`的mx量化场景，且不输出amax时，当x1的shape为(m, k)、x1Scale的shape为(m, ceildiv(k, 64), 2)时，入参x1、x2进行matmul计算和dequant计算后，再进行ReduceScatter通信。x2支持转置/不转置场景：当x2转置时，x2的shape为(n, k)、x2Scale的shape为(n, ceildiv(k, 64), 2)；当x2不转置时，x2的shape为(k, n)、x2Scale的shape为(ceildiv(k, 64), n, 2)。

    $$
    output=ReduceScatter(\sum_{0}^{ceildiv} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
    $$

    其中

    $$
    ceildiv = \left \lceil \frac{k}{blockSize=32} \right \rceil
    $$

> [!NOTE]
>
> `comm_mode`未传入时按平台取默认值：
> <!-- npu="A3,910b" id24 -->
> <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>默认值为"aiv"。
> <!-- end id24 -->
> <!-- npu="950" id25 -->
> <term>Ascend 950DT系列产品</term>默认值为"ai_cpu"。
> <!-- end id25 -->

## 函数原型

```python
torch_npu.npu_quant_mm_reduce_scatter(self, x2, hcom, world_size, *, reduce_op='sum', bias=None,
                                      x1_scale=None, x2_scale=None, quant_scale=None, block_size=0,
                                      comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None,
                                      x1_dtype=None, x2_dtype=None, x1_scale_dtype=None,
                                      x2_scale_dtype=None, comm_mode=None) -> (Tensor, Tensor)
```

## 参数说明

- **self**（`Tensor`）：必选参数，MM左矩阵，即计算公式中的$x1$。当前版本仅支持两维输入，shape为[m, k]，且仅支持不转置场景，m须为卡数（world\_size）的整数倍，k轴取值范围为[256, 65535)。数据格式支持ND，不支持非连续Tensor。数据类型支持如下：

  <!-- npu="A3,910b" id5 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - <term>Ascend 950DT系列产品</term>：支持`torch.float16`、`torch.bfloat16`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`。
  <!-- end id6 -->

- **x2**（`Tensor`）：必选参数，MM右矩阵，即计算公式中的$x2$。当前版本仅支持两维输入，支持转置/不转置场景，具体shape要求详见约束说明，仅支持转置带来的非连续Tensor。数据类型支持如下：

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`，数据格式支持ND、FRACTAL_NZ。
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950DT系列产品</term>：支持`torch.float16`、`torch.bfloat16`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`，数据格式仅支持ND。
  <!-- end id8 -->

- **hcom**（`str`）：必选参数，通信域名称。通过get\_hccl\_comm\_name接口获取。
- **world\_size**（`int`）：必选参数，卡数（rank\_size）。

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：支持2、4、8卡。
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950DT系列产品</term>：支持2、4、8、16、32、64卡。
  <!-- end id10 -->

- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **reduce\_op**（`str`）：可选参数，reduce操作类型。当前版本仅支持"sum"。默认值为'sum'。
- **bias**（`Tensor`）：可选参数，即计算公式中的$bias$。当前版本仅支持一维输入，shape为(n,)，支持传入None。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：当x1和x2数据类型为`torch.int8`时，bias数据类型可以是`torch.float16`、`torch.bfloat16`、`torch.float32`；当x1和x2数据类型为`torch.float16`时，bias数据类型必须为`torch.float16`、`torch.float32`；当x1和x2数据类型为`torch.bfloat16`时，bias数据类型必须为`torch.bfloat16`、`torch.float32`。
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950DT系列产品</term>：如果x1的数据类型是`torch.float16`、`torch.bfloat16`，则bias的数据类型必须为`torch.float16`、`torch.bfloat16`。如果x1的数据类型是`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`时，在pertensor和mx量化场景下，bias的数据类型必须为`torch.float32`。在perblock场景下，仅支持输入为None。
  <!-- end id12 -->

- **x1\_scale**（`Tensor`）：可选参数，mm左矩阵反量化参数，支持传入None。当x1和x2数据类型为`torch.float16`/`torch.bfloat16`时，仅支持传入None。

  <!-- npu="A3,910b" id13 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：pertoken场景shape为(m, 1)，数据类型支持`torch.float32`。
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950DT系列产品</term>：pertensor场景shape为[1]，perblock场景shape为[ceildiv(m, 128), ceildiv(k, 128)]，数据类型支持`torch.float32`；mx量化场景（MXFP8和MXFP4）数据类型为`torch.float8_e8m0fnu`，shape为(m, ceilDiv(k, 64), 2)。
  <!-- end id14 -->

- **x2\_scale**（`Tensor`）：可选参数，mm右矩阵反量化参数，支持传入None。当x1和x2数据类型为`torch.float16`/`torch.bfloat16`时，仅支持传入None。

  <!-- npu="A3,910b" id15 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：perchannel场景shape为(1, n)，数据类型支持`torch.float32`、`torch.int64`（仅在output数据类型为`torch.float16`场景支持）。
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950DT系列产品</term>：pertensor场景shape为[1]，perblock场景shape为[ceildiv(k, 128), ceildiv(n, 128)]，数据类型支持`torch.float32`；mx量化场景数据类型为`torch.float8_e8m0fnu`，shape跟随`x2`转置/不转置，具体shape要求详见约束说明，仅支持转置带来的非连续Tensor。
  <!-- end id16 -->

- **quant\_scale**（`Tensor`）：可选参数，输出矩阵量化scale。当前仅支持传入None。默认值为None。
- **block\_size**（`int`）：可选参数，用于表示mm输出矩阵在M轴方向和N轴方向上可以用于对应方向上的多少个数的量化。由blockSizeM、blockSizeN、blockSizeK三个值拼接而成，每个值占16位，计算公式为blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32，mm输出矩阵不涉及K轴，blockSizeK固定为0，当前版本只支持blockSizeM=blockSizeN=0。默认值为0。
- **comm\_turn**（`int`）：可选参数，通信数据切分数，即总数据量/单次通信量。当前版本仅支持输入0。默认值为0。
- **group\_sizes**（`List[int]`）：可选参数，用于表示反量化中x1_scale/x2_scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。长度为3，依次为[groupSizeM, groupSizeN, groupSizeK]，每个值取值范围[0, 65535]，内部拼接公式为groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。传None或全0表示使用默认值。默认值为None。

  <!-- npu="950" id17 -->
  - <term>Ascend 950DT系列产品</term>：仅当x1Scale和x2Scale输入都是2维及以上数据时，groupSize取值有效，其他场景需传入0。当groupSizeM/groupSizeN/groupSizeK中有1个或多个为0时，会根据x1/x2/x1Scale/x2Scale输入shape重新推导：groupSizeM = m / scaleM（需保证m能被scaleM整除），groupSizeK = k / scaleK，groupSizeN = n / scaleN。一般情况下，当x1Scale、x2Scale输入都是2维且数据类型都为`torch.float32`时，[groupSizeM, groupSizeN, groupSizeK]取值组合会推导为[128, 128, 128]；当x1Scale、x2Scale输入都是3维且数据类型都为`torch.float8_e8m0fnu`时，会推导为[1, 1, 32]。
  <!-- end id17 -->
  <!-- npu="A3,910b" id18 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：当前版本仅支持输入0。
  <!-- end id18 -->

- **amax\_output**（`bool`）：可选参数，是否输出MM计算的最大值结果。为True时返回MM计算的最大值结果amax。默认值为False。
- **y\_dtype**（`int`）：可选参数，输出数据类型（ScalarType枚举值）。当输入为`torch.float16`或`torch.bfloat16`时，输出应与输入数据类型保持一致（可不传）；输入为其他数据类型时必须传入。支持取值`torch.float16`、`torch.bfloat16`、`torch.float32`。默认值为None。
- **x1\_dtype**（`int`）：可选参数，x1的数据类型（ScalarType枚举值），用于将x1的存储类型解释为指定低精度类型。mx量化场景（MXFP4）下必须传入，取值须为`torch_npu.float4_e2m1fn_x2`。默认值为None。
- **x2\_dtype**（`int`）：可选参数，x2的数据类型（ScalarType枚举值），用于将x2的存储类型解释为指定低精度类型。mx量化场景（MXFP4）下必须传入，取值须为`torch_npu.float4_e2m1fn_x2`。默认值为None。
- **x1\_scale\_dtype**（`int`）：可选参数，x1\_scale的数据类型（ScalarType枚举值）。mx量化场景（MXFP4）下必须传入，取值须为`torch.float8_e8m0fnu`。默认值为None。
- **x2\_scale\_dtype**（`int`）：可选参数，x2\_scale的数据类型（ScalarType枚举值）。mx量化场景（MXFP4）下必须传入，取值须为`torch.float8_e8m0fnu`。默认值为None。
- **comm\_mode**（`str`）：可选参数，通信模式。

  <!-- npu="A3,910b" id19 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：当前版本仅支持输入"aiv"，默认值为"aiv"。
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950DT系列产品</term>：当前版本支持输入"ai\_cpu"或"ccu"。默认值为"ai\_cpu"）。
  <!-- end id20 -->

## 返回值说明

- **output**（`Tensor`）：ReduceScatter通信与MatMul计算的结果，即计算公式中的$output$。shape为(m / world\_size, n)，其中world\_size为卡数。当x1类型为`torch.float16`、`torch.bfloat16`时，输出类型与x1保持一致；当x1类型为`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`时，输出数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。数据格式支持ND。仅当输出类型为`torch.float16`、`torch.bfloat16`时支持空Tensor。
- **amax**（`Tensor`）：MM计算的最大值结果，即公式中的$amaxOut$。`amax_output`为True时返回shape为[1]的`torch.float32`Tensor；`amax_output`为False时返回空Tensor。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 确定性计算：默认采用确定性计算实现。
- 输入约束（所有产品）：
  - 只支持x2矩阵转置/不转置，x1矩阵仅支持不转置场景。
  - 输入x1为2维，其shape为(m, k)，m须为卡数rank\_size的整数倍，k轴取值范围为[256, 65535)。
  - 输入x2必须是2维，其shape为(k, n)，轴满足mm算子入参要求，k轴相等。
  - bias为1维，shape为(n,)。
  - 输出为2维，其shape为(m / rank\_size, n)，rank\_size为卡数。

<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>约束：
  - 通信约束：当前版本仅支持输入comm\_mode为"ai\_cpu"或"ccu"，支持CCU通信和AICPU通信，CCU仅支持单机UB域内互联，AICPU可支持跨机UB域内互联。
  - 当x1、x2的数据类型为`torch.float16`/`torch.bfloat16`时，x1/x2支持空tensor场景，m和n可以为空，k不可为空；当x1、x2的数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`时，不支持空tensor。
  - 当x1、x2的数据类型为`torch.float16`/`torch.bfloat16`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`时，x1和x2的数据类型需要保持一致；当x1、x2的数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`时，x1和x2的数据类型可以为其中任意一种。
  - 输入x2支持转置和不转置场景，但输入约束里要求其shape为(k, n)，这是为了满足mm算子入参要求，即左矩阵x1的shape应该是(输出行维度，规约轴)，右矩阵x2的shape应该是(规约轴，输出列维度)，其中输出行维度为m，规约轴为k，输出列维度为n。支持转置场景，是指x2可以通过(n, k)的shape构造连续Tensor，在调用本接口前执行x2.t()操作，以此得到视图上为(k, n)的x2，满足mm算子入参要求，此时x2非连续，x2也仅支持该场景下的非连续Tensor。不转置场景时，x2的shape为(k, n)，此时x2连续，可以直接作为入参输入。
  - 输入x2\_scale在mx量化场景时为3维，其shape跟随x2转置/不转置，x2转置时x2\_scale的shape为(n, ceildiv(k, 64), 2)，x2不转置时x2\_scale的shape为(ceildiv(k, 64), n, 2)。
  - mx量化场景下，x1和x2输入为`torch_npu.float4_e2m1fn_x2`（MXFP4量化）时，k必须是偶数，当x2不转置时，n也必须是偶数。
  - 支持2、4、8、16、32、64卡。
  - ReduceScatter集合通信数据总量不能超过16 \* 256MB，集合通信数据总量计算方式为：m \* n \* sizeof(output\_dtype)。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。
<!-- end id21 -->
<!-- npu="A3,910b" id22 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>约束：
  - 仅支持comm\_mode为"aiv"，且通信缓冲区大于等于200MB。
  - 不支持空tensor。
  - x1和x2的数据类型需要保持一致。
  - 支持2、4、8卡。
<!-- end id22 -->
- `world_size`取值必须在[2, 4, 8, 16, 32, 64]范围内。
- 输入self和x2的K轴必须相等，且self的M轴必须能被world\_size整除。
- `group_sizes`必须传长度为3的列表，每个元素取值范围[0, 65535]。
- mx量化场景（MXFP4）下，`x1_dtype`、`x2_dtype`、`x1_scale_dtype`、`x2_scale_dtype`为必传参数，且x1\_dtype与x2\_dtype必须为`torch_npu.float4_e2m1fn_x2`，x1\_scale\_dtype与x2\_scale\_dtype必须为`torch.float8_e8m0fnu`。

## 调用示例

### 场景1：`x1，x2`均为`torch.float16`或`torch.bfloat16`

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_npu_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    self_t = torch.randn(x1_shape, dtype=dtype).npu()
    x2 = torch.randn(x2_shape, dtype=dtype).npu()
    output, amax = torch_npu.npu_quant_mm_reduce_scatter(self_t, x2, hcomm_info, world_size, bias=None, x1_scale=None, x2_scale=None, amax_output=amax_output)
    print(output.shape, output.dtype)
    # amax_output为False时返回空Tensor
    print(amax)  # torch.Size([0])

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16
    amax_output = False  # amax_output为False时返回空Tensor
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)
    amax_output = True  # amax_output为True时返回[1]的float32最大值结果
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)
```

输出如下所示

```text
torch.Size([64, 64]) torch.float16
None
torch.Size([64, 64]) torch.float16
None

torch.Size([64, 64]) torch.float16
torch.Size([64, 64]) torch.float16
tensor([0.], device='npu:0')
tensor([0.], device='npu:1')
```

### 场景2：`x1，x2`数据类型为`torch.int8`的perchannel、pertoken场景（不输出amax）

> [!NOTE]
>
> 场景2中`torch.int8`的perchannel、pertoken场景（<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>）与此类似：x1、x2数据类型为`torch.int8`，pertoken时`x1_scale`的shape为(m, 1)、`x2_scale`不传；perchannel时`x2_scale`的shape为(1, n)、`x1_scale`不传，scale数据类型支持`torch.float32`（perchannel的`x2_scale`还支持`torch.int64`）。

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_npu_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    self_t = torch.randint(-128, 127, x1_shape, dtype=dtype).npu()
    x2 = torch.randint(-128, 127, x2_shape, dtype=dtype).npu()
    # pertoken场景：x1_scale的shape为(m, 1)；perchannel场景：x2_scale的shape为(1, n)
    x1_scale = torch.rand((x1_shape[0], 1), dtype=torch.float32).npu()
    x2_scale = torch.rand((1, x2_shape[1]), dtype=torch.float32).npu()
    output, amax = torch_npu.npu_quant_mm_reduce_scatter(self_t, x2, hcomm_info, world_size,
                                                         bias=None, x1_scale=x1_scale, x2_scale=x2_scale,
                                                         amax_output=amax_output, y_dtype=torch.float16)
    print(output.shape, output.dtype) # (torch.Size([64, 64]), torch.float16)


if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50002'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.int8
    amax_output = False  # 场景2不输出amax
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)
```

输出如下所示

```text
torch.Size([64, 64]) torch.float16
torch.Size([64, 64]) torch.float16
```

### 场景3：`x1，x2`数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`的perblock场景（不输出amax）

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import math
def run_npu_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    m, k = x1_shape
    n = x2_shape[1]
    self_t = torch.randn(x1_shape).to(dtype).npu()
    x2 = torch.randn(x2_shape).to(dtype).npu()
    # perblock场景：blockSize为128，scale的shape分别为(ceildiv(m, 128), ceildiv(k, 128))、(ceildiv(k, 128), ceildiv(n, 128))
    x1_scale = torch.rand((math.ceil(m / 128), math.ceil(k / 128)), dtype=torch.float32).npu()
    x2_scale = torch.rand((math.ceil(k / 128), math.ceil(n / 128)), dtype=torch.float32).npu()
    output, amax = torch_npu.npu_quant_mm_reduce_scatter(self_t, x2, hcomm_info, world_size, bias=None,
                                                          x1_scale=x1_scale, x2_scale=x2_scale,
                                                          amax_output=amax_output, y_dtype=torch.float16)
    print(output.shape, output.dtype) # (torch.Size([64, 64]), torch.float16)
    # amax_output为False时返回空Tensor
    print(amax)  # torch.Size([0])

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 128]
    dtype = torch.float8_e4m3fn
    amax_output = False  # 场景3不输出amax
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)

```

输出如下所示

```text
torch.Size([64, 128]) torch.float16
None
torch.Size([64, 128]) torch.float16
None
```

### 场景4：`x1，x2`数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.float4_e2m1fn_x2`的mx量化场景（不输出amax，以MXFP8为例）

> [!NOTE]
>
> 场景4的MXFP4场景（x1、x2数据类型为`torch_npu.float4_e2m1fn_x2`）还需额外传入`x1_dtype`、`x2_dtype`（取值为`torch_npu.float4_e2m1fn_x2`）以及`x1_scale_dtype`、`x2_scale_dtype`（取值为`torch.float8_e8m0fnu`），且k须为偶数。

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import math
def run_npu_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    m, k = x1_shape   # x1_shape为(m, k)
    n = x2_shape[1]   # mx量化仅支持x2、x2_scale转置输入，x2转置后shape为(k, n)
    self_t = torch.randn(x1_shape).to(dtype).npu()
    x2 = torch.randn(x2_shape).to(dtype).npu()
    # mx量化场景：scale数据类型为float8_e8m0fnu，shape分别为(m, ceildiv(k, 64), 2)、(ceildiv(k, 64), n, 2)
    x1_scale = torch.rand((m, math.ceil(k / 64), 2)).to(torch.float8_e8m0fnu).npu()
    x2_scale = torch.rand((math.ceil(k / 64), n, 2)).to(torch.float8_e8m0fnu).npu()
    output, amax = torch_npu.npu_quant_mm_reduce_scatter(self_t, x2, hcomm_info, world_size, bias=None,
                                                          x1_scale=x1_scale, x2_scale=x2_scale,
                                                          amax_output=amax_output, y_dtype=torch.float16)
    print(output.shape, output.dtype) # (torch.Size([64, 64]), torch.float16)
    # amax_output为False时返回空Tensor
    print(amax)  # torch.Size([0])

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 256]
    x2_shape = [256, 64] 
    dtype = torch.float8_e4m3fn
    amax_output = False  # 场景4不输出amax
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)
```

输出如下所示

```text
torch.Size([64, 64]) torch.float16
None
torch.Size([64, 64]) torch.float16
None
```
