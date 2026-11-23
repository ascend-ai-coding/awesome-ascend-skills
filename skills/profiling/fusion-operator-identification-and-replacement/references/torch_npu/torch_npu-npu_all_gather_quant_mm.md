# torch_npu.npu_all_gather_quant_mm

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

- API功能：在TP切分场景下，对输入的MM左矩阵x1执行AllGather集合通信后，与右矩阵x2进行MatMul计算，支持量化场景下的反量化计算，并可同时输出AllGather通信结果gatherOut与MatMul计算结果的最大值amaxOut。该接口是`torch_npu.npu_all_gather_base_mm`接口的功能扩展：
  <!-- npu="950" id4 -->
  - <term>Ascend 950DT系列产品</term> 新增了对低精度数据类型`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`的支持（支持pertensor、perblock、mx量化方式，其中mx量化支持mxfp8和mxfp4场景）；
  <!-- end id4 -->
  <!-- npu="A3,910b" id21 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>新增了对`torch.int8`的支持（支持pertoken/perchannel量化方式）。
  <!-- end id21 -->

- 计算公式：

  - 场景1：当x1和x2数据类型为`torch.float16`/`torch.bfloat16`时，对入参x1进行AllGather后，对x1、x2进行MatMul计算：

    $$
    output=AllGather(x1)@x2 + bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

  - 场景2：当x1和x2数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`的pertensor场景，或x1和x2数据类型为`torch.int8`/`torch.int4`的perchannel、pertoken场景，且不输出amaxOut时，对入参x1进行AllGather后，对x1、x2进行MatMul计算，然后进行dequant操作：

    $$
    output=(x1Scale*x2Scale)*(AllGather(x1)@x2 + bias)
    $$

    $$
    gatherOut=AllGather(x1)
    $$

  - 场景3：当x1和x2数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`的perblock场景，且不输出amaxOut时，当x1为(m, k)、x2为(k, n)，x1Scale为(ceilDiv(m, 128), ceilDiv(k, 128))、x2Scale为(ceilDiv(k, 128), ceilDiv(n, 128))时，对入参x1和x1Scale进行AllGather后，对x1、x2进行perblock量化MatMul计算，然后进行dequant操作：

    $$
    output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
    $$

    $$
    gatherOut=AllGather(x1)
    $$

  - 场景4：当x1和x2数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.float4_e2m1fn_x2`的mx量化场景，x1为(m, k)、x2非转置场景下为(k, n)，转置场景下为(n, k)，且x1Scale为(m, ceilDiv(k, 64), 2)、x2Scale非转置场景下为(ceilDiv(k, 64), n, 2)，转置场景下为(n, ceilDiv(k, 64), 2)，入参x1和x1Scale进行AllGather后，对x1、x2进行MatMul计算，然后进行dequant操作：

    $$
    output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
    $$

    $$
    gatherOut=AllGather(x1)
    $$

> [!NOTE]
>
> - 量化方式（pertensor、perchannel、pertoken、perblock、mx）由x1_scale、x2_scale的shape隐式确定，接口不提供显式的量化方式参数。
> <!-- npu="950" id22 -->
> - <term>Ascend 950DT系列产品</term>：支持pertensor、perblock、mx量化方式。
> <!-- end id22 -->
> <!-- npu="A3,910b" id23 -->
> - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：支持pertoken、perchannel量化方式。
> <!-- end id23 -->
> - 对于PyTorch原生不支持的数据类型（`torch.int4`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`等），需要通过x1_dtype、x2_dtype、x1_scale_dtype、x2_scale_dtype参数指定实际数据类型。

## 函数原型

​```
torch_npu.npu_all_gather_quant_mm(self, x2, hcom, world_size, *, bias=None, x1_scale=None, x2_scale=None, quant_scale=None, block_size=0, gather_index=0, gather_output=True, comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None, comm_mode=None) -> (Tensor, Tensor, Tensor)
​```

## 参数说明

- **self** (`Tensor`)：必选参数，MM左矩阵，即计算公式中的$x1$。shape为2维$(m, k)$，仅支持不转置场景，数据格式支持$ND$。

  <!-- npu="950" id5 -->
  - <term>Ascend 950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`。
  <!-- end id5 -->
  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch.int4`。
  <!-- end id6 -->

- **x2** (`Tensor`)：必选参数，MM右矩阵，即计算公式中的$x2$。shape为2维，转置时为$(n, k)$，非转置时为$(k, n)$，支持转置/不转置场景，仅转置场景支持非连续Tensor，数据格式支持$ND$。数据类型支持范围与`self`一致，且x1和x2的数据类型需保持一致（`torch.float8_e4m3fn`与`torch.float8_e5m2`可混用）。
- **hcom** (`str`)：必选参数，通信域名称。可通过`group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)`获取，其中`group`为`torch.distributed`的进程组。
- **world_size** (`int`)：必选参数，通信域内的rank总数，必须为2的幂。

  <!-- npu="950" id7 -->
  - <term>Ascend 950DT系列产品</term>：支持2、4、8、16、32、64卡。
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：支持2、4、8卡。
  <!-- end id8 -->

- <strong>*</strong>：位置参数与关键字参数的分隔符。其之前的参数为位置参数，需按顺序传入；其之后的参数为关键字参数，需通过键值对方式传入，未赋值时使用默认值。
- **bias** (`Tensor`)：可选参数，偏置，即计算公式中的$bias$。shape为1维$(n,)$，数据格式支持$ND$，默认值为`None`。

  <!-- npu="950" id9 -->
  - <term>Ascend 950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。当x1为`torch.float16`、`torch.bfloat16`时，bias的数据类型必须与x1一致；当x1为`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`时，pertensor和mx量化场景下bias的数据类型必须为`torch.float32`，perblock场景下仅支持传入`None`。
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：当前版本仅支持传入`None`。
  <!-- end id10 -->

- **x1_scale** (`Tensor`)：可选参数，MM左矩阵反量化参数，默认值为`None`。当x1和x2数据类型为`torch.float16`/`torch.bfloat16`时，仅支持传入`None`。

  <!-- npu="950" id11 -->
  - <term>Ascend 950DT系列产品</term>：pertensor场景shape为$[1]$，perblock场景shape为$(ceilDiv(m, 128), ceilDiv(k, 128))$，以上场景数据类型支持`torch.float32`；mx量化场景数据类型为`torch.float8_e8m0fnu`，shape为$(m, ceilDiv(k, 64), 2)$。
  <!-- end id11 -->
  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.float32`，pertoken场景shape为$(m, 1)$。
  <!-- end id12 -->

- **x2_scale** (`Tensor`)：可选参数，MM右矩阵反量化参数，默认值为`None`。当x1和x2数据类型为`torch.float16`/`torch.bfloat16`时，仅支持传入`None`。

  <!-- npu="950" id13 -->
  - <term>Ascend 950DT系列产品</term>：pertensor场景shape为$[1]$，perblock场景shape为$(ceilDiv(k, 128), ceilDiv(n, 128))$，以上场景数据类型支持`torch.float32`；mx量化场景数据类型为`torch.float8_e8m0fnu`，转置场景下shape为$(n, ceilDiv(k, 64), 2)$，非转置场景下shape为$(ceilDiv(k, 64), n, 2)$，仅支持转置/不转置场景。
  <!-- end id13 -->
  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.float32`、`torch.int64`（`torch.int64`仅在x1和x2数据类型为`torch.int8`或output数据类型为`torch.float16`场景支持），perchannel场景shape为$(1, n)$。
  <!-- end id14 -->

- **quant_scale** (`Tensor`)：可选参数，量化参数，默认值为`None`。当前版本仅支持传入`None`。
- **block_size** (`int`)：可选参数，用于表示MM输出矩阵在M轴方向和N轴方向上可用于对应方向上的多少个数的量化，默认值为`0`。block_size由blockSizeM、blockSizeN、blockSizeK三个值拼接而成，每个值占16位，计算公式为block_size = blockSizeK | blockSizeN << 16 | blockSizeM << 32，MM输出矩阵不涉及K轴，blockSizeK固定为0。当前版本仅支持blockSizeM=blockSizeN=0，即仅支持传入0。
- **gather_index** (`int`)：可选参数，标识gather目标，默认值为`0`。`0`表示目标为x1，`1`表示目标为x2。当前版本仅支持输入`0`。
- **gather_output** (`bool`)：可选参数，是否需要输出AllGather通信结果gatherOut，默认值为`True`。为`True`时输出gatherOut；为`False`时gatherOut返回空Tensor。
- **comm_turn** (`int`)：可选参数，通信数据切分数，即总数据量/单次通信量，默认值为`0`。当前版本仅支持输入`0`。
- **group_sizes** (`List[int]`)：可选参数，反量化分组大小，默认值为`None`。列表长度必须为3，元素依次为groupSizeM、groupSizeN、groupSizeK，每个元素取值范围为$[0, 65535]$，拼接公式为group_size = groupSizeK | groupSizeN << 16 | groupSizeM << 32。当group_sizes为`None`或空列表时，group_size为0。

  <!-- npu="A3,910b" id15 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：当前版本仅支持group_sizes为`None`（group_size为0）。
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950DT系列产品</term>：仅当x1_scale和x2_scale输入都是2维及以上数据时group_size取值有效，其他场景需传0。当groupSizeM、groupSizeN、groupSizeK中有1个或多个为0时，接口根据x1/x2/x1_scale/x2_scale的shape重新推导：groupSizeM = m / scaleM（需整除，m为x1的shape第一维，scaleM为x1Scale的shape第一维）、groupSizeK = k / scaleK、groupSizeN = n / scaleN。常见取值组合：x1_scale、x2_scale为2维`torch.float32`时推导为[128, 128, 128]，对应group_size值为549764202624；x1_scale、x2_scale为3维`torch_npu.float8_e8m0fnu`时推导为[1, 1, 32]，对应group_size值为4295032864。
  <!-- end id16 -->

- **amax_output** (`bool`)：可选参数，是否需要输出MatMul计算结果的最大值amaxOut，默认值为`False`。当前版本仅支持`False`，此时amax返回空Tensor。
- **y_dtype** (`int`)：可选参数，输出y的数据类型（例如：`torch.float16`），默认值为`None`。当x1为`torch.float16`/`torch.bfloat16`时，y的数据类型与x1保持一致，若指定y_dtype必须与x1一致；当x1为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`/`torch.int8`/`torch.int4`等低精度类型时，必须指定y_dtype，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
- **x1_dtype** (`int`)：可选参数，x1（self）实际参与计算的数据类型，默认值为`None`。当x1为PyTorch原生不支持的数据类型（如`torch.int4`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`）时需指定，且该数据类型的itemsize须与x1 Tensor的itemsize一致。
- **x2_dtype** (`int`)：可选参数，x2实际参与计算的数据类型，默认值为`None`，取值规则同x1_dtype。
- **x1_scale_dtype** (`int`)：可选参数，x1_scale的实际数据类型，默认值为`None`。当x1_dtype为`torch_npu.float4_e2m1fn_x2`时，必须为`torch.float8_e8m0fnu`。
- **x2_scale_dtype** (`int`)：可选参数，x2_scale的实际数据类型，默认值为`None`。当x2_dtype为`torch_npu.float4_e2m1fn_x2`时，必须为`torch.float8_e8m0fnu`。
- **comm_mode** (`str`)：可选参数，通信模式，默认值为`None`（等效`"ai_cpu"`）。

  <!-- npu="950" id17 -->
  - <term>Ascend 950DT系列产品</term>：支持`"ai_cpu"`、`"ccu"`。
  <!-- end id17 -->
  <!-- npu="A3,910b" id18 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：仅支持`"aiv"`。
  <!-- end id18 -->

## 返回值说明

- **y** (`Tensor`)：AllGather通信与MatMul计算的结果，即计算公式中的$output$。shape为2维$(m*world\_size, n)$，world_size为卡数。当x1为`torch.float16`/`torch.bfloat16`时，数据类型与x1保持一致；当x1为低精度数据类型时，数据类型由y_dtype指定，支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
- **gather_out** (`Tensor`)：AllGather通信的结果，即计算公式中的$gatherOut$。gather_output为`True`时，shape为2维$(m*world\_size, k)$，数据类型与x1保持一致；gather_output为`False`时返回空Tensor。
- **amax** (`Tensor`)：MatMul计算结果的最大值，即计算公式中的$amaxOut$。amax_output为`True`时shape为$[1]$，数据类型为`torch.float32`；amax_output为`False`（当前版本仅支持该场景）时返回空Tensor。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式。
- 默认确定性实现。
- 输入self必须是2维，其shape为$(m, k)$；输入x2必须是2维，非转置时其shape为$(k, n)$，转置时其shape为$(n, k)$，k轴相等，且k轴取值范围为$[256, 65535)$，m和n的值不得超过2147483647。
- self仅支持不转置场景，x2支持转置/不转置场景。
- x1和x2的数据类型需保持一致；当x1、x2数据类型为`torch.float8_e4m3fn`/`torch.float8_e5m2`时，两者可以为其中任意一种。

<!-- npu="950" id19 -->
- <term>Ascend 950DT系列产品</term>：
  - 支持2、4、8、16、32、64卡。
  - 支持空Tensor场景：m和n可以为空，k不可为空，且需满足以下条件：m为空、k不为空、n不为空；m不为空、k不为空、n为空；m为空、k不为空、n为空。
  - 当x1、x2数据类型为`torch_npu.float4_e2m1fn_x2`时，x2矩阵支持转置/不转置场景，x1矩阵只支持不转置场景，k轴需要为偶数，且当x2矩阵非转置时，n轴也需要为偶数。
  - 当group_size取值为549764202624时，bias必须为`None`。
  - comm_mode为`"ccu"`时仅支持单机UB域内互联，`"ai_cpu"`可支持跨机UB域内互联；使用`"ccu"`通信引擎时，单个通信域内allgather(x1)集合通信数据总量不能超过63*256MB，集合通信数据总量计算方式为：m \* k \* sizeof(x1_dtype) \* 卡数。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。
<!-- end id19 -->
<!-- npu="A3,910b" id20 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - 支持2、4、8卡。
  - bias仅支持传入`None`。
  - 不支持空Tensor。
  - 当x1和x2数据类型为`torch.int4`时，k与n必须为偶数。
  - comm_mode仅支持`"aiv"`。
  - 通信缓冲区大于等于200MB。
<!-- end id20 -->

## 调用示例

### 场景1：8卡场景，每张卡上的m为16，AllGather后的总m为128

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_all_gather_base_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    tensor_allgather_shape = x1_shape
    single_shape = [x1_shape[0] // world_size, x1_shape[1]]

    self_t = torch.randn(single_shape, dtype=dtype).npu()
    x2 = torch.randn(x2_shape, dtype=dtype).npu()
    output, gather_out, _ = torch_npu.npu_all_gather_quant_mm(self_t, x2, hcomm_info, world_size, comm_mode="aiv") #运行于A2/A3上，需要添加comm_mode="aiv"
    print(output.shape, output.dtype)
    print(gather_out.shape, gather_out.dtype)

if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 256]
    dtype = torch.float16

    mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
```

输出如下所示

```text
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
```

<!-- npu="A3,910b" id24 -->
### 场景2：`x1、x2`数据类型为`torch.int8`的perchannel、pertoken场景（<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>）

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_npu_all_gather_quant_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    single_shape = [x1_shape[0] // world_size, x1_shape[1]]
    self_t = torch.randint(-128, 127, single_shape, dtype=dtype).npu()
    x2 = torch.randint(-128, 127, x2_shape, dtype=dtype).npu()
    # pertoken场景：x1_scale的shape为(m, 1)；perchannel场景：x2_scale的shape为(1, n)
    x1_scale = torch.rand((single_shape[0], 1), dtype=torch.float32).npu()
    x2_scale = torch.rand((1, x2_shape[1]), dtype=torch.float32).npu()
    output, gather_out, amax = torch_npu.npu_all_gather_quant_mm(
        self_t, x2, hcomm_info, world_size, bias=None, x1_scale=x1_scale, x2_scale=x2_scale,
        gather_output=True, y_dtype=torch.float16, comm_mode="aiv")
    print(output.shape, output.dtype)  # (torch.Size([128, 256]), torch.float16)
    print(gather_out.shape, gather_out.dtype)  # (torch.Size([128, 512]), torch.int8)

if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 256]
    dtype = torch.int8
    mp.spawn(run_npu_all_gather_quant_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
```

输出如下所示

```text
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
```
<!-- end id24 -->

<!-- npu="950" id25 -->
### 场景3：`x1、x2`数据类型为`torch.float8_e4m3fn`的perblock场景（仅<term>Ascend 950DT系列产品</term>）

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import math
def run_npu_all_gather_quant_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    single_shape = [x1_shape[0] // world_size, x1_shape[1]]
    m_rank, k = single_shape
    n = x2_shape[1]
    self_t = torch.randn(single_shape).to(dtype).npu()
    x2 = torch.randn(x2_shape).to(dtype).npu()
    # perblock场景：blockSize为128，x1_scale、x2_scale均为float32，shape分别为(ceilDiv(m,128), ceilDiv(k,128))、(ceilDiv(k,128), ceilDiv(n,128))
    x1_scale = torch.rand((math.ceil(m_rank / 128), math.ceil(k / 128)), dtype=torch.float32).npu()
    x2_scale = torch.rand((math.ceil(k / 128), math.ceil(n / 128)), dtype=torch.float32).npu()
    output, gather_out, amax = torch_npu.npu_all_gather_quant_mm(
        self_t, x2, hcomm_info, world_size, bias=None, x1_scale=x1_scale, x2_scale=x2_scale,
        gather_output=True, y_dtype=torch.float16)
    print(output.shape, output.dtype)  # (torch.Size([256, 256]), torch.float16)
    print(gather_out.shape, gather_out.dtype)  # (torch.Size([256, 512]), torch.float8_e4m3fn)
    print(amax)  # amax_output为False时返回空Tensor

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [256, 512]
    x2_shape = [512, 256]
    dtype = torch.float8_e4m3fn
    mp.spawn(run_npu_all_gather_quant_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
```

输出如下所示

```text
torch.Size([256, 256]) torch.float16
torch.Size([256, 512]) torch.float8_e4m3fn
None
torch.Size([256, 256]) torch.float16
torch.Size([256, 512]) torch.float8_e4m3fn
None
```
<!-- end id25 -->

<!-- npu="950" id26 -->
### 场景4：`x1、x2`数据类型为`torch.float8_e4m3fn`的mx量化场景（仅<term>Ascend 950DT系列产品</term>）

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import math
def run_npu_all_gather_quant_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    single_shape = [x1_shape[0] // world_size, x1_shape[1]]
    m_rank, k = single_shape
    n = x2_shape[1]
    self_t = torch.randn(single_shape).to(dtype).npu()
    x2 = torch.randn(x2_shape).to(dtype).npu()
    # mx场景：scale数据类型为float8_e8m0fnu，shape分别为(m, ceilDiv(k, 64), 2)、(ceilDiv(k, 64), n, 2)
    x1_scale = torch.rand((m_rank, math.ceil(k / 64), 2)).to(torch.float8_e8m0fnu).npu()
    x2_scale = torch.rand((math.ceil(k / 64), n, 2)).to(torch.float8_e8m0fnu).npu()
    output, gather_out, amax = torch_npu.npu_all_gather_quant_mm(
        self_t, x2, hcomm_info, world_size, bias=None, x1_scale=x1_scale, x2_scale=x2_scale,
        gather_output=True, y_dtype=torch.float16)
    print(output.shape, output.dtype)
    print(gather_out.shape, gather_out.dtype)

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [64, 256]
    x2_shape = [256, 128]  # 转置输入(n, k)
    dtype = torch.float8_e4m3fn
    mp.spawn(run_npu_all_gather_quant_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
```

输出如下所示

```text
torch.Size([64, 128]) torch.float16
torch.Size([64, 256]) torch.float8_e4m3fn
torch.Size([64, 128]) torch.float16
torch.Size([64, 256]) torch.float8_e4m3fn
```

<!-- end id26 -->