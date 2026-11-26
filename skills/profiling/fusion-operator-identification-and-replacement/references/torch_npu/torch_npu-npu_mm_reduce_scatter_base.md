# torch_npu.npu_mm_reduce_scatter_base

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- **API功能**：TP切分场景下，完成Matmul计算和ReduceScatter通信的融合，先计算后通信。融合算子内部实现计算和通信流水并行。
    <!-- npu="A3,910b" id4 -->
    - 支持如下两种量化方式（量化场景仅<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>支持）：
      - perchannel量化：针对权重（Weight）张量，沿输出特征维度（Output Channel）为每个通道维护独立的缩放因子与零点。
      - pertoken量化：针对激活（Activation）张量，沿序列长度维度（Sequence Length / Token）为每个Token维护独立的量化参数。
    <!-- end id4 -->
- 计算公式：
    `x1`（即`input`）代表Matmul计算的左矩阵，`x2`代表Matmul计算的右矩阵，`bias`代表Matmul计算的偏置。场景由`comm_mode`与`input`数据类型共同决定：

    <!-- npu="A3,910b" id5 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`comm_mode`为`ai_cpu`时，始终为基础场景；`comm_mode`为`aiv`（aiv模式下使用AI Vector核完成通信任务）时，`input`为`torch.float16`或`torch.bfloat16`走基础场景，`input`为`torch.int8`走量化场景。
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT系列产品</term>：仅支持基础场景。
    <!-- end id6 -->

    基础场景：
    $$
    output = ReduceScatter(x_1 \mathbin{@} x_2 + bias)
    $$

    <!-- npu="A3,910b" id34 -->
    量化场景（仅<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>支持）：
    $$
    output = ReduceScatter((x1\_scale * x2\_scale) * (x_1 \mathbin{@} x_2 + bias))
    $$
    量化场景scale参数使用说明：
    - 仅提供`x2_scale`时：
        $$
        output = ReduceScatter(x2\_scale * (x_1 \mathbin{@} x_2 + bias))
        $$
    - 同时提供`x1_scale`和`x2_scale`时：
        $$
        output = ReduceScatter((x1\_scale * x2\_scale) * (x_1 \mathbin{@} x_2 + bias))
        $$
    - 其中，`x1_scale`按\(m, 1\)广播，`x2_scale`按\(1, n\)广播。

    <!-- end id34 -->

> [!NOTE]
> 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者更高版本，否则将会引发BUS ERROR等报错。

## 函数原型

```python
torch_npu.npu_mm_reduce_scatter_base(input, x2, hcom, world_size, *, reduce_op='sum', bias=None, x1_scale=None, x2_scale=None, comm_turn=0, output_dtype=None, comm_mode=None) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：**必选参数**，表示Matmul计算的左矩阵输入。数据格式支持$ND$，输入shape支持2维，形如\(m, k\)。

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
    <!-- end id8 -->

- **x2**（`Tensor`）：**必选参数**，表示Matmul计算的右矩阵输入，数据类型需要和`input`保持一致。输入shape支持2维，形如\(k, n\)，轴满足matmul算子入参要求，第一轴与`input`的第二轴相等，且k的取值范围为\[256, 65535\)。

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据格式支持$ND$、$NZ$，$NZ$仅在`comm_mode`为`aiv`时支持。
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据格式仅支持$ND$。
    <!-- end id10 -->

- **hcom**（`str`）：**必选参数**，Host侧标识列组的字符串，即通信域handle名，通过get\_hccl\_comm\_name接口获取。
- **world\_size**（`int`）：**必选参数**，通信域内的rank总数。

    <!-- npu="910b" id11 -->
    - <term>Atlas A2系列产品</term>：支持2、4、8卡，支持HCCS链路all mesh组网（每张卡和其它卡两两相连）。
    <!-- end id11 -->
    <!-- npu="A3" id12 -->
    - <term>Atlas A3系列产品</term>：支持2、4、8、16、32卡，支持HCCS链路double ring组网（多张卡按顺序组成一个圈，每张卡只和左右卡相连）。
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持2、4、8、16、32、64卡，CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。
    <!-- end id13 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **reduce\_op**（`str`）：**可选参数**，reduce操作类型，当前仅支持'sum'，默认值为'sum'。
- **bias**（`Tensor`）：**可选参数**，表示Matmul计算的偏置。数据类型支持`torch.float16`、`torch.bfloat16`，需要和`input`保持一致。数据格式支持$ND$。`bias`仅支持一维，且维度大小与`output`的第1维大小相同。

    <!-- npu="A3,910b" id14 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：**当前版本暂不支持`bias`输入为非0的场景。**
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持`bias`输入为非0的场景。
    <!-- end id15 -->

- **x1\_scale**（`Tensor`）：**可选参数**，mm左矩阵反量化参数。数据维度为\(m, 1\)，支持pertoken量化。数据格式支持$ND$。

    <!-- npu="A3,910b" id16 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float32`。
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950PR&950DT系列产品</term>：**暂不支持该参数。**
    <!-- end id17 -->

- **x2\_scale**（`Tensor`）：**可选参数**，mm右矩阵反量化参数。数据维度为\(1, n\)，支持perchannel量化。数据格式支持$ND$。

    <!-- npu="A3,910b" id18 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float32`、`torch.int64`。如需传入`torch.int64`数据类型，需要提前调用torch_npu.npu_trans_quant_param来获取`torch.int64`数据类型的`x2_scale`。
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT系列产品</term>：**暂不支持该参数。**
    <!-- end id19 -->

- **comm\_turn**（`int`）：**可选参数**，表示rank间通信切分粒度，默认值为0，表示默认的切分方式。**当前版本仅支持输入0。**
- **output\_dtype**（`ScalarType`）：**可选参数**，表示输出数据类型。仅支持在量化场景且`x1_scale`和`x2_scale`均为`torch.float32`时，可指定输出数据类型为`torch.bfloat16`或`torch.float16`，默认值为`torch.bfloat16`。

    <!-- npu="950" id20 -->
    - <term>Ascend 950PR&950DT系列产品</term>：**暂不支持该参数。**
    <!-- end id20 -->

- **comm\_mode**（`str`）：**可选参数**，表示通信模式，默认值为`None`，`None`表示AI CPU通信。

    <!-- npu="A3,910b" id21 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持`ai_cpu`、`aiv`两种模式，`aiv`模式下使用AI Vector核完成通信任务。`ai_cpu`模式仅支持基础场景，`aiv`模式支持基础场景和量化场景。
    <!-- end id21 -->
    <!-- npu="950" id22 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持`ai_cpu`、`ccu`两种模式，仅支持基础场景。
    <!-- end id22 -->

## 返回值说明

`Tensor`

输出shape为\(m // world\_size, n\)。

<!-- npu="A3,910b" id23 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：基础场景下，数据类型和`input`保持一致。量化场景下，当`x2_scale`为`torch.int64`时，输出数据类型为`torch.float16`；当`x1_scale`和`x2_scale`均为`torch.float32`时，输出数据类型由`output_dtype`指定，默认为`torch.bfloat16`。
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950PR&950DT系列产品</term>：仅支持基础场景，数据类型和`input`保持一致。
<!-- end id24 -->

## 约束说明

- 该接口支持单算子模式和TorchAir图模式。
- `input`不支持输入转置后的tensor，`x2`转置后输入，需要满足shape的第一维大小与`input`的最后一维相同，满足matmul的计算条件。
- `world_size`必须等于实际通信域中的rank总数，且`input`的m轴必须能够被`world_size`整除。
- **通信模式约束**：

  <!-- npu="A3,910b" id25 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`comm_mode`支持`ai_cpu`、`aiv`两种模式。`ai_cpu`模式仅在训练场景下使用，`aiv`模式训练和推理场景均可使用。
  <!-- end id25 -->
  <!-- npu="950" id26 -->
  - <term>Ascend 950PR&950DT系列产品</term>：`comm_mode`支持`ai_cpu`、`ccu`两种模式，两种模式均仅在训练场景下使用。ReduceScatter集合通信数据总量不能超过16\*256MB，集合通信数据总量计算方式为：m \* n \* sizeof\(output\_dtype\)。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。
  <!-- end id26 -->

- **comm_mode支持矩阵**：

  <!-- npu="A3,910b" id27 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - 场景：基础场景
    - `comm_mode`：`ai_cpu` 
    - `input`数据类型：`torch.float16` / `torch.bfloat16`
    - 是否支持：支持
  <!-- end id27 -->
  <!-- npu="A3,910b" id28 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - 场景：量化场景
    - `comm_mode`：`ai_cpu` 
    - `input`数据类型：`torch.int8`
    - 是否支持：不支持
  <!-- end id28 -->
  <!-- npu="A3,910b" id29 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - 场景：基础场景
    - `comm_mode`：`aiv` 
    - `input`数据类型：`torch.float16` / `torch.bfloat16`
    - 是否支持：支持
  <!-- end id29 -->
  <!-- npu="A3,910b" id30 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - 场景：量化场景
    - `comm_mode`：`aiv` 
    - `input`数据类型：`torch.int8`
    - 是否支持：支持
  <!-- end id30 -->
  <!-- npu="950" id31 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - 场景：基础场景
    - `comm_mode`：`ai_cpu` / `ccu` 
    - `input`数据类型：`torch.float16` / `torch.bfloat16`
    - 是否支持：支持
  <!-- end id31 -->
  <!-- npu="950" id32 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - 场景：量化场景
    - `comm_mode`：- 
    - `input`数据类型：-
    - 是否支持：不支持（仅支持基础场景）
  <!-- end id32 -->
<!-- npu="A3,910b" id33 -->
- **scale参数组合约束**（量化场景，仅<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>支持）：

    |  场景   |  `x1_scale`  | `x2_scale`  |     输出数据类型 |
    | ------- | ---------- | --------- |   ----------- |
    | 基础场景 |    不传    |  不传     |   与`input`一致 |
    | 量化场景 |  不传      | `torch.float32` |  由`output_dtype`指定，默认`torch.bfloat16` |
    | 量化场景 | `torch.float32` | `torch.float32` |  由`output_dtype`指定，默认`torch.bfloat16` |
    | 量化场景 | `torch.float32` |  `torch.int64`  |  由`output_dtype`指定，默认`torch.bfloat16` |
    | 量化场景 |  不传     |  `torch.int64`  |   由`output_dtype`指定，默认`torch.bfloat16` |
<!-- end id33 -->

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    def run_mm_reduce_scatter_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)

        input_ = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = torch_npu.npu_mm_reduce_scatter_base(input_, weight, hcomm_info, world_size)

    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16

        mp.spawn(run_mm_reduce_scatter_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    class MM_REDUCESCATTER_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, weight, hcomm_info, world_size, reduce_op):
            output = torch_npu.npu_mm_reduce_scatter_base(input, weight, hcomm_info, world_size,
                                                          reduce_op=reduce_op)
            return output
    def define_model(model, graph_type):
        import torchair
        if graph_type == 1:  # 传统入图模式，静态shape+在线编译场景
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN入图模式，动态shape+二进制
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        else:
            print("Error type")
        return model
    def get_graph(input, weight, hcomm_info, world_size):
        model = MM_REDUCESCATTER_GRAPH_Model()
        model = define_model(model, 2)
        model_output = model(input, weight, hcomm_info, world_size, reduce_op="sum")
        return model_output
    def run_mm_reduce_scatter_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
        input = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = get_graph(input, weight, hcomm_info, world_size)
        print("output:", output)
    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
        mp.spawn(run_mm_reduce_scatter_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```
