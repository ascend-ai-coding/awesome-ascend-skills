# torch_npu.npu_all_gather_base_mm

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

- **API功能**：TP切分（Tensor Parallelism，张量并行）场景下，实现allgather和matmul的融合，实现通信和计算流水并行。

- **计算公式**：
    `x1`（即`input`）代表Matmul计算的左矩阵，`x2`代表Matmul计算的右矩阵，`bias`代表Matmul计算的偏置。

    基础场景：
    $$
    output = allgather(x1) \mathbin{@} x2 + bias
    $$
    $$
    gather\_out = allgather(x1)
    $$

    <!-- npu="A3,910b" id22 -->
    量化场景（仅<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>支持）：
    $$
    output = (allgather(x1\_scale) * x2\_scale) * (allgather(x1) \mathbin{@} x2 + bias)
    $$
    $$
    gather\_out = allgather(x1)
    $$

    量化场景公式中，`allgather(x1_scale)`由底层算子内部完成，用户无需手动对`x1_scale`执行allgather操作。用户只需将当前rank本地的`x1_scale`（shape为(m, 1)）直接传入接口即可，底层算子会自动聚合所有rank的`x1_scale`后参与计算。`x2_scale`为全局共享的perchannel量化参数，各rank相同，无需执行allgather操作。
    <!-- end id22 -->

> [!NOTE]
> 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者更高版本，否则将会引发报错，比如BUS ERROR等。

## 函数原型

```python
torch_npu.npu_all_gather_base_mm(input, x2, hcom, world_size, *, bias=None, x1_scale=None, x2_scale=None, gather_index=0, gather_output=True, comm_turn=0, output_dtype=None, comm_mode=None) -> (Tensor, Tensor)
```

## 参数说明

- **input**（`Tensor`）：**必选参数**，表示矩阵乘法中的左矩阵。数据类型支持`torch.float16`、`torch.bfloat16`，数据格式支持$ND$，输入shape支持2维，形如\(m, k\)、\(k, n\)，轴满足matmul算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
- **x2**（`Tensor`）：**必选参数**，表示矩阵乘法中的右矩阵，数据类型、输入shape维度需要和`input`保持一致。

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据格式支持$ND$、$NZ$，$NZ$仅在`comm_mode`为`aiv`时支持。
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据格式支持$ND$。
    <!-- end id5 -->

- **hcom**（`str`）：**必选参数**，通信域handle名，通过get\_hccl\_comm\_name接口获取。
- **world\_size**（`int`）：**必选参数**，通信域内的rank总数。

    <!-- npu="910b" id6 -->
    - <term>Atlas A2系列产品</term>：支持2、4、8卡，支持HCCS链路all mesh组网（每张卡和其它卡两两相连）。
    <!-- end id6 -->
    <!-- npu="A3" id7 -->
    - <term>Atlas A3系列产品</term>：支持2、4、8、16、32卡，支持HCCS链路double ring组网（多张卡按顺序组成一个圈，每张卡只和左右卡相连）。
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持2、4、8、16、32、64卡，CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。
    <!-- end id8 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **bias**（`Tensor`）：**可选参数**，表示Matmul计算的偏置。数据类型支持`torch.float16`、`torch.bfloat16`，数据格式支持$ND$。数据类型需要和`input`保持一致。`bias`仅支持一维，且维度大小与`output`的第1维大小相同。

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：**当前版本暂不支持`bias`输入为非0的场景。**
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持`bias`输入为非0的场景。
    <!-- end id10 -->

- **x1\_scale**（`Tensor`）：**可选参数**，mm左矩阵反量化参数。数据格式支持$ND$。数据维度为\(m, 1\)，支持pertoken量化。

    <!-- npu="A3,910b" id11 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float32`。
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT系列产品</term>：**暂不支持该参数。**
    <!-- end id12 -->

- **x2\_scale**（`Tensor`）：**可选参数**，mm右矩阵反量化参数。数据格式支持$ND$。数据维度为\(1, n\)，支持perchannel量化。

    <!-- npu="A3,910b" id13 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float32`、`torch.int64`。如需传入`torch.int64`数据类型，需要提前调用torch\_npu.npu\_trans\_quant\_param来获取`torch.int64`数据类型的`x2_scale`。
    <!-- end id13 -->
    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT系列产品</term>：**暂不支持该参数。**
    <!-- end id14 -->

- **gather\_index**（`int`）：**可选参数**，表示gather操作对象，0：对`input`做gather，1：对`x2`做gather。默认值0。**当前版本仅支持输入0。**
- **gather\_output**（`bool`）：**可选参数**，表示是否需要gather输出。默认值为`True`。`comm_mode`为`ai_cpu`时，`gather_output`支持`True`和`False`；`comm_mode`为`aiv`时，`gather_output`仅支持`True`。
- **comm\_turn**（`int`）：**可选参数**，表示rank间通信切分粒度，默认值为0，表示默认的切分方式。**当前版本仅支持输入0。**
- **output\_dtype**（`ScalarType`）：**可选参数**，表示第一个输出的数据类型。仅支持在量化场景且`x1_scale`和`x2_scale`均为`torch.float32`时，可指定输出数据类型为`torch.bfloat16`或`torch.float16`，默认值为`torch.bfloat16`。

    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT系列产品</term>：**暂不支持该参数。**
    <!-- end id15 -->

- **comm\_mode**（`str`）：**可选参数**，表示通信模式。默认值为`None`。

    <!-- npu="A3,910b" id16 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持`ai_cpu`、`aiv`，`None`表示AI CPU通信。`ai_cpu`模式仅支持基础场景，`aiv`模式支持基础场景和量化场景。
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持`ai_cpu`、`ccu`、`None`。当为`None`时，走AI CPU通信。
    <!-- end id17 -->

## 返回值说明

- **output**（`Tensor`）：第一个输出是allgather+matmul的结果。

    <!-- npu="A3,910b" id18 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：基础场景下，数据类型和`input`保持一致。量化场景下，当`x2_scale`为`torch.int64`数据类型时，输出数据类型为`torch.float16`；当`x1_scale`和`x2_scale`均为`torch.float32`时，输出数据类型由`output_dtype`指定，默认为`torch.bfloat16`。
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT系列产品</term>：仅支持基础场景，输出数据类型和`input`保持一致。
    <!-- end id19 -->

- **gather\_out**（`Tensor`）：第二个输出是allgather的结果。由`gather_output`参数控制是否输出，`gather_output`为`False`时，返回空Tensor。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式。
- `input`不支持输入转置后的tensor，`x2`转置后输入，需要满足shape的第一维大小与`input`的最后一维相同，满足matmul的计算条件。
- **通信模式约束**：
  - `comm_mode`为`ai_cpu`时：
      - 该接口支持训练场景下使用。
      - 该接口支持图模式。

      <!-- npu="910b" id20 -->
      - <term>Atlas A2系列产品</term>：一个模型中的通算融合算子（AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce），仅支持相同通信域。
      <!-- end id20 -->

  - `comm_mode`为`aiv`时，训练和推理场景均可使用。

<!-- npu="950" id21 -->
- <term>Ascend 950PR&950DT系列产品</term>：allgather\(x1\)集合通信数据总量不能超过63\*256MB，集合通信数据总量计算方式为：m \* k \* sizeof\(x1\_dtype\) \* 卡数。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。
<!-- end id21 -->

## 调用示例

- 单算子模式调用

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

        input_ = torch.randn(single_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output, gather_out = torch_npu.npu_all_gather_base_mm(input_, weight, hcomm_info, world_size)

    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16

        mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    class ALLGATHER_MM_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, weight, hcomm_info, world_size, gather_output):
            output, gather_output = torch_npu.npu_all_gather_base_mm(input, weight, hcomm_info, world_size,
                                                                     gather_output=gather_output)
            return output, gather_output
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
    def get_graph(input, weight, hcomm_info, world_size, gather_output):
        model = ALLGATHER_MM_GRAPH_Model()
        model = define_model(model, 2)
        model_output = model(input, weight, hcomm_info, world_size, gather_output=gather_output)
        output_npu = model_output[0]
        gather_output_npu = model_output[1]
        return output_npu, gather_output_npu
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
        single_shape = [x1_shape[0] // world_size, x1_shape[1]]
        input = torch.randn(single_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        is_gather_out = True
        output, gather_out = get_graph(input, weight, hcomm_info, world_size, is_gather_out)
        print("output:", output)
    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
        mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```
