# torch_npu.npu_quant_scatter

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->

## 功能说明

先将`updates`进行量化，然后将`updates`中的值按指定的轴`axis`和索引`indices`更新`self`中的值，并将结果保存到输出tensor，`self`本身的数据不变。

## 函数原型

```python
torch_npu.npu_quant_scatter(self, indices, updates, quant_scales, quant_zero_points=None, axis=-2, quant_axis=-1, reduce='update', dst_type=None, round_mode='rint') -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选输入，源数据张量，数据格式支持$ND$，支持非连续的Tensor，维数只能是3~8维。

    <!-- npu="310p" id4 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.int8`。
    <!-- end id4 -->
    <!-- npu="910b" id5 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.int8`。
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。
    <!-- end id6 -->

- **indices** (`Tensor`)：必选输入，索引张量，数据类型支持`torch.int32`，数据格式支持$ND$，支持非连续的Tensor。
- **updates** (`Tensor`)：必选输入，更新数据张量，数据格式支持$ND$，支持非连续的Tensor。

    <!-- npu="310p" id7 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`。
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`。
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float16`。
    <!-- end id9 -->

- **quant_scales** (`Tensor`)：必选输入，量化缩放张量，数据格式支持$ND$，支持非连续的Tensor。

    <!-- npu="310p" id10 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float32`。
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float32`。
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.bfloat16`、`torch.float32`。
    <!-- end id12 -->

- **quant_zero_points** (`Tensor`)：可选输入，量化偏移张量，数据格式支持$ND$，支持非连续的Tensor。

    <!-- npu="310p" id13 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.int32`。
    <!-- end id13 -->
    <!-- npu="910b" id14 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.bfloat16`、`torch.int32`。
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.bfloat16`、`torch.int32`。
    <!-- end id15 -->

- **axis** (`int`)：可选参数，`updates`上用来更新的轴，默认值为`-2`。
- **quant_axis** (`int`)：可选参数，`updates`上用来量化的轴，默认值为`-1`。
- **reduce** (`str`)：可选参数，表示数据操作方式；当前只支持`'update'`，即更新操作。
- **dst_type** (`int`)：可选参数，指定输出`self`的数据类型，默认值为`None`，表示`torch.int8`。

    <!-- npu="910b,310p" id16 -->
    - <term>Atlas推理系列产品</term>、<term>Atlas A2系列产品</term>：暂不支持该参数，默认按`torch.int8`处理。
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持取值`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。
    <!-- end id17 -->

- **round_mode** (`str`)：可选参数，指定量化时的取整方式，默认值为`"rint"`。

    <!-- npu="910b,310p" id18 -->
    - <term>Atlas推理系列产品</term>、<term>Atlas A2系列产品</term>：暂不支持该参数，默认按`"rint"`处理。
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT系列产品</term>：当`dst_type`为`torch.int8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`时，支持`"rint"`；当`dst_type`为`torch_npu.hifloat8`时，支持`"round"`、`"hybrid"`。
    <!-- end id19 -->

## 返回值说明

`Tensor`

代表`self`被更新后的结果。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式。

- `indices`的维数只能是1维或者2维；如果是2维，其第2维的大小必须是2；不支持索引越界，用户需自行确保索引合法，框架不进行越界检查；`indices`映射的`self`数据段不能重合，若重合则会因为多核并发原因导致多次执行结果不一样。
- `updates`的维数需要与`self`的维数一样；其第1维的大小等于`indices`的第1维的大小，且不大于`self`的第1维的大小；其`axis`轴的大小不大于`self`的`axis`轴的大小；其余维度的大小要跟`self`对应维度的大小相等；其最后一维的大小必须32B对齐。
- `quant_scales`的元素个数需要等于`updates`在`quant_axis`轴的大小。
- `quant_zero_points`的元素个数需要等于`updates`在`quant_axis`轴的大小。
- `axis`不能为`updates`的第1维或最后1维。
- `quant_axis`只能为`updates`的最后1维。
<!-- npu="950,910b" id20 -->
- `updates`、`quant_scales`、`quant_zero_points`数据类型输入组合包括（适用于如下产品）：
    - <term>Atlas A2系列产品</term>
    - <term>Ascend 950PR&950DT系列产品</term>

    | updates | quant_scales | quant_zero_points |
    | --- | --- | --- |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` |
    | `torch.float16` | `torch.float32` | `torch.int32` |
<!-- end id20 -->

## 调用示例

- 单算子模式调用

   ```python
   >>> import torch
   >>> import torch_npu
   >>> import numpy as np
   >>>
   >>> data_var = np.random.uniform(0, 1, [24, 4096, 128]).astype(np.int8)
   >>> var = torch.from_numpy(data_var).to(torch.int8).npu()
   >>>
   >>> data_indices = np.random.uniform(0, 1, [24]).astype(np.int32)
   >>> indices = torch.from_numpy(data_indices).to(torch.int32).npu()
   >>>
   >>> data_updates = np.random.uniform(1, 2, [24, 1, 128]).astype(np.float16)
   >>> updates = torch.from_numpy(data_updates).to(torch.bfloat16).npu()
   >>>
   >>> data_quant_scales = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
   >>> quant_scales = torch.from_numpy(data_quant_scales).to(torch.bfloat16).npu()
   >>>
   >>> data_quant_zero_points = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
   >>> quant_zero_points = torch.from_numpy(data_quant_zero_points).to(torch.bfloat16).npu()
   >>>
   >>> axis = -2
   >>> quant_axis = -1
   >>> reduce = "update"
   >>>
   >>> out = torch_npu.npu_quant_scatter(var, indices, updates, quant_scales, quant_zero_points, axis=axis, quant_axis=quant_axis, reduce=reduce)
   >>> print(out.shape)
   torch.Size([24, 4096, 128])
   >>> print(out.dtype)
   torch.int8
   >>> print(out)
   tensor([[[2, 2, 2,  ..., 4, 9, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 2, 2,  ..., 5, 6, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[3, 2, 3,  ..., 6, 8, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         ...,

         [[2, 2, 2,  ..., 4, 8, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 2, 2,  ..., 4, 8, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 2, 3,  ..., 4, 9, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]]], device='npu:0', dtype=torch.int8)
   ```

- 图模式调用

    ```python
    # 入图方式
    import torch
    import torch_npu
    import math
    import torchair as tng
    import numpy as np
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"
    
    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    
    # 数据生成
    dtype_list2 =["fp16","int8","int32","fp32","fp16"]
    dtype_list =[np.float16,np.int8,np.int32,np.float32]
    updates_shape =[1,11,1,32]
    var_shape =[1,11,1,32]
    indices_shape =[1,2]
    quant_scales_shape =[1,1,1,32]
    quant_zero_points_shape =[1,1,1,32]
    
    axis =-2
    quant_axis=-1
    reduce = "update"

    updates_data = np.random.uniform(-1,1,size=updates_shape)
    var_data = np.random.uniform(1,2,size=var_shape).astype(dtype_list[1])
    quant_scales_data = np.random.uniform(1,2,size=quant_scales_shape)
    indices_data = np.random.uniform(0,1,size=indices_shape).astype(dtype_list[2])
    quant_zero_points_data = np.random.uniform(0,1,size=quant_zero_points_shape)

    updates_npu = torch.from_numpy(updates_data).npu().to(torch.bfloat16).npu()
    var_npu = torch.from_numpy(var_data).npu()
    quant_scales_npu = torch.from_numpy(quant_scales_data).npu().to(torch.bfloat16).npu()
    quant_zero_points_npu = torch.from_numpy(quant_zero_points_data).to(torch.bfloat16).npu()
    indices_npu = torch.from_numpy(indices_data).npu()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_quant_scatter(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)

    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_quant_scatter(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)
        print("single op output with mask:", single_op, single_op.shape)
        print("graph output with mask:", graph_output, graph_output.shape)

    if __name__ == "__main__":
        MetaInfershape()
    
    # 执行上述代码的输出类似如下
    single op output with mask: tensor([[[[ 1,  1,  0,  1,  0, -1,  0,  0,  0,  1,  0,  1,  0, -1,  1,  0,  0,
               0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  2,  1,  0,  0]],
            [[ 1,  0,  0,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  1,  1,  0,  1,
               1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0]],
            [[ 1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0, -1,  1,  1,  1,  1,
               0,  1,  0,  2,  0,  0,  0,  1,  0,  1,  1,  2,  0,  1,  1]],
            [[ 1,  1,  0,  1,  0, -1,  0,  1,  0,  1,  0,  0, -1,  0,  1,  0,  0,
               1,  0,  2,  2,  0,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0]],
            [[ 1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,
               0,  0,  1,  2,  0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1]],
            [[ 0,  1,  0,  1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  0,
               0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0]],
            [[ 0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0, -1,  1,  0,  0,
               1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1]],
            [[ 1,  1,  1,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1,
               0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  1]],
            [[ 1,  1,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  1, -1,  0,
               1,  1,  0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,  1]],
            [[ 1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,
               0,  1,  1,  1, -1,  0,  1,  0,  0,  0,  1,  1,  1,  0,  0]],
            [[ 1,  0, -1,  1,  0,  0,  1,  0,  1,  2,  0,  1,  0, -1,  1,  1,  1,
               1,  0,  0,  2,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0]]]],
           device='npu:0', dtype=torch.int8) torch.Size([1, 11, 1, 32])           
    graph output with mask: tensor([[[[ 1,  1,  0,  1,  0, -1,  0,  0,  0,  1,  0,  1,  0, -1,  1,  0,  0,
               0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  2,  1,  0,  0]],
            [[ 1,  0,  0,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  1,  1,  0,  1,
               1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0]],
            [[ 1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0, -1,  1,  1,  1,  1,
               0,  1,  0,  2,  0,  0,  0,  1,  0,  1,  1,  2,  0,  1,  1]],
            [[ 1,  1,  0,  1,  0, -1,  0,  1,  0,  1,  0,  0, -1,  0,  1,  0,  0,
               1,  0,  2,  2,  0,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0]],
            [[ 1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,
               0,  0,  1,  2,  0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1]],
            [[ 0,  1,  0,  1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  0,
               0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0]],
            [[ 0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0, -1,  1,  0,  0,
               1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1]],
            [[ 1,  1,  1,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1,
               0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  1]],
            [[ 1,  1,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  1, -1,  0,
               1,  1,  0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,  1]],
            [[ 1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,
               0,  1,  1,  1, -1,  0,  1,  0,  0,  0,  1,  1,  1,  0,  0]],
            [[ 1,  0, -1,  1,  0,  0,  1,  0,  1,  2,  0,  1,  0, -1,  1,  1,  1,
               1,  0,  0,  2,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0]]]],
           device='npu:0', dtype=torch.int8) torch.Size([1, 11, 1, 32])
    ```
