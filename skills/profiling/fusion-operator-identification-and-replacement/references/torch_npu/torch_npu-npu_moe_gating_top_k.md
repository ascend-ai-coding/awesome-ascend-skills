# torch_npu.npu_moe_gating_top_k

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

- API功能：MoE计算中，对输入x做Sigmoid/Softmax计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。
- 计算公式：

    对输入做sigmoid（`bias`可选）：
    - 当`norm_type`为1时：

      ![](figures/zh-cn_formulaimage_0000002258672873.png)

    - 当`norm_type`为0时：

      ![](figures/zh-cn_formulaimage_0000002313785750.png)

    对计算结果按照`group_count`进行分组：
    
    - 如果`group_select_mode`为1，每组按照topk2的sum值对group进行排序，取前`k_group`个组：

      ![](figures/zh-cn_formulaimage_0000002219010398.png)

    - 如果`group_select_mode`为0，每组按照组内的最大值对group进行排序，取前`k_group`个组：

      ![](figures/zh-cn_formulaimage_0000002347834049.png)

    根据上一步的groupId获取`normOut`中对应的元素，将数据再做TopK，得到`expertIdxOut`的结果：

    ![](figures/zh-cn_formulaimage_0000002219172722.png)

    对y按照输入的`routed_scaling_factor`和`eps`参数进行计算，得到yOut的结果：

    ![](figures/zh-cn_formulaimage_0000002219173660.png)

- 等价计算逻辑：

    ```python
    import torch
    import numpy

    def moe_gating_top_k_numpy(x: torch.tensor, k: int, *, bias: torch.tensor = None, k_group: int = 1, group_count: int = 1,
                                group_select_mode: int = 0, renorm: int = 0, norm_type: int = 0, out_flag: bool = False,
                                routed_scaling_factor: float = 1.0, eps: float = 1e-20) -> tuple:
        dtype = x.dtype
        if dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            bias = bias.to(dtype=torch.float32)

        x = x.numpy()
        bias = bias.numpy()
        if norm_type == 0:
            x = numpy.exp(x - numpy.expand_dims(numpy.log(numpy.sum(numpy.exp(x),
                            axis=-1, keepdims=True)), axis=-1))  # softmax
        else:
            x = 1 / (1 + numpy.exp(-x))  # sigmoid
        original_x = x
        if bias is not None:
            x = x + bias
            
        if group_count > 1:
            x = x.reshape(x.shape[0], group_count, -1)
            if group_select_mode == 0:
                group_x = numpy.amax(x, axis=-1)
            else:
                group_x = numpy.partition(x, -2, axis=-1)[..., -2:].sum(axis=-1)
        indices = numpy.argsort(-group_x, axis=-1, kind='stable')[:, :k_group]  # Indices of top-k_group

        mask = numpy.ones((x.shape[0], group_count), dtype=bool)  # Create a mask with all 1
        mask[numpy.arange(x.shape[0])[:, None], indices] = False  # Set to false at the indices
        x = numpy.where(mask[..., None], float('-inf'), x)  # Fill with -inf when mask value is true
        x = x.reshape(x.shape[0], -1)

        indices = numpy.argsort(-x, axis=-1, kind='stable')
        indices = indices[:, :k]
        y = numpy.take_along_axis(original_x, indices, axis=1)

        if norm_type == 1:
            y /= (numpy.sum(y, axis=-1, keepdims=True) + eps)
        y *= routed_scaling_factor
        if out_flag:
            out = original_x
        else:
            out = None

        y = torch.tensor(y, dtype=dtype)
        return y, indices.astype(numpy.int32), out   

    k = 6
    k_group = 4
    group_count = 8
    group_select_mode = 1
    renorm = 0
    norm_type = 1
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20
        

    x = numpy.random.uniform(-2, 2, (8, 256)).astype(numpy.float32)
    bias = numpy.random.uniform(-2, 2, (256,)).astype(numpy.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)

    y, expert_idx, out = moe_gating_top_k_numpy(x_tensor, k, bias = bias_tensor, k_group = k_group, group_count = group_count,
                                group_select_mode = group_select_mode, renorm = renorm, norm_type = norm_type, out_flag = out_flag,
                                routed_scaling_factor = routed_scaling_factor, eps = eps)

    print(f"yOut shape: {y.shape}")              
    print(f"expertIdxOut shape: {expert_idx.shape}")  
    print(f"Selected experts: {expert_idx[0]}")
    ```

## 函数原型

```python
npu_moe_gating_top_k(x, k, *, bias=None, input_ids=None, tid2eid=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选参数，表示待计算的输入。要求是一个2D的Tensor，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式要求为ND。支持非连续Tensor。最后一维的大小（即专家数）要求不大于`2048`。

- **k**（`int`）：必选参数，表示每个token最终筛选得到的专家个数，数据类型为`torch.int64`。要求`1 <= k <= x_shape[-1] / group_count * k_group`。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
 
- **bias**（`Tensor`）：可选参数，表示与输入`x`进行计算的bias值。要求是1D的Tensor，要求shape值与`x`的最后一维相等。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据类型需要与`x`保持一致，数据格式要求为ND。支持非连续`Tensor`。

- **input_ids**（`Tensor`）：可选参数，表示Hash模式的输入索引，用于从`tid2eid`中查找专家索引。要求是1D的Tensor，要求shape值与`x`的第一维相等。数据类型支持`torch.int32`、`torch.int64`，数据格式要求为ND。支持非连续`Tensor`。

- **tid2eid**（`Tensor`）：可选参数，表示Hash映射表，存储预计算的专家索引。要求是2D的Tensor，要求shape值的最后一维与`k`相等。数据类型支持`torch.int32`、`torch.int64`，数据格式要求为ND。支持非连续`Tensor`。

- **k_group**（`int`）：可选参数，表示每个token组筛选过程中，选出的专家组个数，数据类型为`torch.int64`，默认值为`1`。要求`1 <= k_group <= group_count`，并且`k_group * x_shape[-1] / group_count`的值要大于等于`k`。

- **group_count**（`int`）：可选参数，表示将全部专家划分的组数，数据类型为`torch.int64`，默认值为`1`。要求group_count > 0，x_shape[-1]能够被`group_count`整除且整除后的结果大于`group_select_mode`，并且整除的结果按照32个数对齐后乘`group_count`的结果不大于`2048`。

- **group_select_mode**（`int`）：可选参数，表示一个专家组的总得分计算方式。默认值为`0`，`0`表示组内取最大值，作为专家组得分；`1`表示取组内Top2的专家进行得分累加，作为专家组得分。

- **renorm**（`int`）：可选参数，表示renorm标记，默认值为`0`。支持`0`和`1`，`0`表示不做renorm，`1`表示做renorm。仅`norm_type`为`0`时该参数生效，`norm_type`为`1`或`2`时无效。
- **norm_type**（`int`）：可选参数，表示norm函数类型，默认值为`0`。

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持取值`0`、`1`，`0`表示使用Softmax函数，`1`表示使用Sigmoid函数。
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持取值`0`、`1`、`2`，`0`表示使用Softmax函数，`1`表示使用Sigmoid函数，`2`表示使用SqrtSoftplus函数。
    <!-- end id5 -->

- **out_flag**（`bool`）：可选参数，是否输出norm函数中间结果。默认值为`False`。
- **routed_scaling_factor**（`float`）：可选参数，表示计算`yOut`使用的`routed_scaling_factor`系数，默认值为`1.0`。
- **eps**（`float`）：可选参数，表示计算`yOut`使用的`eps`系数，默认值为`1e-20`。

## 返回值说明

- **yOut**（`Tensor`）：表示对`x`做norm操作和分组排序topk后计算的结果。要求是一个2D的Tensor，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据类型与`x`需要保持一致，数据格式要求为ND，第一维的大小要求与`x`的第一维相同，最后一维的大小与`k`相同。不支持非连续Tensor。
- **expertIdxOut**（`Tensor`）：表示对`x`做norm操作和分组排序topk后的索引，即专家的序号。shape要求与yOut一致，数据类型支持`torch.int32`，数据格式要求为ND。不支持非连续Tensor。
- **normOut**（`Tensor`）：表示norm计算的输出结果。shape要求与`x`保持一致，数据类型为`torch.float32`，数据格式要求为ND。不支持非连续Tensor。

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy
    
    k = 1
    k_group = 4
    group_count = 8
    group_select_mode = 1
    renorm = 0
    norm_type = 1
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20
    
    # 生成随机数据, 并发送到npu
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()
    
    # 调用MoeGatingTopK算子
    y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
    ```

- 图模式调用

    ```python
    # 入图方式
    import torch
    import torch_npu
    import torchair
    import numpy
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x_tensor, bias_tensor):
            return torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
    # 实例化模型model
    model = Model().npu()
    # 从TorchAir获取NPU提供的默认backend
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # 使用TorchAir的backend去调用compile接口编译模型
    model = torch.compile(model, backend=npu_backend)
    
    k = 1
    k_group = 4
    group_count = 8
    group_select_mode = 1
    renorm = 0
    norm_type = 1
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20
    
    # 生成随机数据, 并发送到npu
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()
    
    # 调用MoeGatingTopK算子
    y_npu, expert_idx_npu, out_npu = model(x_tensor, bias_tensor)
    ```
    