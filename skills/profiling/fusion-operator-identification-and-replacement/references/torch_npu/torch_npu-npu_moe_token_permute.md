# torch_npu.npu_moe_token_permute

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

- API功能：MoE（Mixture of Experts）的permute计算，根据索引indices将tokens广播并排序。

- 计算公式：当`padded_mode`为`False`时（当前仅支持`False`），公式如下，其中`topK`指一个token选择的专家个数，`indices`维度为2时`topK`等于`indices`最后一维大小，`indices`维度为1时`topK`等于1：

  $$
  sortedIndicesFirst=argSort(flatten(Indices))
  $$

  $$
  sortedIndicesOut=argSort(sortedIndicesFirst)
  $$

  $$
  permuteTokensOut[sortedIndicesOut[i]]=tokens[i//topK]
  $$

  当`padded_mode`为`True`时（当前暂不支持）：

  $$
  permuteTokensOut[i]=tokens[indices[i]]
  $$

  $$
  sortedIndicesOut=indices
  $$

> [!NOTE]
>
> - 本接口为确定性计算。
> - 本接口通常与`torch_npu.npu_moe_token_unpermute`接口配套使用：先对tokens按专家进行permute排序，完成专家计算后再通过unpermute将数据还原回原始顺序。

## 函数原型

```python
torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=None, padded_mode=False) -> (Tensor, Tensor)
```

## 参数说明

- **tokens**（`Tensor`）：必选参数，表示输入token特征。要求是一个2维的Tensor，shape为(num_tokens, hidden_size)，其中第一维的大小为num_tokens。支持空tensor。数据格式支持$ND$，支持非连续Tensor。

  <!-- npu="950" id4 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`、`torch.int8`，其中`torch.int8`按非量化方式处理。
  <!-- end id4 -->
  <!-- npu="A3" id5 -->
  - <term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id5 -->
  <!-- npu="910b" id6 -->
  - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id6 -->

- **indices**（`Tensor`）：必选参数，表示输入indices索引。要求是一个1D或2D的Tensor，支持空tensor。数据类型支持`torch.int32`、`torch.int64`，数据格式支持$ND$，支持非连续Tensor。
  - 当`padded_mode`为`False`时，表示每一个输入token对应的topK个处理专家索引：维度为2时shape为(num_tokens, topK)，维度为1时shape为(num_tokens)，此时topK视为1。
  - 当`padded_mode`为`True`时，表示每个专家选中的token索引（当前暂不支持）。
  - 元素个数要求小于16777215，元素值要求大于等于0且小于16777215。

  <!-- npu="950" id7 -->
  - 在<term>Ascend 950PR&950DT系列产品</term>上调用本接口且`tokens`数据类型为`torch.int8`时，元素表示expert ID，取值范围为[0, 10240)，最大值为10239，不支持10240。
  <!-- end id7 -->

- **num_out_tokens**（`int`）：可选参数，表示有效输出token数，默认值为`None`，数据类型为`torch.int64`。
  - 值为`None`或`0`时，表示不会删除任何token。
  - 值大于`0`时，会按照num_out_tokens对按照专家排序好的token进行切片，保留前num_out_tokens个token。
  - 值小于`0`时，按负的切片索引进行处理。

- **padded_mode**（`bool`）：可选参数，表示是否为填充模式，默认值为`False`。
  - `False`：表示非填充模式，对indices进行排序。
  - `True`：表示填充模式，indices已被填充为代表每个专家选中的token索引，此时不对indices进行排序（当前暂不支持）。

## 返回值说明

- **permuted_tokens**（`Tensor`）：根据indices进行扩展并排序过的tokens。要求是一个2维的Tensor，数据类型与`tokens`保持一致，数据格式支持$ND$，不支持非连续Tensor。第一维的大小为min(num_tokens \* topK, num_out_tokens)，其中`num_out_tokens`为`None`或`0`时第一维的大小为num_tokens \* topK，`num_out_tokens`小于`0`时按负的切片索引处理；除第一维外其余维度大小与`tokens`保持一致。
- **sorted_indices**（`Tensor`）：表示`permuted_tokens`和`tokens`的映射关系。要求是一个1D的Tensor，shape为(num_tokens * topK)，即`indices`的元素个数，数据类型支持`torch.int32`，数据格式支持$ND$，不支持非连续Tensor。其中sorted_indices[i]表示`tokens`的第i // topK行在`permuted_tokens`中的位置，即permuted_tokens[sorted_indices[i]] = tokens[i // topK]，可配合`torch_npu.npu_moe_token_unpermute`接口将数据还原回原始顺序。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口仅支持图模式。
- 该接口为确定性计算。
- `tokens`与`permuted_tokens`的数据类型必须一致。
<!-- npu="950" id8 -->
- `torch.int8`类型的`tokens`和`permuted_tokens`仅支持<term>Ascend 950PR&950DT系列产品</term>。
<!-- end id8 -->

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> tokens = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float16, device='npu')
    >>> indices = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0]], dtype=torch.int32, device='npu')
    >>> permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices)
    >>> permuted_tokens
    tensor([[1., 1.],
            [2., 2.],
            [3., 3.],
            [4., 4.],
            [1., 1.],
            [2., 2.],
            [3., 3.],
            [4., 4.]], device='npu:0', dtype=torch.float16)
    >>> sorted_indices
    tensor([0, 4, 5, 1, 2, 6, 7, 3], device='npu:0', dtype=torch.int32)
    >>> permuted_tokens[sorted_indices]
    tensor([[1., 1.],
            [1., 1.],
            [2., 2.],
            [2., 2.],
            [3., 3.],
            [3., 3.],
            [4., 4.],
            [4., 4.]], device='npu:0', dtype=torch.float16)
    ```

    指定`num_out_tokens`对排序后的token进行切片：

    ```python
    >>> permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=4)
    >>> permuted_tokens
    tensor([[1., 1.],
            [2., 2.],
            [3., 3.],
            [4., 4.]], device='npu:0', dtype=torch.float16)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import numpy

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, tokens, indices):
            return torch_npu.npu_moe_token_permute(tokens, indices)
    # 实例化模型model
    model = Model().npu()
    # 从TorchAir获取NPU提供的默认backend
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # 使用TorchAir的backend去调用compile接口编译模型
    model = torch.compile(model, backend=npu_backend)

    # 生成随机数据, 并发送到npu
    tokens = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float16).npu()
    indices = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0]], dtype=torch.int32).npu()

    # 调用MoeTokenPermute算子
    permuted_tokens, sorted_indices = model(tokens, indices)
    ```
