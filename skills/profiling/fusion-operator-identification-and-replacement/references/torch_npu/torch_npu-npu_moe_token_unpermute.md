# torch_npu.npu_moe_token_unpermute

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

- API功能：MoE的unpermute计算，根据sorted_indices存储的下标，获取permuted_tokens中存储的输入数据；如果存在probs数据，permuted_tokens会与probs相乘；最后进行累加求和，并输出计算结果。

- 计算公式：
  - 当probs非None时，计算公式如下，此时topK_num为probs的最后一维大小：

    $$
    T[k] = T[S[k]]
    $$

    $$
    T[k] = T[k] * P[i][j]
    $$

    $$
    O[i] = \sum_{k=i*topK}^{(i+1)*topK - 1 } T[k]
    $$

    其中$i \in {0,1,...,tokens-1}$；$j = k - i * topK$，$j \in {0,1,...,topK-1}$；$k \in {0,1,...,tokens*topK-1}$；T表示permuted_tokens；S表示sorted_indices；P表示probs；O表示out；topK表示topK_num；tokens表示tokens_num。

  - 当probs为None时，此时topK_num=1，计算公式如下：

    $$
    T[i] = T[S[i]]
    $$

    $$
    O[i] = T[i]
    $$

    其中$i \in {0,1,...,tokens-1}$；T表示permuted_tokens；S表示sorted_indices；O表示out；tokens表示tokens_num。

> [!NOTE]
>
> - 本接口为确定性计算。
<!-- npu="950" id5 -->
> - 在<term>Ascend 950PR&950DT系列产品</term>上调用本接口时，框架内部会转调用aclnnMoeFinalizeRoutingV2接口，参数映射关系如下：permuted_tokens等同于aclnnMoeFinalizeRoutingV2接口的expandedX输入，sorted_indices等同于expandedRowIdx输入，probs等同于scalesOptional输入，padded_mode等同于dropPadMode输入，输出out等同于aclnnMoeFinalizeRoutingV2接口的out输出。如出现参数错误提示，请参考该映射关系。
<!-- end id5 -->

## 函数原型

```python
torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None) -> Tensor
```

## 参数说明

- **permuted_tokens**（`Tensor`）：必选参数，表示经过permute计算后的输入tokens。要求是一个2D的Tensor，shape为(tokens_num * topK_num, hidden_size)，其中tokens_num表示输入token的个数，topK_num表示处理每个token的专家个数，hidden_size表示每个token的向量表示的长度。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式支持$ND$，支持非连续Tensor。
- **sorted_indices**（`Tensor`）：必选参数，表示需要计算的数据在permuted_tokens中的位置。要求是一个1D的Tensor，shape为(tokens_num \* topK_num)。数据类型支持`torch.int32`，数据格式支持$ND$，支持非连续Tensor。取值范围是[0, tokens_num \* topK_num - 1]，且没有重复索引。
- **probs**（`Tensor`）：可选参数，表示与permuted_tokens相乘的概率值，默认值为`None`。要求是一个2D的Tensor，shape为(tokens_num, topK_num)。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式支持$ND$，支持非连续Tensor。当probs传入时，topK_num等于probs的最后一维大小；当probs不传时，topK_num等于1。
- **padded_mode**（`bool`）：可选参数，表示是否开启paddedMode，默认值为`False`。padded_mode为`True`时，restore_shape生效，输出结果的shape与restore_shape保持一致。当前仅支持`False`。
- **restore_shape**（`int[]`）：可选参数，表示padded_mode为`True`时输出结果的shape，默认值为`None`。当前仅支持`None`。

## 返回值说明

`Tensor`

表示unpermute计算后的输出结果。要求是一个2D的Tensor，padded_mode为`False`时shape为(tokens_num, hidden_size)，padded_mode为`True`时shape与restore_shape保持一致（当前padded_mode仅支持`False`）。数据类型与permuted_tokens保持一致。数据格式支持$ND$，不支持非连续Tensor。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口仅支持图模式。
- 该接口为确定性计算。

<!-- npu="A3,910b" id4 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：topK_num要求小于等于512。
<!-- end id4 -->

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> permuted_tokens = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [1, 1], [2, 2], [3, 3], [4, 4]],
    ...                                dtype=torch.float16, device='npu')
    >>> sorted_indices = torch.tensor([0, 4, 5, 1, 2, 6, 7, 3], dtype=torch.int32, device='npu')
    >>> probs = torch.ones((4, 2), dtype=torch.float16, device='npu') / 2
    >>> unpermuted_tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)
    >>> unpermuted_tokens
    tensor([[1., 1.],
            [2., 2.],
            [3., 3.],
            [4., 4.]], device='npu:0', dtype=torch.float16)
    ```

    probs为None时的调用（此时topK_num为1）：

    ```python
    >>> unpermuted_tokens = torch_npu.npu_moe_token_unpermute(
    ...     permuted_tokens[:4], torch.tensor([0, 1, 2, 3], dtype=torch.int32, device='npu'))
    >>> unpermuted_tokens
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
        def forward(self, permuted_tokens, sorted_indices, probs=None):
            return torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)
    # 实例化模型model
    model = Model().npu()
    # 从TorchAir获取NPU提供的默认backend
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # 使用TorchAir的backend去调用compile接口编译模型
    model = torch.compile(model, backend=npu_backend)

    # 生成随机数据, 并发送到npu
    permuted_tokens = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [1, 1], [2, 2], [3, 3], [4, 4]],
                                   dtype=torch.float16).npu()
    sorted_indices = torch.tensor([0, 4, 5, 1, 2, 6, 7, 3], dtype=torch.int32).npu()
    probs = torch.ones((4, 2), dtype=torch.float16).npu() / 2

    # 调用MoeTokenUnpermute算子
    unpermuted_tokens = model(permuted_tokens, sorted_indices, probs=probs)
    ```
