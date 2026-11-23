# torch_npu.npu_moe_token_unpermute_with_routing_map

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

- API功能：对经过permute计算处理的permuted_tokens，根据sorted_indices和routing_map累加回原unpermuted_tokens，支持drop_and_pad填充模式。

- 计算公式：

  $$
  topK\_num= permutedTokens.size(0) // routingMapOptional.size(0)
  $$

  其中`topK_num`表示每个token预留的最大专家槽位数。`drop_and_pad`为`False`时，每个token实际选择的专家数可以小于等于`topK_num`；未使用的槽位在`sorted_indices`中以`-1`表示，计算时跳过该槽位。

  $$
  numExperts = probs.size(1)
  $$

  $$
  numTokens = probs.size(0)
  $$

  $$
  capacity = sortedIndices.size(0) // numExperts
  $$

  （1）probs不为None，drop_and_pad为`True`时：

  $$
  permuteProbs[i//capacity,sortedIndices[i]]=probs[i]
  $$

  $$
  permutedTokens = permutedTokens * permuteProbs
  $$

  $$
  unpermutedTokens = zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  （2）probs不为None，drop_and_pad为`False`时（T为转置操作）：

  $$
  permuteProbs = probs.T.maskedSelect(routingMap.T)
  $$

  $$
  permutedTokens = permutedTokens * permuteProbs
  $$

  $$
  unpermutedTokens = zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  if \space sortedIndices[i] >= 0:
      unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]] * permuteProbs[i]
  $$

  （3）probs为None，drop_and_pad为`True`时：

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  （4）probs为None，drop_and_pad为`False`时：

  $$
  if \space sortedIndices[i] >= 0:
      unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$

> [!NOTE]
>
> - 本接口默认非确定性实现，支持通过alcrtCtxSetSysParamOpt开启确定性。

## 函数原型

```python
torch_npu.npu_moe_token_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape, *, probs=None, routing_map=None, drop_and_pad=False) -> Tensor
```

## 参数说明

- **permuted_tokens**（`Tensor`）：必选参数，表示经过permute计算后的输入tokens。要求是一个2D的Tensor：drop_and_pad为`False`时shape为(tokens_num \* topK_num, hidden_size)，drop_and_pad为`True`时shape为(experts_num \* capacity, hidden_size)，其中capacity表示每个专家能够处理的token个数。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式支持$ND$，支持非连续Tensor。
- **sorted_indices**（`Tensor`）：必选参数，表示输入输出梯度的映射关系。要求是一个1D的Tensor：drop_and_pad为`False`时shape为(tokens_num \* topK_num)，索引取值范围[0, tokens_num \* topK_num - 1]，允许使用`-1`表示无效槽位，计算时跳过该槽位；drop_and_pad为`True`时shape为(experts_num \* capacity)，索引取值范围[0, tokens_num - 1]。数据类型支持`torch.int32`，数据格式支持$ND$，支持非连续Tensor。
- **restore_shape**（`int[]`）：必选参数，表示输出unpermuted_tokens的shape。size为2，即(tokens_num, hidden_size)。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **probs**（`Tensor`）：可选参数，表示对应位置的token被对应专家处理后的结果在最终结果中的权重，默认值为`None`。要求是一个2D的Tensor，shape与routing_map一致，为(tokens_num, experts_num)。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`：数据类型与permuted_tokens一致，当permuted_tokens的数据类型为`torch.bfloat16`时，probs额外支持`torch.float32`。数据格式支持$ND$，支持非连续Tensor。当probs为`None`时，routing_map不需要传入。
- **routing_map**（`Tensor`）：可选参数，表示对应位置的token是否被对应专家处理，默认值为`None`。要求是一个2D的Tensor，shape为(tokens_num, experts_num)。数据类型支持`torch.int8`、`torch.bool`：数据类型为`torch.int8`时取值支持0、1，数据类型为`torch.bool`时取值支持true、false。数据格式支持$ND$，支持非连续Tensor。当probs为`None`时该参数不需要传入。
- **drop_and_pad**（`bool`）：可选参数，表示填充模式是否开启，默认值为`False`。
  - `False`：表示关闭填充模式。
  - `True`：表示开启填充模式。

## 返回值说明

 `Tensor`

 表示unpermute计算后的正向输出结果。要求是一个2D的Tensor，shape为restore_shape，即(tokens_num, hidden_size)。数据类型与permuted_tokens保持一致。数据格式支持$ND$。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口仅支持图模式。
- 该接口默认非确定性实现，支持通过alcrtCtxSetSysParamOpt开启确定性。
- topK_num要求小于等于512。drop_and_pad为`False`时，每个token最多预留topK_num个专家槽位，routing_map中每行为1或true的个数小于等于topK_num；sorted_indices中允许使用`-1`表示无效槽位。
- 以下场景后续版本会拦截，如果提示warning，建议整改：
  - drop_and_pad为`True`，且topK_num大于experts_num。
  - drop_and_pad为`True`，且capacity大于tokens_num。
  - routing_map的数据类型或shape不符合要求。
  - 输入tensor的数据格式不为ND。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> permuted_tokens = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.float16, device='npu')
    >>> sorted_indices = torch.tensor([0, 1], dtype=torch.int32, device='npu')
    >>> routing_map = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8, device='npu')
    >>> probs = torch.tensor([[1, 0], [0, 1]], dtype=torch.float16, device='npu')
    >>> unpermuted_tokens = torch_npu.npu_moe_token_unpermute_with_routing_map(
    ...     permuted_tokens, sorted_indices, [2, 4], probs=probs, routing_map=routing_map)
    >>> unpermuted_tokens
    tensor([[1., 1., 1., 1.],
            [2., 2., 2., 2.]], device='npu:0', dtype=torch.float16)
    ```

    drop_and_pad模式调用：

    ```python
    >>> permuted_tokens = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.float16, device='npu')
    >>> sorted_indices = torch.tensor([0, 1], dtype=torch.int32, device='npu')
    >>> probs = torch.tensor([[0.5, 0], [0, 0.5]], dtype=torch.float16, device='npu')
    >>> unpermuted_tokens = torch_npu.npu_moe_token_unpermute_with_routing_map(
    ...     permuted_tokens, sorted_indices, [2, 4], probs=probs, routing_map=routing_map, drop_and_pad=True)
    >>> unpermuted_tokens
    tensor([[0.5000, 0.5000, 0.5000, 0.5000],
            [1.0000, 1.0000, 1.0000, 1.0000]], device='npu:0', dtype=torch.float16)
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
        def forward(self, permuted_tokens, sorted_indices, restore_shape, probs=None, routing_map=None, drop_and_pad=False):
            return torch_npu.npu_moe_token_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape,
                                                                      probs=probs, routing_map=routing_map,
                                                                      drop_and_pad=drop_and_pad)
    # 实例化模型model
    model = Model().npu()
    # 从TorchAir获取NPU提供的默认backend
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # 使用TorchAir的backend去调用compile接口编译模型
    model = torch.compile(model, backend=npu_backend)

    # 生成随机数据, 并发送到npu
    permuted_tokens = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.float16).npu()
    sorted_indices = torch.tensor([0, 1], dtype=torch.int32).npu()
    routing_map = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8).npu()
    probs = torch.tensor([[1, 0], [0, 1]], dtype=torch.float16).npu()

    # 调用MoeTokenUnpermuteWithRoutingMap算子
    unpermuted_tokens = model(permuted_tokens, sorted_indices, [2, 4], probs=probs, routing_map=routing_map)
    ```
