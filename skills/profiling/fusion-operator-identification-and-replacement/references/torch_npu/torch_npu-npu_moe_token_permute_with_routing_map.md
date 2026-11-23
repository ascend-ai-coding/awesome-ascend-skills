# torch_npu.npu_moe_token_permute_with_routing_map

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

- API功能：MoE的permute计算，将token和expert的标签作为routing_map传入，根据routing_map将tokens和可选probs广播后排序。

- 计算公式：tokens_num为routing_map的第0维大小，expert_num为routing_map的第1维大小。

  当drop_and_pad为`False`时：

  $$
  expertIndex=arange(tokens\_num).expand(expert\_num,-1)
  $$

  $$
  sortedIndicesFirst=expertIndex.masked\_select(routingMap.T)
  $$

  $$
  sortedIndicesOut=argsort(sortedIndicesFirst)
  $$

  $$
  topK = numOutTokens // tokens\_num
  $$

  $$
  outToken = topK * tokens\_num
  $$

  $$
  permuteTokensOut[sortedIndicesOut[i]]=tokens[i//topK]
  $$

  如果probs不是None：

  $$
  permuteProbsOutOptional=probsOptional.T.masked\_select(routingMap.T)
  $$

  当drop_and_pad为`True`时：

  $$
  capacity = numOutTokens // expert\_num
  $$

  $$
  outToken = capacity * expert\_num
  $$

  $$
  sortedIndicesOut = argsort[routingMap.T,dim=-1](:, :capacity)
  $$

  $$
  permutedTokensOut = tokens.index\_select(0, sortedIndicesOut)
  $$

  如果probs不是None：

  $$
  probs\_T\_1D = probsOptional.T.view(-1)
  $$

  $$
  indices\_dim0 = arange(expert\_num).view(expert\_num, 1)
  $$

  $$
  indices\_dim1 = sortedIndicesOut.view(expert\_num, capacity)
  $$

  $$
  indices\_1D = (indices\_dim0 * tokens\_num + indices\_dim1).view(-1)
  $$

  $$
  permuteProbsOutOptional = probs\_T\_1D.index\_select(0, indices\_1D)
  $$

> [!NOTE]
>
> - 本接口为确定性计算。
> - 非drop_and_pad模式下，routing_map每行中为1或true的个数要求固定（记为topK），且与num_out_tokens // tokens_num保持一致。

## 函数原型

```python
torch_npu.npu_moe_token_permute_with_routing_map(tokens, routing_map, *, probs=None, num_out_tokens=None, drop_and_pad=False) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **tokens**（`Tensor`）：必选参数，表示输入token特征。要求是一个2D的Tensor，shape为(tokens_num, hidden_size)。支持空tensor。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式支持$ND$，支持非连续Tensor。
- **routing_map**（`Tensor`）：必选参数，表示token到expert的映射关系。要求是一个2D的Tensor，shape为(tokens_num, experts_num)。支持空tensor。数据类型支持`torch.int8`、`torch.bool`：数据类型为`torch.int8`时取值支持0、1，数据类型为`torch.bool`时取值支持true、false。数据格式支持$ND$，支持非连续Tensor。
  - 非drop_and_pad模式下，要求每行中为1或true的个数固定，该个数记为topK。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **probs**（`Tensor`）：可选参数，表示与routing_map对应的概率值，默认值为`None`。支持空tensor。要求元素个数与routing_map相同（shape为(tokens_num, experts_num)）。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`：仅当probs的数据类型为`torch.float32`且tokens的数据类型为`torch.bfloat16`时，probs的数据类型可以不和tokens一致，其他场景probs的数据类型需要和tokens一致。数据格式支持$ND$，支持非连续Tensor。当probs为`None`时，输出permuted_probs为空。

- **num_out_tokens**（`int`）：可选参数，表示有效输出token数，用于计算topK和capacity，默认值为`None`，此时等效于tokens_num。取值范围为[0, tokens_num * experts_num]。
  - 非drop_and_pad模式下，topK = num_out_tokens // tokens_num，实际输出token数outToken = topK * tokens_num。
  - drop_and_pad模式下，capacity = num_out_tokens // experts_num，实际输出token数outToken = capacity * experts_num。

- **drop_and_pad**（`bool`）：可选参数，表示是否开启dropAndPad模式，默认值为`False`。
  - `False`：表示非dropAndPad模式。
  - `True`：表示dropAndPad模式，每个专家按固定capacity对token进行选择，不足的部分进行pad。

## 返回值说明

- **permuted_tokens**（`Tensor`）：根据routing_map进行扩展并排序筛选过的tokens。要求是一个2D的Tensor，shape为(outToken, hidden_size)。数据类型同tokens，数据格式支持$ND$。
- **permuted_probs**（`Tensor`）：根据routing_map进行排序并筛选过的probs。Shape为(outToken)，数据类型同probs。当probs为`None`时，该输出为空。
- **sorted_indices**（`Tensor`）：permuted_tokens和tokens的映射关系。要求是一个1D的Tensor，Shape为(outToken)，数据类型支持`torch.int32`，数据格式支持$ND$。非drop_and_pad模式下，sorted_indices[i]表示tokens的第i // topK行在permuted_tokens中的位置，即permuted_tokens[sorted_indices[i]] = tokens[i // topK]。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口仅支持图模式。
- 该接口为确定性计算。

- 由于float无损转int的限制，tokens_num和experts_num要求小于16777215。
- drop_and_pad为`False`时，routing_map中每行为1或true的个数要求固定且小于512，topK（num_out_tokens // tokens_num）小于512，且topK * tokens_num小于16777215。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> tokens = torch.tensor([[0.1, 0.1, 0.1, 0.1],
    ...                        [0.2, 0.2, 0.2, 0.2],
    ...                        [0.3, 0.3, 0.3, 0.3]], dtype=torch.float16, device='npu')
    >>> routing_map = torch.ones((3, 2), dtype=torch.int8, device='npu')
    >>> permuted_tokens, _, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(tokens, routing_map, num_out_tokens=6)
    >>> permuted_tokens
    tensor([[0.1000, 0.1000, 0.1000, 0.1000],
            [0.2000, 0.2000, 0.2000, 0.2000],
            [0.3000, 0.3000, 0.3000, 0.3000],
            [0.1000, 0.1000, 0.1000, 0.1000],
            [0.2000, 0.2000, 0.2000, 0.2000],
            [0.3000, 0.3000, 0.3000, 0.3000]], device='npu:0', dtype=torch.float16)
    >>> sorted_indices
    tensor([0, 3, 1, 4, 2, 5], device='npu:0', dtype=torch.int32)
    >>> permuted_tokens[sorted_indices]
    tensor([[0.1000, 0.1000, 0.1000, 0.1000],
            [0.1000, 0.1000, 0.1000, 0.1000],
            [0.2000, 0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000, 0.2000],
            [0.3000, 0.3000, 0.3000, 0.3000],
            [0.3000, 0.3000, 0.3000, 0.3000]], device='npu:0', dtype=torch.float16)
    ```

    指定probs进行permute计算：

    ```python
    >>> probs = torch.tensor([[0.5, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float16, device='npu')
    >>> permuted_tokens, permuted_probs, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(
    ...     tokens, routing_map, probs=probs, num_out_tokens=6)
    >>> permuted_probs
    tensor([0.5000, 0.4000, 0.2000, 0.3000, 0.6000, 0.8000], device='npu:0',
           dtype=torch.float16)
    ```

    drop_and_pad模式调用：

    ```python
    >>> permuted_tokens, _, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(
    ...     tokens, routing_map, num_out_tokens=4, drop_and_pad=True)
    >>> permuted_tokens.shape, sorted_indices.shape
    (torch.Size([4, 4]), torch.Size([4]))
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
        def forward(self, tokens, routing_map, probs=None, num_out_tokens=None, drop_and_pad=False):
            return torch_npu.npu_moe_token_permute_with_routing_map(tokens, routing_map, probs=probs,
                                                                    num_out_tokens=num_out_tokens,
                                                                    drop_and_pad=drop_and_pad)
    # 实例化模型model
    model = Model().npu()
    # 从TorchAir获取NPU提供的默认backend
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # 使用TorchAir的backend去调用compile接口编译模型
    model = torch.compile(model, backend=npu_backend)

    # 生成随机数据, 并发送到npu
    tokens = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]],
                          dtype=torch.float16).npu()
    routing_map = torch.ones((3, 2), dtype=torch.int8).npu()

    # 调用MoeTokenPermuteWithRoutingMap算子
    permuted_tokens, _, sorted_indices = model(tokens, routing_map, num_out_tokens=6)
    ```
