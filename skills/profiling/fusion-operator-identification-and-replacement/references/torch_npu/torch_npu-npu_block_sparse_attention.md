# torch_npu.npu_block_sparse_attention

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

- API功能：BlockSparseAttention，即稀疏注意力计算，支持块级稀疏模式，通过`block_sparse_mask`指定每个Q块选择的KV块，实现高效稀疏注意力。

- 计算公式：稀疏块大小为blockShapeX × blockShapeY，`block_sparse_mask`指定稀疏模式：

$$
attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T) \cdot value_{sparse}
$$

## 函数原型

```python
torch_npu.npu_block_sparse_attention(query, key, value, block_sparse_mask, block_shape, *, q_input_layout='TND', kv_input_layout='TND', num_key_value_heads=1, scale_value=0.0, inner_precise=1, actual_seq_lengths=None, actual_seq_lengths_kv=None, softmax_lse_flag=0) -> (Tensor, Tensor)
```

## 参数说明

- **query** (`Tensor`)：必选参数，表示Attention中的query，对应公式中的$query$。数据格式为$ND$；数据类型支持`torch.float16`、`torch.bfloat16`。
  - TND：shape为`[totalQTokens, headNum, headDim]`；
  - BNSD：shape为`[batch, headNum, maxQSeqLength, headDim]`；

- **key** (`Tensor`)：必选参数，表示Attention中的key，对应公式中的$key$。数据格式为$ND$；数据类型与`query`一致。
  - TND：shape为`[totalKTokens, numKeyValueHeads, headDim]`；
  - BNSD：shape为`[batch, numKeyValueHeads, maxKvSeqLength, headDim]`；

- **value** (`Tensor`)：必选参数，表示Attention中的value，对应公式中的$value$。shape和数据类型与`key`一致。

- **block_sparse_mask** (`Tensor`)：必选参数，块稀疏掩码。shape为`[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]`，表示按块划分后哪些块参与计算（当取值为1则对应的块参与注意力计算，当取值为0则表示不参与）。数据类型为`torch.int8`。

- **block_shape** (`list[int]`)：必选参数，稀疏块shape。至少需要包含两个元素，如`[blockShapeX, blockShapeY]`，且均大于0。blockShapeX：Q方向块大小；blockShapeY：KV方向块大小。
  <!-- npu="A3,910b" id4 -->
  - 在<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term> ：**blockShapeY必须为128的倍数**。
  <!-- end id4 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950DT系列产品</term>：**blockShapeY必须为16的倍数**。
  <!-- end id8 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **q_input_layout** (`str`)：可选参数，`query`的排布，默认值为`"TND"`。当前仅支持`"TND"`、`"BNSD"`。

- **kv_input_layout** (`str`)：可选参数，`key`、`value`的排布，默认值为`"TND"`。当前仅支持`"TND"`、`"BNSD"`，需与`q_input_layout`一致。

- **num_key_value_heads** (`int`)：可选参数，`key`、`value`的head数，默认值为`1`。

- **scale_value** (`float`)：可选参数，缩放系数，默认值为`0.0`，通常设置为$D^{-0.5}$。

- **inner_precise** (`int`)：可选参数，Softmax计算精度，默认值为`1`。`0`表示`torch.float32`中间结果（高精度），`1`表示`torch.float16`中间结果（性能更优），`4`表示混合精度运算。**当`query`/`key`/`value`为`torch.bfloat16`时，仅支持`0`**。
  <!-- npu="950" id5 -->
  - <term>Ascend 950DT系列产品</term>仅支持`4`。
  <!-- end id5 -->
  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>仅支持`0`或`1`。
  <!-- end id9 -->
- **actual_seq_lengths** (`list[int]`)：可选参数，代表每个batch的`query`实际序列长度，用于变长序列场景。
  - **当`q_input_layout`为`"TND"`时必选**：TND下`query`的shape为`[totalQTokens, headNum, headDim]`，若无batch维度，算子无法从shape推断各batch的`query`长度。
  - **当`q_input_layout`为`"BNSD"`时可选**：BNSD下`query`的 shape为`[batch, headNum, maxQSeqLength, headDim]`，不传时算子按shape中的S（maxQSeqLength）作为序列长度处理；传入时按本参数指定的实际长度处理。

- **actual_seq_lengths_kv** (`list[int]`)：可选参数，代表每个batch的`key`/`value`实际序列长度，用于变长序列场景。
  - **当`kv_input_layout`为`"TND"`时必选**：TND下`key`/`value`的shape为`[totalKvTokens, numKeyValueHeads, headDim]`，若无batch维度，算子无法从shape推断各batch的kv长度。
  - **当`kv_input_layout`为`"BNSD"`时可选**：BNSD下`key`/`value`的shape为`[batch, numKeyValueHeads, maxKvSeqLength, headDim]`，不传时算子按shape中的S作为序列长度处理；传入时按本参数指定的实际长度处理。

- **softmax_lse_flag** (`int`)：可选参数，默认值为`0`。`0`表示不输出softmax_lse；`1`表示输出softmax_lse（可能有性能损失）。

## 返回值说明

- **attention_out** (`Tensor`)：公式中的$attentionOut$，与`query`的数据类型和排布一致，最后一维与`value`的headDim一致。
- **softmax_lse** (`Tensor`)：Softmax计算的log-sum-exp中间结果，当`softmax_lse_flag=1`时有效；**数据类型为`torch.float32`**。
  - TND时，shape为`[totalQTokens, headNum, 1]`；
  - BNSD时，shape为`[batch, headNum, maxQSeqLength, 1]`。

## 约束说明

- `query`、`key`、`value`数据类型必须一致，且为`torch.float16`或`torch.bfloat16`。
- `query`的head数$N1$与`key`/`value`的head数$N2$需满足$N1 ≥ N2$且$N1 \% N2 = 0$。
- `actual_seq_lengths`与`actual_seq_lengths_kv`当前必须同时配置或同时不配置，仅配置其中之一会被算子拦截。
- 序列长度不需要被`block_shape`整除，分块数按向上取整计算。
- 正向路径当前支持headDim=64或128；反向路径当前支持headDim=128。
- 反向路径支持`q_input_layout`和`kv_input_layout`同为`"BNSD"`或同为`"TND"`，并支持MHA/GQA场景。MHA场景下$N1 = N2$，GQA场景下需满足$N1 > N2$且$N1 \% N2 = 0$，其中$N1$为`query`的head数，$N2$为`key`/`value`的head数。

## 调用示例

单算子模式调用：

- BNSD 布局

    ```python
    import torch
    import torch_npu

    B, N, S, D = 2, 8, 32, 64
    num_kv_heads = 8
    scale_value = 1.0 / (D ** 0.5)
    block_shape = [128, 128]  # blockShapeY 须为 128 的倍数
    ceil_q = (S + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

    query = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    key = torch.randn(B, num_kv_heads, S, D, dtype=torch.float16).npu()
    value = torch.randn(B, num_kv_heads, S, D, dtype=torch.float16).npu()
    block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8).npu()

    attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
        query, key, value, block_sparse_mask, block_shape,
        q_input_layout="BNSD", kv_input_layout="BNSD",
        num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
    )
    print(attention_out.shape)  # (B, N, S, D)
    ```

- TND 布局

    ```python
    import torch
    import torch_npu

    T, N, D = 32, 8, 64
    num_kv_heads = 8
    scale_value = 1.0 / (D ** 0.5)
    block_shape = [128, 128]  # blockShapeY 须为 128 的倍数
    ceil_q = (T + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (T + block_shape[1] - 1) // block_shape[1]

    query = torch.randn(T, N, D, dtype=torch.float16).npu()
    key = torch.randn(T, num_kv_heads, D, dtype=torch.float16).npu()
    value = torch.randn(T, num_kv_heads, D, dtype=torch.float16).npu()
    block_sparse_mask = torch.ones(1, N, ceil_q, ceil_kv, dtype=torch.int8).npu()

    attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
        query, key, value, block_sparse_mask, block_shape,
        q_input_layout="TND", kv_input_layout="TND",
        num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
        actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
    )
    print(attention_out.shape)  # (T, N, D)
    ```
