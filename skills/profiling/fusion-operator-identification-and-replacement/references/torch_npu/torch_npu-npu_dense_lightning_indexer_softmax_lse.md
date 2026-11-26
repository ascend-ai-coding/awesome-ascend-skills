# torch_npu.npu_dense_lightning_indexer_softmax_lse

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：是`npu_dense_lightning_indexer_grad_kl_loss`接口的前置接口，通过把Lightning Indexer组件的Softmax求最大值和求和运算提前来降低接口的显存占用。

- 计算公式：

$$
res=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(\tilde{Q}@\tilde{K}^T\right)\right)\right)
$$

$$
maxIndex=\text{max}\left(res\right)
$$

$$
sumIndex=\text{ReduceSum}\left(\text{exp}\left(res-maxIndex\right)\right)
$$

$maxIndex$，$sumIndex$作为输出传递给接口npu_dense_lightning_indexer_grad_kl_loss，作为输入用于计算Softmax。

## 函数原型

```python
npu_dense_lightning_indexer_softmax_lse(query_index, key_index, weights, *, actual_seq_qlen=None, actual_seq_klen=None, layout='BSND', sparse_mode=3, pre_tokens=2**63-1, next_tokens=2**63-1) -> (Tensor, Tensor)
```

## 参数说明

- **query_index**(`Tensor`)：必选参数，表示Lightning Indexer正向的输入query，对应公式中的$\tilde{Q}$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S1, N1index, D)$、$(T1, N1index, D)$。

- **key_index**(`Tensor`)：必选参数，表示Lightning Indexer正向的输入key，对应公式中的$\tilde{K}$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S2, N2index, D)$、$(T2, N2index, D)$。

- **weights**(`Tensor`)：必选参数，表示Lightning Indexer的权重系数，对应公式中的$W$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`。shape支持$(B, S1, N1index)$、$(T1, N1index)$。

- **actual_seq_qlen**(`list[int]`)：可选参数，TND场景时需传入此参数。表示query每个S的累加和长度，数据类型支持`torch.int64`，数据格式支持$ND$，默认值为`None`。

- **actual_seq_klen**(`list[int]`)：可选参数，TND场景时需传入此参数。表示key每个S的累加和长度，数据类型支持`torch.int64`，数据格式支持$ND$，默认值为`None`。

- **layout**(`str`)：可选参数，用于标识输入`query_index`的数据排布格式。当前支持$BSND$、$TND$，默认值为$BSND$。

- **sparse_mode**(`int`)：可选参数，表示sparse的模式，数据类型支持`torch.int32`，默认值为`3`。当前仅支持模式`3`。

- **pre_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和前几个token计算关联。数据类型支持`torch.int64`，默认值为$2^{63}-1$。

- **next_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和后几个token计算关联。数据类型支持`torch.int64`，默认值为$2^{63}-1$。

## 返回值说明

- **softmax_max_index**(`Tensor`)：表示softmax计算使用的max值，对应公式中的$maxIndex$，数据格式支持$ND$，数据类型支持`torch.float32`。
- **softmax_sum_index**(`Tensor`)：表示softmax计算使用的sum值，对应公式中的$sumIndex$，数据格式支持$ND$，数据类型支持`torch.float32`。

## 约束说明

- 参数query_index、key_index的数据类型应保持一致。
- 参数weights不为`torch.float32`时，参数query_index、key_index、weights的数据类型应保持一致。
- shape数值约束：

| 规格项    | 规格       | 规格说明         |
|-----------|------------|-----------------|
| B         | 1~256      | -               |
| S1、S2    | 1~128K     | S1、S2支持不等长，当`layout`为BSND时，S1<=S2；`layout`为TND时，`actual_seq_qlen`小于等于`actual_seq_klen`相同索引位置的值，且相同索引位置S1<=S2 |
| N1index   | 8、16、32、64 | -               |
| N2index   | 1          | -               |
| D         | 128        | -               |

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    def gen_inputs(isTnd=False):
        B = 20
        N1 = 32
        N2 = 1
        S1 = 511
        S2 = 2049
        D = 128

        output_dtype = torch.float16
        query_index = torch.randn(B, S1, N1, D, dtype=output_dtype, device=torch.device('npu'))
        key_index = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))
        weights = torch.randn(B, S1, N1, dtype=output_dtype, device=torch.device('npu'))
        if isTnd:
            query_index = query_index.reshape(B*S1, N1, D)
            key_index = key_index.reshape(B*S2, N2, D)
            weights = weights.reshape(B*S1, N1)
            layout = 'TND'
            actual_seq_qlen = [S1*(i+1) for i in range(B)]
            actual_seq_klen = [S2*(i+1) for i in range(B)]
        else :
            layout = 'BSND'
            actual_seq_qlen = None
            actual_seq_klen = None

        sparse_mode = 3
        pre_tokens = 9223372036854775807
        next_tokens = 9223372036854775807

        return query_index, key_index, weights, actual_seq_qlen, actual_seq_klen, layout, sparse_mode, pre_tokens, next_tokens

    query_index, key_index, weights, actual_seq_qlen, actual_seq_klen, layout, sparse_mode, pre_tokens, next_tokens = gen_inputs(isTnd=False)
    softmax_max_index, softmax_sum_index = torch_npu.npu_dense_lightning_indexer_softmax_lse(
        query_index, 
        key_index, 
        weights, 
        actual_seq_qlen=actual_seq_qlen, 
        actual_seq_klen=actual_seq_klen, 
        layout=layout, 
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens)
    ```
