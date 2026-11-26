# torch_npu.npu_dense_lightning_indexer_grad_kl_loss

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

- API功能：该接口实现了Lightning Indexer组件warmup阶段训练的反向梯度计算，并融合了Loss的计算。
     - 相比较于`npu_sparse_lightning_indexer_grad_kl_loss`接口，此API适用于稠密场景，其输入`key`、`key_index`不用做稀疏化处理。
     - 与`npu_dense_lightning_indexer_softmax_lse`配合使用，`npu_dense_lightning_indexer_softmax_lse`计算出来的`softmax_max_index`与`softmax_sum_index`作为本接口的输入，通过两个接口配合使用可以优化显存占用。

- 计算公式：

    1. Top-k value的计算公式：
        $$
        I_{t,:}=W_{t,:}@ReLU(\tilde{q}_{t,:}@\tilde{K}_{:t,:}^\top)
        $$
    
        - $W_{t,:}$是$W$矩阵中第$t$个token对应的$weights$；
        - $\tilde{q}_{t,:}$是$\tilde{Q}$矩阵第$t$个token对应的$G$个query头合轴后的结果；
        - $\tilde{K}_{:t,:}$为$t$行$\tilde{K}$矩阵。
  
    2. 正向的Softmax对应公式：

        $$
        p_{t,:} = \text{Softmax}(q_{t,:} @ K_{:t,:}^\top/\sqrt{d})
        $$
        
        - $p_{t,:}$是第$t$个token对应的Softmax结果；
        - $q_{t,:}$是$Q$矩阵第$t$个token对应的$G$个query头合轴后的结果；
        - $K_{:t,:}$为$t$行$K$矩阵。

    3. Lightning Indexer会单独训练，对应的loss function为：
        $$
        Loss{=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
        $$

        其中，$p_{t,:}$是target distribution，通过对main attention score 进行所有的head的求和，然后把求和结果沿着上下文方向进行L1正则化得到。$D_{KL}$为KL散度，其表达式为：

        $$
        D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
        $$

    4. 通过求导可得Loss的梯度表达式：
        $$
        dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
        $$

        利用链式法则可以进行`weights`，`query`和`key`矩阵的梯度计算：
        $$
        dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left) \left) \mathop{{}}\nolimits^{\top}\right. \right. \right. \right.
        $$
    
        $$
        d\mathop{{\tilde{q}}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@\tilde{K}\mathop{{}}\nolimits_{{:t,:}}
        $$
    
        $$
        d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}=\left(dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{\top}@\tilde{q}\mathop{{}}\nolimits_{{:t, :}}\right. \right.
        $$

        其中，$S$为$\tilde{Q}$和$\tilde{K}$矩阵乘的结果，$S_{t,:}$是$S$矩阵中第$t$行。$dW_{t,:}$是$dW$矩阵中第$t$行，$d\mathop{{\tilde{q}}}\nolimits_{{t,:}}$是$d\tilde{Q}$矩阵中第$t$行，$d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}$是$d\tilde{K}$矩阵中$t$行累加值。

## 函数原型

```python
npu_dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, scale_value, *, query_rope=None, key_rope=None, actual_seq_qlen=None, actual_seq_klen=None, layout='BSND', sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **query**(`Tensor`)：必选参数，表示Attention中的query，对应公式中的$Q$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S1, N1, D)$、$(T1, N1, D)$。

- **key**(`Tensor`)：必选参数，表示Attention中的key，对应公式中的$K$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S2, N2, D)$、$(T2, N2, D)$。

- **query_index**(`Tensor`)：必选参数，表示Lightning Indexer正向的输入query，对应公式中的$\tilde{Q}$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S1, N1index, D)$、$(T1, N1index, D)$。

- **key_index**(`Tensor`)：必选参数，表示Lightning Indexer正向的输入key，对应公式中的$\tilde{K}$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S2, N2index, D)$、$(T2, N2index, D)$。

- **weights**(`Tensor`)：必选参数，表示Lightning Indexer的权重系数，对应公式中的$W$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`。shape支持$(B, S1, N1index)$、$(T1, N1index)$。

- **softmax_max**(`Tensor`)：必选参数，表示Attention softmax结果中的最大值。数据格式支持$ND$，数据类型支持`torch.float32`。shape支持$(B, N2, S1, G)$、$(N2, T1, G)$。$G$等于$N1/N2$。

- **softmax_sum**(`Tensor`)：必选参数，表示Attention softmax结果的求和。数据格式支持$ND$，数据类型支持`torch.float32`。shape支持$(B, N2, S1, G)$、$(N2, T1, G)$。$G$等于$N1/N2$。

- **softmax_max_index**(`Tensor`)：必选参数，表示Index attention softmax结果中的最大值。数据格式支持$ND$，数据类型支持`torch.float32`。shape支持$(B, N2index, S1)$、$(N2index, T1)$。

- **softmax_sum_index**(`Tensor`)：必选参数，表示Index attention softmax结果的求和。数据格式支持$ND$，数据类型支持`torch.float32`。shape支持$(B, N2index, S1)$、$(N2index, T1)$。

- **scale_value**(`float`)：必选参数，表示缩放系数，数据类型支持`torch.float32`。

- **query_rope**(`Tensor`)：可选参数，表示MLA结构中的query的rope信息。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S1, N1, Dr)$、$(T1, N1, Dr)$。

- **key_rope**(`Tensor`)：可选参数，表示MLA结构中的key的rope信息。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S2, N2, Dr)$、$(T2, N2, Dr)$。

- **actual_seq_qlen**(`list[int]`)：可选参数，TND场景时需传入此参数。表示query每个S的累加和长度，数据类型支持`torch.int64`，数据格式支持$ND$，默认值为`None`。

- **actual_seq_klen**(`list[int]`)：可选参数，TND场景时需传入此参数。表示key每个S的累加和长度，数据类型支持`torch.int64`，数据格式支持$ND$，默认值为`None`。

- **layout**(`str`)：可选参数，用于标识输入`query`的数据排布格式。当前支持$BSND$、$TND$，默认值为$BSND$。

- **sparse_mode**(`int`)：可选参数，表示sparse的模式，数据类型支持`torch.int32`，默认值为`3`。当前仅支持模式`3`。

- **pre_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和前几个token计算关联。数据类型支持`torch.int64`，默认值2^63-1。

- **next_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和后几个token计算关联。数据类型支持`torch.int64`，默认值2^63-1。

## 返回值说明

- **d\_query\_index**(`Tensor`)：对应公式中的$d\tilde{Q}$，表示`query_index`的梯度，数据类型支持`torch.bfloat16`、`torch.float16`。
- **d\_key\_index**(`Tensor`)：对应公式中的$d\tilde{K}$，表示`key_index`的梯度，数据类型支持`torch.bfloat16`、`torch.float16`。
- **d\_weights**(`Tensor`)：对应公式中的$dW$，表示`weights`的梯度，数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`。
- **loss**(`Tensor`)：对应公式中的$Loss$，表示网络正向输出和目标分布(target distribution)的差异，数据类型支持`torch.float32`。

## 约束说明

- 参数query、key、query_index、key_index的数据类型应保持一致。
- 参数weights不为`torch.float32`时，参数query、key、query_index、key_index、weights的数据类型应保持一致。
- shape数值约束：

| 规格项    | 规格       | 规格说明         |
|-----------|------------|-----------------|
| B         | 1~256      | -               |
| S1、S2    | 1~128K     | S1、S2支持不等长 |
| N1        | 32、64、128| -               |
| N1index   | 8、16、32、64 | -               |
| N2        | 32、64、128| N2=N1           |
| N2index   | 1          | -               |
| D         | 128        | -               |
| Dr        | 64         | -               |

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    def gen_inputs(isTnd):
        B = 1
        N1 = 64
        N2 = N1
        N1_index = 64
        N2_index = 1
        S1 = 128
        S2 = 256
        D = 128
        Dr = 64
        output_dtype = torch.float16
        q = torch.randn(B, S1, N1, D, dtype=output_dtype, device=torch.device('npu'))
        k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))
    
        q_index = torch.randn(B, S1, N1_index, D, dtype=output_dtype, device=torch.device('npu'))
        k_index = torch.randn(B, S2, N2_index, D, dtype=output_dtype, device=torch.device('npu'))
        if Dr != 0:
            q_rope = torch.randn(B, S1, N1, Dr, dtype=output_dtype, device=torch.device('npu'))
            k_rope = torch.randn(B, S2, N2, Dr, dtype=output_dtype, device=torch.device('npu'))
        else:
            q_rope = None
            k_rope = None
        weights = torch.randn(B, S1, N1_index, dtype=output_dtype, device=torch.device('npu'))
        if isTnd:
            q_tnd = q.squeeze(dim=0)
            k_tnd = k.squeeze(dim=0)
            q_index_tnd = q_index.squeeze(dim=0)
            k_index_tnd = k_index.squeeze(dim=0)
            if q_rope is not None:
                q_rope_tnd = q_rope.squeeze(dim=0)
                k_rope_tnd = k_rope.squeeze(dim=0)
            else :
                q_rope_tnd = None
                k_rope_tnd = None
            weights_tnd = weights.squeeze(dim=0)
            
            softmax_max = (torch.randn(N2, S1, 1, dtype=torch.float32, device=torch.device('npu')).abs() + 0.4) * D
            softmax_sum = torch.ones(N2, S1, 1, dtype=torch.float32, device=torch.device('npu'))
            actual_seq_qlen = [S1]
            actual_seq_klen = [S2]
            return q_tnd, k_tnd, q_index_tnd, k_index_tnd, q_rope_tnd, k_rope_tnd, weights_tnd, softmax_max, softmax_sum, actual_seq_qlen, actual_seq_klen
        else :
            softmax_max = (torch.randn(B, N2, S1, 1, dtype=torch.float32, device=torch.device('npu')).abs() + 0.4) * D
            softmax_sum = torch.ones(B, N2, S1, 1, dtype=torch.float32, device=torch.device('npu'))
            actual_seq_qlen = None
            actual_seq_klen = None
            return q, k, q_index, k_index, q_rope, k_rope, weights, softmax_max, softmax_sum, actual_seq_qlen, actual_seq_klen

    input_layout = 'TND'
    isTnd = True
    sparse_mode = 3
    scale = 1.0
    q, k, q_index, k_index, q_rope, k_rope, weights, softmax_max, softmax_sum, actual_seq_qlen, actual_seq_klen = gen_inputs(isTnd)
    softmax_max_index, softmax_sum_index = torch_npu.npu_dense_lightning_indexer_softmax_lse(q_index, k_index, weights,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=input_layout,
            sparse_mode=sparse_mode)
    
    torch_npu.npu_dense_lightning_indexer_grad_kl_loss(
            q, k, q_index, k_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, scale,
            query_rope=q_rope, key_rope=k_rope, actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen, layout=input_layout, sparse_mode=sparse_mode)
    ```
