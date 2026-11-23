# torch_npu.npu_sparse_lightning_indexer_grad_kl_loss

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：该接口实现了npu_lightning_indexer的反向功能，并融合了Loss的计算。npu_lightning_indexer用于筛选Attention的`query`与`key`间最高内在联系的Top-k项，存放在`sparse_indices`中，以减少长序列场景下的Attention计算量，提升训练性能。

- 计算公式：

    1. Top-k value的计算公式：
        $$
        I_{t,:}=W_{t,:}@ReLU(\tilde{q}_{t,:}@\tilde{K}_{:t,:}^\top)
        $$

        - $W_{t,:}$是第$t$个token对应的$weights$；
        - $\tilde{q}_{t,:}$是$Q_{index}$矩阵第$t$个token对应的$G$个头合轴后的结果；
        - $\tilde{K}_{:t,:}$为$K_{index}$矩阵$t$行。

    2. 正向的Softmax对应公式：

        $$
        p_{t,:} = \text{Softmax}(q_{t,:} @ K_{:t,:}^\top/\sqrt{d})
        $$

        - $p_{t,:}$是第$t$个token对应的Softmax结果；
        - $q_{t,:}$是$Q$矩阵第$t$个token对应的$G$个query头合轴后的结果；
        - $K$为$K$矩阵$t$行。

    3. npu_lightning_indexer会单独训练，对应的loss function为：
        $$
        Loss{=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
        $$

        其中，$p_{t,:}$是target distribution，通过对main attention score 进行所有的head的求和，然后把求和结果沿着上下文方向进行sinks修正后的L1正则化得到。$D_{KL}$为KL散度，其表达式为：

        $$
        D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
        $$

    4. 通过求导可得Loss的梯度表达式：
        $$
        dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
        $$

        利用链式法则可以进行weights、query和key矩阵的梯度计算：
        $$
        dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left) \left) \mathop{{}}\nolimits^{\top}\right. \right. \right. \right.
        $$

        $$
        d\mathop{{\tilde{q}}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@\tilde{K}\mathop{{}}\nolimits_{{:t,:}}
        $$

        $$
        d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}=\left(dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{\top}@\tilde{q}\mathop{{}}\nolimits_{{:t, :}}\right. \right.
        $$

        其中，$S$为$Q_{index}$和$K_{index}$矩阵乘的结果。

## 函数原型

```python
npu_sparse_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, sparse_indices, softmax_max, softmax_sum, scale_value, *, query_rope=None, key_rope=None, actual_seq_qlen=None, actual_seq_klen=None, layout='BSND', sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1, sinks=None) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **query**(`Tensor`)：必选参数，表示Attention中的query，对应公式中的$q_{t}$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S1, N1, D)$、$(T1, N1, D)$。

- **key**(`Tensor`)：必选参数，表示Attention中的key，对应公式中的$K_{t}$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S2, N2, D)$、$(T2, N2, D)$。

- **query_index**(`Tensor`)：必选参数，表示lightning_indexer正向的输入`query`，对应公式中的$\tilde{q}_{t}$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S1, N1index, D)$、$(T1, N1index, D)$。

- **key_index**(`Tensor`)：必选参数，表示lightning_indexer正向的输入`key`，对应公式中的$\tilde{K}_{t}$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S2, N2index, D)$、$(T2, N2index, D)$。

- **weights**(`Tensor`)：必选参数，表示lightning_indexer的权重系数，对应公式中的$W_{t}$。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`。shape支持$(B, S1, N1index)$、$(T1, N1index)$。

- **sparse_indices**(`Tensor`)：必选参数，表示排序后`key`和`key_index`的token序号。数据格式支持$ND$，数据类型支持`torch.int32`。shape支持$(B, S1, N2index, topK)$、$(T1, N2index, topK)$。

- **softmax_max**(`Tensor`)：必选参数，表示Attention softmax结果中的最大值。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`。shape支持$(B, N2, S1, G)$、$(N2, T1, G)$。

- **softmax_sum**(`Tensor`)：必选参数，表示Attention softmax结果的求和。数据格式支持$ND$，数据类型支持`torch.float32`。shape支持$(B, N2, S1, G)$、$(N2, T1, G)$。

- **scale_value**(`float`)：必选参数，表示缩放系数，数据类型支持`float`。

- **query_rope**(`Tensor`)：可选参数，表示MLA结构中的query的rope信息。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S1, N1, Dr)$、$(T1, N1, Dr)$。

- **key_rope**(`Tensor`)：可选参数，表示MLA结构中的key的rope信息。数据格式支持$ND$，数据类型支持`torch.bfloat16`、`torch.float16`。shape支持$(B, S2, N2, Dr)$、$(T2, N2, Dr)$。

- **actual_seq_qlen**(`list[int]`)：可选参数，TND场景时需传入此参数。表示query每个S的累加和长度，数据类型支持`torch.int64`，数据格式支持$ND$，默认值为`None`。

- **actual_seq_klen**(`list[int]`)：可选参数，TND场景时需传入此参数。表示key每个S的累加和长度，数据类型支持`torch.int64`，数据格式支持$ND$，默认值为为`None`。

- **layout**(`str`)：可选参数，用于标识输入`query`的数据排布格式。当前支持$BSND$、$TND$，默认值为"BSND"。

- **sparse_mode**(`int`)：可选参数，表示sparse的模式，数据类型支持`torch.int32`，默认值为`3`。

- **pre_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和前几个token计算关联。数据类型支持`torch.int64`，默认值为2^63-1。

- **next_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和后几个token计算关联。数据类型支持`torch.int64`，默认值2^63-1。

- **sinks**(`float`)：可选参数，表示attention结构中的sinks信息，不支持非连续，数据格式支持$ND$，数据类型支持`torch.float32`，shape为$(N1)$。

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>：暂不支持可选输入sinks，传默认值即可
  <!-- end id4 -->

## 返回值说明

- **d\_query\_index**(`Tensor`)：对应公式中的$d\tilde{q}_{t}$，表示`query_index`的梯度，数据类型支持`torch.bfloat16`、`torch.float16`。
- **d\_key\_index**(`Tensor`)：对应公式中的$d\tilde{K}_{t}$，表示`key_index`的梯度，数据类型支持`torch.bfloat16`、`torch.float16`。
- **d\_weights**(`Tensor`)：对应公式中的$dW_{t}$，表示`weights`的梯度，数据类型支持`torch.bfloat16`、`torch.float16`、`torch.float32`。
- **loss**(`Tensor`)：对应公式中的$dI_{t}$，表示网络正向输出和golden值的差异，数据类型支持`torch.float`。

## 约束说明

- 参数query、key、query_index、key_index的数据类型应保持一致。
- 参数weights不为`torch.float32`时，参数query、key、query_index、key_index、weights的数据类型应保持一致。
- 该接口暂不支持图模式调用。
- 规格约束：

  | 规格项     | 规格                                    | 规格说明                  |
  |---------|---------------------------------------|-----------------------|
  | B       | 1~256                                 | -                     |
  | S1、S2   | S1支持1~8K，S2支持1~128K                   | S1、S2支持不等长            |
  | N1      | 32、64、128                             | -                     |
  | N1index | 8、16、32、64                            | -                     |
  | N2      | 1                                       | -                     |
  | N2index | 1                                     | -                     |
  | D       | 512（query/key）/ 128（query_index/key_index） | query与query_index的D可取不同值 |
  | Dr      | 64                                    | -                     |
  | K       | 1024、2048、3081、4096、5120、6144、7168、8192 | -                     |
  | layout  | BSND/TND                              | -                     |

  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>：N1额外支持48，N1index额外支持24，二者仅允许(48，24)组合，禁止其余数值配对；B,S1,S2均支持泛化。
  <!-- end id5 -->

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import math

    def gen_inputs(seqlens_list_array, seqlens_list_kv_array, isTnd):
        B = 1
        NQuery = 64
        NQueryIndex = 64
        N2 = 1
        S1 = 128
        S2 = 128
        topK = 2048
        D = 512
        DIndex = 128
        DR = 64
        output_dtype = torch.float16
        q = torch.randn(B, S1, NQuery, D, dtype=output_dtype, device=torch.device('npu'))
        k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))

        q_index = torch.randn(B, S1, NQueryIndex, DIndex, dtype=output_dtype, device=torch.device('npu'))
        k_index = torch.randn(B, S2, N2, DIndex, dtype=output_dtype, device=torch.device('npu'))
        if DR != 0:
            q_rope = torch.randn(B, S1, NQuery, DR, dtype=output_dtype, device=torch.device('npu'))
            k_rope = torch.randn(B, S2, N2, DR, dtype=output_dtype, device=torch.device('npu'))
        else:
            q_rope = None
            k_rope = None
        weights = torch.randn(B, S1, NQueryIndex, dtype=output_dtype, device=torch.device('npu'))
        a = -0.05  # 最小值
        b = 0.05 # 最大值
        kk = 3.0  # 控制分布范围（3σ 覆盖绝大多数值）
        scale = (b - a) / (2 * kk)
        shift = (a + b) / 2
        weights = weights * scale + shift
        if isTnd:
            sparse_indices = torch.zeros(S1, N2, topK).to(torch.int32).npu()
            tIdx = 0
            for bIdx in range(B):
                for s1Idx in range(seqlens_list_array[bIdx]):
                    s2RealSize = (int)((seqlens_list_kv_array[bIdx] - seqlens_list_array[bIdx]) + s1Idx + 1)
                    if s2RealSize <= 0:
                        s2RealSize = seqlens_list_kv_array[bIdx]

                    if s2RealSize > topK:
                        s2RealLen = topK
                    else:
                        s2RealLen = s2RealSize
                    #处理S2无效行场景，把对应的sparse indices置为-1
                    sparse_indices[tIdx, :, 0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
                    sparse_indices[tIdx, :, s2RealLen : topK] = -1
                    tIdx = tIdx + 1
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

            softmax_max = torch.randn(N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
            softmax_sum = torch.randn(N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
            return q_tnd, k_tnd, q_index_tnd, k_index_tnd, q_rope_tnd, k_rope_tnd, weights_tnd, sparse_indices, softmax_max, softmax_sum
        else :
            sparse_indices = torch.zeros(B, S1, N2, topK).to(torch.int32).npu()
            for s1Idx in range(S1):
                s2RealSize = (int)(S2 - S1 + s1Idx + 1)
                if s2RealSize <= 0:
                    s2RealSize = S2

                if s2RealSize > topK:
                    s2RealLen = topK
                else:
                    s2RealLen = s2RealSize
                sparse_indices[:, s1Idx, 0,  0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
                sparse_indices[:, s1Idx, 0,  s2RealLen : topK] = -1

            softmax_max = torch.randn(B, N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
            softmax_sum = torch.randn(B, N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
            return q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum


   actual_seq_qlen = [128]
   actual_seq_kvlen = [128]
   input_layout = 'TND'
   isTnd = True
   sparse_mode = 3
   scale = 1.0 / math.sqrt(512)
   q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum = gen_inputs(actual_seq_qlen, actual_seq_kvlen, isTnd)

   torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
          q, k, q_index, k_index, weights, sparse_indices, softmax_max, softmax_sum, scale,
          query_rope=q_rope, key_rope=k_rope, actual_seq_qlen=actual_seq_qlen,
          actual_seq_klen=actual_seq_kvlen, layout=input_layout, sparse_mode=sparse_mode,
          pre_tokens=65536, next_tokens=65536, sinks=None)
    ```
