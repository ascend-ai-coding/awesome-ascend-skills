# torch_npu.npu_sparse_flash_attention

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

- 计算公式：

    $$
    \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
    $$

    其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。对应于`sparse_indices`中的索引值为`b`，表示选择KV序列中从`b * sparse_block_size`开始的一段连续token。将`key/value`按`sparse_indices`收集后，即得到公式中的$\tilde{K},\tilde{V}$。

    当`attention_mode = 2`时，表示MLA-absorb模式。此时原公式中的$Q$和$\tilde{K}$由`nope`部分和`rope`部分拼接而成：

    $$
    Q = [Q_{nope}, Q_{rope}], \quad \tilde{K} = [\tilde{K}_{nope}, \tilde{K}_{rope}]
    $$

    其中，`query/key`提供$Q_{nope}, \tilde{K}_{nope}$，`query_rope/key_rope`提供$Q_{rope}, \tilde{K}_{rope}$。

    不同`sparse_mode`下，公式中$\tilde{K},\tilde{V}$对应的“有效KV集合”有所不同：

    - 当`sparse_mode = 0`时，仅按照`sparse_indices`选择出的KV位置构造$\tilde{K},\tilde{V}$，即只做稀疏选择，不额外施加因果mask。
    - 当`sparse_mode = 3`时，在`sparse_indices`选择出的KV位置基础上，还会继续叠加`rightDownCausal`约束，限定当前query允许访问的因果范围。若当前batch的query和kv有效长度分别为$L_q$和$L_{kv}$，则对第$i$个query位置，允许访问的kv位置$j$为：

      $$
      j \le (L_{kv} - L_q) + i
      $$

    本次公布的`sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

    典型使用流程如下：
    1. 使用选择算法（如`lightning_indexer`）根据Query和Key计算出重要token的索引；
    2. 将索引填充为`sparse_indices`参数所要求的格式；
    3. 将`sparse_indices`传入本API，完成稀疏注意力计算。
    > [!NOTE]
    >
    > `lightning_indexer` 是一种稀疏KV选择算法，用于识别长序列推理中与当前查询最相关的Key/Value区块。它的输出即为本API所需的`sparse_indices`张量。若不使用`lightning_indexer`，用户需自行实现索引选择逻辑，保证`sparse_indices`满足[参数说明](#参数说明)中的格式要求。

## 函数原型

```python
torch_npu.npu_sparse_flash_attention(query, key, value, sparse_indices, scale_value, *, block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None, query_rope=None, key_rope=None, sparse_block_size=1, layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807, attention_mode=0, return_softmax_lse=False, sinks=None) -> (Tensor, Tensor, Tensor)
```

> [!NOTE]
>
> `value`参数为位置参数（必填），类型为`Tensor`（可空），支持传入`None`。`value=None`仅在CANN >= 9.2.0时支持，更低版本会抛出`NOT_SUPPORT`错误。

## 参数说明

> [!NOTE]
>
>- query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- Q\_S和S1表示query shape中的S，KV\_S和S2表示key shape中的S，Q\_N和N1表示num\_query\_heads，KV\_N和N2表示num\_key\_value\_heads，T1表示query shape中的T，T2表示key shape中的输入样本序列长度的累加和。
>
- **query**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`和`torch.float16`。`layout_query`为BSND时shape为[B,S1,N1,D]，当`layout_query`为TND时shape为[T1,N1,D]。
    <!-- npu="A3,910b" id6 -->
    - Atlas A3推理系列产品/Atlas A2推理系列产品的N1支持1/2/4/8/16/32/64/128。
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - Ascend 950PR&950DT系列产品的N1支持1~128。
    <!-- end id7 -->
- **key**（`Tensor`）：必选参数，对应公式中的$\tilde{K}$，在`layout_kv`为PA_BSND时支持0轴非连续，数据格式支持ND，数据类型支持`torch.bfloat16`和`torch.float16`，`layout_kv`为PA_BSND时shape为[block\_num, block\_size, KV\_N, D]，其中block\_num为Paged Attention时block总数，block\_size为一个block的token数，block\_size取值为16的倍数，最大支持1024。`layout_kv`为BSND时shape为[B, S2, KV\_N, D]，`layout_kv`为TND时shape为[T2, KV\_N, D]，其中KV\_N只支持1。

- **value**（`Tensor`）：必选参数（可空），对应公式中的$\tilde{V}$。传入`Tensor`时，在`layout_kv`为PA_BSND时支持0轴非连续，维度N只支持1，数据格式支持ND，数据类型支持`torch.bfloat16`和`torch.float16`，shape与`key`的shape一致。传入`None`时，底层kernel将D和V合并处理，减少冗余访存与计算，提升性能，此时反向传播中`d_value`为空张量。`value=None`仅在CANN >= 9.2.0时支持，更低版本会抛出`NOT_SUPPORT`错误。

- **sparse\_indices**（`Tensor`）：必选参数，代表离散取kvCache的索引，该索引通常由稀疏选择算法（如`lightning_indexer`）生成，具体生成流程见[功能说明](#功能说明)中的典型使用流程。不支持非连续，数据格式支持ND，数据类型支持`torch.int32`。当`layout_query`为BSND时，shape需要传入[B, Q\_S, KV\_N, sparse\_size]，当`layout_query`为TND时，shape需要传入[Q\_T, KV\_N, sparse\_size]，其中sparse\_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse\_size大于0。

- **scale\_value**（`double`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持`double`。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **block\_table**（`Tensor`）：可选参数，表示Paged Attention中kvCache存储使用的block映射表。数据格式支持ND，数据类型支持`torch.int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整。

- **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`torch.int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

- **actual\_seq\_lengths\_kv**（`Tensor`）：可选参数，表示不同Batch中`key`和`value`的有效token数，数据类型支持`torch.int32`。如果不指定，可传入None，表示和key的shape的S长度相同。该参数中每个Batch的有效token数不超过`key/value`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_kv`为TND或PA_BSND时，该入参必须传入，`layout_kv`为TND时，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

- **query\_rope**（`Tensor`）：可选参数，表示MLA结构中的query的rope信息，不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`和`torch.float16`。

- **key\_rope**（`Tensor`）：可选参数，表示MLA结构中的key的rope信息，数据格式支持ND，数据类型支持`torch.bfloat16`和`torch.float16`。

- **sparse\_block\_size**（`int`）：可选参数，代表sparse阶段的block大小，在计算importance score时使用，数据类型支持`torch.int64`，取值范围为[1,128]，且为2的幂次方。
    - sparse_block_size为1时，为Token-wise稀疏化场景，将每个token视为独立单元，在计算重要性分数时，评估每个查询token与每个键值token之间的独立关联程度。
    - sparse_block_size为大于1小于等于128时，为Block-wise稀疏化场景，将token序列划分为固定大小的连续块，以块为单位进行重要性评估，块内token共享相同的稀疏化决策。

- **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，用户不特意指定时可传入默认值"BSND"，支持传入BSND和TND。

- **layout\_kv**（`str`）：可选参数，用于标识输入`key`的数据排布格式，用户不特意指定时可传入默认值"BSND"，支持传入TND、BSND和PA\_BSND，其中PA\_BSND在开启Paged Attention时使用。

- **sparse\_mode**（`int`）：可选参数，表示sparse的模式。数据类型支持`torch.int64`。
    - sparse\_mode为0时，代表全部计算。
    - sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。

- **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`torch.int64`，仅支持默认值9223372036854775807。

- **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`torch.int64`，仅支持默认值9223372036854775807。

- **attention\_mode**（`int`）：可选参数，表示attention的模式，数据类型支持`torch.int64`，仅支持传入2，表示MLA-absorb模式，即计算过程中会将query和key的nope部分分别和query_rope和key_rope的rope部分沿头维度（D）拼接，合并形成最终的query和key用于后续计算，且key和value共享同一份底层张量数据。

- **return\_softmax\_lse**（`bool`）：可选参数，用于表示是否返回softmax_max和softmax_sum。True表示返回，但图模式下不支持，False表示不返回；默认值为False。该参数仅在训练且`layout_kv`不为PA_BSND场景支持。

- **sinks**（`Tensor`）：可选参数，表示attention结构中的sinks信息。传入该参数时调用`aclnnSparseFlashAttentionV2`接口；不传入时调用原`aclnnSparseFlashAttention`接口。数据格式支持ND，数据类型支持`torch.float32`，shape为[N1]。

## 返回值说明

- **attention\_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`torch.bfloat16`和`torch.float16`。当layout\_query为BSND时shape为[B,S1,N1,D]，当layout\_query为TND时shape为[T1,N1,D]。
- **softmax\_max**（`Tensor`）：可选输出，Attention算法对query乘key的结果，取max得到softmax_max，数据类型支持`torch.float`。当layout\_query为BSND时shape为[B,N2,S1,N1/N2]，当layout\_query为TND时shape为[N2,T1,N1/N2]。
- **softmax\_sum**（`Tensor`）：可选输出，Attention算法query乘key的结果减去softmax_max, 再取exp，接着求sum，得到softmax_sum，数据类型支持`torch.float`。当layout\_query为BSND时shape为[B,N2,S1,N1/N2]，当layout\_query为TND时shape为[N2,T1,N1/N2]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 参数query中的D和key、value的D值相等为512，参数query\_rope中的D和key\_rope的D值相等为64。
- 参数query、key、value的数据类型必须保持一致。
- `value=None`仅在CANN >= 9.2.0时支持，更低版本会抛出`NOT_SUPPORT`错误。
- 支持sparse\_block\_size整除block\_size。
- `layout_kv`为PA_BSND时，`layout_query`和`layout_kv`无需一致；`layout_kv`为BSND或TND时，`layout_query`和`layout_kv`需保持一致。

<!-- npu="950" id4 -->
- Ascend 950PR&950DT系列产品：
  - Q\_N支持1~128。
  - sparse\_block\_size仅支持1。
  - 仅在`layout_kv`为PA_BSND时，key、value和key_rope支持0轴非连续。
<!-- end id4 -->
<!-- npu="A3,910b" id5 -->
- Atlas A3推理系列产品/Atlas A2推理系列产品：
  - Q\_N支持1/2/4/8/16/32/64/128。
  - 参数sparse\_block\_size支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparse\_block\_size整除block\_size。
  - 不支持sinks。
  - key、value和key_rope不支持非连续。
<!-- end id5 -->

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random

    query_type = torch.float16
    scale_value = 0.041666666666666664
    sparse_block_size = 1
    sparse_block_count = 2048
    t = 10
    b = 4
    s1 = 1
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    dr = 64
    tile_size = 128
    block_size = 256
    s2_act = 4096
    attention_mode = 2
    return_softmax_lse = False

    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
    key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
    value = key.clone()
    # 注意：此处使用 random.sample 仅为演示目的，生产环境中应使用稀疏选择算法（如 lightning_indexer）生成 sparse_indices
    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
        to(torch.int32)
    query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
    act_seq_q = [s1] * b
    act_seq_kv = [s2_act] * b

    query = query.npu()
    key = key.npu()
    value = value.npu()
    sparse_indices = sparse_indices.npu()
    query_rope = query_rope.npu()
    key_rope = key_rope.npu()
    act_seq_q = torch.tensor(act_seq_q).to(torch.int32).npu()
    act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).npu()

    attention_out, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        query, key, value, sparse_indices, scale_value, block_table=None,
        actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
        layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1,
        attention_mode = attention_mode, return_softmax_lse = return_softmax_lse)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import torch.nn as nn
    import numpy as np
    import random

    query_type = torch.float16
    scale_value = 0.041666666666666664
    sparse_block_size = 1
    sparse_block_count = 2048
    t = 10
    b = 4
    s1 = 1
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    dr = 64
    tile_size = 128
    block_size = 256
    s2_act = 4096
    attention_mode = 2
    return_softmax_lse = False

    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
    key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
    value = key.clone()
    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
        to(torch.int32)
    query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
    act_seq_q = [s1] * b
    act_seq_kv = [s2_act] * b

    query = query.npu()
    key = key.npu()
    value = value.npu()
    sparse_indices = sparse_indices.npu()
    query_rope = query_rope.npu()
    key_rope = key_rope.npu()
    act_seq_q = torch.tensor(act_seq_q).to(torch.int32).npu()
    act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).npu()

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, query, key, value, sparse_indices, scale_value,
            block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
            query_rope, key_rope, sparse_block_size, layout_query, layout_kv,
            sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse):

            attention_out, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
                query, key, value, sparse_indices, scale_value, block_table=None,
                actual_seq_lengths_query=actual_seq_lengths_query, actual_seq_lengths_kv=actual_seq_lengths_kv,
                query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
                layout_query=layout_query, layout_kv=layout_kv, sparse_mode=sparse_mode,
                pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1, attention_mode = attention_mode,
                return_softmax_lse = return_softmax_lse)
            return attention_out, softmax_max, softmax_sum

    mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True)

    attention_out, softmax_max, softmax_sum = mod(query, key, value, sparse_indices,
        scale_value, block_table=None, actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
        layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1,
        attention_mode = attention_mode, return_softmax_lse = return_softmax_lse)
    ```
