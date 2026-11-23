# torch_npu.npu_fusion_attention_v2

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

- API功能：训练场景下，使用FlashAttention算法实现self-attention（自注意力）的融合计算，支持非varlen（BSH、SBH、BSND、BNSD）与varlen（TND）两种场景，支持query/key的RoPE扩展输入（`query_rope, key_rope`）、每个注意力头的偏置sink，以及GQA（grouped-query attention）等特性。
- 计算公式：
  - pse\_type为1（默认值）时，注意力的正向计算公式如下：

    $$
    attention\_out = Dropout(Softmax(Mask(scale*(query*key^T + pse), atten\_mask)), keep\_prob)*value
    $$

  - pse\_type为其他取值时，公式如下：

    $$
    attention\_out = Dropout(Softmax(Mask(scale*(query*key^T) + pse), atten\_mask), keep\_prob)*value
    $$

  - 传入query\_rope、key\_rope时（varlen场景），公式如下：

    $$
    attention\_out = Dropout(Softmax(Mask(scale*(query*key^T + query\_rope*key\_rope^T) + pse), atten\_mask), keep\_prob)*value
    $$

  - 传入sink时，计算逻辑增加sink偏置，主要修改softmax\_max和softmax\_sum的计算部分：

    $$
    S = Q * K^{T}
    $$

    $$
    m = max(sink, max(S))
    $$

    $$
    Attention = \frac{e^{S - m} * V}{\sum e^{S-m} + e^{sink - m}}
    $$

<!-- npu="A3,910b" id12 -->
> [!NOTE]
> <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>上，query\_rope/key\_rope仅在varlen场景（input\_layout为TND）下生效，且要求actual\_seq\_qlen、actual\_seq\_kvlen传入有效值。
<!-- end id12 -->

## 函数原型

```python
torch_npu.npu_fusion_attention_v2(query, key, value, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None, query_rope=None, key_rope=None, scale=1., keep_prob=1., pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None, softmax_layout="", sink=None, dropout_mask=None, seed=0, offset=0) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)
```

## 参数说明

- **query**（`Tensor`）：必选参数，公式中的$query$。维度支持3-4维，数据格式支持$ND$，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`（传入query\_rope/key\_rope时的数据类型分产品约束参见`query_rope`参数说明），需与key、value的数据类型一致。
- **key**（`Tensor`）：必选参数，公式中的$key$。维度支持3-4维，数据格式支持$ND$，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`（传入query\_rope/key\_rope时的数据类型分产品约束参见`query_rope`参数说明），需与query、value的数据类型一致。
- **value**（`Tensor`）：必选参数，公式中的$value$。维度支持3-4维，数据格式支持$ND$，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`（传入query\_rope/key\_rope时的数据类型分产品约束参见`query_rope`参数说明），需与query、key的数据类型一致。
- **head\_num**（`int`）：必选参数，代表单卡的head个数，即输入query的N轴长度。
- **input\_layout**（`str`）：必选参数，代表输入`query`、`key`、`value`的数据排布格式，支持BSH、SBH、BSND、BNSD、TND（不区分大小写）。`input_layout`为TND时即为varlen场景，此时`actual_seq_qlen`/`actual_seq_kvlen`需传值。后续章节如无特殊说明，S表示`query`或`key`、`value`的sequence length，Sq表示query的sequence length，Skv表示`key`、`value`的sequence length，SS表示Sq\*Skv。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **pse**（`Tensor`）：可选参数，公式中的$pse$，位置编码，需与`pse_type`配套使用，默认值为None。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式支持$ND$。
  - 非varlen场景支持四维输入，包含BNSS格式、BN1Skv格式、1NSS格式。
  - 若非varlen场景Sq大于1024或varlen场景、每个batch的Sq与Skv等长且是`sparse_mode`为0、2、3的下三角掩码场景，可开启alibi位置编码压缩，此时只需要输入原始PSE最后1024行进行内存优化，即alibi\_compress = ori\_pse[:, :, -1024:, :]，参数每个batch不相同时输入BNHSkv(H=1024)，每个batch相同时输入1NHSkv(H=1024)。varlen场景压缩的适用情况如下：

    <!-- npu="950" id4 -->
    - <term>Ascend 950PR&950DT系列产品</term>：varlen场景亦支持压缩。
    <!-- end id4 -->
    <!-- npu="A3,910b" id13 -->
    - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：仅非varlen场景支持压缩，且需同时满足Sq大于1024、每个batch的Sq与Skv等长、下三角掩码条件。
    <!-- end id13 -->
  - varlen场景支持BNHSkv(H=1024)、1NHSkv(H=1024)、pseTotalLen三种shape；pseTotalLen为所有batch段pse元素个数之和，第i个batch段的pse元素个数为N \* Sq\_i \* Skv\_i。
  - 当`pse_type`为2或3时，数据类型需为`torch.float32`，对应shape支持范围是\[B, N\]或\[N\]。
- **padding\_mask**（`Tensor`）：可选参数，预留参数，暂未使用，传入None即可。
- **atten\_mask**（`Tensor`）：可选参数，公式中的$atten\_mask$。取值为1（或True）代表该位不参与计算（被遮蔽），取值为0（或False）代表该位参与计算（被保留），数据类型支持`torch.bool`、`torch.uint8`，数据格式支持$ND$，输入shape类型支持BNSS格式、B1SS格式、11SS格式、SS格式。varlen场景只支持SS格式，SS分别是maxSq和maxSkv。
- **query\_rope**（`Tensor`）：可选参数，公式中的$query\_rope$，为query的RoPE扩展输入，需与`key_rope`同时传入，默认值为None。仅在varlen场景（input\_layout为TND）下支持，此时必须传入`atten_mask`。数据格式支持$ND$，shape类型支持\[TND\]。Head-Dim必须满足(qRoPED == kRoPED)、D为8的整数倍且小于等于query、key、value的D。数据类型支持如下：

  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>：与`query`一致（`torch.float16`、`torch.bfloat16`、`torch.float32`）。
  <!-- end id5 -->
  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：仅支持`torch.bfloat16`。
  <!-- end id14 -->
- **key\_rope**（`Tensor`）：可选参数，公式中的$key\_rope$，为key的RoPE扩展输入，需与`query_rope`同时传入，默认值为None。仅在varlen场景（input\_layout为TND）下支持，此时必须传入`atten_mask`。数据类型、数据格式、shape及Head-Dim约束与`query_rope`一致。
- **scale**（`float`）：可选参数，代表缩放系数，作为计算流中Muls的scalar值，默认值为1.。
- **keep\_prob**（`float`）：可选参数，代表Dropout中1的比例，取值范围为(0, 1\]，默认值为1.，表示全部保留。传入query\_rope/key\_rope时keep\_prob必须为1。
- **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示sliding window的左边界，默认值为2147483647。
- **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示sliding window的右边界，默认值为2147483647。`next_tokens`和`pre_tokens`取值与`atten_mask`的关系请参见`sparse_mode`参数，参数取值与`atten_mask`分布不一致会导致精度问题。
- **inner\_precise**（`int`）：可选参数，用于提升精度，默认值为0。当前0、1为保留配置值，2为开启无效行计算，其功能是避免在计算过程中存在整行mask进而导致精度有损失，但是该配置会导致性能下降。如果算子可判断出存在无效行场景，会自动开启无效行计算，例如`sparse_mode`为3、Sq > Skv场景。
- **prefix**（`List[int]`）：可选参数，代表prefix稀疏计算场景每个Batch的N值，默认值为None。数据类型支持`torch.int64`。当Sq > Skv时，prefix的N值取值范围\[0, Skv\]，当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。varlen场景不支持非压缩prefix（即不支持sparse\_mode=5）。
- **actual\_seq\_qlen**（`List[int]`）：可选参数，varlen场景时需要传入此参数，表示`query`每个S的累加和长度，默认值为None。数据类型支持`torch.int64`，长度取值范围为1\~2K，每个元素取值小于等于1M。比如真正的S长度列表为：\[2, 2, 2, 2, 2\]，则`actual_seq_qlen`传：\[2, 4, 6, 8, 10\]。
- **actual\_seq\_kvlen**（`List[int]`）：可选参数，varlen场景时需要传入此参数，表示`key`/`value`每个S的累加和长度，默认值为None。数据类型支持`torch.int64`，长度取值范围为1\~2K，每个元素取值小于等于1M。比如真正的S长度列表为：\[2, 2, 2, 2, 2\]，则`actual_seq_kvlen`传：\[2, 4, 6, 8, 10\]。
- **sparse\_mode**（`int`）：可选参数，表示sparse的模式，默认值为0。当`atten_mask`输入为None时，`sparse_mode`、`pre_tokens`、`next_tokens`参数不生效，固定为全计算。不同模式的原理参见[参考资源](torch_npu-npu_fusion_attention.md#参考资源)。不同取值场景说明如下表：

  | sparse_mode | 含义 | 备注 |
  | --- | --- | --- |
  | 0 | defaultMask模式 | - |
  | 1 | allMask模式 | - |
  | 2 | leftUpCausal模式 | - |
  | 3 | rightDownCausal模式 | - |
  | 4 | band模式 | - |
  | 5 | prefix非压缩模式 | varlen场景不支持 |
  | 6 | prefix压缩模式 | 传入query\_rope/key\_rope时varlen场景不支持 |
  | 7 | varlen外切场景，rightDownCausal模式 | 仅varlen场景支持 |
  | 8 | varlen外切场景，leftUpCausal模式 | 仅varlen场景支持 |

  - varlen场景（input\_layout为TND）支持取值0、1、2、3、4、6、7、8；非varlen场景支持取值0\~6。
  - 当整网的`atten_mask`都相同且shape小于2048\*2048时，建议使用defaultMask模式，来减少内存使用量。

- **gen\_mask\_parallel**（`bool`）：可选参数，DSA生成dropout随机数向量mask的控制开关。默认值为True：同AI Core并行计算；设为False：同AI Core串行计算。
- **sync**（`bool`）：可选参数，在`gen_mask_parallel=True`时控制是否同步等待dropout mask生成完成。默认值为False：dropout mask异步生成；设为True：dropout mask同步生成。
- **pse\_type**（`int`）：可选参数，控制mul与add计算顺序，默认值为1。不同取值含义如下表：

  | pse_type | 含义 | 备注 |
  | --- | --- | --- |
  | 0 | 外部传入pse先mul再add | - |
  | 1 | 外部传入pse先add再mul | 默认值 |
  | 2 | 内部生成pse先mul再add | pse需为`torch.float32`，shape为\[B, N\]或\[N\] |
  | 3 | 内部生成pse先mul再add再sqrt | pse需为`torch.float32`，shape为\[B, N\]或\[N\] |

  - `pse_type`为2或3时，当前只支持Sq和Skv等长。
  - 传入query\_rope/key\_rope时，`pse_type`仅支持取值为1。

- **q\_start\_idx**（`List[int]`）：可选参数，代表外切场景，当前分块的query的sequence在全局中的起始索引，默认值为None。数据类型支持`torch.int64`。
- **kv\_start\_idx**（`List[int]`）：可选参数，代表外切场景，当前分块的key和value的sequence在全局中的起始索引，默认值为None。数据类型支持`torch.int64`。
- **softmax\_layout**（`str`）：可选参数，用于控制TND场景下softmax的输出（softmax\_max和softmax\_sum）的数据排布方式，默认值为""。仅支持传入""和"TND"，且仅当`input_layout`为TND时才可传入"TND"。TND场景下softmax\_max和softmax\_sum的输出格式如下：

  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT系列产品</term>：传入"TND"时，softmax的输出排布为TN8，否则为BNS8。
  <!-- end id6 -->
  <!-- npu="A3,910b" id15 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：传入"TND"时，softmax的输出排布为TND排布，否则为NTD。
  <!-- end id15 -->
- **sink**（`Tensor`）：可选参数，每个注意力头的偏置，默认值为None。shape为\[head\_num\]，数据类型仅支持`torch.float32`。
- **dropout\_mask**（`Tensor`）：可选参数，外部传入的dropout mask，默认值为None。不传入时，由接口内部根据`keep_prob`、`seed`、`offset`自动生成。
- **seed**（`int`）：可选参数，DSA生成dropout mask中Philox算法的seed，默认值为0。返回值中的seed为实际使用的seed，可用于反向计算。
- **offset**（`int`）：可选参数，DSA生成dropout mask中Philox算法的offset，默认值为0。返回值中的offset为实际使用的offset，可用于反向计算。

## 返回值说明

共7个输出，类型依次为**Tensor、Tensor、Tensor、Tensor、int、int、int。**

- **attention\_out**（`Tensor`）：计算公式的最终输出，数据类型和shape类型与query保持一致（BNSD场景输出shape为\[B, N, Sq, Dv\]，TND场景输出shape为\[T, N, Dv\]，其中Dv为value的Head-Dim）。
- **softmax\_max**（`Tensor`）：Softmax计算的Max中间结果，用于反向计算。数据类型为`torch.float32`。非varlen场景shape为\[B, N, Sq, 8\]，varlen场景（TND）shape为\[T, N, 8\]。
- **softmax\_sum**（`Tensor`）：Softmax计算的Sum中间结果，用于反向计算。数据类型为`torch.float32`。非varlen场景shape为\[B, N, Sq, 8\]，varlen场景（TND）shape为\[T, N, 8\]。
- **softmax\_out**（`Tensor`）：预留参数，暂未使用，返回空Tensor。
- **seed**（`int`）：DSA生成dropout mask中Philox算法的seed，返回实际使用的seed，可用于反向计算。
- **offset**（`int`）：DSA生成dropout mask中Philox算法的offset，返回实际使用的offset，可用于反向计算。
- **numels**（`int`）：dropout mask的元素个数。

## 约束说明

- 该接口仅在训练场景下使用。
- 该接口仅支持单算子模式。
- 该接口暂不支持图模式，不支持aclgraph。
- 确定性计算：默认确定性实现。当`keep_prob`小于1时，dropout mask由随机数生成，计算结果的确定性受`seed`、`offset`参数控制。
- 输入`query`、`key`、`value`的维度必须为3维或4维，且`input_layout`必须一致。
- 输入`query`、`key`、`value`的数据类型必须一致。传入query\_rope/key\_rope时：

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型不变，仍支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id7 -->
  <!-- npu="A3,910b" id16 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：仅支持`torch.bfloat16`。
  <!-- end id16 -->
- 输入`pse`时数据类型必须与`query`、`key`、`value`的数据类型一致。
- 输入`key`和`value`的shape必须一致（除Head-Dim外）。有一个例外场景，D不等长场景下，`key`的D跟`value`的D可以不同，但是`value`的D要求小于`key`的D，适用情况如下：

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持该例外场景。
  <!-- end id8 -->
  <!-- npu="A3,910b" id17 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：不支持，`key`和`value`的shape必须完全一致（含Head-Dim）。
  <!-- end id17 -->
- D：Head-Dim必须满足(qD == kD && kD >= vD)，取值范围1\~768。其中D不等长（kD > vD）的适用情况：

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持。
  <!-- end id9 -->
  <!-- npu="A3,910b" id18 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：不支持，须满足qD == kD == vD。
  <!-- end id18 -->
- 关于数据shape的约束（B表示batchsize，N表示head个数，S表示sequence length，T表示B\*S）：
  
  | 场景 | B | N | S | T |
  | --- | --- | --- | --- | --- |
  | 非varlen | 1\~2M（带prefix时最大2K） | 1\~256 | 1\~1M | - |
  | varlen | 1\~20000（带prefix时最大1K）；<term>Ascend 950PR&950DT系列产品</term>为1\~2K | 1\~256 | 1\~1M | 1\~1M |

- varlen场景（input\_layout为TND）：
  - 必须传入`actual_seq_qlen`和`actual_seq_kvlen`，且两者长度相等、不能为空。
  - `atten_mask`输入不支持补pad，即`atten_mask`中不能存在某一行全1的场景。
  - 支持`actual_seq_qlen`中某个Batch上的S长度为0，此时不支持`pse`输入。假设真实的S长度为\[2, 2, 0, 2, 2\]，则传入的`actual_seq_qlen`为\[2, 4, 4, 6, 8\]。
  - 支持尾部部分Batch不参与计算，此时`actual_seq_qlen`和`actual_seq_kvlen`尾部传入对应个数个0即可。假设真实的S长度为\[2, 3, 4, 5, 6\]，此时后两个Batch不参与计算，则传入的`actual_seq_qlen`为\[2, 5, 9, 0, 0\]。
  - `actual_seq_qlen`、`actual_seq_kvlen`的长度取值范围为1\~2K，存在prefix输入时长度最大支持1K。
- 支持输入query的N和key/value的N不相等，但必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1\~256。当Nq/Nkv > 1时，即为GQA（grouped-query attention）；当Nq/Nkv = 1时，即为MHA（multi-head attention）。本文如无特殊说明，N表示的是Nq。
- `keep_prob`取值范围为(0, 1\]。传入query\_rope/key\_rope时，`keep_prob`必须为1（不支持dropout）。
- `sparse_mode`取值约束：
  - `sparse_mode`为1、2、3、4、5、6、7、8时，应传入对应正确的`atten_mask`，否则将导致计算结果错误。
  - `sparse_mode`配置为1、2、3、5、6时，用户配置的`pre_tokens`、`next_tokens`不会生效。
  - `sparse_mode`配置为0、4时，需保证`atten_mask`与`pre_tokens`、`next_tokens`的范围一致。
  - band场景，`pre_tokens`和`next_tokens`之间必须要有交集。
  - `sparse_mode`配置为3时，Sq > Skv的场景自动开启无效行计算。
  - `sparse_mode`配置为7、8时，不支持可选参数`pse`。
- `prefix`稀疏计算场景B不大于32；varlen场景不支持非压缩prefix，即不支持sparse\_mode=5。
- 传入`query_rope`、`key_rope`时：
  - 仅在varlen场景（input_layout为TND）下支持，且必须传入`atten_mask`。
  - qRoPED必须等于kRoPED，且D必须是8的整数倍、小于等于query、key和value的D。

  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT系列产品</term>：输入`query_rope`与`query`的输入shape仅在D维度不同，其他shape参数需要相同。当前只支持：query的D=128，query_rope的D=64这1种场景。
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：输入`key_rope`与`key`的输入shape仅在D维度不同，其他shape参数需要相同。当前只支持：key的D=128，key_rope的D=64这1种场景。
  <!-- end id11 -->
  - 不支持传入`pse`和dropout mask。
- 传入`sink`时，sink的shape必须为\[head\_num\]。
- `input_layout`为TND时，`sparse_mode`取值范围为\[0, 5)或(5, 8\]；`input_layout`为其他取值时，`sparse_mode`取值范围为\[0, 6\]。
- 部分场景下，如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。

## 调用示例

- 单算子模式调用，非varlen场景（BNSD）

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    BNSD = (1, 8, 16, 64)
    query = torch.randn(BNSD, dtype=torch.float16).npu()
    key = torch.randn(BNSD, dtype=torch.float16).npu()
    value = torch.randn(BNSD, dtype=torch.float16).npu()
    atten_out, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention_v2(
        query, key, value, head_num=8, input_layout="BNSD", scale=0.088, keep_prob=0.9)
    print(atten_out.shape, atten_out.dtype)
    print(atten_out[0, 0, 0, :8])
    print(softmax_max.shape, softmax_max.dtype)
    print(softmax_max[0, 0, 0])
    print(softmax_out.shape)
    print(seed, offset, numels)
    ```

    输出如下所示

    ```text
    torch.Size([1, 8, 16, 64]) torch.float16
    tensor([ 0.6816, -0.0999, -0.5698,  0.2695, -0.4182,  0.1345,  0.5288,  0.4639],
           dtype=torch.float16)
    torch.Size([1, 8, 16, 8]) torch.float32
    tensor([1.2028, 1.2028, 1.2028, 1.2028, 1.2028, 1.2028, 1.2028, 1.2028])
    torch.Size([0])
    0 0 2048
    ```

- 单算子模式调用，varlen场景（TND）

    ```python
    import torch
    import torch_npu
    torch.manual_seed(0)
    TND = (8, 2, 64)# B=2，两个sequence长度分别为3和5，则T=8
    query = torch.randn(TND, dtype=torch.float16).npu()
    key = torch.randn(TND, dtype=torch.float16).npu()
    value = torch.randn(TND, dtype=torch.float16).npu()
    atten_out, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention_v2(
        query, key, value, head_num=2, input_layout="TND", scale=0.088,
        actual_seq_qlen=[3, 8], actual_seq_kvlen=[3, 8])
    print(atten_out.shape, atten_out.dtype)
    print(atten_out[:2, :, :4])
    print(softmax_max.shape)
    print(numels)
    ```

    输出如下所示

    ```text
    torch.Size([8, 2, 64]) torch.float16
    tensor([[[ 0.2325, -1.2969, -0.8823,  0.2834],
             [ 0.5112, -0.1545,  0.9160, -0.2747]],
            [[ 0.3386, -0.9658, -0.9204,  0.3330],
             [ 0.2206, -0.4248,  0.4963,  0.6211]]], device='npu:0',
           dtype=torch.float16)
    torch.Size([8, 2, 8])
    68
    ```

- 单算子模式调用，传入`sink`（非varlen场景）

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    BNSD = (1, 8, 16, 64)
    query = torch.randn(BNSD, dtype=torch.float16).npu()
    key = torch.randn(BNSD, dtype=torch.float16).npu()
    value = torch.randn(BNSD, dtype=torch.float16).npu()
    sink = torch.randn(8, dtype=torch.float32).npu()
    atten_out, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention_v2(
        query, key, value, head_num=8, input_layout="BNSD", scale=0.088, sink=sink)
    print(atten_out.shape)
    ```

    输出如下所示

    ```text
    torch.Size([1, 8, 16, 64])
    ```

- 单算子模式调用，传入`query_rope`、`key_rope`（varlen场景）

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    TND = (8, 2, 64)
    query = torch.randn(TND, dtype=torch.bfloat16).npu()
    query_rope = torch.randn(TND, dtype=torch.bfloat16).npu()
    key = torch.randn(TND, dtype=torch.bfloat16).npu()
    key_rope = torch.randn(TND, dtype=torch.bfloat16).npu()
    value = torch.randn(TND, dtype=torch.bfloat16).npu()
    atten_mask = torch.zeros(8, 8, dtype=torch.bool).npu()
    atten_out, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention_v2(
        query, key, value, head_num=2, input_layout="TND", scale=0.088,
        query_rope=query_rope, key_rope=key_rope, atten_mask=atten_mask,
        actual_seq_qlen=[3, 8], actual_seq_kvlen=[3, 8])
    print(atten_out.shape, atten_out.dtype)
    ```

    输出如下所示

    ```text
    torch.Size([8, 2, 64]) torch.bfloat16
    ```
