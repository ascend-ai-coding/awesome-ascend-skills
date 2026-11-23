# torch\_npu.npu\_fused\_infer\_attention\_score\_v2<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id4 -->

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

- API功能：适配增量&全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。当不涉及system prefix、左padding、kv量化参数合一、pertensor全量化的场景，推荐使用本接口，否则使用老接口`npu_fused_infer_attention_score`。
- 计算公式：

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，$d$表示隐藏层最小的单元尺寸。为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵，$n$为输出矩阵的行数。

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```python
torch_npu.npu_fused_infer_attention_score_v2(query, key, value, *, query_rope=None, key_rope=None, pse_shift=None, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None, block_table=None, dequant_scale_query=None, dequant_scale_key=None, dequant_offset_key=None, dequant_scale_value=None, dequant_offset_value=None, dequant_scale_key_rope=None, quant_scale_out=None, quant_offset_out=None, quant_scale_p=None, learnable_sink=None, num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", sparse_mode=0, block_size=0, query_quant_mode=0, key_quant_mode=0, value_quant_mode=0, inner_precise=0, return_softmax_lse=False, query_dtype=None, key_dtype=None, value_dtype=None, query_rope_dtype=None, key_rope_dtype=None, key_shared_prefix_dtype=None, value_shared_prefix_dtype=None, dequant_scale_query_dtype=None, dequant_scale_key_dtype=None, dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None, out_dtype=None) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

> [!NOTE]
>
> - `query`、`key`、`value`参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示隐藏层的大小、N（Head Num）表示多头数、D（Head Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - Q_S和S1表示`query` shape中的S，KV_S和S2表示`key`和`value` shape中的S，Q_N表示`num_query_heads`，KV_N表示`num_key_value_heads`。
>

**参数快速参考**

| 参数 | 必选/可选 | 类型 | 默认值 | 说明 |
| ------ | ----------- | ------ | -------- | ------ |
| query | 必选 | Tensor | - | Query输入 |
| key | 必选 | Tensor | - | Key输入 |
| value | 必选 | Tensor | - | Value输入 |
| query_rope | 可选 | Tensor | None | MLA结构中`query`的rope信息 |
| key_rope | 可选 | Tensor | None | MLA结构中`key`的rope信息 |
| pse_shift | 可选 | Tensor | None | 位置编码参数 |
| atten_mask | 可选 | Tensor | None | 注意力掩码 |
| actual_seq_qlen | 可选 | List[Int] | None | `query`的有效seqlen |
| actual_seq_kvlen | 可选 | List[Int] | None | `key/value`的有效seqlen |
| block_table | 可选 | Tensor | None | PageAttention的block映射表 |
| dequant_scale_query | 可选 | Tensor | None | `query`的反量化参数 |
| dequant_scale_key | 可选 | Tensor | None | `key`的反量化因子 |
| dequant_offset_key | 可选 | Tensor | None | `key`的反量化偏移 |
| dequant_scale_value | 可选 | Tensor | None | `value`的反量化因子 |
| dequant_offset_value | 可选 | Tensor | None | `value`的反量化偏移 |
| dequant_scale_key_rope | 可选 | Tensor | None | 预留参数，暂未使用 |
| quant_scale_out | 可选 | Tensor | None | 输出的量化因子 |
| quant_offset_out | 可选 | Tensor | None | 输出的量化偏移 |
| quant_scale_p | 可选 | Tensor | None | 预留参数，暂未使用 |
| learnable_sink | 可选 | Tensor | None | 可学习的Sink Token |
| num_query_heads | 可选 | int | 1 | `query`的head个数 |
| num_key_value_heads | 可选 | int | 0 | `key/value`的head个数，0表示与`query`相同 |
| softmax_scale | 可选 | float | 1.0 | 缩放系数，建议传入1/√D |
| pre_tokens | 可选 | int | 2147483647 | 稀疏计算前向Token数 |
| next_tokens | 可选 | int | 2147483647 | 稀疏计算后向Token数 |
| input_layout | 可选 | str | "BSH" | 输入数据排布格式 |
| sparse_mode | 可选 | int | 0 | sparse模式 |
| block_size | 可选 | int | 0 | PageAttention每个block最大token数 |
| query_quant_mode | 可选 | int | 0 | `query`的伪量化方式 |
| key_quant_mode | 可选 | int | 0 | `key`的伪量化方式 |
| value_quant_mode | 可选 | int | 0 | `value`的伪量化方式 |
| inner_precise | 可选 | int | 0 | 精度模式 |
| return_softmax_lse | 可选 | bool | False | 是否输出softmax_lse |
| query_dtype | 可选 | int | None | 预留参数，暂未使用 |
| key_dtype | 可选 | int | None | 表示`key`的数据类型 |
| value_dtype | 可选 | int | None | 表示`value`的数据类型 |
| query_rope_dtype | 可选 | int | None | 预留参数，暂未使用 |
| key_rope_dtype | 可选 | int | None | 预留参数，暂未使用 |
| key_shared_prefix_dtype | 可选 | int | None | 预留参数，暂未使用 |
| value_shared_prefix_dtype | 可选 | int | None | 预留参数，暂未使用 |
| dequant_scale_query_dtype | 可选 | int | None | 预留参数，暂未使用 |
| dequant_scale_key_dtype | 可选 | int | None | 表示`dequant_scale_key`的数据类型 |
| dequant_scale_value_dtype | 可选 | int | None | 表示`dequant_scale_value`的数据类型 |
| dequant_scale_key_rope_dtype | 可选 | int | None | 预留参数，暂未使用 |
| out_dtype | 可选 | int | None | 输出的数据类型 |

- **query**（`Tensor`）：必选参数，表示attention结构的Query输入，对应公式中的`Q`。不支持非连续的Tensor，数据格式支持$ND$。

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch.hifloat8`、`torch.float8_e4m3fn`。
  <!-- end id7 -->

- **key**（`Tensor`）：必选参数，表示attention结构的Key输入，对应公式中的`K`。除MXFP8单算子直调场景外，均不支持非连续的Tensor，数据格式支持$ND$。

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch_npu.int4`（`torch.int32`）。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch.int4`（`torch.int32`）、`torch.hifloat8`、`torch.float8_e4m3fn`、`torch.float4_e2m1fn_x2`。
  <!-- end id9 -->

- **value**（`Tensor`）：必选参数，表示attention结构的Value输入，对应公式中的`V`。除MXFP8单算子直调场景外，均不支持非连续的Tensor，数据格式支持$ND$。

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch_npu.int4`（`torch.int32`）。
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch.int4`（`torch.int32`）、`torch.hifloat8`、`torch.float8_e4m3fn`、`torch.float4_e2m1fn_x2`。
  <!-- end id11 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **query_rope**（`Tensor`）：可选参数，表示MLA（Multi-head Latent Attention）结构中`query`的rope信息，数据类型支持`torch.float16`、`torch.bfloat16`，不支持非连续的Tensor，数据格式支持$ND$。
- **key_rope**（`Tensor`）：可选参数，表示MLA（Multi-head Latent Attention）结构中的`key`的rope信息，数据类型支持`torch.float16`、`torch.bfloat16`，不支持非连续的Tensor，数据格式支持$ND$。
- **pse_shift**（`Tensor`）：可选参数，表示attention结构内部的位置编码参数，数据类型支持`torch.float16`、`torch.bfloat16`，数据类型与`query`数据类型需满足类型推导规则。不支持非连续的Tensor，数据格式支持$ND$。如不使用该功能可传入None。

    - Q\_S大于1，当`pse_shift`为`torch.float16`类型时，要求`query`为`torch.float16`或`torch.int8`类型；当`pse_shift`为`torch.bfloat16`类型时，要求`query`为`torch.bfloat16`类型。输入shape类型需为\(B, Q\_N, Q\_S, KV\_S\)或\(1, Q\_N, Q\_S, KV\_S\)。对于`pse_shift`的KV\_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
    - Q\_S为1，当`pse_shift`为`torch.float16`类型时，要求`query`为`torch.float16`类型；当`pse_shift`为`torch.bfloat16`类型时，要求`query`为`torch.bfloat16`类型。输入shape类型需为\(B, Q\_N, 1, KV\_S\)或\(1, Q\_N, 1, KV\_S\)。对于`pse_shift`的KV\_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。

- **atten_mask**（`Tensor`）：可选参数，对QK结果进行mask，用来指示是否计算Token间的相关性。数据类型支持`torch.bool`、`torch.int8`和`torch.uint8`。不支持非连续的Tensor，数据格式支持$ND$。如不使用该功能可传入None。

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - `sparse_mode`为0、1时：
        - 支持shape传入(1,Q_S,KV_S)、(B,1,Q_S,KV_S)、(1,1,Q_S,KV_S)。
        - 当输入`input_layout`为BSH、BSND、BNSD、BNSD_BSND时，且`query`、`key`、`value`的D相等，并且不传`query_rope`和`key_rope`时，Q_S为1可支持传入(B,KV_S)，Q_S大于1时可支持传入(Q_S,KV_S)。
        - 如果Q\_S、KV\_S非16或32对齐，可以取到向上对齐的值。综合约束请见[Q_S>1约束](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)。
    - `sparse_mode`为2、3、4时，shape输入支持(2048,2048)或(1,2048,2048)或(1,1,2048,2048)。
    - `sparse_mode`为9时：
        - `input_layout`为BSH、BSND或BNSD时，shape输入支持(B, Q_S, Q_S)。
        - `input_layout`为TND时，shape输入支持(∑Q_Si²,)，即每个batch的Q_Si×Q_Si mask拼接为1D tensor。
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - Q_S不为1时建议shape输入(B, Q_S, KV_S)、(1, Q_S, KV_S)、(B, 1, Q_S, KV_S)、(1, 1, Q_S, KV_S)。
    - Q_S为1时建议shape输入(B, 1, KV_S)、(B, 1, 1, KV_S)。
    - 如果Q_S、KV_S非16或32对齐，可以向上取到对齐的S。
  <!-- end id13 -->

- **actual_seq_qlen**（`List[Int]`）：可选参数，表示不同Batch中`query`的有效seqlen，数据类型支持`torch.int64`。默认值为None，表示和`query`的shape的S长度相同。

  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：  
    该入参中每个Batch的有效seqlen不超过`query`中对应batch的seqlen。当seqlen传入长度为1时，每个Batch使用相同seqlen；当seqlen传入长度>=Batch时，取seqlen的前Batch个数；其他长度不支持。当`query`的`input_layout`为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须>=前一个元素的值，且不能出现负值。
  <!-- end id14 -->

- **actual_seq_kvlen**（`List[Int]`）：可选参数，表示不同Batch中`key`/`value`的有效seqlenKv，数据类型支持`torch.int64`。默认值为None，表示和`key`/`value`的shape的S长度相同。不同Q_S值有不同的约束，具体参见[Q_S>1约束](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)和[Q_S=1约束](#zh-cn_topic_0000001832267082_section_qs_eq1_constraint)。
- **block_table**（`Tensor`）：可选参数，表示PageAttention中KV存储使用的block映射表，数据类型支持`torch.int32`。数据格式支持$ND$。如不使用该功能可传入None。
- **dequant\_scale\_query**（`Tensor`）：可选参数，表示`query`的反量化参数。数据格式支持$ND$，如不使用该功能可传入None，综合约束请见[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。

  <!-- npu="A3,910b" id15 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持pertoken叠加perhead。数据类型支持`torch.float32`。
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持per-token叠加per-head和per-block，此时数据类型为`torch.float32`；支持per-channel-group，此时数据类型为`torch.float8_e8m0fnu`。
  <!-- end id16 -->

- **dequant_scale_key**（`Tensor`）：可选参数，kv伪量化参数分离时表示`key`的反量化因子。数据格式支持$ND$。通常支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理scale、pertoken叠加perhead并使用page attention模式管理scale。如不使用该功能可传入None。综合约束请见[Q_S>1约束](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)、[Q_S=1约束](#zh-cn_topic_0000001832267082_section_qs_eq1_constraint)、[GQA伪量化+KV NZ格式约束](#zh-cn_topic_0000001832267082_section_gqa_nz_constraint)和[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。

  <!-- npu="A3,910b" id17 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id17 -->
  <!-- npu="950" id18 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`、`torch.float8_e8m0fnu`。
  <!-- end id18 -->

- **dequant_offset_key**（`Tensor`）：可选参数，kv伪量化参数分离时表示`key`的反量化偏移。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。数据格式支持$ND$。支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理offset、pertoken叠加perhead并使用page attention模式管理offset。如不使用该功能可传入None。综合约束请见[Q_S>1约束](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)、[Q_S=1约束](#zh-cn_topic_0000001832267082_section_qs_eq1_constraint)、[GQA伪量化+KV NZ格式约束](#zh-cn_topic_0000001832267082_section_gqa_nz_constraint)和[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。
- **dequant_scale_value**（`Tensor`）：可选参数，kv伪量化参数分离时表示`value`的反量化因子。数据格式支持$ND$。支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理scale、pertoken叠加perhead并使用page attention模式管理scale。如不使用该功能可传入None，综合约束请见[Q_S>1约束](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)、[Q_S=1约束](#zh-cn_topic_0000001832267082_section_qs_eq1_constraint)、[GQA伪量化+KV NZ格式约束](#zh-cn_topic_0000001832267082_section_gqa_nz_constraint)和[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。

  <!-- npu="A3,910b" id19 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`、`torch.float8_e8m0fnu`。
  <!-- end id20 -->

- **dequant_offset_value**（`Tensor`）：可选参数，kv伪量化参数分离时表示`value`的反量化偏移。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。数据格式支持$ND$。支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理offset、pertoken叠加perhead并使用page attention模式管理offset。如不使用该功能可传入None，综合约束请见[Q_S>1约束](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)、[Q_S=1约束](#zh-cn_topic_0000001832267082_section_qs_eq1_constraint)、[GQA伪量化+KV NZ格式约束](#zh-cn_topic_0000001832267082_section_gqa_nz_constraint)和[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。
- **dequant_scale_key_rope**（`Tensor`）：可选参数，**预留参数，暂未使用，使用默认值即可。**
- **quant_scale_out**（`Tensor`）：可选参数，表示输出的量化因子。数据类型支持`torch.float32`、`torch.bfloat16`。数据格式支持$ND$。支持pertensor、perchannel。当输入为`torch.bfloat16`时，同时支持`torch.float32`、`torch.bfloat16`，否则仅支持`torch.float32`。perchannel格式，当输出layout为BSH时，要求`quant_scale_out`所有维度的乘积等于H；其他layout要求乘积等于Q\_N\*D（建议输出layout为BSH时，`quant_scale_out` shape传入\(1, 1, H\)或\(H,\)；输出为BNSD时，建议传入\(1, Q\_N, 1, D\)或\(Q\_N, D\)；输出为BSND时，建议传入\(1, 1, Q\_N, D\)或\(Q\_N, D\)）。如不使用该功能可传入None，综合约束请见[通用约束](#zh-cn_topic_0000001832267082_section_general_constraint)。
- **quant_offset_out**（`Tensor`）：可选参数，表示输出的量化偏移。数据类型支持`torch.float32`、`torch.bfloat16`。数据格式支持$ND$。支持pertensor、perchannel。若传入`quant_offset_out`，需保证其类型和shape信息与`quant_scale_out`一致。如不使用该功能可传入None，综合约束请见[通用约束](#zh-cn_topic_0000001832267082_section_general_constraint)。
- **quant_scale_p**（`Tensor`）：可选参数，表示P矩阵的量化因子。

  <!-- npu="A3,910b" id21 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：预留参数，暂未使用，使用默认值即可。
  <!-- end id21 -->
  <!-- npu="950" id22 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float32`。数据格式支持ND。支持per-tensor，shape必须是(1)。如不使用该功能可传入None。
  <!-- end id22 -->

- **learnable_sink**（`Tensor`）：可选参数，表示通过可学习的“Sink Token”起到吸收Attention Score的作用，数据类型支持`torch.bfloat16`，数据格式支持$ND$，shape输入为(Q_N,)。默认值为None，综合约束请见[learnable_sink约束](#zh-cn_topic_0000001832267082_section_learnable_sink_constraint)。

- **num_query_heads**（`int`）：可选参数，代表`query`的head个数，数据类型支持`torch.int64`，在BNSD场景下，需要与shape中的`query`的N轴shape值相同，否则执行异常。综合约束请见[GQA伪量化+KV NZ格式约束](#zh-cn_topic_0000001832267082_section_gqa_nz_constraint)。
- **num_key_value_heads**（`int`）：可选参数，代表`key`、`value`中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景，数据类型支持`torch.int64`。默认值为0，表示`key`/`value`/`query`的head个数相等，需要满足`num_key_value_heads`整除`num_query_heads`，`num_query_heads`与`num_key_value_heads`的比值不能大于64。在BSND、BNSD、BNSD\_BSND（仅支持Q\_S大于1）场景下，还需要与shape中的`key`/`value`的N轴shape值相同，否则执行异常。综合约束请见[GQA伪量化+KV NZ格式约束](#zh-cn_topic_0000001832267082_section_gqa_nz_constraint)。
- **softmax_scale**（`float`）：可选参数，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持`torch.float32`。数据类型与`query`数据类型需满足数据类型推导规则。默认值为1.0，即不做缩放。**建议传入`1/√D`（D为Head Dim）**，例如当D=128时传入`1/math.sqrt(128.0)`，以获得正确的注意力计算结果。
- **pre_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`torch.int64`。默认值为2147483647，Q\_S为1时该参数无效。
- **next_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`torch.int64`。默认值为2147483647，Q\_S为1时该参数无效。
- **input_layout**（`str`）：可选参数，用于标识输入`query`、`key`、`value`的数据排布格式，默认值为"BSH"。

    > [!NOTE]
    > 注意排布格式带下划线时，下划线左边表示输入`query`的layout，下划线右边表示输出output的格式，算子内部会进行layout转换。

    <!-- npu="A3,910b" id23 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持BSH、BSND、BNSD、BNSD\_BSND（输入为BNSD时，输出格式为BSND，仅支持Q\_S大于1）、BSH\_NBSD、BSND\_NBSD、BNSD\_NBSD（输出格式为NBSD时，仅支持Q\_S大于1且小于等于16）、TND、TND\_NTD、NTD\_TND（TND相关场景综合约束请见[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)）。其中BNSD\_BSND含义指当输入为BNSD，输出格式为BSND，仅支持Q\_S大于1。
    <!-- end id23 -->
    <!-- npu="950" id24 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持BSH、BSND、BNSD、BNSD\_BSND（输入为BNSD时，输出格式为BSND，仅支持Q\_S大于1）、BSH\_NBSD、BSND\_NBSD、BNSD\_NBSD（输出格式为NBSD时，仅支持Q\_S大于1且小于等于16）、TND、NTD、TND\_NTD、NTD\_TND（TND相关场景综合约束请见[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)）。
    <!-- end id24 -->

    | `input_layout` | `query` shape | `key` shape | `value` shape | 输出(`attention_out`) shape | 说明 |
    | ------------- | ----------- | --------- | ----------- | ---------- | ---- |
    | BSH | (B, Q\_S, H) | (B, KV\_S, H) | (B, KV\_S, H) | (B, Q\_S, H) | H=N\*D |
    | BSND | (B, Q\_S, Q\_N, D) | (B, KV\_S, KV\_N, D) | (B, KV\_S, KV\_N, D) | (B, Q\_S, Q\_N, D) | N和D分开 |
    | BNSD | (B, Q\_N, Q\_S, D) | (B, KV\_N, KV\_S, D) | (B, KV\_N, KV\_S, D) | (B, Q\_N, Q\_S, D) | N和D分开，N在前 |
    | BNSD\_BSND | (B, Q\_N, Q\_S, D) | (B, KV\_N, KV\_S, D) | (B, KV\_N, KV\_S, D) | (B, Q\_S, Q\_N, D) | 输入BNSD，输出BSND，仅Q\_S>1 |
    | BSH\_NBSD | (B, Q\_S, H) | (B, KV\_S, H) | (B, KV\_S, H) | (Q\_N, B, Q\_S, D) | 输入BSH，输出NBSD |
    | BSND\_NBSD | (B, Q\_S, Q\_N, D) | (B, KV\_S, KV\_N, D) | (B, KV\_S, KV\_N, D) | (Q\_N, B, Q\_S, D) | 输入BSND，输出NBSD |
    | BNSD\_NBSD | (B, Q\_N, Q\_S, D) | (B, KV\_N, KV\_S, D) | (B, KV\_N, KV\_S, D) | (Q\_N, B, Q\_S, D) | 输入BNSD，输出NBSD，仅支持Q\_S大于1且小于等于16 |
    | TND | (T, Q\_N, D) | (T, KV\_N, D) | (T, KV\_N, D) | (T, Q\_N, D) | T为所有Batch的S累加和 |
    | NTD | (Q\_N, T, D) | (KV\_N, T, D) | (KV\_N, T, D) | (Q\_N, T, D) | T为所有Batch的S累加和 |
    | TND\_NTD | (T, Q\_N, D) | (T, KV\_N, D) | (T, KV\_N, D) | (Q\_N, T, D) | 输入TND，输出NTD |
    | NTD\_TND | (Q\_N, T, D) | (KV\_N, T, D) | (KV\_N, T, D) | (T, Q\_N, D) | 输入NTD，输出TND |

- **sparse_mode**（`int`）：可选参数，表示sparse的模式，默认值为0。数据类型支持`torch.int64`。Q\_S为1且不带rope输入时该参数无效。`input_layout`为TND、TND\_NTD、NTD\_TND时，综合约束请见[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)；GQA伪量化场景下综合约束请见[GQA伪量化+KV NZ格式约束](#zh-cn_topic_0000001832267082_section_gqa_nz_constraint)。取值5、6、7、8（分别代表prefix、global、dilated、block\_local）暂未实现，请勿使用。

    <!-- npu="A3,910b" id25 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
        - 当前仅支持取值0、1、2、3、4、9。
    <!-- end id25 -->
    <!-- npu="950" id26 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
        - 当前仅支持取值0、1、2、3、4。
    <!-- end id26 -->

    | 取值 | 模式名称 | 说明 | `atten_mask`要求 |
    | ------ | ---------- | ------ | ----------------- |
    | 0 | defaultMask | 如果`atten_mask`未传入则不做mask操作，忽略`pre_tokens`和`next_tokens`（内部赋值为INT\_MAX）；如果传入，则需要传入完整的`atten_mask`矩阵（S1\*S2），表示`pre_tokens`和`next_tokens`之间的部分需要计算 | 可选 |
    | 1 | allMask | 必须传入完整的`atten_mask`矩阵（S1\*S2） | 必须传入(S1\*S2) |
    | 2 | leftUpCausal | 左上角因果模式的mask | 优化后的`atten_mask`矩阵(2048\*2048) |
    | 3 | rightDownCausal | 右下角因果模式的mask，对应以右顶点为划分的下三角场景 | 优化后的`atten_mask`矩阵(2048\*2048) |
    | 4 | band | band模式的mask | 优化后的`atten_mask`矩阵(2048\*2048) |
    | 9 | treeMask | 推测解码场景的树形注意力掩码。仅MLA场景（`query_rope`和`key_rope`不为空）支持。不支持左padding、`pse_shift`、sharedPrefix，输出dtype不支持`torch.int8`，每个batch需满足Q\_S ≤ KV\_S | 需传入自定义tree mask |

- **block_size**（`int`）：可选参数，表示PageAttention中KV存储每个block中最大的token个数，默认为0，数据类型支持`torch.int64`。
- **query_quant_mode**（`int`）：可选参数， 表示`query`的伪量化方式。

  <!-- npu="A3,910b" id27 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持传入3，代表模式3：pertoken叠加perhead模式。
  <!-- end id27 -->
  <!-- npu="950" id28 -->
  - <term>Ascend 950PR&950DT系列产品</term>：代表模式3：per-token叠加per-head模式。代表模式6：per-token-group全量化模式。代表模式7：FP8 per-block全量化模式。
  <!-- end id28 -->

- **key_quant_mode**（`int`）：可选参数，表示`key`的伪量化方式，默认值为0。取值除了`key_quant_mode`为0且`value_quant_mode`为1、`key_quant_mode`为6且`value_quant_mode`为8的场景外，其他场景取值需要与`value_quant_mode`一致。综合约束请见[Q_S>1约束](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)和[Q_S=1约束](#zh-cn_topic_0000001832267082_section_qs_eq1_constraint)。

    <!-- npu="A3,910b" id29 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当Q\_S>=2时，仅支持传入值为0、1；当Q\_S=1时，支持取值0、1、2、3、4、5。
    <!-- end id29 -->
    <!-- npu="950" id30 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持取值0、1、2、3、4、5、6、7。
    <!-- end id30 -->

    | 取值 | 说明 |
    | ------ | ------ |
    | 0 | perchannel模式（perchannel包含pertensor） |
    | 1 | pertoken模式 |
    | 2 | pertensor叠加perhead模式 |
    | 3 | pertoken叠加perhead模式 |
    | 4 | pertoken叠加使用page attention模式管理scale/offset模式 |
    | 5 | pertoken叠加perhead并使用page attention模式管理scale/offset模式 |
    | 6 | per-token-group模式 |
    | 7 | FP8 per-block全量化模式 |
    | 8 | per-channel-group全量化模式 |

- **value_quant_mode**（`int`）：可选参数，表示`value`的伪量化方式，模式编号与`key_quant_mode`一致，默认值为0。取值除了`key_quant_mode`为0且`value_quant_mode`为1、`key_quant_mode`为6且`value_quant_mode`为8的场景外，其他场景取值需要与`key_quant_mode`一致。综合约束请见[Q_S>1约束](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)、[Q_S=1约束](#zh-cn_topic_0000001832267082_section_qs_eq1_constraint)和[GQA伪量化+KV NZ格式约束](#zh-cn_topic_0000001832267082_section_gqa_nz_constraint)。

    <!-- npu="A3,910b" id31 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当Q\_S>=2时，仅支持传入值为0、1；当Q\_S=1时，支持取值0、1、2、3、4、5。
    <!-- end id31 -->
    <!-- npu="950" id32 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持取值0、1、2、3、4、5、6、7、8。
    <!-- end id32 -->

- **inner_precise**（`int`）：可选参数，数据类型支持`torch.int64`，支持4种模式：0、1、2、3。一共两位bit位，第0位（bit0）表示高精度或者高性能选择，第1位（bit1）表示是否做行无效修正。当Q\_S\>1时，`sparse_mode`为0或1，并传入用户自定义mask的情况下，建议开启行无效；Q\_S为1时该参数仅支持取0和1。综合约束请见[Q_S>1约束](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)、[Q_S=1约束](#zh-cn_topic_0000001832267082_section_qs_eq1_constraint)和[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。

    - `inner_precise`为0时，代表开启高精度模式，且不做行无效修正。
    - `inner_precise`为1时，代表高性能模式，且不做行无效修正。
    - `inner_precise`为2时，代表开启高精度模式，且做行无效修正。
    - `inner_precise`为3时，代表高性能模式，且做行无效修正。

    > [!NOTE]
    > `torch.bfloat16`和`torch.int8`不区分高精度和高性能，行无效修正对`torch.float16`、`torch.bfloat16`和`torch.int8`均生效。当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来开启行无效功能以提升精度，但是该配置会导致性能下降。

- **return_softmax_lse**（`bool`）：可选参数，表示是否输出`softmax_lse`，支持S轴外切（增加输出）。true表示输出，false表示不输出；默认值为false。
- **query_dtype**（`int`）：可选参数，表示`query`的数据类型，**预留参数，暂未使用，使用默认值即可。**
- **key_dtype**（`int`）：可选参数，表示`key`的数据类型。

  <!-- npu="A3,910b" id33 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：预留参数，暂未使用，使用默认值即可。
  <!-- end id33 -->
  <!-- npu="950" id34 -->
  - <term>Ascend 950PR&950DT系列产品</term>：`key`的数据类型为`torch_npu.float4_e2m1fn_x2`、`torch_npu.hifloat8`时，需要传值。
  <!-- end id34 -->

- **value_dtype**（`int`）：可选参数，表示`value`的数据类型。

  <!-- npu="A3,910b" id35 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：预留参数，暂未使用，使用默认值即可。
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - <term>Ascend 950PR&950DT系列产品</term>：`value`的数据类型为`torch_npu.float4_e2m1fn_x2`、`torch_npu.hifloat8`时，需要传值。
  <!-- end id36 -->

- **query_rope_dtype**（`int`）：可选参数，表示`query_rope`的数据类型，**预留参数，暂未使用，使用默认值即可。**
- **key_rope_dtype**（`int`）：可选参数，表示`key_rope`的数据类型，**预留参数，暂未使用，使用默认值即可。**
- **key_shared_prefix_dtype**（`int`）：可选参数，表示key_shared_prefix的数据类型，**预留参数，暂未使用，使用默认值即可。**
- **value_shared_prefix_dtype**（`int`）：可选参数，表示value_shared_prefix的数据类型，**预留参数，暂未使用，使用默认值即可。**
- **dequant_scale_query_dtype**（`int`）：可选参数，表示`dequant_scale_query`的数据类型，**预留参数，暂未使用，使用默认值即可。**
- **dequant_scale_key_dtype**（`int`）：可选参数，表示`dequant_scale_key`的数据类型。

  <!-- npu="A3,910b" id37 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：预留参数，暂未使用，使用默认值即可。
  <!-- end id37 -->
  <!-- npu="950" id38 -->
  - <term>Ascend 950PR&950DT系列产品</term>：`dequant_scale_key`的数据类型为`torch.float8_e8m0fnu`时，需要传值。
  <!-- end id38 -->

- **dequant_scale_value_dtype**（`int`）：可选参数，表示`dequant_scale_value`的数据类型。

  <!-- npu="A3,910b" id39 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：预留参数，暂未使用，使用默认值即可。
  <!-- end id39 -->
  <!-- npu="950" id40 -->
  - <term>Ascend 950PR&950DT系列产品</term>：`dequant_scale_value`的数据类型为`torch.float8_e8m0fnu`时，需要传值。
  <!-- end id40 -->

- **dequant_scale_key_rope_dtype**（`int`）：可选参数，表示`dequant_scale_key_rope`的数据类型，**预留参数，暂未使用，使用默认值即可。**
- **out_dtype**（`int`）：可选参数，表示输出的数据类型。当输入为`torch.int8`或`torch.float8_e4m3fn`时，可通过该参数指定输出的数据类型（如`torch.float8_e5m2`）。如不使用该功能可传入None。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

- **attention_out**（`Tensor`）：公式中的输出，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。数据格式支持$ND$。限制：输出的D维度需要与`value`的D保持一致，其余维度需要与入参`query`的shape保持一致。
- **softmax_lse**（`Tensor`）：ring attention算法对`query`乘`key`的结果先取max得到softmax\_max，`query`乘`key`的结果减去softmax\_max，再取exp，最后取sum，得到softmax\_sum，最后对softmax\_sum取log，再加上softmax\_max得到的结果。数据类型支持`torch.float32`，当`return_softmax_lse`为True时，一般情况下输出shape为\(B, Q\_N, Q\_S, 1\)，若`input_layout`为TND/NTD\_TND时，输出shape为\(T,Q\_N,1\)；当`return_softmax_lse`为False时，输出shape为\[1\]的值为0的Tensor。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

> [!NOTICE]
> 约束说明按场景组织，可按需查阅：
>
> - [**通用约束**](#zh-cn_topic_0000001832267082_section_general_constraint)：入参为空处理、`key/value` shape一致性、`torch.int8`量化限制。
> - [**MLA场景约束**](#zh-cn_topic_0000001832267082_section_mla_constraint)（`query_rope`和`key_rope`输入时）：D=512约束、D=128约束、TND场景约束。
> - [**GQA伪量化+KV NZ格式约束**](#zh-cn_topic_0000001832267082_section_gqa_nz_constraint)：KV NZ输入格式、dequant_scale约束、`sparse_mode`限制、`num_query_heads/num_key_value_heads`组合限制。
> - [**learnable_sink约束**](#zh-cn_topic_0000001832267082_section_learnable_sink_constraint)：入参`learnable_sink`在使用时的场景限制。
> - [**Q_S>1（全量推理）约束**](#zh-cn_topic_0000001832267082_section_qs_gt1_constraint)：输入shape限制、`sparse_mode`限制、page attention限制、量化限制、`pse_shift`限制、kv伪量化参数分离。
> - [**Q_S=1（增量推理）约束**](#zh-cn_topic_0000001832267082_section_qs_eq1_constraint)：输入shape限制、page attention限制、kv伪量化参数分离。

### 通用约束<a name="zh-cn_topic_0000001832267082_section_general_constraint"></a>

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 入参为空的处理：算子内部需要判断参数`query`是否为空，如果是空则直接返回空。参数`query`不为空Tensor，参数`key`、`value`为空Tensor（即S2为0），则`attention_out`按照对应shape大小返回全0。`attention_out`为空Tensor时，返回空。
- 参数`key`、`value`中对应tensor的shape需要完全一致；非连续场景下`key`、`value`的tensorlist中的batch只能为1，个数等于`query`的B，N和D需要相等。
- `torch.int8`量化相关入参数量与输出数据格式的综合限制：
    - 输出为`torch.int8`的场景：入参`quant_scale_out`需要存在，`quant_offset_out`可选，不传时默认为0。

        <!-- npu="A3,910b" id41 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：输入为`torch.int8`。
        <!-- end id41 -->

    - 输出为`torch.float16`的场景：若存在入参`quant_offset_out`或`quant_scale_out`（即不为None），则报错并返回。
    - 入参`quant_offset_out`和`quant_scale_out`支持pertensor或perchannel格式，数据类型支持`torch.float32`、`torch.bfloat16`。

### MLA场景约束<a name="zh-cn_topic_0000001832267082_section_mla_constraint"></a>

- `query_rope`和`key_rope`输入时即为MLA场景，参数约束如下：
    - `query_rope`的数据类型、数据格式与`query`一致。
    - `key_rope`的数据类型、数据格式与`key`一致。
    - `query_rope`和`key_rope`要求同时配置或同时不配置，不支持只配置其中一个。
    - 当`query_rope`和`key_rope`非空时，`query`的D只支持512、128；
        - 当`query`的D等于512时：

            <!-- npu="A3,910b" id42 -->
            - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
                - sparse：支持0/3/4/9；
                - `query_rope`配置时要求`query`的N为1/2/4/8/16/32/64/128，`query_rope`的shape中D为64，其余维度与`query`一致；
                - `key_rope`配置时要求`key`的N为1、D为512，`key_rope`的shape中D为64，其余维度与`key`一致；
                - 支持`key`、`value`、`key_rope`的数据格式为$ND$或$NZ$。当数据格式为$NZ$时，若数据类型为`torch.float16`或`torch.bfloat16`，输入参数`key`和`value`的格式为\[blockNum, KV\_N, D/16, blockSize, 16\]；若输入数据类型为`torch.int8`，输入参数`key`和`value`的格式为\[blockNum, KV\_N, D/32, blockSize, 32\]；
                - `input_layout`形状支持BSH、BSND、BNSD、BNSD\_NBSD、BSND\_NBSD、BSH\_NBSD、TND、TND\_NTD；
                - 支持开启page attention，此时`block_size`支持16的倍数且不大于1024；
                - 不支持开启左padding、tensorlist、pse、prefix、伪量化、后量化、空Tensor。
                - 支持全量化场景，即输入`query`/`key`/`value`全为`torch.int8`，`query_rope`和`key_rope`为`torch.bfloat16`，输出为`torch.bfloat16`的场景：
                    - 入参`dequant_scale_query`、`dequant_scale_key`、`dequant_scale_value`需要同时存在，且其数据类型仅支持`torch.float32`。
                    - 不支持传入`quant_scale_out`、`quant_offset_out`、`dequant_offset_key`、`dequant_offset_value`，否则报错并返回。
                    - `query_quant_mode`仅支持pertoken叠加perhead模式，`key_quant_mode`和`value_quant_mode`仅支持pertensor模式。
                    - 支持`key`、`value`、`key_rope`的`input_layout`格式为$NZ$。
            <!-- end id42 -->
            <!-- npu="950" id43 -->
            - <term>Ascend 950PR&950DT系列产品</term>：
                - sparse：Q_S等于1时只支持sparse=0且不传mask，Q_S大于1时只支持sparse=3且传入mask。
                - `query_rope`配置时要求`query`的s为1-16、fp8全量化场景n支持32、64、128，`torch.int8`场景n支持1、2、4、8、16、32、64、128，d为512，`query_rope`的shape中b、n、s与`query`一致，d为64。
                - `key_rope`配置时要求`key`的n为1，d为512，`key_rope`的shape中b、n、s与`key`一致，d为64。
                - sparse：支持sparse=0且不传mask、sparse=3且传入mask。
                - `key`&`value`支持$ND$输入，`torch.int8`全量化场景仅支持PANZ输入。
                - `input_layout`：BSH、BSND、BNSD、BNSD\_BSND、TND、TND\_NTD。
                - fp8全量化支持`input_layout`：BSH、BSND、BNSD、TND。
                - `torch.int8`全量化支持`input_layout`：BSH、BSND、TND、BSH\_NBSD、BSND\_NBSD、TND\_NTD。
                - 支持`actual_seq_qlen`、`actual_seq_kvlen`参数。
            <!-- end id43 -->

        - 当`query`的D等于128时：

            <!-- npu="A3,910b" id44 -->
            - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
                - `input_layout`：BSH、BSND、TND、BNSD、NTD、BSH\_BNSD、BSND\_BNSD、BNSD\_BSND、NTD\_TND。
                - `query_rope`配置时要求`query_rope`的shape中D为64，其余维度与`query`一致。
                - `key_rope`配置时要求`key_rope`的shape中D为64，其余维度与`key`一致。
                - 不支持开启左padding、tensorlist、pse、prefix、伪量化、全量化、后量化、空Tensor。
                - 其余约束同TND、NTD\_TND场景下的综合限制保持一致。
            <!-- end id44 -->
            <!-- npu="950" id45 -->
            - <term>Ascend 950PR&950DT系列产品</term>：
                - `query_rope`配置时要求`query_rope`的shape中b、n、s与`query`一致，d为64。
                - `key_rope`配置时要求`key_rope`的shape中b、n、s与`key`一致，d为64。
                - `input_layout`：BSH、BSND、BNSD、BNSD\_BSND、TND、NTD、NTD\_TND。
                - 不支持pse、prefix、伪量化、全量化、后量化。
                - 当kv为tensorlist时，`key_rope`的shape中b需要与tensorlist长度保持一致，n、s需要与tensorlist中每个tensor的n、s相等，d为64。
            <!-- end id45 -->

    <!-- npu="A3,910b" id46 -->
    - TND、TND\_NTD、NTD\_TND场景下`query`、`key`、`value`输入的综合限制（<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>）：
        - `actual_seq_qlen`和`actual_seq_kvlen`必须传入，且以该入参元素数量作为Batch值（注意入参元素数量要小于等于4096）。该入参中每个元素的值表示当前Batch与之前所有Batch的Sequence Length和，因此后一个元素的值必须大于等于前一个元素的值；
        - 当`query`的D等于512时：
            - sparse：支持0/3/4/9；
            - 支持TND、TND\_NTD；
            - 支持开启page attention，此时`actual_seq_kvlen`长度等于`key`/`value`的batch值，代表每个batch的实际长度，值不大于KV\_S；
            - 要求`query`的N为1/2/4/8/16/32/64/128，`key`、`value`的N为1；
            - 要求`query_rope`和`key_rope`不等于空，`query_rope`和`key_rope`的D为64；
            - 不支持开启左padding、tensorlist、pse、prefix、伪量化、全量化、后量化、空Tensor。

        - 当`query`的D不等于512时：
            - 当`query_rope`和`key_rope`为空时：TND场景，要求Q\_D（`query`的D维度）、K\_D（`key`的D维度）、V\_D（`value`的D维度）等于128，或者Q\_D、K\_D等于192，V\_D等于128/192；NTD场景，不支持V\_D等于192；NTD\_TND场景，要求Q\_D、K\_D等于128/192，V\_D等于128。当`query_rope`和`key_rope`不为空时，要求Q\_D、K\_D、V\_D等于128；GQA和PA场景不支持V_D等于192; MHA（Multi-Head Attention）场景Q\_D、K\_D、V\_D都等于64，或Q\_D、K\_D、V\_D都等于128，或Q\_D和K\_D等于192时V\_D等于128。
            - 支持TND、NTD、NTD\_TND；
            - page attention场景下仅支持`block_size`为16对齐且小于等于1024;
            - MHA场景下仅支持数据类型为`torch.float16`、`torch.bfloat16`，当数据类型为`torch.float16`，`inner_precise`仅支持0和1，当数据类型为`torch.bfloat16`，`inner_precise`仅支持0。当`sparse_mode`=0不传`atten_mask`矩阵，`sparse_mode`为3/4传优化后的`atten_mask`矩阵。page attention仅支持BnBsH格式，BnBsH表示KV Cache的排布格式为（blockNum, blocksize, H），其中blockNum为块数量、blocksize为每个块中的token个数、H为隐藏层大小；
            - 不支持开启左padding、tensorlist、pse、prefix、伪量化、全量化；
    <!-- end id46 -->
    <!-- npu="950" id47 -->
    - TND场景下`query`、`key`、`value`输入的综合限制（<term>Ascend 950PR&950DT系列产品</term>）：
        - `actual_seq_qlen`和`actual_seq_kvlen`必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的Sequence Length和，因此后一个元素的值必须大于等于前一个元素的值。
        - 不支持左padding、tensorlist、pse、prefix。
    <!-- end id47 -->

### GQA伪量化+KV NZ格式约束<a name="zh-cn_topic_0000001832267082_section_gqa_nz_constraint"></a>

- GQA伪量化场景下KV为$NZ$格式时的参数约束如下：
    - `input_layout`仅支持BSH、BSND、BNSD；
    - `key`&`value`仅支持$NZ$输入，输入格式为\[blockNum, KV\_N, D/32, blockSize, 32\]；
    - 仅支持KV分离；
    - 仅支持高性能模式；
    - 不支持配置`query_rope`和`key_rope`；
    - `num_query_heads`与`num_key_value_heads`支持组合有(10, 1)、(64, 8)、(80, 8)、(128, 16)。

    <!-- npu="A3,910b" id48 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
        - 仅支持page_attention场景，`block_size`仅支持128或512；
        - 支持perchannel和pertoken模式，`query`数据类型固定为`torch.bfloat16`，`key`&`value`固定为`torch.int8`；`query`&`key`&`value`的D仅支持128；`query` Sequence Length仅支持1-16；
        - `dequant_scale_key`和`dequant_scale_value`的dtype：perchannel模式下，仅支持`torch.bfloat16`类型；pertoken模式下，仅支持`torch.float32`类型；
        - `dequant_scale_key`和`dequant_scale_value`的shape：perchannel模式下，当layout为BSH时，必须传入\[H\]；layout为BNSD时，必须传入\[KV\_N,1,D\]；输出为BSND时，必须传入\[KV\_N, D\]；pertoken模式下，必须传入\[B,KV\_S\]，S需要大于等于`block_table`的第二维*`block_size`；
        - 当MTP等于0时，支持`sparse_mode`=0且不传mask；当MTP大于0、小于16时，支持`sparse_mode`=3（传入优化后的`atten_mask`矩阵，shape必须为2048\*2048）或`sparse_mode`=9（传入tree mask，`input_layout`为BSH/BSND/BNSD时shape为\(B, Q\_S, Q\_S\)，`input_layout`为TND时shape为\(∑Q\_Si²,\)）；
        - 不支持配置`dequant_offset_key`和`dequant_offset_value`;
        - 不支持左padding、tensorlist、pse、prefix、后量化；
    <!-- end id48 -->
    <!-- npu="950" id49 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
        - 仅支持per-channel模式，query数据类型固定为`torch.bfloat16`，key&value固定为`torch.int8`；`query&key&value`的D仅支持128；`query` Sequence Length仅支持1-16；
        - `dequant_scale_key`和`dequant_scale_value`的shape：当layout为BSH时，必须传入[H]；layout为BNSD时，必须传入[N,1,D]；输出为BSND时，必须传入[N, D]；
        - 当MTP等于0时，支持`sparse_mode`=0且不传mask；当MTP大于0、小于16时，支持`sparse_mode`=3且传入优化后的`atten_mask`矩阵，`atten_mask`矩阵shape必须传入（2048*2048）；
        - 不支持tensorlist、pse、page attention、后量化；
    <!-- end id49 -->

### learnable_sink约束<a name="zh-cn_topic_0000001832267082_section_learnable_sink_constraint"></a>

- `learnable_sink`的参数约束如下：

    <!-- npu="A3,910b" id50 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
        - 仅支持TND、NTD\_TND；
        - 仅支持`value`的D小于等于128；
    <!-- end id50 -->
    <!-- npu="950" id51 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
        - 仅支持`value`的D等于64、128；
    <!-- end id51 -->
    - 仅支持非量化场景。
    - 不支持pse、左padding、公共前缀、后量化。

### Q_S>1（全量推理）约束<a name="zh-cn_topic_0000001832267082_section_qs_gt1_constraint"></a>

- **当Q\_S大于1时：**
    - `query`、`key`、`value`输入，功能使用限制如下：

        <!-- npu="A3,910b" id52 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持B轴小于等于65536，D轴32byte不对齐时仅支持B轴小于等于128。
        <!-- end id52 -->
        <!-- npu="950" id53 -->
        - <term>Ascend 950PR&950DT系列产品</term>：支持B轴小于等于65536。
        <!-- end id53 -->
        - 支持N轴小于等于256，支持D轴小于等于512；`input_layout`为BSH或者BSND时，建议N\*D小于65535。
        - S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致PFA算子执行超时（aicore error类型报错，errorStr为timeout or trap error），此场景下建议做S切分处理（注：这里计算量会受B、S、N、D等的影响，值越大计算量越大），典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：
            - B=1，Q\_N=20，Q\_S=2097152，D=256，KV\_N=1，KV\_S=2097152。
            - B=1，Q\_N=2，Q\_S=20971520，D=256，KV\_N=2，KV\_S=20971520。
            - B=20，Q\_N=1，Q\_S=2097152，D=256，KV\_N=1，KV\_S=2097152。
            - B=1，Q\_N=10，Q\_S=2097152，D=512，KV\_N=1，KV\_S=2097152。

        - `query`、`key`、`value`输入类型包含`torch.int8`时，D轴需要32对齐；输入类型全为`torch.float16`、`torch.bfloat16`时，D轴需16对齐。
        - D轴限制：

            <!-- npu="A3,910b" id54 -->
            - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`query`、`key`、`value`输入类型包含`torch.int8`时，D轴需要32对齐；`query`、`key`、`value`或attentionOut类型包含`torch_npu.int4`时，D轴需要64对齐；输入类型全为`torch.float16`、`torch.bfloat16`时，D轴需16对齐。
            <!-- end id54 -->
            <!-- npu="950" id55 -->
            - <term>Ascend 950PR&950DT系列产品</term>：非量化场景：`query`，`key`，`value`的类型全部为`torch.float16`、`torch.bfloat16`，D轴1-512全部支持。伪量化场景：`query`类型为`torch.float16`、`torch.bfloat16`，`key`、`value`类型为`torch.int8`/`torch.hifloat8`/`torch.float8_e4m3fn`/`torch.float4_e2m1`/`torch.int4`（`torch.int32`），其中当`key`、`value`类型为`torch.int4`（`torch.int32`）D轴仅支持64对齐（`torch.int32`仅支持D 8对齐）。全量化场景：per-block全量化D轴支持1-128，MxFP8全量化D轴仅支持64或128。
            <!-- end id55 -->

    - `actual_seq_qlen`：

        <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：该入参中每个batch的有效Sequence Length应该不大于`query`中对应batch的Sequence Length。seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度大于等于Batch时取seqlen的前Batch个数。其他长度不支持。当`query`的`input_layout`为TND/NTD\_TND时，综合约束请见[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。

        <term>Ascend 950PR&950DT系列产品</term>：该参数应为非负数，在`input_layout`不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，其长度为1或大于等于`query`的Batch值，该入参中的值代表每个Batch的实际长度，其值应该不大于Q\_S。当`input_layout`为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。

    - `actual_seq_kvlen`：

        <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：该入参中每个batch的有效Sequence Length应该不大于`key/value`中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当`key/value`的`input_layout`为TND/NTD\_TND时，综合约束请见[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。

        <term>Ascend 950PR&950DT系列产品</term>：该参数传入时应为非负数，在`input_layout`不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，该入参中每个Batch的有效seqlenKv应该不大于`key/value`中对应Batch的seqlenKv。当本参数的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当`key/value`的`input_layout`为TND时，该入参必须传入，且该入参元素的数量等于Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlenKv和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。

    - `sparse_mode`：

        <!-- npu="A3,910b" id56 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
            - 当前仅支持值为0、1、2、3、4、9的场景，取其它值时会报错。
        <!-- end id56 -->
        <!-- npu="950" id57 -->
        - <term>Ascend 950PR&950DT系列产品</term>：
            - 当前仅支持值为0、1、2、3、4的场景，取其它值时会报错。
        <!-- end id57 -->
        - `sparse_mode`=0时，`atten_mask`如果为None，则忽略入参`pre_tokens`、`next_tokens`（内部赋值为INT\_MAX）。
        - `sparse_mode`=2、3、4时，`atten_mask`的shape需要为\(S, S\)或\(1, S, S\)或\(1, 1, S, S\)，其中S的值需要固定为2048，且需要用户保证传入的`atten_mask`为下三角，不传入`atten_mask`或者传入的shape不正确报错。
        - `sparse_mode`=1、2、3的场景忽略入参`pre_tokens`、`next_tokens`并按照相关规则赋值。

        <!-- npu="A3,910b" id58 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
            - `sparse_mode`=9时，仅MLA场景（`query_rope`和`key_rope`不为空）支持。`atten_mask`不能为None。`input_layout`为BSH/BSND/BNSD时shape为\(B, Q\_S, Q\_S\)；`input_layout`为TND时shape为\(∑Q\_Si², \)。不支持左padding、`pse_shift`、sharedPrefix。
        <!-- end id58 -->

    - page attention场景：
        - page attention的开启必要条件是`block_table`存在且有效，同时`key`、`value`是按照`block_table`中的索引在一片连续内存中排布，支持`key`、`value`数据类型为`torch.float16`、`torch.bfloat16`。在该场景下`key`、`value`的`input_layout`参数无效。`block_table`中填充的是blockid，当前不会对blockid的合法性进行校验，需用户自行保证。
        - `block_size`是用户自定义的参数，该参数的取值会影响page attention的性能，在开启page attention场景下，`block_size`最小为128，最大为512，且要求是128的倍数。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。

        - page attention场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且KV\_N\*D超过65535时，受硬件指令约束，会被拦截报错。可通过开启GQA（减小KV\_N）或调整kv cache排布格式为（blocknum, KV\_N, blocksize, D）解决。当`query`的`input_layout`为BNSD、TND时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV\_N, blocksize, D）两种格式，当`query`的`input_layout`为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据`actual_seq_kvlen`和`block_size`计算的每个batch的block数量之和。且`key`和`value`的shape需保证一致。

        <!-- npu="A3,910b" id59 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：page attention不支持伪量化场景。
        <!-- end id59 -->
        <!-- npu="950" id60 -->
        - <term>Ascend 950PR&950DT系列产品</term>：page attention伪量化场景下，支持`query`为`torch.float16`/`torch.bfloat16`，支持`key`、`value` dtype为`torch.int8`/`torch.hifloat8`/`torch.float8_e4m3fn`/`torch.float4_e2m1fn_x2`/`torch.int4`（`torch.int32`）。当kv cache为五维时，kv cache排布为（blocknum, KV_N, D/16, blocksize, 16）；同时，当`key`、`value` dtype为`torch.int32`时，kv cache排布为（blocknum, KV_N, D/2, blocksize, 2）。
        <!-- end id60 -->
        - page attention不支持tensorlist场景。
        - page attention场景下，必须传入`actual_seq_kvlen`。
        - page attention场景下，`block_table`必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大`actual_seq_kvlen`对应的block数量）。

        <!-- npu="A3,910b" id61 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
            - page attention场景下，支持两种格式和`torch.float32`/`torch.bfloat16`，不支持输入`query`为`torch.int8`的场景。
        <!-- end id61 -->
        <!-- npu="950" id62 -->
        - <term>Ascend 950PR&950DT系列产品</term>：
            - page attention场景下，支持两种格式和`torch.float32`/`torch.bfloat16`，输入`query`为`torch.int8`的场景仅支持PA_NZ。
        <!-- end id62 -->
        - page attention开启场景下，以下场景输入需满足KV\_S\>=maxBlockNumPerSeq\*`block_size`：
            - 传入`atten_mask`时，如mask shape为（B, 1, Q\_S, KV\_S）。
            - 传入`pse_shift`时，如`pse_shift` shape为（B, Q\_N, Q\_S, KV\_S）。

            <!-- npu="950" id63 -->
            - <term>Ascend 950PR&950DT系列产品</term>：
                - 开启伪量化per-token模式：输入参数`antiquantScale`和`antiquantOffset`的shape均为(2, B, S)。
                - 开启per-token叠加per-head模式：两个参数的shape均为(B, N, S)，数据类型固定为`torch.float32`，当key、value数据类型为`torch.int8`、`torch.int4`(`torch.int32`)时支持。
            <!-- end id63 -->

    - 入参`quant_scale_out`和`quant_offset_out`支持pertensor、perchannel量化，支持`torch.float32`、`torch.bfloat16`类型。若传入`quant_offset_out`，需保证其类型和shape信息与`quant_scale_out`一致。当输入为`torch.bfloat16`时，同时支持`torch.float32`和`torch.bfloat16`，否则仅支持`torch.float32`。perchannel场景下，当输出layout为BSH时，要求`quant_scale_out`所有维度的乘积等于H；其他layout要求乘积等于Q\_N\*D。当输出layout为BSH时，`quant_scale_out` shape建议传入\(1, 1, H\)或\(H,\)；当输出layout为BNSD时，建议传入\(1, Q\_N, 1, D\)或\(Q\_N, D\)；当输出为BSND时，建议传入\(1, 1, Q\_N, D\)或\(Q\_N, D)。
    - 输出为`torch.int8`，`quant_scale_out`和`quant_offset_out`为perchannel时，暂不支持Ring Attention或者D非32Byte对齐的场景。
    - 输出为`torch.int8`时，暂不支持sparse为band且`pre_tokens`/`next_tokens`为负数。
    - `pse_shift`功能使用限制如下：

        - 支持`query`数据类型为`torch.float16`、`torch.bfloat16`、`torch.int8`场景下使用该功能。
        - `query`、`key`、`value`数据类型为`torch.float16`且`pse_shift`存在时，强制走高精度模式，对应的限制继承自高精度模式的限制。
        - Q\_S需大于等于`query`的S长度，KV\_S需大于等于`key`的S长度。

        <!-- npu="950" id64 -->
        - <term>Ascend 950PR&950DT系列产品</term>：非量化场景无对齐限制。
        <!-- end id64 -->

    - 输出为`torch.int8`，入参`quant_offset_out`传入非None和非空tensor值，并且`sparse_mode`、`pre_tokens`和`next_tokens`满足以下条件，矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截：
        - `sparse_mode`=0，`atten_mask`如果非None，每个batch `actual_seq_qlen`-`actual_seq_kvlen`-`pre_tokens`\>0或`next_tokens`<0时，满足拦截条件。
        - `sparse_mode`=1或2，不会出现满足拦截条件的情况。
        - `sparse_mode`=3，每个batch `actual_seq_kvlen`-`actual_seq_qlen`<0，满足拦截条件。
        - `sparse_mode`=4，`pre_tokens`<0或每个batch `next_tokens`+`actual_seq_kvlen`-`actual_seq_qlen`<0时，满足拦截条件。

    - kv伪量化参数分离：
        - 当伪量化参数和KV分离量化参数同时传入时，以KV分离量化参数为准。
        - `key_quant_mode`和`value_quant_mode`取值需要保持一致。
        - `dequant_scale_key`和`dequant_scale_value`要么都为空，要么都不为空；`dequant_offset_key`和`dequant_offset_value`要么都为空，要么都不为空。
        - `dequant_scale_key`和`dequant_scale_value`都不为空时，其shape需要保持一致；`dequant_offset_key`和`dequant_offset_value`都不为空时，其shape需要保持一致。

        <!-- npu="A3,910b" id65 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
            - 仅支持pertoken和perchannel模式，pertoken模式下要求两个参数的shape均为\(B, KV\_S\)，数据类型固定为`torch.float32`；perchannel模式下要求两个参数的shape为（KV\_N, D），\(KV\_N, D\)，\(H\)，数据类型固定为`torch.bfloat16`,H为KV\_N*D。
            - `dequant_scale_key`与`dequant_scale_value`非空场景，要求`query`的s小于等于16；要求`query`的dtype为`torch.bfloat16`，`key`、`value`的dtype为`torch.int8`，输出的dtype为`torch.bfloat16`；不支持tensorlist、page attention特性。
        <!-- end id65 -->
        <!-- npu="950" id66 -->
        - <term>Ascend 950PR&950DT系列产品</term>：
            - 支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token使用page attention模式管理scale/offset、per-token叠加per-head并使用page attention模式管理scale/offset、`key`支持per-channel叠加`value`支持per-token和per-token-group九种模式。
            - `dequant_scale_key`和`dequant_scale_value`都不为空时，除了`key_quant_mode`为0并且`value_quant_mode`为1的场景外，其shape需要保持一致；`dequant_offset_key`和`dequant_offset_value`都不为空时，除了`key_quant_mode`为0并且`value_quant_mode`为1的场景外，其shape需要保持一致。
        <!-- end id66 -->
        - 管理scale/offset的量化模式如下：

            > [!NOTE]
            > 注意scale、offset具体指`dequant_scale_key`、`dequant_scale_value`、`dequant_offset_key`、`dequant_offset_value`参数。
            > 当Q\_S大于1时，不同产品支持不同量化模式：
            > <!-- npu="A3,910b" id81 -->
            > - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持per-channel模式、per-token模式。
            > <!-- end id81 -->
            > <!-- npu="950" id82 -->
            > - <term>Ascend 950PR&950DT系列产品</term>：支持下方所有量化模式。
            > <!-- end id82 -->

            | 量化模式 | 该场景下scale和offset条件 | 该场景下`key`和`value`条件 |
            | --- | --- | --- |
            | per-channel模式 | 两个参数shape支持(N, 1, D)，(N, D)，(H)，(1, N, 1, D)，(1, N, D)，(1, H)数据类型和`query`数据类型相同。 | <li><term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持`key`、`value`数据类型为`torch.int8`。</li><li><term>Ascend 950PR&950DT系列产品</term>：当`key`、`value`数据类型为`torch.int4`（`torch.int32`）、`torch.int8`、`torch.hifloat8`、`torch.float8_e4m3fn`时支持。其中，当为`torch.hifloat8`、`torch.float8_e4m3fn`时，不支持带`dequant_offset`。</li> |
            | per-token模式 | 两个参数的shape均为(B, S)，数据类型固定为`torch.float32`。 | <li><term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持`key`、`value`数据类型为`torch.int8`。</li><li><term>Ascend 950PR&950DT系列产品</term>：当`key`、`value`数据类型为`torch.int4`（`torch.int32`）、`torch.int8`、`torch.float8_e4m3fn`时支持。</li> |
            | per-tensor模式 | 两个参数的shape均为(1,)，数据类型和`query`数据类型相同。 | 当`key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
            | per-tensor叠加per-head模式 | 两个参数的shape均为(N,)，数据类型和`query`数据类型相同。 | 当`key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
            | per-token叠加per-head模式 | 两个参数的shape均为(B, N, S)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
            | per-token叠加使用page attention模式 | 两个参数的shape均为(blocknum, blocksize)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int8`、`torch.float8_e4m3fn`时支持。 |
            | per-token叠加per head并使用page attention模式 | 两个参数的shape均为(blocknum, N, blocksize)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int8`时支持。 |
            | `key`支持per-channel叠加`value`支持per-token模式 | 对`key`支持per-channel，两个参数的shape可支持(N, 1, D)、(N, D)、(H)，(1, N, 1, D)、(1, N, D)、(1, H)且数据类型和`query`数据类型相同。 | 当`key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
            | 对于`value`支持per-token，两个参数的shape均为(B, S)并且数据类型固定为`torch.float32`。 | | |
            | per-token-group模式 | `dequant_scale`的shape为(1, B, N, S, D/32)，数据类型固定为`torch.float8_e8m0fnu`，不支持带`dequant_offset`。 | 当`key`、`value`数据类型为`torch.float4_e2m1fn_x2`时支持。 |

    <!-- npu="950" id67 -->
    - MXFP8全量化：

        - <term>Ascend 950PR&950DT系列产品</term> ：
            - `query`、`key`和`value`的数据类型支持`torch.float8_e4m3fn`。
            - `query_quant_mode`、`key_quant_mode`和`value_quant_mode`分别为6、6、8。
            - `dequant_scale_query`、`dequant_scale_key`和`dequant_scale_value`的数据类型固定为`torch.float8_e8m0fnu`。
            - `input_layout`仅支持TND，同时支持PageAttention场景和非PageAttention场景。`query`的shape为(Q\_T, Q\_N, D)，`key`和`value`按照PageAttention场景的shape传入，kv cache排布支持(blocknum, KV\_N, blocksize, D)或(blocknum，KV\_N，D/D0，blocksize，D0)，其中blocksize固定为512。
            - 当Q\_S \* G \> 80场景，`dequant_scale_query`的shape推荐为(Q\_T, Q\_N, D/64, 2)，当Q\_S \* G <= 80场景，入参`dequant_scale_query`的shape推荐为(KV\_N, Q\_T, G, D/64, 2)；非PagedAttention场景，`dequant_scale_key`的shape为(KV\_T, KV\_N, D/64, 2)，`dequant_scale_value`的shape为(KV\_T/64, KV\_N, D, 2)；PagedAttention场景下，kv cache排布为BnNBsD时，`dequant_scale_key`的shape为(Bn, KV\_N, Bs, D/64, 2)，`dequant_scale_value`的shape为(Bn, KV\_N, Bs/64, D, 2)；kv cache排布为$NZ$时，`dequant_scale_key`的shape为(Bn, KV\_N, Bs/16, D/64, 16, 2)，`dequant_scale_value`的shape为(Bn, KV\_N, D/16, Bs/64, 16, 2)。
            - 支持`sparse_mode` 0和3，且`sparse_mode`为0时不支持传入mask矩阵。
            - 支持`inner_precise` 0和1。
            - 支持softmax_lse。
            - 单算子直调时，PagedAttention场景下，kv cache排布为BnNBsD/$NZ$时，`key`、`value`、`dequant_scale_key`、`dequant_scale_value`和`key_rope`支持0轴、1轴的非连续。
    <!-- end id67 -->

    <!-- npu="950" id68 -->
    - GQA FP8全量化：

        - <term>Ascend 950PR&950DT系列产品</term> ：
            - `query`、`key`和`value`的数据类型支持`torch.float8_e4m3fn`。
            - `query_quant_mode`、`key_quant_mode`和`value_quant_mode`分别为3、3、2。
            - `dequant_scale_query`、`dequant_scale_key`和`dequant_scale_value`的数据类型固定为`torch.float32`。
            - `input_layout`支持NTD\_TND，且仅支持PageAttention场景。`query`的shape为(Q\_N, Q\_T, D)，入参`key`和`value`的shape为(blocknum, KV\_N, blocksize, D)，其中blocksize固定为128，D固定为128。
            - `dequant_scale_query`的shape为(Q\_N, Q\_T)，`dequant_scale_key`的shape为(blocknum, KV\_N, blocksize)，`dequant_scale_value`的shape为(KV\_N)。
            - 支持`sparse_mode` 0和3，且`sparse_mode`为0时不支持传入mask矩阵。
            - 支持`inner_precise` 0和1。
            - 支持softmax_lse。
            - 支持PScale静态量化，PScale的shape为(1)，数据类型为`torch.float32`，可选参数，不传默认值为1.0。
            - 单算子直调时，PagedAttention场景下，`key`、`value`、`dequant_scale_key`仅支持0轴、1轴的非连续场景。
            - 不支持rope、左padding、tensorlist、prefix、pse。
    <!-- end id68 -->

### Q_S=1（增量推理）约束<a name="zh-cn_topic_0000001832267082_section_qs_eq1_constraint"></a>

- **当Q\_S等于1时：**
    - `query`、`key`、`value`输入，功能使用限制如下：
        - 支持B轴小于等于65536，支持N轴小于等于256，支持S轴小于等于262144，支持D轴小于等于512。
        - `query`、`key`、`value`输入类型均为`torch.int8`的场景暂不支持。
        - 在`torch_npu.int4`（`torch.int32`）伪量化场景下，PyTorch入图调用仅支持KV `torch.int4`拼接成`torch.int32`输入（建议通过dynamicQuant生成`torch.int4`格式的数据，算子运行时会将8个`torch.int4`打包存储于一个`torch.int32`中）。
        - 在`torch_npu.int4`（`torch.int32`）伪量化场景下，若KV `torch.int4`拼接成`torch.int32`输入，那么KV的N、D或者H是实际值的八分之一。并且，`torch.int4`伪量化仅支持D 64对齐（`torch.int32`支持D 8对齐）。

        <!-- npu="950" id69 -->
        - <term>Ascend 950PR&950DT系列产品</term>：`key`、`value`输入类型为`torch.float4_e2m1fn_x2`/`torch.int4`（`torch.int32`）时，`query`的D轴以及`key`、`value`的D轴需要64对齐（`torch.int32`仅支持`key`、`value`的D 8对齐）。
        <!-- end id69 -->

    - `actual_seq_qlen`：

        <!-- npu="A3,910b" id70 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当`query`的`input_layout`不为TND时，Q\_S为1时该参数无效。当`query`的`input_layout`为TND/TND\_NTD时，综合约束请见[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。
        <!-- end id70 -->
        <!-- npu="950" id71 -->
        - <term>Ascend 950PR&950DT系列产品</term>：该参数在未输入rope参数时不生效。输入rope参数时生效，传入时应为非负数，在`input_layout`不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，其长度为1或大于等于`query`的Batch值，该入参中的值代表每个Batch的实际长度，其值应该不大于Q\_S。当`input_layout`为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
        <!-- end id71 -->

    - `actual_seq_kvlen`：

        <!-- npu="A3,910b" id72 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：该入参中每个batch的有效Sequence Length应该不大于`key`/`value`中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当`key`/`value`的`input_layout`为TND/TND\_NTD时，综合约束请见[MLA场景约束](#zh-cn_topic_0000001832267082_section_mla_constraint)。
        <!-- end id72 -->
        <!-- npu="950" id73 -->
        - <term>Ascend 950PR&950DT系列产品</term>：该参数应为非负数，在`input_layout`不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，该入参中每个Batch的有效Sequence Length应该不大于`key/value`中对应Batch的seqlenKv。当本参数的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当`input_layout`为TND/TND\_NTD时，该入参必须传入，在非PA场景下，第b个值表示前b个Batch的S轴累加长度，其值应递增（大于等于前一个值）排列，且该入参元素的数量代表总Batch数，在PA场景下，其长度等于`key/value`的Batch值，代表每个Batch的实际长度，值不大于KV\_S。
        <!-- end id73 -->

    - page attention场景：
        - 开启必要条件是`block_table`存在且有效，同时`key`、`value`是按照`block_table`中的索引在一片连续内存中排布，在该场景下`key`、`value`的`input_layout`参数无效。

        <!-- npu="A3,910b" id74 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
            - 支持`key`、`value`数据类型为`torch.float16`、`torch.bfloat16`、`torch.int8`。
            - 不支持Q为`torch.bfloat16`、`torch.float16`、`key`、`value`为`torch_npu.int4`（`torch.int32`）的场景。
        <!-- end id74 -->
        <!-- npu="950" id75 -->
        - <term>Ascend 950PR&950DT系列产品</term>：支持`key`、`value`数据类型为`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch.int4`（`torch.int32`）、`torch.float8_e4m3fn`、`torch.hifloat8`、`torch.float4_e2m1fn_x2`。
        <!-- end id75 -->
        - 该场景下，`block_size`是用户自定义的参数，该参数的取值会影响page attention的性能。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。

            <!-- npu="A3,910b" id76 -->
            - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`key`、`value`输入类型为`torch.float16`、`torch.bfloat16`时需要16对齐，`key`、`value`输入类型为`torch.int8`时需要32对齐，推荐使用128。
            <!-- end id76 -->
            <!-- npu="950" id77 -->
            - <term>Ascend 950PR&950DT系列产品</term>：`key`、`value`输入类型为`torch.float16`/`torch.bfloat16`时需要16对齐；`key`、`value`输入类型为`torch.int8`/`torch.float8_e4m3fn`/`torch.hifloat8`时需要32对齐；`key`、`value`输入类型为`torch.float4_e2m1fn_x2`/`torch.int4`（`torch.int32`）时需要64对齐。
            <!-- end id77 -->

        - 参数`key`、`value`各自对应tensor的shape所有维度相乘不能超过`torch.int32`的表示范围。
        - page attention场景下，`block_table`必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大`actual_seq_kvlen`对应的block数量）。

        <!-- npu="A3,910b" id78 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
            - page attention场景下，当`query`的`input_layout`为BNSD、TND时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV\_N, blocksize, D）两种格式，当`query`的`input_layout`为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据`actual_seq_kvlen`和`block_size`计算的每个batch的block数量之和。且`key`和`value`的shape需保证一致。
        <!-- end id78 -->
        <!-- npu="950" id79 -->
        - <term>Ascend 950PR&950DT系列产品</term>：
            - page attention场景下，当`query`的`input_layout`为BNSD、TND时，kv cache排布支持（blocknum, blocksize, H）、（blocknum, KV_N, blocksize, D）和（blocknum, KV_N, D/16, blocksize, 16）三种格式，当`query`的`input_layout`为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）和（blocknum, KV_N, D/16, blocksize, 16）两种格式。blocknum不能小于根据`actual_seq_kvlen`和`block_size`计算的每个batch的block数量之和。且`key`和`value`的shape需保证一致。
            - page attention伪量化场景下，当kv cache为五维时，kv cache排布为（blocknum, KV\_N, D/16, blocksize, 16）；同时，当`key`、`value` dtype为`torch.int32`时，kv cache排布为（blocknum, KV\_N, D/2, blocksize, 2）。
        <!-- end id79 -->
        - page attention场景下，kv cache排布为（blocknum, KV\_N, blocksize, D）时性能通常优于kv cache排布为（blocknum, blocksize, H）时的性能，建议优先选择（blocknum, KV\_N, blocksize, D）格式。
        - page attention开启场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且numKvHeads \* headDim超过64k时，受硬件指令约束，会被拦截报错。可通过开启GQA（减小numKvHeads）或调整kv cache排布格式为（blocknum, numKvHeads, blocksize, D）解决。
        - page attention场景的参数`key`、`value`各自对应tensor的shape所有维度相乘不能超过`torch.int32`的表示范围。

    - kv伪量化参数分离：
        - 除了`key_quant_mode`为0并且`value_quant_mode`为1的场景外，`key_quant_mode`和`value_quant_mode`取值需要保持一致。
        - `dequant_scale_key`和`dequant_scale_value`要么都为空，要么都不为空；`dequant_offset_key`和`dequant_offset_value`要么都为空，要么都不为空。
        - `dequant_scale_key`和`dequant_scale_value`都不为空时，除了`key_quant_mode`为0并且`value_quant_mode`为1的场景外，其shape需要保持一致；`dequant_offset_key`和`dequant_offset_value`都不为空时，除了`key_quant_mode`为0并且`value_quant_mode`为1的场景外，其shape需要保持一致。
        - `torch_npu.int4`（`torch.int32`）伪量化场景不支持后量化。
        - 管理scale/offset的量化模式如下：
    
            > [!NOTE]   
            > 注意scale、offset两个参数指`dequant_scale_key`、`dequant_scale_value`、`dequant_offset_key`、`dequant_offset_value`。
            > 当Q\_S等于1时，不同产品支持不同量化模式：
            > <!-- npu="A3,910b" id83 -->
            > - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持per-channel模式、per-tensor模式、per-token模式、per-tensor叠加per-head模式、per-token叠加per-head模式、per-token叠加使用page attention模式、per-token叠加per head并使用page attention模式、`key`支持per-channel叠加value支持per-token模式。
            > <!-- end id83 -->
            > <!-- npu="950" id84 -->
            > - <term>Ascend 950PR&950DT系列产品</term>: 支持下方所有量化模式。
            > <!-- end id84 -->

            | 量化模式 | 该场景下scale和offset条件 | 该场景下`key`和value条件 |
            | --- | --- | --- |
            | per-channel模式 | 两个参数shape支持(1, KV_N, 1, D)，(1, KV_N, D)，(1, H)，数据类型和`query`数据类型相同。 | <li><term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当`key`、`value`数据类型为`torch_npu.int4`（`torch.int32`）或`torch.int8`时支持。</li><li><term>Ascend 950PR&950DT系列产品</term>：当`key`、`value`数据类型为`torch.int4`（`torch.int32`）、`torch.int8`、`torch.hifloat8`、`torch.float8_e4m3fn`时支持。其中，当为`torch.hifloat8`、`torch.float8_e4m3fn`时，不支持带`dequant_offset`。</li> |
            | per-tensor模式 | 两个参数的shape为(1,)，数据类型和`query`数据类型相同。 | 当`key`、`value`数据类型为`torch.int8`时支持。 |
            | per-token模式 | 两个参数的shape均为(1, B, KV_S)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
            | per-tensor叠加per-head模式 | 两个参数的shape为(KV_N,)，数据类型和`query`数据类型相同。 | 当`key`、`value`数据类型为`torch.int8`时支持。 |
            | per-token叠加per-head模式 | 两个参数的shape均为(B, KV_N, KV_S)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
            | per-token叠加使用page attention模式 | 两个参数的shape为(blocknum, blocksize)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int8`时支持。 |
            | per-token叠加per head并使用page attention模式 | 两个参数的shape为(blocknum, KV_N, blocksize)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int8`时支持。 |
            | `key`支持per-channel叠加`value`支持per-token模式 | <li>对于`key`支持per-channel，两个参数的shape可支持(1, KV_N, 1, D)、(1, KV_N, D)、(1, H)，且数据类型和`query`数据类型相同。</li><li>对于`value`支持per-token，两个参数的shape均为(1, B, KV_S)并且数据类型固定为`torch.float32`。</li> | <li><term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当`key`、`value`数据类型为`torch_npu.int4`（`torch.int32`）或`torch.int8`时支持；当`key`和`value`的数据类型为`torch.int8`时，仅支持query和输出的dtype为`torch.float16`。</li><li><term>Ascend 950PR&950DT系列产品</term>：当`key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。</li> |
            | per-token-group模式 | `dequant_scale`的shape为(1, B, KV\_N, KV\_S, D/32)，数据类型固定为`torch.float8_e8m0fnu`，不支持带`dequant_offset`。 | 当`key`、`value`数据类型为`torch.float4_e2m1fn_x2`时支持。 |

    - `pse_shift`功能使用限制如下：
        - `pse_shift`数据类型需与`query`数据类型保持一致。

        <!-- npu="A3,910b" id80 -->
        - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
            - 仅支持D轴对齐，即D轴可以被16整除。
        <!-- end id80 -->

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

### 示例 1：MHA + 非量化 + BNSD（PromptFlashAttention）

**场景说明**：`query/key/value`均为`torch.float16`，非量化场景，BNSD layout。无BMM1/BMM2量化参数，也不做输出量化。

**关键参数**：

- `num_query_heads=8`：必选参数，表示head个数。
- `num_key_value_heads`：可选参数，为空表示单头大小与`num_query_heads`一致，即单头注意力机制。
- `input_layout="BNSD"`：必选参数，输入数据的排布格式。
- `softmax_scale=1/√D`：必选参数，防止点积值过大导致softmax梯度消失。
- `pre_tokens=65535, next_tokens=65535`：可选参数，表示不限制attention范围。

**相关约束**：[通用约束](#通用约束)

```python
import torch
import torch_npu
import math
# 生成随机数据, 并发送到npu
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
softmax_scale = 1/math.sqrt(128.0)
actseqlen = [164]
actseqlenkv = [1024]

# 调用FIA算子
out, _ = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, 
         actual_seq_qlen = actseqlen, actual_seq_kvlen = actseqlenkv,
         num_query_heads = 8, input_layout = "BNSD", softmax_scale = softmax_scale, pre_tokens=65535, next_tokens=65535)

print(out)
```

执行上述代码的输出out类似如下：

```text
tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ..
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16)
```

### 示例 2：GQA + KV NZ + PageAttention

**场景说明**：Decode阶段，Q_S=1（逐token增量推理）。GQA下KV_N=1 < Q_N=10，KV Cache以Page Attention方式管理，`key`/`value` 为`torch.int8`并通过伪量化反量化为`torch.float16`计算，选择NZ格式时仅支持高性能模式。

**关键参数**：

- `num_query_heads=10, num_key_value_heads=1`：必选参数，GQA支持组合：(10,1)/(64,8)/(80,8)/(128,16)。
- `block_size=128`：必选参数，开启PageAttention，仅支持128或512。
- `dequant_scale_value, dequant_scale_key`：使用per-channel伪量化时必选参数，dtype固定为`torch.bfloat16`。
- `key_quant_mode=0,value_quant_mode=0`：使用per-channel伪量化时必选参数，固定为0。
- `key, value`: 必选参数，类型为`torch.int8`，NZ格式[blockNum, KV_N, D/32, blockSize, 32]。
- `inner_precise=1`：必选参数，KV NZ格式约束下仅支持高性能模式。

**相关约束**：[GQA伪量化+KV NZ格式约束](#gqa伪量化kv-nz格式约束)，[增量推理约束](#q_s1增量推理约束)

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 10, 1, 128
KV_N = 1            # 组合 (10, 1)
block_size = 128     # 仅支持 128 或 512
KV_S = 1024          # key/value序列长度

# 计算block数量
block_num = (KV_S + block_size - 1) // block_size

# query: bfloat16, BNSD
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.bfloat16).npu()

# key/value: int8, NZ格式 [blockNum, KV_N, D/32, blockSize, 32]
k = torch.randint(-128, 127, (block_num, KV_N, D // 32, block_size, 32), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (block_num, KV_N, D // 32, block_size, 32), dtype=torch.int8).npu()

# dequant_scale: perchannel模式, bfloat16
# layout=BNSD时shape为 [KV_N, 1, D]
dequant_scale_key = torch.randn(KV_N, 1, D, dtype=torch.bfloat16).npu()
dequant_scale_value = torch.randn(KV_N, 1, D, dtype=torch.bfloat16).npu()

# block_table: page attention映射表
block_table = torch.arange(block_num, dtype=torch.int32).reshape(B, block_num).npu()

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(q, k, v,
         dequant_scale_key=dequant_scale_key, dequant_scale_value=dequant_scale_value, 
         key_quant_mode=0, value_quant_mode=0, 
         block_table=block_table, actual_seq_kvlen=[KV_S],
         num_query_heads=Q_N, num_key_value_heads=KV_N,
         input_layout="BNSD", softmax_scale=softmax_scale, block_size=block_size,
         inner_precise=1 #高性能模式
         )

print(out)
```

### 示例 3：MLA + TND约束

**场景说明**：MLA场景，独立传入RoPE位置编码信息。TND layout，D=128。

**关键参数**：

- `input_layout="TND"`：TND场景下必选参数。
- `actual_seq_qlen, actual_seq_kvlen`：TND场景下必选参数。
- `query_rope, key_rope`：必选参数，MLA模式参数。

**相关约束**：[MLA场景约束](#mla场景约束)

```python
import torch
import torch_npu
import math

Q_N, KV_N = 8, 8
D = 128
D_rope = 64

# TND场景：T为所有Batch的seqlen累加和
# actual_seq_qlen表示每个batch的累加seqlen
# 单batch场景: T = S
S_q = 164
S_kv = 1024

# query:  (T, Q_N, D)     → (164, 8, 128)
# key:    (T, KV_N, D)    → (1024, 8, 128)
# value:  (T, KV_N, D)    → (1024, 8, 128)
q = torch.randn(S_q, Q_N, D, dtype=torch.float16).npu()
k = torch.randn(S_kv, KV_N, D, dtype=torch.float16).npu()
v = torch.randn(S_kv, KV_N, D, dtype=torch.float16).npu()

# query_rope: (T, Q_N, 64)
# key_rope:   (T, KV_N, 64)
query_rope = torch.randn(S_q, Q_N, D_rope, dtype=torch.float16).npu()
key_rope   = torch.randn(S_kv, KV_N, D_rope, dtype=torch.float16).npu()

# TND场景必须传入actual_seq_qlen / actual_seq_kvlen
# 元素为当前batch与之前所有batch的S累加和（单batch直接给S值）
actual_seq_qlen = [S_q]
actual_seq_kvlen = [S_kv]

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(
    q, k, v,
    query_rope=query_rope,
    key_rope=key_rope,
    actual_seq_qlen=actual_seq_qlen,       # TND场景必须传入
    actual_seq_kvlen=actual_seq_kvlen,     # TND场景必须传入
    num_query_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="TND",                    # TND布局
    softmax_scale=softmax_scale,
    sparse_mode=0,                         # 不传atten_mask
    inner_precise=1                        # 仅支持 0 或 1
)

print(out)  #tensor([[[...]]], device='npu:0', dtype=torch.float16)
```

### 示例4：learnable_sink + 非量化约束

**场景说明**：添加**learnable_sink**参数，并且注意其他约束条件。

**关键参数**：

- `learnable_sink`：必选参数，learnable_sink约束关键参数。
- `actual_seq_qlen, actual_seq_kvlen`：TND场景下必选参数。
- `input_layout="TND"`：必选参数，仅支持TND、NTD_TND场景。

**相关约束**：[learnable_sink约束](#learnable_sink约束)

```python
import torch
import torch_npu
import math

S_q = 164
S_kv = 1024
Q_N, KV_N = 8, 8
D = 128               # value的D ≤ 128

# query/key/value: TND布局, bfloat16
q = torch.randn(S_q, Q_N, D, dtype=torch.bfloat16).npu()
k = torch.randn(S_kv, KV_N, D, dtype=torch.bfloat16).npu()
v = torch.randn(S_kv, KV_N, D, dtype=torch.bfloat16).npu()

# learnable_sink: shape=(Q_N,), dtype=bfloat16
learnable_sink = torch.randn(Q_N, dtype=torch.bfloat16).npu()

# TND场景必须传入actual_seq_qlen / actual_seq_kvlen
actual_seq_qlen = [S_q]
actual_seq_kvlen = [S_kv]

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(
    q, k, v,
    learnable_sink=learnable_sink,          # 可学习的Sink Token
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_kvlen=actual_seq_kvlen,
    num_query_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="TND",
    softmax_scale=softmax_scale,
    sparse_mode=0
)

print(out)
```

### 示例5：`torch.int8`后量化

**场景说明**：输入为`torch.bfloat16`，输出为`torch.int8`。除参数名称外与`torch_npu.npu_fused_infer_attention_score`同名示例差别不大。

**关键参数**：

- `quant_scale_out`: 必选参数，`torch.bfloat16`输入时同时支持`torch.float32` / `torch.bfloat16`。
- `quant_offset_out`: 可选参数，类型和shape与scale一致，不传则默认0。

**相关约束**：[通用约束](#通用约束)

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N = 8
KV_S = 1024

# query/key/value: bfloat16, BNSD
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.bfloat16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.bfloat16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.bfloat16).npu()

# quant_scale_out: 必传，bfloat16 输入时同时支持float32 / bfloat16
# perchannel: 输出BNSD推荐 (1, Q_N, 1, D)
quant_scale_out = torch.randn(1, Q_N, 1, D, dtype=torch.float32).npu()
# quant_offset_out: 可选，类型和shape与scale一致，不传则默认0

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(
    q, k, v,
    quant_scale_out=quant_scale_out,        # 输出int8 必传
    num_query_heads=Q_N,
    input_layout="BNSD",
    softmax_scale=softmax_scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.dtype)  # torch.int8
print(out.shape)  # torch.Size([1, 8, 164, 128])
```

### 示例6：伪量化（KV分离模式）+ Decode

**场景说明**：KV分离传入`dequant_scale_key`与`dequant_scale_value`，与`torch_npu.npu_fused_infer_attention_score`的同名示例相比用法相同，仅有参数名的差异。

**关键参数**：`key_quant_mode,value_quant_mode`：必选参数，取值需保持一致，为0时使用perchannel模式。

**相关约束**：[Q_S=1（增量推理）约束](#q_s1增量推理约束)

```python
import torch
import torch_npu
import math


B, Q_N, Q_S, D = 1, 8, 1, 128       # Q_S=1 增量推理
KV_N = 2                            
KV_S = 2048

# query: bfloat16, BNSD
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.bfloat16).npu()
k = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()

# dequant_scale: 必传，perchannel，dtype与query一致
# shape = (1, KV_N, 1, D)
dequant_scale_key   = torch.randn(1, KV_N, 1, D, dtype=torch.bfloat16).npu()
dequant_scale_value = torch.randn(1, KV_N, 1, D, dtype=torch.bfloat16).npu()
# dequant_offset: 可选，shape与scale一致

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(
    q, k, v,
    dequant_scale_key=dequant_scale_key,            # KV分离：key反量化因子
    dequant_scale_value=dequant_scale_value,        # KV分离：value反量化因子
    key_quant_mode=0,                               # perchannel
    value_quant_mode=0,                             # perchannel，需与key一致
    num_query_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="BNSD",
    softmax_scale=softmax_scale
)

print(out.dtype)   # torch.bfloat16
print(out.shape)   # torch.Size([1, 8, 1, 128])
```

### 示例7：通用约束+aclgraph模式

**场景说明**：当显式传入`backend="npugraph_ex"`时，使用[aclgraph模式](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/quick_start.md)。

**关键参数**：`backend="npugraph_ex"`：`torch.compile`的必选参数，使用aclgraph模式。

**相关约束**：[通用约束](#通用约束)

```python
import torch
import torch_npu
import math
import torchair as tng

import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
from torch.library import Library, impl

# 数据生成
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
softmax_scale = 1/math.sqrt(128.0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535)

def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
        graph_output = model()
    single_op = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535)
    print("single op output with mask:", single_op[0], single_op[0].shape)
    print("graph output with mask:", graph_output[0], graph_output[0].shape)

if __name__ == "__main__":
    MetaInfershape()
```

执行上述代码的输出类似如下：

```text
single op output with mask: tensor([[[[-0.0417,  0.0780,  0.0827,  ...,  0.0655, -0.0575,  0.0035],
          [-0.0108, -0.0408, -0.0359,  ...,  0.0466,  0.0022,  0.0015],
          [ 0.0052, -0.0380,  0.1528,  ...,  0.0970,  0.0013,  0.0779],
          ...,
          [ 0.0104, -0.0696, -0.0266,  ..., -0.0652, -0.0572, -0.0585],
          [-0.0435, -0.0248, -0.0358,  ..., -0.0403,  0.0307, -0.0343],
          [ 0.0108, -0.0703, -0.0366,  ..., -0.0621,  0.0567, -0.0403]]]],
       device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
graph output with mask: tensor([[[[-0.0417,  0.0780,  0.0827,  ...,  0.0655, -0.0575,  0.0035],
          [-0.0108, -0.0408, -0.0359,  ...,  0.0466,  0.0022,  0.0015],
          [ 0.0052, -0.0380,  0.1528,  ...,  0.0970,  0.0013,  0.0779],
          ...,
          [ 0.0104, -0.0696, -0.0266,  ..., -0.0652, -0.0572, -0.0585],
          [-0.0435, -0.0248, -0.0358,  ..., -0.0403,  0.0307, -0.0343],
          [ 0.0108, -0.0703, -0.0366,  ..., -0.0621,  0.0567, -0.0403]]]],
       device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])

```
