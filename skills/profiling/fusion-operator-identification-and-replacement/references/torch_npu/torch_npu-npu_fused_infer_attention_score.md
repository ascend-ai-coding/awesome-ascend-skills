# torch_npu.npu_fused_infer_attention_score

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

- **API功能**：适配增量&全量推理场景的FlashAttention算子。当`query`矩阵的S=1，进入IncreFlashAttention（decode）分支；其余场景进入PromptFlashAttention（prefill）分支。

- **计算公式**：

  $$
  attention\_out = softmax \left(scale \cdot (query \cdot key^\top) + atten\_mask \right) \cdot value
  $$

## 函数原型

```python
torch_npu.npu_fused_infer_attention_score(
    query, key, value, *,
    pse_shift=None, atten_mask=None,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    dequant_scale1=None, quant_scale1=None,
    dequant_scale2=None, quant_scale2=None, quant_offset2=None,
    antiquant_scale=None, antiquant_offset=None,
    block_table=None, query_padding_size=None, kv_padding_size=None,
    key_antiquant_scale=None, key_antiquant_offset=None,
    value_antiquant_scale=None, value_antiquant_offset=None,
    key_shared_prefix=None, value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None, key_rope=None, key_rope_antiquant_scale=None,
    num_heads=1, scale=1.0,
    pre_tokens=2147483647, next_tokens=2147483647,
    input_layout="BSH", num_key_value_heads=0,
    sparse_mode=0, inner_precise=0,
    block_size=0, antiquant_mode=0,
    softmax_lse_flag=False,
    key_antiquant_mode=0, value_antiquant_mode=0
) -> (Tensor, Tensor)
```

  > [!NOTE]
  > `*`之前的`query, key, value`为位置参数（必须按顺序传入）；`*`之后的为关键字参数（键值对赋值，可选，不传使用默认值）。

## 参数说明

### 按场景分类的参数速查

| 场景 | 必选参数 | 关键可选参数 | 对应约束章节 |
| ---- | -------- | ------------ | ------------ |
| 基础Prefill（MHA） | `query, key, value` | `atten_mask, num_heads, scale, input_layout` | [通用基础约束](#base_constraints), [Q_S > 1约束](#qs_gt_1) |
| 基础Decode（MHA） | `query, key, value` | `atten_mask, num_heads, scale, input_layout, actual_seq_lengths_kv` | [通用基础约束](#base_constraints), [Q_S = 1约束](#qs_eq_1) |
| Decode + PageAttention | `query, key, value, block_table, block_size` | `actual_seq_lengths_kv` | [PageAttention约束](#pa_constraints) |
| GQA | `query, key, value, num_heads, num_key_value_heads` | — | [通用基础约束](#base_constraints) |
| MLA | `query, key, value, query_rope, key_rope` | `num_heads, scale, input_layout` | [MLA约束](#mla_constraints) |
| Prefix（共享前缀） | `query, key, value, key_shared_prefix, value_shared_prefix` | `actual_shared_prefix_len` | [Prefix约束](#prefix_constraints) |
| `torch.int8`全量化 | `query, key, value, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2` | `quant_offset2` | [`torch.int8`量化约束](#int8_constraints) |
| `torch.int8`后量化 | `query, key, value, quant_scale2` | `quant_offset2` | [`torch.int8`量化约束](#int8_constraints) |
| 伪量化（KV由`torch.int8`反量化为`torch.float16`） | `query, key, value, antiquant_scale` | `antiquant_offset, antiquant_mode` | [伪量化约束](#pseudo_quant_constraints) |
| 伪量化分离模式 | `query, key, value, key_antiquant_scale, value_antiquant_scale` | `key_antiquant_mode, value_antiquant_mode` | [Q_S > 1约束](#qs_gt_1) / [Q_S = 1约束](#qs_eq_1) |
| 左Padding | `query, key, value, actual_seq_lengths, query_padding_size` | `kv_padding_size` | [Padding约束](#padding_constraints) |
| Ring Attention | `query, key, value, softmax_lse_flag=True` | — | — |
| TND布局 | `query, key, value, actual_seq_lengths, actual_seq_lengths_kv, input_layout="TND"` | — | [TND布局约束](#tnd_constraints) |

### 各轴取值范围速查

> **维度约定**：B = Batch Size, S = Sequence Length, H = Hidden Size, N = Head Num, D = Head Dim（满足D = H / N）, T =所有Batch序列长度的累加和。
> Q_S/S1 = `query`的S, KV_S/S2 = `key/value`的S, Q_N = num_query_heads, KV_N = `num_key_value_heads`。

| 轴 | 含义 | Q_S > 1上限 | Q_S = 1上限 | 对齐要求 |
| -- | ---- | ------------ | ------------ | -------- |
| B | Batch Size | ≤ 65536 | ≤ 65536 | — |
| N (Q_N) | `query` head数 | ≤ 256 | ≤ 256 | — |
| N (KV_N) | `key/value` head数 | ≤ 256 | ≤ 256 | — |
| D | Head Dim | ≤ 512 | ≤ 512 | `torch.float16`/`torch.bfloat16`： 16字节对齐、 `torch.int8`： 32字节对齐、 `torch.int4`： 64字节对齐 |
| S (Q_S) | `query`序列长度 | ≤ 20971520 (20M) | = 1 | — |
| S (KV_S) | `key/value`序列长度 | ≤ 20971520 (20M) | ≤ 262144 | — |

### 详细参数说明

#### 必选参数（位置参数）

- **query** (`Tensor`)：attention的Query输入。数据格式：$ND$，不支持非连续Tensor。

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型：`torch.float16`、`torch.bfloat16`。
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型：`torch.float16`、`torch.bfloat16`。
  <!-- end id5 -->

- **key** (`Tensor`)：attention的Key输入。数据类型：`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch.int4`（`torch.int32`容器，即8个`torch.int4`拼接为一个`torch.int32`），数据格式：$ND$，不支持非连续Tensor。

- **value** (`Tensor`)：attention的Value输入。数据类型：`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch.int4`（`torch.int32`容器），数据格式：$ND$，不支持非连续Tensor。

#### 位置编码参数

- **pse_shift** (`Tensor`,可选)：位置编码参数。数据类型：`torch.float16`、`torch.bfloat16`（需与`query`类型满足推导规则），数据格式：$ND$，不支持非连续，默认值：None。

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型：Q_S > 1时：shape为(B, Q_N, Q_S, KV_S)或(1, Q_N, Q_S, KV_S)；KV_S非32字节对齐建议padding到32字节。Q_S = 1时：shape为(B, Q_N, 1, KV_S)或(1, Q_N, 1, KV_S)；仅支持D轴16整除。
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - Q_S不为1，要求在`pse_shift`为`torch.float16`类型时，此时的`query`为`torch.float16`类型；而在`pse_shift`为`torch.bfloat16`类型时，要求此时`query`为`torch.bfloat16`类型。输入shape类型需为(B, N, Q_S, KV_S)或(1, N, Q_S, KV_S)，其中Q_S为query的shape中的S，KV_S为key和value的shape中的S。对于`pse_shift`的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
    - Q_S为1，要求在`pse_shift`为`torch.float16`类型时，此时的`query`为`torch.float16`类型；而在`pse_shift`为`torch.bfloat16`类型时，要求此时`query`为`torch.bfloat16`类型。输入shape类型需为(B, N, 1, KV_S)或(1, N, 1, KV_S)，其中N为num_heads，KV_S为key和value的shape中的S。对于`pse_shift`的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
  <!-- end id7 -->

#### Mask与序列长度参数

- **atten_mask** (`Tensor`,可选)：对Q×K结果做mask，指示Token间相关性是否计算。数据类型：`torch.bool`、`torch.int8`、`torch.uint8`，数据格式：$ND$，不支持非连续。默认值：None。

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`sparse_mode`=0/1时：支持shape (1, Q_S, KV_S)、(B, 1, Q_S, KV_S)、(1, 1, Q_S, KV_S)；当`input_layout`为BSH/BSND/BNSD/BNSD_BSND且不传rope时，Q_S=1支持(B, KV_S)，Q_S>1支持(Q_S, KV_S)。`sparse_mode`=2/3/4时：shape必须为(2048, 2048)或(1, 2048, 2048)或(1, 1, 2048, 2048)，需用户保证为下三角。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - Q_S不为1时建议shape输入(B, Q_S, KV_S)、(1, Q_S, KV_S)、(B, 1, Q_S, KV_S)、(1, 1, Q_S, KV_S)。
    - Q_S为1时建议shape输入(B, 1, KV_S)、(B, 1, 1, KV_S)。
    - 其中Q_S为`query`的shape中的S，KV_S为`key`和`value`的shape中的S，但如果Q_S、KV_S非16或32对齐，可以向上取到对齐的S。
  <!-- end id9 -->

- **actual_seq_lengths** (`List[int]`,可选)：各Batch中`query`的有效序列长度。数据类型：`torch.int64`，默认值：None（表示与`query`的S相同）。

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：每个batch有效seqlen不大于`query`中对应batch的seqlen。传入长度为1时，所有batch共用；长度 ≥ batch时取前batch个。**TND布局时必须传入**：每个元素值为当前batch与之前所有batch的seqlen **累加和**（后值 ≥ 前值，非负），元素个数即batch值（≤ 4096）。
  <!-- end id10 -->

- **actual_seq_lengths_kv** (`List[int]`,可选)：各Batch中`key/value`的有效序列长度。数据类型：`torch.int64`。默认值：None（表示与`key/value`的S相同）。

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：约束同`actual_seq_lengths`。PageAttention场景下必须传入。
  <!-- end id11 -->

#### `torch.int8`量化参数

- **dequant_scale1** (`Tensor`,可选)：BMM1后面的反量化因子，支持pertensor。数据类型：`torch.uint64`、`torch.float32`。默认值：None。

- **quant_scale1** (`Tensor`,可选)：BMM2前面的量化因子，支持pertensor。数据类型：`torch.float32`。默认值：None。

- **dequant_scale2** (`Tensor`,可选)：BMM2后面的反量化因子，支持pertensor。数据类型：`torch.uint64`、`torch.float32`。默认值：None。

- **quant_scale2** (`Tensor`,可选)：输出的量化因子，支持pertensor、perchannel。
  - 数据类型：`torch.float32`、`torch.bfloat16`（`torch.bfloat16`输入时两种均可，否则仅`torch.float32`）。默认值：None。
  - perchannel下：输出BSH：所有维度乘积= H，建议shape (1, 1, H)或(H,)。输出BNSD：乘积= Q_N × D，建议shape (1, Q_N, 1, D)或(Q_N, D)。输出BSND：乘积= Q_N × D，建议shape (1, 1, Q_N, D)或(Q_N, D)。

- **quant_offset2** (`Tensor`,可选)：输出的量化偏移，支持pertensor、perchannel。若传入，类型和shape须与`quant_scale2`一致，数据类型：`torch.float32`、`torch.bfloat16`。默认值：None（等效于0）。

#### 伪量化参数

- **antiquant_scale** (`Tensor`,可选)：伪量化因子，pertensor或perchannel或pertoken。数据类型：`torch.float16`、`torch.bfloat16`（Q_S ≥ 2时仅`torch.float16`；Q_S = 1时perchannel与`query`同类型，pertoken为`torch.float32`），默认值：None。建议使用KV伪量化分离模式（`key_antiquant_scale`和`value_antiquant_scale`）。

  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT系列产品</term>：不支持此参数。
  <!-- end id12 -->

- **antiquant_offset** (`Tensor`,可选)：伪量化偏移，pertensor或perchannel或pertoken。约束同`antiquant_scale`（shape须一致）。默认值：None。对称量化时不传（None），非对称量化时必须传入。

  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT系列产品</term>：不支持此参数。
  <!-- end id13 -->

- **antiquant_mode** (`int`,可选)：伪量化方式。0=perchannel（含pertensor），1=pertoken。默认值：0。

  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：Q_S ≥ 2时该参数无效；Q_S = 1时传入0和1之外的其他值会执行异常。
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持取值0、1。
  <!-- end id15 -->

#### PageAttention参数

- **block_table** (`Tensor`,可选)：PageAttention中KV存储的block映射表。数据类型：`torch.int32`，数据格式：$ND$，必须为二维：[B, maxBlockNumPerSeq]，blockid合法性由用户保证，传入此参数即开启PageAttention。默认值：None。

- **block_size** (`Tensor`,可选)：PageAttention中每个block最多token个数。数据类型：`torch.int64`，Q_S > 1：最小128，最大512，须为128倍数。 Q_S = 1： 非0值，最大512。`torch.float16`/`torch.bfloat16`需16字节对齐，`torch.int8`需32字节对齐，推荐128。默认值：0（不开启PageAttention）。

  <!-- npu="950" id16 -->
  - <term>Ascend 950PR&950DT系列产品</term>：Q_S = 1： key、value输入类型是`torch.int4（torch.int32）`时需要64对齐。
  <!-- end id16 -->

#### Padding参数

- **query_padding_size** (`Tensor`,可选)：`query`每个batch右对齐的padding数量。数据类型：`torch.int64`，仅Q_S > 1生效。需与`actual_seq_lengths`一起开启，否则默认为右padding。默认值：None。

- **kv_padding_size** (`Tensor`,可选)：`key`或`value`每个batch右对齐的padding数量。数据类型：`torch.int64`，小于0时被置为0。需与`actual_seq_lengths_kv`一起开启，否则默认为右padding。默认值：None。

#### KV伪量化分离参数

- **key_antiquant_scale** (`Tensor`,可选)：KV伪量化分离时`key`的反量化因子。数据类型：`torch.float16`、`torch.bfloat16`、`torch.float32`，支持perchannel、pertensor、pertoken、pertensor+perhead、pertoken+perhead、pertoken+PA等模式。与`value_antiquant_scale`必须同时为空或同时非空。默认值：None。

- **key_antiquant_offset** (`Tensor`,可选)：KV伪量化分离时`key`的反量化偏移。约束同`key_antiquant_scale`。与`value_antiquant_offset`必须同时为空或同时非空。默认值：None。

- **value_antiquant_scale** (`Tensor`,可选)：KV伪量化分离时`value`的反量化因子。约束同`key_antiquant_scale`。默认值：None。

- **value_antiquant_offset** (`Tensor`,可选)：KV伪量化分离时`value`的反量化偏移。约束同`key_antiquant_scale`。默认值：None。

- **key_antiquant_mode** (`int`,可选)：`key`伪量化方式。取值0~5,详见表格。默认值：0。

- **value_antiquant_mode** (`int`,可选)：`value`伪量化方式。取值0~5，详见表格。默认值：0。

  | 值 | 模式 | 说明 |
  | -- | ---- | ---- |
  | 0 | perchannel | perchannel模式（包含pertensor）。默认值。 |
  | 1 | pertoken | pertoken模式。 |
  | 2 | pertensor + perhead | pertensor叠加perhead模式。 |
  | 3 | pertoken + perhead | pertoken叠加perhead模式。 |
  | 4 | pertoken + page attention | pertoken叠加page attention管理scale/offset。 |
  | 5 | pertoken + perhead + page attention | pertoken叠加perhead并使用page attention管理scale/offset。 |

> **特殊约束**：
>
><!-- npu="A3,910b" id47 -->
> - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：Q_S ≥ 2时仅支持0、1；Q_S = 1时支持0~5。
><!-- end id47 -->
><!-- npu="950" id48 -->
> - <term>Ascend 950PR&950DT系列产品</term>：`key_antiquant_mode`支持取值0、1、2、3、4、5。`value_antiquant_mode`：Q_S ≥ 2时支持0、1、2、3、4、5；Q_S = 1时支持0、1、2、3、4、5。除`key_antiquant_mode=0`且`value_antiquant_mode=1`的场景外，`key_antiquant_mode`与`value_antiquant_mode`必须一致。
><!-- end id48 -->

#### Prefix（共享前缀）参数

- **key_shared_prefix** (`Tensor`,可选)：Key的共享前缀部分。数据类型：`torch.float16`、`torch.bfloat16`、`torch.int8`。数据格式：$ND$，不支持非连续。shape第一维batch必须为1。与`value_shared_prefix`必须同时为空或同时非空。默认值：None。

- **value_shared_prefix** (`Tensor`,可选)：Value的共享前缀部分。约束同`key_shared_prefix`。默认值：None。

- **actual_shared_prefix_len** (`List[int]`,可选)：共享前缀的有效序列长度。数据类型：`torch.int64`。shape为[1]，值不能大于prefix的S。默认值：None（表示与prefix的S相同）。

#### MLA（Multi-head Latent Attention）参数

- **query_rope** (`Tensor`,可选)：MLA中Query的RoPE信息。数据类型：`torch.float16`、`torch.bfloat16`。数据格式：$ND$，不支持非连续。shape中D为64，其余维度与`query`一致。与`key_rope`必须同时配置或同时不配置。默认值：None。

- **key_rope** (`Tensor`,可选)：MLA中Key的RoPE信息。约束同`query_rope`（shape中D为64，其余维度与`key`一致）。默认值：None。

- **key_rope_antiquant_scale** (`Tensor`,可选)：预留参数，暂未使用。使用默认值即可。

#### 其他标量参数

- **num_heads** (`int`,可选)：`query`的head个数。默认值：1。

  <!-- npu="A3,910b" id17 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：BNSD场景下需与`query`的N轴一致。
  <!-- end id17 -->
  <!-- npu="950" id18 -->
  - <term>Ascend 950PR&950DT系列产品</term>：BNSD、BSND、BNSD_BSND、TND场景下需与`query`的N轴一致。
  <!-- end id18 -->

- **input_layout** (`str`,可选)：输入`query/key/value`的数据排布格式。**格式中带下划线时**，下划线左边为输入layout，右边为输出layout，算子内部自动layout转换，详见表格，默认值："BSH"。

  | `input_layout` | 输入shape（`query`） | 输入shape（`key/value`） | 输出shape | 适用场景 |
  | ------------ | ------------------- | ----------------------- | ---------- | -------- |
  | `"BSH"` | (B, Q_S, H) | (B, KV_S, H) | (B, Q_S, H) | 通用，H = N × D |
  | `"BSND"` | (B, Q_S, Q_N, D) | (B, KV_S, KV_N, D) | (B, Q_S, Q_N, D) | 通用，N/D分离 |
  | `"BNSD"` | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, Q_N, Q_S, D) | 通用，N轴在前 |
  | `"BNSD_BSND"` | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, Q_S, Q_N, D) | 仅Q_S > 1 |
  | `"BSH_NBSD"` | (B, Q_S, H) | (B, KV_S, H) | (B, Q_N, Q_S, D) | layout转换 |
  | `"BSND_NBSD"` | (B, Q_S, Q_N, D) | (B, KV_S, KV_N, D) | (B, Q_N, Q_S, D) | layout转换 |
  | `"BNSD_NBSD"` | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, Q_N, Q_S, D) | 仅Q_S > 1且Q_S ≤ 16 |
  | `"TND"` | (T, Q_N, D) | (T, KV_N, D) | (T, Q_N, D) | 变长序列，T = Σseqlen |
  | `"TND_NTD"` | (T, Q_N, D) | (T, KV_N, D) | (Q_N, T, D) | 变长序列+ transpose |
  | `"NTD_TND"` | (Q_N, T, D) | (KV_N, T, D) | (T, Q_N, D) | 变长序列+ transpose |

<!-- npu="950" id46 -->
> **特殊约束**：
>
> - <term>Ascend 950PR&950DT系列产品</term>：支持BSH、BSND、BNSD、BNSD_BSND、TND（不支持左padding、tensorlist、pse）。
<!-- end id46 -->

- **scale** (`float`,可选)：缩放系数，通常为**1/√D**。 默认值为1.0，**大多数场景需手动设置，否则计算结果将不正确**。默认值：1.0。

- **pre_tokens** (`int`,可选)：稀疏计算中attention与前几个token的关联范围。Q_S=1时无效。默认值：2147483647。

- **next_tokens** (`int`,可选)：稀疏计算中attention与后几个token的关联范围。Q_S=1时无效。默认值：2147483647。

- **num_key_value_heads** (`int`,可选)：`key/value`中head个数，用于GQA。0表示与`query`的head数相等。需满足`num_heads`能被`num_key_value_heads`整除（即`num_heads` ÷ `num_key_value_heads`为整数），且比值 ≤ 64。默认值：0。

  <!-- npu="A3,910b" id19 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：BSND/BNSD/BNSD_BSND场景需与`key/value`的N轴一致。
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950PR&950DT系列产品</term>：BSND/BNSD/BNSD_BSND/TND场景需与`key/value`的N轴一致。
  <!-- end id20 -->

- **sparse_mode** (`int`,可选)：稀疏模式，详见表格，默认值：0。

  | `sparse_mode` | 模式 | 说明 | `atten_mask`要求 |
  | ----------- | ---- | ---- | --------------- |
  | 0 | defaultMask | 默认模式。不传`atten_mask`时不做mask；传入时使用完整mask矩阵，`pre_tokens`和`next_tokens`之间的部分参与计算。Q_S=1且不带rope时此参数无效。 | shape：(Q_S, KV_S)或(1, Q_S, KV_S)等 |
  | 1 | allMask | 必须传入完整`atten_mask`矩阵(S1×S2)。忽略`pre_tokens`和`next_tokens`。 | shape：(Q_S, KV_S)或(1, Q_S, KV_S)等 |
  | 2 | leftUpCausal | 左上角causal mask。需传入优化后的`atten_mask`矩阵。 | shape：(2048, 2048)或(1, 2048, 2048)或(1, 1, 2048, 2048) |
  | 3 | rightDownCausal | 右下角causal mask（以右顶点为划分的下三角）。需传入优化后的mask。 | shape：(2048, 2048)或(1, 2048, 2048)或(1, 1, 2048, 2048) |
  | 4 | band | band模式mask。需传入优化后的`atten_mask`矩阵。 | shape：(2048, 2048)或(1, 2048, 2048)或(1, 1, 2048, 2048) |
  | 5 | prefix | **暂不支持** | — |
  | 6 | global | **暂不支持** | — |
  | 7 | dilated | **暂不支持** | — |
  | 8 | block_local | **暂不支持** | — |

> **特殊约束**：Q_S = 1且不带rope输入时，`sparse_mode`参数无效。

- **inner_precise** (`int`,可选)：精度/性能模式，详见表格，默认值：0。

  | `inner_precise` | 精度模式 | 行无效修正 | 适用场景 |
  | ------------- | -------- | ---------- | -------- |
  | 0 | 高精度 | 否 | 默认，精度优先 |
  | 1 | 高性能 | 否 | 性能优先 |
  | 2 | 高精度 | 是 | mask存在整行全1时提升精度 |
  | 3 | 高性能 | 是 | mask存在整行全1时提升精度（性能略降） |

> [!NOTE]
>
>`torch.bfloat16`和`torch.int8`不区分高精度/高性能（bit0无效）。行无效修正对`torch.float16`、`torch.bfloat16`、`torch.int8`均生效。当mask存在整行全1时精度可能损失，可尝试配置2或3。Q_S = 1时仅支持0和1。若算子可判断出存在无效行场景（如`sparse_mode`=3且Q_S > KV_S），会自动开启行无效计算。

- **softmax_lse_flag** (`bool`,可选)：是否额外输出softmax_lse（用于Ring Attention分布式合并）。默认值：False。

## 返回值说明

- **attention_out** (`Tensor`)：注意力输出。
  - 数据类型：`torch.float16`、`torch.bfloat16`、`torch.int8`。数据格式：$ND$。D维度与`value`的D一致，其余维度与`query`的shape一致。

- **softmax_lse** (`Tensor`)：softmax的log-sum-exp值（用于Ring Attention分布式场景合并各设备结果）。
  - `softmax_lse_flag=True`时：`input_layout`为TND/NTD_TND时：shape (T, Q_N, 1)，dtype 为`torch.float32`。其他情况shape (B, Q_N, Q_S, 1)，dtype为`torch.float32`。
  - `softmax_lse_flag=False`时：shape (1,)，值为0的Tensor。

  Ring Attention多设备合并公式：

  $$
  O_{global} = \frac{e^{lse_1 - lse_{max}} \cdot O_1 + e^{lse_2 - lse_{max}} \cdot O_2}{e^{lse_1 - lse_{max}} + e^{lse_2 - lse_{max}}}
  $$

  其中$lse_{max} = max(lse_1, lse_2)$。

## 约束说明

### <a id="base_constraints"></a>通用基础约束

- 该接口支持推理场景、图模式。
- 与PyTorch配合使用时需CANN与PyTorch版本匹配。
- **入参为空处理**：`query`为空则直接返回空。`query`非空、`key/value`为空（S2=0）则`attention_out`按对应shape返回全0。`attention_out`为空则返回空。
- `key`与`value`的shape必须完全一致；非连续场景下`key/value` tensorlist中batch只能为1，个数等于`query`的B，N和D需相等。
- `scale`默认值1.0，通常应设置为**1/√D**，否则计算结果不正确。

### <a id="int8_constraints"></a>`torch.int8`量化约束

入参与输入/输出数据类型组合限制：

| 输入dtype | 输出dtype | 必须传入 | 可选传入 | 禁止传入 |
| ---------- | ---------- | -------- | -------- | -------- |
| `torch.int8` | `torch.int8` | `dequant_scale1, quant_scale1, dequant_scale2, quant_scale2` | `quant_offset2` | — |
| `torch.int8` | `torch.float16` | `dequant_scale1, quant_scale1, dequant_scale2` | — | `quant_offset2, quant_scale2` |
| `torch.float16`/`torch.bfloat16` | `torch.int8` | `quant_scale2` | `quant_offset2` | `dequant_scale1, quant_scale1, dequant_scale2` |

- `quant_scale2`和`quant_offset2`支持pertensor或perchannel，dtype支持`torch.float32`/`torch.bfloat16`。`quant_offset2`传入时须与`quant_scale2`类型和shape一致。
- `input_layout`仅支持BSH、BNSD、BSND、BNSD_BSND。

<!-- npu="A3,910b" id21 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  - 输出`torch.int8`，参数`quant_scale2`和`quant_offset2` 为perchannel时：不支持左padding、Ring Attention或D非32字节对齐。
  - 输出`torch.int8`时暂不支持sparse=band且`pre_tokens`/`next_tokens`为负。
  - 输出为`torch.int8`且`quant_offset2`非None时，若矩阵存在不参与计算的行，会触发拦截（建议外部做后量化）。
<!-- end id21 -->

### <a id="pseudo_quant_constraints"></a>伪量化约束（antiquant_scale / antiquant_offset）

- 仅支持kv dtype为`torch.int8`。
- perchannel：shape为(2, KV_N, 1, D)或(2, KV_N, D)或(2, H)，dtype与`query`相同。`antiquant_mode`=0。
- pertensor：shape为(2,)，dtype与`query`相同。`antiquant_mode`=0。
- pertoken：shape为(2, B, KV_S)，dtype固定为`torch.float32`。`antiquant_mode`=1。
- 对称量化：`antiquant_offset`=None；非对称量化：两者必须同时存在。
- 算子根据shape dim自动判断模式（dim=1 时为 pertensor，否则为 perchannel）。
- **建议使用KV伪量化分离模式**（用`key_antiquant_scale`和`value_antiquant_scale`代替`antiquant_scale`和`antiquant_offset`）。

### <a id="pa_constraints"></a>PageAttention约束

**开启条件**：`block_table`存在且有效。开启后`key/value`的`input_layout`参数无效，`key/value`按`block_table`索引在连续内存中排布。

| 约束项 | Q_S > 1 | Q_S = 1 |
| ------ | ------- | ------- |
| `block_size` | 最小128，最大512，须128倍数 | <li><term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：非0，最大512，推荐128。</li><li><term>Ascend 950PR&950DT系列产品</term>：`key`、`value`输入类型为`torch.float16`、`torch.bfloat16`时需要16对齐，`key`、`value`输入类型为`torch.int8`时需要32对齐，推荐使用128，`key`、`value`输入类型是`torch.int4（torch.int32）`时需要64对齐。</li> |
| kv dtype | `torch.float16`、`torch.bfloat16` | <li><term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.float16`、`torch.bfloat16`、`torch.int8`。</li><li><term>Ascend 950PR&950DT系列产品</term>：`torch.float16`、`torch.bfloat16`、`torch.int8`、`torch.int4`（`torch.int32`）</li> |
| `query` `torch.int8` | 不支持 | 不支持 |
| kv cache排布 | BSH/BSND仅BnBsH；BNSD/TND支持BnBsH和BnNBsD | 同左 |
| 必须传入`actual_seq_lengths_kv` | ✓ | ✓ |
| `block_table`二维 | [B, ≥maxBlockNumPerSeq] | [B, ≥maxBlockNumPerSeq] |
| blocknum下限 | ≥ 各batch block数之和 | 同左 |
| 支持tensorlist | ✗ | ✗ |
| 支持左padding | ✗ | ✗ |
| 支持伪量化 | ✗ | ✓（`torch.int8` kv） |
| 性能提示 | BnNBsD优于BnBsH | BnNBsD优于BnBsH |
| H超65535时 | 开GQA或用BnNBsD | 开GQA或用BnNBsD |

- PageAttention开启时，以下场景需KV_S ≥ maxBlockNumPerSeq × `block_size`：
  - 传入`atten_mask`（如shape (B, 1, Q_S, KV_S)）
  - 传入`pse_shift`（如shape (B, Q_N, Q_S, KV_S)）

  <!-- npu="A3,910b" id22 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - Q_S = 1时开启`atten_mask`（`sparse_mode`非2/3/4）：`atten_mask`最后一维 ≥ `block_table`第二维 × `block_size`。
    - Q_S = 1时开启`pse_shift`：`pse_shift`最后一维 ≥ `block_table`第二维 × `block_size`。
    - Q_S = 1伪量化pertoken模式：`antiquant_scale`/`antiquant_offset` shape的S ≥ 该值。
  <!-- end id22 -->
  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - 开启per-token叠加per-head模式：两个参数的shape均为(B, N, S)，数据类型固定为`torch.float32`，当key、value数据类型为`torch.int8`、`torch_npu.int4`(`torch.int32`)时支持。
  <!-- end id23 -->

### <a id="mla_constraints"></a>MLA约束

`query_rope`和`key_rope`输入时即进入MLA场景。

- `query_rope`和`key_rope`必须同时配置或同时不配置。
- `query_rope`的dtype/format与`query`一致；`key_rope`的dtype/format与`key`一致。
- `query`的D仅支持512或128。

**D = 512时**：

<!-- npu="A3,910b" id24 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  - sparse支持0/3/4。
  - `query_rope`：要求`query`的N为1/2/4/8/16/32/64/128，shape中D=64，其余维度与`query`一致。
  - `key_rope`：要求key的N=1、D=512，shape中D=64，其余维度与`key`一致。
  - `key/value/key_rope`支持$ND$或$NZ$。$NZ$格式：`torch.float16`/`torch.bfloat16` 的shape为 [blockNum, KV_N, D/16, blockSize, 16]；`torch.int8` 的shape为 [blockNum, KV_N, D/32, blockSize, 32]。
  - `input_layout`：BSH、BSND、BNSD、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND、TND_NTD。
  - 支持PageAttention（`block_size`：16的倍数、≤1024）。
  - 不支持：softmax_lse、左padding、tensorlist、pse、prefix、伪量化、全量化、后量化。
<!-- end id24 -->
<!-- npu="950" id25 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - sparse：Q_S等于1时只支持sparse=0且不传mask，Q_S大于1时支持sparse=0且不传mask或sparse=3且传入mask；
  - 支持`key、value`的`input_layout`格式为$ND$。
  - `input_layout`：BNSD、BSND、BSH、TND。
  - 该场景下，`input_layout`形状为BNSD/BSH/BSND仅支持`actual_seq_lengths_kv`参数。`input_layout`形状为TND必须传入`actual_seq_lengths`和`actual_seq_lengths_kv`参数。
<!-- end id25 -->

**D = 128时**：

<!-- npu="A3,910b" id26 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  - `input_layout`：BSH、BSND、TND、BNSD、NTD、BSH_BNSD、BSND_BNSD、BNSD_BSND、NTD_TND。
  - `query_rope`/`key_rope` shape中D=64，其余维度不变。
  - 不支持左padding、tensorlist、pse、prefix、伪量化、全量化、后量化。
  - 其余约束同TND/NTD_TND场景。
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - `query_rope`配置时要求`query_rope`的shape中b、n、s与`query`一致，d为64。
  - `key_rope`配置时要求`key_rope`的shape中b、n、s与`key`一致，d为64。
  - `input_layout`：BSH、BSND、BNSD、BNSD_BSND、TND。
  - 不支持page attention、prefix、伪量化、全量化、后量化。
  - 当kv为tensorlist时，`key_rope`的shape中b需要与tensorlist长度保持一致，n、s需要与tensorlist中每个tensor的n、s相等，d为64。
<!-- end id27 -->

### <a id="prefix_constraints"></a>Prefix（共享前缀）约束

- `key_shared_prefix`和`value_shared_prefix`必须同时为空或同时非空。
- 非空时：两者与`key`/`value`维度相同、dtype一致。
- `key_shared_prefix`的shape第一维batch=1；BNSD/BSND下N、D轴与`key`一致；BSH下H与`key`一致。`value_shared_prefix`同理。两者的S应相等。
- `actual_shared_prefix_len`存在时：shape=[1]，值 ≤ prefix的S。
- prefix的S + `key/value`的S ≤ 原`key/value`的S限制。
- 不支持：PageAttention、左padding、tensorlist、qkv全`torch.int8`。
- sparse=0/1且传入`atten_mask`时：S2 ≥ `actual_shared_prefix_len` + `key`的S。

### <a id="padding_constraints"></a>Padding约束

**query左padding（仅Q_S > 1）**：

- 搬运起点= Q_S − `query_padding_size` − `actual_seq_lengths`（须 ≥ 0）。
- 搬运终点= Q_S − `query_padding_size`（须 ≤ Q_S）。
- `kv_padding_size` < 0时被置为0。
- 需与`actual_seq_lengths`一起开启。
- 不支持PageAttention。

**kv左padding**：

- 搬运起点= KV_S − `kv_padding_size` − `actual_seq_lengths_kv`（Q_S > 1时须 ≥ 0；Q_S = 1时若< 0则返回全0）。
- 搬运终点= KV_S − `kv_padding_size`（Q_S > 1时须 ≤ KV_S；Q_S = 1时若< 0则返回全0）。
- 需与`actual_seq_lengths_kv`一起开启。
- 不支持PageAttention。

<!-- npu="A3,910b" id28 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  - 不支持Q为`torch.bfloat16`/`torch.float16`、KV为`torch.int4`的场景。
<!-- end id28 -->

### <a id="tnd_constraints"></a>TND布局约束

- `actual_seq_lengths`和`actual_seq_lengths_kv`必须传入，元素个数即batch值（≤ 4096）。每个元素值为累加和（后值 ≥ 前值）。

<!-- npu="A3,910b" id29 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  
  `input_layout`为TND、TND_NTD、NTD_TND时的综合限制：

  - **query D = 512时**：

    - sparse：0/3/4。
    - 支持TND、TND_NTD。
    - 支持PageAttention（`actual_seq_lengths_kv`长度 = kv batch数，值 ≤ KV_S）。
    - `query` N = 1/2/4/8/16/32/64/128，kv N = 1。
    - `query_rope`和`key_rope`非空（D=64）。
    - 不支持：左padding、tensorlist、pse、prefix、伪量化、全量化。
  - **query D ≠ 512时**：
    - 无rope：Q_D、K_D、V_D = 128，或Q_D、K_D = 192且V_D = 128或192（NTD不支持V_D=192；NTD_TND要求Q_D、K_D = 128或192，V_D = 128）。
    - 有rope：Q_D、K_D、V_D = 128。
    - 支持TND、NTD、NTD_TND。
    - PageAttention仅`block_size`须为16的倍数且 ≤ 1024。
    - 不支持：左padding、tensorlist、pse、prefix、伪量化、全量化、后量化。
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - 不支持左padding、tensorlist、pseType=0、prefix。
<!-- end id30 -->

### <a id="qs_gt_1"></a>Q_S > 1（Prefill）专属约束

- **B轴**：≤ 65536。

  <!-- npu="A3,910b" id31 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：D轴非32字节对齐时 ≤ 128。
  <!-- end id31 -->
  <!-- npu="950" id32 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持B轴小于等于65536。
  <!-- end id32 -->

- **N轴**：≤ 256。
- **D轴**：≤ 512。BSH/BSND时建议N×D < 65535。
- **S轴**：≤ 20971520 (20M)。
- **D对齐**：

  <!-- npu="A3,910b" id33 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.int8` 须32字节对齐，`torch_npu.int4` 须64字节对齐，`torch.float16`/`torch.bfloat16` 须16字节对齐。
  <!-- end id33 -->
  <!-- npu="950" id34 -->
  - <term>Ascend 950PR&950DT系列产品</term>：非量化场景：`query`，`key`，`value`的类型全部为`torch.float16`、`torch.bfloat16`，D轴1-512全部支持。伪量化场景：`query`类型为`torch.float16`、`torch.bfloat16`，`key`、`value`类型为`torch.int8`/`torch.int4`（`torch.int32`），其中当`key`、`value`类型为`torch.int4`（`torch.int32`）D轴仅支持64对齐（`torch.int32`仅支持D轴8对齐）。
  <!-- end id34 -->

<!-- npu="950" id35 -->
- **actual_seq_lengths**：

  - <term>Ascend 950PR&950DT系列产品</term>：该参数应为非负数，在`input_layout`不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，其长度为1或大于等于`query`的Batch值，该入参中的值代表每个Batch的实际长度，其值应该不大于Q_S。当`input_layout`为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
<!-- end id35 -->

<!-- npu="950" id36 -->
- **actual_seq_lengths_kv**：

  - <term>Ascend 950PR&950DT系列产品</term>：该参数传入时应为非负数，在`input_layout`不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，该入参中每个Batch的有效seqlenKv应该不大于`key/value`中对应Batch的seqlenKv。当本参数的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当`key/value`的`input_layout`为TND时，该入参必须传入，且该入参元素的数量等于Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlenKv和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
<!-- end id36 -->
- **长序列超时风险**：B×N×S×D过大时可能超时（aicore timeout/trap error），建议S切分。典型高风险场景：
  - B=1, Q_N=20, Q_S=2097152, D=256, KV_N=1, KV_S=2097152
  - B=1, Q_N=2, Q_S=20971520, D=256, KV_N=2, KV_S=20971520
  - B=20, Q_N=1, Q_S=2097152, D=256, KV_N=1, KV_S=2097152
  - B=1, Q_N=10, Q_S=2097152, D=512, KV_N=1, KV_S=2097152
- **sparse_mode**：仅支持0/1/2/3/4，其他值报错。
  - 0：`atten_mask`=None或在左padding时忽略`pre_tokens`和`next_tokens`。
  - 2/3/4：mask shape必须为(S,S)或(1,S,S)或(1,1,S,S)，其中S=2048，且用户保证下三角。
  - 1/2/3：忽略`pre_tokens`和`next_tokens`。
- **kvCache反量化合成参数**：仅支持`torch.int8` 反量化为`torch.float16`。`key/value`的data range × `antiquant_scale`的data range须在(−1, 1)内（高性能模式）否则需高精度模式。
- **pse_shift**：`query` dtype为`torch.float16`/`torch.bfloat16`/`torch.int8`时可用。`torch.float16` `query` + `pse_shift` 时强制高精度模式。Q_S ≥ `query` S，KV_S ≥ `key` S。prefix时KV_S ≥ `actual_shared_prefix_len` + `key` S。
- **GQA伪量化+ KV NZ格式**：

  <!-- npu="A3,910b" id37 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - `query`: `torch.bfloat16`, `key/value`: `torch.int8`, D=128, Q_S=1~16。
    - `input_layout`: BSH/BSND/BNSD。
    - 仅PageAttention（`block_size`=128或512）。
    - `key/value` NZ格式：[blockNum, KV_N, D/32, blockSize, 32]。
    - `key/value_antiquant_scale` dtype：perchannel 时为 `torch.bfloat16`，pertoken 时为 `torch.float32`。
    - 仅KV分离模式，不支持offset、rope、左padding、tensorlist、pse、prefix、后量化。
    - `num_heads`与`num_key_value_heads`支持组合：(10,1)、(64,8)、(80,8)、(128,16)。
    - MTP=0：`sparse_mode`=0不传mask。MTP>0且<16：`sparse_mode`=3传优化mask (2048×2048)。
    - 仅高性能模式。
  <!-- end id37 -->

<!-- npu="950" id38 -->
- **kv伪量化参数分离**：

  - <term>Ascend 950PR&950DT系列产品</term>：
    - 当伪量化参数和KV分离量化参数同时传入时，以KV分离量化参数为准。
    - 除`key_antiquant_mode=0`且`value_antiquant_mode=1`的场景外，`key_antiquant_mode`与`value_antiquant_mode`取值需要保持一致。
    - `key_antiquant_scale`和`value_antiquant_scale`要么都为空，要么都不为空；`key_antiquant_offset`和`value_antiquant_offset`要么都为空，要么都不为空。
    - `key_antiquant_scale`和`value_antiquant_scale`都不为空时，其shape需要保持一致；`key_antiquant_offset`和`value_antiquant_offset`都不为空时，其shape需要保持一致。
    - 当`key_antiquant_mode=0`且`value_antiquant_mode=1`时，`key_antiquant_mode`和`value_antiquant_mode`取值无需一致，其他场景均需保持一致。
    - `key_antiquant_scale`和`value_antiquant_scale`要么都为空，要么都不为空；`key_antiquant_offset`和`value_antiquant_offset`要么都为空，要么都不为空。
    - `key_antiquant_scale`和`value_antiquant_scale`都不为空时，除`key_antiquant_mode=0`且`value_antiquant_mode=1`的场景外，其shape需要保持一致。
    - `key_antiquant_offset`和`value_antiquant_offset`都不为空时，除`key_antiquant_mode=0`且`value_antiquant_mode=1`的场景外，其shape需要保持一致。
    - 管理scale/offset的量化模式如下：

      > [!NOTE]
      > 注意scale、offset具体指`key_antiquant_scale`、`value_antiquant_scale`、`key_antiquant_offset`、`value_antiquant_offset`参数。

      | 量化模式 | 该场景下scale和offset条件 | 该场景下`key`和`value`条件 |
      | --- | --- | --- |
      | per-channel模式 | 两个参数shape支持(N, 1, D)，(N, D)，(H)，(1, N, 1, D)，(1, N, D)，(1, H)数据类型和`query`数据类型相同。 | 当`key、value`数据类型为`torch.int4`（`torch.int32`）、`torch.int8`时支持。 |
      | per-token模式 | 两个参数的shape均为(B, S)，数据类型固定为`torch.float32`。 | 当`key、value`数据类型为`torch.int4`（`torch.int32`）、`torch.int8`时支持。 |
      | per-tensor模式 | 两个参数的shape均为(1)，数据类型和`query`数据类型相同。 | 当`key、value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
      | per-tensor叠加per-head模式 | 两个参数的shape均为(N)，数据类型和`query`数据类型相同。 | 当`key、value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
      | per-token叠加per-head模式 | 两个参数的shape均为(B, N, S)，数据类型固定为`torch.float32`。 | `key、value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
      | per-token叠加使用page attention模式 | 两个参数的shape均为(blocknum, blocksize)，数据类型固定为`torch.float32`。 | `key、value`数据类型为`torch.int8`时支持。 |
      | per-token叠加per head并使用page attention模式 | 两个参数的shape均为(blocknum, N, blocksize)，数据类型固定为`torch.float32`。 | `key、value`数据类型为`torch.int8`时支持。 |
      | `key`支持per-channel叠加`value`支持per-token模式 | 对`key`支持per-channel，两个参数的shape可支持(N, 1, D)、(N, D)、(H)，(1, N, 1, D)、(1, N, D)、(1, H)且数据类型和`query`数据类型相同。对`value`支持per-token，两个参数的shape均为(B, S)并且数据类型固定为`torch.float32`。 | 当`key、value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
<!-- end id38 -->

### <a id="qs_eq_1"></a>Q_S = 1（Decode）专属约束

- **B轴**：≤ 65536。
- **N轴**：≤ 256。
- **D轴**：≤ 512。
- **S轴**（KV）：≤ 262144。
- **qkv全`torch.int8`**：不支持。
- **`torch_npu.int4`（`torch.int32`）伪量化**：
  - PyTorch入图调用仅支持KV `torch.int4`拼接为`torch.int32`输入（建议通过dynamic_quant生成：1个`torch.int32`包含8个`torch.int4`）。
  - KV N、D或H为实际值的1/8（prefix同理）。
  - `torch.int4`仅支持D 64字节对齐（`torch.int32`支持D 8字节对齐）。
- **actual_seq_lengths**：

  <!-- npu="A3,910b" id39 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：非TND布局时该参数无效。
  <!-- end id39 -->
  <!-- npu="950" id40 -->
  - <term>Ascend 950PR&950DT系列产品</term>：该参数在未输入rope参数且layout不为TND时不生效。输入rope参数时生效，传入时应为非负数，在`input_layout`不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，其长度为1或大于等于`query`的Batch值，该入参中的值代表每个Batch的实际长度，其值应该不大于Q_S。当`input_layout`为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
  <!-- end id40 -->

- **actual_seq_lengths_kv**：

  <!-- npu="A3,910b" id41 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：非TND布局时传入长度为1/≥batch；传入值 ≤ 对应KV_S。
  <!-- end id41 -->
  <!-- npu="950" id42 -->
  - <term>Ascend 950PR&950DT系列产品</term>：该参数应为非负数，在`input_layout`不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，该入参中每个Batch的有效Sequence Length应该不大于`key/value`中对应Batch的seqlenKv。当本参数的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当`input_layout`为TND/TND_NTD时，该入参必须传入，在非PA场景下，第b个值表示前b个Batch的S轴累加长度，其值应递增（大于等于前一个值）排列，且该入参元素的数量代表总Batch数，在PA场景下，其长度等于`key/value`的Batch值，代表每个Batch的实际长度，值不大于KV_S。
  <!-- end id42 -->

- **pse_shift**：dtype与`query`一致。

  <!-- npu="A3,910b" id43 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持D轴16整除。
  <!-- end id43 -->

- **kv左padding**：搬运起点/终点< 0时返回全0；需与`actual_seq_lengths_kv`一起开启。
- **`torch.int4` kv**：

  <!-- npu="A3,910b" id44 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：不支持左padding、page attention（Q: `torch.bfloat16`/`torch.float16` + KV: `torch_npu.int4`）。
  <!-- end id44 -->

<!-- npu="950" id45 -->
- **kv伪量化参数分离**：

  - <term>Ascend 950PR&950DT系列产品</term>：
    - 除`key_antiquant_mode=0`且`value_antiquant_mode=1`的场景外，`key_antiquant_mode`与`value_antiquant_mode`取值需要保持一致。
    - `key_antiquant_scale`和`value_antiquant_scale`要么都为空，要么都不为空；`key_antiquant_offset`和`value_antiquant_offset`要么都为空，要么都不为空。
    - `key_antiquant_scale`和`value_antiquant_scale`都不为空时，除`key_antiquant_mode=0`且`value_antiquant_mode=1`的场景外，其shape需要保持一致；`key_antiquant_offset`和`value_antiquant_offset`都不为空时，除`key_antiquant_mode=0`且`value_antiquant_mode=1`的场景外，其shape需要保持一致。
    - `torch_npu.int4`（`torch.int32`）伪量化场景不支持后量化。
    - 管理scale/offset的量化模式如下：

      > [!NOTE]
      > 注意scale、offset两个参数指`key_antiquant_scale`、`value_antiquant_scale`、`key_antiquant_offset`、`value_antiquant_offset`。
      > 当Q_S等于1时，支持下表所示的量化模式。

      | 量化模式 | 该场景下scale和offset条件 | 该场景下`key`和`value`条件 |
      | --- | --- | --- |
      | per-channel模式 | 两个参数shape支持(1, N, 1, D)，(1, N, D)，(1, H)，数据类型和`query`数据类型相同。 | 当`key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
      | per-tensor模式 | 两个参数的shape均为(1)，数据类型和`query`数据类型相同。 | 当`key`、`value`数据类型为`torch.int8`时支持。 |
      | per-token模式 | 两个参数的shape均为(1, B, S)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
      | per-tensor叠加per-head模式 | 两个参数的shape均为(N)，数据类型和`query`数据类型相同。 | 当`key`、`value`数据类型为`torch.int8`时支持。 |
      | per-token叠加per-head模式 | 两个参数的shape均为(B, N, S)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |
      | per-token叠加使用page attention模式 | 两个参数的shape均为(blocknum, blocksize)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int8`时支持。 |
      | per-token叠加per head并使用page attention模式 | 两个参数的shape均为(blocknum, N, blocksize)，数据类型固定为`torch.float32`。 | `key`、`value`数据类型为`torch.int8`时支持。 |
      | `key`支持per-channel叠加`value`支持per-token模式 | 对`key`支持per-channel，两个参数的shape可支持(1, N, 1, D)、(1, N, D)、(1, H)，且数据类型和`query`数据类型相同。对`value`支持per-token，两个参数的shape均为(1, B, S)并且数据类型固定为`torch.float32`。 | 当`key`、`value`数据类型为`torch.int4`（`torch.int32`）或`torch.int8`时支持。 |

<!-- end id45 -->
## 调用示例

### 示例1：MHA + Prefill（基础场景）

最简单的全量计算调用：单batch、8 head、D=128、BNSD布局。

```python
import torch
import torch_npu
import math

# =====数据准备=====
B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

# query/key/value形状: (B, N, S, D) — BNSD布局
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# scale必须设置为1/sqrt(D)
scale = 1.0 / math.sqrt(D)

# actual_seq_lengths:长度为1表示所有batch共用
# actual_seq_lengths_kv: key/value的有效长度
act_seq_len = [Q_S]
act_seq_len_kv = [KV_S]

# =====调用算子=====
# MHA: num_heads == num_key_value_heads (或不传num_key_value_heads默认为0)
out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    actual_seq_lengths=act_seq_len,
    actual_seq_lengths_kv=act_seq_len_kv,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> **关键参数说明**：
>
> - `scale=1/√128`：必设，控制softmax之前QK^T的缩放。设错会导致attention分布异常。
>
> - `pre_tokens=65535, next_tokens=65535`：表示不限制attention范围（等价于双向attention）。
>
> - `input_layout="BNSD"`：表示输入为(B,N,S,D)。
>
> - 此场景命中Q_S > 1，走PromptFlashAttention分支。

### 示例2：GQA + Decode + PageAttention

分组查询注意力+ KV Cache + PageAttention的增量推理场景。

```python
import torch
import torch_npu
import math

# =====数据准备=====
B = 1
Q_N = 8           # query heads
KV_N = 2          # key/value heads (GQA: Q_N / KV_N = 4)
Q_S = 1           # decode阶段query长度= 1
D = 128
KV_S = 2048       # KV Cache总长度
block_size = 128
#计算需要的block数量
max_block_num = (KV_S + block_size - 1) // block_size  # = 16

# BSH layout: query为3维(B, S, H_q)，其中H_q = Q_N * D
H_q = Q_N * D    # = 1024
H_kv = KV_N * D  # = 256
q = torch.randn(B, Q_S, H_q, dtype=torch.float16).npu()
# KV Cache: BnBsH格式(blocknum, block_size, H_kv)
k = torch.randn(max_block_num, block_size, H_kv, dtype=torch.float16).npu()
v = torch.randn(max_block_num, block_size, H_kv, dtype=torch.float16).npu()

# block_table: (B, max_block_num)，存储block id
block_table = torch.arange(max_block_num, dtype=torch.int32).unsqueeze(0).npu()
scale = 1.0 / math.sqrt(D)

# PageAttention场景必须传入actual_seq_lengths_kv
act_seq_len_kv = [KV_S]

# =====调用算子=====
out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    actual_seq_lengths_kv=act_seq_len_kv,
    num_heads=Q_N,
     num_key_value_heads=KV_N,   # GQA:指定KV head数
    input_layout="BSH",          # kv cache为BnBsH时用BSH
    scale=scale,
    block_table=block_table,
    block_size=block_size
)

print(out.shape)  # torch.Size([1, 1, 1024])
```

> **关键参数说明**：
>
> - `num_key_value_heads=2`：启用GQA，num_heads / num_key_value_heads = 4。该参数不传时默认为0（即MHA），见[参数说明](#详细参数说明)。
>
> - `block_table` + `block_size`：传入即开启PageAttention，kv cache排布见[PageAttention约束](#pa_constraints)。
>
> - `input_layout="BSH"`：kv cache为BnBsH格式时必须用BSH。若kv cache为BnNBsD格式且query为BNSD，则可用BNSD。
>
> - `actual_seq_lengths_kv`：PageAttention场景必须传入，对应约束见[PageAttention约束](#pa_constraints)。
>
> -此场景Q_S=1，走IncreFlashAttention分支，约束见[Q_S = 1约束](#qs_eq_1)。

### 示例3：MHA + Causal Mask（带atten_mask）

使用causal attention mask的标准decoder场景。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 164

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

#下三角causal mask: (1, Q_S, KV_S)
mask = torch.tril(torch.ones(Q_S, KV_S, dtype=torch.bool)).unsqueeze(0).npu()

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    atten_mask=mask,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    sparse_mode=0,     # defaultMask:使用完整mask矩阵
    pre_tokens=65535,
    next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> **关键参数说明**：
>
> - `atten_mask`：shape (1, Q_S, KV_S)，`torch.bool`类型。更多mask shape选项见[atten_mask参数说明](#详细参数说明)。
>
> - `sparse_mode=0`：defaultMask模式，使用`pre_tokens`/`next_tokens`控制计算范围。此处为完整双向attention。
>
> -也可使用`sparse_mode=3`（rightDownCausal）+ 2048×2048优化mask。

### 示例4：MLA + Decode（Multi-head Latent Attention）

DeepSeek类模型的MLA推理场景。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 16, 1, 512   # MLA场景D=512
KV_N, KV_S = 1, 2048
rope_D = 64                       # RoPE维度固定64

# query: (B, Q_N, Q_S, D)
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
# key/value: (B, KV_N, KV_S, D)
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# RoPE张量: shape中D=64，其余维度与q/k一致
q_rope = torch.randn(B, Q_N, Q_S, rope_D, dtype=torch.float16).npu()
k_rope = torch.randn(B, KV_N, KV_S, rope_D, dtype=torch.float16).npu()

scale = 1.0 / math.sqrt(D + rope_D)  # MLA场景有效维度= D + 64

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    query_rope=q_rope,
    key_rope=k_rope,
    num_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="BNSD",
    scale=scale,
    sparse_mode=0
)

print(out.shape)  # torch.Size([1, 16, 1, 512])
```

> **关键参数说明**：
>
> - `query_rope` + `key_rope`：传入即进入MLA分支，必须同时传入。RoPE维度固定64。二者dtype/format须与query/key一致。
>
> - D=512：MLA prefill场景的D为512或128。D=512时可开启PageAttention、支持NZ格式等。约束详见[MLA约束](#mla_constraints)。
>
> -此场景不支持：softmax_lse、左padding、pse、prefix、伪量化、全量化。

### 示例5：`torch.int8`后量化

仅对输出做`torch.int8`量化（post-training quantization），输入保持`torch.float16`。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# pertensor量化因子: shape (1, 1, 1)或标量shape
quant_scale2 = torch.tensor([0.01], dtype=torch.float32).npu()
#不传quant_offset2则默认为0

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    quant_scale2=quant_scale2,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.dtype)  # torch.int8
print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> **关键参数说明**：
>
> - `quant_scale2`：传入即触发后量化（输出`torch.int8`）。不传`quant_offset2`则偏移为0。
>
> - `torch.int8`输出场景下，**禁止**同时传入`dequant_scale1`/`quant_scale1`/`dequant_scale2`。约束详见[`torch.int8`量化约束](#int8_constraints)。
>
> -如需perchannel量化，shape建议(Q_N, D)。详见[quant_scale2参数说明](#详细参数说明)。

### 示例6：`torch.int8`全量化

输入`torch.int8`，内部反量化为`torch.float16`计算，再量化输出`torch.int8`。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

# int8输入（需确保D轴32字节对齐）
q = torch.randint(-128, 127, (B, Q_N, Q_S, D), dtype=torch.int8).npu()
k = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()

# BMM1后反量化因子(pertensor)
deq_scale1 = torch.tensor([0.01], dtype=torch.float32).npu()
# BMM2前量化因子
quant_scale1 = torch.tensor([0.02], dtype=torch.float32).npu()
# BMM2后反量化因子
deq_scale2 = torch.tensor([0.03], dtype=torch.float32).npu()
#输出量化因子
quant_scale2 = torch.tensor([0.04], dtype=torch.float32).npu()

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    dequant_scale1=deq_scale1,
    quant_scale1=quant_scale1,
    dequant_scale2=deq_scale2,
    quant_scale2=quant_scale2,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.dtype)  # torch.int8
```

> **关键参数说明**：
>
> -四个量化参数必须**同时存在**。`quant_offset2`可选，不传时偏移为0。
>
> - D轴需32字节对齐。input_layout仅支持BSH/BNSD/BSND/BNSD_BSND。
>
> -约束详见[`torch.int8`量化约束](#int8_constraints)。

### 示例7：伪量化（KV分离模式）+ Decode

key/value为`torch.int8`，通过perchannel反量化因子转为`torch.float16`参与计算。使用推荐的KV分离模式。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 1, 128
KV_N, KV_S = 2, 2048

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.bfloat16).npu()
k = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()

# KV分离perchannel量化因子: shape (1, KV_N, 1, D)或(1, KV_N, D)或(1, H)
key_antiquant_scale = torch.ones(1, KV_N, 1, D, dtype=torch.bfloat16).npu() * 0.01
value_antiquant_scale = torch.ones(1, KV_N, 1, D, dtype=torch.bfloat16).npu() * 0.01
#对称量化：不传offset

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    key_antiquant_scale=key_antiquant_scale,
    value_antiquant_scale=value_antiquant_scale,
    key_antiquant_mode=0,     # perchannel
    value_antiquant_mode=0,
    num_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="BNSD",
    scale=scale
)

print(out.dtype)   # torch.bfloat16
print(out.shape)   # torch.Size([1, 8, 1, 128])
```

> **关键参数说明**：
>
> -使用KV分离模式（优于`antiquant_scale`和`antiquant_offset`模式）。
>
> - `key_antiquant_mode=0`：perchannel模式。此处Q_S=1 + query `torch.bfloat16` + kv `torch.int8` + key/value_antiquant_scale `torch.bfloat16` → 满足[Q_S = 1约束](#qs_eq_1)中的kv 伪量化参数分离条件。
>
> -当`key_antiquant_scale`和`value_antiquant_scale`与`antiquant_scale`同时传入时，以KV分离参数为准。

### 示例8：Prefix + Prefill（共享前缀）

系统提示词/共享前缀场景：prefix部分作为公共KV，与每次请求的KV拼接计算。

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024
prefix_S = 32

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

#共享前缀: batch=1, S=prefix_S, N/D与k/v一致
key_prefix = torch.randn(1, KV_N, prefix_S, D, dtype=torch.float16).npu()
value_prefix = torch.randn(1, KV_N, prefix_S, D, dtype=torch.float16).npu()

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    key_shared_prefix=key_prefix,
    value_shared_prefix=value_prefix,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> **关键参数说明**：
>
> - `key_shared_prefix` + `value_shared_prefix`：必须同时传入。shape第一维batch=1。约束见[Prefix约束](#prefix_constraints)。
>
> -如果S2（KV_S + prefix_S）超出原KV_S限制则报错。若传入`atten_mask`，S2需大于prefix_S + KV_S。
>
> -不支持PageAttention、左padding、tensorlist、qkv全`torch.int8`。

### 示例9：ACL Graph模式（torch_npu.npu.NPUGraph）

通过NPUGraph捕获计算图，实现图模式执行和参数更新。

```python
import torch
import torch_npu
import math

# ===== 1.数据准备=====
B, Q_N, Q_S, D = 1, 8, 164, 128
KV_S = 1024
scale = 1.0 / math.sqrt(D)

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, Q_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, Q_N, KV_S, D, dtype=torch.float16).npu()

# ===== 2.定义模型=====
class Model(torch.nn.Module):
    def __init__(self, num_heads, scale, pre_tokens=65535, next_tokens=65535):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens

    def forward(self, q, k, v):
        return torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=self.num_heads,
            input_layout="BNSD",
            scale=self.scale,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens
        )[0]  # 只取第一个返回值

# ===== 3.图捕获=====
model = Model(Q_N, scale).npu()
graph = torch_npu.npu.NPUGraph()

with torch_npu.npu.graph(graph):
    graph_out = model(q, k, v)

print(f"图捕获完成，输出 shape: {graph_out.shape}")

# ===== 4.图重放（输入数据更新） =====
#生成新输入数据（内容变化，shape不变）
q_new = torch.randn_like(q)
k_new = torch.randn_like(k)
v_new = torch.randn_like(v)

#重要：通过copy_更新数据到捕获时的内存地址
q.copy_(q_new)
k.copy_(k_new)
v.copy_(v_new)

#重放图
graph.replay()
print(f"图重放完成（输入数据更新），输出 shape: {graph_out.shape}")

# ===== 5.参数更新（模型配置变化） =====
#注意：pre_tokens/next_tokens变化需要重新捕获图
new_pre_tokens = 32768
model_updated = Model(Q_N, scale, new_pre_tokens, 32768).npu()

graph_updated = torch_npu.npu.NPUGraph()
with torch_npu.npu.graph(graph_updated):
    updated_out = model_updated(q, k, v)

graph_updated.replay()
print(f"图重放完成（pre_tokens={new_pre_tokens}），输出 shape: {updated_out.shape}")

# ===== 6.结果验证=====
with torch.no_grad():
    single_out = model(q, k, v)

match = torch.allclose(graph_out, single_out, atol=1e-2)
print(f"\nGraph vs Eager 匹配: {'✓' if match else '✗'}")
print(f"  graph output shape: {graph_out.shape}")
print(f"  single op output shape: {single_out.shape}")
```

## 参考资源

- **BMM1（Batch Matrix Multiply 1）**：第一次矩阵乘法，计算attention score：$attention\_score = query \cdot key^{\top}$。
- **BMM2（Batch Matrix Multiply 2）**：第二次矩阵乘法，将attention权重应用到value：$attention\_out = atten\_prob \cdot value$，其中$atten\_prob = softmax(\cdot)$。
- **Ring Attention**：分布式注意力算法，将长序列按block切分到多设备，通过环形通信逐块计算注意力，降低单卡显存占用。单卡退化为标准注意力。多设备合并依赖`softmax_lse_flag=True`输出的LSE值。详见[返回值说明](#返回值说明)。
