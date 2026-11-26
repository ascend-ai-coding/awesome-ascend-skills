# torch_npu.npu_prompt_flash_attention

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id19 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id19 -->

## 功能说明

- API功能：全量FA实现。

- 计算公式：

$$
atten\_out = softmax\left(scale \cdot (Q \cdot K) + atten\_mask\right) \cdot V
$$

## 函数原型

```python
torch_npu.npu_prompt_flash_attention(query, key, value, *, pse_shift=None, padding_mask=None, atten_mask=None, actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147483647, next_tokens=0, input_layout="BSH",num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0) -> Tensor
```

## 参数说明

- **query** (`Tensor`)：必选参数，对应公式中的输入$Q$，数据类型与`key`的数据类型需满足数据类型推导规则，即保持与`key`、`value`的数据类型一致。不支持非连续的`Tensor`，数据格式支持$ND$。
    <!-- npu="310p" id20 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`。
    <!-- end id20 -->

    <!-- npu="910b" id3 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id3 -->
    <!-- npu="A3" id4 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id4 -->

- **key** (`Tensor`)：必选参数，对应公式中的输入$K$，数据类型与`query`的数据类型需满足数据类型推导规则，即保持与`query`、`value`的数据类型一致。不支持非连续的`Tensor`，数据格式支持$ND$。
    <!-- npu="310p" id21 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`。
    <!-- end id21 -->

    <!-- npu="910b" id5 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id5 -->
    <!-- npu="A3" id6 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id6 -->

- **value** (`Tensor`)：必选参数，对应公式中的输入$V$，数据类型与`query`的数据类型需满足数据类型推导规则，即保持与`query`、`key`的数据类型一致。不支持非连续的`Tensor`，数据格式支持$ND$。
    <!-- npu="310p" id22 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`。
    <!-- end id22 -->

    <!-- npu="910b" id7 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id7 -->
    <!-- npu="A3" id8 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.int8`。
    <!-- end id8 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **pse_shift** (`Tensor`)：可选参数。不支持非连续的`Tensor`，数据格式支持$ND$。输入shape类型需为$(B, N, Q\_S, KV\_S)$或$(1, N, Q\_S, KV\_S)$，其中$Q\_S$为`query`的shape中的$S$，$KV\_S$为`key`和`value`的shape中的$S$。对于`pse_shift`的$KV\_S$为非32字节对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。如不使用该功能时可传入`None`。综合约束请见[约束说明](#section12345537164214)。
    <!-- npu="310p" id23 -->
    - <term>Atlas推理系列产品</term>：暂不支持该参数。
    <!-- end id23 -->

    <!-- npu="910b" id9 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。当`pse_shift`为`torch.float16`时，要求`query`为`torch.float16`或`torch.int8`；当`pse_shift`为`torch.bfloat16`时，要求`query`为`torch.bfloat16`。在`query`、`key`、`value`为`torch.float16`且`pse_shift`存在的情况下，默认走高精度模式。
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。当`pse_shift`为`torch.float16`时，要求`query`为`torch.float16`或`torch.int8`；当`pse_shift`为`torch.bfloat16`时，要求`query`为`torch.bfloat16`。在`query`、`key`、`value`为`torch.float16`且`pse_shift`存在的情况下，默认走高精度模式。
    <!-- end id10 -->

- **padding_mask**：预留参数，暂未使用，默认值为`None`。
- **atten_mask** (`Tensor`)：可选参数，对应公式中$atten\_mask$，代表下三角全为0、上三角全为负无穷的倒三角mask矩阵，数据类型支持`torch.bool`、`torch.int8`和`torch.uint8`。数据格式支持$ND$，不支持非连续的`Tensor`。如果不使用该功能可传入`None`。通常建议shape输入$(Q\_S, KV\_S)$、$(B, Q\_S, KV\_S)$、$(1, Q\_S, KV\_S)$、$(B, 1, Q\_S, KV\_S)$、$(1, 1, Q\_S, KV\_S)$，其中$Q\_S$为`query`的shape中的$S$，$KV\_S$为`key`和`value`的shape中的$S$，对于`atten_mask`的$KV\_S$为非32字节对齐的场景，建议padding到32字节对齐来提高性能，多余部分填充成1。综合约束请见[约束说明](#section12345537164214)。
- **actual_seq_lengths** (`List[int]`)：可选参数，代表不同Batch中`query`的有效Sequence Length，数据类型支持`torch.int64`。如果不指定`seqlen`可以传入`None`，表示和`query`的shape的s长度相同。限制：该入参中每个batch的有效Sequence Length应该不大于`query`中对应batch的Sequence Length。`seqlen`的传入长度为1时，每个Batch使用相同`seqlen`；传入长度大于等于Batch数时取`seqlen`的前Batch个数。其它长度不支持。
    <!-- npu="310p" id24 -->
    - <term>Atlas推理系列产品</term>：暂不支持该参数。
    <!-- end id24 -->

    <!-- npu="910b" id11 -->
    - <term>Atlas A2系列产品</term>：支持TND格式。当`query`的`input_layout`为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
    <!-- end id11 -->
    <!-- npu="A3" id12 -->
    - <term>Atlas A3系列产品</term>：支持TND格式。当`query`的`input_layout`为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
    <!-- end id12 -->

- **deq_scale1** (`Tensor`)：可选参数，表示BMM1后面的反量化因子，支持pertensor。数据类型支持`torch.uint64`、`torch.float32`，数据格式支持$ND$。如不使用该功能时可传入`None`。
    <!-- npu="310p" id25 -->
    - <term>Atlas推理系列产品</term>暂不支持该参数。
    <!-- end id25 -->
- **quant_scale1** (`Tensor`)：可选参数，数据类型支持`torch.float32`。数据格式支持$ND$，表示BMM2前面的量化因子，支持pertensor。如不使用该功能时可传入`None`。
    <!-- npu="310p" id26 -->
    - <term>Atlas推理系列产品</term>暂不支持该参数。
    <!-- end id26 -->
- **deq_scale2** (`Tensor`)：可选参数，数据类型支持`torch.uint64`、`torch.float32`。数据格式支持$ND$，表示BMM2后面的反量化因子，支持pertensor。如不使用该功能时可传入`None`。
    - <term>Atlas推理系列产品</term>暂不支持该参数。
- **quant_scale2** (`Tensor`)：可选参数，数据格式支持$ND$，表示输出的量化因子，支持pertensor、perchannel。如不使用该功能时可传入`None`。
    <!-- npu="310p" id27 -->
    - <term>Atlas推理系列产品</term>：暂不支持该参数。
    <!-- end id27 -->

    <!-- npu="910b" id13 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。当输入为`torch.bfloat16`时，同时支持`torch.float32`和`torch.bfloat16`，否则仅支持`torch.float32`。perchannel格式，当输出layout为$BSH$时，要求`quant_scale2`所有维度的乘积等于$H$；其他layout要求乘积等于$N*D$。当输出layout为$BSH$，`quant_scale2` shape建议传入$(1, 1, H)$或$(H,)$；输出为$BNSD$时，建议传入$(1, N, 1, D)$或$(N, D)$；输出为$BSND$时，建议传入$(1, 1, N, D)$或$(N, D)$。
    <!-- end id13 -->
    <!-- npu="A3" id14 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。当输入为`torch.bfloat16`时，同时支持`torch.float32`和`torch.bfloat16`，否则仅支持`torch.float32`。perchannel格式，当输出layout为$BSH$时，要求`quant_scale2`所有维度的乘积等于$H$；其他layout要求乘积等于$N*D$。当输出layout为$BSH$，`quant_scale2` shape建议传入$(1, 1, H)$或$(H,)$；输出为$BNSD$时，建议传入$(1, N, 1, D)$或$(N, D)$；输出为$BSND$时，建议传入$(1, 1, N, D)$或$(N, D)$。
    <!-- end id14 -->

- **quant_offset2** (`Tensor`)：可选参数，数据格式支持$ND$，表示输出的量化偏移，支持pertensor、perchannel。若传入 `quant_offset2`，需保证其类型和`shape`信息与`quant_scale2`一致。如不使用该功能时可传入`None`。
    <!-- npu="310p" id28 -->
    - <term>Atlas推理系列产品</term>：暂不支持该参数。
    <!-- end id28 -->

    <!-- npu="910b" id15 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。
    <!-- end id15 -->
    <!-- npu="A3" id16 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。
    <!-- end id16 -->

- **num_heads** (`int`)：可选参数，代表`query`的head个数，数据类型支持`torch.int64`。
- **scale_value** (`float`)：可选参数，对应公式中$scale$，值通常是$d$开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持`float`。数据类型与`query`的数据类型需满足数据类型推导规则。默认值为`1.0`。
- **pre_tokens** (`int`)：可选参数，用于稀疏计算，表示Attention（注意力机制）需要和前几个Token计算关联，数据类型支持`torch.int64`。默认值为2147483647。
    <!-- npu="310p" id29 -->
    - <term>Atlas推理系列产品</term>仅支持默认值2147483647。
    <!-- end id29 -->
- **next_tokens** (`int`)：可选参数，用于稀疏计算，表示Attention需要和后几个Token计算关联。数据类型支持`torch.int64`。默认值为`0`。
    <!-- npu="310p" id30 -->
    - <term>Atlas推理系列产品</term>仅支持0和2147483647。
    <!-- end id30 -->
- **input_layout** (`str`)：可选参数，用于标识输入`query`、`key`、`value`的数据排布格式，当前支持$BSH$、$BSND$、$BNSD$、$BNSD\_BSND$（输入为$BNSD$时，输出格式为$BSND$）。默认值为`"BSH"`。
- **num_key_value_heads**：可选参数，代表`key`、`value`中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景，GQA是介于MHA（Multi-Head Attention，多头注意力）和MQA（Multi-Query Attention，多查询注意力）之间的折中方案：多个query head共享同一组key/value head，从而在保持模型效果接近MHA的同时，显著降低KV Cache显存占用和访存开销，数据类型支持`torch.int64`。默认值为`0`，表示`key`/`value`和`query`的head个数相等。限制：需要满足`num_key_value_heads`整除`num_heads`，`num_heads`与`num_key_value_heads`的比值不能大于64，且在$BSND$、$BNSD$、$BNSD\_BSND$场景下，需要与shape中的`key`/`value`的$N$轴shape值相同，否则报错。
    <!-- npu="310p" id32 -->
    - <term>Atlas推理系列产品</term>仅支持默认值`0`。
    <!-- end id32 -->
- **actual_seq_lengths_kv** (`List[int]`)：可选参数，代表不同batch中`key`/`value`的有效seqlenKV。数据类型支持`torch.int64`。限制：该入参中每个batch的有效seqlenKV应该不大于`key`/`value`中对应batch的seqlenKV。seqlenKV的传入长度为1时，每个Batch使用相同seqlenKV；传入长度大于等于Batch数时取seqlenKV的前Batch个数，其它长度不支持。
    <!-- npu="310p" id31 -->
    - <term>Atlas推理系列产品</term>：暂不支持该参数。
    <!-- end id31 -->

    <!-- npu="910b" id17 -->
    - <term>Atlas A2系列产品</term>：支持TND格式。当key/value的input_layout为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlenKV和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
    <!-- end id17 -->
    <!-- npu="A3" id18 -->
    - <term>Atlas A3系列产品</term>：支持TND格式。当key/value的input_layout为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlenKV和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
    <!-- end id18 -->

- **sparse_mode** (`int`)：可选参数，表示sparse的模式，数据类型支持`torch.int64`。默认值为`0`，综合约束请见[约束说明](#section12345537164214)。

    <!-- npu="310p" id33 -->
    - <term>Atlas推理系列产品</term>仅支持默认值`0`。
    <!-- end id33 -->
    - `sparse_mode`为0时，代表`defaultMask`模式，如果`atten_mask`未传入则不做mask操作，忽略`pre_tokens`和`next_tokens`（内部赋值为`INT_MAX`）；如果传入，则需要传入完整的`atten_mask`矩阵$(S1 * S2)$，表示`pre_tokens`和`next_tokens`之间的部分需要计算。不支持传入的mask矩阵中参与计算的部分整行为1的情况。
    - `sparse_mode`为1时，代表`allMask`。不支持传入的mask矩阵中参与计算的部分整行为1的情况。
    - `sparse_mode`为2时，代表`leftUpCausal`模式的mask，需要传入优化后的`atten_mask`矩阵（2048*2048）。
    - `sparse_mode`为3时，代表`rightDownCausal`模式的mask，均对应以左顶点为划分的下三角场景，需要传入优化后的`atten_mask`矩阵（2048*2048）。
    - `sparse_mode`为4时，代表`band`模式的mask，需要传入优化后的`atten_mask`矩阵（2048*2048）。
    - `sparse_mode`为5、6、7、8时，分别代表`prefix、global、dilated、block_local`，均暂不支持。

## 返回值说明

`Tensor`

公式中的$atten\_out$，表示计算的最终结果。当`input_layout`为$BNSD\_BSND$时，输入`query`的shape是$BNSD$，输出shape为$BSND$，其余情况shape与`query`的shape保持一致。

## 约束说明<a name="section12345537164214"></a>

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 入参为空的处理：算子内部需要判断参数`query`是否为空，如果是空则直接返回。参数`query`不为空Tensor，参数`key`、`value`为空Tensor(即$S2$为0)，则填充全零的对应shape的输出（填充`atten_out`）。`atten_out`为空Tensor时，框架会处理。
- `query`、`key`、`value`输入，功能使用限制如下：

    <a name="zh-cn_topic_0000001798619409_table382212695610"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001798619409_row1882282685619"><th class="cellrowborder" valign="top" width="30.570000000000004%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001798619409_p182262616569"><a name="zh-cn_topic_0000001798619409_p182262616569"></a><a name="zh-cn_topic_0000001798619409_p182262616569"></a>产品型号</p>
    </th>
    <th class="cellrowborder" valign="top" width="69.43%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001798619409_p1682262645612"><a name="zh-cn_topic_0000001798619409_p1682262645612"></a><a name="zh-cn_topic_0000001798619409_p1682262645612"></a>轴约束</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001798619409_row88221926155613"><td class="cellrowborder" valign="top" width="30.570000000000004%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001798619409_p282310268569"><a name="zh-cn_topic_0000001798619409_p282310268569"></a><a name="zh-cn_topic_0000001798619409_p282310268569"></a><span id="zh-cn_topic_0000001798619409_ph999933035615"><a name="zh-cn_topic_0000001798619409_ph999933035615"></a><a name="zh-cn_topic_0000001798619409_ph999933035615"></a><a name="zh-cn_topic_0000001798619409_zh-cn_topic_0000001312391781_term15651172142210_16"></a><a name="zh-cn_topic_0000001798619409_zh-cn_topic_0000001312391781_term15651172142210_16"></a><term>Atlas推理系列产品</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.43%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000001798619409_ul849244711561"></a><a name="zh-cn_topic_0000001798619409_ul849244711561"></a><ul id="zh-cn_topic_0000001798619409_ul849244711561"><li>支持B轴小于等于128。</li><li>支持N轴小于等于256。</li><li>支持S轴小于等于65535(64k)。</li><li>支持D轴小于等于512。</li></ul>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001798619409_row1782312260569"><td class="cellrowborder" valign="top" width="30.570000000000004%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001798619409_p128231426175620"><a name="zh-cn_topic_0000001798619409_p128231426175620"></a><a name="zh-cn_topic_0000001798619409_p128231426175620"></a><span id="zh-cn_topic_0000001798619409_ph12807852115620"><a name="zh-cn_topic_0000001798619409_ph12807852115620"></a><a name="zh-cn_topic_0000001798619409_ph12807852115620"></a><a name="zh-cn_topic_0000001798619409_zh-cn_topic_0000001312391781_term11962195213215_7"></a><a name="zh-cn_topic_0000001798619409_zh-cn_topic_0000001312391781_term11962195213215_7"></a><term>Atlas A2系列产品</term></span></p>
    <p id="zh-cn_topic_0000001798619409_p16629103195810"><a name="zh-cn_topic_0000001798619409_p16629103195810"></a><a name="zh-cn_topic_0000001798619409_p16629103195810"></a><span id="zh-cn_topic_0000001798619409_ph1464135665612"><a name="zh-cn_topic_0000001798619409_ph1464135665612"></a><a name="zh-cn_topic_0000001798619409_ph1464135665612"></a><a name="zh-cn_topic_0000001798619409_zh-cn_topic_0000001312391781_term1253731311225_7"></a><a name="zh-cn_topic_0000001798619409_zh-cn_topic_0000001312391781_term1253731311225_7"></a><term>Atlas A3系列产品</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.43%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000001798619409_ul103175710575"></a><a name="zh-cn_topic_0000001798619409_ul103175710575"></a><ul id="zh-cn_topic_0000001798619409_ul103175710575"><li>支持B轴小于等于65536(64k)，D轴32byte不对齐时仅支持到128。</li><li>支持N轴小于等于256。</li><li>支持S轴小于等于20971520（20M）。长序列场景下，如果计算量过大可能会导致PFA算子执行超时（AI Core error类型报错，errorStr为timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：<a name="zh-cn_topic_0000001798619409_ul103175718575"></a><a name="zh-cn_topic_0000001798619409_ul103175718575"></a><ul id="zh-cn_topic_0000001798619409_ul103175718575"><li>B=1，Q<sub>N</sub>=20，Q<sub>S</sub>=1048576，D = 256，KV<sub>N</sub>=1，KV<sub>S</sub>=1048576。</li><li>B=1，Q<sub>N</sub>=2，Q<sub>S</sub>=10485760，D = 256，KV<sub>N</sub>=2，KV<sub>S</sub>=10485760。</li><li>B=20，Q<sub>N</sub>=1，Q<sub>S</sub>=1048576，D = 256，KV<sub>N</sub>=1，KV<sub>S</sub>=1048576。</li><li>B=1，Q<sub>N</sub>=10，Q<sub>S</sub>=1048576，D = 512，KV<sub>N</sub>=1，KV<sub>S</sub>=1048576。</li></ul>
    </li><li>支持D轴小于等于512。`input_layout`为BSH或者BSND时，要求N*D小于65535。</li></ul>
    </td>
    </tr>
    </tbody>
    </table>

- 参数`sparse_mode`当前仅支持值为`0、1、2、3、4`的场景，取其它值时会报错。
    - `sparse_mode`为0时，`atten_mask`如果为空指针，则忽略入参`pre_tokens`、`next_tokens`（内部赋值为`INT_MAX`）。
    - `sparse_mode`为2、3、4时，`atten_mask`的shape需要为$(S, S)$或$(1, S, S)$或$(1, 1, S, S)$，其中$S$的值需要固定为2048，且需要用户保证传入的`atten_mask`为下三角，不传入`atten_mask`或者传入的shape不正确报错。
    - `sparse_mode`为1、2、3的场景忽略入参`pre_tokens`、`next_tokens`并按照相关规则赋值。

- `torch.int8`量化相关入参数量与输入、输出数据格式的综合限制：
    - 输入为`torch.int8`，输出为`torch.int8`的场景：入参`deq_scale1`、`quant_scale1`、`deq_scale2`、`quant_scale2`需要同时存在，`quant_offset2`可选，不传时默认为`0`。
    - 输入为`torch.int8`，输出为`torch.float16`的场景：入参`deq_scale1`、`quant_scale1`、`deq_scale2`需要同时存在，若存在入参`quant_offset2`或`quant_scale2`（即不为`None`），则报错并返回。
    - 输入为`torch.float16`或`torch.bfloat16`，输出为`torch.int8`的场景：入参`quant_scale2`需存在，`quant_offset2`可选，不传时默认为`0`，若存在入参`deq_scale1`或`quant_scale1`或`deq_scale2`（即不为`None`），则报错并返回。
    - 入参 `quant_offset2`和`quant_scale2`支持pertensor/perchannel两种格式和`torch.float32`/`torch.bfloat16`两种数据类型。若传入`quant_offset2`，需保证其类型和shape信息与`quant_scale2`一致。当输入为`torch.bfloat16`时，同时支持`torch.float32`和`torch.bfloat16`，否则仅支持`torch.float32`。perchannel格式，当输出layout为$BSH$时，要求`quant_scale2`所有维度的乘积等于$H$；其他layout要求乘积等于$N*D$。当输出layout为$BSH$，`quant_scale2` shape传入$(1, 1, H)$或$(H,)$；输出为$BNSD$时，建议传入$(1, N, 1, D)$或$(N, D)$；输出为$BSND$时，建议传入$(1, 1, N, D)$或$(N, D)$。pertensor格式，建议$D$轴对齐到32Byte。
    - perchannel格式，入参`quant_scale2`和`quant_offset2`暂不支持左padding、Ring Attention（一种面向超长序列的分布式注意力计算方案，将KV序列切分到多张卡上，通过卡间环形传递KV分块来完成注意力计算）或者$D$非32Byte对齐的场景。
    - 输出为`torch.int8`时，暂不支持sparse为`band`且`pre_tokens`/`next_tokens`为负数。

- `pse_shift`功能使用限制如下：
    - 支持`query`数据类型为`torch.float16`或`torch.bfloat16`或`torch.int8`场景下使用该功能。
    - `query`，`key`，`value`数据类型为`torch.float16`且`pse_shift`存在时，强制走高精度模式，对应的限制继承自高精度模式的限制。
    - $Q\_S$需大于等于`query`的$S$长度，$KV\_S$需大于等于`key`的$S$长度。

- 输出为`torch.int8`，入参`quant_offset2`传入非空指针和非空`Tensor`值，并且`sparse_mode`、`pre_tokens`和`next_tokens`满足以下条件，矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截：
    - `sparse_mode=0`，`atten_mask`如果非空指针，每个batch `actual_seq_lengths-actual_seq_lengths_kv-pre_tokens>0`或`next_tokens<0`时，满足拦截条件。
    - `sparse_mode=1`或`2`，不会出现满足拦截条件的情况。
    - `sparse_mode=3`，每个batch `actual_seq_lengths_kv-actual_seq_lengths<0`，满足拦截条件。
    - `sparse_mode=4`，`pre_tokens<0`或每个batch `next_tokens+actual_seq_lengths_kv-actual_seq_lengths<0`时，满足拦截条件。

- kv伪量化参数分离当前暂不支持。
- 暂不支持维度不对齐场景。

## 调用示例

- 单算子调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>> import math
    >>>
    >>> # 生成随机数据，并发送到npu
    >>> q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    >>> k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    >>> v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    >>> scale = 1/math.sqrt(128.0)
    >>> actseqlen = [164]
    >>> actseqlenkv = [1024]
    >>> atten_mask = torch.triu(torch.ones((164, 1024), dtype=torch.bool), diagonal=1).npu()
    >>>
    >>> # 调用PFA算子
    >>> out = torch_npu.npu_prompt_flash_attention(q, k, v,
    ... atten_mask = atten_mask, actual_seq_lengths = actseqlen, actual_seq_lengths_kv = actseqlenkv,
    ... num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)
    >>> print(out.shape)
    torch.Size([1, 8, 164, 128])
    >>> print(out.dtype)
    torch.float16
    ```

- 图模式调用

    ```python
    # 入图方式
    import torch
    import torch_npu
    import math

    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"

    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # 数据生成
    q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    scale = 1/math.sqrt(128.0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self):
            return torch_npu.npu_prompt_flash_attention(q, k, v, num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)

    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()

        single_op = torch_npu.npu_prompt_flash_attention(q, k, v, num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)
        print("single op output with mask:", single_op, single_op.shape)
        print("graph output with mask:", graph_output, graph_output.shape)

    if __name__ == "__main__":
        MetaInfershape()

    # 执行上述代码的输出类似如下
    single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])

    graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
    ```
