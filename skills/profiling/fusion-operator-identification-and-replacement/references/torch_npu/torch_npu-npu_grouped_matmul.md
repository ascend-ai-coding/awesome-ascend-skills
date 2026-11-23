# torch\_npu.npu\_grouped\_matmul

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
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

- **API功能**：npu\_grouped\_matmul是一种对多个矩阵乘法（matmul）操作进行分组计算的高效方法。该API实现了对多个矩阵乘法操作的批量处理，通过将具有相同形状或相似形状的矩阵乘法操作组合在一起，减少内存访问开销和计算资源的浪费，从而提高计算效率。

    > **说明：**
    > 本算子当前支持非量化、全量化、伪量化场景，其中全量化和非量化场景适用于训练或推理场景，伪量化场景仅适用于推理场景。

- **计算公式**：

    注意公式中@符号表示矩阵乘法，×符号表示矩阵Hadamard乘积。

  - 非量化场景（公式1）：

    $$
    y_i = x_i \text{@} \mathit{weight}_i + \mathit{bias}_i
    $$

  - 静态量化（T-T、T-C）场景（公式2）：

    $$
    y_i = (x_i \text{@} \mathit{weight}_i) \times \mathit{scale}_i + \mathit{offset}_i
    $$

    - x为`torch.int8`输入，bias为`torch.int32`输入（公式2-1）：

        $$
        y_i = (x_i \text{@} \mathit{weight}_i + \mathit{bias}_i) \times \mathit{scale}_i + \mathit{offset}_i
        $$

    - x为`torch.int8`输入，bias为`torch.bfloat16`、`torch.float16`、`torch.float32`输入，无offset（公式2-2）：

        $$
        y_i = (x_i \text{@} \mathit{weight}_i) \times \mathit{scale}_i + \mathit{bias}_i
        $$

  - 动态量化（K-T、K-C、T-T、T-C）场景（公式3）：

    $$
    y_i = (x_i \text{@} \mathit{weight}_i + \mathit{bias}_i) \times \mathit{scale}_i \times \mathit{pertokenscale}_i
    $$

    - x为`torch.int8`输入，bias为`torch.int32`输入（公式3-1）：

        $$
        y_i = (x_i \text{@} \mathit{weight}_i + \mathit{bias}_i) \times \mathit{scale}_i \times \mathit{pertokenscale}_i
        $$

    - x为`torch.int8`输入，bias为`torch.bfloat16`、`torch.float16`、`torch.float32`输入（公式3-2）：

        $$
        y_i = (x_i \text{@} \mathit{weight}_i) \times \mathit{scale}_i \times \mathit{pertokenscale}_i + \mathit{bias}_i
        $$

  - 伪量化（perchannel、pergroup）场景（公式4）：

    $$
    y_i = x_i \text{@} \left((\mathit{weight}_i + \mathit{antiquant\_offset}_i) \times \mathit{antiquant\_scale}_i\right) + \mathit{bias}_i
    $$

  <!-- npu="950" id5 -->
  - 动态量化（G-B、mx）场景（公式5）：仅适用于<term>Ascend 950PR&950DT系列产品</term>

    $$
    \mathit{y}_i[m,n] = \sum_{j=0}^{\mathit{k}_{\mathit{loops}}-1}\left(\left(\sum_{k=0}^{\mathit{gsk}-1}\left(\mathit{x}_{\mathit{slice},i} \times \mathit{weight}_{\mathit{slice},i}\right)\right)\times\left(\mathit{pertokenscale}_i\left[\frac{m}{\mathit{gsm}},j\right] \times \mathit{scale}_i\left[j,\frac{n}{\mathit{gsn}}\right]\right)\right) + \mathit{bias}_i[n]
    $$

    其中gsm、gsn和gsk分别代表M、N、K轴的量化的block size，K<sub>i</sub>为每个分组的K的大小，x\_slice<sub>i</sub>代表x<sub>i</sub>第m行长度为gsk的向量，weight\_slice<sub>i</sub>代表weight<sub>i</sub>第n列长度为gsk的向量，K轴均从j\*gsk起始切片，j的取值范围\[0, k\_loops\)，k\_loops=ceil\(K<sub>i</sub>/gsk\)，支持最后的切片长度不足gsk。
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - 伪量化（mx）场景（公式6）：仅适用于<term>Ascend 950PR&950DT系列产品</term>
    - x为`torch.float16`、`torch.bfloat16`输入，weight为`torch.float32`（`torch_npu.float4_e2m1fn_x2`）输入（公式6-1）：

        $$
        y_i = x_i \text{@} (\mathit{weight}_i \times \mathit{antiquant\_scale}_i) + \mathit{bias}_i
        $$

    - x为`torch.float8_e4m3fn`输入，weight为`torch.float32`（`torch_npu.float4_e2m1fn_x2`）输入（公式6-2）：

        $$
        y_i = (x_i \times \mathit{pertokenscale}_i) \text{@} (\mathit{weight}_i \times \mathit{antiquant\_scale}_i) + \mathit{bias}_i
        $$
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - 伪量化（K-CG）场景（公式7）：仅适用于<term>Ascend 950PR&950DT系列产品</term>

    $$
    y_i = \left(x_i \text{@} (\mathit{weight}_i \times \mathit{antiquant\_scale}_i)\right) \times \mathit{scale}_i \times \mathit{pertokenscale}_i + \mathit{bias}_i
    $$

    其中antiquant\_scale<sub>i</sub>为weight矩阵pergroup量化参数，scale<sub>i</sub>为weight矩阵perchannel量化参数，pertokenscale<sub>i</sub>为pertoken量化参数。
  <!-- end id7 -->

## 函数原型

```python
torch_npu.npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None, per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None, activation_quant_offset=None, split_item=0, group_type=-1, group_list_type=0, act_type=0, output_dtype=None, tuning_config=None, x_dtype=None, weight_dtype=None, scale_dtype=None, per_token_scale_dtype=None)
```

## 参数说明

- **`x`**（`List[Tensor]`）：**必选参数**，输入矩阵列表，表示矩阵乘法中的左矩阵。
  - 数据格式支持$ND$，支持的数据类型如下：

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch.int8`。
    <!-- end id8 -->
    <!-- npu="310p" id9 -->
    - <term>Atlas推理系列产品</term>：`torch.float16`。
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT系列产品</term>：`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch.float16`、`torch.bfloat16`、`torch.float32`、`torch.int8`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`、`torch.int8`、`torch_npu.int4`，其中`torch_npu.int4`实际上是用`torch.int8`或者`torch.int32`承载，保持[-8,7]值域，而`torch_npu.hifloat8`/float4系列需配置可选参数`x_dtype`为对应类型，此时`x`本身的`dtype`不再生效，但仍需保证`x`本身的`dtype`为8bit位的数据类型，以保证shape正确；其中float4内轴`K`需为偶数，以保证8bits可以转换为2个float4；另外，`torch_npu.float4_e1m2fn_x2`数据类型仅在`weight`为NZ时支持。
    <!-- end id10 -->

  - 列表最大长度为128。

    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT系列产品</term>：非量化场景支持列表最大长度为1024。
    <!-- end id11 -->

  - 当`split_item`=0，1时，部分场景张量支持2至6维输入，其他情况下，张量仅支持2维输入。

- **`weight`**（`List[Tensor]`）：**必选参数**，权重矩阵列表，表示矩阵乘法中的右矩阵。
  - 数据格式支持$ND$/`FRACTAL_NZ`，支持的数据类型如下：

    <!-- npu="A3,910b" id12 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
      - 当`group_list`输入类型为`List[int]`时，支持`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch.int8`。
      - 当`group_list`输入类型为`Tensor`时，支持`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch_npu.int4`、`torch.int8`。
    <!-- end id12 -->
    <!-- npu="310p" id13 -->
    - <term>Atlas推理系列产品</term>：`torch.float16`。
    <!-- end id13 -->
    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch.int8`、`torch_npu.hifloat8`、`torch.bfloat16`、`torch.float16`、`torch.float32`、`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`、`torch.int32`、`torch_npu.int4`，其中`torch_npu.int4`实际上是用`torch.int8`或者`torch.int32`承载，保持[-8,7]值域，而`torch_npu.hifloat8`/float4系列需配置可选参数`weight_dtype`为对应类型，此时`weight`本身的`dtype`不再生效，但仍需保证`weight`本身的`dtype`为8bit位的数据类型，以保证shape正确；另外，`torch_npu.float4_e1m2fn_x2`数据类型仅在`weight`为NZ时支持。
      - 全量化场景下，当`x`为float4系列、`weight`为float4系列输入时，仅支持推理场景，此时输入`x`的`K`需为偶数，且当`weight`不转置时内轴`N`需为偶数，以保证8bits可以转换为2个float4。当输入`x`/`weight`数据类型为`torch.int8`或`torch.float8_e4m3fn`（`scale_dtype`与`per_token_scale_dtype`为`torch_npu.float8_e8m0fnu`）或`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`（`scale_dtype`与`per_token_scale_dtype`为`torch_npu.float8_e8m0fnu`），`weight`数据格式支持`FRACTAL_NZ`，可通过`torch_npu.npu_format_cast`接口实现$ND$转`FRACTAL_NZ`格式。当`x`为`torch_npu.int4`、`weight`为`torch_npu.int4`输入时，同样需要输入`x`的`K`为偶数，且当`weight`不转置时内轴`N`需为偶数。`weight`数据格式支持`FRACTAL_NZ`，可通过`torch_npu.npu_format_cast`接口实现$ND$转`FRACTAL_NZ`格式。
      - 伪量化场景下，当`weight`为`torch.float32`/`torch.int32`类型时，鉴于PyTorch原生不支持部分类型数据，可通过[torch\_npu.npu\_convert\_weight\_to\_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)实现**1个`torch.float32`承载8个`torch_npu.float4_e2m1fn_x2`的输入、1个`torch.int32`承载8个`torch_npu.int4`的输入**，此时不传入`weight_dtype`。同理，当`weight`为`torch_npu.int4`时，也需要进行pack操作。
      - 伪量化场景下，MxA8W4数据流支持`weight`使用`torch.uint8`承载`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`（1个`torch.uint8`承载两个`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`），该场景下，静态图/动态图不支持E=1或K=64，其中E表示Group大小。
      - 非量化场景下，当`weight`输入为`torch.float16`或`torch.bfloat16`，且`weight` shape后两维对32B对齐时，数据格式还支持`FRACTAL_NZ`，可通过`torch_npu.npu_format_cast`接口实现$ND$转`FRACTAL_NZ`格式。
    <!-- end id14 -->

  - 列表最大长度为128。

    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT系列产品</term>：非量化场景支持列表最大长度为1024。全量化在mx量化且输入且`w`为`FRACTAL_NZ`格式时，支持多`weight`多tensor输入，支持tensorlist最大长度为128。
    <!-- end id15 -->

  - 每个张量支持2维或3维输入。

- <strong>*</strong>：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **`bias`**（`List[Tensor]`）：**可选参数**，每个分组的矩阵乘法输出的独立偏置项。
  - 数据格式支持$ND$，支持的数据类型如下：

    <!-- npu="A3,910b" id16 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.float16`、`torch.float32`和`torch.int32`。
    <!-- end id16 -->
    <!-- npu="310p" id17 -->
    - <term>Atlas推理系列产品</term>：`torch.float16`。
    <!-- end id17 -->
    <!-- npu="950" id18 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
      - 量化场景下，在输入`x`为`torch.int8`时支持`torch.int32`、`torch.bfloat16`、`torch.float16`、`torch.float32`，在输入`x`为`torch_npu.float4_e2m1fn_x2`时，仅支持`torch.float32`，其它类型输入需传None。
      - 伪量化场景下，在输入`x`为`torch.float16`时，`bias`支持`torch.float16`；在输入`x`为`torch.bfloat16`时，`bias`支持`torch.bfloat16`、`torch.float32`；在输入`x`为`torch.float8_e4m3fn`（MXFP8等场景）时，`bias`支持`torch.float16`、`torch.bfloat16`；其它类型输入需传None。
      - 非量化场景下，在输入`x`为`torch.float16`时支持`torch.float16`、`torch.float32`，在`x`为`torch.bfloat16`时支持`torch.bfloat16`、`torch.float32`，在输入`x`为`torch.float32`时支持`torch.float32`，其它类型输入需传None。
    <!-- end id18 -->

  - 列表长度与`weight`列表长度相同。
  - 每个张量支持1维或2维输入。

- **`scale`**（`List[Tensor]`）：**可选参数**，用于缩放原数值以匹配量化后的范围值，代表量化参数中的缩放因子。
  - 数据格式和数据类型支持如下：

    <!-- npu="A3,910b" id19 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持$ND$格式。
      - 当`group_list`输入类型为`List[int]`时，支持`torch.int64`。
      - 当`group_list`输入类型为`Tensor`时，支持`torch.float32`、`torch.bfloat16`、`torch.int64`。
    <!-- end id19 -->
    <!-- npu="310p" id20 -->
    - <term>Atlas推理系列产品</term>：仅支持传入None。
    <!-- end id20 -->
    <!-- npu="950" id21 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
      - $ND$格式：当`group_list`输入类型为`torch.Tensor`且`x`不为`torch.float16`、`torch.bfloat16`时，支持`torch.int64`、`torch.bfloat16`、`torch.float32`、`torch_npu.float8_e8m0fnu`，其中`torch_npu.float8_e8m0fnu`需配置`scale_dtype`为对应类型，此时`scale`本身的`dtype`不再生效，但仍需保证`scale`本身的`dtype`为8bit位的数据类型，以保证shape正确。
      - `FRACTAL_NZ`格式：仅支持`torch_npu.float8_e8m0fnu`，且需配置`scale_dtype`为对应类型。
    <!-- end id21 -->

  - 列表长度与`weight`列表长度相同。

  <!-- npu="A3,910b" id22 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：每个张量仅支持1维输入。
  <!-- end id22 -->
  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx量化的M轴分组场景时，每个张量的维度需要为4，在`weight`多tensor输入场景下，每个张量维度为3维，并且与`weight`具有相同的转置属性。mx量化的K轴分组场景时，每个张量的维度需要为3，并且与`weight`具有相同转置属性。K-CG伪量化的M轴分组场景，每个张量的维度需要为2，并且与`weight`具有相同转置属性。
  <!-- end id23 -->

- **`offset`**（`List[Tensor]`）：**可选参数**，用于调整量化后的数值偏移量，从而更准确地表示原始浮点数值。数据格式支持$ND$。
  <!-- npu="950" id88 -->
  - <term>Ascend 950PR&950DT系列产品</term>：S8S4 K-C伪量化场景支持`torch.float32`。
  <!-- end id88 -->
- **`antiquant_scale`**（`List[Tensor]`）：**可选参数**，用于缩放原数值以匹配伪量化后的范围值，代表伪量化参数中的缩放因子。
  - 数据格式支持$ND$，支持的数据类型如下：

    <!-- npu="A3,910b" id24 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.float16`、`torch.bfloat16`。
    <!-- end id24 -->
    <!-- npu="310p" id25 -->
    - <term>Atlas推理系列产品</term>：仅支持传入None。
    <!-- end id25 -->
    <!-- npu="950" id26 -->
    - <term>Ascend 950PR&950DT系列产品</term>：`torch.float16`、`torch.bfloat16`、`torch_npu.float8_e8m0fnu`，其中`torch_npu.float8_e8m0fnu`需传入`torch.uint8`表示。
    <!-- end id26 -->

  - 列表长度与`weight`列表长度相同。
  - 每个张量支持输入维度如下（其中g为matmul组数，G为pergroup数，G<sub>i</sub>为第i个tensor的pergroup数）：
    - 伪量化perchannel场景，`weight`为单tensor时，shape限制为\[g, n\]；weight为多tensor时，shape限制为\[n<sub>i</sub>\]。
    - 伪量化pergroup场景，`weight`为单tensor时，shape限制为\[g, G, n\]；weight为多tensor时，shape限制为\[G<sub>i</sub>, n<sub>i</sub>\]。
    - 伪量化mx场景，`weight`为单tensor时，shape限制为\(g, n, G/2, 2\)

- **`antiquant_offset`**（`List[Tensor]`）：**可选参数**，用于调整伪量化后的数值偏移量，从而更准确地表示原始浮点数值。
  - 数据格式支持$ND$，支持的数据类型如下：

    <!-- npu="A3,910b" id27 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.float16`、`torch.bfloat16`。
    <!-- end id27 -->
    <!-- npu="310p" id28 -->
    - <term>Atlas推理系列产品</term>：仅支持传入None。
    <!-- end id28 -->
    <!-- npu="950" id29 -->
    - <term>Ascend 950PR&950DT系列产品</term>：`torch.float16`、`torch.bfloat16`。
    <!-- end id29 -->

  - 列表长度与`weight`列表长度相同。
  - 每个张量输入维度和`antiquant_scale`输入维度一致。

- **`per_token_scale`**（`List[Tensor]`）：**可选参数**，用于缩放原数值以匹配量化后的范围值，代表量化参数中由`x`量化引入的缩放因子，数据格式支持$ND$。
  - `group_list`输入类型为`List[int]`时，当前只支持传入None。
  - `group_list`输入类型为`Tensor`时：

    <!-- npu="A3,910b" id30 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float32`。
    <!-- end id30 -->
    <!-- npu="950" id31 -->
    - <term>Ascend 950PR&950DT系列产品</term>：当`group_list`输入类型为`Tensor`时，数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`，其中`torch_npu.float8_e8m0fnu`需配置`per_token_scale_dtype`为对应类型，此时`per_token_scale`本身的`dtype`不再生效，但仍需保证`per_token_scale`本身的`dtype`为8bit位的数据类型，以保证shape正确。
    <!-- end id31 -->
    - 列表长度与`x`列表长度相同。

    <!-- npu="A3,910b" id32 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：每个张量仅支持1维输入。
    <!-- end id32 -->
    <!-- npu="950" id33 -->
    - <term>Ascend 950PR&950DT系列产品</term>：mx量化时，每个张量需要为3维，且与`x`具有相同的转置属性。mx伪量化场景，张量需要为3维，转置属性与x相同，数据类型支持`torch_npu.float8_e8m0fnu`。K-CG伪量化场景，张量需要为1维，数据类型支持`torch.float32`。
    <!-- end id33 -->

- **`group_list`**（`List[int]`/`torch.Tensor`）：**可选参数**，用于指定分组的索引，表示`x`的第0维矩阵乘法的索引情况。数据格式支持$ND$，数据类型支持`torch.int64`。

  <!-- npu="310p" id34 -->
  - <term>Atlas推理系列产品</term>：仅支持`torch.Tensor`类型。仅支持1维输入，长度与`weight`列表长度相同。
  <!-- end id34 -->
  <!-- npu="A3,910b" id35 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持`List[int]`或`torch.Tensor`类型。若为`torch.Tensor`类型，仅支持1维输入，长度与`weight`列表长度相同。
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - <term>Ascend 950PR&950DT系列产品</term>：仅支持`torch.Tensor`类型。仅支持1维输入，在`group_type`为0时，长度与`weight`\(单tensor，\[\[E, N, K\]\]\)的最高轴E大小相同。在`group_type`为2时，长度与输出\(单tensor，\[\[E, M, N\]\]\)的最高轴E大小相同。
  <!-- end id36 -->
  - 配置值要求如下：
    - `group_list`输入类型为`List[int]`时，配置值必须为非负递增数列，且长度不能为0。
    - `group_list`输入类型为`Tensor`时：
      - 当`group_list_type`为0时，`group_list`必须为非负、单调非递减数列。
      - 当`group_list_type`为1时，`group_list`必须为非负数列，且长度不能为0。
      - 当`group_list_type`为2时，`group_list`的shape为\[E, 2\]，E表示Group大小，数据排布为\[\[groupIdx0, groupSize0\], \[groupIdx1, groupSize1\], ...\]，其中groupSize为分组轴上每组大小，必须为非负数。所有groupSize非0的分组按groupIdx有序排列在前，所有groupSize为0的分组按groupIdx有序排列在后，确保非零组前置、零值组后置，且组内有序。

- **`activation_input`**（`List[Tensor]`）：**可选参数**，代表激活函数的反向输入，当前仅支持传入None。
- **`activation_quant_scale`**（`List[Tensor]`）：**可选参数**，预留参数，当前只支持传入None。
- **`activation_quant_offset`**（`List[Tensor]`）：**可选参数**，预留参数，当前只支持传入None。
- **`split_item`**（`int`）：**可选参数**，用于指定切分模式。数据类型支持`torch.int64`。
  - 0、1：输出为多个张量，数量与weight相同。
  - 2、3：输出为单个张量。

  <!-- npu="950" id37 -->
  - <term>Ascend 950PR&950DT系列产品</term>：非量化和伪量化支持取0/1/2/3，量化当前支持取2和3。
  <!-- end id37 -->

- **`group_type`**（`int`）：**可选参数**，代表需要分组的轴。数据类型支持`torch.int64`。
  - `group_list`输入类型为`List[int]`时仅支持传入None。
  - `group_list`输入类型为`Tensor`时，若矩阵乘为C\[m,n\]=A\[m,k\]xB\[k,n\]，group\_type支持的枚举值为：-1代表不分组；0代表m轴分组；1代表n轴分组，2代表k轴分组。

    <!-- npu="A3,910b" id38 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当前支持取-1、0、2。
    <!-- end id38 -->
    <!-- npu="310p" id39 -->
    - <term>Atlas推理系列产品</term>：当前只支持取0。
    <!-- end id39 -->
    <!-- npu="950" id40 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
      - 非量化场景支持取-1、0、2，weightNz场景时不支持取2。
      - 量化场景支持取0、2。
      - 伪量化场景支持取-1、0。
    <!-- end id40 -->

- **`group_list_type`**（`int`）：**可选参数**，代表`group_list`的表达形式。数据类型支持`torch.int64`。
  - `group_list`输入类型为`List[int]`时仅支持传入None。
  - `group_list`输入类型为`Tensor`时可取值0、1或2：
    - 0：默认值，`group_list`中数值为分组轴大小的cumsum（累积和）结果。
    - 1：`group_list`中数值为分组轴上每组大小。
    - 2：`group_list` shape为\[E, 2\]，E表示Group大小，数据排布为\[\[groupIdx0, groupSize0\], \[groupIdx1, groupSize1\], ...\]，其中groupSize为分组轴上每组大小。

    <!-- npu="A3,910b" id41 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅当`x`和`weight`参数输入类型为`torch.int8`，并且`group_type`取0（m轴分组）时，支持取2。
    <!-- end id41 -->
    <!-- npu="310p" id42 -->
    - <term>Atlas推理系列产品</term>：不支持取2。
    <!-- end id42 -->
    <!-- npu="950" id43 -->
    - <term>Ascend 950PR&950DT系列产品</term>：仅当非量化、全量化场景或MX A8W4伪量化场景下，并且`group_type`取0（m轴分组）时，支持取2。
    - <term>Ascend 950PR&950DT系列产品</term>：K-CG伪量化场景下，`group_list_type`仅支持取1。
    <!-- end id43 -->

- **`act_type`**（`int`）：**可选参数**，代表激活函数类型。数据类型支持`torch.int64`。
  - `group_list`输入类型为`List[int]`时仅支持传入None。
  - `group_list`输入类型为`Tensor`时，支持的枚举值包括：0代表NONE，不启用激活；1代表RELU激活；2代表GELU\_TANH激活；3代表GELU\_ERR\_FUNC激活，目前暂不支持；4代表FAST\_GELU激活；5代表SILU激活。

    <!-- npu="A3,910b" id44 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：取值范围为0-5。
    <!-- end id44 -->
    <!-- npu="310p" id45 -->
    - <term>Atlas推理系列产品</term>：当前只支持传入0。
    <!-- end id45 -->
    <!-- npu="950" id46 -->
    - <term>Ascend 950PR&950DT系列产品</term>：当前支持传入0、1、2、4、5。
    <!-- end id46 -->

- **`output_dtype`**（`int`）：**可选参数**，输出数据类型。支持的配置包括：
  - None：默认值，表示输出数据类型与输入`x`的数据类型相同。
  - 与输出y数据类型一致的类型，具体参考[约束说明](#约束说明)。

- **`tuning_config`**（`List[int]`）：**可选参数**，调优参数。如不使用该参数不传即可。
  <!-- npu="310p" id47 -->
  - <term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id47 -->
  <!-- npu="950" id48 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当前仅S8S4 K-C和S8S4 K-CG伪量化场景支持。
    - 第一个元素表示各个专家处理的token数的预期值。支持0和正整数，取值范围为\[0, min(M, UINT32_MAX)\]，其中M为输入`x`的行数。0表示不指定预期值；正整数表示各个专家处理的token数的预期值，且不能大于M。当前仅校验并保存该值，暂不参与实际调优计算。
    - 第二个元素仅支持0或1。设置为0或不传入该元素时，不开启`weight`特殊格式，`weight`按照常规\[E, K, N\]逻辑布局解析；设置为1时，开启`weight`特殊格式，仅支持`offset`不为空的perchannel场景，此时`weight`需按照\[E, N, K\]排布后转换为`FRACTAL_NZ`格式，且`weight`列表长度为1。例如，传入`[0, 1]`表示不指定预期token数，并开启`weight`特殊格式。
  <!-- end id48 -->

- **`x_dtype`**（`int`）：**可选参数**，输入`x`的真实数据类型。默认值None，表示输入`x`真实的数据类型与输入`x`的`dtype`相同。

  <!-- npu="A3,910b,310p" id49 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id49 -->
  <!-- npu="950" id50 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`。
  <!-- end id50 -->

- **`weight_dtype`**（`int`）：**可选参数**，输入`weight`的真实数据类型。默认值None，表示输入`weight`真实的数据类型与输入`weight`的`dtype`相同。伪量化场景使用`torch.float32`表示`torch_npu.float4_e2m1fn_x2`时无需传入。

  <!-- npu="A3,910b,310p" id51 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id51 -->
  <!-- npu="950" id52 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`。
  <!-- end id52 -->

- **`scale_dtype`**（`int`）：**可选参数**，输入`scale`的真实数据类型。默认值None，表示输入`scale`真实的数据类型与输入`scale`的`dtype`相同。

  <!-- npu="A3,910b,310p" id53 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id53 -->
  <!-- npu="950" id54 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当前仅支持`torch_npu.float8_e8m0fnu`。
  <!-- end id54 -->

- **`per_token_scale_dtype`**（`int`）：**可选参数**，输入`per_token_scale`的真实数据类型。默认值None，表示`per_token_scale`真实的数据类型与输入`per_token_scale`的`dtype`相同。

  <!-- npu="A3,910b,310p" id55 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id55 -->
  <!-- npu="950" id56 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当前仅支持`torch_npu.float8_e8m0fnu`。
  <!-- end id56 -->

## 返回值说明

**`y`**（`List[Tensor]`）：输出Tensor，数据格式支持$ND$。

- 当`split_item`为0或1时，返回的张量数量与`weight`相同。
- 当`split_item`为2或3时，返回的张量数量为1。

## 约束说明

- 该接口在全量化和非量化场景适用于训练或推理场景，伪量化场景仅适用于推理场景。
- 该接口支持单算子模式和TorchAir图模式。

<!-- npu="950" id57 -->
- 图模式场景下参数类型和格式的约束如下（仅适用于<term>Ascend 950PR&950DT系列产品</term>）：
  - 伪量化场景下，`offset`、`activation_input`、`activation_quant_scale`、`activation_quant_offset`只支持传入None。
  - 伪量化场景下，`x_dtype`只支持传入None，`weight_dtype`支持传入None或`torch_npu.hifloat8`。
  - 伪量化场景下，`x_dtype`为`torch.float16`/`torch.bfloat16`、`weight_dtype`为`torch_npu.int4`的PerGroup量化场景下，不支持图模式运行。
  - 非量化场景下，`x_dtype`只支持传入None，`weight_dtype`只支持传入None。
  - 全量化场景下，若输出数据类型`y`为`torch.int8`，则不支持图模式运行。全量化MX量化且`weight`输入格式为FRACTAL\_NZ格式，若weight多tensor输入，则不支持图模式运行。若`x`和`weight`输入类型都为`torch_npu.int4`，则暂不支持图模式。
<!-- end id57 -->
<!-- npu="950" id58 -->
- WeightNZ场景说明（仅适用于<term>Ascend 950PR&950DT系列产品</term>）：
  - 仅伪量化与全量化（输入`x`/`weight`数据类型为`torch.int8`、`x`/`weight`数据类型为`torch.float8_e4m3fn`/`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`且scale\_dtype与per\_token\_scale\_dtype为`torch_npu.float8_e8m0fnu`）weight支持FRATCAL\_NZ数据格式。
  - 全量化场景下，`weight`输入为FRATCAL\_NZ数据格式时，K轴和N轴均不能为1。MXFP4场景，K必须大于2，weight不转置时N必须大于2。MXFP4场景支持静态图模式，不支持动态图模式。MXFP4静态图模式场景E必须大于1且K必须大于64。x和weight均为`torch_npu.int4`时，不支持图模式运行，K必须保持8对齐。
<!-- end id58 -->
<!-- npu="950" id59 -->
- scale NZ亲和格式使用约束（仅适用于<term>Ascend 950PR&950DT系列产品</term>）：
  - 仅支持MX量化场景，`scale`数据类型为`torch_npu.float8_e8m0fnu`。
  - `scale`使用NZ亲和格式时，`weight`必须为`FRACTAL_NZ`格式。
  - `x`和`weight`数据类型仅支持`torch.float8_e4m3fn`。
  - 仅支持`weight`不转置场景。
  - 仅支持`group_type`为0，即M轴分组。
  - 不支持图模式，仅支持单算子模式。
  - 单tensor场景下，scale NZ亲和格式的storage shape为$[E, \lceil N/16 \rceil, \lceil K/64 \rceil, 16, 2]$；多tensor场景下，每个scale tensor的storage shape为$[\lceil N_i/16 \rceil, \lceil K_i/64 \rceil, 16, 2]$。
<!-- end id59 -->
- 内轴限制如下：

  <!-- npu="A3,910b" id60 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：内轴限制InnerLimit为65536。
  <!-- end id60 -->

- `x`和`weight`中每一组tensor的最后一维大小都应小于InnerLimit。x<sub>i</sub>的最后一维指当x不转置时x<sub>i</sub>的K轴或当x转置时x<sub>i</sub>的M轴。weight<sub>i</sub>的最后一维指当weight不转置时weight<sub>i</sub>的N轴或当weight转置时weight<sub>i</sub>的K轴。

<!-- npu="950" id61 -->
- tuning\_config使用场景限制：

    - S8S4 K-C和S8S4 K-CG伪量化场景，仅适用于<term>Ascend 950PR&950DT系列产品</term>。
    - 量化场景：输入为`torch.int8`，输出为`torch.int32`/`torch.bfloat16`/`torch.float16`/`torch.int8`，且为单tensor单专家场景，数据类型如下表。

    | x | weight | output_dtype | y |
    | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int8` | `torch.int8` |
    | `torch.int8` | `torch.int8` | `torch.bfloat16` | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.float16` | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.int32` |
<!-- end id61 -->
- 各场景输入与输出数据类型使用约束：

  <!-- npu="A3,910b" id62 -->
  - **group\_list输入类型为List\[int\]时**（针对<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>）

    **表 1**  数据类型约束

    | 场景 | x | weight | bias | scale | antiquant_scale | antiquant_offset | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 非量化 | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | `torch.float16` | `torch.float16` |
    | 非量化 | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | None | None | None | `torch.bfloat16` | `torch.bfloat16` |
    | 非量化 | `torch.float32` | `torch.float32` | `torch.float32` | None | None | None | `torch.float32` | `torch.float32` |
    | perchannel全量化 | `torch.int8` | `torch.int8` | `torch.int32` | `torch.int64` | None | None | `torch.int8` | `torch.int8` |
    | 伪量化 | `torch.float16` | `torch.int8` | `torch.float16` | None | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` |
    | 伪量化 | `torch.bfloat16` | `torch.int8` | `torch.float32` | None | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` |
  <!-- end id62 -->
  <!-- npu="A3,910b" id63 -->
  - **group\_list输入类型为Tensor时**（针对<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>）

    **表 2**  数据类型约束

    | 场景 | x | weight | bias | scale | antiquant_scale | antiquant_offset | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 非量化 | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | None | None/`torch.float16` | `torch.float16` |
    | 非量化 | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | None | None | None | None | None/`torch.bfloat16` | `torch.bfloat16` |
    | 非量化 | `torch.float32` | `torch.float32` | `torch.float32` | None | None | None | None | None/`torch.float32`（仅x/weight/y均为单张量） | `torch.float32` |
    | perchannel全量化 | `torch.int8` | `torch.int8` | `torch.int32` | `torch.int64` | None | None | None | None/`torch.int8` | `torch.int8` |
    | perchannel全量化 | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | None | None | None | `torch.bfloat16` | `torch.bfloat16` |
    | perchannel全量化 | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float32` | None | None | None | `torch.float16` | `torch.float16` |
    | pertoken全量化 | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | None | None | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` |
    | pertoken全量化 | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float32` | None | None | `torch.float32` | `torch.float16` | `torch.float16` |
    | 伪量化 | `torch.float16` | `torch.int8`/`torch_npu.int4` | `torch.float16` | None | `torch.float16` | `torch.float16` | None | None/`torch.float16` | `torch.float16` |
    | 伪量化 | `torch.bfloat16` | `torch.int8`/`torch_npu.int4` | `torch.float32` | None | `torch.bfloat16` | `torch.bfloat16` | None | None/`torch.bfloat16` | `torch.bfloat16` |

    > **说明：**
    > - 伪量化场景，若weight的类型为`torch.int8`，仅支持perchannel模式；若weight的类型为`torch_npu.int4`，支持perchannel和pergroup两种模式。若为pergroup，pergroup数G或G<sub>i</sub>必须要能整除对应的k<sub>i</sub>。若weight为多tensor，定义pergroup长度s<sub>i</sub>  = k<sub>i</sub>  / G<sub>i</sub>，要求所有s<sub>i</sub>\(i=1,2,...g\)都相等。
    > - 伪量化场景，若weight的类型为`torch_npu.int4`，则weight中每一组tensor的最后一维大小都应是偶数。weight<sub>i</sub>的最后一维指weight不转置时weight<sub>i</sub>的N轴或当weight转置时weight<sub>i</sub>的K轴。并且在pergroup场景下，当weight转置时，要求pergroup长度s<sub>i</sub>是偶数。tensor转置：指若tensor shape为\[M,K\]时，则stride为\[1,M\]，数据排布为\[K,M\]的场景，即非连续tensor。
    > - 当前PyTorch不支持`torch_npu.int4`类型数据，需要使用时可以通过[torch\_npu.npu\_quantize](torch_npu-npu_quantize.md)接口使用`torch.int32`数据表示`torch_npu.int4`。
  <!-- end id63 -->
  <!-- npu="310p" id64 -->
  - **group\_list输入类型为Tensor时**（针对<term>Atlas推理系列产品</term>）

    **表 3**  数据类型约束

    | x | weight | bias | scale | antiquant_scale | antiquant_offset | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | `torch.float32` | `torch.float16` | `torch.float16` |
  <!-- end id64 -->
  <!-- npu="950" id65 -->
  - **group\_list输入类型为Tensor时**（针对<term>Ascend 950PR&950DT系列产品</term>）

    **表 4**  数据类型约束

    | 场景 | x | weight | bias | scale | antiquant_scale | antiquant_offset | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | perchannel伪量化 | `torch.float16`/`torch.bfloat16` | `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float16`/`torch.bfloat16`/`torch.float32`/None | None | `torch.float16`/`torch.bfloat16` | None | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | perchannel伪量化 | `torch.float16`/`torch.bfloat16` | `torch.int8`/`torch.int32`（`torch_npu.int4`） | `torch.float16`/`torch.bfloat16`/`torch.float32`/None | None | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16`/None | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | pergroup伪量化 | `torch.float16`/`torch.bfloat16` | `torch.int32`(`torch_npu.int4`) | `torch.float16`/`torch.float32`/None | None | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16`/None | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | mx伪量化 | `torch.float16`/`torch.bfloat16` | `torch.float32`（`torch_npu.float4_e2m1fn_x2`） | `torch.float16`/`torch.bfloat16`/`torch.float32`/None | None | `torch_npu.float8_e8m0fnu` | None | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | mx伪量化 | `torch.float8_e4m3fn` | `torch.uint8`/`torch.int8`（`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`） | `torch.float16`/`torch.bfloat16` | None | `torch_npu.float8_e8m0fnu` | None | `torch_npu.float8_e8m0fnu` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | K-CG伪量化 | `torch.int8` | `torch.int32`（`torch_npu.int4`） | `torch.float32` | `torch.float32` | `torch.float16` | None | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | MxA8W8（全量化） | ND支持`torch.float8_e4m3fn`/`torch.float8_e5m2`，FRACTAL_NZ支持`torch.float8_e4m3fn` | ND支持`torch.float8_e4m3fn`/`torch.float8_e5m2`，<br>FRACTAL_NZ支持`torch.float8_e4m3fn` | None | `torch_npu.float8_e8m0fnu` | None | None | `torch_npu.float8_e8m0fnu` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | MxA4W4（全量化） | ND支持`torch_npu.float4_e2m1fn_x2`，FRACTAL_NZ支持`torch_npu.float4_e1m2fn_x2` | ND支持`torch_npu.float4_e2m1fn_x2`，FRACTAL_NZ支持`torch_npu.float4_e1m2fn_x2` | `torch.float32`/None | `torch_npu.float8_e8m0fnu` | None | None | `torch_npu.float8_e8m0fnu` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | T-T && T-C静态量化 | `torch.int8` | `torch.int8` | `torch.int32`/None | `torch.int64` | None | None | None | `torch.float16`/`torch.bfloat16`/`torch.int8` | `torch.float16`/`torch.bfloat16`/`torch.int8` |
    | T-T && T-C静态量化 | `torch.int8` | `torch.int8` | `torch.int32`/`torch.bfloat16`/`torch.float32`/None | `torch.bfloat16`/`torch.float32` | None | None | None | `torch.bfloat16` | `torch.bfloat16` |
    | T-T && T-C静态量化 | `torch.int8` | `torch.int8` | `torch.int32`/`torch.float16`/`torch.float32`/None | `torch.float32` | None | None | None | `torch.float16` | `torch.float16` |
    | T-T && T-C静态量化 | `torch.int8` | `torch.int8` | `torch.int32`/None | None/`torch.int64` | None | None | None | `torch.int32` | `torch.int32` |
    | T-T && T-C静态量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.int64`/`torch.float32` | None | None | None | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | T-T && T-C静态量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.int64`/`torch.float32` | None | None | None | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | K-T && K-C动态量化 | `torch.int8` | `torch.int8` | `torch.int32`/`torch.bfloat16`/`torch.float32`/None | `torch.bfloat16`/`torch.float32` | None | None | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` |
    | K-T && K-C动态量化 | `torch.int8` | `torch.int8` | `torch.int32`/`torch.float16`/`torch.float32`/None | `torch.float32` | None | None | `torch.float32` | `torch.float16` | `torch.float16` |
    | K-T && K-C动态量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | K-T && K-C动态量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | K-C动态量化 | `torch_npu.int4` | `torch_npu.int4` | None | `torch.uint64` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | K-G量化 | `torch_npu.int4` | `torch_npu.int4` | None | `torch.uint64` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | T-C && T-T动态量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | T-C && T-T动态量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | G-B动态量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | G-B动态量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | 非量化 | `torch.float16` | `torch.float16` | `torch.float16`/`torch.float32` | None | None | None | None | `torch.float16` | `torch.float16` |
    | 非量化 | `torch.bfloat16` | `torch.bfloat16` | `torch.float16`/`torch.float32` | None | None | None | None | `torch.bfloat16` | `torch.bfloat16` |
    | 非量化 | `torch.float32` | `torch.float32` | `torch.float32` | None | None | None | None | `torch.float32` | `torch.float32` |
    | S8S4 K-C/K-CG伪量化 | `torch.int8` | `torch_npu.int4` | `torch.float32` | `torch.uint64` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
  <!-- end id65 -->

- 根据输入x、输入weight与输出y的Tensor数量不同，支持以下几种场景。场景中的“单”表示单个张量，“多”表示多个张量。场景顺序为x、weight、y，例如“单多单”表示x为单张量，weight为多张量，y为单张量。

  <!-- npu="A3,910b" id66 -->
  - **group\_list输入类型为List\[int\]时支持的场景如下**（针对<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>）

    | 支持场景 | 场景说明 | 场景限制 |
    | --- | --- | --- |
    | 多多多 | x和weight为多张量，y为多张量。每组数据的张量是独立的。 | 1. 仅支持split_item为0或1。<br>  2. x中tensor要求维度一致且支持2-6维，weight中tensor需为2维，y中tensor维度和x保持一致。<br>  3. x中tensor大于2维，group_list必须传空。<br>  4. x中tensor为2维且传入group_list，group_list的差值需与x中tensor的第一维一一对应。 |
    | 单多单 | x为单张量，weight为多张量，y为单张量。 | 1. 仅支持split_item为2或3。<br>  2. 必须传group_list，且最后一个值与x中tensor的第一维相等。<br>  3. x、weight、y中tensor需为2维。<br>  4. weight中每个tensor的N轴必须相等。 |
    | 单多多 | x为单张量，weight为多张量，y为多张量。 | 1. 仅支持split_item为0或1。<br>  2. 必须传group_list，group_list的差值需与y中tensor的第一维一一对应。<br>  3. x、weight、y中tensor需为2维。 |
    | 多多单 | x和weight为多张量，y为单张量。每组矩阵乘法的结果连续存放在同一个张量中。 | 1. 仅支持split_item为2或3。<br>  2. x、weight、y中tensor需为2维。<br>  3. weight中每个tensor的N轴必须相等。<br>  4. 若传入group_list，group_list的差值需与x中tensor的第一维一一对应。 |
  <!-- end id66 -->
  <!-- npu="A3,910b" id67 -->
  - **group\_list输入类型为Tensor时支持的场景如下**（针对<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>）

    > **说明：**
    > - 量化、伪量化仅支持group\_type为-1和0场景。
    > - 仅pertoken量化场景支持激活函数计算。

    | group_type | 支持场景 | 场景说明 | 场景限制 |
    | --- | --- | --- | --- |
    | -1 | 多多多 | x和weight为多张量，y为多张量。每组数据的张量是独立的。 | 1. 仅支持split_item为0或1。<br>  2. x中tensor要求维度一致且支持2-6维，weight中tensor需为2维，y中tensor维度和x保持一致。<br>  3. group_list必须传空。<br>  4. 支持weight转置，但weight中每个tensor是否转置需保持统一。<br>  5. x不支持转置。 |
    | 0 | 单单单 | x、weight与y均为单张量。 | 1. 仅支持split_item为2或3。<br>  2. weight中tensor需为3维，x、y中tensor需为2维。<br>  3. 必须传group_list，且当group_list_type为0时，最后一个值与x中tensor的第一维相等，当group_list_type为1时，数值的总和与x中tensor的第一维相等，当group_list_type为2时，第二列数值的总和小于等于x中tensor的第一维。<br>  4. group_list第1维最大支持1024，即最多支持1024个group。<br>  5. 支持weight转置。<br>  6. x不支持转置。 |
    | 0 | 单多单 | x为单张量，weight为多张量，y为单张量。 | 1. 仅支持split_item为2或3。<br>  2. 必须传group_list，且当group_list_type为0时，最后一个值与x中tensor的第一维相等，当group_list_type为1时，数值的总和与x中tensor的第一维相等且长度最大为128，当group_list_type为2时，第二列数值的总和小于等于x中tensor的第一维且长度最大为128。<br>  3. x、weight、y中tensor需为2维。<br>  4. weight中每个tensor的N轴必须相等。<br>  5. 支持weight转置，但weight中每个tensor是否转置需保持统一。<br>  6. x不支持转置。 |
    | 0 | 多多单 | x和weight为多张量，y为单张量。每组矩阵乘法的结果连续存放在同一个张量中。 | 1. 仅支持split_item为2或3。<br>  2. x、weight、y中tensor需为2维。<br>  3. weight中每个tensor的N轴必须相等。<br>  4. 若传入group_list，当group_list_type为0时，group_list的差值需与x中tensor的第一维一一对应，当group_list_type为1时，group_list的数值需与x中tensor的第一维一一对应且长度最大为128，当group_list_type为2时，group_list第二列的数值需与x中tensor的第一维一一对应且长度最大为128。<br>  5. 支持weight转置，但weight中每个tensor是否转置需保持统一。<br>  6. x不支持转置。 |
    | 2 | 单单单 | x、weight与y均为单张量 | 1. 仅支持split_item为2或3。<br>  2. x、weight中tensor需为2维，y中tensor需为3维。<br>  3. 必须传group_list，且当group_list_type为0时，最后一个值与x中tensor的第二维相等，当group_list_type为1时，数值的总和与x中tensor的第二维相等，当group_list_type为2时，第二列数值的总和小于等于x中tensor的第二维。<br>  4. group_list第1维最大支持1024，即最多支持1024个group。<br>  5. x必须转置，weight不能转置。 |
  <!-- end id67 -->
  <!-- npu="310p" id68 -->
  - **group\_list输入类型为Tensor时支持的场景如下**（针对<term>Atlas推理系列产品</term>）

    输入输出只支持`torch.float16`的数据类型，输出y的n轴大小需要是16的倍数。

    | group_type | 支持场景 | 场景说明 | 场景限制 |
    | --- | --- | --- | --- |
    | 0 | 单单单 | x、weight与y均为单张量 | 1. 仅支持split_item为2或3。<br>  2. weight中tensor需为3维，x、y中tensor需为2维。<br>  3. 必须传group_list，且当group_list_type为0时，最后一个值与x中tensor的第一维相等，当group_list_type为1时，数值的总和与x中tensor的第一维相等。<br>  4. group_list第1维最大支持1024，即最多支持1024个group。<br>  5. 支持weight转置，不支持x转置。 |
  <!-- end id68 -->
  <!-- npu="950" id69 -->
  - **group\_list输入类型为Tensor时支持的场景如下**（<term>Ascend 950PR&950DT系列产品</term>）

    > **说明：**
    > - 全量化仅支持group\_type为0和2的单单单场景，仅在输入x和weight为`torch_npu.hifloat8`、`torch.float8_e4m3fn`或`torch.float8_e5m2`的全量化支持训练场景即group\_type为2。
    > - 伪量化场景支持`group_type`为-1的多多多场景，以及`group_type`为0的单单单和单多单场景。其中，单多单表示`x`为单张量、`weight`为多张量、`y`为单张量。
    >   - perchannel伪量化场景，且输入`x`为`torch.float16`/`torch.bfloat16`、`weight`为`torch.int8`/`torch.int32`（`torch_npu.int4`）时。
    >   - pergroup伪量化场景，且输入`x`为`torch.float16`/`torch.bfloat16`、`weight`为`torch.int32`（`torch_npu.int4`）时。
    > - 伪量化场景不支持激活函数计算。全量化场景下，当x和weight为`torch.int8`，量化模式为静态T-C量化或动态K-C量化，且scale类型为`torch.float32`/`torch.bfloat16`时，act\_type参数支持传入0、1、2、4、5。其余全量化场景act\_type参数仅支持传入0。

    | group_type | 支持场景 | 场景说明 | 场景限制 |
    | --- | --- | --- | --- |
    | -1 | 多多多 | x和weight为多张量，y为多张量。每组数据的张量是独立的。 | 1. 仅支持split_item为0或1（非量化仅支持split_item为0）。<br>  2. x中tensor要求维度一致，伪量化pergroup量化场景下仅支持2维，其他场景支持2-6维，非量化支持2维，weight中tensor需为2维，y中tensor维度和x保持一致。<br>  3. group_list必须传空。<br>  4. 支持weight转置，但weight中每个tensor是否转置需保持统一。<br>  5. x不支持转置。<br>  6. 非量化/伪量化场景bias可选且shape仅支持1维的tensorList[(n),(n),...,(n)]，list长度与weight长度相同。<br>  7. pergroup伪量化场景，weight最后一维需要为偶数，K为group_size整数倍；多多多场景下不支持weight非转置。 |
    | 0 | 单单单 | `x`、`weight`与`y`均为单张量。 | 1. 仅支持`split_item`为2或3。<br>  2. 必须传`group_list`，且当`group_list_type`为0时，最后一个值不大于`x`中tensor的第一维，当`group_list_type`为1时，数值的总和不大于`x`中tensor的第一维。当`group_list_type`为2时，第二列数值的总和不大于`x`中tensor的第一维。<br>  3. `group_list`第1维最大支持1024，即最多支持1024个group。<br>  4. 量化场景，仅在`x`输入`dtype`为`torch.int8`/`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`时支持`bias`；非量化场景和伪量化场景都支持`bias`。<br>  5. `x`仅支持不转置，`weight`支持转置或不转置。但是`x`、`weight`均为`torch_npu.int4`场景下，`weight`格式为ND的话，仅支持不转置。<br>  6. `weight`中单tensor需为3维，`x`、`y`中单tensor需为2维。<br>  7. K轴不能为0。<br>  8. `x`输入`dtype`为`torch.float16`/`torch.bfloat16`、`weight`为`torch.float32`（`torch_npu.float4_e2m1fn_x2`）的mx伪量化场景，仅支持`x`不转置，`weight`不转置；`x`输入`dtype`为`torch.float16`/`torch.bfloat16`、`weight`为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`的perchannel伪量化场景和`x`输入`dtype`为`torch.float8_e4m3fn`、`weight`为`torch.float32`（`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`）的mx伪量化场景，仅支持`x`不转置，`weight`转置。<br>  9. mx伪量化和K-CG伪量化场景，`weight`的最后2维需要满足32B对齐。<br>  10. `x`输入`dtype`为`torch.float8_e4m3fn`、`weight`为`torch.float32`（`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`）的mx伪量化场景，`bias`类型需要与`output_dtype`数据类型一致，此时必须传入可选参数`output_dtype`。<br>  11. `bias`可选且shape仅支持2维的（g,n），其中g为M的分组数。 |
    | 0 | 单多单 | `x`为单张量、`weight`为多张量、`y`为单张量。 | 1. 仅支持`split_item`为2或3。<br>  2. 必须传`group_list`，且当`group_list_type`为0时，最后一个值不大于`x`中tensor的第一维，当`group_list_type`为1时，数值的总和不大于`x`中tensor的第一维。当`group_list_type`为2时，第二列数值的总和不大于`x`中tensor的第一维。支持长度最大为128，非量化场景长度最大为1024。<br>  3. `x`、`y`中tensor需为2维，shape分别为（M, K）和（M, N）。<br>  4. `weight`中tensor需为2维，shape分别为（N, K）或（K, N）。<br>  5. `weight`中每个tensor的N轴必须相等。<br>  6. 支持`weight`转置，但`weight`中每个tensor是否转置需保持统一。<br>  7. `x`不支持转置。<br>  8. 非量化和伪量化场景支持传入`bias`，`bias`可选且shape仅支持1维的tensorList[(n),(n),...,(n)]，list长度与`weight`列表长度相同。<br>  9. 全量化MX量化且输入`w`为`FRACTAL_NZ`格式时，支持单多单场景，此时不支持传入`bias`。 |
    | 0 | 多多单 | x和weight为多张量，y为单张量。每组矩阵乘法的结果连续存放在同一个张量中。 | 1. 仅支持split_item为2或3。<br>  2. x、weight、y中tensor需为2维。<br>  3. weight中每个tensor的N轴必须相等。<br>  4. 若传入group_list，当group_list_type为0时，group_list的差值需与x中tensor的第一维一一对应，当group_list_type为1时，group_list的数值需与x中tensor的第一维一一对应且长度最大为128，非量化场景长度最大为1024。<br>  5. 支持weight转置，但weight中每个tensor是否转置需保持统一。<br>  6. x不支持转置。<br>  7. 非量化场景bias可选且shape仅支持1维的tensorList[(n),(n),...,(n)]，list长度与weight长度相同。 |
    | 2 | 单单单 | x、weight与y均为单张量。 | 1. 仅支持split_item为2或3。<br>  2. 必须传group_list，且当group_list_type为0时，最后一个值不大于x中tensor的第一维，当group_list_type为1时，数值的总和不大于x中tensor的第一维。<br>  3. group_list第1维最大支持1024，即最多支持1024个group。<br>  4. 在全量化场景下，不支持scale为`torch.int64`。<br>  5. 仅支持x转置，weight不转置。<br>  6. y中单tensor需为3维，x、weight中单tensor需为2维。<br>  7. 图模式场景下，K轴不能为0。<br>  8. 仅支持ND进ND出。 |
    | 2 | 单多多 | x为单张量，weight为多张量，y为多张量。 | 1. 仅支持split_item为0或1。<br>  2. x、weight、y中tensor需为2维。<br>  3. 若传入grouplist，当group_list_type为0时，group_list的差值需与x中tensor的第一维一一对应，当group_list_type为1时，group_list的数值需与x中tensor的第一维一一对应且长度最大为1024。<br>  4. 仅支持x转置，weight不转置。<br>  5. 非分组的轴的shape不能为0。<br>  6. 仅支持ND进ND出。<br>  7. 不支持bias。<br>  8. 仅支持非量化。 |

    - **单单单的全量化场景**：各场景对scale和per\_token\_scale的shape规格限制如下（其中g为group\_list shape大小，M与x的shape m一致，K与x的shape k一致，N与weight的shape n一致）：

        | group_type | 量化模式 | shape限制 |
        | --- | --- | --- |
        | 0/2 | T-T && T-C静态量化 | <li>scale为单tensor，每个tensor 2维或1维，C量化场景shape为（g, N）；T量化场景shape为（g, 1）或（g,）。`torch.int8`输出场景下，仅支持C量化，shape为（g, N）。</li><li>per_token_scale无需赋值。</li> |
        | 0/2 | K-T && K-C动态量化 | <li>scale为单tensor，每个tensor 2维或1维，C量化场景shape为（g, N）；T量化场景shape为（g, 1）或（g,）。</li><li>per_token_scale为单tensor，group_type等于0时，shape为1维（M,）；group_type等于2时，shape为2维（g, M）。</li> |
        | 0/2 | T-T动态量化 | scale和per_token_scale均为单tensor，2维时shape为（g, 1）或1维时shape为（g, ）。 |
        | 0/2 | T-C动态量化 | <li>scale为单tensor，每个tensor 2维，shape为（g, N）。</li><li>per_token_scale为单tensor，每个tensor 2维或1维，2维时shape为（g, 1）或1维时shape为（g, ）。</li> |
        | 0 | mx量化 | 计算公式（4）中gsm = gsn = 1，gsk = 32。scale和per_token_scale均为单tensor，scale的shape支持（g, N, ceil(K/64), 2）或  (g, ceil(K/64), N, 2)，per_token_scale的shape仅支持（M, ceil(K/64), 2），输入x为`torch_npu.float4_e2m1fn_x2`时，需要满足K为偶数并且K不为2。当weight非转置时还需满足N为偶数。 |
        | 2 | mx量化 | 计算公式（4）中gsm = gsn = 1，gsk = 32。scale和per_token_scale均为单tensor。scale shape仅支持(K//64+g, N, 2)，per_token_scale shape仅支持(K//64 + g, M, 2)。 |
        | 0 | G-B动态量化 | 计算公式（4）中gsm = 1，gsn = gsk = 128。scale为单tensor，scale为3维，shape为（g, ceil(K/128), ceil(N/128)）或（g, ceil(N/128), ceil(K/128)）。<br>per_token_scale为单tensor，per_token_scale为2维，shape为（M,ceil(K/128)）。 |
        | 2 | G-B动态量化 | 计算公式（4）中gsm = 1，gsn = gsk = 128。scale为单tensor，scale为2维，shape为（K//128+g, ceil(N/128)）。<br>per_token_scale为单tensor，per_token_scale为2维，shape为（K//128+g, M） |
        | 0 | K-G量化 | <li>每个tensor 3维，shape为（E, G, N），$G$必须要能整除$K$，且$k/G$需为偶数</li><li>per_token_scale为单tensor，shape为1维（M,）。</li> |

        > **单单单的全量化特殊场景说明**：
        > - 当group\_type为0或2，N=1且scale的shape为\(g, 1\)时，weight既可以pertensor量化也可以perchannel量化时，优先选择pertensor量化模式。
        > - 当group\_type为2，M=1且per\_token\_scale的shape为\(g, 1\)时，x既可以pertoken量化也可以pertensor量化时，优先选择pertensor量化模式。
        > - 在动态量化场景，当group\_type为2、K<128、N≤128，且scale的shape为2维\(g, 1\)时，按照已有量化模式区分规则，既可以为非G-B动态量化，又可以为G-B动态量化，这种场景现在一律按照G-B动态量化处理。如果期望使用非G-B动态量化，scale推荐1维\(g,\)，以防与G-B动态量化混淆。
        > - 在动态量化场景，当group\_type为0、g = M、K \> 128且per\_token\_scale的shape为\(g,\)时，x选择pertoken量化模式；当group\_type为0、g = M，K <= 128且per\_token\_scale的shape为\(g, 1\)时，根据weight的量化模式选择x的量化模式（weight如果是perchannel或者pertensor量化，x选择pertensor量化；weight如果是perblock量化，x选择pergroup量化）。
        > - 在动态量化场景，当group\_type为2、K<128、M不等于1时，如果N小于等于128，x则选择pergroup量化；如果N大于128，根据weight的量化模式选择x的量化模式（weight如果是perchannel或者pertensor量化，x选择pertoken量化；weight如果是perblock量化，x选择pergroup量化）。
        > - 在动态量化场景，当group\_type为2、K<128、M等于1且per\_token\_scale的shape为\(g, 1\)时，如果N小于等于128，x则选择pergroup量化；如果N大于128，根据weight的量化模式选择x的量化模式（weight如果是perchannel或者pertensor量化，x选择pertensor量化；weight如果是perblock量化，x选择pergroup量化）。
        > - 在全量化`torch_npu.int4`场景，x为pertoken量化、weight为pergroup量化时，pergroup数G必须要整除K，且K/G需为偶数,weightNZ转置后，K/G必须按照64对齐，K按照64对齐，N按照16对齐。

    - **伪量化场景**：各场景对antiquant\_scale、scale和per\_token\_scale的shape规格限制如下（其中g为group\_list的shape大小，M和K与x的m和k一致，N与weight的n一致）：

        | group_type | 量化模式 | shape限制 |
        | --- | --- | --- |
        | -1 | perchannel | antiquant_scale为多tensor，每个tensor为1维，shape为（ni）。 |
        | 0 | perchannel | antiquant_scale为单tensor，每个tensor为2维，shape为（g, N）。 |
        | -1 | pergroup | <li>group_size支持32/64/128/256，pergroup数G=k/gs，要求K可以被gs整除。</li><li>antiquant_scale/antiquant_offset为多tensor，每个tensor为2维，shape为 (Gi, ni)，固定为非转置。</li> |
        | 0 | pergroup | <li>group_size支持32/64/128/256，pergroup数G=k/gs，要求K可以被gs整除。</li><li>antiquant_scale/antiquant_offset为单tensor，每个tensor为3维，shape为 (g, G, N)，固定为非转置。</li> |
         | 0 | mx | pergroup数G=K/32，要求K可以被32整除。<br><ul><li>`x`为`torch.float16`、`torch.bfloat16`输入，`weight`为`torch.float32`（`torch_npu.float4_e2m1fn_x2`）输入场景：`antiquant_scale`为单tensor，每个tensor为3维，shape为（g, K/32, N）。</li><li>`x`为`torch.float8_e4m3fn`输入，`weight`为`torch.uint8`/`torch.int8`（`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`）输入场景：<ul><li>G需要为偶数。</li><li>单单单：`antiquant_scale`为单tensor，每个tensor为4维，shape为（g, N, K/64, 2）。</li><li>单单单：`per_token_scale`为单tensor，每个tensor为3维，shape为（M, K/64, 2）。</li><li>单多单：`antiquant_scale`为单tensor，每个tensor为3维，shape为（N, K/64, 2）。</li><li>单多单：`per_token_scale`为单tensor，每个tensor为3维，shape为（M, K/64, 2）。</li></ul></li></ul> |
        | 0 | K-CG | <li>仅支持`group_list_type`为1，此时`group_list`中的数值表示分组轴上每组大小。</li><li>antiquant_scale为单tensor，每个tensor为3维，shape为（g, G, N），其中pergroup数G=K/gs，gs支持取值为128、192、256、512，要求K可以被gs整除。</li><li>per_token_scale为单tensor，每个tensor为1维，shape为（M）。</li><li>scale为单tensor，每个tensor为2维，shape为，shape为（g, N）。</li> |
        | 0 | S8S4 K-CG | scale为单tensor，shape为（g, K/256, N）；per_token_scale为单tensor，shape为（M）。 |
        | 0 | S8S4 K-C | scale和offset均为单tensor，shape为（g, 1, N）；per_token_scale为单tensor，shape为（M）。 |
  <!-- end id69 -->

## 调用示例

- 单算子模式调用

  - 非量化场景示例：

    ```python
    import torch
    import torch_npu
    M = 256
    K = 256
    N = 256
    g = 2
    x = torch.randint(-1, 1, (M, K), dtype=torch.float16).npu()
    weight = torch.randint(-1, 1, (g, K, N), dtype=torch.float16).npu().transpose(1,2)
    group_list = torch.Tensor([128, 256]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.float16, group_list_type=0)
    ```

  <!-- npu="950" id70 -->
  - <term>Ascend 950PR&950DT系列产品</term>：`torch_npu.hifloat8`输入，K-T动态量化场景

    ```python
    import torch
    import torch_npu

    M = 576
    K = 512
    N = 7168
    g = 4
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).npu()
    weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).npu().transpose(1,2)
    x2_scale = torch.randint(-1, 1, (g, 1), dtype=torch.float32).npu()
    x1_scale = torch.randint(-1, 1, (M,), dtype=torch.float32).npu()

    group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch_npu.hifloat8, weight_dtype=torch_npu.hifloat8, group_list_type=0)
    ```
  <!-- end id70 -->
  <!-- npu="950" id71 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx量化场景示例-mxfp8

    ```python
    import math
    import torch
    import torch_npu
    M = 576
    K = 512
    N = 7168
    g = 4
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(1,2)
    x2_scale = torch.randint(-1, 1, (g, N, math.ceil(K/64), 2), dtype=torch.int8).npu().transpose(1,2)
    x1_scale = torch.randint(-1, 1, (M, math.ceil(K/64), 2), dtype=torch.int8).npu()

    group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)
    ```
  <!-- end id71 -->
  <!-- npu="950" id72 -->
  - <term>Ascend 950PR&950DT系列产品</term>：G-B动态量化场景示例

    ```python
    # G-B
    import math
    import torch
    import torch_npu

    M = 576
    K = 512
    N = 7168
    g = 4
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(1,2)
    x2_scale = torch.randint(-1, 1, (g, math.ceil(N/128), math.ceil(K/128)), dtype=torch.float32).npu().transpose(1,2)
    x1_scale = torch.randint(-1, 1, (M, math.ceil(K/128)), dtype=torch.float32).npu()

    group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0)
    ```
  <!-- end id72 -->
  <!-- npu="950" id73 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化weightNZ场景示例（x/weight为`torch.int8`）

    ```python
    import torch
    import torch_npu
    M = 4
    K = 4
    N = 5
    g = 2
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).npu()
    weight1 = torch.randint(-1, 1, (g, K, N), dtype=torch.int8).npu()
    weightnz = torch_npu.npu_format_cast(weight1, 29)
    scale = torch.randint(-1, 1, (g, N), dtype=torch.bfloat16).npu()
    pertoken = torch.randint(-1, 1, (M,), dtype=torch.float32).npu()
    group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
    split_item = 3
    bias = torch.randint(-1, 1, (g, N), dtype=torch.int32).npu()

    npu_out = torch_npu.npu_grouped_matmul([x], [weightnz], scale=[scale], bias = [bias], per_token_scale = [pertoken], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0)
    ```
  <!-- end id73 -->
  <!-- npu="950" id74 -->
  - <term>Ascend 950PR&950DT系列产品</term>：非量化weightNZ场景示例

    ```python
    import torch
    import torch_npu
    M = 1
    K = 256
    N = 256
    g = 1
    x = torch.randint(-1, 1, (M, K), dtype=torch.float16).npu()
    weight = torch.randint(-1, 1, (g, K, N), dtype=torch.float16).npu().transpose(1,2)

    group_list = torch.Tensor([1]).to(torch.int64).npu()
    split_item = 2
    weight_flag = 29
    weight = torch_npu.npu_format_cast(weight, weight_flag)
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.float16, group_list_type=0)
    ```
  <!-- end id74 -->
  <!-- npu="950" id75 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx（A4W4）全量化场景示例-mxfp4

    ```python
    import math
    import torch
    import torch_npu
    M = 64
    N = 128
    K = 128
    g = 1
    #使用uint8代替两个mxfp4，此处uint8的内轴需要扩展2倍才是真实的mxfp4的内轴
    x = torch.randint(1, 3, (M, int(K/2)), dtype=torch.uint8).npu()

    weight = torch.randint(1, 3, (g, K, int(N/2)), dtype=torch.uint8).npu()

    scale = torch.randint(-1, 1, (g, math.ceil(K/64),N,2), dtype=torch.int8).npu()
    per_token_scale = torch.randint(-1, 1, (M, math.ceil(K/64),2), dtype=torch.int8).npu()

    group_list = torch.Tensor([1]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[scale], per_token_scale = [per_token_scale], group_list=group_list,
    split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch_npu.float4_e2m1fn_x2, weight_dtype=torch_npu.float4_e2m1fn_x2,
    scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)
    ```
  <!-- end id75 -->
  <!-- npu="950" id76 -->
  - <term>Ascend 950PR&950DT系列产品</term>：（A16W4）伪量化场景示例

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def call_pta(x, weight, bias, antiquant_scale, antiquant_offset, group_type, split_item, group_list, group_list_type):
            for i in range(len(weight)):
                weight[i] = weight[i].transpose(-2, -1)
            return torch_npu.npu_grouped_matmul(x, weight, bias=bias, antiquant_scale=antiquant_scale,
                                                antiquant_offset=antiquant_offset, group_list=group_list,
                                                split_item=split_item, group_type=group_type, output_dtype=torch.float16,
                                                group_list_type=group_list_type)

    def main():
        g, m, k, n = 2, 5, 64, 128
        x1 = torch.ones(m, k, dtype=torch.float16)
        x = [x1.npu()]
        weight1 = torch.ones(g, n, k, dtype=torch.int32)
        weight_packed = torch_npu.npu_convert_weight_to_int4pack(weight1.npu())
        weight = [weight_packed.npu()]
        bias1 = torch.ones(g, n, dtype=torch.float16)
        bias = [bias1.npu()]
        antiquant_scale1 = torch.ones(g, n, dtype=torch.float16)
        print(f"antiquant_scale1 shape {antiquant_scale1.shape}")
        antiquant_scale = [antiquant_scale1.npu()]
        antiquant_offset1 = torch.zeros(g, n, dtype=torch.float16)
        antiquant_offset = [antiquant_offset1.npu()]
        group_list = torch.Tensor([2, 3]).to(torch.int64).npu()
        group_list_type = 1
        split_item = 3
        group_type = 0
        custom_output = call_pta(x, weight, bias, antiquant_scale, antiquant_offset, group_type, split_item, group_list,
                              group_list_type)

    if __name__ == '__main__':
        main()
    ```
  <!-- end id76 -->
  <!-- npu="950" id77 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx伪量化场景示例（x为`torch.float8_e4m3fn`，weight使用`torch.uint8`承载`torch_npu.float4_e2m1fn_x2`（1个`torch.uint8`承载两个float4\_e2m1），antiquant\_scale和per\_token\_scale使用`torch.uint8`承载`torch_npu.float8_e8m0fnu`）

    ```python
    import torch
    import torch_npu

    MX_GROUP_SIZE = 32

    class NetPTA(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, weight, per_token_scale, antiquant_scale, bias, group_list, group_list_type,
                    output_dtype=torch.bfloat16):
            weight = weight.transpose(-2, -1)
            antiquant_scale = antiquant_scale.transpose(-3, -2)
            bias = [bias] if bias is not None else None
            output = torch_npu.npu_grouped_matmul([x], [weight], bias=bias, group_list=group_list, group_type=0,
                                                  group_list_type=group_list_type, per_token_scale=[per_token_scale],
                                                  antiquant_scale=[antiquant_scale], split_item=2,
                                                  output_dtype=output_dtype, per_token_scale_dtype=torch_npu.float8_e8m0fnu, weight_dtype=torch_npu.float4_e2m1fn_x2)
            return output

    def main():
        g, m, k, n = 4, 16, 512, 128
        output_dtype = torch.bfloat16
        x = torch.ones((m, k)).to(torch.float8_e4m3fn).npu()
        weight = torch.randint(0, 256, (g, n, k//2), dtype=torch.uint8).npu()
        weight = torch_npu.npu_format_cast(weight, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2)
        per_token_scale = torch.randint(124, 130, (m, k // MX_GROUP_SIZE // 2 , 2)).to(torch.uint8).npu()
        antiquant_scale = torch.randint(124, 130, (g, n, k // MX_GROUP_SIZE // 2, 2), dtype=torch.uint8).npu()
        bias = torch.randint(-5, 5, (g, n)).to(output_dtype).npu()
        group_list = torch.Tensor([2, 4, 6, 16]).to(torch.int64).npu()
        group_list_type = 0
        model = NetPTA().npu()
        out = model(x, weight, per_token_scale, antiquant_scale, bias, group_list, group_list_type, output_dtype)
        print("output")
        print(out[0].cpu())

    if __name__ == '__main__':
        main()
    ```
  <!-- end id77 -->
  <!-- npu="950" id78 -->
  - <term>Ascend 950PR&950DT系列产品</term>：pergroup伪量化场景示例（x为`torch.float16`/`torch.bfloat16`，weight使用`torch.int32`承载`torch_npu.int4`，即1个`torch.int32`承载8个`torch_npu.int4`）

    ```python
    import torch
    import torch_npu

    E, M, N, K, group_size = 2, 32, 128, 256, 32
    group_list = [M // E] * E
    xdtype = torch.bfloat16
    is_multi = True
    transpose_weight = True

    if __name__ == "__main__":
        assert not (is_multi and not transpose_weight), "multi/multi/multi and non-transpose is not supported"

        x = [torch.rand((group_list[i], K), dtype=xdtype, device="npu") for i in range(E)] if is_multi else \
            [torch.rand((M, K), dtype=xdtype, device="npu")]
        weight = [torch_npu.npu_convert_weight_to_int4pack(torch.randint(-8, 7, (N, K) if transpose_weight else (K, N), dtype=torch.int32, device="npu"))
                    for _ in range(E)] if is_multi else \
                  [torch_npu.npu_convert_weight_to_int4pack(torch.randint(-8, 7, (E, N, K) if transpose_weight else (E, K, N), dtype=torch.int32, device="npu"))]
        weight = [w.transpose(-1, -2) if transpose_weight else w for w in weight]
        bias = [torch.rand((N, ), dtype=xdtype if xdtype == torch.float16 else torch.float32, device="npu") for _ in range(E)] if is_multi else \
                [torch.rand((E, N), dtype=xdtype if xdtype == torch.float16 else torch.float32, device="npu")]
        antiquantScale = [torch.rand((K // group_size, N), dtype=xdtype, device="npu") for _ in range(E)] if is_multi else \
                [torch.rand((E, K // group_size, N), dtype=xdtype, device="npu")]
        antiquantOffset = [torch.rand((K // group_size, N), dtype=xdtype, device="npu") for _ in range(E)] if is_multi else \
                [torch.rand((E, K // group_size, N), dtype=xdtype, device="npu")]

        out = torch_npu.npu_grouped_matmul(x, weight, bias=bias,
                                            antiquant_scale=antiquantScale, antiquant_offset=antiquantOffset,
                                            group_list=None if is_multi else torch.Tensor(group_list).to(torch.int64).npu(),
                                            split_item=0 if is_multi else 2,
                                            group_type=-1 if is_multi else 0,
                                            group_list_type=1)
        print([o.shape for o in out])
    ```

  <!-- end id78 -->
  <!-- npu="950" id79 -->
  - <term>Ascend 950PR&950DT系列产品</term>：S8S4 K-C伪量化示例

    ```python
    import os
    import numpy as np
    import torch
    import torch_npu

    def encode_a8w4_scale(scale_fp32: torch.Tensor) -> torch.Tensor:
        """Encode FP32 scales as the UINT64 carrier required by A8W4."""
        scale_np = scale_fp32.cpu().contiguous().numpy().astype(np.float32)
        scale_bits = scale_np.view(np.uint32)
        scale_bits &= np.uint32(0xFFFFE000)
        encoded = scale_bits.astype(np.uint64)
        encoded |= np.uint64(1 << 46)
        return torch.from_numpy(encoded.copy())

    def main() -> None:
        device_id = int(os.getenv("ASCEND_DEVICE_ID", "0"))
        torch.npu.set_device(device_id)

        m, n, k, expert_num = 64, 128, 256, 1

        x = torch.randint(-128, 128, (m, k), dtype=torch.int8).npu()

        # INT32 carries logical INT4 values in [-8, 7] before NPU packing.
        weight_i32 = torch.randint(
            -8, 8, (expert_num, k, n), dtype=torch.int32
        ).npu()
        weight_i4 = torch_npu.npu_convert_weight_to_int4pack(weight_i32)

        # A non-empty offset selects Per-channel mode. Both tensors are [E, 1, N].
        scale_fp32 = torch.rand(expert_num, 1, n) * 0.01 + 0.001
        scale = encode_a8w4_scale(scale_fp32).npu()
        offset = torch.zeros((expert_num, 1, n), dtype=torch.float32).npu()
        per_token_scale = (torch.rand(m, dtype=torch.float32) * 0.01 + 0.001).npu()
        bias = torch.zeros((expert_num, n), dtype=torch.float32).npu()
        group_list = torch.tensor([m], dtype=torch.int64).npu()

        output = torch_npu.npu_grouped_matmul(
            [x],
            [weight_i4],
            bias=[bias],
            scale=[scale],
            offset=[offset],
            antiquant_scale=None,
            antiquant_offset=None,
            per_token_scale=[per_token_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
            act_type=0,
            tuning_config=[0],
            # Asymmetric S8S4 K-C (offset is non-empty) requires FP16 output.
            output_dtype=torch.float16,
        )[0]

        torch.npu.synchronize()
    ```
  <!-- end id79 -->
  <!-- npu="950" id80 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化`torch_npu.int4`，perchannel示例

    ```python
    import numpy as np
    import torch
    import torch_npu

    # npu_quantize 打 int4 包时需要开启内部格式 (allow_internal_format)
    torch_npu.npu.config.allow_internal_format = True


    def main():
        m, k, n, e = 2048, 1024, 1024, 16
        out_dtype = torch.bfloat16
        npu = "npu"
        print(f"[run] perchannel  shape M={m} K={k} N={n} E={e}  out={out_dtype}")

        x_int = torch.full((m, k), 1, dtype=torch.int8)
        w_int = torch.full((e, k, n), 1, dtype=torch.int8)
        scale_ref = torch.full((e, 1, n), 0.1, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        pt_scale = torch.full((m,), 1.0, dtype=torch.float32)
        group_list = torch.arange(1, e + 1) * (m // e)

        # ---- NPU 侧: int4 量化打包 (ND, 非转置) ----
        x_quant = torch_npu.npu_quantize(
            x_int.to(torch.float32).to(npu),
            torch.tensor([1.], device=npu), None, torch.quint4x2, -1, False)
        weight_quant = torch_npu.npu_quantize(
            w_int.to(torch.float32).to(npu),
            torch.tensor([1.], device=npu), None, torch.quint4x2, -1, False)

        # ---- scale 打包: (E,1,N) fp32 比特位放入 int64 低 32 位, 高 32 位=0 ----
        scale_arr = scale_ref.cpu().numpy().astype(np.float32)
        scale_arr.dtype = np.uint32                               # 比特重解释, 不改数值
        packed = np.zeros((e, 1, n * 2), dtype=np.uint32)
        packed[..., ::2] = scale_arr
        packed.dtype = np.int64
        scale_i64 = torch.from_numpy(packed).to(npu)

        y = torch_npu.npu_grouped_matmul(
            [x_quant], [weight_quant],
            bias=None,
            scale=[scale_i64],
            offset=None,
            antiquant_scale=None,
            antiquant_offset=None,
            per_token_scale=[pt_scale.to(npu)],
            group_list=group_list.to(npu),
            split_item=3,
            group_type=0,
            group_list_type=0,
            act_type=0,
            output_dtype=out_dtype,
        )[0]
        torch_npu.npu.synchronize()

    if __name__ == "__main__":
        main()
    ```
  <!-- end id80 -->
  <!-- npu="950" id81 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化`torch.int8`，动态K-C量化支持激活示例

    ```python
    import torch
    import torch_npu
    x = torch.randint(-1, 1, (4, 4), dtype=torch.int8).npu()
    weight1 = torch.randint(-1, 1, (2, 4, 5), dtype=torch.int8).npu()
    scale = torch.randint(-1, 1, (2, 5), dtype=torch.float32).npu()
    pertoken = torch.randint(-1, 1, (4,), dtype=torch.float32).npu()
    group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
    split_item = 3
    bias = torch.randint(-1, 1, (2, 5), dtype=torch.int32).npu()
    npu_out = torch_npu.npu_grouped_matmul([x], [weight1], scale=[scale], bias = [bias],    per_token_scale = [pertoken],group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, act_type=1)
    print("npu_out[0].shape: ", npu_out[0].shape)
    ```
  <!-- end id81 -->
  <!-- npu="950" id82 -->
  - <term>Ascend 950PR&950DT系列产品</term>：伪量化MXA8W4, 单多单场景

    ```python
    import torch
    import torch_npu
    MX_GROUP_SIZE = 32

    def test():
        # 1. 基础参数
        g, m, k, n = 4, 16, 512, 128
        output_dtype = torch.bfloat16
        # 2. 生成 Input x: (m, k)
        x_list = [torch.ones((m, k)).to(torch.float8_e4m3fn).npu()]
        # 3. 生成 group_list
        m_per_group = m // g
        group_list = torch.tensor([m_per_group] * g, dtype=torch.int64).npu()
        group_list_type = 1  # group_list中数值为分组轴上每组大小。
        # 4. 生成参数
        weight_list = []
        antiquant_scale_list = []
        bias_list = []
        for i in range(g):
            # 单组权重形状：(n, k)
            w = torch.randint(0, 256, (n, k//2), dtype=torch.uint8).npu()
            w_cast = torch_npu.npu_format_cast(w, 29,
                                                customize_dtype=torch.float8_e4m3fn,
                                                input_dtype=torch_npu.float4_e2m1fn_x2).t()
            weight_list.append(w_cast)
            # 假设 scale 也是按组或全局，这里生成列表
            s_k = k // MX_GROUP_SIZE // 2
            s = torch.randint(124, 130, (n, s_k, 2), dtype=torch.uint8).npu().permute(1, 0, 2)
            antiquant_scale_list.append(s)
            # Bias 列表 (g 个 (n))
            b = torch.randint(-5, 5, (n,), dtype=output_dtype).npu()
            bias_list.append(b)
        # 5. 准备 per_token_scale
        per_token_scale_list = [torch.randint(124, 130, (m, k // MX_GROUP_SIZE // 2, 2), dtype=torch.uint8).npu()]
        # 6. 执行
        out_list = torch_npu.npu_grouped_matmul(
            x_list,
            weight_list,
            bias=bias_list,
            group_list=group_list,
            group_type=0,
            group_list_type=group_list_type,
            per_token_scale=per_token_scale_list,
            antiquant_scale=antiquant_scale_list,
            split_item=2,
            output_dtype=output_dtype,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            weight_dtype=torch_npu.float4_e2m1fn_x2
        )
    if __name__ == '__main__':
        test()
    ```
  <!-- end id82 -->

- 图模式调用

  - 非量化场景示例：

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    M = 256
    N = 256
    K = 256
    g = 2
    class GMMModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, group_list, split_item):
            return torch_npu.npu_grouped_matmul(x, weight, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.float16, group_list_type=0)

    def main():
        x1 = torch.ones(M, K, dtype=torch.float16)
        x = [x1.npu()]
        weight1 = torch.ones(g, K, N, dtype=torch.float16)
        weight = [weight1.npu()]
        group_list = torch.Tensor([128, 256]).to(torch.int64).npu()
        split_item = 3
        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        custom_output = model(x, weight, group_list, split_item)
        print(custom_output[0].cpu())

    if __name__ == '__main__':
        main()
    ```

  <!-- npu="950" id83 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化场景示例

      ```python
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"
      M = 4
      N = 5
      K = 4
      g = 2
      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, scale, group_list, split_item, pertoken):
              return torch_npu.npu_grouped_matmul(x, weight, scale=scale, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, per_token_scale=pertoken)

      def main():
          x1 = torch.ones(M, K, dtype=torch.int8)
          x = [x1.npu()]
          weight1 = torch.ones(g, K, N, dtype=torch.int8)
          weight = [weight1.npu()]
          scale1 = torch.ones(g, N, dtype=torch.bfloat16).npu()
          scale = [scale1]
          pertoken1 = torch.ones(M, dtype=torch.float32).npu()
          pertoken = [pertoken1]
          group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
          split_item = 3

          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight, scale, group_list, split_item, pertoken)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id83 -->
  <!-- npu="950" id84 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化weightNz场景示例（x/weight为`torch.int8`）

      ```python
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os
      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"
      M = 4
      N = 5
      K = 4
      g = 2

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, scale, group_list, split_item, pertoken):
              return torch_npu.npu_grouped_matmul(x, [weight.transpose(1,2)], scale=scale, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, per_token_scale=pertoken)
      def main():
          x1 = torch.ones(M, K, dtype=torch.int8)
          x = [x1.npu()]

          weight3 = torch.ones(g, K, N, dtype=torch.int8).npu().transpose(1,2)
          weight = torch_npu.npu_format_cast(weight3, 29) # 将weight转为nz格式

          scale1 = torch.ones(g, N, dtype=torch.bfloat16).npu()
          scale = [scale1]
          pertoken1 = torch.ones(M, dtype=torch.float32).npu()
          pertoken = [pertoken1]
          group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
          split_item = 3
          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight, scale, group_list, split_item, pertoken)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id84 -->
  <!-- npu="950" id85 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx伪量化场景示例（x为`torch.float16`，weight为`torch.float32`（`torch_npu.float4_e2m1fn_x2`），antiquant\_scale使用`torch.uint8`承载`torch_npu.float8_e8m0fnu`）

      ```python
      import numpy as np
      from ml_dtypes import float4_e2m1fn
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      m = 1
      k = 64
      n = 64
      g = 2
      E2M1_MIN, E2M1_MAX = -6, 6
      antiquant_group_size = 32
      trans_weight = False

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, antiquant_scale, group_list, split_item):
              return torch_npu.npu_grouped_matmul(x, weight, antiquant_scale=antiquant_scale,
                                                  group_list=group_list,
                                                  split_item=split_item,
                                                  group_type=0, output_dtype=torch.float16, group_list_type=1)

      def main():
          x1 = torch.ones(m, k, dtype=torch.float16)
          x = [x1.npu()]
          weight1 = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(g * k * n).reshape((g, k, n))).astype(float4_e2m1fn)
          weight1 = torch.from_numpy(weight1.astype(np.float32))
          npu_weight = torch_npu.npu_format_cast(weight1.npu(), 29, customize_dtype=torch.float16)
          weight_packed = torch_npu.npu_convert_weight_to_int4pack(npu_weight)
          weight = [weight_packed.npu()]
          antiquant_scale1 = torch.randint(127, 128, (g, k // antiquant_group_size, n), dtype=torch.uint8)
          print(antiquant_scale1)
          antiquant_scale = [antiquant_scale1.npu()]
          group_list = torch.Tensor([0, 1]).to(torch.int64).npu()
          split_item = 3

          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=True)
          custom_output = model(x, weight, antiquant_scale, group_list, split_item)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id85 -->
  <!-- npu="950" id86 -->
  - <term>Ascend 950PR&950DT系列产品</term>：G-B动态量化场景示例

      ```python
      import math
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"
      M = 576
      N = 7168
      K = 512
      g = 4

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight1, scale1, group_list, split_item, pertoken):
              weight1 = weight1.transpose(1,2)
              scale1 = scale1.transpose(1,2)
              scale = [scale1]
              weight = [weight1]
              return torch_npu.npu_grouped_matmul(x, weight, scale=scale, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, per_token_scale=pertoken)

      def main():
          x1 = torch.ones(M, K, dtype=torch.int8).to(torch.float8_e4m3fn)
          x = [x1.npu()]
          weight1 = torch.ones(g, N, K, dtype=torch.int8).to(torch.float8_e4m3fn).npu()
          scale1 = torch.randint(-1, 1, (g, math.ceil(N/128), math.ceil(K/128)), dtype=torch.float32).npu()
          pertoken1 = torch.randint(-1, 1, (M, math.ceil(K/128)), dtype=torch.float32).npu()
          pertoken = [pertoken1]
          group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
          split_item = 3

          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight1, scale1, group_list, split_item, pertoken)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id86 -->
  <!-- npu="950" id87 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx（A4W4）全量化场景示例-mxfp4

      ```python
      import math
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, scale, per_token_scale, group_list, split_item):
              return torch_npu.npu_grouped_matmul([x], [weight], scale=[scale], per_token_scale = [per_token_scale], group_list=group_list,
        split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch_npu.float4_e2m1fn_x2, weight_dtype=torch_npu.float4_e2m1fn_x2,
        scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)

      def main():
          M = 64
          N = 128
          K = 128
          g = 1
          x = torch.randint(1, 3, (M, int(K/2)), dtype=torch.uint8).npu()
          weight = torch.randint(1, 3, (g, K, int(N/2)), dtype=torch.uint8).npu()
          scale = torch.randint(-1, 1, (g, math.ceil(K/64),N,2), dtype=torch.int8).npu()
          per_token_scale = torch.randint(-1, 1, (M,math.ceil(K/64),2), dtype=torch.int8).npu()
          group_list = torch.Tensor([1]).to(torch.int64).npu()
          split_item = 2
          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight, scale, per_token_scale, group_list, split_item)

      if __name__ == '__main__':
          main()
      ```
  <!-- end id87 -->
