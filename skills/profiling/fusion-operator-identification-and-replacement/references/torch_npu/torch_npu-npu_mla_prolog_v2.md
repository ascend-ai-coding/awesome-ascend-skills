# torch_npu.npu_mla_prolog_v2

> [!NOTICE]  
> 该接口中kv_cache和kr_cache进行原地计算，未按in-place算子实现接口，推荐使用`torch_npu.npu_mla_prolog_v3`接口进行替换。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：推理场景下，Multi-Head Latent Attention（MLA）前处理的计算操作。主要计算过程分为五路：
    - 首先对输入x乘以W<sup>DQ</sup>进行下采样和RmsNorm后分成两路，第一路乘以W<sup>UQ</sup>和W<sup>UK</sup>经过两次上采样后得到q<sup>N</sup>；
    - 第二路乘以W<sup>QR</sup>后经过旋转位置编码（ROPE）得到q<sup>R</sup>。
    - 第三路是输入x乘以W<sup>DKV</sup>进行下采样和RmsNorm后传入Cache中得到k<sup>C</sup>。
    - 第四路是输入x乘以W<sup>KR</sup>后经过旋转位置编码后传入另一个Cache中得到k<sup>R</sup>。
    - 第五路是输出q<sup>N</sup>经过DynamicQuant后得到的量化参数。

- 计算公式：
    - RmsNorm公式
        $$RmsNorm(x) = \gamma \cdot \frac{x} {\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}}$$

    - Query计算公式
        $$c^Q = RmsNorm(x \cdot W^{DQ})$$
        $$q^C = c^Q \cdot W^{UQ}$$
        $$q^N = q^C \cdot W^{UK}$$

    - Query ROPE旋转位置编码

        $$q^R = ROPE(c^Q \cdot W^{QR})$$

    - Key计算公式

        $$k^C = Cache(RmsNorm(x \cdot W^{DKV}))$$

    - Key ROPE旋转位置编码

        $$k^R = Cache(ROPE(x \cdot W^{KR}))$$

    - 反量化缩放因子（Dequant Scale Query Nope）计算公式
        $$\text{dequantScaleQNope} = \frac{\text{RowMax}(\text{abs}(q^N))}{127}$$
        $$q^N = \text{round}(\frac{q^N}{\text{dequantScaleQNope}})$$

## 函数原型

```python
torch_npu.npu_mla_prolog_v2(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, *, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None, quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None, rmsnorm_epsilon_cq=1e-05, rmsnorm_epsilon_ckv=1e-05, cache_mode="PA_BSND") -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **token\_x**（`Tensor`）：必选参数，对应公式中x。shape支持2维和3维，格式为\(T, He\)和\(B, S, He\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。
- **weight\_dq**（`Tensor`）：必选参数，表示计算Query的下采样权重矩阵，即公式中W<sup>DQ</sup>。shape支持2维，格式为\(He, Hcq\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持FRACTAL\_NZ（可通过`torch_npu.npu_format_cast`将ND格式转为FRACTAL\_NZ格式）。
- **weight\_uq\_qr**（`Tensor`）：必选参数，表示计算Query的上采样权重矩阵和Query的位置编码权重矩阵，即公式中W<sup>UQ</sup>和W<sup>QR</sup>。shape支持2维，格式为\(Hcq, N\*\(D+Dr\)\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持FRACTAL\_NZ。
- **weight\_uk**（`Tensor`）：必选参数，表示计算Key的上采样权重，即公式中W<sup>UK</sup>。shape支持3维，格式为\(N, D, Hckv\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **weight\_dkv\_kr**（`Tensor`）：必选参数，表示计算Key的下采样权重矩阵和Key的位置编码权重矩阵，即公式中W<sup>DKV</sup>和W<sup>KR</sup>。shape支持2维，格式为\(He, Hckv+Dr\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持FRACTAL\_NZ。
- **rmsnorm\_gamma\_cq**（`Tensor`）：必选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为\(Hcq\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **rmsnorm\_gamma\_ckv**（`Tensor`）：必选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为\(Hckv\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **rope\_sin**（`Tensor`）：必选参数，表示用于计算旋转位置编码的正弦参数矩阵。shape支持2维和3维，格式为\(T, Dr\)和\(B, S, Dr\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **rope\_cos**（`Tensor`）：必选参数，表示用于计算旋转位置编码的余弦参数矩阵。shape支持2维和3维，格式为\(T, Dr\)和\(B, S, Dr\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **cache\_index**（`Tensor`）：必选参数，表示写入`kv_cache`、`kr_cache`的页式线性槽位索引。shape支持1维和2维，格式为\(T\)和\(B, S\)，dtype支持`torch.int64`，数据格式支持ND。每个元素取值范围为[0, L)，其中 **L** 为`kv_cache`第0维与第1维长度之积（须与`kr_cache`前两维之积一致）。
- **kv\_cache**（`Tensor`）：必选参数，表示用于cache索引的aclTensor。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Hckv\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。
- **kr\_cache**（`Tensor`）：必选参数，表示用于key位置编码的cache。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Dr\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **dequant\_scale\_x**（`Tensor`）：可选参数，输入`token_x`为`torch.int8`类型时下采样后进行反量化操作时的参数，`token_x`量化方式为pertoken。其shape支持2维，格式为\(T, 1\)和\(BS, 1\)，dtype支持`torch.float`，数据格式支持ND。
- **dequant\_scale\_w\_dq**（`Tensor`）：可选参数，输入`weight_dq`为`torch.int8`类型时下采样后进行反量化操作时的参数，`weight_dq`量化方式为perchannel。其shape支持2维，格式为\(1, Hcq\)，dtype支持`torch.float`，数据格式支持ND。
- **dequant\_scale\_w\_uq\_qr**（`Tensor`）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化方式为perchannel。shape支持2维，格式为\(1, N\*\(D+Dr\)\)，dtype支持`torch.float`，数据格式支持ND。
- **dequant\_scale\_w\_dkv\_kr**（`Tensor`）：可选参数，输入`weight_dkv_kr`为`torch.int8`类型时矩阵乘后进行反量化操作时的参数，`weight_dkv_kr`量化方式为perchannel。其shape支持2维，格式为\(1, Hckv+Dr\)，dtype支持`torch.float`，数据格式支持ND。
- **quant\_scale\_ckv**（`Tensor`）：可选参数，用于对输出到`kv_cache_out`中的数据做量化操作时的参数。shape支持2维，格式为\(1, Hckv\)，dtype支持`torch.float`，数据格式支持ND。
- **quant\_scale\_ckr**（`Tensor`）：可选参数，用于对输出到`kr_cache_out`中的数据做量化操作时的参数。shape支持2维，格式为\(1, Dr\)，dtype支持`torch.float`，数据格式支持ND。
- **smooth\_scales\_cq**（`Tensor`）：可选参数，用于对RmsNormCq输出做动态量化操作时的参数。shape支持2维，格式为\(1, Hcq\)，dtype支持`torch.float`，数据格式支持ND。
- **rmsnorm\_epsilon\_cq**（`float`）：可选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的ε参数，默认值为1e-05。
- **rmsnorm\_epsilon\_ckv**（`float`）：可选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的ε参数，默认值为1e-05。
- **cache\_mode**（`str`）：可选参数，表示`kv_cache`的模式，支持“PA\_BSND”、“PA\_NZ”，默认值为“PA\_BSND”。

## 返回值说明

- **query**（`Tensor`）：表示Query的输出Tensor，即公式中q<sup>N</sup>。shape支持3维和4维，格式为\(T, N, Hckv\)和\(B, S, N, Hckv\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。
- **query\_rope**（`Tensor`）：表示Query位置编码的输出Tensor，即公式中q<sup>R</sup>。shape支持3维和4维，格式为\(T, N, Dr\)和\(B, S, N, Dr\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **kv\_cache\_out**（`Tensor`）：表示Key输出到`kv_cache`中的Tensor（本质in-place更新），即公式中k<sup>C</sup>。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Hckv\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。
- **kr\_cache\_out**（`Tensor`）：表示Key的位置编码输出到`kr_cache`中的Tensor（本质in-place更新），即公式中k<sup>R</sup>。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Dr\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。
- **dequant\_scale\_q\_nope**（`Tensor`）：表示Query的输出Tensor的反量化参数。其shape支持1维和3维，全量化kv\_cache量化场景下，其shape为\(T, N, 1\)和\(B\*S, N, 1\)；其他场景下，其shape为\(1\)，dtype支持`torch.float`，数据格式支持ND。

## 约束说明

- 该接口支持在推理场景下使用。
- 该接口支持图模式。
- 接口参数中shape格式字段含义：
    - B：Batch表示输入样本批量大小，取值范围为0\~65536。
    - S：Seq-Length表示输入样本序列长度，取值范围为0\~16。
    - He：Head-Size表示隐藏层的大小，取值为7168。

    - Hcq：q低秩矩阵维度，取值为1536。
    - N：Head-Num表示多头数，取值范围为1、2、4、8、16、32、64、128。

    - Hckv：kv低秩矩阵维度，取值为512。
    - D：qk不含位置编码维度，取值为128。
    - Dr：qk位置编码维度，取值为64。
    - Nkv：kv的head数，取值为1。
    - BlockNum：PagedAttention场景下的块数，即`kv_cache`、`kr_cache`的第0维长度。已知Batch为B、kv序列长度为Skv、块大小为BlockSize时，通常应满足：BlockNum ≥ ⌈B×Skv / BlockSize⌉（Skv允许取0）。
    - BlockSize：PagedAttention场景下的块大小，取值范围为16、128。
    - T：BS合轴后的大小，取值范围：0\~1048576。

- shape约束：
    - 若`token_x`的维度采用BS合轴，即\(T, He\)，则`rope_sin`和rope\_cos的shape为\(T, Dr\)，cache\_index的shape为\(T,\)，dequant\_scale\_x的shape为\(T, 1\)，query的shape为\(T, N, Hckv\)，query\_rope的shape为\(T, N, Dr\)。全量化kv\_cache量化场景下，dequant\_scale\_q\_nope的shape为\(T, N, 1\)，其他场景下dequant\_scale\_q\_nope的shape为\(1\)。
    - 若`token_x`的维度不采用BS合轴，即\(B, S, He\)，则`rope_sin`和rope\_cos的shape为\(B, S, Dr\)，cache\_index的shape为\(B, S\)，dequant\_scale\_x的shape为\(B\*S, 1\)，query的shape为\(B, S, N, Hckv\)，query\_rope的shape为\(B, S, N, Dr\)。全量化kv\_cache量化场景下，dequant\_scale\_q\_nope的shape为\(B\*S, N, 1\)，其他场景下dequant\_scale\_q\_nope的shape为\(1\)。
    - B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        - 如果B、S、T取值为0，则query、query_rope、dequant_scale_q_nope输出空Tensor，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新。
        - 如果Skv取值为0，则query、query_rope、dequant_scale_q_nope正常计算，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新，即输出空Tensor。

- 本算子支持以下场景：
    <a name="zh-cn_topic_0000002313328922_table664817810310"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000002313328922_row9649788313"><th class="cellrowborder" colspan="2" valign="top" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002313328922_p14649381739"><a name="zh-cn_topic_0000002313328922_p14649381739"></a><a name="zh-cn_topic_0000002313328922_p14649381739"></a>场景</p>
    </th>
    <th class="cellrowborder" valign="top" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002313328922_p1649781312"><a name="zh-cn_topic_0000002313328922_p1649781312"></a><a name="zh-cn_topic_0000002313328922_p1649781312"></a>含义</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000002313328922_row36491488316"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002313328922_p16649987312"><a name="zh-cn_topic_0000002313328922_p16649987312"></a><a name="zh-cn_topic_0000002313328922_p16649987312"></a>非量化</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002313328922_p96491986311"><a name="zh-cn_topic_0000002313328922_p96491986311"></a><a name="zh-cn_topic_0000002313328922_p96491986311"></a>算子所有入参全传入非量化数据，出参全返回非量化数据。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row230913715311"><td class="cellrowborder" rowspan="2" valign="top" width="8.780000000000001%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002313328922_p330973715314"><a name="zh-cn_topic_0000002313328922_p330973715314"></a><a name="zh-cn_topic_0000002313328922_p330973715314"></a>部分量化</p>
    </td>
    <td class="cellrowborder" valign="top" width="18.18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002313328922_p9145842144313"><a name="zh-cn_topic_0000002313328922_p9145842144313"></a><a name="zh-cn_topic_0000002313328922_p9145842144313"></a>kv_cache非量化</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.04%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002313328922_p18277152811188"><a name="zh-cn_topic_0000002313328922_p18277152811188"></a><a name="zh-cn_topic_0000002313328922_p18277152811188"></a>入参：weight_uq_qr传入pertoken量化数据，其他入参传入非量化数据。</p>
    <p id="zh-cn_topic_0000002313328922_p8284162318619"><a name="zh-cn_topic_0000002313328922_p8284162318619"></a><a name="zh-cn_topic_0000002313328922_p8284162318619"></a>出参：全返回非量化数据。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row6013117434"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002313328922_p714610427438"><a name="zh-cn_topic_0000002313328922_p714610427438"></a><a name="zh-cn_topic_0000002313328922_p714610427438"></a>kv_cache量化</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002313328922_p4752029876"><a name="zh-cn_topic_0000002313328922_p4752029876"></a><a name="zh-cn_topic_0000002313328922_p4752029876"></a>入参：weight_uq_qr传入pertoken量化数据，kv_cache、kr_cache传入perchannel量化数据，其他入参全传入非量化数据。</p>
    <p id="zh-cn_topic_0000002313328922_p4262738101812"><a name="zh-cn_topic_0000002313328922_p4262738101812"></a><a name="zh-cn_topic_0000002313328922_p4262738101812"></a>出参：kv_cache_out、kr_cache_out返回perchannel量化数据，其他出参返回非量化数据。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row3649881538"><td class="cellrowborder" rowspan="2" valign="top" width="8.780000000000001%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002313328922_p1764928538"><a name="zh-cn_topic_0000002313328922_p1764928538"></a><a name="zh-cn_topic_0000002313328922_p1764928538"></a>全量化</p>
    </td>
    <td class="cellrowborder" valign="top" width="18.18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002313328922_p71465427433"><a name="zh-cn_topic_0000002313328922_p71465427433"></a><a name="zh-cn_topic_0000002313328922_p71465427433"></a>kv_cache非量化</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.04%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002313328922_p20649887310"><a name="zh-cn_topic_0000002313328922_p20649887310"></a><a name="zh-cn_topic_0000002313328922_p20649887310"></a>入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，其他入参传入非量化数据。</p>
    <p id="zh-cn_topic_0000002313328922_p940213169195"><a name="zh-cn_topic_0000002313328922_p940213169195"></a><a name="zh-cn_topic_0000002313328922_p940213169195"></a>出参：全返回非量化数据。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row576033534319"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002313328922_p1714664220437"><a name="zh-cn_topic_0000002313328922_p1714664220437"></a><a name="zh-cn_topic_0000002313328922_p1714664220437"></a>kv_cache量化</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002313328922_p14760153518431"><a name="zh-cn_topic_0000002313328922_p14760153518431"></a><a name="zh-cn_topic_0000002313328922_p14760153518431"></a>入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，kv_cache传入pertensor量化数据，其他入参传入非量化数据。</p>
    <p id="zh-cn_topic_0000002313328922_p1280815811110"><a name="zh-cn_topic_0000002313328922_p1280815811110"></a><a name="zh-cn_topic_0000002313328922_p1280815811110"></a>出参：query返回pertoken_head动态量化数据、kv_cache_out返回pertensor量化数据，其他出参返回非量化数据。</p>
    </td>
    </tr>
    </tbody>
    </table>

- 在不同量化场景下，参数的dtype和shape组合需满足如下条件：

    <a name="zh-cn_topic_0000002313328922_table1311951423117"></a>
    <table><tbody><tr id="zh-cn_topic_0000002313328922_row510181463115"><td class="cellrowborder" rowspan="3" valign="top"><p id="zh-cn_topic_0000002313328922_p21013144313"><a name="zh-cn_topic_0000002313328922_p21013144313"></a><a name="zh-cn_topic_0000002313328922_p21013144313"></a><strong id="zh-cn_topic_0000002313328922_b1423515521358"><a name="zh-cn_topic_0000002313328922_b1423515521358"></a><a name="zh-cn_topic_0000002313328922_b1423515521358"></a>参数名</strong></p>
    </td>
    <td class="cellrowborder" rowspan="2" colspan="2" valign="top"><p id="zh-cn_topic_0000002313328922_p201012147314"><a name="zh-cn_topic_0000002313328922_p201012147314"></a><a name="zh-cn_topic_0000002313328922_p201012147314"></a><strong id="zh-cn_topic_0000002313328922_b1824916521457"><a name="zh-cn_topic_0000002313328922_b1824916521457"></a><a name="zh-cn_topic_0000002313328922_b1824916521457"></a>非量化场景</strong></p>
    </td>
    <td class="cellrowborder" colspan="4" valign="top"><p id="zh-cn_topic_0000002313328922_p1810121413112"><a name="zh-cn_topic_0000002313328922_p1810121413112"></a><a name="zh-cn_topic_0000002313328922_p1810121413112"></a><strong id="zh-cn_topic_0000002313328922_b122631152959"><a name="zh-cn_topic_0000002313328922_b122631152959"></a><a name="zh-cn_topic_0000002313328922_b122631152959"></a>部分量化场景</strong></p>
    </td>
    <td class="cellrowborder" colspan="4" valign="top"><p id="zh-cn_topic_0000002313328922_p10101131423113"><a name="zh-cn_topic_0000002313328922_p10101131423113"></a><a name="zh-cn_topic_0000002313328922_p10101131423113"></a><strong id="zh-cn_topic_0000002313328922_b172646522059"><a name="zh-cn_topic_0000002313328922_b172646522059"></a><a name="zh-cn_topic_0000002313328922_b172646522059"></a>全量化场景</strong></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row6101181419318"><td class="cellrowborder" colspan="2" valign="top"><p id="zh-cn_topic_0000002313328922_p1101101414315"><a name="zh-cn_topic_0000002313328922_p1101101414315"></a><a name="zh-cn_topic_0000002313328922_p1101101414315"></a><strong id="zh-cn_topic_0000002313328922_b42651521553"><a name="zh-cn_topic_0000002313328922_b42651521553"></a><a name="zh-cn_topic_0000002313328922_b42651521553"></a>kv_cache非量化</strong></p>
    </td>
    <td class="cellrowborder" colspan="2" valign="top"><p id="zh-cn_topic_0000002313328922_p01012143316"><a name="zh-cn_topic_0000002313328922_p01012143316"></a><a name="zh-cn_topic_0000002313328922_p01012143316"></a><strong id="zh-cn_topic_0000002313328922_b1226655212514"><a name="zh-cn_topic_0000002313328922_b1226655212514"></a><a name="zh-cn_topic_0000002313328922_b1226655212514"></a>kv_cache量化</strong></p>
    </td>
    <td class="cellrowborder" colspan="2" valign="top"><p id="zh-cn_topic_0000002313328922_p3101101493112"><a name="zh-cn_topic_0000002313328922_p3101101493112"></a><a name="zh-cn_topic_0000002313328922_p3101101493112"></a><strong id="zh-cn_topic_0000002313328922_b1526718527511"><a name="zh-cn_topic_0000002313328922_b1526718527511"></a><a name="zh-cn_topic_0000002313328922_b1526718527511"></a>kv_cache非量化</strong></p>
    </td>
    <td class="cellrowborder" colspan="2" valign="top"><p id="zh-cn_topic_0000002313328922_p41011614203115"><a name="zh-cn_topic_0000002313328922_p41011614203115"></a><a name="zh-cn_topic_0000002313328922_p41011614203115"></a><strong id="zh-cn_topic_0000002313328922_b826811521456"><a name="zh-cn_topic_0000002313328922_b826811521456"></a><a name="zh-cn_topic_0000002313328922_b826811521456"></a>kv_cache量化</strong></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row210212145314"><td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p71014149318"><a name="zh-cn_topic_0000002313328922_p71014149318"></a><a name="zh-cn_topic_0000002313328922_p71014149318"></a><strong id="zh-cn_topic_0000002313328922_b152701452558"><a name="zh-cn_topic_0000002313328922_b152701452558"></a><a name="zh-cn_topic_0000002313328922_b152701452558"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p191011214113115"><a name="zh-cn_topic_0000002313328922_p191011214113115"></a><a name="zh-cn_topic_0000002313328922_p191011214113115"></a><strong id="zh-cn_topic_0000002313328922_b1127115521059"><a name="zh-cn_topic_0000002313328922_b1127115521059"></a><a name="zh-cn_topic_0000002313328922_b1127115521059"></a>shape</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p210171493118"><a name="zh-cn_topic_0000002313328922_p210171493118"></a><a name="zh-cn_topic_0000002313328922_p210171493118"></a><strong id="zh-cn_topic_0000002313328922_b52724521557"><a name="zh-cn_topic_0000002313328922_b52724521557"></a><a name="zh-cn_topic_0000002313328922_b52724521557"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p15101114203119"><a name="zh-cn_topic_0000002313328922_p15101114203119"></a><a name="zh-cn_topic_0000002313328922_p15101114203119"></a><strong id="zh-cn_topic_0000002313328922_b32734521655"><a name="zh-cn_topic_0000002313328922_b32734521655"></a><a name="zh-cn_topic_0000002313328922_b32734521655"></a>shape</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p1010113147310"><a name="zh-cn_topic_0000002313328922_p1010113147310"></a><a name="zh-cn_topic_0000002313328922_p1010113147310"></a><strong id="zh-cn_topic_0000002313328922_b1527420527515"><a name="zh-cn_topic_0000002313328922_b1527420527515"></a><a name="zh-cn_topic_0000002313328922_b1527420527515"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p161011014123117"><a name="zh-cn_topic_0000002313328922_p161011014123117"></a><a name="zh-cn_topic_0000002313328922_p161011014123117"></a><strong id="zh-cn_topic_0000002313328922_b32751752355"><a name="zh-cn_topic_0000002313328922_b32751752355"></a><a name="zh-cn_topic_0000002313328922_b32751752355"></a>shape</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p610216144319"><a name="zh-cn_topic_0000002313328922_p610216144319"></a><a name="zh-cn_topic_0000002313328922_p610216144319"></a><strong id="zh-cn_topic_0000002313328922_b192768521753"><a name="zh-cn_topic_0000002313328922_b192768521753"></a><a name="zh-cn_topic_0000002313328922_b192768521753"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p510211149315"><a name="zh-cn_topic_0000002313328922_p510211149315"></a><a name="zh-cn_topic_0000002313328922_p510211149315"></a><strong id="zh-cn_topic_0000002313328922_b192771952258"><a name="zh-cn_topic_0000002313328922_b192771952258"></a><a name="zh-cn_topic_0000002313328922_b192771952258"></a>shape</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p1410221423116"><a name="zh-cn_topic_0000002313328922_p1410221423116"></a><a name="zh-cn_topic_0000002313328922_p1410221423116"></a><strong id="zh-cn_topic_0000002313328922_b12278105218513"><a name="zh-cn_topic_0000002313328922_b12278105218513"></a><a name="zh-cn_topic_0000002313328922_b12278105218513"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="zh-cn_topic_0000002313328922_p5102814163116"><a name="zh-cn_topic_0000002313328922_p5102814163116"></a><a name="zh-cn_topic_0000002313328922_p5102814163116"></a><strong id="zh-cn_topic_0000002313328922_b132796521952"><a name="zh-cn_topic_0000002313328922_b132796521952"></a><a name="zh-cn_topic_0000002313328922_b132796521952"></a>shape</strong></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row19103514123115"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1310271493114"><a name="zh-cn_topic_0000002313328922_p1310271493114"></a><a name="zh-cn_topic_0000002313328922_p1310271493114"></a>token_x</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1102141414313"><a name="zh-cn_topic_0000002313328922_p1102141414313"></a><a name="zh-cn_topic_0000002313328922_p1102141414313"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="zh-cn_topic_0000002313328922_ul17461162913811"></a><a name="zh-cn_topic_0000002313328922_ul17461162913811"></a><ul id="zh-cn_topic_0000002313328922_ul17461162913811"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p8102514163119"><a name="zh-cn_topic_0000002313328922_p8102514163119"></a><a name="zh-cn_topic_0000002313328922_p8102514163119"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul472585016425"></a><a name="zh-cn_topic_0000002313328922_ul472585016425"></a><ul id="zh-cn_topic_0000002313328922_ul472585016425"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p17102314193112"><a name="zh-cn_topic_0000002313328922_p17102314193112"></a><a name="zh-cn_topic_0000002313328922_p17102314193112"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="zh-cn_topic_0000002313328922_ul53761545381"></a><a name="zh-cn_topic_0000002313328922_ul53761545381"></a><ul id="zh-cn_topic_0000002313328922_ul53761545381"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1210291473118"><a name="zh-cn_topic_0000002313328922_p1210291473118"></a><a name="zh-cn_topic_0000002313328922_p1210291473118"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul1124455434211"></a><a name="zh-cn_topic_0000002313328922_ul1124455434211"></a><ul id="zh-cn_topic_0000002313328922_ul1124455434211"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p010219141312"><a name="zh-cn_topic_0000002313328922_p010219141312"></a><a name="zh-cn_topic_0000002313328922_p010219141312"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="zh-cn_topic_0000002313328922_ul7208162853214"></a><a name="zh-cn_topic_0000002313328922_ul7208162853214"></a><ul id="zh-cn_topic_0000002313328922_ul7208162853214"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row310371413111"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p191032141314"><a name="zh-cn_topic_0000002313328922_p191032141314"></a><a name="zh-cn_topic_0000002313328922_p191032141314"></a>weight_dq</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p31032147310"><a name="zh-cn_topic_0000002313328922_p31032147310"></a><a name="zh-cn_topic_0000002313328922_p31032147310"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p171031714193111"><a name="zh-cn_topic_0000002313328922_p171031714193111"></a><a name="zh-cn_topic_0000002313328922_p171031714193111"></a>(He,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p110361423111"><a name="zh-cn_topic_0000002313328922_p110361423111"></a><a name="zh-cn_topic_0000002313328922_p110361423111"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p101038148313"><a name="zh-cn_topic_0000002313328922_p101038148313"></a><a name="zh-cn_topic_0000002313328922_p101038148313"></a>(He,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p161031714153111"><a name="zh-cn_topic_0000002313328922_p161031714153111"></a><a name="zh-cn_topic_0000002313328922_p161031714153111"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p4103191483114"><a name="zh-cn_topic_0000002313328922_p4103191483114"></a><a name="zh-cn_topic_0000002313328922_p4103191483114"></a>(He,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p201031114103115"><a name="zh-cn_topic_0000002313328922_p201031114103115"></a><a name="zh-cn_topic_0000002313328922_p201031114103115"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p15103121413119"><a name="zh-cn_topic_0000002313328922_p15103121413119"></a><a name="zh-cn_topic_0000002313328922_p15103121413119"></a>(He,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p51032014193110"><a name="zh-cn_topic_0000002313328922_p51032014193110"></a><a name="zh-cn_topic_0000002313328922_p51032014193110"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p51031414143117"><a name="zh-cn_topic_0000002313328922_p51031414143117"></a><a name="zh-cn_topic_0000002313328922_p51031414143117"></a>(He,Hcq)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row161042141311"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p910314141313"><a name="zh-cn_topic_0000002313328922_p910314141313"></a><a name="zh-cn_topic_0000002313328922_p910314141313"></a>weight_uq_qr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p19103151418313"><a name="zh-cn_topic_0000002313328922_p19103151418313"></a><a name="zh-cn_topic_0000002313328922_p19103151418313"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1610321416310"><a name="zh-cn_topic_0000002313328922_p1610321416310"></a><a name="zh-cn_topic_0000002313328922_p1610321416310"></a>(Hcq,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p6103131473113"><a name="zh-cn_topic_0000002313328922_p6103131473113"></a><a name="zh-cn_topic_0000002313328922_p6103131473113"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p10104191410311"><a name="zh-cn_topic_0000002313328922_p10104191410311"></a><a name="zh-cn_topic_0000002313328922_p10104191410311"></a>(Hcq,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p9104171413118"><a name="zh-cn_topic_0000002313328922_p9104171413118"></a><a name="zh-cn_topic_0000002313328922_p9104171413118"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p189211613184413"><a name="zh-cn_topic_0000002313328922_p189211613184413"></a><a name="zh-cn_topic_0000002313328922_p189211613184413"></a>(Hcq,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p0104101413314"><a name="zh-cn_topic_0000002313328922_p0104101413314"></a><a name="zh-cn_topic_0000002313328922_p0104101413314"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p17104214133117"><a name="zh-cn_topic_0000002313328922_p17104214133117"></a><a name="zh-cn_topic_0000002313328922_p17104214133117"></a>(Hcq,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p51041914123119"><a name="zh-cn_topic_0000002313328922_p51041914123119"></a><a name="zh-cn_topic_0000002313328922_p51041914123119"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p1910410144316"><a name="zh-cn_topic_0000002313328922_p1910410144316"></a><a name="zh-cn_topic_0000002313328922_p1910410144316"></a>(Hcq,N*(D+Dr))</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row3105131493117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1610421413118"><a name="zh-cn_topic_0000002313328922_p1610421413118"></a><a name="zh-cn_topic_0000002313328922_p1610421413118"></a>weight_uk</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p710491419315"><a name="zh-cn_topic_0000002313328922_p710491419315"></a><a name="zh-cn_topic_0000002313328922_p710491419315"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p14104121453113"><a name="zh-cn_topic_0000002313328922_p14104121453113"></a><a name="zh-cn_topic_0000002313328922_p14104121453113"></a>(N,D,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p41041414163116"><a name="zh-cn_topic_0000002313328922_p41041414163116"></a><a name="zh-cn_topic_0000002313328922_p41041414163116"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p17295153120442"><a name="zh-cn_topic_0000002313328922_p17295153120442"></a><a name="zh-cn_topic_0000002313328922_p17295153120442"></a>(N,D,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p91041114113114"><a name="zh-cn_topic_0000002313328922_p91041114113114"></a><a name="zh-cn_topic_0000002313328922_p91041114113114"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p299953374419"><a name="zh-cn_topic_0000002313328922_p299953374419"></a><a name="zh-cn_topic_0000002313328922_p299953374419"></a>(N,D,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p121041145312"><a name="zh-cn_topic_0000002313328922_p121041145312"></a><a name="zh-cn_topic_0000002313328922_p121041145312"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p25501735154412"><a name="zh-cn_topic_0000002313328922_p25501735154412"></a><a name="zh-cn_topic_0000002313328922_p25501735154412"></a>(N,D,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p410441493120"><a name="zh-cn_topic_0000002313328922_p410441493120"></a><a name="zh-cn_topic_0000002313328922_p410441493120"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p14409163815449"><a name="zh-cn_topic_0000002313328922_p14409163815449"></a><a name="zh-cn_topic_0000002313328922_p14409163815449"></a>(N,D,Hckv)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row510581423111"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p6105414123114"><a name="zh-cn_topic_0000002313328922_p6105414123114"></a><a name="zh-cn_topic_0000002313328922_p6105414123114"></a>weight_dkv_kr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p13105014123115"><a name="zh-cn_topic_0000002313328922_p13105014123115"></a><a name="zh-cn_topic_0000002313328922_p13105014123115"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p41059141311"><a name="zh-cn_topic_0000002313328922_p41059141311"></a><a name="zh-cn_topic_0000002313328922_p41059141311"></a>(He,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p161051514173116"><a name="zh-cn_topic_0000002313328922_p161051514173116"></a><a name="zh-cn_topic_0000002313328922_p161051514173116"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p4135250144412"><a name="zh-cn_topic_0000002313328922_p4135250144412"></a><a name="zh-cn_topic_0000002313328922_p4135250144412"></a>(He,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1910571416319"><a name="zh-cn_topic_0000002313328922_p1910571416319"></a><a name="zh-cn_topic_0000002313328922_p1910571416319"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p9441185294416"><a name="zh-cn_topic_0000002313328922_p9441185294416"></a><a name="zh-cn_topic_0000002313328922_p9441185294416"></a>(He,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p191051614143117"><a name="zh-cn_topic_0000002313328922_p191051614143117"></a><a name="zh-cn_topic_0000002313328922_p191051614143117"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p12951554154420"><a name="zh-cn_topic_0000002313328922_p12951554154420"></a><a name="zh-cn_topic_0000002313328922_p12951554154420"></a>(He,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p121058149314"><a name="zh-cn_topic_0000002313328922_p121058149314"></a><a name="zh-cn_topic_0000002313328922_p121058149314"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p52995561441"><a name="zh-cn_topic_0000002313328922_p52995561441"></a><a name="zh-cn_topic_0000002313328922_p52995561441"></a>(He,Hckv+Dr)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row10106161463117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1510561410319"><a name="zh-cn_topic_0000002313328922_p1510561410319"></a><a name="zh-cn_topic_0000002313328922_p1510561410319"></a>rmsnorm_gamma_cq</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p201051914193111"><a name="zh-cn_topic_0000002313328922_p201051914193111"></a><a name="zh-cn_topic_0000002313328922_p201051914193111"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p9105514123113"><a name="zh-cn_topic_0000002313328922_p9105514123113"></a><a name="zh-cn_topic_0000002313328922_p9105514123113"></a>(Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p1910551410318"><a name="zh-cn_topic_0000002313328922_p1910551410318"></a><a name="zh-cn_topic_0000002313328922_p1910551410318"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p20412171411451"><a name="zh-cn_topic_0000002313328922_p20412171411451"></a><a name="zh-cn_topic_0000002313328922_p20412171411451"></a>(Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p11106914183110"><a name="zh-cn_topic_0000002313328922_p11106914183110"></a><a name="zh-cn_topic_0000002313328922_p11106914183110"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p996811614518"><a name="zh-cn_topic_0000002313328922_p996811614518"></a><a name="zh-cn_topic_0000002313328922_p996811614518"></a>(Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p51061814193115"><a name="zh-cn_topic_0000002313328922_p51061814193115"></a><a name="zh-cn_topic_0000002313328922_p51061814193115"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p3552171824517"><a name="zh-cn_topic_0000002313328922_p3552171824517"></a><a name="zh-cn_topic_0000002313328922_p3552171824517"></a>(Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p1110681414313"><a name="zh-cn_topic_0000002313328922_p1110681414313"></a><a name="zh-cn_topic_0000002313328922_p1110681414313"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p14312204455"><a name="zh-cn_topic_0000002313328922_p14312204455"></a><a name="zh-cn_topic_0000002313328922_p14312204455"></a>(Hcq)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row8107151423115"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1110616141319"><a name="zh-cn_topic_0000002313328922_p1110616141319"></a><a name="zh-cn_topic_0000002313328922_p1110616141319"></a>rmsnorm_gamma_ckv</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1110631463118"><a name="zh-cn_topic_0000002313328922_p1110631463118"></a><a name="zh-cn_topic_0000002313328922_p1110631463118"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p15106111423114"><a name="zh-cn_topic_0000002313328922_p15106111423114"></a><a name="zh-cn_topic_0000002313328922_p15106111423114"></a>(Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p71064146316"><a name="zh-cn_topic_0000002313328922_p71064146316"></a><a name="zh-cn_topic_0000002313328922_p71064146316"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p1110619142318"><a name="zh-cn_topic_0000002313328922_p1110619142318"></a><a name="zh-cn_topic_0000002313328922_p1110619142318"></a>(Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p151061114153117"><a name="zh-cn_topic_0000002313328922_p151061114153117"></a><a name="zh-cn_topic_0000002313328922_p151061114153117"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p6106181416316"><a name="zh-cn_topic_0000002313328922_p6106181416316"></a><a name="zh-cn_topic_0000002313328922_p6106181416316"></a>(Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p5106714193110"><a name="zh-cn_topic_0000002313328922_p5106714193110"></a><a name="zh-cn_topic_0000002313328922_p5106714193110"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p61061314183118"><a name="zh-cn_topic_0000002313328922_p61061314183118"></a><a name="zh-cn_topic_0000002313328922_p61061314183118"></a>(Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p141063148312"><a name="zh-cn_topic_0000002313328922_p141063148312"></a><a name="zh-cn_topic_0000002313328922_p141063148312"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p1310611418319"><a name="zh-cn_topic_0000002313328922_p1310611418319"></a><a name="zh-cn_topic_0000002313328922_p1310611418319"></a>(Hckv)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row1107191463116"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p810731415313"><a name="zh-cn_topic_0000002313328922_p810731415313"></a><a name="zh-cn_topic_0000002313328922_p810731415313"></a>rope_sin</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1810781411315"><a name="zh-cn_topic_0000002313328922_p1810781411315"></a><a name="zh-cn_topic_0000002313328922_p1810781411315"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="zh-cn_topic_0000002313328922_ul6327114983818"></a><a name="zh-cn_topic_0000002313328922_ul6327114983818"></a><ul id="zh-cn_topic_0000002313328922_ul6327114983818"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p181071814133111"><a name="zh-cn_topic_0000002313328922_p181071814133111"></a><a name="zh-cn_topic_0000002313328922_p181071814133111"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul254803553916"></a><a name="zh-cn_topic_0000002313328922_ul254803553916"></a><ul id="zh-cn_topic_0000002313328922_ul254803553916"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p610721403111"><a name="zh-cn_topic_0000002313328922_p610721403111"></a><a name="zh-cn_topic_0000002313328922_p610721403111"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="zh-cn_topic_0000002313328922_ul16821173773914"></a><a name="zh-cn_topic_0000002313328922_ul16821173773914"></a><ul id="zh-cn_topic_0000002313328922_ul16821173773914"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p7107201413315"><a name="zh-cn_topic_0000002313328922_p7107201413315"></a><a name="zh-cn_topic_0000002313328922_p7107201413315"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul199164010395"></a><a name="zh-cn_topic_0000002313328922_ul199164010395"></a><ul id="zh-cn_topic_0000002313328922_ul199164010395"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p1310701410316"><a name="zh-cn_topic_0000002313328922_p1310701410316"></a><a name="zh-cn_topic_0000002313328922_p1310701410316"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="zh-cn_topic_0000002313328922_ul241342173912"></a><a name="zh-cn_topic_0000002313328922_ul241342173912"></a><ul id="zh-cn_topic_0000002313328922_ul241342173912"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row16108131453117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p16107114103120"><a name="zh-cn_topic_0000002313328922_p16107114103120"></a><a name="zh-cn_topic_0000002313328922_p16107114103120"></a>rope_cos</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p510781414315"><a name="zh-cn_topic_0000002313328922_p510781414315"></a><a name="zh-cn_topic_0000002313328922_p510781414315"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="zh-cn_topic_0000002313328922_ul6804164519391"></a><a name="zh-cn_topic_0000002313328922_ul6804164519391"></a><ul id="zh-cn_topic_0000002313328922_ul6804164519391"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p15107131419311"><a name="zh-cn_topic_0000002313328922_p15107131419311"></a><a name="zh-cn_topic_0000002313328922_p15107131419311"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul1171554793920"></a><a name="zh-cn_topic_0000002313328922_ul1171554793920"></a><ul id="zh-cn_topic_0000002313328922_ul1171554793920"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p0108141463115"><a name="zh-cn_topic_0000002313328922_p0108141463115"></a><a name="zh-cn_topic_0000002313328922_p0108141463115"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="zh-cn_topic_0000002313328922_ul1998674913913"></a><a name="zh-cn_topic_0000002313328922_ul1998674913913"></a><ul id="zh-cn_topic_0000002313328922_ul1998674913913"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p4108414123111"><a name="zh-cn_topic_0000002313328922_p4108414123111"></a><a name="zh-cn_topic_0000002313328922_p4108414123111"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul66071251153916"></a><a name="zh-cn_topic_0000002313328922_ul66071251153916"></a><ul id="zh-cn_topic_0000002313328922_ul66071251153916"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p151081314153116"><a name="zh-cn_topic_0000002313328922_p151081314153116"></a><a name="zh-cn_topic_0000002313328922_p151081314153116"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="zh-cn_topic_0000002313328922_ul1621625303919"></a><a name="zh-cn_topic_0000002313328922_ul1621625303919"></a><ul id="zh-cn_topic_0000002313328922_ul1621625303919"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row910901403114"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p71081514113115"><a name="zh-cn_topic_0000002313328922_p71081514113115"></a><a name="zh-cn_topic_0000002313328922_p71081514113115"></a>cache_index</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1510861463119"><a name="zh-cn_topic_0000002313328922_p1510861463119"></a><a name="zh-cn_topic_0000002313328922_p1510861463119"></a>`torch.int64`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="zh-cn_topic_0000002313328922_ul19841958123912"></a><a name="zh-cn_topic_0000002313328922_ul19841958123912"></a><ul id="zh-cn_topic_0000002313328922_ul19841958123912"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p181081514133118"><a name="zh-cn_topic_0000002313328922_p181081514133118"></a><a name="zh-cn_topic_0000002313328922_p181081514133118"></a>`torch.int64`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul1850621614011"></a><a name="zh-cn_topic_0000002313328922_ul1850621614011"></a><ul id="zh-cn_topic_0000002313328922_ul1850621614011"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p18108314143115"><a name="zh-cn_topic_0000002313328922_p18108314143115"></a><a name="zh-cn_topic_0000002313328922_p18108314143115"></a>`torch.int64`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="zh-cn_topic_0000002313328922_ul1814741813406"></a><a name="zh-cn_topic_0000002313328922_ul1814741813406"></a><ul id="zh-cn_topic_0000002313328922_ul1814741813406"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p15109131433112"><a name="zh-cn_topic_0000002313328922_p15109131433112"></a><a name="zh-cn_topic_0000002313328922_p15109131433112"></a>`torch.int64`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul14946151914019"></a><a name="zh-cn_topic_0000002313328922_ul14946151914019"></a><ul id="zh-cn_topic_0000002313328922_ul14946151914019"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p31091614153119"><a name="zh-cn_topic_0000002313328922_p31091614153119"></a><a name="zh-cn_topic_0000002313328922_p31091614153119"></a>`torch.int64`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="zh-cn_topic_0000002313328922_ul12670621204012"></a><a name="zh-cn_topic_0000002313328922_ul12670621204012"></a><ul id="zh-cn_topic_0000002313328922_ul12670621204012"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row110918146313"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1010914146318"><a name="zh-cn_topic_0000002313328922_p1010914146318"></a><a name="zh-cn_topic_0000002313328922_p1010914146318"></a>kv_cache</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1610919146315"><a name="zh-cn_topic_0000002313328922_p1610919146315"></a><a name="zh-cn_topic_0000002313328922_p1610919146315"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p91099145315"><a name="zh-cn_topic_0000002313328922_p91099145315"></a><a name="zh-cn_topic_0000002313328922_p91099145315"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p510911413112"><a name="zh-cn_topic_0000002313328922_p510911413112"></a><a name="zh-cn_topic_0000002313328922_p510911413112"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p1248419487458"><a name="zh-cn_topic_0000002313328922_p1248419487458"></a><a name="zh-cn_topic_0000002313328922_p1248419487458"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1610921483112"><a name="zh-cn_topic_0000002313328922_p1610921483112"></a><a name="zh-cn_topic_0000002313328922_p1610921483112"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p102779548454"><a name="zh-cn_topic_0000002313328922_p102779548454"></a><a name="zh-cn_topic_0000002313328922_p102779548454"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p61097141319"><a name="zh-cn_topic_0000002313328922_p61097141319"></a><a name="zh-cn_topic_0000002313328922_p61097141319"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p269055604512"><a name="zh-cn_topic_0000002313328922_p269055604512"></a><a name="zh-cn_topic_0000002313328922_p269055604512"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p101091814193117"><a name="zh-cn_topic_0000002313328922_p101091814193117"></a><a name="zh-cn_topic_0000002313328922_p101091814193117"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p188285815456"><a name="zh-cn_topic_0000002313328922_p188285815456"></a><a name="zh-cn_topic_0000002313328922_p188285815456"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row1011013147312"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p110971483115"><a name="zh-cn_topic_0000002313328922_p110971483115"></a><a name="zh-cn_topic_0000002313328922_p110971483115"></a>kr_cache</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p16109101411314"><a name="zh-cn_topic_0000002313328922_p16109101411314"></a><a name="zh-cn_topic_0000002313328922_p16109101411314"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p20110161453119"><a name="zh-cn_topic_0000002313328922_p20110161453119"></a><a name="zh-cn_topic_0000002313328922_p20110161453119"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p1311013144317"><a name="zh-cn_topic_0000002313328922_p1311013144317"></a><a name="zh-cn_topic_0000002313328922_p1311013144317"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p811019146316"><a name="zh-cn_topic_0000002313328922_p811019146316"></a><a name="zh-cn_topic_0000002313328922_p811019146316"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p11110414103116"><a name="zh-cn_topic_0000002313328922_p11110414103116"></a><a name="zh-cn_topic_0000002313328922_p11110414103116"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p420731810463"><a name="zh-cn_topic_0000002313328922_p420731810463"></a><a name="zh-cn_topic_0000002313328922_p420731810463"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p4110121414313"><a name="zh-cn_topic_0000002313328922_p4110121414313"></a><a name="zh-cn_topic_0000002313328922_p4110121414313"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p111101814123114"><a name="zh-cn_topic_0000002313328922_p111101814123114"></a><a name="zh-cn_topic_0000002313328922_p111101814123114"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p9110121463118"><a name="zh-cn_topic_0000002313328922_p9110121463118"></a><a name="zh-cn_topic_0000002313328922_p9110121463118"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p345722317462"><a name="zh-cn_topic_0000002313328922_p345722317462"></a><a name="zh-cn_topic_0000002313328922_p345722317462"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row1211161411319"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p111020146314"><a name="zh-cn_topic_0000002313328922_p111020146314"></a><a name="zh-cn_topic_0000002313328922_p111020146314"></a>dequant_scale_x</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p511031410310"><a name="zh-cn_topic_0000002313328922_p511031410310"></a><a name="zh-cn_topic_0000002313328922_p511031410310"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p121101014193110"><a name="zh-cn_topic_0000002313328922_p121101014193110"></a><a name="zh-cn_topic_0000002313328922_p121101014193110"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p81101214163111"><a name="zh-cn_topic_0000002313328922_p81101214163111"></a><a name="zh-cn_topic_0000002313328922_p81101214163111"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p5111214133114"><a name="zh-cn_topic_0000002313328922_p5111214133114"></a><a name="zh-cn_topic_0000002313328922_p5111214133114"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p11111114153110"><a name="zh-cn_topic_0000002313328922_p11111114153110"></a><a name="zh-cn_topic_0000002313328922_p11111114153110"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p5111111412314"><a name="zh-cn_topic_0000002313328922_p5111111412314"></a><a name="zh-cn_topic_0000002313328922_p5111111412314"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p14111161416312"><a name="zh-cn_topic_0000002313328922_p14111161416312"></a><a name="zh-cn_topic_0000002313328922_p14111161416312"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul1875112920408"></a><a name="zh-cn_topic_0000002313328922_ul1875112920408"></a><ul id="zh-cn_topic_0000002313328922_ul1875112920408"><li>(B*S,1)</li><li>(T,1)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p511141420316"><a name="zh-cn_topic_0000002313328922_p511141420316"></a><a name="zh-cn_topic_0000002313328922_p511141420316"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="zh-cn_topic_0000002313328922_ul187764617408"></a><a name="zh-cn_topic_0000002313328922_ul187764617408"></a><ul id="zh-cn_topic_0000002313328922_ul187764617408"><li>(B*S,1)</li><li>(T,1)</li></ul>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row16112614153117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p2011112146316"><a name="zh-cn_topic_0000002313328922_p2011112146316"></a><a name="zh-cn_topic_0000002313328922_p2011112146316"></a>dequant_scale_w_dq</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p511141433113"><a name="zh-cn_topic_0000002313328922_p511141433113"></a><a name="zh-cn_topic_0000002313328922_p511141433113"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p111111214133114"><a name="zh-cn_topic_0000002313328922_p111111214133114"></a><a name="zh-cn_topic_0000002313328922_p111111214133114"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p2111131411312"><a name="zh-cn_topic_0000002313328922_p2111131411312"></a><a name="zh-cn_topic_0000002313328922_p2111131411312"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p151123140315"><a name="zh-cn_topic_0000002313328922_p151123140315"></a><a name="zh-cn_topic_0000002313328922_p151123140315"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1011291411315"><a name="zh-cn_topic_0000002313328922_p1011291411315"></a><a name="zh-cn_topic_0000002313328922_p1011291411315"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p131120149316"><a name="zh-cn_topic_0000002313328922_p131120149316"></a><a name="zh-cn_topic_0000002313328922_p131120149316"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p9112111473120"><a name="zh-cn_topic_0000002313328922_p9112111473120"></a><a name="zh-cn_topic_0000002313328922_p9112111473120"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p12112121423118"><a name="zh-cn_topic_0000002313328922_p12112121423118"></a><a name="zh-cn_topic_0000002313328922_p12112121423118"></a>(1,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p011241416315"><a name="zh-cn_topic_0000002313328922_p011241416315"></a><a name="zh-cn_topic_0000002313328922_p011241416315"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p1611251453115"><a name="zh-cn_topic_0000002313328922_p1611251453115"></a><a name="zh-cn_topic_0000002313328922_p1611251453115"></a>(1,Hcq)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row411310144314"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p61124144317"><a name="zh-cn_topic_0000002313328922_p61124144317"></a><a name="zh-cn_topic_0000002313328922_p61124144317"></a>dequant_scale_w_uq_qr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1511391413316"><a name="zh-cn_topic_0000002313328922_p1511391413316"></a><a name="zh-cn_topic_0000002313328922_p1511391413316"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1611321414310"><a name="zh-cn_topic_0000002313328922_p1611321414310"></a><a name="zh-cn_topic_0000002313328922_p1611321414310"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p191131214143118"><a name="zh-cn_topic_0000002313328922_p191131214143118"></a><a name="zh-cn_topic_0000002313328922_p191131214143118"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p1011381416313"><a name="zh-cn_topic_0000002313328922_p1011381416313"></a><a name="zh-cn_topic_0000002313328922_p1011381416313"></a>(1,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p171131514113119"><a name="zh-cn_topic_0000002313328922_p171131514113119"></a><a name="zh-cn_topic_0000002313328922_p171131514113119"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p1774723914611"><a name="zh-cn_topic_0000002313328922_p1774723914611"></a><a name="zh-cn_topic_0000002313328922_p1774723914611"></a>(1,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p11139145317"><a name="zh-cn_topic_0000002313328922_p11139145317"></a><a name="zh-cn_topic_0000002313328922_p11139145317"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p13113171420315"><a name="zh-cn_topic_0000002313328922_p13113171420315"></a><a name="zh-cn_topic_0000002313328922_p13113171420315"></a>(1,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p16113314153120"><a name="zh-cn_topic_0000002313328922_p16113314153120"></a><a name="zh-cn_topic_0000002313328922_p16113314153120"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p1081344154613"><a name="zh-cn_topic_0000002313328922_p1081344154613"></a><a name="zh-cn_topic_0000002313328922_p1081344154613"></a>(1,N*(D+Dr))</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row19114101463113"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p13113181419315"><a name="zh-cn_topic_0000002313328922_p13113181419315"></a><a name="zh-cn_topic_0000002313328922_p13113181419315"></a>dequant_scale_w_dkv_kr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p2113114133115"><a name="zh-cn_topic_0000002313328922_p2113114133115"></a><a name="zh-cn_topic_0000002313328922_p2113114133115"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p91131145310"><a name="zh-cn_topic_0000002313328922_p91131145310"></a><a name="zh-cn_topic_0000002313328922_p91131145310"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p15113111415314"><a name="zh-cn_topic_0000002313328922_p15113111415314"></a><a name="zh-cn_topic_0000002313328922_p15113111415314"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p18113191433113"><a name="zh-cn_topic_0000002313328922_p18113191433113"></a><a name="zh-cn_topic_0000002313328922_p18113191433113"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p91131914103116"><a name="zh-cn_topic_0000002313328922_p91131914103116"></a><a name="zh-cn_topic_0000002313328922_p91131914103116"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p911341412318"><a name="zh-cn_topic_0000002313328922_p911341412318"></a><a name="zh-cn_topic_0000002313328922_p911341412318"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p18114131413113"><a name="zh-cn_topic_0000002313328922_p18114131413113"></a><a name="zh-cn_topic_0000002313328922_p18114131413113"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p711471414312"><a name="zh-cn_topic_0000002313328922_p711471414312"></a><a name="zh-cn_topic_0000002313328922_p711471414312"></a>(1,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p91141614143115"><a name="zh-cn_topic_0000002313328922_p91141614143115"></a><a name="zh-cn_topic_0000002313328922_p91141614143115"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p152341456204613"><a name="zh-cn_topic_0000002313328922_p152341456204613"></a><a name="zh-cn_topic_0000002313328922_p152341456204613"></a>(1,Hckv+Dr)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row1811491433118"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p151141514103110"><a name="zh-cn_topic_0000002313328922_p151141514103110"></a><a name="zh-cn_topic_0000002313328922_p151141514103110"></a>quant_scale_ckv</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p10114514143118"><a name="zh-cn_topic_0000002313328922_p10114514143118"></a><a name="zh-cn_topic_0000002313328922_p10114514143118"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1311418145312"><a name="zh-cn_topic_0000002313328922_p1311418145312"></a><a name="zh-cn_topic_0000002313328922_p1311418145312"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p1114141413111"><a name="zh-cn_topic_0000002313328922_p1114141413111"></a><a name="zh-cn_topic_0000002313328922_p1114141413111"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p9114171417313"><a name="zh-cn_topic_0000002313328922_p9114171417313"></a><a name="zh-cn_topic_0000002313328922_p9114171417313"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p2011441433113"><a name="zh-cn_topic_0000002313328922_p2011441433113"></a><a name="zh-cn_topic_0000002313328922_p2011441433113"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p61141614173119"><a name="zh-cn_topic_0000002313328922_p61141614173119"></a><a name="zh-cn_topic_0000002313328922_p61141614173119"></a>(1,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p51141714123118"><a name="zh-cn_topic_0000002313328922_p51141714123118"></a><a name="zh-cn_topic_0000002313328922_p51141714123118"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p911415141314"><a name="zh-cn_topic_0000002313328922_p911415141314"></a><a name="zh-cn_topic_0000002313328922_p911415141314"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p161143149317"><a name="zh-cn_topic_0000002313328922_p161143149317"></a><a name="zh-cn_topic_0000002313328922_p161143149317"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p1513369122213"><a name="zh-cn_topic_0000002313328922_p1513369122213"></a><a name="zh-cn_topic_0000002313328922_p1513369122213"></a>(1,Hckv)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row711512142317"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p10114181412318"><a name="zh-cn_topic_0000002313328922_p10114181412318"></a><a name="zh-cn_topic_0000002313328922_p10114181412318"></a>quant_scale_ckr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p5115131412312"><a name="zh-cn_topic_0000002313328922_p5115131412312"></a><a name="zh-cn_topic_0000002313328922_p5115131412312"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p3115814193118"><a name="zh-cn_topic_0000002313328922_p3115814193118"></a><a name="zh-cn_topic_0000002313328922_p3115814193118"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p121151414103110"><a name="zh-cn_topic_0000002313328922_p121151414103110"></a><a name="zh-cn_topic_0000002313328922_p121151414103110"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p18115314183111"><a name="zh-cn_topic_0000002313328922_p18115314183111"></a><a name="zh-cn_topic_0000002313328922_p18115314183111"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p4115114113119"><a name="zh-cn_topic_0000002313328922_p4115114113119"></a><a name="zh-cn_topic_0000002313328922_p4115114113119"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p511518143318"><a name="zh-cn_topic_0000002313328922_p511518143318"></a><a name="zh-cn_topic_0000002313328922_p511518143318"></a>(1,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1411591443113"><a name="zh-cn_topic_0000002313328922_p1411591443113"></a><a name="zh-cn_topic_0000002313328922_p1411591443113"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p16115181419312"><a name="zh-cn_topic_0000002313328922_p16115181419312"></a><a name="zh-cn_topic_0000002313328922_p16115181419312"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p1011511473112"><a name="zh-cn_topic_0000002313328922_p1011511473112"></a><a name="zh-cn_topic_0000002313328922_p1011511473112"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p171151914133118"><a name="zh-cn_topic_0000002313328922_p171151914133118"></a><a name="zh-cn_topic_0000002313328922_p171151914133118"></a>/</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row10116114103112"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p11115161418311"><a name="zh-cn_topic_0000002313328922_p11115161418311"></a><a name="zh-cn_topic_0000002313328922_p11115161418311"></a>smooth_scales_cq</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p12115191483119"><a name="zh-cn_topic_0000002313328922_p12115191483119"></a><a name="zh-cn_topic_0000002313328922_p12115191483119"></a>无需赋值</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p111151147315"><a name="zh-cn_topic_0000002313328922_p111151147315"></a><a name="zh-cn_topic_0000002313328922_p111151147315"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p81152014153118"><a name="zh-cn_topic_0000002313328922_p81152014153118"></a><a name="zh-cn_topic_0000002313328922_p81152014153118"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p1411591483118"><a name="zh-cn_topic_0000002313328922_p1411591483118"></a><a name="zh-cn_topic_0000002313328922_p1411591483118"></a>(1,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p311514144313"><a name="zh-cn_topic_0000002313328922_p311514144313"></a><a name="zh-cn_topic_0000002313328922_p311514144313"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p171151147319"><a name="zh-cn_topic_0000002313328922_p171151147319"></a><a name="zh-cn_topic_0000002313328922_p171151147319"></a>(1,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p0116191443116"><a name="zh-cn_topic_0000002313328922_p0116191443116"></a><a name="zh-cn_topic_0000002313328922_p0116191443116"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p104931614134716"><a name="zh-cn_topic_0000002313328922_p104931614134716"></a><a name="zh-cn_topic_0000002313328922_p104931614134716"></a>(1,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p19116614173110"><a name="zh-cn_topic_0000002313328922_p19116614173110"></a><a name="zh-cn_topic_0000002313328922_p19116614173110"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p1880211619471"><a name="zh-cn_topic_0000002313328922_p1880211619471"></a><a name="zh-cn_topic_0000002313328922_p1880211619471"></a>(1,Hcq)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row1611711147313"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p18116191473120"><a name="zh-cn_topic_0000002313328922_p18116191473120"></a><a name="zh-cn_topic_0000002313328922_p18116191473120"></a>query</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p0116171433116"><a name="zh-cn_topic_0000002313328922_p0116171433116"></a><a name="zh-cn_topic_0000002313328922_p0116171433116"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="zh-cn_topic_0000002313328922_ul15120190144120"></a><a name="zh-cn_topic_0000002313328922_ul15120190144120"></a><ul id="zh-cn_topic_0000002313328922_ul15120190144120"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p11116191483116"><a name="zh-cn_topic_0000002313328922_p11116191483116"></a><a name="zh-cn_topic_0000002313328922_p11116191483116"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul1945521894110"></a><a name="zh-cn_topic_0000002313328922_ul1945521894110"></a><ul id="zh-cn_topic_0000002313328922_ul1945521894110"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p31167143319"><a name="zh-cn_topic_0000002313328922_p31167143319"></a><a name="zh-cn_topic_0000002313328922_p31167143319"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="zh-cn_topic_0000002313328922_ul5222182074120"></a><a name="zh-cn_topic_0000002313328922_ul5222182074120"></a><ul id="zh-cn_topic_0000002313328922_ul5222182074120"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p181162148319"><a name="zh-cn_topic_0000002313328922_p181162148319"></a><a name="zh-cn_topic_0000002313328922_p181162148319"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul4809112113415"></a><a name="zh-cn_topic_0000002313328922_ul4809112113415"></a><ul id="zh-cn_topic_0000002313328922_ul4809112113415"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p611671419319"><a name="zh-cn_topic_0000002313328922_p611671419319"></a><a name="zh-cn_topic_0000002313328922_p611671419319"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="zh-cn_topic_0000002313328922_ul1831425194112"></a><a name="zh-cn_topic_0000002313328922_ul1831425194112"></a><ul id="zh-cn_topic_0000002313328922_ul1831425194112"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row1411711410316"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p12117111413112"><a name="zh-cn_topic_0000002313328922_p12117111413112"></a><a name="zh-cn_topic_0000002313328922_p12117111413112"></a>query_rope</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p4117514153118"><a name="zh-cn_topic_0000002313328922_p4117514153118"></a><a name="zh-cn_topic_0000002313328922_p4117514153118"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="zh-cn_topic_0000002313328922_ul10316123710419"></a><a name="zh-cn_topic_0000002313328922_ul10316123710419"></a><ul id="zh-cn_topic_0000002313328922_ul10316123710419"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p1511719141313"><a name="zh-cn_topic_0000002313328922_p1511719141313"></a><a name="zh-cn_topic_0000002313328922_p1511719141313"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul225985618412"></a><a name="zh-cn_topic_0000002313328922_ul225985618412"></a><ul id="zh-cn_topic_0000002313328922_ul225985618412"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p711741414318"><a name="zh-cn_topic_0000002313328922_p711741414318"></a><a name="zh-cn_topic_0000002313328922_p711741414318"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="zh-cn_topic_0000002313328922_ul183131705429"></a><a name="zh-cn_topic_0000002313328922_ul183131705429"></a><ul id="zh-cn_topic_0000002313328922_ul183131705429"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p3117201420317"><a name="zh-cn_topic_0000002313328922_p3117201420317"></a><a name="zh-cn_topic_0000002313328922_p3117201420317"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="zh-cn_topic_0000002313328922_ul383917194210"></a><a name="zh-cn_topic_0000002313328922_ul383917194210"></a><ul id="zh-cn_topic_0000002313328922_ul383917194210"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p111731415319"><a name="zh-cn_topic_0000002313328922_p111731415319"></a><a name="zh-cn_topic_0000002313328922_p111731415319"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="zh-cn_topic_0000002313328922_ul537183164214"></a><a name="zh-cn_topic_0000002313328922_ul537183164214"></a><ul id="zh-cn_topic_0000002313328922_ul537183164214"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row111871453119"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p711719148317"><a name="zh-cn_topic_0000002313328922_p711719148317"></a><a name="zh-cn_topic_0000002313328922_p711719148317"></a>kv_cache</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1117814153111"><a name="zh-cn_topic_0000002313328922_p1117814153111"></a><a name="zh-cn_topic_0000002313328922_p1117814153111"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p115809401478"><a name="zh-cn_topic_0000002313328922_p115809401478"></a><a name="zh-cn_topic_0000002313328922_p115809401478"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p11178144316"><a name="zh-cn_topic_0000002313328922_p11178144316"></a><a name="zh-cn_topic_0000002313328922_p11178144316"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p3418442164713"><a name="zh-cn_topic_0000002313328922_p3418442164713"></a><a name="zh-cn_topic_0000002313328922_p3418442164713"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p12118141419319"><a name="zh-cn_topic_0000002313328922_p12118141419319"></a><a name="zh-cn_topic_0000002313328922_p12118141419319"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p86601644114719"><a name="zh-cn_topic_0000002313328922_p86601644114719"></a><a name="zh-cn_topic_0000002313328922_p86601644114719"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p5118201443110"><a name="zh-cn_topic_0000002313328922_p5118201443110"></a><a name="zh-cn_topic_0000002313328922_p5118201443110"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p47124724714"><a name="zh-cn_topic_0000002313328922_p47124724714"></a><a name="zh-cn_topic_0000002313328922_p47124724714"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p5118014193110"><a name="zh-cn_topic_0000002313328922_p5118014193110"></a><a name="zh-cn_topic_0000002313328922_p5118014193110"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p980949134717"><a name="zh-cn_topic_0000002313328922_p980949134717"></a><a name="zh-cn_topic_0000002313328922_p980949134717"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row0119171483111"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p101182014173116"><a name="zh-cn_topic_0000002313328922_p101182014173116"></a><a name="zh-cn_topic_0000002313328922_p101182014173116"></a>kr_cache</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p41180146312"><a name="zh-cn_topic_0000002313328922_p41180146312"></a><a name="zh-cn_topic_0000002313328922_p41180146312"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p18382381877"><a name="zh-cn_topic_0000002313328922_p18382381877"></a><a name="zh-cn_topic_0000002313328922_p18382381877"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p8118141411318"><a name="zh-cn_topic_0000002313328922_p8118141411318"></a><a name="zh-cn_topic_0000002313328922_p8118141411318"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p17382411973"><a name="zh-cn_topic_0000002313328922_p17382411973"></a><a name="zh-cn_topic_0000002313328922_p17382411973"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p10118101413114"><a name="zh-cn_topic_0000002313328922_p10118101413114"></a><a name="zh-cn_topic_0000002313328922_p10118101413114"></a>`torch.int8`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p127212315481"><a name="zh-cn_topic_0000002313328922_p127212315481"></a><a name="zh-cn_topic_0000002313328922_p127212315481"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p51181614113119"><a name="zh-cn_topic_0000002313328922_p51181614113119"></a><a name="zh-cn_topic_0000002313328922_p51181614113119"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p1088320441076"><a name="zh-cn_topic_0000002313328922_p1088320441076"></a><a name="zh-cn_topic_0000002313328922_p1088320441076"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p18118181493113"><a name="zh-cn_topic_0000002313328922_p18118181493113"></a><a name="zh-cn_topic_0000002313328922_p18118181493113"></a>`torch.bfloat16`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="zh-cn_topic_0000002313328922_p18451144612718"><a name="zh-cn_topic_0000002313328922_p18451144612718"></a><a name="zh-cn_topic_0000002313328922_p18451144612718"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002313328922_row17119414153117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p17119121411315"><a name="zh-cn_topic_0000002313328922_p17119121411315"></a><a name="zh-cn_topic_0000002313328922_p17119121411315"></a>dequant_scale_q_nope</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1111920146316"><a name="zh-cn_topic_0000002313328922_p1111920146316"></a><a name="zh-cn_topic_0000002313328922_p1111920146316"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1511911418312"><a name="zh-cn_topic_0000002313328922_p1511911418312"></a><a name="zh-cn_topic_0000002313328922_p1511911418312"></a>(1<span id="zh-cn_topic_0000002313328922_ph18586141419271"><a name="zh-cn_topic_0000002313328922_ph18586141419271"></a><a name="zh-cn_topic_0000002313328922_ph18586141419271"></a>,</span>)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p1611918142315"><a name="zh-cn_topic_0000002313328922_p1611918142315"></a><a name="zh-cn_topic_0000002313328922_p1611918142315"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p19351141464819"><a name="zh-cn_topic_0000002313328922_p19351141464819"></a><a name="zh-cn_topic_0000002313328922_p19351141464819"></a>(1<span id="zh-cn_topic_0000002313328922_ph35291817152714"><a name="zh-cn_topic_0000002313328922_ph35291817152714"></a><a name="zh-cn_topic_0000002313328922_ph35291817152714"></a>,</span>)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p6119101414312"><a name="zh-cn_topic_0000002313328922_p6119101414312"></a><a name="zh-cn_topic_0000002313328922_p6119101414312"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="zh-cn_topic_0000002313328922_p14893131519488"><a name="zh-cn_topic_0000002313328922_p14893131519488"></a><a name="zh-cn_topic_0000002313328922_p14893131519488"></a>(1<span id="zh-cn_topic_0000002313328922_ph166319122718"><a name="zh-cn_topic_0000002313328922_ph166319122718"></a><a name="zh-cn_topic_0000002313328922_ph166319122718"></a>,</span>)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="zh-cn_topic_0000002313328922_p1611911414313"><a name="zh-cn_topic_0000002313328922_p1611911414313"></a><a name="zh-cn_topic_0000002313328922_p1611911414313"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="zh-cn_topic_0000002313328922_p13273111784815"><a name="zh-cn_topic_0000002313328922_p13273111784815"></a><a name="zh-cn_topic_0000002313328922_p13273111784815"></a>(1<span id="zh-cn_topic_0000002313328922_ph10953620192719"><a name="zh-cn_topic_0000002313328922_ph10953620192719"></a><a name="zh-cn_topic_0000002313328922_ph10953620192719"></a>,</span>)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="zh-cn_topic_0000002313328922_p1911971463117"><a name="zh-cn_topic_0000002313328922_p1911971463117"></a><a name="zh-cn_topic_0000002313328922_p1911971463117"></a>`torch.float`</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="zh-cn_topic_0000002313328922_ul5238121054219"></a><a name="zh-cn_topic_0000002313328922_ul5238121054219"></a><ul id="zh-cn_topic_0000002313328922_ul5238121054219"><li>(B*S,N,1)</li><li>(T,N,1)</li></ul>
    </td>
    </tr>
    </tbody>
    </table>

## 调用示例<a name="zh-cn_topic_0000002313328922_section983519211229"></a>

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import math
    torch.npu.config.allow_internal_format = True
    # 生成随机数据, 并发送到npu
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    cache_num_slots = kv_cache.size(0) * kv_cache.size(1)
    cache_index = torch.randint(0, cache_num_slots, (B, S), dtype=torch.int64).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    
    # 调用MlaProlog算子
    query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla, dequant_scale_q_nope_mla = torch_npu.npu_mla_prolog_v2(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    print(query_mla)
    # 执行上述代码的输出类似如下
    tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ..
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16)
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
    
    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    torch.npu.config.allow_internal_format = True
    
    # 数据生成
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    cache_num_slots = kv_cache.size(0) * kv_cache.size(1)
    cache_index = torch.randint(0, cache_num_slots, (B, S), dtype=torch.int64).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_mla_prolog_v2(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla, dequant_scale_q_nope_mla = torch_npu.npu_mla_prolog_v2(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
        print("single op output:", query_mla)
        print("graph output:", graph_output)
        
    if __name__ == "__main__":
        MetaInfershape()
    
    # 执行上述代码的输出类似如下
    single op output: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16)
    
    graph output: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16) 
    ```
