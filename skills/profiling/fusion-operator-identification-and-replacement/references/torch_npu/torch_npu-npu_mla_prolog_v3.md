# torch_npu.npu_mla_prolog_v3

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

- 接口功能：推理场景下，Multi-Head Latent Attention（MLA）前处理的计算操作。v3版本相比v2新增query_norm和dequant_scale_q_norm输出、pertoken-pergroup量化模式、尺度矫正因子、多种cache_mode格式支持及多个量化配置参数，并调整cache_index为可选参数。该算子实现四条并行的计算路径：
    1. 标准Query路径：输入$x$ → $W^{DQ}$下采样 → RmsNorm → $W^{UQ}$上采样 → $W^{UK}$上采样 → $q^N$
    2. 位置编码Query路径：输入$x$ → $W^{DQ}$下采样 → RmsNorm → $W^{QR}$ → ROPE旋转位置编码 → $q^R$
    3. 标准Key路径：输入$x$ → $W^{DKV}$下采样 → RmsNorm → Cache存储 → $k^C$
    4. 位置编码Key路径：输入$x$ → $W^{KR}$ → ROPE旋转位置编码 → Cache存储 → $k^R$

- 相比torch_npu.npu_mla_prolog_v2的主要差异如下：
    - 新增输出`query_norm`和`dequant_scale_q_norm`，用于支持DeepSeekV3.2网络。
    - 新增`kv_cache`的pertoken-pergroup量化模式。
    - 新增query与key的尺度矫正因子，分别对应qc_qr_scale（$\alpha_q$）与kc_scale（$\alpha_{kv}$）。
    - 新增`cache_mode`对"PA_BLK_BSND"、"PA_BLK_NZ"、"BSND"和"TND"格式的支持。
    - 新增可选参数`weight_quant_mode`、`kv_cache_quant_mode`、`query_quant_mode`、`ckvkr_repo_mode`、`quant_scale_repo_mode`，用于配置量化场景。
    - 调整`cache_index`为可选参数。

- 计算公式：
    - RmsNorm公式
        $$
        \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
        $$

        $$
        \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
        $$

    - 路径1：标准Query计算

        包括下采样、RmsNorm和两次上采样：

        $$
        c^Q = RmsNorm(x \cdot W^{DQ})
        $$

        $$
        q^C = c^Q \cdot W^{UQ}
        $$

        $$
        q^N = q^C \cdot W^{UK}
        $$

    - 路径2：位置编码Query计算

        对Query进行ROPE旋转位置编码：

        $$
        q^R = ROPE(c^Q \cdot W^{QR})
        $$

    - 路径3：标准Key计算

        包括下采样、RmsNorm，将计算结果存入Cache：

        $$
        c^{KV} = RmsNorm(x \cdot W^{DKV})
        $$

        $$
        k^C = Cache(c^{KV})
        $$

    - 路径4：位置编码Key计算

        对Key进行ROPE旋转位置编码，并将结果存入Cache：

        $$
        k^R = Cache(ROPE(x \cdot W^{KR}))
        $$

- ROPE操作补充说明

    torch_npu.npu_mla_prolog_v3算子实现中融合rotary操作的interleave-half模式，该模式会先将奇数位的输入抽取到前半部分，将偶数位的输入抽取到后半部分，再进行half处理。

    $$
    x_1 = x[..., ::2]
    $$

    $$
    x_2 = x[..., 1::2]
    $$

    $$
    x_{part1} = torch.cat((x_1, x_2), dim = -1)
    $$

    $$
    x_{part2} = torch.cat((-x_2, x_1), dim = -1)
    $$

    $$
    y = x_{part1} * cos + x_{part2} * sin
    $$

## 函数原型

```python
torch_npu.npu_mla_prolog_v3(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index=None, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None, quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None, actual_seq_len=None, k_nope_clip_alpha=None, rmsnorm_epsilon_cq=1e-05, rmsnorm_epsilon_ckv=1e-05, cache_mode='PA_BSND', query_norm_flag=False, weight_quant_mode=0, kv_cache_quant_mode=0, query_quant_mode=0, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, qc_qr_scale=1.0, kc_scale=1.0, token_x_dtype=None, weight_dq_dtype=None, weight_uq_qr_dtype=None, weight_dkv_kr_dtype=None, kv_cache_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

  > [!NOTE]
  >
  > - B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、He（Head Size）表示隐藏层大小、N（Head Num）表示多头数、Hcq表示q低秩矩阵维度、Hckv表示kv低秩矩阵维度、Dtile表示kv_cache的D轴维度、D表示qk不含位置编码维度、Dr表示qk位置编码维度、Nkv表示kv的head数、BlockNum表示PagedAttention场景下的块数、BlockSize表示PagedAttention场景下的块大小、T表示BS合轴后的大小。

- **token_x**（`Tensor`）：必选参数，公式中用于计算Query和Key的输入tensor。不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`、`torch.int8`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。BS合轴时，shape为[T, He]；BS非合轴时，shape为[B, S, He]。

- **weight_dq**（`Tensor`）：必选参数，公式中用于计算Query的下采样权重矩阵$W^{DQ}$。不支持非连续，数据格式支持FRACTAL_NZ，数据类型支持`torch.bfloat16`、`torch.int8`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`，shape为[He, Hcq]。

- **weight_uq_qr**（`Tensor`）：必选参数，公式中用于计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$。不支持非连续，数据格式支持FRACTAL_NZ，数据类型支持`torch.bfloat16`、`torch.int8`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`，shape为[Hcq, N*(D+Dr)]。

- **weight_uk**（`Tensor`）：必选参数，公式中用于计算Query的上采样权重$W^{UK}$。不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`，shape为[N, D, Hckv]。

- **weight_dkv_kr**（`Tensor`）：必选参数，公式中用于计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$。不支持非连续，数据格式支持FRACTAL_NZ，数据类型支持`torch.bfloat16`、`torch.int8`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`，shape为[He, Hckv+Dr]。

- **rmsnorm_gamma_cq**（`Tensor`）：必选参数，计算$c^Q$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`，shape为[Hcq]。

- **rmsnorm_gamma_ckv**（`Tensor`）：必选参数，计算$c^{KV}$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`，shape为[Hckv]。

- **rope_sin**（`Tensor`）：必选参数，用于计算旋转位置编码的正弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`。BS合轴时，shape为[T, Dr]；BS非合轴时，shape为[B, S, Dr]。支持B=0,S=0,T=0的空Tensor。

- **rope_cos**（`Tensor`）：必选参数，用于计算旋转位置编码的余弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`；BS合轴时，shape为[T, Dr]；BS非合轴时，shape为[B, S, Dr]。支持B=0,S=0,T=0的空Tensor。

- **kv_cache**（`Tensor`）：必选参数，表示cache的索引，计算结果原地更新（对应公式中的$k^C$）。不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`、`torch.int8`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`，当cache_mode为"PA_BSND"、"PA_NZ"、"PA_BLK_BSND"、"PA_BLK_NZ"时shape为[BlockNum, BlockSize, Nkv, Dtile]，支持B=0,Skv=0的空Tensor;当cache_mode为"BSND"时shape为[B, S, Nkv, Dtile]，不支持空Tensor；当cache_mode为"TND"时shape为[T, Nkv, Dtile]，不支持空Tensor；Nkv与N关联，N是超参，故不支持Nkv=0。

- **kr_cache**（`Tensor`）：必选参数，用于key位置编码的cache，计算结果原地更新（对应公式中的$k^R$）。不支持非连续，数据格式支持ND，数据类型支持`torch.bfloat16`、`torch.int8`，当cache_mode为"PA_BSND"、"PA_NZ"、"PA_BLK_BSND"、"PA_BLK_NZ"时shape为[BlockNum, BlockSize, Nkv, Dr]，支持B=0,Skv=0的空Tensor；当cache_mode为"BSND"时shape为[B, S, Nkv, Dr]，不支持空Tensor；当cache_mode为"TND"时shape为[T, Nkv, Dr]，不支持空Tensor；Nkv与N关联，N是超参，故不支持Nkv=0。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **cache_index**（`Tensor`）：可选参数，用于存储`kv_cache`和`kr_cache`的索引。不支持非连续，数据格式支持ND，数据类型支持`torch.int64`，当`cache_mode`为"PA_BSND"或"PA_NZ"：BS合轴时shape为[T]，BS非合轴时shape为[B, S]，取值范围需在[0, BlockNum*BlockSize)内; 当`cache_mode`为"PA_BLK_BSND"或"PA_BLK_NZ"：BS合轴时shape为[Sum(Ceil(S_i/BlockSize))]（S_i表示第i个batch的序列长度），BS非合轴时shape为[B, Ceil(S/BlockSize)]，取值范围需在[0, BlockNum)内；当`cache_mode`为"BSND"或"TND"：`cache_index`无需传入。当前不会对传入值的合法性进行校验，需用户自行保证。

- **dequant_scale_x**（`Tensor`）：可选参数，token_x的反量化参数。不支持非连续，数据格式支持ND，数据类型支持`torch.float`、`torch.float8_e8m0fnu`。weight_quant_mode=2/4/5时，shape为[T, 1]或[B\*S, 1]；weight_quant_mode=3时，shape为[T, He/32]或[B*S, He/32]；weight_quant_mode=1时无需赋值。支持B=0,S=0,T=0的空Tensor。

- **dequant_scale_w_dq**（`Tensor`）：可选参数，weight_dq的反量化参数。不支持非连续，数据格式支持ND，数据类型支持`torch.float`、`torch.float8_e8m0fnu`。weight_quant_mode=2/4/5时，shape为[1,Hcq]；weight_quant_mode=3时，shape为[Hcq, He/32]；weight_quant_mode=1时无需赋值。

- **dequant_scale_w_uq_qr**（`Tensor`）：可选参数，用于MatmulQcQr矩阵乘后反量化操作的perchannel参数。不支持非连续，数据格式支持ND，数据类型支持`torch.float`、`torch.float8_e8m0fnu`。weight_quant_mode=1/2/4/5时，shape为[1,N*(D+Dr)]；weight_quant_mode=3时，shape为[N*(D+Dr), Hcq/32]。

- **dequant_scale_w_dkv_kr**（`Tensor`）：可选参数，weight_dkv_kr的反量化参数。不支持非连续，数据格式支持ND，数据类型支持`torch.float`、`torch.float8_e8m0fnu`。weight_quant_mode=2/4/5时，shape为[(1,Hckv+Dr)]；weight_quant_mode=3时，shape为[Hckv+Dr, He/32]。weight_quant_mode=1时无需赋值。

- **quant_scale_ckv**（`Tensor`）：可选参数，用于对kv_cache输出数据做量化操作的参数。不支持非连续，数据格式支持ND，数据类型支持`torch.float`，kv_cache_quant_mode=1时shape为[1]；kv_cache_quant_mode=2时shape为[1, Hckv]；kv_cache_quant_mode=3时无需赋值。支持非空Tensor。

- **quant_scale_ckr**（`Tensor`）：可选参数，用于对kr_cache输出数据做量化操作的参数。不支持非连续，数据格式支持ND，数据类型支持`torch.float`。kv_cache_quant_mode=2时，shape为[1, Dr]；kv_cache_quant_mode=1/3时无需赋值。支持非空Tensor。

- **smooth_scales_cq**（`Tensor`）：可选参数，用于对RmsNorm_cq输出做动态量化操作的参数。不支持非连续，数据格式支持ND，数据类型支持`torch.float`。weight_quant_mode=1/2/4/5时，shape为[1, Hcq]；weight_quant_mode=3时无需赋值。支持非空Tensor。

- **actual_seq_len**（`Tensor`）：可选参数，表示每个batch中的序列长度，以前缀和的形式存储。不支持非连续，数据格式支持ND，数据类型支持`torch.int32`，shape为[B]，支持非空tensor（仅BS合轴且cache_mode为"PA_BLK_BSND"或"PA_BLK_NZ"时需要传入）。当前不会对传入值的合法性进行校验，需用户自行保证。

- **k_nope_clip_alpha**（`Tensor`）：可选参数，表示kv_cache做clip操作时的缩放因子，在部分量化pertoken-pergroup场景和`torch.int8`全量化pertoken-pergroup场景下使用，其他场景无需赋值。不支持非连续，数据格式支持ND，数据类型支持`torch.float`，shape为[1]。

- **rmsnorm_epsilon_cq**（`double`）：可选参数，计算$c^Q$的RmsNorm公式中的$\epsilon$参数。默认值为1e-05。

- **rmsnorm_epsilon_ckv**（`double`）：可选参数，计算$c^{KV}$的RmsNorm公式中的$\epsilon$参数。默认值为1e-05。

- **cache_mode**（`str`）：可选参数，表示kv_cache的模式。可选值为"PA_BSND"、"PA_NZ"、"PA_BLK_BSND"、"PA_BLK_NZ"、"TND"（对应BS合轴）和"BSND"（对应非BS合轴），默认为"PA_BSND"。

- **query_norm_flag**（`bool`）：可选参数，表示是否输出query_norm。仅支持bool类型，False表示不输出query_norm，True表示输出query_norm（量化场景下伴随输出dequant_scale_q_norm），默认值为False。

- **weight_quant_mode**（`int`）：可选参数，表示weight_dq、weight_uq_qr、weight_uk、weight_dkv_kr的量化模式。0表示非量化，1表示weightUqQr量化，2表示weightDq、weightUqQr、weightDkvKr `torch.int8`量化，3表示weightDq、weightUqQr、weightDkvKr mxfp8量化，4表示weightDq、weightUqQr、weightDkvKr fp8量化，5表示weightDq、weightUqQr、weightDkvKr hif8量化，默认值为0。

- **kv_cache_quant_mode**（`int`）：可选参数，表示kv_cache的量化模式。0表示非量化，1表示pertensor量化，2表示perchannel量化，3-表示pertoken-pergroup量化，默认值为0。

- **query_quant_mode**（`int`）：可选参数，表示query的量化模式。0表示非量化，1表示per-token-head量化，默认值为0。

- **ckvkr_repo_mode**（`int`）：可选参数，表示kv_cache和kr_cache的存储模式。0表示kv_cache和kr_cache分别存储，1表示kv_cache和kr_cache合并存储，默认值为0。

- **quant_scale_repo_mode**（`int`）：可选参数，表示量化scale的存储模式。0表示量化scale和数据分别存储，1表示量化scale和数据合并存储，默认值为0。

- **tile_size**（`int`）：可选参数，表示pertoken-pergroup量化时每个tile的大小，仅在kv_cache_quant_mode为3时有效，默认值为128；kv_cache_quant_mode=1/2时无需赋值。

- **qc_qr_scale**（`double`）：可选参数，表示Query的尺度矫正系数，默认值为1.0。

- **kc_scale**（`double`）：可选参数，表示Key的尺度矫正系数，默认值为1.0。

- **token_x_dtype**（`int`）：可选参数，表示参数token_x的传入dtype，在hif8全量化场景为`torch_npu.hifloat8`，其他场景为None。

- **weight_dq_dtype**（`int`）：可选参数，表示参数weight_dq的传入dtype，在hif8全量化场景为`torch_npu.hifloat8`，其他场景为None。

- **weight_uq_qr_dtype**（`int`）：可选参数，表示参数weight_uq_qr的传入dtype，在hif8全量化场景为`torch_npu.hifloat8`，其他场景为None。

- **weight_dkv_kr_dtype**（`int`）：可选参数，表示参数weight_dkv_kr的传入dtype，在hif8全量化场景为`torch_npu.hifloat8`，其他场景为None。

- **kv_cache_dtype**（`int`）：可选参数，表示参数kv_cache的传入dtype，在hif8 kv_cache pertensor量化和hif8 kv_cache pertoken-pergroup量化场景为`torch_npu.hifloat8`，其他场景为None。

  <!-- npu="A3,910b" id5 -->
  > [!NOTE]
  >
  > <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  > 
  > - token_x、weight_dq、weight_uq_qr、weight_dkv_kr、kv_cache不支持`torch.float8_e4m3fn`、`torch_npu.hifloat8`数据类型。
  > - dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr不支持`torch.float8_e8m0fnu`数据类型。
  <!-- end id5 -->

## 返回值说明

- **query_out**（`Tensor`）：表示Query的输出Tensor，即公式中q<sup>N</sup>。数据格式支持ND，dtype支持`torch.bfloat16`、`torch.int8`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。shape支持3维和4维，格式为[T, N, Hckv]和[B, S, N, Hckv]。

- **query_rope_out**（`Tensor`）：表示Query位置编码的输出Tensor，即公式中q<sup>R</sup>。数据格式支持ND，dtype支持`torch.bfloat16`。shape支持3维和4维，格式为[T, N, Dr]和[B, S, N, Dr]。

- **dequant_scale_q_nope**（`Tensor`）：表示Query的输出Tensor的反量化参数。数据格式支持ND，dtype支持`torch.float`。shape支持3维，weightQuantMode=2/3/4/5时，其shape为[T, N, 1]和[B*S, N, 1]；weightQuantMode=0/1时为nullptr。

- **query_norm**（`Tensor`）：Query做RmsNorm_cq后的输出tensor（对应$q^C$）。数据格式支持ND，dtype支持`torch.bfloat16`、`torch.int8`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。shape支持2维和3维，`query_norm_flag=True`时有效，shape为[T, Hcq]或[B, S, Hcq]；`query_norm_flag=False`时为nullptr。

- **dequant_scale_q_norm**（`Tensor`）：Query做RmsNorm_cq后的反量化参数。数据格式支持ND，数据类型支持`torch.float`、`torch.float8_e8m0fnu`。shape支持2维，`query_norm_flag=True`且`weight_quant_mode=1/2/3/4/5`时有效。`weight_quant_mode=0`时为nullptr。`weight_quant_mode=1/2/4/5`时，shape为[T,1]或[B\*S,1]；`weightQuantMode=3`时，shape为[T, Hcq/32]或[B*S, Hcq/32]。

  <!-- npu="A3,910b" id6 -->
  > [!NOTE]
  >
  > <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  > 
  > - query、query_norm不支持`torch.float8_e4m3fn`、`torch_npu.hifloat8`数据类型。
  > - dequant_scale_q_norm不支持`torch.float8_e8m0fnu`数据类型。
  <!-- end id6 -->

## 约束说明

- 该接口支持推理场景下使用。

- 该接口支持单算子模式和图模式。

- shape 格式字段含义说明

    | 字段名       | 英文全称/含义                  | 取值规则与说明                                                                 |
    |--------------|--------------------------------|------------------------------------------------------------------------------|
    | B            | Batch（输入样本批量大小）      | 取值范围：0~65536                                                           |
    | S            | Seq-Length（输入样本序列长度） | 取值范围：不限制                                                              |
    | He           | Head-Size（隐藏层大小）        | 取值固定为：1024、2048、3072、4096、5120、6144、7168、7680、8192                                                      |
    | Hcq          | q 低秩矩阵维度                 | 取值固定为：1536、 2048                                                           |
    | N            | Head-Num（多头数）             | 取值范围：1、2、4、8、16、32、64、128                                       |
    | Hckv         | kv 低秩矩阵维度                | 取值固定为：512                                                             |
    | Dtile        | kv_cache的D轴维度              | 取值固定为：pertoken-pergroup场景下为656，非pertoken-pergroup场景下为512                                                          |
    | D            | qk 不含位置编码维度            | 取值固定为：128、 192                                                             |
    | Dr           | qk 位置编码维度                | 取值固定为：64                                                              |
    | Nkv          | kv 的 head 数                  | 取值固定为：1                                                               |
    | BlockNum     | PagedAttention 场景下pertoken-pergroup量的块数    | 取值为计算 `B*Skv/BlockSize` 的结果后向上取整（Skv 表示 kv 的序列长度，允许取 0） |
    | BlockSize    | PagedAttention 场景下的块大小  | 取值范围：16-1024，且为16的倍数                                                           |
    | T            | BS 合轴后的大小                | 取值范围：不限制；注：若采用 BS 合轴，此时 token_x、rope_sin、rope_cos、query_norm 均为 2 维，query_out、query_rope_out 为 3维，cache_index 为 1 维 |

- 支持场景：

  <!-- npu="A3,910b" id4 -->
  > [!NOTE]
  >
  > <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  > 
  > - 当前不支持 fp8/hif8/mxfp8 全量化场景。
  <!-- end id4 -->

  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">场景</th>
      <th>约束</th>
    </tr>
    <tr>
      <td colspan="2"><em>非量化</em></td>
      <td>
          weight_quant_mode=0, kv_cache_quant_mode=0, query_quant_mode=0<br>
          入参：所有入参皆为非量化数据<br>
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3"><em>部分量化</em></td>
      <td>kv_cache非量化 </td>
      <td>
          weight_quant_mode=1, kv_cache_quant_mode=0, query_quant_mode=0<br>
          入参：weight_uq_qr传入pertoken量化数据，其余入参皆为非量化数据。dequant_scale_w_uq_qr字段必须传入，smooth_scales_cq字段可选传入<br>
          出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kv_cache perchannel量化 </td>
      <td>
          weight_quant_mode=1, kv_cache_quant_mode=2, query_quant_mode=0<br>
          入参：weight_uq_qr传入pertoken量化数据，kv_cache、kr_cache传入perchannel量化数据，其余入参皆为非量化数据 <br>
          dequant_scale_w_uq_qr、quant_scale_ckv、quant_scale_ckr字段必须传入，smooth_scales_cq字段可选传入 <br>
          出参：kv_cache、kr_cache返回perchannel量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kv_cache pertoken-pergroup量化 </td>
      <td>
          weight_quant_mode=1, kv_cache_quant_mode=3, query_quant_mode=0<br>
          入参：weight_uq_qr传入pertoken量化数据，kv_cache传入pertoken-pergroup量化数据，其余入参皆为非量化数据 <br>
          dequant_scale_w_uq_qr字段必须传入，smooth_scales_cq字段可选传入 <br>
          出参：kv_cache_out返回pertoken-pergroup量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3"><em>`torch.int8`/fp8/hif8全量化</em></td>
      <td> kv_cache非量化</td>
      <td>
          weight_quant_mode=2/4/5, kv_cache_quant_mode=0, query_quant_mode=0<br>
          入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，其余入参皆为非量化数据 <br>
          dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr字段必须传入，smooth_scales_cq字段可选传入 <br>
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache pertensor量化 </td>
      <td>
          weight_quant_mode=2/4/5, kv_cache_quant_mode=1, query_quant_mode=1<br>
          入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，kv_cache传入pertensor量化数据，其余入参皆为非量化数据 <br>
          dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr、quant_scale_ckv字段必须传入，smooth_scales_cq字段可选传入 <br>
          出参：query_out返回pertoken_head量化数据，kv_cache出参返回pertensor量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache pertoken-pergroup量化 </td>
      <td>
          weight_quant_mode=2/4/5, kv_cache_quant_mode=3, query_quant_mode=0<br>
          入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，其余入参皆为非量化数据 <br>
          dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr字段必须传入，smooth_scales_cq字段可选传入 <br>
          出参：query_out返回pertoken_head量化数据，kv_cache出参返回pertoken-pergroup量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3"><em>mxfp8全量化</em></td>
      <td> kv_cache非量化</td>
      <td>
        weight_quant_mode=3, kv_cache_quant_mode=0, query_quant_mode=0<br>
        入参：tokenX传入pertoken量化数据，weightDq、weightUqQr、weightDkvKr传入perchannel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入 <br>
        出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache pertensor量化 </td>
      <td>
        weight_quant_mode=3, kv_cache_quant_mode=1, query_quant_mode=1<br>
        入参：tokenX传入pertoken量化数据，weightDq、weightUqQr、weightDkvKr传入perchannel量化数据，kvCacheRef传入pertensor量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr、quantScaleCkv字段必须传入 <br>
        出参：query_out返回per-token-head量化数据，kv_cache出参返回pertensor量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache pertoken-pergroup量化 </td>
      <td>
        weight_quant_mode=3, kv_cache_quant_mode=3, query_quant_mode=0<br>
        入参：tokenX传入pertoken量化数据，weightDq、weightUqQr、weightDkvKr传入perchannel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入 <br>
        出参：kv_cache出参返回pertoken-pergroup量化数据，其余出参返回非量化数据
      </td>
    </tr>
  </table>

- 在不同量化场景下，参数的dtype需要满足如下条件：
  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="2">参数名</th>
      <th><em>非量化</em></th>
      <th colspan="3"><em>部分量化</em></th>
      <th colspan="3"><em>`torch.int8`全量化</em></th>
      <th colspan="3"><em>mxfp8全量化</em></th>
      <th colspan="3"><em>fp8全量化</em></th>
      <th colspan="3"><em>hif8全量化</em></th>
    </tr>
    <tr>
      <th>dtype</th>
      <th>kv_cache非量化<br>dtype</th>
      <th>kv_cache量化<br>dtype</th>
      <th>kv_cache pertoken-pergroup量化<br>dtype</th>
      <th>kv_cache非量化<br>dtype</th>
      <th>kv_cache量化<br>dtype</th>
      <th>kv_cache pertoken-pergroup量化<br>dtype</th>
      <th>kvCache非量化<br>dtype</th>
      <th>kvCache pertensor量化<br>dtype</th>
      <th>kvCache pertoken-pergroup量化<br>dtype</th>
      <th>kvCache非量化<br>dtype</th>
      <th>kvCache pertensor量化<br>dtype</th>
      <th>kvCache pertoken-pergroup量化<br>dtype</th>
      <th>kvCache非量化<br>dtype</th>
      <th>kvCache pertensor量化<br>dtype</th>
      <th>kvCache pertoken-pergroup量化<br>dtype</th>
    </tr>
    <tr>
      <td>token_x</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
    </tr>
    <tr>
      <td>weight_dq</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
    </tr>
    <tr>
      <td>weight_uq_qr</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
    </tr>
    <tr>
      <td>weight_uk</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
    </tr>
    <tr>
      <td>weight_dkv_kr</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
    </tr>
    <tr>
      <td> rmsnorm_gamma_cq </td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
    </tr>
    <tr>
      <td> rmsnorm_gamma_ckv </td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
    </tr>
    <tr>
      <td> rope_sin </td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
    </tr>
    <tr>
      <td> rope_cos </td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
    </tr>
    <tr>
      <td> kv_cache </td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
    </tr>
    <tr>
      <td> kr_cache </td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.int8`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
    </tr>
    <tr>
      <td> cache_index </td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
      <td>`torch.int64`</td>
    </tr>
    <tr>
      <td> dequant_scale_x </td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
    </tr>
    <tr>
      <td> dequant_scale_w_dq </td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
    </tr>
    <tr>
      <td> dequant_scale_w_uq_qr </td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
    </tr>
    <tr>
      <td> dequant_scale_w_dkv_kr </td>
      <td> 无需赋值 </td>
      <td> 无需赋值 </td>
      <td> 无需赋值 </td>
      <td> 无需赋值 </td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
    </tr>
    <tr>
      <td> quant_scale_ckv </td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
    </tr>
    <tr>
      <td> quant_scale_ckr </td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
    </tr>
    <tr>
      <td> smooth_scales_cq </td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
    </tr>
    <tr>
      <td> actual_seq_len </td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
      <td>`torch.int32`</td>
    </tr>
    <tr>
      <td> k_nope_clip_alpha </td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
    </tr>
    <tr>
      <td> query_out </td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.int8`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch.bfloat16`</td>
    </tr>
    <tr>
      <td> query_rope_out </td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
      <td>`torch.bfloat16`</td>
    </tr>
    <tr>
      <td> dequant_scale_q_nope_out </td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>无需赋值</td>
    </tr>
    <tr>
      <td> query_norm_out </td>
      <td>`torch.bfloat16`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.int8`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch.float8_e4m3fn`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
      <td>`torch_npu.hifloat8`</td>
    </tr>
    <tr>
      <td> dequant_scale_q_norm_out </td>
      <td>无需赋值</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float8_e8m0fnu`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
      <td>`torch.float`</td>
    </tr>
  </table>
  </div>

## 调用示例

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
    cache_index = torch.randint(0, BlockNum * BlockSize, (B, S), dtype=torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"

    # 调用MlaProlog算子
    query_mla, query_rope_mla, dequant_scale_q_nope_mla, query_norm_mla, dequant_scale_q_norm_mla = torch_npu.npu_mla_prolog_v3(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
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
    cache_index = torch.randint(0, BlockNum * BlockSize, (B, S), dtype=torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_mla_prolog_v3(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        query_mla, query_rope_mla, dequant_scale_q_nope_mla, query_norm_mla, dequant_scale_q_norm_mla = torch_npu.npu_mla_prolog_v3(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
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
