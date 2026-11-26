# torch_npu.npu_grouped_matmul_swiglu_quant_v2

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

- API功能：`torch_npu.npu_grouped_matmul_swiglu_quant_v2`是一种融合分组矩阵乘法（GroupedMatmul）、SwiGLU混合激活函数、量化（quant）的计算方法。该方法适用于需要对矩阵乘法结果进行SwiGLU激活函数激活的场景，融合算子在底层能够对部分过程并行，达到性能优化的效果。

- 计算公式：

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
    <details>
    <summary>量化场景A8W8（A指激活矩阵，W指权重矩阵，8指`torch.int8`数据类型）：</summary>

    - **输入**：
      - $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
      - $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
      - $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵的逐通道缩放因子。
      - $x\_scale∈\mathbb{R}^{M}$：激活矩阵的逐token缩放因子。
      - $groupList∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
    - **输出**：
      - $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
      - $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
    - **计算过程**：
      1. 根据groupList\[i\]确定当前分组的token，$i \in [0,Len(groupList)]$。
      2. 根据分组确定的入参进行如下计算：

          $$
          C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}
          $$

          $$
          C_{i,act}, gate_{i} = split(C_{i})
          $$

          $$
          S_{i}=Swish(C_{i,act})\odot gate_{i}
          $$

          其中$Swish(x)=\frac{x}{1+e^{-x}}$

      3. 量化输出结果：

          $$
          Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
          $$

          $$
          Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
          $$

    </details>

    <details>
    <summary>MSD场景A8W4（A指激活矩阵，W指权重矩阵，8指`torch.int8`数据类型，4指`torch.int4`数据类型）：</summary>

    - **输入**：
      - $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
      - $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
      - $weightAssistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（离线生成，非算子内部完成）。
      - $w\_scale$：分组权重矩阵的缩放因子，perchannel时shape为$\mathbb{R}^{E \times N}$，pergroup时shape为$\mathbb{R}^{E \times K\_group\_num \times N}$。
      - $x\_scale∈\mathbb{R}^{M}$：激活矩阵的逐token缩放因子。
      - $groupList∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
    - **输出**：
      - $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
      - $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
    - **计算过程**：
      1. 根据groupList\[i\]确定当前分组的token，分组逻辑与A8W8相同。
      2. 将左矩阵`torch.int8`拆为高低4bit两部分：

          $$
          X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor
          $$

          $$
          X\_low\_4bits_{i} = X_{i}\ \&\ 0x0f - 8
          $$

      3. 分别与权重做矩阵乘并应用perchannel或pergroup量化缩放，合并高低位结果：

          $$
          C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}
          $$

          $$
          C_{i,act}, gate_{i} = split(C_{i})
          $$

          $$
          S_{i}=Swish(C_{i,act})\odot gate_{i}
          $$

          $$
          Swish(x)=\frac{x}{1+e^{-x}}
          $$

      4. 量化输出结果：

          $$
          Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
          $$

          $$
          Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
          $$

    </details>

    <details>
    <summary>量化场景A4W4（A指激活矩阵，W指权重矩阵，4指`torch.int4`数据类型）：</summary>

    - **输入**：
      - $X∈\mathbb{Z_4}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
      - $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
      - $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵的逐通道缩放因子。
      - $x\_scale∈\mathbb{R}^{M}$：激活矩阵的逐token缩放因子。
      - $smoothScale∈\mathbb{R}^{E \times N/2}$：平滑缩放因子，E是专家个数，N是输出维度。支持shape为(E,)时广播。
      - $groupList∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
    - **输出**：
      - $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
      - $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
    - **计算过程**：
      1. 根据groupList\[i\]确定当前分组的token，分组逻辑与A8W8相同。
      2. 根据分组确定的入参进行如下计算：

          $$
          C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}
          $$

          $$
          C_{i,act}, gate_{i} = split(C_{i})
          $$

          $$
          S_{i}=Swish(C_{i,act})\odot gate_{i}
          $$

          $$
          Swish(x)=\frac{x}{1+e^{-x}}
          $$

          $$
          S_{i} = S_{i} \odot smoothScale_{i\ Broadcast}
          $$

          注：当smoothScale形状为(E,)时，会对其进行广播，使其与$S_{i}$的形状匹配。

      3. 量化输出结果：

          $$
          Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
          $$

          $$
          Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
          $$

    </details>
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    <details>
    <summary>MX量化场景：</summary>

      1. 根据分组确定的入参进行如下计算：

         $$
         C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}
         $$

         $$
         C_{i,act}, gate_{i} = split(C_{i})
         $$

         $$
         S_{i}=Swish(C_{i,act})\odot gate_{i}
         $$

         $$
         Swish(x)=\frac{x}{1+e^{-x}}
         $$

      2. 量化输出结果：

         $$
         shared\_exp = \left\lfloor \log_2(max_i(|S_i|)) \right\rceil - emax
         $$

         $$
         Q\_scale = 2 ^ {shared\_exp}
         $$

         $$
         Q_i = quantize\_to\_element\_format(S_i/Q\_scale), \space i\space from\space 1\space to\space blocksize
         $$

         其中，$emax$表示对应数据类型的最大正则数的指数位：

         |   DataType    | emax |
         | :-----------: | :--: |
         | `torch.float8_e4m3fn` |  8   |
         | `torch.float8_e5m2`  |  15  |
         | `torch.float4_e2m1fn_x2`  |  2   |

         其中，$blocksize$表示每次量化的元素个数，仅支持32。

    </details>

    <details>
    <summary>A8W8 Pertoken量化场景：</summary>

      - **定义**：
        - **⋅** 表示矩阵乘法。
        - **⊙** 表示逐元素乘法。
      - **输入**：
        - $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        - $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        - $w\_scale∈\mathbb{R}^{E \times N}$：权重矩阵的逐channel缩放因子。
        - $x\_scale∈\mathbb{R}^{M}$：激活矩阵的逐token缩放因子。
        - $groupList∈\mathbb{N}^{E}$：cumsum或count形式的分组索引列表。
      - **输出**：
        - $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        - $Q\_scale∈\mathbb{R}^{M}$：输出矩阵的逐token量化缩放因子。
      - **计算过程**：

      1. 根据groupList\[i\]确定当前分组的token，$i \in [0,Len(groupList)]$。

      2. 根据分组确定的入参进行如下计算：

         $$
         C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i} \odot w\_scale_{i}
         $$

         $$
         C_{i,act}, gate_{i} = split(C_{i})
         $$

         $$
         S_{i}=Swish(C_{i,act})\odot gate_{i}
         $$

         $$
         Swish(x)=\frac{x}{1+e^{-x}}
         $$

         其中，$x\_scale_{i}$表示对应token的量化因子。

      3. 量化输出结果：

         $$
         Q\_scale_{i} = \frac{max(|S_{i}|)}{max(type)}
         $$

         $$
         Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
         $$

    </details>
  <!-- end id5 -->

## 函数原型

```python
torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, x_scale, group_list, *, smooth_scale=None, weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=6, quant_mode=0, quant_dtype=0, group_list_type=0, tuning_config=None, x_dtype=None, weight_dtype=None, weight_scale_dtype=None, x_scale_dtype=None) -> (Tensor, Tensor)
```

## 参数说明

- **`x`**（`Tensor`）：**必选参数**，矩阵乘法的左矩阵。`shape`支持2维\[m, k\]，数据格式支持$ND$，支持非连续的`Tensor`。

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int8`。
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch.float4_e2m1fn_x2`、`torch.int8`、`torch_npu.hifloat8`、`torch_npu.float4_e1m2fn_x2`（仅`weight`为$FRACTAL\_NZ$格式时支持）。其中`torch_npu.hifloat8`和`torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`系列需配置可选参数`x_dtype`为对应类型，此时输入`x`自身的`dtype`不再生效，但仍需保证输入`x`自身的`dtype`为8 bit数据类型，以保证`shape`正确；其中float4内轴`K`需为偶数，以保证8 bit数据可以转换为2个float4。数据格式支持$ND$。
  <!-- end id7 -->

- **`weight`**（`List[Tensor]`）：**必选参数**，权重矩阵（矩阵乘法右矩阵），支持非连续的`Tensor`。

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int8`。
    - 数据格式为$ND$时，`shape`支持3维\[e, k, n\]。
    - 数据格式为$FRACTAL\_NZ$（通过接口`npu_format_cast`可实现格式转换）时，`shape`支持5维。以非转置为例，`torch.float8_e4m3fn`场景的`shape`为\[e, k/32, n/16, 16, 32\]，`torch.float4_e2m1fn_x2`场景的`shape`为\[e, k/64, n/16, 16, 64\]。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持单个`Tensor`（`Tensor`列表长度必须为1）和多个`Tensor`（`Tensor`列表长度为e）。
    - 数据格式为$ND$时，`shape`支持3维，非转置`shape`为\[\[e, k, n\]\]，转置`shape`为\[\[e, n, k\]\]。数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch.float4_e2m1fn_x2`、`torch.int8`、`torch_npu.hifloat8`，其中`torch_npu.hifloat8`和`torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`系列需配置可选参数`weight_dtype`为对应类型，此时输入`weight`自身的`dtype`不再生效，但仍需保证输入`weight`自身的`dtype`为8 bit数据类型，以保证`shape`正确；其中float4内轴需为偶数，以保证8 bit数据可以转换为2个float4。
    - 数据格式为$FRACTAL\_NZ$（通过接口`npu_format_cast`可实现格式转换）时，单单单场景其中`Tensor`的`shape`支持5维，单多单场景其中`Tensor`的`shape`支持4维；数据类型仅支持`torch.float8_e4m3fn`、`torch_npu.float4_e1m2fn_x2`、`torch.float4_e2m1fn_x2`。
  <!-- end id9 -->

- **`weight_scale`**（`List[Tensor]`）：**必选参数**，右矩阵的量化因子。数据格式支持$ND$，支持非连续的`Tensor`。

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`shape`支持2维\[\[e, n\]\]，数据类型支持`torch.float32`。
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：MX量化场景下，单多单且`weight`数据格式为$FRACTAL\_NZ$时，其中`Tensor`的`shape`支持3维；其余场景其中`Tensor`的`shape`支持4维。数据类型支持`torch.float8_e8m0fnu`。Pertoken量化场景下，`shape`支持2维，`shape`为\[\[e, n\]\]；当`x`为`torch.int8`时，`weight_scale`需支持`torch.bfloat16`、`torch.float32`、`torch.float16`；当`x`为`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`时，`weight_scale`支持`torch.bfloat16`、`torch.float32`。目前仅支持`Tensor`列表长度为1。
  <!-- end id11 -->

- **`x_scale`**（`Tensor`）：**必选参数**，左矩阵的量化因子。数据格式支持$ND$，支持非连续的`Tensor`。

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`shape`支持1维\[m\]，数据类型支持`torch.float32`。
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT系列产品</term>：MX量化场景下，`shape`支持3维\[m, ceil\(k / 64\), 2\]，数据类型支持`torch.float8_e8m0fnu`；Pertoken量化场景下，`shape`支持1维\[m\]，数据类型支持`torch.float32`。
  <!-- end id13 -->

- **`*`**：代表`*`之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **`group_list`**（`Tensor`）：**必选参数**，指示每个分组参与计算的Token个数。`shape`支持1维\[e\]，数据类型支持`torch.int64`，数据格式支持$ND$，支持非连续的`Tensor`。当`group_list_type`为0时，最后一个值不大于输入`x`中`Tensor`的第一维；当`group_list_type`为1时，数值的总和不大于输入`x`中`Tensor`的第一维。`group_list`中的值约束了输出数据的有效部分，`group_list`中未指定的部分将不会参与更新。

- **`smooth_scale`**（`Tensor`）：**可选参数**，平滑缩放因子。数据类型为`torch.float32`，数据格式支持$ND$，当前仅支持传入默认值`None`。

- **`weight_assist_matrix`**（`List[Tensor]`）：**可选参数**，右矩阵的辅助矩阵。数据类型支持`torch.float32`，数据格式支持$ND$，当前仅支持传入默认值`None`。

- **`bias`**（`Tensor`）：**可选参数**，矩阵乘计算的偏移值。数据类型支持`torch.int32`，当前仅支持传入默认值`None`。

- **`dequant_mode`**（`int`）：**可选参数**，表示反量化模式，数据类型为`torch.int32`，默认值为`0`。取值为`0`时，表示激活矩阵pertoken、权重矩阵perchannel；取值为`1`时，表示激活矩阵pertoken、权重矩阵pergroup；取值为`2`时，表示MX量化。

  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当前仅支持传入默认值`0`。
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当前仅支持传入`0`以及`2`。
  <!-- end id15 -->

- **`dequant_dtype`**（`int`）：**可选参数**，表示反量化类型，数据类型为`torch.int32`。

  <!-- npu="A3,910b" id16 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：预留输入，当前仅支持传入默认值`torch.float32`。
  <!-- end id16 -->
  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT系列产品</term>：默认值为`torch.float32`，当前仅支持传入`torch.float32`、`torch.bfloat16`、`torch.float16`。
  <!-- end id17 -->

- **`quant_mode`**（`int`）：**可选参数**，表示SwiGLU后的量化模式，数据类型为`torch.int32`。支持取值：`0`（默认值）表示pertoken量化；`1`表示pergroup量化；`2`表示MX量化。

  <!-- npu="A3,910b" id18 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当前仅支持传入默认值`0`。
  <!-- end id18 -->
  <!-- npu="950" id19 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当前仅支持传入`0`以及`2`。
  <!-- end id19 -->

- **`quant_dtype`**（`int`）：**可选参数**，表示量化后低比特数据类型，数据类型为`torch.int32`。

  <!-- npu="A3,910b" id20 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当前仅支持传入默认值`torch.int8`。
  <!-- end id20 -->
  <!-- npu="950" id21 -->
  - <term>Ascend 950PR&950DT系列产品</term>：默认值为`torch.int8`，当前支持传入`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`、`torch.int8`、`torch_npu.hifloat8`。
  <!-- end id21 -->

- **`group_list_type`**（`int`）：**可选参数**，表示`group_list`的输入类型，数据类型为`torch.int32`，默认值为`0`。
  - 取值为`0`时，表示cumsum模式，`group_list`中的每个元素代表当前分组的累计长度。
  - 取值为`1`时，表示count模式，`group_list`中的每个元素代表该分组包含的元素个数。

- **`tuning_config`**（`List[int]`）：**可选参数**，数组中的第一个元素表示各个专家处理的token数的预期值。从第二个元素开始预留，用户无须填写，未来会进行扩展。默认值为`None`。

- **`x_dtype`**（`int`）：**可选参数**，指定输入`x`的真实数据类型。当前仅支持默认值`None`，表示输入`x`真实的`dtype`。

  <!-- npu="A3,910b" id22 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数，使用默认值。
  <!-- end id22 -->
  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当`x`为`torch.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`、`torch_npu.hifloat8`时，`x_dtype`需要传入`torch.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`、`torch_npu.hifloat8`。
  <!-- end id23 -->

- **`weight_dtype`**（`int`）：**可选参数**，指定输入`weight`的真实数据类型。当前仅支持默认值`None`，表示输入`weight`真实的`dtype`。

  <!-- npu="A3,910b" id24 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数，使用默认值。
  <!-- end id24 -->
  <!-- npu="950" id25 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当`weight`为`torch.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`、`torch_npu.hifloat8`时，`weight_dtype`需要传入`torch.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`、`torch_npu.hifloat8`。
  <!-- end id25 -->

- **`weight_scale_dtype`**（`int`）：**可选参数**，指定输入`weight_scale`的真实数据类型。默认值为`None`，表示输入`weight_scale`真实的`dtype`。

  <!-- npu="A3,910b" id26 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数，使用默认值。
  <!-- end id26 -->
  <!-- npu="950" id27 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当`weight_scale`为`torch.float8_e8m0fnu`时，`weight_scale_dtype`需要传入`torch.float8_e8m0fnu`。
  <!-- end id27 -->

- **`x_scale_dtype`**（`int`）：**可选参数**，指定输入`x_scale`的真实数据类型。默认值为`None`，表示输入`x_scale`真实的`dtype`。

  <!-- npu="A3,910b" id28 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数，使用默认值。
  <!-- end id28 -->
  <!-- npu="950" id29 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当`x_scale`为`torch.float8_e8m0fnu`时，`x_scale_dtype`需要传入`torch.float8_e8m0fnu`。
  <!-- end id29 -->

## 返回值说明

- **`output`**（`Tensor`）：输出的量化结果。数据格式支持$ND$，支持非连续的`Tensor`。

  <!-- npu="A3,910b" id30 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int8`，`shape`支持2维\[m, n / 2\]。
  <!-- end id30 -->
  <!-- npu="950" id31 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch.float4_e2m1fn_x2`、`torch.int8`、`torch_npu.hifloat8`、`torch_npu.float4_e1m2fn_x2`（仅`weight`为$FRACTAL\_NZ$格式时支持），`shape`支持2维\[m, n / 2\]。
  <!-- end id31 -->

- **`output_scale`**（`Tensor`）：输出的量化因子。数据格式支持$ND$，支持非连续的`Tensor`。

  <!-- npu="A3,910b" id32 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float`，`shape`支持1维\[m\]。
  <!-- end id32 -->
  <!-- npu="950" id33 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - MX量化场景：数据类型支持`torch.float8_e8m0fnu`，`shape`支持3维\[m, ceil\(\(n / 2\) / 64\), 2\]。
    - Pertoken量化场景：`shape`支持1维\[m\]，数据类型支持`torch.float32`。
  <!-- end id33 -->

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- group\_list第1维最大支持1024，即最多支持1024个group。

<!-- npu="950" id34 -->
- WeightNZ场景说明（仅适用于Ascend 950PR&950DT系列产品）：
  - MXFP4、MXFP8场景支持静态图模式，不支持动态图模式。
  - 单多单MXFP4、MXFP8场景支持单算子模式，不支持图模式。
  - MX量化、`weight`为ND格式场景下，当输入为`torch.float8_e4m3fn`或`torch.float8_e5m2`数据类型时，需满足N为2对齐；当输入为`torch.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`数据类型时，需满足N为4对齐。
  - MX量化、`weight`为NZ格式场景下，当输入为`torch.float8_e4m3fn`或`torch.float8_e5m2`数据类型时，需满足N为64对齐；当输入为`torch.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`数据类型时，需满足N为128对齐。
  - MXFP4场景不支持k=2，MXFP4场景需满足K为偶数。
<!-- end id34 -->
- 参数说明里Shape使用的变量说明：
  - e：表示分组数目，取值范围为1-1024。
  - m：输出矩阵output的倒数第二维大小，取值范围为1-2147483647。
  - n：输出矩阵output的倒数第一维大小的两倍，取值范围为1-2147483647。

    MX量化场景下要求（仅适用于Ascend 950PR&950DT系列产品）：

    - MX量化、`weight`为ND格式场景下，当输入为`torch.float8_e4m3fn`或`torch.float8_e5m2`数据类型时，需满足`n`为2对齐；当输入为`torch.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`数据类型时，需满足`n`为4对齐。
    - MX量化、`weight`为NZ格式场景下，当输入为`torch.float8_e4m3fn`或`torch.float8_e5m2`数据类型时，需满足`n`为64对齐；当输入为`torch.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`数据类型时，需满足`n`为128对齐。

  - k：矩阵乘法reduce轴的大小，取值范围为1-2147483647。

- 输入和输出Tensor支持的数据类型组合如下：

    <!-- npu="A3,910b" id44 -->
    **表 1** Atlas A2系列产品、Atlas A3系列产品

    | x | weight | group_list | weight_scale | x_scale | bias | weight_assist_matrix | smooth_scale | output | output_scale |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32` | `torch.float32` | `torch.int32` | `torch.float32` | `torch.float32` | `torch.int8` | `torch.float32` |
    <!-- end id44 -->

    <!-- npu="950" id45 -->
    **表 2** Ascend 950PR&950DT系列产品

    | 量化模式 | x | weight | group_list | weight_scale | x_scale | bias | weight_assist_matrix | smooth_scale | output | output_scale |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | MXFP8量化（ND格式） | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e8m0fnu` |
    | MXFP4量化（ND格式） | `torch.float4_e2m1fn_x2` | `torch.float4_e2m1fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` |
     | MXFP4量化（ND格式） | `torch.float4_e2m1fn_x2` | `torch.float4_e2m1fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e8m0fnu` |
    | MXFP8量化（FRACTAL_NZ格式） | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` |
    | MXFP4量化（FRACTAL_NZ格式） | `torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` | `torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float4_e2m1fn_x2`/`torch.float8_e4m3fn`/`torch_npu.float4_e1m2fn_x2` | `torch.float8_e8m0fnu` |
    | Pertoken量化 | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.int8` | `torch.float32` |
    | Pertoken量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.int64` | `torch.float32`/`torch.bfloat16` | `torch.float32` | 暂不支持 | 暂不支持 | 暂不支持 | `torch_npu.hifloat8` | `torch.float32` |
    | Pertoken量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.int64` | `torch.float32`/`torch.bfloat16` | `torch.float32` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` |
    | MxFP8FP4量化（FRACTAL_NZ格式） | `torch.float8_e4m3fn` | `torch.float4_e2m1fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` |

    > **MxA8W4场景**：
    > - `x`数据类型为`torch.float8_e4m3fn`，`weight`数据类型为`torch.float4_e2m1fn_x2`。`weight`数据格式要求$FRACTAL\_NZ$格式，可通过`torch_npu.npu_format_cast`接口实现$ND$转$FRACTAL\_NZ$格式。`k`要求32对齐，N要求128对齐。
    > - 支持单单单和单多单场景。
    <!-- end id45 -->

- 根据输入x、输入weight与输出y的Tensor数量不同，支持以下几种场景。场景中的“单”表示单个张量，“多”表示多个张量。场景顺序为x、weight、y，例如“单多单”表示x为单张量，weight为多张量，y为单张量。

    | 支持场景 | 场景说明 | 场景限制 |
    | --- | --- | --- |
    | 单多单 | x为单张量，weight为多张量，y为单张量。 | 1. 必须传group_list，且最后一个值与x中tensor的第一维相等。<br>2. x、weight、y中tensor需为2维。<br>3. weight中每个tensor的N轴必须相等。<br>4. 必须传group_list，且当group_list_type为0时，最后一个值与x中tensor的第一维相等，当group_list_type为1时，数值的总和需与x中tensor的第一维一一对应且长度最大为128 |

<!-- npu="950" id46 -->
- 输入和输出Tensor支持的shape组合如下：

    **表 3** Ascend 950PR&950DT系列产品

    | 支持场景 | 量化模式 | x | weight | weight_scale | xScale | output | outputscale |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | 单单单 | MX量化（ND格式） | (m, k) | <li>非转置shape形如{(e, k, n)}</li><li>转置shape形如{(e, n, k)}</li> | <li>非转置shape形如{(e, ceil(k / 64), n, 2)}</li><li>转置shape形如{(e, n, ceil(k / 64), 2)}</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | 单单单 | MXFP8量化（FRACTAL_NZ格式） | (m, k) | <li>非转置shape形如{(e, n/32, k/16, 16, 32)}</li><li>转置shape形如{(e, k/32, n/16,16, 32)}</li> | <li>非转置shape形如{(e, ceil(k / 64), n, 2)}</li><li>转置shape形如{(e, n, ceil(k / 64), 2)}</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | 单单单 | MXFP4量化（FRACTAL_NZ格式） | (m, k) | <li>非转置shape形如{(e, n/64, k/16, 16, 64)}</li><li>转置shape形如{(e, k/64, n/16,16, 64)}</li> | <li>非转置shape形如{(e, ceil(k / 64), n, 2)}</li><li>转置shape形如{(e, n, ceil(k / 64), 2)}</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | 单单单 | Pertoken量化 | (m, k) | <li>非转置shape形如{(e, k, n)}</li><li>转置shape形如{(e, n, k)}</li> | shape形如{(e, n)} | (m, ) | (m, n / 2) | (m, ) |
    | 单单单 | MxFP8FP4量化（FRACTAL_NZ格式） | (m, k) | 转置shape形如{(e, k/32, n/16,16, 32)} | 转置shape形如{(e, n, ceil(k / 64), 2)} | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | 单多单 | MXFP8量化（FRACTAL_NZ格式） | (m, k) | <li>非转置shape形如{e个(n/32, k/16, 16, 32)}</li><li>转置shape形如{e个(k/32, n/16,16, 32)}</li> | <li>非转置shape形如{e个(ceil(k / 64), n, 2)}</li><li>转置shape形如{e个(n, ceil(k / 64), 2)}</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | 单多单 | MXFP4量化（FRACTAL_NZ格式） | (m, k) | <li>非转置shape形如{e个(n/64, k/16, 16, 64)}</li><li>转置shape形如{e个(k/64, n/16,16, 64)}</li> | <li>非转置shape形如{e个(ceil(k / 64), n, 2)}</li><li>转置shape形如{e个(n, ceil(k / 64), 2)}</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
<!-- end id46 -->

## 调用示例

- 单算子模式调用

  <!-- npu="A3,910b" id35 -->
  - Atlas A2系列产品、Atlas A3系列产品：

    ```python
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax

    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weight_scale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList
    E = 2
    M = 512
    K = 7168
    N = 4096
    x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
    output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(), [weight_npu], [weight_scale.npu()], xScale.npu(), groupList.npu())
    ```
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - Ascend 950PR&950DT系列产品：mx量化场景示例-mxfp8

    ```python
    import unittest
    import itertools
    import numpy as np
    import torch
    import torch_npu
    import math

    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)

        weight_scale = torch.randint(low=-128, high=127, size=(E, math.ceil(K / 64), N, 2), dtype=torch.int8)
        xScale = torch.randint(low=-128, high=127, size=(M, math.ceil(K / 64), 2), dtype=torch.int8)
        groupList = torch.tensor([int(M/2), int(M/2)], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList
    K = 2
    E = 2
    M = 16
    N = 128
    x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = weight.npu()
    weight_scale = weight_scale.npu()
    output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(),
    [weight_npu], [weight_scale],
    xScale.npu(), groupList.npu(),
    dequant_mode = 2,
    quant_mode=2,
    dequant_dtype=torch.float32,
    quant_dtype=torch.float8_e4m3fn,
    weight_scale_dtype=torch_npu.float8_e8m0fnu,
    x_scale_dtype=torch_npu.float8_e8m0fnu)
    ```
  <!-- end id36 -->
  <!-- npu="950" id37 -->
  - Ascend 950PR&950DT系列产品：mx量化场景示例-mxfp4

    ```python
    import numpy as np
    import torch
    import torch_npu
    import math

    K = 9
    E = 2
    M = 2255
    N = 896

    x = torch.randint(0, 256, (M, K), dtype=torch.uint8).npu()
    weight = torch.randint(0, 256, (E, K * 2, N), dtype=torch.uint8).npu()
    weight_scale = torch.randint(0, 256, (E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8).npu()
    xScale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
    groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64).npu()

    y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x,
        [weight], [weight_scale],
        xScale, groupList,
        dequant_mode=2,
        dequant_dtype=torch.float32,
        quant_mode=2,
        quant_dtype=torch_npu.float4_e2m1fn_x2,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu,
        x_dtype=torch_npu.float4_e2m1fn_x2,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        group_list_type=1)

    print("y.shape: ", y.shape)
    print("y_scale.shape: ", y_scale.shape)
    ```
  <!-- end id37 -->
  <!-- npu="950" id38 -->
  - Ascend 950PR&950DT系列产品：Pertoken量化场景示例

    ```python
    import numpy as np
    import torch
    import torch_npu
    import math
    K = 9
    E = 2
    M = 2255
    N = 896
    x = torch.randint(0, 256, (M, K), dtype=torch.uint8).to(torch.float8_e5m2).npu()
    weight = torch.randint(0, 256, (E, K, N), dtype=torch.uint8).to(torch.float8_e5m2).npu()
    weight_scale = torch.randint(0, 256, (E, N), dtype=torch.float).npu()
    xScale = torch.randint(0, 256, (M,), dtype=torch.float).npu()
    groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64).npu()
    y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x,
        [weight], [weight_scale],
        xScale, groupList,
        dequant_mode=0,
        quant_mode=0,
        quant_dtype=torch_npu.float8_e5m2,
        dequant_dtype=torch.float,
        group_list_type=1)
    print("y.shape: ", y.shape)
    print("y_scale.shape: ", y_scale.shape)
    ```
  <!-- end id38 -->
  <!-- npu="950" id39 -->
  - Ascend 950PR&950DT系列产品：MxA8W4伪量化场景示例

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    from ml_dtypes import float8_e4m3fn

    def ceil_div(a, b):
        return math.ceil(a / b)
    class NetPTA(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, x_scale, weight_scale, group_list, group_list_type, dequant_mode, dequant_dtype,
                    quant_mode, quant_dtype):
            weight = weight.transpose(-1, -2)
            weight_scale = weight_scale.transpose(-2, -3)
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, [weight],
                                                                    weight_scale=[weight_scale],
                                                                    bias=None,
                                                                    x_scale=x_scale,
                                                                    dequant_mode=dequant_mode,
                                                                    dequant_dtype=dequant_dtype,
                                                                    quant_mode=quant_mode,
                                                                    quant_dtype=quant_dtype,
                                                                    group_list_type=group_list_type,
                                                                    group_list=group_list,
                                                                    weight_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    x_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    weight_dtype=torch_npu.float4_e2m1fn_x2)
            return output

    def fp32_to_fp4_e2m1_u8packed(tensor_in):
        fp4_values = np.array([
            +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
        ], dtype=np.float32)
        x = tensor_in.numpy()
        x_flat = x.reshape(-1, 1)  # (N, 1)
        fp4_values = fp4_values.reshape(1, -1)  # (1, M)

        dist = np.abs(x_flat - fp4_values)  # (N, M)
        indices = np.argmin(dist, axis=1)  # (N,)
        tmp = indices.reshape(-1, 2)
        packed = tmp[:, 0] + (tmp[:, 1] << 4)
        packed = packed.astype(np.uint8)
        shape_out = list(tensor_in.shape)
        shape_out[-1] = shape_out[-1] // 2
        out = torch.from_numpy(packed).reshape(shape_out)
        return out

    def generate_data_mxa8w4(m, n, k, group_num, x_range, weight_range, weight_scale_range, x_scale_range, group_size):
        x = torch.rand((m, k), dtype=torch.float32) * (x_range[1] - x_range[0]) + x_range[0]
        x = x.to(torch.float8_e4m3fn)
        weight = torch.rand((group_num, n, k), dtype=torch.float32) * (weight_range[1] - weight_range[0]) + weight_range[0]
        weight = fp32_to_fp4_e2m1_u8packed(weight)
        pertoken_scale = torch.rand((m, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
                x_scale_range[1] - x_scale_range[0]) + x_scale_range[0]
        pertoken_scale = pertoken_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
        scale = torch.rand((group_num, n, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
                weight_scale_range[1] - weight_scale_range[0]) + weight_scale_range[0]
        scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
        return x, weight, pertoken_scale, scale

    def main():
        g, m, k, n, is_dynamic = 4, 128, 32, 512, True

        groupType = 0
        group_list_type = 1  # 0: cumsun 1: count
        dequant_mode = 2  # mx量化
        dequant_dtype = torch.float32
        quant_mode = 2  # mx量化
        quant_dtype = torch.float8_e4m3fn
        group_size = 32

        # generate data range
        x_range = [-1, 1]
        weight_range = [-6, 6]
        weight_scale_range = [0, 2]
        x_scale_range = [0, 2]
        x, weight, x_scale, weight_scale = generate_data_mxa8w4(m, n, k, g, x_range=x_range, weight_range=weight_range,
                                                                weight_scale_range=weight_scale_range,
                                                                x_scale_range=x_scale_range,
                                                                group_size=group_size)
        group_list = torch.Tensor([32, 32, 32, 32]).to(torch.int64)
        # npu
        x_npu = x.npu()
        weight_npu = weight.npu()
        x_scale_npu = x_scale.npu()
        weight_scale_npu = weight_scale.npu()
        group_list_npu = group_list.npu()
        # npu_format_cast
        weight_npu = torch_npu.npu_format_cast(weight_npu, 29, customize_dtype=torch.float8_e4m3fn,
                                                input_dtype=torch_npu.float4_e2m1fn_x2)
        model = NetPTA().npu()
        output, output_scale = model(x_npu, weight_npu, x_scale_npu, weight_scale_npu, group_list_npu, group_list_type,
                                        dequant_mode, dequant_dtype, quant_mode, quant_dtype)
        print("output")
        print(output)
    if __name__ == '__main__':
        main()
    ```
  <!-- end id39 -->
  <!-- npu="950" id40 -->
  - Ascend 950PR&950DT系列产品：MxA8W4伪量化单多单场景示例

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    from ml_dtypes import float8_e4m3fn
    def ceil_div(a, b):
        return math.ceil(a / b)
    class NetPTA(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, x_scale, weight_scale, group_list, group_list_type, dequant_mode, dequant_dtype,
                    quant_mode, quant_dtype):
            for i in range(len(weight)):
                weight[i] = weight[i].transpose(-1, -2)
                weight_scale[i] = weight_scale[i].transpose(-2, -3)
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight,
                                                                    weight_scale=weight_scale,
                                                                    bias=None,
                                                                    x_scale=x_scale,
                                                                    dequant_mode=dequant_mode,
                                                                    dequant_dtype=dequant_dtype,
                                                                    quant_mode=quant_mode,
                                                                    quant_dtype=quant_dtype,
                                                                    group_list_type=group_list_type,
                                                                    group_list=group_list,
                                                                    weight_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    x_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    weight_dtype=torch_npu.float4_e2m1fn_x2)
            return output
    def fp32_to_fp4_e2m1_u8packed(tensor_in):
        fp4_values = np.array([
            +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
        ], dtype=np.float32)
        x = tensor_in.numpy()
        x_flat = x.reshape(-1, 1)  # (N, 1)
        fp4_values = fp4_values.reshape(1, -1)  # (1, M)
        dist = np.abs(x_flat - fp4_values)  # (N, M)
        indices = np.argmin(dist, axis=1)  # (N,)
        tmp = indices.reshape(-1, 2)
        packed = tmp[:, 0] + (tmp[:, 1] << 4)
        packed = packed.astype(np.uint8)
        shape_out = list(tensor_in.shape)
        shape_out[-1] = shape_out[-1] // 2
        out = torch.from_numpy(packed).reshape(shape_out)
        return out
    def generate_data_mxa8w4(m, n, k, group_num, x_range, weight_range, weight_scale_range, x_scale_range, group_size):
        x = torch.rand((m, k), dtype=torch.float32) * (x_range[1] - x_range[0]) + x_range[0]
        x = x.to(torch.float8_e4m3fn)

        pertoken_scale = torch.rand((m, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
                x_scale_range[1] - x_scale_range[0]) + x_scale_range[0]
        pertoken_scale = pertoken_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
        weight_list = []
        weight_scale_list = []
        for i in range(group_num):
            weight = torch.rand((n, k), dtype=torch.float32) * (weight_range[1] - weight_range[0]) + weight_range[0]
            weight = fp32_to_fp4_e2m1_u8packed(weight)
            scale = torch.rand((n, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
                    weight_scale_range[1] - weight_scale_range[0]) + weight_scale_range[0]
            scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
            weight_list.append(weight.npu())
            weight_scale_list.append(scale.npu())
        return x, weight_list, pertoken_scale, weight_scale_list
    def main():
        g, m, k, n, is_dynamic = 4, 128, 32, 512, True
        groupType = 0
        group_list_type = 1  # 0: cumsun 1: count
        dequant_mode = 2  # mx量化
        dequant_dtype = torch.float32
        quant_mode = 2  # mx量化
        quant_dtype = torch.float8_e4m3fn
        group_size = 32
        # generate data range
        x_range = [-1, 1]
        weight_range = [-6, 6]
        weight_scale_range = [0, 2]
        x_scale_range = [0, 2]
        x, weight, x_scale, weight_scale = generate_data_mxa8w4(m, n, k, g, x_range=x_range, weight_range=weight_range,
                                                                weight_scale_range=weight_scale_range,
                                                                x_scale_range=x_scale_range,
                                                                group_size=group_size)
        group_list = torch.Tensor([32, 32, 32, 32]).to(torch.int64)
        # npu
        x_npu = x.npu()
        x_scale_npu = x_scale.npu()
        group_list_npu = group_list.npu()
        # npu_format_cast
        for idx in range(g):
            weight[idx] = torch_npu.npu_format_cast(weight[idx], 29, customize_dtype=torch.float8_e4m3fn,
                                                    input_dtype=torch_npu.float4_e2m1fn_x2)
        model = NetPTA().npu()
        output, output_scale = model(x_npu, weight, x_scale_npu, weight_scale, group_list_npu, group_list_type,
                                        dequant_mode, dequant_dtype, quant_mode, quant_dtype)
        print("output")
        print(output)
    if __name__ == '__main__':
        main()
    ```
  <!-- end id40 -->

- 图模式调用
  <!-- npu="A3,910b" id47 -->
  - Atlas A2系列产品/Atlas A3系列产品

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from scipy.special import softmax
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, weight_scale, xscale, group_list):
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, xscale, group_list)
            return output

    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weight_scale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList
    E = 2
    M = 512
    K = 7168
    N = 4096
    x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), [weight_npu], [weight_scale.npu()], xScale.npu(), groupList.npu())
    ```
  <!-- end id47 -->

  <!-- npu="950" id41 -->
  - Ascend 950PR&950DT系列产品：mx量化场景示例-mxfp8

    ```python
    import os
    import unittest
    import itertools
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    import torchair as tng
    from typing import Tuple
    import logging
    import torch_npu
    from torchair import logger
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()

    npu_backend = tng.get_npu_backend(compiler_config=config)

    os.environ["ENABLE_ACLNN"] = "false"

    class GMMModel(nn.Module):
        def __init__(self, weight_npu, weight_scale, xScale, transpose=True):
            super().__init__()
            self.transpose = transpose
            self.weight = nn.Parameter(weight_npu, requires_grad=False)
            self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
            self.xScale = nn.Parameter(xScale, requires_grad=False)

        def forward(self, x_npu: torch.Tensor, w: torch.Tensor, group_list_npu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                weight = self.weight
                weight_scale = self.weight_scale.npu()
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x_npu, [weight.transpose(1, 2)], [weight_scale.transpose(1, 2)], xScale.npu(), group_list_npu, quant_mode=2, quant_dtype=torch.float8_e5m2, dequant_mode=2, dequant_dtype=torch.float32,weight_scale_dtype=torch_npu.float8_e8m0fnu, x_scale_dtype=torch_npu.float8_e8m0fnu)
                return y, y_scale

    def gen_input_data(E, M, K, N, transpose):
        if transpose:
            x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
            weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn)
            weight_scale = torch.randint(low=0, high=256, size=(E, N, math.ceil(K / 64), 2), dtype=torch.uint8)
            xScale = torch.randint(low=0, high=256, size=(M, math.ceil(K / 64), 2), dtype=torch.uint8)
            groupList = torch.tensor([M//2, M//2], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList

    def run_npu(x, weight_npu, weight_scale, xScale, groupList, transpose):
        model = GMMModel(weight_npu, weight_scale, xScale, transpose).npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)

        for k in range(1):
            torch_npu.npu.synchronize()
            custom_output, y_scale = model(x, None, groupList)
            torch_npu.npu.synchronize()

    if __name__ == "__main__":
        K = 1
        E = 2
        M = 16
        N = 128
        transpose = True
        x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N, transpose)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weight_scale_npu = weight_scale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        run_npu(x_npu, weight_npu, weight_scale_npu, xScale_npu, groupList_npu, transpose)
    ```
  <!-- end id41 -->
  <!-- npu="950" id42 -->
  - Ascend 950PR&950DT系列产品：mx量化场景示例-mxfp4

    ```python
    import os
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    os.environ["ENABLE_ACLNN"] = "false"

    class GMMModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,
                    x,
                    weight,
                    weight_scale,
                    x_scale,
                    group_list,
                    dequant_mode,
                    dequant_dtype,
                    quant_mode,
                    quant_dtype,
                    group_list_type,
                    weight_scale_dtype,
                    x_scale_dtype,
                    x_dtype=None,
                    weight_dtype=None,
                    transpose_w=False):
            if quant_dtype is None:
                quant_dtype = torch.float8_e5m2
            if weight_scale_dtype is None:
                weight_scale_dtype = torch_npu.float8_e8m0fnu
            if x_scale_dtype is None:
                x_scale_dtype = torch_npu.float8_e8m0fnu

            processed_weight = []
            for w in weight:
                if transpose_w:
                    w = w.transpose(1, 2)
                processed_weight.append(w)

            processed_weight_scale = []
            for ws in weight_scale:
                if transpose_w:
                    ws = ws.transpose(1, 2)
                processed_weight_scale.append(ws)

            with torch.no_grad():
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                    x,
                    processed_weight,
                    processed_weight_scale,
                    x_scale,
                    group_list,
                    dequant_mode=dequant_mode,
                    dequant_dtype=dequant_dtype,
                    quant_mode=quant_mode,
                    quant_dtype=quant_dtype,
                    group_list_type=group_list_type,
                    weight_scale_dtype=weight_scale_dtype,
                    x_scale_dtype=x_scale_dtype,
                    x_dtype=x_dtype,
                    weight_dtype=weight_dtype
                )
                return y, y_scale

    def gen_input_data(E, M, K, N):
        x = torch.randint(0, 256, (M, K), dtype=torch.uint8)
        weight = torch.randint(0, 256, (E, K * 2, N), dtype=torch.uint8)
        weight_scale = torch.randint(0, 256, (E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8)
        xScale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8)
        groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList

    if __name__ == "__main__":
        K = 9
        E = 2
        M = 2255
        N = 896
        transpose = False

        x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weight_scale_npu = weight_scale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        weight_list = [weight_npu]
        weight_scale_list = [weight_scale_npu]

        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)

        y, y_scale = model(
            x_npu,
            weight_list,
            weight_scale_list,
            xScale_npu,
            groupList_npu,
            dequant_mode=2,
            dequant_dtype=torch.float32,
            quant_mode=2,
            quant_dtype=torch.float8_e4m3fn,
            group_list_type=1,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            transpose_w=transpose
        )

        print("y shape: ", y.shape)
        print("y_scale shape: ", y_scale.shape)
    ```
  <!-- end id42 -->
  <!-- npu="950" id43 -->
  - Ascend 950PR&950DT系列产品：Pertoken量化场景示例

    ```python
    import os
    import unittest
    import itertools
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    import torchair as tng
    from typing import Tuple
    import logging
    import torch_npu
    from torchair import logger
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    os.environ["ENABLE_ACLNN"] = "false"
    class GMMModel(nn.Module):
        def __init__(self, weight_npu, weight_scale, xScale, transpose=True):
            super().__init__()
            self.transpose = transpose
            self.weight = nn.Parameter(weight_npu, requires_grad=False)
            self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
            self.xScale = nn.Parameter(xScale, requires_grad=False)
        def forward(self, x_npu: torch.Tensor, w: torch.Tensor, group_list_npu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                weight = self.weight
                weight_scale = self.weight_scale.npu()
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x_npu, [weight.transpose(1, 2)], [weight_scale], xScale.npu(), group_list_npu, quant_mode=0, quant_dtype=torch.float8_e5m2, dequant_mode=0, dequant_dtype=torch.float)
                return y, y_scale
    def gen_input_data(E, M, K, N, transpose):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight_scale = torch.randint(low=0, high=256, size=(E, N), dtype=torch.float)
        xScale = torch.randint(low=0, high=256, size=(M,), dtype=torch.float)
        groupList = torch.tensor([M//2, M//2], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList
    def run_npu(x, weight_npu, weight_scale, xScale, groupList, transpose):
        model = GMMModel(weight_npu, weight_scale, xScale, transpose).npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        for k in range(1):
            torch_npu.npu.synchronize()
            customyy_output, y_scale = model(x, None, groupList)
            print(customyy_output, y_scale)
            torch_npu.npu.synchronize()
    if __name__ == "__main__":
        K = 1
        E = 2
        M = 16
        N = 128
        transpose = False
        x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N, transpose)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weight_scale_npu = weight_scale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        run_npu(x_npu, weight_npu, weight_scale_npu, xScale_npu, groupList_npu, transpose)
    ```
  <!-- end id43 -->
