# torch\_npu.npu\_grouped\_matmul\_finalize\_routing

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

API功能：GroupedMatMul和MoeFinalizeRouting的融合算子，其中，GroupedMatMul负责分专家计算，MoeFinalizeRouting负责按路由关系回填并聚合GroupedMatMul结果，两者串联形成完整的MoE输出路径。

- MoE: Mixture of Experts，混合专家模型。每个token按路由结果分配到一个或多个专家进行计算。
- MoeFinalizeRouting: MoE路由最终化过程。将各专家计算结果按路由索引回填到token原始顺序，并对同一token的多个专家结果进行聚合，得到最终输出。

## 函数原型

```python
torch_npu.npu_grouped_matmul_finalize_routing(x, w, group_list, *, scale=None, bias=None, offset=None, pertoken_scale=None, shared_input=None, logit=None, row_index=None, dtype=None, shared_input_weight=1.0, shared_input_offset=0, output_bs=0, group_list_type=1, x_dtype=None, w_dtype=None, scale_dtype=None, pertoken_scale_dtype=None)
```

## 参数说明

- **`x`**（`Tensor`）：**必选参数**，矩阵计算的左矩阵，支持非连续的Tensor。数据格式支持$ND$，维度为\(m, k\)。`m`取值范围为\[0, 16\*1024\*8\]。

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int8`。
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - mx量化场景下，数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1fn_x2`，其中float4系列需配置可选参数`x_dtype`为对应类型，此时`x`本身的`dtype`不再生效，但仍需保证`x`本身的`dtype`为8bit位的数据类型，以保证shape正确；其中float4内轴`K`需为偶数，以保证8bits可以转换为2个float4。
    - pertoken量化场景下，数据类型支持`torch.int8`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。
    - x支持m为0，其他空Tensor场景不支持。
  <!-- end id5 -->

- **`w`**（`Tensor`）：**必选参数**，矩阵计算的右矩阵，支持非连续的Tensor。数据格式支持$ND$。

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.int8`、`torch_npu.int4`。
    - A8W8量化场景下，数据格式支持`FRACTAL_NZ`，维度为\(e, n1, k1, k0, n0\)，其中`k0`=16、`n0`=32，`x` shape中的`k`和`w` shape中的`k1`需要满足以下关系：ceilDiv(k, 16) = k1，`e`取值范围为\[1, 256\]，`k`取值为16整倍数，`n`取值为32整倍数，且`n`大于等于256。
    - A8W4场景下数据格式支持$ND$，维度为\(e, k, n\)，`k`支持2048，`n`只支持7168。
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - mx量化场景下，数据类型支持`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1fn_x2`，其中float4系列需配置可选参数`w_dtype`为对应类型，此时`w`本身的`dtype`不再生效，但仍需保证`w`本身的`dtype`为8bit位的数据类型，以保证shape正确；其中float4场景，此时输入`x`的`K`需为大于2的偶数，且当`weight`不转置时内轴`N`需为偶数，以保证8bits可以转换为2个float4。维度为\(e, k, n\)。
    - pertoken量化场景下，数据类型支持`torch.int8`、`torch.float8_e4m3fn`、`torch_npu.hifloat8`。数据格式支持`ND`和`FRACTAL_NZ`，可通过`torch_npu.npu_format_cast`接口实现$ND$转`FRACTAL_NZ`格式。
    - w支持n为0，其他空Tensor场景不支持。
  <!-- end id7 -->

- <strong>*</strong>：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **`group_list`**（`Tensor`）：**必选参数**，GroupedMatMul的各分组大小，支持非连续的Tensor。数据类型支持`torch.int64`，数据格式支持$ND$，维度为\(e,\)，`e`与`w`的`e`一致。`group_list`的值总和要求≤`m`，`group_list`的长度不超过1024。
- **`scale`**（`Tensor`）：**可选参数**，矩阵计算反量化参数，对应`weight`矩阵。数据格式支持$ND$。

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：A8W8场景下支持per-channel量化方式，不支持非连续的Tensor，数据类型支持`torch.float32`，维度\(e, n\)，这里的n=n1\*n0；A8W4量化场景下，数据类型支持`torch.int64`，维度为\(e, 1, n\)。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - mx量化场景下，数据类型支持`torch_npu.float8_e8m0fnu`，其中`torch_npu.float8_e8m0fnu`需配置`scale_dtype`为对应类型，此时`scale`本身的`dtype`不再生效，但仍需保证`scale`本身的`dtype`为8bit位的数据类型，以保证shape正确，`weight`非转置维度为\(e, k/64, n, 2\)，`weight`转置维度为\(e, n, k/64, 2\)。
    - pertoken量化场景下，数据类型支持`torch.float32`、`torch.bfloat16`。维度为\(e, 1, n\)，`e`、`n`与`w`的`e`、`n`一致。
  <!-- end id9 -->

- **`bias`**（`Tensor`）：**可选参数**，矩阵计算的bias参数，支持非连续的Tensor。数据格式支持$ND$，维度为\(e, n\)。

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：只支持A8W4场景，数据类型支持`torch.float32`。
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持mx量化、pertoken量化场景，数据类型支持`torch.bfloat16`。
  <!-- end id11 -->

- **`offset`**（`Tensor`）：**可选参数**，矩阵计算量化参数的偏移量，支持非连续的Tensor。支持3D Tensor输入，数据类型支持`torch.float32`，数据格式支持$ND$。

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：只支持A8W4量化场景。
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT系列产品</term>：暂不支持该参数，使用默认值即可。
  <!-- end id13 -->

- **`pertoken_scale`**（`Tensor`）：**可选参数**，矩阵计算的反量化参数，对应`x`矩阵，pertoken量化方式，支持非连续的Tensor。数据格式支持$ND$。

  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float32`。维度为\(m,\)。
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - mx量化场景下，**必选参数**，数据类型支持`torch_npu.float8_e8m0fnu`，其中`torch_npu.float8_e8m0fnu`需配置`scale_dtype`为对应类型，此时`pertoken_scale`本身的`dtype`不再生效，但仍需保证`pertoken_scale`本身的`dtype`为8bit位的数据类型，以保证shape正确。维度为\(m, k/64, 2\)。
    - pertoken量化场景下，**可选参数**，数据类型支持`torch.float32`。维度为\(m,\)。
  <!-- end id15 -->

- **`shared_input`**（`Tensor`）：**可选参数**，MoE计算中共享专家的输出，需要与MoE专家的输出进行combine操作，支持非连续的Tensor。数据类型支持`bfloat16`，数据格式支持$ND$，维度\(batch/dp, n\)，`n`与`scale`的`n`一致，`batch/dp`取值范围\[1, 2\*1024\]，`batch`取值范围\[1, 16\*1024\]。
- **`logit`**（`Tensor`）：**可选参数**，MoE专家对各个token的logit大小，矩阵乘的计算输出与该logit做乘法，然后索引进行combine，支持非连续的Tensor。数据格式支持$ND$，维度\(m,\)，`m`与`x`的`m`一致。

  <!-- npu="A3,910b" id16 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`float32`。
  <!-- end id16 -->
  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT系列产品</term>：该参数必须传入。数据类型支持`float32`；mx量化且输出为`bfloat16`场景下，还支持`bfloat16`。
  <!-- end id17 -->

- **`row_index`**（`Tensor`）：**可选参数**，MoE专家输出按照该rowIndex进行combine，其中的值即为combine做scatter add的索引，支持非连续的Tensor。数据格式支持$ND$，维度为\(m,\)，`m`与`x`的`m`一致。

  <!-- npu="A3,910b" id18 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`int32`、`int64`。
  <!-- end id18 -->
  <!-- npu="950" id19 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - mx量化场景下，必选参数，输出为`float32`时数据类型支持`int64`；输出为`bfloat16`时数据类型支持`int32`。
    - pertoken量化场景下，**必选参数**。当输入的`x`数据类型为`int8`时，`row_index`数据类型支持`int64`和`int32`；当输入的`x`数据类型为`float8_e4m3fn`或`hifloat8`时，`row_index`数据类型支持`int64`；
  <!-- end id19 -->

- **`dtype`**（`ScalarType`）：**可选参数**，指定GroupedMatMul计算的输出类型。当前接口仅支持`dtype`为0或2，不支持1。

  <!-- npu="A3,910b" id20 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：0表示`float32`，1表示`float16`，2表示`bfloat16`。默认值为0。
  <!-- end id20 -->
  <!-- npu="950" id21 -->
  - <term>Ascend 950PR&950DT系列产品</term>：0表示`float32`，1表示`float16`，2表示`bfloat16`。默认值为0。mx量化、权重为FRACTAL\_NZ格式且`bias`为None场景下支持输出`bfloat16`。
  <!-- end id21 -->

- **`shared_input_weight`**（`float`）：**可选参数**，共享专家与MoE专家进行combine的系数，`shared_input`先与该参数乘，然后再和MoE专家结果累加。默认为1.0。
- **`shared_input_offset`**（`int`）：**可选参数**，共享专家输出的在总输出中的偏移。默认值为0，`shared_input_offset`+`shared_input`的第一维度总和不允许超过`batch`。
- **`output_bs`**（`int`）：**可选参数**，输出的最高维大小。默认值为0。
- **`group_list_type`**（`int`）：**可选参数**，GroupedMatMul的分组模式。默认为1，表示count模式；若配置为0，表示cumsum模式，即为前缀和。
- **`x_dtype`**（`int`）：**可选参数**，输入`x`的真实数据类型，默认为None。

  <!-- npu="A3,910b" id22 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数，使用默认值。
  <!-- end id22 -->
  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - mx量化场景下，若传入None表示输入`x`的真实数据类型与输入的`dtype`相同；若非None，`x`的真实数据类型支持`float4_e2m1fn_x2`。
    - pertoken量化场景下，若传入None表示输入`x`的真实数据类型与输入的`dtype`相同；若非None，`x`的真实数据类型支持`hifloat8`。
  <!-- end id23 -->

- **`w_dtype`**（`int`）：**可选参数**，输入`w`的真实数据类型，默认为None。

  <!-- npu="A3,910b" id24 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数，使用默认值。
  <!-- end id24 -->
  <!-- npu="950" id25 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - mx量化场景下，若传入None表示输入`x`的真实数据类型与输入的`dtype`相同；若非None，`x`的真实数据类型支持`float4_e2m1fn_x2`。
    - pertoken量化场景下，若传入None表示输入`x`的真实数据类型与输入的`dtype`相同；若非None，`x`的真实数据类型支持`hifloat8`。
  <!-- end id25 -->

- **`scale_dtype`**（`int`）：**可选参数**，输入`scale`的真实数据类型，默认为None。

  <!-- npu="A3,910b" id26 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数，使用默认值。
  <!-- end id26 -->
  <!-- npu="950" id27 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx量化场景下，若传入None表示输入`scale`的真实数据类型与输入的`dtype`相同；若非None，`scale`的真实数据类型支持`float8_e8m0`。
  <!-- end id27 -->

- **`pertoken_scale_dtype`**（`int`）：**可选参数**，输入`pertoken_scale`的真实数据类型，默认为None。

  <!-- npu="A3,910b" id28 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持该参数，使用默认值。
  <!-- end id28 -->
  <!-- npu="950" id29 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx量化场景下，若传入None表示输入`pertoken_scale`的真实数据类型与输入的`dtype`相同；若非None，`pertoken_scale`的真实数据类型支持`float8_e8m0`。
  <!-- end id29 -->

## 返回值说明

**`y`**（`Tensor`）：一个2D的Tensor，支持非连续的Tensor，维度为\(batch, n\)。

<!-- npu="A3,910b" id30 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：输出的数据类型支持`float32`。
<!-- end id30 -->
<!-- npu="950" id31 -->
- <term>Ascend 950PR&950DT系列产品</term>：输出的数据类型支持`float32`；mx量化、权重为FRACTAL\_NZ格式且`bias`为None场景下还支持`bfloat16`。
<!-- end id31 -->

## 约束说明

- 该接口支持推理和训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式；输出为`bfloat16`时，仅支持单算子模式，暂不支持图模式。
- 输入和输出Tensor支持的数据类型组合如下：

  <!-- npu="A3,910b" id32 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

    | x | w | group_list | scale | bias | offset | pertoken_scale | shared_input | logit | row_index | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | int8 | int8 | int64 | float32 | None | None | float32 | bfloat16 | float32 | int64 | float32 |
    | int8 | int8 | int64 | float32 | None | None | float32 | None | None | int64 | float32 |
    | int8 | int4 | int64 | int64 | float32 | None | float32 | bfloat16 | float32 | int64 | float32 |
    | int8 | int4 | int64 | int64 | float32 | float32 | float32 | bfloat16 | float32 | int64 | float32 |
  <!-- end id32 -->
  <!-- npu="950" id33 -->
  - <term>Ascend 950PR&950DT系列产品</term>：

    | x | w | group_list | scale | bias | offset | pertoken_scale | shared_input | logit | row_index | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | float8_e5m2 | float8_e5m2 | int64 | float8_e8m0 | bfloat16 | float32 | float8_e8m0 | bfloat16 | float32 | int64 | float32 |
    | float8_e5m2 | float8_e4m3fn | int64 | float8_e8m0 | bfloat16 | float32 | float8_e8m0 | bfloat16 | float32 | int64 | float32 |
    | float8_e4m3fn | float8_e5m2 | int64 | float8_e8m0 | bfloat16 | float32 | float8_e8m0 | bfloat16 | float32 | int64 | float32 |
    | float8_e4m3fn | float8_e4m3fn | int64 | float8_e8m0 | bfloat16 | float32 | float8_e8m0 | bfloat16 | float32 | int64 | float32 |
    | float4_e2m1fn_x2 | float4_e2m1fn_x2 | int64 | float8_e8m0 | bfloat16 | float32 | float8_e8m0 | bfloat16 | float32 | int64 | float32 |
    | float8_e4m3fn | float8_e4m3fn | int64 | float8_e8m0 | None | float32 | float8_e8m0 | bfloat16/None | float32/bfloat16 | int32 | bfloat16 |
    | float4_e2m1fn_x2 | float4_e2m1fn_x2 | int64 | float8_e8m0 | None | float32 | float8_e8m0 | bfloat16/None | float32/bfloat16 | int32 | bfloat16 |
    | int8 | int8 | int64 | float32 | bfloat16/None | None | float32/None | bfloat16 | float32 | int64 | float32 |
    | int8 | int8 | int64 | bfloat16 | bfloat16/None | None | float32/None | bfloat16 | float32 | int64 | float32 |
    | int8 | int8 | int64 | float32 | bfloat16/None | None | float32/None | bfloat16 | float32 | int32 | float32 |
    | int8 | int8 | int64 | bfloat16 | bfloat16/None | None | float32/None | bfloat16 | float32 | int32 | float32 |
    | float8_e4m3fn | float8_e4m3fn | int64 | float32 | bfloat16/None | None | float32/None | bfloat16 | float32 | int64 | float32 |
    | float8_e4m3fn | float8_e4m3fn | int64 | bfloat16 | bfloat16/None | None | float32/None | bfloat16 | float32 | int64 | float32 |
    | hifloat8 | hifloat8 | int64 | float32 | bfloat16/None | None | float32/None | bfloat16 | float32 | int64 | float32 |
    | hifloat8 | hifloat8 | int64 | bfloat16 | bfloat16/None | None | float32/None | bfloat16 | float32 | int64 | float32 |
    | float8_e4m3fn | float4_e2m1fn_x2 | int64 | float8_e8m0 | bfloat16/None | None | float8_e8m0 | bfloat16 | float32 | int64 | float32 |

    > **MxA8W4场景说明**：
    > - `x`数据类型为`torch.float8_e4m3fn`，`w`数据类型为`torch_npu.float4_e2m1fn_x2`。`w`数据格式要求FRACTAL\_NZ格式，可通过torch\_npu.npu\_format\_cast接口实现ND转FRACTAL\_NZ格式。
    > - `x`要求非转置，`w`要求转置，`w`的shape支持\(e, n, k\), 要求e<=1024、k和n满足32对齐。图模式场景下要求e不等于1，同时k\>64。
    > - `shared_input_offset`+`shared_input`的第一维度总和不允许超过`output_bs`。
    > - mx量化输出为`bfloat16`时，`w`数据格式要求FRACTAL\_NZ格式，`bias`必须为None，`row_index`必须为`int32`，仅支持单算子模式，暂不支持图模式。
  <!-- end id33 -->

## 调用示例

- 单算子模式调用

  <!-- npu="A3,910b" id34 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

    ```python
    # 非量化场景
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax

    m, k, n = 576, 2048, 7168
    batch = 72
    topK = 8
    group_num = 8

    x = np.random.randint(-1, 1, (m, k)).astype(np.int8)
    weight = np.random.randint(-1, 1, (group_num, k, n)).astype(np.int8)
    scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
    pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)

    x_clone = torch.from_numpy(x).npu()
    weight_clone = torch.from_numpy(weight).npu()
    weightNz = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.from_numpy(scale).npu()
    pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    output_bs = batch
    y = torch_npu.npu_grouped_matmul_finalize_routing(x_clone, weightNz,
                group_list_clone, scale=scale_clone, pertoken_scale=pertoken_scale_clone,
                shared_input=shared_input_clone, logit=logit_clone, row_index=row_index_clone,
                shared_input_offset=shared_input_offset, output_bs=output_bs)
    ```
  <!-- end id34 -->
  <!-- npu="950" id35 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx量化场景示例-mxfp8

    ```python
    import torch
    import torch_npu
    import math
    import numpy as np
    from scipy.special import softmax
    def gen_input_data_mxfp8_wtrans(M, K, N, E):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        Scale = torch.randint(low=0, high=256, size=(E, N, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
        pertoken_scale = torch.randint(low=0, high=256, size=(M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
        groupList = torch.tensor([M//2, M//2], dtype=torch.int64).npu()
        return x, weight, Scale, pertoken_scale, groupList
    m, k, n, batch, topK, group_num = 72, 32, 7168, 72, 1, 2
    x_clone, weight_clone, scale_clone, pertoken_scale_clone, group_list_clone = gen_input_data_mxfp8_wtrans(m, k, n, group_num)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    group_list_type = 1
    output_bs = batch
    x_dtype = None
    w_dtype = None
    scale_dtype = torch_npu.float8_e8m0fnu
    pertoken_scale_dtype = torch_npu.float8_e8m0fnu
    # w transpose
    weight_clone = weight_clone.transpose(1,2)
    scale_clone = scale_clone.transpose(1,2)
    bias = torch.randint(-1, 1, (group_num, n), dtype=torch.bfloat16).npu()
    out = torch_npu.npu_grouped_matmul_finalize_routing(x_clone, weight_clone, group_list_clone,bias=bias,
        scale=scale_clone, pertoken_scale=pertoken_scale_clone, shared_input=shared_input_clone,
        logit=logit_clone, row_index=row_index_clone, shared_input_offset=shared_input_offset, output_bs=output_bs,
        group_list_type=group_list_type,x_dtype=x_dtype, w_dtype=w_dtype, scale_dtype=scale_dtype, pertoken_scale_dtype=pertoken_scale_dtype)
    print(out.cpu())
    print(out.cpu().shape)
    ```
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - <term>Ascend 950PR&950DT系列产品</term>：pertoken量化weightNZ场景示例-int8

    ```python
    import torch
    import torch_npu
    import math
    import numpy as np
    from scipy.special import softmax
    m, k, n, batch, topK, group_num = 72, 32, 7168, 72, 1, 2
    x_clone = torch.randint(-128, 127, (m, k), dtype=torch.int8).npu()
    weight_clone = torch.randint(-128, 127, (group_num, k, n), dtype=torch.int8).npu()
    # w transpose
    # weight_clone = weight_clone.transpose(1,2)
    weightNZ_clone = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.randint(low=0, high=256, size=(group_num, 1, n), dtype=torch.float32).npu()
    pertoken_scale_clone = (torch.rand(m, dtype=torch.float32) * 256).npu()
    group_list_clone = torch.tensor([m//2, m//2], dtype=torch.int64).npu()
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    group_list_type = 1
    output_bs = batch
    x_dtype = None
    w_dtype = None
    scale_dtype = None
    pertoken_scale_dtype = None
    bias = torch.randint(-1, 1, (group_num, n), dtype=torch.bfloat16).npu()
    out = torch_npu.npu_grouped_matmul_finalize_routing(x_clone, weightNZ_clone, group_list_clone,bias=bias,
        scale=scale_clone, pertoken_scale=pertoken_scale_clone, shared_input=shared_input_clone,
        logit=logit_clone, row_index=row_index_clone, shared_input_offset=shared_input_offset, output_bs=output_bs,
        group_list_type=group_list_type,x_dtype=x_dtype, w_dtype=w_dtype, scale_dtype=scale_dtype, pertoken_scale_dtype=pertoken_scale_dtype)
    print(out.cpu())
    print(out.cpu().shape)
    ```
  <!-- end id36 -->
  <!-- npu="950" id37 -->
  - <term>Ascend 950PR&950DT系列产品</term>：伪量化场景mxA8W4

    ```python
    import numpy as np
    from ml_dtypes import float4_e2m1fn
    import torch
    import torch_npu
    import math

    def ceil_div(a, b):
        return math.ceil(a / b)
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

    g, group_size, m, k, n = 4, 32, 128, 64, 512
    group_list_type = 1  # 0: cumsun 1: count
    output_dtype = torch.float32
    shared_input_weight = 0.5
    shared_input_offset = 2
    output_bs = 8
    # generate data
    x_range = [-1, 1]
    weight_range = [-6, 6]
    scale_range = [0, 2]
    bias_range = [0, 2]
    pertoken_scale_range = [0, 2]
    logit_range = [0, 2]
    shared_input_range = [0, 2]
    row_index_range = [0, 2]
    x = torch.rand((m, k), dtype=torch.float32) * (x_range[1] - x_range[0]) + x_range[0]
    x = x.to(torch.float8_e4m3fn)
    weight = torch.rand((g, n, k), dtype=torch.float32) * (weight_range[1] - weight_range[0]) + weight_range[0]
    weight = fp32_to_fp4_e2m1_u8packed(weight)
    pertoken_scale = torch.rand((m, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
        pertoken_scale_range[1] - pertoken_scale_range[0]) + pertoken_scale_range[0]
    pertoken_scale = pertoken_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    scale = torch.rand((g, n, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
        scale_range[1] - scale_range[0]) + scale_range[0]
    scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    bias = torch.rand((g, n), dtype=torch.bfloat16) * (
        bias_range[1] - bias_range[0]) + bias_range[0]
    logit = torch.rand((m,), dtype=torch.float32) * (logit_range[1] - logit_range[0]) + logit_range[0]
    shared_input = torch.rand((output_bs // 2, n), dtype=torch.bfloat16) * (
        shared_input_range[1] - shared_input_range[0]) + shared_input_range[0]
    row_index = torch.randint(low=row_index_range[0], high=row_index_range[1], size=(m,), dtype=torch.int64)
    group_list = torch.Tensor([32, 32, 32, 32]).to(torch.int64)
    weight_npu = weight.npu()
    weight_npu = torch_npu.npu_format_cast(weight_npu, 29, customize_dtype=torch.float8_e4m3fn,
                                                input_dtype=torch_npu.float4_e2m1fn_x2)
    # npu
    x_npu = x.npu()
    pertoken_scale_npu = pertoken_scale.npu()
    scale_npu = scale.npu()
    bias_npu = bias.npu()
    logit_npu = logit.npu()
    shared_input_npu = shared_input.npu()
    row_index_npu = row_index.npu()
    group_list_npu = group_list.npu()
    weight_npu = weight_npu.transpose(-1, -2)
    scale_npu = scale_npu.transpose(-2, -3)
    output = torch_npu.npu_grouped_matmul_finalize_routing(x_npu, weight_npu, group_list=group_list_npu,             scale=scale_npu,bias=bias_npu, pertoken_scale=pertoken_scale_npu, shared_input=shared_input_npu,               logit=logit_npu,row_index=row_index_npu,shared_input_weight=shared_input_weight,               shared_input_offset=shared_input_offset, group_list_type=group_list_type,               output_bs=output_bs, scale_dtype=torch_npu.float8_e8m0fnu, pertoken_scale_dtype=torch_npu.float8_e8m0fnu,               w_dtype=torch_npu.float4_e2m1fn_x2)
    ```
  <!-- end id37 -->

- 图模式调用

  <!-- npu="A3,910b" id38 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

    ```python
    # 非量化场景
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
        def forward(self, x, weight, group_list, scale, pertoken_scale, shared_input, logit, row_index, shared_input_offset, output_bs):
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list,
                        scale=scale, pertoken_scale=pertoken_scale, shared_input=shared_input,
                        logit=logit, row_index=row_index, shared_input_offset=shared_input_offset, output_bs=output_bs)
            return output

    m, k, n = 576, 2048, 7168
    batch = 72
    topK = 8
    group_num = 8

    x = np.random.randint(-10, 10, (m, k)).astype(np.int8)
    weight = np.random.randint(-10, 10, (group_num, k, n)).astype(np.int8)
    scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
    pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)

    x_clone = torch.from_numpy(x).npu()
    weight_clone = torch.from_numpy(weight).npu()
    weightNz = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.from_numpy(scale).npu()
    pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    output_bs = batch

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x_clone, weightNz, group_list_clone, scale_clone, pertoken_scale_clone, shared_input_clone, logit_clone, row_index_clone, shared_input_offset, output_bs)
    ```
  <!-- end id38 -->
  <!-- npu="950" id39 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx量化场景示例-mxfp4图模式调用

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
        import torch
        import torch_npu
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, bias, group_list, scale, pertoken_scale, shared_input, logit, row_index, shared_input_offset, output_bs, group_list_type,
            x_dtype, w_dtype, scale_dtype, pertoken_scale_dtype):
            weight = weight.transpose(1, 2)
            scale = scale.transpose(1, 2)
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list,bias=bias,
            scale=scale, pertoken_scale=pertoken_scale, shared_input=None,
            logit=logit, row_index=row_index, shared_input_offset=shared_input_offset, output_bs=output_bs, group_list_type=group_list_type,
                x_dtype=x_dtype, w_dtype=w_dtype, scale_dtype=scale_dtype, pertoken_scale_dtype=pertoken_scale_dtype)
            return output
    m, k, n = 72, 32, 7168
    batch = 72
    topK = group_num = 1
    scale_clone = torch.full((group_num, n, k//64, 2), 1, dtype=torch.int8).npu()
    pertoken_scale_clone = torch.full((m, k//64, 2), 1, dtype=torch.int8).npu()
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
    x_clone = torch.randint(1, 2, (m, k // 2), dtype=torch.uint8).npu()
    weight_clone = torch.randint(1, 2, (group_num, n // 2, k), dtype=torch.uint8).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    group_list_type = 1
    output_bs = batch
    x_dtype=torch_npu.float4_e2m1fn_x2
    w_dtype=torch_npu.float4_e2m1fn_x2
    scale_dtype=torch_npu.float8_e8m0fnu
    pertoken_scale_dtype=torch_npu.float8_e8m0fnu
    bias = torch.randint(-1, 1, (group_num, n), dtype=torch.bfloat16).npu()
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x_clone, weight_clone, bias, group_list_clone, scale_clone, pertoken_scale_clone, shared_input_clone, logit_clone, row_index_clone, shared_input_offset, output_bs, group_list_type, x_dtype, w_dtype, scale_dtype, pertoken_scale_dtype)
    print(y.cpu())
    print(y.cpu().shape)
    ```
  <!-- end id39 -->
  <!-- npu="950" id40 -->
  - <term>Ascend 950PR&950DT系列产品</term>：pertoken量化weightNZ场景示例-int8图模式调用

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
        import torch
        import torch_npu
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, bias, group_list, scale, pertoken_scale, shared_input, logit, row_index, shared_input_offset, output_bs, group_list_type,
            x_dtype, w_dtype, scale_dtype, pertoken_scale_dtype):
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list,bias=bias,
            scale=scale, pertoken_scale=pertoken_scale, shared_input=None,
            logit=logit, row_index=row_index, shared_input_offset=shared_input_offset, output_bs=output_bs, group_list_type=group_list_type,
                x_dtype=x_dtype, w_dtype=w_dtype, scale_dtype=scale_dtype, pertoken_scale_dtype=pertoken_scale_dtype)
            return output
    m, k, n, batch, topK, group_num = 72, 256, 7168, 72, 1, 2
    x_clone = torch.randint(-128, 127, (m, k), dtype=torch.int8).npu()
    weight_clone = torch.randint(-128, 127, (group_num, k, n), dtype=torch.int8).npu()
    # w transpose
    # weight_clone = weight_clone.transpose(1,2)
    weightNZ_clone = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.randint(low=0, high=256, size=(group_num, 1, n), dtype=torch.float32).npu()
    pertoken_scale_clone = (torch.rand(m, dtype=torch.float32) * 256).npu()
    group_list_clone = torch.tensor([m//2, m//2], dtype=torch.int64).npu()
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    group_list_type = 1
    output_bs = batch
    x_dtype = None
    w_dtype = None
    scale_dtype = None
    pertoken_scale_dtype = None
    bias = torch.randint(-1, 1, (group_num, n), dtype=torch.bfloat16).npu()
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x_clone, weightNZ_clone, bias, group_list_clone, scale_clone, pertoken_scale_clone, shared_input_clone, logit_clone, row_index_clone, shared_input_offset, output_bs, group_list_type, x_dtype, w_dtype, scale_dtype, pertoken_scale_dtype)
    print(y.cpu())
    print(y.cpu().shape)
    ```
  <!-- end id40 -->
  <!-- npu="950" id41 -->
  - <term>Ascend 950PR&950DT系列产品</term>：伪量化mxA8W4场景图模式

    ```python
    import numpy as np
    from ml_dtypes import float4_e2m1fn
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    import math
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    # config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class NetPTA(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, pertoken_scale, scale, bias, logit, shared_input, row_index, group_list,
                    group_list_type,
                    output_bs, shared_input_weight, shared_input_offset):
            weight = weight.transpose(-1, -2)
            scale = scale.transpose(-2, -3)
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list=group_list,
                                                                    scale=scale,
                                                                    bias=bias,
                                                                    pertoken_scale=pertoken_scale,
                                                                    shared_input=shared_input,
                                                                    logit=logit,
                                                                    row_index=row_index,
                                                                    shared_input_weight=shared_input_weight,
                                                                    shared_input_offset=shared_input_offset,
                                                                    group_list_type=group_list_type,
                                                                    output_bs=output_bs,
                                                                    scale_dtype=torch_npu.float8_e8m0fnu,
                                                                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    w_dtype=torch_npu.float4_e2m1fn_x2)
            return output

    def ceil_div(a, b):
        return math.ceil(a / b)

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

    g, group_size, m, k, n, is_dynamic = 4, 32, 128, 256, 512, True
    # mode控制单算子/图模式，静态图 : 1 ; 动态图 ：2
    mode = 2
    group_list_type = 1  # 0: cumsun 1: count
    output_dtype = torch.float32
    shared_input_weight = 0.5
    shared_input_offset = 2
    output_bs = 8
    # generate data
    x_range = [-1, 1]
    weight_range = [-6, 6]
    scale_range = [0, 2]
    bias_range = [0, 2]
    pertoken_scale_range = [0, 2]
    logit_range = [0, 2]
    shared_input_range = [0, 2]
    row_index_range = [0, 2]
    x = torch.rand((m, k), dtype=torch.float32) * (x_range[1] - x_range[0]) + x_range[0]
    x = x.to(torch.float8_e4m3fn)
    weight = torch.rand((g, n, k), dtype=torch.float32) * (weight_range[1] - weight_range[0]) + weight_range[0]
    weight = fp32_to_fp4_e2m1_u8packed(weight)
    pertoken_scale = torch.rand((m, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
        pertoken_scale_range[1] - pertoken_scale_range[0]) + pertoken_scale_range[0]
    pertoken_scale = pertoken_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    scale = torch.rand((g, n, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
        scale_range[1] - scale_range[0]) + scale_range[0]
    scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    bias = torch.rand((g, n), dtype=torch.bfloat16) * (
        bias_range[1] - bias_range[0]) + bias_range[0]
    logit = torch.rand((m,), dtype=torch.float32) * (logit_range[1] - logit_range[0]) + logit_range[0]
    shared_input = torch.rand((output_bs // 2, n), dtype=torch.bfloat16) * (
        shared_input_range[1] - shared_input_range[0]) + shared_input_range[0]
    row_index = torch.randint(low=row_index_range[0], high=row_index_range[1], size=(m,), dtype=torch.int64)
    group_list = torch.Tensor([32, 32, 32, 32]).to(torch.int64)
    x_npu = x.npu()
    weight_npu = weight.npu()
    pertoken_scale_npu = pertoken_scale.npu()
    scale_npu = scale.npu()
    bias_npu = bias.npu()
    logit_npu = logit.npu()
    shared_input_npu = shared_input.npu()
    row_index_npu = row_index.npu()
    group_list_npu = group_list.npu()
    weight_npu = torch_npu.npu_format_cast(weight_npu, 29, customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2)
    model = NetPTA().npu()
    if mode == 1:
        model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif mode == 2:
        model = torch.compile(model, backend=npu_backend, dynamic=True)
    out = model(x_npu, weight_npu, pertoken_scale_npu, scale_npu, bias_npu, logit_npu, shared_input_npu, row_index_npu,group_list_npu, group_list_type, output_bs, shared_input_weight, shared_input_offset)
    ```
  <!-- end id41 -->
