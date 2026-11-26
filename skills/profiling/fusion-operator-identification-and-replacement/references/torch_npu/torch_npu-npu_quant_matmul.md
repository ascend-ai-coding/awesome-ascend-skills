# torch_npu.npu_quant_matmul

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

- API功能：完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。

- 计算公式：

  公式中的x1Scale、x2Scale、yScale分别对应参数`pertoken_scale`、`scale`、`y_scale`；x2Offset和yOffset均由参数`offset`提供，具体含义由量化场景决定。量化模式的具体介绍参见《CANN算子库》中的[量化介绍](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/quant_mode_introduction.md)。

  <!-- npu="A3,910b" id5 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

    支持K-C、K-T、T-C、T-T、G-B和K-G量化模式，不同量化模式对应的输入输出数据类型组合参见[约束说明](#约束说明)。

    <details>

    <summary><strong>K-G量化模式</strong></summary>

      - x1为`torch.int8`，x2为`torch.int32`，x1Scale为`torch.float32`，x2Scale为`torch.uint64`/`torch.int64`，yOffset为`torch.float32`：

        $$
        out = ((x1 \mathbin{@} (x2 * \text{x2Scale})) + \text{yOffset}) * \text{x1Scale}
        $$

      - x1、x2为`torch_npu.int4`，x1Scale、x2Scale为`torch.float32`，x2Offset为`torch.float16`，out为`torch.float16`/`torch.bfloat16`（pertoken-pergroup非对称量化）：

        $$
        out = \text{x1Scale} * \text{x2Scale} * (x1 \mathbin{@} x2 - x1 \mathbin{@} \text{x2Offset})
        $$

    </details>

    <details>

    <summary><strong>K-C和K-T量化模式</strong></summary>

      - 有x1Scale、无bias：

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale}
        $$

      - 有x1Scale，bias为`torch.int32`（此场景无offset）：

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} * \text{x1Scale}
        $$

      - 有x1Scale，bias为`torch.bfloat16`/`torch.float16`/`torch.float32`（此场景无offset）：

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale} + bias
        $$

    </details>

    <details>

    <summary><strong>T-C和T-T量化模式</strong></summary>

      - 无x1Scale、无bias：

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} + \text{x2Offset}
        $$

      - bias为`torch.int32`：

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} + \text{x2Offset}
        $$

      - bias为`torch.bfloat16`/`torch.float32`（此场景无offset）：

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} + bias
        $$

    </details>

    <details>

    <summary><strong>G-B量化模式</strong></summary>

      x1、x2为`torch.int8`，x1Scale、x2Scale为`torch.float32`，bias为`torch.float32`，out为`torch.float16`/`torch.bfloat16`（pergroup-perblock量化）：

      $$
      out = (x1 \mathbin{@} x2) * \text{x1Scale} * \text{x2Scale} + bias
      $$

    </details>
  <!-- end id5 -->
  <!-- npu="310p" id6 -->
  - <term>Atlas推理系列产品</term>：

    支持K-C量化模式，不同量化模式对应的输入输出数据类型组合参见[约束说明](#约束说明)。

    <details>

    <summary><strong>K-C量化模式</strong></summary>

      - 有x1Scale、无bias：

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale}
        $$

      - 有x1Scale，bias为`torch.int32`（此场景无offset）：

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} * \text{x1Scale}
        $$

    </details>
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：

    支持T-C、T-T、K-C、K-T、G-B、B-B、MX、T-CG和K-G量化模式，不同量化模式对应的输入输出数据类型组合参见[约束说明](#约束说明)。

    <details>

    <summary><strong>K-G量化模式</strong></summary>

      x1、x2为`torch_npu.int4`，x1Scale、x2Scale为`torch.float32`，x2Offset为`torch.float16`，out为`torch.float16`/`torch.bfloat16`（pertoken-pergroup非对称量化）：

      $$
      out = \text{x1Scale} * \text{x2Scale} * (x1 \mathbin{@} x2 - x1 \mathbin{@} \text{x2Offset})
      $$

    </details>

    <details>

    <summary><strong>T-C和T-T量化模式</strong></summary>

      - x1、x2为`torch.int8`，无x1Scale，x2Scale为`torch.int64`/`torch.uint64`，可选参数x2Offset为`torch.float32`，可选参数bias为`torch.int32`：

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} + \text{x2Offset}
        $$

      - 参数满足如下任一条件：
        - x1、x2为`torch.int8`，无x1Scale，x2Scale为`torch.int64`/`torch.uint64`，可选参数bias为`torch.int32`；
        - x1、x2为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`，无x1Scale，x2Scale为`torch.int64`/`torch.uint64`，可选参数bias为`torch.float32`；
        - x1、x2为`torch_npu.int4`，无x1Scale，x2Scale为`torch.int64`/`torch.uint64`，可选参数bias为`torch.int32`。

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale}
        $$

      - x1、x2为`torch.int8`，无x1Scale，x2Scale为`torch.bfloat16`/`torch.float32`，可选参数bias为`torch.bfloat16`/`torch.float32`：

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} + bias
        $$

      - x1、x2为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`，x1Scale、x2Scale为`torch.float32`，可选参数bias为`torch.float32`：

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale} + bias
        $$

    </details>

    <details>

    <summary><strong>K-C和K-T量化模式</strong></summary>

      - 参数满足如下任一条件：
        - x1、x2为`torch.int8`，x1Scale为`torch.float32`，x2Scale为`torch.bfloat16`/`torch.float32`，可选参数bias为`torch.int32`；
        - x1、x2为`torch_npu.int4`，x1Scale为`torch.float32`，x2Scale为`torch.bfloat16`/`torch.float32`，可选参数bias为`torch.int32`。

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} * \text{x1Scale}
        $$

      - 参数满足如下任一条件：
        - x1、x2为`torch.int8`，x1Scale为`torch.float32`，x2Scale为`torch.bfloat16`/`torch.float32`，可选参数bias为`torch.bfloat16`/`torch.float32`；
        - x1、x2为`torch.int8`，x1Scale为`torch.float32`，x2Scale为`torch.float32`，可选参数bias为`torch.float16`/`torch.float32`；
        - x1、x2为`torch_npu.int4`，x1Scale为`torch.float32`，x2Scale为`torch.bfloat16`/`torch.float32`，可选参数bias为`torch.bfloat16`/`torch.float32`；
        - x1、x2为`torch_npu.int4`，x1Scale为`torch.float32`，x2Scale为`torch.float32`，可选参数bias为`torch.float16`/`torch.float32`；
        - x1、x2为`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`，x1Scale、x2Scale为`torch.float32`，可选参数bias为`torch.float32`。

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale} + bias
        $$

    </details>

    <details>

    <summary><strong>G-B、B-B和MX量化模式</strong></summary>

      $$
      out[m,n] = \sum_{j=0}^{kLoops-1} \left(\left(\sum_{k=0}^{gsK-1} \text{x1Slice} * \text{x2Slice}\right) * \left(\text{x1Scale}[m/gsM,j] * \text{x2Scale}[j,n/gsN]\right)\right) + bias[n]
      $$

      其中，gsM、gsN和gsK分别代表groupSizeM、groupSizeN和groupSizeK；x1Slice代表x1第m行长度为groupSizeK的向量，x2Slice代表x2第n列长度为groupSizeK的向量；K轴均从j * groupSizeK起始切片，j的取值范围为$[0, kLoops)$，$kLoops = \lceil K / groupSizeK \rceil$，K为K轴长度，支持最后的切片长度不足groupSizeK。对于G-B、B-B和MX量化模式，`[groupSizeM, groupSizeN, groupSizeK]`取值分别仅支持`[1, 128, 128]`、`[128, 128, 128]`和`[1, 1, 32]`。

    </details>

    <details>

    <summary><strong>T-CG量化模式</strong></summary>

      $$
      out = (x1 \mathbin{@} (x2 * \text{x2Scale})) * \text{yScale}
      $$

      其中，x1为`torch.float8_e4m3fn`，x2为`torch_npu.float4_e2m1fn_x2`，x2Scale为`torch.float16`/`torch.bfloat16`，yScale为`torch.int64`/`torch.uint64`，out类型与x2Scale类型一致。x2为pergroup量化，对输出进行perchannel反量化。

    </details>
  <!-- end id7 -->

## 函数原型

```python
torch_npu.npu_quant_matmul(x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None, output_dtype=None, x1_dtype=None, x2_dtype=None, pertoken_scale_dtype=None, scale_dtype=None, group_sizes=None, y_scale=None) -> Tensor
```

## 参数说明

- **x1** (`Tensor`)：必选参数，输入张量，表示矩阵乘法中的左矩阵，数据格式支持$ND$，shape需要在2-6维范围。

    <!-- npu="310p" id8 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.int8`。
    <!-- end id8 -->
    <!-- npu="910b" id9 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.int8`和`torch.int32`。其中`torch.int32`表示`torch_npu.int4`类型矩阵乘计算，每个`torch.int32`数据存放8个`torch_npu.int4`数据。
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.int8`和`torch.int32`。其中`torch.int32`表示`torch_npu.int4`类型矩阵乘计算，每个`torch.int32`数据存放8个`torch_npu.int4`数据。
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`、`torch.int32`。
      - 其中`torch.int32`表示`torch_npu.int4`类型矩阵乘计算，每个`torch.int32`数据存放8个`torch_npu.int4`数据。
      - 对于`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`，需配置可选参数`x1_dtype`为对应类型，此时x1本身dtype不再生效，但仍需保证x1 dtype为8bit位数据类型，以保证shape正确。
      - 当数据类型为`torch_npu.float4_e2m1fn_x2`时，用两个float4的数拼成一个8bit类型的数。
    <!-- end id11 -->

- **x2** (`Tensor`)：必选参数，输入张量，表示矩阵乘法中的右矩阵，数据格式支持$ND$，shape需要在2-6维范围。

    <!-- npu="310p" id12 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.int8`。
    <!-- end id12 -->
    <!-- npu="910b" id13 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.int8`和`torch.int32`，须与`x1`的数据类型保持一致（`torch.int32`含义同`x1`，表示`torch_npu.int4`类型计算）。
    <!-- end id13 -->
    <!-- npu="A3" id14 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.int8`和`torch.int32`，须与`x1`的数据类型保持一致（`torch.int32`含义同`x1`，表示`torch_npu.int4`类型计算）。
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`、`torch.float32`、`torch.int32`。
        - 对于`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`，需配置可选参数`x2_dtype`为对应类型，此时x2本身dtype不再生效，但仍需保证x2 dtype为8bit位数据类型，以保证shape正确。
        - 当数据类型为`torch_npu.float4_e2m1fn_x2`时，用两个float4的数拼成一个8bit类型的数。
        - 当数据类型为`torch.float32`时，通过`torch.float32`承载`torch_npu.float4_e2m1fn_x2`的输入，具体参考[torch\_npu.npu\_convert\_weight\_to\_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)调用示例。仅在FRACTAL\_NZ场景支持。
        - 全量化场景且数据类型为`torch.int8`（仅T-C量化、T-T量化、K-C量化或K-T量化）、`torch_npu.hifloat8`\(T-C量化、T-T量化、K-C量化、K-T量化、G-B量化或B-B量化\)、`torch.float8_e4m3fn`（mx全量化、T-C量化、T-T量化、K-C量化、K-T量化、G-B量化或B-B量化）、`torch_npu.float4_e2m1fn_x2`\(mx全量化\)，或mx伪量化场景、T-CG伪量化场景，且数据类型为`torch.float32`时，数据格式还支持FRACTAL\_NZ，可通过torch\_npu.npu\_format\_cast接口实现ND转FRACTAL\_NZ格式。
    <!-- end id15 -->

- **scale** (`Tensor`)：必选参数，量化缩放因子，数据格式支持$ND$。如需传入`torch.int64`数据类型的`scale`，需要提前调用`torch_npu.npu_trans_quant_param`来获取`torch.int64`数据类型的`scale`。

    <!-- npu="310p" id16 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.float32`、`torch.int64`。shape需要是1维$(t, )$，其中$t=1$或$n$，$n$表示`x2`的最后一维。
    <!-- end id16 -->
    <!-- npu="910b" id17 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`、`torch.int64`、`torch.bfloat16`。shape需要是1维$(t, )$，其中$t=1$或$n$，$n$表示`x2`的最后一维。
    <!-- end id17 -->
    <!-- npu="A3" id18 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.float32`、`torch.int64`、`torch.bfloat16`。shape需要是1维$(t, )$，其中$t=1$或$n$，$n$表示`x2`的最后一维。
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float32`、`torch.int64`、`torch.float16`、`torch.bfloat16`、`torch_npu.float8_e8m0fnu`（需配置可选参数scale\_dtype为对应类型，此时scale本身dtype不再生效，但仍需保证scale dtype为8bit位数据类型，以保证shape正确）。shape支持1维或多维，1维场景要求shape为\(t, \)，t=1或n，其中n与x2的n一致；多维场景的shape和dtype约束参见[约束说明](#约束说明)。
    <!-- end id19 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **offset** (`Tensor`)：仅当`scale`为2维时为必选参数，并且数据类型仅支持`torch.float16`，shape需要是2维且与`scale`相同；其他场景下为可选参数，用于调整量化后的数值偏移量。数据类型支持`torch.float32`，数据格式支持$ND$，shape需要是1维$(t,)$，$t=1$或$n$，其中$n$与`x2`的$n$一致。

- **pertoken_scale** (`Tensor`)：可选参数，用于缩放原数值以匹配量化后的范围值。数据类型支持`torch.float32`，数据格式支持$ND$，shape需要是1维$(m,)$，其中$m$与`x1`的$m$一致，表示`x1`的倒数第二维。

  <!-- npu="310p" id20 -->
  - <term>Atlas推理系列产品</term>当前不支持`pertoken_scale`。
  <!-- end id20 -->
  <!-- npu="910b" id21 -->
  - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`。shape需要是1维\(m,\)，其中m与`x1`的m一致。
  <!-- end id21 -->
  <!-- npu="A3" id22 -->
  - <term>Atlas A3系列产品</term>：数据类型支持`torch.float32`。shape需要是1维\(m,\)，其中m与`x1`的m一致。
  <!-- end id22 -->
  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`（需配置可选参数`pertoken_scale_dtype`为对应类型，此时`pertoken_scale`本身的dtype不再生效，但仍需保证`pertoken_scale`本身的dtype为8bit位的数据类型，以保证shape正确）。shape支持1维或多维，1维场景要求shape为\(m,\)或\(1,\)；多维场景的shape和dtype约束参见[约束说明](#约束说明)。
  <!-- end id23 -->

- **bias** (`Tensor`)：可选参数，偏置项，数据格式支持$ND$，$n$与`x2`的$n$一致，同时$batch$值需要等于`x1`和`x2` broadcast后推导出的$batch$值。当输出是4、5、6维时，`bias`的shape必须为1维$(n,)$；当输出是3维时，`bias`的shape可以为1维$(n,)$或3维$(batch, 1, n)$；当输出是2维时，<term>Ascend 950PR&950DT系列产品</term>的`bias`支持1维$(n,)$或2维$(1, n)$，其他产品仅支持1维$(n,)$。
    <!-- npu="310p" id24 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.int32`。
    <!-- end id24 -->
    <!-- npu="910b" id71 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.int32`、`torch.bfloat16`、`torch.float16`、`torch.float32`。
    <!-- end id71 -->
    <!-- npu="A3" id72 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.int32`、`torch.bfloat16`、`torch.float16`、`torch.float32`。
    <!-- end id72 -->
    <!-- npu="950" id73 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int32`、`torch.bfloat16`、`torch.float16`、`torch.float32`。shape支持1维\(n,\)、2维\(1, n\)或3维（batch, 1, n）。需注意的是，当输出是2维，bias的shape可为1维或2维。
    <!-- end id73 -->
- **output_dtype** (`int`)：可选参数，表示输出Tensor的数据类型。默认值为`None`，代表输出Tensor数据类型为`torch.int8`。

    <!-- npu="310p" id25 -->
    - <term>Atlas推理系列产品</term>：数据类型支持`torch.int8`、`torch.float16`。
    <!-- end id25 -->
    <!-- npu="910b" id26 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.int8`、`torch.float16`、`torch.bfloat16`、`torch.int32`。
    <!-- end id26 -->
    <!-- npu="A3" id27 -->
    - <term>Atlas A3系列产品</term>：数据类型支持`torch.int8`、`torch.float16`、`torch.bfloat16`、`torch.int32`。
    <!-- end id27 -->
    <!-- npu="950" id28 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持输入`torch.int8`、`torch.float16`、`torch.bfloat16`、`torch.float32`、`torch.int32`。
    <!-- end id28 -->

- **x1\_dtype**（`int`）：可选参数，表示`x1`实际数据类型，传入值时表示忽略x1本身的dtype，将x1中的数据视为x1\_dtype传入的类型进行计算，不传入时则直接取x1的dtype进行计算。

  <!-- npu="310p" id29 -->
  - <term>Atlas推理系列产品</term>：预留参数，当前不支持设置。
  <!-- end id29 -->
  <!-- npu="910b" id30 -->
  - <term>Atlas A2系列产品</term>：预留参数，当前不支持设置。
  <!-- end id30 -->
  <!-- npu="A3" id31 -->
  - <term>Atlas A3系列产品</term>：预留参数，当前不支持设置。
  <!-- end id31 -->
  <!-- npu="950" id32 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持`torch_npu.float4_e2m1fn_x2`、`torch_npu.hifloat8`类型。
  <!-- end id32 -->

- **x2\_dtype**（`int`）：可选参数，表示`x2`实际数据类型，传入值时表示忽略x2本身的dtype，将x2中的数据视为x2\_dtype传入的类型进行计算，不传入时则直接取x2的dtype进行计算。

  <!-- npu="310p" id33 -->
  - <term>Atlas推理系列产品</term>：预留参数，当前不支持设置。
  <!-- end id33 -->
  <!-- npu="910b" id34 -->
  - <term>Atlas A2系列产品</term>：预留参数，当前不支持设置。
  <!-- end id34 -->
  <!-- npu="A3" id35 -->
  - <term>Atlas A3系列产品</term>：预留参数，当前不支持设置。
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持`torch_npu.float4_e2m1fn_x2`、`torch_npu.hifloat8`类型。
  <!-- end id36 -->

- **pertoken\_scale\_dtype**（`int`）：可选参数，表示`pertoken_scale`实际数据类型，传入值时表示忽略pertoken\_scale本身的dtype，将pertoken\_scale中的数据视为pertoken\_scale\_dtype传入的类型进行计算，不传入时则直接取pertoken\_scale的dtype进行计算。

  <!-- npu="310p" id37 -->
  - <term>Atlas推理系列产品</term>：预留参数，当前不支持设置。
  <!-- end id37 -->
  <!-- npu="910b" id38 -->
  - <term>Atlas A2系列产品</term>：预留参数，当前不支持设置。
  <!-- end id38 -->
  <!-- npu="A3" id39 -->
  - <term>Atlas A3系列产品</term>：预留参数，当前不支持设置。
  <!-- end id39 -->
  <!-- npu="950" id40 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持`torch_npu.float8_e8m0fnu`类型。
  <!-- end id40 -->

- **scale\_dtype**（`int`）：可选参数，表示`scale`实际数据类型，传入值时表示忽略scale本身的dtype，将scale中的数据视为scale\_dtype传入的类型进行计算，不传入时则直接取scale的dtype进行计算。

  <!-- npu="310p" id41 -->
  - <term>Atlas推理系列产品</term>：预留参数，当前不支持设置。
  <!-- end id41 -->
  <!-- npu="910b" id42 -->
  - <term>Atlas A2系列产品</term>：预留参数，当前不支持设置。
  <!-- end id42 -->
  <!-- npu="A3" id43 -->
  - <term>Atlas A3系列产品</term>：预留参数，当前不支持设置。
  <!-- end id43 -->
  <!-- npu="950" id44 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持`torch_npu.float8_e8m0fnu`类型。
  <!-- end id44 -->

- **group_sizes** (`list[int]`)：可选参数，表示分组量化粒度，用于输入$m$（`x1`倒数第二维）、$n$（`x2`最后一维）、$k$（`x1`最后一维/`x2`倒数第二维）方向上的量化分组大小，格式为 [group_m, group_n, group_k]，列表必须包含3个元素，且每个元素须为0或正整数。

- **y\_scale**（`Tensor`）：可选参数，表示对输出结果做反量化。

  <!-- npu="310p" id45 -->
  - <term>Atlas推理系列产品</term>：预留参数，当前不支持设置。
  <!-- end id45 -->
  <!-- npu="910b" id46 -->
  - <term>Atlas A2系列产品</term>：预留参数，当前不支持设置。
  <!-- end id46 -->
  <!-- npu="A3" id47 -->
  - <term>Atlas A3系列产品</term>：预留参数，当前不支持设置。
  <!-- end id47 -->
  <!-- npu="950" id48 -->
  - <term>Ascend 950PR&950DT系列产品</term>：该参数仅当x1为`torch.float8_e4m3fn`、x2为`torch_npu.float4_e2m1fn_x2`时才支持。shape是2维\(1, n\)，其中n与`x2`的n一致。数据类型支持`torch.int64`，如需传入`torch.int64`类型数据，需借助torch\_npu.npu\_trans\_quant\_param来获取`torch.int64`数据类型的scale。参数约束参见[约束说明](#约束说明)。
  <!-- end id48 -->

## 返回值说明

`Tensor`

代表量化matmul的计算结果。shape支持2\~6维，形如\(batch, m, n\)，batch可不存在，支持`x1`与`x2`的batch维度broadcast，输出batch与broadcast之后的batch一致，m与`x1`的m一致，n与`x2`的n一致。

- 如果`output_dtype`为`torch.float16`，输出的数据类型为`torch.float16`。
- 如果`output_dtype`为`torch.int8`或者None，输出的数据类型为`torch.int8`。

<!-- npu="950,A3,910b" id49 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Ascend 950PR&950DT系列产品</term>：
  - 如果`output_dtype`为`torch.bfloat16`，输出的数据类型为`torch.bfloat16`。
  - 如果`output_dtype`为`torch.int32`，输出的数据类型为`torch.int32`。
<!-- end id49 -->
<!-- npu="950" id50 -->
- <term>Ascend 950PR&950DT系列产品</term>：如果`output_dtype`为`torch.float32`，输出的数据类型为`torch.float32`。
<!-- end id50 -->

## 约束说明

- 该接口支持单算子模式和TorchAir图模式。
- **公共约束**：
  - 当`x2`的数据格式需要为FRACTAL\_NZ时，一般通过`torch_npu.npu_format_cast`将ND格式转为FRACTAL\_NZ格式。
  - 当输出`out`的数据类型为`torch.int8`或`torch.float16`且无`pertoken_scale`时，图模式不支持`scale`直接传入`torch.float32`。

<!-- npu="950" id51 -->
- <term>Ascend 950PR&950DT系列产品</term>约束：
  - **空Tensor说明**：`x1`、`x2`、`scale`通常不能是空Tensor。特殊情况下，若`x2`为ND格式，对于m或n=0的空Tensor，返回空Tensor作为输出；若`x2`为FRACTAL\_NZ格式，对于m=0的空Tensor，返回空Tensor作为输出。
  - **x1、x2相关约束**：
    - `x2`数据格式（全量化）：当为T-C量化、T-T量化、K-C量化或K-T量化场景（且`x2`的数据类型为`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e4m3fn`），或为mx量化场景（且`x2`的数据类型为`torch.float8_e4m3fn`或`torch_npu.float4_e2m1fn_x2`），或为G-B、B-B量化场景（且`x2`的数据类型为`torch.float8_e4m3fn`或`torch_npu.hifloat8`）时，支持将`x2`转为FRACTAL\_NZ格式。如需将`x2`转为FRACTAL\_NZ，`x2`和`scale`的shape的所有维度不支持为1（某些特殊场景下存在1可以正常运行，但不保证使能了FRACTAL\_NZ特性）。
    - `x2`数据格式（伪量化）：`x2`支持ND格式的场景包括mx伪量化的eager模式，以及T-CG伪量化的eager模式、静态图和动态图模式。ND格式下，`x2`的数据类型为`torch_npu.float4_e2m1fn_x2`（由1个`torch.uint8`或`torch.int8`承载两个`torch_npu.float4_e2m1fn_x2`）。`x2`支持FRACTAL\_NZ格式的场景包括mx伪量化和T-CG伪量化的eager模式、静态图和动态图模式。FRACTAL\_NZ格式下，`x2`支持`torch.float32`或`torch_npu.float4_e2m1fn_x2`：所有FRACTAL\_NZ场景均支持`torch.float32`，表示1个`torch.float32`承载8个`torch_npu.float4_e2m1fn_x2`；`torch_npu.float4_e2m1fn_x2`仅支持mx伪量化场景。
  - **scale、pertoken_scale相关约束**：
    - 在mx、G-B、B-B量化中，`scale`的转置应和`x2`保持一致，`pertoken_scale`的转置应和`x1`保持一致。在mx量化中，转置节点应写在图中，`scale`的batch维度应和`x2`的batch维度保持一致，`pertoken_scale`的batch维度应和`x1`的batch维度保持一致。
    - 当输入`x1`、`x2`的数据类型为`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`且无`pertoken_scale`时，图模式不支持`scale`直接传入`torch.float32`。
  - **batch一致性相关约束**：
    - B-B量化场景不支持batch一致性。即使开启batch一致性开关，也不能保证输出满足batch一致性要求。
    - T-T量化和T-C量化场景若需满足batch一致性，在对不同m值的输入进行对比时，当`pertoken_scale`输入不为空时，`pertoken_scale`的输入值不得随`x1`的输入值动态变化，必须保持不变。
<!-- end id51 -->
<!-- npu="A3,910b,310p" id52 -->
- <term>Atlas推理系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>约束：
  - 空Tensor说明：传入的`x1`、`x2`、`scale`不能是空Tensor。
  - `x2`相关约束：
    - <term>Atlas推理系列产品</term>：必须先将`x2`转置后再转FRACTAL\_NZ。
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：推荐`x2`不转置直接转FRACTAL\_NZ。
<!-- end id52 -->
<!-- npu="310p" id53 -->
- <term>Atlas推理系列产品</term>场景下参数的数据类型约束：

    **表1** dtype组合<a id="table1"></a>

    | x1 | x2 | scale | offset | bias | pertoken_scale | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int64`/`torch.float32` | None | `torch.int32`/None | None | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int64`/`torch.float32` | `torch.float32`/None | `torch.int32`/None | None | `torch.int8` |
<!-- end id53 -->
<!-- npu="A3,910b" id54 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>场景下参数的数据类型约束：

    **表2** dtype组合<a id="table2"></a>

    | x1 | x2 | scale | offset | bias | pertoken_scale | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int64`/`torch.float32` | None | `torch.int32`/None | None | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int64`/`torch.float32` | `torch.float32`/None | `torch.int32`/None | None | `torch.int8` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | None | `torch.int32`/`torch.bfloat16`/`torch.float32`/`torch.float16`/None | `torch.float32`/None | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.float32` | None | `torch.int32`/`torch.bfloat16`/`torch.float32`/`torch.float16`/None | `torch.float32` | `torch.float16` |
    | `torch.int32` | `torch.int32` | `torch.int64`/`torch.float32` | None | `torch.int32`/None | None | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | None | `torch.int32`/None | None | `torch.int32` |
<!-- end id54 -->
<!-- npu="950" id55 -->
- <term>Ascend 950PR&950DT系列产品</term>各量化场景下参数的数据类型约束：

    `x1`、`x2`、`scale`、`pertoken_scale`与`group_sizes`参数在不同量化场景下的dtype、shape和取值等方面相互影响。以`x1`不转置、`x2`也不转置为例，关系如下。

    **表3** T-C量化和T-T量化数据类型组合<a id="table3"></a>

    | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int32` | `torch.int32` | `torch.int64`/`torch.uint64` | None | None | None/`torch.int32` | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int64` | None | None | None/`torch.int32` | `torch.float16`/`torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.int64` | None | None/`torch.float32` | None/`torch.int32` | `torch.int8` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | None | None | None/`torch.int32`/`torch.float32`/`torch.bfloat16` | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | None | None | None/`torch.int32` | `torch.int32` |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.int64` | None | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.int64` | None | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |

    > **T-C量化和T-T量化场景说明**：
    > - T-T量化场景下，`pertoken_scale`的shape为\(1,\)或None，`scale`的shape为\(1,\)。
    > - T-C量化场景下，`pertoken_scale`的shape为\(1,\)或None，`scale`的shape为\(n,\)，其中n与`x2`的n一致。
    > - `x1`、`x2`的数据类型为`torch.float8_e4m3fn`、`torch.float8_e5m2`或`torch_npu.hifloat8`时，区分静态量化和动态量化。静态量化时`scale`的数据类型为`torch.int64`，动态量化时`scale`的数据类型为`torch.float32`；`x1`、`x2`的数据类型为`torch.int8`或`torch.int32`时，不支持动态T-C或动态T-T量化。
    > - 静态量化场景下，当`x1`、`x2`为`torch_npu.int4`或`torch.int32`时，`x1`支持2～6维，`x2`仅支持2维。
    > - 静态量化场景下，`x1`与`x2`的输入类型均为`torch_npu.hifloat8`时，当`x2`的数据格式为ND时支持静态图和动态图模式；当`x2`的数据格式为FRACTAL\_NZ时仅支持静态图模式，不支持动态图模式。

    **表4** K-C量化和K-T量化数据类型组合<a id="table4"></a>

    | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int32` | `torch.int32` | `torch.float32`/`torch.bfloat16` | `torch.float32` | None | None/`torch.int32`/`torch.float32`/`torch.bfloat16` | `torch.bfloat16` |
    | `torch.int32` | `torch.int32` | `torch.float32` | `torch.float32` | None | None/`torch.int32`/`torch.float32`/`torch.float16` | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | `torch.float32` | None | None/`torch.int32`/`torch.float32`/`torch.bfloat16` | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.float32` | `torch.float32` | None | None/`torch.int32`/`torch.float32`/`torch.float16` | `torch.float16` |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |

    > **K-C量化和K-T量化场景说明**：
    > - K-C量化场景下，`pertoken_scale`的shape为\(m,\)，`scale`的shape为\(n,\)，其中m与`x1`的m一致，n与`x2`的n一致。
    > - K-T量化场景下，`pertoken_scale`的shape为\(m,\)，`scale`的shape为\(1,\)，其中m与`x1`的m一致。
    > - 当`x2`的数据格式为FRACTAL\_NZ时，仅支持`x1`不转置。

    **表5** G-B量化和B-B量化数据类型组合<a id="table5"></a>

    | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | None | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | None | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch.int8` | `torch.int8` | `torch.float32` | `torch.float32` | None | `torch.float32` | `torch.bfloat16` |

    **表6** G-B量化和B-B量化参数shape和dtype的关系<a id="table6"></a>

    | 量化模式 | x1_dtype | x2_dtype | scale数据类型 | pertoken_scale数据类型 | y_scale数据类型 | x1 shape | x2 shape | scale shape | pertoken_scale shape | y_scale shape | group_sizes值 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | B-B全量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, ceil(m/128), ceil(k/128)) | None | [128,128,128] |
    | B-B全量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, ceil(m/128), ceil(k/128)) | None | [128,128,128] |
    | G-B全量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, m, ceil(k/128)) | None | [1,128,128] |
    | G-B全量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, m, ceil(k/128)) | None | [1,128,128] |
    | G-B全量化 | `torch.int8` | `torch.int8` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, m, ceil(k/128)) | None | [1,128,128] |

    > **G-B量化和B-B量化场景说明**：
    > - G-B量化场景下，仅`torch.int8`输入支持`bias`，其余场景不支持`bias`。
    > - B-B量化场景下，不支持`torch.int8`输入，且不支持`bias`。
    > - `group_sizes`中为0的维度会自动推导。上述表中的`group_sizes`是不使用自动推导时的取值。
    > - 当`x2`的数据格式为FRACTAL\_NZ时，仅支持`x1`不转置。

    **表7** mx量化数据类型组合<a id="table7"></a>

    | 量化模式 | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | mx全量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | mx全量化 | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | mx伪量化 | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2`/`torch.float32` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | None/`torch.bfloat16`/`torch.float16` | `torch.bfloat16`/`torch.float16` |

    **表8** mx量化参数shape和dtype的关系<a id="table8"></a>

    | 量化模式 | x1_dtype | x2_dtype | scale数据类型 | pertoken_scale数据类型 | y_scale数据类型 | x1 shape | x2 shape | scale shape | pertoken_scale shape | y_scale shape | group_sizes值 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | mx全量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/64), n, 2) | (batch, m, ceil(k/64), 2) | None | [1,1,32] |
    | mx全量化 | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/64), n, 2) | (batch, m, ceil(k/64), 2) | None | [1,1,32] |
    | mx伪量化 | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | (m, k) | (n, k/2) | (n, k/64, 2) | (m, k/64, 2) | None | [1,1,32] |
    | mx伪量化 | `torch.float8_e4m3fn` | `torch.float32` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | (m, k) | (n, k/8) | (n, k/64, 2) | (m, k/64, 2) | None | [1,1,32] |

    > **mx量化场景说明**：
    > - mx全量化场景下，`x1`与`x2`的输入类型均为`torch_npu.float4_e2m1fn_x2`时，内轴必须为偶数，k必须大于2。
    > - mx全量化场景下，`x1`与`x2`的输入类型均为`torch_npu.float4_e2m1fn_x2`时，若`x2`的数据格式为FRACTAL\_NZ，仅支持`x1`不转置，且k或n不能为1。
    > - mx全量化场景下，`x1`与`x2`的输入类型均为`torch_npu.float4_e2m1fn_x2`时，当`x2`的数据格式为ND时支持静态图和动态图模式；当`x2`的数据格式为FRACTAL\_NZ时仅支持静态图模式，不支持动态图模式。
    > - 在mx全量化动态图模式下，若`x2`为`torch.float8_e4m3fn`/`torch.float8_e5m2`（ND格式）、`torch.float8_e4m3fn`（FRACTAL\_NZ格式）或`torch_npu.float4_e2m1fn_x2`（ND格式），必须同时满足：k值大于64，且`x1`与`x2`的batch轴维度均不为1。
    > - mx全量化场景下，若`x2`为FRACTAL\_NZ格式，则`x1`和`x2`的数据类型必须均为`torch_npu.float4_e2m1fn_x2`或均为`torch.float8_e4m3fn`。
    > - mx全量化场景下，`scale`、`pertoken_scale`仅最后三轴支持非连续的Tensor。
    > - mx伪量化场景下，当`x1`的数据类型为`torch.float8_e4m3fn`，`x2`的数据类型为`torch_npu.float4_e2m1fn_x2`时，仅支持`x1`不转置、`x2`转置。如果`x2`的数据格式为ND，要求k是8的倍数，k、n大小不能超过2<sup>31</sup>-1，该场景不支持图模式。如果`x2`的数据格式为FRACTAL\_NZ，要求k、n是8的倍数且大小不能超过2<sup>31</sup>-1，该场景支持图模式，但要求k大于64。
    > - mx伪量化场景下，当`x1`的数据类型为`torch.float8_e4m3fn`，`x2_dtype`为`torch.float32`时，仅支持`x1`不转置、`x2`转置，且`x2`必须为FRACTAL\_NZ格式。`x1`、`x2`的k值必须是8的倍数且大小不能超过2<sup>31</sup>-1，`x2`的n值必须是8的倍数且大小不能超过2<sup>31</sup>-1。该场景支持图模式，但要求k大于64。
    > - mx伪量化场景下，`bias`为可选参数，数据类型支持`torch.bfloat16`或`torch.float16`，且与输出数据类型保持一致。数据格式支持ND，shape支持2维，表示为\(1, n\)。
    > - `group_sizes`中为0的维度会自动推导。上述表中的`group_sizes`是不使用自动推导时的取值。

    **表9** T-CG伪量化数据类型组合<a id="table9"></a>

    | 量化模式 | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | T-CG伪量化 | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2`/`torch.float32` | `torch.bfloat16`/`torch.float16` | None | None | None | `torch.bfloat16`/`torch.float16` |

    **表10** T-CG伪量化参数shape和dtype的关系<a id="table10"></a>

    | x1_dtype | x2_dtype | scale数据类型 | pertoken_scale数据类型 | y_scale数据类型 | x1 shape | x2 shape | scale shape | pertoken_scale shape | y_scale shape | group_sizes值 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2` | `torch.bfloat16`/`torch.float16` | None | `torch.int64` | (m, k) | (n, k/2) | (n, k/32) | None | (1, n) | [1,1,32] |
    | `torch.float8_e4m3fn` | `torch.float32` | `torch.bfloat16`/`torch.float16` | None | `torch.int64` | (m, k) | (k, n/8) | (k/32, n) | None | (1, n) | [1,1,32] |

    > **T-CG伪量化场景说明**：
    > - T-CG伪量化模式下，`y_scale`的数据类型支持`torch.int64`，数据格式支持ND，shape支持2维，表示为\(1, n\)。
    > - T-CG伪量化场景下，当`x1`的数据类型为`torch.float8_e4m3fn`，`x2_dtype`为`torch_npu.float4_e2m1fn_x2`时，`x2`要求为ND格式，仅支持`x1`不转置、`x2`转置。`x1`、`x2`的k值必须是32的倍数且不等于32，并且大小不能超过2<sup>31</sup>-1；`x2`的n值大小不能超过2<sup>31</sup>-1。
    > - T-CG伪量化场景下，当`x1`的数据类型为`torch.float8_e4m3fn`，`x2_dtype`为`torch.float32`时，`x2`要求为FRACTAL\_NZ格式，仅支持`x1`不转置、`x2`不转置。`x1`、`x2`的k值必须是32的倍数且不等于32，并且大小不能超过2<sup>31</sup>-1；`x2`的n值必须是8的倍数且大小不能超过2<sup>31</sup>-1。
    > - T-CG伪量化场景下，不支持`bias`。
    > - `group_sizes`中为0的维度会自动推导。上述表中的`group_sizes`是不使用自动推导时的取值。
<!-- end id55 -->
<!-- npu="A3,910b,310p" id74 -->
- **`torch_npu.int4`类型计算的额外约束**：

  仅适用于<term>Atlas推理系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>。

  - 当`x1`、`x2`的数据类型均为`torch.int32`时，每个`torch.int32`类型的数据存放8个`torch_npu.int4`数据。输入的`torch.int32` shape需要将数据原本为`torch_npu.int4`类型时shape的最后一维缩小8倍。`torch_npu.int4`数据的shape最后一维应为8的倍数。例如，进行\(m, k\)乘\(k, n\)的`torch_npu.int4`类型矩阵乘计算时，需要输入`torch.int32`类型、shape为\(m, k//8\)、\(k, n//8\)的数据，其中k与n都应是8的倍数。`x1`只能接受shape为\(m, k//8\)且数据排布连续的数据，`x2`可以接受shape为\(k, n//8\)且数据排布连续的数据，或shape为\(k//8, n\)且由数据连续排布的\(n, k//8\)转置而来的数据。
  - 如果在PyTorch图模式中使用本接口，且环境变量`ENABLE_ACLNN=false`，则在调用接口前需要对shape为\(n, k//8\)的`x2`数据进行转置，转置过程应写在图中。

    > [!NOTE]  
    > 数据排布连续是指数组中所有相邻的数，包括换行时内存地址连续。使用`Tensor.is_contiguous`返回值为`True`，则表明Tensor数据排布连续。
<!-- end id74 -->

## 调用示例

- 单算子调用
  - `torch.int8`类型输入场景：

    ```python
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    B = 31
    cpu_x1 = torch.randint(-5, 5, (1, M, K), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (B, K, N), dtype=torch.int8)
    scale = torch.randn(N, dtype=torch.float32)
    offset = torch.randn(N, dtype=torch.float32)
    bias = torch.randint(-5, 5, (B, 1, N), dtype=torch.int32)
    # Method 1：You can directly call npu_quant_matmul
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), offset=offset.npu(), bias=bias.npu())

    # Method 2: You can first call npu_trans_quant_param to convert scale and offset from float32 to int64 when output dtype is not torch.bfloat16 and pertoken_scale is none
    scale_1 = torch_npu.npu_trans_quant_param(scale.npu(), offset.npu())
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale_1,  bias=bias.npu())
    ```

  <!-- npu="950" id56 -->
  - `torch_npu.hifloat8`类型+双路scale场景，示例代码如下，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    B = 31
    cpu_x1 = torch.randint(-5, 5, (1, M, K), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (B, K, N), dtype=torch.int8)
    scale_x2 = torch.randn(1, dtype=torch.float32)
    scale_x1 = torch.randn(1, dtype=torch.float32)
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale_x2.npu(), pertoken_scale=scale_x1.npu(), x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8, output_dtype=torch.float16)
    ```
  <!-- end id56 -->
  <!-- npu="950" id57 -->
  - `torch.float8_e4m3fn`类型+pertoken\_scale场景，示例代码如下，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    B = 31
    cpu_x1 = torch.randint(-5, 5, (1, M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    cpu_x2 = torch.randint(-5, 5, (B, K, N), dtype=torch.int8).to(torch.float8_e5m2)
    scale_x2 = torch.randn(N, dtype=torch.float32)
    scale_x1 = torch.randn(M, dtype=torch.float32)
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale_x2.npu(), pertoken_scale=scale_x1.npu(), output_dtype=torch.float16)
    ```
  <!-- end id57 -->
  <!-- npu="950" id58 -->
  - `torch.float8_e4m3fn`类型+`torch_npu.float4_e2m1fn_x2`类型+双路scale场景，示例代码如下，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import math
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    cpu_x1 = torch.randint(-5, 5, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    # float4_e2m1fn_x2数据类型可用int8或uint8表示
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8)
    # float8_e8m0fnu数据类型可用int8或uint8表示
    scale_x2 = torch.randint(1, 5, (N, int(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(1, 5, (M, int(K/64), 2), dtype=torch.int8)
    bias = torch.randint(-5, 5, (1, N), dtype=torch.bfloat16)
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu().transpose(-1, -2), scale_x2.npu().transpose(-3,-2), bias=bias.npu(), pertoken_scale=scale_x1.npu(), output_dtype=torch.bfloat16, x2_dtype=torch_npu.float4_e2m1fn_x2, pertoken_scale_dtype=torch_npu.float8_e8m0fnu, scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[0, 0, 32])
    ```
  <!-- end id58 -->
  <!-- npu="950" id59 -->
  - `torch.float8_e4m3fn`类型+`torch_npu.float4_e2m1fn_x2`类型+双路scale场景+weightNZ场景，示例代码如下，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import torch
    import torch_npu
    M = 256
    K = 768
    N = 128
    cpu_x1 = torch.randint(-5, 5, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    # float4_e2m1fn_x2数据类型可用int8或uint8表示
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8)
    npu_x2 = cpu_x2.npu()
    npu_x2 = torch_npu.npu_format_cast(npu_x2, 29, customize_dtype = torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2)
    # float8_e8m0fnu数据类型可用int8或uint8表示
    scale_x2 = torch.randint(1, 5, (N, int(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(1, 5, (M, int(K/64), 2), dtype=torch.int8)
    bias = torch.randint(-5, 5, (1, N), dtype=torch.int16).to(torch.bfloat16)
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), npu_x2.transpose(-1, -2), scale_x2.npu().transpose(-3,-2), bias=bias.npu(), pertoken_scale=scale_x1.npu(), output_dtype=torch.bfloat16, x2_dtype=torch_npu.float4_e2m1fn_x2, pertoken_scale_dtype=torch_npu.float8_e8m0fnu, scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[0, 0, 32])
    ```
  <!-- end id59 -->
  <!-- npu="950" id60 -->
  - `torch.float8_e4m3fn`+`torch_npu.float4_e2m1fn_x2`+FRACTAL\_NZ场景，代码示例如下，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import torch
    import torch_npu
    import numpy as np
    from ml_dtypes import float4_e2m1fn
    from ml_dtypes import float8_e4m3fn

    m = 128
    k = 64
    n = 64
    group_sizes = 32
    E2M1_MIN, E2M1_MAX = -6, 6

    cpu_x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).to(torch.float8_e4m3fn)
    cpu_x2 = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((k, n))).astype(float4_e2m1fn)
    cpu_x2 = torch.from_numpy(cpu_x2.astype(np.float32))

    x2_scale = torch.randint(1, 5, (k//group_sizes, n), dtype=torch.bfloat16).npu()
    y_scale = torch.randint(-5, 5, (1, n), dtype=torch.float32).npu()
    x2_dtype = None
    bias = None
    pertoken_scale=None
    pertoken_scale_dtype = None
    scale_dtype = None

    npu_x2 = cpu_x2.npu()
    npu_x2 = torch_npu.npu_format_cast(npu_x2, 29, customize_dtype=torch.float8_e4m3fn)
    npu_x2 = torch_npu.npu_convert_weight_to_int4pack(npu_x2)
    # per-group场景把y_scale从float32变为int64
    if (pertoken_scale == None) :
        y_scale = torch_npu.npu_trans_quant_param(y_scale)

    npu_out = torch_npu.npu_quant_matmul(
        cpu_x1.npu(),
        npu_x2,
        x2_scale,
        bias=bias,
        pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=pertoken_scale_dtype,
        scale_dtype=scale_dtype,
        output_dtype=torch.bfloat16,
        x2_dtype=x2_dtype,
        group_sizes=[1, 1, group_sizes],
        y_scale=y_scale)
    ```
  <!-- end id60 -->
  <!-- npu="950" id61 -->
  - mx全量化：`torch_npu.float4_e2m1fn_x2`+`torch_npu.float4_e2m1fn_x2` weightNZ类型，示例代码如下，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import math
    import torch
    import torch_npu
    import logging
    import os
    M = 64
    N = 7168
    K = 2112
    cpu_x1 = torch.randint(-5, 5, (M, int(K/2)), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8)
    scale_x2 = torch.randint(-5, 5, (N, math.ceil(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(-5, 5, (M, math.ceil(K/64), 2), dtype=torch.int8)
    x1_npu = cpu_x1.npu()
    x2_npu = cpu_x2.npu()
    x2_npu = torch_npu.npu_format_cast(x2_npu, 29)
    scale_x2_npu = scale_x2.npu()
    scale_x1_npu = scale_x1.npu()
    # 调用npu_quant_matmul函数，指定x1_dtype和x2_dtype为torch_npu.float4_e2m1fn_x2
    npu_out = torch_npu.npu_quant_matmul(
    x1_npu,
    x2_npu.transpose(-1,-2),
    scale_x2_npu.transpose(0, 1),
    pertoken_scale=scale_x1_npu,
    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
    output_dtype=torch.float16,
    group_sizes=[1,1,32],
    x1_dtype=torch_npu.float4_e2m1fn_x2,
    x2_dtype=torch_npu.float4_e2m1fn_x2,
    scale_dtype=torch_npu.float8_e8m0fnu
    )
    print('npu_out is ', npu_out.cpu())
    ```
  <!-- end id61 -->
  <!-- npu="950" id62 -->
  - mx全量化：`torch_npu.float4_e2m1fn_x2`+`torch_npu.float4_e2m1fn_x2` weightND类型，示例代码如下，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import math
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    cpu_x1 = torch.randint(-5, 5, (M, int(K/2)), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8)
    scale_x2 = torch.randint(-5, 5, (N, math.ceil(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(-5, 5, (M, math.ceil(K/64), 2), dtype=torch.int8)

    x1_npu = cpu_x1.npu()
    x2_npu = cpu_x2.npu().transpose(-1,-2)
    scale_x2_npu = scale_x2.npu().transpose(0, 1)
    scale_x1_npu = scale_x1.npu()

    # 调用npu_quant_matmul函数，指定x1_dtype和x2_dtype为torch_npu.float4_e2m1fn_x2
    npu_out = torch_npu.npu_quant_matmul(
        x1_npu,
        x2_npu,
        scale_x2_npu,
        pertoken_scale=scale_x1_npu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        output_dtype=torch.float16,
        group_sizes=[1,1,32],
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
        scale_dtype=torch_npu.float8_e8m0fnu
    )
    ```
  <!-- end id62 -->

- 图模式调用（ND数据格式）
  - 输出`torch.float16`

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    # "ENABLE_ACLNN"是否开启走aclnn, true: 回调走aclnn, false: 在线编译
    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    M = 1
    K = 512
    N = 128
    B = 15
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias):
            return torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, bias=bias, output_dtype=torch.float16)
    cpu_model = MyModel()
    model = cpu_model.npu()
    cpu_x1 = torch.randint(-1, 1, (B, M, K), dtype=torch.int8)
    cpu_x2 = torch.randint(-1, 1, (B, K, N), dtype=torch.int8)
    scale = torch.randn(1, dtype=torch.float32)
    # pertoken_scale为空时，输出fp16必须先调用npu_trans_quant_param，将scale(offset)从float转为int64.
    scale_1 = torch_npu.npu_trans_quant_param(scale.npu(), None)
    bias = torch.randint(-1,1, (B, 1, N), dtype=torch.int32)
    # dynamic=True: 动态图模式，dynamic=False: 静态图模式
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale_1, None, bias.npu())
    ```

  <!-- npu="A3,910b" id75 -->
  - 输出`torch.bfloat16`，示例代码如下，仅支持如下产品：

    <!-- npu="910b" id63 -->
    - <term>Atlas A2系列产品</term>
    <!-- end id63 -->
    <!-- npu="A3" id64 -->
    - <term>Atlas A3系列产品</term>
    <!-- end id64 -->

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
            return torch_npu.npu_quant_matmul(x1, x2.t(), scale, offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16)
    cpu_model = MyModel()
    model = cpu_model.npu()
    m = 15
    k = 11264
    n = 6912
    bias_flag = True
    cpu_x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
    cpu_x2 = torch.randint(-1, 1, (n, k), dtype=torch.int8)
    scale = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
    pertoken_scale = torch.randint(-1,1, (m,), dtype=torch.float32)

    bias = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    if bias_flag:
        npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), None, bias.npu(), pertoken_scale.npu())
    else:
        npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), None, None, pertoken_scale.npu())
    ```
  <!-- end id75 -->

- 图模式调用（FRACTAL\_NZ）

  <!-- npu="950" id65 -->
  - mx全量化：`torch_npu.float4_e2m1fn_x2`+`torch_npu.float4_e2m1fn_x2` weightNZ类型，示例代码如下，仅支持<term>Ascend 950PR&950DT系列产品</term>

    ```python
    import torch
    import math
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    # "ENABLE_ACLNN"是否开启走aclnn, true: 回调走aclnn, false: 在线编译
    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
            return torch_npu.npu_quant_matmul(x1, x2.transpose(-1,-2), scale.transpose(0,1), offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16,group_sizes=[1,1,32],
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        scale_dtype=torch_npu.float8_e8m0fnu
        )
    cpu_model = MyModel()
    model = cpu_model.npu()
    M = 64
    N = 7168
    K = 2112
    cpu_x1 = torch.randint(-5, 5, (M, int(K/2)), dtype=torch.int8).npu()
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8).npu()
    scale_x2 = torch.randint(-5, 5, (N, math.ceil(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(-5, 5, (M, math.ceil(K/64), 2), dtype=torch.int8)
    cpu_x2_t_29 = torch_npu.npu_format_cast(cpu_x2, 29)
    #cpu_x2_t_29 = cpu_x2
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    npu_out = model(cpu_x1, cpu_x2_t_29, scale_x2.npu(), None, None, scale_x1.npu())
    ```
  <!-- end id65 -->
  <!-- npu="950,310p" id76 -->
  - 将x2转置\(batch,  **n, k**\)后转format，示例代码如下，仅支持如下产品：

    <!-- npu="310p" id66 -->
    - <term>Atlas推理系列产品</term>
    <!-- end id66 -->
    <!-- npu="950" id67 -->
    - <term>Ascend 950PR&950DT系列产品</term>
    <!-- end id67 -->

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    M = 1
    K = 512
    N = 128
    B = 15
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias):
            return torch_npu.npu_quant_matmul(x1, x2.transpose(2,1), scale, offset=offset, bias=bias)
    cpu_model = MyModel()
    model = cpu_model.npu()
    cpu_x1 = torch.randint(-1, 1, (B, M, K), dtype=torch.int8).npu()
    cpu_x2 = torch.randint(-1, 1, (B, K, N), dtype=torch.int8).npu()
    # Process x2 into a high-bandwidth format(29) offline to improve performance, please ensure that the input is continuous with (batch,n,k) layout
    cpu_x2_t_29 = torch_npu.npu_format_cast(cpu_x2.transpose(2,1).contiguous(), 29)
    scale = torch.randn(1, dtype=torch.float32).npu()
    offset = torch.randn(1, dtype=torch.float32).npu()
    bias = torch.randint(-1,1, (N,), dtype=torch.int32).npu()
    # Process scale from float32 to int64 offline to improve performance
    scale_1 = torch_npu.npu_trans_quant_param(scale, offset)
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=False)
    npu_out = model(cpu_x1, cpu_x2_t_29, scale_1, offset, bias)
    ```
  <!-- end id76 -->

  <!-- npu="950,A3,910b" id77 -->
  - 将x2非转置\(batch,  **k, n**\)后转format，示例代码如下，仅支持如下产品：

    <!-- npu="910b" id68 -->
    - <term>Atlas A2系列产品</term>
    <!-- end id68 -->
    <!-- npu="A3" id69 -->
    - <term>Atlas A3系列产品</term>
    <!-- end id69 -->
    <!-- npu="950" id70 -->
    - <term>Ascend 950PR&950DT系列产品</term>
    <!-- end id70 -->

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
            return torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16)
    cpu_model = MyModel()
    model = cpu_model.npu()
    m = 15
    k = 11264
    n = 6912
    bias_flag = True
    cpu_x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
    cpu_x2 = torch.randint(-1, 1, (n, k), dtype=torch.int8)
    # Process x2 into a high-bandwidth format(29) offline to improve performance, please ensure that the input is continuous with (batch,k,n) layout
    x2_notranspose_29 = torch_npu.npu_format_cast(cpu_x2.npu().transpose(1,0).contiguous(), 29)
    scale = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
    pertoken_scale = torch.randint(-1,1, (m,), dtype=torch.float32)

    bias = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    if bias_flag:
        npu_out = model(cpu_x1.npu(), x2_notranspose_29, scale.npu(), None, bias.npu(), pertoken_scale.npu())
    else:
        npu_out = model(cpu_x1.npu(), x2_notranspose_29, scale.npu(), None, None, pertoken_scale.npu())
    ```
  <!-- end id77 -->
