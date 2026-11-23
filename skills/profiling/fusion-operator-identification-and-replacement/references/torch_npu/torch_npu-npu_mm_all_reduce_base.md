# torch_npu.npu_mm_all_reduce_base

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- **API功能**：TP切分（Tensor Parallelism，张量并行）场景下，实现mm和all\_reduce的融合，融合算子内部实现计算和通信流水并行。

- 计算公式：
    $$
    output = allreduce\left(x1 \mathbin{@} \left((x2 + antiquantOffset) * antiquantScale\right) + bias + x3\right)
    $$

> [!NOTE]
> 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者更高版本，否则将会引发报错，比如BUS ERROR等。

## 函数原型

```python
torch_npu.npu_mm_all_reduce_base(x1, x2, hcom, *, reduce_op='sum', bias=None, antiquant_scale=None, antiquant_offset=None, x3=None, dequant_scale=None, pertoken_scale=None, comm_quant_scale_1=None, comm_quant_scale_2=None, comm_turn=0, antiquant_group_size=0, group_sizes=None, y_dtype=None, x1_dtype=None, x2_dtype=None, dequant_scale_dtype=None, pertoken_scale_dtype=None, comm_quant_mode=0, comm_mode=None) -> Tensor
```

## 参数说明

- **x1**（`Tensor`）：**必选参数**，数据格式支持$ND$，输入shape支持2维或者3维。

    <!-- npu="910b" id3 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.int8`、`torch.float16`、`torch.bfloat16`。
    <!-- end id3 -->
    <!-- npu="950" id4 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch.float16`、`torch.bfloat16`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1fn_x2`。
    <!-- end id4 -->

- **x2**（`Tensor`）：**必选参数**，数据类型需要和`x1`保持一致，输入shape维度第0维和`x1`的最后一维保持一致。

    <!-- npu="910b" id5 -->
    - <term>Atlas A2系列产品</term>：数据格式支持$NZ$（昇腾亲和排布格式）、$ND$。
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据格式支持$ND$。
    <!-- end id6 -->

- **hcom**（`str`）：**必选参数**，通信域handle名，通过get\_hccl\_comm\_name接口获取。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **reduce\_op**（`str`）：**可选参数**，reduce操作类型，当前版本仅支持'sum'（默认值）。
- **bias**（`Tensor`）：**可选参数**，默认值为`None`，数据格式支持$ND$。bias当前仅支持一维，且维度大小与`output`/`x2`的最后一维大小相同。

    <!-- npu="910b" id7 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.int32`、`torch.float16`、`torch.bfloat16`。
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int32`、`torch.float16`、`torch.bfloat16`、`torch.float32`。perblock场景，仅支持bias输入空。
    <!-- end id8 -->

- **antiquant\_scale**（`Tensor`）：**可选参数**，默认值为`None`，伪量化场景对`x2`进行去量化的系数，数据类型支持`torch.float16`、`torch.bfloat16`，数据格式支持$ND$。伪量化场景数据类型需要和`x1`保持一致。
    - pertensor场景：shape为\[1\]。
    - perchannel场景：shape为\[1,n\]或者\[n\]，n为`x2`最后一维的大小。
    - pergroup场景：shape为\[ceil\(k, antiquant\_group\_size\), n\]。其中k为`x2`第一维的大小，n为`x2`最后一维的大小，`antiquant_group_size`为伪量化场景对输入`x2`进行反量化计算的groupSize输入。

        > [!NOTE]
        > ceil\(k, antiquant\_group\_size\)的计算逻辑为：\(k+antiquant\_group\_size-1\)/antiquant\_group\_size，并对计算结果取整数部分。

- **antiquant\_offset**（`Tensor`）：**可选参数**，默认值为`None`，伪量化场景对`x2`进行去量化的系数，数据类型支持`torch.float16`、`torch.bfloat16`，数据格式支持$ND$。数据类型、shape需要和`antiquant_scale`保持一致。
- **x3**（`Tensor`）：**可选参数**，默认值为`None`，matmul计算后的偏移。数据格式支持$ND$。数据类型、shape需要和输出`output`保持一致。

    <!-- npu="910b" id9 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
    <!-- end id10 -->

- **dequant\_scale**（`Tensor`）：**可选参数**，默认值为`None`，matmul计算后的去量化系数。数据格式支持$ND$。支持的量化场景如下：
    - pertensor场景：shape为\[1\]。
    - perchannel场景：shape为\[n\]/\[1,n\]，n为`x2`最后一维的大小。
    - mx量化场景：数据类型为`torch_npu.float8_e8m0fnu`时，仅支持转置，`x2` shape为\[n, k\]时，x2Scale的shape为\[n, ceilDiv\(k, 64\), 2\]，且必须保证ceilDiv\(k, 32\)为偶数。
    - perblock场景：shape为\[ceilDiv\(n, 128\), ceilDiv\(k, 128\)\]。

    <!-- npu="910b" id11 -->
    - <term>Atlas A2系列产品</term>：支持pertensor、perchannel场景。数据类型支持`torch.int64`、`torch.uint64`、`torch.bfloat16`、`torch.float32`。
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持pertensor、perchannel、mx、perblock场景。数据类型支持`torch.int64`、`torch.uint64`、`torch.bfloat16`、`torch.float32`、`torch_npu.float8_e8m0fnu`。
    <!-- end id12 -->

- **pertoken\_scale**（`Tensor`）：**可选参数**，默认值为`None`，matmul计算后的pertoken去量化系数。若数据类型为`torch.float32`，当`x1`为\[m,k\]时，`pertoken_scale` shape为\[m\]；当`x1`为\[b, s, k\]时，`pertoken_scale` shape为\[b\*s\]。

    <!-- npu="910b" id13 -->
    - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`。
    <!-- end id13 -->
    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float32`、`torch_npu.float8_e8m0fnu`。若数据类型为`torch_npu.float8_e8m0fnu`，mx场景：shape为\[s, ceilDiv\(k, 64\), 2\]或者\[m, ceilDiv\(k, 64\), 2\]，且必须保证ceilDiv\(k, 32\)为偶数；perblock场景：shape为\[ceilDiv\(m, 128\), ceilDiv\(k, 128\)\]。
    <!-- end id14 -->

- **comm\_quant\_scale\_1**（`Tensor`）：**可选参数**，默认值为`None`，alltoall通信前后的量化、去量化系数。支持`torch.float16`、`torch.bfloat16`，支持$ND$格式。`x2`为\[k, n\]时shape为\[1, n\]或\[n\]，用户需保证每张卡上数据保持一致且正确。
- **comm\_quant\_scale\_2**（`Tensor`）：**可选参数**，默认值为`None`，allgather通信前后的量化、去量化系数。支持`torch.float16`、`torch.bfloat16`，支持$ND$格式。`x2`为\[k, n\]时shape为\[1, n\]或\[n\]，用户需保证每张卡上数据保持一致且正确。
- **comm\_turn**（`int`）：**可选参数**，表示rank间通信切分粒度，默认值为0，表示默认的切分方式。**当前版本仅支持输入0。**
- **antiquant\_group\_size**（`int`）：**可选参数**，表示伪量化pre-group算法模式下，对输入`x2`进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在k轴方向的大小。当伪量化算法模式不为pre-group时传入0；当伪量化算法模式为pre-group时传入值的范围为\[32, min\(k-1, INT\_MAX\)\]且值要求是32的倍数，其中k为`x2`第一维的大小。默认值0，为0则表示非per-group场景。
- **group\_sizes**（`List[int]`）：**可选参数**，默认值为`None`，用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向`x1`/`x2`输入的多少个数的反量化。group\_sizes为\[groupSizeM，groupSizeN，groupSizeK\]。groupSizeM，groupSizeN，groupSizeK表示一个反量化系数在各个维度对应的数的个数。支持参数自动推导，当根据计算公式分解的groupSizeM/groupSizeN/groupSizeK任一或多个参数为0时，算子自动推导对应的参数值，推导原理为：假设groupSizeM=0，表示m方向量化分组值由接口推断，推断公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与`x1` shape中的m一致，scaleM与`x1Scale` shape中的m一致。

    <!-- npu="910b" id15 -->
    - <term>Atlas A2系列产品</term>：**暂不支持该参数。**
    <!-- end id15 -->

- **y\_dtype**（`int`）：**可选参数**，默认值为`None`，代表输出数据类型。支持取值：5表示`torch.float16`、6表示`torch.float32`、15表示`torch.bfloat16`。

    <!-- npu="910b" id16 -->
    - <term>Atlas A2系列产品</term>：**暂不支持该参数。**
    <!-- end id16 -->

- **x1\_dtype**（`int`）：**可选参数**，默认值为`None`，代表输入数据类型，当`x1` tensor的数据类型为`torch_npu.hifloat8`时，需要输入`torch_npu.hifloat8`，类型为`torch_npu.float4_e2m1fn_x2`时输入`torch_npu.float4_e2m1fn_x2`。

    <!-- npu="910b" id17 -->
    - <term>Atlas A2系列产品</term>：**暂不支持该参数。**
    <!-- end id17 -->

- **x2\_dtype**（`int`）：**可选参数**，默认值为`None`，代表输入数据类型，当`x2` tensor的数据类型为`torch_npu.hifloat8`时，需要输入`torch_npu.hifloat8`，类型为`torch_npu.float4_e2m1fn_x2`时输入`torch_npu.float4_e2m1fn_x2`。

    <!-- npu="910b" id18 -->
    - <term>Atlas A2系列产品</term>：**暂不支持该参数。**
    <!-- end id18 -->

- **dequant\_scale\_dtype**（`int`）：**可选参数**，默认值为`None`，代表输入数据类型，当`dequant_scale`数据类型为`torch_npu.float8_e8m0fnu`时，需输入`torch_npu.float8_e8m0fnu`类型。

    <!-- npu="910b" id19 -->
    - <term>Atlas A2系列产品</term>：**暂不支持该参数。**
    <!-- end id19 -->

- **pertoken\_scale\_dtype**（`int`）：**可选参数**，默认值为`None`，代表输入数据类型，当`pertoken_scale`数据类型为`torch_npu.float8_e8m0fnu`时，需输入`torch_npu.float8_e8m0fnu`类型。

    <!-- npu="910b" id20 -->
    - <term>Atlas A2系列产品</term>：**暂不支持该参数。**
    <!-- end id20 -->

- **comm\_quant\_mode**（`int`）：**可选参数**，默认值为`0`，代表低比特通信的量化模式，取值为0或1，当需要动态量化的低比特通信时，需要输入为1。

    <!-- npu="910b" id21 -->
    - <term>Atlas A2系列产品</term>：**暂不支持该参数。**
    <!-- end id21 -->

- **comm\_mode**（`str`）：**可选参数**，表示通信引擎模式，默认值为`None`。

    <!-- npu="910b" id22 -->
    - <term>Atlas A2系列产品</term>：取值支持`None`和`ai_cpu`，均使用AI CPU通信。
    <!-- end id22 -->
    <!-- npu="950" id23 -->
    - <term>Ascend 950PR&950DT系列产品</term>：取值支持`None`、`ai_cpu`和`ccu`。`ai_cpu`和`ccu`直接指定AI CPU和CCU通信；`None`默认使用AI CPU通信。
    <!-- end id23 -->

## 返回值说明

`Tensor`

输出张量。数据类型非量化场景以及伪量化场景与`x1`保持一致，全量化场景输出数据类型为`torch.float16`或`torch.bfloat16`。shape第0维度和`x1`的0维保持一致，若`x1`为2维，shape第1维度和`x2`的1维保持一致，若`x1`为3维，shape第1维度和`x1`的1维保持一致，shape第2维度和`x2`的1维保持一致。

## 约束说明

- 该接口支持推理场景下使用。增量场景不开启该融合算子，全量场景开启该融合算子。
- **通信引擎约束**：

  <!-- npu="910b" id24 -->
  - <term>Atlas A2系列产品</term>：支持Host CPU+TS通信。
  <!-- end id24 -->
  <!-- npu="950" id25 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持CCU和AI CPU通信，CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。
  <!-- end id25 -->

- 该接口支持单算子模式和TorchAir图模式。
- 输入`x1`可为2维或者3维、`x2`必须是2维，分别为\(b, s, k\)/\(m, k\), \(k, n\)，k轴满足mm算子入参要求，k轴相等。bias当前仅支持一维，且维度大小与output的最后一维大小相同。x3的shape与output的shape相同。
- `x1`不支持输入转置后的tensor，`x2`转置后输入，需要满足shape的第一维大小与`x1`的最后一维相同，满足matmul的计算条件。
- `antiquant_group_size`中k值的范围与matmul一致，为\[1,65535\]，INT\_MAX大于\(k-1\)。

<!-- npu="910b" id26 -->
- <term>Atlas A2系列产品</term>：
  - 数据类型支持`torch.bfloat16`。
  - `x1`、`x2`不支持为空tensor。
  - 支持1、2、4、8卡，并且仅支持HCCS链路all mesh组网。
  - `comm_quant_scale_1`，`comm_quant_scale_2`的shape应保持一致，dtype与输出的dtype保持一致，且只在全量化场景支持。
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - 非量化场景：支持k为0的场景，输出为bias + x3。支持bs/m/n为0，此时传入的输出也应是空tensor，此场景不进入kernel计算，直接返回。
  - 全量化场景：不支持空tensor。
  - 伪量化场景：仅支持k轴为0的空tensor。
  - 仅支持1、2、4、8、16、32、64卡，CCU模式下不支持1卡。
<!-- end id27 -->
- 非量化场景：b\*s、m、k、n的值均不得超过2147483647\(`torch.int32`\_MAX\)。
- 全量化场景：b\*s、m取值范围均为\[1, 2147483647\]，`x1`、`x2`的最后一维范围为\[1, 65535\]，即k的取值范围为\[1, 65535\]、仅当x2\(shape=\[n, k\]\)为转置时n可以大于65535。
- 伪量化场景：b\*s、m取值范围均为\[1, 2147483647\]，k、n的取值范围为\[1, 65535\]。
- 一个模型中的通算融合MC2算子，仅支持相同通信域。

<!-- npu="910b" id28 -->
- <term>Atlas A2系列产品</term>：一个模型中的通算融合算子（AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce），仅支持相同通信域。
<!-- end id28 -->
- 在长序列场景，随着b/s或者m的增大，可能出现内存不足或者计算超时。
- 不同量化场景下参数支持的数据类型组合：

    **表 1**  非量化场景

    | x1 | x2 | bias | x3 | output（输出） | antiquant_scale | antiquant_offset | dequant_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | None | None |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | None | None | None | None | None |

    <!-- npu="910b" id29 -->
    **表 2**  伪量化场景(针对<term>Atlas A2系列产品</term>)

    | x1 | x2 | bias | x3 | output（输出） | antiquant_scale | antiquant_offset | dequant_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.int8` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None |
    | `torch.bfloat16` | `torch.int8` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | None | None | None |
    <!-- end id29 -->

    <!-- npu="950" id30 -->
    **表 3**  伪量化场景(针对<term>Ascend 950PR&950DT系列产品</term>)

    | x1 | x2 | bias | x3 | output（输出） | antiquant_scale | antiquant_offset | dequant_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch_npu.hifloat8`/`torch.float8_e4m3fn` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | None |
    | `torch.bfloat16` | `torch_npu.hifloat8`/`torch.float8_e4m3fn` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | None | None | None | None |
    <!-- end id30 -->

    <!-- npu="910b" id31 -->
    **表 4**  全量化场景(针对<term>Atlas A2系列产品</term>)

    | x1 | x2 | bias | x3 | output（输出） | antiquant_scale | antiquant_offset | dequant_scale | pertoken_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float16` | `torch.float16` | None | None | `torch.uint64`/`torch.int64` | None | None/`torch.float16` | None/`torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | `torch.bfloat16` | None | None | `torch.bfloat16` | None | None/`torch.bfloat16` | None/`torch.bfloat16` |
    <!-- end id31 -->

    <!-- npu="950" id32 -->
    **表 5**  全量化场景(针对<term>Ascend 950PR&950DT系列产品</term>)

    | x1 | x2 | bias | x3 | output（输出） | antiquant_scale | antiquant_offset | dequant_scale | pertoken_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float16` | `torch.float16` | None | None | `torch.float32` | `torch.float32` | None/`torch.float16` | None/`torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | `torch.bfloat16` | None | None | `torch.bfloat16` | `torch.float32` | None/`torch.bfloat16` | None/`torch.bfloat16` |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch.float32` | `torch.float32` | None | None |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch.float32` | `torch.float32` | None | None |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | None |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | None |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | None | None |

    > [!NOTE]
    > 全量化场景：若`dequant_scale`需要以`torch.float32`类型传入，在调用torch\_npu.npu\_mm\_all\_reduce\_base前，需通过torch\_npu.npu\_trans\_quant\_param接口对`dequant_scale`进行处理为`torch.int64`类型（处理方法见对应的接口使用说明）。
    <!-- end id32 -->

- 全量化场景中，`x1`、`x2`、`dequant_scale`、`pertoken_scale`、groupSize在不同量化场景下的dtype与shape取值关系如下表。

    | 量化类型 | x1 | x2 | dequant_scale | pertoken_scale | [groupSizeM,groupSizeN,groupSizeK] | 约束 |
    | --- | --- | --- | --- | --- | --- | --- |
    | perblock量化 | (b,s,k) | (k,n)/(n,k) | (ceilDiv(k,128),ceilDiv(n,128))/(ceilDiv(n,128),ceilDiv(k,128)) | (b,ceilDiv(m,128),ceilDiv(k,128)) | [128,128,128] | - |
    | perblock量化 | (m,k) | (k,n)/(n,k) | (ceilDiv(k,128),ceilDiv(n,128))/(ceilDiv(n,128),ceilDiv(k,128)) | (ceilDiv(m,128),ceilDiv(k,128)) | [128,128,128] | - |
    | MXFP量化 | (b,s,k) | (n,k) | (n,ceilDiv(k,64),2) | (s,ceilDiv(k,64),2) | [1,1,32] | x1、x2输入类型为`torch_npu.float4_e2m1fn_x2`时，必须保证ceilDiv\(k, 32\)为偶数。 |
    | MXFP量化 | (m,k) | (n,k) | (n,ceilDiv(k,64),2) | (m,ceilDiv(k,64),2) | [1,1,32] | x1、x2输入类型为`torch_npu.float4_e2m1fn_x2`时，必须保证ceilDiv\(k, 32\)为偶数。 |

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    def run_mm_all_reduce_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)

        input_ = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = torch_npu.npu_mm_all_reduce_base(input_, weight, hcom_info, reduce_op='sum')
        print("output: ", output)

    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16

        mp.spawn(run_mm_all_reduce_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

- 图模式调用

    非量化、伪量化、全量化场景下FRACTAL\_NZ格式示例如下：

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    class MM_ALLREDUCE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale):
            output_npu = torch_npu.npu_mm_all_reduce_base(x1=x1,
                                                          x2=x2,
                                                          hcom=hcom,
                                                          reduce_op=reduce_op,
                                                          bias=bias,
                                                          antiquant_scale=antiquant_scale,
                                                          antiquant_offset=antiquant_offset,
                                                          x3=x3,
                                                          dequant_scale=dequant_scale
                                                          )
            return output_npu

    class MM_ALLREDUCE_A8W8_GRAPH_Model(MM_ALLREDUCE_GRAPH_Model):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale):
            output_npu = torch_npu.npu_mm_all_reduce_base(x1=x1,
                                                          x2=x2.t(),
                                                          hcom=hcom,
                                                          reduce_op=reduce_op,
                                                          bias=bias,
                                                          antiquant_scale=antiquant_scale,
                                                          antiquant_offset=antiquant_offset,
                                                          x3=x3,
                                                          dequant_scale=dequant_scale
                                                          )
            return output_npu

    def define_model(model, graph_type):
        import torchair
        if graph_type == 1:  # 传统入图模式，静态shape+在线编译场景
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN入图模式，动态shape+二进制
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        else:
            print("Error type")
        return model

    def get_graph(input, weight, hcomm_info, dequant_scale, bias, antiquant_scale, antiquant_offset, x3):
        model = MM_ALLREDUCE_A8W8_GRAPH_Model()
        model = define_model(model, 2) # 1:静态入图;2:动态入图;
        output = model(x1=input, x2=weight, hcom=hcomm_info, reduce_op="sum", bias=bias, antiquant_scale=antiquant_scale,
                       antiquant_offset=antiquant_offset, x3=x3, dequant_scale=dequant_scale)
        return output

    def run_mc2_a16w16(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.float16)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        output_a16w16 = get_graph(input, weight, hcom_info, None, None, None, None, None)
        return output_a16w16

    def run_mc2_a8w8(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.int8)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        # weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
        dequant_scale = torch.randn(x2_shape[0], dtype=torch.float32).uniform_(float(-10), float(10)).npu()
        dequant_scale = torch_npu.npu_trans_quant_param(dequant_scale)
        output_a8w8 = get_graph(input, weight, hcom_info, dequant_scale, None, None, None, None)
        return output_a8w8

    def run_mc2_a16w8(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        # weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
        antiquant_scale = torch.randn(x2_shape[0], dtype=torch.float16).uniform_(float(-1), float(1)).npu()
        antiquant_offset = torch.ones(x2_shape[0], dtype=torch.float16).npu()
        output_a16w8 = get_graph(input, weight, hcom_info, None, None, antiquant_scale, antiquant_offset, None)
        return output_a16w8

    def run_mm_all_reduce_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, op_type):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        output = None
        # 非量化调用
        if op_type == "a16w16":
            output = run_mc2_a16w16(x1_shape, x2_shape, hcom_info)
        # 伪量化调用
        if op_type == "a16w8":
            output = run_mc2_a16w8(x1_shape, x2_shape, hcom_info)
        # 全量化调用
        if op_type == "a8w8":
            output = run_mc2_a8w8(x1_shape, x2_shape, hcom_info)
        print("output:", output)
    if __name__ == "__main__":
        worksize = 2
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [1280, 5120]
        x2_shape = [640, 5120]
        op_type = "a16w16" # Options: a16w16, a16w8, a8w8
        mp.spawn(run_mm_all_reduce_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, op_type), nprocs=worksize)
    ```
