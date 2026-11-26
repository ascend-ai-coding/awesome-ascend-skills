# torch_npu.npu_rms_norm_quant

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
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->

## 功能说明

- API功能：RmsNormQuant算子是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。RmsNormQuant算子将RmsNorm算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出的操作。
- 计算公式：
  
  $$
  quant\_in_i = \frac{x_i}{Rms(x)}g_i+b_i, where \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  y = round((quant\_in * scale) + offset)
  $$
  上面公式中的`round`操作支持CAST_RINT模式。

## 函数原型

```python
torch_npu.npu_rms_norm_quant(x, gamma, beta, scale, offset, epsilon=1e-06, dst_dtype=torch.int8) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，表示标准化过程中的源数据张量，对应公式中的$x$，数据格式支持$ND$，shape支持 1-8 维，支持非连续的Tensor，不支持空Tensor。

  <!-- npu="310p,310b" id6 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>：数据类型支持`torch.float16`。
  <!-- end id6 -->
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id8 -->

- **gamma** (`Tensor`)：必选参数，表示标准化过程中的缩放张量，对应公式中的$g$，shape支持1-2维，若shape为1维，则需与`x`最后一维维度一致；若shape为2维，则第一维必须为1，第二维需与`x`最后一维维度一致。数据类型需与`x`保持一致，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。

  <!-- npu="310p,310b" id9 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>：数据类型支持`torch.float16`。
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id11 -->

- **beta** (`Tensor`)：必选参数，表示标准化过程中的偏移张量，对应公式中的$b$。shape支持1-2维，规则同`gamma`。数据类型需与`x`保持一致，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。

  <!-- npu="310p,310b" id12 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>：数据类型支持`torch.float16`。
  <!-- end id12 -->
  <!-- npu="A3,910b" id13 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id14 -->

- **scale** (`Tensor`)：必选参数，表示量化过程中得到y进行的scale张量，对应公式中的$scale$。shape为1，维度为1。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。该参数的值不能为0。

  <!-- npu="310p,310b" id15 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>：数据类型支持`torch.float16`。
  <!-- end id15 -->
  <!-- npu="A3,910b" id16 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
  <!-- end id16 -->
  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id17 -->

- **offset** (`Tensor`)：必选参数，表示量化过程中的偏移张量，对应公式中的$offset$。shape需与 `scale` 保持一致，数据格式支持$ND$，支持非连续Tensor，不支持空Tensor。

  <!-- npu="A3,910b,310p,310b" id18 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：数据类型支持 `torch.int8`。
  <!-- end id18 -->
  <!-- npu="950" id19 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch.int32`、`torch.float16`、`torch.bfloat16`、`torch.float32`。
  <!-- end id19 -->

- **epsilon** (`float`)：可选参数，对应公式中的$eps$，用于防止除零错误，默认值为1e-6。建议传入较小的正数。
- **dst_dtype** (`int`): 可选参数，指定量化输出的类型，默认值为`torch.int8`。传None时当做`torch.int8`处理。

  <!-- npu="A3,910b,310p,310b" id20 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：支持取值 `torch.int8`、`torch.quint4x2`。
  <!-- end id20 -->
  <!-- npu="950" id21 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持取值 `torch.int8`、`torch.quint4x2`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`。
  <!-- end id21 -->

## 返回值说明
  
  `Tensor`

  返回结果，对应公式中的$y$，即最终量化输出张量，数据类型由`dst_dtype`指定。当`dst_dtype`是`torch.quint4x2`时，`y`的数据类型为`torch.int32`，形状最后一维为`x`最后一维除以8，其余维度与`x`一致，每个`torch.int32`元素包含8个`torch_npu.int4`结果。其他场景下`y`形状与输入`x`一致，数据类型由`dst_dtype`指定。

## 约束说明

<!-- npu="310p" id22 -->
- <term>Atlas推理系列产品</term>：x、y的尾轴长度，以及gamma的尾轴长度必须大于等于32 Bytes。
<!-- end id22 -->
<!-- npu="910b" id23 -->
- <term>Atlas A2系列产品</term>：当`dst_dtype`取`torch.quint4x2`时，`x`、`gamma`以及`beta`的最后一维必须为偶数，并且`x`最后一维必须能够被8整除。
<!-- end id23 -->
- 各产品型号支持数据类型说明：
  
  <!-- npu="A3,910b" id24 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

    | x | gamma | beta | scale | offset | epsilon | y |
    | --------- | ------------- | ------------- | ------------- | -------------- | --------- |--------- |
    | `torch.float16`   | `torch.float16`       | `torch.float16`       | `torch.float16`       | `torch.int8`           | torch.double      |`torch.int8`      |
    | `torch.bfloat16`  | `torch.bfloat16`      | `torch.bfloat16`      | `torch.bfloat16`      | `torch.int8`           | torch.double      |`torch.int8`      |
    | `torch.float16`   | `torch.float16`       | `torch.float16`       | `torch.float16`       | `torch.int8`           | torch.double      |`torch.int32`      |
    | `torch.bfloat16`  | `torch.bfloat16`      | `torch.bfloat16`      | `torch.bfloat16`      | `torch.int8`           | torch.double      |`torch.int32`      |
  <!-- end id24 -->
  <!-- npu="310p,310b" id25 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>：

    | x | gamma | beta | scale | offset | epsilon | y |
    | --------- | ------------- | ------------- | ------------- | -------------- | --------- |--------- |
    | `torch.float16`   | `torch.float16`       | `torch.float16`       | `torch.float16`       | `torch.int8`           | torch.double      |`torch.int8`      |
    | `torch.float16`   | `torch.float16`       | `torch.float16`       | `torch.float16`       | `torch.int8`           | torch.double      |`torch.int32`      |
  <!-- end id25 -->
  <!-- npu="950" id26 -->
  - <term>Ascend 950PR&950DT系列产品</term>：

    | x | gamma | beta | scale | offset | epsilon | y |
    | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.int8` | torch.double | `torch.int8`、`torch.int32`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.int8` | torch.double | `torch.int8`、`torch.int32`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | torch.double | `torch.int8`、`torch.int32`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | torch.double | `torch.int8`、`torch_npu.int4`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8` |
    | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | torch.double | `torch.int8`、`torch.int32`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` | `torch.int32` | torch.double | `torch.int8`、`torch.int32`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.int32` | torch.double | `torch.int8`、`torch.int32`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | torch.double | `torch.int8`、`torch.int32`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | torch.double | `torch.int8`、`torch.int32`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8` |
  <!-- end id26 -->

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> eps = 1e-6
>>> x = torch.randn(16, dtype=torch.float16).npu()
>>> gamma = torch.randn(16, dtype=torch.float16).npu()
>>> beta = torch.zeros(16, dtype=torch.float16).npu()
>>> scale = torch.ones(1, dtype=torch.float16).npu()
>>> offset = torch.zeros(1, dtype=torch.int8).npu()
>>> y = torch_npu.npu_rms_norm_quant(x, gamma, beta, scale, offset, eps)
>>> y.cpu().numpy()
    tensor([ 1, -1,  2,  0, -2,  1,  0,  1,  2,  0,  2,  0,  0,  0,  0,  0],
        device='npu:0', dtype=torch.int8)
```
