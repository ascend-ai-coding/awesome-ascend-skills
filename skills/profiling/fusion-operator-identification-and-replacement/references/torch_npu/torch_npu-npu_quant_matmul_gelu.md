# torch_npu.npu_quant_matmul_gelu

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：完成量化的矩阵乘和GELU激活函数的融合计算，支持A8W8和A4W4量化。该接口融合了量化矩阵乘和GELU激活，减少内存访问，提升性能。

- 计算公式：
    - 量化矩阵乘计算：
        - 当`bias`为`torch.int32`时：
        $$
        qbmmout = (x1 \mathbin{@} x2 + \text{bias}) * x2Scale * x1Scale
        $$
        - 当`bias`为`torch.bfloat16`/`torch.float16`/`torch.float32`时：
        $$
        qbmmout = x1 \mathbin{@} x2 * x2Scale * x1Scale + \text{bias}
        $$
        - 当`bias`为None时:
        $$
        qbmmout = x1@x2 * x2Scale * x1Scale
        $$

    - GELU激活函数，GELU类型由`approximate`输入指定：
        - 当`approximate`为`gelu_tanh`时：
        $$
        out = gelu\_tanh(qbmmout)
        $$
        - 当`approximate`为`gelu_erf`时（默认值）：
        $$
        out = gelu\_erf(qbmmout)
        $$

## 函数原型

```python
torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, *, bias=None, approximate="gelu_erf") -> Tensor
```

## 参数说明

- **x1** (`Tensor`)：必选参数，输入张量，表示矩阵乘法中的左矩阵（激活值），数据格式支持$ND$，shape需要在2-6维范围。数据类型支持`torch.int8`（A8W8量化）、`torch.int32`（A4W4量化，每个`torch.int32`数据存放8个`torch_npu.int4`数据）和`torch_npu.int4`（A4W4量化，直接`torch_npu.int4`类型）。

- **x2** (`Tensor`)：必选参数，输入张量，表示矩阵乘法中的右矩阵（权重），其与`x1`的数据类型须保持一致。数据格式支持$ND$或$NZ$（昇腾亲和排布格式），shape需要在2-6维范围。数据类型支持`torch.int8`（A8W8量化）、`torch.int32`（A4W4量化，每个`torch.int32`数据存放8个`torch_npu.int4`数据）和`torch_npu.int4`（A4W4量化，直接`torch_npu.int4`类型）。<br>
A8W8量化场景下，支持昇腾亲和的$NZ$数据排布格式，可通过`torch_npu.npu_format_cast`转换为$NZ$格式以提升性能。

- **x1_scale** (`Tensor`)：必选参数，`x1`的量化缩放因子，数据格式支持$ND$。数据类型支持`torch.float32`。shape需要是1维$(m,)$，其中$m$与`x1`的$m$一致。采用pertoken量化方式，每个token（行）有一个独立的scale值。

- **x2_scale** (`Tensor`)：必选参数，`x2`的量化缩放因子，数据格式支持$ND$。数据类型支持`torch.float32`、`torch.bfloat16`。shape需要是1维$(n,)$或$(1,)$，其中$n$与`x2`的$n$一致。采用perchannel量化方式，每个输出通道有一个独立的scale值，或使用pertensor量化（shape为$(1,)$）。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **bias** (`Tensor`)：可选参数，默认值为`None`，偏置项，数据格式支持$ND$。数据类型支持`torch.int32`、`torch.float32`、`torch.bfloat16`、`torch.float16`。

    - A4W4量化场景下：shape仅支持1维$(n,)$，$n$与`x2`的$n$一致。
    - A8W8量化场景下：shape支持1维$(n,)$或3维$(batch, 1, n)$，$n$与`x2`的$n$一致，同时batch值需要等于`x1`和`x2` broadcast后推导得出的batch值。

- **approximate** (`str`)：可选参数，默认值为`"gelu_erf"`。指定GELU激活函数的类型。支持`"gelu_tanh"`（GELU的tanh近似版本）和`"gelu_erf"`（GELU的erf精确版本）。

## 返回值说明

`Tensor`

代表量化矩阵乘融合GELU激活的计算结果。

- 输出数据类型的确定规则：
  - 如果`x2_scale`的数据类型为`torch.float32`，输出的数据类型为`torch.float16`。
  - 如果`x2_scale`的数据类型为`torch.bfloat16`，输出的数据类型为`torch.bfloat16`。
  - 如果`bias`的数据类型为`torch.bfloat16`，输出的数据类型强制为`torch.bfloat16`（优先级高于`x2_scale`）。
- 输出shape为$(batch, m, n)$，其中$batch$根据`x1`和`x2`的batch通过broadcast推导得出。

## 约束说明

- 该接口支持推理场景下使用。
- 传入的`x1`、`x2`、`x1_scale`、`x2_scale`不能为空。
- `x1`与`x2`最后一维的shape大小不能超过65535。

- **A4W4量化（`torch_npu.int4`/`torch.int32`类型输入）的额外约束**：

    A4W4量化场景支持两种输入类型：
    - **`torch_npu.int4`类型**：直接使用`torch_npu.int4`数据类型。
    - **`torch.int32`类型**：每个`torch.int32`数据存放8个`torch_npu.int4`数据。
    
    当使用`torch.int32`类型时，输入的`torch.int32` shape需要将数据原本`torch_npu.int4`类型时shape的最后一维缩小8倍。`torch_npu.int4`数据的shape最后一维应为8的倍数。

    - `x1`和`x2`的内轴（k轴）必须为偶数。
    - 当`x2`为`torch.int32`类型时，`x2`的shape为$(k, n//8)$，$n$必须是8的倍数。
    - 当`x2`为`torch_npu.int4`类型时，`x2`的shape为$(k, n)$，$n$必须是8的倍数。
    - A4W4量化仅支持$ND$格式，不支持$NZ$格式。
    - 转置信息由算子内部根据tensor的stride自动推导，无需手动指定。

- **A8W8量化的约束**：

    - 支持$ND$格式和$NZ$格式。
    - 如果需要使用$NZ$格式以提升性能，可以手动调用`torch_npu.npu_format_cast`完成输入`x2`（weight）的$NZ$格式转换。
    - 转置信息由算子内部根据tensor的stride自动推导，无需手动指定。

<!-- npu="A3,910b" id3 -->
- 输入参数间支持的数据类型组合情况如下：

    **表 1** <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>

    | x1    | x2    | x1_scale | x2_scale  | bias                                | 输出数据类型    |
    |-------|-------|----------|-----------|-------------------------------------|-----------|
    | `torch.int8`  | `torch.int8`  | `torch.float32`  | `torch.float32`   | `torch.int32`/`torch.float32`/`torch.bfloat16`/`torch.float16`/None | `torch.float16`   |
    | `torch.int8`  | `torch.int8`  | `torch.float32`  | `torch.bfloat16`  | `torch.int32`/`torch.float32`/`torch.bfloat16`/`torch.float16`/None | `torch.bfloat16`  |
    | `torch.int32` | `torch.int32` | `torch.float32`  | `torch.float32`   | `torch.int32`/None                          | `torch.float16`   |
    | `torch.int32` | `torch.int32` | `torch.float32`  | `torch.bfloat16`  | `torch.int32`/None                          | `torch.bfloat16`  |
    | `torch_npu.int4`  | `torch_npu.int4`  | `torch.float32`  | `torch.float32`   | `torch.int32`/None                          | `torch.float16`   |
    | `torch_npu.int4`  | `torch_npu.int4`  | `torch.float32`  | `torch.bfloat16`  | `torch.int32`/None                          | `torch.bfloat16`  |
<!-- end id3 -->

## 调用示例

- 单算子调用（A8W8量化，ND格式，gelu_tanh激活）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- 单算子调用（A8W8量化，ND格式，gelu_erf激活，带bias）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>> bias = torch.randn(n, dtype=torch.float32).npu() * 0.1
    >>>
    >>> # 使用gelu_erf激活并添加bias
    >>> output = torch_npu.npu_quant_matmul_gelu(
    ...     x1, x2, x1_scale, x2_scale, bias=bias, approximate="gelu_erf"
    ... )
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- 单算子调用（A8W8量化，NZ格式，gelu_tanh激活）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>>
    >>> # 将x2转换为NZ格式以提升性能
    >>> x2_nz = torch_npu.npu_format_cast(x2.contiguous(), 29)  # 29为ACL_FORMAT_FRACTAL_NZ
    >>>
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # 自动识别NZ格式并调用对应接口
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2_nz, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- 单算子调用（A8W8量化，`torch.bfloat16`输出）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 64, 128, 256
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.bfloat16).abs().npu() * 0.01  # bfloat16 scale
    >>>
    >>> # 输出数据类型由x2_scale的类型决定，此处输出为bfloat16
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.dtype)  # torch.bfloat16
    ```

- 单算子调用（A4W4量化）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> # 生成int4数据（以int32格式存储）
    >>> # 注意：实际使用时需要通过量化接口将float32数据量化为int4并打包为int32
    >>> x1 = torch.randint(-8, 8, (m, k // 8), dtype=torch.int32).npu()
    >>> x2 = torch.randint(-8, 8, (k, n // 8), dtype=torch.int32).npu()
    >>>
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # A4W4量化仅支持ND格式，不支持NZ格式
    >>> # 转置信息由算子内部根据tensor的stride自动推导
    >>> output = torch_npu.npu_quant_matmul_gelu(
    ...     x1, x2, x1_scale, x2_scale, 
    ...     approximate="gelu_tanh"
    ... )
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- 单算子调用（使用默认approximate="gelu_erf"）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 64, 128, 256
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # 不指定approximate参数，使用默认值"gelu_erf"
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale)
    >>> print(output.dtype)  # torch.float16
    ```
