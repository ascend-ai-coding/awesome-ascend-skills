# （beta）torch_npu.npu_format_cast_

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->

## 功能说明

原地修改`input`的数据格式为目标格式。

## 函数原型

```python
torch_npu.npu_format_cast_(input, src, *, customize_dtype=None, input_dtype=None) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：必选参数，待处理的输入张量。

  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据维度支持2、3维，数据类型支持`torch.int8`、`torch.float8_e4m3fn`、`torch.int32`、`torch.float16`、`torch.bfloat16`。
  <!-- end id6 -->

- **src**（`Tensor`/`torch.int`/`Format`）：必选参数，目标格式。可输入张量、整数或torch_npu.Format类型。

  - 若输入张量，则将`input`的数据格式修改为此张量的格式。例如将`input`的数据格式转换为ND格式时，此处可以输入*ND格式的张量*。
  - 若输入整数，则将`input`的数据格式修改为整数值对应的torch_npu.Format。例如将`input`的数据格式转换为ND格式时，此处可以输入`2`。

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT系列产品</term>：当前仅支持取29（ACL\_FORMAT\_FRACTRAL\_NZ）。
    <!-- end id7 -->

  - 若输入torch_npu.Format，则将`input`的数据格式修改为该格式。例如将`input`的数据格式转换为ND格式时，此处可以输入`torch_npu.Format.ND`。torch_npu.Format表示torch_npu的数据格式，torch_npu支持如下数据格式：

    |torch_npu.Format类型|整数值|说明|
    | ------| ------|:------: |
    |torch_npu.Format.UNDEFINED|-1|未知数据格式。对应的AscendCL数据格式为ACL_FORMAT_UNDEFINED。|
    |torch_npu.Format.NCHW|0|NCHW格式。对应的AscendCL数据格式为ACL_FORMAT_NCHW。|
    |torch_npu.Format.NHWC|1|NHWC格式。对应的AscendCL数据格式为ACL_FORMAT_NHWC。|
    |torch_npu.Format.ND|2|表示支持任意格式，除了Square、Tanh等这些单输入对自身处理的算子外，其他算子需谨慎使用。对应的AscendCL数据格式为ACL_FORMAT_ND。|
    |torch_npu.Format.NC1HWC0|3|5维数据格式。其中，C0与微架构强相关，该值等于cube单元的size，例如16；C1是将C维度按照C0切分：C1=C/C0， 若结果不整除，最后一份数据需要填充到C0。对应的AscendCL数据格式为ACL_FORMAT_NC1HWC0。|
    |torch_npu.Format.FRACTAL_Z|4|卷积的权重的格式。对应的AscendCL数据格式为ACL_FORMAT_FRACTAL_Z。|
    |torch_npu.Format.NC1HWC0_C04|12|5维数据格式。其中，C0固定为4，C1是将C维度按照C0切分：C1=C/C0， 若结果不整除，最后一份数据需要padding到C0。当前版本不支持。对应的AscendCL数据格式为ACL_FORMAT_NC1HWC0_C04。|
    |torch_npu.Format.HWCN|16|HWCN格式。对应的AscendCL数据格式为ACL_FORMAT_HWCN。|
    |torch_npu.Format.NDHWC|27|NDHWC格式。对于3维图像就需要使用带D（Depth）维度的格式。对应的AscendCL数据格式为ACL_FORMAT_NDHWC。|
    |torch_npu.Format.FRACTAL_NZ|29|内部格式，用户目前无需使用。对应的AscendCL数据格式为ACL_FORMAT_FRACTAL_NZ。|
    |torch_npu.Format.NCDHW|30|NCDHW格式。对于3维图像就需要使用带D（Depth）维度的格式。对应的AscendCL数据格式为ACL_FORMAT_NCDHW。|
    |torch_npu.Format.NDC1HWC0|32|6维数据格式。相比于NC1HWC0，仅多了D（Depth）维度。对应的AscendCL数据格式为ACL_FORMAT_NDC1HWC0。|
    |torch_npu.Format.FRACTAL_Z_3D|33|3D卷积权重格式，例如Conv3D/MaxPool3D/AvgPool3D这些算子均需以这种格式来表达。对应的AscendCL数据格式为ACL_FORMAT_FRACTAL_Z_3D。|
    |torch_npu.Format.NC|35|2维数据格式。对应的AscendCL数据格式为ACL_FORMAT_NC。|
    |torch_npu.Format.NCL|47|3维数据格式。对应的AscendCL数据格式为ACL_FORMAT_NCL。|

    > [!NOTE]
    > 数据排布格式具体可参考《CANN Ascend C算子开发》中的“<a href="https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/technical_appendix/concepts_and_terms/neural_networks_and_operators/data_layout.md">数据排布格式</a>”章节。

<!-- npu="950,A3,910b" id14 -->
- **customize_dtype**（`int`）：可选参数，用于指定格式转换时的目标数据类型。该参数可控制C0值，默认值为`None`，`torch.float32`和`torch.int32`数据类型的默认C0值为16，`torch.int8`数据类型的默认C0值为32。

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2训练系列产品</term>、<term>Atlas A3训练系列产品</term>：传入`3`（对应int(`torch.int32`)）时，FRACTAL_NZ格式的C0值为8。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：对于仅对权重量化的MatMul场景，对权重W做私有格式转换时，需传入A矩阵的数据类型来推断W的C0轴大小。数据类型支持`torch.int8`、`torch.float8_e4m3fn`、`torch.float16`、`torch.bfloat16`。若使用默认值`None`，表示A的dtype和W的dtype一样，推断出W的C0轴大小。
  <!-- end id9 -->

<!-- end id14 -->
- **input\_dtype**（`int`）：表示`input`的真实数据类型，主要用于非torch原生数据类型，例如`float4`。

  <!-- npu="A3,910b,910,310p" id10 -->
  - <term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>、<term>Atlas A2训练系列产品</term>、<term>Atlas A3训练系列产品</term>：暂不支持该参数，使用默认值`None`即可。
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持该参数。默认值`None`表示`input`的真实类型和tensor的数据类型一致。
  <!-- end id11 -->

## 约束说明

<!-- npu="950,A3,910b" id15 -->
`customize_dtype`参数仅在Atlas A2训练系列产品/Atlas A3训练系列产品/Ascend 950PR&950DT系列产品且CANN版本为9.1.0及以上的场景下支持。其他产品或CANN 9.1.0以下版本，传入该参数将导致异常。

<!-- end id15 -->
<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  目前输入参数支持如下组合，当传入为第三种组合时，转换出来的format为50（ACL\_FORMAT\_FRACTRAL\_NZ\_C0\_16）。

    | `input`数据类型 | `input` format | `acl_format`取值 | `customize_dtype`取值 | `input_dtype`取值 |
    | --- | --- | --- | --- | --- |
    | `torch.int8` | $ND$ | 29 | `torch.int8` | NA |
    | `torch.float8_e4m3fn` | $ND$ | 29 | `torch.float8_e4m3fn` | NA |
    | `torch.int32` | $ND$ | 29 | `torch.float16`/`torch.bfloat16` | NA |
    | `torch.float16` | $ND$ | 29 | `torch.float16` | NA |
    | `torch.bfloat16` | $ND$ | 29 | `torch.bfloat16` | NA |
    | `torch_npu.float4_e2m1fn_x2`（`torch.float32`承载） | $ND$ | 29 | `torch.float8_e4m3fn` | NA |
    | `torch_npu.float4_e2m1fn_x2`（`torch.uint8`/`torch.int8`承载） | $ND$ | 29 | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2` |
    | `torch_npu.float4_e2m1fn_x2`（`torch.uint8`/`torch.int8`承载） | $FRACTAL\_NZ$ | 2 | NA | `torch_npu.float4_e2m1fn_x2` |
    | `torch_npu.float4_e2m1fn_x2`（`torch.uint8`/`torch.int8`承载） | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | `torch_npu.float4_e2m1fn_x2` |
    | `torch_npu.float4_e1m2fn_x2`（`torch.uint8`/`torch.int8` 承载） | $ND$ | 29 | NA | `torch_npu.float4_e1m2fn_x2` |
    | `torch.uint8` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.uint8` | $FRACTAL\_NZ\_C0\_16$ | 2 | NA | NA |
    | `torch.uint8` | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | NA |
    | `torch.int8` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.int8` | $FRACTAL\_NZ\_C0\_16$ | 2 | NA | NA |
    | `torch.int8` | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | NA |
    | `torch.float8_e4m3fn` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.float16` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.bfloat16` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.int32` | $FRACTAL\_NZ\_C0\_2$ | 2 | NA | NA |
    | `torch.int32` | $FRACTAL\_NZ\_C0\_4$ | 2 | NA | NA |
    | `torch.int32` | $FRACTAL\_NZ\_C0\_16$ | 2 | NA | NA |
    | `torch.int32` | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | NA |
    | `torch.float32` | $FRACTAL\_NZ\_C0\_2$ | 2 | NA | NA |
    | `torch.float32` | $FRACTAL\_NZ\_C0\_4$ | 2 | NA | NA |
    | `torch.float32` | $FRACTAL\_NZ\_C0\_16$ | 2 | NA | NA |
    | `torch.float32` | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | NA |

  当前不支持的特殊场景：

  - $ND$转$FRACTAL\_NZ$场景，当`srcTensor.dtype`和`additionalDtype`相同且类型为`torch.float16`、`torch.bfloat16`时，若维度表示为\[k, n\]，则k为1场景暂不支持。
  - 调用本接口转为$FRACTAL\_NZ$格式后，不支持进行任何能修改Tensor的操作，包括contiguous、pad、view、slice等。
  - `srcTensor`的shape后两维任意一维度shape等于1场景，不允许转$FRACTAL\_NZ$后进行任何能修改Tensor的操作，包括transpose等。
  - $FRACTAL\_NZ$转$ND$场景，不支持输入`srcTensor`非连续。
<!-- end id12 -->

## 返回值说明

`Tensor`

返回原地修改后的`input`。

## 调用示例

- 整数值调用示例：

    ```python
     >>> import torch
     >>> import torch_npu
     >>> x = torch.rand(2, 3, 4, 5).npu()
     >>> torch_npu.get_npu_format(x)
     0
     >>> torch_npu.get_npu_format(torch_npu.npu_format_cast_(x, 2))
     2
    ```

- Format类型调用示例：

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.rand(2, 3, 4, 5).npu()
    >>> x2 = torch_npu.npu_format_cast_(x, torch_npu.Format.NHWC)
    >>> torch_npu.get_npu_format(x2)
    1
    ```

<!-- npu="950" id13 -->
- 使用 `input_dtype` （仅<term>Ascend 950PR&950DT系列产品</term>）：

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randint(-5, 5, (1,64), dtype=torch.int8).npu()
    >>> y = torch_npu.npu_format_cast(x, 29, customize_dtype=torch.int8, input_dtype=torch.int8)
    >>> torch_npu.get_npu_format(y)
    29
    ```
<!-- end id13 -->
