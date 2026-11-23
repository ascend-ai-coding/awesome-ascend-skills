# torch_npu.npu_grouped_matmul_add

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：实现分组矩阵乘累加（GroupedMatmulAdd）计算，每组矩阵乘的维度大小可以不同，基本功能为矩阵乘加，如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i]+y_i[m_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应shape，输入输出数据类型均为Tensor，K轴分组。
  - k轴分组：$k_i$各不相同，但$m_i/n_i$每组相同。
  - group\_list支持两种解读方式（由group\_list\_type指定）：
    - group\_list\_type为0（默认值）：group\_list中的数值为分组轴大小的cumsum结果（累积和）。
    - group\_list\_type为1：group\_list中的数值为分组轴上每组大小。
- 计算公式：

  $$
  yRef_i^{out}=x_i\times weight_i + yRef_i^{in}
  $$

  其中$yRef$对应输入参数self，输出为self与分组矩阵乘结果的累加。

> [!NOTE]
>
> 该接口提供原地版本`torch_npu.npu_grouped_matmul_add_`，参数与返回值一致，直接在输入self上累加并返回self本身。

## 函数原型

```python
torch_npu.npu_grouped_matmul_add(self, x, weight, group_list, *, transpose_x=True, transpose_weight=False,
                                 group_type=2, group_list_type=0) -> Tensor
```

## 参数说明

- **self**（`Tensor`）：必选参数，表示原地累加的输入矩阵，公式中的$yRef$。shape为[g, M, N]，其中g为分组个数。数据类型支持`torch.float32`，数据格式支持ND。
- **x**（`Tensor`）：必选参数，公式中的输入$x$。维度支持2维，数据格式支持ND，数据类型支持`torch.float16`、`torch.bfloat16`，需与weight的数据类型一致。支持非连续Tensor。x必须转置，即`transpose_x`仅支持True。
- **weight**（`Tensor`）：必选参数，表示权重，公式中的$weight$。维度支持2维，数据格式支持ND，数据类型支持`torch.float16`、`torch.bfloat16`，需与x的数据类型一致。支持非连续Tensor。weight不支持转置，即`transpose_weight`仅支持False。
- **group\_list**（`Tensor`）：必选参数，表示输入和输出分组轴方向的matmul大小分布。数据类型支持`torch.int64`，数据格式支持ND。第1维最大支持1024，即最多支持1024个group。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **transpose\_x**（`bool`）：可选参数，表示x矩阵是否转置，当前仅支持True。默认值为True。
- **transpose\_weight**（`bool`）：可选参数，表示weight矩阵是否转置，当前仅支持False。默认值为False。
- **group\_type**（`int`）：可选参数，表示分组类型。当前仅支持2（K轴分组），即$x_i$与$weight_i$的K轴（$k_i$）按组切分，每组$m_i/n_i$相同。默认值为2。
- **group\_list\_type**（`int`）：可选参数，表示group\_list中数值的解读方式。目前仅支持两个取值，默认值为0。
  - 0：group\_list中的数值为分组轴大小的cumsum结果（累积和），group\_list须为非负单调非递减数列，最后一个值不大于x中tensor的第一维。以K=256、E=4（各组大小依次为64、0、128、64）为例：`[64, 64, 192, 256]`。
  - 1：group\_list中的数值为分组轴上每组大小，group\_list须为非负数列，数值总和不大于x中tensor的第一维。以K=256、E=4为例：`[64, 0, 128, 64]`。

## 返回值说明

**output**(`Tensor`)：表示分组矩阵乘累加的结果，公式中的$yRef^{out}$。shape为[g, M, N]，与输入self的shape一致，数据类型为`torch.float32`。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 确定性计算：默认确定性实现。
- 支持的输入类型组合为：
  - x为`torch.float16`、weight为`torch.float16`、self为`torch.float32`。
  - x为`torch.bfloat16`、weight为`torch.bfloat16`、self为`torch.float32`。
- x与weight的数据类型必须一致。
- x和weight中每一组tensor的每一维大小在32字节对齐后都应小于`torch.int32`的最大值2147483647。
- x和weight中每一组tensor的最后一维大小都应小于65536。x的最后一维指转置后的M轴；weight的最后一维指不转置时的N轴。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    # x的shape为[512, 256]，转置后M=256、K=512；group_list将K轴切分为[0:256]、[256:512]两组
    x = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    weight = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    self_t = torch.zeros(2, 256, 256, dtype=torch.float32).npu()
    group_list = torch.tensor([256, 512], dtype=torch.int64).npu()
    out = torch_npu.npu_grouped_matmul_add(self_t, x, weight, group_list, transpose_x=True)
    print(out.shape, out.dtype)
    print(out[0, 0, :6])
    print(out[1, 0, :4])
    ```

    输出为

    ```text
    torch.Size([2, 256, 256]), torch.float32
    tensor([-14.8446,  -4.4171,   7.6527,  10.9340, -22.6788, -15.8647], device='npu:0')
    tensor([5.2137, 8.1306, 1.2067, 2.9381], device='npu:0')
    ```

- 单算子模式调用，非零self（累加语义）

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    x = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    weight = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    self_t = torch.full((2, 256, 256), 0.5, dtype=torch.float32).npu()
    group_list = torch.tensor([256, 512], dtype=torch.int64).npu()
    out = torch_npu.npu_grouped_matmul_add(self_t, x, weight, group_list, transpose_x=True)
    print(out[0, 0, :3]) 
    ```

    输出为

    ```text
    tensor([-14.3446,  -3.9171,   8.1527], device='npu:0')
    ```

- 单算子模式调用，原地版本

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    x = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    weight = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    self_t = torch.zeros(2, 256, 256, dtype=torch.float32).npu()
    group_list = torch.tensor([256, 512], dtype=torch.int64).npu()
    out = torch_npu.npu_grouped_matmul_add_(self_t, x, weight, group_list, transpose_x=True)
    print(out is self_t)  # True
    print(out[0, 0, :3]) # tensor([-14.8446,  -4.4171,   7.6527], device='npu:0')
    ```

    输出为

    ```text
    True
    tensor([-14.8446,  -4.4171,   7.6527], device='npu:0')
    ```
