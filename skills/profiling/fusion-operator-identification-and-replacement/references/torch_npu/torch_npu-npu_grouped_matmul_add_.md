# torch_npu.npu_grouped_matmul_add_

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：在micro-batch训练场景，需要做micro-batch的梯度累计，会存在大量GroupedMatmul操作接InplaceAdd操作的融合场景。本算子（GroupedMatmulAdd）在非量化场景中将上述算子融合起来，以提高网络性能。

- **计算公式**：

  $$
  self = self + x @ weight
  $$

## 函数原型

```python
torch_npu.npu_grouped_matmul_add_(self, x, weight, group_list, *, transpose_x=True, transpose_weight=False, group_type=2, group_list_type=0) -> torch.Tensor
```

## 参数说明

- **self**(`Tensor`)：必选参数，待累加矩阵。数据类型支持`torch.float32`，tensor支持3维，shape为(g, M, N)，数据格式支持$ND$。
- **x**(`Tensor`)：必选参数，表示矩阵乘法中的左矩阵。数据类型支持`torch.float16`、`torch.bfloat16`，tensor支持2维，shape为(K, M)，数据格式支持$ND$。
- **weight**(`Tensor`)：必选参数，表示矩阵乘法中的右矩阵。数据类型支持`torch.float16`、`torch.bfloat16`，tensor支持2维，shape为(K, N)，数据格式支持$ND$。
- **group_list**(`Tensor`)：必选参数，代表输入和输出分组轴方向的matmul大小分布。数据类型支持`torch.int64`，1维tensor，shape为(g, )，数据格式支持$ND$。
  - 当`group_list_type`为0时，`group_list`必须为非负单调非递减数列；最后一个值不大于`x`中tensor的第一维。
  - 当`group_list_type`为1时，`group_list`必须为非负数列。数值总和不大于`x`中tensor的第一维。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **transpose_x**(`bool`)：**可选参数**，代表`x`矩阵是否转置，True表示`x`转置（默认），False表示`x`非转置。当前仅支持True。
- **transpose_weight**(`bool`)：**可选参数**，代表`weight`矩阵是否转置，False表示`weight`不转置（当前仅支持），True表示`weight`转置。
- **group_type**(`int`)：**可选参数**，代表需要分组的轴。数据类型支持`torch.int64`，仅支持2（默认）。
- **group_list_type**(`int`)：**可选参数**，代表`group_list`的表达形式。数据类型支持`torch.int64`。
  - 0（默认）：`group_list`中数值为分组轴大小的cumsum结果（累积和）。
  - 1：`group_list`中数值为分组轴上每组大小。

## 返回值说明

`Tensor`

返回`self`张量本身，表示groupedMatmul计算完成后与待累加矩阵相加得到的最后结果矩阵，支持的数据类型、shape、数据格式均与输入`self`保持一致。

## 约束说明

- 该接口仅支持训练场景下使用。
- 该接口仅支持单算子模式。
- 参数说明里Shape使用的变量说明：
  - g：表示分组数目，取值范围为1-1024。
  - `x`和`weight`矩阵每一维大小在32字节对齐后都应小于`torch.int32`的最大值2147483647。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    M = 576
    K = 512
    N = 7168
    g = 4
    y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.float16).npu()
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.float16).npu()
    group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
    y = torch_npu.npu_grouped_matmul_add_(y, x1, x2, group_list, transpose_x=True, transpose_weight=False, group_type=2, group_list_type=0)
    print(y)
    ```
