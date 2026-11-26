# torch_npu.npu_gather_sparse_index

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明 

- API功能：从输入tensor的指定维度，按照`index`中的下标序号提取元素，保存到输出tensor中。

- 计算公式：假设$x$是输入`input`，$idx$是输入`index`。
  $$
  y_{i,j} = x_{idx[i],j}
  $$

- 示例：

  输入$x$如下：
  $$
  \begin{bmatrix}
    1& 2 & 3\\
    4& 5 & 6\\
    7& 8 &9
  \end{bmatrix}
  $$
  索引$idx$如下：
  $$
  \begin{bmatrix}
  1 & 0
  \end{bmatrix}
  $$
  此时输入$x$的shape为$[3, 3]$，索引$idx$的shape为$[2]$(一维向量)，以上索引的含义为分别提取$x$中的第1行和第0行，则输出shape为$[2, 3]$，输出结果如下：
  $$
  \begin{bmatrix}
  4 & 5 & 6\\
  1 & 2 & 3
  \end{bmatrix}
  $$

## 函数原型

```python
torch_npu.npu_gather_sparse_index(input, index) -> Tensor
```

## 参数说明

**input**(`Tensor`)：必选参数，输入张量，数据格式支持ND。数据类型支持`torch.float32`, `torch.float16`, `torch.bfloat16`, `torch.int64`, `torch.int32`, `torch.int16`, `torch.int8`, `torch.uint8`, `torch.bool`, `torch.float64`, `torch.complex64`, `torch.complex128`。

**index**(`Tensor`)：必选参数，代表目标元素下标序号的张量。数据维度不超过7维。数据类型支持`torch.int64`, `torch.int32`。取值范围$[0, input.shape[0] - 1]$，不支持负数索引。

## 返回值说明

`Tensor`
接口计算获得的结果，包含按照`index`中的下标序号提取的元素。数据类型与`input`一致，输出张量的维度数为$index.dim + input.dim - 1$。例如`input.shape = [16, 32]`, `index.shape = [2, 3]`，则输出张量 `out.shape = [2, 3, 32]`。

## 约束说明

- `input`的维度与`index`的维度之和减1不能超过8，即$index.dim + input.dim - 1<=8$。
- 为获取性能收益，`input`和`index`需要满足如下约束：
     1. `input`的shape各维度乘积需要大于$150 * 1024 / itemsize$，其中itemsize为`input` dtype对应元素大小，可以通过`torch.dtype.itemsize`查询。
     2. `index`的shape各维度乘积大于960。
     3. 数据需要聚合，即`input`中0值元素和非0值元素分别集中在连续区域内，而不是频繁交错分布。可将 `input != 0`按行优先顺序展开为一维布尔序列$mask$，统计相邻元素状态切换比例：
     $$
     \mathrm{switch\_ratio} = \frac{\sum_{i=1}^{N-1} \mathbf{1}(\mathrm{mask}_i \ne \mathrm{mask}_{i-1})}{N - 1}
     $$
     其中，$N$为`input`元素总数，$\mathrm{mask}_i$表示布尔序列$\mathrm{mask}$中第$i$个元素。$\mathrm{switch\_ratio}$越小，说明0值和非0值连续分布越明显，数据聚合程度越高；$\mathrm{switch\_ratio}$越大，说明0值和非0值越分散，可能无法获得预期性能收益。

## 调用示例

```python
import torch
import torch_npu

inputs = torch.randn(16, 32).npu()
index = torch.randint(0, 16, [2, 3]).npu()
out = torch_npu.npu_gather_sparse_index(inputs, index)
```
