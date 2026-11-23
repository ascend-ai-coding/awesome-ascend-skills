# torch\_npu.npu\_top\_k\_top\_p

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2推理系列产品</term>：支持
<!-- end id5 -->

## 功能说明

- API功能：对原始输入`logits`进行`top-k`和`top-p`采样过滤。

- 计算公式：
  - 对输入`logits`按最后一轴进行升序排序，得到对应的排序结果sortedValue和sortedIndices。
  $$sortedValue, sortedIndices = sort(logits, dim=-1, descend=false, stable=true)$$
  - 计算保留的阈值（第k大的值）。
  $$topKValue[b][v] = sortedValue[b][sortedValue.size(1) - k[b]]$$
  - 生成top-k需要过滤的mask。
  $$topKMask = sortedValue < topKValue$$
  - 通过topKMask将小于阈值的部分置为-inf。
  $$
  sortedValue[b][v] = 
  \begin{cases}
  -inf & \text{topKMask[b][v]=true}\\
  sortedValue[b][v] & \text{topKMask[b][v]=false}
  \end{cases}
  $$
  - 通过softmax将经过top-k过滤后的数据按最后一轴转换为概率分布。
  $$probsValue = softmax(sortedValue, dim=-1)$$
  - 按最后一轴计算累计概率（从最小的概率开始累加）。
  $$probsSum = cumsum(probsValue, dim=-1)$$
  - 生成top-p的mask，累计概率小于等于1-p的位置需要过滤掉，并保证每个batch至少保留一个元素。
  $$topPMask[b][v] = probsSum[b][v] <= 1-p[b]$$
  $$topPMask[b][-1] = false$$
  - 通过topPMask将小于阈值的部分置为-inf。
  $$
  sortedValue[b][v] = 
  \begin{cases}
  -inf & \text{topPMask[b][v]=true}\\
  sortedValue[b][v] & \text{topPMask[b][v]=false}
  \end{cases}
  $$
  - 将过滤后的结果按sortedIndices还原到原始顺序。
  $$out[b][v] = sortedValue[b][sortedIndices[b][v]]$$
  其中$0 \le b \lt logits.size(0), 0 \le v \lt logits.size(1)$。

## 函数原型

```python
torch_npu.npu_top_k_top_p(logits, p, k) -> torch.Tensor
```

## 参数说明

- **logits** (`Tensor`)：必选参数，表示要处理的数据，数据类型支持`torch.float32`、`torch.float16`和`torch.bfloat16`，数据格式支持$ND$，支持非连续的Tensor，维数支持2维。
- **p** (`Tensor`)：必选参数，表示`top-p`张量，值域为`[0, 1]`，数据类型支持`torch.float32`、`torch.float16`和`torch.bfloat16`，数据类型需要与`logits`一致，shape支持1维且需要与`logits`的第一维相同，数据格式支持$ND$，支持非连续的Tensor。
- **k** (`Tensor`)：必选参数，表示`top-k`的阈值张量，值域为`[1, 1024]`，且最大值需要小于等于logits.size(1)，数据类型支持`torch.int32`，shape支持1维且需要与`logits`的第一维相同，数据格式支持$ND$，支持非连续的Tensor。

## 返回值说明

`Tensor`

表示过滤后的数据。数据类型支持`torch.float32`、`torch.float16`和`torch.bfloat16`，数据类型与`logits`一致，shape支持2维且需要与`logits`一致，支持非连续Tensor，数据格式支持$ND$。

## 约束说明

在输入`logits`第二维大于1024场景下平均性能优于小算子实现，建议在`logits`第二维大于1024场景下使用该接口。

## 调用示例

单算子模式调用

  ```python
   >>> import torch
   >>> import torch_npu
   >>>
   >>> logits = torch.randn(16, 2048).npu()
   >>> p = torch.rand(16).npu()
   >>>
   >>> k = torch.randint(10, 1024, (16,)).npu().to(torch.int32)
   >>> out = torch_npu.npu_top_k_top_p(logits, p, k)
   >>>
   >>> print(out)
   tensor([[0.0000, 0.0000, 0.0000,  ...,   -inf,   -inf,   -inf],
        [0.0000, 0.0000, 0.0000,  ...,   -inf,   -inf,   -inf],
        [0.0000, 0.0000, 0.0000,  ...,   -inf,   -inf,   -inf],
        ...,
        [0.0000, 0.0000, 1.4379,  ...,   -inf,   -inf,   -inf],
        [0.0000, 0.0000, 0.0000,  ...,   -inf,   -inf,   -inf],
        [1.5425, 0.0000, 0.0000,  ...,   -inf, 1.5491,   -inf]],
       device='npu:0')
  ```
