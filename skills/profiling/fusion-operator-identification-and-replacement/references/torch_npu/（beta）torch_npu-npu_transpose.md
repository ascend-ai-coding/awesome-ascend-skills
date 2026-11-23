# （beta）torch_npu.npu_transpose

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议使用torch.permute。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

返回原始张量视图，其维度已permute，结果连续。支持FakeTensor模式。

## 函数原型

```python
torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：输入张量。
- **perm** (`List[int]`)：对应维度排列。
- **require_contiguous** (`bool`)：用户是否需要在调用函数前对输入Tensor做转连续，默认值为True。设置为False时，表示在调用函数前无需对输入Tensor做转连续。当用户明确输入Tensor为连续Tensor或转置Tensor时，才能设置为True。

## 调用示例

```python
>>> x = torch.randn(2, 3, 5).npu()
>>> print(x.shape)
torch.Size([2, 3, 5])
>>> x1 = torch_npu.npu_transpose(x, (2, 0, 1))
>>> print(x1.shape)
torch.Size([5, 2, 3])
>>> x2 = torch_npu.npu_transpose(x, (2, 0, 1))
>>> print(x2.shape)
torch.Size([5, 2, 3])
```
