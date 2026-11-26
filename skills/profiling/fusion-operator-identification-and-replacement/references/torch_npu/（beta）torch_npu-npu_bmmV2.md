# （beta）torch_npu.npu_bmmV2

> [!NOTICE]  
> 该接口计划废弃，其内部通过`torch.view`/`torch.Tensor.expand`对输入张量进行形状变换（1D扩展、batch维度广播），再调用`torch.bmm`等价操作（BatchMatMul），最后`torch.view`至目标形状。可使用`torch.bmm`和`Tensor.view()`接口进行替换。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

将矩阵“a”乘以矩阵“b”，生成“a*b”。支持FakeTensor模式。

## 函数原型

```python
torch_npu.npu_bmmV2(self, mat2, output_sizes) -> Tensor
```

## 参数说明

- **self**（`Tensor`）：2D或更高维度矩阵张量。数据类型支持`torch.float16`、`torch.float32`、`torch.int32`。格式支持$[ND, NHWC, FRACTAL\_NZ]$。
- **mat2**（`Tensor`）：2D或更高维度矩阵张量。数据类型支持`torch.float16`、`torch.float32`、`torch.int32`。格式支持$[ND, NHWC, FRACTAL\_NZ]$。
- **output_sizes**（`List[int]`）：默认值为[]，输出的shape，用于matmul的反向传播。

## 调用示例

示例一：

```python
>>> mat1 = torch.randn(10, 3, 4).npu()
>>> mat2 = torch.randn(10, 4, 5).npu()
>>> res = torch_npu.npu_bmmV2(mat1, mat2, [])
>>> print(res.shape)
torch.Size([10, 3, 5])
```

示例二：

```python
# FakeTensor模式
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     mat1 = torch.randn(10, 3, 4).npu()
...     mat2 = torch.randn(10, 4, 5).npu()
...     result = torch_npu.npu_bmmV2(mat1, mat2, [])
...
>>> print(result)
FakeTensor(..., device='npu:0', size=(10, 3, 5))
```
