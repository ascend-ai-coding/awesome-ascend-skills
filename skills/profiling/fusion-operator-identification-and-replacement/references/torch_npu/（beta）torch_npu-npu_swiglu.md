# （beta）torch\_npu.npu\_swiglu

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：Swish门控线性单元激活函数，实现张量`input`的swiglu计算。

- 计算公式：

     公式中$x$是输入参数`input`的Tensor。$dim$是切分维度，默认为-1。$A$和$B$是`input`沿`dim`维度切分的Tensor。A表示前半部分张量，B表示后半部分张量。
    $$
    outputs=swiglu(x,dim=-1)=swish(A)*B=A*sigmoid(A)*B
    $$

## 函数原型

```python
torch_npu.npu_swiglu(Tensor input, int dim=-1) -> (Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，表示待计算的数据，对应公式中的$x$。shape支持1-8维，且shape必须在入参`dim`对应维度上可被2整除。不支持非连续的Tensor，不支持空Tensor。数据类型支持`torch.float32`、`torch.float16`或`torch.bfloat16`类型。

- **dim** (`int`)：可选参数，默认为-1。需要进行切分的维度序号，对`input`相应轴进行对半切。数据类型为`torch.int64`，取值范围为\[-input.dim\(\), input.dim\(\)-1\]。

## 返回值说明

`Tensor`

对应公式中的$outputs$。数据类型与计算输入`input`的类型一致，数据格式支持$ND$，不支持非连续的Tensor。

## 调用示例

- 单算子模式调用

  ```python
  import torch
  import torch_npu
  input_tensor = torch.randn(2, 32, 6, 6)
  output = torch_npu.npu_swiglu(input_tensor.npu(), dim = -1)
  ```

<!-- npu="950" id3 -->
- 图模式调用：仅适用于<term>Ascend 950PR&950DT系列产品</term>。

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, dim):
            return torch_npu.npu_swiglu(x, dim=dim)

    x = torch.rand((2, 32), dtype=torch.float32, device="npu")
    model = Net()
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    output = model(x, dim=-1)
    ```
<!-- end id3 -->
