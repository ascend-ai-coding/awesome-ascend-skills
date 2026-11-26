# torch_npu.print_npugraph_tensor

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

aclgraph模式下，由于原生Python的print函数包含同步处理，在`torch.compile`时会触发断图，并且无法被aclgraph捕获，导致图模式下无法使用print接口观察aclgraph模式执行过程中的tensor数据。

当前接口提供了类似原生Python的print接口特性且不影响aclgraph捕获、重放的tensor打印能力，允许将aclgraph中间节点的tensor数据、数据类型、shape信息直接打印出来，以便用户观察aclgraph的执行过程中的tensor数据，以快速定位问题。

## 函数原型

```python
torch_npu.print_npugraph_tensor(input, tensor_name=None) -> None
```

## 参数说明

- **input** (`Tensor`)：必选参数，用于打印的tensor。
- **tensor_name** (`str`)：可选参数，指定打印tensor的tensor name，用于区分不同的tensor，默认为None。
  - tensor_name为None时：直接输出tensor数据内容。
  - tensor_name不为None时：以`{tensor_name}:`格式作为前缀，后接tensor数据。

## 返回值说明

`None`

无返回值。

## 约束说明

该接口支持在Eager模式和aclgraph模式下使用。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>> a = torch.randn([5, 5]).npu()
    >>> torch_npu.print_npugraph_tensor(a, tensor_name = "a")
    ```

- 基于`torch.npu.graph`调用

    ```python
    import torch
    import torch_npu

    x = torch.randn([5, 5]).npu()

    graph1 = torch.npu.NPUGraph()
    with torch.npu.graph(graph1):
        torch_npu.print_npugraph_tensor(x, tensor_name = "x")
        output = torch.square(x)
        torch_npu.print_npugraph_tensor(output, tensor_name = "output")

    graph1.replay()
    ```

- 基于`torch.compile`调用

    ```python
    import torch
    import torch_npu

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            x = torch.add(x, x)
            torch_npu.print_npugraph_tensor(x, tensor_name ="x")
            x = torch.add(x, 2)
            torch_npu.print_npugraph_tensor(x, tensor_name="added_x")
            return x

    x = torch.randn([5, 5]).npu()
    model = Model()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    model (x)
    ```
