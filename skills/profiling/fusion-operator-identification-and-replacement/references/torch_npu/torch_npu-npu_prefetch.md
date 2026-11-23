# torch_npu.npu_prefetch

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

提供网络`weight`预取功能，用于在计算执行前将指定的权重数据预先加载到L2 Cache中，减少算子访问这些权重时的访存等待时间。例如，在MatMul等算子之前进行预取，算子执行时可直接从低时延的L2 Cache中读取权重，进而提升算子数据访问与计算效率。实际性能收益取决于用户采用的并行方式和配置。

## 函数原型

```python
torch_npu.npu_prefetch(input, dependency, max_size, offset=0) -> None
```

## 参数说明

- **input** (`Tensor`)：表示需要预取的权重，不做数据处理，与数据类型和数据格式无关；输入不能含有空指针。
- **dependency** (`Tensor`)：表示开始预取的节点，单算子下不生效可为None，图模式下不可为None；不做数据处理，与数据类型和数据格式无关。
- **max_size** (`int`)：取值需大于0，表示权重预取的最大size，超过预取权重的size时，会设置为权重的最大size。数据类型为`torch.int32`、`torch.int64`。
- **offset** (`int`)：默认值0，取值大于等于0，表示权重预取内存地址偏移，不允许超过权重地址范围。数据类型为`torch.int32`、`torch.int64`。

## 返回值说明

无

## 约束说明

该接口支持图模式。

## 调用示例

- 单算子多流并发调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> s_cmo = torch.npu.Stream()
    >>> x = torch.randn(10000, 10000, dtype=torch.float32).npu()
    >>> y = torch.randn(10000, 1, dtype=torch.float32).npu()
    tensor_size = 元素个数 × 每个元素的字节数
            = (10000 × 1) × 4    # float32 占 4 字节
            = 40000 字节
    >>> add = torch.add(x, 1)
    >>>
    >>> with torch.npu.stream(s_cmo):
    ...     torch_npu.npu_prefetch(y, None, 40000)
    ...     torch_npu.npu_prefetch(y, None, 20000, offset=20000)
    ...
    >>> abs = torch.abs(add)
    >>> mul = torch.matmul(abs, abs)
    >>> out = torch.matmul(mul, y)
    >>> print(out)
    [W428 20:51:15.589252880 ToKernelNpu.cpp:41] Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
    tensor([[-993366.4375],
            [-987970.6250],
            [-998181.7500],
            ...,
            [-992221.5625],
            [-980903.0000],
            [-999433.1875]], device='npu:0')
    >>> print(out.shape)
    torch.Size([10000, 1])
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)

    x = torch.randn(10000, 10000, dtype=torch.float32).npu()
    y = torch.randn(10000, 1, dtype=torch.float32).npu()


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            add = torch.add(x, 1)
            torch_npu.npu_prefetch(y, add, 40000)
            torch_npu.npu_prefetch(y, add, 20000, offset=20000)
            abs = torch.abs(add)
            mul = torch.matmul(abs, abs)
            out = torch.matmul(mul, y)
            return out


    npu_model = Model().npu()
    model = torch.compile(npu_model, backend=npu_backend, dynamic=False, fullgraph=True)
    output = model(x, y)
    print(output)
    print(output.shape)

    # 执行上述代码的输出类似如下    
    tensor([[-516592.8438],
            [-510890.9375],
            [-518402.5938],
            ...,
            [-518633.8750],
            [-523125.4062],
            [-509616.8750]], device='npu:0')
    torch.Size([10000, 1])
    ```
