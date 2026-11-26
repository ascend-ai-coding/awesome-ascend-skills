# torch_npu.npu_anti_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

- API功能：对张量`x`进行反量化操作，即将量化后的低精度数据恢复为浮点数。反量化是量化的逆过程，用于将低精度数据（如`torch.int8`、`torch_npu.int4`、`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`）转换回高精度浮点数。

- 计算公式：其中`out`是输出张量，`x`是量化后的输入，`scale`是缩放因子，`offset`是偏移量，`dst_dtype`指定输出的数据类型。
  
  - 当`offset`存在时：

    $out = cast\_to\_dst\_dtype((x + offset) * scale)$
  
  - 当`offset`不存在时：

    $out = cast\_to\_dst\_dtype(x * scale)$

## 函数原型

```python
torch_npu.npu_anti_quant(x, scale, *, offset=None, dst_dtype=None, src_dtype=None) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数，需要做反量化的输入，数据格式支持$ND$，支持非连续的Tensor，支持空Tensor。最大支持8维。

  <!-- npu="310p" id5 -->
  - <term>Atlas推理系列产品</term>：数据类型支持`torch.int8`。
  <!-- end id5 -->
  <!-- npu="910b" id6 -->
  - <term>Atlas A2系列产品</term>：数据类型支持`torch.int8`、`torch.int32`，其中`torch.int32`类型数据的每个值是由8个`torch_npu.int4`数值拼成。
  <!-- end id6 -->
  <!-- npu="A3" id7 -->
  - <term>Atlas A3系列产品</term>：数据类型支持`torch.int8`、`torch.int32`，其中`torch.int32`类型数据的每个值是由8个`torch_npu.int4`数值拼成。
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.int8`、`torch.int32`、`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`，其中`torch.int32`类型数据的每个值是由8个`torch_npu.int4`数值拼成。
  <!-- end id8 -->

- **scale** (`Tensor`)：必选参数，反量化中的`scale`值。`scale`的维数必须与`x`相同或是1维；如果`x`是一维，`scale`的形状必须是[1]或与`x`相同；如果`scale`是一维，其大小必须是1、`x`[-1]或`x`[-2]；如果`scale`是多维，最多只能有一个非1的维度，且这个非1的维度只能是-1或-2轴；当`x`为`torch.int32`类型且反量化轴为尾轴时，`scale`的尾轴必须等于`x`尾轴大小的8倍。数据格式支持$ND$，支持非连续的Tensor。

  <!-- npu="310p" id9 -->
  - <term>Atlas推理系列产品</term>：数据类型支持`torch.float32`。`scale`仅支持1维，且只能等于`x`尾轴大小或1。当`x`为`torch.int32`类型时，必须为`x`尾轴大小的8倍。
  <!-- end id9 -->
  <!-- npu="910b" id10 -->
  - <term>Atlas A2系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。`scale`仅支持1维，且只能等于`x`尾轴大小或1；当`x`为`torch.int32`类型时，必须为`x`尾轴大小的8倍。
  <!-- end id10 -->
  <!-- npu="A3" id11 -->
  - <term>Atlas A3系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。`scale`仅支持1维，且只能等于`x`尾轴大小或1；当`x`为`torch.int32`类型时，必须为`x`尾轴大小的8倍。
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。当`x`的数据类型为`torch_npu.hifloat8`、`torch.float8_e4m3fn`或`torch.float8_e5m2`时，`scale`仅支持`torch.float32`。
  <!-- end id12 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **offset** (`Tensor`)：可选参数，反量化中的`offset`值。数据类型和shape必须与`scale`一致。数据格式支持$ND$，支持非连续的Tensor。

- **dst_dtype** (`int`)：可选参数，指定输出的数据类型，默认值为`torch.float16`。

  <!-- npu="310p" id13 -->
  - <term>Atlas推理系列产品</term>：数据类型支持`torch.float16`。
  <!-- end id13 -->
  <!-- npu="910b" id14 -->
  - <term>Atlas A2系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
  <!-- end id14 -->
  <!-- npu="A3" id15 -->
  - <term>Atlas A3系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
  <!-- end id16 -->

- **src_dtype** (`int`)：可选参数，指定源输入的数据类型，默认值为`torch.int8`。当不传入时，根据`x`的数据类型自动推断。`src_dtype`必须与`x`的实际数据类型匹配。

  <!-- npu="310p" id17 -->
  - <term>Atlas推理系列产品</term>：数据类型支持`torch.int8`。
  <!-- end id17 -->
  <!-- npu="910b" id18 -->
  - <term>Atlas A2系列产品</term>：数据类型支持`torch.quint4x2`或`torch.int8`。
  <!-- end id18 -->
  <!-- npu="A3" id19 -->
  - <term>Atlas A3系列产品</term>：数据类型支持`torch.quint4x2`或`torch.int8`。
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.quint4x2`、`torch_npu.int4`、`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。
  <!-- end id20 -->

## 返回值说明

`Tensor`

代表`npu_anti_quant`的计算结果，对应公式中的$out$。支持非连续的Tensor，支持空Tensor。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- `x`、`scale`这两个输入均不能为None。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int8).npu()
    >>> scale = torch.tensor([2.0], dtype=torch.float).npu()
    >>> offset = torch.tensor([2.0], dtype=torch.float).npu()
    >>> out = torch_npu.npu_anti_quant(x_tensor, scale, offset=offset, dst_dtype=torch.float16)
    >>> print(out)
    tensor([ 6.,  8., 10., 12.], device='npu:0', dtype=torch.float16)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    
    config = CompilerConfig()
    config.debug.graph_dump.type = 'pbtxt'
    npu_backend = tng.get_npu_backend(compiler_config=config)
    x_tensor = torch.tensor([1,2,3,4], dtype=torch.int8).npu()
    scale = torch.tensor([2.0], dtype=torch.float).npu()
    offset = torch.tensor([2.0], dtype=torch.float).npu()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,x,scale,offset):
            return torch_npu.npu_anti_quant(x, scale, offset=offset, dst_dtype=torch.float16)

    cpu_model = Model()
    model = cpu_model.npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    output = model(x_tensor,scale,offset)
    print(output)

    # 执行上述代码的输出类似如下
    tensor([ 6.,  8., 10., 12.], device='npu:0', dtype=torch.float16)
    ```
    