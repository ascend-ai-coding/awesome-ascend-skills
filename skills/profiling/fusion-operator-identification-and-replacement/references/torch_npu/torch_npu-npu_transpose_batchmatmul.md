# torch_npu.npu_transpose_batchmatmul

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- **API功能**：完成张量`input`与张量`weight`的矩阵乘计算。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的数列进行变更。`perm_x1`代表张量`input`的转置序列，`perm_x2`代表张量`weight`的转置序列，序列值为0的是batch维度，其余两个维度做矩阵乘法。

- **计算公式**：

  - 非量化融合：

    $$
    Y = (input^{T_1} @ weight^{T_2} + bias)^{T_y}
    $$

  - 量化融合：

    $$
    Y = (input^{T_1} @ weight^{T_2} + bias)^{T_y} * scale
    $$

    T1、T2、Ty分别通过参数`perm_x1`、`perm_x2`、`perm_y`描述转置序列。

## 函数原型

```python
torch_npu.npu_transpose_batchmatmul(input, weight, *, bias=None, scale=None, perm_x1=[0,1,2], perm_x2=[0,1,2], perm_y=[1,0,2], batch_split_factor=1) -> Tensor
```

## 参数说明

- **input**(`Tensor`)：必选参数，表示矩阵乘的第一个矩阵。数据格式支持$ND$。shape维度支持3维(B, M, K)或者(M, B, K)。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。支持非连续的Tensor。

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：-1轴（末轴）<=65535，B取值范围为[1, 65536)。
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>：shape维度支持3维(B, M, K)或者(M, B, K)。
  <!-- end id5 -->

- **weight**(`Tensor`)：必选参数，表示矩阵乘的第二个矩阵。数据格式支持$ND$，`weight`的Reduce维度需要与`input`的Reduce维度大小相等。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。支持非连续的Tensor。

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：-1轴（末轴）<=65535。shape维度支持3维(B, K, N)，N的取值范围为[1, 65536)。
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：shape维度支持3维(B, K, N)或者(B, N, K)。
  <!-- end id7 -->

- \*：代表其之前的变量支持按位置输入，也可使用键值对赋值；之后的变量仅支持使用键值对赋值，其中带默认值的变量不赋值时使用默认值，不带默认值的变量必须赋值。
- **bias**(`Tensor`)：**可选参数**，表示矩阵乘的偏置矩阵，当前版本暂不支持该参数，使用默认值即可。
- **scale**(`Tensor`)：**可选参数**，表示量化输入。数据格式支持$ND$，数据类型支持`torch.int64`、`torch.uint64`，shape维度支持1维(B*N)。支持非连续的Tensor。

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：B*N的取值范围为[1, 65536)。
  <!-- end id8 -->

- **perm_x1**(`List[int]`)：**可选参数**，表示矩阵乘的第一个矩阵的转置序列，size为3，数据类型为`torch.int64`，数据格式支持$ND$，支持[0, 1, 2]、[1, 0, 2]。
- **perm_x2**(`List[int]`)：**可选参数**，表示矩阵乘的第二个矩阵的转置序列，size为3，数据类型为`torch.int64`，数据格式支持$ND$。

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：只支持[0, 1, 2]。
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持[0, 1, 2]、[0, 2, 1]。
  <!-- end id10 -->

- **perm_y**(`List[int]`)：**可选参数**，表示矩阵乘输出矩阵的转置序列，size为3，数据类型为`torch.int64`，数据格式支持$ND$，只支持[1, 0, 2]。
- **batch_split_factor**(`int`)：**可选参数**，用于指定矩阵乘输出矩阵中B维的切分大小。数据类型支持`torch.int32`。取值范围为[1, B]且能被B整除，默认值为1。注：当`scale`有值时，`batch_split_factor`只能为1。

## 返回值说明

- **y**(`Tensor`)：表示最终计算结果，数据格式支持$ND$，shape维度支持3维。

  - 当输入`scale`有值时，数据类型仅为`torch.int8`类型，shape为(M, 1, B*N)；否则数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
  - 当`batch_split_factor`>1时，shape大小计算公式为[batch_split_factor, M, B*N/batch_split_factor]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。

<!-- npu="A3,910b" id11 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  - 当`perm_x1`为[1, 0, 2]时，即`input`矩阵需要转置时，K*B的取值范围[1, 65536)；当`perm_x1`为[0, 1, 2]时，K需要小于65536。
  - K和N需要能被16整除。
<!-- end id11 -->

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    M, K, N, Batch = 32, 512, 128, 16
    x1 = torch.randn((M, Batch, K), dtype=torch.float16)
    x2 = torch.randn((Batch, K, N), dtype=torch.float16)
    batch_split_factor = 1
    output = torch_npu.npu_transpose_batchmatmul(x1.npu(), x2.npu(), bias=None, scale=None, perm_x1=(1,0,2), perm_x2=(0,1,2), perm_y=(1,0,2), batch_split_factor=batch_split_factor)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    M, K, N, Batch = 32, 512, 128, 16
    x1 = torch.randn((M, Batch, K), dtype=torch.float16)
    x2 = torch.randn((Batch, K, N), dtype=torch.float16)

    class MyModel1(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2, perm_x1, perm_y, batch_split_factor=1):
            output = torch_npu.npu_transpose_batchmatmul(x1, x2, perm_x1=perm_x1, perm_y=perm_y, batch_split_factor=batch_split_factor)
            output = output.add(1)
            return output

    model = MyModel1().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(x1.npu(), x2.npu(), (1, 0, 2), (1, 0, 2)).to("cpu")
    ```
