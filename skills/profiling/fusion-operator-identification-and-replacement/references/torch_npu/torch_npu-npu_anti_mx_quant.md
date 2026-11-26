# torch\_npu.npu\_anti\_mx\_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：将调用`npu_dynamic_mx_quant`量化得到的`torch_npu.float4`/`torch.float8`的Tensor反量化为`torch.float16`/`torch.bfloat16`/`torch.float32`格式，是`npu_dynamic_mx_quant`的逆过程。

- 计算公式：
  $$
  X_{dq} = X_q \times 2^{sf - bias}
  $$

  其中$sf$是缩放因子，由输入`mxscale`提供；$bias$是指数位的偏移，对于`torch_npu.float8_e8m0`格式，$bias=127$；$X_q$是量化得到的`torch_npu.float4`/`torch.float8`张量；$X_{dq}$是反量化得到的`torch.float16`/`torch.bfloat16`/`torch.float32`张量。

## 函数原型

```python
torch_npu.npu_anti_mx_quant(x, mxscale, *, axis=-1, dst_type=15, src_type=292) -> Tensor
```

## 参数说明

- **x**（`Tensor`）：必选参数，待反量化的输入Tensor。数据类型支持`torch_npu.float4_e2m1`、`torch_npu.float4_e1m2`、`torch.float8_e5m2`、`torch.float8_e4m3fn`，数据格式支持ND。当前shape支持1-7维。`torch.float8`输入时，`x`的数据类型为对应的`torch.float8`类型，支持非连续的Tensor；`torch_npu.float4`输入时，`x`的实际数据类型为`torch.uint8`（每两个`float4`数据打包为一个`torch.uint8`，shape的最后一维为真实元素个数的一半），不支持非连续的Tensor。
- **mxscale**（`Tensor`）：必选参数，参与反量化计算的量化尺度，由`npu_dynamic_mx_quant`输出得到。数据类型为`torch_npu.float8_e8m0`（实际存储数据类型为`torch.uint8`），数据格式支持ND。当前shape支持2-8维。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **axis**（`int`）：可选参数，指定反量化轴。当前仅支持尾轴，即取值为`-1`或`D-1`，其中`D`为输入`x`的维度数，默认值为`-1`。
- **dst\_type**（`int`）：可选参数，指定输出结果的数据类型。当前支持取值为`{5, 6, 15}`，分别对应输出的数据类型为`{5: torch.float16, 6: torch.float32, 15: torch.bfloat16}`，默认值为`15`（`torch.bfloat16`）。也支持直接传入`torch.float16`、`torch.float32`、`torch.bfloat16`。
- **src\_type**（`int`）：可选参数，指定输入`x`的数据类型。当前支持取值为`{291, 292, 296, 297}`，分别对应输入`x`的数据类型为`{291: torch.float8_e5m2, 292: torch.float8_e4m3fn, 296: torch_npu.float4_e2m1, 297: torch_npu.float4_e1m2}`，默认值为`292`（`torch.float8_e4m3fn`）。

## 返回值说明

`Tensor`

反量化结果。数据类型由`dst_type`指定，支持`torch.float16`/`torch.bfloat16`/`torch.float32`，数据格式支持ND。`torch.float8`输入时，结果的shape与输入`x`保持一致；`torch_npu.float4`输入时，结果的shape最后一维为输入`x`（`torch.uint8`存储）最后一维的2倍，其余维度与输入`x`保持一致。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输入`x`和`mxscale`需配套使用，建议`mxscale`通过`torch_npu.npu_dynamic_mx_quant`（`block_size`为32）量化获得。
- 输入`x`和`mxscale`的shape约束关系（`block_size`为32时）：
  - rank(mxscale) = rank(x) + 1
  - axis_change = axis if axis >= 0 else axis + rank(x)（当前仅支持尾轴）
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / block_size) + 2 - 1) / 2
  - mxscale.shape[-1] = 2
  - 其他维度与输入x一致

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    def anti_mx_quant_test(x_dtype, dst_type, src_type):
        # method 1: generate inputs manually
        x = torch.randn((16, 128), dtype=torch.float32).to(dtype=torch.float8_e5m2).npu()
        mxscale = torch.randint(120, 140, (16, 2, 2), dtype=torch.float32).to(dtype=torch.float8_e8m0fnu).npu()
        y = torch_npu.npu_anti_mx_quant(x, mxscale, axis=-1, dst_type=dst_type, src_type=src_type)
        print("AntiMxQuant result:")
        print("x:\n", x.cpu())
        print("mxscale:\n", mxscale.cpu())
        print("y:\n", y.cpu())

        # method 2: generate inputs by npu_dynamic_mx_quant
        x_tmp = torch.randn((16, 128), dtype=x_dtype).npu()
        x, mxscale = torch_npu.npu_dynamic_mx_quant(x_tmp, axis=-1, round_mode="rint", dst_type=torch.float8_e5m2, block_size=32, scale_alg=0)
        y = torch_npu.npu_anti_mx_quant(x, mxscale, axis=-1, dst_type=dst_type, src_type=src_type)
        print("AntiMxQuant result:")
        print("x:\n", x.cpu())
        print("mxscale:\n", mxscale.cpu())
        print("y:\n", y.cpu())

    if __name__ == "__main__":
        anti_mx_quant_test(torch.bfloat16, 15, 291)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class AntiMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, mxscale, axis=-1, dst_type=15, src_type=291):
            return torch_npu.npu_anti_mx_quant(x, mxscale, axis=axis, dst_type=dst_type, src_type=src_type)

    def anti_mx_quant_test(x_dtype, dst_type):
        # 构造x tensor
        x_tmp = torch.randn((16, 128), dtype=x_dtype).npu()
        x, mxscale = torch_npu.npu_dynamic_mx_quant(x_tmp, axis=-1, round_mode="rint", dst_type=torch.float8_e5m2, block_size=32, scale_alg=0)
        # 图模式下需将mxscale由torch.uint8存储还原为torch_npu.float8_e8m0fnu类型
        mxscale = mxscale.view(torch.float8_e8m0fnu)
        model = AntiMxQuantModel()
        model.to('npu')
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        y = model(x, mxscale, axis=-1, dst_type=dst_type, src_type=291)
        y = y.cpu()
        print("AntiMxQuant result:")
        print("x:\n", x.cpu())
        print("y:\n", y)

    if __name__ == "__main__":
        anti_mx_quant_test(torch.bfloat16, 15)
    ```
