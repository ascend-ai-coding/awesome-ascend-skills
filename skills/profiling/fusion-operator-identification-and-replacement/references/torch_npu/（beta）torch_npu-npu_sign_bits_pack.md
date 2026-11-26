# （beta）torch_npu.npu_sign_bits_pack

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

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

- API功能：将`float`类型的输入打包为`torch.uint8`类型。每8个浮点数打包为一个`torch.uint8`数值，-1.0编码为二进制位0，1.0编码为二进制位1，并按小端序进行打包。

- 等价计算逻辑：
    
     可使用`sign_pack`等价替换`torch_npu.npu_sign_bits_pack`，两者计算逻辑一致。
     
     ```python
    import torch
    import numpy as np
    
    def sign_pack(in_data, size):
        sign_data = np.sign(in_data)
        sign_data = sign_data + 1
        bool_data = np.bool_(sign_data)
        pack_bit = np.packbits(bool_data, bitorder="little")
        return pack_bit.reshape(size, pack_bit.shape[0] // size)
    ```

## 函数原型

```python
torch_npu.npu_sign_bits_pack(self, size) -> Tensor
```

## 参数说明

- **self** (`Tensor`): 必选参数，1D `torch.float`张量。 支持`torch.float32`和`torch.float16`类型输入。
- **size**(`int`): 必选参数，用于reshape输出张量的第一个维度。

## 返回值说明

`Tensor`

打包后的张量。

## 约束说明

`size`可被`float`打包的输出整除。如果`self`的size可被8整除，则输出的size为(size of self)/8；否则，输出的size为(size of self // 8) + 1。将在小端位置添加-1浮点值以填充可整除性。

## 调用示例

```python
>>> import torch, torch_npu
>>> a = torch.tensor([5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2],dtype=torch.float32).npu()
>>> b = torch_npu.npu_sign_bits_pack(a, 2)
>>> print(b)
tensor([[159],
        [ 15]], device='npu:0', dtype=torch.uint8)
>>> c = torch_npu.npu_sign_bits_pack(a, 1)
>>> print(c)
tensor([[159, 15]], device='npu:0', dtype=torch.uint8)
```
