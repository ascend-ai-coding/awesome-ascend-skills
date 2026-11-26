# （beta）torch_npu.npu_conv3d

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.nn.functional.conv3d`接口进行替换。

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

在由多个输入平面组成的输入图像上应用一个3D卷积。

## 函数原型

```python
torch_npu.npu_conv3d(input, weight, bias, stride, padding, dilation, groups) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：必选参数，输入张量，shape为(minibatch, in_channels, iT, iH, iW)。
- **weight**（`Tensor`）：必选参数，过滤器张量，shape为(out_channels, in_channels/groups, kT, kH, kW)。
- **bias**（`Tensor`）：可选参数，默认为None，偏置张量，shape为(out_channels)。
- **stride**（`List[int]`）：可选参数，默认为[1,1,1]，卷积核步长。
- **padding**（`List[int]`）：可选参数，默认为[1,1,1]，输入两侧的隐式填充。
- **dilation**（`List[int]`）：可选参数，默认为[1,1,1]，内核元素间距。
- **groups**（`int`）：可选参数，默认为1，对输入进行分组。in_channels可被组数整除。
