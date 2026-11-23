# （beta）torch_npu.npu_bounding_box_decode

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

根据rois和deltas生成标注框。自定义Faster R-CNN算子。

## 函数原型

```python
torch_npu.npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip) -> Tensor
```

## 参数说明

- **rois** (`Tensor`)：区域候选网络（RPN）生成的regions of interest（ROIs）。shape为(N,4)，数据类型为`torch.float32`或`torch.float16`的2D张量。“N”表示ROI的数量，“4”表示“x0”、“x1”、“y0”和“y1”。
- **deltas** (`Tensor`)：RPN生成的ROI和真值框之间的绝对变化。shape为(N,4)，数据类型为`torch.float32`或`torch.float16`的2D张量。“N”表示区域数，“4”表示“dx”、“dy”、“dw”和“dh”。
- **means0** (`float`)：“x0”的偏差值。默认值为0。
- **means1** (`float`)：“y0”的偏差值。默认值为0。
- **means2** (`float`)：“x1”的偏差值。默认值为0。
- **means3** (`float`)：“y1”的偏差值。默认值为0。
- **stds0** (`float`)：“x0”的缩放值。默认值为1.0。
- **stds1** (`float`)：“y0”的缩放值。默认值为1.0。
- **stds2** (`float`)：“x1”的缩放值。默认值为1.0。
- **stds3** (`float`)：“y1”的缩放值。默认值为1.0。
- **max_shape** (`List[int]`/`Tuple[int]` of length 2)：shape[h, w]，指定传输到网络的图像大小。用于确保转换后的bbox shape不超过“max_shape”。
- **wh_ratio_clip** (`float`)：“dw”和“dh”的值在(-wh_ratio_clip, wh_ratio_clip)范围内。

## 返回值说明

`Tensor`

表示解码后的边界框。

## 调用示例

```python
>>> rois = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
>>> deltas = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
>>> output = torch_npu.npu_bounding_box_decode(rois, deltas, 0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)
>>> print(output)
tensor([[2.5000, 6.5000, 9.0000, 9.0000],
        [9.0000, 9.0000, 9.0000, 9.0000]], device='npu:0')
```
