# （beta）torch\_npu.npu\_nms\_with\_mask

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

生成值0或1，用于NMS算子确定有效位。

## 函数原型

```python
torch_npu.npu_nms_with_mask(input, iou_threshold) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **input** \(`Tensor`\)：输入张量。
- **iou\_threshold** \(`Scalar`\)：阈值。如果超过此阈值，则值为1，否则值为0。

## 返回值说明

- **selected\_boxes**\(`Tensor`\)：shape为\[N,5\]的2D张量，表示filtered box，包括proposal box和相应的置信度分数。
- **selected\_idx** \(`Tensor`\)：shape为\[N\]的1D张量，表示输入建议框的index。
- **selected\_mask** \(`Tensor`\)：shape为\[N\]的1D张量，判断输出建议框是否有效。

## 约束说明

输入input的2nd-dim必须等于8。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.6, 0.5, 0.4, 0.3], [6.0, 7.0, 8.0, 9.0, 0.4, 0.5, 0.6, 0.7]], dtype=torch.float16).to("npu")
>>> iou_threshold = 0.5
>>> output1, output2, output3, = torch_npu.npu_nms_with_mask(input, iou_threshold)
>>> print(output1)
tensor([[0.0000, 1.0000, 2.0000, 3.0000, 0.6001],
        [6.0000, 7.0000, 8.0000, 9.0000, 0.3999]], device='npu:0',      
        dtype=torch.float16)
>>> print(output2)
tensor([0, 1], device='npu:0', dtype=torch.int32)
>>> print(output3)
tensor([1, 1], device='npu:0', dtype=torch.uint8)
```
