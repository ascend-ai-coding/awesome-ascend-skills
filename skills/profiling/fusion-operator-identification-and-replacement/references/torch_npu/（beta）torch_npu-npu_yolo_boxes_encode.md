# （beta）torch_npu.npu_yolo_boxes_encode

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

根据YOLO的锚点框（anchor box）和真值框（ground-truth box）生成标注框。自定义mmdetection算子。

## 函数原型

```python
torch_npu.npu_yolo_boxes_encode(self, gt_bboxes, stride, performance_mode=False) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，YOLO训练集生成的锚点框。shape为(N, 4)，数据类型为`torch.float32`或`torch.float16`的2D张量。“N”表示ROI的数量，值“4”表示(tx, ty, tw, th)。
- **gt_bboxes** (`Tensor`)：必选参数，转换目标，例如真值框。shape为(N, 4)，数据类型为`torch.float32`或`torch.float16`的2D张量。“N”表示ROI的数量，值“4”表示(dx, dy, dw, dh)。
- **stride** (`Tensor`)：必选参数，各框比例。shape为(N,)，数据类型为`torch.int32`的1D张量。“N”表示ROI的数量。
- **performance_mode** (`bool`)：可选参数，选择性能模式为“high_precision”或“high_performance”。如果值为True，则性能模式为“high_performance”；如果值为False，则性能模式为“high_precision”，默认值为False。当输入数据类型为`torch.float32`时，选择“high_precision”，输出张量误差将小于0.0001。当输入数据类型为`torch.float16`时，选择“high_performance”，ops将是最佳性能，但误差将只小于0.005。

## 约束说明

输入锚点框支持的最大N为20480。

## 调用示例

```python
>>> anchor_boxes = torch.rand(2, 4).npu()
>>> gt_bboxes = torch.rand(2, 4).npu()
>>> stride = torch.tensor([2, 2], dtype=torch.int32).npu()
>>> output = torch_npu.npu_yolo_boxes_encode(anchor_boxes, gt_bboxes, stride, False)
>>> print(output.shape)
torch.Size([2, 4])
```
