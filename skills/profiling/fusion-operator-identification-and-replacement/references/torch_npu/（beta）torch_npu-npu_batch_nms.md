# （beta）torch\_npu.npu\_batch\_nms

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="310p" id2 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id2 -->

## 功能说明

以批量处理方式对每个类别的检测框进行非极大值抑制（Non-Maximum Suppression，NMS），从而去除冗余检测框，输出保留下来的检测框及其对应的类别和得分。

## 函数原型

```python
torch_npu.npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame=False, transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **self** \(`Tensor`)：必选参数，输入框的tensor，包含batch大小，数据类型为`torch.float16`，输入示例：\[batch\_size, num\_anchors, q, 4\]，其中q=1或q=num\_classes。
- **scores** \(`Tensor`)：必选参数，输入tensor，数据类型为`torch.float16`，输入示例：\[batch\_size, num\_anchors, num\_classes\]。
- **score\_threshold** \(`float`)：必选参数，指定评分过滤器的阈值，用于筛选框，去除得分较低的框，数据类型为`torch.float32`。
- **iou\_threshold** \(`float`)：必选参数，指定nms的iou\_threshold，用于设定阈值，去除高于阈值的框，数据类型为`torch.float32`。
- **max\_size\_per\_class** \(`int`)：必选参数，指定每个类别的最大可选的框数。
- **max\_total\_size** \(`int`)：必选参数，指定每个batch最大可选的框数。
- **change\_coordinate\_frame** \(`bool`)：可选参数，是否归一化输出框坐标矩阵，默认值为False。
- **transpose\_box** \(`bool`)：可选参数，确定是否在此op之前插入转置，默认值为False。True表示boxes使用(4, N)排布。False表示boxes使用(N, 4)排布。

## 返回值说明

- **nmsed_boxes** (`Tensor`)：shape为\(batch, max\_total\_size, 4\)的3D张量，指定每批次输出的nms框，数据类型`torch.float16`。
- **nmsed_scores** (`Tensor`)：shape为\(batch, max\_total\_size\)的2D张量，指定每批次输出的nms分数，数据类型`torch.float16`，batch和参数`nmsed_boxes`的 batch保持一致。
- **nmsed_classes**(`Tensor`)：shape为\(batch, max\_total\_size\)的2D张量，指定每批次输出的nms类，数据类型`torch.float16`。
- **nmsed_num**(`Tensor`)：shape为\(batch\)的1D张量，指定nmsed\_boxes的有效数量，数据类型为`torch.int32`。

## 调用示例

```python
>>> import torch, torch_npu
>>> boxes = torch.randn(8, 2, 4, 4, dtype = torch.float16).to("npu")
>>> scores = torch.randn(8, 2, 4, dtype = torch.float16).to("npu")
>>> nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 3, 4)
```
