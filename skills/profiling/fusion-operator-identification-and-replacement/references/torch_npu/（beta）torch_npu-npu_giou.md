# （beta）torch_npu.npu_giou

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

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

首先计算两个框的最小封闭面积和IoU，然后计算封闭区域中不属于两个框的封闭面积的比例，最后从IoU中减去这个比例，得到GIoU。

## 函数原型

```python
torch_npu.npu_giou(self, gtboxes, trans=False, is_cross=False, mode=0) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，标注框，shape为(N, 4)，数据类型为`torch.float16`或`torch.float32`的2D张量。“N”表示标注框的数量，值“4”表示[x1, y1, x2, y2]或[x, y, w, h]。
- **gtboxes** (`Tensor`)：必选参数，真值框，shape为(M, 4)，数据类型为`torch.float16`或`torch.float32`的2D张量。“M”表示真值框的数量，值“4”表示[x1, y1, x2, y2]或[x, y, w, h]。
- **trans** (`bool`)：可选参数，值为True代表“xywh”，值为False代表“xyxy”，默认值为False。
- **is_cross** (`bool`)：可选参数，控制输出shape是[M, N]还是[1,N]。如果值为True，则输出shape为[M,N]。如果为False，则输出shape为[1,N]。默认值为False。
- **mode** (`int`)：可选参数，计算模式，取值为0或1。0表示IoU，1表示IoF。默认值为0。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> a=np.random.uniform(0,1,(10,4)).astype(np.float16)
>>> b=np.random.uniform(0,1,(10,4)).astype(np.float16)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(b).to("npu")
>>> output = torch_npu.npu_giou(box1, box2, trans=True, is_cross=False, mode=0)
```
