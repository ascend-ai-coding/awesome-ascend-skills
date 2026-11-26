# （beta）torch_npu.npu_rotated_iou

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

计算旋转框的IoU。

## 函数原型

```python
torch_npu.npu_rotated_iou(self, query_boxes, trans=False, mode=0, is_cross=True,v_threshold=0.0, e_threshold=0.0) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，第一组旋转框数据，shape为(B, 5, N)，数据类型为`torch.float32`的3D张量。
- **query_boxes** (`Tensor`)：必选参数，标注框，shape为(B, 5, K)，数据类型为`torch.float32`的3D张量，其中K值不大于1600。
- **trans** (`bool`)：可选参数，值为True表示“xyxyt”，值为False表示“xywht”。默认值为False。
- **is_cross** (`bool`)：可选参数，值为True时表示交叉计算，为False时表示一对一计算。默认值为True。
- **mode** (`int`)：可选参数，计算模式，取值为0或1。0表示IoU，1表示IoF。默认值为0。
- **v_threshold** (`float`)：可选参数，旋转框的高度阈值。默认值为0.0。
- **e_threshold** (`float`)：可选参数，旋转框的角度阈值。默认值为0.0。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> a=np.random.uniform(0,1,(2,2,5)).astype(np.float32)
>>> b=np.random.uniform(0,1,(2,3,5)).astype(np.float32)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(b).to("npu")
>>> output = torch_npu.npu_rotated_iou(box1, box2, trans=False, mode=0, is_cross=True)
```
