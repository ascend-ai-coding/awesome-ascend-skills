# （beta）torch_npu.npu_ciou

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

基于NPU的CIoU操作，计算预测边界框与真实边界框之间的CIoU（Complete Intersection over Union）损失。

## 函数原型

```python
torch_npu.npu_ciou(boxes1, boxes2, trans=False, is_cross=True, mode=0, atan_sub_flag=False) -> Tensor
```

## 参数说明

- **boxes1** (`Tensor`)：必选参数，格式为xywh、shape为(4, n)的预测检测框。
- **boxes2** (`Tensor`)：必选参数，相应的gt检测框，shape为(4, n)。
- **trans** (`bool`)：可选参数，是否有偏移。默认值为False。
- **is_cross** (`bool`)：可选参数，boxes1和boxes2之间是否有交叉操作。默认值为True。
- **mode** (`int`)：可选参数，选择CIoU的计算方式。0表示IoU，1表示IoF。默认值为0。
- **atan_sub_flag** (`bool`)：可选参数，是否将正向的第二个值传递给反向。默认值为False。

## 返回值说明

`Tensor`

CIoU计算结果。

## 约束说明

<!-- npu="950" id6 -->
- Ascend 950DT系列产品：`boxes1`或`boxes2`的第二维度只支持1024的倍数，`is_cross`只支持False，且暂不支持反向计算。
<!-- end id6 -->
<!-- npu="A3,910b,910,310p" id7 -->
- Atlas A3训练系列产品、Atlas A2训练系列产品、Atlas推理系列产品、Atlas训练系列产品：到目前为止，CIoU反向计算只支持trans==True、is_cross==False、mode==0('iou')。如果需要反向传播，确保参数正确。
<!-- end id7 -->

## 调用示例

```python
>>> box1 = torch.randn(4, 32).npu()
>>> box1.requires_grad = True
>>> box2 = torch.randn(4, 32).npu()
>>> box2.requires_grad = True
>>> ciou = torch_npu.npu_ciou(box1, box2, trans=True, is_cross=False, mode=0)
>>> l = ciou.sum()
>>> l.backward()
```
