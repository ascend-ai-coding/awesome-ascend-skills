# （beta）torch_npu.npu_ptiou

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

根据ground-truth和预测区域计算交并比（IoU）或前景交叉比（IoF）。

## 函数原型

```python
torch_npu.npu_ptiou(bboxes, gtboxes, mode=0) -> Tensor
```

## 参数说明

- **bboxes**（`Tensor`）：输入张量。
- **gtboxes**（`Tensor`）：输入张量。
- **mode**（`int`）：默认值为0。0为IoU模式，1为IoF模式。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> bboxes = torch.tensor([[0, 0, 10, 10],
                           [10, 10, 20, 20],
                           [32, 32, 38, 42]], dtype=torch.float16).to("npu")
>>> gtboxes = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20]], dtype=torch.float16).to("npu")
>>> output_iou = torch_npu.npu_ptiou(bboxes, gtboxes, 0)
>>> print(output_iou)
tensor([[0.4985, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000], 
       [0.0000, 0.9961, 0.0000]], device='npu:0', dtype=torch.float16)
```
