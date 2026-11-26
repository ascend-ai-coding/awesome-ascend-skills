# torch.npu.get_device_limit

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

## 功能说明

- 通过该接口，获取指定Device上的资源限制。
- 当前支持资源类型为Cube Core、Vector Core。

## 函数原型

```python
torch.npu.get_device_limit(device) ->Dict
```

## 参数说明

**device** (`Device`)：必选参数，指定要查询资源限制的设备ID。

## 返回值说明

`Dict`

代表`Device`的Cube和Vector核数。

## 约束说明

无

## 调用示例

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_device(0)
>>> torch.npu.set_device_limit(0,12,20)
>>> print(torch.npu.get_device_limit(0))
{"cube_core_num":12, "vector_core_num":20}
 ```
