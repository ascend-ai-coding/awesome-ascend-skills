# torch_npu.npu.set_deterministic_level

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id7 -->

## 功能说明

该接口用于控制CANN侧确定性计算等级。具体为重新配置CANN侧参数，参数详细说明可见：[aclSysParamOpt](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1010.html)。

实际level对应配置如下表所示：

| torch_npu.npu.set_deterministic_level配置 | 调用aclrtSetSysParamOpt配置 | 实际功能 | 与原生接口torch.use_deterministic_algorithms的对应关系 |
| :---------------------------------------- | :-------------------------------------------------------------------------------------------- | :--------- | :------------------------|
| 0                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 0)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | 关闭确定性 | torch.use_deterministic_algorithms(False) |
| 1                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | 仅开启确定性 | torch.use_deterministic_algorithms(True) |
| 2                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 1) | 开启强一致性 | 用于解决部分场景下，某些算子因为累加顺序不一致而导致结果不一致的问题。|
| 3                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 1) | 开启batch一致性 | 用于保证无论批处理大小如何变化，都能产生相同的输出结果 |

开启强一致性计算后，会导致算子执行速度变慢，影响性能。仅建议在需要严格保证不同位置上的相同数据计算结果一致，或进行精度调优时，才开启强一致性计算，以辅助模型调试和优化。

## 函数原型

```python
torch_npu.npu.set_deterministic_level(level)
```

## 参数说明

**level**(`int`)：所配置的确定性等级，目前可配置为0/1/2/3，默认值为0。

## 返回值说明

无

## 约束说明

- 当`level`取值为2时，该功能当前仅影响`torch.mm`计算结果。

- 该接口支持动态修改，后续新下发算子按新的确定性等级生效。

- 调用该接口时，会根据所配置的`torch.use_deterministic_algorithms`的bool值，推导出应该下发的确定性等级。

- 该接口不支持多线程配置。

- 8.5.0及以上的CANN版本支持配置`level`为0/1/2，仅9.2.0及以上的CANN版本支持配置`level`为3。

## 调用示例

```python
import torch
import torch_npu
torch_npu.npu.set_deterministic_level(2)
```
