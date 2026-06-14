> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_sub_block_num.md

# asc_get_sub_block_num

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

分离模式下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_sub_block_num()
```

## 参数说明

无

## 返回值说明


不同Kernel类型下，在AIC和AIV上调用该接口的返回值如下：

表1 返回值说明

|Kernel类型|KERNEL_TYPE_AIV_ONLY|KERNEL_TYPE_AIC_ONLY|KERNEL_TYPE_MIX__AIC_1_2|KERNEL_TYPE_MIX_AIC_1_1|KERNEL_TYPE_MIX_AIC_1_0|KERNEL_TYPE_MIX_AIV_1_0|
| :------ | :------------------ | :----------------- | :-------------------- | :--------------------- | :-------------------- | :-------------------- |
|AIV      |1                    |-                   |2                      |1                       |-                      |1                      |
|AIC      |-                    |1                   |1                      |1                       |1                      |-                      |

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无