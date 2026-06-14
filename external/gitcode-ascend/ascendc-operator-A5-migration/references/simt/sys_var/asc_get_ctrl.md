> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_ctrl.md

# asc_get_ctrl

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

读取CTRL寄存器（控制寄存器）特定比特位上的值。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_ctrl()
```

## 参数说明

无

## 返回值说明

CTRL寄存器的值。具体含义请参考[asc_set_ctrl](asc_set_ctrl.md)中的描述。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无