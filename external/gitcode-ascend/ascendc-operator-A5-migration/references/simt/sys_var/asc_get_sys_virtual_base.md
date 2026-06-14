> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_sys_virtual_base.md

# asc_get_sys_virtual_base

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

获取系统虚拟基地址。

## 函数原型

```cpp
 __aicore__ inline int64_t asc_get_sys_virtual_base()
```

## 参数说明

无

## 返回值说明

系统虚拟基地址。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
int64_t sys_virtual_base = asc_get_sys_virtual_base();
printf("sys virtual base is %x", sys_virtual_base);// 需用%x将其打印成十六进制的数
```