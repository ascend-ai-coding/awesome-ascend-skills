> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_arch_ver.md


# asc_get_arch_ver

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √ </cann-filter>|

## 功能说明

获取当前AI处理器架构版本号。


## 函数原型

```cpp
__aicore__ inline void asc_get_arch_ver(uint32_t& core_version)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| core_version | 输出 | AI处理器架构版本，数据类型uint32_t。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

在调用asc_get_arch_ver接口前，需先定义core_version,调用asc_get_arch_ver接口后core_version会变成相对应架构版本号的值。
由于硬件约束，在查看转换后的AI处理器架构版本号时需要将其打印成十六进制的数或者自行转换成十六进制的数。

## 调用示例

```cpp
uint32_t core_version = 0;// 定义AI处理器版本
asc_get_arch_ver(core_version);
printf("core version is %x", core_version);// 需用%x将其打印成十六进制的数
```