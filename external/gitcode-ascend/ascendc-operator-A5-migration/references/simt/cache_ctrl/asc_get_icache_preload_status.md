> **原始文档路径**: asc-devkit/docs/api/context/c_api/cache_ctrl/asc_get_icache_preload_status.md

# asc_get_icache_preload_status

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

获取ICache的Preload的状态。

## 函数原型

```c++
__aicore__ inline int64_t asc_get_icache_preload_status()
```

## 参数说明

无

## 返回值说明

int64_t类型，0表示空闲，1表示忙。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
int64_t status = asc_get_icache_preload_status();
```