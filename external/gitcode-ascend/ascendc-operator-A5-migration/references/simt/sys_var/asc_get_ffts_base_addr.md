> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_ffts_base_addr.md

# asc_get_ffts_base_addr

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |

## 功能说明

获取核间同步寄存器的基地址。需要在Host侧调用接口aclrtGetHardwareSyncAddr获取核间同步寄存器的基地址，并作为参数传入[asc_set_ffts_base_addr](asc_set_ffts_base_addr.md)后使用。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_ffts_base_addr()
```

## 参数说明

无

## 返回值说明

核间同步寄存器的基地址。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
// Host侧调用接口aclrtGetHardwareSyncAddr获取核间同步基地址ffts_addr
uint64_t config = *(__gm__ uint64_t*)ffts_addr;
asc_set_ffts_base_addr(config);
int64_t ffts_base = asc_get_ffts_base_addr();
```
