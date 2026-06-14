> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_vf_len.md

# asc_get_vf_len

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |

## 功能说明

获取Tensor位宽VL（Vector Length）的大小。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_vf_len()
```

## 参数说明

无

## 返回值说明

位宽VL（Vector Length）的大小。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
int64_t len = asc_get_vf_len();
```
