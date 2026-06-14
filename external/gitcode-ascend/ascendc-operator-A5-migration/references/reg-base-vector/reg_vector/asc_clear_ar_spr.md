> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_clear_ar_spr.md

# asc_clear_ar_spr

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

对AR寄存器进行清理，AR寄存器通常由[asc_squeeze](./asc_squeeze.md)接口使用。

## 函数原型

```cpp
__simd_callee__ inline void asc_clear_ar_spr()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
asc_clear_ar_spr();
```