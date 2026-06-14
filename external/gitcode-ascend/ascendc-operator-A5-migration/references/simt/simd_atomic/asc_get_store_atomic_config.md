> **原始文档路径**: asc-devkit/docs/api/context/c_api/simd_atomic/asc_get_store_atomic_config.md

# asc_get_store_atomic_config

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Ascend 950PR/Ascend 950DT | √    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

获取原子操作使能位与原子操作类型的值。

## 函数原型

```c++
__aicore__ inline void asc_get_store_atomic_config(asc_store_atomic_config& config)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输出 | 用于获取原子操作使能位和原子操作类型的值，详细说明请参考[asc_store_atomic_config](../struct/asc_store_atomic_config.md)。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 需配合[asc_set_store_atomic_config_v1](./asc_set_store_atomic_config_v1.md)和[asc_set_store_atomic_config_v2](./asc_set_store_atomic_config_v2.md)接口使用，该接口用于设置原子操作使能位和原子操作类型的值。

## 调用示例

```c++
asc_store_atomic_config set_config;
set_config.atomic_type = 1; // 使能原子操作,进行原子操作的数据类型为float，值为1
set_config.atomic_op = 0; // 求和操作,值为0
asc_set_store_atomic_config(set_config);
asc_store_atomic_config get_config;    // 用于获取原子操作使能位和原子操作类型的值
asc_get_store_atomic_config(get_config);    // get_config.atomic_type = 1; get_config.atomic_op = 0;
```
