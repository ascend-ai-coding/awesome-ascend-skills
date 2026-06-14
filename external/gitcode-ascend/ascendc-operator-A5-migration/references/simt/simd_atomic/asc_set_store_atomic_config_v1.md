> **原始文档路径**: asc-devkit/docs/api/context/c_api/simd_atomic/asc_set_store_atomic_config_v1.md

# asc_set_store_atomic_config_v1

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

设置原子操作使能位与原子操作类型的值。

## 函数原型

```cpp
__aicore__ inline void asc_set_store_atomic_config_v1(uint16_t type, uint16_t op)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| type | 输入 | 原子操作使能位。<br>0：无原子操作。<br>1：使能原子操作，进行原子操作的数据类型为float。<br>2：使能原子操作，进行原子操作的数据类型为half。<br>3：使能原子操作，进行原子操作的数据类型为int16_t。<br>4：使能原子操作，进行原子操作的数据类型为int32_t。<br>5：使能原子操作，进行原子操作的数据类型为int8_t。<br>6：使能原子操作，进行原子操作的数据类型为bfloat16_t。<br>其余值无具体含义。 |
| op | 输入 | 原子操作类型。<br>0：求和操作。<br>其余值无具体含义。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint16_t type = 1;
uint16_t op = 0;
asc_set_store_atomic_config_v1(type, op);
```