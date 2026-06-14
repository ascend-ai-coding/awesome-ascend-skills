> **原始文档路径**: asc-devkit/docs/api/context/c_api/simd_atomic/asc_set_atomic_none.md

# asc_set_atomic_none

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Ascend 950PR/Ascend 950DT | √    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

清空原子操作的状态。一般和[asc_set_atomic_add](asc_set_atomic_add.md)，[asc_set_atomic_max](asc_set_atomic_max.md)，[asc_set_atomic_min](asc_set_atomic_min.md)接口配合使用，用于在完成原子操作后关闭原子操作，避免影响后续功能。

## 函数原型

```c++
__aicore__ inline void asc_set_atomic_none()
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

```c++
//total_length指参与搬运的数据总长度。dst是外部输入的int16_t类型的GM内存。
constexpr uint32_t total_length = 256;
__ubuf__ int16_t src0[total_length];
__ubuf__ int16_t src1[total_length];
asc_set_atomic_add_int16();
asc_copy_ub2gm_sync(dst, src0, total_length * sizeof(int16_t));
asc_copy_ub2gm_sync(dst, src1, total_length * sizeof(int16_t));
asc_set_atomic_none();
```

结果示例：

```
输入数据src0：[1, 1, 1, ..., 1]  // int16_t类型
输入数据src1：[2, 2, 2, ..., 2]  // int16_t类型
输出数据dst：[3, 3, 3, ..., 3]   // int16_t类型
```