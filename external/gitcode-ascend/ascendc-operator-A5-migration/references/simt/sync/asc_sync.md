> **原始文档路径**: asc-devkit/docs/api/context/c_api/sync/asc_sync.md

# asc_sync

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| <cann-filter npu_type="950">Ascend 950PR/Ascend 950DT | √    </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

等待所有流水线操作完成。

## 函数原型

```cpp
__aicore__ inline void asc_sync()
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
// 本例中total_length指参与计算的数据总长度。src0_gm，src1_gm，dst_gm是外部输入的float类型的源操作数、目的操作数，指向GM内存空间。
constexpr uint32_t total_length = 128;
__ubuf__ float src0[total_length];
__ubuf__ float src1[total_length];
__ubuf__ float dst[total_length];

asc_copy_gm2ub((__ubuf__ void*)src0, (__gm__ void*)src0_gm, total_length * sizeof(float));
asc_copy_gm2ub((__ubuf__ void*)src1, (__gm__ void*)src1_gm, total_length * sizeof(float));

// 同步操作：前置操作完成后才能启动后续操作。
asc_sync();

asc_add(dst, src1, src0, total_length);

// 同步操作：前置操作完成后才能启动后续操作。
asc_sync();

asc_copy_ub2gm((__gm__ void*)dst_gm, (__ubuf__ void*)dst, blockLength * sizeof(float));
```
