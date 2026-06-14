> **原始文档路径**: asc-devkit/docs/api/context/c_api/sync/asc_sync_vec.md

# asc_sync_vec

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| <cann-filter npu_type="950">Ascend 950PR/Ascend 950DT | √    </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

针对所有流水线执行同步操作。

## 函数原型

```cpp
__aicore__ inline void asc_sync_vec()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

```cpp
asc_sync_vec();
```
