> **原始文档路径**: asc-devkit/docs/api/context/c_api/cache_ctrl/asc_dci.md

# asc_dci

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

数据缓存失效，该指令用于使整个数据缓存无效化，且不会对处于“脏”状态的缓存行执行写回操作。换言之，所有缓存行将被直接标记为无效，其修改过但尚未同步至主存的数据将被丢弃。

## 函数原型

```cpp
__aicore__ inline void asc_dci()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

在调用asc_dci之前，需调用asc_sync_data_barrier插入DSB_ALL指令, 等待所有内存访问指令执行结束。

## 调用示例

```cpp
asc_sync_data_barrier(mem_dsb_t::DSB_ALL);
asc_dci();
```
