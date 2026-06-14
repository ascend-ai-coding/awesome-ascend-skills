> **原始文档路径**: asc-devkit/docs/api/context/c_api/cache_ctrl/asc_icache_preload.md

# asc_icache_preload

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |


## 功能说明

从指令所在DDR地址预加载数据到对应的cacheline中。

## 函数原型

```cpp
__aicore__ inline void asc_icache_preload(const void* addr, int64_t prefetch_len)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| addr | 输入| 预加载数据的地址。 |
| prefetch_len | 输入 | 预加载数据的长度。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
int64_t prefetch_length = 32;
int64_t pc = asc_get_program_counter() & 0xFFFFFFFFFFFF;
asc_icache_preload(reinterpret_cast<void *>(pc), prefetch_length);
```
