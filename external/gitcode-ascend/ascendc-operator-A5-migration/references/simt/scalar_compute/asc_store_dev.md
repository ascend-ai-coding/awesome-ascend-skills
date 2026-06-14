> **原始文档路径**: asc-devkit/docs/api/context/c_api/scalar_compute/asc_store_dev.md

# asc_store_dev

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

不经过DCache向GM地址上写数据。‌
当多核操作GM地址时，如果数据无法对齐到Cache Line，经过DCache的方式下，由于按照Cache Line大小进行读写，会导致多核数据随机覆盖的问题。此时，可以采用不经过DCache直接读写GM地址的方式，从而避免上述随机覆盖的问题。

## 函数原型

```cpp
__aicore__ inline void asc_store_dev(__gm__ int8_t* addr, int8_t value)

__aicore__ inline void asc_store_dev(__gm__ uint8_t* addr, uint8_t value)

__aicore__ inline void asc_store_dev(__gm__ int16_t* addr, int16_t value)

__aicore__ inline void asc_store_dev(__gm__ uint16_t* addr, uint16_t value)

__aicore__ inline void asc_store_dev(__gm__ int32_t* addr, int32_t value)

__aicore__ inline void asc_store_dev(__gm__ uint32_t* addr, uint32_t value)

__aicore__ inline void asc_store_dev(__gm__ int64_t* addr, int64_t value)

__aicore__ inline void asc_store_dev(__gm__ uint64_t* addr, uint64_t value)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| addr     | 输入     | 目标GM地址。|
| value     | 输入     | 待写入目标的数据。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
__gm__ int32_t* addr;
int32_t value = 2;
asc_store_dev(addr, value);
```