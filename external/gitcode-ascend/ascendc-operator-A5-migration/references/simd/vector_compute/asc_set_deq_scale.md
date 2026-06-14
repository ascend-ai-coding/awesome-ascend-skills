> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_set_deq_scale.md
# asc_set_deq_scale

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

设置DEQSCALE（量化参数）寄存器的值。

## 函数原型

```cpp
__aicore__ inline void asc_set_deq_scale(__ubuf__ uint64_t* tmp, float scale_arr[ASC_VDEQ_SIZE], int16_t offset_arr[ASC_VDEQ_SIZE], bool sign_mode_arr[ASC_VDEQ_SIZE])
__aicore__ inline void asc_set_deq_scale(float scale, int16_t offset, bool sign_mode)
__aicore__ inline void asc_set_deq_scale(half scale)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|tmp|输入|一块临时内存区域，用于存储一组大小为16的量化参数。|
|scale_arr|输出|float数组类型，大小为16，用于存储一组量化参数。|
|offset_arr|输出|int16_t数组类型，大小为16，用于存储一组量化参数。|
|sign_mode_arr|输出|bool数组类型，大小为16，用于存储一组量化参数。|
|scale（half）|输入|量化参数，half类型。适用于cast_deq等接口的int32_t转化为half的场景。|
|scale（float）|输入|量化参数，float类型。设置DEQSCALE寄存器的值。|
|offset|输入|量化参数，int16_t类型，只有前9位有效。|
|sign_mode|输入|bool类型，表示量化结果是否带符号。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
constexpr uint32_t src_size = 256;
__ubuf__ half src[src_size];
__ubuf__ half dst[src_size];
float scale = 1.0;
int16_t offset = 2;
bool sign_mode = false;
asc_set_deq_scale(scale, offset, sign_mode);
```
