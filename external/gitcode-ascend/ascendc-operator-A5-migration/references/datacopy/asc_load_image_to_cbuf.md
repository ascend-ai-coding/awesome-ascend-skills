> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_load_image_to_cbuf.md


# asc_load_image_to_cbuf

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

将图像数据从Global Memory搬运到L1 Buffer。

## 函数原型

- 常规计算

    ```cpp
    __aicore__ inline void asc_load_image_to_cbuf(__cbuf__ int8_t* dst, uint16_t hor_size, uint16_t ver_size, uint16_t hor_start_pos, uint16_t ver_start_pos, uint16_t src_hor_size, uint8_t top_pad_size, uint8_t bot_pad_size, uint16_t left_pad_size, uint16_t right_pad_size)
    __aicore__ inline void asc_load_image_to_cbuf(__cbuf__ half* dst, uint16_t hor_size, uint16_t ver_size, uint16_t hor_start_pos, uint16_t ver_start_pos, uint16_t src_hor_size, uint8_t top_pad_size, uint8_t bot_pad_size, uint16_t left_pad_size, uint16_t right_pad_size)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_load_image_to_cbuf_sync(__cbuf__ int8_t* dst, uint16_t hor_size, uint16_t ver_size, uint16_t hor_start_pos, uint16_t ver_start_pos, uint16_t src_hor_size, uint8_t top_pad_size, uint8_t bot_pad_size, uint16_t left_pad_size, uint16_t right_pad_size)
    __aicore__ inline void asc_load_image_to_cbuf_sync(__cbuf__ half* dst, uint16_t hor_size, uint16_t ver_size, uint16_t hor_start_pos, uint16_t ver_start_pos, uint16_t src_hor_size, uint8_t top_pad_size, uint8_t bot_pad_size, uint16_t left_pad_size, uint16_t right_pad_size)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| hor_size | 输入 | 从源图中加载图片的水平宽度，单位为像素。取值范围：[1, 4094]。 |
| ver_size | 输入 | 从源图中加载图片的垂直高度，单位为像素。取值范围：[1, 4094]。 |
| hor_start_pos | 输入 | 加载图片在源图片上的水平起始地址，单位为像素。取值范围：[0, 4095]。 |
| ver_start_pos | 输入 | 加载图片在源图片上的垂直起始地址，单位为像素。取值范围：[0, 4095]。 |
| src_hor_size | 输入 | 源图片水平宽度，单位为像素。取值范围：[1, 4094]。 |
| top_pad_size | 输入 | 目的图片顶部填充的像素数。取值范围：[0, 32]。 |
| bot_pad_size | 输入 | 目的图片底部填充的像素数。取值范围：[0, 32]。 |
| left_pad_size | 输入 | 目的图片左边填充的像素数。取值范围：[0, 32]。 |
| right_pad_size | 输入 | 目的图片右边填充的像素数。取值范围：[0, 32]。 |

## 返回值说明

无

## 流水类型

PIPE_MTE2

## 约束说明

- dst的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
constexpr uint64_t total_length = 128;    // total_length指参与搬运的数据总长度
__ca__ half dst[total_length];
// 加载图片时的高度与宽度均为1，起始地址为(0,0)，源图片水平宽度为1，不做padding
asc_load_image_to_cbuf(dst, 1, 1, 0, 0, 1, 0, 0, 0, 0);
```