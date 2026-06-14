> **原始文档路径**: asc-devkit/docs/api/context/c_api/struct/asc_l13d_fmatrix_config.md
# asc_l13d_fmatrix_config

asc_l13d_fmatrix_config用于设置[asc_copy_l12l0a](../cube_datamove/asc_copy_l12l0a)/[asc_copy_l12l0b](../cube_datamove/asc_copy_l12l0b)的3D格式搬运接口的Feature map属性参数。

## 结构体具体定义

```cpp
constexpr uint64_t ASC_DEFAULT_L13D_FMATRIX_CONFIG_VALUE = 0;
union asc_l13d_fmatrix_config {
    uint64_t config = ASC_DEFAULT_L13D_FMATRIX_CONFIG_VALUE;
    struct {
        uint16_t l1_height;
        uint16_t l1_width;
        uint8_t padding_left_size;
        uint8_t padding_right_size;
        uint8_t padding_top_size;
        uint8_t padding_bottom_size;
    };
};
```

## 字段详解

|字段名|字段含义|
|----------|----------|
| l1_height | Feature map的height，取值范围：l1_height∈[1, 32767]。 |
| l1_width | Feature map的width，取值范围：l1_width∈[1, 32767]。 |
| padding_left_size | 左侧填充的大小，取值范围：padding_left_size ∈[0, 255] 。|
| padding_right_size | 右侧填充的大小，取值范围：padding_right_size ∈[0, 255] 。|
| padding_top_size | 上侧填充的大小，取值范围：padding_top_size ∈[0, 255] 。|
| padding_bottom_size | 下侧填充的大小，取值范围：padding_bottom_size ∈[0, 255] 。|
