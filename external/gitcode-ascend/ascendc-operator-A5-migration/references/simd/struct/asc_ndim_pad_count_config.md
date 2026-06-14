> **原始文档路径**: asc-devkit/docs/api/context/c_api/struct/asc_ndim_pad_count_config.md
# asc_ndim_pad_count_config

asc_ndim_pad_count_config用于[asc_set_ndim_pad_count](../vector_datamove/asc_set_ndim_pad_count.md)接口中，设置[asc_ndim_copy_gm2ub](../vector_datamove/asc_ndim_copy_gm2ub.md)接口的各个维度左右侧的padding元素个数。

## 结构体具体定义

```cpp
constexpr uint64_t ASC_DEFAULT_NDIM_PAD_COUNT_CONFIG_VALUE = 0;
union asc_ndim_pad_count_config {
    uint64_t config = ASC_DEFAULT_NDIM_PAD_COUNT_CONFIG_VALUE;
    struct {
        uint8_t loop1_lp_count;
        uint8_t loop1_rp_count;
        uint8_t loop2_lp_count;
        uint8_t loop2_rp_count;
        uint8_t loop3_lp_count;
        uint8_t loop3_rp_count;
        uint8_t loop4_lp_count;
        uint8_t loop4_rp_count;
    };
};
```

## 字段详解

|字段名|字段含义|
|----------|----------
| loop1_lp_count | 表示1维左侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]，默认值为0。 |
| loop1_rp_count | 表示1维右侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]，默认值为0。 |
| loop2_lp_count | 表示2维左侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]，默认值为0。 |
| loop2_rp_count | 表示2维右侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]，默认值为0。 |
| loop3_lp_count | 表示3维左侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]，默认值为0。 |
| loop3_rp_count | 表示3维右侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]，默认值为0。 |
| loop4_lp_count | 表示4维左侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]，默认值为0。 |
| loop4_rp_count | 表示4维右侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]，默认值为0。 |
