> **原始文档路径**: asc-devkit/docs/api/context/c_api/struct/asc_load3d_v2_config.md

# asc_load3d_v2_config

asc_load3d_v2_config用于设置Load3Dv2接口的repeat参数。

## 结构体具体定义

```cpp
constexpr uint64_t ASC_DEFAULT_LOAD3D_V2_CONFIG_VALUE = 0x0000000000010000;
union asc_load3d_v2_config {
    uint64_t config = ASC_DEFAULT_LOAD3D_V2_CONFIG_VALUE;
    struct {
        uint64_t rpt_stride : 16;
        uint64_t rpt_time : 8;
        uint64_t rpt_mode : 1;
        uint64_t reserved2 : 39;
    };
};
```

## 字段详解

|字段名|字段含义|
|----------|----------|
| rpt_stride | height/width方向上的前一个迭代与后一个迭代起始地址的距离，取值范围：n∈[0, 65535]，默认值为0。 |
| rpt_time | height/width方向上的迭代次数，取值范围：repeatTime ∈[0, 255] 。默认值为1。|
| rpt_mode | 控制repeat迭代的方向，取值范围：k∈[0, 1] 。默认值为0。<br> 0：迭代沿height方向；<br>1：迭代沿width方向 |
