> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_get_smmu_tag_version.md

# asc_get_smmu_tag_version

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

获取SMMU（System Memory Management Unit）版本信息。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_smmu_tag_version()
```

## 参数说明

无

## 返回值说明

SMMU版本信息。各bit含义如下：
| bit范围    | 含义 |
| ----------- |:----|
| 63:28 | 保留位。 |
| 27:0 | SMMU标签版本。 |
| 27:14 | 年。格式：YY-YYYY-YYYY。 |
| 13:8 | 月。格式：MM-MMMM。 |
| 7:2 | 日。格式：DD-DDDD。 |
| 1:0 | 同一日期的子版本。 |

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
int64_t smmu_tag_version = asc_get_smmu_tag_version();
printf("smmu tag version is %x", smmu_tag_version);// 需用%x将其打印成十六进制的数
```