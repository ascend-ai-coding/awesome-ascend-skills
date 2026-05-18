# AVI 模板参考

AVI 模板定义了 vNPU 实例的资源分配规格，包括 AI Core 数量、内存大小、各类加速器资源等。

## 查询模板

### 查询所有设备的模板

```bash
npu-smi info -t template-info
```

### 查询指定设备的模板

```bash
npu-smi info -t template-info -i <设备ID>
```

## 模板参数说明

| 参数   | 说明            |
| ------ | --------------- |
| Name   | 模板名称        |
| AICORE | AI Core 数量    |
| Memory | 内存大小（GB）  |
| AICPU  | AI CPU 数量     |
| VPC    | VPC 数量        |
| VENC   | 视频编码器数量  |
| VDEC   | 视频解码器数量  |
| JPEGD  | JPEG 解码器数量 |
| JPEGE  | JPEG 编码器数量 |
| PNGD   | PNG 解码器数量  |

## 常用模板

### Atlas 910 系列常见模板

| 模板名          | AICORE | 内存(GB) | AICPU | VPC  | 说明               |
| --------------- | ------ | -------- | ----- | ---- | ------------------ |
| vir10_3c_16g    | 10     | 16       | 3     | 4    | 标准模板，媒体功能 |
| vir10_4c_16g_m  | 10     | 16       | 4     | 9    | 媒体增强模板       |
| vir10_3c_16g_nm | 10     | 16       | 3     | 0    | 非媒体模板         |
| vir05_1c_8g     | 5      | 8        | 1     | 2    | 轻量级模板         |

### 模板选择建议

| 场景       | 推荐模板        |
| ---------- | --------------- |
| 轻量级推理 | vir05_1c_8g     |
| 标准推理   | vir10_3c_16g    |
| 媒体处理   | vir10_4c_16g_m  |
| 无媒体需求 | vir10_3c_16g_nm |

## 计算可创建的 vNPU 数量

根据模板资源需求和设备总资源计算：

```
最大 vNPU 数量 = min(总AICORE / 模板AICORE, 总内存 / 模板内存)
```

### 示例

设备资源：20 AICORE, 32GB 内存
模板 vir10_3c_16g：10 AICORE, 16GB 内存

```
最大数量 = min(20/10, 32/16) = min(2, 2) = 2 个
```

## 模板资源限制

- **最小模板**：通常需要至少 1 个 AICORE 和 2GB 内存
- **最大模板**：取决于设备硬件规格
- **vNPU ID 范围**：`[phy_id*16+100, phy_id*16+115]`

例如：

- 设备 0 的 vNPU ID 范围：100-115
- 设备 1 的 vNPU ID 范围：116-131

## 参考链接

- [华为企业支持 - npu-smi 命令参考](https://support.huawei.com/enterprise/zh/doc/EDOC1100540373/e5658bf0?idPath=23710424|251366513|22892968|252309139|251052354)
- [华为昇腾官方文档](https://www.hiascend.com/document)