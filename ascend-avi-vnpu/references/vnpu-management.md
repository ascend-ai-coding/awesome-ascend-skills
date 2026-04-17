# vNPU 管理参考

vNPU（Virtual NPU）是昇腾 NPU 的虚拟化实例，允许将物理 NPU 资源划分为多个虚拟实例。

## vNPU 生命周期

```
创建 → 查询 → 使用 → 销毁
```

## 查询 vNPU 信息

### 查询指定芯片的 vNPU

```bash
npu-smi info -t info-vnpu -i <设备ID> -c <芯片ID>
```

### 输出字段说明

| 字段                 | 说明                    |
| -------------------- | ----------------------- |
| AICORE               | 已用/总 AI Core 数量    |
| Memory               | 已用/总内存大小 (GB)    |
| AICPU                | 已用/总 AI CPU 数量     |
| VPC                  | 已用/总 VPC 数量        |
| VENC                 | 已用/总视频编码器数量   |
| VDEC                 | 已用/总视频解码器数量   |
| JPEGD                | 已用/总 JPEG 解码器数量 |
| JPEGE                | 已用/总 JPEG 编码器数量 |
| Total number of vnpu | vNPU 实例总数           |
| Vnpu ID              | vNPU 实例 ID            |
| Vgroup ID            | vNPU 组 ID              |
| Container ID         | 容器 ID                 |
| Status               | 状态（0=正常）          |
| Template Name        | 使用的模板名称          |

## 创建 vNPU

### 命令

```bash
npu-smi set -t create-vnpu -i <设备ID> -c <芯片ID> -f <模板名> [-v <vnpu_id>] [-g <vgroup_id>]
```

### 参数说明

| 参数 | 必填 | 说明        | 范围                          |
| ---- | ---- | ----------- | ----------------------------- |
| i    | 是   | 物理 NPU ID | 0-7                           |
| c    | 是   | 芯片 ID     | 0-N                           |
| f    | 是   | 模板名称    | 见模板列表                    |
| v    | 否   | vNPU ID     | phy_id*16+100 ~ phy_id*16+115 |
| g    | 否   | vNPU 组 ID  | 0-N                           |

### 示例

```bash
# 使用自动分配的 ID 创建
npu-smi set -t create-vnpu -i 0 -c 0 -f vir10_3c_16g

# 指定 vNPU ID 创建
npu-smi set -t create-vnpu -i 0 -c 0 -f vir10_3c_16g -v 100

# 在指定组中创建
npu-smi set -t create-vnpu -i 0 -c 0 -f vir10_3c_16g -v 101 -g 1
```

## 销毁 vNPU

### 命令

```bash
npu-smi set -t destroy-vnpu -i <设备ID> -c <芯片ID> -v <vnpu_id>
```

### 参数说明

| 参数 | 必填 | 说明             |
| ---- | ---- | ---------------- |
| i    | 是   | 物理 NPU ID      |
| c    | 是   | 芯片 ID          |
| v    | 是   | 要销毁的 vNPU ID |

### 示例

```bash
# 销毁指定 vNPU
npu-smi set -t destroy-vnpu -i 0 -c 0 -v 100
```

## vNPU 配置恢复

### 查询配置恢复状态

```bash
npu-smi info -t vnpu-cfg-recover -i <设备ID> -c <芯片ID>
```

返回：

- `Enable`: 启用
- `Disable`: 禁用

### 设置配置恢复状态

```bash
npu-smi set -t vnpu-cfg-recover -i <设备ID> -c <芯片ID> -d <0|1>
```

参数：

- `0`: 禁用
- `1`: 启用

### 示例

```bash
# 查询状态
npu-smi info -t vnpu-cfg-recover -i 0 -c 0

# 启用配置恢复
npu-smi set -t vnpu-cfg-recover -i 0 -c 0 -d 1

# 禁用配置恢复
npu-smi set -t vnpu-cfg-recover -i 0 -c 0 -d 0
```

## vNPU 组（Vgroup）

vNPU 组用于实现资源隔离和多租户场景。

- **Vgroup 0-N**：共 N 个组
- 不同组的 vNPU 资源相互隔离
- 同一组内 vNPU 共享资源池

## 最佳实践

1. **资源规划**：根据工作负载选择合适的模板
2. **ID 管理**：使用统一的 vNPU ID 命名规则便于管理
3. **及时清理**：不再使用的 vNPU 应及时销毁
4. **监控状态**：定期检查 vNPU 运行状态
5. **隔离策略**：使用 vgroup 实现 workload 隔离

## 参考链接

- [华为企业支持 - npu-smi 命令参考](https://support.huawei.com/enterprise/zh/doc/EDOC1100540373/e5658bf0?idPath=23710424|251366513|22892968|252309139|251052354)
- [华为昇腾官方文档](https://www.hiascend.com/document)