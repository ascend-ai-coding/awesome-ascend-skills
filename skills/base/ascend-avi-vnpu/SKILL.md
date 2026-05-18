---
name: ascend-avi-vnpu
description: 昇腾虚拟化实例（AVI）和vNPU管理技能。用于查询/设置AVI模式、管理vNPU实例（创建/销毁）、查询vNPU配置恢复状态等操作。支持安全确认机制，在执行修改操作前检查设备状态。
keywords:

   - AVI
   - vNPU
   - 虚拟化
   - 昇腾
   - Ascend
   - npu-smi
   - 模板
   - 实例管理
   - 算力切分
---

# 昇腾虚拟化实例（AVI）vNPU 管理

本技能用于管理华为昇腾 NPU 的虚拟化实例（AVI）和 vNPU。

> **注意**: 在执行任何修改操作（设置、创建、销毁）之前，系统会检查设备状态并请您确认。

## 快速开始

### 查询操作

```bash
# 查询AVI模式
npu-smi info -t vnpu-mode

# 查询所有AVI模板
npu-smi info -t template-info

# 查询指定设备的模板
npu-smi info -t template-info -i <设备ID>

# 查询指定芯片的vNPU信息
npu-smi info -t info-vnpu -i <设备ID> -c <芯片ID>

# 查询vNPU配置恢复使能状态
npu-smi info -t vnpu-cfg-recover -i <设备ID> -c <芯片ID>
```

### 修改操作

```bash
# 设置AVI模式 (0=Container, 1=VM)
npu-smi set -t vnpu-mode -d <0|1>

# 创建vNPU实例
npu-smi set -t create-vnpu -i <设备ID> -c <芯片ID> -f <模板名> [-v <vnpu_id>] [-g <vgroup_id>]

# 销毁vNPU实例
npu-smi set -t destroy-vnpu -i <设备ID> -c <芯片ID> -v <vnpu_id>

# 设置vNPU配置恢复使能状态
npu-smi set -t vnpu-cfg-recover -i <设备ID> -c <芯片ID> -d <0|1>
```

## 查询操作详情

### 1. 查询昇腾虚拟化实例（AVI）模式

```bash
npu-smi info -t vnpu-mode
```

返回说明：

- `0`: Container 模式（Docker/容器虚拟化）
- `VM`: VM 模式（虚拟机虚拟化）

### 2. 查询昇腾虚拟化实例（AVI）模板信息

```bash
# 查询所有模板
npu-smi info -t template-info

# 查询指定设备的模板
npu-smi info -t template-info -i <设备ID>
```

常用模板：

| 模板名 | AI Core数 | 内存 | 适用场景       |
| ------ | --------- | ---- | -------------- |
| vir02  | 2         | 2GB  | 轻量级工作负载 |
| vir04  | 4         | 4GB  | 中等工作负载   |
| vir08  | 8         | 8GB  | 重工作负载     |

### 3. 查询指定设备的昇腾虚拟化实例（AVI）模板信息

```bash
npu-smi info -t template-info -i <设备ID>
```

### 4. 查询指定芯片的vNPU

```bash
npu-smi info -t info-vnpu -i <设备ID> -c <芯片ID>
```

返回字段：

| 字段          | 说明              |
| ------------- | ----------------- |
| vNPU ID       | vNPU标识符        |
| vNPU Group ID | vNPU组ID          |
| AI Core Num   | 分配的AI Core数量 |
| Memory Size   | 分配的内存大小    |
| Status        | 当前状态          |

### 5. 查询vNPU的配置恢复使能状态

```bash
npu-smi info -t vnpu-cfg-recover -i <设备ID> -c <芯片ID>
```

返回：`Enable` (启用) 或 `Disable` (禁用)

## 修改操作详情

### 6. 设置昇腾虚拟化实例（AVI）模式

```bash
npu-smi set -t vnpu-mode -d <0|1>
```

参数说明：

- `0`: Container 模式
- `1`: VM 模式

**注意**: 更改AVI模式可能需要重启设备。

### 7. 创建指定芯片的vNPU

```bash
npu-smi set -t create-vnpu -i <设备ID> -c <芯片ID> -f <模板名> [-v <vnpu_id>] [-g <vgroup_id>]
```

参数说明：

| 参数 | 必填 | 说明                                            |
| ---- | ---- | ----------------------------------------------- |
| i    | 是   | 物理NPU ID                                      |
| c    | 是   | 芯片ID                                          |
| f    | 是   | 模板名称 (如 vir02, vir04, vir08)               |
| v    | 否   | 自定义vNPU ID: `[phy_id*16+100, phy_id*16+115]` |
| g    | 否   | 组ID: 0, 1, 2, 或 3                             |

示例：

```bash
# 使用自动分配的vNPU ID创建
npu-smi set -t create-vnpu -i 0 -c 0 -f vir04

# 指定vNPU ID创建
npu-smi set -t create-vnpu -i 0 -c 0 -f vir04 -v 103

# 在指定组中创建
npu-smi set -t create-vnpu -i 0 -c 0 -f vir04 -v 104 -g 1
```

### 8. 销毁指定芯片的vNPU

```bash
npu-smi set -t destroy-vnpu -i <设备ID> -c <芯片ID> -v <vnpu_id>
```

参数说明：

| 参数 | 必填 | 说明            |
| ---- | ---- | --------------- |
| i    | 是   | 物理NPU ID      |
| c    | 是   | 芯片ID          |
| v    | 是   | 要销毁的vNPU ID |

示例：

```bash
npu-smi set -t destroy-vnpu -i 0 -c 0 -v 103
```

### 9. 设置vNPU的配置恢复使能状态

```bash
npu-smi set -t vnpu-cfg-recover -i <设备ID> -c <芯片ID> -d <0|1>
```

参数说明：

- `0`: 禁用配置恢复
- `1`: 启用配置恢复

## 安全检查机制

在执行任何修改操作之前，系统会自动执行以下检查：

1. **设备状态检查**: 执行 `npu-smi info -l` 和 `npu-smi info -t health` 查看设备健康状态
2. **进程占用检查**: 执行 `npu-smi info -t proc-mem` 查看是否有程序占用NPU
3. **用户确认**: 在执行修改操作前，会请求您确认是否继续

如果检测到有程序占用NPU，系统会：

- 显示占用程序信息
- 询问您是否要kill掉这些程序后再执行操作

### ⚠️ 重要：模板名称确认

当用户使用模糊表达（如"模板1"、"第一个模板"、"第二个模板"）时，**必须先确认具体的模板名称**，不能直接猜测执行。

**正确的处理流程：**

1. 用户说"模板1" → 显示可用模板列表，让用户确认是哪一个
2. 用户说"第一个模板" → 同上
3. 用户说"用模板 vir10_3c_16g" → 直接执行

**可用模板列表示例：**

| 序号 | 模板名          | AICORE | 内存 |
| ---- | --------------- | ------ | ---- |
| 1    | vir10_3c_16g    | 10     | 16GB |
| 2    | vir10_4c_16g_m  | 10     | 16GB |
| 3    | vir10_3c_16g_nm | 10     | 16GB |
| 4    | vir05_1c_8g     | 5      | 8GB  |

## 参数参考

| 参数      | 说明         | 获取方式                               |
| --------- | ------------ | -------------------------------------- |
| `设备ID`  | NPU设备ID    | `npu-smi info -l`                      |
| `芯片ID`  | 运行时芯片ID | `npu-smi info -m` (通常为0)            |
| `vNPU ID` | vNPU实例ID   | 范围: `[phy_id*16+100, phy_id*16+115]` |

## 常见问题

### Q: 如何查看当前有哪些vNPU实例？

```bash
npu-smi info -t info-vnpu -i 0 -c 0
```

### Q: 创建vNPU时模板如何选择？

根据工作负载需求选择：

| 模板名          | AICORE | 内存 | 说明         |
| --------------- | ------ | ---- | ------------ |
| vir10_3c_16g    | 10     | 16GB | 标准模板     |
| vir10_4c_16g_m  | 10     | 16GB | 媒体增强模板 |
| vir10_3c_16g_nm | 10     | 16GB | 非媒体模板   |
| vir05_1c_8g     | 5      | 8GB  | 轻量级模板   |

> **注意**: 当您说"模板1"、"第一个模板"等模糊表达时，我会列出模板请您确认后再执行。

### Q: vNPU ID可以自定义吗？

可以，范围是 `[phy_id*16+100, phy_id*16+115]`。例如设备0的vNPU ID范围是 `100-115`。

### Q: 更改AVI模式需要重启吗？

通常需要重启设备才能生效。请提前做好业务规划。

## 参考资料

### 本地文档

- [[AVI 模式说明](https://github.com/Jingbo-gao/test/issues/references/avi-mode.md)](references/avi-mode.md)
- [[模板信息详解](https://github.com/Jingbo-gao/test/issues/references/template-info.md)](references/template-info.md)
- [[vNPU 管理指南](https://github.com/Jingbo-gao/test/issues/references/vnpu-management.md)](references/vnpu-management.md)
- [[安全操作指南](https://github.com/Jingbo-gao/test/issues/references/security-checklist.md)](references/security-checklist.md)
- [[常见问题与故障排查](https://github.com/Jingbo-gao/test/issues/references/troubleshooting.md)](references/troubleshooting.md)

### 官方文档

- [[npu-smi 命令参考](https://support.huawei.com/enterprise/zh/doc/EDOC1100540373/e5658bf0?idPath=23710424%7C251366513%7C22892968%7C252309139%7C251052354)](https://support.huawei.com/enterprise/zh/doc/EDOC1100540373/e5658bf0?idPath=23710424|251366513|22892968|252309139|251052354)
- [[华为昇腾官方文档](https://www.hiascend.com/document)](https://www.hiascend.com/document)