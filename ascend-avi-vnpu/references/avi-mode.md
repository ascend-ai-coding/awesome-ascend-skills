# AVI 模式参考

昇腾虚拟化实例（AVI，Ascend Virtualization Infrastructure）模式定义了 NPU 资源的虚拟化方式。

## AVI 模式类型

| 模式值 | 模式名称  | 说明                  |
| ------ | --------- | --------------------- |
| 0      | Container | Docker/容器虚拟化模式 |
| 1      | VM        | 虚拟机虚拟化模式      |

## 查询 AVI 模式

```bash
npu-smi info -t vnpu-mode
```

### 输出示例

```
vnpu-mode                      : docker
```

或

```
vnpu-mode                      : vm
```

## 设置 AVI 模式

```bash
npu-smi set -t vnpu-mode -d <0|1>
```

参数说明：

- `0`: 设置为 Container 模式
- `1`: 设置为 VM 模式

### 示例

```bash
# 设置为 Container 模式
npu-smi set -t vnpu-mode -d 0

# 设置为 VM 模式
npu-smi set -t vnpu-mode -d 1
```

## 注意事项

1. **更改模式需要重启**：修改 AVI 模式后，通常需要重启设备才能生效
2. **资源分配影响**：不同模式下，vNPU 的资源分配策略可能有所不同
3. **业务中断**：更改模式会导致现有 vNPU 实例不可用，请提前规划

## 适用场景

### Container 模式

- 容器化部署场景
- Kubernetes/Docker 环境
- 轻量级虚拟化需求

### VM 模式

- 虚拟机隔离需求
- 传统虚拟化环境
- 强隔离场景

## 参考链接

- [[华为企业支持 - npu-smi 命令参考](https://support.huawei.com/enterprise/zh/doc/EDOC1100540373/e5658bf0?idPath=23710424%7C251366513%7C22892968%7C252309139%7C251052354)](https://support.huawei.com/enterprise/zh/doc/EDOC1100540373/e5658bf0?idPath=23710424|251366513|22892968|252309139|251052354)
- [[华为昇腾官方文档](https://www.hiascend.com/document)](https://www.hiascend.com/document)