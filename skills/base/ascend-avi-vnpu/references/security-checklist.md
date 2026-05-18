# 安全操作指南

在进行任何 vNPU 管理操作之前，请遵循以下安全检查流程，确保操作安全和系统稳定。

## 修改操作前的安全检查

### 1. 设备状态检查

查询设备健康状态：

```bash
# 列出所有设备
npu-smi info -l

# 查看指定设备健康状态
npu-smi info -t health -i <设备ID>
```

**健康状态说明**：

- `OK`：设备正常
- `Warning`：设备有警告
- `Error`：设备错误

### 2. 进程占用检查

查询设备上的运行进程：

```bash
npu-smi info -t proc-mem -i <设备ID> -c <芯片ID>
```

如果发现有进程占用 NPU：

- 记录进程信息（PID、名称、内存使用）
- 评估是否需要 kill 进程

### 3. 资源可用性检查

查询 vNPU 资源和模板：

```bash
# 查看当前 vNPU 资源使用
npu-smi info -t info-vnpu -i <设备ID> -c <芯片ID>

# 查看可用模板
npu-smi info -t template-info -i <设备ID>
```

### 4. 用户确认

在执行以下修改操作前，**必须**请求用户确认：

- 设置 AVI 模式
- 创建 vNPU
- 销毁 vNPU
- 设置配置恢复状态

## 进程管理

### kill 占用进程

```bash
# 查看进程详情
ps -p <PID> -o pid,ppid,cmd,etime

# 杀掉进程（需要 root 权限）
kill -9 <PID>
```

### 注意事项

- Kill 进程会导致正在运行的任务中断
- 请提前保存任务状态
- 评估是否可以等任务自然结束

## 资源分配计算

### 可创建 vNPU 数量计算

```
可创建数量 = min(
    floor(可用AICORE / 模板AICORE),
    floor(可用内存 / 模板内存)
)
```

### 示例

设备资源：20 AICORE, 32GB 内存
已用资源：10 AICORE, 16GB 内存
模板：vir10_3c_16g (10 AICORE, 16GB)

```
可用AICORE = 20 - 10 = 10
可用内存 = 32 - 16 = 16GB

可创建数量 = min(floor(10/10), floor(16/16)) = min(1, 1) = 1
```

## 常见操作安全 Checklist

### 创建 vNPU 前

- [ ] 检查设备健康状态
- [ ] 确认无重要进程运行
- [ ] 计算资源是否充足
- [ ] 确认模板选择正确

### 销毁 vNPU 前

- [ ] 确认要销毁的 vNPU ID
- [ ] 确认业务已迁移或结束
- [ ] 确认不影响其他 vNPU

### 修改 AVI 模式前

- [ ] 确认业务可以中断
- [ ] 提前通知相关人员
- [ ] 规划重启时间窗口

## 故障恢复

### 创建失败

可能原因：

1. 资源不足 → 销毁空闲 vNPU 或等待任务结束
2. 模板不存在 → 使用 `npu-smi info -t template-info` 确认模板名
3. vNPU ID 冲突 → 使用自动分配或更换 ID

### 销毁失败

可能原因：

1. vNPU ID 不存在 → 使用 `npu-smi info -t info-vnpu` 确认
2. 设备忙 → 等待或强制销毁

## 参考链接

- [华为企业支持 - npu-smi 命令参考](https://support.huawei.com/enterprise/zh/doc/EDOC1100540373/e5658bf0?idPath=23710424|251366513|22892968|252309139|251052354)
- [华为昇腾官方文档](https://www.hiascend.com/document)

