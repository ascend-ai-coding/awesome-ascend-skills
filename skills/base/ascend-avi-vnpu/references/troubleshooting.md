# 常见问题与故障排查

## vNPU 操作常见问题

### Q1: 如何查看当前有哪些 vNPU 实例？

```bash
npu-smi info -t info-vnpu -i <设备ID> -c <芯片ID>
```

### Q2: 创建 vNPU 时模板如何选择？

根据工作负载需求选择：

| 模板名          | AICORE | 内存 | 适用场景     |
| --------------- | ------ | ---- | ------------ |
| vir10_3c_16g    | 10     | 16GB | 标准推理     |
| vir10_4c_16g_m  | 10     | 16GB | 媒体处理增强 |
| vir10_3c_16g_nm | 10     | 16GB | 无媒体需求   |
| vir05_1c_8g     | 5      | 8GB  | 轻量级任务   |

### Q3: vNPU ID 可以自定义吗？

可以，范围是 `[phy_id*16+100, phy_id*16+115]`。

例如：

- 设备 0：ID 范围 100-115
- 设备 1：ID 范围 116-131

### Q4: 更改 AVI 模式需要重启吗？

是的，通常需要重启设备才能生效。请提前做好业务规划。

### Q5: 为什么创建 vNPU 失败？

常见原因：

1. **资源不足**：设备资源已被其他 vNPU 占用
2. **模板不存在**：模板名称拼写错误
3. **vNPU ID 冲突**：指定的 ID 已被使用
4. **权限不足**：需要 root 权限

### Q6: 为什么销毁 vNPU 失败？

常见原因：

1. **vNPU ID 不存在**：ID 错误或已销毁
2. **设备忙**：设备正在执行任务
3. **权限不足**：需要 root 权限

### Q7: 如何计算还能创建多少个 vNPU？

```bash
# 先查询当前资源使用
npu-smi info -t info-vnpu -i 0 -c 0
```

计算公式：

```
最大数量 = min(可用AICORE/模板AICORE, 可用内存/模板内存)
```

### Q8: 设备上有进程占用怎么办？

1. 先查看进程信息：

```bash
npu-smi info -t proc-mem -i <设备ID> -c <芯片ID>
```

2. 评估是否可以等待任务结束

3. 如需强制终止，询问用户确认后再执行：

```bash
kill -9 <PID>
```

### Q9: vNPU 实例状态说明

| 状态值 | 说明   |
| ------ | ------ |
| 0      | 正常   |
| 1      | 创建中 |
| 2      | 销毁中 |
| 3      | 异常   |

### Q10: 如何查询 AVI 模式？

```bash
npu-smi info -t vnpu-mode
```

返回：

- `docker`：Container 模式
- `vm`：VM 模式

## 错误码参考

### npu-smi 常见错误

| 错误码 | 说明     | 解决方法         |
| ------ | -------- | ---------------- |
| 0      | 成功     | -                |
| 1      | 失败     | 查看详细错误信息 |
| 215    | 参数错误 | 检查命令参数     |
| 255    | 连接失败 | 检查网络/设备    |

## 性能优化建议

1. **选择合适模板**：避免资源浪费
2. **及时销毁空闲 vNPU**：释放资源供其他任务使用
3. **使用 vgroup 隔离**：不同业务使用不同组
4. **监控资源使用**：定期检查资源利用率

## 技术支持

如遇到无法解决的问题，请收集以下信息：

```bash
# 设备信息
npu-smi info -l
npu-smi info -m

# 健康状态
npu-smi info -t health -i <id>

# vNPU 状态
npu-smi info -t info-vnpu -i <id> -c <chip>

# 进程信息
npu-smi info -t proc-mem -i <id> -c <chip>

# 模板信息
npu-smi info -t template-info -i <id>
```

## 参考链接

- [华为企业支持 - npu-smi 命令参考](https://support.huawei.com/enterprise/zh/doc/EDOC1100540373/e5658bf0?idPath=23710424|251366513|22892968|252309139|251052354)
- [华为昇腾官方文档](https://www.hiascend.com/document)