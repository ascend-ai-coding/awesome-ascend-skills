# Gate 2 YAML 与 Generator 设计

## 输入

- Gate 1 通过结果
- 输入来源（设计文档 / 已有 YAML / 用户参数）
- 目标算子与 dtype/shape/op_param 约束

## 命令

```bash
# 本 Gate 以文件生成与一致性检查为主，按项目脚本或手工生成：
# 1) 生成 ATB_<OpName>_gen.yaml
# 2) 生成 generator_<op>.py
# 3) 生成 node.yaml / node_perf.yaml
```

## 通过标准

- `ATB_<OpName>_gen.yaml`
- `generator_<op>.py`
- `node.yaml`
- `node_perf.yaml`

## 一致性约束

- `api_type` 与 `@register(...)` 一致
- `generate` 与 `@GENERATOR_REGISTRY.register(...)` 一致
- `name` 与 `Operations.cpp` 注册名一致
- 必须包含 `in_formats` 和 `op_param`

### node.yaml / node_perf.yaml 后端约束

- ATK 调度时，**只有 `backend: atb` 的节点**会进入 ATB 侧精度/性能路径；若仅有 `backend: cpu` 等其它后端，不会覆盖 ATB 逻辑验证。
- `node.yaml`、`node_perf.yaml` 各自须**至少包含一条** `backend: atb`（可与 `backend: cpu` 等多节点并存，例如对照路径）。
- 常用约定（可按项目约定微调）：`node.yaml` 精度任务使用 `task: ['accuracy']`；`node_perf.yaml` 设备性能使用 `task: ['performance_device']`。

最小片段示例（与 knowledge 侧约定一致）：

```yaml
# node.yaml（精度）
nodes:
   - backend: atb
     task: ['accuracy']
     devices: [0]
```

```yaml
# node_perf.yaml（性能）
nodes:
   - backend: atb
     task: ['performance_device']
     devices: [1]
```

- [ ] dtype 组合覆盖完整
- [ ] shape 边界合理
- [ ] op_param 字段完整
- [ ] node 配置可执行
- [ ] `node.yaml` 至少包含一条 `backend: atb`
- [ ] `node_perf.yaml` 至少包含一条 `backend: atb`（需要跑 ATB 性能时）
- [ ] 用户确认 Gate 2 通过

## 失败回流

- 一致性字段不匹配：修复 YAML/Generator 命名并重试 Gate 2
- shape 或参数覆盖不足：补充约束后重试 Gate 2
- 缺少 `backend: atb`：补全节点后再试 Gate 2
- 用户未确认：停止在 Gate 2，等待确认后再进入 Gate 3
