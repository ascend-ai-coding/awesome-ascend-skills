# 分册 6：风险点与回滚策略

> 对应主流程 **§9**。**失败或未通过评测时必用**；成功归档后可选用作风险复核。留痕 **`Mig_report` §7**；排查细节见 [part-09-examples-troubleshooting.md](part-09-examples-troubleshooting.md)。

---

## 9.1 何时触发回滚 / 回流

| 触发 | 典型现象 | 优先回流 |
|------|----------|----------|
| part-04 smoke 失败 | 无法 load 权重、前向报错、NaN/Inf | part-03（环境）或 part-04（代码） |
| part-05 未达标 | 精度/延迟/吞吐超出允许范围 | part-04（算子/dtype）→ part-05 重测 |
| 环境变更 | CANN/驱动升级后行为变化 | part-03 刷新 `environment.md` |
| 训练 >500 step 无改善 | loss 平盘（见 part-05 §8.2） | part-04 代码/模型实现，**非**继续加步数 |

---

## 9.2 回滚类型与保留物

| 类型 | 保留 | 用途 |
|------|------|------|
| **代码回滚** | 上一可运行 commit/分支 + diff 摘要 | 快速恢复 smoke |
| **权重回滚** | 基线 checkpoint 路径 + 校验 hash | 排除权重损坏 |
| **配置回滚** | 上一版 YAML/ENV（device、AMP、batch） | 隔离配置引入问题 |
| **环境回滚** | `environment.md` 历史快照或 CANN 版本号 | 版本不兼容排查 |

要求 agent 给出回滚与迭代建议：

- 保留基线 checkpoint 与输入数据版本  
- 保留「可运行但未必最优」的中间分支或配置快照  
- 出现运行失败/精度显著下降时，优先调整顺序：**环境 → device/框架插件 → 数据管线/dtype → 算子替换/CPU 回退 → 后处理**  
- 算子不支持：算子替换/回退 → 换精度或固定 shape → 无法代码级解决则 §7 阻塞  
- **训练**：iteration 加大仍无效且 **>500 step** → part-05 §8.2 + `Mig_Readme` §5.3 查代码/模型  

---

## 9.3 决策：回滚到哪一层

```text
报错含 CANN/driver/version / npu-smi 不可见?
  └─ 是 → 暂停 NPU 自动化，刷新 part-03，MANUAL_STOP 清单
  └─ 否 → 报错含 .cuda / unsupported op / HCCL?
        └─ 是 → part-04 + reference-code-patterns
        └─ 否 → 能跑但精度/性能差?
              └─ 是 → Golden + 预处理对齐 → part-05 口径 → Compare
              └─ 否 → 记录 §7，保留日志，小步重试
```

---

## 9.4 §7 留痕模板（复制到 Mig_report）

```markdown
### 问题：<简短标题>（YYYY-MM-DD）

- **触发命令**：（完整命令行）
- **现象**：（报错摘要 / 指标）
- **复现步骤**：1… 2…
- **环境**：（CANN、torch_npu、设备；链到 environment.md）
- **已尝试**：
  | 序号 | 方案 | 结果 |
  |:--:|------|------|
  | 1 | | 失败/部分/成功 |
- **根因**：（确认后填写）
- **修复**：（文件/配置变更摘要）
- **验证**：（smoke / 短测 / Compare 结论）
- **回流**：（part-03 / 04 / 05）
```

- 每轮尝试单独一行，避免重复试错  
- 修复后同步 `Compare.md`、`Summary.md`，并按 [workflow.md](workflow.md)「文档一致性校验矩阵」核对  

---

## 9.5 回滚最小交付（对话末尾固定块）

1. **回滚目标**：分支/配置/checkpoint 版本标识  
2. **回滚命令**：可复制命令与路径  
3. **回滚后验证**：smoke 或约定指标通过标准  
4. **关联**：`Mig_report` §7 问题标题或日期  
5. **日志**：`Mig_report` §8 路径  

---

## 关联索引

- **触发**：part-05 未通过，或 part-04 运行失败（见 [workflow.md](workflow.md) 回流）  
- **配合**：[part-09-examples-troubleshooting.md](part-09-examples-troubleshooting.md)、[reference-code-patterns.md](reference-code-patterns.md)  
- **回流**：part-03（环境）、part-04（代码）、part-05（评测）  
- **落盘**：`Mig_report` §7、§8；必要时更新 `Compare.md`  
- **流程总览**：[workflow.md](workflow.md) 失败路径
