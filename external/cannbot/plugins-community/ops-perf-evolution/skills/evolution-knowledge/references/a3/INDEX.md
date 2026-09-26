# Evolution Knowledge Base

面向进化优化 Agent 的领域知识库。采用渐进式披露：先读 guide.md 快速参考，按需深入详细文件。

## 分类概览

| 目录 | 类型 | 内容 | 快速参考 |
|------|------|------|---------|
| `hardware/` | 硬件约束 | 芯片架构、流水线模型、Roofline | [guide.md](hardware/guide.md) |
| `optimization_patterns/` | 优化模板 | 已验证的代码模式（含模板和实测数据） | [guide.md](optimization_patterns/guide.md) |
| `algorithm_insights/` | 算法洞见 | 按算子族分类的算法级优化 | [guide.md](algorithm_insights/guide.md) |
| `ascendc_api/` | API 避坑 | 常见编码陷阱和正确用法 | [guide.md](ascendc_api/guide.md) |
| `proven_solutions/` | 方案沉淀 | 历次进化中发现的新技术 | [INDEX.md](proven_solutions/INDEX.md) |

## 工作流绑定（谁在什么时候必须/建议读什么）

### 主 Agent（lingxi-evo / ops-evo）

| 工作流步骤 | 必读文件 | 建议读文件 | 触发条件 |
|-----------|---------|-----------|---------|
| Init（世界模型初始化） | `hardware/guide.md` | `hardware/ascend910b_arch.md` (hw_params 不足时) | 每次进化必执行 |
| Init（节点设计） | `optimization_patterns/guide.md` + **`optimization_patterns/scalar_to_vector.md`** | 按算子族读 `algorithm_insights/{family}.md` | 每次进化必执行（标量检查最高优先） |
| 4.4.S 策略提炼 | `proven_solutions/INDEX.md` | — | open_exploration 成功且提升 ≥ 10% |
| Init（同类算子存在时） | — | `proven_solutions/` 中匹配的方案 | 算子类型与已有方案匹配 |

### 子 Agent（lingxi-partial / ops-partial）

| 优化模式 | 必读文件 | 建议读文件 | 禁读文件 |
|---------|---------|-----------|---------|
| strategy_guided | `ascendc_api/guide.md` | 策略引用的 `optimization_patterns/*.md` | — |
| open_exploration | `ascendc_api/guide.md` | `algorithm_insights/{family}.md` | 策略库文件 (strategies/*.md) |
| profiling_driven | `ascendc_api/guide.md` | 父节点 profiling 相关的 `optimization_patterns/*.md` | 策略库文件 |

## 检索优先级链

当知识库不足时，按以下顺序扩展检索：

```
1. plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/         (本地知识库 — 已验证的领域知识)
     ↓ 不足
2. plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/ (策略库 — 指令级优化模板)
     ↓ 不足
3. proven_solutions/                  (历史方案 — 跨算子迁移)
     ↓ 不足
4. WebSearch                          (外部知识 — 论文、开源实现)
```

## 知识质量标准

- 必须包含具体代码示例（非纯文字描述）
- 必须标注硬件约束和适用条件
- 必须包含实测性能影响（优化前后对比）
- 优先收录跨算子验证过的模式
