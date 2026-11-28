# MatMul 族 Tiling 算法路由

> MatMul 族（matmul / matmul_a16w16 / matmul_mxfp4 / batch_matmul / group_matmul）Tiling 分析入口。默认进行理论分析；实测求解器仅在用户明确要求执行时进入。
>
> 先按目标代际路由。`arch22`（A2/A3）读取目标专用知识；`arch35`（A5）才进入本目录的 SWAT / FullLoad / StreamK fallback。

---

## 1. 路由规则

默认读取 [通用矩阵乘 Tiling 原理](solver/references/matmul-tiling-theory.md)，再结合目标架构参考、当前源码、Shape、dtype 和已有性能资料进行分析。通用理论与架构参考共同支撑分析，不互相替代。

理论分析给出几个可能的优化方向即可：例如调整分块以改善复用、增加任务数以提高并行度，或调整缓冲以改善流水。每个方向简述依据和主要取舍；资料不足的判断标为待验证假设。无需完整参数表、候选集或实测排名，不将推测写成已验证的最优配置。

```
给定：目标 SoC/架构 + 算子类型 + Shape + dtype

读取通用 Tiling 原理后，匹配目标代际：
  ├─ arch22（DAV_2201 / 220x）→ arch22 MatMul 指南
  ├─ arch35（DAV_3510）       → 遍历已注册算法
  └─ 未知目标                 → 返回 architecture_reference_gap

arch35 内按优先级匹配：
  ├─ 命中扩展算法 → 路由到该算法
  └─ 未命中       → 回退到 fallback 兜底算法
```

---

## 2. 已注册算法

| 优先级 | 算法/知识入口 | 选择条件 | 目录 |
|--------|------|---------|------|
| 100 | arch22 MatMul 优化参考 | DAV_2201 / 220x；A2/A3 | [A2/A3 MatMul 指南](../../../../ascendc-performance-best-practices/references/matmul/arch22/guide.md) |
| 0（兜底） | arch35 官方参考 | DAV_3510 且变体/能力匹配 | [fallback/](fallback/) |
| —（显式触发） | 实测求解工具包 | 用户明确要求执行实测，或接受实测方案；详见下文 | [solver.md](solver.md) |

arch22 条目是 A2/A3 的分块、复用与流水优化知识入口，优先级用于路由选择，不代表性能排名。

arch22 路线继续读取 [流水、预取与大包搬运](../../../../ascendc-performance-best-practices/references/matmul/arch22/pipeline_design.md) 和 [复用、切分与缓存](../../../../ascendc-performance-best-practices/references/matmul/arch22/reuse_and_split_design.md)。A2 样例参数和性能不能直接写成 A3 结论；A3 必须重新代入核数、L0/L1/L2、事件/API 能力并逐 case 验证。

`fallback/` 的 SWAT / FullLoad / StreamK、MX scale 和固定阈值是 arch35 资料，A2/A3 不得把它作为默认兜底。

[实测求解工具包](solver.md) 不参与上表的算法优先级排序。仅当用户明确要求运行上板调优、TTK 候选扫描或完整实测求解，或者接受已说明范围与预计耗时的实测方案后，才进入其完整流程。沿用用户已经授权的范围与预算，无需逐阶段重复确认。

普通「调优算子」「找最优 tiling」请求，以及仅讨论求解器或 TTK 的问题，均不自动触发实测。默认理论分析不要求生成算子档案、创建编译 worktree、构建安装包、运行基线、采集锚点或扫描候选；不因设备可用或缺少实测证据自动启动这些步骤。

给出优化方向即完成本轮理论分析，并继续主工作流中适用的流水分析；实测不是理论分析后的必做阶段。

---

## 3. 贡献新算法

在 `matmul/` 下创建新目录，在此文件的路由表中注册选择条件和优先级：

```
matmul/
├── index.md              # 本文件（算法路由）
├── fallback/             # arch35 官方参考算法
│   ├── fallback-algorithm.md
│   ├── tiling-flow.md
│   ├── tiling-fields.md
│   └── tiling-variants.md
├── solver.md             # 实测求解工具包入口（候选扫描 + 真机实测裁决）
├── solver/               # 自包含求解器（scripts/ + references/ + 运行时生成的 profiles/）
└── <算法名>/             # 社区贡献
    └── algorithm.md
```

选择条件必须先包含适用架构，再组合 dtype、shape 范围、核数等可判定字段。
