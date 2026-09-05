# 性能相关 Skill 审视备忘

> 供 Agent 维护 skill 时参考；交付给用户时引用关键结论即可。

## 已覆盖能力（2026-07 更新）

| 能力 | 文档位置 |
| --- | --- |
| baseline C++ 接入 + perf_compare | baseline-compare-workflow.md |
| Phase 6 聊天自动输出 | perf-chat-output-spec.md §2 |
| Phase 6.5 每轮 Δ% 自动输出 | perf-chat-output-spec.md §3~§4 |
| 性能评估唯一标准 | timing-and-metrics-standard.md |
| S 档 + L 档采集（Round 0 + 最终轮） | SKILL.md 写入规则 |

## 历史修正记录（2026-07）

| 项 | 原状 | 现规范 |
| --- | --- | --- |
| steady_bus_bandwidth_GBps | 独立指标，需 sweep 确认平台区 | **已删除**；统一使用 `kernel_bus_bandwidth_GBps`（kernel 口径，达标主指标），见 timing-and-metrics-standard.md |
| platform-perf-spec.md | 四算子独立目标表 | **已删除**；规模门禁与统一达标线迁入 [performance-eval-guide.md §4](performance-eval-guide.md)（有 baseline ≥ 80%，无 baseline 利用率 ≥ 20%） |
| sweep 平台区 | perf-workflow.md §2 倍增 sweep | **已删除**；性能采集不再要求 sweep |
| S 档数据流 | Round 0 暂存 → 最终轮写入 §6.2 | **已修正**；Round 0 直接写入 §6.2 baseline 列 |
| 达标指标不一致 | optim 用 e2e，eval 用 steady_bus | **已统一**：timing-and-metrics-standard.md 为唯一标准 |
| Phase 6.5 入口 | 曾写「未达标才进入」与「达标也跑满」矛盾 | **`performance_auto_optim:true` 且未达标 → Phase 6 后 MUST 进入**；进入后 MUST 跑满 `max_opt_rounds`（默认 5）轮（达标 **不是** 提前停止理由），用户明确指定轮次时 MUST 跑满指定轮次（即使已达标） |
| Phase 0 | 业务开关未确认 | Torch/性能/`performance_auto_optim` 已写入 `phase0_intake`（先探测环境，缺项再问） |
| Skill/仓路径 | skill 相对路径链到 `docs/` | **先定位 `SHMEM_REPO`**（shmem-repo-resolution.md） |
| 聊天输出 | 「不得等用户追问」分散多处 | 集中 **perf-chat-output-spec.md** |

## 仍待工程化（非 skill 条文能单独解决）

1. **连续 8PE 测试资源冲突** → 已改为 **HCCL/SHMEM 分阶段 + 离线对比**（见 [perf-workflow.md](perf-workflow.md)）
2. ~~HCCL 不得在 SHMEM init 之后调用~~ → skill + 脚本已强制隔离
3. **peak_bandwidth 单链路 28GB/s**：utilization>100% 需在报告中注释；未改公式
4. **Device Frame 未强制自动化**：eval 仍要求 §3.4/§4 Frame 表，采集依赖 SHMEMI_PROF 手工步骤

## 建议后续增强（可选）

- performance_report §5 轮次总览与聊天累计表字段对齐代码生成
- timing-and-metrics-standard.md 作为唯一标准后，移除其余文件中重复的性能指标定义