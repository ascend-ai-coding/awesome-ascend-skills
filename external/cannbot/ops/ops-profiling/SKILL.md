---
name: external-cannbot-ops-ops-profiling
description: NPU 性能采集与分析，用于采集算子性能数据、定位性能瓶颈并给出优化建议。当用户在算子开发过程中提到"上板性能"、"算子性能测试"、"硬件性能验证"、"NPU性能采集"、"NPU
  profiling"等场景时触发。
original-name: ops-profiling
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-06-14'
synced-commit: f2b0db65abdced502b1cd1956e2e48e6e0197add
license: UNKNOWN
---

# 上板性能采集与调优

在真实 NPU 上采集算子性能数据，系统化解读指标文件，判定性能是否达标，定位瓶颈类型，并给出可操作的优化建议。


| 工具 | 流程文档 |
|------|----------|
| **`msprof op`** | [`references/msprof-op-guide.md`](references/msprof-op-guide.md) |
| **`msprof`** | [`references/msprof-guide.md`](references/msprof-guide.md) |

---

## 适用场景

| 场景 | 说明 |
|------|------|
| 算子开发完成后的性能验收 | 确认算子达到预期性能水平 |
| 性能问题定位 | 通过指标精确定位瓶颈 |
| 优化效果验证 | 对比优化前后的归档数据 |
| Agent team 测试阶段 | tester / developer 调用，自动化性能分析 |

---

## 选用哪个工具：决策树

按顺序执行，命中即停止。

1. **用户显式指定工具**
   - 指定 `msprof op` / msopprof → 加载 [`references/msprof-op-guide.md`](references/msprof-op-guide.md)；若环境缺 `msopprof`，不要硬跑，向用户说明并提示安装或换用另一工具。
   - 指定 `msprof` → 加载 [`references/msprof-guide.md`](references/msprof-guide.md)。

2. **用户未指定** — 探测环境：
   - 仅 `msopprof` 可用 → [`references/msprof-op-guide.md`](references/msprof-op-guide.md)
   - 仅 `msprof` 可用 → [`references/msprof-guide.md`](references/msprof-guide.md)
   - 两者皆可用 → 须向用户确认或按项目约定选用其一；不得在未约定时自行替用户拍板。
   - 两者皆不可用 → 报错，提示检查 CANN / `ASCEND_HOME` 安装

3. **上游流程强制指定采集工具** → 以该指定为准（不适用本条环境探测逻辑）。

---

## 参考资源

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| [`references/msprof-op-guide.md`](references/msprof-op-guide.md) | `msprof op`：构建 / 采集 / 归档 / 判定 / 瓶颈 / 回归 | 选用 `msprof op` 时 |
| [`references/msprof-guide.md`](references/msprof-guide.md) | `msprof`：构建 / 采集 / 归档 / **主 Bound 判定** / 瓶颈 | 选用 `msprof` 时 |
| [`references/csv_fields_reference.md`](references/csv_fields_reference.md) | CSV 字段定义与阈值 | 理解指标含义时 |
| [`references/optimization_quickref.md`](references/optimization_quickref.md) | 瓶颈类型与优化方法 | 定位瓶颈后 |
| `scripts/perf_summary.py` | `msprof op` 归档与摘要 | 对应 guide Step 3 |
| `scripts/msprof_profile_run.sh` | `msprof` 一键采集 | 对应 guide Step 2 |
| `scripts/msprof_perf_summary.py` | `msprof` 归档与逐核摘要 | 对应 guide Step 3 |
