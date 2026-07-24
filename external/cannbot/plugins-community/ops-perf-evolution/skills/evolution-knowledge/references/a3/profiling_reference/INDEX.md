# Profiling Reference — Stage 2 LLM 诊断参考材料

> 来源：cannbot `ops/ops-profiling/references/` (引入版本：2026-05)
> 用途：Phase B 三阶段诊断管线的 Stage 2（LLM narrative）参考资料
> 关联：[knowledge-strategy-architecture-v3.2](../../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.4

## 文件说明

| 文件 | 用途 | 何时读 |
|---|---|---|
| `optimization_quickref.md` | 瓶颈判定标准 + 跨指标交叉诊断表 | Stage 2 LLM 给出 bottleneck_labels 前必读 |
| `csv_fields_reference.md` | msprof 输出的 8 个 CSV 字段完整定义 | 需要精确解读某指标含义时读 |

## 标准化 bottleneck_labels 词表

供 Stage 2 LLM 诊断输出使用，**必须从此表选**，超出词表的 labels 由 wm_ops.refine 校验拒收。

### 主标签（14 个，对齐 evolution-strategies/cards/SCHEMA.md）

| Label | 触发判定 | 典型来源指标 |
|---|---|---|
| `mte2_stall` | MTE2（Scalar→UB）停等明显 | `ai*_mte2_ratio` 最高 + `bw_usage_rate` 中等 |
| `mte3_stall` | MTE3（UB→GM）停等明显 | `ai*_mte3_ratio` 最高 |
| `tiling_imbalance` | 分核负载不均 | 各核 `aiv_time` 差异 >10% |
| `scalar_loading` | Scalar 加载瓶颈 | `ai*_scalar_ratio` 高 + tiling 数据加载 |
| `scalar_compute` | Scalar 计算瓶颈 | `ai*_scalar_ratio` >30% |
| `compute_bound` | Vector/Cube 计算密集 | `aiv_vec_ratio` 或 `aic_cube_ratio` >70% |
| `near_optimal` | 接近硬件极限 | 各 ratio < 50%，task duration ≈ 理论最长流水 |
| `no_overlap` | 无流水重叠 | MTE2 / VEC / CUBE 完全串行 |
| `partial_overlap` | 部分流水重叠 | 重叠比 10-30% |
| `undersize_transfer` | 搬运粒度过小 | `mte2_instructions` 高 + 单次搬运 < 16KB |
| `icache_miss` | 指令缓存未命中 | `icache_miss_rate` >5% |
| `bus_contention` | 总线竞争 | 多核同地址访问，各核 mte2 差异大 |
| `l2_cache_thrash` | L2 Cache 抖动 | `l2_total_hit_rate` < 50% |
| `ub_memory_pressure` | UB 内存压力 | UB 占用 >85% |

### 增量标签（Stage 1 facts 补充判定，4 个）

来自 cannbot quickref 的"真假 bound"区分：

| Label | 触发判定 | 区分自 |
|---|---|---|
| `fake_mte2_bound` | `mte2_ratio` 高（>60%）+ `bw_usage_rate` <70% | `mte2_stall`（前者是搬运结构问题，后者是带宽真饱和）|
| `fake_compute_bound` | `vec_ratio` 高 + `bank_cflt_ratio` >5% | `compute_bound`（前者是 conflict 放大耗时） |
| `bank_conflict` | `aiv_vec_total_cflt_ratio` >5% | 独立标签，常与 `compute_bound` 同时出现 |
| `db_not_effective` | MTE2/VEC 重叠 <5% + `BUFFER_NUM>1` | 独立标签，意味着 P1 已用但配置错 |

**总计 18 个 labels**，比单纯映射 12 类 bottleneck_type 更精细，能区分根因。

### 多 label 组合的典型场景

来自 cannbot `optimization_quickref.md` §10 交叉关联诊断：

| 现象组合（labels） | 根因 | 优先策略 |
|---|---|---|
| `compute_bound + bank_conflict` | Bank conflict 放大 vec 耗时 | 先解 bank（P65） |
| `mte2_stall + l2_cache_thrash` | L2 miss 导致搬运慢 | 设 CacheMode（P52） |
| `mte2_stall + undersize_transfer` | 单次搬运过小 | 增大 tile（P22 / 适应性 tiling） |
| `tiling_imbalance + scalar_compute` | 核数不合理 | P4 多核负载均衡 |
| `no_overlap + db_not_effective` | DoubleBuffer 配置错 | 检查 P1 Playbook Step 5 自检 |

## 与 Stage 2 LLM 诊断输出格式的对应

LLM 在 refine 时输出：

```yaml
diagnosis_text: |
  MTE2 busy 85% 但带宽利用 32%，bw_usage_rate 低说明不是真带宽饱和，
  根因是 scale 小块重复 DMA 搬运
bottleneck_labels: [fake_mte2_bound, undersize_transfer]
confidence: 0.75
```

wm_ops.refine 校验：
- `bottleneck_labels` ⊂ 18 个标签词表
- `confidence` ∈ [0, 1]
- `diagnosis_text` 非空且 ≥ 20 字符

校验失败 → 拒收 + 触发重做。

## 阈值判定权威

`csv_fields_reference.md` 给出每个指标的"达标/警告/严重"三档阈值。Stage 1 `extract_facts()` 输出原始数值；Stage 2 LLM 对照本目录文档给标签。

阈值不在脚本里硬编码，避免脆性——文档可以更新，脚本只看 `bw_usage_rate < 0.7` 这种带置信度的事实。
