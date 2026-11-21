# Profiler field reference

Field meanings for captures produced by [performance collection](performance-collection.md).
Current values, units and missing columns always come from the installed CANN version;
treat the tables below as a reading aid, not a capability promise.

## Application-level `op_summary_*.csv`

Each `PROF_<Metric>/.../op_summary_<Metric>.csv` holds one row per profiled task.
Names and duration columns are matched case-sensitively by the collector; the keys
below are the ones it accepts.

| Group | Main columns | Read for |
| --- | --- | --- |
| `PipeUtilization` | `aic_*_ratio`, `aiv_*_ratio`, `aic_time(us)`, `aiv_time(us)` | Dominant pipe, scalar/fixpipe share |
| `ArithmeticUtilization` | `aic_cube_ratio`, `aiv_vec_ratio`, `*_fops`, `*_instr_number` | Instruction mix and useful compute |
| `Memory` | `*_main_mem_*_bw(GB/s)`, `*_mte2_ratio`, `*_mte3_ratio` | Main-memory and transfer pressure |
| `MemoryL0` | `aic_l0a_*`, `aic_l0b_*`, `aic_l0c_*` | Cube-local bandwidth |
| `MemoryUB` | `aiv_ub_*_bw_*` | UB read/write pressure |
| `L2Cache` | `ai*_read_hit_rate(%)`, `ai*_total_hit_rate(%)` | Cache reuse |
| `ResourceConflictRatio` | `aiv_vec_*_cflt_ratio`, `ai*_mte*_wait_ratio` | Bank, bank-group and wait stalls |

`aicore_time(us)` on the `PipeUtilization` row is what the per-core conversion needs.
Byte fields such as `read_main_memory_datas(KB)` are absent on A5 / CANN 9.2.0; only
bandwidth fields are available there.

## Per-core samples from `aicore.db`

Sample-based runs write two databases. Cube kernels use `aicore.db`
(`AICoreOriginalData.task_cyc`); Vector kernels use `ai_vector_core.db`
(`EventCount.task_cyc`). The family is resolved from `Task Type` and from which time
field is populated: a row with `VECTOR` in its task type, or with `aiv_time(us)` set and
`aicore_time(us)` unset, uses the vector database. Cycles and reference time always come
from the same family; when that database is absent, `per_core` stays `null` with a reason
instead of mixing the other database. When neither the task type nor a populated time
field identifies the family, the documented cube-then-vector fallback applies,
`coverage.notes` records `family_unresolved`, and the reference time may be the task
duration.

```sql
-- Cube
SELECT coreid, SUM(CAST(task_cyc AS INTEGER)) FROM AICoreOriginalData
WHERE CAST(task_cyc AS INTEGER)>0 GROUP BY coreid
-- Vector
SELECT coreid, SUM(task_cyc) FROM EventCount WHERE task_cyc>0 GROUP BY coreid
```

Use `aicore_time(us)` for Cube and `aiv_time(us)` for Vector as the reference time.
Frequency inference and conversion:

```
ns_per_cycle = core_time_us * 1000 / max(task_cyc)
freq_ghz     = 1 / ns_per_cycle
time_us      = task_cyc * ns_per_cycle / 1000
imbalance_%  = (max(time_us) - min(time_us)) / max(time_us) * 100
```

Record this as a derived, process-level sample. It does not prove that the cycles
belong to one `Op Name`.

## `OPPROF_*` files (msprof op / msopprof)

Real collectors write one directory per operator and append a capture timestamp to each
file name, for example `OPPROF_<timestamp>_XXX/<kernel>/0/OpBasicInfo_<timestamp>.csv`.
The table lists the file stems; a flat `OpBasicInfo.csv` from an older collector has the
same columns.

| File | Content |
| --- | --- |
| `OpBasicInfo.csv` | `Op Name`, `Op Type`, `Task Duration(us)`, `Block Dim`, `Current Freq`, `Rated Freq` |
| `PipeUtilization.csv` | One row per `block_id`/`sub_block_id`; `aic_time(us)`, `aiv_time(us)`, `*_total_cycles`, pipe ratios |
| `ArithmeticUtilization.csv` | Cube/Vector instruction ratios, FOPS and instruction counts |
| `Memory.csv` | Main-memory and L1 bandwidth, MTE instruction counts |
| `MemoryL0.csv` | L0A/L0B/L0C bandwidth |
| `MemoryUB.csv` | UB bandwidth from Vector and Scalar |
| `L2Cache.csv` | Hit and miss counters, read/write hit rates |
| `ResourceConflictRatio.csv` | Bank, bank-group, resource and MTE conflict shares, wait ratios |

Per-core time in `OPPROF` uses each row's `aic_time(us)` or `aiv_time(us)`; `block_id`
and `sub_block_id` identify the core position. Frequency comes from
`OpBasicInfo.csv`; `Current < Rated` indicates DVFS.

## Derived quantities

- Pipe and conflict ratios overlap; they are not additive and cannot be summed into
  a time budget.
- `Task Duration(us)` includes scheduling, execution and response. It is not a DMA
  floor and does not include host gaps.
- Bandwidth fields describe active bandwidth; do not integrate them as total DRAM
  traffic.
- Missing columns and missing samples stay missing. Never substitute zero.

## Compact diagnostic tiers

These tiers are a reading aid for ordering follow-up work. They are not an acceptance
gate or a completion criterion. Use the [optimization workflow](../workflows/optimization.md)
to choose comparisons and assess the stopping evidence.

| Reading | Low | Medium | High | Next check |
| --- | --- | --- | --- | --- |
| MTE2 utilization | <50% | 50–80% | ≥80% | Transfer extent, alignment, overlap |
| MAC utilization | <75% | 75–90% | ≥90% | Tile shape, K tail, L0 residency |
| VEC utilization | <70% | 70–80% | ≥80% | VF instruction mix, layout, masks |
| FIXPIPE utilization | <30% | 30–40% | ≥40% | Output alignment, drain overlap |
| MTE3 utilization | <30% | 30–50% | ≥50% | Output layout, store width |
| SCALAR utilization | <60% | 60–80% | ≥80% | Address/mask/descriptor work |
| MTE1 utilization | <50% | 50–80% | ≥80% | L1 residency, panel shape |
| Core imbalance | <5% | 5–15% | ≥15% | Block assignment, tail wave |
| L2 total hit rate | <50% | 50–80% | >80% | Reuse distance, working set |
| Vector conflict | <5% | 5–15% | ≥15% | Bank stride, UB offsets, scheduling |

A ratio above a tier says where to look next, not what the fix is.

## Metric to collector

| Question | Collector |
| --- | --- |
| Is one implementation faster than another? | `profile_kernel.py` AB/BA |
| Which pipe dominates a lowered kernel? | `msprof` `PipeUtilization` or `msopprof` `PipeUtilization.csv` |
| Is the work spread across cores? | `msprof` sample-based `aicore.db` |
| Which core is slow and why? | `msopprof` per-row `PipeUtilization.csv` |
| Is DVFS active? | `msopprof` `OpBasicInfo.csv` frequency columns |
| Are there bank or bank-group conflicts? | `ResourceConflictRatio.csv` |
| What are the exact lowered names? | `perf_collect.py discover` |
