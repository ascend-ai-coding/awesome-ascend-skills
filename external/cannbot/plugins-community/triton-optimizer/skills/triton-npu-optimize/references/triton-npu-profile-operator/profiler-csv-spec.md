# Ascend Profiler CSV Files

## Layout

`msprof` generates a `PROF_*` directory. The timing CSV files that matter for operator summaries live under:

```text
{PROF_XXX}/
└── mindstudio_profiler_output/
    ├── op_statistic_<timestamp>.csv
    └── op_summary_<timestamp>.csv
```

## `op_statistic`

This is the primary per-operator summary file. Expected columns commonly include:

```text
Device_id, OP Type, Core Type, Count, Total Time(us), Min Time(us), Avg Time(us), Max Time(us), Ratio(%)
```

Use it to identify the hottest operators and to report high-level timing metrics.

## `op_summary`

This file may be much larger than `op_statistic`. Read it carefully and prefer streaming instead of eager whole-file loading.

The exact columns may vary by profiler version. When available, use:

- an operator-identifying column such as `Op Name`, `OP Type`, or `Op Type`
- a duration column such as `Task Duration(us)`, `Task Duration`, or `Duration(us)`

Use `op_summary` as a cross-check for the target operator:

- count matching rows
- sum matching task durations
- compute min, average, and max duration if a numeric duration column exists

## Operator selection

- Prefer an explicit user-provided operator name.
- If none is provided, infer the target operator from the largest `Total Time(us)` row in `op_statistic`.
- When inference is used, say so in the report.
