# External performance collection: msprof and msprof op

Use this page when a task needs application-level timing, the two metric groups that
`torch_npu.profiler` does not expose here, or per-core samples. For paired implementation
comparisons on the default path, keep using [native-task timing](../verification/native-timing.md)
and `scripts/profile_kernel.py`; the two paths do not replace each other.

| Path | Entry | Data | Use |
| --- | --- | --- | --- |
| In-process | `scripts/profile_kernel.py` | `kernel_details.csv` task sums, 5 metric groups, AB/BA pairs | Default diagnosis and comparisons |
| Application level | `scripts/perf_collect.py run --backend msprof` | 7 `op_summary_*` groups plus sample-based `aicore.db` | Per-core load and missing metric groups |
| Operator level | `scripts/perf_collect.py run --backend msopprof-op` | The requested `OPPROF_*` CSVs and per-core `PipeUtilization.csv` | One lowered kernel, replay diagnostics |

<!-- kernel-query: {"id":"performance-counters","aliases":["bank conflict profiling","UB bank conflict counters","UB bank 冲突采集","ResourceConflictRatio"],"term_groups":[["bank conflict","bank group","bank 冲突"],["profiling","profile","profiler","collector","counter","counters","metric","metrics","采集","指标"]],"when":"Discover a documented collector or metric; source discovery does not establish installed profiler capability or measured conflicts.","missing_facts":["Identify the target device, installed collector/version, native kernel symbol and requested metric; keep documentation, capture support and measured results separate."]} -->

## Check availability

```bash
python scripts/perf_collect.py probe --format json
```

`probe` reports `ascend_home`, an optional `arch`, and each collector's `path`,
`version` and `available`. It resolves `PATH` first, then
`$ASCEND_HOME_PATH/tools/profiler/bin/msprof` and
`$ASCEND_HOME_PATH/tools/msopprof/bin/msopprof`. A missing binary is reported, not
guessed. A successful `--version` establishes discovery only, not a working capture.
The probe payload uses schema `pypto-perf-probe/1`; capture metadata and evidence use
`pypto-perf-collection/1` and `pypto-perf-evidence/1` respectively.

## Application-level msprof

```bash
python scripts/perf_collect.py run --backend msprof \
  --app "python3 test_op.py --case-id 0" \
  --metrics all --warm-up 3 --output ./msprof_output
```

The runner warms up the application, then records each group separately:
`PipeUtilization`, `ArithmeticUtilization`, `Memory`, `MemoryL0`, `MemoryUB`,
`L2Cache`, `ResourceConflictRatio`. One additional `--aic-mode=sample-based` run
writes `PROF_Sample/.../sqlite/aicore.db` for per-core cycles. `--timeout` and `--no-l2-clean` are accepted; L2 is cleaned before each measured run by
default, using `TILE_FWK_DEVICE_ID` when set and logical device 0 otherwise. Runners that
need another device protocol must receive it through `--app` or the environment.
Collector output streams to `msprof.log` in each stage directory, so a long capture is
not buffered in memory.

Output:

```
msprof_output/PROF_GROUP_<timestamp>_<pid>/
  collection.json
  PROF_<Metric>/PROF_*/mindstudio_profiler_output/op_summary_*.csv
  PROF_<Metric>/msprof.log
  PROF_Sample/PROF_*/device_*/sqlite/aicore.db
```

The application typically launches many kernels, including input generation and
accuracy checks. Before parsing, list the exact lowered names from the capture:

```bash
python scripts/perf_collect.py discover --capture ./msprof_output --format json
```

Choose the name from the discovery result; do not select a fuzzy grep hit or the
largest task duration. A name that matches several rows is ambiguous and fails.

```bash
python scripts/perf_collect.py summarize --capture ./msprof_output \
  --op-name '<exact Op Name>' --out evidence.json
```

## Operator-level msprof op / msopprof

```bash
python scripts/perf_collect.py run --backend msopprof-op --app ./runner \
  --kernel-name '*stage2*' --aic-metrics PipeUtilization --output ./msprof_op
```

Pass the groups to request as one comma-separated `--aic-metrics` value; the default is
`PipeUtilization`. Groups that were not requested are reported under
`coverage.not_requested`, never as missing, so a deliberate single-group capture is not
mistaken for an incomplete one. Without
`--kernel-name`, only the first scheduled operator is captured. PyPTO-Pro
kernels are mangled symbols; use a wildcard such as `*stage2*` or a `_Z...` prefix.
The capture nests one directory per operator and suffixes every CSV with the capture
timestamp:

```
msprof_op/OPPROF_<timestamp>_XXX/
  <kernel>/0/OpBasicInfo_<timestamp>.csv
  <kernel>/0/PipeUtilization_<timestamp>.csv
  collection.json
```

Flat `OPPROF_*/OpBasicInfo.csv` captures from older collectors remain accepted. A
capture that holds several operators, for example `--launch-count 2` without
`--kernel-name`, must be summarized with `--op-name`; otherwise `summarize` fails and
lists the candidate directories. Collection replays the operator, so replay timing can
differ from application timing; label it as replay.

Use the capture directory printed by `run` and the exact operator name from `discover`:

```bash
CAPTURE_DIR=$(python scripts/perf_collect.py run --backend msopprof-op --app ./runner \
  --kernel-name '*stage2*' --aic-metrics PipeUtilization --output ./msprof_op)
OP_NAME='<exact Op Name from discover>'

python scripts/perf_collect.py summarize \
  --capture "$CAPTURE_DIR" \
  --backend auto --op-name "$OP_NAME" --out evidence.json
```

`--capture` takes a directory or a path inside one, never a glob: a quoted `OPPROF_*`
is passed to the CLI literally and fails as a missing directory, while an unquoted one
expands into several arguments. `auto` uses the backend recorded in `collection.json`, so
a capture taken through the `msprof op` fallback summarizes without relabelling it.
`--backend` also accepts `msprof-op` explicitly, and rejects a value that contradicts the
recorded backend.

`--op-name` accepts the `Op Name` value from `OpBasicInfo.csv` or the operator
directory name, which is the C++-mangled symbol used by `--kernel-name`. Passing the
operator directory itself also works. Without `OpBasicInfo.csv` only the directory name
identifies the operator, and a non-matching `--op-name` still fails even for a single
capture.

## Collection metadata

Every capture writes `collection.json` with the collector that actually ran
(`backend`, path, version), the invoked argv, the target device, the requested warm-up
and launch counts, and the return code of each stage. When the binary exposes no
version flag, the installed package metadata (`share/info/<collector>/version.info`)
supplies it; otherwise the version stays `null`. The toolkit version is recorded
separately. `summarize` reads the capture
identity from the file owned by the selected capture only. When several captures share one
output root, a neighbouring `collection.json` is never borrowed: selecting an operator
through the common root and through its own capture directory must give the same
identity, warm-up count and metrics. When the file is missing, as with an older capture,
a hand-written capture or a directory collected without this runner, `identity.source` is
`unknown` and the identity fields stay `null`; an ambiguous selection fails with exit
code 2 and lists the candidates. The current parsing environment is reported separately
under `parse_environment` and is not attributed to the capture.

A recorded failure stays visible in the evidence. `collection.status` is `complete` when
metadata exists without an error, `failed` when the metadata carries a stage error, and
`unknown` when there is no metadata.

## Evidence output

`summarize` writes one `pypto-perf-evidence/1` JSON object:

| Field | Content |
| --- | --- |
| `collector` | `backend` (`msprof`, `msopprof-op` or the `msprof-op` fallback) plus the path and version recorded at capture time |
| `identity` | capture-time arch/soc/CANN, toolkit version, exact `op_name`, optional `kernel_name`, and `source` (`collection` or `unknown`) |
| `collection` | `status` (`complete`, `failed` or `unknown`), the recorded `error` and the stages that completed |
| `parse_environment` | arch/soc/CANN/`ascend_home` of the process that ran `summarize`; never the capture identity |
| `scope` | collected groups, warm-up, measured runs, `single_target` or `unverified` |
| `metrics.task_duration_us` | selected task duration with its sample count; `value` is `null` and `samples` is `0` when no row carries a duration |
| `metrics.groups` | raw rows or row per metric with source path |
| `metrics.per_core` | per-core cycles/time plus frequency and imbalance, or `null`; `coverage.notes` records `family_unresolved` when the family could not be derived |
| `coverage` | requested groups, collected groups, requested groups that are missing, groups that were not requested, and reason notes |

`summarize` exits non-zero when `collection.status` is `failed`; the parsed metrics are
still written so the capture can be diagnosed. Coverage and collection status are
separate: a failed capture can have complete metrics.

Warm-up and measured-run counts come from `collection.json`; without it they are
`null` rather than an assumed `1`. A requested metric that was not collected stays in
`coverage.missing`; a group that was never requested is listed in
`coverage.not_requested` instead, so it never looks like a failed capture. When the
metadata is absent both fields stay `null` and every absent group is reported as missing.
An unattributable per-core sample is `null` with a reason. Never fill missing evidence with
zero. Keep the raw capture when a number will be quoted; the JSON is a digest, not a
replacement.

Per-core samples are process-level. Cube kernels use `aicore.db` with `aicore_time(us)`;
Vector kernels use `ai_vector_core.db` with `aiv_time(us)`. The family comes from
`Task Type` and from which time field is populated (a `MIX_AIV` row without `VECTOR` in
its task type still uses the vector time). Cycles and reference time always come from the
same family; when that database is absent, `per_core` is `null` with a reason instead of
mixing the other database. Use them for load-spread
diagnosis only; do not claim they belong to one target operator unless the current
CANN schema proves the binding.

## Failure semantics

| Situation | Behavior |
| --- | --- |
| Collector not found | `probe` reports unavailable; `run` exits non-zero |
| Warm-up application exits non-zero | Error names the warm-up stage and return code; nothing is measured |
| A metric or sample stage exits non-zero | Error names the stage and return code; the capture is kept, and application-level collection writes `collection.json` with the error and the stages that completed |
| The collector exits zero but its log reports the application raised | Error names the stage and the reported reason; the log marker or a Python traceback fails the run, so a collector that hides the application exit cannot produce a successful run |
| Collector output contains non-UTF-8 bytes | Decoded with replacement, never raising `UnicodeDecodeError`; the stage is judged by its return code and failure markers |
| The collector exits zero but creates an empty capture (no OPPROF metric CSV) | Error; a `--kernel-name` that matched nothing is reported instead of returning a capture |
| `OpBasicInfo*.csv` is absent but other metric CSVs exist | Not an error: `op_name` comes from the operator directory, other CSVs still parse, `task_duration_us` may be `null`, and `coverage.notes` records `no_op_basic_info`. A `--op-name` that does not match the operator directory still fails |
| L2 flush fails | `coverage.notes` records `l2_clean_failed`, so the run is not reported as clean |
| A stage times out | The stage's whole process group is terminated (SIGTERM, then SIGKILL after a grace period) before the timeout error is raised, so an application the collector left running cannot keep using the device |
| `collection.json` records a failed stage | `summarize` writes the evidence with `collection.status=failed` and the error, and exits non-zero |
| Several captures share one output root | Metadata is read only from the capture that owns the selected operator; selecting an operator whose capture has none keeps the identity `unknown` instead of borrowing a sibling's |
| Capture holds several operators and `--op-name` is absent | Error lists the candidate operator directories |
| `discover` finds no `op_summary_*.csv`, `OpBasicInfo*.csv` or OPPROF metric CSV | Error reports the capture path; an OPPROF capture without `OpBasicInfo` lists the operator directories and reports `0.0` for an unknown duration |
| `--op-name` missing from the capture | Error lists bounded candidates |
| `--op-name` matches several rows | Error; isolate one measured launch |
| `--kernel-name` misses or matches several operators | Error; refine the mangled wildcard |
| The selected row's duration is missing, non-finite or non-positive | `summarize` errors instead of substituting zero; `discover` omits such application-level rows but still lists an `OpBasicInfo`-less operator directory with `0.0` |
| Several timestamped CSVs for one metric in the same directory | Error; keep one capture per directory |
| The resolved family's database is absent, or `task_cyc` is empty | `per_core` is `null` and the reason is recorded; the other family's database is never mixed in |

## Collector traps

Trace directory names containing `=` are rejected by the collector. Exported `Name`/`Type`
values use C++-mangled symbols rather than `@pl.jit(name=)`; factory variants that share an
inner function name can also share a symbol. Use separate captures or unambiguous variant
names, and do not split rows by the requested JIT label. The official performance guide in
the matching PyPTO source checkout owns collector metric and output-file descriptions.

## Measurement boundaries

- `Task Duration(us)`, `Duration(us)`, replay timing, `torch.npu.Event` intervals and
  wall time are different measurements. Do not convert or mix them in one comparison.
- Pipe ratios are overlapping shares, not additive time slices.
- Per-core time is derived from cycles and an inferred frequency; record it as derived.
- Whether a change is faster is still decided by the paired AB/BA comparison in
  [native-task timing](../verification/native-timing.md), not by one msprof capture.

## Optional analysis resources

Case manifests, seeded inputs, multi-device batches, cross-round comparisons and
diagnostic digests are documented in the repository's
[performance-analysis resources](https://gitcode.com/cann/pypto-gym/tree/master/cannbot-skills/ops/pypto-pro-op-perf-tune/references/perf-analysis/).
Those resources and their companion scripts are separate from this plugin's installation.
Timeline, Occupancy and Roofline block capture additionally use the optional external
[`ops-profiling` skill](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ops-profiling/SKILL.md);
follow the resource guide for its dependencies and explicit paths.

For A5 SKU, units and nominal-rate calibration, use the bundled
[A5 roofline guide](../diagnostics/a5-roofline.md). A diagnostic digest does not establish
a hardware ceiling or a measured optimization benefit.
