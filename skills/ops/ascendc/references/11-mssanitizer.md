# Phase 11 — Memory check (mssanitizer, optional)

Detect operator memory problems with mssanitizer (CANN's memory-correctness suite):
illegal access, illegal free, memory leak, UB out-of-bounds, data race, uninitialized
memory, sync issues.

Run after Phase 4+ when verifying memory safety, or on any suspected memory error.

## Project type → mode

```
op_graph/ or op_host/op_api/ present?       → ops repo → C++ mode (GE IR / aclnn)
examples/test_geir_*.cpp present?           → C++ mode (GE IR submode)
examples/test_aclnn_*.cpp present?          → C++ mode (aclnn submode)
none of the above                           → custom-op repo → Python mode
```

The ascend-kernel custom-op project uses **Python mode**.

## Environment

```bash
source ${CANN_PATH}/*/set_env.sh
conda activate <env_name>
# mssanitizer needs ASCEND_HOME_PATH and CANN lib64 on LD_LIBRARY_PATH
```

Constraints from practice:
- `--check-device-heap` and `--check-cann-heap` are **mutually exclusive** → run memcheck
  twice.
- mssanitizer may drop `libhccl.so` from `LD_LIBRARY_PATH` (breaks `torch_npu` import) —
  the scripts re-prepend CANN lib64.
- **Empty log = no error**: mssanitizer only writes a log when it finds an error; a 0-byte
  log means that check passed.

## Bundled scripts (`../scripts/`)

| Script | Purpose | Mode |
|---|---|---|
| `gen_test_script.py` | generate a Python test script for the operator | Python |
| `run_mssanitizer.sh` | run all 5 checks + summary | Python |
| `run_mssanitizer_geir.sh` | build + check + summary (GE IR / aclnn) | C++ |
| `parse_mssanitizer_log.py` | parse memcheck logs into a report | both |

## Python mode flow

```bash
# 1. generate test
python3 ../scripts/gen_test_script.py \
    --operator <op> --fallback <torch.nn.functional fn> \
    --dtypes float16 float32 \
    --output <project>/mssanitizer_test/<op>_mssanitizer_test.py
# 2. run all checks (memcheck device-heap, memcheck cann-heap, racecheck, initcheck, synccheck) + summary
bash ../scripts/run_mssanitizer.sh <project>/mssanitizer_test/<op>_mssanitizer_test.py <cann_root>
```

`--fallback` keeps the NPU path running (and monitored) when the custom op is not
registered.

## Detection tools

| Tool | Flag | Detects |
|---|---|---|
| memcheck | `-t memcheck` | illegal access/free, leak, UB out-of-bounds |
| racecheck | `-t racecheck` | multi-core data race |
| initcheck | `-t initcheck` | uninitialized memory read |
| synccheck | `-t synccheck` | sync errors |

## Severity

| Severity | Types | Priority |
|---|---|---|
| critical | illegal_write / illegal_read / illegal_free | fix now |
| medium | memory_leak | fix soon |
| minor | uninitialized memory | recommended |

## Common causes

- **DataCopy arg**: `DataCopy(dst, src, count)` — `count` is **element count**, not bytes
  (`* sizeof(T)` is wrong) → illegal write/read.
- **Alloc/Free mismatch**: every `AllocTensor` needs a `FreeTensor`; `EnQue`/`DeQue`
  paired → memory leak.
- **UB buffer too small**: `InitBuffer` must account for temp space and 32B alignment → UB
  out-of-bounds.

## Tips

- Compile the operator with `-g -O0` for precise source locations (else `<unknown>:0`).
- mssanitizer is 10–100x slower — debug only.
- Some issues are intermittent — run multiple times.
- Fix illegal access before leaks; re-run to confirm.
