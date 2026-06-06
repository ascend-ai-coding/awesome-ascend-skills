# Phase 0 — Environment and requirements

Resolve the build/run environment and collect operator requirements **before any
development action**. Hardcoding paths or guessing the environment is the most common
cause of downstream failures.

## Environment resolution

### CANN toolkit

1. `echo $ASCEND_HOME_PATH`.
2. If set → use it as `CANN_PATH`.
3. If unset → **MUST** ask the user for the CANN install path (do not search the
   filesystem to guess it).
4. Activate per shell command:

```bash
source ${CANN_PATH}/*/set_env.sh
```

### Conda environment

1. `echo $CONDA_DEFAULT_ENV`.
2. If non-empty and not `base` → use the current env.
3. Otherwise → **MUST** ask the user for the conda env name.
4. Activate per shell command:

```bash
conda activate <env_name>
```

> Re-`source` the CANN env and re-activate conda **before every shell command**.
> Skipping this leads to: (1) `build.sh` cannot find the AscendC compiler;
> (2) wrong `pip`/`python`; (3) `torch_npu` import errors when running tests.

### Quick sanity checks (optional)

```bash
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -c "import torch, torch_npu; print(torch.npu.is_available())"
```

## Requirements collection

| Item | Required | Default | Notes |
|---|---|---|---|
| Operator name | Yes | — | snake_case, e.g. `acosh`, `rms_norm` |
| Functional / math spec | Yes | — | formula or reference behavior |
| Supported dtypes | No | `float16, float32` | may add `bfloat16` |
| SoC / platform | No | `ascend910b` | obtained via platform API at runtime, never hardcoded in kernel |
| Input domain constraints | No | — | e.g. `x >= 1` for `acosh`; needed for test data |

## Gate (must pass before Phase 1)

- [ ] `CANN_PATH` resolved and `set_env.sh` sourceable.
- [ ] Conda env resolved and activatable.
- [ ] Operator name (snake_case) confirmed.
- [ ] Functional/math spec confirmed.

## Anti-patterns (NEVER)

- NEVER hardcode CANN or conda paths.
- NEVER search the filesystem to guess CANN/conda locations.
- NEVER start coding before the operator name and functional spec are confirmed.
