# Phase 1 — Project init

Locate or create the **ascend-kernel** project, then scaffold the operator directory.

## What an ascend-kernel project looks like

```
ascend-kernel/
├── build.sh                  # one-shot build → produces output/ascend_kernel*.whl
├── CMakeLists.txt
├── cmake/                    # DO NOT MODIFY
├── csrc/
│   ├── ops.h                 # operator declarations (registration point 1)
│   ├── register.cpp          # TORCH_LIBRARY schema + impl (registration point 2)
│   ├── CMakeLists.txt        # source lists (registration point 3)
│   ├── utils/                # DO NOT MODIFY
│   └── ops/
│       └── <op>/
│           ├── op_host/<op>.cpp
│           ├── op_kernel/<op>.cpp
│           ├── CMakeLists.txt
│           ├── design.md
│           └── test/
├── python/ascend_kernel/...  # python package wrapping the built .so
├── tests/
└── output/                   # build artifacts (.whl)
```

## Step 1 — Detect or create the project

Run the bundled detector:

```bash
bash scripts/detect_ascend_kernel_project.sh [search_root]
```

- If a project is found, use it.
- If none is found, copy the bundled template and make the build script executable:

```bash
cp -r templates/ascend-kernel <target_dir>/ascend-kernel
cd <target_dir>/ascend-kernel && chmod +x build.sh
```

## Step 2 — Scaffold `csrc/ops/<op>/`

Create the operator skeleton (placeholder files that later phases fill):

```
csrc/ops/<op>/
├── op_host/<op>.cpp          # placeholder
├── op_kernel/<op>.cpp        # placeholder
├── CMakeLists.txt            # per-op cmake (copy from a sibling op)
├── design.md                 # empty / template → filled in Phase 2
└── test/                     # created when test cases are generated
```

## Step 3 — Note the three registration points

These are updated in **Phase 4** (code-gen), but flag them now so nothing is missed:

| Point | File | Update |
|---|---|---|
| 1 | `csrc/ops.h` | function declaration in `namespace ascend_kernel` |
| 2 | `csrc/register.cpp` | `m.def(schema)` + `m.impl(name, TORCH_FN(...))` |
| 3 | `csrc/CMakeLists.txt` | host source in `OP_SRCS`; kernel source in `ascendc_library(...)` |

## Gate (must pass before Phase 2)

- [ ] Project root exists with `build.sh`, `CMakeLists.txt`, `csrc/`.
- [ ] `csrc/ops/<op>/` created with `op_host/<op>.cpp`, `op_kernel/<op>.cpp`,
      `CMakeLists.txt`, `design.md`.
- [ ] `build.sh` is executable.

## Anti-patterns (NEVER)

- NEVER modify files under `cmake/` or `csrc/utils/`.
- NEVER delete or overwrite other operators' registration entries.
- NEVER skip creating `design.md` — it is the input for code generation.
