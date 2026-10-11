# Debug playbook — INDEX (split into focused leaf files)

This file used to be a single ~1200-line playbook. It has been split into smaller topic files in this directory so that an agent only needs to read the section that matches the current failure, instead of loading the whole playbook into context.

**External references using `§X.Y` notation still resolve via this index** — find the section number below, then open the linked leaf file.

## Quick map: section → leaf file

| Section | Topic | File |
|---------|-------|------|
| §1–§7 | Opaque error / FFFFF / UNKNOWN / F21004 / AICore / error-code reference / stop conditions | [`error-codes.md`](error-codes.md) |
| §8 (8.1, 8.2, 8.3) | Example kernels — debug practice | [`examples-debug.md`](examples-debug.md) |
| §9.1 | JIT signature parsing (`from __future__ import annotations` breaks JIT) | [`jit-signature.md`](jit-signature.md) |
| §9.2 | Dynamic shapes, `pypto.loop`, symbolic indexing, view dimension matching | [`dynamic-shapes.md`](dynamic-shapes.md) |
| §9.4 | Comprehensive `pypto.view` guide (signature, padding, common mistakes) | [`pypto-view.md`](pypto-view.md) |
| §9.6 | SIM mode limitations | [`sim-mode.md`](sim-mode.md) |
| §9.7 | Tensor operations (matmul dim requirements, broadcasting, zeros, element-wise) | [`tensor-ops.md`](tensor-ops.md) |
| §9.8 | Common patterns (multi-session, tile constants, host precompute, reverse iteration) | [`common-patterns.md`](common-patterns.md) |
| §9.9 | Debug checklist | [`checklist-and-api.md`](checklist-and-api.md) |
| §9.10 | Testing strategy | [`checklist-and-api.md`](checklist-and-api.md) |
| §9.11 | Common error messages (quick table) | [`checklist-and-api.md`](checklist-and-api.md) |
| §9.12 | Key PyPTO API notes | [`checklist-and-api.md`](checklist-and-api.md) |
| §9.13 | Tensor shape DYNAMIC + `pypto.view` + `pypto.loop` pattern (avoid INT32_MAX overflow correctly; empty `[]` is NOT a production workaround) | [`jit-signature.md`](jit-signature.md) |
| §9.14 | Python operators inside PyPTO JIT (`*`, `+`, `.exp()`) | [`python-operators.md`](python-operators.md) |
| §9.15 | Tile shape configuration | [`tile-shapes.md`](tile-shapes.md) |
| §9.16 | Transpose operations | [`tensor-ops.md`](tensor-ops.md) |
| §9.17 | Pattern quick reference (verbose vs preferred) | [`common-patterns.md`](common-patterns.md) |
| §9.18 | Key takeaways | [`common-patterns.md`](common-patterns.md) |
| §9.19 | matmul API and tile shapes (transpose flags, vec+cube, assemble, reshape, `.T`, reduction alignment, K-dim mismatch, 5D views) | [`matmul.md`](matmul.md) |
| §9.20 | NPU kernel launch failures (delivered ops lessons) | [`npu-launch-failures.md`](npu-launch-failures.md) |
| §9.21 | Large-axis scan/cumulative & UB-exceeding reduction (`F40005` UB OOM; block + carry; element-wise scan anti-pattern) | [`scan-and-reduction.md`](scan-and-reduction.md) |

## Quick map: situation → leaf file

| Situation | Open |
|-----------|------|
| Opaque error code (`FFFFF`, `UNKNOWN`, `0x3FFFF`), silent failure, no detail | [`error-codes.md`](error-codes.md) |
| Tile-shape failure: `L0A/L0B/L0C/L1 size exceeded`, tile align / not aligned, "tile shape not set", `enable_split_k`, or `set_cube_tile_shapes` misuse | [`tile-shapes.md`](tile-shapes.md) |
| `F21004` / `REGISTER_COPY` invalid vector tile errors | [`error-codes.md`](error-codes.md) §4; then use [`tile-shapes.md`](tile-shapes.md) if the fix is tile configuration |
| About to write a `@pypto.frontend.jit` function | [`jit-signature.md`](jit-signature.md) |
| About to write `pypto.view` / `pypto.assemble` | [`pypto-view.md`](pypto-view.md) |
| About to write `pypto.matmul` | [`matmul.md`](matmul.md) |
| About to write `.sum()` / reduction | [`matmul.md`](matmul.md) (32-byte alignment + matmul workaround) |
| `F40005` UB alloc exceeds `MEM_UB` on a scan/cumulative or large-axis reduction, or about to scan an axis element-by-element | [`scan-and-reduction.md`](scan-and-reduction.md) (block + carry) |
| About to write `pypto.cumsum` / `pypto.cumprod` (scan over a large axis) | [`scan-and-reduction.md`](scan-and-reduction.md) (block + carry) |
| Working with dynamic shapes / `pypto.loop` | [`dynamic-shapes.md`](dynamic-shapes.md) |
| Element-wise ops inside JIT | [`python-operators.md`](python-operators.md) |
| Configuring tile shapes | [`tile-shapes.md`](tile-shapes.md) + [`matmul.md`](matmul.md) (vec+cube) |
| SIM mode produces garbage / fails but NPU works | [`sim-mode.md`](sim-mode.md) |
| NPU launch fails with `Errcode: FFFFFF! launch aicpu` | [`npu-launch-failures.md`](npu-launch-failures.md) |
| Looking up an error message in a quick table | [`checklist-and-api.md`](checklist-and-api.md) §9.11 |
| Need an API quick reference | [`checklist-and-api.md`](checklist-and-api.md) §9.12 |
| Debugging an example kernel from `examples/` | [`examples-debug.md`](examples-debug.md) |

---

**Note:** The agent SHOULD NOT read this whole index plus every leaf in one pass. Pick the leaf file that matches the current failure mode and read only that file. Cross-references between leaves are explicit (each leaf links the related ones).
