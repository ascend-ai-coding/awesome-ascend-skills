# API examples: scope and coverage contract

The basic-example plan is complete. Current directories and API usage are computed by `python scripts/check.py coverage` / `surface`; batch progress, historical ST call counts and third-party implementation coverage are not maintained.

## 1. Scope

The basic library covers data movement, VF, SIMT, supporting APIs and Fixpipe parameters. Cube/GEMM belongs to [matmul examples](../../examples/matmul/); measured boundaries are maintained by family in the [API findings](../index.md#api-topics). Each example's docstring owns its inputs, reference and limits.

Examples follow the same [Vector authoring policy](../../SKILL.md#constraints-and-workspace-policy): VF computation, with only `sort32` / `mrgsort` / `mrgsort2` permitted as tile compute. Teaching is not an exception. The sort family should be added for actual needs; mask state, padding and data movement retain their native APIs. The policy does not establish a universal speed advantage; measure each complete implementation.

**API-name coverage is not parameter coverage.** Fixpipe parameters on move/store/insert require separate combination tests. The following contract assigns selected VF APIs to examples; it does not establish validation of all public APIs, dtypes or modes.

## 7. Example organization

### 7.3. Writing-style declaration

The first kernel.py docstring line declares `写法：语法糖 / 原始 / 混合`; do not add another directory layer for style. `writing-style-declaration` enforces this convention; see [tools](checks.md) for the current default scope.

**`写法：原始` declares mechanism teaching, not a shipping recommendation.** Shipping forms follow skill policy: VF for vector compute, `pl.gemm`/`gemm_mx` first for matmul. Manual-chain examples state their teaching reason and shipping guidance beside the declaration, so an agent landing there sees the warning locally. Observed example-copying behavior outweighed a routing sentence, which is why this guidance belongs in the copied file.

Do not duplicate teaching prose from metadata.json or commit generated reverse API indexes; docstrings and `surface` already provide entry points. CPU references are independent of the compiler, and negative cases that damage device context require separate processes.

## 8. VF coverage contract

### 八点二、81 项逐项

The exact heading above is a retained machine-readable marker. It and the table's first three columns are consumed by `scripts/_checks/coverage.py`. Quote example-directory names in backticks and separate multiple examples with `+`; descriptions may follow each name, and code spans in those descriptions are not example names. Rerun coverage after assignment changes.

The checker counts APIs and verifies each assigned example actually calls its API. Missing example directories are reported separately; rows without recognizable example names fail. This establishes neither numerics nor hardware support.

| # | `vf.` API | Examples |
| --- | --- | --- |
| 1 | `create_mask` | `vf_entry` |
| 2 | `update_mask` | `vf_entry` |
| 3 | `full` | `vf_entry` |
| 4 | `move` | `vf_entry` |
| 5 | `load` | `vf_entry` |
| 6 | `store` | `vf_entry` |
| 7 | `load_align` | `vf_align` (distribution modes)+ `vf_align_widths` (width binding: UNPK widening / E2B) |
| 8 | `store_align` | `vf_align` (distribution modes)+ `vf_align_widths` (width binding: direct PACK_B32 / PACK4_B32 stores and PACK traps) |
| 9 | `load_unalign_init` | `vf_unalign` |
| 10 | `load_unalign_pre` | `vf_unalign` |
| 11 | `load_unalign` | `vf_unalign` |
| 12 | `store_unalign` | `vf_unalign` |
| 13 | `store_unalign_post` | `vf_unalign` |
| 14 | `squeeze_store_unalign` | `vf_unalign` |
| 15 | `squeeze_store_unalign_post` | `vf_unalign` |
| 16 | `unalign_reg_for_store` | `vf_unalign` |
| 17 | `add` | `vf_arith` |
| 18 | `sub` | `vf_arith` |
| 19 | `mul` | `vf_arith` |
| 20 | `div` | `vf_arith` |
| 21 | `muls` | `vf_arith` |
| 22 | `adds` | `vf_arith` |
| 23 | `mins` | `vf_arith` |
| 24 | `maxs` | `vf_arith` |
| 25 | `mull` | `vf_arith` |
| 26 | `addc` | `vf_arith` |
| 27 | `subc` | `vf_arith` |
| 28 | `abs_sub` | `vf_arith` |
| 29 | `axpy` | `vf_arith` |
| 30 | `min` | `vf_arith` |
| 31 | `max` | `vf_arith` |
| 32 | `relu` | `vf_activation` |
| 33 | `neg` | `vf_activation` |
| 34 | `prelu` | `vf_activation` |
| 35 | `leaky_relu` | `vf_activation` |
| 36 | `mul_add_dst` | `vf_activation` |
| 37 | `mul_dst_add` | `vf_activation` |
| 38 | `reduce_sum` | `vf_reduce` |
| 39 | `reduce_max` | `vf_reduce` |
| 40 | `reduce_min` | `vf_reduce` |
| 41 | `pair_reduce_sum` | `vf_reduce` |
| 42 | `ln` | `vf_transcendental` |
| 43 | `log` | `vf_transcendental` |
| 44 | `log2` | `vf_transcendental` |
| 45 | `log10` | `vf_transcendental` |
| 46 | `exp` | `vf_transcendental` |
| 47 | `exp_sub` | `vf_transcendental` |
| 48 | `sqrt` | `vf_transcendental` |
| 49 | `abs` | `vf_transcendental` |
| 50 | `eq` | `vf_predicate` |
| 51 | `ne` | `vf_predicate` |
| 52 | `lt` | `vf_predicate` |
| 53 | `gt` | `vf_predicate` |
| 54 | `le` | `vf_predicate` |
| 55 | `ge` | `vf_predicate` |
| 56 | `select` | `vf_predicate` |
| 57 | `not_` | `vf_predicate` |
| 58 | `get_mask_spr` | `vf_predicate` |
| 59 | `mask_gen_with_reg_tensor` | `vf_predicate` |
| 60 | `interleave` | `vf_lane_layout` |
| 61 | `de_interleave` | `vf_lane_layout` |
| 62 | `squeeze` | `vf_lane_layout` |
| 63 | `unsqueeze` | `vf_lane_layout` |
| 64 | `pack` | `vf_lane_layout` |
| 65 | `unpack` | `vf_lane_layout` |
| 66 | `truncate` | `vf_lane_layout` |
| 67 | `astype` | `vf_lane_layout` |
| 68 | `bit_cast` | `vf_lane_layout` |
| 69 | `muls_cast` | `vf_lane_layout` |
| 70 | `gather` | `vf_gather_scatter` |
| 71 | `scatter` | `vf_gather_scatter` |
| 72 | `arange` | `vf_gather_scatter` |
| 73 | `create_addr_reg` | `vf_gather_scatter` |
| 74 | `and_` | `vf_bitwise` |
| 75 | `or_` | `vf_bitwise` |
| 76 | `xor` | `vf_bitwise` |
| 77 | `shift_left` | `vf_bitwise` |
| 78 | `shift_right` | `vf_bitwise` |
| 79 | `histograms` | `vf_spr_misc` |
| 80 | `clear_spr` | `vf_spr_misc` |
| 81 | `mem_bar` | `vf_spr_misc` |

### 8.5. Evidence for mode combinations

See [vf_align](../../examples/basic_api/vf/vf_align/test.py) for load/store `dist`, stride and post-update combinations; [VF behavior](../api/vf.md) owns cross-example mechanisms.

### 8.6. Entry APIs and register budgets

[vf_entry](../../examples/basic_api/vf/vf_entry/test.py) compares `vf.load/store/move/full` semantics. UnalignReg limits simultaneous liveness, not declaration/call counts; four positive/negative forms appear in [vf_entry kernel](../../examples/basic_api/vf/vf_entry/kernel.py) and `scripts/_checks/tests/test_hardware.py`. Evidence and limits belong to [VF behavior](../api/vf.md), without copying probe histories here.
