# Example library

Each example explains one mechanism, boundary or small integrated pattern. This page owns selection and admission. Detailed contracts, toolchain identity, measurement conditions and performance tradeoffs remain in each example's `kernel.py` / `test.py`; the index does not duplicate experiment narratives or historical pass counts.

## Selection principles

Use `check.py surface` for `basic_api/` calls, then read the nearest kernel docstring and test header; use the tables below for matmul and complete algorithms. Examples establish only their verified scope, not the current task's shapes, precision or configuration. If none fits, consult upstream Pro STs (`pypto/python/tests/st/pypto_pro/frontend`), retain provenance and independently revalidate. Tensor API examples in `pypto/examples/` are not the Pro path.

Add examples for recurring mechanisms or verification gaps, not mechanically per API. An `algorithms/` example also needs a pattern or actual task consumer; that group is neither in the VF assignment table nor indexed by `surface`.

## Matmul directory

Enter Cube work through the [GEMM mapping](../patterns/gemm-desugar.md). Select by mechanism:

| Mechanism | Examples |
| --- | --- |
| GEMM expansion, explicit addresses and automatic mutex IDs | [gemm_vs_matmul](../../examples/matmul/gemm_vs_matmul/) |
| Reusing an unchanged L0B operand across independent M blocks | [l0_operand_reuse](../../examples/matmul/l0_operand_reuse/): matched sugar/native chains, explicit slot lifetimes and repeated-call numerical controls |
| K/N splitting, physical tails and transpose views | [split_k](../../examples/matmul/split_k/), [split_n](../../examples/matmul/split_n/), [tail_extent](../../examples/matmul/tail_extent/), [transpose_reinterpret](../../examples/matmul/transpose_reinterpret/) |
| MX matrix formats | [gemm_mx](../../examples/matmul/gemm_mx/) |
| Shape/dtype specialization | [tilingkey_specialize](../../examples/matmul/tilingkey_specialize/), [dynamic_shape](../../examples/matmul/fundamental_usage/dynamic_shape/) |
| Teaching mechanisms using basic matrix APIs | [acc_phase](../../examples/matmul/fundamental_usage/acc_phase/), [bias_fusion](../../examples/matmul/fundamental_usage/bias_fusion/), [hif8_fp32](../../examples/matmul/fundamental_usage/hif8_fp32/), [l1_layout](../../examples/matmul/fundamental_usage/l1_layout/), [matmul_add_chain](../../examples/matmul/fundamental_usage/matmul_add_chain/), [matmul_mx](../../examples/matmul/fundamental_usage/matmul_mx/) |

## Algorithms directory: advanced composition

### Indexing, selection, routing and sampling

| Example | What it demonstrates |
| --- | --- |
| [segmented_scan](../../examples/algorithms/segmented_scan/) | Segmented prefix sums, jointly propagated values/head flags and state carried across registers |
| [streaming_segmented_scan](../../examples/algorithms/streaming_segmented_scan/) | Bounded UB chunks, ordered block summaries and three native stages; local versus incoming carry, dynamic scalar broadcasts and changed metadata at reused addresses |
| [stable_filter](../../examples/algorithms/stable_filter/) | Predicate ranking, stable value/index compaction, counts and tail fill values |
| [row_gather](../../examples/algorithms/row_gather/) | Indirect tile-to-register reads, negative-index normalization, physical padding and tail masks |
| [indexed_batched_gemm](../../examples/algorithms/indexed_batched_gemm/) | Runtime-selected rows staged in GM and consumed by a separate Cube product; exact panel bytes, explicit placement, independent grids and short/partial M |
| [strided_transpose](../../examples/algorithms/strided_transpose/) | SIMT 1-D physical views with actual stride addressing and bit-preserving payloads |
| [row_argmax](../../examples/algorithms/row_argmax/) | First-index extrema and dense FP32/INT32 output transport |
| [row_top2](../../examples/algorithms/row_top2/) | Stable paired candidates, independent register state, exact indices and input bit patterns |
| [row_topk](../../examples/algorithms/row_topk/) | Variable K, bounded candidates, ordered UB updates and dense outputs; merge scans complete candidate lists |
| [row_topk_heads](../../examples/algorithms/row_topk_heads/) | Complete-order insertion/pruning and head/cursor emission for larger K; displaced ties, exhausted streams and exact payloads |
| [stable_expert_dispatch](../../examples/algorithms/stable_expert_dispatch/) | Interval counts/stable positions, payload mapping, inverse maps and unused tails |
| [moe_weighted_combine](../../examples/algorithms/moe_weighted_combine/) | Bounded-error integer accumulation for reproducible indexed sums, worker/group private slots and complete-call scratch/traffic tradeoffs |
| [embedding_bag](../../examples/algorithms/embedding_bag/) | Runtime offsets/indices, empty segments and direct versus partial/merge reduction |
| [simt_scatter_add](../../examples/algorithms/simt_scatter_add/) | Global ordering between native initialization and atomic accumulation on one Stream |
| [byte_histogram](../../examples/algorithms/byte_histogram/) | UINT8 input, UINT16 local counts, INT32 partial/merge and overflow boundaries |
| [grouped_int8](../../examples/algorithms/grouped_int8/) | FP32 multiplication, RNE, integer range and `PACK4_B32`, separating group buffers from logical GM addresses |
| [sliding_max](../../examples/algorithms/sliding_max/) | Output versus clipped-load coordinates, read-only halos and transfer reuse across VF segments |
| [bilinear_sampling](../../examples/algorithms/bilinear_sampling/) | Record/channel groups, shared coordinates/weights, per-neighbor zero extension and direct integer-point movement |
| [bilinear_sampling_backward](../../examples/algorithms/bilinear_sampling_backward/) | Conflicting dX scatter, owned dGrid partial/merge and selected-cell derivatives at integers; configuration trades off large and small shapes |
| [rope_kv_scatter](../../examples/algorithms/rope_kv_scatter/) | Coupled adjacent-element units, record/lane groups, conditional unique-slot writes and unconditional Q output; bit-preserving identity regions and configuration-neighbor comparisons |

### Elementwise state updates

| Example | What it demonstrates |
| --- | --- |
| [fused_adamw](../../examples/algorithms/fused_adamw/) | In-place FP32 state updates from FP16 inputs, UNPK predicate reuse, independent buffer IDs and per-call FP64 references across changing gradients/scalars |

### Reductions and normalization

| Example | What it demonstrates |
| --- | --- |
| [column_statistics](../../examples/algorithms/column_statistics/) | Noncontiguous-axis reduction, one retained column per lane and centered two-pass variance |
| [streaming_moments](../../examples/algorithms/streaming_moments/) | Bounded row chunks, lane/partial compensation, centered variance and independent long-row cancellation probes |
| [csr_softmax_aggregate](../../examples/algorithms/csr_softmax_aggregate/) | Bounded exponential reference, shared weights, runtime compact segments and owned payload partials; empty rows and FP32 cancellation probes |
| [layernorm](../../examples/algorithms/layernorm/) | VF dual reductions and tail handling after mean subtraction; one zero fill cannot protect all subsequent reductions |
| [engram_gate](../../examples/algorithms/engram_gate/) | Dual RMS and weighted-dot VF reductions, resident weights, key/output UB reuse, tiling-key slot selection and masked residual updates |
| [layernorm_backward](../../examples/algorithms/layernorm_backward/) | Separate row/column mappings for outputs and batched statistic loads/broadcasts |
| [masked_softmax](../../examples/algorithms/masked_softmax/) | Runtime valid prefixes, exact zero for empty rows, dynamic rows and unaligned widths |
| [streaming_masked_softmax](../../examples/algorithms/streaming_masked_softmax/) | Rows exceeding UB, fixed transfer blocks, streamed state and dense LSE transport |
| [prefix_cross_entropy](../../examples/algorithms/prefix_cross_entropy/) | Prefix normalization, smoothing, loss/gradient outputs and distinct dense-scalar/block store paths |
| [residual_rms_quant](../../examples/algorithms/residual_rms_quant/) | Contractual FP16 rounding, shared binary scaling for tiny normalized ratios, INT32-to-INT8 packing and invariant-vector reuse; two native stages with explicit task/GM costs |

### Matrix fusion and attention

| Example | What it demonstrates |
| --- | --- |
| [mla_prolog](../../examples/algorithms/mla_prolog/) | Projection/RMSNorm chains retaining FP32 intermediates with BF16 weights; independent K partials, residual components and thin-row VF/Cube alternatives |
| [mla_bounded](../../examples/algorithms/mla_bounded/) | Split score features and independent V: materialized FP16/BF16, short-axis FP16 fusion and single-query FP16 streaming; choose the entry through [normalized-contraction anchors](../patterns/normalized-contractions.md#select-a-correctness-anchor) |
| [mm_softmax_mm_materialized](../../examples/algorithms/mm_softmax_mm_materialized/) | Batched/scaled chain with GM intermediates, independent phase grids, explicit placement, short/partial M, runtime prefixes and FP16/BF16 conversion |
| [batched_transposed_gemm](../../examples/algorithms/batched_transposed_gemm/) | Batch addressing, L1 transpose, explicit groups with one slot and M tails under N/K alignment preconditions |
| [ragged_grouped_gemm](../../examples/algorithms/ragged_grouped_gemm/) | Runtime group boundaries, short-group weight reuse, long-group sharing, dynamic M tails and fixed FP64 term-sum criteria |
| [grouped_gemm_swiglu_quant](../../examples/algorithms/grouped_gemm_swiglu_quant/) | Across-group M/N ownership, empty local consumers, independent L1/L0 K sizes, paired N accumulators retaining L0A and batched VF quantization; `kernel.py` also accepts Host-known group attributes for full-task evaluation |
| [paged_gqa_decode](../../examples/algorithms/paged_gqa_decode/) | Runtime page lookup, regular VF inner blocks, bounded exponential reference and multi-page partials; GQA input reuse versus intermediate traffic, exact empty outputs and metadata changes |
| [broadcast_gemm_relu](../../examples/algorithms/broadcast_gemm_relu/) | Broadcast weight reuse, disjoint Mat A/W/bias placement, GEMM bias initialization and Acc-output ReLU |
| [gemm_bias_silu](../../examples/algorithms/gemm_bias_silu/) | K splitting, bias fusion and one on-chip Cube→Vector handover |
| [gemm_bias_silu_banded](../../examples/algorithms/gemm_bias_silu_banded/) | Minimal 1+1 B-layer topology, an internal band loop, explicit forward/reverse events and `DualModeSplitM` |
| [gemm_bias_silu_multicore](../../examples/algorithms/gemm_bias_silu_multicore/) | Mix-kernel execution groups/core indices and disjoint output partitions |
| [mm_softmax_mm](../../examples/algorithms/mm_softmax_mm/) | Fixed unscaled FP16 chain, explicit c-v-c events, VF→Cube NZ packing and serial/positive-preload consistency |
| [swiglu_ffn](../../examples/algorithms/swiglu_ffn/) | Dual-projection L0 reuse, three explicit handover edges, VF SiLU×mul, final GEMM and finite-output/term-sum criteria |
| [rmsnorm_gemm](../../examples/algorithms/rmsnorm_gemm/) | v-c topology, order-preserving FP16 UNPK widening, explicit placement and two-slot VF→Cube handovers |
| [rmsnorm_gemm_tail](../../examples/algorithms/rmsnorm_gemm_tail/) | Effective extents across core/band/AIV partitions, compact GEMM tiles, tail localization, guards and sanitizer |
| [gemm_rmsnorm](../../examples/algorithms/gemm_rmsnorm/) | Independent GEMM/whole-row reduction blocking and explicit Mat/Vec placement, with output as the stage intermediate |
| [quant_gemm_w8a16](../../examples/algorithms/quant_gemm_w8a16/) | VF byte widening, explicit W handover events, separate FIX scale-payload event and own-layer numerical checks |
| [quant_gemm_w4a16_grouped](../../examples/algorithms/quant_gemm_w4a16_grouped/) | Packed INT4, K-axis groups, explicit Vec/Mat placement and the FP32-dequantization→FP16-RNE intermediate contract with GM weights |
| [quant_gemm_w4a16_onchip](../../examples/algorithms/quant_gemm_w4a16_onchip/) | Bounded Vec conversion fragments, explicit forward/reverse events, ND→NZ→shared L1 handover and zero auxiliary GM; row ownership trades reuse for parallelism |
| [attn_epilogue](../../examples/algorithms/attn_epilogue/) | Inverse RoPE and two MXFP8 projections, independent CPU mathematics, native intermediate-rounding reference and exact payload checks |
| [quant_lightning_indexer](../../examples/algorithms/quant_lightning_indexer/) | MXFP4 QK, data/scale slot pairing, BF16 sequential head FMA, bounded Top-K selection and native alias/event ownership |
| [quant_sparse_lightning_indexer](../../examples/algorithms/quant_sparse_lightning_indexer/) | Candidate-block gather, LI/LD metadata, retained Cube operands, explicit VF/Cube rings and exact BF16 Top-K checks |
| [flash_attention](../../examples/algorithms/flash_attention/) | Fixed single-head online softmax, explicit placement and per-slot handovers, transposed scores, same-core alpha ring and an input-dependent precision budget |
| [sparse_attention](../../examples/algorithms/sparse_attention/) | CANN FP16 sparse attention with V as a K prefix: dense-mask and selected-row paths, five native pipeline stages, static first/last VF dispatch, causal/empty rows and slot-reuse controls; original-evaluator contract tests additionally require `CANN_BENCH_ROOT` and importable `kernel_eval` |
| [flash_attention_mha](../../examples/algorithms/flash_attention_mha/) | Fixed BSND causal MHA, Host-known weighted partitioning, owner-cache/prefetch guards, explicit 512 KiB Mat plan and three handovers using 16 event IDs |
| [causal_attention_backward](../../examples/algorithms/causal_attention_backward/) | Streamed FP32 VF gradients with explicit single-slot Vec placement, completed row statistics and independent query/key owners; K/V blocks and query reuse |
| [decode_attention_gqa](../../examples/algorithms/decode_attention_gqa/) | Thin grouped contractions, explicit resources/events, physical/valid row separation, guarded prefetch and optional workspace/MIX merge; [short multi-position queries](../patterns/online-softmax.md#decode--very-thin-m-gqa) need their own coordinate/mask adaptation |
| [window_gqa_attention](../../examples/algorithms/window_gqa_attention/) | Runtime windows/lengths, O/LSE outputs, shared VF row state and query blocking |

Use the shared [native-task timer](../verification/native-timing.md) for complete calls. Remeasure current source/toolchains instead of inferring performance from this index.

### Run the grouped GEMM controls

After selecting the device and taking its lock through the external environment, run from the skill root (`pypto-pro-kernel/`):

```bash
python scripts/check.py run --source examples/algorithms/grouped_gemm_swiglu_quant -- python -m pytest examples/algorithms/grouped_gemm_swiglu_quant/test.py -q
```

For a paired timing, prepare inputs with `reference.make_inputs`, validate both `kernel.run` configurations against `reference.reference`, then pass their complete two-kernel calls to the shared timer. Compare `global_work=False/True` at fixed geometry to isolate ownership. At fixed ownership, compare `reuse_l0=False/True` and `row_batch=1/2` separately. Resident-input timings do not reproduce an external evaluator's cache protocol.

For the CANN-Bench grouped-matmul/SwiGLU/quantization task, copy the example's `kernel.py` unchanged into the submission's operator package. Have its selected entry import `grouped_matmul_swiglu_quant` from that module. This wrapper accepts Host-known cumulative `group_list` attributes; it constructs device offsets from those attributes without reading back device data. Keep the package's normal dispatcher, verify the imported module path and matching file hashes, then run the original evaluator on all cases with its unchanged precision and performance settings. Verify source-mode support before using an evaluator's skip-install option. This keeps the delivered library code in the measured path; a temporary implementation's score cannot validate a later rewrite. Save generated reports outside tracked source. Hardware and shape changes require remeasuring `configuration()` choices.

## Example layout

This layout is the standard for every repository example. Standalone operator handoffs outside the example library follow the [delivery format](../workflows/delivery.md).

```text
examples/<group>/<name>/
    __init__.py
    kernel.py       native pypto_pro kernel
    reference.py    independent mathematics and fixed comparison criteria
    test.py         explanation and ordinary pytest cases
    _kernels/       optional native computation blocks or tiling-key variants
    _references/    optional mathematical reference blocks
    _tests/         optional evaluator adapters or large test-support blocks
```

Each example's top level contains only `kernel.py`, `reference.py`, `test.py` and the package marker `__init__.py`. `kernel.py` owns or exports the public native implementation; `reference.py` owns or imports the independent mathematics and fixed comparison criteria; `test.py` collects every ordinary pytest case and owns local fixtures. Keep small helpers in the appropriate entry file. When substantial functional blocks or distinct tiling-key paths need separate modules, place them in subpackages and expose them through these entries. Subpackages describe real computation, reference or verification roles; they must not serve as storage for retired scripts.

Do not add per-example benches, READMEs, reports, precision-comparator files or license copies. Put current contracts and usage in implementation docstrings and the test header; use the shared timer for optional performance measurements, and keep generated evidence outside tracked source. The shared [CANN agreement](../../LICENSE.CANN) accompanies files carrying its notice; retain source copyright and license headers. Package markers support relative imports and pytest collection of identically named `test.py` files. The library's shared `conftest.py` and `pytest.ini` supply device selection and collection settings.

An external evaluator adapter must identify its dependency and preserve the original golden and precision rules. For QLI, setting `CANNBOT_DSL_ROOT` includes the pinned upstream evaluator in `test.py`; its adapter requires CANNBot DSL 0.7.0. An unset dependency may skip optional evaluator cases, but target-board acceptance must enable and pass them. Debugging examples put expected values in `reference.py` and check logs, exceptions and process behavior in `test.py`.

## Admission rules

- When adding, removing or renaming a `matmul/` or `algorithms/` example, update its direct directory or `kernel.py` link in this page's selection tables in the same change. A parent-group link does not cover its children. Run the default `check.py` before admission; its [catalog gate](checks.md#consistency-and-maintenance) rejects missing entries and broken example targets without maintaining a second inventory. `basic_api/` uses the separate API assignment and live surface checks.
- Supply the kernel, independent reference and ordinary pytest tests. The test.py header answers: what it solves, physical accesses, synchronization rationale and independent verification.
- Derive the reference mathematically rather than duplicating tile/slot/blocked accumulation order. Consume actual quantized inputs; explicitly specified intermediate rounding is contractual too. Do not infer tolerances backwards from the candidate. See the [HF8 correct/incorrect-reference comparison](../../examples/matmul/fundamental_usage/hif8_fp32/test.py).
- Kernels compile without agentkit and follow [VF/GEMM shipping policy](../../SKILL.md#constraints-and-workspace-policy), including teaching examples. The Vector tile-compute exception is limited to the sorting family; manual Cube examples state their allowed scope explicitly.
- The kernel.py module docstring begins with `写法：原始。`, `写法：语法糖。` or `写法：混合。`. These are checker conventions, not compiler requirements.
- Convert temporary top-level imports to package-relative imports and rerun `python -m pytest examples/<group>/<name>/test.py` on the device from the final library path. Temporary-directory success does not establish package collection. Toolchain identity, source hash and fixed criteria identify the evidence; identical historical source does not establish a current pass.
- Library device/environment fixtures come from [conftest.py](../../examples/conftest.py). Standalone scripts must import/initialize torch_npu and provide a device fixture or select the logical device directly. Copying a test signature does not copy fixtures. `pytest.importorskip` can handle missing runtime dependencies, but skips on the target board do not count as acceptance.
- Isolate teaching negative cases from correct calls and state the `反例：` marker, verification mode and no-board conditions explicitly. Potential hangs cannot enter default numerical tests. See [check waivers](checks.md#counterexample-declarations) and [device isolation](../verification/board-runs.md#isolating-device-faults).
- Algorithms must pass on-board within their stated scope and have a corresponding pattern or task; do not add them just to increase count. [Authoring](../workflows/authoring.md#preflight-decisions) owns the complete boundary matrix.

## Pipeline examples

Serial rewriting and synchronization at `preload=0` do not validate positive-preload fill/drain or slot reuse. An example claiming to demonstrate preload's scheduling effect needs a later stage on at least one core, at least one tested `preload>0` value and enough tasks to substantially exceed the longest fill and exercise steady-state slot reuse. [Cross-core synchronization](../api/crosscore-sync.md) owns the mechanism and version scope.

These are conditions for demonstrating that mechanism, not requirements to add stages to every production kernel. The 1+1 [banded GEMM](../../examples/algorithms/gemm_bias_silu_banded/) can demonstrate a minimal handover; its preload comparison cannot be presented as a scheduling-rewrite gain. [mm_softmax_mm](../../examples/algorithms/mm_softmax_mm/) and [flash_attention](../../examples/algorithms/flash_attention/) provide same-core later-stage comparisons.

Create an independent kernel for each preload instead of mutating `_pipeline` on an existing compiled object. Historical build-directory keys omitted preload and could return an older binary. Preserve the numerical contract; the deterministic examples above also compare serial/shipping variants bitwise. Framework references are `test_pipelined_crosscore_handover.py` and `test_auto_pipeline_two_loops.py` under `python/tests/st/pypto_pro/frontend/auto_pipeline/`.

## Language and naming

Agent-facing instructions in this skill are English. Bundled example explanations/docstrings use Chinese, with English identifiers, API names and machine-readable fields. Do not name ordinary intra-core double buffering `pipeline` / `pipelined`: `pl.jit(pipeline=)` denotes cross-core pipelining. Use an explicit name such as `double_buffer` for the intra-core mechanism.
