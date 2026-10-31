# SIMT payloads: transport units and record ownership

Use for routing, indirect reads, sampling and packing where several contiguous elements share coordinates, weights, indices or a destination. First run the [element-per-thread direct-GM baseline](simt-first.md), then compare mappings. Even a simple copy can dominate time; use [native-task decomposition](../verification/native-timing.md) to locate the worthwhile change.

## Derive ownership

For `T` threads per block, choose a logical group width `G` with `T % G == 0`. `G` need not equal a hardware warp width. Groups own records and lanes own contiguous elements:

```text
lane = thread_id % G
groups_per_block = T // G
record_start = block_id * groups_per_block + thread_id // G
record_step = grid_blocks * groups_per_block
record = record_start, record_start + record_step, ...
element = lane, lane + G, ...
```

These quotient/remainder mappings give each valid `(record, element)` one writer. Read metadata only for valid records; bound the element loop by the true width. Cover record counts beyond one grid and tail widths. Assign separate unique writers to compact per-record outputs such as indices or counts.

Define the independent unit before flattening: it may be a scalar, a tuple of adjacent elements with coupled outputs, or a transport word. One unit per thread preserves work shared within that unit. In the grouped mapping, `G=1` instead makes each thread traverse an entire record serially; it is not the flat control. Name comparisons by their actual ownership and loops.

Hoist only element-independent reads, address calculations and computation outside each thread's element loop. Each lane can compute metadata once and reuse it; lane 0's local scalar is not automatically visible to other lanes. No shared UB or barrier is needed without cross-thread communication. A whole record per thread is also a candidate, but addresses across threads may become widely strided: logical reuse does not guarantee coalescing.

## Use wider words for pure transport

When a phase only copies contiguous bytes, its numerical element type need not be its transfer unit. A native `pl.make_tensor(..., dtype=...)` view can group adjacent short elements into one integer word for SIMT load/store. This changes address/count units, not values; `simt.cast` is a numerical conversion and is not a substitute. Keep arithmetic consumers in their original dtype.

Before testing a wider carrier, prove base alignment, row/panel byte pitch, whole-word extent and unique output ownership. Express the view's shape, stride and loop bounds in carrier elements, preserving the original byte span and record selector. Do not round a partial word up into padding or another record; use a narrower path or separately owned tail when the contract permits it. A contiguous tensor can have a misaligned storage offset, so contiguity alone does not prove alignment. Input/output overlap needs its own ordering proof.

The [indexed word-view probe](../../examples/basic_api/datamove/strided_view/kernel.py) uses ordinary native kernels and tests FP16/BF16 storage through 32/64-bit integer views. Its tests compare original 16-bit patterns, including NaN payloads, signed zero, repeated indices, aligned offset bases and guards; they do not reproduce the carrier's grouping as the reference. Official `make_tensor` docs, Python/C++ declarations and these probes agree on behavior `75a6c2600` / CANN 9.2.0. Fewer address calculations and scalar memory operations can matter even when bytes moved are unchanged; this is not a universal speed guarantee.

## Compare flat and grouped mappings

Retain both a flat unit mapping and a cooperative record mapping, even when starting from grouped code. Grouping can reduce metadata work from roughly `width` to `G` operations per record, but its contiguous access span, active records and tail-lane waste also change. The flat control may repeat more metadata yet win through coalescing or more independent memory operations. Narrow payloads and `G > width` can waste lanes; small `G` can increase scattered accesses.

Make `T/G` explicit factory/cache parameters whose offsets and strides match the [actual launch](simt-first.md#launch-variants). At fixed workers, compare at least two legal `T` values for the flat control and a small grouped set spanning at least two `T` and two `G` values. Sweeping only `G` at the original `T` does not validate the grouped launch choice. Keep the same consumers and include the complete producer/consumer call when ranking the two mappings.

When changing carrier width, first hold launch geometry fixed to isolate that change, then retune both mappings in the new units. Losing at an old scalar path's thread count does not reject the flat carrier path. If its cost is still falling at a tested edge, follow that trend within budget before comparing the families' best configurations. The optimization workflow owns [screening](../workflows/optimization.md#screen-candidates-before-confirmation) and [neighborhood stopping conditions](../workflows/optimization.md#check-the-neighbors-of-a-winning-configuration).

## Workload and validation

Use the [data-dependent workload rules](../workflows/optimization.md#data-dependent-work): indexed reads need dispersed and high-reuse addresses; conditional payloads also need active, skipped and mixed records when those branches are contractual. Production Host handles metadata only; tests may report valid fractions, but Host must not inspect runtime tensors to select a configuration.

Validate tail widths, small and multi-grid record counts, worker counts, poisoned repeats and guards on the selected configuration. For pure transport, compare original payload bit patterns independently of the carrier grouping. Apply the workflow's [final confirmation](../workflows/optimization.md#exit-criteria); a locally faster group width or one cached workload does not establish a general mapping win.

Initialization is work too. Reducing pre-zeroing requires proving valid output ranges, fill ranges and order so every element has a writer; old caller values cannot supply correctness. For ordered categories, derive [stable positions](stable-placement.md) first and choose payload mapping separately. One block owning a category is not the same as a group cooperating on one record.

When a record produces both scatter and a payload-reduced small output, use [multiple-output ownership](multi-output-reductions.md) to separate final writers from reduction participants. Uniquely writing a statistic does not require that thread to read every channel serially; atomic outputs do not prevent per-record metadata reuse. Compare initialization fusion and partial/merge choices using their work domains and complete-call cost.

[Bilinear sampling](../../examples/algorithms/bilinear_sampling/) demonstrates shared coordinates and boundary decisions with cooperative channels. [RoPE and KV scatter](../../examples/algorithms/rope_kv_scatter/) demonstrates adjacent-element units and separate output-validity domains. Their headers and independent references define tested shapes, locality regimes and configuration limits; their defaults are not universal launch rules.
