# Stable grouping: logical positions and parallel ownership

Use for stable filtering, keyed packing, bucketing and scatter that preserves original record order. Derive each record's deterministic output position before choosing who moves its payload. Stability follows from logical rank; it does not require one worker to process an entire category sequentially.

Partition input into logical intervals ordered by original position. Let `count[g,r]` be the number of valid records of category `g` in interval `r`; local rank counts only earlier records of the same category. For a valid record `i`:

```text
destination(i) = total records in earlier categories
               + same-category records in earlier logical intervals
               + same-category rank of i within this interval
```

These terms establish disjoint output intervals and stable order. Logical interval IDs follow input ranges, not worker IDs, completion order or atomic-counter acquisition order. The device computes runtime categories, valid counts and positions; Host supplies only shapes, layouts and launch parameters. Empty intervals contribute zero, adjacent offsets may be equal for empty categories, and invalid records consume no output position.

Choose count-table layout for its consumer's traversal. With category-major, interval-minor counts, one exclusive prefix over the table contains the first two terms; category boundaries come from corresponding prefix positions. The final boundary is the total valid count: save the total separately or allocate a complete boundary array rather than reading beyond the prefix's logical length. Other layouts require rederiving indices and scan axes. Budget count/rank/intermediate-map storage and scanning work; count dtype must hold the total, and oversized tables need hierarchical or streaming treatment.

One worker per category scanning the whole input is a valid simple baseline, but count its repeated `category_count × input_length` work and utilization under few or concentrated categories. More launch workers cannot split a long task already assigned to one owner. Interval counts and prefix positions allow several workers to process different input ranges of one category while preserving order; that choice is independent of payload group width.

Once positions are known, assign movement by input record, output record or contiguous payload block. Compare [SIMT metadata reuse and contiguous access](simt-payload.md), avoiding both per-element rank recomputation and serializing an entire category onto one block just to reuse metadata. Prove unique writers, disjoint intervals and producer completion before consumption separately; native-kernel stages on one Stream can express those dependencies.

Build loop bounds directly from the valid range. If only a tail needs filling, traversing the entire output with a tail predicate retains idle work; multiple owners can cover the known tail interval directly. Fully define compact outputs, inverse maps and invalid records too; correctness cannot depend on old caller values. Test empty sets, one category, interleaved categories, interval boundaries, non-divisible interval counts, multiple workers, repeated calls and guards. Performance comparisons also retain concentrated and dispersed distributions.

Runnable two-stage reference: [stable_expert_dispatch](../../examples/algorithms/stable_expert_dispatch/). It computes interval counts explicitly, then token owners accumulate earlier intervals and copy payloads. This is an inspectable correctness starting point, not an optimal count/prefix/movement schedule. Independent stable-sort tests cover empty categories, tails, inverse maps, special FP32 bit patterns, offset outputs and repeated calls.
