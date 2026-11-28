---
type: "Reference"
title: "Query and relate performance cards"
description: "How the bundled card source participates in kernel search."
status: "stable"
tags: ["pypto-pro", "performance", "knowledge-card"]
---

# Query and relate performance cards

The card bundle ships with this skill. It uses a separate, explicitly selected `perf-cards` source. The [Active table](../index.md#active-items) is the only admission list. Neither a file in a category directory nor a code fragment in the contributor example becomes a kernel contract. The [optimization workflow](../../workflows/optimization.md) chooses experiments; the card only supplies a conditional mechanism.

## Link only the part that overlaps

```yaml
knowledge_links:
  schema: pypto-knowledge-links/1
  relations:
    - kind: same_mechanism
      target:
        skill: pypto-pro-kernel
        id: contract:grouped-ownership
      scope: Number independent GEMM output tiles continuously across group boundaries.
      boundary: Fixed group lengths do not establish the runtime prefix mapping or a speedup.
```

Use 1–32 relations if the field is present; omit it when there is no accurate target. A relation contains exactly `kind`, `target`, `scope` and `boundary`. Targets use a stable `contract:<id>` or `document:skill/<relative-path>.md#<optional-anchor>` identity. Do not use an absolute path, line number or static-check rule name. Do not repeat a (kind, skill, id) tuple. The reader resolves the section and hashes the sources at read time.

| Kind | Example and boundary |
| --- | --- |
| `same_mechanism` | [cube-06](../cube/cube-06-grouped-continuous-core-assign.md) can be grouped with continuous worker assignment; the current shape, ownership and benefit still need verification. |
| `overlaps_with` | [vec-09](../vec/vec-09-loop-sink-into-vf.md) shares multi-row loop organization but does not imply the same DMA or metadata strategy. |
| `uses_contract` | [pipe-07](../pipe/pipe-07-ub-to-l1-handoff.md) relies on the UB-to-L1 layout/visibility contract. Removing a GM round trip is different from removing a local layout conversion. |
| `alternative_to` | [pipe-05](../pipe/pipe-05-gm-to-l1-direct-load.md) compares direct loading only where the original operand requires no Vector transform. |

Relations are not transitive. A capacity contract shared by two cards neither equates their changes nor transfers test results. A cache-maintenance API is not an L2 bypass switch, and UNPK/PACK does not perform BF16 numerical conversion.

## Validate the library

These commands check retrieval, metadata and link targets. For a new card, prepare the factual material described in the [contributor guide](../CONTRIBUTING.md) and obtain technical review before merging.

From the installed skill root, run:

```bash
python -m pip install -r scripts/requirements-cards.txt
python scripts/check.py --cards-root references/knowledge-cards query cube-06 --kind document --format json
python scripts/check.py --pypto-root "$PYPTO_SRC" --cards-root references/knowledge-cards query --validate --format json
```

For subsequent reads and continuations, reuse the returned request and file-content SHA-256 (`source_sha256`). Keep the same selected roots for CLI calls. An unavailable external framework is a coverage limitation, not evidence of an absent API. For exact document IDs, candidate grouping and continuation reads see [optimization tooling](../../tooling/optimization.md#optional-performance-card-source).
