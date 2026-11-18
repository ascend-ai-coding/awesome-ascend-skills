# Derive tests from implementation decisions

This is a selection guide, not a mandatory matrix. Choose entries supported by
the inspected source. The unit of coverage is an implementation decision or
interaction, not the number of shapes/dtypes executed.

## Hypothesis quality

Validity precedes sensitivity. For the baseline, bind each native constraint to
an assertion over the actual generated inputs: dtype/layout, argument-dependent
counts, alignment, fixed cardinality, capacity or broadcasting as relevant.
Check the installed op docs/tests as well as the wrapper. Recheck preconditions
when a risk case changes a mode or related argument. A schema validates argument
types, not these semantic relationships. Do not turn an unsupported combination's
output mismatch into a supported-input defect.

A useful hypothesis predicts a different observable result for a plausible
incorrect implementation. Write the expected invariant and its source before
execution. Use the documented contract, mathematical definition, an actual caller
requirement or an explicit cross-implementation compatibility requirement.
If none establishes the expected behavior, mark the probe exploratory.

Prefer a small input that makes the alternatives visibly different. State why a
baseline cannot expose the issue. Do not duplicate all preprocessing in the
oracle: two implementations can share the same wrong ordering or cast.

## Source cue -> discriminating experiment

| Source cue | Test design and limits |
|---|---|
| Split/slice, reorder, cast, bias, clamp, scaling before or after another operation | Ask whether swapping order changes semantics. Use unequal sentinel coordinates, nonzero bias, a boundary-crossing value, partial dimensions or non-unit scale as applicable. Verify supposedly untouched slices separately. Check whether the caller ever uses the distinguishing combination. |
| Backward saves/allocates tensors under a conjunction of requires_grad flags | Compare the reachable flag classes: neither, each individually and together, if meaningful. Use analytic/CPU gradient or an independent autograd reference. Verify broadcast reductions into parameter shape. An unsupported flag combination is an interface limit, not automatically a model training defect. |
| Global cache, lazy init, default or config override | Reset to a controlled fresh state, repeat identical calls, then vary one argument/config source according to its supported lifetime. Establish precedence from contract separately; idempotence does not determine which argument should win. |
| Broad shape signature but dimension-dependent native branches | Partition by the branch boundary: full/partial, broadcast/nonbroadcast, aligned/tail, empty/nonempty only where relevant. Confirm the actual device tensor layout after transfers; constructing a CPU slice does not prove NPU noncontiguity. |
| Parameter/input precision or cast ordering differs by flag | Use caller-relevant mixed dtypes and nontrivial weights. Check output promotion and both input/parameter gradients before value casts hide the difference. |
| Multiple imports, registration layers or fallback implementations | Find the last effective registration and references retained by importing callers. Where possible probe the callable actually held by the caller. A correct alternate function is not proof the selected function is correct, or vice versa. |

## Interactions and controls

Start from the actual caller's ordinary input as a positive control. For a
suspected interaction, compare each condition alone and the joint condition when
feasible. Equal/zero/unit inputs often conceal ordering and scaling differences.
Choose values to expose the specific invariant; random tests supplement them.

Build each variant from immutable value templates through a fresh tensor factory.
Keep the valid baseline's dimensions/data/preconditions when varying only a
gradient flag or option. Do not let `.to()` aliases, in-place requires_grad changes,
existing .grad values or saved graphs leak between variants. Verify a reported
case still behaves the same when run alone if a dependency is suspected.

Give independent hypotheses separate assertions/cases. A failing permutation
must not silently prevent an independently constructible inverse-gradient test;
conversely do not feed an invalid mapping into the inverse and call its mismatch
a second defect. Reuse a validated mapping or construct a contract-valid one.

Useful invariants include: identity/zero-effect settings, unchanged pass-through
regions, deterministic repeated calls, equivalent formulations, output promotion,
and gradient reduction shape. Verify each invariant's assumptions: for example,
an identity rotation may still intentionally include a separate reorder or scale.

Use simple fault reasoning before an expensive sweep: would the planned assertion
reject skipping the suspected transform, changing its order, or dropping one
required gradient? If the same expected answer would result, redesign the input.
This is reasoning about test sensitivity, not permission to modify source or a
requirement to run a mutation-testing framework.

## Prioritize and close a finding

Choose first by caller impact and concrete source evidence, then expected cost.
One branch-discriminating gradient/contract test may be more informative than
another nominal dtype case. Do not promise exhaustive combinations.

For a mismatch, separate:

1. What executed and differed, with unchanged original output.
2. Why the oracle is justified and whether the input is supported.
3. Which layer is responsible: model selection, wrapper, native forward/backward.
4. Whether a real configured caller can reach it; otherwise label potential or
   interface-only impact. Preserve unknowns instead of inferring production failure.

When time is limited, prefer resolving one plausible high-impact finding to
collecting many unclassified anomalies. Stop unrelated exploration at the budget;
record remaining branches for follow-up.
