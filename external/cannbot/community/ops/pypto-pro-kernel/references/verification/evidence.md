# Evidence assessment: counterexamples and next actions

Retain the original item numbers for existing references. Each item keeps the mistaken interpretation and next action; Git retains the history.

## 1. Misclassifying `addc` / `subc` as framework bugs and committing that conclusion

Official call examples are part of the contract. Multi-result destinations for `addc/subc` require prior declaration with the correct types; reading only prototypes can produce a false defect claim. See [constraints](../api/constraints.md).

## 2. Declaring static final axes for 4-bit types impossible after an incorrectly scoped exhaustive search

For cached compilation behavior, test cold objects, cache hits and different call orders. See the [static FP4 final-axis defect](../diagnostics/defects/subbyte-static-dim-order-dependent.md); one rejection cannot prove all paths reject it.

## 3. Calling arena an existing framework term: circular evidence

Use `git log -S <name>` to check a term's origin. A name or docstring you introduced is not independent evidence for itself. See the [allocation contract](../api/memory-allocation.md) for explicit placement and mutex ownership.

## 4. Treating differences introduced by another change as measurement mistakes

When a result differs from a previous run, first check source hashes, workspace changes and the toolchain. Tests added in a parallel session are not differences caused by the current change.

## 5. Using unspecified floating-point reduction order to justify wider tolerances

Locate the error source and numerical scale before relaxing a tolerance; continue diagnosis if neither documentation nor experiments justify it. Bitwise equality may depend on selected inputs and need not generalize to real workloads.

## 6. Writing the `reduce_max` lane-1 reference from its docstring and failing the test

When a docstring and official return-value description differ, keep both pieces of evidence distinct and use inputs that separate their interpretations. Undefined reduction-result lanes must not enter the numerical oracle; see [VF behavior](../api/vf.md).

## 7. Observing masks with `vf.move` and obtaining contradictory conclusions

Validate the observation tool first. `vf.move` with MERGING retains old destination values; unselected lanes of a new name have no reliable expected value. Do not treat it as an unconditional copy when observing masks; see [VF behavior](../api/vf.md).

## 8. Assuming `astype` rounds toward zero despite an explicit docstring default

Read the signature default first. Select inputs distinguishing rounding modes rather than inputs where candidates happen to agree. See [VF behavior](../api/vf.md) for conversion boundaries.

## 9. Assuming independent prefix sums in two halves because one half happened to pass

Cover all four combinations of two independent mode switches. One tested combination does not establish the other three. Treat any corresponding-region restriction as a hypothesis to verify.

## 10. Replacing a missing numerical criterion with a test that greps its own source

Assert program behavior rather than searching your prose to prove behavior. Consistency checks are appropriate only for contract fields or index tables that are actually consumed.

## 11. Treating `order=[1, 0]` as transpose based on a summary line whose body says otherwise

Read the actual calls and assertions; a test-file summary is not evidence. `order=[1,0]` is not equivalent to physical transposition; see [data movement](../api/datamove.md).

## 12. Predicting silent removal of dual + relu after reading a different ISA path

Confirm the instantiated C++ template arguments with a minimal compilation case. One helper may serve different paths. ISA checks are usually necessary conditions; the union of several paths cannot prove support on one particular path.

## 13. Declaring fixpipe offset absent in pypto_pro after searching eight names

Not finding a parameter with one name does not prove the capability is absent. Continue through bit fields, type inference and mode parameters. See [Fixpipe behavior](../api/fixpipe.md) for the current offset entry.

## 14. A criterion passes for the wrong reason: the diagnostic quotes the code being checked

Match only the target layer's own diagnostic. A regex inside a source excerpt or a quoted error code from another layer can make a test pass vacuously. Remove the correct check and rerun; see [validation traps](../diagnostics/validation-traps.md).

## 15. Calling a table generated when its generator was never checked in

Before claiming an index is automatically maintained, confirm the generator is in the repo, can rerun and has failure criteria. `check.py coverage` validates VF assignments; other historical statistics are not automatic contracts.

## 16. Blaming a one-position TilePad enum shift for missing synchronization despite four reruns

Before blaming the framework, inspect synchronization, IR/codegen and initialization. Change prior state, process or call order between reruns: four identical-state repetitions can reproduce the same UB residue. See [data movement](../api/datamove.md).

## 17. Treating `pl.get_spr()` as a constant 128

A neat constant can be stale content in an unwritten register. Add a control that does not use the feature under test at all. `pl.get_spr()` is not an observation port for squeeze byte counts; see [supporting interfaces](../api/support.md).

## 18. Using a table with dtype collapsed out as a sufficient condition

A compressed table cannot answer questions about dimensions it omits. Capability tables must distinguish dtype, path and parameters; layout alone is not sufficient evidence of support.

## 19. Attributing init_output temporary-tile safety to auto_mutex instead of make_tile_group

Change one independent variable per comparison. Do not attribute `init_output` safety to `auto_mutex=True`; distinguish tile groups from bare tiles. See [validation traps](../diagnostics/validation-traps.md) for the mechanism and workaround.

## 20. Declaring lrelu / prelu absent after missing their registrations while vf.prelu is already used

`pl`, `vf` and `pl.simt` are different namespaces. Before claiming absence, search the repo's executed examples, then Python/C++ registrations and lowerings.

## 21. Missing an existing gated parser check because its name differs from the guessed name

Search predicate structure and call paths, not only guessed feature names. A gated check and a completely missing check require different fixes. See the [allocation contract](../api/memory-allocation.md) for overlap-check coverage.

## 22. Assuming automatic synchronization is disabled when auto_mutex is omitted

Use the current signature's default: `auto_mutex` defaults to True. Test an excluded form before narrowing a check. If the local machine cannot run it, first read the target project’s external remote-execution configuration; a local limitation does not establish that the project cannot verify it.
