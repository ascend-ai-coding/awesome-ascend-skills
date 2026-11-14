# Compile Diagnostics: Output and Location Contract

Warnings and structured source locations are in place; a standalone resource-allocation report has not been implemented. Implementation entry points: `pypto_pro/_errors.py`, `_info.py`, and `language/parser/_call_parser.py::parse_op_call`. The mechanism was verified with CANN 9.2.0 / Ascend950PR on 2026-09-25; check the target installation for current behavior.

## Output levels

| Output | Default | Usage and boundary |
| --- | --- | --- |
| `CapacityUnknown` | On | Capacity should be known but could not be retrieved; check the environment first. DDR is not covered. |
| `CheckSkipped` | Off | Enable with `PYPTOPRO_INFO=skipped-checks`; a dynamic offset did not receive a static out-of-bounds check. |

Skip warnings also occur in ordinary loops, so they are disabled by default. They do not mean that no checks ran; runtime checking can be enabled with `@pl.jit(sanitizer=True)`. Warnings use Python's warnings module; pytest captures logging by default, so successful tests may not display it.

```bash
PYPTOPRO_INFO=skipped-checks python -m pytest <target> -W error::pypto_pro._errors.CheckSkipped
```

Enable both the environment gate and the pytest filter. Bare `python -W` may fail to import the custom category during startup and cannot replace this command. `PYPTOPRO_INFO=all` enables the existing categories; it does not mean that layout or resource reports exist.

## Spans and error matching

- Exceptions raised during DSL parsing carry the user's source location through `current_span`; `parse_op_call` covers both automatic and handwritten handlers.
- When a span exists, pass it using `span=` instead of stitching it into the message. When there is no location, use `None`; do not block ambient fallback with an unknown span.
- For ordinary runtime-call errors, first check whether the traceback already contains a user frame; do not batch-fix based on the “number of missing spans.”
- Use `e.error_code/e.span/e.module/e.origin`; parameter details and expected/actual values still need to be read from the message. Do not assume those are already structured fields.
- A source excerpt can cause `pytest.raises(match=)` to match the test's own regular expression. The UT `conftest.py` strips excerpts, and the regression is in `test_match_guard.py`; ST does not share this `conftest.py`. For new diagnostics, verify that the span points to the user's line and use a reverse comparison to show that the criterion can fail.

See [parse-error localization and regressions](#parse-error-localization-and-regressions) for a specific case of “caught but not clearly explained.” Do not retain counts of how many batches were fixed or how many `raise` sites have spans.

## Minimum contract if a resource report is started

For each tile, report its space, address range, byte size, mutex IDs, and depth, plus the capacity actually used by each space. Keep output silent by default and export on request, binding it to the source and toolchain; do not equate declared size with peak live allocation. See [capacity conventions](../api/memory-limits.md) for capacity sources.

`ir.IRVerifier` exposes batched `Diagnostic` objects whose field shape can serve as a reference, but historical probes confirmed that `Diagnostic` has no Python constructor, Python cannot register rules, and severity does not include Info. Probe capabilities first; do not rely on it as a generic report container. Existing `python_print` can read layouts, so the report is a convenience for consumers, not a prerequisite for allocator correctness.

Regressions: `python/tests/ut/pypto_pro/test_compile_warnings.py`, `test_error_format.py`, and `test_match_guard.py`. Search the source tree for their actual locations. Listing these tests as entry points does not mean they were rerun for this revision.

Argument-shape validation is performed against the public signature by `_op_registry.py::op_impl(validate_signature=True)`; the first group covers set/get of saturation flags (`22d9428f1`). Before evaluating arguments, it reports keyword-only violations, missing arguments, and unknown keywords, with the call span attached. It does not catch a `KeyError` inside a handler and does not claim protection for APIs that have not been wired in. The regression is in `test_keyword_only_contract.py`.

## Parse-error localization and regressions

Describe concrete diagnostic gaps with a minimal reproducer and the target toolchain identity.

| Call form | Current diagnostic | Regression |
| --- | --- | --- |
| Calling `bar_m/bar_mte1/bar_fix` in Vector | Parse-time `NotSupported`, with op, Cube/Vector side, and user source location (`3322ea999`) | `test_barrier_scope.py`: explicit/implicit sections, helpers, mixed kernels, and valid public barriers |
| Calling set/get saturation flags with positional arguments, missing arguments, or extra keywords | `InvalidArgument`, explaining the public argument contract and call location (`22d9428f1`) | `test_keyword_only_contract.py` |

Both test groups are under the framework's `python/tests/ut/pypto_pro/language/parser/`. On 2026-09-28, with CANN 9.2.0 / Ascend950PR, 1,150 parser cases and 8 on-board `move_sync` cases passed.

Source-location propagation for `TileType` is covered by `current_span` in `parse_op_call` (`b623212f9`). New checks should reuse this page's output and location contract; do not replace argument validation with an internal `KeyError`, or bulk-wrap ordinary runtime tracebacks because the parser lacks a span.
