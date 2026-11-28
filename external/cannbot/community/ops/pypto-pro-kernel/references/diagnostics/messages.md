# Compile Diagnostics: Output and Location Contract

Warnings and structured source locations are in place; a standalone resource-allocation report has not been implemented. Implementation entry points: `pypto_pro/_errors.py`, `_info.py`, and `language/parser/_call_parser.py::parse_op_call`. The mechanism was verified with CANN 9.2.0 / Ascend950PR on 2026-09-25; check the target installation for current behavior.

## Output levels

| Output | Default | Usage and boundary |
| --- | --- | --- |
| `CapacityUnknown` | On | Capacity should be known but could not be retrieved; check the environment first. DDR is not covered. |
| `CheckSkipped` | Off | Enable with `PYPTOPRO_INFO=skipped-checks`; a dynamic offset did not receive a static out-of-bounds check. |

Skip warnings also occur in ordinary loops, so they are disabled by default. They do not mean that no checks ran; runtime checking can be enabled with `@pl.jit(sanitizer=True)`. Warnings use Python's warnings module; pytest captures logging by default, so successful tests may not display it.

## Tool feedback contract

`query`, `rules --format json`, `native`, `suggest` and `run --format json` return `pypto-kernel-feedback/1`. The default text runner emits its final envelope after `kernel_feedback`; JSON mode writes one compact result to stdout and forwards live child output to stderr. Default feedback retains source identities, diagnostics and selected context, summarizes file inventories by digest and omits non-query transfer inventories. Use `--details` to expand the identity/descriptor fields; coverage limits remain visible in either form. [Tool usage](../tooling/checks.md#scenario-query-and-native-preflight) owns invocation and the metadata specification. `suggest` adds bounded `candidates` with unresolved predictions and measured-cost references; `experiment` uses the separate [current-state contract](../tooling/optimization.md#one-best-snapshot-and-one-current-trial).

| Field | Meaning |
| --- | --- |
| `stage`, `status` | The phase that produced the result and `passed`, `failed` or `unverified`. Passed means that phase completed without its rejection conditions, not numerical correctness |
| `identity` | Checked file hashes, checker implementation digest, host Python and selected framework-source hashes. `selected_source.complete` means the fingerprinted declaration, allocation and runtime entry files are present; API coverage and capability guards have their own checks. `installed_runtime` is null until native preflight establishes it; a checkout identity is not an installed binary identity |
| `diagnostics` | Rule, severity (`error`, `advisory`, `unverified`), observed message/location, evidence and next action. Counterexample waivers remain explicit |
| `context` | Selected maintained reference sections and example identities. Query returns the complete pilot kernel; execution feedback links it without repeatedly including its full source |
| `related_contracts` | Bounded conditional owner routes with query IDs. Query routes distinguish `match: symbol` from natural-language `match: discovery`; neither proves an API declaration or native support. After an unclassified failure, `source_matches` bind statically resolved API references to checked source hashes/lines; `contract_review_scope` states that these are reviews rather than a diagnosis |
| `coverage`, `transfers` | What the phase inspected and its unresolved scope. Source descriptors do not prove producer bytes, actual loop counts or synchronization; native descriptors do not prove runtime access footprints |
| `validation` | Native parse, compilation, numerical and performance states are separate. An arbitrary child exit of zero does not establish any of the last three; source drift invalidates the result |
| `command` | Whether the command started and its original exit status; preflight rejection never launches it |

When the task supplies an independent verifier, keep its source-bound receipt separate from the generic runner's `validation` fields. Report the stages established by that verifier's declared contract, naming its source identity, execution target and covered cases. A generic `not_established` field does not erase a separate verified receipt; conversely, exit zero or an untyped `passed` string is not such a receipt. CPU parsing does not establish device execution, and numerical acceptance does not establish performance. Do not repeat a completed verification merely to clear a generic unknown field.

Native preflight reads structured exception fields when available. Ordinary subprocess diagnostics are matched only from bounded diagnostic lines, excluding quoted Python/pytest source. Recognized rank and transfer/store-contract symptoms select a next check; generic move/insert errors select UB→L1 context only when a source transfer at the reported location establishes that path. NZ pitch and direct-store diagnostics carry their own narrower context. Error numbers alone do not establish a root cause. Unknown framework diagnostics retain their module/code/location without an unrelated recipe; unmatched failures retain a bounded raw tail. The runner cannot prove that an arbitrary command imports every checked source or uses the preflight specialization: keep its public entry, wrappers, arguments and evaluator aligned.

For an unclassified rejection without selected diagnostic context, the runner can offer up to three owner-declared reviews from source API references. Namespace parameters and rebinding do not count as imported API references. A route may involve an unused helper or several unrelated operations; matching symbols do not prove that they executed, share a dataflow edge, or caused the failure. Read its `when` condition and query the applicable owner. Cast reviews require both a conversion and a corresponding load/store reference, rather than every ordinary load receiving a conversion recipe. Missing or changed source/spec identities suppress these routes; they never change the original status or numerical/performance acceptance.

```bash
PYPTOPRO_INFO=skipped-checks python -m pytest <target> -W error::pypto_pro._errors.CheckSkipped
```

Enable both the environment gate and the pytest filter. Bare `python -W` may fail to import the custom category during startup and cannot replace this command. `PYPTOPRO_INFO=all` enables the existing categories; it does not mean that layout or resource reports exist.

<!-- kernel-query: {"id":"execution-environment","aliases":["execution-environment","verification transport","SSH verification failure","验证通道失败"],"when":"Verification output reports a transport or import boundary, rather than a kernel result.","rules":["execution-transport-failed","execution-import-failed"]} -->
### Transport and import failures

Preserve the original stderr, exit status, interpreter and request identity. A reported SSH connection, authentication, host-key or control-socket failure does not establish a kernel API or numerical failure. If the controller confirms that no kernel request was dispatched, independently established source defects may still be repaired; keep the new source unverified until execution is available. If dispatch or result delivery is unknown, preserve the outstanding request and its checked source, reconcile its process/artifacts before replay, and stage independent edits separately. A missing reply is not proof that execution never started. Transport evidence alone never justifies changing kernel arithmetic.

For optional SSH connection sharing, use a per-command configuration that fits the allowed filesystem; creating a control socket outside writable roots can fail even when direct TCP authentication works. Keep global configuration unchanged. An import failure instead requires checking the execution interpreter, package path and named symbol against the installed version; do not infer general API absence or change arithmetic precision. These reported failures select this context instead of speculative source-API reviews. The runner still cannot authenticate arbitrary child output or prove which remote code ran.

## Spans and error matching

Runtime enrichment preserves the reported message, exception/code/module, location and observed stage separately. Known `entry_import` or `argument_binding` failures do not select a kernel-body parser repair; plain logs carry stage `unknown`. Framework error headers are separated from the diagnostic body for matching while the complete original message remains available. Entry classification, annotation parsing and installed support remain separate questions; see [entry classification](../api/constraints.md#check-the-imported-entry-before-changing-its-layout).

For transfer feedback, `condition_status=unverified` means that no unique compatible source call has been bound. A shared line with several compatible calls remains ambiguous. `source_site_matched` identifies the compared source descriptors; only a matching Vec→Vec NZ move with a positive, non-16-aligned physical row count selects its pitch rule. This does not extend that rejection to Vec→Mat insert, prove producer bytes or establish synchronization. A reported failure can retain severity `error` while its specific cause remains unknown. Read-only `explain --error-log` shares this routing with execution feedback; it does not authenticate a log or establish a new execution result.

- Exceptions raised during DSL parsing carry the user's source location through `current_span`; `parse_op_call` covers both automatic and handwritten handlers.
- When a span exists, pass it using `span=` instead of stitching it into the message. When there is no location, use `None`; do not block ambient fallback with an unknown span.
- For ordinary runtime-call errors, first check whether the traceback already contains a user frame; do not batch-fix based on the “number of missing spans.”
- Use `e.error_code/e.span/e.module/e.origin`; parameter details and expected/actual values still need to be read from the message. Do not assume those are already structured fields.
- A source excerpt can cause `pytest.raises(match=)` to match the test's own regular expression. The UT `conftest.py` strips excerpts, and the regression is in `test_match_guard.py`; ST does not share this `conftest.py`. For new diagnostics, verify that the span points to the user's line and use a reverse comparison to show that the criterion can fail.

See [parse-error localization and regressions](#parse-error-localization-and-regressions) for a specific case of “caught but not clearly explained.” Do not retain counts of how many batches were fixed or how many `raise` sites have spans.

## Minimum contract if a resource report is started

For each tile, report its space, address range, byte size, mutex IDs, and depth, plus the capacity actually used by each space. Keep output silent by default and export on request, binding it to the source and toolchain; do not equate declared size with peak live allocation. See [capacity conventions](../api/memory-limits.md) for capacity sources.

`ir.IRVerifier` exposes batched `Diagnostic` objects whose field shape can serve as a reference, but historical probes confirmed that `Diagnostic` has no Python constructor, Python cannot register rules, and severity does not include Info. Probe capabilities first; do not rely on it as a generic report container. The IR `text_dump` printer can read layouts, so the report is a convenience for consumers, not a prerequisite for allocator correctness.

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
