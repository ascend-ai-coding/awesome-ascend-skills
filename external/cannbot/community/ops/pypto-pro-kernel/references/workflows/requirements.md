# Requirements and optional specifications

Use this when the mathematical contract is unclear or a task needs a written specification. Reuse the user's requirements, supplied code, tests and evaluator; distinguish their declared support domain from the cases sampled by a test. Existing evidence can establish the contract without generating another document.

## Resolve the contract

- Establish the mathematical operation, public input/output order, shapes, dtypes, layouts, dynamic axes and optional-argument semantics. Preserve stateful behavior, aliasing, rounding and nonfinite inputs when the contract includes them.
- Identify the independent reference, comparison criteria and meaningful acceptance boundaries using [reference and evaluator checks](authoring.md#reference-and-evaluator). Keep implementation choices separate from externally required behavior.
- Trace non-obvious requirements to the supplied source. Resolve conflicting evidence and ask only about missing facts that affect implementation or acceptance; continue work supported by the known contract.
- Keep user performance targets explicit, including their metric and scope. A reference latency, pipe ratio or implementation preference does not supply an unstated performance target.

Return the resolved requirements and remaining uncertainty in the current task context or requested document, then continue with [authoring preflight](authoring.md#preflight-decisions). Do not require a separate specification or approval round when the task already supplies enough information.

## Optional SPEC format

Use [templates/spec-template.md](../../templates/spec-template.md) when the user requests a machine-readable specification, the project already uses this format, or the selected Golden generator needs it. Its single `json machine-contract` block contains the machine fields; prose explains semantics, supported domain and evidence without duplicating those values. Keep the requested destination and update authorization.

```bash
python scripts/validate_spec.py /path/to/SPEC.md
```

The checker validates the encoded contract and case consistency without importing PyPTO, running a reference or accessing a device. It does not establish mathematical correctness, kernel support or full-domain coverage. Its positive-dimension, finite-range and case encoding rules describe this optional schema; they are not universal Pro limitations. If the task cannot be represented faithfully, retain its original contract instead of inventing bounds or narrowing it to fit the template.

`p0_cases` are concrete acceptance configurations, not the complete support domain. Preserve the broader task contract in the specification and apply [generalization checks](authoring.md#generalization-and-specialization). Record special inputs explicitly, retain the original numerical criteria, and use `perf_target: null` and `exit_criteria: null` when no such target is requested.

The adjacent [exit_criteria.py](../../scripts/exit_criteria.py) validates and evaluates explicitly declared conditions against caller-supplied observations. Missing or ambiguous evidence remains unknown; the caller owns source identity and evidence authentication. This helper neither measures hardware nor determines a default stopping condition. Performance claims and optimization decisions follow the [core evidence and experiment rules](../tooling/optimization.md).
