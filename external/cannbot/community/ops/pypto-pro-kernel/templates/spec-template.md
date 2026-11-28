# Kernel specification

Use this optional format with the [requirements workflow](../references/workflows/requirements.md) when the task or selected generator needs a SPEC. Replace placeholders with confirmed facts and remove unused prose sections. The schema's encoding limits do not narrow the task's supported domain.

## Machine contract

```json machine-contract
{
  "schema_version": 1,
  "op_name": "{{OP_NAME}}",
  "formula": "{{FORMULA_OR_NUMBERED_STEPS}}",
  "supported_dtypes": [],
  "inputs": [],
  "outputs": [],
  "default_params": {},
  "tolerance": {
    "atol": null,
    "rtol": null
  },
  "dynamic_axes_ranges": {},
  "shape_constraints": [],
  "p0_cases": [],
  "perf_target": null,
  "exit_criteria": null
}
```

The single machine-contract block owns machine fields. Prose explains semantics and evidence without repeating those values.

- Use lower_snake_case for `op_name`. Define every public output in `formula` using reviewable assignments or numbered algorithm steps; a natural-language label alone is insufficient.
- Declare `inputs` and `outputs` in public signature order. Each entry has `name`, `shape`, `dtype`, a finite `value_range`, and optional `is_list`. Rank-zero tensors use `shape: []`. List the actual default and case dtypes in `supported_dtypes`, preserving first occurrence.
- Shapes use positive integers or symbolic expressions containing integers, `+`, `-`, `*`, `//` and parentheses. Each symbol needs a standalone input dimension and a positive integer interval in `dynamic_axes_ranges`. A rank-polymorphic tensor may use exactly `["..."]`; concrete case shapes remain explicit. These are schema limits, not restrictions on native Pro support.
- `default_params` contains confirmed public scalar defaults in signature order. Use `{}` when there are none. Nonfinite scalar values use JSON strings `"inf"`, `"+inf"`, `"-inf"` or `"nan"`.
- `tolerance` preserves the task's explicit criteria. Use confirmed `atol` and `rtol` for the elementwise form, with `integer_atol` only when requested. An existing explicit `{"policy":"pro_scheme_a"}` contract remains readable; do not select that policy as a default or replace a supplied evaluator with it.
- `p0_cases` contains at least one concrete acceptance configuration with `name`, `params`, `input_shapes` and `output_shapes`. Case keys follow the declarations' order. The first case uses the declared defaults. A dtype variation supplies both complete `input_dtypes` and `output_dtypes` mappings. Cases sample the supported domain; they do not define its full extent.
- Input `value_range` covers finite values. A case containing nonfinite tensor data declares `input_special_values` by input name using the actual `"-inf"`, `"+inf"` and `"nan"` values. Do not replace these inputs with finite samples.
- For TensorList entries, set `is_list: true`. Case shapes are nonempty arrays of member shapes, such as `[[8,64],[16,64]]`; member dtypes match, member ranks match and are at least one, and members are contiguous on one device. Special-value declarations cover the union across members. Keep the task's list-length and member-shape domain explicit. If the task needs an empty list or another unsupported encoding, retain its original contract instead of narrowing it to fit this format.
- `perf_target` and `exit_criteria` describe explicitly requested goals only. Use `null` when absent. A reference's latency or an observed pipe ratio does not create a default target. Schema validation does not establish that any goal has been met.

## Semantics and evidence

{{MATHEMATICS_INTERFACE_OPTIONAL_ARGUMENTS_STATE_AND_SOURCES}}

## Supported domain and acceptance

{{SUPPORTED_DOMAIN_BOUNDARIES_ROUNDING_NONFINITE_BEHAVIOR_AND_EVALUATOR}}

## Unresolved requirements

{{REMAINING_UNCERTAINTY_OR_NOT_APPLICABLE}}
