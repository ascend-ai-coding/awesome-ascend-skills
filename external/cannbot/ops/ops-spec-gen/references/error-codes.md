# Validator rule_id 目录

> 由 `scripts/dump_rule_ids.py` 自动生成；改动 validator 后跑 `--write` 同步。CI 由
> `tests/test_doc_drift.py` 守门：源文件中的 rule_id 与本文件不一致即 fail。

本目录列出 9-stage L0 校验器输出的所有 `rule_id` 字面量。新增 finding 时把字符串
登记到对应 stage；若是 evaluator（numpy AST 求值器）内部抛 `DslError`，code 会在 stages.py 拼成
`<stage>.<code>`，所以**新增 evaluator 错误码必须同时拓宽 SKILL.md §4.x 表**。

## Stage 2 category_paradigm_consistency

- `category_paradigm_consistency.fused_composite_basics`
- `category_paradigm_consistency.mutually_exclusive`
- `category_paradigm_consistency.required_paradigm_missing`

## Stage 2 paradigm_constraint

- `paradigm_constraint.argreduce_dtype`
- `paradigm_constraint.collective_hccl`
- `paradigm_constraint.dataflow_unclosed`
- `paradigm_constraint.dtype_chip_mismatch`
- `paradigm_constraint.error_code_not_declared`
- `paradigm_constraint.error_codes_undeclared`
- `paradigm_constraint.fused_composite_composition_missing`
- `paradigm_constraint.fused_composite_min_primitives`
- `paradigm_constraint.intermediate_leak`
- `paradigm_constraint.numerical_stable`
- `paradigm_constraint.primitive_not_whitelisted`
- `paradigm_constraint.quantization_attrs`
- `paradigm_constraint.random_sampling_seed`
- `paradigm_constraint.reduction_axis_missing`
- `paradigm_constraint.stateful_state_or_inplace`
- `paradigm_constraint.unknown_anti_pattern`
- `paradigm_constraint.unknown_chip`
- `paradigm_constraint.variable_output_flag`

## Stage 2 invariant_kind_resolved

- `invariant_kind_resolved.missing_field`
- `invariant_kind_resolved.tolerance_inherit_forbidden`
- `invariant_kind_resolved.tolerance_inherit_required`
- `invariant_kind_resolved.unknown_kind`

## Stage 3 shape_closure

- `shape_closure.data_dependent_flag_missing`
- `shape_closure.data_dependent_missing_bounds`
- `shape_closure.data_dependent_missing_description`
- `shape_closure.rank_overflow`
- `shape_closure.shape_rule_kind_missing`
- `shape_closure.shape_rule_kind_unknown`
- `shape_closure.unregistered_symbol`

## Stage 4 dtype_closure

- `dtype_closure.combination_mismatch`
- `dtype_closure.combination_missing_output`
- `dtype_closure.dtype_rule_kind_unknown`

## Stage 6 boundary_min_set

- `boundary_min_set.missing_required_case`

## Stage 7 tolerance_coverage

- `tolerance_coverage.tolerance_too_tight`
- `tolerance_coverage.uncovered_output_dtype`

## Stage 8 formula_smoke_eval

- `formula_smoke_eval.dtype_mismatch_at_runtime`
- `formula_smoke_eval.empty_formula`
- `formula_smoke_eval.missing_output`
- `formula_smoke_eval.no_combination`
- `formula_smoke_eval.numpy_eval_error`
- `formula_smoke_eval.numpy_not_installed`
- `formula_smoke_eval.output_not_array`
- `formula_smoke_eval.produces_unexpected_nan`
- `formula_smoke_eval.skipped_non_numpy`
- `formula_smoke_eval.syntax_error`

## Stage 9 oracle_reachable

- `oracle_reachable.absent`
- `oracle_reachable.api_framework_mismatch`
- `oracle_reachable.api_not_callable`
- `oracle_reachable.api_not_found`
- `oracle_reachable.composition_arg_unresolved`
- `oracle_reachable.composition_id_collision`
- `oracle_reachable.composition_id_shadows_input`
- `oracle_reachable.composition_node_invalid`
- `oracle_reachable.composition_output_unresolved`
- `oracle_reachable.dtype_unsupported`
- `oracle_reachable.framework_not_installed`
- `oracle_reachable.incomplete`
- `oracle_reachable.placeholder_unresolved`
- `oracle_reachable.unknown_framework`

## 其他 / 未归类

- `composition_single_primitive`
- `composition_without_fused_composite`
- `error_type_unknown`
- `machine_check_kind_unknown`
- `schema_static`
- `stage_skipped`
- `synthesize_legacy_format`
- `synthesize_pattern_unknown`

## Evaluator 错误码（`DslError(code, ...)`）

以下错误码由 `scripts/evaluators/` 内部抛出，被 stage 3-5 / 8 / 9 包装为 `<stage>.<code>`。

- `dsl_eval_error`
- `dsl_parse_error`
- `explicit_rules_uncovered`
- `folded_dim_misuse`
- `incompatible_dims`
- `numpy_violation`
- `unresolved_symbol`
