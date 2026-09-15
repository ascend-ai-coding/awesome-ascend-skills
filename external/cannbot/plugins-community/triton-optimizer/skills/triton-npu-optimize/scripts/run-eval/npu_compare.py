# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch


_FLOAT_FAMILY = {
    "float8_e4m3",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float16",
    "bfloat16",
    "float32",
    "float64",
}
_COMPLEX_FAMILY = {"complex64", "complex128"}
_INPUT_FLOAT_FAMILY = _FLOAT_FAMILY | _COMPLEX_FAMILY
_OUTPUT_FLOAT_FAMILY = _FLOAT_FAMILY | _COMPLEX_FAMILY
_INTEGER_FAMILY = {"bool", "uint8", "int8", "int16", "int32", "int64"}
_DTYPE_PRIORITY = {
    "complex128": 0,
    "complex64": 1,
    "float64": 2,
    "float32": 3,
    "float16": 4,
    "bfloat16": 5,
    "float8_e4m3": 6,
    "float8_e4m3fn": 6,
    "float8_e4m3fnuz": 6,
    "float8_e5m2": 7,
    "float8_e5m2fnuz": 7,
    "int64": 8,
    "int32": 9,
    "int16": 10,
    "int8": 11,
    "uint8": 12,
    "bool": 13,
}
_MATCH_THRESHOLDS = {
    "float16": {
        "small_value_threshold": 2 ** -11,
        "small_value_error": 2 ** -16,
        "rel_threshold": 2 ** -10,
    },
    "bfloat16": {
        "small_value_threshold": 2 ** -8,
        "small_value_error": 2 ** -16,
        "rel_threshold": 2 ** -7,
    },
    "float32": {
        "small_value_threshold": 2 ** -14,
        "small_value_error": 2 ** -30,
        "rel_threshold": 2 ** -13,
    },
    "hifloat32": {
        "small_value_threshold": 2 ** -12,
        "small_value_error": 2 ** -28,
        "rel_threshold": 2 ** -11,
    },
    "float8_e4m3": {
        "small_value_threshold": 2 ** -4,
        "small_value_error": 2 ** -6,
        "rel_threshold": 2 ** -3,
    },
    "float8_e5m2": {
        "small_value_threshold": 2 ** -3,
        "small_value_error": 2 ** -5,
        "rel_threshold": 2 ** -2,
    },
    "fallback": {
        "small_value_threshold": 2 ** -14,
        "small_value_error": 2 ** -30,
        "rel_threshold": 2 ** -13,
    },
}
_MAX_ERROR_THRESHOLDS = {
    "float16": {
        "atol": 9e-2,
        "rtol": 2 ** -10,
    },
    "bfloat16": {
        "atol": 1e-1,
        "rtol": 2 ** -7,
    },
    "float32": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
    "hifloat32": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
    "float8_e4m3": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
    "float8_e5m2": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
    "fallback": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
}


@dataclass(frozen=True)
class CaseCompareResult:
    passed: bool
    case_id: str
    compute: bool
    input_type: str
    input_dtype: str | None
    output_dtype: str | None
    comparison_path: str
    message: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class ArtifactCompareResult:
    passed: bool
    failed_case_count: int
    case_results: tuple[CaseCompareResult, ...]
    message: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class _InputInfo:
    input_type: str
    input_dtype: str | None


@dataclass(frozen=True)
class _CompareContext:
    case_id: str
    compute: bool
    input_info: _InputInfo
    output_path: str


@dataclass(frozen=True)
class _FloatingContext:
    case_id: str
    compute: bool
    input_info: _InputInfo
    output_path: str
    output_dtype: str
    finite_count: int
    thresholds: dict[str, float]
    max_diff: float
    max_index: tuple[int, ...] | tuple[()]


@dataclass(frozen=True)
class _TensorContext:
    actual: torch.Tensor
    golden: torch.Tensor
    case_id: str
    compute: bool
    input_info: _InputInfo
    output_path: str
    output_dtype: str
    comparison_path: str


@dataclass(frozen=True)
class _ResultContext:
    case_id: str
    compute: bool
    input_info: _InputInfo
    output_dtype: str | None
    comparison_path: str
    output_path: str


def compare_case_result(
    *,
    case_id: str,
    actual: object,
    golden: object,
    inputs: object,
    compute: bool = True,
) -> CaseCompareResult:
    input_info = _infer_input_info(inputs)
    leaf_results = _compare_value(
        actual=actual,
        golden=golden,
        context=_CompareContext(case_id, compute, input_info, "output"),
    )
    if not leaf_results:
        return _empty_output_case_result(case_id, compute, input_info)
    failed = next((result for result in leaf_results if not result.passed), None)
    if failed is not None:
        return failed
    return _passed_leaf_results_case_result(case_id, compute, input_info, leaf_results)


def _empty_output_case_result(
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
) -> CaseCompareResult:
    return _case_result(
        passed=True,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=None,
        comparison_path="empty-output",
        message=f"PASS case '{case_id}' has no comparable outputs.",
        diagnostics={"case_id": case_id, "output_path": "output"},
    )


def _passed_leaf_results_case_result(
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
    leaf_results: list[CaseCompareResult],
) -> CaseCompareResult:
    first = leaf_results[0]
    output_dtypes = {result.output_dtype for result in leaf_results}
    output_dtype = first.output_dtype if len(output_dtypes) == 1 else "multiple"
    comparison_paths = {result.comparison_path for result in leaf_results}
    comparison_path = first.comparison_path if len(comparison_paths) == 1 else "composite"
    return _case_result(
        passed=True,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
        message=f"PASS case '{case_id}' matched across {len(leaf_results)} output leaf/leaves.",
        diagnostics={
            "case_id": case_id,
            "compute": compute,
            "input_type": input_info.input_type,
            "input_dtype": input_info.input_dtype,
            "output_path": "output",
            "leaf_count": len(leaf_results),
        },
    )


def compare_result_payloads(
    oracle_payload: object,
    actual_payload: object,
) -> ArtifactCompareResult:
    oracle_cases, actual_cases, validation_failure = _validated_payload_cases(
        oracle_payload, actual_payload
    )
    if validation_failure is not None:
        return validation_failure
    oracle_compute = _payload_compute_flag(oracle_payload)
    actual_compute = _payload_compute_flag(actual_payload)
    if oracle_compute != actual_compute:
        return _compute_kind_mismatch(oracle_compute, actual_compute)
    return _summarize_payload_case_results(
        _compare_payload_cases(oracle_cases, actual_cases, oracle_compute)
    )


def _validated_payload_cases(
    oracle_payload: object,
    actual_payload: object,
) -> tuple[list[_CaseRecord], list[_CaseRecord], ArtifactCompareResult | None]:
    oracle_cases, oracle_error = _extract_case_records(oracle_payload, "oracle")
    if oracle_error is not None:
        return [], [], _payload_validation_failure("oracle", oracle_error)
    actual_cases, actual_error = _extract_case_records(actual_payload, "actual_payload")
    if actual_error is not None:
        return [], [], _payload_validation_failure("actual_payload", actual_error)
    if len(oracle_cases) != len(actual_cases):
        failure = ArtifactCompareResult(
            passed=False, failed_case_count=1, case_results=(),
            message=("FAIL: payload case count mismatch: "
                     f"oracle={len(oracle_cases)}, actual_payload={len(actual_cases)}"),
            diagnostics={"oracle_case_count": len(oracle_cases), "actual_case_count": len(actual_cases)},
        )
        return [], [], failure
    return oracle_cases, actual_cases, None


def _payload_validation_failure(label: str, message: str) -> ArtifactCompareResult:
    return ArtifactCompareResult(
        passed=False, failed_case_count=1, case_results=(),
        message=f"FAIL: {message}", diagnostics={"payload": label},
    )


def _compute_kind_mismatch(oracle_compute: bool, actual_compute: bool) -> ArtifactCompareResult:
    return ArtifactCompareResult(
        passed=False, failed_case_count=1, case_results=(),
        message=("FAIL: payload compute-kind mismatch: "
                 f"oracle={oracle_compute}, actual_payload={actual_compute}"),
        diagnostics={"oracle_compute": oracle_compute, "actual_compute": actual_compute},
    )


def _compare_payload_cases(
    oracle_cases: list[_CaseRecord],
    actual_cases: list[_CaseRecord],
    compute: bool,
) -> list[CaseCompareResult]:
    results: list[CaseCompareResult] = []
    for index, (oracle_case, actual_case) in enumerate(zip(oracle_cases, actual_cases)):
        if oracle_case.case_id == actual_case.case_id:
            results.append(compare_case_result(
                case_id=oracle_case.case_id, actual=actual_case.result, golden=oracle_case.result,
                inputs=oracle_case.inputs, compute=compute,
            ))
        else:
            results.append(_payload_case_order_mismatch(oracle_case, actual_case, index, compute))
    return results


def _payload_case_order_mismatch(
    oracle_case: _CaseRecord,
    actual_case: _CaseRecord,
    index: int,
    compute: bool,
) -> CaseCompareResult:
    return _case_result(
        passed=False, case_id=oracle_case.case_id, compute=compute,
        input_info=_infer_input_info(oracle_case.inputs), output_dtype=None,
        comparison_path="payload-case-order",
        message=("FAIL payload case order mismatch at index "
                 f"{index}: oracle={oracle_case.case_id!r}, actual_payload={actual_case.case_id!r}"),
        diagnostics={"failure_stage": "case_id_mismatch", "index": index,
                     "oracle_case_id": oracle_case.case_id, "actual_case_id": actual_case.case_id},
    )


def _summarize_payload_case_results(case_results: list[CaseCompareResult]) -> ArtifactCompareResult:
    failed = [result for result in case_results if not result.passed]
    if failed:
        return ArtifactCompareResult(
            passed=False, failed_case_count=len(failed), case_results=tuple(case_results),
            message=(f"FAIL: {len(failed)} of {len(case_results)} case(s) failed. "
                     f"First failure: {failed[0].message}"),
            diagnostics={"case_count": len(case_results)},
        )
    return ArtifactCompareResult(
        passed=True, failed_case_count=0, case_results=tuple(case_results),
        message=f"PASS: all {len(case_results)} case(s) matched the NPU accuracy contract.",
        diagnostics={"case_count": len(case_results)},
    )


def format_artifact_compare_result(result: ArtifactCompareResult) -> str:
    return "\n".join([result.message, *_format_failed_case_results(result)])


def _format_failed_case_results(result: ArtifactCompareResult) -> list[str]:
    if result.passed:
        return []
    lines: list[str] = []
    for case_result in result.case_results:
        if case_result.passed:
            continue
        lines.extend(_format_failed_case(case_result))
    return lines


def _format_failed_case(case_result: CaseCompareResult) -> list[str]:
    lines = [case_result.message]
    diagnostics = case_result.diagnostics
    threshold_info = diagnostics.get("thresholds")
    if isinstance(threshold_info, Mapping):
        lines.append(f"  thresholds={dict(cast(Mapping[object, object], threshold_info))}")
    lines.append(f"  diagnostics={dict(diagnostics)}")
    return lines


@dataclass(frozen=True)
class _CaseRecord:
    case_id: str
    inputs: object
    result: object


def _extract_case_records(payload: object, label: str) -> tuple[list[_CaseRecord], str | None]:
    if not isinstance(payload, Mapping):
        return [], f"{label} payload must be a dict with a 'cases' entry"
    payload_map = cast(Mapping[str, object], payload)
    if "results" in payload_map:
        return [], (
            f"{label} payload uses the legacy payload format. "
            "Expected {'compute': <bool>, 'cases': [...]} instead of {'results': [...]}."
        )
    raw_cases = payload_map.get("cases")
    if not isinstance(raw_cases, list):
        return [], f"{label} payload 'cases' must be a list"
    records: list[_CaseRecord] = []
    for raw_case in cast(list[object], raw_cases):
        if not isinstance(raw_case, Mapping):
            return [], f"{label} payload cases must be mappings"
        case_map = cast(Mapping[str, object], raw_case)
        case_id = case_map.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            return [], f"{label} payload case is missing required string field 'id'"
        if "inputs" not in case_map:
            return [], f"{label} payload case '{case_id}' is missing required field 'inputs'"
        if "result" not in case_map:
            return [], f"{label} payload case '{case_id}' is missing required field 'result'"
        records.append(
            _CaseRecord(
                case_id=case_id,
                inputs=case_map["inputs"],
                result=case_map["result"],
            )
        )
    return records, None


def _payload_compute_flag(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return True
    payload_map = cast(Mapping[str, object], payload)
    raw_compute = payload_map.get("compute")
    if raw_compute is None:
        return True
    if isinstance(raw_compute, bool):
        return raw_compute
    return True


def _compare_value(*, actual: object, golden: object, context: _CompareContext) -> list[CaseCompareResult]:
    if isinstance(golden, Mapping):
        return _compare_mapping_value(actual, golden, context)
    if _is_sequence_output(golden):
        return _compare_sequence_value(actual, golden, context)
    return [_compare_leaf(actual, golden, context)]


def _structure_failure(context: _ResultContext, **kwargs: object) -> list[CaseCompareResult]:
    expected = str(kwargs["expected"])
    actual = kwargs["actual"]
    stage = str(kwargs["stage"])
    message = str(kwargs["message"])
    extra = kwargs.get("extra")
    extra_map = extra if isinstance(extra, dict) else None
    diagnostics: dict[str, object] = {"failure_stage": stage, "output_path": context.output_path}
    if extra_map is None:
        diagnostics.update({"expected_type": expected, "actual_type": type(actual).__name__})
    else:
        diagnostics.update(extra_map)
    return [
        _case_result(
            passed=False,
            case_id=context.case_id,
            compute=context.compute,
            input_info=context.input_info,
            output_dtype=context.output_dtype,
            comparison_path=context.comparison_path,
            message=message,
            diagnostics=diagnostics,
        )
    ]


def _structure_failure_for_compare(
    context: _CompareContext, **kwargs: object,
) -> list[CaseCompareResult]:
    return _structure_failure(
        _ResultContext(
            context.case_id,
            context.compute,
            context.input_info,
            None,
            "structure-mismatch",
            context.output_path,
        ),
        **kwargs,
    )


def _compare_mapping_value(
    actual: object,
    golden: Mapping[object, object],
    context: _CompareContext,
) -> list[CaseCompareResult]:
    if not isinstance(actual, Mapping):
        return _structure_failure_for_compare(
            context,
            expected="mapping",
            actual=actual,
            stage="type_mismatch",
            message=(
                f"FAIL case '{context.case_id}' expected mapping at "
                f"{context.output_path}, got {type(actual).__name__}."
            ),
        )
    actual_map = cast(Mapping[object, object], actual)
    golden_keys, actual_keys = set(golden), set(actual_map)
    if golden_keys != actual_keys:
        return _structure_failure_for_compare(
            context,
            expected="mapping",
            actual=actual,
            stage="key_mismatch",
            message=f"FAIL case '{context.case_id}' mapping keys differ at {context.output_path}.",
            extra={"expected_keys": sorted(golden_keys, key=str), "actual_keys": sorted(actual_keys, key=str)},
        )
    results: list[CaseCompareResult] = []
    for key in sorted(golden_keys, key=str):
        child = replace(context, output_path=f"{context.output_path}.{key}")
        results.extend(_compare_value(actual=actual_map[key], golden=golden[key], context=child))
    return results


def _compare_sequence_value(
    actual: object,
    golden: Sequence[object],
    context: _CompareContext,
) -> list[CaseCompareResult]:
    if not _is_sequence_output(actual):
        return _structure_failure_for_compare(
            context,
            expected="sequence",
            actual=actual,
            stage="type_mismatch",
            message=(
                f"FAIL case '{context.case_id}' expected sequence at "
                f"{context.output_path}, got {type(actual).__name__}."
            ),
        )
    actual_seq, golden_seq = list(cast(Sequence[object], actual)), list(golden)
    if len(actual_seq) != len(golden_seq):
        return _structure_failure_for_compare(
            context,
            expected="sequence",
            actual=actual,
            stage="length_mismatch",
            message=f"FAIL case '{context.case_id}' sequence length differs at {context.output_path}.",
            extra={"expected_length": len(golden_seq), "actual_length": len(actual_seq)},
        )
    results: list[CaseCompareResult] = []
    for index, (actual_item, golden_item) in enumerate(zip(actual_seq, golden_seq)):
        child = replace(context, output_path=f"{context.output_path}[{index}]")
        results.extend(_compare_value(actual=actual_item, golden=golden_item, context=child))
    return results


def _compare_leaf(actual: object, golden: object, context: _CompareContext) -> CaseCompareResult:
    actual_tensor = _coerce_output_leaf(actual)
    golden_tensor = _coerce_output_leaf(golden)
    if actual_tensor is None or golden_tensor is None:
        return _compare_non_tensor_leaf(actual, golden, context)
    tensor_context, output_dtype = _prepare_tensor_context(actual_tensor, golden_tensor, context)
    if tensor_context is None:
        return _unsupported_dtype_result(
            context.case_id,
            context.compute,
            context.input_info,
            context.output_path,
            output_dtype,
        )
    return _compare_tensor_context(tensor_context)


def _prepare_tensor_context(
    actual: torch.Tensor,
    golden: torch.Tensor,
    context: _CompareContext,
) -> tuple[_TensorContext | None, str]:
    actual_cpu = actual.detach().cpu()
    golden_cpu = golden.detach().cpu()
    output_dtype = _dtype_name(golden_cpu.dtype)
    comparison_path = _select_comparison_path(
        compute=context.compute,
        input_type=context.input_info.input_type,
        output_dtype=output_dtype,
    )
    if comparison_path is None:
        return None, output_dtype
    return _TensorContext(
        actual_cpu,
        golden_cpu,
        context.case_id,
        context.compute,
        context.input_info,
        context.output_path,
        output_dtype,
        comparison_path,
    ), output_dtype


def _compare_tensor_context(context: _TensorContext) -> CaseCompareResult:
    if context.comparison_path == "non-compute":
        return _compare_non_compute(context.actual, context.golden, context)
    precheck_failure = _run_prechecks(context.actual, context.golden, context)
    if precheck_failure is not None:
        return precheck_failure
    return _compare_numeric_leaf(context)


def _compare_non_tensor_leaf(
    actual: object, golden: object, context: _CompareContext,
) -> CaseCompareResult:
    case_id = context.case_id
    compute = context.compute
    input_info = context.input_info
    output_path = context.output_path
    if not compute and actual == golden:
        return _case_result(
            passed=True, case_id=case_id, compute=compute, input_info=input_info, output_dtype=None,
            comparison_path="non-compute", message=f"PASS case '{case_id}' matched non-tensor output at {output_path}.",
            diagnostics={"output_path": output_path},
        )
    return _case_result(
        passed=False, case_id=case_id, compute=compute, input_info=input_info, output_dtype=None,
        comparison_path="unsupported-output-type",
        message=(f"FAIL case '{case_id}' produced unsupported output types at {output_path}: "
                 f"expected {type(golden).__name__}, got {type(actual).__name__}."),
        diagnostics={"failure_stage": "unsupported_output_type", "output_path": output_path,
                     "expected_type": type(golden).__name__, "actual_type": type(actual).__name__},
    )


def _unsupported_dtype_result(
    case_id: str, compute: bool, input_info: _InputInfo, output_path: str, output_dtype: str,
) -> CaseCompareResult:
    return _case_result(
        passed=False, case_id=case_id, compute=compute, input_info=input_info, output_dtype=output_dtype,
        comparison_path="unsupported-output-type",
        message=f"FAIL case '{case_id}' output dtype '{output_dtype}' is unsupported at {output_path}.",
        diagnostics={"failure_stage": "unsupported_output_type", "output_path": output_path,
                     "output_dtype": output_dtype},
    )


def _compare_numeric_leaf(
    context: _TensorContext,
) -> CaseCompareResult:
    actual_cpu, golden_cpu = context.actual, context.golden
    case_id = context.case_id
    compute = context.compute
    input_info = context.input_info
    output_path = context.output_path
    output_dtype, comparison_path = context.output_dtype, context.comparison_path
    if comparison_path == "bool-output":
        return _compare_bool_output(
            actual_cpu=actual_cpu, golden_cpu=golden_cpu, context=context,
        )
    if comparison_path in {"integer-compute", "quantized-fp-to-int"}:
        return _compare_integer_output(
            actual_cpu=actual_cpu, golden_cpu=golden_cpu, context=context,
            bound=1 if comparison_path == "quantized-fp-to-int" else 0,
        )
    return _compare_floating_output(
        context=context,
    )


def _compare_non_compute(
    actual_cpu: torch.Tensor, golden_cpu: torch.Tensor, context: _TensorContext,
) -> CaseCompareResult:
    structural_failure = _non_compute_structure_failure(
        actual_cpu, golden_cpu, context
    )
    if structural_failure is not None:
        return structural_failure
    actual_bytes = actual_cpu.contiguous().view(torch.uint8)
    golden_bytes = golden_cpu.contiguous().view(torch.uint8)
    return _non_compute_bitwise_result(
        actual_bytes, golden_bytes, context
    )


def _non_compute_structure_failure(
    actual_cpu: torch.Tensor, golden_cpu: torch.Tensor, context: _TensorContext,
) -> CaseCompareResult | None:
    if tuple(actual_cpu.shape) != tuple(golden_cpu.shape):
        return _shape_mismatch_result(
            case_id=context.case_id,
            compute=context.compute,
            input_info=context.input_info,
            output_path=context.output_path,
            output_dtype=context.output_dtype, comparison_path="non-compute", actual_shape=tuple(actual_cpu.shape),
            golden_shape=tuple(golden_cpu.shape),
        )
    if actual_cpu.dtype == golden_cpu.dtype:
        return None
    return _case_result(
        passed=False,
        case_id=context.case_id,
        compute=context.compute,
        input_info=context.input_info,
        output_dtype=context.output_dtype,
        comparison_path="non-compute",
        message=f"FAIL case '{context.case_id}' dtype mismatch at {context.output_path} for non-compute comparison.",
        diagnostics={"failure_stage": "dtype_mismatch", "output_path": context.output_path,
                     "expected_dtype": _dtype_name(golden_cpu.dtype), "actual_dtype": _dtype_name(actual_cpu.dtype)},
    )


def _non_compute_bitwise_result(
    actual_bytes: torch.Tensor, golden_bytes: torch.Tensor, context: _TensorContext,
) -> CaseCompareResult:
    if torch.equal(actual_bytes, golden_bytes):
        return _case_result(
            passed=True,
            case_id=context.case_id,
            compute=context.compute,
            input_info=context.input_info,
            output_dtype=context.output_dtype,
            comparison_path="non-compute",
            message=(
                f"PASS case '{context.case_id}' matched raw bits at "
                f"{context.output_path}."
            ),
            diagnostics={"output_path": context.output_path},
        )
    return _case_result(
        passed=False,
        case_id=context.case_id,
        compute=context.compute,
        input_info=context.input_info,
        output_dtype=context.output_dtype,
        comparison_path="non-compute",
        message=f"FAIL case '{context.case_id}' raw bits differ at {context.output_path}.",
        diagnostics={
            "failure_stage": "binary_equal",
            "output_path": context.output_path,
            "expected_dtype": _dtype_name(context.golden.dtype),
            "actual_dtype": _dtype_name(context.actual.dtype),
        },
    )


def _run_prechecks(
    actual_cpu: torch.Tensor, golden_cpu: torch.Tensor, context: _TensorContext,
) -> CaseCompareResult | None:
    case_id = context.case_id
    compute = context.compute
    input_info = context.input_info
    output_path = context.output_path
    output_dtype, comparison_path = context.output_dtype, context.comparison_path
    if tuple(actual_cpu.shape) != tuple(golden_cpu.shape):
        return _shape_mismatch_result(
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path=output_path,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            actual_shape=tuple(actual_cpu.shape),
            golden_shape=tuple(golden_cpu.shape),
        )
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    return _special_value_precheck(actual_cast, golden_cpu, context)


def _special_value_precheck(
    actual_cast: torch.Tensor, golden_cpu: torch.Tensor, context: _TensorContext,
) -> CaseCompareResult | None:
    case_id = context.case_id
    compute = context.compute
    input_info = context.input_info
    output_path = context.output_path
    output_dtype, comparison_path = context.output_dtype, context.comparison_path
    nan_mask_actual = _nan_mask(actual_cast)
    nan_mask_golden = _nan_mask(golden_cpu)
    if not torch.equal(nan_mask_actual, nan_mask_golden):
        return _special_mask_mismatch(
            actual_mask=nan_mask_actual, golden_mask=nan_mask_golden, name="NaN", stage="nan_mask_mismatch",
            context=context,
        )
    inf_mask_actual = _inf_mask(actual_cast)
    inf_mask_golden = _inf_mask(golden_cpu)
    if not torch.equal(inf_mask_actual, inf_mask_golden):
        return _special_mask_mismatch(
            actual_mask=inf_mask_actual, golden_mask=inf_mask_golden, name="Inf", stage="inf_mask_mismatch",
            context=context,
        )
    return _inf_sign_mismatch(
        actual_cast, golden_cpu, inf_mask_golden,
        context,
    )


def _special_mask_mismatch(
    *, actual_mask: torch.Tensor, golden_mask: torch.Tensor, name: str, stage: str,
    context: _TensorContext,
) -> CaseCompareResult:
    mismatch_count = int(torch.count_nonzero(actual_mask != golden_mask).item())
    return _case_result(
        passed=False,
        case_id=context.case_id,
        compute=context.compute,
        input_info=context.input_info,
        output_dtype=context.output_dtype,
        comparison_path=context.comparison_path,
        message=(
            f"FAIL case '{context.case_id}' {name} locations differ at "
            f"{context.output_path}."
        ),
        diagnostics={"failure_stage": stage, "output_path": context.output_path,
                     f"{name.lower()}_mismatch_count": mismatch_count},
    )


def _inf_sign_mismatch(
    actual_cast: torch.Tensor, golden_cpu: torch.Tensor, inf_mask: torch.Tensor,
    context: _TensorContext,
) -> CaseCompareResult | None:
    if int(torch.count_nonzero(inf_mask).item()) == 0:
        return None
    if torch.equal(actual_cast[inf_mask], golden_cpu[inf_mask]):
        return None
    return _case_result(
            passed=False,
            case_id=context.case_id,
            compute=context.compute,
            input_info=context.input_info,
            output_dtype=context.output_dtype,
            comparison_path=context.comparison_path,
            message=f"FAIL case '{context.case_id}' Inf values/signs differ at {context.output_path}.",
            diagnostics={
                "failure_stage": "inf_sign_mismatch",
                "output_path": context.output_path,
                "inf_mismatch_count": int(
                    torch.count_nonzero(actual_cast[inf_mask] != golden_cpu[inf_mask]).item()
                ),
            },
        )


def _compare_bool_output(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    context: _TensorContext,
) -> CaseCompareResult:
    case_id = context.case_id
    compute = context.compute
    input_info = context.input_info
    output_path = context.output_path
    output_dtype = context.output_dtype
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    if torch.equal(actual_cast, golden_cpu):
        return _case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path="bool-output",
            message=f"PASS case '{case_id}' matched bool output at {output_path}.",
            diagnostics={"output_path": output_path},
        )
    return _case_result(
        passed=False,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path="bool-output",
        message=f"FAIL case '{case_id}' bool outputs differ at {output_path}.",
        diagnostics={
            "failure_stage": "bool_equal",
            "output_path": output_path,
        },
    )


def _compare_integer_output(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    bound: int,
    context: _TensorContext,
) -> CaseCompareResult:
    actual_cpu, golden_cpu = context.actual, context.golden
    case_id = context.case_id
    compute = context.compute
    input_info = context.input_info
    output_path = context.output_path
    output_dtype = context.output_dtype
    comparison_path = context.comparison_path
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    diff = (actual_cast.to(dtype=torch.int64) - golden_cpu.to(dtype=torch.int64)).abs()
    max_diff = int(diff.max().item()) if diff.numel() > 0 else 0
    if max_diff <= bound:
        return _case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"PASS case '{case_id}' matched integer output at {output_path}.",
            diagnostics={
                "output_path": output_path,
                "max_abs_diff": max_diff,
                "error_bound": bound,
            },
        )
    flat_index = int(torch.argmax(diff).item())
    return _case_result(
        passed=False,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
        message=(
            f"FAIL case '{case_id}' integer comparison exceeded the error bound at {output_path}: "
            f"max_abs_diff={max_diff}, bound={bound}."
        ),
        diagnostics={
            "failure_stage": "integer_error_bound",
            "output_path": output_path,
            "max_abs_diff": max_diff,
            "max_abs_diff_index": _unravel_index(flat_index, tuple(golden_cpu.shape)),
            "error_bound": bound,
        },
    )


def _compare_floating_output(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    context: _TensorContext,
) -> CaseCompareResult:
    actual_cpu, golden_cpu = context.actual, context.golden
    case_id = context.case_id
    compute = context.compute
    input_info = context.input_info
    output_path = context.output_path
    output_dtype = context.output_dtype
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    finite_mask = torch.isfinite(actual_cast) & torch.isfinite(golden_cpu)
    finite_count = int(torch.count_nonzero(finite_mask).item())
    thresholds = _thresholds_for_output_dtype(output_dtype)
    if finite_count:
        return _compare_finite_floating_values(
            actual_cast,
            golden_cpu,
            finite_mask,
            _TensorContext(
                actual_cast,
                golden_cpu,
                case_id,
                compute,
                input_info,
                output_path,
                output_dtype,
                "floating-point-compute",
            ),
        )
    return _case_result(
        passed=True,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path="floating-point-compute",
        message=f"PASS case '{case_id}' had no finite floating-point elements at {output_path}.",
        diagnostics={
            "output_path": output_path,
            "finite_count": 0,
            "matched_ratio": 1.0,
            "mere": 0.0,
            "mere_threshold": thresholds["rel_threshold"],
            "output_dtype": output_dtype,
            "thresholds": dict(thresholds),
        },
    )


def _compare_finite_floating_values(
    actual_cast: torch.Tensor, golden_cpu: torch.Tensor, finite_mask: torch.Tensor,
    context: _TensorContext,
) -> CaseCompareResult:
    finite_count = int(torch.count_nonzero(finite_mask).item())
    thresholds = _thresholds_for_output_dtype(context.output_dtype)
    case_id = context.case_id
    compute = context.compute
    input_info = context.input_info
    output_path = context.output_path
    output_dtype = context.output_dtype
    actual_finite, golden_finite = _finite_float_views(actual_cast, golden_cpu, finite_mask)
    diff, abs_golden = (actual_finite - golden_finite).abs(), golden_finite.abs()
    finite_indices = finite_mask.nonzero(as_tuple=False)
    max_diff = float(diff.max().item())
    max_index = _indices_tensor_to_tuple(finite_indices[int(torch.argmax(diff).item())])
    context = _FloatingContext(
        case_id,
        compute,
        input_info,
        output_path,
        output_dtype,
        finite_count,
        thresholds,
        max_diff,
        max_index,
    )
    cap_failure = _floating_cap_failure(diff, abs_golden, finite_indices, context)
    if cap_failure is not None:
        return cap_failure
    return _floating_ratio_result(diff, abs_golden, context)


def _floating_cap_failure(
    diff: torch.Tensor, abs_golden: torch.Tensor, finite_indices: torch.Tensor,
    context: _FloatingContext,
) -> CaseCompareResult | None:
    error_cap = context.thresholds["atol"] + context.thresholds["rtol"] * abs_golden
    cap_mask = diff <= error_cap
    if bool(torch.all(cap_mask).item()):
        return None
    index = int(torch.nonzero(~cap_mask, as_tuple=False)[0].item())
    failing_index = _indices_tensor_to_tuple(finite_indices[index])
    return _case_result(
        passed=False,
        case_id=context.case_id,
        compute=context.compute,
        input_info=context.input_info,
        output_dtype=context.output_dtype,
        comparison_path="floating-point-compute",
        message=(f"FAIL case '{context.case_id}' max error cap failed at {context.output_path}: "
                 f"diff={float(diff[index].item())}, cap={float(error_cap[index].item())}."),
        diagnostics={
            "failure_stage": "max_error_cap",
            "output_path": context.output_path,
            "finite_count": context.finite_count,
            "max_abs_diff": context.max_diff,
            "max_abs_diff_index": context.max_index,
            "first_failing_index": failing_index,
            "max_error_cap_at_index": float(error_cap[index].item()),
            "output_dtype": context.output_dtype,
            "thresholds": dict(context.thresholds),
        },
    )


def _floating_ratio_result(
    diff: torch.Tensor, abs_golden: torch.Tensor, context: _FloatingContext,
) -> CaseCompareResult:
    small_mask = abs_golden < context.thresholds["small_value_threshold"]
    matched = torch.where(
        small_mask, diff <= context.thresholds["small_value_error"],
        diff / (abs_golden + 1e-7) <= context.thresholds["rel_threshold"],
    )
    matched_ratio = float(matched.to(dtype=torch.float32).mean().item())
    mere = float((diff / (abs_golden + 1e-7)).mean().item())
    stage = (
        "matched_ratio"
        if matched_ratio < 0.9
        else "mere"
        if mere >= context.thresholds["rel_threshold"]
        else None
    )
    if stage is None:
        return _floating_success(context, matched_ratio, mere)
    return _floating_ratio_failure(context, matched_ratio, mere, stage)


def _floating_success(
    context: _FloatingContext, matched_ratio: float, mere: float,
) -> CaseCompareResult:
    return _case_result(
        passed=True,
        case_id=context.case_id,
        compute=context.compute,
        input_info=context.input_info,
        output_dtype=context.output_dtype,
        comparison_path="floating-point-compute",
        message=f"PASS case '{context.case_id}' matched floating-point output at {context.output_path}.",
        diagnostics=_floating_diagnostics(context, matched_ratio, mere),
    )


def _floating_ratio_failure(
    context: _FloatingContext, matched_ratio: float, mere: float, stage: str,
) -> CaseCompareResult:
    if stage == "matched_ratio":
        message = (
            f"FAIL case '{context.case_id}' matched ratio failed at "
            f"{context.output_path}: matched_ratio={matched_ratio}."
        )
    else:
        message = (
            f"FAIL case '{context.case_id}' MERE failed at {context.output_path}: "
            f"mere={mere}, threshold={context.thresholds['rel_threshold']}."
        )
    return _case_result(
        passed=False,
        case_id=context.case_id,
        compute=context.compute,
        input_info=context.input_info,
        output_dtype=context.output_dtype,
        comparison_path="floating-point-compute", message=message,
        diagnostics=_floating_diagnostics(context, matched_ratio, mere, failure_stage=stage),
    )


def _floating_diagnostics(
    context: _FloatingContext,
    matched_ratio: float,
    mere: float,
    *,
    failure_stage: str | None = None,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    if failure_stage is not None:
        diagnostics["failure_stage"] = failure_stage
    diagnostics.update({
        "output_path": context.output_path,
        "finite_count": context.finite_count,
        "matched_ratio": matched_ratio,
        "mere": mere,
        "mere_threshold": context.thresholds["rel_threshold"],
        "max_abs_diff": context.max_diff,
        "max_abs_diff_index": context.max_index,
        "output_dtype": context.output_dtype,
        "thresholds": dict(context.thresholds),
    })
    return diagnostics


def _finite_float_views(
    actual_cast: torch.Tensor,
    golden_cpu: torch.Tensor,
    finite_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_finite = actual_cast[finite_mask]
    golden_finite = golden_cpu[finite_mask]
    if actual_finite.is_complex() or golden_finite.is_complex():
        return (
            torch.view_as_real(actual_finite.to(dtype=torch.complex64))
            .reshape(-1)
            .to(dtype=torch.float32),
            torch.view_as_real(golden_finite.to(dtype=torch.complex64))
            .reshape(-1)
            .to(dtype=torch.float32),
        )
    return (
        actual_finite.to(dtype=torch.float32),
        golden_finite.to(dtype=torch.float32),
    )


def _thresholds_for_output_dtype(output_dtype: str) -> dict[str, float]:
    matched_key = _threshold_key(output_dtype)
    matched = _MATCH_THRESHOLDS.get(matched_key, _MATCH_THRESHOLDS["fallback"])
    max_error = _MAX_ERROR_THRESHOLDS.get(matched_key, _MAX_ERROR_THRESHOLDS["fallback"])
    return {
        "small_value_threshold": float(matched["small_value_threshold"]),
        "small_value_error": float(matched["small_value_error"]),
        "rel_threshold": float(matched["rel_threshold"]),
        "atol": float(max_error["atol"]),
        "rtol": float(max_error["rtol"]),
    }


def _threshold_key(output_dtype: str) -> str:
    if output_dtype.startswith("float8_e4m3"):
        return "float8_e4m3"
    if output_dtype.startswith("float8_e5m2"):
        return "float8_e5m2"
    if output_dtype in {"float16", "bfloat16", "float32", "hifloat32"}:
        return output_dtype
    return "fallback"


def _select_comparison_path(
    *,
    compute: bool,
    input_type: str,
    output_dtype: str,
) -> str | None:
    if not compute:
        return "non-compute"
    if output_dtype == "bool":
        return "bool-output"
    if output_dtype in _INTEGER_FAMILY:
        if input_type == "float":
            return "quantized-fp-to-int"
        if input_type in {"int", "no_tensor"}:
            return "integer-compute"
        return None
    if output_dtype in _OUTPUT_FLOAT_FAMILY:
        return "floating-point-compute"
    return None


def _infer_input_info(inputs: object) -> _InputInfo:
    direct_values = _direct_input_values(inputs)
    direct_tensors = [value for value in direct_values if isinstance(value, torch.Tensor)]
    if direct_tensors:
        dtype_name = min(
            (_dtype_name(tensor.dtype) for tensor in direct_tensors),
            key=lambda name: _DTYPE_PRIORITY.get(name, len(_DTYPE_PRIORITY)),
        )
        return _InputInfo(
            input_type="float" if dtype_name in _INPUT_FLOAT_FAMILY else "int",
            input_dtype=dtype_name,
        )
    for value in direct_values:
        if isinstance(value, (list, tuple)):
            sequence_values = cast(list[object] | tuple[object, ...], value)
            tensor_items = [item for item in sequence_values if isinstance(item, torch.Tensor)]
            if tensor_items and len(tensor_items) == len(sequence_values):
                first_tensor = tensor_items[0]
                dtype_name = _dtype_name(first_tensor.dtype)
                return _InputInfo(
                    input_type="float" if dtype_name in _INPUT_FLOAT_FAMILY else "int",
                    input_dtype=dtype_name,
                )
    return _InputInfo(input_type="no_tensor", input_dtype=None)


def _direct_input_values(inputs: object) -> list[object]:
    if isinstance(inputs, Mapping):
        return list(cast(Mapping[object, object], inputs).values())
    if isinstance(inputs, tuple):
        return list(cast(tuple[object, ...], inputs))
    if isinstance(inputs, list):
        return list(cast(list[object], inputs))
    if inputs is None:
        return []
    return [inputs]


def _coerce_output_leaf(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, bool):
        return torch.tensor(value, dtype=torch.bool)
    if isinstance(value, int):
        return torch.tensor(value, dtype=torch.int64)
    if isinstance(value, float):
        return torch.tensor(value, dtype=torch.float64)
    if isinstance(value, complex):
        return torch.tensor(value, dtype=torch.complex64)
    return None


def _nan_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() or tensor.is_complex():
        return torch.isnan(tensor)
    return torch.zeros_like(tensor, dtype=torch.bool)


def _inf_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() or tensor.is_complex():
        return torch.isinf(tensor)
    return torch.zeros_like(tensor, dtype=torch.bool)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _is_sequence_output(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _indices_tensor_to_tuple(indices: torch.Tensor) -> tuple[int, ...]:
    flat_indices = indices.reshape(-1)
    return tuple(int(flat_indices[index].item()) for index in range(flat_indices.numel()))


def _unravel_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...] | tuple[()]:
    if not shape:
        return ()
    if len(shape) == 1:
        return (flat_index,)
    values: list[int] = []
    remaining = flat_index
    for size in reversed(shape):
        values.append(remaining % size)
        remaining //= size
    return tuple(reversed(values))


def _shape_mismatch_result(
    **kwargs: object,
) -> CaseCompareResult:
    case_id = str(kwargs["case_id"])
    compute = bool(kwargs["compute"])
    input_info = kwargs["input_info"]
    output_path = str(kwargs["output_path"])
    output_dtype = str(kwargs["output_dtype"])
    comparison_path = str(kwargs["comparison_path"])
    actual_shape = kwargs["actual_shape"]
    golden_shape = kwargs["golden_shape"]
    return _case_result(
        passed=False,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
        message=(
            f"FAIL case '{case_id}' shape mismatch at {output_path}: "
            f"expected {golden_shape}, got {actual_shape}."
        ),
        diagnostics={
            "failure_stage": "shape_mismatch",
            "output_path": output_path,
            "shape_expected": golden_shape,
            "shape_actual": actual_shape,
            "output_dtype": output_dtype,
        },
    )


def _case_result(
    *, passed: bool, **kwargs: object,
) -> CaseCompareResult:
    case_id = str(kwargs["case_id"])
    compute = bool(kwargs["compute"])
    input_info = kwargs["input_info"]
    output_dtype = kwargs.get("output_dtype")
    comparison_path = str(kwargs["comparison_path"])
    message = str(kwargs["message"])
    diagnostics = kwargs["diagnostics"]
    return CaseCompareResult(
        passed=passed,
        case_id=case_id,
        compute=compute,
        input_type=input_info.input_type,
        input_dtype=input_info.input_dtype,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
        message=message,
        diagnostics={
            "case_id": case_id,
            "compute": compute,
            "input_type": input_info.input_type,
            "input_dtype": input_info.input_dtype,
            **diagnostics,
        },
    )
