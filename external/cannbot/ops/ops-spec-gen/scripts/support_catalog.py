# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0; see LICENSE.
"""Read official dtype/format universes and metadata from op-spec.json."""

import json
from functools import lru_cache
from pathlib import Path


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas/op-spec.json"


@lru_cache(maxsize=1)
def load_schema() -> dict:
    """Share the same schema snapshot across generation and validation."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_catalog(schema: dict | None = None) -> dict:
    """Expose schema metadata in the catalog shape used by existing callers."""
    if schema is None:
        schema = load_schema()
    metadata = schema["x-support-catalog"]
    return {
        **metadata,
        "id": schema["properties"]["support_universe"]["const"],
        "dtypes": metadata["dtype_aliases"],
        "formats": schema["$defs"]["official_format"]["enum"],
    }


def canonical(value: str, kind: str) -> str:
    if kind == "dtypes":
        return load_catalog()["dtypes"].get(value, value)
    value = value.removeprefix("FORMAT_")
    return "FRACTAL_NZ" if value == "NZ" else value


def universe(kind: str) -> list[str]:
    definition = {"dtypes": "official_dtype", "formats": "official_format"}[kind]
    return list(load_schema()["$defs"][definition]["enum"])


def partition(
    supported: list[str], excluded: list[str], kind: str
) -> tuple[list, list]:
    """The caller declares support exhaustive; exclusions must agree with it."""
    supported = [canonical(value, kind) for value in supported]
    excluded = [canonical(value, kind) for value in excluded]
    all_values = universe(kind)
    unknown = (set(supported) | set(excluded)) - set(all_values)
    if unknown:
        raise ValueError(f"Outside official {kind} universe: {sorted(unknown)}")
    if not supported or len(supported) != len(set(supported)):
        raise ValueError(f"Supported {kind} must be nonempty and unique")
    if len(excluded) != len(set(excluded)):
        raise ValueError(f"Unsupported {kind} assertions must be unique")
    overlap = set(supported) & set(excluded)
    if overlap:
        raise ValueError(f"Conflicting supported/unsupported {kind}: {sorted(overlap)}")
    forbidden = set(load_catalog()[f"non_tensor_{kind}"]) & set(supported)
    if forbidden:
        raise ValueError(
            f"Non-tensor enum markers cannot be supported: {sorted(forbidden)}"
        )
    return supported, [value for value in all_values if value not in supported]
