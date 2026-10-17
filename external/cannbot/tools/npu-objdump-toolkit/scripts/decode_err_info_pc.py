#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Audit a caller-selected platform ERR_INFO rule and byte-PC calculation."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any


from cli_common import configure_logging, integer


ERROR_TYPES = ("CCU", "MTE", "IFU", "CUBE", "VEC")
COMPARISON_SCOPE = (
    "values are comparable only when ERR_INFO PC, base, and direct result use "
    "the same byte-address domain"
)
LOGGER = logging.getLogger(__name__)
OUTPUT_LOGGER = logging.getLogger(f"{__name__}.output")


def decode_err_info(error_type: str, err_info: int) -> dict[str, Any]:
    """Apply the selected unit's ERR_INFO PC rule."""
    if error_type == "IFU":
        return {
            "rule": "the complete IFU_ERR_INFO value is the byte PC",
            "highByte": None,
            "lowByte": None,
            "instructionWordAddress": None,
            "decodedPcBytes": err_info,
        }

    high_bit_ranges = {
        "CCU": (30, 23),
        "MTE": (39, 32),
        "CUBE": (31, 24),
        "VEC": (39, 32),
    }
    bit_range = high_bit_ranges.get(error_type)
    if bit_range is None:
        raise ValueError(f"unsupported error type: {error_type}")
    high_msb, high_lsb = bit_range
    high = (err_info >> high_lsb) & 0xFF
    low = err_info & 0xFF
    word_address = (high << 8) | low
    return {
        "rule": (
            f"ERR_INFO[{high_msb}:{high_lsb}] || ERR_INFO[7:0], "
            "then multiply the 16-bit instruction address by 4"
        ),
        "highByte": high,
        "lowByte": low,
        "instructionWordAddress": word_address,
        "decodedPcBytes": word_address * 4,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode a selected AIC ERROR ERR_INFO register. Select the unit from "
            "EXCP DS before invoking this tool."
        )
    )
    parser.add_argument(
        "--error-type",
        required=True,
        type=str.upper,
        choices=ERROR_TYPES,
        help="Error unit selected from EXCP DS.",
    )
    parser.add_argument(
        "--err-info", required=True, type=integer, help="Selected ERR_INFO value."
    )
    direct = parser.add_mutually_exclusive_group()
    direct.add_argument(
        "--direct-offset",
        type=integer,
        help="Previously calculated byte offset to compare with the decoded result.",
    )
    direct.add_argument(
        "--current-pc",
        type=integer,
        help="Current byte PC used by the direct current_pc - start_pc method.",
    )
    parser.add_argument(
        "--start-pc", type=integer, help="Start byte PC paired with --current-pc."
    )
    parser.add_argument(
        "--pc-base",
        type=integer,
        help=(
            "Base in the same byte-address domain as decoded PC. Subtract it before "
            "comparing with --direct-offset; defaults to --start-pc for a PC pair."
        ),
    )
    return parser.parse_args()


def hex_or_none(value: int | None) -> str | None:
    return None if value is None else hex(value)


def validate_pc_pair(args: argparse.Namespace) -> None:
    if (args.current_pc is None) != (args.start_pc is None):
        raise ValueError("--current-pc and --start-pc must be provided together")
    if args.current_pc is None or args.pc_base is None:
        return
    if args.pc_base != args.start_pc:
        raise ValueError(
            "--pc-base must equal --start-pc when comparing a current/start PC pair"
        )


def resolve_base(args: argparse.Namespace) -> int | None:
    if args.pc_base is not None:
        return args.pc_base
    if args.current_pc is not None:
        return args.start_pc
    return None


def normalize_pc(decoded_pc: int, base: int | None) -> tuple[int | None, str]:
    if base is None:
        return decoded_pc, COMPARISON_SCOPE
    if decoded_pc >= base:
        return decoded_pc - base, COMPARISON_SCOPE
    return None, (
        "decoded PC is below the supplied base; the inputs are not in the same "
        "address domain or a truncated/wrapped PC needs a platform rule"
    )


def direct_calculation(args: argparse.Namespace) -> tuple[str | None, int | None, str]:
    if args.current_pc is None:
        method = "provided direct offset" if args.direct_offset is not None else None
        return method, args.direct_offset, COMPARISON_SCOPE
    if args.current_pc < args.start_pc:
        return (
            "current_pc - start_pc",
            None,
            (
                "current PC is below start PC; apply the target platform's wrap or "
                "truncation rule before comparison"
            ),
        )
    return "current_pc - start_pc", args.current_pc - args.start_pc, COMPARISON_SCOPE


def add_comparison(
    report: dict[str, Any], args: argparse.Namespace, decoded_pc: int
) -> None:
    base = resolve_base(args)
    normalized_pc, normalize_reason = normalize_pc(decoded_pc, base)
    method, direct_offset, direct_reason = direct_calculation(args)
    if args.direct_offset is not None and base is None:
        normalized_pc = None
        normalize_reason = (
            "--direct-offset is relative but decoded ERR_INFO is a PC; supply "
            "--pc-base from the same byte-address domain (use 0 explicitly only "
            "when the decoded PC is already an object-relative offset)"
        )

    if base is not None or method is not None:
        report["normalization"] = {
            "pcBase": hex_or_none(base),
            "decodedOffsetBytes": hex_or_none(normalized_pc),
        }
    if method is None:
        return

    reason = normalize_reason if normalized_pc is None else direct_reason
    comparable = normalized_pc is not None and direct_offset is not None
    report["directCalculation"] = {
        "method": method,
        "currentPc": hex_or_none(args.current_pc),
        "startPc": hex_or_none(args.start_pc),
        "directOffsetBytes": hex_or_none(direct_offset),
    }
    report["comparison"] = {
        "comparable": comparable,
        "equal": normalized_pc == direct_offset if comparable else None,
        "reason": reason,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    validate_pc_pair(args)
    decoded = decode_err_info(args.error_type, args.err_info)
    decoded_pc = decoded.get("decodedPcBytes")
    if not isinstance(decoded_pc, int):
        raise ValueError("decoded PC is missing or invalid")
    report: dict[str, Any] = {
        "errorType": args.error_type,
        "errInfo": hex(args.err_info),
        "scopeNote": (
            "Arithmetic audit for the caller-selected unit/platform rule; do not "
            "override a corrected PC or a target CANN/SoC rule."
        ),
        "extraction": {
            "rule": decoded.get("rule"),
            "highByte": hex_or_none(decoded.get("highByte")),
            "lowByte": hex_or_none(decoded.get("lowByte")),
            "instructionWordAddress": hex_or_none(
                decoded.get("instructionWordAddress")
            ),
        },
        "decodedPcBytes": hex(decoded_pc),
        "alignmentBytes": 4,
    }
    add_comparison(report, args, decoded_pc)
    if args.error_type == "IFU":
        report["formatNote"] = (
            "This applies the requested rule that IFU_ERR_INFO is already a byte PC. "
            "If the target CANN/SoC exposes a packed IFU register, verify its "
            "documented address field before using this result."
        )
    return report


def main() -> int:
    configure_logging(OUTPUT_LOGGER)
    args = parse_args()
    try:
        report = build_report(args)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2
    OUTPUT_LOGGER.info("%s", json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
