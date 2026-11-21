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
"""Use the active CANN llvm-objdump for one physical symbol or PC window."""

from __future__ import annotations

import argparse
import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple


from cli_common import configure_logging, integer


LOGGER = logging.getLogger(__name__)
OUTPUT_LOGGER = logging.getLogger(f"{__name__}.output")


class FunctionSymbol(NamedTuple):
    start: int
    size: int
    name: str


class DisassemblyPlan(NamedTuple):
    command: list[str]
    object_pc: int | None


class CommandError(Exception):
    """Expected CLI failure with an explicit process exit code."""

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def find_objdump(explicit: str | None) -> str:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise CommandError(f"llvm-objdump not found: {candidate}")

    candidate = shutil.which("llvm-objdump")
    if candidate:
        return candidate
    raise CommandError(
        "llvm-objdump not found in PATH; source the matching CANN set_env.sh "
        "or pass --llvm-objdump"
    )


def read_function_symbols(objdump: str, object_path: Path) -> list[FunctionSymbol]:
    """Read function symbols before disassembly; llvm-objdump returns 0 for misses."""
    completed = subprocess.run(
        [objdump, "-t", str(object_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown llvm-objdump error"
        raise CommandError(f"failed to read symbol table: {detail}")

    symbols: list[FunctionSymbol] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[2] != "F":
            continue
        try:
            start = int(fields[0], 16)
            size = int(fields[4], 16)
        except ValueError:
            continue
        symbols.append(FunctionSymbol(start, size, fields[-1]))
    return symbols


def contains_instruction(output: str, address: int) -> bool:
    """Return whether disassembly contains an instruction at the exact address."""
    for line in output.splitlines():
        fields = line.lstrip().split(maxsplit=1)
        if not fields or not fields[0].endswith(":"):
            continue
        try:
            current_address = int(fields[0][:-1], 16)
        except ValueError:
            continue
        if current_address == address:
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Narrowly disassemble one Ascend physical symbol or normalized "
            "object-PC window."
        )
    )
    parser.add_argument("--obj", required=True, help="Confirmed Ascend device ELF.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--symbol", help="Exact physical symbol from ELF metadata.")
    target.add_argument("--pc", type=integer, help="Object or runtime byte PC.")
    parser.add_argument(
        "--symbol-offset",
        type=integer,
        help="Byte offset within --symbol; disassemble a narrow window around it.",
    )
    parser.add_argument(
        "--load-base",
        type=integer,
        default=0,
        help="Runtime load base to subtract from --pc; default 0.",
    )
    parser.add_argument(
        "--before", type=integer, default=64, help="Bytes before normalized PC."
    )
    parser.add_argument(
        "--after", type=integer, default=128, help="Bytes after normalized PC."
    )
    parser.add_argument("--llvm-objdump", help="Explicit CANN llvm-objdump path.")
    return parser.parse_args()


def validate_request(args: argparse.Namespace) -> Path:
    object_path = Path(args.obj).expanduser().resolve()
    if not object_path.is_file():
        raise CommandError(f"kernel object not found: {object_path}")
    if args.after == 0:
        raise CommandError("--after must be positive")
    if args.symbol is not None and not args.symbol:
        raise CommandError("--symbol must not be empty")
    if args.symbol_offset is not None and args.symbol is None:
        raise CommandError("--symbol-offset requires --symbol")
    return object_path


def find_symbol(symbols: list[FunctionSymbol], name: str) -> FunctionSymbol | None:
    for symbol in symbols:
        if symbol.name == name:
            return symbol
    return None


def find_containing_symbol(
    symbols: list[FunctionSymbol], object_pc: int
) -> FunctionSymbol | None:
    for symbol in symbols:
        if symbol.size <= 0:
            continue
        if symbol.start <= object_pc < symbol.start + symbol.size:
            return symbol
    return None


def add_window(command: list[str], start: int, stop: int) -> None:
    command.extend([f"--start-address={start:#x}", f"--stop-address={stop:#x}"])


def require_instruction_alignment(object_pc: int) -> None:
    if object_pc % 4 != 0:
        raise CommandError(
            f"normalized object PC is not 4-byte aligned: {object_pc:#x}"
        )


def build_symbol_plan(
    args: argparse.Namespace,
    objdump: str,
    object_path: Path,
    symbols: list[FunctionSymbol],
) -> DisassemblyPlan:
    if args.load_base != 0:
        raise CommandError("--load-base is only valid with --pc")
    selected = find_symbol(symbols, args.symbol)
    if selected is None:
        raise CommandError(
            f"physical function symbol not found: {args.symbol}", exit_code=1
        )

    command = [objdump, "-d"]
    if args.symbol_offset is None:
        command.append(f"--disassemble-symbols={args.symbol}")
        command.append(str(object_path))
        return DisassemblyPlan(command, None)
    if args.symbol_offset >= selected.size:
        raise CommandError(
            f"symbol offset {args.symbol_offset:#x} is outside {args.symbol} "
            f"(size {selected.size:#x})",
            exit_code=1,
        )

    object_pc = selected.start + args.symbol_offset
    require_instruction_alignment(object_pc)
    LOGGER.info(
        "physical_symbol=%s symbol_start=%#x symbol_offset=%#x object_pc=%#x",
        selected.name,
        selected.start,
        args.symbol_offset,
        object_pc,
    )
    start = max(selected.start, object_pc - args.before) & ~0x3
    stop = min(selected.start + selected.size, (object_pc + args.after + 3) & ~0x3)
    add_window(command, start, stop)
    command.append(str(object_path))
    return DisassemblyPlan(command, object_pc)


def build_pc_plan(
    args: argparse.Namespace,
    objdump: str,
    object_path: Path,
    symbols: list[FunctionSymbol],
) -> DisassemblyPlan:
    if args.pc is None:
        raise CommandError("--pc is required when --symbol is absent")
    object_pc = args.pc - args.load_base
    if object_pc < 0:
        raise CommandError(
            "normalized object PC is negative; check --pc and --load-base"
        )
    require_instruction_alignment(object_pc)
    containing = find_containing_symbol(symbols, object_pc)
    if containing is None:
        raise CommandError(
            f"no function symbol contains object PC {object_pc:#x}", exit_code=1
        )

    LOGGER.info(
        "physical_symbol=%s symbol_start=%#x symbol_offset=%#x",
        containing.name,
        containing.start,
        object_pc - containing.start,
    )
    LOGGER.info(
        "runtime_pc=%#x load_base=%#x object_pc=%#x",
        args.pc,
        args.load_base,
        object_pc,
    )
    command = [objdump, "-d"]
    start = max(0, object_pc - args.before) & ~0x3
    stop = (object_pc + args.after + 3) & ~0x3
    add_window(command, start, stop)
    command.append(str(object_path))
    return DisassemblyPlan(command, object_pc)


def build_plan(
    args: argparse.Namespace,
    objdump: str,
    object_path: Path,
    symbols: list[FunctionSymbol],
) -> DisassemblyPlan:
    if args.symbol is not None:
        return build_symbol_plan(args, objdump, object_path, symbols)
    return build_pc_plan(args, objdump, object_path, symbols)


def run_disassembly(plan: DisassemblyPlan) -> subprocess.CompletedProcess[str]:
    LOGGER.info("command=%s", shlex.join(plan.command))
    try:
        return subprocess.run(plan.command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise CommandError(f"failed to execute llvm-objdump: {exc}") from exc


def validate_disassembly(
    args: argparse.Namespace,
    plan: DisassemblyPlan,
    completed: subprocess.CompletedProcess[str],
) -> None:
    if completed.returncode != 0:
        raise CommandError("llvm-objdump failed", exit_code=completed.returncode)
    if args.symbol is not None and args.symbol_offset is None:
        if f"<{args.symbol}>:" not in completed.stdout:
            raise CommandError(
                f"llvm-objdump produced no instructions for symbol: {args.symbol}",
                exit_code=1,
            )
        return
    if plan.object_pc is None or contains_instruction(completed.stdout, plan.object_pc):
        return
    raise CommandError(
        f"llvm-objdump produced no instruction at object PC {plan.object_pc:#x}",
        exit_code=1,
    )


def emit_completed_output(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        OUTPUT_LOGGER.info("%s", completed.stdout.rstrip("\n"))
    if completed.stderr:
        LOGGER.error("%s", completed.stderr.rstrip("\n"))


def main() -> int:
    configure_logging(OUTPUT_LOGGER)
    args = parse_args()
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        object_path = validate_request(args)
        objdump = find_objdump(args.llvm_objdump)
        symbols = read_function_symbols(objdump, object_path)
        plan = build_plan(args, objdump, object_path, symbols)
        completed = run_disassembly(plan)
        emit_completed_output(completed)
        validate_disassembly(args, plan, completed)
    except CommandError as exc:
        if completed is None or str(exc) != "llvm-objdump failed":
            LOGGER.error("%s", exc)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
