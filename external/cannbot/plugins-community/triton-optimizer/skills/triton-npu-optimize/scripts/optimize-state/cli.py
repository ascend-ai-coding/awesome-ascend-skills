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

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from state_manage import start_round as start_round_check
from state_manage import set_current_round_state as set_current_round_state_check
from state_manage import submit_baseline as baseline_submit
from state_manage import submit_round as submit_round_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Validate optimize baseline state, start a gated round, or validate a completed round.",
        epilog=(
            "Subcommands:\n"
            "  submit-baseline  Validate canonical baseline artifacts and advance workflow state.\n"
            "  start-round     Open the next optimize round through workflow state.\n"
            "  set-current-round-state  Update the active round's strategy state.\n"
            "  submit-round    Validate and submit a completed optimize round."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", help="Subcommand to run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = args[0]
    remaining = args[1:]
    if command == "submit-baseline":
        return baseline_submit.main(
            [command, *remaining],
            prog_name=f"{parser.prog} {command}",
        )
    if command == "start-round":
        return start_round_check.main(
            [command, *remaining],
            prog_name=f"{parser.prog} {command}",
        )
    if command == "set-current-round-state":
        return set_current_round_state_check.main(
            [command, *remaining],
            prog_name=f"{parser.prog} {command}",
        )
    if command == "submit-round":
        return submit_round_check.main(
            [command, *remaining],
            prog_name=f"{parser.prog} {command}",
        )
    raise ValueError(f"unknown command: {command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
