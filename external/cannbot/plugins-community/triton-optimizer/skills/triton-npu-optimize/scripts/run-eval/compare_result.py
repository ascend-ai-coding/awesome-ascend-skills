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
from pathlib import Path
from typing import TextIO

import torch

from npu_compare import compare_result_payloads, format_artifact_compare_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-result", "--oracle-result", dest="ref_result", required=True)
    parser.add_argument("--new-result", required=True)
    args = parser.parse_args()
    return compare_result_files(args.ref_result, args.new_result)


def compare_result_files(
    ref_result: str | Path,
    new_result: str | Path,
) -> int:
    oracle_payload = _load_result_payload(ref_result)
    candidate_payload = _load_result_payload(new_result)
    result = compare_result_payloads(oracle_payload, candidate_payload)
    print(format_artifact_compare_result(result))
    return 0 if result.passed else 1


def compare_remote_result_files(
    ref_result: Path,
    new_result: Path,
    remote: str,
    remote_workdir: str | None,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    from run_runtime import (
        cleanup_remote_workspace,
        copy_file_to_remote,
        create_remote_workspace,
        run_remote_command_streaming,
    )

    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    compare_script = Path(__file__).resolve()
    compare_helper = compare_script.with_name("npu_compare.py")
    remote_script = f"{remote_workspace}/{compare_script.name}"
    remote_helper = f"{remote_workspace}/{compare_helper.name}"
    remote_ref = f"{remote_workspace}/{ref_result.name}"
    remote_new = f"{remote_workspace}/{new_result.name}"
    try:
        copy_file_to_remote(spec, compare_script, remote_script, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, compare_helper, remote_helper, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, ref_result, remote_ref, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, new_result, remote_new, verbose=verbose, stderr=stderr)
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                compare_script.name,
                "--ref-result",
                ref_result.name,
                "--new-result",
                new_result.name,
            ],
            verbose=verbose,
            stderr=stderr,
        )
        return int(result["return_code"])
    finally:
        cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def _load_result_payload(path: str | Path) -> object:
    return torch.load(Path(path), map_location="cpu")


if __name__ == "__main__":
    raise SystemExit(main())
