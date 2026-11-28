#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR
# A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------
"""Install local Git gates without changing tracked files, index, or history."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

MARKER = "# CANN deprecated error-code gate v1"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def install(repo: Path, catalog: Path, scanner: Path) -> Path:
    repo = Path(git(repo, "rev-parse", "--show-toplevel")).resolve()
    # Honor Git's effective hooks path, including linked worktrees and core.hooksPath.
    hooks_path = Path(git(repo, "rev-parse", "--git-path", "hooks"))
    hooks = hooks_path if hooks_path.is_absolute() else repo / hooks_path
    hooks = hooks.resolve()
    payload = hooks / "cann-error-code"
    if payload.is_symlink():
        raise ValueError(f"refusing symlink payload directory: {payload}")
    for name in ["pre-commit", "pre-push"]:
        target = hooks / name
        if target.is_symlink() or (
            target.exists() and MARKER not in target.read_text()
        ):
            raise ValueError(
                f"existing custom hook preserved: {target}; integrate explicitly instead of overwriting"
            )
    spec = importlib.util.spec_from_file_location("cann_error_code_gate", scanner)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    banned = module.load_policy(catalog)  # Validate the baseline before writing.
    for name in [
        "clause.check_error_codes.py",
        "documented-error-codes.json",
        "manifest.json",
    ]:
        if (payload / name).is_symlink():
            raise ValueError(f"refusing symlink payload: {payload / name}")
    payload.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scanner, payload / "clause.check_error_codes.py")
    shutil.copyfile(catalog, payload / "documented-error-codes.json")
    metadata = json.loads(catalog.read_text(encoding="utf-8"))
    manifest = {
        "source_skill": "cann-error-code-standard",
        "source_catalog": str(catalog.resolve()),
        "documentation_version": metadata["documentation_version"],
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "scanner_source": str(scanner.resolve()),
        "scanner_sha256": hashlib.sha256(scanner.read_bytes()).hexdigest(),
        "scope": "C/C++ additions in index and outgoing commits",
        "blocked_codes": sorted(banned),
        "maintenance": "Regenerate this deployment snapshot by rerunning the installer after baseline updates.",
    }
    (payload / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, mode in [("pre-commit", "--staged"), ("pre-push", "--pre-push")]:
        # Resolve the hook payload relative to the hook itself, independent of invocation CWD.
        script = (
            "#!/usr/bin/env python3\n" + MARKER + "\n"
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "payload = Path(__file__).resolve().parent / 'cann-error-code'\n"
            "repo = subprocess.run(\n"
            "    ['git', 'rev-parse', '--show-toplevel'],\n"
            "    text=True, capture_output=True, check=True,\n"
            ").stdout.strip()\n"
            f"python = {sys.executable!r}\n"
            "args = [python, str(payload / 'clause.check_error_codes.py'),\n"
            "        '--root', repo, '--catalog',\n"
            f"        str(payload / 'documented-error-codes.json'), {mode!r}]\n"
            + ("args.extend(sys.argv[1:])\n" if name == "pre-push" else "")
            + "os.execv(python, args)\n"
        )
        hook = hooks / name
        hook.write_text(script, encoding="utf-8", newline="\n")
        hook.chmod(hook.stat().st_mode | 0o111)
    return hooks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "cann-error-code-standard/references/documented-error-codes.json",
    )
    args = parser.parse_args()
    try:
        hooks = install(
            args.repo,
            args.catalog,
            Path(__file__).with_name("clause.check_error_codes.py"),
        )
        print(f"Installed pre-commit and pre-push gates: {hooks}")
        print(
            "The catalog is a deployment snapshot; update from cann-error-code-standard using this installer."
        )
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: hooks were not installed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
