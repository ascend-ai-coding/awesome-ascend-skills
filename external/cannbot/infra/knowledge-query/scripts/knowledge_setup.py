#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Prepare the knowledge checkout and search runtime without installing extra skills."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

REPOSITORY = "https://gitcode.com/cann/cannbot-knowledge.git"
QUERY_DIR = Path(".agents/skills/knowledge-query/scripts")
ROOT_KEY = "CANNBOT_KNOWLEDGE_ROOT"
RUNTIME_CHECK = """
import sqlite3, sys, yaml
assert sys.version_info >= (3, 10)
assert 6 <= int(yaml.__version__.split('.')[0]) < 7
if sys.version_info < (3, 11):
    import tomli
    assert 2 <= int(tomli.__version__.split('.')[0]) < 3
with sqlite3.connect(':memory:') as db:
    db.execute('CREATE VIRTUAL TABLE probe USING fts5(text)')
"""


def split_root(argv):
    """Accept the global root option before or after the query subcommand."""
    remaining, roots = [], []
    values = iter(argv)
    for value in values:
        if value == "--knowledge-root":
            roots.append(next(values, ""))
        elif value.startswith("--knowledge-root="):
            roots.append(value.split("=", 1)[1])
        else:
            remaining.append(value)
    if any(not value.strip() or value.startswith("--") for value in roots):
        raise ValueError("--knowledge-root requires a non-empty path")
    if len(set(roots)) > 1:
        raise ValueError("conflicting --knowledge-root values")
    return remaining, roots[0] if roots else None


def configured_root(explicit=None):
    """Read configuration as data; never source a project shell file."""
    if explicit is not None and not explicit.strip():
        raise ValueError("--knowledge-root requires a non-empty path")
    value = explicit or os.environ.get(ROOT_KEY)
    if value:
        return Path(value).expanduser().absolute()
    for parent in (Path.cwd(), *Path.cwd().parents):
        config = parent / ".cannbot/knowledge.env"
        if config.is_file():
            for line in config.read_text(encoding="utf-8").splitlines():
                tokens = shlex.split(line, comments=True)
                if tokens[:1] == ["export"]:
                    tokens = tokens[1:]
                for token in tokens:
                    if token.startswith(ROOT_KEY + "="):
                        value = token.split("=", 1)[1]
                        if not value.strip():
                            raise ValueError(f"empty knowledge root in {config}")
                        path = Path(value).expanduser()
                        return path if path.is_absolute() else config.parent / path
            break
    for parent in (Path.cwd(), *Path.cwd().parents):
        if (parent / "knowledge").is_dir() and (parent / QUERY_DIR).is_dir():
            return parent
    return Path.home() / ".cannbot/cannbot-knowledge"


def validate_root(root):
    required = (
        "governance/schemas/registries.yaml",
        "governance/contracts/__init__.py",
        "requirements.txt",
        QUERY_DIR / "knowledge_query.py",
        QUERY_DIR / "knowledge_index.py",
    )
    if not (root / "knowledge").is_dir() or any(
        not (root / name).is_file() for name in required
    ):
        raise ValueError(
            f"incomplete knowledge checkout: {root}; existing data was not replaced"
        )


def run(argv, *, check=True, quiet=False):
    return subprocess.run(
        [str(value) for value in argv],
        check=check,
        stdout=subprocess.DEVNULL if quiet else sys.stderr,
        stderr=subprocess.DEVNULL if quiet else sys.stderr,
        timeout=1200,
    )


def acquire(root):
    if root.exists() or root.is_symlink():
        validate_root(root)
        return
    print(f"Cloning {REPOSITORY} into {root}", file=sys.stderr)
    with tempfile.TemporaryDirectory(
        prefix=".knowledge-clone-", dir=root.parent
    ) as temp:
        checkout = Path(temp) / "repository"
        run(
            [
                "git",
                "clone",
                "--depth=1",
                "--branch",
                "master",
                "--",
                REPOSITORY,
                checkout,
            ]
        )
        validate_root(checkout)
        if root.exists() or root.is_symlink():
            raise ValueError(f"destination appeared during clone: {root}")
        checkout.rename(root)


def prepare_python(root):
    python = root / ".venv/bin/python"
    candidate = python if python.is_file() else Path(sys.executable)
    if run([candidate, "-c", RUNTIME_CHECK], check=False, quiet=True).returncode == 0:
        return candidate
    if not python.is_file():
        run([sys.executable, "-m", "venv", root / ".venv"])
    run([python, "-m", "pip", "install", "-r", root / "requirements.txt"])
    run([python, "-c", RUNTIME_CHECK])
    return python


def prepare(explicit=None):
    if sys.version_info < (3, 10):
        raise ValueError("Python 3.10 or newer is required")
    requested = configured_root(explicit)
    if requested.is_symlink() and not requested.exists():
        raise ValueError(f"broken knowledge-root symlink: {requested}")
    root = requested.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    # Serialize bootstrap from query and ingest using the same checkout.
    with (root.parent / f".{root.name}.setup.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        acquire(root)
        python = prepare_python(root)
        index = [
            python,
            root / QUERY_DIR / "knowledge_index.py",
            "--knowledge-root",
            root,
        ]
        if run(
            [*index, "verify", "--level", "quick"], check=False, quiet=True
        ).returncode:
            print("Preparing knowledge search index...", file=sys.stderr)
            run([*index, "build"])
            run([*index, "verify", "--level", "full"])
    return root, python


def report_error(exc):
    print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--knowledge-root", help="checkout path; missing paths are cloned"
    )
    args = parser.parse_args(argv)
    try:
        root, python = prepare(args.knowledge_root)
        print(
            json.dumps({"root": str(root), "python": str(python)}, ensure_ascii=False)
        )
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return report_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
