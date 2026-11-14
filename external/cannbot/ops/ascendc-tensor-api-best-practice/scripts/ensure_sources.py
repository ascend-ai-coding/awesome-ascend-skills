#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License"). You should refer to the License for details.
# You should not use this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS"
# BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the License for the specific
# language governing permissions and limitations under the License.

"""Inspect local CANN community source repositories used by this skill.

These repositories are read-only knowledge sources. This script never clones,
fetches, rebases, or modifies them. It only verifies path existence, origin,
branch, commit, and dirty state, then emits JSON for reproducible reference.
Exit status is 0 when every requested repository is usable.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml

DEFAULT_SOURCE_FILE = Path(__file__).with_name("sources.yaml")


def load_sources(path):
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("sources"), dict):
        raise ValueError(f"invalid source registry: {path}")
    sources = data["sources"]
    for name, spec in sources.items():
        required = ("root", "url", "required_paths")
        missing = [field for field in required if field not in spec]
        if missing:
            raise ValueError(f"source {name} missing fields: {', '.join(missing)}")
        if not isinstance(spec["required_paths"], list) or not spec["required_paths"]:
            raise ValueError(f"source {name} must define a non-empty required_paths list")
        spec["root"] = Path(spec["root"]).expanduser()
    return sources


def run(command):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or " ".join(command)
        raise RuntimeError(message)
    return result.stdout.strip()


def normalize_url(url):
    return url.replace("git@gitcode.com:", "https://gitcode.com/").removesuffix(".git")


def inspect(name, spec):
    root = spec["root"].expanduser().resolve()
    result = {
        "name": name,
        "path": str(root),
        "required_paths": spec["required_paths"],
    }
    try:
        if not root.exists():
            result.update(state="missing", usable=False)
            return result
        if not (root / ".git").exists():
            result.update(state="invalid", usable=False, reason="path is not a Git repository")
            return result
        origin = run(["git", "-C", str(root), "remote", "get-url", "origin"])
        missing_paths = [path for path in spec["required_paths"] if not (root / path).exists()]
        origin_matches = normalize_url(origin) == normalize_url(spec["url"])
        dirty = bool(run(["git", "-C", str(root), "status", "--porcelain"]))
        result.update(
            origin=origin,
            origin_matches=origin_matches,
            branch=run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]),
            commit=run(["git", "-C", str(root), "rev-parse", "HEAD"]),
            dirty=dirty,
            missing_paths=missing_paths,
        )
        # Dirty state is evidence, not a hard blocker: source repositories may
        # carry local build outputs or developer changes. Required paths and
        # origin remain hard requirements; callers must quote the dirty flag
        # and commit when using an affected repository.
        result["usable"] = origin_matches and not missing_paths
        result["state"] = "ready" if result["usable"] else "blocked"
        return result
    except Exception as error:
        result.update(state="error", usable=False, reason=str(error))
        return result


def check_map(markdown_path, sources):
    text = markdown_path.read_text(encoding="utf-8")
    referenced = set()
    for value in re.findall(r"`([^`]+)`", text):
        if not value or value.startswith(("~/", "scripts/")) or value.endswith(".md"):
            continue
        if "/" not in value:
            continue
        for name in sources:
            prefix = name + "/"
            if value.startswith(prefix):
                referenced.add((name, value[len(prefix):]))
                break
    registered = {
        (name, path)
        for name, spec in sources.items()
        for path in spec["required_paths"]
    }
    return {
        "markdown": str(markdown_path),
        "referenced_paths": len(referenced),
        "registered_paths": len(registered),
        "not_registered": sorted(f"{name}/{path}" for name, path in referenced - registered),
        "not_referenced": sorted(f"{name}/{path}" for name, path in registered - referenced),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, default=DEFAULT_SOURCE_FILE,
                        help=f"source registry YAML (default: {DEFAULT_SOURCE_FILE})")
    parser.add_argument("--source", action="append",
                        help="inspect only named sources; repeatable")
    parser.add_argument("--check-map", type=Path,
                        help="verify that a Markdown source map and source registry use the same paths")
    args = parser.parse_args()

    try:
        sources = load_sources(args.source_file)
    except Exception as error:
        print(json.dumps({"status": "blocked", "failures": [str(error)]},
                         ensure_ascii=False, indent=2))
        return 2
    unknown = [name for name in (args.source or []) if name not in sources]
    if unknown:
        print(json.dumps({"status": "blocked", "failures": [f"unknown sources: {unknown}"]},
                         ensure_ascii=False, indent=2))
        return 2
    if args.check_map:
        result = check_map(args.check_map.resolve(), sources)
        consistent = not result["not_registered"] and not result["not_referenced"]
        print(json.dumps({
            "status": "ready" if consistent else "blocked",
            **result,
        }, ensure_ascii=False, indent=2))
        return 0 if consistent else 1
    selected = args.source or list(sources)
    output = {name: inspect(name, sources[name]) for name in selected}
    blocked = [name for name, state in output.items() if not state.get("usable")]
    print(json.dumps({
        "sources": output,
        "status": "blocked" if blocked else "ready",
        "blocked_sources": blocked,
    }, ensure_ascii=False, indent=2))
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
