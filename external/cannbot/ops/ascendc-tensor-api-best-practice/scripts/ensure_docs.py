#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ensure the local asc-devkit documentation and sample root is available.

双工作区机制：
- 主工作区（默认 ~/asc-devkit）：在 master 上时直接 clone / fast-forward；
  被其他分支占用（如 vendor 头锁定的 feature 分支）或与 master 历史分叉时
  不做任何修改，转用 fallback 镜像。
- fallback 镜像（默认 ~/asc-devkit-master）：脚本自有的 master 浅镜像，
  用于手册与样例检索；主工作区不可用时自动 clone / 更新，绝不触碰主工作区。
两个工作区 dirty 或 origin 不匹配时均阻断，不强制覆盖。
输出 JSON 的 selected_root 字段为当前事实源根目录。
"""

import argparse
import json
import subprocess
from pathlib import Path

ASC_DEVKIT_URL = "https://gitcode.com/cann/asc-devkit.git"
DEFAULT_ROOT = Path.home() / "asc-devkit"
FALLBACK_ROOT = Path.home() / "asc-devkit-master"
# 检查具体文件而非目录：空目录树（如残留的 ppt/images）不会造成假阳性。
REQUIRED_PATHS = [
    "docs/zh/api/SIMD-API/tensor_api/algorithm.md",
    "docs/zh/api/SIMD-API/tensor_api/tensor/make_tensor.md",
    "docs/zh/api/SIMD-API/tensor_api/layout/layout.md",
    "examples/01_simd_cpp_api/07_tensor_api/matmul_tensor_api/CMakeLists.txt",
    "examples/01_simd_cpp_api/07_tensor_api/experimental/reg_vector_compute/add",
]


def run(command, check=True):
    result = subprocess.run(command, text=True, capture_output=True)
    if check and result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or " ".join(command)
        raise RuntimeError(message)
    return result


def git(root, *args, check=True):
    return run(["git", "-C", str(root), *args], check=check)


def normalize_url(url):
    return url.replace("git@gitcode.com:", "https://gitcode.com/").removesuffix(".git")


def missing_paths(root):
    return [str(path) for path in REQUIRED_PATHS if not (root / path).exists()]


def inspect(root, ref=None):
    if not root.exists():
        return {"state": "missing"}
    if not (root / ".git").exists():
        return {"state": "invalid", "reason": "path exists but is not a Git repository"}
    if "origin" not in git(root, "remote").stdout.split():
        return {"state": "invalid", "reason": "origin remote is missing"}
    origin = git(root, "remote", "get-url", "origin").stdout.strip()
    if normalize_url(origin) != normalize_url(ASC_DEVKIT_URL):
        return {"state": "invalid", "reason": f"origin mismatch: {origin}"}
    return {
        "state": "present",
        "path": str(root),
        "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
        "commit": git(root, "rev-parse", "HEAD").stdout.strip(),
        "dirty": bool(git(root, "status", "--porcelain").stdout.strip()),
        "origin": origin,
        "requested_ref": ref,
        "missing_paths": missing_paths(root),
    }


def is_usable(result):
    return (
        result.get("state") == "present"
        and not result.get("dirty")
        and not str(result.get("action", "")).startswith("blocked")
        and not result.get("missing_paths")
    )


def ensure_primary(root, ref=None):
    """主工作区：master 上直接 clone / fast-forward；dirty、分叉时阻断。"""
    result = inspect(root, ref)
    if result["state"] == "invalid":
        return result
    if result["state"] == "missing":
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", ASC_DEVKIT_URL, str(root)])
        if ref:
            git(root, "checkout", ref)
        result = inspect(root, ref)
        result["action"] = "cloned"
        return result
    if result.get("dirty"):
        result["action"] = "blocked-dirty"
        return result
    git(root, "fetch", "origin", ref or "master")
    remote = git(root, "rev-parse", "FETCH_HEAD").stdout.strip()
    if result["commit"] != remote:
        # 分支门禁：仅默认 master（或显式 ref）允许推进，避免静默移动
        # feature/vendor-lock 分支（含 detached HEAD 返回的 "HEAD"）。
        if ref is None and result["branch"] != "master":
            result["action"] = "blocked-branch"
            return result
        ancestry = git(root, "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD", check=False)
        if ancestry.returncode:
            result["action"] = "blocked-diverged"
            return result
        if ref:
            git(root, "checkout", ref)
            git(root, "reset", "--hard", "FETCH_HEAD")
        else:
            git(root, "merge", "--ff-only", "FETCH_HEAD")
        result = inspect(root, ref)
        result["action"] = "updated"
    else:
        result["action"] = "up-to-date"
    return result


def ensure_fallback(root):
    """fallback 镜像：脚本自有的 master 浅镜像，fetch --depth + reset --hard 更新。"""
    result = inspect(root)
    if result["state"] == "missing":
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", "master", ASC_DEVKIT_URL, str(root)])
        result = inspect(root)
        result["action"] = "cloned"
        return result
    if result["state"] != "present":
        return result
    if result.get("dirty"):
        result["action"] = "blocked-dirty"
        return result
    git(root, "fetch", "--depth", "1", "origin", "master")
    remote = git(root, "rev-parse", "FETCH_HEAD").stdout.strip()
    if result["commit"] != remote:
        # 浅镜像不含本地提交，reset --hard 等价于 fast-forward。
        git(root, "reset", "--hard", remote)
        result = inspect(root)
        result["action"] = "updated"
    else:
        result["action"] = "up-to-date"
    return result


def summarize(output, selected):
    failures = []
    if selected is None:
        failures.append("no usable asc-devkit worktree for manual and samples")
    else:
        if selected.get("missing_paths"):
            failures.append(f"missing paths: {', '.join(selected['missing_paths'])}")
        if selected.get("state") in {"invalid", "error"} or str(selected.get("action", "")).startswith("blocked"):
            failures.append(str(selected))
    output["selected_root"] = selected.get("path") if selected else None
    output["status"] = "blocked" if failures else "ready"
    if failures:
        output["failures"] = failures
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asc-devkit-root", type=Path, default=DEFAULT_ROOT,
                        help=f"主工作区（默认 {DEFAULT_ROOT}；显式指定为非默认值时禁用 fallback）")
    parser.add_argument("--fallback-root", type=Path, default=FALLBACK_ROOT,
                        help=f"master 浅镜像工作区（默认 {FALLBACK_ROOT}）")
    parser.add_argument("--asc-devkit-ref")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    primary = args.asc_devkit_root.expanduser().resolve()
    fallback = args.fallback_root.expanduser().resolve()
    fallback_enabled = primary == DEFAULT_ROOT.expanduser().resolve()
    output = {}

    if args.check_only:
        primary_state = inspect(primary, args.asc_devkit_ref)
        fallback_state = inspect(fallback) if fallback_enabled else {"state": "disabled"}
        output["asc-devkit"] = primary_state
        output["fallback"] = fallback_state
        if is_usable(primary_state):
            return summarize(output, primary_state)
        if fallback_enabled and is_usable(fallback_state):
            return summarize(output, fallback_state)
        return summarize(output, None)

    primary_state = None
    try:
        primary_state = ensure_primary(primary, args.asc_devkit_ref)
    except Exception as error:
        primary_state = {"state": "error", "reason": str(error), "path": str(primary)}
    output["asc-devkit"] = primary_state

    if is_usable(primary_state):
        return summarize(output, primary_state)

    if not fallback_enabled:
        return summarize(output, None)

    fallback_state = None
    try:
        fallback_state = ensure_fallback(fallback)
    except Exception as error:
        fallback_state = {"state": "error", "reason": str(error), "path": str(fallback)}
    output["fallback"] = fallback_state

    if is_usable(fallback_state):
        return summarize(output, fallback_state)
    if fallback_state.get("state") == "present" and not fallback_state.get("missing_paths"):
        return summarize(output, fallback_state)
    return summarize(output, None)


if __name__ == "__main__":
    raise SystemExit(main())
