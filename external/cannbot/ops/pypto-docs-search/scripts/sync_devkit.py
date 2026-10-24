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
"""装配 PyPTO 算子开发资源到本地缓存：docs（主仓）、ops/tests（算子仓）。

工作树已含某类资源则符号链接复用、免重复下载；否则 sparse-checkout 下载。

用法: python3 sync_devkit.py [--pin <git-ref>]
  环境变量覆盖:
    PYPTO_DEVKIT_DIR                       缓存目录
    PYPTO_SRC_URL / PYPTO_SRC              docs 主仓：远程 URL / 本地已有工作树
    PYPTO_GYM_URL / PYPTO_GYM_SRC          ops+tests 算子仓：远程 URL / 本地已有工作树
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class _BelowErrorFilter(logging.Filter):
    """仅允许 ERROR 以下日志通过，避免 stdout/stderr 重复。"""

    def filter(self, record):
        return record.levelno < logging.ERROR


def _configure_logging():
    formatter = logging.Formatter("%(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(_BelowErrorFilter())
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stdout_handler, stderr_handler],
        force=True,
    )


@dataclass(frozen=True)
class ProvisionRequest:
    """描述一次资源装配，避免调用点依赖位置参数顺序。"""

    devkit: Path
    tag: str
    marker: str
    url: str
    explicit: str
    pairs: list[tuple[str, str]]
    pin: str


def cache_dir():
    d = os.environ.get("PYPTO_DEVKIT_DIR")
    if d:
        return Path(d)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "pypto-devkit"


def find_up(marker):
    """从 $PWD 向上查找含标志子路径的仓根（复用已有工作树，避免重复下载）。"""
    d = Path.cwd()
    while True:
        if (d / marker).exists():
            return d
        if d.parent == d:
            return None
        d = d.parent


def _remove_managed_path(path):
    """删除缓存内由本脚本管理的单个资源目标。"""
    if not (path.is_symlink() or path.exists()):
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def relink(dst, src):
    """等价 ln -sfn：替换已存在的目标为指向 src 的符号链接。"""
    _remove_managed_path(dst)
    dst.symlink_to(src)


def git(args, cwd=None):
    return subprocess.run(
        ["git"] + args,
        cwd=(str(cwd) if cwd else None),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode


def _link_resources(request, base, manifest, mode, source_for):
    for name, sub in request.pairs:
        source = base / sub
        if not source.exists():
            _remove_managed_path(request.devkit / name)
            LOGGER.info("[skipped] %s:%s (path not in source)", request.tag, sub)
            continue
        relink(request.devkit / name, source)
        manifest[name] = {"mode": mode, "source": source_for(sub, source)}


def _reuse_local(request, base, manifest):
    _link_resources(
        request,
        base,
        manifest,
        "symlink",
        lambda _sub, source: str(source),
    )
    LOGGER.info("[reuse] %s ← %s", request.tag, base)


def _checkout_sparse_repo(request, tmp):
    clone_args = [
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--no-checkout",
        request.url,
        str(tmp),
    ]
    if git(clone_args) != 0:
        LOGGER.error(
            "[download failed] %s (set corresponding *_URL to override, or check the network)",
            request.url,
        )
        return 4
    # 自定义 Git 模板可能省略 info；旧版 Git 的 sparse-checkout 不会补建。
    (tmp / ".git/info").mkdir(parents=True, exist_ok=True)
    if (
        git(["config", "core.sparseCheckout", "true"], cwd=tmp) != 0
        or git(["config", "core.sparseCheckoutCone", "true"], cwd=tmp) != 0
    ):
        LOGGER.error("[sparse init failed] %s", request.tag)
        return 4
    subs = [sub for _name, sub in request.pairs]
    if git(["sparse-checkout", "set"] + subs, cwd=tmp) != 0:
        LOGGER.error("[sparse failed] %s", request.tag)
        return 4
    if not request.pin:
        if git(["checkout", "--detach", "HEAD"], cwd=tmp) != 0:
            LOGGER.error("[checkout failed] %s", request.tag)
            return 4
        return 0
    if git(["fetch", "--depth", "1", "origin", request.pin], cwd=tmp) != 0:
        LOGGER.error("[pin failed] %s unable to fetch %s", request.tag, request.pin)
        return 4
    if git(["checkout", "--detach", "FETCH_HEAD"], cwd=tmp) != 0:
        LOGGER.error("[pin failed] %s unable to switch to %s", request.tag, request.pin)
        return 4
    return 0


def _download_resources(request, manifest):
    tmp = request.devkit / (".repo-" + request.tag)
    if tmp.exists() or tmp.is_symlink():
        shutil.rmtree(tmp, ignore_errors=True)
    result = _checkout_sparse_repo(request, tmp)
    if result != 0:
        return result
    revision = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(tmp),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
        or "unknown"
    )
    _link_resources(
        request,
        tmp,
        manifest,
        "download@" + revision,
        lambda sub, _source: f"{request.url}:{sub}",
    )
    LOGGER.info("[download] %s ← %s@%s", request.tag, request.url, revision)
    return 0


def provision(request, manifest):
    """装配一类资源；返回 0 表示成功，非 0 表示下载或切换失败。"""
    if request.explicit and (Path(request.explicit) / request.marker).exists():
        base = Path(request.explicit)
    else:
        base = find_up(request.marker)
    if base:
        _reuse_local(request, base, manifest)
        return 0
    return _download_resources(request, manifest)


def _provision_requests(devkit, pin):
    src_url = os.environ.get("PYPTO_SRC_URL", "https://gitcode.com/cann/pypto.git")
    gym_url = os.environ.get("PYPTO_GYM_URL", "https://gitcode.com/cann/pypto-gym.git")
    return [
        ProvisionRequest(
            devkit,
            "pypto",
            "docs/zh/api",
            src_url,
            os.environ.get("PYPTO_SRC", ""),
            [("docs", "docs/zh")],
            pin,
        ),
        ProvisionRequest(
            devkit,
            "gym",
            "ops",
            gym_url,
            os.environ.get("PYPTO_GYM_SRC", ""),
            [("ops", "ops"), ("tests", "tests/ops")],
            pin,
        ),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", default="", help="固定所有远端资源到指定 git ref")
    args = parser.parse_args(argv)
    _configure_logging()
    devkit = cache_dir()
    if shutil.which("git") is None:
        LOGGER.error("[missing git] Please install git first")
        return 3
    try:
        devkit.mkdir(parents=True, exist_ok=True)
    except OSError:
        LOGGER.error("[cannot create cache] %s", devkit)
        return 3

    manifest = {}
    for request in _provision_requests(devkit, args.pin):
        rc = provision(request, manifest)
        if rc != 0:
            return rc

    (devkit / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dirs = " ".join(sorted(name + "/" for name in manifest))
    LOGGER.info("[done] Cache: %s (%s + MANIFEST.json)", devkit, dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
