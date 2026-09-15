#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""装配 PyPTO 算子开发资源到本地缓存：docs（主仓）、ops/tests（算子仓）、pro_ops（PyPTO-Pro 样例，可选）。

工作树已含某类资源则符号链接复用、免重复下载；否则 sparse-checkout 下载。
pro_ops 为 PyPTO-Pro 工作流额外拉取项：默认跟随主仓源，源仓无该路径时自动跳过，不影响 docs/ops/tests。

用法: python3 sync_devkit.py [--pin <git-ref>]
  环境变量覆盖:
    PYPTO_DEVKIT_DIR                       缓存目录
    PYPTO_SRC_URL / PYPTO_SRC              docs 主仓：远程 URL / 本地已有工作树
    PYPTO_GYM_URL / PYPTO_GYM_SRC          ops+tests 算子仓：远程 URL / 本地已有工作树
    PYPTO_PRO_OPS_URL / PYPTO_PRO_OPS_SRC  pro_ops 样例：远程 URL（默认=PYPTO_SRC_URL）/ 本地已有工作树
"""
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger("sync_devkit")


def _setup_logging():
    """Route INFO status to stdout and warnings/errors to stderr (mirrors the
    original print() stream split)."""
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    out = logging.StreamHandler(sys.stdout)
    out.addFilter(lambda record: record.levelno < logging.WARNING)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    for handler in (out, err):
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


class SourceSpec(NamedTuple):
    """One provisionable source: its tag, repo marker path, URL, local override and link pairs."""

    tag: str
    marker: str
    url: str
    explicit: str
    pairs: list


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


def relink(dst, src):
    """等价 ln -sfn：替换已存在的目标为指向 src 的符号链接。"""
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def git(args, cwd=None):
    return subprocess.run(
        ["git"] + args, cwd=(str(cwd) if cwd else None),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode


def provision(devkit, spec, pin, manifest):
    """spec.pairs: [(类名, 仓内子路径), ...]。返回 0 成功，非 0 失败退出码。"""
    if spec.explicit and (Path(spec.explicit) / spec.marker).exists():
        base = Path(spec.explicit)
    else:
        base = find_up(spec.marker)

    if base:
        for name, sub in spec.pairs:
            src = base / sub
            if not src.exists():
                logger.info("[跳过] %s:%s（源无此路径）", spec.tag, sub)
                continue
            relink(devkit / name, src)
            manifest[name] = {"mode": "symlink", "source": str(src)}
        logger.info("[复用] %s ← %s", spec.tag, base)
        return 0

    tmp = devkit / (".repo-" + spec.tag)
    if tmp.exists() or tmp.is_symlink():
        shutil.rmtree(tmp, ignore_errors=True)
    if git(["clone", "--depth", "1", "--filter=blob:none", "--sparse", spec.url, str(tmp)]) != 0:
        logger.error("[下载失败] %s（设对应 *_URL 覆盖，或检查网络）", spec.url)
        return 4
    subs = [sub for _name, sub in spec.pairs]
    if git(["sparse-checkout", "set"] + subs, cwd=tmp) != 0:
        logger.error("[sparse 失败] %s", spec.tag)
        return 4
    if pin:
        git(["fetch", "--depth", "1", "origin", pin], cwd=tmp)
        git(["checkout", pin], cwd=tmp)
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(tmp),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    head = r.stdout.strip() or "unknown"
    for name, sub in spec.pairs:
        if not (tmp / sub).exists():
            logger.info("[跳过] %s:%s（源无此路径）", spec.tag, sub)
            continue
        relink(devkit / name, tmp / sub)
        manifest[name] = {"mode": "download@" + head, "source": "%s:%s" % (spec.url, sub)}
    logger.info("[下载] %s ← %s@%s", spec.tag, spec.url, head)
    return 0


def main():
    _setup_logging()
    devkit = cache_dir()
    src_url = os.environ.get("PYPTO_SRC_URL", "https://gitcode.com/cann/pypto.git")
    gym_url = os.environ.get("PYPTO_GYM_URL", "https://gitcode.com/cann/pypto-gym.git")
    # pro_ops 样例为 PyPTO-Pro 工作流额外拉取项，独立 URL（默认跟随 PYPTO_SRC_URL），
    # 不与 docs 共用，避免拉 pro_ops 时覆盖 docs 的来源。
    pro_ops_url = os.environ.get("PYPTO_PRO_OPS_URL", src_url)
    pin = ""
    argv = sys.argv[1:]
    if argv and argv[0] == "--pin":
        pin = argv[1] if len(argv) > 1 else ""

    if shutil.which("git") is None:
        logger.error("[缺 git] 请先安装 git")
        return 3
    try:
        devkit.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.error("[无法创建缓存] %s", devkit)
        return 3

    manifest = {}
    # pro_ops 与 docs 同源时并入同一次 clone（来源一致，不覆盖 docs）；异源时独立拉取。
    # 缓存目录名为 pro_ops，源仓子路径仍是 python/tests/ut/block/frontend/a5。
    pro_ops_pair = ("pro_ops", "python/tests/ut/block/frontend/a5")
    docs_pairs = [("docs", "docs/zh")]
    if pro_ops_url == src_url:
        docs_pairs.append(pro_ops_pair)
    rc = provision(devkit, SourceSpec("pypto", "docs/zh/api", src_url,
                                      os.environ.get("PYPTO_SRC", ""), docs_pairs), pin, manifest)
    if rc != 0:
        return rc
    rc = provision(devkit, SourceSpec("gym", "src/pypto_gym/ops", gym_url,
                                      os.environ.get("PYPTO_GYM_SRC", ""),
                                      [("ops", "src/pypto_gym/ops"), ("tests", "tests/ops")]), pin, manifest)
    if rc != 0:
        return rc
    # pro_ops 与 docs 异源时，作为独立拉取项从 pro_ops_url 获取（源无此路径时 provision 内部自动跳过）。
    if pro_ops_url != src_url:
        rc = provision(devkit, SourceSpec("pro_ops", "python/tests/ut/block/frontend/a5", pro_ops_url,
                                          os.environ.get("PYPTO_PRO_OPS_SRC", ""), [pro_ops_pair]), pin, manifest)
        if rc != 0:
            return rc

    (devkit / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dirs = " ".join(sorted(name + "/" for name in manifest))
    logger.info("[完成] 缓存: %s （%s + MANIFEST.json）", devkit, dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
