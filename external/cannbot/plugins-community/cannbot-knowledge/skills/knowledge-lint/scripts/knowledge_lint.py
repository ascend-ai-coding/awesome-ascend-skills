#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""knowledge_lint.py — deterministic, repo-wide, bundle-aware lint of OKF knowledge cards.

One aggregating entry that enforces the project's basic principles (SPEC-Reference /
SPEC-Source / SPEC-Graph / SPEC-Retrieve) across ALL bundles, and folds in the existing
`okf_graph verify` + `knowledge_query verify`. Zero LLM, deterministic, read-only (reports,
never edits). Semantic principles (distilled-not-copied, description accuracy, index
progressive-disclosure) are out of scope here — see knowledge-lint/SKILL.md §3 (`--sample`).

Usage:
  knowledge_lint.py [--bundle B] [--kind K] [--json]   # all deterministic checks + aggregate
  knowledge_lint.py --aggregate-only                   # only shell okf_graph/knowledge_query verify
  knowledge_lint.py --sample N [--bundle B] [--kind K]  # emit JSON worklist for semantic deep-audit
Exit code: 1 if any blocker (or aggregate verify fails), else 0.
"""
import argparse
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))   # skills/knowledge-lint/scripts/
ING = os.path.join(SKILL_DIR, "..", "..", "ops-knowledge-ingest", "scripts")        # 图簇 (okf_graph + shell verify)
REF_ING = os.path.join(SKILL_DIR, "..", "..", "ops-knowledge-reference-ingest", "scripts")  # reference 源簇 (okf_source)
SRCH = os.path.join(SKILL_DIR, "..", "..", "knowledge-query", "scripts")


def _prepend_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_script_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load module %s from %s" % (name, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_prepend_path(ING)
_prepend_path(REF_ING)
G = _load_script_module("cannbot_knowledge_okf_graph", os.path.join(ING, "okf_graph.py"))
S = _load_script_module("cannbot_knowledge_okf_source", os.path.join(REF_ING, "okf_source.py"))
# NB: do NOT `import knowledge_query` — it is now a transitional shim that runs an argparse
# CLI on import (retrieval engine moved to the knowledge-query skill). Reuse okf_graph.kind_of,
# and shell the retrieval `verify` as a subprocess in run_aggregate().

ROOT = G.ROOT


def _emit(message, *args):
    """Write one unprefixed log record to stdout, preserving the CLI contract."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord(__name__, logging.INFO, "", 0, message, args, None)
    handler.handle(record)
    handler.close()


def _read_text(path):
    with open(path, encoding="utf-8", errors="replace") as source_file:
        return source_file.read()

# --- per-bundle provenance policy (see plan §1; calibrated against the live repo) ---
GIT_BUNDLES = {"asc-devkit", "ops"}                      # resource/sources = GitCode @sha blob|tree
DOCSITE_BUNDLES = {"ascend-c-op-dev-guide", "ascend-c-profiling"}  # hiascend.com doc URLs
EMPTY_OK_BUNDLES = {"glossary", "runbooks"}             # resource may be empty
SNAKE_BUNDLES = {"asc-devkit", "glossary", "ascend-c-op-dev-guide", "ascend-c-profiling"}
ROLE_VOCAB = {"primary", "variant", "guide", "example", "faq",
              "constraint", "migration", "header", "impl"}
KNOWN_KINDS = {
    "api", "guide", "example", "operator", "glossary", "operator_optimization",
    "implementation_trap", "debugging_journey", "cross_skill_gap",
}
KNOWN_INDEX_KIND = "index"
KNOWN_SOURCE_FAMILIES = {
    "asc_devkit", "ascend_c_op_dev_guide", "ascend_c_profiling", "ops", "curated", "community",
}
RE_UTC_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
RE_SHA = re.compile(r"https://gitcode\.com/.+/(?:blob|tree)/[0-9a-f]{40}/")
RE_URL = re.compile(r"^https?://")
RE_IMG = re.compile(r"!\[[^\]]*\]\(|<img\b")
RE_CONFLICT = re.compile(r"^(<<<<<<< |=======$|>>>>>>> )", re.M)
RE_LEAD_NUM = re.compile(r"^\d+[_-]")
RE_FENCE = re.compile(r"```.*?```", re.S)
RE_INLINE = re.compile(r"`[^`]*`")
RE_MDLINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
RE_INDEX_ENTRY = re.compile(r"^\s*[*-]\s+\[[^\]]*\]\([^)]+\)(.*)$")


class Findings:
    def __init__(self):
        self.items = []   # (severity, group, card, msg)

    def add(self, sev, group, card, msg):
        t = (sev, group, card, msg)
        if t not in self.items:                        # dedupe identical findings
            self.items.append(t)

    def blockers(self):
        return [i for i in self.items if i[0] == "blocker"]


def rel(path):
    return os.path.relpath(path, ROOT)


# ---------------- card-level checks ----------------
def _check_card_naming(nid, card, findings):
    bundle = card["bundle"]
    fname = os.path.basename(nid)
    if RE_LEAD_NUM.match(fname):
        findings.add("blocker", "A 命名/结构", nid, "文件名含前导序号（剥除：snake_case 无数字前缀）")
    if nid.count("/") > 3:
        findings.add("blocker", "A 命名/结构", nid, "目录深度 >3（自内容根 ≤3 层，更深用 tags）")
    stem = fname[:-3]
    # snake_case applies to doc/concept bundles; runbooks/ops field-notes use s<hash>_<TYPE><N> IDs
    if " " in fname or (bundle in SNAKE_BUNDLES and re.search(r"[A-Z]", stem)):
        findings.add("warn", "A 命名/结构", nid, "文件名非 snake_case（含空格/大写）")


def _check_card_frontmatter(nid, raw, frontmatter, findings):
    required_fields = (
        "schema_version", "kind", "type", "source_family", "title", "description", "tags",
        "created_at", "updated_at",
    )
    for field in required_fields:
        if not (frontmatter.get(field) or "").strip():
            findings.add("blocker", "B frontmatter", nid, "缺 frontmatter `%s`" % field)
    if frontmatter.get("schema_version") and frontmatter.get("schema_version") != "okf.v1":
        message = "`schema_version` 非 okf.v1：%r" % frontmatter.get("schema_version")
        findings.add("blocker", "B frontmatter", nid, message)
    if frontmatter.get("kind") and frontmatter.get("kind") not in KNOWN_KINDS:
        message = "`kind=%r` 未在 okf.v1 profile 中定义" % frontmatter.get("kind")
        findings.add("blocker", "B frontmatter", nid, message)
    elif (
        frontmatter.get("type")
        and G.kind_of(frontmatter.get("type", ""), frontmatter.get("kind", "")) == "other"
    ):
        findings.add(
            "blocker", "B frontmatter", nid,
            "`type=%r` 未映射到已知 kind" % frontmatter.get("type"),
        )
    source_family = frontmatter.get("source_family")
    if source_family and source_family not in KNOWN_SOURCE_FAMILIES:
        message = "`source_family=%r` 未在受控词表中定义" % source_family
        findings.add("blocker", "B frontmatter", nid, message)
    if frontmatter.get("timestamp"):
        findings.add("blocker", "B frontmatter", nid, "仍使用 legacy `timestamp`，应迁移为 `created_at`/`updated_at`")
    for field in ("created_at", "updated_at"):
        if frontmatter.get(field) and not RE_UTC_Z.match(frontmatter.get(field, "")):
            message = "`%s` 非 UTC ISO-8601 Z 格式：%r" % (field, frontmatter.get(field))
            findings.add("blocker", "B frontmatter", nid, message)
    tags = G.parse_tags(frontmatter.get("tags", ""))
    if len(tags) != len(set(tags)):
        findings.add("blocker", "B frontmatter", nid, "tags 存在重复项")
    structural_tags = (
        "cann", "ascend-c", "asc-devkit", "op-dev-guide", "profiling", "example", "glossary", "index",
    )
    for bad in structural_tags:
        if bad in tags:
            message = "tags 仍含结构型标签 `%s`（应由 kind/source_family/type 表达）" % bad
            findings.add("warn", "B frontmatter", nid, message)
    if RE_CONFLICT.search(raw):
        findings.add("blocker", "B frontmatter", nid, "正文含 merge 冲突标记")


def _check_card_provenance(nid, bundle, resource, sources, findings):
    if sources:
        primary_sources = [url for url, role in sources if role == "primary"]
        if len(primary_sources) != 1:
            findings.add(
                "blocker", "C 多源溯源", nid,
                "primary 源数=%d（须有且仅一）" % len(primary_sources),
            )
        elif resource != primary_sources[0]:
            findings.add("blocker", "C 多源溯源", nid, "`resource` != primary.url")
        for url, role in sources:
            if role not in ROLE_VOCAB:
                findings.add("blocker", "C 多源溯源", nid, "role 不在受控词表：%r" % role)
            _check_url_form(nid, bundle, url, findings, "sources[].url")
    if resource or bundle not in EMPTY_OK_BUNDLES:
        _check_url_form(nid, bundle, resource, findings, "resource")


@dataclass(frozen=True)
class _CardText:
    path: str
    raw: str
    body: str


def _check_card_body(nid, card, text, findings):
    if RE_IMG.search(text.body):
        findings.add("blocker", "D 正文", nid, "正文嵌入图片（改文字链接）")
    if text.raw.count("<!-- okf:related:start") > 1 or text.raw.count("<!-- okf:related:end") > 1:
        findings.add("blocker", "D 正文", nid, "多个 `# 相关` 托管块（至多一对）")
    if card.get("status") != "stub":
        # Authoring-intent markers, excluding descriptive words about an upstream stub.
        for placeholder in ("TODO", "FIXME", "待补充", "待完善", "待填"):
            if placeholder in text.body:
                findings.add("warn", "D 正文", nid, "正文含未完成标记 %r" % placeholder)
                break
    _check_body_links(nid, text.path, text.body, findings)


def check_card(nid, card, findings):
    bundle = card["bundle"]
    path = card["path"]
    raw = _read_text(path)
    body = G.strip_managed(raw)
    fm = S.read_fm(path)
    srcs = S.parse_sources(path)
    res = (fm.get("resource", "") or "").strip().strip("'\"")
    _check_card_naming(nid, card, findings)
    _check_card_frontmatter(nid, raw, fm, findings)
    _check_card_provenance(nid, bundle, res, srcs, findings)
    _check_card_body(nid, card, _CardText(path, raw, body), findings)


def _check_url_form(nid, bundle, url, findings, field):
    if not url:
        if bundle in EMPTY_OK_BUNDLES:
            return
        findings.add("blocker", "C 多源溯源", nid, "%s 为空（该 bundle 须有溯源）" % field)
        return
    if bundle in GIT_BUNDLES:
        if not RE_SHA.search(url):
            findings.add("blocker", "C 多源溯源", nid, "%s 非 GitCode @sha blob/tree 永久链接：%s" % (field, url[:70]))
    else:  # docsite / empty-ok with a value → just require a real URL
        if not RE_URL.match(url):
            findings.add("blocker", "C 多源溯源", nid, "%s 非合法 URL：%s" % (field, url[:70]))


def _check_body_links(nid, path, body, findings):
    # strip code so C++ signatures like operator[](const T&) aren't read as md links
    txt = RE_INLINE.sub("", RE_FENCE.sub("", body))
    cdir = os.path.dirname(path)
    for m in RE_MDLINK.finditer(txt):
        u = m.group(1).split()[0].strip("<>")          # markdown [t](<path>) angle form
        if u.startswith(("http", "#", "mailto:")):
            continue
        u = u.split("#")[0]
        if not u or ("/" not in u and not u.endswith(".md")):  # path-like only
            continue
        tgt = os.path.normpath(os.path.join(cdir, urllib.parse.unquote(u)))
        if not os.path.exists(tgt):
            findings.add("warn", "D 正文", nid, "正文死链：%s" % u[:70])


# ---------------- index.md checks ----------------
def _iter_index_paths(roots):
    for base in roots:
        for current_root, _, filenames in os.walk(base):
            if "index.md" in filenames:
                yield current_root, os.path.join(current_root, "index.md")


def _check_index_file(current_root, index_path, pinned, findings):
    text = _read_text(index_path)
    for line in text.splitlines():
        match = RE_INDEX_ENTRY.match(line)
        if match and not match.group(1).strip(" -—:："):
            message = "index 条目缺描述：%s" % line.strip()[:60]
            findings.add("blocker", "E index/导航", rel(index_path), message)
    if current_root in pinned:
        frontmatter = S.read_fm(index_path)
        for key in ("upstream_repo", "upstream_ref", "upstream_commit"):
            if not frontmatter.get(key):
                message = "bundle 根 index.md 缺 pin 字段 `%s`" % key
                findings.add("blocker", "E index/导航", rel(index_path), message)


def check_indexes(findings):
    roots = [G.REFERENCE] + [b for b, _, _ in G.EXTRA_ROOTS if os.path.isdir(b)]
    pinned = {os.path.join(G.REFERENCE, b) for b in GIT_BUNDLES}  # git-bundle roots need a pin
    for current_root, index_path in _iter_index_paths(roots):
        _check_index_file(current_root, index_path, pinned, findings)


# ---------------- log/ checks ----------------
RE_LOGFILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
RE_LOGHEAD = re.compile(r"^## \[(\d{2}):(\d{2})\]")


def check_logs(findings):
    """Validate day-log filenames, timestamps, and descending order.

    This deterministic, read-only check runs only during repo-wide lint. The autofix is
    ``log/sort_logs.py``; operation vocabulary is intentionally out of scope.
    """
    logdir = os.path.join(ROOT, "log")
    if not os.path.isdir(logdir):
        return
    for fn in sorted(fn for fn in os.listdir(logdir) if RE_LOGFILE.match(fn)):
        date, card = fn[:-3], "log/" + fn
        lines = _read_text(os.path.join(logdir, fn)).split("\n")
        h = lines[0].strip() if lines else ""
        if h != "# " + date:
            findings.add("blocker", "G 日志", card, "首行应为 `# %s`，实为 `%s`（跑 sort_logs.py 规范化）" % (date, h[:40]))
        times, bad = [], False
        for ln in lines:
            m = RE_LOGHEAD.match(ln)
            if not m:
                continue
            hh, mm = int(m.group(1)), int(m.group(2))
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                findings.add("blocker", "G 日志", card, "非法时间 [%s:%s]" % (m.group(1), m.group(2)))
                bad = True
            times.append(hh * 60 + mm)
        if not bad and times != sorted(times, reverse=True):
            message = "条目非倒序（最新应在前）；跑 `python3 log/sort_logs.py %s` 修复" % card
            findings.add("blocker", "G 日志", card, message)


# ---------------- aggregate (existing verifies) ----------------
def _last_line(text):
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


def run_aggregate(findings):
    for script, sub in ((os.path.join(ING, "okf_graph.py"), "verify"),
                        (os.path.join(SRCH, "knowledge_query.py"), "verify")):
        try:
            p = subprocess.run(
                [sys.executable, script, sub],
                capture_output=True,
                text=True,
                cwd=ROOT,
                check=False,
            )
            tail = _last_line(p.stdout)
            if p.returncode != 0:
                err_tail = _last_line(p.stdout + p.stderr) or "no output"
                findings.add("blocker", "F 聚合 verify", os.path.basename(script),
                             "%s verify 失败：%s" % (os.path.basename(script), err_tail[:120]))
            else:
                findings.add("ok", "F 聚合 verify", os.path.basename(script), tail[:120])
        except (OSError, subprocess.SubprocessError) as e:
            findings.add("blocker", "F 聚合 verify", os.path.basename(script), "无法运行：%s" % e)


# ---------------- driver ----------------
def select(nodes, bundle, kind):
    out = []
    for nid in G.concept_ids(nodes):
        d = nodes[nid]
        if bundle and d["bundle"] != bundle:
            continue
        if kind and d["kind"] != kind:
            continue
        out.append(nid)
    return sorted(out)


def cmd_sample(nodes, n, bundle, kind):
    ids = select(nodes, bundle, kind)
    step = max(1, len(ids) // n) if n else 1
    picks = ids[::step][:n]
    out = []
    for nid in picks:
        d = nodes[nid]
        body = G.strip_managed(_read_text(d["path"]))
        out.append({"id": nid, "path": rel(d["path"]), "bundle": d["bundle"], "kind": d["kind"],
                    "sources": [{"url": u, "role": r} for u, r in S.parse_sources(d["path"])],
                    "resource": d["resource"], "excerpt": body[:600]})
    _emit("%s", json.dumps(out, ensure_ascii=False, indent=1))


def main():
    if G.report_root_argument_error():
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge-root", help="OKF knowledge-base root")
    ap.add_argument("--bundle")
    ap.add_argument("--kind")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--sample", type=int)
    a = ap.parse_args()

    nodes = G.load_nodes()
    if a.sample:
        return cmd_sample(nodes, a.sample, a.bundle, a.kind)

    findings = Findings()
    if not a.aggregate_only:
        for nid in select(nodes, a.bundle, a.kind):
            check_card(nid, nodes[nid], findings)
        if not a.bundle and not a.kind:
            check_indexes(findings)
            check_logs(findings)
    run_aggregate(findings)

    items = sorted(
        findings.items,
        key=lambda item: ({"blocker": 0, "warn": 1, "ok": 2}[item[0]], item[1], item[2], item[3]),
    )
    nb = sum(1 for i in items if i[0] == "blocker")
    nw = sum(1 for i in items if i[0] == "warn")
    if a.json:
        output = {
            "blocker": nb,
            "warn": nw,
            "findings": [
                {"severity": severity, "group": group, "card": card, "msg": message}
                for severity, group, card, message in items
            ],
        }
        _emit("%s", json.dumps(output, ensure_ascii=False, indent=1))
    else:
        groups = {}
        for s, g, c, m in items:
            groups.setdefault(g, []).append((s, c, m))
        _emit("=== knowledge-lint%s ===", "" if not a.bundle else " [bundle=%s]" % a.bundle)
        for g in sorted(groups):
            _emit("\n## %s", g)
            for s, c, m in groups[g]:
                mark = {"blocker": "✗", "warn": "!", "ok": "✓"}[s]
                _emit("  %s %s — %s", mark, c, m)
        _emit("\n--- blocker: %d  warn: %d ---", nb, nw)
        _emit("OK" if nb == 0 else "FAIL")
    return 1 if nb else 0


if __name__ == "__main__":
    raise SystemExit(main())
