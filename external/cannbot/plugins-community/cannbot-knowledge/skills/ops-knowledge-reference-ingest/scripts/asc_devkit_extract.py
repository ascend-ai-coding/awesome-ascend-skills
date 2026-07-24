#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""asc_devkit_extract.py — extract asc-devkit @ 9.0.0 into an OKF knowledge bundle.

See SPEC-Reference.md / SPEC-Source.md. Division of labour:
  * THIS SCRIPT is pure-Python and deterministic: pin / scaffold / index / source / verify.
  * concept BODIES are LLM-distilled by the agent orchestration layer (NOT this script) — it
    reads .build/worklist.json, dispatches subagents that read .build/<source> and write the
    concise concept files. This script only prepares the worklist and validates the result.

Source/provenance (SPEC-Source.md): provenance lives per-concept in frontmatter `resource:`;
the bundle-level pin (repo/ref/commit) is written to the bundle-root index.md frontmatter; the
concept->source table is SYNTHESIZED on demand by `source` (delegates to the
plugin-local ops-knowledge-reference-ingest source adapter), never固化 into a checked-in manifest.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import okf_source as _source_runtime

CliError = _source_runtime.CliError
STDOUT_LOGGER = _source_runtime.make_cli_logger(__name__ + ".stdout", sys.stdout)
STDERR_LOGGER = _source_runtime.make_cli_logger(__name__ + ".stderr", sys.stderr)

ROOT = _source_runtime.ROOT
BUILD = os.path.join(ROOT, ".build", "asc-devkit")
REFERENCE = os.path.join(ROOT, "reference")
BUNDLE = "asc-devkit"
BUNDLE_REF = os.path.join(REFERENCE, BUNDLE)
WORKLIST = os.path.join(ROOT, ".build", "worklist.json")
MANIFEST_LIST = os.path.join(ROOT, ".build", "manifest_list.json")

UPSTREAM_REPO = "https://gitcode.com/cann/asc-devkit"
UPSTREAM_REF = "9.0.0"
TIMESTAMP = "2026-06-24T00:00:00Z"

# TOC concept-domain heading (zh) -> english snake_case category dir
CATEGORY_SLUG = {
    "基础数据结构": "basic_data_structures",
    "基础API": "basic_api",
    "高阶API": "advanced_api",
    "SIMT API": "simt_api",
    "Utils API": "utils_api",
    "AI CPU API": "aicpu_api",
    "C API": "c_api",
}
API_SUBDIRS = ("c_api", "tensor_api", "utils")     # context/<subdir>/ -> api/_subtree/<subdir>/
INCLUDE_TIERS = ("basic_api", "adv_api", "c_api", "simt_api", "aicpu_api", "utils", "experimental")
CODE_MARKERS = (".asc", ".cpp", ".cc", ".cu")
CODE_FILES = ("CMakeLists.txt",)

RE_HREF = re.compile(r'href="context[\\/]+([^"]+\.md)"')
RE_MDLINK = re.compile(r'href="(?:context[\\/]+)?([^"]+\.md)"|\]\(([^)]+\.md)\)')
RE_H1 = re.compile(r"^#\s+(.*?)\s*$")
RE_ANCHOR = re.compile(r'<a name="[^"]*"></a>')


# --- helpers ---------------------------------------------------------------
def _git_executable():
    """Resolve git through PATH while passing an absolute executable to subprocess."""
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError(2, os.strerror(2), "git")
    return os.path.abspath(executable)


def sha():
    return subprocess.check_output(
        [_git_executable(), "-C", BUILD, "rev-parse", "HEAD"], text=True
    ).strip()


def _url(p):
    return "/".join(urllib.parse.quote(s) for s in p.split("/"))


def blob_url(s, relpath):
    return "%s/blob/%s/%s" % (UPSTREAM_REPO, s, _url(relpath))


def tree_url(s, relpath):
    return "%s/tree/%s/%s" % (UPSTREAM_REPO, s, _url(relpath))


def slug(name):
    name = re.sub(r"\.(md|h)$", "", name)
    name = re.sub(r"^\d+[_\-]+", "", name)        # drop leading ordering prefix (01_, 00-, 001-)
    out = re.sub(r"[^0-9A-Za-z一-鿿]+", "_", name).strip("_").lower()
    return out or "x"


def hash6(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:6]


def read(relpath):
    with open(os.path.join(BUILD, relpath), encoding="utf-8", errors="replace") as f:
        return f.read()


def first_heading(text, fallback):
    for line in text.splitlines():
        m = RE_H1.match(line)
        if m:
            return RE_ANCHOR.sub("", m.group(1)).strip() or fallback
    return fallback


def first_sentence(text, limit=140):
    """First substantial prose sentence (skip headings/html/list/code markers)."""
    for raw in RE_ANCHOR.sub("", text).splitlines():
        s = re.sub(r"<[^>]+>", "", raw).strip()
        if not s or s.startswith(("#", "|", "-", "*", ">", "!", "```", "<")):
            continue
        s = re.split(r"[。.!?！？\n]", s)[0].strip()
        if len(s) >= 6:
            return s[:limit]
    return ""


def md_rel(p):
    return "/".join(urllib.parse.quote(s) for s in p.split(os.sep))


def _slug_path_parts(path):
    """Return normalized non-empty path components."""
    parts = []
    for part in path.split(os.sep):
        if part and part != ".":
            parts.append(slug(part))
    return parts


# --- API top-level categorization: 3-level deterministic cascade -----------
def _api_links(filename, top_level_files):
    """Return top-level API pages linked by one API context page."""
    text = read("docs/api/context/" + filename)
    links = set()
    for match in RE_MDLINK.finditer(text):
        rel = (match.group(1) or match.group(2)).replace("\\", "/").split("/")[-1]
        if rel in top_level_files:
            links.add(rel)
    return links


def _api_base_name(filename):
    """Return the normalized basename used for category inheritance."""
    name = filename[:-3]
    name = re.sub(r"-\d+$", "", name)
    name = re.sub(r"(接口|-Tiling|-tiling)$", "", name)
    return re.sub(r"（[^）]*）$", "", name)


def categorize_api_toplevel():
    """Return dict basename(context/*.md) -> category_slug for TOC-reachable files."""
    ctx = os.path.join(BUILD, "docs/api/context")
    top = set(f for f in os.listdir(ctx) if f.endswith(".md"))
    readme = read("docs/api/README.md")
    cat = {}
    cur = None
    for line in readme.splitlines():
        h = re.match(r"^##\s+(.*?)(?:<a name=.*)?$", line)
        if h:
            cur = h.group(1).strip()
            continue
        for m in RE_HREF.finditer(line):
            rel = m.group(1).replace("\\", "/")
            if "/" not in rel and cur in CATEGORY_SLUG:
                cat.setdefault(rel, CATEGORY_SLUG[cur])

    # L2 transitive link-graph propagation to fixpoint
    pending = list(cat)
    cursor = 0
    while cursor < len(pending):
        filename = pending[cursor]
        cursor += 1
        category = cat.get(filename)
        for child in _api_links(filename, top):
            if child not in cat:
                cat[child] = category
                pending.append(child)

    # L3 base-name inheritance
    for fn in top:
        if fn in cat:
            continue
        base_name = _api_base_name(fn)
        for cand in (base_name + ".md", base_name + "接口.md"):
            if cand in cat:
                cat[fn] = cat[cand]
                break
    return cat, top


# --- worklist construction -------------------------------------------------
@dataclass(frozen=True)
class BuilderEntry:
    """Related inputs for one deterministic worklist entry."""

    concept_rel: str
    source_rel: str
    kind: str
    title: str
    path_tags: list
    category: str = ""
    is_tree: bool = False
    url_target: str | None = None


def _include_entries(tier, tier_dir):
    for root, _, filenames in os.walk(tier_dir):
        for filename in sorted(filenames):
            if not filename.endswith(".h"):
                continue
            source = os.path.relpath(os.path.join(root, filename), BUILD)
            middle_tags = _slug_path_parts(os.path.relpath(root, tier_dir))
            concept = "include/%s/%s.md" % (tier, slug(filename))
            yield BuilderEntry(
                concept, source, "header", filename, ["header", tier] + middle_tags,
                category=tier,
            )


class Builder:
    def __init__(self, s):
        self.s = s
        self.used = {}            # output concept relpath -> source relpath
        self.items = []

    def add(self, entry):
        # flat dir: collision within a category -> stable hash6 suffix on the filename
        concept_rel = entry.concept_rel
        source_rel = entry.source_rel
        if concept_rel in self.used and self.used[concept_rel] != source_rel:
            stem, ext = os.path.splitext(concept_rel)
            concept_rel = "%s__%s%s" % (stem, hash6(source_rel), ext)
        self.used[concept_rel] = source_rel
        target = entry.url_target if entry.url_target is not None else source_rel
        url = tree_url(self.s, target) if entry.is_tree else blob_url(self.s, target)
        self.items.append({
            "concept": concept_rel, "source": source_rel, "kind": entry.kind,
            "title": entry.title, "url": url, "is_tree": entry.is_tree,
            "category": entry.category, "path_tags": _dedup(entry.path_tags),
        })

    # --- guides (flat: guides/<name>.md) ---
    def guides(self):
        for f in sorted(os.listdir(os.path.join(BUILD, "docs"))):
            if f.endswith(".md") and f.lower() != "readme.md":
                src = "docs/" + f
                self.add(BuilderEntry(
                    "guides/%s.md" % slug(f), src, "guide",
                    first_heading(read(src), f[:-3]), ["guide"], category="guides",
                ))

    # --- api (flat: api/<category>/<name>.md ; deeper path -> tags) ---
    def api(self):
        cat, _ = categorize_api_toplevel()
        ctx = "docs/api/context"
        for r, _, fs in os.walk(os.path.join(BUILD, ctx)):
            for f in fs:
                if not f.endswith(".md") or f.lower() == "readme.md":
                    continue
                src = os.path.relpath(os.path.join(r, f), BUILD)
                parts = os.path.relpath(os.path.join(r, f),
                                        os.path.join(BUILD, ctx)).split(os.sep)
                if len(parts) > 1 and parts[0] in API_SUBDIRS:     # context/<subdir>/...
                    category = parts[0]
                    tags = [category] + [slug(p) for p in parts[1:-1]]
                else:                                              # top-level context md
                    category = cat.get(f) or "uncategorized"
                    tags = [category]
                out = "api/%s/%s.md" % (category, slug(f))
                self.add(BuilderEntry(
                    out, src, "api", first_heading(read(src), f[:-3]), tags,
                    category=category,
                ))

    # --- examples (flat: examples/<topcat>/<leaf>.md ; deeper path -> tags) ---
    # Only LEAF dirs (README + no child README) become concepts; branch dirs are
    # navigation only (surfaced via index.md), not concepts.
    def examples(self):
        base = os.path.join(BUILD, "examples")
        for r, _, fs in os.walk(base):
            if "README.md" not in fs:
                continue
            if any("README.md" in cfs for cr, _, cfs in os.walk(r) if cr != r):
                continue                                  # branch dir -> skip (index.md only)
            rel = os.path.relpath(r, base)
            if rel == ".":
                continue
            parts = rel.split(os.sep)
            topcat = slug(parts[0])
            leaf = slug(parts[-1])
            mid = [slug(p) for p in parts[1:-1]]
            out = "examples/%s/%s.md" % (topcat, leaf)
            src = os.path.relpath(os.path.join(r, "README.md"), BUILD)
            self.add(BuilderEntry(
                out, src, "example", first_heading(read(src), os.path.basename(r)),
                ["example", topcat] + mid, category=topcat, is_tree=True,
                url_target=os.path.relpath(r, BUILD),
            ))

    # --- include (flat: include/<tier>/<name>.md ; deeper path -> tags) ---
    def include(self):
        base = os.path.join(BUILD, "include")
        for tier in INCLUDE_TIERS:
            tdir = os.path.join(base, tier)
            if not os.path.isdir(tdir):
                continue
            for entry in _include_entries(tier, tdir):
                self.add(entry)


# --- pilot fixture (deterministic by SOURCE, hits all hard cases) ----------
def pilot_select(all_items):
    by_src = {it["source"]: it for it in all_items}
    pick, seen = [], set()

    def add_item(it):
        if it and it["concept"] not in seen:
            seen.add(it["concept"])
            pick.append(it)

    def take(pred):
        for it in sorted(all_items, key=lambda x: x["source"]):
            if pred(it) and it["concept"] not in seen:
                add_item(it)
                return

    add_item(by_src.get("docs/api/context/Add.md"))                      # api w/ formula image
    take(lambda it: it["kind"] == "api" and it["category"] == "uncategorized")
    take(lambda it: it["kind"] == "api" and it["category"] == "advanced_api")  # L2/L3 categorized
    take(lambda it: it["kind"] == "api" and it["category"] == "c_api"
         and "context/c_api/" in it["source"])                          # subtree c_api
    take(lambda it: it["kind"] == "api" and it["category"] == "tensor_api")    # subtree tensor_api
    take(lambda it: it["kind"] == "example")                            # a flattened example leaf
    add_item(by_src.get("include/basic_api/kernel_operator_vec_binary_intf.h")
             or next((it for it in sorted(all_items, key=lambda x: x["source"])
                      if it["kind"] == "header"), None))                # a header
    take(lambda it: it["kind"] == "guide")                             # a guide
    return pick


def _dedup(seq):
    out = []
    for x in seq:
        if x and x not in out:
            out.append(x)
    return out


STRUCTURAL_TAGS = {"cann", "ascend-c", "asc-devkit", "example", "guide", "index", "api", "reference"}


def _strip_frontmatter(text):
    if not text.startswith("---\n"):
        return text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1:]).lstrip("\n")
    return text


@dataclass(frozen=True)
class IndexOptions:
    """Optional metadata used to render an index frontmatter block."""

    source_family: str = "asc_devkit"
    idx_type: str = "section_index"
    pins: dict | None = None
    created_at: str | None = None
    updated_at: str | None = None


def _index_frontmatter(title, desc, options=None):
    options = options or IndexOptions()
    pins = options.pins or {}
    lines = ["---",
             "schema_version: okf.v1",
             "kind: index",
             "type: %s" % options.idx_type,
             "source_family: %s" % options.source_family,
             "title: %s" % json.dumps(title, ensure_ascii=False),
             "description: %s" % json.dumps(desc or ("导航索引：%s。" % title), ensure_ascii=False),
             "tags: []"]
    for k in ("upstream_repo", "upstream_doc", "upstream_ref", "upstream_commit"):
        if pins.get(k):
            val = pins[k]
            lines.append("%s: %s" % (k, json.dumps(val, ensure_ascii=False) if k == "upstream_ref" else val))
    lines += ["created_at: '%s'" % (options.created_at or TIMESTAMP),
              "updated_at: '%s'" % (options.updated_at or TIMESTAMP),
              "---", ""]
    return "\n".join(lines)


# ============================ subcommands ==================================
def cmd_pin():
    STDOUT_LOGGER.info("pinned %s @ %s", BUNDLE, sha())


def cmd_scaffold(mode, n):
    s = sha()
    b = Builder(s)
    b.guides()
    b.api()
    b.examples()
    b.include()
    items = b.items
    if mode == "pilot":
        items = pilot_select(items)
    elif mode == "sample":
        # deterministic: first n per CATEGORY directory (<section>/<category>)
        groups = {}
        for it in sorted(b.items, key=lambda x: x["source"]):
            g = os.path.dirname(it["concept"])      # the flat category dir
            groups.setdefault(g, [])
            if len(groups[g]) < n:
                groups[g].append(it)
        items = [it for g in sorted(groups) for it in groups[g]]
    os.makedirs(os.path.dirname(WORKLIST), exist_ok=True)
    with open(WORKLIST, "w", encoding="utf-8") as f:
        json.dump({"sha": s, "mode": mode, "items": items}, f, ensure_ascii=False, indent=1)
    with open(MANIFEST_LIST, "w", encoding="utf-8") as f:
        json.dump(sorted(it["concept"] for it in items), f, ensure_ascii=False, indent=1)
    _write_dir_desc()
    STDOUT_LOGGER.info("scaffold mode=%s items=%d -> %s", mode, len(items), WORKLIST)
    # report
    kinds = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    STDOUT_LOGGER.info("  by kind: %s", kinds)


DIR_DESC = os.path.join(ROOT, ".build", "dir_desc.json")


def _write_dir_desc():
    """Per-directory descriptions sourced from each dir's OWN upstream README/landing
    page, so every reference dir level gets a specific (not borrowed) description."""
    desc = {}
    # examples: each dir with README -> first sentence of that dir's README
    exbase = os.path.join(BUILD, "examples")
    for r, _, fs in os.walk(exbase):
        if "README.md" not in fs:
            continue
        rel = os.path.relpath(r, exbase)
        ref = "examples" if rel == "." else "examples/" + "/".join(
            slug(p) for p in rel.split(os.sep))
        d = first_sentence(read(os.path.relpath(os.path.join(r, "README.md"), BUILD)))
        if d:
            desc[ref] = d
    # api concept domains: README intro blurbs
    desc.update(_readme_category_desc())
    with open(DIR_DESC, "w", encoding="utf-8") as f:
        json.dump(desc, f, ensure_ascii=False, indent=1)


# concept discovery (exclude index.md)
def concept_files():
    out = []
    for r, _, fs in os.walk(BUNDLE_REF):
        for f in fs:
            if f.endswith(".md") and f != "index.md":
                out.append(os.path.relpath(os.path.join(r, f), BUNDLE_REF))
    return sorted(out)


def read_fm(path):
    fm = {}
    with open(path, encoding="utf-8") as f:
        if f.readline().strip() != "---":
            return fm
        for line in f:
            if line.strip() == "---":
                break
            m = re.match(r"^(\w+):\s*(.*)$", line)
            if m:
                fm[m.group(1)] = m.group(2).strip().strip("'\"")
    return fm


def kind_of(fm):
    t = fm.get("type", "")
    if t.endswith("Example"):
        return "example"
    if t.endswith("Header"):
        return "header"
    return "doc"


# --- curated directory descriptions (so parent indexes are meaningful) -----
TITLE_MAP = {
    ".": "asc-devkit (Ascend C DevKit)",
    "guides": "Guides", "api": "API Reference", "examples": "Examples",
    "include": "Include Headers", "api/_subtree": "API — 子目录镜像（_subtree）",
    "api/uncategorized": "API — 未归类（uncategorized）",
}
CURATED_DESC = {
    ".": "Ascend C DevKit（asc-devkit @ 9.0.0）结构化抽取的 OKF 知识库，涵盖指南、API 参考、样例与头文件。",
    "guides": "面向开发者的上手与贡献指南（quick start、各层 API 贡献规范等）。",
    "api": "Ascend C 编程类库 API 参考，按概念域平铺组织（上游更深的目录层级通过 tags 区分）。",
    "api/tensor_api": "Tensor/Layout 相关接口（上游 context/tensor_api，更细分类见 tags）。",
    "api/utils": "Utils 辅助接口（上游 context/utils）。",
    "api/uncategorized": "未被 TOC、链接图传播、基名继承覆盖的顶层 API（多为独立页）。",
    "api/aicpu_api": "AI CPU API：运行在 AI CPU 上的算子接口。",
    "examples": "asc-devkit 样例算子，保留上游目录层级（SIMD C++ / SIMD C / SIMT 三类）。",
    "include": "asc-devkit 公共头文件，按 API 层级（tier）组织。",
    "include/basic_api": "基础 API（硬件能力抽象）头文件。",
    "include/adv_api": "高阶 API 头文件。",
    "include/c_api": "语言扩展层 C API 头文件。",
    "include/simt_api": "SIMT API 头文件。",
    "include/aicpu_api": "AI CPU API 头文件。",
    "include/utils": "Utils 公共辅助头文件。",
    "include/experimental": "实验特性头文件。",
}


BUNDLE_INTRO = (
    "本知识库由华为 **Ascend C 开发套件 asc-devkit**（GitCode，`9.0.0` 分支，锁定 commit）结构化抽取而成。"
    "asc-devkit 是面向昇腾 AI 处理器的算子开发工具集，提供 Ascend C 编程语言的多层接口与配套样例。\n\n"
    "本库把上游内容按知识域**扁平**组织为简洁的 OKF concept（蒸馏而非照搬），上游更深的目录层级用 `tags` 表达，"
    "并由逐层 `index.md` 渐进式导航；每个 concept 的 `resource` 指向上游 @commit 的永久链接。包含以下部分："
)


def _bundle_pin():
    """Bundle-level upstream pin (the ONLY source datum not derivable per-concept).
    Commit comes from live .build git when available, else preserved from the
    existing bundle-root index.md frontmatter (idempotent without .build)."""
    p = os.path.join(BUNDLE_REF, "index.md")
    existing = read_fm(p) if os.path.exists(p) else {}
    commit = existing.get("upstream_commit", "")
    if os.path.isdir(BUILD):
        try:
            commit = sha()
        except (OSError, subprocess.SubprocessError):
            STDERR_LOGGER.debug("unable to refresh bundle commit", exc_info=True)
    return {"upstream_repo": UPSTREAM_REPO, "upstream_ref": UPSTREAM_REF,
            "upstream_commit": commit,
            "created_at": existing.get("created_at", existing.get("timestamp", TIMESTAMP)),
            "updated_at": TIMESTAMP}


def _write_bundle_pin(pin):
    """Prepend the bundle-level pin as frontmatter on the bundle-root index.md
    (an OKF-reserved, consumer-visible file). See SPEC-Source.md §2."""
    p = os.path.join(BUNDLE_REF, "index.md")
    with open(p, encoding="utf-8") as index_file:
        body = _strip_frontmatter(index_file.read())
    options = IndexOptions(
        idx_type="bundle_index", pins=pin,
        created_at=pin.get("created_at"), updated_at=pin.get("updated_at"),
    )
    fm = _index_frontmatter(
        TITLE_MAP.get(".", BUNDLE), CURATED_DESC.get(".", BUNDLE_INTRO.split("\n", 1)[0]), options
    )
    with open(p, "w", encoding="utf-8") as index_file:
        index_file.write(fm + body)


def cmd_index(no_pin=False):
    curated = {}
    if os.path.exists(DIR_DESC):                 # per-dir descriptions from own README (deep dirs)
        with open(DIR_DESC, encoding="utf-8") as descriptions_file:
            curated.update(json.load(descriptions_file))
    curated.update(CURATED_DESC)                 # curated structural text wins for top sections
    curated.update(_readme_category_desc())      # api/<slug> blurbs from README intro
    if no_pin:
        # version-bump mode: preserve existing pin from bundle-root index.md (don't refresh from .build HEAD)
        p = os.path.join(BUNDLE_REF, "index.md")
        pin = _bundle_pin_preserved(p)
    else:
        pin = _bundle_pin()                      # capture BEFORE deleting stale index.md
    # clear stale index.md, then regenerate
    for r, _, fs in os.walk(BUNDLE_REF):
        if "index.md" in fs:
            os.remove(os.path.join(r, "index.md"))
    _gen_index_rec(BUNDLE_REF, curated)
    _write_bundle_pin(pin)                        # bundle-level pin -> bundle-root index.md frontmatter
    _append_top_index(BUNDLE, "华为 Ascend C 开发套件 asc-devkit（GitCode @ 9.0.0）的结构化知识库："
                      "涵盖多层 API 参考、样例算子、公共头文件与开发指南。")
    STDOUT_LOGGER.info("indexes rebuilt (no_pin=%s)", no_pin)


def _bundle_pin_preserved(index_path):
    """Read pin from existing bundle-root index.md frontmatter without touching .build.
    Used by `index --no-pin` during version-bump (advance-pin is deferred to finalize)."""
    existing = read_fm(index_path) if os.path.exists(index_path) else {}
    return {"upstream_repo": UPSTREAM_REPO, "upstream_ref": existing.get("upstream_ref", UPSTREAM_REF),
            "upstream_commit": existing.get("upstream_commit", ""),
            "created_at": existing.get("created_at", existing.get("timestamp", TIMESTAMP)),
            "updated_at": existing.get("updated_at", existing.get("timestamp", TIMESTAMP))}


def _append_top_index(bundle, short):
    """Append this bundle to reference/index.md if missing (never clobber other bundles)."""
    top = os.path.join(REFERENCE, "index.md")
    if os.path.exists(top):
        with open(top, encoding="utf-8") as top_index_file:
            cur = top_index_file.read()
    else:
        cur = "# OKF reference 知识库\n\n"
    if "(%s/index.md)" % bundle not in cur:
        cur = cur.rstrip("\n") + "\n* [%s](%s/index.md) - %s\n" % (bundle, bundle, short)
        with open(top, "w", encoding="utf-8") as top_index_file:
            top_index_file.write(cur)


def _readme_category_desc():
    """Extract per concept-domain blurbs from docs/api/README.md intro bullets."""
    out = {}
    try:
        txt = read("docs/api/README.md")
    except Exception:
        return out
    for line in txt.splitlines():
        m = re.match(r"^-\s*\*\*(.+?)\*\*[：:]\s*(.+)$", line.strip())
        if not m:
            continue
        name, desc = m.group(1).strip(), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        for zh, sl in CATEGORY_SLUG.items():
            if zh in name or name in zh:
                out["api/" + sl] = desc[:160]
                break
    return out


def _dir_desc(full, curated):
    """Description OF the folder itself (curated / its own README), never borrowed
    from a descendant concept."""
    rel = os.path.relpath(full, BUNDLE_REF)
    return curated.get(rel) or os.path.basename(full)


def _gen_index_rec(d, curated):
    """Each index.md lists ONLY this level's immediate children — concept files in
    this dir (with their own description) and immediate subfolders (with the folder's
    description). It never reaches into a subfolder; that subfolder's own index.md
    describes its contents."""
    rel = os.path.relpath(d, BUNDLE_REF)
    entries, subdirs = [], []
    for name in sorted(os.listdir(d)):
        full = os.path.join(d, name)
        if name == "index.md":
            continue
        if os.path.isdir(full):
            entries.append(("[%s](%s/index.md)" % (name, md_rel(name)), _dir_desc(full, curated)))
            subdirs.append(full)
        elif name.endswith(".md"):
            fm = read_fm(full)
            entries.append(("[%s](%s)" % (fm.get("title", name[:-3]), md_rel(name)),
                            fm.get("description", "")))
    title = TITLE_MAP.get(rel, os.path.basename(d) if rel != "." else BUNDLE)
    desc = curated.get(rel) or ("导航索引：%s。" % title)
    idx_type = "bundle_index" if rel == "." else "section_index"
    fm = _index_frontmatter(title, desc, IndexOptions(idx_type=idx_type))
    lines = ["# %s" % title, ""]
    if rel == ".":
        lines += [BUNDLE_INTRO, ""]
    for link, desc in entries:
        lines.append("* %s - %s" % (link, desc or "（待补描述）"))
    with open(os.path.join(d, "index.md"), "w", encoding="utf-8") as f:
        f.write(fm + "\n".join(lines) + "\n")
    for full in subdirs:
        _gen_index_rec(full, curated)


def cmd_tags():
    """Merge deterministic hierarchy tags into each authored concept.

    This keeps the flattened upstream hierarchy queryable.
    """
    if not os.path.exists(WORKLIST):
        raise CliError("no worklist; run scaffold first")
    with open(WORKLIST, encoding="utf-8") as worklist_file:
        wl = json.load(worklist_file)["items"]
    n = 0
    for it in wl:
        path = os.path.join(BUNDLE_REF, it["concept"])
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as concept_file:
            lines = concept_file.read().split("\n")
        for i, ln in enumerate(lines[:40]):
            m = re.match(r"^tags:\s*\[(.*)\]\s*$", ln)
            if not m:
                continue
            cur = [t.strip() for t in m.group(1).split(",") if t.strip()]
            merged = _source_runtime.clean_tags(cur + it.get("path_tags", []), STRUCTURAL_TAGS)
            lines[i] = "tags: [%s]" % ", ".join(merged)
            with open(path, "w", encoding="utf-8") as concept_file:
                concept_file.write("\n".join(lines))
            n += 1
            break
    STDOUT_LOGGER.info("tags merged into %d concepts", n)


def cmd_source():
    """Synthesize the concept->source table ON DEMAND and print to stdout (never
    persisted). Delegates to the plugin-local source adapter in ops-knowledge-reference-ingest/scripts/okf_source.py
    (single authority). Aligns with OKF: provenance lives per-concept in
    frontmatter `resource:`; the aggregate is synthesized at consumption time,
    not固化 into a checked-in manifest. See SPEC-Source.md.
    """
    sys.path.insert(0, SCRIPT_DIR)
    from okf_source import get_source
    src = get_source(BUNDLE)
    refs = src.list_concepts()
    STDOUT_LOGGER.info("%s", src.render_table(refs))
    STDERR_LOGGER.info("source table synthesized: %d concepts (stdout, not persisted)", len(refs))


def _verify_concept_names(concepts, errs):
    for rel in concepts:
        if any(re.match(r"^\d{3}-", p) for p in rel.split("/")):
            errs.append("numeric prefix: " + rel)


def _verify_worklist_paths(errs):
    if os.path.exists(WORKLIST):
        with open(WORKLIST, encoding="utf-8") as worklist_file:
            wl = json.load(worklist_file)["items"]
        seen = {}
        for it in wl:
            seen.setdefault(it["concept"], []).append(it["source"])
        for c, srcs in seen.items():
            if len(srcs) > 1:
                errs.append("worklist path collision unresolved: " + c)


def _verify_resources(concepts, old_sha, new_sha, errs):
    migrate_mode = bool(old_sha and new_sha)
    if migrate_mode:
        _verify_migration_provenance(concepts, old_sha, new_sha, errs)
        return
    current_sha = sha()
    for rel in concepts:
        with open(os.path.join(BUNDLE_REF, rel), encoding="utf-8") as concept_file:
            body = concept_file.read()
        if re.search(r"!\[[^\]]*\]\(", body) or re.search(r"<img", body):
            errs.append("embedded image: " + rel)
        resource = read_fm(os.path.join(BUNDLE_REF, rel)).get("resource", "")
        pattern = r"https://gitcode\.com/.+/(blob|tree)/%s/" % re.escape(current_sha)
        if not re.match(pattern, resource):
            errs.append("bad resource url: %s -> %s" % (rel, resource))


def _bare_index_entries(index_path, relative_dir):
    errors = []
    with open(index_path, encoding="utf-8") as index_file:
        for line in index_file:
            if not line.startswith("* ["):
                continue
            description = line.split(") - ", 1)
            if len(description) < 2 or not description[1].strip():
                errors.append("bare index entry in %s: %s" % (relative_dir, line.strip()))
    return errors


def _verify_index_entries(errs):
    for root, _, filenames in os.walk(BUNDLE_REF):
        if "index.md" not in filenames:
            continue
        errs.extend(
            _bare_index_entries(
                os.path.join(root, "index.md"), os.path.relpath(root, BUNDLE_REF)
            )
        )


def _verify_manifest(concepts, mode, errs, warns):
    if not os.path.exists(MANIFEST_LIST):
        return
    with open(MANIFEST_LIST, encoding="utf-8") as manifest_file:
        planned = set(json.load(manifest_file))
    actual = set(concepts)
    if mode == "full":
        _coverage_full(errs)
    missing = planned - actual
    extra = actual - planned
    if missing:
        errs.append("planned-but-missing: %d e.g. %s" % (len(missing), sorted(missing)[:3]))
    if extra:
        warns.append("extra-not-in-plan: %d e.g. %s" % (len(extra), sorted(extra)[:3]))


def _verify_bundle_layout(concepts, mode, errs):
    root_fm = read_fm(os.path.join(BUNDLE_REF, "index.md"))
    for k in ("upstream_repo", "upstream_ref", "upstream_commit"):
        if not root_fm.get(k):
            errs.append("bundle-root index.md missing pin: " + k)
    for rel in concepts:
        if rel.count("/") > 2:
            errs.append("too deep (flat layout violated): " + rel)
    if mode != "full":
        return
    leaves = len(_leaf_example_readmes())
    examples = sum(1 for concept in concepts if concept.startswith("examples/"))
    if examples != leaves:
        errs.append("examples bijection: concepts %d != leaf READMEs %d" % (examples, leaves))
    context_dir = os.path.join(BUILD, "docs/api/context")
    top_level = sum(1 for filename in os.listdir(context_dir) if filename.endswith(".md"))
    uncategorized = sum(1 for concept in concepts if concept.startswith("api/uncategorized/"))
    if top_level and uncategorized / top_level > 0.20:
        errs.append("uncategorized %.1f%% > 20%%" % (100 * uncategorized / top_level))


def cmd_verify(mode, old_sha=None, new_sha=None):
    errs, warns = [], []
    concepts = concept_files()
    migrate_mode = bool(old_sha and new_sha)
    _verify_concept_names(concepts, errs)
    _verify_worklist_paths(errs)
    _verify_resources(concepts, old_sha, new_sha, errs)
    _verify_index_entries(errs)
    _verify_manifest(concepts, mode, errs, warns)
    _verify_bundle_layout(concepts, mode, errs)

    STDOUT_LOGGER.info("=== verify (%s%s) ===", mode, " · migration" if migrate_mode else "")
    STDOUT_LOGGER.info("concepts: %d", len(concepts))
    for w in warns:
        STDOUT_LOGGER.info("  WARN: %s", w)
    for e in errs:
        STDOUT_LOGGER.info("  FAIL: %s", e)
    if errs:
        return 1
    STDOUT_LOGGER.info("OK")
    return 0


# ============================ version-bump helpers ============================
RE_SHA40 = re.compile(r"/(blob|tree)/([0-9a-f]{40})/")
KNOWLEDGE_BUILD = os.path.join(ROOT, ".build", "knowledge", "asc-devkit")
GRAPH_DIR = os.path.join(ROOT, "graph")
ING_SCRIPTS = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "ops-knowledge-ingest", "scripts"))
LOG_DIR = os.path.join(ROOT, "log")


def _extract_sha(url):
    """Extract 40-hex sha from a GitCode blob/tree URL; None if not found."""
    m = RE_SHA40.search(url or "")
    return m.group(2) if m else None


def _url_repo_path(url):
    """Extract the <path> after /blob|tree/<sha>/ in a GitCode URL; None if not parseable."""
    m = RE_SHA40.search(url or "")
    if not m:
        return None
    return url[m.end():]


def _git_path_exists(repo, commit_sha, path):
    """Return whether a path exists in a local git object."""
    try:
        result = subprocess.run(
            [_git_executable(), "-C", repo, "cat-file", "-e", "%s:%s" % (commit_sha, path)],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        STDERR_LOGGER.debug("unable to inspect git path %s:%s", commit_sha, path, exc_info=True)
        return False


def _fetch_git_object(repo, commit_sha):
    """Best-effort fetch of one commit into the current upstream clone."""
    try:
        subprocess.run(
            [_git_executable(), "-C", repo, "fetch", "--depth=1", "origin", commit_sha],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        STDERR_LOGGER.debug("unable to fetch git object %s", commit_sha, exc_info=True)


def _git_object_exists(repositories, commit_sha):
    """Return whether a commit object exists in any available repository."""
    for repo in repositories:
        if not repo or not os.path.isdir(repo):
            continue
        try:
            result = subprocess.run(
                [_git_executable(), "-C", repo, "cat-file", "-t", commit_sha],
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            STDERR_LOGGER.debug("unable to inspect git object %s", commit_sha, exc_info=True)
            continue
        if result.returncode == 0:
            return True
    return False


def _git_cat_file_exists(commit_sha, path, old_sha=None):
    """Check one git path, fetching the object once before classifying failure."""

    new_build = BUILD                                    # .build/asc-devkit
    old_build = os.path.join(ROOT, ".build", "asc-devkit@%s" % old_sha) if old_sha else None
    # 1. new clone
    if os.path.isdir(new_build) and _git_path_exists(new_build, commit_sha, path):
        return True, "ok"
    # 2. old clone (if exists)
    if old_build and os.path.isdir(old_build) and _git_path_exists(old_build, commit_sha, path):
        return True, "ok"
    # 3. fetch --depth=1 origin <sha> (may be rejected by remote — that's 'missing_object', not 404)
    if os.path.isdir(new_build):
        _fetch_git_object(new_build, commit_sha)
        if _git_path_exists(new_build, commit_sha, path):
            return True, "ok"
    # 4. distinguish: does the object exist at all (in either repo)?
    obj_exists = _git_object_exists((new_build, old_build), commit_sha)
    return False, ("missing_path" if obj_exists else "missing_object")


def _parse_fm_sources(fm):
    """Parse the nested `sources:` block from frontmatter (best-effort, line-based).
    Returns list of (url, role). Empty if no sources block."""
    out = []
    src = fm.get("_raw_sources", "")
    if not src:
        return out
    for line in src.splitlines():
        m = re.match(r"\s*-\s*url:\s*(\S+)", line)
        if not m:
            continue
        url = m.group(1)
        rm = re.search(r"role:\s*(\w+)", line)
        role = rm.group(1) if rm else ""
        out.append((url, role))
    return out


def _read_fm_with_sources(path):
    """Like read_fm but also captures the raw `sources:` block for _parse_fm_sources."""
    fm = {}
    with open(path, encoding="utf-8") as frontmatter_file:
        lines = frontmatter_file.read().splitlines()
    if not lines or lines[0].strip() != "---":
        return fm
    in_sources = False
    src_block = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = re.match(r"^(\w+):\s*(.*)$", line)
        if match:
            in_sources = match.group(1) == "sources"
            fm[match.group(1)] = match.group(2).strip().strip("'\"")
            if in_sources and not match.group(2).strip():
                src_block = []
            continue
        if in_sources:
            src_block.append(line)
    fm["_raw_sources"] = "\n".join(src_block)
    return fm


def _migration_changed_cards():
    """Return cards whose migrated URLs need an existence check."""
    ledger_path = os.path.join(KNOWLEDGE_BUILD, "ledger.json")
    changed_cards = set()
    if not os.path.exists(ledger_path):
        return changed_cards
    try:
        with open(ledger_path, encoding="utf-8") as ledger_file:
            ledger = json.load(ledger_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        STDERR_LOGGER.debug("unable to read migration ledger %s", ledger_path, exc_info=True)
        return changed_cards
    for item in ledger:
        card_path = item.get("card_path") or item.get("card_id")
        if card_path:
            changed_cards.add(card_path)
    return changed_cards


def _migration_expected_sha(frontmatter, new_sha, rel, errs):
    """Return the required source SHA for one active or deprecated card."""
    is_deprecated = frontmatter.get("deprecated", "").lower() in ("true", '"true"')
    if not is_deprecated:
        return new_sha, False
    expected_sha = frontmatter.get("deprecated_commit", "").strip().strip("'\"")
    if not expected_sha:
        errs.append("deprecate card missing deprecated_commit: %s" % rel)
        return None, True
    return expected_sha, True


@dataclass(frozen=True)
class MigrationCheck:
    """Shared context for validating the URLs of one migrated concept."""

    rel: str
    expected_sha: str
    is_deprecated: bool
    old_sha: str | None
    check_url: bool


def _verify_migration_url(url, check, errs):
    """Validate the SHA and optional git existence of one provenance URL."""
    if not url:
        return
    source_sha = _extract_sha(url)
    if source_sha != check.expected_sha:
        status = "deprecate" if check.is_deprecated else "active"
        errs.append("sha mismatch (%s expected %s, got %s): %s -> %s"
                    % (status, check.expected_sha, source_sha, check.rel, url[:70]))
        return
    if not check.check_url:
        return
    repo_path = _url_repo_path(url)
    if repo_path is None:
        return
    exists, error_kind = _git_cat_file_exists(source_sha, repo_path, check.old_sha)
    if exists:
        return
    if error_kind == "missing_object":
        errs.append("object missing (need manual fetch): %s sha=%s" % (check.rel, source_sha))
    else:
        errs.append("404 path not in tree (legit sha but missing path): %s -> %s" % (check.rel, url[:70]))


def _verify_migration_provenance(concepts, old_sha, new_sha, errs):
    """Validate migration-mode provenance SHAs and changed-card URL existence."""
    changed_cards = _migration_changed_cards()

    for rel in concepts:
        path = os.path.join(BUNDLE_REF, rel)
        with open(path, encoding="utf-8") as concept_file:
            body = concept_file.read()
        if re.search(r"!\[[^\]]*\]\(", body) or re.search(r"<img", body):
            errs.append("embedded image: " + rel)
        fm = _read_fm_with_sources(path)
        expected_sha, is_deprecated = _migration_expected_sha(fm, new_sha, rel, errs)
        if expected_sha is None:
            continue
        urls = [fm.get("resource", "")]
        urls += [u for u, _ in _parse_fm_sources(fm)]
        check = MigrationCheck(
            rel, expected_sha, is_deprecated, old_sha, rel in changed_cards
        )
        for url in urls:
            _verify_migration_url(url, check, errs)


def cmd_advance_pin():
    """Advance bundle-root pin to current .build/asc-devkit HEAD sha.
    Standalone use (debug); normally called inside finalize-version-bump.
    """
    pin = _bundle_pin()    # reads .build HEAD
    _write_bundle_pin(pin)
    STDOUT_LOGGER.info("advance-pin: %s @ %s", pin["upstream_ref"], pin["upstream_commit"])


def _backup_file(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def _atomic_write(path, content):
    """Write to path via .tmp + os.replace (atomic)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def _gen_bump_log_entry(impact_inv, ledger):
    """Generate the sync log entry body (without the leading # header) from
    .build/knowledge/asc-devkit/impact_inventory.json + ledger.json."""
    from collections import Counter
    actions = Counter(it.get("action", "?") for it in (ledger or []))
    summary = "新增 %d、更新 %d、弃用 %d、重命名 %d、架构变更 %d" % (
        actions.get("new", 0), actions.get("update", 0), actions.get("deprecate", 0),
        actions.get("move_rename", 0), actions.get("structural", 0))
    by_action = {}
    for it in (ledger or []):
        by_action.setdefault(it.get("action", "?"), []).append(it.get("card_path", it.get("card_id", "?")))
    changes = []
    if by_action.get("new"):
        changes.append("- created: " + ", ".join(sorted(by_action.get("new", []))))
    if by_action.get("update"):
        changes.append("- updated: " + ", ".join(sorted(by_action.get("update", []))))
    if by_action.get("deprecate"):
        changes.append("- deprecated: " + ", ".join(sorted(by_action.get("deprecate", []))))
    if by_action.get("move_rename"):
        changes.append("- move_rename: " + ", ".join(sorted(by_action.get("move_rename", []))))
    changes.append("- created: runbooks/version-migration/%s_to_%s.md" % (impact_inv.get("from"), impact_inv.get("to")))
    return summary, changes


@dataclass(frozen=True)
class _FinalizeState:
    date: str
    hhmm: str
    rollback_dir: str
    index_path: str
    log_path: str
    log_existed: bool
    impact_inventory: dict
    ledger: list


def _load_optional_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as source_file:
        return json.load(source_file)


def _prepare_finalize_state():
    now = datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    date = now.split()[0]
    hhmm = now.split()[1]
    rollback_dir = os.path.join(KNOWLEDGE_BUILD, "rollback")
    if os.path.isdir(rollback_dir):
        shutil.rmtree(rollback_dir)
    index_path = os.path.join(BUNDLE_REF, "index.md")
    _backup_file(index_path, rollback_dir)
    for filename in os.listdir(GRAPH_DIR):
        if filename.startswith("viz") and filename.endswith(".html"):
            _backup_file(os.path.join(GRAPH_DIR, filename), rollback_dir)
    log_path = os.path.join(LOG_DIR, "%s.md" % date)
    _backup_file(log_path, rollback_dir)
    log_existed = os.path.exists(log_path)
    impact_inventory = _load_optional_json(
        os.path.join(KNOWLEDGE_BUILD, "impact_inventory.json"), {}
    )
    ledger = _load_optional_json(os.path.join(KNOWLEDGE_BUILD, "ledger.json"), [])
    return _FinalizeState(
        date, hhmm, rollback_dir, index_path, log_path, log_existed,
        impact_inventory, ledger,
    )


def _run_graph_step(command, timeout):
    result = subprocess.run(
        [sys.executable, os.path.join(ING_SCRIPTS, "okf_graph.py"), command],
        capture_output=True, text=True, cwd=ROOT, timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()[:200]
        raise RuntimeError("okf_graph %s failed: %s" % (command, detail))
    STDOUT_LOGGER.info("[finalize] %s OK", command)


def _bump_log_entry(state):
    inventory = state.impact_inventory
    summary, changes = _gen_bump_log_entry(inventory, state.ledger)
    entry_template = (
        "## [%s] sync | asc-devkit %s→%s 版本升级\n\n"
        "### Summary\n\n%s\n\n"
        "### Changes\n\n%s\n\n"
        "### References\n\n- source: %s\n- revision: %s..%s\n\n"
        "### Details\n\n%s\n\n---\n\n"
    )
    old_sha = inventory.get("old_sha")
    source = old_sha[:12] + "..." if old_sha else "?"
    old_revision = inventory.get("old_sha", "?")
    return entry_template % (
        state.hhmm, inventory.get("from", "?"), inventory.get("to", "?"),
        summary, "\n".join(changes), source, old_revision, inventory.get("new_sha", "?"),
        "按 SPEC-Version-update §4.2 版本升级流程处理。",
    )


def _updated_log_content(state, entry):
    if not state.log_existed:
        return "# %s\n\n> 当天维护日志。\n\n%s" % (state.date, entry)
    with open(state.log_path, encoding="utf-8") as log_file:
        current = log_file.read()
    lines = current.split("\n", 3)
    if len(lines) >= 3 and lines[0].startswith("# "):
        return "\n".join(lines[:3]) + "\n\n" + entry + "\n".join(lines[3:])
    return entry + current


def _rollback_finalize(state):
    index_backup = os.path.join(state.rollback_dir, "index.md")
    if os.path.exists(index_backup):
        shutil.copy2(index_backup, state.index_path)
    for filename in os.listdir(state.rollback_dir):
        if filename.startswith("viz") and filename.endswith(".html"):
            shutil.copy2(
                os.path.join(state.rollback_dir, filename), os.path.join(GRAPH_DIR, filename)
            )
    log_backup = os.path.join(state.rollback_dir, "%s.md" % state.date)
    if state.log_existed and os.path.exists(log_backup):
        shutil.copy2(log_backup, state.log_path)
    elif not state.log_existed and os.path.exists(state.log_path):
        os.remove(state.log_path)


def cmd_finalize_version_bump():
    """Atomically advance the pin, refresh the graph, verify, and update the log."""
    state = _prepare_finalize_state()

    try:
        pin = _bundle_pin()
        _write_bundle_pin(pin)
        STDOUT_LOGGER.info(
            "[finalize] advance-pin OK: %s @ %s",
            pin["upstream_ref"],
            pin["upstream_commit"],
        )

        _run_graph_step("viz", 300)
        _run_graph_step("verify", 120)
        _atomic_write(state.log_path, _updated_log_content(state, _bump_log_entry(state)))
        STDOUT_LOGGER.info("[finalize] log written to %s", state.log_path)
        STDOUT_LOGGER.info(
            "[finalize] version-bump LANDED. watermark → %s", pin["upstream_commit"]
        )
    except Exception as error:
        STDOUT_LOGGER.info("[finalize] FAILED: %s — rolling back", error)
        _rollback_finalize(state)
        return 1
    return 0


def _file_sha1(path):
    with open(path, "rb") as source_file:
        return hashlib.sha1(source_file.read()).hexdigest()


def _tree_file_hashes(tree_root, sub):
    """Walk <tree_root>/<sub>, return {relpath: sha1(content)}. relpath relative to tree_root."""
    out = {}
    base = os.path.join(tree_root, sub)
    if not os.path.isdir(base):
        return out
    for root, _, filenames in os.walk(base):
        for filename in filenames:
            is_doc = filename.endswith(".md") and filename.lower() != "readme.md"
            is_header = filename.endswith(".h") and sub == "include"
            if (is_doc and sub.startswith("docs")) or is_header:
                path = os.path.join(root, filename)
                out[os.path.relpath(path, tree_root)] = _file_sha1(path)
    return out


def _leaf_readmes(tree_root):
    """Set of LEAF example README relpaths (README dir with no child README)."""
    base = os.path.join(tree_root, "examples")
    out = set()
    if not os.path.isdir(base):
        return out
    for r, _, fs in os.walk(base):
        if "README.md" not in fs or r == base:
            continue
        if any("README.md" in cfs for cr, _, cfs in os.walk(r) if cr != r):
            continue
        out.add(os.path.relpath(os.path.join(r, "README.md"), tree_root))
    return out


@dataclass(frozen=True)
class _TreeDiff:
    old_sha: str
    new_sha: str
    old_hashes: dict
    new_hashes: dict
    new_paths: list
    updated_paths: list
    deprecated_paths: list


def _combined_tree_hashes(tree):
    hashes = _tree_file_hashes(tree, "docs")
    hashes.update(_tree_file_hashes(tree, "include"))
    return hashes


def _collect_tree_diff(old_tree, new_tree):
    old_sha = subprocess.check_output(
        [_git_executable(), "-C", old_tree, "rev-parse", "HEAD"], text=True
    ).strip()
    new_sha = subprocess.check_output(
        [_git_executable(), "-C", new_tree, "rev-parse", "HEAD"], text=True
    ).strip()
    old_all = _combined_tree_hashes(old_tree)
    new_all = _combined_tree_hashes(new_tree)
    old_set, new_set = set(old_all), set(new_all)
    old_ex = _leaf_readmes(old_tree)
    new_ex = _leaf_readmes(new_tree)
    return _TreeDiff(
        old_sha=old_sha,
        new_sha=new_sha,
        old_hashes=old_all,
        new_hashes=new_all,
        new_paths=sorted(new_set - old_set) + sorted(new_ex - old_ex),
        updated_paths=[path for path in sorted(old_set & new_set) if old_all[path] != new_all[path]],
        deprecated_paths=sorted(old_set - new_set) + sorted(old_ex - new_ex),
    )


def _describe_tree(tree, revision):
    if not revision:
        return revision
    return subprocess.check_output(
        [_git_executable(), "-C", tree, "describe", "--tags", "--always", "HEAD"],
        text=True,
    ).strip()


def _version_inventory(diff, old_tree, new_tree):
    return {
        "from": _describe_tree(old_tree, diff.old_sha),
        "to": _describe_tree(new_tree, diff.new_sha),
        "old_sha": diff.old_sha, "new_sha": diff.new_sha,
        "new": [{"doc": path,
                 "kind": "example" if path.startswith("examples/")
                 else ("header" if path.startswith("include/") else "doc"),
                 "target_card_seed": os.path.splitext(os.path.basename(path))[0]}
                for path in diff.new_paths],
        "update": [{"doc": path, "old_hash": diff.old_hashes[path],
                    "new_hash": diff.new_hashes[path], "changed_sections": []}
                   for path in diff.updated_paths],
        "deprecate": [{"doc": path,
                       "old_url": "%s/blob/%s/%s" % (UPSTREAM_REPO, diff.old_sha, path),
                       "reason": "not in new version", "replaced_by": None}
                      for path in diff.deprecated_paths],
        "move_rename": [],   # content-hash matching for rename detection (left to LLM/agent)
        "structural": [],    # filled by LLM agent (§4.2.1.B)
    }


def cmd_version_diff(old_tree, new_tree):
    """Write a deterministic structural diff between two asc-devkit checkouts."""
    diff = _collect_tree_diff(old_tree, new_tree)
    inventory = _version_inventory(diff, old_tree, new_tree)
    os.makedirs(KNOWLEDGE_BUILD, exist_ok=True)
    out = os.path.join(KNOWLEDGE_BUILD, "impact_inventory.json")
    with open(out, "w", encoding="utf-8") as inventory_file:
        json.dump(inventory, inventory_file, ensure_ascii=False, indent=2)
    STDOUT_LOGGER.info("version-diff: %s..%s", diff.old_sha[:12], diff.new_sha[:12])
    STDOUT_LOGGER.info(
        "  new: %d  update: %d  deprecate: %d  (docs+include+examples)",
        len(inventory["new"]),
        len(inventory["update"]),
        len(inventory["deprecate"]),
    )
    STDOUT_LOGGER.info("  -> %s", out)


def _leaf_example_readmes():
    """Return the source set of leaf example README files."""
    return _leaf_readmes(BUILD)


def _coverage_full(errs):
    src = set()
    for f in os.listdir(os.path.join(BUILD, "docs")):
        if f.endswith(".md") and f.lower() != "readme.md":
            src.add("docs/" + f)
    for r, _, fs in os.walk(os.path.join(BUILD, "docs/api/context")):
        for f in fs:
            if f.endswith(".md") and f.lower() != "readme.md":
                src.add(os.path.relpath(os.path.join(r, f), BUILD))
    src |= _leaf_example_readmes()                 # only LEAF example READMEs are concepts
    for r, _, fs in os.walk(os.path.join(BUILD, "include")):
        for f in fs:
            if f.endswith(".h"):
                src.add(os.path.relpath(os.path.join(r, f), BUILD))
    with open(WORKLIST, encoding="utf-8") as worklist_file:
        wl = json.load(worklist_file)["items"]
    wsrc = set(it["source"] for it in wl)
    miss = src - wsrc
    extra = wsrc - src
    if miss:
        errs.append("source-not-in-worklist: %d e.g. %s" % (len(miss), sorted(miss)[:3]))
    if extra:
        errs.append("worklist-not-in-source: %d e.g. %s" % (len(extra), sorted(extra)[:3]))


def _build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-root", help="OKF knowledge-base root")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("pin")
    scaffold = subparsers.add_parser("scaffold")
    scaffold_mode = scaffold.add_mutually_exclusive_group()
    scaffold_mode.add_argument("--pilot", action="store_true")
    scaffold_mode.add_argument("--sample", type=int)
    scaffold_mode.add_argument("--full", action="store_true")
    subparsers.add_parser("tags")
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument(
        "--no-pin", action="store_true",
        help="rebuild indexes but preserve existing bundle pin (version-bump mode)",
    )
    subparsers.add_parser("source")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--mode", default=None)
    verify.add_argument(
        "--old-sha", default=None,
        help="migration-mode: old watermark sha (enables deprecated-split SHA check)",
    )
    verify.add_argument("--new-sha", default=None, help="migration-mode: new target sha")
    subparsers.add_parser(
        "advance-pin", help="advance bundle pin to .build HEAD (debug; normally inside finalize-version-bump)"
    )
    subparsers.add_parser(
        "finalize-version-bump", help="atomic advance-pin + viz + verify + log (version-bump landing)"
    )
    version_diff = subparsers.add_parser(
        "version-diff", help="structural diff of two checkouts → impact_inventory.json"
    )
    version_diff.add_argument("--old-tree", required=True)
    version_diff.add_argument("--new-tree", required=True)
    return parser


def _verify_mode(args):
    mode = args.mode
    if not mode and os.path.exists(WORKLIST):
        with open(WORKLIST, encoding="utf-8") as worklist_file:
            mode = json.load(worklist_file).get("mode", "pilot")
    return mode or "pilot"


def _dispatch_command(args):
    if args.cmd == "pin":
        cmd_pin()
    elif args.cmd == "scaffold":
        mode = "pilot" if args.pilot else "sample" if args.sample else "full"
        cmd_scaffold(mode, args.sample or 8)
    elif args.cmd == "tags":
        cmd_tags()
    elif args.cmd == "index":
        cmd_index(no_pin=args.no_pin)
    elif args.cmd == "source":
        cmd_source()
    elif args.cmd == "verify":
        return cmd_verify(_verify_mode(args), args.old_sha, args.new_sha)
    elif args.cmd == "advance-pin":
        cmd_advance_pin()
    elif args.cmd == "finalize-version-bump":
        return cmd_finalize_version_bump()
    elif args.cmd == "version-diff":
        cmd_version_diff(args.old_tree, args.new_tree)
    return 0


def main():
    _source_runtime.validate_root_argument()
    args = _build_cli_parser().parse_args()
    build_commands = (
        "pin", "scaffold", "verify", "advance-pin", "finalize-version-bump", "version-diff"
    )
    if not os.path.isdir(BUILD) and args.cmd in build_commands:
        raise CliError("missing %s — clone asc-devkit there first" % BUILD)
    return _dispatch_command(args)


if __name__ == "__main__":
    try:
        _EXIT_CODE = main()
    except CliError as cli_error:
        raise SystemExit(str(cli_error)) from None
    raise SystemExit(_EXIT_CODE)
