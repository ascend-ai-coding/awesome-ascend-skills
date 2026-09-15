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
"""cann_doc_extract.py — extract a CANN online doc (a single converted markdown with flat
NUMBERED headings `# 1.3.2.1 Title`) into an OKF bundle under reference/<bundle>/, following
SPEC-Reference.md: flat ≤2 levels (<chapter>/<leaf>.md), deeper section hierarchy via tags,
per-level index.md. One concept per LEAF section (no numbered children).

concept BODIES are LLM-distilled by the agent layer (not this script). This script does the
deterministic parts: scaffold / tags / index / resource / verify. resource = the hiascend doc
URL (canonical upstream; per-section page URLs unavailable, so the section number is recorded
in the manifest for traceability).

Usage: cann_doc_extract.py <cmd> <bundle> [opts]   where bundle ∈ DOCS keys.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.parse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import okf_source as _source_runtime

CliError = _source_runtime.CliError
STDOUT_LOGGER = _source_runtime.make_cli_logger(__name__ + ".stdout", sys.stdout)
STDERR_LOGGER = _source_runtime.make_cli_logger(__name__ + ".stderr", sys.stderr)

ROOT = _source_runtime.ROOT
REFERENCE = os.path.join(ROOT, "reference")
TIMESTAMP = "2026-06-24T00:00:00Z"
UPSTREAM_REF = "9.0.0"   # CANN 商用版 9.0.0 (all DOCS bundles); hiascend docs have no commit pin
DETAIL_BASE = "https://www.hiascend.com/document/detail/"   # + nodeUrl -> real per-section page

# ---- per-document configuration ------------------------------------------
DOCS = {
    "ascend-c-op-dev-guide": {
        "doc": "raw/Ascend C算子开发指南/Ascend C算子开发指南.md",
        "url": "https://www.hiascend.com/document/detail/zh/canncommercial/900/"
               "programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html",
        "title": "Ascend C 算子开发指南（CANN 商用版 9.0.0）",
        "short": "CANN《Ascend C 算子开发指南》(9.0.0) 的结构化知识库：入门、编程指南、算子实践、兼容迁移与可视化。",
        "chapters": {  # ch6 (API 参考) excluded — overlaps the asc-devkit api/ bundle
            "1": ("getting_started", "入门教程：Ascend C 是什么、环境准备，以及基于 SIMD/SIMT 的快速上手样例。"),
            "2": ("programming_guide", "编程指南：Kernel 直调、算子工程、Tiling、内存与流水、调试调优等编程方法。"),
            "3": ("operator_practice", "算子实践参考：典型算子的实现思路与端到端开发样例。"),
            "4": ("compatibility_migration", "兼容性与迁移指南：跨硬件版本/编程范式的兼容性说明与迁移方法。"),
            "5": ("visualization", "可视化专区：算子开发相关的可视化工具与用法。"),
        },
        "intro": (
            "本知识库由 CANN 商用版 9.0.0 的 **《Ascend C 算子开发指南》**（hiascend 在线文档）结构化抽取而成。"
            "Ascend C 是 CANN 面向算子开发场景的编程语言，原生支持 C/C++ 标准、兼顾开发效率与运行性能。\n\n"
            "本库把指南正文（第 1–5 章，不含与 asc-devkit 重叠的第 6 章 API 参考）按章节**扁平**组织为简洁的 OKF "
            "concept（蒸馏而非照搬），上游更深的小节层级用 `tags` 表达，并由逐层 `index.md` 导航；每个 concept 的 "
            "`resource` 指向该在线文档。"),
    },
    "ascend-c-profiling": {
        "doc": "raw/Ascend C 性能调优工具用户指南/Ascend C 性能调优工具用户指南.md",
        "url": "https://www.hiascend.com/document/detail/zh/canncommercial/900/"
               "devaids/Profiling/atlasprofiling_16_0001.html",
        "title": "Ascend C 性能调优工具用户指南（CANN 商用版 9.0.0）",
        "short": (
            "CANN《Ascend C 性能调优工具用户指南》(9.0.0) 的结构化知识库："
            "msprof 采集/解析、数据采集方式与文件参考、FAQ 等。"
        ),
        "chapters": {
            "1": ("introduction", "简介：性能调优工具（msprof 等）与整体能力概述。"),
            "2": ("prerequisites", "使用前准备：环境与依赖准备。"),
            "3": ("quickstart_offline_inference", "离线推理场景性能分析快速入门。"),
            "4": ("quickstart_pytorch_training", "PyTorch 训练场景性能分析快速入门。"),
            "5": ("msprof_collect", "msprof 采集命令：性能数据采集的命令行用法。"),
            "6": ("msprof_parse_export", "使用 msprof 命令解析、查询与导出性能数据。"),
            "7": ("other_collection", "性能数据其他采集方式（API/环境变量等）。"),
            "8": ("profiling_data_files", "性能数据文件参考：目录结构与各文件说明。"),
            "9": ("framework_data_files", "MindSpore & PyTorch 框架性能数据文件参考。"),
            "10": ("appendix", "附录。"),
            "11": ("faq", "常见问题（FAQ）。"),
        },
        "intro": (
            "本知识库由 CANN 商用版 9.0.0 的 **《Ascend C 性能调优工具用户指南》**（hiascend 在线文档）结构化抽取而成，"
            "覆盖 msprof 采集/解析导出、各类性能数据采集方式、性能数据文件结构与 FAQ。\n\n"
            "本库把文档正文按章节**扁平**组织为简洁的 OKF concept（蒸馏而非照搬），上游更深的小节层级用 `tags` 表达，"
            "并由逐层 `index.md` 导航；每个 concept 的 `resource` 指向该在线文档。"),
    },
}

RE_NUM = re.compile(r"^#\s+(\d+(?:\.\d+)*)\s+(.*\S)\s*$")
RE_TOC = re.compile(r"\.{3,}\s*\d+\s*$")
RE_FENCE = re.compile(r"^\s*(```|~~~)")


class RuntimeConfig:
    """Mutable process-wide configuration for the selected document bundle."""

    bundle = doc = url = title = short = intro = None
    chapters = {}
    ref = worklist = manifest_list = tree = resolved = docdir = None


def load(bundle):
    if bundle not in DOCS:
        raise CliError("unknown bundle %r (have: %s)" % (bundle, ", ".join(DOCS)))
    cfg = DOCS[bundle]
    RuntimeConfig.bundle = bundle
    RuntimeConfig.doc = os.path.join(ROOT, cfg["doc"])
    RuntimeConfig.url = cfg["url"]
    RuntimeConfig.title = cfg["title"]
    RuntimeConfig.short = cfg["short"]
    RuntimeConfig.intro = cfg["intro"]
    RuntimeConfig.chapters = cfg["chapters"]
    RuntimeConfig.ref = os.path.join(REFERENCE, bundle)
    RuntimeConfig.worklist = os.path.join(ROOT, ".build", "%s_worklist.json" % bundle)
    RuntimeConfig.manifest_list = os.path.join(ROOT, ".build", "%s_manifest_list.json" % bundle)
    RuntimeConfig.tree = os.path.join(ROOT, ".build", "%s_tree.json" % bundle)
    RuntimeConfig.resolved = os.path.join(ROOT, ".build", "%s_resolved.json" % bundle)
    # doc path-prefix up to the docid (for verify: real URLs live under this)
    RuntimeConfig.docdir = (
        RuntimeConfig.url[len(DETAIL_BASE):].rsplit("/", 1)[0]
        if RuntimeConfig.url.startswith(DETAIL_BASE) else ""
    )


def slug(name):
    name = re.sub(r"^\d+(?:\.\d+)*\s*", "", name)
    name = re.sub(r"^\d+[_\-]+", "", name)
    out = re.sub(r"[^0-9A-Za-z一-鿿]+", "_", name).strip("_").lower()
    return out or "x"


def hash6(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:6]


def md_rel(p):
    return "/".join(urllib.parse.quote(s) for s in p.split(os.sep))


def _dedup(seq):
    out = []
    for x in seq:
        if x and x not in out:
            out.append(x)
    return out


STRUCTURAL_TAGS = {"cann", "ascend-c", "profiling", "guide", "index", "api", "reference"}


def _strip_frontmatter(text):
    if not text.startswith("---\n"):
        return text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1:]).lstrip("\n")
    return text


def _index_frontmatter(title, desc, idx_type="section_index", pins=None):
    pins = pins or {}
    lines = ["---",
             "schema_version: okf.v1",
             "kind: index",
             "type: %s" % idx_type,
             "source_family: %s" % (
                 "ascend_c_profiling" if RuntimeConfig.bundle == "ascend-c-profiling" else "ascend_c_op_dev_guide"
             ),
             "title: %s" % json.dumps(title, ensure_ascii=False),
             "description: %s" % json.dumps(desc or ("导航索引：%s。" % title), ensure_ascii=False),
             "tags: []"]
    for k in ("upstream_doc", "upstream_ref"):
        if pins.get(k):
            val = pins[k]
            lines.append("%s: %s" % (k, json.dumps(val, ensure_ascii=False) if k == "upstream_ref" else val))
    lines += ["created_at: '%s'" % TIMESTAMP,
              "updated_at: '%s'" % TIMESTAMP,
              "---", ""]
    return "\n".join(lines)


def parse_sections():
    with open(RuntimeConfig.doc, encoding="utf-8", errors="replace") as source_file:
        lines = source_file.read().split("\n")
    heads, incode = [], False
    for i, ln in enumerate(lines):
        if RE_FENCE.match(ln):
            incode = not incode
            continue
        if incode:
            continue
        m = RE_NUM.match(ln)
        if m and not RE_TOC.search(ln):
            heads.append((m.group(1), m.group(2).strip(), i + 1))
    out = []
    for k, (num, title, ln) in enumerate(heads):
        end = heads[k + 1][2] if k + 1 < len(heads) else len(lines) + 1
        out.append((num, title, ln, end))
    return out


def leaves():
    sections = parse_sections()
    nums = set(s[0] for s in sections)
    title = {s[0]: s[1] for s in sections}
    items = []
    for num, ttl, ln, end in sections:
        if any(o != num and o.startswith(num + ".") for o in nums):
            continue
        ch = num.split(".")[0]
        if ch not in RuntimeConfig.chapters:
            continue
        chslug = RuntimeConfig.chapters[ch][0]
        parts = num.split(".")
        anc = [".".join(parts[:i]) for i in range(2, len(parts))]
        path_tags = [chslug] + [slug(title[a]) for a in anc if a in title]
        items.append({"concept": "%s/%s.md" % (chslug, slug(ttl)), "section": num,
                      "title": ttl, "start": ln, "end": end,
                      "path_tags": _dedup(path_tags), "chapter": chslug,
                      "resource": RuntimeConfig.url})
    used = {}
    for it in items:
        c = it["concept"]
        if c in used and used[c] != it["section"]:
            stem, ext = os.path.splitext(c)
            c = "%s__%s%s" % (stem, hash6(it["section"]), ext)
        used[c] = it["section"]
        it["concept"] = c
    items.sort(key=lambda x: [int(p) for p in x["section"].split(".")])
    return items


def read_fm(path):
    fm = {}
    with open(path, encoding="utf-8") as frontmatter_file:
        lines = frontmatter_file.read().splitlines()
    if not lines or lines[0].strip() != "---":
        return fm
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = re.match(r"^(\w+):\s*(.*)$", line)
        if match:
            fm[match.group(1)] = match.group(2).strip().strip("'\"")
    return fm


def concept_files():
    out = []
    for r, _, fs in os.walk(RuntimeConfig.ref):
        for f in fs:
            if f.endswith(".md") and f != "index.md":
                out.append(os.path.relpath(os.path.join(r, f), RuntimeConfig.ref))
    return sorted(out)


def cmd_scaffold(mode, n):
    items = leaves()
    if mode == "sample":
        g = {}
        for it in items:
            g.setdefault(it["chapter"], [])
            if len(g[it["chapter"]]) < n:
                g[it["chapter"]].append(it)
        items = [it for ch in RuntimeConfig.chapters.values() for it in g.get(ch[0], [])]
    os.makedirs(os.path.dirname(RuntimeConfig.worklist), exist_ok=True)
    with open(RuntimeConfig.worklist, "w", encoding="utf-8") as worklist_file:
        json.dump(
            {"mode": mode, "bundle": RuntimeConfig.bundle, "doc": RuntimeConfig.doc, "items": items},
            worklist_file,
            ensure_ascii=False,
            indent=1,
        )
    with open(RuntimeConfig.manifest_list, "w", encoding="utf-8") as manifest_file:
        json.dump(
            sorted(it["concept"] for it in items),
            manifest_file,
            ensure_ascii=False,
            indent=1,
        )
    import collections
    STDOUT_LOGGER.info("scaffold[%s] mode=%s items=%d", RuntimeConfig.bundle, mode, len(items))
    STDOUT_LOGGER.info(
        "  by chapter: %s", dict(collections.Counter(it["chapter"] for it in items))
    )


def _merge_worklist_tags(item):
    path = os.path.join(RuntimeConfig.ref, item["concept"])
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as concept_file:
        lines = concept_file.read().split("\n")
    for index, line in enumerate(lines[:40]):
        match = re.match(r"^tags:\s*\[(.*)\]\s*$", line)
        if not match:
            continue
        current = [tag.strip() for tag in match.group(1).split(",") if tag.strip()]
        normalized = _source_runtime.clean_tags(current + item["path_tags"], STRUCTURAL_TAGS)
        lines[index] = "tags: [%s]" % ", ".join(normalized)
        with open(path, "w", encoding="utf-8") as concept_file:
            concept_file.write("\n".join(lines))
        return True
    return False


def cmd_tags():
    with open(RuntimeConfig.worklist, encoding="utf-8") as worklist_file:
        wl = json.load(worklist_file)["items"]
    n = 0
    for item in wl:
        if _merge_worklist_tags(item):
            n += 1
    STDOUT_LOGGER.info("tags merged into %d concepts", n)


def _write_bundle_pin():
    """Prepend the bundle-level pin as frontmatter on the bundle-root index.md
    (OKF-reserved, consumer-visible). hiascend docs have no commit; the pin is
    the doc-root URL + version. See SPEC-Source.md §2."""
    p = os.path.join(RuntimeConfig.ref, "index.md")
    with open(p, encoding="utf-8") as index_file:
        body = _strip_frontmatter(index_file.read())
    fm = _index_frontmatter(RuntimeConfig.title, RuntimeConfig.short, idx_type="bundle_index",
                            pins={"upstream_doc": RuntimeConfig.url, "upstream_ref": UPSTREAM_REF})
    with open(p, "w", encoding="utf-8") as index_file:
        index_file.write(fm + body)


def cmd_index():
    for r, _, fs in os.walk(RuntimeConfig.ref):
        if "index.md" in fs:
            os.remove(os.path.join(r, "index.md"))
    _gen(RuntimeConfig.ref)
    _write_bundle_pin()                          # bundle-level pin -> bundle-root index.md frontmatter
    top = os.path.join(REFERENCE, "index.md")
    if os.path.exists(top):
        with open(top, encoding="utf-8") as top_index_file:
            cur = top_index_file.read()
    else:
        cur = "# OKF reference 知识库\n\n"
    if "(%s/index.md)" % RuntimeConfig.bundle not in cur:
        cur = cur.rstrip("\n") + "\n* [%s](%s/index.md) - %s\n" % (
            RuntimeConfig.bundle, RuntimeConfig.bundle, RuntimeConfig.short
        )
        with open(top, "w", encoding="utf-8") as top_index_file:
            top_index_file.write(cur)
    STDOUT_LOGGER.info("indexes rebuilt")


def _chap_desc(full):
    base = os.path.basename(full)
    for _, (sl, desc) in RuntimeConfig.chapters.items():
        if sl == base:
            return desc
    return os.path.relpath(full, RuntimeConfig.ref)


def _gen(d):
    rel = os.path.relpath(d, RuntimeConfig.ref)
    entries, subs = [], []
    for name in sorted(os.listdir(d)):
        full = os.path.join(d, name)
        if name == "index.md":
            continue
        if os.path.isdir(full):
            entries.append(("[%s](%s/index.md)" % (name, md_rel(name)), _chap_desc(full)))
            subs.append(full)
        elif name.endswith(".md"):
            fm = read_fm(full)
            entries.append(("[%s](%s)" % (fm.get("title", name[:-3]), md_rel(name)),
                            fm.get("description", "")))
    title = RuntimeConfig.title if rel == "." else os.path.basename(d)
    desc = RuntimeConfig.short if rel == "." else _chap_desc(d)
    idx_type = "bundle_index" if rel == "." else "section_index"
    fm = _index_frontmatter(title, desc, idx_type=idx_type)
    lines = ["# %s" % title, ""]
    if rel == ".":
        lines += [RuntimeConfig.intro + " 包含以下部分：", ""]
    for link, desc in entries:
        lines.append("* %s - %s" % (link, desc or "（待补描述）"))
    index_path = os.path.join(d, "index.md")
    with open(index_path, "w", encoding="utf-8") as index_file:
        index_file.write(fm + "\n".join(lines) + "\n")
    for s in subs:
        _gen(s)


def section_urls():
    """Map our positional section number -> real per-section page URL, from the cached
    /doc/node/tree json (node order mirrors the doc's heading numbering).
    """
    with open(RuntimeConfig.tree, encoding="utf-8") as tree_file:
        d = json.load(tree_file)["data"]["directory"]
    out = {}
    family = re.match(r"[a-zA-Z]+", RuntimeConfig.url.rstrip("/").split("/")[-1]).group(0)  # e.g. atlasprofiling

    def walk(nodes, prefix):
        for i, n in enumerate(nodes, 1):
            sec = "%s.%d" % (prefix, i) if prefix else str(i)
            node_url = (n.get("nodeUrl") or n.get("nodeHtml") or "").lstrip("/")
            if node_url:
                if "/" in node_url:                          # full path -> use as-is
                    out[sec] = DETAIL_BASE + node_url
                elif node_url.startswith(family):            # bare same-doc page -> add doc dir
                    out[sec] = DETAIL_BASE + RuntimeConfig.docdir + "/" + node_url
                # else: bare cross-doc reference (another product doc) -> leave unmapped
            walk(n.get("children") or [], sec)
    walk(d, "")
    return out


def cmd_urls():
    """Rewrite each concept's resource (and any body citation) from the doc-root URL to
    its real per-section page URL resolved from the TOC tree.
    """
    if not os.path.exists(RuntimeConfig.tree):
        raise CliError("missing tree json %s — fetch /doc/node/tree first" % RuntimeConfig.tree)
    urls = section_urls()
    with open(RuntimeConfig.worklist, encoding="utf-8") as worklist_file:
        wl = json.load(worklist_file)["items"]
    resolved, miss = {}, []
    for it in wl:
        p = os.path.join(RuntimeConfig.ref, it["concept"])
        target = urls.get(it["section"]) or RuntimeConfig.url     # cross-doc/unmapped -> doc-root fallback
        if it["section"] not in urls:
            miss.append(it["section"])
        resolved[it["concept"]] = target
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as concept_file:
            txt = concept_file.read()
        cur = read_fm(p).get("resource", "")           # idempotent: replace whatever is there now
        if cur and cur != target:
            with open(p, "w", encoding="utf-8") as concept_file:
                concept_file.write(txt.replace(cur, target))
    with open(RuntimeConfig.resolved, "w", encoding="utf-8") as resolved_file:
        json.dump(resolved, resolved_file, ensure_ascii=False, indent=1)
    mapped = sum(1 for v in resolved.values() if v != RuntimeConfig.url)
    STDOUT_LOGGER.info(
        "resolved %d/%d concepts to real per-section URLs (%d cross-doc -> doc-root fallback)",
        mapped,
        len(wl),
        len(resolved) - mapped,
    )
    if miss:
        STDOUT_LOGGER.info("  cross-doc/unmapped sections: %s", miss[:15])


def cmd_source():
    """Synthesize the concept->source table ON DEMAND and print to stdout (never
    persisted). Delegates to the plugin-local source adapter in ops-knowledge-reference-ingest/scripts/okf_source.py
    (single authority): `section` from the .build worklist, doc url from the
    resolved per-section URL (or frontmatter `resource:`). Aligns with OKF: the
    aggregate is synthesized at consumption time, not固化 into a checked-in
    manifest. See SPEC-Source.md.
    """
    sys.path.insert(0, SCRIPT_DIR)
    from okf_source import get_source
    src = get_source(RuntimeConfig.bundle)
    refs = src.list_concepts()
    STDOUT_LOGGER.info("%s", src.render_table(refs))
    STDERR_LOGGER.info("source table synthesized: %d concepts (stdout, not persisted)", len(refs))


def _verify_concept(rel, errs):
    """Validate one generated concept without changing error ordering."""
    if rel.count("/") > 1:
        errs.append("too deep: " + rel)
    if any(re.match(r"^\d{3}-", part) for part in rel.split("/")):
        errs.append("numeric prefix: " + rel)
    concept_path = os.path.join(RuntimeConfig.ref, rel)
    with open(concept_path, encoding="utf-8") as concept_file:
        body = concept_file.read()
    if re.search(r"!\[[^\]]*\]\(", body) or re.search(r"<img", body):
        errs.append("embedded image: " + rel)
    resource = read_fm(concept_path).get("resource", "")
    valid_resource = resource == RuntimeConfig.url or (
        resource.startswith(DETAIL_BASE) and RuntimeConfig.docdir in resource
    )
    if not valid_resource:
        errs.append("bad resource: %s -> %s" % (rel, resource))


def _bare_index_lines(index_path):
    errors = []
    with open(index_path, encoding="utf-8") as index_file:
        for line in index_file:
            missing_description = ") - " not in line or not line.split(") - ", 1)[1].strip()
            if line.startswith("* [") and missing_description:
                errors.append("bare index entry: " + line.strip())
    return errors


def _verify_index_entries(errs):
    """Validate that every generated index entry carries a description."""
    for r, _, fs in os.walk(RuntimeConfig.ref):
        if "index.md" not in fs:
            continue
        errs.extend(_bare_index_lines(os.path.join(r, "index.md")))


def _verify_pin_and_manifest(concepts, errs):
    """Validate the bundle pin and optional scaffold manifest."""
    root_fm = read_fm(os.path.join(RuntimeConfig.ref, "index.md"))   # bundle-level pin (SPEC-Source.md §2)
    for k in ("upstream_doc", "upstream_ref"):
        if not root_fm.get(k):
            errs.append("bundle-root index.md missing pin: " + k)
    if not os.path.exists(RuntimeConfig.manifest_list):
        return
    with open(RuntimeConfig.manifest_list, encoding="utf-8") as manifest_file:
        planned = set(json.load(manifest_file))
    actual = set(concepts)
    if planned - actual:
        errs.append("planned-but-missing: %d e.g. %s" % (len(planned - actual), sorted(planned - actual)[:3]))
    if actual - planned:
        errs.append("extra-not-in-plan: %d e.g. %s" % (len(actual - planned), sorted(actual - planned)[:3]))


def cmd_verify():
    errs, concepts = [], concept_files()
    for rel in concepts:
        _verify_concept(rel, errs)
    _verify_index_entries(errs)
    _verify_pin_and_manifest(concepts, errs)
    STDOUT_LOGGER.info("=== verify[%s] ===\nconcepts: %d", RuntimeConfig.bundle, len(concepts))
    for e in errs:
        STDOUT_LOGGER.info("  FAIL: %s", e)
    if errs:
        return 1
    STDOUT_LOGGER.info("OK")
    return 0


def main():
    _source_runtime.validate_root_argument()
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge-root", help="OKF knowledge-base root")
    ap.add_argument("cmd", choices=["scaffold", "tags", "urls", "index", "source", "verify"])
    ap.add_argument("bundle")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    load(a.bundle)
    exit_code = 0
    if a.cmd == "scaffold":
        if not os.path.exists(RuntimeConfig.doc):
            raise CliError("missing source doc: %s" % RuntimeConfig.doc)
        cmd_scaffold("sample" if a.sample else "full", a.sample or 0)
    elif a.cmd == "tags":
        cmd_tags()
    elif a.cmd == "urls":
        cmd_urls()
    elif a.cmd == "index":
        cmd_index()
    elif a.cmd == "source":
        cmd_source()
    elif a.cmd == "verify":
        exit_code = cmd_verify()
    return exit_code


if __name__ == "__main__":
    try:
        _EXIT_CODE = main()
    except CliError as cli_error:
        raise SystemExit(str(cli_error)) from None
    raise SystemExit(_EXIT_CODE)
