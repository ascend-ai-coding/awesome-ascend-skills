#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""okf_graph.py — LLM-judged knowledge-graph layer over the OKF reference bundles.

See SPEC-Graph.md (v3). Recall is pure stdlib; edges are judged by an LLM and cached.

  RECALL   `candidates` — deterministic recall (term overlap + tag-IDF + #include, NO mentions)
           emits per-card candidate sets (.build/candidates.json) for the judge.
  JUDGE    an LLM reads each focal card + its candidates and decides which are genuinely
           related, the type, and a one-line reason. Verdicts are written to the committed
           graph/edge_judgments.json, keyed for fingerprint-based incremental reuse.
  RECORD   the canonical edge record is the card body `# 相关` block (LLM type + reason).
           The graph = kept judgments + deterministic part_of tree, synthesized on read.
  RETRIEVE in-card `# 相关`; or `related/explain` (from the cache); or self-contained viz.html.

Subcommands: candidates / inject / viz / verify / related / explain  (build = legacy v2 report).
Node ids are reference-relative (root id = "index.md"). All content reads strip the managed
`<!-- okf:related:start … end -->` block first (so injection never feeds back into recall).
"""
import argparse
import collections
import hashlib
import itertools
import json
import logging
import math
import os
import re
import sys
import urllib.parse


def _log(message, *args, stream):
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord(__name__, logging.INFO, "", 0, message, args, None)
    handler.handle(record)
    handler.close()


def _log_output(message, *args):
    """Write one unprefixed log record to stdout."""
    _log(message, *args, stream=sys.stdout)


def _log_error(message, *args):
    """Write one unprefixed log record to stderr."""
    _log(message, *args, stream=sys.stderr)


def _knowledge_root_from_argv(argv):
    root = None
    error = None
    rest = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--knowledge-root":
            if index + 1 >= len(argv):
                error = "--knowledge-root requires a path"
                break
            root = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--knowledge-root="):
            root = arg.split("=", 1)[1]
            if not root:
                error = "--knowledge-root requires a path"
                break
            index += 1
            continue
        rest.append(arg)
        index += 1
    if error:
        return None, error
    argv[:] = rest
    if not root:
        return None, None
    root = os.path.abspath(os.path.expanduser(root))
    os.environ["CANNBOT_KNOWLEDGE_ROOT"] = root
    os.environ["CANNBOT_KNOWLEDGE_ROOTS"] = root
    os.environ["OKF_KNOWLEDGE_ROOT"] = root
    os.environ["OKF_KNOWLEDGE_ROOTS"] = root
    os.environ["KNOWLEDGE_ROOT"] = root
    os.environ["KNOWLEDGE_ROOTS"] = root
    return root, None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARGV_ROOT, ROOT_ARGUMENT_ERROR = _knowledge_root_from_argv(sys.argv)
ROOT = (
    ARGV_ROOT
    or os.environ.get("CANNBOT_KNOWLEDGE_ROOT")
    or os.environ.get("OKF_KNOWLEDGE_ROOT")
    or os.environ.get("KNOWLEDGE_ROOT")
    or os.getcwd()
)
ROOT = os.path.abspath(ROOT)
REFERENCE = os.path.join(ROOT, "reference")
OPS = os.path.join(ROOT, "ops")
RUNBOOKS = os.path.join(ROOT, "runbooks")
GRAPH_DIR = os.path.join(ROOT, "graph")
BUILD_DIR = os.path.join(ROOT, ".build")          # gitignored: graph cache + report live here

# Extra content roots beyond reference/ — each maps to a id-prefix and a root-index path.
# reference/ nodes keep their existing id scheme (relpath from reference/), unchanged.
# ops/ nodes get id = "ops/" + relpath from ops/.
# runbooks/ nodes get id = "runbooks/" + relpath from runbooks/.
EXTRA_ROOTS = [
    (OPS, "ops/", "ops/index.md"),
    (RUNBOOKS, "runbooks/", "runbooks/index.md"),
]

# --- params (SPEC-Graph §3) ----------------------------------------------
K = 8            # related top-K / degree cap
THETA_RATIO = 0.35
M = 10           # mentions cap per source
MAX_DF = 15      # term frequency window upper bound
GROUP_CAP = 12   # term group size cap for peer (same_topic/demonstrates) edges
INJECT_N = 8     # max related links injected per card

START, END = "<!-- okf:related:start -->", "<!-- okf:related:end -->"
RE_MANAGED = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
RE_H1 = re.compile(r"^#\s+(.*\S)\s*$")
RE_ASCII = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
RE_INCLUDE = re.compile(r'#include\s+"([^"]+\.h)"')
RE_LINK = re.compile(r"\]\(([^)]+\.md)\)")

ALLOWLIST = {  # forced-in operator/API terms (high confidence)
    "add", "matmul", "datacopy", "localtensor", "globaltensor", "tque", "tpipe", "tbuf",
    "softmax", "layernorm", "gelu", "geglu", "acos", "abs", "abssub", "crd2idx", "fixpipe",
    "loaddata", "broadcast", "doublebuffer", "msprof", "msserviceprofiler", "mstx",
    # wiki-enriched terms (ops/runbooks body APIs not covered above)
    "reducemax", "reducesum", "datacopypad", "nonaligned", "inplace", "ub_fusion",
    "binary_add", "bank_conflict", "regbase", "arch35", "cast", "exp", "muls",
    "rotate_half", "mermaid", "apply_rotary_pos_emb",
}
DENYLIST = {  # structure / domain words that must never be terms
    "api", "example", "header", "guide", "index", "reference", "overview", "readme",
    "cann", "ascend", "ascendc", "asc", "devkit", "asc_devkit",
    "basic", "advanced", "data", "structures", "kernel", "operator", "intf", "impl",
    "simd", "simt", "tiling", "profiling", "vector", "scalar", "cube", "context",
    "com", "https", "html", "gitcode", "hiascend", "document", "detail", "blob", "tree",
    "md", "cpp", "asc", "the", "and", "for", "with", "use", "using",
}

FIELD_NOTE_KINDS = {"implementation_trap", "debugging_journey", "cross_skill_gap", "field_note"}
RUNBOOK_KINDS = {"operator_optimization", "runbook"}
CONCEPT_KINDS = {
    "api", "example", "header", "guide", "other", "operator", "glossary",
    "operator_optimization", "implementation_trap", "debugging_journey", "cross_skill_gap",
    # legacy grouping kinds
    "field_note", "runbook",
}
KNOWN_KINDS = CONCEPT_KINDS | {"index", "root"}
TYPE_KIND = dict(
    api_reference="api",
    code_example="example",
    devkit_guide="guide",
    programming_guide="guide",
    profiling_guide="guide",
    migration_guide="guide",
    term="glossary",
    paradigm="glossary",
    operator_spec="operator",
    optimization_runbook="operator_optimization",
    implementation_trap="implementation_trap",
    debugging_journey="debugging_journey",
    cross_skill_gap="cross_skill_gap",
    root_index="index",
    bundle_index="index",
    section_index="index",
    **{
    # legacy display/source types
    "Glossary": "glossary",
    "ASC-DevKit Header": "header",
    "ASC-DevKit Example": "example",
    "ASC-DevKit API Reference": "api",
    "ASC-DevKit API Guide": "api",
    "ASC-DevKit Guide": "guide",
    "Ascend C Dev Guide": "guide",
    "Ascend C Profiling Guide": "guide",
    "Guide": "guide",
    "Operator": "operator",
    "Implementation Trap": "implementation_trap",
    "Debugging Journey": "debugging_journey",
    "Cross-Skill Gap": "cross_skill_gap",
    "Runbook": "operator_optimization",
    "OKF Bundle Index": "index",
        "Glossary Index": "index",
    },
)


# ====================== node model ========================================
def strip_managed(text):
    return RE_MANAGED.sub("", text)


def _parse_card_text(raw):
    fm, body = {}, raw
    if not raw.startswith("---"):
        return fm, body
    end = raw.find("\n---", 3)
    if end == -1:
        return fm, body
    for line in raw[3:end].splitlines():
        match = re.match(r"^(\w+):\s*(.*)$", line)
        if match:
            fm[match.group(1)] = match.group(2).strip()
    return fm, raw[end + 4:]


def read_card(path):
    with open(path, encoding="utf-8", errors="replace") as card_file:
        raw = strip_managed(card_file.read())
    fm, body = _parse_card_text(raw)
    return fm, body


def parse_tags(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    return [t.strip().strip("'\"").lower() for t in v.split(",") if t.strip()]


def kind_of(type_raw, kind_raw=""):
    if kind_raw in KNOWN_KINDS:
        return kind_raw
    t = type_raw or ""
    if t in TYPE_KIND:
        return TYPE_KIND[t]
    if "Glossary" in t:
        return "glossary"
    if "Header" in t:
        return "header"
    if "Example" in t:
        return "example"
    if "API Reference" in t:
        return "api"
    if "Guide" in t:
        return "guide"
    if "Operator" in t:
        return "operator"
    if "Trap" in t:
        return "implementation_trap"
    if "Journey" in t:
        return "debugging_journey"
    if "Gap" in t:
        return "cross_skill_gap"
    if "Runbook" in t or "Optimization" in t:
        return "operator_optimization"
    return "other"


def edge_kind(kind):
    if kind in FIELD_NOTE_KINDS:
        return "field_note"
    if kind in RUNBOOK_KINDS:
        return "runbook"
    return kind


def _first_paragraph(lines):
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "*", "-", "|", "<", "!", "```")):
            return re.sub(r"<[^>]+>", "", stripped)[:140]
    return ""


def first_h1_and_para(body):
    lines = body.splitlines()
    for index, line in enumerate(lines):
        match = RE_H1.match(line)
        if match:
            return match.group(1).strip(), _first_paragraph(lines[index + 1:])
    return "", ""


def _load_dir(base_dir, id_prefix, root_id, nodes):
    """Scan one content root, add nodes to `nodes` dict.
    id_prefix: "" for reference/ (backward compat), "ops/" for ops/, "runbooks/" for runbooks/.
    root_id: the root index.md node id for this tree (e.g. "index.md" or "ops/index.md")."""
    for r, _, fs in os.walk(base_dir):
        for f in sorted(fs):
            if not f.endswith(".md"):
                continue
            path = os.path.join(r, f)
            rel = os.path.relpath(path, base_dir).replace(os.sep, "/")
            nid = (id_prefix + rel) if id_prefix else rel
            parts = rel.split("/")
            fm, body = read_card(path)
            if f == "index.md":
                title, desc = first_h1_and_para(body)
                kind = "root" if nid == root_id else "index"
                nodes[nid] = {"id": nid, "kind": kind, "path": path,
                              "title": title or (parts[-2] if len(parts) > 1 else id_prefix.rstrip("/") or "reference"),
                              "description": desc, "resource": "", "tags": [],
                              "bundle": id_prefix.rstrip("/") if id_prefix else (parts[0] if parts else ""),
                              "category": parts[-2] if len(parts) >= 2 else "", "body": body}
            else:
                nodes[nid] = {"id": nid, "kind": kind_of(fm.get("type", ""), fm.get("kind", "")), "path": path,
                              "title": fm.get("title", f[:-3]).strip("'\""),
                              "description": fm.get("description", "").strip("'\""),
                              "resource": fm.get("resource", "").strip("'\""),
                              "tags": parse_tags(fm.get("tags", "")),
                              "aliases": parse_tags(fm.get("aliases", "")),
                              "bundle": id_prefix.rstrip("/") if id_prefix else parts[0],
                              "category": parts[-2] if len(parts) >= 2 else "",
                              "status": fm.get("status", "").strip("'\""),
                              "body": body}


def is_stub(d):
    """Return whether a provisional card must be excluded from graph operations.

    A card whose frontmatter status is ``stub`` is not yet concept knowledge.
    """
    return d.get("status") == "stub"


def load_nodes():
    nodes = {}
    # reference/ — backward-compatible ids (relpath from reference/)
    _load_dir(REFERENCE, "", "index.md", nodes)
    # ops/ and runbooks/ — prefixed ids
    for base, prefix, root_id in EXTRA_ROOTS:
        if os.path.isdir(base):
            _load_dir(base, prefix, root_id, nodes)
    return nodes


def concept_ids(nodes):
    concepts = []
    for node_id, node in nodes.items():
        if node["kind"] in CONCEPT_KINDS and not is_stub(node):
            concepts.append(node_id)
    return concepts


# ====================== terms (SPEC-Graph §3.1) ===========================
def candidate_tokens(node):
    toks = set(node["tags"])
    toks |= set(node.get("aliases", []))  # glossary aliases: normalize variant spellings for recall
    fields = [node["title"], os.path.basename(node["id"])[:-3], node["category"]]
    fields += node["id"].split("/")
    fields.append(os.path.basename(node["resource"]))
    for fld in fields:
        for m in RE_ASCII.findall(fld or ""):
            toks.add(m.lower())
    return {t for t in toks if t not in DENYLIST}


def build_terms(nodes, cids):
    deny = set(DENYLIST)
    for c in cids:
        deny |= set(c.split("/")[:-1])              # every dir segment is structural
    cand = {c: {t for t in candidate_tokens(nodes[c]) if t not in deny} for c in cids}
    df = {}
    for c in cids:
        for t in cand[c]:
            df[t] = df.get(t, 0) + 1
    vocabulary = set(ALLOWLIST)
    for t, d in df.items():
        if 2 <= d <= MAX_DF and t not in deny:
            vocabulary.add(t)
    vocabulary -= deny
    cjk_allow = {t for t in ALLOWLIST if re.search(r"[一-鿿]", t)}
    terms = {}
    for c in cids:
        ct = {t for t in cand[c] if t in vocabulary}
        for t in cjk_allow:
            if t in nodes[c]["title"]:
                ct.add(t)
        terms[c] = ct
    by_term = {}
    for c in cids:
        for t in terms[c]:
            by_term.setdefault(t, []).append(c)
    return terms, by_term, df, vocabulary


# ====================== edge helpers ======================================
PRIORITY = {"declares": 6, "exemplifies": 6, "explains": 6, "demonstrates": 5,
            "same_topic": 4, "related": 3, "mentions": 2, "part_of": 1}
SYMMETRIC = {"same_topic", "related", "demonstrates"}
DIRECTED_EDGE_RULES = {
    frozenset(("example", "api")): ("example", "exemplifies"),
    frozenset(("guide", "api")): ("guide", "explains"),
    frozenset(("header", "api")): ("header", "declares"),
    frozenset(("operator", "guide")): ("operator", "exemplifies"),
    frozenset(("operator", "api")): ("operator", "exemplifies"),
    frozenset(("runbook", "guide")): ("runbook", "explains"),
    frozenset(("field_note", "guide")): ("field_note", "exemplifies"),
    frozenset(("operator", "runbook")): ("operator", "exemplifies"),
}


def typed_edge(a, b, ka, kb):
    ka, kb = edge_kind(ka), edge_kind(kb)
    pair = frozenset((ka, kb))
    if pair == frozenset(("guide", "example")):
        source, target = sorted((a, b))
        return source, target, "demonstrates"
    rule = DIRECTED_EDGE_RULES.get(pair)
    if rule:
        source_kind, edge_type = rule
        source, target = (a, b) if ka == source_kind else (b, a)
        return source, target, edge_type
    source, target = sorted((a, b))
    return source, target, "same_topic"


# ====================== build (SPEC-Graph §3) =============================
def build_graph(nodes):
    """Return (edges, ctx). ctx carries provenance for the build report."""
    cids = concept_ids(nodes)
    raw = []   # (src, tgt, type, weight, signal, why)
    terms, by_term, df, vocabulary = build_terms(nodes, cids)
    capped = sorted(t for t, g in by_term.items() if len(g) > GROUP_CAP)

    for t, group in by_term.items():
        if len(group) < 2:
            continue
        for a, b in itertools.combinations(group, 2):
            source, target, edge_type = typed_edge(a, b, nodes[a]["kind"], nodes[b]["kind"])
            if edge_type not in ("exemplifies", "explains", "declares") and len(group) > GROUP_CAP:
                continue
            raw.append((source, target, edge_type, 3.0, "term", "shared term: %s" % t))

    rel, stop, idf = related_edges(nodes, cids)
    raw += rel
    raw += include_edges(nodes, cids)
    raw += part_of_edges(nodes)
    raw += mention_edges(nodes, cids, terms, by_term, df)

    edges = merge_edges(raw)
    ctx = {"terms": terms, "by_term": by_term, "df": df, "V": vocabulary, "capped": capped,
           "stop": stop, "idf": idf, "cids": cids}
    return edges, ctx


def related_edges(nodes, cids):
    bundles = {}
    for c in cids:
        bundles.setdefault(nodes[c]["bundle"], []).append(c)
    tag_bundle_df = {}
    for c in cids:
        for g in set(nodes[c]["tags"]):
            tag_bundle_df.setdefault(g, {})
            tag_bundle_df[g][nodes[c]["bundle"]] = tag_bundle_df[g].get(nodes[c]["bundle"], 0) + 1
    stop = set()
    for g, perb in tag_bundle_df.items():
        for bnd, c in perb.items():
            if c == len(bundles.get(bnd, [])):
                stop.add(g)
    stop |= {nodes[c]["category"] for c in cids if nodes[c]["category"]}
    node_count = len(cids)
    tdf = {}
    for c in cids:
        for g in set(nodes[c]["tags"]):
            if g not in stop:
                tdf[g] = tdf.get(g, 0) + 1
    idf = {g: math.log((node_count - d + 0.5) / (d + 0.5) + 1) for g, d in tdf.items()}
    tagset = {c: {g for g in nodes[c]["tags"] if g in idf} for c in cids}
    cand = {}
    for c in cids:
        sims = {}
        for o in cids:
            if o == c:
                continue
            shared = tagset[c] & tagset[o]
            if shared:
                sims[o] = round(sum(idf[g] for g in shared), 6)
        if not sims:
            continue
        mx = max(sims.values())
        ranked = sorted(sims.items(), key=lambda kv: (-kv[1], kv[0]))
        cand[c] = [(o, s) for o, s in ranked if s >= THETA_RATIO * mx][:K]
    pairw, pairtags = {}, {}
    for c, lst in cand.items():
        for o, s in lst:
            key = tuple(sorted([c, o]))
            pairw[key] = max(pairw.get(key, 0.0), s)
            pairtags[key] = sorted(tagset[c] & tagset[o], key=lambda g: -idf[g])[:4]
    out, deg = [], {}
    for (a, b), w in sorted(pairw.items(), key=lambda kv: (-kv[1], kv[0])):
        if deg.get(a, 0) < K and deg.get(b, 0) < K:
            why = "shared tags: %s" % ", ".join(pairtags[(a, b)])
            out.append((a, b, "related", w, "tag", why))
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
    return out, stop, idf


def include_edges(nodes, cids):
    by_basename = {}
    for c in cids:
        bn = os.path.basename(nodes[c]["resource"])
        if bn.endswith(".h"):
            by_basename.setdefault(bn, []).append(c)
    out = []
    for c in cids:
        if nodes[c]["kind"] != "header":
            continue
        for inc in set(RE_INCLUDE.findall(nodes[c]["body"])):
            targets = by_basename.get(os.path.basename(inc), [])
            out.extend(
                (c, target, "declares", 3.0, "include", '#include "%s"' % inc)
                for target in targets if target != c
            )
    return out


def part_of_edges(nodes):
    out = []
    for nid, d in nodes.items():
        if d["kind"] == "root":
            continue
        parent = parent_index(nodes, nid)
        if parent:
            out.append((nid, parent, "part_of", 1.0, "hier", "directory hierarchy"))
    return out


def parent_index(nodes, nid):
    parts = nid.split("/")
    # Determine the root id for this node's tree
    if nid.startswith("ops/"):
        root_id = "ops/index.md"
    elif nid.startswith("runbooks/"):
        root_id = "runbooks/index.md"
    else:
        root_id = "index.md"
    # Don't ascend above the tree's own root
    if nid == root_id:
        return None
    up = parts[:-2] if parts[-1] == "index.md" else parts[:-1]
    while True:
        cand = "/".join(up + ["index.md"]) if up else root_id
        if cand in nodes and cand != nid:
            return cand
        if not up:
            return root_id if root_id in nodes and nid != root_id else None
        up = up[:-1]


def _mention_scores(node_id, nodes, terms, by_term, df):
    body = nodes[node_id]["body"].lower()
    own_terms = terms[node_id]
    scored, reasons = {}, {}
    for term, group in by_term.items():
        if term in own_terms or len(group) > GROUP_CAP:
            continue
        hit = (
            re.search(r"\b" + re.escape(term) + r"\b", body)
            if re.match(r"[a-z]", term) else term in nodes[node_id]["body"]
        )
        if not hit:
            continue
        weight = math.log(len(terms) / df.get(term, 1) + 1)
        for other in group:
            if other != node_id and weight > scored.get(other, 0.0):
                scored[other] = round(weight, 6)
                reasons[other] = "mentions term: %s" % term
    return scored, reasons


def mention_edges(nodes, cids, terms, by_term, df):
    out = []
    for node_id in cids:
        scored, reasons = _mention_scores(node_id, nodes, terms, by_term, df)
        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:M]
        for other, weight in ranked:
            out.append((node_id, other, "mentions", weight, "mention", reasons[other]))
    return out


def merge_edges(raw):
    groups = {}
    for s, t, et, w, sig, why in raw:
        if s == t:
            continue
        groups.setdefault(frozenset([s, t]), []).append((s, t, et, w, sig, why))
    edges = []
    for lst in groups.values():
        bt = max(lst, key=lambda e: PRIORITY[e[2]])[2]
        winners = [e for e in lst if e[2] == bt]
        s, t = sorted(winners, key=lambda e: (e[0], e[1]))[0][:2]
        sym = bt in SYMMETRIC
        if sym:
            s, t = sorted([s, t])
        edges.append({"source": s, "target": t, "type": bt,
                      "weight": round(sum(e[3] for e in lst), 6),
                      "via": sorted({e[4] for e in lst}),
                      "why": "; ".join(sorted({e[5] for e in lst})),
                      "symmetric": sym})
    edges.sort(key=lambda e: (e["source"], e["target"], e["type"]))
    return edges


def fingerprint(nodes):
    h = hashlib.sha1()
    for nid in sorted(nodes):
        with open(nodes[nid]["path"], encoding="utf-8", errors="replace") as card_file:
            raw = strip_managed(card_file.read())
        h.update(nid.encode())
        h.update(hashlib.sha1(raw.encode("utf-8")).digest())
    return h.hexdigest()


# ====================== graph object (synthesized, not committed) =========
def graph_json(nodes, edges, with_bodies=True):
    cids = set(concept_ids(nodes))
    deg = collections.Counter()
    for e in edges:
        if e["type"] != "part_of":
            deg[e["source"]] += 1
            deg[e["target"]] += 1
    out_nodes = []
    for nid, d in sorted(nodes.items()):
        out_nodes.append({"id": d["id"], "title": d["title"], "kind": d["kind"],
                          "bundle": d["bundle"], "category": d["category"], "tags": d["tags"],
                          "resource": d["resource"], "description": d["description"],
                          "degree": deg.get(nid, 0)})
    for e in edges:
        e["cross"] = nodes[e["source"]]["bundle"] != nodes[e["target"]]["bundle"] \
            and bool(nodes[e["source"]]["bundle"]) and bool(nodes[e["target"]]["bundle"])
    g = {"meta": {"card_count": len(cids),
                  "index_count": sum(1 for d in nodes.values() if d["kind"] in ("index", "root")),
                  "edge_count": len(edges),
                  "bundles": sorted({nodes[c]["bundle"] for c in cids}),
                  "fingerprint": fingerprint(nodes),
                  "params": {"K": K, "theta_ratio": THETA_RATIO, "M": M,
                             "max_df": MAX_DF, "group_cap": GROUP_CAP}},
         "nodes": out_nodes, "edges": edges}
    if with_bodies:
        g["bodies"] = {nid: nodes[nid]["body"] for nid in nodes if nid in cids}
    return g


# ====================== candidates (SPEC-Graph v3 §3, recall) =============
MAX_CAND = 30        # candidate cap per card fed to the LLM judge
EXCERPT = 700        # body excerpt chars sent for judging


def body_excerpt(node, n=EXCERPT):
    txt = re.sub(r"\n{3,}", "\n\n", node["body"].strip())
    return txt[:n]


def card_payload(node):
    return {"id": node["id"], "title": node["title"], "kind": node["kind"],
            "description": node["description"], "tags": node["tags"],
            "excerpt": body_excerpt(node)}


def _add_term_candidate_scores(cids, by_term, df, score, via):
    node_count = len(cids)
    for term, group in by_term.items():
        if len(group) < 2:
            continue
        weight = math.log(node_count / df.get(term, len(group)) + 1)
        tag = "term:%s" % term
        for source, target in itertools.permutations(group, 2):
            score[source][target] = score[source].get(target, 0.0) + weight
            via[source].setdefault(target, set()).add(tag)


def _add_tag_candidate_scores(nodes, cids, score, via):
    _, _, idf = related_edges(nodes, cids)
    tagset = {c: {g for g in nodes[c]["tags"] if g in idf} for c in cids}
    for source in cids:
        for target in cids:
            if target == source:
                continue
            shared = tagset[source] & tagset[target]
            if shared:
                score[source][target] = score[source].get(target, 0.0) + sum(idf[g] for g in shared)
                top = sorted(shared, key=lambda g: -idf[g])[:3]
                via[source].setdefault(target, set()).add("tag:%s" % ",".join(top))


def _candidate_output(nodes, cids, score, via, only_bundle):
    forced = {c: set() for c in cids}
    out = {}
    for node_id in cids:
        if only_bundle and nodes[node_id]["bundle"] != only_bundle:
            continue
        ranked = sorted(score[node_id].items(), key=lambda item: (-item[1], item[0]))
        keep = [other for other, _ in ranked][:MAX_CAND]
        keep += [other for other in forced[node_id] if other not in keep]
        candidates = []
        for other in keep:
            provisional_type = typed_edge(
                node_id, other, nodes[node_id]["kind"], nodes[other]["kind"],
            )[2]
            payload = card_payload(nodes[other])
            payload.update({
                "via": sorted(via[node_id].get(other, [])),
                "provisional_type": provisional_type,
                "score": round(score[node_id].get(other, 0.0), 4),
                "forced": other in forced[node_id],
            })
            candidates.append(payload)
        out[node_id] = {"card": card_payload(nodes[node_id]), "candidates": candidates}
    return out


def candidate_sets(nodes, only_bundle=None):
    """Return deterministic term-overlap and tag-IDF recall candidates.

    Mentions are excluded and ``part_of`` is handled separately.
    """
    cids = concept_ids(nodes)
    _, by_term, df, _ = build_terms(nodes, cids)
    score = {node_id: {} for node_id in cids}
    via = {node_id: {} for node_id in cids}
    _add_term_candidate_scores(cids, by_term, df, score, via)
    _add_tag_candidate_scores(nodes, cids, score, via)
    return _candidate_output(nodes, cids, score, via, only_bundle)


def cmd_candidates(only_bundle=None):
    nodes = load_nodes()
    cand = candidate_sets(nodes, only_bundle)
    os.makedirs(BUILD_DIR, exist_ok=True)
    path = os.path.join(BUILD_DIR, "candidates.json")
    with open(path, "w", encoding="utf-8") as candidates_file:
        json.dump(cand, candidates_file, ensure_ascii=False, indent=1)
    sizes = sorted((len(v["candidates"]) for v in cand.values()))
    n = len(sizes)
    suffix = "" if not only_bundle else " (bundle=%s)" % only_bundle
    _log_output("candidates: %d focal cards%s", n, suffix)
    if n:
        _log_output(
            "  cand/card: min=%d median=%d max=%d  (MAX_CAND=%d)",
            sizes[0], sizes[n // 2], sizes[-1], MAX_CAND,
        )
        _log_output("  total candidate pairs: %d", sum(sizes))
    _log_output("  -> %s", os.path.relpath(path, ROOT))


def cmd_build():
    nodes = load_nodes()
    edges, ctx = build_graph(nodes)
    os.makedirs(BUILD_DIR, exist_ok=True)
    g = graph_json(nodes, edges, with_bodies=False)
    graph_path = os.path.join(BUILD_DIR, "okf.graph.json")
    with open(graph_path, "w", encoding="utf-8") as graph_file:
        json.dump(g, graph_file, ensure_ascii=False, indent=1)
    write_report(nodes, edges, ctx)
    by_t = collections.Counter(e["type"] for e in edges)
    by_v = collections.Counter(v for e in edges for v in e["via"])
    _log_output(
        "build: %d concept + %d index nodes, %d edges",
        g["meta"]["card_count"], g["meta"]["index_count"], len(edges),
    )
    _log_output("  by type: %s", dict(by_t))
    _log_output("  by via : %s", dict(by_v))
    _log_output("  report : %s", os.path.relpath(os.path.join(BUILD_DIR, "graph_report.md"), ROOT))


def write_report(nodes, edges, ctx):
    df, vocabulary = ctx["df"], ctx["V"]
    capped, stop = ctx["capped"], ctx["stop"]
    incident = collections.Counter()
    for e in edges:
        if e["type"] != "part_of":
            incident[e["source"]] += 1
            incident[e["target"]] += 1
    isolated = [c for c in ctx["cids"] if incident.get(c, 0) == 0]
    by_t = collections.Counter(e["type"] for e in edges)
    by_v = collections.Counter(v for e in edges for v in e["via"])
    top_terms = sorted(
        ((term, df[term]) for term in vocabulary if term in df),
        key=lambda item: (-item[1], item[0]),
    )[:40]
    lines = ["# OKF graph build report", "",
             "params: K=%d theta_ratio=%.2f M=%d max_df=%d group_cap=%d" %
             (K, THETA_RATIO, M, MAX_DF, GROUP_CAP), "",
             "## edges", "- total: %d" % len(edges),
             "- by type: %s" % dict(by_t), "- by via: %s" % dict(by_v), "",
             "## term vocabulary V (|V|=%d) — top by df" % len(vocabulary),
             "| term | df |", "|---|---|"]
    lines += ["| %s | %d |" % (term, count) for term, count in top_terms]
    lines += ["", "## group_cap-skipped terms (group > %d, no same_topic clique): %d" %
              (GROUP_CAP, len(capped)),
              ", ".join(capped) or "(none)", "",
              "## stop tags (whole-bundle / category, |stop|=%d)" % len(stop),
              ", ".join(sorted(stop)), "",
              "## isolated concept nodes (no non-part_of edge): %d" % len(isolated)]
    lines += ["- " + node_id for node_id in isolated[:40]]
    report_path = os.path.join(BUILD_DIR, "graph_report.md")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def _edge_perspective(edge, me, nodes):
    s, t, et = edge["source"], edge["target"], edge["type"]
    other = t if s == me else s
    k = edge_kind(nodes[me]["kind"])
    lab = {"exemplifies": "实现示例" if k in ("api", "guide") else "实践案例",
           "explains": "开发指南" if k in ("api", "operator", "field_note", "runbook") else "讲解接口",
           "declares": "声明头文件" if k == "api" else "声明的接口",
           "demonstrates": "配套样例" if k == "guide" else "对应指南"}.get(et, "相关主题")
    return lab, other


LABEL_ORDER = ["实现示例", "实践案例", "对应接口", "开发指南", "讲解接口", "声明头文件",
               "声明的接口", "配套样例", "对应指南", "相关主题"]


def related_block(me, ebn, nodes):
    # v3: no hard cap — the LLM already pruned to genuinely related edges; carry its reason.
    incident = [e for e in ebn.get(me, []) if e["type"] != "part_of"]
    incident.sort(key=lambda e: (-PRIORITY[e["type"]], e["source"], e["target"]))
    rows, seen = [], set()
    me_path = nodes[me]["path"]
    me_dir = os.path.dirname(me_path)
    for e in incident:
        lab, other = _edge_perspective(e, me, nodes)
        if other in seen or nodes.get(other, {}).get("kind") in ("index", "root"):
            continue
        seen.add(other)
        other_path = nodes[other]["path"]
        rows.append((lab, nodes[other]["title"],
                     os.path.relpath(other_path, me_dir), e.get("why", "")))
    if not rows:
        return None
    rows.sort(key=lambda r: (LABEL_ORDER.index(r[0]) if r[0] in LABEL_ORDER else 99, r[2]))
    lines = [START, "", "# 相关", ""]
    for lab, title, rel, why in rows:
        suffix = " — %s" % why.strip() if why and why.strip() else ""
        lines.append("- %s: [%s](%s)%s" % (lab, title, _mdlink(rel), suffix))
    lines += ["", END]
    return "\n".join(lines)


def _mdlink(rel):
    return "/".join(urllib.parse.quote(s) for s in rel.split("/"))


def edges_by_node_map(edges):
    m = {}
    for e in edges:
        m.setdefault(e["source"], []).append(e)
        m.setdefault(e["target"], []).append(e)
    return m


# ====================== v3: edges from LLM judgment cache =================
JUDGMENTS_PATH = os.path.join(GRAPH_DIR, "edge_judgments.json")


def load_judgments():
    if os.path.exists(JUDGMENTS_PATH):
        with open(JUDGMENTS_PATH, encoding="utf-8") as judgments_file:
            return json.load(judgments_file)
    return {"version": 3, "model": "", "card_fp": {}, "judgments": {}}


def judgment_fingerprint(node):
    """Fingerprint the semantic card body used for edge judgment freshness.

    Frontmatter governance changes (schema_version/kind/type/source_family/time/tag
    cleanup) should not force a full LLM re-judge when the card body is unchanged.
    """
    body = re.sub(r"\n{3,}", "\n\n", node["body"].strip())
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def card_fp_map(nodes):
    fp = {}
    for nid in concept_ids(nodes):
        fp[nid] = judgment_fingerprint(nodes[nid])
    return fp


def judgment_to_edge(nodes, a, b, jtype, reason):
    """Convert one kept LLM judgment to a canonical typed edge.

    Cross-kind type and direction follow ``typed_edge``; same-kind edges retain the
    LLM's ``same_topic`` or ``related`` classification.
    """
    s, t, conv = typed_edge(a, b, nodes[a]["kind"], nodes[b]["kind"])
    if conv in ("exemplifies", "explains", "declares", "demonstrates"):
        et, src, tgt = conv, s, t
    else:
        et = jtype if jtype in ("same_topic", "related") else "same_topic"
        src, tgt = sorted([a, b])
    return {"source": src, "target": tgt, "type": et, "weight": 1.0,
            "via": ["llm"], "why": reason, "symmetric": et in SYMMETRIC}


def v3_edges(nodes):
    """Build the v3 graph from LLM judgments and the deterministic ``part_of`` tree.

    Mention and keyword-defined edges are intentionally excluded.
    """
    judg = load_judgments()
    edges, seen = [], set()
    for j in judg.get("judgments", {}).values():
        if not j.get("related"):
            continue
        a, b = j.get("a"), j.get("b")
        if a not in nodes or b not in nodes or a == b:
            continue
        key = frozenset([a, b])
        if key in seen:
            continue
        seen.add(key)
        edges.append(judgment_to_edge(nodes, a, b, j.get("type", "related"), j.get("reason", "")))
    for source, target, *_, why in part_of_edges(nodes):
        edges.append({"source": source, "target": target, "type": "part_of", "weight": 1.0,
                      "via": ["hier"], "why": why, "symmetric": False})
    edges.sort(key=lambda e: (e["source"], e["target"], e["type"]))
    return edges


def cmd_inject(only_bundle=None):
    nodes = load_nodes()
    edges = v3_edges(nodes)
    ebn = edges_by_node_map(edges)
    n = 0
    for nid in concept_ids(nodes):
        if only_bundle and nodes[nid]["bundle"] != only_bundle:
            continue
        path = nodes[nid]["path"]
        with open(path, encoding="utf-8", errors="replace") as card_file:
            raw = card_file.read()
        base = strip_managed(raw).rstrip("\n")
        block = related_block(nid, ebn, nodes)
        new = base + ("\n\n" + block + "\n" if block else "\n")
        if new != raw:
            with open(path, "w", encoding="utf-8") as card_file:
                card_file.write(new)
            n += 1
    suffix = " [%s]" % only_bundle if only_bundle else ""
    _log_output("inject%s: updated %d cards", suffix, n)


# --- viz (synthesized on read, self-contained; combined + per-bundle) ----
def subgraph(g, bundle):
    """Bundle subgraph: nodes in `bundle` + their 1-hop external neighbors (flagged)."""
    keep = {n["id"] for n in g["nodes"] if n["bundle"] == bundle}
    ext = set()
    edges = []
    for e in g["edges"]:
        s, t = e["source"], e["target"]
        if s in keep or t in keep:
            edges.append(e)
            if s not in keep:
                ext.add(s)
            if t not in keep:
                ext.add(t)
    nid = keep | ext
    nodes = []
    for n in g["nodes"]:
        if n["id"] in nid:
            n = dict(n)
            n["external"] = n["id"] in ext
            nodes.append(n)
    sub = {"meta": dict(g["meta"], focus=bundle, card_count=len(keep)),
           "nodes": nodes, "edges": edges,
           "bodies": {k: v for k, v in g.get("bodies", {}).items() if k in nid}}
    return sub


def cmd_viz(only_bundle=None):
    nodes = load_nodes()
    edges = v3_edges(nodes)
    g = graph_json(nodes, edges, with_bodies=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    written = []
    targets = ([only_bundle] if only_bundle else [None] + g["meta"]["bundles"])
    for b in targets:
        if b is None:
            data, title, out = g, "OKF 知识图谱 · 全部 bundle", "viz.html"
        else:
            data, title, out = subgraph(g, b), "OKF 知识图谱 · %s" % b, "viz.%s.html" % b
        html = (VIZ_TEMPLATE.replace("__TITLE__", title)
                .replace("__GRAPH__", json.dumps(data, ensure_ascii=False)))
        with open(os.path.join(GRAPH_DIR, out), "w", encoding="utf-8") as visualization_file:
            visualization_file.write(html)
        written.append("%s (%d nodes, %d edges)" % (out, len(data["nodes"]), len(data["edges"])))
    for w in written:
        _log_output("viz: %s", w)


# --- verify ---------------------------------------------------------------
def _verify_roots_and_kinds(nodes):
    errors = []
    if nodes.get("index.md", {}).get("kind") != "root":
        errors.append("missing unique root node index.md")
    for extra_root in ("ops/index.md", "runbooks/index.md"):
        if extra_root in nodes and nodes[extra_root].get("kind") != "root":
            errors.append("missing root for %s" % extra_root)
    for node_id, node in nodes.items():
        if node["kind"] == "other":
            errors.append("unmapped kind: %s" % node_id)
    return errors


def _verify_edges(nodes, edges):
    errors, node_ids, seen = [], set(nodes), set()
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source == target:
            errors.append("self-loop: " + source)
        if source not in node_ids or target not in node_ids:
            errors.append("dangling: %s->%s" % (source, target))
        if edge["type"] not in PRIORITY:
            errors.append("bad type: " + edge["type"])
        if edge["type"] == "mentions":
            errors.append("mentions edge present — removed in v3: " + source)
        key = frozenset((source, target))
        if key in seen:
            errors.append("dup pair: %s/%s" % (source, target))
        seen.add(key)
        if edge["symmetric"] and (source, target) != tuple(sorted((source, target))):
            errors.append("symmetric not canonical: %s/%s" % (source, target))
    return errors


def _verify_judgments(nodes, judgments):
    errors = []
    fingerprints = card_fp_map(nodes)
    for judgment in judgments.get("judgments", {}).values():
        endpoint_missing = judgment.get("a") not in nodes or judgment.get("b") not in nodes
        if judgment.get("related") and endpoint_missing:
            errors.append("judgment endpoint missing: %s/%s" % (judgment.get("a"), judgment.get("b")))
    cached = judgments.get("card_fp", {})
    stale = [node_id for node_id, value in cached.items()
             if node_id in fingerprints and fingerprints[node_id] != value]
    new_unjudged = [node_id for node_id in fingerprints if node_id not in cached]
    if stale:
        errors.append(
            "%d cards changed since last judge (run `judge` to refresh): %s"
            % (len(stale), ", ".join(stale[:5]))
        )
    if new_unjudged:
        errors.append("%d cards never judged: %s" % (len(new_unjudged), ", ".join(new_unjudged[:5])))
    return errors


def _expected_root(node_id):
    if node_id.startswith("ops/"):
        return "ops/index.md"
    if node_id.startswith("runbooks/"):
        return "runbooks/index.md"
    return "index.md"


def _verify_part_of(nodes, edges):
    errors = []
    parent_map = {edge["source"]: edge["target"] for edge in edges if edge["type"] == "part_of"}
    for node_id, node in nodes.items():
        if node["kind"] == "root":
            continue
        current, steps = node_id, 0
        while current in parent_map and steps < 50:
            current = parent_map[current]
            steps += 1
        expected = _expected_root(node_id)
        if current != expected:
            errors.append(
                "part_of not reaching root: %s (reached %s, expected %s)"
                % (node_id, current, expected)
            )
    return errors


def _dead_related_links(match, path):
    if match is None:
        return []
    dead_links = []
    for link in RE_LINK.findall(match.group(0)):
        target = os.path.normpath(os.path.join(os.path.dirname(path), urllib.parse.unquote(link)))
        if not os.path.exists(target):
            dead_links.append(link)
    return dead_links


def _verify_managed_blocks(nodes, edges):
    errors = []
    edges_by_node = edges_by_node_map(edges)
    for node_id in concept_ids(nodes):
        path = nodes[node_id]["path"]
        with open(path, encoding="utf-8", errors="replace") as card_file:
            raw = card_file.read()
        if raw.count(START) > 1 or raw.count(END) > 1:
            errors.append("multiple managed blocks: " + node_id)
        match = RE_MANAGED.search(raw)
        errors.extend(
            "dead related link in %s -> %s" % (node_id, link)
            for link in _dead_related_links(match, path)
        )
        if START in raw:
            expected = related_block(node_id, edges_by_node, nodes)
            actual = match.group(0).strip() if match else ""
            if (expected or "").strip() != actual:
                errors.append("injection not idempotent: " + node_id)
    return errors


def cmd_verify():
    nodes = load_nodes()
    edges = v3_edges(nodes)
    judgments = load_judgments()
    errors = _verify_roots_and_kinds(nodes)
    errors += _verify_edges(nodes, edges)
    errors += _verify_judgments(nodes, judgments)
    errors += _verify_part_of(nodes, edges)
    errors += _verify_managed_blocks(nodes, edges)
    if os.path.exists(os.path.join(GRAPH_DIR, "okf.graph.json")):
        errors.append("graph/okf.graph.json is committed — must be gitignored (synthesize on read)")
    _log_output(
        "=== verify (v3) ===\nnodes: %d  edges: %d  judgments: %d",
        len(nodes), len(edges), len(judgments.get("judgments", {})),
    )
    for error in errors[:40]:
        _log_output("  FAIL: %s", error)
    if errors:
        return 1
    _log_output("OK")
    return 0


def cmd_related(path):
    nodes = load_nodes()
    edges = v3_edges(nodes)
    for e in sorted(edges_by_node_map(edges).get(path, []),
                    key=lambda e: (-PRIORITY[e["type"]], -e["weight"])):
        o = e["target"] if e["source"] == path else e["source"]
        arrow = "→" if e["source"] == path or e["symmetric"] else "←"
        _log_output(
            "%-12s %s w=%.2f via=%s  %-55s  (%s)",
            e["type"], arrow, e["weight"], ",".join(e["via"]), o, e["why"],
        )


def cmd_explain(a, b):
    nodes = load_nodes()
    edges = v3_edges(nodes)
    for e in edges:
        if {e["source"], e["target"]} == {a, b}:
            _log_output(
                "%s  %s →%s %s", e["type"], e["source"],
                "" if e["symmetric"] else ">", e["target"],
            )
            _log_output("  via    : %s", ", ".join(e["via"]))
            _log_output("  why    : %s", e["why"])
            _log_output("  weight : %s | symmetric: %s", e["weight"], e["symmetric"])
            return
    _log_output("no edge between:\n  %s\n  %s", a, b)
    _log_output("(not connected by any of: term / tag-IDF / #include·hierarchy / mentions)")


VIZ_TEMPLATE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:#1b2733}
#app{display:flex;flex-direction:column;height:100%}
header{padding:8px 12px;border-bottom:1px solid #e2e8f0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
header .t{font-weight:700}header .muted{color:#64748b;font-size:12px}
input,select{font-size:12px;padding:4px 6px;border:1px solid #cbd5e1;border-radius:4px}
.legend{font-size:11px;color:#475569;display:flex;gap:8px;flex-wrap:wrap}
.legend b{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px;vertical-align:middle}
main{flex:1;display:flex;min-height:0}
#cy{flex:1;min-width:0}#detail{width:40%;max-width:560px;border-left:1px solid #e2e8f0;overflow:auto;padding:14px 16px;font-size:13px}
.chip{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
.tag{display:inline-block;background:#eef2f7;color:#334155;border-radius:8px;padding:1px 7px;font-size:11px;margin:1px}
#detail h2{font-size:18px;margin:6px 0 2px}#detail .id{color:#94a3b8;font-size:11px;font-family:monospace;word-break:break-all}
.grp{margin:8px 0 2px;font-weight:700;color:#0f4c92;font-size:12px}
.nb{display:block;padding:2px 0;color:#1d4ed8;cursor:pointer;text-decoration:none}.nb:hover{text-decoration:underline}
.nb .w{color:#94a3b8;font-size:11px}
#body{margin-top:10px;border-top:1px solid #eef2f7;padding-top:8px}
#body h1{font-size:15px}#body h2{font-size:13px}#body pre{background:#0f172a;color:#e2e8f0;padding:8px;border-radius:6px;overflow:auto;font-size:12px}
#body code{background:#f1f5f9;padding:1px 4px;border-radius:3px;font-family:monospace}#body a.ext{color:#2563eb}#body a.int{color:#7c3aed;cursor:pointer}
#body table{border-collapse:collapse}#body td,#body th{border:1px solid #e2e8f0;padding:2px 6px;font-size:12px}
.filters{font-size:11px;color:#475569;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.filters label{white-space:nowrap}
</style></head>
<body><div id="app">
<header>
  <span class="t">__TITLE__</span><span class="muted" id="meta"></span>
  <input id="q" placeholder="搜索 title / id / tag" style="width:200px">
  <select id="layout"><option value="cose">力导向 cose</option><option value="concentric">同心</option><option value="breadthfirst">层次</option><option value="circle">环形</option><option value="grid">网格</option></select>
  <button id="reset">重置</button>
  <span class="filters" id="bundleF"></span>
  <span class="filters" id="kindF"></span>
  <span class="filters" id="typeF"></span>
  <span class="legend" id="legend"></span>
</header>
<main><div id="cy"></div><div id="detail">点击节点查看详情、邻居与「被引用」。</div></main>
</div>
<script>
const G=__GRAPH__;
const KC={api:'#1f77b4',example:'#2e9e6b',header:'#7e57c2',guide:'#ff7f0e',index:'#cbd5e1',root:'#334155',other:'#94a3b8',operator:'#e6194b',glossary:'#0f766e',operator_optimization:'#3cb371',implementation_trap:'#f032e6',debugging_journey:'#c026d3',cross_skill_gap:'#a21caf',field_note:'#f032e6',runbook:'#3cb371'};
const EC={exemplifies:'#2e9e6b',explains:'#ff7f0e',declares:'#7e57c2',demonstrates:'#0891b2',same_topic:'#1f77b4',related:'#94a3b8',part_of:'#e2e8f0',mentions:'#f1d6c0'};
const ETZH={exemplifies:'实现示例',explains:'开发指南',declares:'声明',demonstrates:'配套',same_topic:'同主题',related:'相关',part_of:'层级',mentions:'提及'};
const nodeById={};G.nodes.forEach(n=>nodeById[n.id]=n);
const back={};G.edges.forEach(e=>{(back[e.target]=back[e.target]||[]).push(e);});
document.getElementById('meta').textContent=`· ${G.meta.card_count} concept / ${G.nodes.length} 节点 / ${G.edges.length} 边`;
const sz=n=>10+Math.min(28,(n.degree||0)*2);
const els=[];
for(const n of G.nodes)els.push({data:{id:n.id,label:n.title,kind:n.kind,bundle:n.bundle,ext:!!n.external,sz:sz(n)}});
for(const e of G.edges)els.push({data:{id:e.source+'|'+e.target,source:e.source,target:e.target,type:e.type,e:e}});
const cy=cytoscape({container:document.getElementById('cy'),elements:els,wheelSensitivity:0.2,
 style:[
  {selector:'node',style:{'background-color':n=>KC[n.data('kind')]||'#94a3b8','width':'data(sz)','height':'data(sz)','label':'data(label)','font-size':6,'color':'#334155','text-opacity':0,'text-valign':'bottom','text-margin-y':2,'text-wrap':'ellipsis','text-max-width':90,'border-width':n=>n.data('ext')?2:0,'border-color':'#94a3b8','border-style':'dashed'}},
  {selector:'node[?ext]',style:{'background-opacity':0.45}},
  {selector:'node:selected',style:{'border-width':3,'border-color':'#f59e0b','border-style':'solid','text-opacity':1,'font-size':9}},
  {selector:'.nbr',style:{'text-opacity':1,'font-size':8}},
  {selector:'edge',style:{'width':e=>e.data('e').cross?2:1,'line-color':e=>EC[e.data('type')]||'#ccc','line-style':e=>e.data('e').cross?'dashed':'solid','curve-style':'haystack','opacity':0.55,'target-arrow-shape':e=>e.data('e').symmetric?'none':'triangle','target-arrow-color':e=>EC[e.data('type')]||'#ccc','arrow-scale':0.6}},
  {selector:'edge:selected',style:{'opacity':1,'width':3}},
  {selector:'.dim',style:{'opacity':0.06,'text-opacity':0}}
 ],
 layout:{name:'cose',animate:false,nodeRepulsion:9000,idealEdgeLength:55,padding:30}});

function resolveLink(fromId,href){let base=fromId.split('/').slice(0,-1);for(const p of href.split('/')){if(p==='..')base.pop();else if(p!=='.')base.push(p);}let id=base.join('/');return nodeById[id]?id:null;}
function showDetail(id){const n=nodeById[id];if(!n)return;cy.$(':selected').unselect();const node=cy.getElementById(id);node.select();cy.elements().removeClass('nbr');node.connectedNodes().addClass('nbr');cy.animate({center:{eles:node},duration:200});
 let h=`<span class="chip" style="background:${KC[n.kind]}">${n.kind}</span> ${n.external?'<span class="chip" style="background:#94a3b8">外部</span>':''}<h2>${esc(n.title)}</h2><div class="id">${n.id}</div>`;
 if(n.description)h+=`<p>${esc(n.description)}</p>`;
 if(n.resource)h+=`<p><a class="ext" href="${n.resource}" target="_blank" rel="noopener">上游 resource ↗</a></p>`;
 if(n.tags&&n.tags.length)h+=`<div>${n.tags.map(t=>`<span class="tag">${esc(t)}</span>`).join('')}</div>`;
 const inc=cy.getElementById(id).connectedEdges().map(x=>x.data('e'));
 const groups={};inc.forEach(e=>{if(e.type==='part_of')return;const me=id;const other=e.source===me?e.target:e.source;const lab=persp(e,me);(groups[lab]=groups[lab]||[]).push([other,e]);});
 const order=['实现示例','对应接口','开发指南','讲解接口','声明头文件','声明的接口','配套样例','对应指南','同主题','相关','提及'];
 Object.keys(groups).sort((a,b)=>order.indexOf(a)-order.indexOf(b)).forEach(lab=>{h+=`<div class="grp">${lab} (${groups[lab].length})</div>`;groups[lab].forEach(([o,e])=>{const t=nodeById[o];h+=`<a class="nb" onclick="showDetail('${o}')">${esc(t?t.title:o)} <span class="w">· ${ETZH[e.type]||e.type} ${e.weight.toFixed(1)}</span></a>`;});});
 const bl=(back[id]||[]).filter(e=>e.type!=='part_of'&&!e.symmetric);
 if(bl.length){h+=`<div class="grp">被引用 / 指向本卡 (${bl.length})</div>`;bl.forEach(e=>{const t=nodeById[e.source];h+=`<a class="nb" onclick="showDetail('${e.source}')">${esc(t?t.title:e.source)} <span class="w">· ${ETZH[e.type]||e.type}</span></a>`;});}
 const body=(G.bodies||{})[id];if(body){h+=`<div id="body">${marked.parse(body,{gfm:true,breaks:false})}</div>`;}
 const d=document.getElementById('detail');d.innerHTML=h;d.scrollTop=0;
 d.querySelectorAll('#body a[href]').forEach(a=>{const href=a.getAttribute('href');if(href&&href.endsWith('.md')&&!href.includes('://')){const tid=resolveLink(id,href);if(tid){a.className='int';a.removeAttribute('href');a.onclick=()=>showDetail(tid);return;}}a.className='ext';a.target='_blank';a.rel='noopener';});}
function kg(k){if(['implementation_trap','debugging_journey','cross_skill_gap','field_note'].includes(k))return'field_note';if(['operator_optimization','runbook'].includes(k))return'runbook';return k;}
function persp(e,me){const k=kg(nodeById[me].kind);if(e.type==='exemplifies'){if(k==='api'||k==='guide')return'实现示例';if(k==='operator'||k==='field_note')return'实践案例';return'对应接口';}if(e.type==='explains'){if(k==='api')return'开发指南';if(k==='operator'||k==='field_note'||k==='runbook')return'开发指南';return'讲解接口';}if(e.type==='declares')return k==='api'?'声明头文件':'声明的接口';if(e.type==='demonstrates')return k==='guide'?'配套样例':'对应指南';if(e.type==='same_topic')return'同主题';if(e.type==='related')return'相关';if(e.type==='mentions')return'提及';return'相关';}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
cy.on('tap','node',ev=>showDetail(ev.target.id()));
cy.on('tap','edge',ev=>{const e=ev.target.data('e');document.getElementById('detail').innerHTML=`<div class="grp">边</div><p><b>${ETZH[e.type]||e.type}</b> ${e.symmetric?'(对称)':''}<br><span class="id">${e.source}<br>${e.symmetric?'—':'→'} ${e.target}</span></p><p>via: ${e.via.join(', ')}<br>why: ${esc(e.why)}<br>weight: ${e.weight}</p>`;});
cy.on('tap',ev=>{if(ev.target===cy){cy.elements().removeClass('nbr');cy.$(':selected').unselect();}});
document.getElementById('layout').onchange=e=>cy.layout({name:e.target.value,animate:false,padding:30}).run();
document.getElementById('reset').onclick=()=>{cy.fit(null,30);document.getElementById('q').value='';cy.elements().removeClass('dim');};
document.getElementById('q').oninput=e=>{const q=e.target.value.trim().toLowerCase();if(!q){cy.elements().removeClass('dim');return;}cy.nodes().forEach(n=>{const d=n.data();const hay=(d.label+' '+d.id+' '+d.bundle).toLowerCase();n.toggleClass('dim',!hay.includes(q));});cy.edges().forEach(e=>e.toggleClass('dim',e.source().hasClass('dim')||e.target().hasClass('dim')));};
function mkfilter(host,title,keys,active,attr){let h=`<b>${title}</b>`;keys.forEach(k=>h+=`<label><input type=checkbox ${active(k)?'checked':''} data-f="${attr}" data-v="${k}">${k}</label>`);document.getElementById(host).innerHTML=h;}
mkfilter('bundleF','bundle:',G.meta.bundles,()=>true,'bundle');
mkfilter('kindF','kind:',['api','example','header','guide','glossary','index','other','operator','operator_optimization','implementation_trap','debugging_journey','cross_skill_gap'],()=>true,'kind');
mkfilter('typeF','边:',Object.keys(EC),k=>k!=='mentions'&&k!=='part_of','type');
document.getElementById('legend').innerHTML=Object.keys(EC).map(t=>`<span><b style="background:${EC[t]}"></b>${ETZH[t]}</span>`).join('');
function applyFilters(){const on=a=>new Set([...document.querySelectorAll(`input[data-f="${a}"]:checked`)].map(x=>x.dataset.v));
 const kb=on('bundle'),kk=on('kind'),kt=on('type');
 cy.nodes().forEach(n=>{const d=n.data();n.style('display',(kb.has(d.bundle)||!d.bundle)&&kk.has(d.kind)?'element':'none');});
 cy.edges().forEach(e=>{const vis=kt.has(e.data('type'))&&e.source().style('display')==='element'&&e.target().style('display')==='element';e.style('display',vis?'element':'none');});}
document.querySelectorAll('.filters input').forEach(cb=>cb.onchange=applyFilters);applyFilters();
const first=G.nodes.find(n=>n.kind==='api'&&!n.external)||G.nodes.find(n=>!n.external)||G.nodes[0];if(first)showDetail(first.id);
</script></body></html>"""


def report_root_argument_error():
    """Log a malformed early ``--knowledge-root`` argument, if present.
    """
    if not ROOT_ARGUMENT_ERROR:
        return False
    _log_error("%s", ROOT_ARGUMENT_ERROR)
    return True


def main():
    if report_root_argument_error():
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge-root", help="OKF knowledge-base root")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    cnd = sub.add_parser("candidates")
    cnd.add_argument("--bundle", default=None)
    inj = sub.add_parser("inject")
    inj.add_argument("--bundle", default=None)
    vz = sub.add_parser("viz")
    vz.add_argument("--bundle", default=None)
    sub.add_parser("verify")
    rel = sub.add_parser("related")
    rel.add_argument("path")
    ex = sub.add_parser("explain")
    ex.add_argument("a")
    ex.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "build":
        return cmd_build()
    if a.cmd == "candidates":
        return cmd_candidates(a.bundle)
    if a.cmd == "inject":
        return cmd_inject(a.bundle)
    if a.cmd == "viz":
        return cmd_viz(a.bundle)
    if a.cmd == "verify":
        return cmd_verify()
    if a.cmd == "related":
        return cmd_related(a.path)
    return cmd_explain(a.a, a.b)


if __name__ == "__main__":
    raise SystemExit(main())
