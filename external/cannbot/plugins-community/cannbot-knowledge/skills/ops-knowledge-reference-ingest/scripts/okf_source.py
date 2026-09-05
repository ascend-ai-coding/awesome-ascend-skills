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
"""OKF Source adapters — the producer-side `source` abstraction.

Mirrors knowledge-catalog/okf/src/reference_agent/sources/ (`Source` ABC with
`list_concepts()` / `read_concept()`): a Source advertises the concepts it can
produce and where each one comes from upstream. Provenance lives per-concept in
the card's frontmatter `resource:`; the concept->source aggregate is SYNTHESIZED
on demand here, never固化 into a checked-in manifest. See SPEC-Source.md.

Two listing modes per bundle:
  - list_concepts()  -> the PRODUCED set (scan reference/<bundle> frontmatter);
                        authoritative for the knowledge base's current source view.
  - list_upstream()  -> the ADVERTISED set (asc: Builder over .build/asc-devkit;
                        cann: the .build worklist); used to detect drift/coverage.

CLI:
  python3 scripts/okf_source.py --knowledge-root <知识库根> list  <bundle> [--json] [--upstream]
  python3 scripts/okf_source.py --knowledge-root <知识库根> drift <bundle>
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class _CliLogger(logging.Logger):
    def __init__(self, name, stream_name, fallback_stream):
        super().__init__(name, logging.INFO)
        self.name = name
        self.stream_name = stream_name
        self.fallback_stream = fallback_stream

    def isEnabledFor(self, level):
        """Apply this logger's level without the global logging disable switch."""
        return level >= self.getEffectiveLevel()

    def handle(self, record):
        """Emit one unprefixed record to the current CLI stream."""
        stream = getattr(sys, self.stream_name) if self.stream_name else self.fallback_stream
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.handle(record)
        handler.close()


def make_cli_logger(name, stream):
    """Create a no-prefix logger that preserves the scripts' CLI stream format."""
    stream_name = "stdout" if stream is sys.stdout else "stderr" if stream is sys.stderr else None
    return _CliLogger(name, stream_name, stream)


STDOUT_LOGGER = make_cli_logger(__name__ + ".stdout", sys.stdout)
STDERR_LOGGER = make_cli_logger(__name__ + ".stderr", sys.stderr)


class CliError(RuntimeError):
    """Expected command-line failure that should not produce a traceback."""


def _knowledge_root_from_argv(argv):
    root = None
    rest = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--knowledge-root":
            if index + 1 >= len(argv):
                raise CliError("--knowledge-root requires a path")
            root = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--knowledge-root="):
            root = arg.split("=", 1)[1]
            if not root:
                raise CliError("--knowledge-root requires a path")
            index += 1
            continue
        rest.append(arg)
        index += 1
    argv[:] = rest
    if not root:
        return None
    root = os.path.abspath(os.path.expanduser(root))
    os.environ["CANNBOT_KNOWLEDGE_ROOT"] = root
    os.environ["CANNBOT_KNOWLEDGE_ROOTS"] = root
    os.environ["OKF_KNOWLEDGE_ROOT"] = root
    os.environ["OKF_KNOWLEDGE_ROOTS"] = root
    os.environ["KNOWLEDGE_ROOT"] = root
    os.environ["KNOWLEDGE_ROOTS"] = root
    return root


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_ARGUMENT_ERROR = None
try:
    _ARGV_ROOT = _knowledge_root_from_argv(sys.argv)
except CliError as root_argument_error:
    _ROOT_ARGUMENT_ERROR = root_argument_error
    _ARGV_ROOT = None
ROOT = (
    _ARGV_ROOT
    or os.environ.get("CANNBOT_KNOWLEDGE_ROOT")
    or os.environ.get("OKF_KNOWLEDGE_ROOT")
    or os.environ.get("KNOWLEDGE_ROOT")
    or os.getcwd()
)
ROOT = os.path.abspath(ROOT)
REFERENCE = os.path.join(ROOT, "reference")
BUILD = os.path.join(ROOT, ".build")


def validate_root_argument():
    """Raise a deferred command-line root parsing error at the actual entry point."""
    if _ROOT_ARGUMENT_ERROR is not None:
        raise _ROOT_ARGUMENT_ERROR


@dataclass(frozen=True)
class ConceptRef:
    """A concept advertised by a source.

    `id` = path under the bundle without the
    `.md` suffix (catalog convention); `resource` = the PRIMARY upstream URL.
    `sources` = the full multi-source set `((url, role), ...)`; empty for legacy
    single-`resource` cards (then `resource` is the sole source).
    """
    id: str
    type: str
    resource: str
    hint: dict = field(default_factory=dict)   # bundle-specific extras (kind / section)
    sources: tuple = ()                        # ((url, role), ...) — see SPEC-Reference §3.2


def md_rel(p):
    return "/".join(urllib.parse.quote(s) for s in p.split("/"))


def read_fm(path):
    fm = {}
    with open(path, encoding="utf-8") as source_file:
        lines = source_file.read().splitlines()
    if not lines or lines[0].strip() != "---":
        return fm
    for ln in lines[1:]:
        if ln.strip() == "---":
            break
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", ln)
        if m:
            fm[m.group(1)] = m.group(2).strip().strip("'\"")
    return fm


def parse_sources(path):
    """Read the `sources:` block from frontmatter.

    Returns `[(url, role), ...]`; empty for legacy single-`resource` cards. The
    scalar parsers in okf_graph/knowledge_query skip this block harmlessly (the
    indented `- url:` / `role:` lines don't match their `^(\\w+):` key regex).
    """
    with open(path, encoding="utf-8") as source_file:
        lines = source_file.read().splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    out, cur, in_block = [], None, False
    for ln in lines[1:]:
        if ln.strip() == "---":
            break
        if re.match(r"^sources:\s*$", ln):
            in_block = True
            continue
        if not in_block:
            continue
        if re.match(r"^[A-Za-z_]", ln):        # next top-level key ends the block
            break
        mu = re.match(r"^\s*-\s*url:\s*(.+)$", ln)
        mr = re.match(r"^\s*role:\s*(.+)$", ln)
        if mu:
            if cur:
                out.append(cur)
            cur = {"url": mu.group(1).strip().strip("'\""), "role": ""}
        elif mr and cur is not None:
            cur["role"] = mr.group(1).strip().strip("'\"")
    if cur:
        out.append(cur)
    return [(d["url"], d.get("role", "")) for d in out]


def concept_rels(bundle):
    """Produced concept files under reference/<bundle> (excl. every index.md)."""
    base = os.path.join(REFERENCE, bundle)
    out = []
    for r, _, fs in os.walk(base):
        for f in fs:
            if f.endswith(".md") and f != "index.md":
                out.append(os.path.relpath(os.path.join(r, f), base))
    return sorted(out)


_TAG_ALIASES = {
    "c-api": "c_api", "high-level-api": "high_level_api",
    "RoPE": "rope", "vv-fusion": "vv_fusion",
}


def clean_tags(tags, structural_tags):
    """Normalize tags and remove structural navigation terms."""
    cleaned = []
    for value in tags:
        tag = (value or "").strip().strip("'\"")
        if not tag:
            continue
        tag = _TAG_ALIASES.get(tag, tag)
        if re.match(r"^[A-Za-z0-9_+.-]+$", tag):
            tag = tag.lower().replace("-", "_")
        if tag not in structural_tags and tag not in cleaned:
            cleaned.append(tag)
    return cleaned


class Source(ABC):
    """Producer-side source for one bundle (catalog `Source` analog)."""
    bundle = ""
    columns = ()           # extra table columns after `concept`

    @abstractmethod
    def list_concepts(self) -> list[ConceptRef]:
        """The PRODUCED set: one ConceptRef per concept card in reference/<bundle>."""

    def list_upstream(self) -> list[ConceptRef]:
        """Return the advertised set from upstream or planning artifacts.

        The default is the produced set; adapters with a live or cached upstream
        override this method.
        """
        return self.list_concepts()

    def read_concept(self, ref: ConceptRef) -> str:
        """Raw concept markdown for a ref (catalog `read_concept` analog)."""
        path = os.path.join(REFERENCE, self.bundle, ref.id + ".md")
        with open(path, encoding="utf-8") as concept_file:
            return concept_file.read()

    def render_table(self, refs=None) -> str:
        """Synthesize the ephemeral concept-to-source markdown table.

        Multi-source concepts contribute several rows, one per source.
        """
        refs = self.list_concepts() if refs is None else refs
        head = "| concept | %s |" % " | ".join(self.columns)
        sep = "|%s|" % "|".join(["---"] * (len(self.columns) + 1))
        lines = [head, sep]
        for ref in refs:
            link = "[%s.md](../reference/%s/%s.md)" % (ref.id, self.bundle, md_rel(ref.id))
            for cells in self._rows(ref):
                lines.append("| %s | %s |" % (link, " | ".join(cells)))
        return "\n".join(lines)

    def _cells(self, ref: ConceptRef) -> tuple:
        """Return the non-concept cells for this bundle's table row."""
        raise NotImplementedError

    def _rows(self, ref: ConceptRef) -> list:
        """Return one cell tuple per source, or one tuple for legacy concepts."""
        return [self._cells(ref)]


def _kind_of(fm):
    if fm.get("kind"):
        return fm.get("kind")
    t = fm.get("type", "")
    if t in ("code_example",) or t.endswith("Example"):
        return "example"
    if t.endswith("Header"):
        return "header"
    if t in ("api_reference",):
        return "api"
    if t in ("devkit_guide", "programming_guide", "profiling_guide", "migration_guide"):
        return "guide"
    return "doc"


class AscDevkitSource(Source):
    bundle = "asc-devkit"
    columns = ("upstream url", "role", "kind")

    def list_concepts(self):
        base = os.path.join(REFERENCE, self.bundle)
        refs = []
        for rel in concept_rels(self.bundle):
            path = os.path.join(base, rel)
            fm = read_fm(path)
            srcs = tuple(parse_sources(path))
            primary = next((u for u, r in srcs if r == "primary"), "") or fm.get("resource", "")
            refs.append(ConceptRef(rel[:-3], fm.get("type", ""), primary,
                                   {"kind": _kind_of(fm)}, srcs))
        return refs

    def list_upstream(self):
        """Full advertised set from the cloned upstream via the extract Builder."""
        sys.path.insert(0, SCRIPT_DIR)
        import asc_devkit_extract as ax
        b = ax.Builder(ax.sha())
        b.guides()
        b.api()
        b.examples()
        return [ConceptRef(it["concept"][:-3], it["kind"], it["url"],
                           {"kind": it["kind"]}) for it in b.items]

    def _rows(self, ref):
        kind = ref.hint.get("kind", "")
        if ref.sources:
            return [(url, role or "", kind) for url, role in ref.sources]
        return [(ref.resource, "primary", kind)]   # legacy single-resource card


class CannDocSource(Source):
    columns = ("section", "doc url")

    def __init__(self, bundle):
        self.bundle = bundle
        self._wl = self._load(os.path.join(BUILD, "%s_worklist.json" % bundle), "items")
        self._resolved = self._load(os.path.join(BUILD, "%s_resolved.json" % bundle))

    @staticmethod
    def _load(path, key=None):
        if not os.path.exists(path):
            return {} if key is None else []
        with open(path, encoding="utf-8") as data_file:
            data = json.load(data_file)
        return data[key] if key else data

    def list_concepts(self):
        base = os.path.join(REFERENCE, self.bundle)
        wl = {it["concept"]: it for it in self._wl}
        refs = []
        for rel in concept_rels(self.bundle):
            fm = read_fm(os.path.join(base, rel))
            url = self._resolved.get(rel) or fm.get("resource", "")
            refs.append(ConceptRef(rel[:-3], fm.get("type", ""), url,
                                   {"section": wl.get(rel, {}).get("section", "")}))
        return refs

    def list_upstream(self):
        """Advertised set = the .build worklist (cann is full-coverage, not sampled)."""
        return [ConceptRef(it["concept"][:-3], "", self._resolved.get(it["concept"], it.get("resource", "")),
                           {"section": it.get("section", "")}) for it in self._wl]

    def _cells(self, ref):
        return (ref.hint.get("section", ""), ref.resource)


_CANN_BUNDLES = ("ascend-c-op-dev-guide", "ascend-c-profiling")


def get_source(bundle) -> Source:
    if bundle == "asc-devkit":
        return AscDevkitSource()
    if bundle in _CANN_BUNDLES:
        return CannDocSource(bundle)
    raise CliError("unknown bundle %r (have: asc-devkit, %s)" % (bundle, ", ".join(_CANN_BUNDLES)))


def main():
    import argparse
    validate_root_argument()
    ap = argparse.ArgumentParser(description="OKF source adapter (list / drift)")
    ap.add_argument("--knowledge-root", help="OKF knowledge-base root")
    ap.add_argument("cmd", choices=["list", "drift"])
    ap.add_argument("bundle")
    ap.add_argument("--json", action="store_true", help="emit ConceptRef list as JSON")
    ap.add_argument("--upstream", action="store_true", help="list the advertised set, not the produced set")
    a = ap.parse_args()
    src = get_source(a.bundle)

    if a.cmd == "list":
        refs = src.list_upstream() if a.upstream else src.list_concepts()
        if a.json:
            payload = [{"id": ref.id, "type": ref.type, "resource": ref.resource,
                        "sources": [{"url": url, "role": role} for url, role in ref.sources],
                        "hint": ref.hint}
                       for ref in refs]
            STDOUT_LOGGER.info("%s", json.dumps(payload, ensure_ascii=False, indent=1))
        else:
            STDOUT_LOGGER.info("%s", src.render_table(refs))
        STDERR_LOGGER.info(
            "%s: %d concepts (%s)",
            a.bundle,
            len(refs),
            "upstream" if a.upstream else "produced",
        )
    elif a.cmd == "drift":
        produced = {r.id for r in src.list_concepts()}
        upstream = {r.id for r in src.list_upstream()}
        only_up = sorted(upstream - produced)
        only_pr = sorted(produced - upstream)
        STDOUT_LOGGER.info("produced: %d  upstream: %d", len(produced), len(upstream))
        STDOUT_LOGGER.info("advertised-but-not-produced: %d", len(only_up))
        for x in only_up[:20]:
            STDOUT_LOGGER.info("  + %s", x)
        if len(only_up) > 20:
            STDOUT_LOGGER.info("  … %d more", len(only_up) - 20)
        STDOUT_LOGGER.info("produced-but-not-advertised: %d", len(only_pr))
        for x in only_pr[:20]:
            STDOUT_LOGGER.info("  - %s", x)
    return 0


if __name__ == "__main__":
    try:
        _EXIT_CODE = main()
    except CliError as cli_error:
        raise SystemExit(str(cli_error)) from None
    raise SystemExit(_EXIT_CODE)
