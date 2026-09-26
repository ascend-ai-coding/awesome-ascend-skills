#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Immutable Git source facts for Issue response review; never mutate the repo."""

from pathlib import Path, PurePosixPath
import hashlib
import json
import posixpath
import re
import subprocess
from urllib.parse import quote, unquote, urlsplit
from call_options import bind_extra


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True)
    if result.returncode:
        raise ValueError("source_git_lookup_failed")
    return result.stdout


def _append_markdown_links(record, content, path, tracked):
    record["links"] = []
    if path.lower().endswith((".md", ".markdown")):
        # Inline and reference-style Markdown links. Unsupported syntax stays unknown.
        links = re.findall(r"!?\[[^\]\n]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", content)
        links += re.findall(r"^\s*\[[^\]\n]+\]:\s*(\S+)", content, re.M)
        for target in dict.fromkeys(links):
            url = urlsplit(target.strip("<>"))
            external_link = url.scheme or url.netloc
            if external_link or not url.path or target.startswith("#"):
                continue
            relative = unquote(url.path)
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), relative))
            if relative.startswith("/") or resolved == ".." or resolved.startswith("../"):
                status = "outside_repository"
            elif resolved in tracked or any(p.startswith(resolved.rstrip("/") + "/") for p in tracked):
                status = "tracked_target_exists"
            else:
                status = "target_missing_at_revision"
            record["links"].append({"target": target, "resolved": resolved, "status": status})


def _source_record(snapshot, requested, lines):
    root, repo, commit, tracked = snapshot
    path = PurePosixPath(requested).as_posix()
    if path.startswith("/") or ".." in PurePosixPath(path).parts or path in ("", "."):
        raise ValueError("source_path_must_be_repository_relative")
    members = sorted(p for p in tracked if p.startswith(path.rstrip("/") + "/"))
    kind = "file" if path in tracked else "directory" if members else "missing"
    if lines and kind != "file":
        raise ValueError("source_line_range_requires_file")
    record = {
        "path": path,
        "kind": kind,
        "revision": commit,
        "url": (
            f"https://gitcode.com/{repo}/{'tree' if kind == 'directory' else 'blob'}/{commit}/{quote(path, safe='/')}"
        ),
        "tracked": kind != "missing",
    }
    if kind == "file":
        data = git(root, "show", f"{commit}:{path}")
        record["blob_sha256"] = hashlib.sha256(data).hexdigest()
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            content = ""
            record["binary"] = True
        record["line_count"] = len(content.splitlines())
        start, end = lines or (1, 160)
        if lines and (record.get("binary") or start > record["line_count"]):
            raise ValueError("source_line_range_out_of_bounds")
        selected = content.splitlines()[slice(start - 1, end)]
        excerpt = "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start))
        record["excerpt"] = excerpt[:18000]
        record["excerpt_range"] = {"start": start, "end": min(end, record["line_count"])}
        record["excerpt_truncated"] = start > 1 or record["line_count"] > end or len(excerpt) > 18000
        _append_markdown_links(record, content, path, tracked)
    elif kind == "directory":
        record.update(files=members[:80], files_truncated=len(members) > 80)
    history = git(root, "log", "-5", "--format=%H%x09%an%x09%ae%x09%s", commit, "--", path).decode(
        "utf-8", errors="replace"
    )
    record["history"] = [
        dict(zip(("commit", "author_name", "author_email", "subject"), line.split("\t", 3)))
        for line in history.splitlines()
        if line
    ]
    return record


def inspect_sources(root, repo, paths, revision="HEAD", *args, **kwargs):
    reported_text, lines = bind_extra(args, kwargs, ("reported_text", "lines"), ("", None))
    root = Path(root).resolve()
    if not paths or not re.fullmatch(r"[A-Za-z0-9_./~-]+", revision) or revision.startswith("-"):
        raise ValueError("source_paths_and_valid_revision_required")
    invalid_range = lines and (len(paths) != 1 or lines[0] < 1 or lines[1] < lines[0] or lines[1] - lines[0] >= 160)
    if invalid_range:
        raise ValueError("source_single_path_and_valid_line_range_required")
    commit = git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
    tracked = {p.decode("utf-8") for p in git(root, "ls-tree", "-r", "--name-only", "-z", commit).split(b"\0") if p}
    snapshot = root, repo, commit, tracked
    records = [_source_record(snapshot, requested, lines) for requested in paths]
    # Resolve the report's named Markdown documents across the complete tree,
    # so a search under one sample cannot be mistaken for repository-wide absence.
    names = sorted(set(re.findall(r"([^/`\s\[\]()<>:]+\.md)\b", reported_text, re.I)))
    documents = []
    for name in names:
        matches = sorted(p for p in tracked if PurePosixPath(p).name == name)
        documents.append(
            {"name": name, "matches": matches[:20], "match_count": len(matches), "matches_truncated": len(matches) > 20}
        )
    return {
        "repository": repo,
        "revision": commit,
        "records": records,
        "reported_documents": documents,
        "limits": (
            "Git tracked snapshot only; not a hardware reproduction or remote URL availability test. "
            "Git author names/emails are not verified GitCode logins."
        ),
    }


def _identity_for_commit(session, cfg, token, sha, api_get):
    url = f"{cfg['gitcode_api'].rstrip('/')}/repos/{cfg['repo']}/commits/{sha}"
    try:
        data = api_get(session, url, token)
        author = data.get("author") or {}
        login = author.get("login")
        return (
            {"login": login, "identity_url": url}
            if isinstance(login, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", login)
            else {"identity_status": "no_login_in_commit_metadata"}
        )
    except Exception:
        return {"identity_status": "commit_metadata_unavailable"}


def verify_identities(evidence, cfg):
    """Map authors using GitCode commit metadata; never guess a login from email."""
    import os
    import sys

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent / "gitcode-toolkit/scripts"))
    from gitcode_client import api_get, make_session
    from runtime_paths import rate_limit_path

    token = os.environ.get("GITCODE_TOKEN")
    if not token:
        evidence["identity_status"] = "token_unavailable"
        return
    session = make_session(rate_limit_dir=rate_limit_path(cfg["cache_dir"]))
    checked = {}
    for record in evidence["records"]:
        for item in record["history"][:3]:
            sha = item["commit"]
            if sha not in checked:
                checked[sha] = _identity_for_commit(session, cfg, token, sha, api_get)
            item.update(checked[sha])
    evidence["identity_status"] = "checked"


def content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
