#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
from typing import Any


KEYWORD_WEIGHTS = {
    "train_error": {
        "verl": ["trainer", "worker", "ray", "rollout", "reward", "engine", "actor"],
        "mindspeed": ["mindspeed", "megatron", "parallel", "distributed", "hccl", "npu", "ascend"],
    },
    "accuracy_regression": {
        "verl": ["reward", "loss", "train", "rollout", "sampling", "generation", "dataset"],
        "mindspeed": ["optimizer", "precision", "parallel", "megatron", "mindspeed", "amp", "fp16", "bf16"],
    },
}


def normalize_text(parts: list[str]) -> str:
    return " ".join(parts).lower()


def tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9_./-]+", text.lower()) if token}


def score_commit(commit: dict[str, Any], repo_name: str, issue_type: str, evidence_text: str) -> dict[str, Any]:
    subject = commit.get("subject", "")
    files = commit.get("files", [])
    haystack = normalize_text([subject] + files)
    tokens = tokenize(haystack)
    score = 0
    reasons: list[str] = []

    for keyword in KEYWORD_WEIGHTS.get(issue_type, {}).get(repo_name, []):
        if keyword in tokens:
            score += 3
            reasons.append(f"matched repo keyword '{keyword}'")

    evidence_tokens = {token for token in tokenize(evidence_text) if len(token) >= 4}
    for token in evidence_tokens:
        if token in tokens:
            score += 2
            reasons.append(f"matched evidence token '{token}'")

    if "fix" in subject.lower() or "refactor" in subject.lower():
        score += 1
        reasons.append("subject indicates behavior-changing maintenance")

    return {
        **commit,
        "repo": repo_name,
        "score": score,
        "reasons": reasons,
    }


def rank_candidates(payload: dict[str, Any], top_k: int) -> dict[str, Any]:
    issue_type = payload.get("issue_type", "unknown")
    evidence_text = payload.get("evidence_text", "")
    ranked: list[dict[str, Any]] = []

    for repo_name, commits in payload.get("repos", {}).items():
        for commit in commits:
            ranked.append(score_commit(commit, repo_name, issue_type, evidence_text))

    ranked.sort(key=lambda item: (item["score"], item["authored_at"]), reverse=True)
    return {
        "issue_type": issue_type,
        "evidence_text": evidence_text,
        "top_candidates": ranked[:top_k],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank recent commits against regression evidence.")
    parser.add_argument("payload", help="Path to a JSON payload with issue_type, evidence_text, and repo commit lists.")
    parser.add_argument("--top-k", type=int, default=6, help="Number of candidates to keep.")
    args = parser.parse_args()

    payload = json.loads(pathlib.Path(args.payload).read_text(encoding="utf-8"))
    print(json.dumps(rank_candidates(payload, args.top_k), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
