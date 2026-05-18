#!/usr/bin/env python3
"""
Post review comments to a GitCode Pull Request.

Supports both general comments and line-level inline comments.

Usage:
    # Post inline comments (line-level)
    python post_pr_comment.py --owner OWNER --repo REPO --pull-number NUMBER \\
        --comments-file COMMENTS_JSON [--inline]
    
    # Post a general review comment
    python post_pr_comment.py --owner OWNER --repo REPO --pull-number NUMBER \\
        --body "Review summary" [--event COMMENT|APPROVE|REQUEST_CHANGES]

Comments JSON format for inline comments:
[
    {
        "path": "src/main.py",
        "line": 42,
        "severity": "严重",
        "problem": "Describe the issue",
        "reason": "Explain why this matters",
        "fix": "Show how to fix it",
        "side": "RIGHT"
    }
]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import quote


# GitCode API 配置
API_V4_BASE = "https://api.gitcode.com/api/v4"
API_V5_BASE = "https://api.gitcode.com/api/v5"
COMMENT_SECTION_RE = re.compile(r"\*\*(严重程度|问题|原因|怎么改|应该怎么改)\s*[：:]\*\*")
SEVERITY_ALIASES = {
    "critical": "严重",
    "crit": "严重",
    "严重": "严重",
    "严重问题": "严重",
    "高": "严重",
    "improvement": "建议",
    "improvements": "建议",
    "improve": "建议",
    "suggestion": "建议",
    "suggestions": "建议",
    "建议": "建议",
    "改进": "建议",
    "优化": "建议",
    "nitpick": "提示",
    "nitpicks": "提示",
    "nit": "提示",
    "style": "提示",
    "提示": "提示",
    "细节": "提示",
    "格式": "提示",
    "样式": "提示",
}


def normalize_severity(raw: Optional[str]) -> str:
    """Normalize severity labels to the canonical Chinese review levels."""
    if raw is None:
        return "建议"
    value = str(raw).strip()
    if not value:
        return "建议"

    alias = SEVERITY_ALIASES.get(value.lower())
    if alias:
        return alias

    if value in {"严重", "建议", "提示"}:
        return value

    raise ValueError(
        f"Unsupported severity '{raw}'. Use one of: 严重, 建议, 提示. "
        "English aliases like Critical, Improvement, Nitpick are also supported."
    )


def format_comment_body(severity: str, problem: str, reason: str, fix: str) -> str:
    """Render the canonical four-section inline comment body."""
    return (
        f"**严重程度：** {normalize_severity(severity)}\n\n"
        f"**问题：** {problem.strip()}\n\n"
        f"**原因：** {reason.strip()}\n\n"
        f"**怎么改：**\n{fix.strip()}"
    )


def parse_comment_body(body: str) -> Dict[str, str]:
    """Parse a markdown comment body into structured sections."""
    matches = list(COMMENT_SECTION_RE.finditer(body))
    if not matches:
        return {}

    sections: Dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = body[start:end].strip()

        if label == "严重程度":
            key = "severity"
        elif label == "问题":
            key = "problem"
        elif label == "原因":
            key = "reason"
        else:
            key = "fix"

        sections[key] = content

    return sections


def normalize_inline_comment(comment: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Validate and normalize one inline comment entry."""
    if not isinstance(comment, dict):
        raise ValueError(f"Comment #{index} must be a JSON object.")

    path = str(comment.get("path", "")).strip()
    if not path:
        raise ValueError(f"Comment #{index} is missing a non-empty 'path'.")

    line = comment.get("line")
    if not isinstance(line, int) or line <= 0:
        raise ValueError(f"Comment #{index} has invalid 'line': {line!r}.")

    side = str(comment.get("side", "RIGHT")).upper()
    if side not in {"RIGHT", "LEFT"}:
        raise ValueError(f"Comment #{index} has invalid 'side': {side!r}.")

    body = str(comment.get("body", "")).strip()
    severity_value = comment.get("severity")
    structured = {
        "problem": comment.get("problem"),
        "reason": comment.get("reason"),
        "fix": comment.get("fix"),
    }
    has_structured_content = any(value is not None for value in structured.values())

    if has_structured_content or (severity_value is not None and not body):
        problem = str(structured["problem"] or "").strip()
        reason = str(structured["reason"] or "").strip()
        fix = str(structured["fix"] or "").strip()
        severity = normalize_severity(severity_value)

        missing = [
            name for name, value in {
                "problem": problem,
                "reason": reason,
                "fix": fix,
            }.items() if not value
        ]
        if missing:
            raise ValueError(
                f"Comment #{index} is missing structured fields: {', '.join(missing)}."
            )
    elif body:
        parsed = parse_comment_body(body)
        if not parsed:
            raise ValueError(
                f"Comment #{index} body must use the four-section format, "
                "or provide severity/problem/reason/fix fields."
            )

        severity = normalize_severity(parsed.get("severity") or comment.get("severity"))
        problem = parsed.get("problem", "").strip()
        reason = parsed.get("reason", "").strip()
        fix = parsed.get("fix", "").strip()

        missing = [
            name for name, value in {
                "problem": problem,
                "reason": reason,
                "fix": fix,
            }.items() if not value
        ]
        if missing:
            raise ValueError(
                f"Comment #{index} body is missing sections: {', '.join(missing)}."
            )
    else:
        raise ValueError(
            f"Comment #{index} must include either 'body' or structured fields "
            "severity/problem/reason/fix."
        )

    return {
        "path": path,
        "line": line,
        "body": format_comment_body(severity, problem, reason, fix),
        "side": side,
    }


def normalize_inline_comments(comments: Any) -> list[Dict[str, Any]]:
    """Normalize all inline comments before posting."""
    if not isinstance(comments, list):
        raise ValueError("Comments file must contain a JSON array.")

    return [normalize_inline_comment(comment, index) for index, comment in enumerate(comments, 1)]


def get_gitcode_token(token: Optional[str] = None) -> str:
    """Get GitCode token from argument, environment, or git config."""
    if token:
        return token
    
    # Try environment variable
    env_token = os.environ.get("GITCODE_TOKEN")
    if env_token:
        return env_token
    
    # Try git config
    try:
        result = subprocess.run(
            ["git", "config", "--global", "gitcode.token"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            return result.stdout.strip()
    except subprocess.CalledProcessError:
        pass
    
    raise ValueError(
        "GitCode token not found. Please set it via:\n"
        "1. --token argument\n"
        "2. GITCODE_TOKEN environment variable\n"
        "3. git config --global gitcode.token <token>"
    )


def get_pr_info(owner: str, repo: str, pull_number: str, token: str) -> Dict[str, Any]:
    """Get PR information including head and base SHA."""
    url = f"{API_V5_BASE}/repos/{owner}/{repo}/pulls/{pull_number}"
    headers = {
        "PRIVATE-TOKEN": token,
        "Accept": "application/json",
        "User-Agent": "gitcode-code-reviewer/1.0"
    }
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return {
                "head_sha": data.get("head", {}).get("sha", ""),
                "base_sha": data.get("base", {}).get("sha", ""),
                "start_sha": data.get("base", {}).get("sha", ""),
                "title": data.get("title", ""),
                "state": data.get("state", ""),
                "author": data.get("user", {}).get("login", "")
            }
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to get PR info: {e.code} - {e.reason}")
    except Exception as e:
        raise RuntimeError(f"Network error: {e}")


def post_inline_comment(
    owner: str,
    repo: str,
    pull_number: str,
    path: str,
    line: int,
    body: str,
    sha_info: Dict[str, str],
    token: str,
    side: str = "RIGHT"
) -> bool:
    """
    Post a line-level inline comment to the PR.
    Uses GitLab v4 API for inline comments.
    """
    project_id = f"{owner}/{repo}"
    encoded_project = quote(project_id, safe="")
    url = f"{API_V4_BASE}/projects/{encoded_project}/merge_requests/{pull_number}/discussions"
    
    headers = {
        "PRIVATE-TOKEN": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "gitcode-code-reviewer/1.0"
    }
    
    # Build position object
    position = {
        "position_type": "text",
        "base_sha": sha_info["base_sha"],
        "head_sha": sha_info["head_sha"],
        "start_sha": sha_info["start_sha"],
        "new_path": path,
        "new_line": line
    }
    
    # For deleted lines, use old_path and old_line
    if side == "LEFT":
        position["old_path"] = path
        position["old_line"] = line
        del position["new_path"]
        del position["new_line"]
    
    payload = {
        "body": body,
        "position": position
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return True
    except urllib.error.HTTPError as e:
        error_detail = e.read().decode("utf-8")
        print(f"    Error posting inline comment: {e.code} - {error_detail[:200]}")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def post_general_comment(
    owner: str,
    repo: str,
    pull_number: str,
    body: str,
    token: str
) -> bool:
    """Post a general comment to the PR using v5 API."""
    url = f"{API_V5_BASE}/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    
    headers = {
        "PRIVATE-TOKEN": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "gitcode-code-reviewer/1.0"
    }
    
    payload = {"body": body}
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return True
    except urllib.error.HTTPError as e:
        error_detail = e.read().decode("utf-8")
        print(f"    Error: {e.code} - {error_detail[:200]}")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def post_pr_review(
    owner: str,
    repo: str,
    pull_number: str,
    body: str,
    event: str,
    token: str
) -> bool:
    """Post a PR review with summary."""
    url = f"{API_V5_BASE}/repos/{owner}/{repo}/pulls/{pull_number}/reviews"
    
    headers = {
        "PRIVATE-TOKEN": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "gitcode-code-reviewer/1.0"
    }
    
    payload = {
        "body": body,
        "event": event  # COMMENT, APPROVE, REQUEST_CHANGES
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return True
    except urllib.error.HTTPError as e:
        error_detail = e.read().decode("utf-8")
        print(f"    Error: {e.code} - {error_detail[:200]}")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Post review comments to a GitCode PR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Post inline comments (line-level)
    python post_pr_comment.py --owner Ascend --repo msprof --pull-number 109 \\
        --comments-file comments.json --inline
    
    # Post a general review
    python post_pr_comment.py --owner Ascend --repo msprof --pull-number 109 \\
        --body "LGTM!" --event APPROVE
    
Comments JSON format for inline comments:
[
    {
        "path": "src/main.py",
        "line": 42,
        "severity": "严重",
        "problem": "Describe the issue",
        "reason": "Explain why this matters",
        "fix": "Show how to fix it",
        "side": "RIGHT"
    }
]
"""
    )
    
    parser.add_argument("--owner", required=True, help="Repository owner/namespace")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--pull-number", required=True, help="Pull request number")
    parser.add_argument("--token", help="GitCode access token")
    
    # Comment source options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--comments-file", help="JSON file containing inline comments")
    group.add_argument("--body", help="General comment body text")
    
    # Options
    parser.add_argument("--inline", action="store_true", 
                       help="Post as inline line-level comments (requires --comments-file)")
    parser.add_argument("--review-event", default="COMMENT",
                       choices=["COMMENT", "APPROVE", "REQUEST_CHANGES"],
                       help="Review event type for general comments")
    
    args = parser.parse_args()
    
    # Get token
    try:
        token = get_gitcode_token(args.token)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"GitCode PR Review Comment Tool")
    print(f"Repository: {args.owner}/{args.repo}")
    print(f"PR: #{args.pull_number}")
    print()
    
    # Mode: Inline comments
    if args.inline and args.comments_file:
        # Load comments
        comments_file = Path(args.comments_file)
        if not comments_file.exists():
            print(f"Error: Comments file not found: {comments_file}", file=sys.stderr)
            sys.exit(1)
        
        try:
            with open(comments_file, "r", encoding="utf-8") as f:
                comments = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in comments file: {e}", file=sys.stderr)
            sys.exit(1)
        
        try:
            comments = normalize_inline_comments(comments)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        if not comments:
            print("No comments to post.")
            sys.exit(0)
        
        # Get PR SHA info
        print("Fetching PR information...")
        try:
            sha_info = get_pr_info(args.owner, args.repo, args.pull_number, token)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        print(f"  Head SHA: {sha_info['head_sha'][:8] if sha_info['head_sha'] else 'Unknown'}")
        print(f"  Base SHA: {sha_info['base_sha'][:8] if sha_info['base_sha'] else 'Unknown'}")
        print()
        
        # Post inline comments
        print(f"Posting {len(comments)} inline comments...")
        print()
        
        success_count = 0
        failed_count = 0
        
        for i, comment in enumerate(comments, 1):
            path = comment.get("path", "")
            line = comment.get("line", 0)
            body = comment.get("body", "")
            side = comment.get("side", "RIGHT")
            
            print(f"[{i}/{len(comments)}] {path}:{line}")
            
            if post_inline_comment(
                args.owner, args.repo, args.pull_number,
                path, line, body, sha_info, token, side
            ):
                success_count += 1
            else:
                failed_count += 1
        
        print()
        print(f"Results:")
        print(f"  Posted: {success_count}")
        print(f"  Failed: {failed_count}")
        
        if failed_count > 0:
            sys.exit(1)
    
    # Mode: General comment/review
    elif args.body:
        print(f"Posting general {args.review_event} comment...")
        
        if post_pr_review(
            args.owner, args.repo, args.pull_number,
            args.body, args.review_event, token
        ):
            print("  ✓ Posted successfully")
        else:
            print("  ✗ Failed to post")
            sys.exit(1)
    
    else:
        print("Error: Invalid arguments. Use --inline with --comments-file, or --body for general comments.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
