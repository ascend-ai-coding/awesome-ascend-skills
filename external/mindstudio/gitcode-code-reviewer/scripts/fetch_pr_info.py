#!/usr/bin/env python3
"""
Fetch GitCode Pull Request information.

Usage:
    python fetch_pr_info.py --owner OWNER --repo REPO --pull-number NUMBER [--token TOKEN] [--output-dir DIR]
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

API_BASE = "https://api.gitcode.com/api/v5"


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


def make_api_request(url: str, token: str) -> dict:
    """Make an API request with authentication."""
    headers = {
        "PRIVATE-TOKEN": token,
        "Accept": "application/json",
        "User-Agent": "gitcode-code-reviewer/1.0"
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError("Authentication failed. Please check your GitCode token.")
        elif e.code == 404:
            raise RuntimeError(f"Resource not found: {url}")
        else:
            error_body = e.read().decode("utf-8")[:500]
            raise RuntimeError(f"API error: {e.code} - {e.reason}\n{error_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")


def fetch_pr_metadata(owner: str, repo: str, pull_number: str, token: str) -> dict:
    """Fetch PR metadata using GitCode API."""
    url = f"{API_BASE}/repos/{owner}/{repo}/pulls/{pull_number}"
    return make_api_request(url, token)


def fetch_pr_files(owner: str, repo: str, pull_number: str, token: str) -> list:
    """Fetch PR changed files using GitCode API."""
    url = f"{API_BASE}/repos/{owner}/{repo}/pulls/{pull_number}/files"
    return make_api_request(url, token)


def fetch_pr_diff(owner: str, repo: str, pull_number: str, token: str) -> str:
    """Fetch PR diff using GitCode API."""
    url = f"{API_BASE}/repos/{owner}/{repo}/pulls/{pull_number}/diff"
    
    headers = {
        "PRIVATE-TOKEN": token,
        "Accept": "application/vnd.github.v3.diff",
        "User-Agent": "gitcode-code-reviewer/1.0"
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API error fetching diff: {e.code} - {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")


def fetch_pr_comments(owner: str, repo: str, pull_number: str, token: str) -> list:
    """Fetch PR comments using GitCode API."""
    url = f"{API_BASE}/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    return make_api_request(url, token)


def main():
    parser = argparse.ArgumentParser(description="Fetch GitCode PR information")
    parser.add_argument("--owner", required=True, help="Repository owner/namespace")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--pull-number", required=True, help="Pull request number")
    parser.add_argument("--token", help="GitCode access token")
    parser.add_argument("--output-dir", default=".", help="Output directory for fetched data")
    parser.add_argument("--include-comments", action="store_true", help="Include existing PR comments")
    
    args = parser.parse_args()
    
    try:
        token = get_gitcode_token(args.token)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching PR #{args.pull_number} from {args.owner}/{args.repo}...")
    print()
    
    try:
        # Fetch PR metadata
        print("  Fetching PR metadata...")
        pr_metadata = fetch_pr_metadata(args.owner, args.repo, args.pull_number, token)
        metadata_file = output_dir / "pr_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(pr_metadata, f, indent=2, ensure_ascii=False)
        print(f"    ✓ Saved to {metadata_file}")
        
        # Fetch changed files
        print("  Fetching changed files...")
        pr_files = fetch_pr_files(args.owner, args.repo, args.pull_number, token)
        files_file = output_dir / "pr_files.json"
        with open(files_file, "w", encoding="utf-8") as f:
            json.dump(pr_files, f, indent=2, ensure_ascii=False)
        print(f"    ✓ Saved to {files_file}")
        
        # Fetch diff
        print("  Fetching PR diff...")
        try:
            pr_diff = fetch_pr_diff(args.owner, args.repo, args.pull_number, token)
            diff_file = output_dir / "pr_diff.patch"
            with open(diff_file, "w", encoding="utf-8") as f:
                f.write(pr_diff)
            print(f"    ✓ Saved to {diff_file}")
        except RuntimeError as e:
            print(f"    ⚠ Warning: Could not fetch diff - {e}")
        
        # Fetch comments if requested
        if args.include_comments:
            print("  Fetching PR comments...")
            try:
                pr_comments = fetch_pr_comments(args.owner, args.repo, args.pull_number, token)
                comments_file = output_dir / "pr_comments.json"
                with open(comments_file, "w", encoding="utf-8") as f:
                    json.dump(pr_comments, f, indent=2, ensure_ascii=False)
                print(f"    ✓ Saved to {comments_file}")
            except RuntimeError as e:
                print(f"    ⚠ Warning: Could not fetch comments - {e}")
        
        # Save summary
        summary = {
            "owner": args.owner,
            "repo": args.repo,
            "pull_number": args.pull_number,
            "pr_url": pr_metadata.get("html_url", ""),
            "title": pr_metadata.get("title", ""),
            "author": pr_metadata.get("user", {}).get("login", ""),
            "state": pr_metadata.get("state", ""),
            "files_changed": len(pr_files),
            "additions": sum(f.get("additions", 0) for f in pr_files),
            "deletions": sum(f.get("deletions", 0) for f in pr_files),
            "base_branch": pr_metadata.get("base", {}).get("ref", ""),
            "head_branch": pr_metadata.get("head", {}).get("ref", ""),
            "head_sha": pr_metadata.get("head", {}).get("sha", ""),
            "base_sha": pr_metadata.get("base", {}).get("sha", "")
        }
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print()
        print("=" * 50)
        print("PR Summary:")
        print("=" * 50)
        print(f"  Title:     {summary['title']}")
        print(f"  Author:    @{summary['author']}")
        print(f"  State:     {summary['state']}")
        print(f"  Changes:   +{summary['additions']}/-{summary['deletions']} in {summary['files_changed']} files")
        print(f"  Head SHA:  {summary['head_sha'][:8] if summary['head_sha'] else 'N/A'}")
        print(f"  Base SHA:  {summary['base_sha'][:8] if summary['base_sha'] else 'N/A'}")
        print()
        print(f"All data saved to: {output_dir.absolute()}")
        
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
