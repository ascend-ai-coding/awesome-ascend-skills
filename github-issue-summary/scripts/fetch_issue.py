#!/usr/bin/env python3
"""
Fetch GitHub issue data for case study generation.

Usage:
    python fetch_issue.py owner/repo 123
    python fetch_issue.py owner/repo 123 --token $GITHUB_TOKEN
    python fetch_issue.py https://github.com/owner/repo/issues/123
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def parse_issue_ref(ref: str) -> tuple[str, str, int]:
    """Parse various issue reference formats."""
    # Full URL: https://github.com/owner/repo/issues/123
    url_match = re.match(r'https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)', ref)
    if url_match:
        return url_match.group(1), url_match.group(2), int(url_match.group(3))
    
    # Short format: owner/repo#123 or owner/repo 123
    short_match = re.match(r'([^/\s]+/[^#\s]+)[#\s](\d+)', ref)
    if short_match:
        owner_repo = short_match.group(1)
        owner, repo = owner_repo.split('/')
        return owner, repo, int(short_match.group(2))
    
    raise ValueError(f"Cannot parse issue reference: {ref}")


def fetch_via_gh_cli(owner: str, repo: str, number: int) -> Optional[dict]:
    """Try to fetch issue using GitHub CLI."""
    try:
        result = subprocess.run(
            ['gh', 'issue', 'view', str(number), '--repo', f'{owner}/{repo}',
             '--json', 'number,title,body,state,comments,labels,createdAt,closedAt,author,assignees'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def fetch_via_api(owner: str, repo: str, number: int, token: Optional[str] = None) -> dict:
    """Fetch issue using GitHub REST API."""
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'
    
    # Fetch issue details
    issue_url = f'https://api.github.com/repos/{owner}/{repo}/issues/{number}'
    req = Request(issue_url, headers=headers)
    
    try:
        with urlopen(req, timeout=30) as response:
            issue = json.loads(response.read().decode('utf-8'))
    except HTTPError as e:
        if e.code == 404:
            print(f"Error: Issue #{number} not found in {owner}/{repo}")
            sys.exit(1)
        elif e.code == 403:
            print("Error: Rate limit exceeded. Use --token for authenticated requests.")
            sys.exit(1)
        raise
    
    # Fetch comments
    if issue.get('comments', 0) > 0:
        comments_url = f'{issue_url}/comments'
        req = Request(comments_url, headers=headers)
        with urlopen(req, timeout=30) as response:
            issue['_fetched_comments'] = json.loads(response.read().decode('utf-8'))
    else:
        issue['_fetched_comments'] = []
    
    return issue


def normalize_issue(issue: dict) -> dict:
    """Normalize issue data from different sources."""
    # Handle comments from API vs gh CLI format
    comments = issue.get('_fetched_comments') or issue.get('comments', [])
    if isinstance(comments, list) and comments:
        if isinstance(comments[0], dict):
            if 'body' in comments[0]:
                # API format - already normalized
                pass
            elif 'comments' in issue and isinstance(issue['comments'], list):
                # gh CLI format
                comments = issue.get('comments', [])
    
    return {
        'number': issue.get('number'),
        'title': issue.get('title'),
        'body': issue.get('body', ''),
        'state': issue.get('state'),
        'author': issue.get('user', {}).get('login') if issue.get('user') else issue.get('author', {}).get('login'),
        'labels': [l.get('name') if isinstance(l, dict) else l for l in issue.get('labels', [])],
        'created_at': issue.get('createdAt') or issue.get('created_at'),
        'closed_at': issue.get('closedAt') or issue.get('closed_at'),
        'assignees': [
            a.get('login') if isinstance(a, dict) else a 
            for a in (issue.get('assignees') or [])
        ],
        'comments': comments,
        'url': issue.get('html_url') or f"https://github.com/issue/{issue.get('number')}"
    }


def calculate_duration(created: str, closed: Optional[str]) -> Optional[str]:
    """Calculate time to resolution."""
    if not created or not closed:
        return None
    
    try:
        start = datetime.fromisoformat(created.replace('Z', '+00:00'))
        end = datetime.fromisoformat(closed.replace('Z', '+00:00'))
        delta = end - start
        
        days = delta.days
        hours = delta.seconds // 3600
        
        if days > 0:
            return f"{days}d {hours}h"
        else:
            return f"{hours}h"
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub issue data')
    parser.add_argument('issue_ref', help='Issue reference (URL, owner/repo#123, or owner/repo 123)')
    parser.add_argument('number', nargs='?', type=int, help='Issue number (if not in issue_ref)')
    parser.add_argument('--token', help='GitHub personal access token')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--format', choices=['json', 'summary'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    # Parse issue reference
    try:
        if args.number:
            # Format: owner/repo number
            owner_repo = args.issue_ref.replace('#', '')
            owner, repo = owner_repo.split('/')
            number = args.number
        else:
            owner, repo, number = parse_issue_ref(args.issue_ref)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Fetching issue {owner}/{repo}#{number}...", file=sys.stderr)
    
    # Try gh CLI first, fall back to API
    issue = fetch_via_gh_cli(owner, repo, number)
    if issue:
        print("(via gh CLI)", file=sys.stderr)
    else:
        print("(via API)", file=sys.stderr)
        issue = fetch_via_api(owner, repo, number, args.token)
    
    # Normalize and enrich
    data = normalize_issue(issue)
    data['duration'] = calculate_duration(data['created_at'], data['closed_at'])
    data['owner'] = owner
    data['repo'] = repo
    
    # Output
    if args.format == 'summary':
        print(f"\nIssue #{data['number']}: {data['title']}")
        print(f"Status: {data['state']}")
        print(f"Author: @{data['author']}")
        print(f"Created: {data['created_at']}")
        if data['closed_at']:
            print(f"Closed: {data['closed_at']} ({data['duration']})")
        print(f"Labels: {', '.join(data['labels']) or 'none'}")
        print(f"Comments: {len(data['comments'])}")
    else:
        output = json.dumps(data, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Saved to {args.output}", file=sys.stderr)
        else:
            print(output)


if __name__ == '__main__':
    main()
