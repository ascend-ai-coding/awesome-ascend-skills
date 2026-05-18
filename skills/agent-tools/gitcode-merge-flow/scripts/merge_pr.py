#!/usr/bin/env python3
"""
合并 PR

参数：
1. access_token: GitCode API 访问令牌
2. upstream_repo_info: 开源仓信息，格式为 owner/repo
3. pr_number: PR 编号
4. merge_method: 合并方式，可选，默认 squash
5. commit_count: 提交数量，可选，若大于 1 则强制使用 squash

返回：
JSON 格式的合并结果
"""

import sys
import json
import requests

def get_commit_count(access_token, upstream_owner, upstream_repo, pr_number):
    url = f"https://api.gitcode.com/api/v5/repos/{upstream_owner}/{upstream_repo}/pulls/{pr_number}/commits"
    headers = {
        "Accept": "application/json",
        "private-token": access_token
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    commits = response.json()
    return len(commits)

def main():
    if len(sys.argv) < 4:
        print("Usage: python merge_pr.py <access_token> <upstream_repo_info> <pr_number> [merge_method] [commit_count]")
        sys.exit(1)
    
    access_token = sys.argv[1]
    upstream_repo_info = sys.argv[2]
    pr_number = sys.argv[3]
    merge_method = sys.argv[4] if len(sys.argv) > 4 else "squash"
    commit_count = int(sys.argv[5]) if len(sys.argv) > 5 else None
    
    upstream_owner, upstream_repo = upstream_repo_info.split('/')
    
    if commit_count is None:
        commit_count = get_commit_count(access_token, upstream_owner, upstream_repo, pr_number)
    
    if commit_count > 1:
        merge_method = "squash"
        print(f"提交数 ({commit_count}) > 1，强制使用 squash 合并")
    
    url = f"https://api.gitcode.com/api/v5/repos/{upstream_owner}/{upstream_repo}/pulls/{pr_number}/merge"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "private-token": access_token
    }
    data = {
        "merge_method": merge_method
    }
    
    response = requests.put(url, headers=headers, json=data)
    response.raise_for_status()
    
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
