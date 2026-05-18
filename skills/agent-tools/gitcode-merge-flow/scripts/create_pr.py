#!/usr/bin/env python3
"""
创建 PR 从个人仓到开源仓

参数：
1. access_token: GitCode API 访问令牌
2. personal_repo_info: 个人仓信息，格式为 owner/repo
3. upstream_repo_info: 开源仓信息，格式为 owner/repo
4. branch_name: 源分支名称
5. target_branch: 目标分支名称
6. pr_title: PR 标题
7. pr_body: PR 描述
8. close_related_issue: 是否关闭关联 Issue，可选，默认 true

返回：
JSON 格式的 PR 信息
"""

import sys
import json
import requests

def main():
    if len(sys.argv) < 8:
        print("Usage: python create_pr.py <access_token> <personal_repo_info> <upstream_repo_info> <branch_name> <target_branch> <pr_title> <pr_body> [close_related_issue]")
        sys.exit(1)
    
    access_token = sys.argv[1]
    personal_repo_info = sys.argv[2]
    upstream_repo_info = sys.argv[3]
    branch_name = sys.argv[4]
    target_branch = sys.argv[5]
    pr_title = sys.argv[6]
    pr_body = sys.argv[7]
    close_related_issue = sys.argv[8] if len(sys.argv) > 8 else "true"
    close_related_issue = close_related_issue.lower() == "true"
    
    # 解析仓库信息
    personal_owner, personal_repo = personal_repo_info.split('/')
    upstream_owner, upstream_repo = upstream_repo_info.split('/')
    
    # 构建请求
    url = f"https://api.gitcode.com/api/v5/repos/{upstream_owner}/{upstream_repo}/pulls"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "private-token": access_token
    }
    data = {
        "title": pr_title,
        "head": f"{personal_owner}:{branch_name}",
        "base": target_branch,
        "body": pr_body,
        "fork_path": f"{personal_owner}/{personal_repo}",
        "close_related_issue": close_related_issue
    }
    
    # 发送请求
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    # 打印结果
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
