#!/usr/bin/env python3
"""
创建 Issue 在开源仓中

参数：
1. access_token: GitCode API 访问令牌
2. upstream_repo_info: 开源仓信息，格式为 owner/repo
3. issue_title: Issue 标题
4. issue_body: Issue 描述

返回：
JSON 格式的 Issue 信息
"""

import sys
import json
import requests

def main():
    if len(sys.argv) < 5:
        print("Usage: python create_issue.py <access_token> <upstream_repo_info> <issue_title> <issue_body>")
        sys.exit(1)
    
    access_token = sys.argv[1]
    upstream_repo_info = sys.argv[2]
    issue_title = sys.argv[3]
    issue_body = sys.argv[4]
    
    # 解析仓库信息
    upstream_owner, upstream_repo = upstream_repo_info.split('/')
    
    # 构建请求
    url = f"https://api.gitcode.com/api/v5/repos/{upstream_owner}/issues"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "private-token": access_token
    }
    data = {
        "repo": upstream_repo,
        "title": issue_title,
        "body": issue_body
    }
    
    # 发送请求
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    # 打印结果
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
