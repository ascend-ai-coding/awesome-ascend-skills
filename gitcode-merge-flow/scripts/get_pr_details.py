#!/usr/bin/env python3
"""
获取 PR 详情

参数：
1. access_token: GitCode API 访问令牌
2. upstream_repo_info: 开源仓信息，格式为 owner/repo
3. pr_number: PR 编号

返回：
JSON 格式的 PR 详情
"""

import sys
import json
import requests

def main():
    if len(sys.argv) < 4:
        print("Usage: python get_pr_details.py <access_token> <upstream_repo_info> <pr_number>")
        sys.exit(1)
    
    access_token = sys.argv[1]
    upstream_repo_info = sys.argv[2]
    pr_number = sys.argv[3]
    
    # 解析仓库信息
    upstream_owner, upstream_repo = upstream_repo_info.split('/')
    
    # 构建请求
    url = f"https://api.gitcode.com/api/v5/repos/{upstream_owner}/{upstream_repo}/pulls/{pr_number}"
    headers = {
        "Accept": "application/json",
        "private-token": access_token
    }
    
    # 发送请求
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # 打印结果
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
