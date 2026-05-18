#!/usr/bin/env python3
"""
获取 PR 评论

参数：
1. access_token: GitCode API 访问令牌
2. upstream_repo_info: 开源仓信息，格式为 owner/repo
3. pr_number: PR 编号
4. comment_type: 评论类型，可选，diff_comment 或 pr_comment
5. page: 页码，可选，默认 1
6. per_page: 每页数量，可选，默认 100

返回：
JSON 格式的评论列表
"""

import sys
import json
import requests

def main():
    if len(sys.argv) < 4:
        print("Usage: python get_pr_comments.py <access_token> <upstream_repo_info> <pr_number> [comment_type] [page] [per_page]")
        sys.exit(1)
    
    access_token = sys.argv[1]
    upstream_repo_info = sys.argv[2]
    pr_number = sys.argv[3]
    comment_type = sys.argv[4] if len(sys.argv) > 4 else ""
    page = sys.argv[5] if len(sys.argv) > 5 else "1"
    per_page = sys.argv[6] if len(sys.argv) > 6 else "100"
    
    # 解析仓库信息
    upstream_owner, upstream_repo = upstream_repo_info.split('/')
    
    # 构建查询参数
    params = {
        "page": page,
        "per_page": per_page
    }
    if comment_type:
        params["comment_type"] = comment_type
    
    # 构建请求
    url = f"https://api.gitcode.com/api/v5/repos/{upstream_owner}/{upstream_repo}/pulls/{pr_number}/comments"
    headers = {
        "Accept": "application/json",
        "private-token": access_token
    }
    
    # 发送请求
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    # 打印结果
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
