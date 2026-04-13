#!/usr/bin/env python3
"""
检查流水线状态（轮询）

参数：
1. access_token: GitCode API 访问令牌
2. upstream_repo_info: 开源仓信息，格式为 owner/repo
3. pr_number: PR 编号
4. max_attempts: 最大尝试次数，可选，默认 20
5. interval_seconds: 轮询间隔，可选，默认 30

返回：
成功返回 0，失败返回 1
"""

import sys
import time
import subprocess
import json

def main():
    if len(sys.argv) < 4:
        print("Usage: python check_pipeline_status.py <access_token> <upstream_repo_info> <pr_number> [max_attempts] [interval_seconds]")
        sys.exit(1)
    
    access_token = sys.argv[1]
    upstream_repo_info = sys.argv[2]
    pr_number = sys.argv[3]
    max_attempts = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    interval_seconds = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    
    # 轮询检查
    for i in range(1, max_attempts + 1):
        print(f"Checking pipeline status (attempt {i}/{max_attempts})...")
        
        # 获取 PR 详情
        script_path = f"{subprocess.os.path.dirname(__file__)}/get_pr_details.py"
        result = subprocess.run(
            ["python", script_path, access_token, upstream_repo_info, pr_number],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error getting PR details: {result.stderr}")
            time.sleep(interval_seconds)
            continue
        
        try:
            pr_details = json.loads(result.stdout)
            # 检查状态（这里根据实际 API 返回字段调整）
            if pr_details.get("mergeable"):
                print("Pipeline succeeded!")
                sys.exit(0)
        except json.JSONDecodeError:
            print(f"Error parsing PR details: {result.stdout}")
        
        # 等待
        time.sleep(interval_seconds)
    
    print(f"Pipeline check timed out after {max_attempts} attempts")
    sys.exit(1)

if __name__ == "__main__":
    main()