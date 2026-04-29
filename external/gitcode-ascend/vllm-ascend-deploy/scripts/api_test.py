#!/usr/bin/env python3
"""
vLLM API 测试脚本 - Windows/Linux 通用
测试 /v1/models、/v1/completions、/v1/chat/completions 接口

用法:
  # 安装依赖
  pip install requests

  # 运行测试（修改脚本顶部的 API_BASE 和 MODEL）
  python api_test.py
"""

import requests
import json
import sys

# ============ 配置区域 ============
API_BASE = "http://localhost:8000"  # 修改为你的 API 地址
MODEL = "qwen3.5-27b"               # 修改为你的模型名称
# =================================

def print_separator(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

def test_models():
    """测试 /v1/models 接口"""
    print_separator("测试 /v1/models")
    try:
        resp = requests.get(f"{API_BASE}/v1/models", timeout=10)
        print(f"状态码: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"可用模型: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"错误: {resp.text}")
    except Exception as e:
        print(f"请求失败: {e}")

def test_completions(prompt="你好"):
    """测试 /v1/completions 接口"""
    print_separator("测试 /v1/completions")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 1
    }
    print(f"请求: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        resp = requests.post(
            f"{API_BASE}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        print(f"状态码: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            if 'choices' in data and len(data['choices']) > 0:
                print(f"\n生成文本: {data['choices'][0].get('text', '')}")
        else:
            print(f"错误: {resp.text}")
    except Exception as e:
        print(f"请求失败: {e}")

def test_chat_completions(message="你好"):
    """测试 /v1/chat/completions 接口"""
    print_separator("测试 /v1/chat/completions")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": message}
        ],
        "max_tokens": 50,
        "temperature": 1
    }
    print(f"请求: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        resp = requests.post(
            f"{API_BASE}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        print(f"状态码: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            if 'choices' in data and len(data['choices']) > 0:
                msg = data['choices'][0].get('message', {})
                print(f"\n回复: {msg.get('content', '')}")
        else:
            print(f"错误: {resp.text}")
    except Exception as e:
        print(f"请求失败: {e}")

def main():
    print(f"\nvLLM API 测试工具")
    print(f"API 地址: {API_BASE}")
    print(f"模型: {MODEL}")
    
    # 测试所有接口
    test_models()
    test_completions("你好")
    test_chat_completions("你好")
    
    print_separator("测试完成")

if __name__ == "__main__":
    main()
