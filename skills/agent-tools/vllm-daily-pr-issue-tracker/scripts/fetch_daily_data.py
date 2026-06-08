#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch today's PRs and Issues from vllm-project/vllm and vllm-project/vllm-ascend."""

import json
import os
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

MODEL_KEYWORDS = {
    "DeepSeek": [
        "deepseek", "deepseek-r1", "deepseek-v3", "deepseek-v2",
        "deepseek-mtp", "deepseek_r1", "deepseek_v3", "mtp",
    ],
    "Qwen": [
        "qwen", "qwen3", "qwen2", "qwen-vl", "qwen-coder",
        "qwen3-235b", "qwen3-30b", "qwen_vl", "qwenvl",
        "qwen-dense", "qwen3moe", "qwen2.5",
    ],
    "GLM": [
        "glm", "glm5", "glm-5", "glm4", "glm-4",
        "chatglm", "glm5.0", "glm-z1",
    ],
    "MiniMax": [
        "minimax", "minimax2", "minimax-2.5", "minimax2.5",
        "minimax-text", "abab",
    ],
    "Kimi/Moonshot": [
        "kimi", "moonshot", "moonshot-v1", "kimi-vl",
    ],
}

TECH_KEYWORDS = {
    "PD分离": [
        "prefill", "decode", "disaggregat", "pd disaggregat",
        "prefill-decode", "prefill decode", "kv transfer",
        "kv migration", "disagg", "p/d", "pd split",
    ],
    "MTP/投机解码": [
        "speculative", "spec decode", "spec_decode", "draft model",
        "mtp", "multi-token prediction", "multi_token_prediction",
        "eagle", "medusa", "lookahead", "speculative decoding",
    ],
    "量化": [
        "quantiz", "quant", "w8a8", "fp8", "int8", "int4",
        "awq", "gptq", "gguf", "bitsandbytes", "bnb",
        "ascend quantiz", "kv quant", "kv cache quant",
        "weight quant", "activation quant", "smoothquant",
    ],
    "图模式/TorchAir": [
        "torchair", "torch_air", "graph mode", "graph_mode",
        "torch.compile", "inductor", "dynamo", "aot autograd",
        "cuda graph", "cudagraph", "capture graph",
    ],
    "模型Bug Fix": [
        "bug fix", "bugfix", "fix bug", "incorrect output",
        "wrong result", "numerical error", "nan", "inf",
        "crash", "oom", "out of memory", "segfault",
        "regression", "broken", "fix model",
    ],
    "性能优化": [
        "performance", "throughput", "latency", "speedup",
        "speed up", "optimize", "optimiz", "efficient",
        "kv cache", "attention", "flash attention", "flashattn",
        "expert parallel", "tensor parallel", "pipeline parallel",
        "paged attention", "chunked prefill", "continuous batching",
        "prefix caching", "radix", "memory", "bandwidth",
    ],
    "模型新特性": [
        "support", "add support", "new model", "new feature",
        "enable", "implement", "introduce", "add model",
    ],
}


def get_github_token():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("[警告] 未设置 GITHUB_TOKEN 环境变量，API 请求将受到速率限制（每小时 60 次）")
    return token


def github_api_request(url, token=""):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "vllm-daily-tracker/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"[错误] HTTP {e.code}: {url}")
        if e.code == 403:
            print("  -> 可能是 API 速率限制，请设置 GITHUB_TOKEN")
        elif e.code == 401:
            print("  -> GITHUB_TOKEN 无效或已过期")
        return None
    except Exception as e:
        print(f"[错误] 请求失败: {e}")
        return None


def fetch_all_pages(base_url, token="", max_pages=10):
    all_items = []
    page = 1
    while page <= max_pages:
        url = f"{base_url}&page={page}&per_page=100"
        data = github_api_request(url, token)
        if not data:
            break
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        elif isinstance(data, list):
            items = data
        else:
            break
        if not items:
            break
        all_items.extend(items)
        if len(items) < 100:
            break
        page += 1
    return all_items


def get_today_range():
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    today_start = datetime(
        now_utc.year, now_utc.month, now_utc.day, 0, 0, 0, tzinfo=timezone.utc
    )
    return today_str, today_start


def fetch_prs(repo, token, since_iso):
    print(f"  正在获取 {repo} 的 PR...")

    merged_url = (
        f"https://api.github.com/search/issues"
        f"?q=repo:{repo}+is:pr+is:merged+merged:>={since_iso}"
        f"&sort=updated&order=desc"
    )
    merged_prs = fetch_all_pages(merged_url, token)

    opened_url = (
        f"https://api.github.com/search/issues"
        f"?q=repo:{repo}+is:pr+is:open+created:>={since_iso}"
        f"&sort=updated&order=desc"
    )
    opened_prs = fetch_all_pages(opened_url, token)

    seen_ids = set()
    all_prs = []
    for pr in merged_prs + opened_prs:
        if pr["id"] not in seen_ids:
            seen_ids.add(pr["id"])
            pr["_type"] = "PR"
            pr["_repo"] = repo
            pr["_status"] = (
                "merged"
                if pr.get("pull_request", {}).get("merged_at")
                else ("open" if pr.get("state") == "open" else "closed")
            )
            all_prs.append(pr)

    print(f"    -> 获取到 {len(all_prs)} 个 PR（merged: {len(merged_prs)}, opened: {len(opened_prs)}）")
    return all_prs


def fetch_issues(repo, token, since_iso):
    print(f"  正在获取 {repo} 的 Issue...")

    opened_url = (
        f"https://api.github.com/search/issues"
        f"?q=repo:{repo}+is:issue+created:>={since_iso}"
        f"&sort=updated&order=desc"
    )
    opened_issues = fetch_all_pages(opened_url, token)

    updated_url = (
        f"https://api.github.com/search/issues"
        f"?q=repo:{repo}+is:issue+updated:>={since_iso}"
        f"&sort=updated&order=desc"
    )
    updated_issues = fetch_all_pages(updated_url, token)

    seen_ids = set()
    all_issues = []
    for issue in opened_issues + updated_issues:
        if issue["id"] not in seen_ids:
            seen_ids.add(issue["id"])
            issue["_type"] = "Issue"
            issue["_repo"] = repo
            issue["_status"] = issue.get("state", "open")
            all_issues.append(issue)

    print(
        f"    -> 获取到 {len(all_issues)} 个 Issue"
        f"（opened: {len(opened_issues)}, updated: {len(updated_issues)}）"
    )
    return all_issues


def normalize_text(text):
    return (text or "").lower()


def match_keywords(item, keyword_dict):
    title = normalize_text(item.get("title", ""))
    body = normalize_text(item.get("body", "") or "")
    labels = " ".join(normalize_text(l.get("name", "")) for l in item.get("labels", []))
    full_text = f"{title} {body} {labels}"

    matched = []
    for category, keywords in keyword_dict.items():
        for kw in keywords:
            if kw.lower() in full_text:
                matched.append(category)
                break
    return matched


def classify_item(item):
    model_cats = match_keywords(item, MODEL_KEYWORDS)
    tech_cats = match_keywords(item, TECH_KEYWORDS)
    return model_cats, tech_cats


def is_relevant(model_cats, tech_cats):
    return bool(model_cats or tech_cats)


def assign_priority(model_cats, tech_cats, item):
    core_tech = {"PD分离", "MTP/投机解码", "量化", "图模式/TorchAir"}

    if len(model_cats) >= 2:
        return "P0"
    if model_cats and "模型Bug Fix" in tech_cats:
        return "P0"
    if model_cats and (core_tech & set(tech_cats)):
        return "P1"
    if model_cats and tech_cats:
        return "P1"
    if core_tech & set(tech_cats):
        return "P2"
    if tech_cats:
        return "P2"
    return "P3"


def main():
    print("=" * 60)
    print("vllm Daily PR & Issue Tracker")
    print("=" * 60)

    token = get_github_token()
    today_str, today_start = get_today_range()
    since_iso = today_start.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\n[INFO] 今日日期: {today_str}")
    print(f"[INFO] 过滤时间: >= {since_iso} (UTC)")
    print()

    repos = ["vllm-project/vllm", "vllm-project/vllm-ascend"]
    all_items = []

    for repo in repos:
        print(f"[{repo}]")
        prs = fetch_prs(repo, token, today_str)
        issues = fetch_issues(repo, token, today_str)
        all_items.extend(prs)
        all_items.extend(issues)
        print()

    print(f"[INFO] 共获取 {len(all_items)} 条原始数据")

    relevant_items = []
    for item in all_items:
        model_cats, tech_cats = classify_item(item)
        if is_relevant(model_cats, tech_cats):
            item["_model_cats"] = model_cats
            item["_tech_cats"] = tech_cats
            item["_priority"] = assign_priority(model_cats, tech_cats, item)
            relevant_items.append(item)

    print(f"[INFO] 筛选后相关条目: {len(relevant_items)} 条")

    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    relevant_items.sort(key=lambda x: (priority_order.get(x["_priority"], 9), x["_repo"]))

    output_data = {
        "date": today_str,
        "total_fetched": len(all_items),
        "total_relevant": len(relevant_items),
        "items": [],
    }

    for item in relevant_items:
        output_data["items"].append({
            "repo": item["_repo"],
            "type": item["_type"],
            "number": item.get("number"),
            "title": item.get("title", ""),
            "url": item.get("html_url", ""),
            "status": item["_status"],
            "author": item.get("user", {}).get("login", "unknown"),
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
            "body_preview": (item.get("body") or "")[:500],
            "labels": [l.get("name", "") for l in item.get("labels", [])],
            "model_categories": item["_model_cats"],
            "tech_categories": item["_tech_cats"],
            "priority": item["_priority"],
        })

    output_dir = os.path.join(os.getcwd(), "daily-reports")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"daily-data-{today_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 数据已保存到: {json_path}")
    print("[下一步] 请继续执行 Step 2，由 AI 对数据进行深度分析并生成报告")

    print("\n" + "=" * 60)
    print("数据摘要（供 AI 分析参考）")
    print("=" * 60)

    priority_counter = Counter(item["_priority"] for item in relevant_items)
    print(f"\n优先级分布: {dict(priority_counter)}")

    repo_counter = Counter(item["_repo"] for item in relevant_items)
    print(f"仓库分布: {dict(repo_counter)}")

    model_counter = Counter()
    for item in relevant_items:
        for cat in item["_model_cats"]:
            model_counter[cat] += 1
    print(f"模型分类: {dict(model_counter)}")

    tech_counter = Counter()
    for item in relevant_items:
        for cat in item["_tech_cats"]:
            tech_counter[cat] += 1
    print(f"技术方向: {dict(tech_counter)}")

    print("\n各条目列表:")
    print("-" * 60)
    for item in relevant_items:
        print(
            f"[{item['_priority']}] [{item['_repo'].split('/')[-1]}] "
            f"#{item.get('number')} {item['_type']} | "
            f"{item.get('title', '')[:60]}"
        )
        print(f"       模型: {item['_model_cats']} | 技术: {item['_tech_cats']}")
        print(f"       链接: {item.get('html_url', '')}")
        print()

    return json_path, output_data


if __name__ == "__main__":
    main()
