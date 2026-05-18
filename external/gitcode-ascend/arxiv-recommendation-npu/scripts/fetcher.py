"""论文抓取模块

使用 deepxiv-sdk 抓取 arxiv 论文
"""
import re
import json
import subprocess
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ARXIV_KEYWORDS, TARGET_VENUES
from utils.logger import setup_logger

logger = setup_logger("fetcher")


def run_deepxiv_cmd(args: List[str], timeout: int = 30) -> Optional[str]:
    """运行 deepxiv 命令并返回输出"""
    try:
        result = subprocess.run(
            ["deepxiv"] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout
        else:
            logger.warning(f"deepxiv 命令失败: {' '.join(args)}, 错误: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning(f"deepxiv 命令超时: {' '.join(args)}")
        return None
    except Exception as e:
        logger.warning(f"运行 deepxiv 异常: {e}")
        return None


def search_papers_deepxiv(keywords: List[str], max_results: int = 20, days: int = 3) -> List[Dict]:
    """使用 deepxiv 搜索论文

    Args:
        keywords: 搜索关键词
        max_results: 最大结果数
        days: 搜索最近几天的论文

    Returns:
        论文列表
    """
    papers = []

    # 计算日期范围
    # 注意：deepxiv 日期筛选有 bug，当 date-from 包含 04-15 时返回 0 篇
    # 临时解决方案：将 date_from 往前调一天，避免触发 bug
    date_to = datetime.now().strftime("%Y-%m-%d")
    date_from = (datetime.now() - timedelta(days=days + 1)).strftime("%Y-%m-%d")

    for keyword in keywords:
        logger.info(f"搜索关键词: {keyword}, 日期范围: {date_from} ~ {date_to}")

        # 搜索论文（带日期筛选）
        query = f'"{keyword}" recommendation'
        cmd = ["search", query, "--limit", str(max_results), "--format", "json",
               "--date-from", date_from, "--date-to", date_to]
        output = run_deepxiv_cmd(cmd)

        if not output:
            continue

        try:
            # 解析 JSON 输出（整个输出是一个 JSON 对象）
            data = json.loads(output)
            results = data.get("results", [])

            for item in results:
                arxiv_id = item.get("arxiv_id", "")
                if arxiv_id:
                    papers.append({
                        "arxiv_id": arxiv_id,
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "authors": item.get("authors", []),
                        "categories": item.get("categories", []),
                        "venue": item.get("venue", ""),
                        "published": item.get("publish_at", ""),
                    })
        except Exception as e:
            logger.warning(f"解析搜索结果失败: {e}")
            continue

    # 去重
    seen = set()
    unique_papers = []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique_papers.append(p)

    return unique_papers


def get_paper_details(arxiv_id: str) -> Optional[Dict]:
    """获取论文详细信息

    Args:
        arxiv_id: arXiv ID

    Returns:
        论文详细信息字典
    """
    # 获取简要信息
    output = run_deepxiv_cmd(["paper", arxiv_id, "--brief", "--format", "json"])
    if not output:
        return None

    try:
        data = json.loads(output)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"解析论文详情失败: {arxiv_id}, 错误: {e}")
        return None

    # 从 raw 内容中提取 GitHub URL
    raw_output = run_deepxiv_cmd(["paper", arxiv_id, "--raw"])
    if raw_output:
        # 搜索 GitHub 链接
        github_pattern = r"github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)"
        match = re.search(github_pattern, raw_output)
        if match:
            data["github_url"] = f"https://github.com/{match.group(1)}/{match.group(2)}"

    return data


def enrich_papers_with_details(papers: List[Dict]) -> List[Dict]:
    """为论文列表补充详细信息

    Args:
        papers: 论文列表

    Returns:
        补充详情后的论文列表
    """
    enriched = []

    for paper in papers:
        arxiv_id = paper.get("arxiv_id", "")
        if not arxiv_id:
            continue

        details = get_paper_details(arxiv_id)
        if not details:
            logger.warning(f"无法获取论文详情: {arxiv_id}")
            continue

        # 合并信息
        paper.update(details)
        paper["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}"

        enriched.append(paper)
        logger.info(f"获取论文详情成功: {paper.get('title', arxiv_id)[:50]}...")

    return enriched


def search_papers(days: int = 3, max_results: int = 50) -> List[Dict]:
    """搜索论文

    Args:
        days: 搜索最近几天的论文（默认 3 天）
        max_results: 每个关键词最大结果数

    Returns:
        论文列表，每篇论文包含:
        - title: 标题
        - abstract: 摘要
        - authors: 作者
        - published: 发表日期
        - categories: 分类
        - comments: 备注（可能有 GitHub 链接）
        - pdf_url: PDF 链接
    """
    logger.info(f"开始搜索最近 {days} 天的推荐相关论文...")

    # 使用 deepxiv 搜索论文（带日期范围）
    papers = search_papers_deepxiv(ARXIV_KEYWORDS, max_results, days)
    logger.info(f"初步搜索到 {len(papers)} 篇论文")

    # 补充详细信息（包含 GitHub URL）
    papers = enrich_papers_with_details(papers)
    logger.info(f"补充详情后得到 {len(papers)} 篇论文")

    # 筛选推荐相关论文
    filtered = filter_recommendation_papers(papers)

    logger.info(f"找到 {len(filtered)} 篇推荐相关论文")
    return filtered


def filter_recommendation_papers(papers: List[Dict]) -> List[Dict]:
    """筛选推荐相关论文"""
    filtered = []

    for paper in papers:
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        comments = paper.get("comments", "").lower()

        text = f"{title} {abstract} {comments}"

        # 关键词匹配
        matched = any(kw.lower() in text for kw in ARXIV_KEYWORDS)

        if matched:
            filtered.append(paper)

    return filtered


def extract_venue(paper: Dict) -> Optional[str]:
    """从论文中提取发表场所"""
    comments = paper.get("comments", "")

    # 常见会议模式
    patterns = [
        r"(RecSys\s+\d{4})",
        r"(SIGIR\s+\d{4})",
        r"(KDD\s+\d{4})",
        r"(WWW\s+\d{4})",
        r"(ICML\s+\d{4})",
        r"(NeurIPS\s+\d{4})",
        r"(TOIS)",
    ]

    for pattern in patterns:
        match = re.search(pattern, comments, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def save_paper_list(papers: List[Dict], output_path: str):
    """保存论文列表到文件"""
    lines = ["# 今日推荐相关论文\n"]
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"总计: {len(papers)} 篇\n\n---\n\n")

    for i, paper in enumerate(papers, 1):
        lines.append(f"## {i}. {paper.get('title', 'Unknown')}\n")
        lines.append(f"- 作者: {', '.join(paper.get('authors', [])[:3])}...\n")
        lines.append(f"- 发表: {paper.get('published', 'Unknown')}\n")
        venue = extract_venue(paper)
        if venue:
            lines.append(f"- 会议: {venue}\n")
        lines.append(f"- 分类: {', '.join(paper.get('categories', []))}\n")
        lines.append(f"- 摘要: {paper.get('abstract', '')[:200]}...\n")
        lines.append(f"- PDF: {paper.get('pdf_url', '')}\n")

        if paper.get("github_url"):
            lines.append(f"- **源码: {paper['github_url']}**\n")

        lines.append("\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"论文列表已保存到: {output_path}")