"""源码检测模块

从论文中提取 GitHub 链接，并验证代码仓是否有可执行代码
"""
import re
import os
import subprocess
from typing import List, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MIN_PY_FILES, GITHUB_MIRROR, MAX_RETRIES
from utils.logger import setup_logger

logger = setup_logger("source_detector")


# GitHub 正则匹配
GITHUB_PATTERNS = [
    r"github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)",
    r"https?://github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+?)(?:\.git)?",
]


def extract_github_url(text: str) -> Optional[str]:
    """从文本中提取 GitHub 链接"""
    for pattern in GITHUB_PATTERNS:
        match = re.search(pattern, text)
        if match:
            owner, repo = match.groups()[:2]
            return f"https://github.com/{owner}/{repo}"
    return None


def extract_github_from_paper(paper: Dict) -> Optional[str]:
    """从论文中提取 GitHub 链接"""
    # 优先检查 github_url 字段（enrich_papers_with_details 中提取的）
    github_url = paper.get("github_url", "")
    if github_url:
        logger.info(f"从 github_url 字段找到 GitHub: {github_url}")
        return github_url

    # 检查 comments 字段
    comments = paper.get("comments", "")
    if comments:
        url = extract_github_url(comments)
        if url:
            logger.info(f"从 comments 中找到 GitHub: {url}")
            return url

    # 检查 abstract
    abstract = paper.get("abstract", "")
    if abstract:
        url = extract_github_url(abstract)
        if url:
            logger.info(f"从 abstract 中找到 GitHub: {url}")
            return url

    return None


def check_repo_has_code(github_url: str, timeout: int = 10) -> bool:
    """检查代码仓是否有可执行代码（递归检查子目录）

    Args:
        github_url: GitHub 仓库地址
        timeout: 超时时间（秒）

    Returns:
        True 如果有 .py 文件，否则 False
    """
    try:
        # 提取 owner 和 repo
        match = re.search(r"github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)", github_url)
        if not match:
            return False

        owner, repo = match.groups()
        repo = repo.replace(".git", "")

        # 递归统计 .py 文件
        py_count = count_py_files_recursive(owner, repo, timeout)

        if py_count >= MIN_PY_FILES:
            logger.info(f"仓库 {github_url} 包含 {py_count} 个 .py 文件")
            return True
        else:
            logger.info(f"仓库 {github_url} 只有 {py_count} 个 .py 文件，不足 {MIN_PY_FILES}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning(f"GitHub API 请求超时: {github_url}")
        return False
    except Exception as e:
        logger.warning(f"检查仓库代码失败: {github_url}, 错误: {e}")
        return False


def count_py_files_recursive(owner: str, repo: str, timeout: int = 10, max_depth: int = 3) -> int:
    """递归统计仓库中的 .py 文件数量

    Args:
        owner: GitHub 用户名
        repo: 仓库名
        timeout: 超时时间
        max_depth: 最大递归深度

    Returns:
        .py 文件总数
    """
    import json

    def recursive_count(path: str, depth: int) -> int:
        if depth > max_depth:
            return 0

        try:
            cmd = [
                "curl", "-s", "-L",
                f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
                "-H", "Accept: application/vnd.github.v3+json"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

            if result.returncode != 0:
                return 0

            contents = json.loads(result.stdout)
            if not isinstance(contents, list):
                return 0

            count = 0
            for item in contents:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                item_type = item.get("type", "")

                if name.endswith(".py"):
                    count += 1
                elif item_type == "dir" and name not in [".git", "test", "tests", "docs", "examples"]:
                    # 递归检查子目录（跳过常见非代码目录）
                    count += recursive_count(f"{path}/{name}".strip("/"), depth + 1)

            return count
        except Exception:
            return 0

    return recursive_count("", 0)


def filter_papers_with_code(papers: List[Dict]) -> List[Dict]:
    """筛选出有源码的论文"""
    result = []

    for paper in papers:
        github_url = extract_github_from_paper(paper)

        if not github_url:
            logger.info(f"论文《{paper.get('title', 'Unknown')}》未提供源码链接")
            continue

        # 检查代码仓是否有可执行代码
        if check_repo_has_code(github_url):
            paper["github_url"] = github_url
            result.append(paper)
            logger.info(f"论文《{paper.get('title', 'Unknown')}》有可用源码: {github_url}")
        else:
            logger.info(f"论文《{paper.get('title', 'Unknown')}》源码仓库无可执行代码: {github_url}")

    return result


def convert_to_mirror_url(github_url: str) -> str:
    """将 GitHub URL 转换为镜像 URL

    Args:
        github_url: 原始 GitHub URL

    Returns:
        镜像 URL
    """
    if not GITHUB_MIRROR.get("enabled", False):
        return github_url

    # 提取 owner 和 repo
    match = re.search(r"github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)", github_url)
    if not match:
        return github_url

    owner, repo = match.groups()
    repo = repo.replace(".git", "")

    # 使用模板替换
    template = GITHUB_MIRROR.get("url_template", "https://ghproxy.com/https://github.com/{owner}/{repo}")
    mirror_url = template.format(owner=owner, repo=repo)

    logger.info(f"使用镜像: {mirror_url}")
    return mirror_url


def clone_repo(github_url: str, target_dir: str, timeout: int = 120) -> Dict:
    """克隆代码仓库

    Args:
        github_url: GitHub 仓库地址
        target_dir: 目标目录
        timeout: 超时时间（秒）

    Returns:
        Dict: {
            "success": bool,  # 是否成功
            "message": str,   # 状态信息
            "needs_manual": bool  # 是否需要用户手动处理
        }
    """
    import time

    result = {
        "success": False,
        "message": "",
        "needs_manual": False,
    }

    os.makedirs(target_dir, exist_ok=True)

    # 去除 .git 后缀
    repo_url = github_url.rstrip(".git")

    # 提取 owner/repo 用于错误信息
    match = re.search(r"github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)", repo_url)
    if match:
        repo_name = f"{match.group(1)}/{match.group(2).replace('.git', '')}"
    else:
        repo_name = repo_url

    # 策略1: 直接克隆（优先）
    logger.info(f"尝试克隆仓库: {repo_url} -> {target_dir}")
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"尝试 {attempt}/{MAX_RETRIES}: 直接克隆")

        try:
            proc_result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, target_dir],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if proc_result.returncode == 0:
                logger.info(f"仓库克隆成功: {target_dir}")
                result["success"] = True
                result["message"] = "克隆成功"
                return result
            else:
                logger.warning(f"尝试 {attempt} 失败: {proc_result.stderr[:100]}")

        except subprocess.TimeoutExpired:
            logger.warning(f"尝试 {attempt} 超时")
        except Exception as e:
            logger.warning(f"尝试 {attempt} 异常: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(3)  # 重试间隔

    # 策略2: 尝试镜像
    if GITHUB_MIRROR.get("enabled", False):
        mirror_url = convert_to_mirror_url(repo_url)
        logger.info(f"直接克隆失败，尝试镜像: {mirror_url}")

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(f"镜像尝试 {attempt}/{MAX_RETRIES}")

            try:
                proc_result = subprocess.run(
                    ["git", "clone", "--depth", "1", mirror_url, target_dir],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if proc_result.returncode == 0:
                    logger.info(f"镜像克隆成功: {target_dir}")
                    result["success"] = True
                    result["message"] = "通过镜像克隆成功"
                    return result
                else:
                    logger.warning(f"镜像尝试 {attempt} 失败")

            except subprocess.TimeoutExpired:
                logger.warning(f"镜像尝试 {attempt} 超时")
            except Exception as e:
                logger.warning(f"镜像尝试 {attempt} 异常: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(3)

    # 所有尝试都失败
    logger.error(f"仓库克隆最终失败: {repo_url}")
    result["success"] = False
    result["message"] = f"GitHub 访问不稳定，尝试多次仍无法连接。请手动下载: {repo_url}"
    result["needs_manual"] = True

    return result


def find_entry_script(model_dir: str) -> str:
    """查找模型目录中的入口脚本

    优先级: train.py -> main.py -> run.py -> 其他 .py
    """
    # 常见入口脚本名称
    priority_names = ["train.py", "main.py", "run.py", "evaluate.py", "demo.py", "trainer.py"]

    # 先检查优先级名称
    for name in priority_names:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return name

    # 查找所有 .py 文件
    py_files = []
    for root, dirs, files in os.walk(model_dir):
        # 跳过隐藏目录和常见无关目录
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "venv", "env", "data", "logs", "outputs", "checkpoints")]

        for f in files:
            if f.endswith(".py") and not f.startswith("test_") and not f.startswith("setup"):
                rel_path = os.path.relpath(os.path.join(root, f), model_dir)
                py_files.append(rel_path)

    if not py_files:
        return ""

    # 返回第一个找到的 py 文件
    return py_files[0]