"""arxiv-recommendation-npu 主入口

自动化推荐论文发现与 NPU 适配流水线
"""
import os
import sys
import json
import argparse
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TODAY_DIR
from utils.logger import setup_logger
from fetcher import search_papers, save_paper_list
from source_detector import filter_papers_with_code, clone_repo, find_entry_script

logger = setup_logger("main", os.path.join(TODAY_DIR, "main.log"))

# 默认搜索最近3天的论文
DEFAULT_DAYS = 3


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="arxiv 推荐论文发现流水线")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"搜索最近几天的论文 (默认: {DEFAULT_DAYS} 天)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="每个关键词最大结果数 (默认: 50)"
    )
    return parser.parse_args()


def create_migration_task(model_dir: str, paper: dict, github_url: str) -> dict:
    """创建待迁移任务清单

    这个清单会交给 npu-model-migration skill 进行处理
    """
    # 查找入口脚本
    entry_script = find_entry_script(model_dir)

    task = {
        "task_id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "created_at": datetime.now().isoformat(),
        "paper": {
            "title": paper.get("title", ""),
            "arxiv_id": paper.get("arxiv_id", ""),
            "github_url": github_url,
            "pdf_url": paper.get("pdf_url", ""),
            "authors": paper.get("authors", [])[:3],
        },
        "model_dir": model_dir,
        "entry_script": entry_script,
        "status": "pending",  # pending / migrating / completed / failed
    }

    # 保存任务清单
    task_file = os.path.join(model_dir, "migration_task.json")
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    logger.info(f"已创建迁移任务清单: {task_file}")
    return task


def run_pipeline(days: int = DEFAULT_DAYS, max_results: int = 50):
    """运行完整流水线

    Args:
        days: 搜索最近几天的论文
        max_results: 每个关键词最大结果数
    """
    logger.info("=" * 60)
    logger.info("开始执行 arxiv-recommendation-npu 流水线")
    logger.info(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"搜索范围: 最近 {days} 天")
    logger.info("=" * 60)

    results = []

    # Step 1: 抓取论文
    logger.info("\n[Step 1] 抓取论文...")
    papers = search_papers(days=days, max_results=max_results)

    if not papers:
        logger.warning("未找到推荐相关论文")
        save_paper_list([], os.path.join(TODAY_DIR, "paper_list.md"))
        # 生成空报告
        from reporter import generate_daily_report, notify_user
        generate_daily_report([], [], os.path.join(TODAY_DIR, "daily_report.md"))
        notify_user(os.path.join(TODAY_DIR, "daily_report.md"), 0, 0)
        return

    # 保存论文列表
    save_paper_list(papers, os.path.join(TODAY_DIR, "paper_list.md"))
    logger.info(f"找到 {len(papers)} 篇论文")

    # Step 2: 检测源码
    logger.info("\n[Step 2] 检测源码...")
    papers_with_code = filter_papers_with_code(papers)
    logger.info(f"其中 {len(papers_with_code)} 篇有可用源码")

    # Step 3: 下载源码并创建迁移任务
    logger.info("\n[Step 3] 下载源码...")

    models_dir = os.path.join(TODAY_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    for i, paper in enumerate(papers_with_code, 1):
        title = paper.get("title", "Unknown")[:50]
        github_url = paper.get("github_url", "")
        logger.info(f"\n--- 处理 {i}/{len(papers_with_code)}: {title}")

        # 创建模型目录
        safe_name = "".join(c for c in title if c.isalnum() or c in " -_").strip()
        safe_name = safe_name.replace(" ", "_")[:30]
        model_dir = os.path.join(models_dir, f"model_{i}_{safe_name}")
        os.makedirs(model_dir, exist_ok=True)

        # 克隆仓库
        clone_result = clone_repo(github_url, model_dir)
        if not clone_result["success"]:
            result = {
                "status": "clone_failed",
                "message": clone_result["message"],
                "needs_manual": clone_result.get("needs_manual", False),
                "title": title,
                "github_url": github_url,
            }
            results.append(result)
            logger.warning(f"克隆失败: {title}, {clone_result['message']}")
            continue

        # 创建待迁移任务清单 (交给 npu-model-migration skill)
        task = create_migration_task(model_dir, paper, github_url)

        result = {
            "status": "ready_to_migrate",
            "message": "源码已克隆，请使用 npu-model-migration skill 进行 NPU 适配",
            "task": task,
            "title": title,
            "github_url": github_url,
            "paper": paper,
            "model_dir": model_dir,
        }
        results.append(result)
        logger.info(f"源码已克隆，任务清单已创建: {model_dir}/migration_task.json")

    # Step 4: 生成报告
    logger.info("\n[Step 4] 生成报告...")
    from reporter import generate_daily_report, notify_user
    report_path = os.path.join(TODAY_DIR, "daily_report.md")
    generate_daily_report(papers_with_code, results, report_path)

    # 统计
    ready_count = sum(1 for r in results if r.get("status") == "ready_to_migrate")
    fail_count = len(results) - ready_count

    # 通知用户
    notify_user(report_path, ready_count, fail_count)

    logger.info("\n" + "=" * 60)
    logger.info("流水线执行完成")
    logger.info(f"待迁移: {ready_count}, 失败: {fail_count}")
    logger.info("\n下一步：使用 npu-model-migration skill 进行 NPU 适配")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(days=args.days, max_results=args.max_results)