"""报告生成模块"""
import os
import json
from typing import List, Dict
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import TODAY_DIR, EMAIL_CONFIG, NPU_SKILL_PATH

logger = setup_logger("reporter")


def generate_daily_report(papers: List[Dict], results: List[Dict], output_path: str):
    """生成每日报告

    Args:
        papers: 论文列表
        results: 验证结果列表
        output_path: 输出文件路径
    """
    lines = []
    lines.append(f"# 每日推荐论文 NPU 适配报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"\n## 概览\n")
    lines.append(f"- 检测论文数: {len(papers)}\n")
    lines.append(f"- 成功克隆源码: {sum(1 for r in results if r.get('status') == 'ready_to_migrate')}\n")
    lines.append(f"- 克隆失败: {sum(1 for r in results if r.get('status') == 'clone_failed')}\n")
    lines.append(f"\n---\n\n")

    # 待迁移列表 (ready_to_migrate)
    ready_results = [r for r in results if r.get("status") == "ready_to_migrate"]
    if ready_results:
        lines.append(f"## 待迁移任务 (需调用 npu-model-migration skill)\n\n")
        lines.append(f"以下论文源码已准备好，请使用 **npu-model-migration skill** 进行 NPU 适配：\n\n")

        for r in ready_results:
            task = r.get("task", {})
            model_dir = r.get("model_dir", "")
            lines.append(f"### 📄 {r.get('title', 'Unknown')}\n")
            lines.append(f"- **arXiv**: {r.get('paper', {}).get('arxiv_id', 'N/A')}\n")
            lines.append(f"- **GitHub**: {r.get('github_url', 'N/A')}\n")
            lines.append(f"- **模型目录**: `{model_dir}`\n")
            lines.append(f"- **入口脚本**: `{task.get('entry_script', 'N/A')}`\n")
            lines.append(f"\n**下一步**：\n")
            lines.append(f"```bash\n")
            lines.append(f"# 进入模型目录\n")
            lines.append(f"cd {model_dir}\n")
            lines.append(f"\n")
            lines.append(f"# 使用 npu-model-migration skill 进行迁移\n")
            lines.append(f"# 参考: {NPU_SKILL_PATH}/SKILL.md\n")
            lines.append(f"```\n")
            lines.append(f"\n---\n\n")

    # 失败列表
    fail_results = [r for r in results if r.get("status") == "clone_failed"]
    if fail_results:
        lines.append(f"## 克隆失败\n\n")
        for r in fail_results:
            lines.append(f"### ❌ {r.get('title', 'Unknown')}\n")
            lines.append(f"- 源码: {r.get('github_url', 'N/A')}\n")
            lines.append(f"- 原因: {r.get('message', 'Unknown')}\n")

            # 如果需要用户手动处理
            if r.get("needs_manual"):
                lines.append(f"- ⚠️ **需要手动处理**: 请手动下载源码并放入对应目录\n")

            lines.append("\n")

    # 迁移说明
    lines.append(f"\n---\n\n")
    lines.append(f"## 迁移说明\n\n")
    lines.append(f"1. 进入上述模型目录\n")
    lines.append(f"2. 参考 **npu-model-migration skill** 的流程：\n")
    lines.append(f"   - 阶段 1.5: 快速尝试 (transfer_to_npu)\n")
    lines.append(f"   - 阶段 4: NPU 验证\n")
    lines.append(f"3. 迁移完成后，结果会保存在 `result.json`\n")
    lines.append(f"\n---\n\n")
    lines.append(f"**npu-model-migration skill 位置**: `{NPU_SKILL_PATH}/`\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"报告已生成: {output_path}")


def send_email(report_path: str, success_count: int, fail_count: int) -> bool:
    """发送邮件（预留接口）

    Args:
        report_path: 报告文件路径
        success_count: 成功数量
        fail_count: 失败数量

    Returns:
        是否发送成功
    """
    if not EMAIL_CONFIG.get("enabled", False):
        logger.info("邮件功能未启用，跳过发送")
        return False

    # TODO: 实现邮件发送
    logger.warning("邮件发送功能尚未实现")
    return False


def notify_user(report_path: str, ready_count: int, fail_count: int):
    """通知用户

    Args:
        report_path: 报告路径
        ready_count: 待迁移数量
        fail_count: 失败数量
    """
    logger.info(f"处理完成: 待迁移 {ready_count} 个, 失败 {fail_count} 个")

    # 尝试发送邮件
    if send_email(report_path, ready_count, fail_count):
        logger.info("邮件发送成功")
    else:
        logger.info("请查看本地报告")


def save_paper_results(model_dir: str, paper: Dict, result: Dict):
    """保存单个论文的结果"""
    # 合并论文信息和验证结果
    combined = {
        "paper": {
            "title": paper.get("title"),
            "authors": paper.get("authors"),
            "published": paper.get("published"),
            "github_url": paper.get("github_url"),
            "pdf_url": paper.get("pdf_url"),
        },
        "result": result
    }

    result_file = os.path.join(model_dir, "paper_info.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    logger.info(f"论文结果已保存: {result_file}")