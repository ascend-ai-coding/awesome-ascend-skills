"""配置文件"""
import os
from datetime import datetime

# 基础路径配置（可通过环境变量 ARXIV_RECO_DIR 自定义）
BASE_DIR = os.environ.get("ARXIV_RECO_DIR", "./arxiv-recommendation-models")
TODAY = datetime.now().strftime("%Y-%m-%d")
TODAY_DIR = os.path.join(BASE_DIR, TODAY)

# 创建今日目录
os.makedirs(TODAY_DIR, exist_ok=True)
os.makedirs(os.path.join(TODAY_DIR, "models"), exist_ok=True)

# arxiv 论文筛选配置（推荐系统核心关键词）
ARXIV_KEYWORDS = [
    "recommendation",
    "recommender system",
    "collaborative filtering",
    "CTR prediction",
]

# 权威会议/期刊
TARGET_VENUES = [
    "RecSys",
    "SIGIR",
    "KDD",
    "WWW",
    "ICML",
    "NeurIPS",
    "TOIS",
    "arXiv",
]

# 热门方向
HOT_TOPICS = [
    "Sequential Recommendation",
    "CTR Prediction",
    "Graph Neural Network",
    "Multi-modal Recommendation",
    "Reinforcement Learning",
    "Debiasing",
    "Explainable Recommendation",
    "Cold Start",
    "Federated Recommendation",
]

# 源码检测配置
MIN_PY_FILES = 3  # 最少 .py 文件数量（排除只有 README 的仓库）

# NPU 适配配置（可通过环境变量 NPU_MIGRATION_SKILL_PATH 自定义）
NPU_SKILL_PATH = os.environ.get("NPU_MIGRATION_SKILL_PATH", "../npu-model-migration")

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

# 邮件配置（预留）
EMAIL_CONFIG = {
    "enabled": False,
    "smtp_server": "",
    "smtp_port": 587,
    "sender": "",
    "password": "",
    "recipients": [],
}

# GitHub 镜像配置（解决国内访问不稳定问题）
GITHUB_MIRROR = {
    "enabled": True,
    # 使用 ghproxy 镜像
    "url_template": "https://ghproxy.com/https://github.com/{owner}/{repo}",
    # 备选：GitCode 镜像
    # "url_template": "https://gitcode.net/{owner}/{repo}",
}