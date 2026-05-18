"""日期工具"""
from datetime import datetime, timedelta


def get_today() -> str:
    """获取今天的日期字符串"""
    return datetime.now().strftime("%Y-%m-%d")


def get_date_days_ago(days: int) -> str:
    """获取 N 天前的日期字符串"""
    date = datetime.now() - timedelta(days=days)
    return date.strftime("%Y-%m-%d")


def parse_date(date_str: str) -> datetime:
    """解析日期字符串"""
    return datetime.strptime(date_str, "%Y-%m-%d")


def format_date(date: datetime) -> str:
    """格式化日期"""
    return date.strftime("%Y-%m-%d")