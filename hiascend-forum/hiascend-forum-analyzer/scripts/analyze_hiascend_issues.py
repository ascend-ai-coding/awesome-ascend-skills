#!/usr/bin/env python3
"""
昇腾社区论坛问题分析工具
读取Excel数据，筛选问题类帖子，获取详细内容，导出分析报告
支持从配置文件读取筛选关键词
"""

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    logging.info(f"加载配置文件: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证必需配置项
        required_keys = ['initial_filter', 'deep_filter']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必需项: {key}")
        
        logging.info("配置文件加载成功")
        return config
    except FileNotFoundError:
        logging.error(f"配置文件未找到: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"配置文件格式错误: {e}")
        raise


def setup_logging(level: str = "INFO"):
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_session():
    """创建带有重试机制的请求会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def read_excel_data(file_path: str) -> pd.DataFrame:
    """读取Excel文件并验证字段"""
    logging.info(f"读取Excel文件: {file_path}")
    
    try:
        # 强制将topicId列读取为字符串，保留前导零
        df = pd.read_excel(file_path, dtype={'topicId': str})
    except Exception as e:
        logging.error(f"读取Excel文件失败: {e}")
        raise
    
    # 验证必需字段
    required_columns = ['topicId', 'sectionName', 'topicClassName', 
                       'title', 'createTime', 'topicLink']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Excel文件缺少必需字段: {missing_columns}")
    
    logging.info(f"成功读取 {len(df)} 条记录")
    return df


def matches_keywords(text: str, keywords: List[str], 
                    case_sensitive: bool = False,
                    match_mode: str = 'contains') -> bool:
    """
    检查文本是否匹配关键词
    
    Args:
        text: 待检查的文本
        keywords: 关键词列表
        case_sensitive: 是否区分大小写
        match_mode: 匹配模式 - contains/exact/regex
    """
    if pd.isna(text):
        return False
    
    text = str(text)
    
    if not case_sensitive:
        text = text.lower()
        keywords = [kw.lower() for kw in keywords]
    
    for keyword in keywords:
        if match_mode == 'contains':
            if keyword in text:
                return True
        elif match_mode == 'exact':
            if keyword == text:
                return True
        elif match_mode == 'regex':
            try:
                if re.search(keyword, text):
                    return True
            except re.error:
                logging.warning(f"正则表达式错误: {keyword}")
                continue
    
    return False


def initial_filter(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """初步筛选：基于topicClassName和title"""
    logging.info("执行初步筛选...")
    
    filter_config = config.get('initial_filter', {})
    settings = config.get('filter_settings', {})
    
    class_keywords = filter_config.get('topic_class_keywords', [])
    title_keywords = filter_config.get('title_keywords', [])
    case_sensitive = settings.get('case_sensitive', False)
    match_mode = settings.get('match_mode', 'contains')
    
    # topicClassName匹配
    class_match = df['topicClassName'].apply(
        lambda x: matches_keywords(x, class_keywords, case_sensitive, match_mode)
    )
    
    # title匹配
    title_match = df['title'].apply(
        lambda x: matches_keywords(x, title_keywords, case_sensitive, match_mode)
    )
    
    # 合并条件（任一匹配）
    filtered_df = df[class_match | title_match].copy()
    
    logging.info(f"初步筛选完成: {len(df)} -> {len(filtered_df)} 条")
    return filtered_df


def fetch_topic_content(session: requests.Session, 
                       topic_id: str,
                       retry_limit: int = 3) -> Optional[str]:
    """获取帖子详细内容"""
    url = ("https://www.hiascend.com/ascendgateway/ascendservice/"
           f"devCenter/bbs/servlet/get-topic-detail?topicId={topic_id}")
    
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'),
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.hiascend.com/'
    }
    
    for attempt in range(retry_limit):
        try:
            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == 200:
                result = data.get('data', {}).get('result', {})
                content = result.get('content', '')
                # 简单的HTML标签清洗
                content = re.sub(r'<[^>]+>', '', content)
                content = re.sub(r'\s+', ' ', content).strip()
                return content
            else:
                logging.warning(f"API返回错误: {data.get('msg')}")
                return None
                
        except requests.exceptions.RequestException as e:
            if attempt == retry_limit - 1:
                logging.warning(f"获取帖子 {topic_id} 内容失败: {e}")
                return None
            time.sleep(2 ** attempt)  # 指数退避
    
    return None


def fetch_all_contents(df: pd.DataFrame, 
                      concurrency: int = 5,
                      min_length: int = 10) -> pd.DataFrame:
    """并发获取所有帖子的详细内容"""
    logging.info(f"开始获取帖子详细内容，并发数: {concurrency}")
    
    session = create_session()
    contents = {}
    
    def fetch_with_delay(topic_id):
        content = fetch_topic_content(session, topic_id)
        time.sleep(0.5)  # 请求间隔
        return topic_id, content
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_id = {
            executor.submit(fetch_with_delay, row['topicId']): row['topicId']
            for _, row in df.iterrows()
        }
        
        completed = 0
        total = len(future_to_id)
        
        for future in as_completed(future_to_id):
            topic_id = future_to_id[future]
            try:
                _, content = future.result()
                # 过滤过短的内容
                if content and len(content) >= min_length:
                    contents[topic_id] = content
                else:
                    contents[topic_id] = content if content else ''
            except Exception as e:
                logging.warning(f"处理帖子 {topic_id} 时出错: {e}")
                contents[topic_id] = ''
            
            completed += 1
            if completed % 10 == 0:
                logging.info(f"进度: {completed}/{total}")
    
    # 添加content列
    df['content'] = df['topicId'].map(contents)
    df['content'] = df['content'].fillna('')
    
    logging.info(f"内容获取完成，成功: {sum(1 for v in contents.values() if v)}/{total}")
    return df


def deep_filter(df: pd.DataFrame,
               config: Dict,
               filter_mode: int = 1) -> pd.DataFrame:
    """深度筛选：基于title和content"""
    logging.info(f"执行深度筛选，模式: {filter_mode}")
    
    filter_config = config.get('deep_filter', {})
    settings = config.get('filter_settings', {})
    
    problem_keywords = filter_config.get('problem_keywords', [])
    tech_keywords = filter_config.get('tech_keywords', [])
    case_sensitive = settings.get('case_sensitive', False)
    match_mode = settings.get('match_mode', 'contains')
    
    # 组合文本用于搜索
    df['_search_text'] = (df['title'].fillna('') + ' ' + 
                         df['content'].fillna('')).str.lower()
    
    if filter_mode == 1:
        # 模式1：至少包含1个问题类关键词
        mask = df['_search_text'].apply(
            lambda x: matches_keywords(x, problem_keywords, case_sensitive, match_mode)
        )
    elif filter_mode == 2 and tech_keywords:
        # 模式2：同时包含问题类和技术领域关键词
        problem_mask = df['_search_text'].apply(
            lambda x: matches_keywords(x, problem_keywords, case_sensitive, match_mode)
        )
        tech_mask = df['_search_text'].apply(
            lambda x: matches_keywords(x, tech_keywords, case_sensitive, match_mode)
        )
        mask = problem_mask & tech_mask
    else:
        mask = pd.Series([True] * len(df))
    
    filtered_df = df[mask].copy()
    filtered_df = filtered_df.drop(columns=['_search_text'])
    
    logging.info(f"深度筛选完成: {len(df)} -> {len(filtered_df)} 条")
    return filtered_df


def save_to_excel(df: pd.DataFrame, output_path: str):
    """保存结果到Excel"""
    logging.info(f"保存结果到: {output_path}")
    
    # 调整列顺序（不包含 content 列）
    column_order = ['topicId', 'sectionName', 'topicClassName', 
                   'title', 'createTime', 'topicLink']
    
    # 确保所有列存在
    for col in column_order:
        if col not in df.columns:
            df[col] = ''
    
    df_export = df[column_order].copy()
    
    # 将topicId转换为字符串格式，防止科学计数法
    df_export['topicId'] = df_export['topicId'].astype(str)
    
    # 保存Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='问题帖子分析', index=False)
        
        # 设置列宽
        worksheet = writer.sheets['问题帖子分析']
        column_widths = [25, 20, 20, 50, 15, 70]
        for i, width in enumerate(column_widths, 1):
            worksheet.column_dimensions[chr(64 + i)].width = width
        
        # 设置topicId列（A列）为文本格式，防止科学计数法
        from openpyxl.styles import numbers
        for cell in worksheet['A']:
            if cell.row > 1:  # 跳过表头
                cell.number_format = '@'  # @ 表示文本格式
    
    logging.info(f"成功保存 {len(df_export)} 条记录到Excel")


def main():
    parser = argparse.ArgumentParser(
        description='昇腾社区论坛问题分析工具'
    )
    parser.add_argument('input_excel', help='输入Excel文件路径')
    parser.add_argument('--config', default='references/keywords.json',
                       help='配置文件路径 (默认: references/keywords.json)')
    parser.add_argument('--filter-mode', type=int, default=1,
                       help='筛选模式：1=问题类, 2=问题+技术 (默认1)')
    parser.add_argument('--concurrency', type=int, default=5,
                       help='并发请求数 (默认5)')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--log-level', default='INFO',
                       help='日志级别 (默认INFO)')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        # 1. 加载配置
        config = load_config(args.config)
        
        # 2. 读取数据
        df = read_excel_data(args.input_excel)
        
        # 3. 初步筛选
        df_filtered = initial_filter(df, config)
        
        if len(df_filtered) == 0:
            logging.warning("初步筛选后无数据，请检查筛选条件")
            return
        
        # 4. 获取详细内容
        min_length = config.get('filter_settings', {}).get('min_content_length', 10)
        df_with_content = fetch_all_contents(df_filtered, args.concurrency, min_length)
        
        # 5. 深度筛选
        df_final = deep_filter(df_with_content, config, args.filter_mode)
        
        # 6. 保存结果
        if not args.output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output = f'hiascend_issues_{timestamp}.xlsx'
        
        save_to_excel(df_final, args.output)
        
        logging.info(f"分析完成！输入: {len(df)} -> 输出: {len(df_final)} 条")
        
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
        raise


if __name__ == '__main__':
    main()