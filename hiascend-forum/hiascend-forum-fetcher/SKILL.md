---
name: hiascend-forum-fetcher
description: 读取昇腾社区论坛（Hiascend Community Forum）的帖子内容，支持按时间范围筛选并导出为Excel。支持时间快捷选项（今年、本月、本周、今天）和自定义区间，自动剔除置顶贴（top>0），智能终止查询，时区自动转换（+8小时），并发分页获取提升效率，并将topicId转换为可点击的论坛链接。当用户需要获取昇腾论坛帖子、筛选特定时间范围的论坛内容、导出论坛数据到Excel时触发使用。
---

# 昇腾社区论坛数据获取工具

用于从昇腾社区论坛 API 获取帖子数据，按时间范围筛选，自动剔除置顶贴，支持并发分页获取，并导出为 Excel 格式。

## 功能特性

- **灵活的时间筛选**：支持"今年"、"本月"、"本周"、"今天"快捷选项，或自定义日期区间，最小粒度为天
- **智能截止时间调整**：若查询截止时间超过当天，自动调整为当天23:59:59
- **置顶贴自动剔除**：自动过滤 resultList 中 top > 0 的置顶贴
- **智能查询终止**：当当前批次最早创建时间早于查询区间时，自动停止后续查询
- **时区自动转换**：createTime 自动加8小时（北京时间转换）
- **可配置分页大小**：支持自定义每页获取数据量（默认100条）
- **并发分页获取**：支持多线程并发获取，提升查询效率（默认6并发）
- **Excel 导出**：结构化输出，包含指定字段和可点击链接

## API 端点

```
https://www.hiascend.com/ascendgateway/ascendservice/devCenter/bbs/servlet/get-topic-list?filterCondition=1&pageIndex={page}&pageSize={pageSize}
```

## 工作流程

### 1. 解析并调整时间范围

将用户输入转换为开始日期和结束日期（YYYYMMDDHHMMSS 格式）：

**快捷选项：**
- `今年` → 当年1月1日00:00:00 至 今天23:59:59
- `本月` → 当月1日00:00:00 至 今天23:59:59  
- `本周` → 本周一00:00:00 至 今天23:59:59
- `今天` → 今天00:00:00 至 今天23:59:59

**自定义区间：**
- 格式：`YYYY年MM月DD日` 或 `YYYY-MM-DD` 或 `YYYYMMDD`
- 示例：`2026年1月1号到3月30号` → start=20260101000000, end=20260330235959

**截止时间调整：**
- 若 end_date > today，则自动调整为 today 23:59:59
- 输出警告日志告知用户调整情况

### 2. 分页并发获取数据

**两阶段获取策略：**

**阶段1：获取总页数**
- 请求第1页，获取 `totalCount`
- 计算总页数：`total_pages = ceil(totalCount / pageSize)`

**阶段2：并发获取**
- 使用线程池并发获取剩余页面
- 默认并发数：6（可通过 `--concurrency` 调整）
- 每批次获取后处理数据（剔除置顶贴、时区转换、时间筛选）
- 智能终止：当某批次最早时间早于查询区间时，停止后续获取

### 3. 数据处理流程

**步骤1：剔除置顶贴**
- 过滤 resultList 中 top > 0 的数据
- 记录剔除数量

**步骤2：时区转换**
- 将 createTime（YYYYMMDDHHMMSS）加8小时
- 转换为北京时间

**步骤3：时间筛选**
- 筛选落在查询区间内的帖子
- 获取当前批次最早创建时间

**步骤4：智能终止判断**
- 若当前批次最早 createTime（加8小时后）< 查询区间开始时间
- 则停止后续查询，仅保留区间内数据

### 4. 输出格式（Excel）

将筛选后的数据保存为 Excel 文件（.xlsx），包含以下列：

| 列名 | 说明 | 格式/规则 |
|------|------|-----------|
| topicId | 帖子ID | resultList.topicId |
| sectionName | 版块名称 | resultList.sectionName |
| topicClassName | 话题分类 | resultList.topicClassName |
| title | 帖子标题 | resultList.title |
| createTime | 创建时间 | yyyy-mm-dd（已加8小时） |
| topicLink | 帖子链接 | https://www.hiascend.com/forum/thread-{topicId}-1-1.html |

## CLI 参数说明

```bash
python scripts/fetch_hiascend_forum.py <时间范围> [选项]
```

**位置参数：**
- `time_range`：时间范围文本，例如：今年、本月、2026年3月1日到3月30日

**可选参数：**
- `--page-size N`：每页获取数据量（默认 100）
- `--concurrency N`：并发数（默认 6，建议不超过10）
- `--max-pages N`：最大分页数（默认 500）
- `--output <路径>`：输出目录，默认当前工作目录
- `--log-level <级别>`：日志等级（DEBUG/INFO/WARNING/ERROR，默认 INFO）

## 实现脚本

```python
#!/usr/bin/env python3
"""
昇腾社区论坛数据获取工具
支持：时间灵活筛选、置顶贴剔除、智能终止、时区转换、并发分页获取、Excel导出
"""

import argparse
import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import requests
import sys


def parse_date_range(time_input):
    """
    解析时间范围，支持快捷选项和自定义区间
    若截止时间超过当天，调整为当天23:59:59
    返回: (start_time, end_time) 格式 YYYYMMDDHHMMSS
    """
    today = datetime.now()
    today_end = today.replace(hour=23, minute=59, second=59)
    
    # 快捷选项
    if '今年' in time_input or '本年' in time_input:
        start_time = today.replace(month=1, day=1, hour=0, minute=0, second=0)
        end_time = today_end
    elif '本月' in time_input:
        start_time = today.replace(day=1, hour=0, minute=0, second=0)
        end_time = today_end
    elif '本周' in time_input:
        monday = today - timedelta(days=today.weekday())
        start_time = monday.replace(hour=0, minute=0, second=0)
        end_time = today_end
    elif '今天' in time_input:
        start_time = today.replace(hour=0, minute=0, second=0)
        end_time = today_end
    else:
        # 自定义区间解析
        dates = extract_dates(time_input)
        if len(dates) >= 2:
            start_dt = datetime.strptime(dates[0], '%Y%m%d')
            end_dt = datetime.strptime(dates[1], '%Y%m%d')
            start_time = start_dt.replace(hour=0, minute=0, second=0)
            end_time = end_dt.replace(hour=23, minute=59, second=59)
        else:
            raise ValueError(f"无法解析时间范围：{time_input}")
    
    # 若截止时间超过今天，调整为今天23:59:59
    if end_time.date() > today.date():
        logging.warning(f"查询截止时间 {end_time.strftime('%Y-%m-%d %H:%M:%S')} 超过今天，"
                       f"已自动调整为今天23:59:59")
        end_time = today_end
    
    return start_time.strftime('%Y%m%d%H%M%S'), end_time.strftime('%Y%m%d%H%M%S')


def extract_dates(text):
    """从文本中提取日期"""
    import re
    dates = []
    patterns = [
        r'(\d{4})年(\d{1,2})月(\d{1,2})[号日]?',
        r'(\d{4})-(\d{2})-(\d{2})',
        r'\d{8}'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                y, m, d = match
                dates.append(f"{int(y):04d}{int(m):02d}{int(d):02d}")
            else:
                dates.append(match)
    return dates[:2]


def convert_to_beijing_time(create_time_str):
    """将createTime转换为北京时间（+8小时），返回YYYYMMDDHHMMSS"""
    if len(create_time_str) >= 14:
        try:
            dt = datetime.strptime(create_time_str[:14], '%Y%m%d%H%M%S')
            dt = dt + timedelta(hours=8)
            return dt.strftime('%Y%m%d%H%M%S')
        except:
            pass
    return create_time_str


def format_date_display(beijing_time_str):
    """将北京时间格式化为 yyyy-mm-dd"""
    if len(beijing_time_str) >= 8:
        return f"{beijing_time_str[:4]}-{beijing_time_str[4:6]}-{beijing_time_str[6:8]}"
    return beijing_time_str


def fetch_page(page, base_url, headers, retry_limit=3, backoff=1.5):
    """单页抓取"""
    url = base_url.format(page=page)
    for attempt in range(1, retry_limit + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json()
            if data.get('code') != 200:
                raise RuntimeError(f"API error: {data.get('msg')}")
            return data.get('data', {}).get('resultList', []) or []
        except Exception as e:
            if attempt >= retry_limit:
                logging.warning(f"分页 {page} 请求失败：{e}")
                return []
            time.sleep(min(backoff ** attempt, 5))
    return []


def process_topics(topics, start_time, end_time):
    """
    处理单页数据：
    1. 剔除置顶贴（top > 0）
    2. 时区转换（+8小时）
    3. 时间筛选
    返回: (有效帖子列表, 最早北京时间, 置顶贴数量)
    """
    # 步骤1：剔除置顶贴
    filtered = []
    pinned_count = 0
    for t in topics:
        top_value = t.get('top', 0)
        if top_value and int(top_value) > 0:
            pinned_count += 1
        else:
            filtered.append(t)
    
    if not filtered:
        return [], None, pinned_count
    
    # 步骤2&3：时区转换并筛选
    valid_topics = []
    earliest_beijing_time = None
    
    for t in filtered:
        original_ct = str(t.get('createTime', ''))
        beijing_ct = convert_to_beijing_time(original_ct)
        
        # 记录最早时间
        if earliest_beijing_time is None or beijing_ct < earliest_beijing_time:
            earliest_beijing_time = beijing_ct
        
        # 筛选查询区间内的数据
        if start_time <= beijing_ct <= end_time:
            t['_beijing_createTime'] = beijing_ct
            valid_topics.append(t)
    
    return valid_topics, earliest_beijing_time, pinned_count


def fetch_all_topics(start_time, end_time, page_size=100, concurrency=6, max_pages=500):
    """
    智能分页并发获取：
    1. 先获取第1页，得到totalCount，计算总页数
    2. 并发获取剩余页面
    3. 实时处理数据并智能终止
    """
    base_url = ("https://www.hiascend.com/ascendgateway/ascendservice/"
                "devCenter/bbs/servlet/get-topic-list?"
                f"filterCondition=1&pageIndex={{page}}&pageSize={page_size}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.hiascend.com/',
        'Accept': 'application/json, text/plain, */*',
    }
    
    all_topics = []
    total_pinned = 0
    should_stop = False
    
    logging.info(f"开始获取数据，每页{page_size}条，并发{concurrency}，"
                f"查询区间：{start_time} 至 {end_time}")
    
    # 阶段1：获取第1页，得到totalCount
    first_page_topics = fetch_page(1, base_url, headers)
    if not first_page_topics:
        logging.error("获取第1页失败，无法继续")
        return []
    
    # 处理第1页数据
    valid_topics, earliest_time, pinned = process_topics(first_page_topics, start_time, end_time)
    total_pinned += pinned
    if valid_topics:
        all_topics.extend(valid_topics)
        logging.info(f"第1页：获取 {len(valid_topics)} 条有效数据，"
                    f"剔除置顶贴 {pinned} 条")
    
    # 检查是否需要停止
    if earliest_time and earliest_time < start_time:
        logging.info(f"第1页最早时间 {earliest_time[:8]} 早于查询区间，停止获取")
        return all_topics
    
    # 获取totalCount计算总页数
    try:
        resp = requests.get(base_url.format(page=1), headers=headers, timeout=30)
        data = resp.json()
        total_count = int(data.get('data', {}).get('totalCount', 0))
        total_pages = math.ceil(total_count / page_size)
        total_pages = min(total_pages, max_pages)
        logging.info(f"总记录数：{total_count}，总页数：{total_pages}")
    except Exception as e:
        logging.warning(f"获取总记录数失败：{e}，将逐页获取")
        total_pages = max_pages
    
    if total_pages <= 1:
        return all_topics
    
    # 阶段2：并发获取剩余页面
    pages_to_fetch = list(range(2, total_pages + 1))
    mutex = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_page = {
            executor.submit(fetch_page, p, base_url, headers): p 
            for p in pages_to_fetch
        }
        
        for future in as_completed(future_to_page):
            if should_stop:
                # 取消剩余任务
                for f in future_to_page:
                    f.cancel()
                break
            
            page = future_to_page[future]
            try:
                topics = future.result()
            except Exception as e:
                logging.warning(f"分页 {page} 处理异常：{e}")
                continue
            
            if not topics:
                continue
            
            # 处理该页数据
            valid_topics, earliest_time, pinned = process_topics(topics, start_time, end_time)
            
            with mutex:
                total_pinned += pinned
                if valid_topics:
                    all_topics.extend(valid_topics)
                
                # 检查是否需要停止
                if earliest_time and earliest_time < start_time:
                    logging.info(f"分页 {page} 最早时间 {earliest_time[:8]} 早于查询区间，"
                               f"停止后续获取")
                    should_stop = True
                else:
                    logging.info(f"分页 {page} 完成，获取 {len(valid_topics)} 条，"
                               f"累计 {len(all_topics)} 条")
    
    if total_pinned > 0:
        logging.info(f"总计剔除置顶贴 {total_pinned} 条")
    
    return all_topics


def extract_required_fields(topics):
    """提取字段并生成Excel数据"""
    extracted = []
    for t in topics:
        topic_id = t.get('topicId') or t.get('tid') or ''
        beijing_time = t.get('_beijing_createTime', '')
        
        row = {
            'topicId': topic_id,
            'sectionName': t.get('sectionName', ''),
            'topicClassName': t.get('topicClassName', ''),
            'title': t.get('title', ''),
            'createTime': format_date_display(beijing_time),  # yyyy-mm-dd
            'topicLink': f"https://www.hiascend.com/forum/thread-{topic_id}-1-1.html"
        }
        extracted.append(row)
    return extracted


def save_to_excel(data, start_time, end_time, filename=None):
    """保存为Excel"""
    try:
        import pandas as pd
    except ImportError:
        logging.error("需要安装 pandas 和 openpyxl")
        sys.exit(1)
    
    if not filename:
        filename = f"hiascend_topics_{start_time[:8]}_{end_time[:8]}.xlsx"
    
    df = pd.DataFrame(data)
    cols = ['topicId', 'sectionName', 'topicClassName', 'title', 'createTime', 'topicLink']
    df = df[cols]
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='帖子列表', index=False)
        ws = writer.sheets['帖子列表']
        widths = [18, 20, 20, 50, 15, 70]
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[chr(64 + i)].width = w
    
    logging.info(f"数据已保存：{filename}，共 {len(data)} 条")


def main():
    parser = argparse.ArgumentParser(description="昇腾社区论坛数据获取工具")
    parser.add_argument("time_range", help="时间范围，如：今年、本月、2026年1月1日到3月30日")
    parser.add_argument("--page-size", type=int, default=100, 
                       help="每页获取数据量（默认100）")
    parser.add_argument("--concurrency", type=int, default=6,
                       help="并发数（默认6，建议不超过10）")
    parser.add_argument("--max-pages", type=int, default=500, help="最大页数（默认500）")
    parser.add_argument("--output", help="输出目录")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # 解析时间范围
    start_time, end_time = parse_date_range(args.time_range)
    logging.info(f"查询区间：{start_time} 至 {end_time}")
    logging.info(f"每页获取：{args.page_size} 条，并发数：{args.concurrency}")
    
    # 获取数据
    topics = fetch_all_topics(start_time, end_time, args.page_size, args.concurrency, args.max_pages)
    logging.info(f"共获取 {len(topics)} 条有效帖子")
    
    # 按createTime排序
    topics.sort(key=lambda x: x.get('_beijing_createTime', ''), reverse=True)
    
    # 提取字段
    data = extract_required_fields(topics)
    
    # 保存Excel
    filename = None
    if args.output:
        import os
        os.makedirs(args.output, exist_ok=True)
        filename = os.path.join(args.output, 
                               f"hiascend_topics_{start_time[:8]}_{end_time[:8]}.xlsx")
    
    save_to_excel(data, start_time, end_time, filename)


if __name__ == "__main__":
    main()
```

## 依赖

- Python 3.7+
- requests
- pandas
- openpyxl

```bash
pip install requests pandas openpyxl
```

## 使用示例

```bash
# 快捷选项
python scripts/fetch_hiascend_forum.py "今年"
python scripts/fetch_hiascend_forum.py "本月"

# 自定义区间
python scripts/fetch_hiascend_forum.py "2026年1月1号到3月30号"

# 调整每页数据量和并发数
python scripts/fetch_hiascend_forum.py "今年" --page-size 50 --concurrency 8

# 指定输出目录
python scripts/fetch_hiascend_forum.py "今年" --output ./data
```
