---
name: hiascend-forum-analyzer
description: 昇腾社区论坛问题分析工具，读取Excel格式的论坛帖子数据，自动筛选问题、故障、报错类帖子，获取帖子详细内容，并根据关键词进行多维度筛选分析，最终导出结构化的问题数据Excel报告。用于分析昇腾论坛中的技术问题、故障排查和报错处理相关帖子。
---

# 昇腾社区论坛问题分析工具

## 功能概述

本工具用于分析昇腾社区论坛中的技术问题、故障排查和报错处理相关帖子。支持从标准格式的Excel文件读取数据，自动识别问题类帖子，获取详细内容，并进行多维度筛选分析。

## 输入数据格式

**源文件格式**：`hiascend_topics_temp.xlsx`

**必须包含的字段**：
- `topicId`：帖子唯一标识符
- `sectionName`：版块名称
- `topicClassName`：话题分类
- `title`：帖子标题
- `createTime`：创建时间（格式：yyyy-mm-dd）
- `topicLink`：帖子链接

**文件位置**：用户指定的Excel文件路径

## 配置文件

关键词配置存储在独立的 JSON 文件中，便于维护和定制。

### 配置文件：references/keywords.json

```json
{
  "initial_filter": {
    "topic_class_keywords": [
      "问题", "故障", "报错", "错误", "error", "issue", "bug", "异常",
      "报错信息", "出错", "失败", "技术支持", "求助", "问答", "提问"
    ],
    "title_keywords": [
      "报错", "错误", "error", "失败", "fail", "异常", "bug", "issue", "问题",
      "无法", "不能", "报错信息", "出错", "运行失败", "编译失败", "安装失败",
      "求助", "请问", "怎么解决", "如何解决", "报错码", "error code"
    ]
  },
  "deep_filter": {
    "problem_keywords": [
      "报错", "错误", "error", "exception", "失败", "fail",
      "无法", "不能", "运行失败", "编译失败", "安装失败", "启动失败",
      "求助", "解决", "报错信息", "报错码", "出错信息",
      "卡死", "死机", "崩溃", "闪退", "无响应",
      "error", "fail", "exception", "bug", "issue", "fault",
      "crash", "cannot", "unable", "not working", "failed"
    ],
    "tech_keywords": [
      "ascend", "cann", "mindspore", "atlas",
      "算子", "模型", "训练", "推理", "部署",
      "驱动", "固件", "sdk", "开发环境"
    ]
  },
  "filter_settings": {
    "case_sensitive": false,
    "match_mode": "contains",
    "min_content_length": 10
  }
}
```

### 配置项说明

| 配置项 | 说明 | 类型 |
|--------|------|------|
| `initial_filter.topic_class_keywords` | topicClassName字段筛选关键词 | 字符串数组 |
| `initial_filter.title_keywords` | title字段筛选关键词 | 字符串数组 |
| `deep_filter.problem_keywords` | 深度筛选问题类关键词 | 字符串数组 |
| `deep_filter.tech_keywords` | 技术领域关键词（可选） | 字符串数组 |
| `filter_settings.case_sensitive` | 是否区分大小写 | 布尔值 |
| `filter_settings.match_mode` | 匹配模式：contains/exact/regex | 字符串 |
| `filter_settings.min_content_length` | content最小长度限制 | 整数 |

## 工作流程

### 1. 读取配置文件

- 加载 `references/keywords.json` 配置文件
- 验证必需配置项完整性
- 解析筛选关键词和匹配设置

### 2. 读取并解析Excel数据

- 使用 pandas 读取指定Excel文件
- 验证必需字段是否存在
- 数据类型检查和清洗

### 3. 初步筛选（基于topicClassName和title）

**从配置文件读取筛选关键词**：
- **topicClassName** 匹配关键词：来自 `initial_filter.topic_class_keywords`
- **title** 匹配关键词：来自 `initial_filter.title_keywords`

**匹配设置**（来自 `filter_settings`）：
- `case_sensitive`：是否区分大小写（默认false）
- `match_mode`：匹配模式（contains/exact/regex）
- 支持模糊匹配、精确匹配、正则匹配

**匹配规则**：
- topicClassName 或 title 任一匹配即可
- 记录筛选前后的数据量变化

### 4. 获取帖子详细内容

**API端点**：
```
https://www.hiascend.com/ascendgateway/ascendservice/devCenter/bbs/servlet/get-topic-detail?topicId={topicId}
```

**请求配置**：
- 方法：GET
- 超时：30秒
- 重试次数：3次
- User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36

**响应处理**：
- 检查 HTTP 状态码（200）
- 解析 JSON 响应
- 提取 `result.content` 字段
- HTML 标签清洗（可选）

**并发控制**：
- 默认并发数：5
- 避免对服务器造成压力
- 添加请求间隔（0.5秒）

### 5. 数据结构重组

**新增字段**：
- `content`：帖子详细内容（放在 topicLink 前面）

**输出字段顺序**：
```
topicId, sectionName, topicClassName, title, createTime, content, topicLink
```

### 6. 深度筛选（基于title和content）

**从配置文件读取筛选条件**：

**问题类关键词**（`deep_filter.problem_keywords`）：
- 报错、错误、error、exception、失败、fail
- 无法、不能、运行失败、编译失败、安装失败
- 异常、bug、issue、fault、crash、卡死、死机
- 求助、解决、问题

**技术领域关键词**（`deep_filter.tech_keywords`，可选）：
- Ascend、CANN、MindSpore、Atlas
- 算子、模型、训练、推理、部署
- 驱动、固件、SDK、开发环境

**筛选模式**：
- 模式1：至少包含1个问题类关键词
- 模式2：同时包含问题类和技术领域关键词
- 使用配置文件的 `filter_settings` 控制匹配行为

### 7. 数据导出

**输出文件命名**：
```
hiascend_issues_{timestamp}.xlsx
```

**输出字段**：
- `topicId`：帖子ID
- `sectionName`：版块名称
- `topicClassName`：话题分类
- `title`：帖子标题
- `createTime`：创建时间
- `content`：帖子内容
- `topicLink`：帖子链接

**Excel格式设置**：
- 列宽自适应
- 内容列启用自动换行
- 链接列设置为超链接格式

## CLI 参数说明

```bash
python scripts/analyze_hiascend_issues.py <input_excel> [选项]
```

**位置参数**：
- `input_excel`：输入Excel文件路径（必须符合hiascend_topics_temp.xlsx格式）

**可选参数**：
- `--config <路径>`：配置文件路径（默认 references/keywords.json）
- `--filter-mode`：筛选模式（1=问题类, 2=问题+技术，默认1）
- `--concurrency N`：并发请求数（默认 5）
- `--output <路径>`：输出文件路径（默认自动生成）
- `--log-level <级别>`：日志等级（DEBUG/INFO/WARNING/ERROR，默认 INFO）

## 依赖要求

```
pandas>=1.3.0
openpyxl>=3.0.0
requests>=2.25.0
urllib3>=1.26.0
```

安装命令：
```bash
pip install pandas openpyxl requests urllib3
```

## 文件结构

```
hiascend-forum-analyzer/
├── SKILL.md                        # 技能文档
├── scripts/
│   └── analyze_hiascend_issues.py  # 主程序
├── references/
│   └── keywords.json               # 配置文件（关键词和匹配设置）
├── hiascend_topics_temp.xlsx       # 输入文件（示例）
└── hiascend_issues_*.xlsx          # 输出文件（自动生成）
```

## 使用示例

### 基础用法（使用默认配置）
```bash
python scripts/analyze_hiascend_issues.py hiascend_topics_temp.xlsx
```

### 使用自定义配置文件
```bash
# 创建自定义配置文件 my_keywords.json
{
  "initial_filter": {
    "topic_class_keywords": ["故障", "报错"],
    "title_keywords": ["报错", "错误", "失败"]
  },
  "deep_filter": {
    "problem_keywords": ["报错", "错误", "失败", "无法"],
    "tech_keywords": ["CANN", "MindSpore", "算子"]
  },
  "filter_settings": {
    "case_sensitive": false,
    "match_mode": "contains",
    "min_content_length": 20
  }
}

# 运行分析
python scripts/analyze_hiascend_issues.py hiascend_topics_temp.xlsx \
    --config my_keywords.json \
    --filter-mode 2
```

### 调整筛选模式
```bash
# 仅筛选问题类帖子
python scripts/analyze_hiascend_issues.py hiascend_topics_temp.xlsx --filter-mode 1

# 同时匹配问题类和技术领域关键词
python scripts/analyze_hiascend_issues.py hiascend_topics_temp.xlsx --filter-mode 2
```

### 调整并发和输出
```bash
python scripts/analyze_hiascend_issues.py hiascend_topics_temp.xlsx \
    --concurrency 10 \
    --output ./reports/ascend_issues_march.xlsx
```

## 输出结果说明

**统计信息**：
- 输入数据总量
- 初步筛选后数量
- 成功获取内容数量
- 最终筛选结果数量

**输出文件内容**：
- 所有字段与输入文件一致
- 新增 `content` 列存储帖子详细内容
- 按原始顺序排列
- Excel格式已优化（列宽、换行等）

## 注意事项

1. **配置文件**：首次使用前确保 `references/keywords.json` 存在且格式正确
2. **API访问频率**：默认0.5秒间隔，避免触发反爬机制
3. **内容长度**：默认限制10000字符，超长内容自动截断
4. **网络稳定性**：支持3次重试，指数退避策略
5. **数据隐私**：获取的内容仅用于分析，遵守社区规范
6. **存储空间**：包含content的数据可能较大，请确保磁盘空间充足
7. **配置热加载**：修改配置文件后无需重启程序，下次运行自动生效

## 错误处理

**常见错误及解决方案**：

1. **配置文件错误**：
   - 错误：`配置文件格式错误` → 检查JSON格式是否正确
   - 错误：`配置文件缺少必需项` → 确保包含 `initial_filter` 和 `deep_filter`

2. **Excel字段缺失**：检查输入文件是否包含所有必需字段
3. **API请求失败**：检查网络连接，稍后重试
4. **内存不足**：减少并发数或分批处理数据
5. **权限问题**：确保有读取输入文件和写入输出目录的权限
6. **关键词匹配问题**：调整 `match_mode` 或关键词列表