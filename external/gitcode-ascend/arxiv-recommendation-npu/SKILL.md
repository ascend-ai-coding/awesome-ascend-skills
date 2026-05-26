---
name: external-gitcode-ascend-arxiv-recommendation-npu
description: 自动化推荐系统论文发现流水线。抓取 arxiv 推荐论文，检测源码，生成待迁移任务清单，由 npu-model-migration skill
  完成 NPU 适配。
keywords:
- arxiv
- 推荐系统
- NPU
- 论文
- 迁移
- collaborative filtering
- recommendation
original-name: arxiv-recommendation-npu
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

> ⚠️ **注意**：这是公开版本，自带环境变量配置，可自定义路径：
> - `ARXIV_RECO_DIR`: 论文存储目录（默认: `./arxiv-recommendation-models`）
> - `NPU_MIGRATION_SKILL_PATH`: npu-model-migration skill 路径（默认: `../npu-model-migration`）

# arxiv-recommendation-npu 自动流水线
keywords:
    - arxiv
    - 推荐系统
    - NPU
    - 论文
    - 迁移
    - collaborative filtering
    - recommendation
---

# arxiv-recommendation-npu 自动流水线

自动化推荐系统论文发现与任务生成。

## 触发条件

当用户请求：
- "arxiv 推荐论文"
- "帮我查询最新的推荐论文并适配到NPU"
- "latest recommendation papers npu"
- "找一下一周内的推荐论文"
- "找一下最近一个月的推荐相关论文"

**时间范围规则**：
- 用户明确指定时间范围 → 按用户指定天数搜索
  - 例："找一周内的推荐论文" → `--days 7`
  - 例："最近一个月的推荐相关论文" → `--days 30`
- 用户未明确指定 → 默认搜索最近 **3 天**
  - 例："找最新的推荐论文" → `--days 3`

---

## 前置要求

### 1. 安装依赖

```bash
# 安装 deepxiv (论文搜索)
pip install deepxiv-sdk

# 验证 NPU 环境
python -c "import torch_npu; print('NPU OK')"
```

### 2. 环境检查

```bash

bash scripts/check_npu_env.sh
```

---

## 快速开始

### 一键执行完整流水线

```bash


# 默认搜索最近3天的论文
python scripts/main.py

# 指定搜索时间范围
python scripts/main.py --days 7      # 最近7天
python scripts/main.py --days 30     # 最近30天

# 同时指定时间和结果数量
python scripts/main.py --days 7 --max-results 100
```

或使用脚本：

```bash
# 默认3天
bash scripts/run_full_pipeline.sh

# 指定7天
bash scripts/run_full_pipeline.sh 7
```

---

## 工作流程

本 skill 只负责 **发现论文 + 克隆源码**，NPU 适配由 **npu-model-migration skill** 完成。

```
┌─────────────────────────────────────────────────────────┐
│  arxiv-recommendation-npu skill                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Step 1      │→ │ Step 2      │→ │ Step 3      │    │
│  │ 抓取论文     │  │ 检测源码     │  │ 克隆源码     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                                    │          │
│         │         生成迁移任务清单             │          │
│         │         (migration_task.json)       │          │
│         └──────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
                          ↓ 交接
┌─────────────────────────────────────────────────────────┐
│  npu-model-migration skill                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ 阶段 1      │→ │ 阶段 1.5    │→ │ 阶段 4      │    │
│  │ 目标分析    │  │ 快速尝试    │  │ NPU 验证    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Step 1: 抓取论文

使用 `deepxiv` 搜索推荐系统相关论文。

**时间范围**：
- 用户未指定时，**默认搜索最近 3 天**
- 用户指定时间范围时，按指定天数搜索

```bash
# 默认搜索最近3天
python scripts/main.py

# 指定搜索最近7天
python scripts/main.py --days 7

# 指定搜索最近30天
python scripts/main.py --days 30
```

或在脚本中调用：
```bash
python -c "
import sys
sys.path.insert(0, 'scripts')
from fetcher import search_papers, save_paper_list
from config import TODAY_DIR

# 默认3天
papers = search_papers(days=3, max_results=50)
# 或指定天数
papers = search_papers(days=7, max_results=50)

save_paper_list(papers, f'{TODAY_DIR}/paper_list.md')
print(f'找到 {len(papers)} 篇论文')
"
```

**搜索关键词**：recommendation, recommender system, collaborative filtering, CTR prediction

**输出**：`./arxiv-recommendation-models/{date}/paper_list.md`

> **注意**：paper_list.md 是中间结果记录，方便回顾查看。实际克隆使用内存中的论文数据（包含通过 `enrich_papers_with_details` 补充的 `github_url` 字段）。

---

### Step 2: 检测源码

从论文中提取 GitHub 链接，验证仓库是否有可执行代码。

```bash
python -c "
import sys
sys.path.insert(0, 'scripts')
from fetcher import search_papers
from source_detector import filter_papers_with_code

papers = search_papers(days=7, max_results=50)
papers_with_code = filter_papers_with_code(papers)
print(f'其中 {len(papers_with_code)} 篇有可用源码')
"
```

**验证规则**：
- 优先检查论文的 `github_url` 字段（enrich 阶段提取）
- 备用：从论文 `comments` 和 `abstract` 提取 GitHub 链接
- 调用 GitHub API **递归**检查仓库至少有 **3 个 .py 文件**（含子目录）
- 排除只有 README 的空仓库

---

### Step 3: 克隆源码 & 创建迁移任务

对每篇有源码的论文：
1. 克隆仓库到本地
2. 创建 **迁移任务清单** (`migration_task.json`)
3. 等待 npu-model-migration-skill 处理

```bash
python -c "
import sys
sys.path.insert(0, 'scripts')
from source_detector import clone_repo

github_url = 'https://github.com/owner/repo'
model_dir = './arxiv-recommendation-models/2026-04-13/models/repo'

result = clone_repo(github_url, model_dir)
print(result)
"
```

**迁移任务清单** (`migration_task.json`)：自动生成在模型目录中，详细格式见 [references/output-structure.md](references/output-structure.md)

---

### Step 4: 生成报告

```bash
python -c "
import sys
sys.path.insert(0, 'scripts')
from reporter import generate_daily_report

generate_daily_report(papers_with_code, results, 'daily_report.md')
"
```

**输出**：`./arxiv-recommendation-models/{date}/daily_report.md`

---

## 衔接 npu-model-migration skill

### 报告中的下一步指引

daily_report.md 会包含待迁移论文的信息和下一步操作指引，详细格式见 [references/output-structure.md](references/output-structure.md)

### 手动调用 npu-model-migration

1. 查看报告，找到待迁移的模型目录
2. 进入模型目录
3. 参考 npu-model-migration skill 进行迁移：
   - 阶段 1.5: 快速尝试 (`transfer_to_npu`)
   - 阶段 4: NPU 验证

---

## 配置说明

修改 `scripts/config.py`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `BASE_DIR` | 论文存储根目录 | `./arxiv-recommendation-models` |
| `ARXIV_KEYWORDS` | 搜索关键词 | 推荐系统相关 |
| `MIN_PY_FILES` | 最少 .py 文件数 | 3 |
| `MAX_RETRIES` | 最大重试次数 | 3 |
| `GITHUB_MIRROR` | GitHub 镜像 | ghproxy (国内加速) |

---

## 输出结构

```
./arxiv-recommendation-models/{YYYY-MM-DD}/
├── main.log              # 运行日志
├── paper_list.md         # 论文列表 (Step 1-2)
├── daily_report.md       # 适配报告 (包含迁移指引)
└── models/               # 模型源码
    └── {model_name}/
        ├── migration_task.json  # 迁移任务清单
        └── [原始代码...]
```

---

## 文件结构

```
arxiv-recommendation-npu/
├── SKILL.md                      # 本文件
├── references/                   # 参考文档
│   ├── paper-filtering.md        # 论文筛选规则
│   ├── npu-adaptation.md         # NPU 适配指南 (指向 npu-model-migration skill)
│   └── output-structure.md       # 输出结构说明
└── scripts/                      # 可执行脚本
    ├── main.py                   # 完整流水线入口
    ├── config.py                 # 配置文件
    ├── fetcher.py                # 论文抓取
    ├── source_detector.py        # 源码检测与克隆 + 创建迁移任务
    ├── reporter.py               # 报告生成 (包含迁移指引)
    ├── utils/                    # 工具模块
    │   ├── logger.py
    │   └── date_utils.py
    ├── run_full_pipeline.sh      # 完整流水线脚本
    └── check_npu_env.sh          # 环境检查脚本
```

---

## 依赖 Skill

| Skill | 职责 | 位置 |
|-------|------|------|
| **arxiv-recommendation-npu** | 发现论文、克隆源码、创建任务 | 本 skill |
| **npu-model-migration skill** | NPU 适配验证 | `../npu-model-migration/` |

**衔接方式**：arxiv-recommendation-npu 生成 `migration_task.json`，用户根据报告指引调用 npu-model-migration skill。

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 克隆失败 | 检查网络，或手动 `git clone` 后放入 models 目录 |
| 不会用 npu-model-migration skill | 查看 `../npu-model-migration/SKILL.md` |
| 想只处理特定论文 | 修改 `scripts/main.py` 中 `papers_with_code` 列表手动指定 |

---

## 参考文档

- [references/paper-filtering.md](references/paper-filtering.md) - 论文筛选规则详解
- [references/npu-adaptation.md](references/npu-adaptation.md) - NPU 适配指南 (指向 npu-model-migration skill)
- [references/output-structure.md](references/output-structure.md) - 输出结构说明
- **npu-model-migration-skill**: `../npu-model-migration/SKILL.md`