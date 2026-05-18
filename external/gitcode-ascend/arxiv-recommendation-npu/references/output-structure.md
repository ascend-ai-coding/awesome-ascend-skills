# 输出目录结构

## 根目录

```
./arxiv-recommendation-models/
└── {YYYY-MM-DD}/
    ├── main.log           # 运行日志
    ├── paper_list.md      # 论文列表 (Step 1)
    ├── daily_report.md    # 适配报告 (Step 4)
    └── models/            # 模型源码目录
        ├── model_1_{name}/
        ├── model_2_{name}/
        └── ...
```

## 模型目录结构

```
models/{model_name}/
├── migration_task.json  # 迁移任务清单 (Step 3 自动生成)
├── [原始代码...]        # 克隆的源码
└── __pycache__/         # Python 缓存 (运行时生成)
```


## migration_task.json 格式

```json
{
  "task_id": "20260413103045",
  "created_at": "2026-04-13T10:30:45",
  "paper": {
    "title": "论文标题",
    "arxiv_id": "2604.12345",
    "github_url": "https://github.com/owner/repo",
    "pdf_url": "https://arxiv.org/pdf/2604.12345",
    "authors": ["作者1", "作者2"]
  },
  "model_dir": "/path/to/model",
  "entry_script": "train.py",
  "status": "pending"
}
```

### status 枚举

| 状态 | 说明 |
|------|------|
| `pending` | 待处理 |
| `migrating` | 迁移中 |
| `completed` | 迁移完成 |
| `failed` | 迁移失败 |

## daily_report.md 结构

```markdown
# 2026-04-13 推荐论文 NPU 适配报告

## 统计

- 论文总数: 12
- 有源码: 5
- 适配成功: 3
- 适配失败: 2

## 成功案例

### 1. L2UnRank
- arXiv: 2511.06803
- GitHub: https://github.com/Juniper42/L2UnRank
- NPU 卡: 2
- 运行时间: 12.5s

## 失败案例

### 1. XXXModel
- 原因: xxx
```

## main.log 结构

```
2026-04-13 09:00:00,123 - main - INFO - 开始执行...
2026-04-13 09:00:01,456 - fetcher - INFO - 搜索关键词: recommendation
...
```