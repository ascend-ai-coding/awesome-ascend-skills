# generation_validation

生成验证流水线模块 - 验证生成文档质量、一致性检查。

## STRUCTURE
```
generation_validation/
├── __init__.py                     # 有包初始化 (唯一有 __init__.py 的模块)
├── main.py                         # 验证主入口
├── config.py                       # 配置 (TEST_PREFIX, DOC_PARTS, 置信阈值)
├── dashboard.py                    # 并行 doc_part 实时面板 (--dashboard)
├── core/
│   └── models.py                   # 16个Pydantic模型 (Evidence, Anchor, DocBlock, etc.)
├── generator/
│   ├── generator.py                # DocumentGenerator
│   ├── writer.py                   # TextWriter
│   └── get_example.py              # 574行, C++示例匹配/提取
├── validator/
│   ├── validator.py                # DocumentValidator
│   ├── consistency.py              # ConsistencyChecker
│   ├── metric.py                   # ModificationMetric
│   └── example_validator.py
├── evidence/
│   ├── extractor.py                # 944行, EvidenceExtractor (渐进式提取)
│   ├── parser.py                   # EvidenceParser
│   ├── ranker.py                   # EvidenceRanker
│   ├── chip_mapper.py
│   └── support_verifier.py
├── style/
│   ├── summarizer.py               # StyleSummarizer
│   ├── extractor.py                # StyleExtractor
│   └── loader.py                   # TemplateLoader
├── pipeline/
│   ├── stages.py                   # 4阶段类: Evidence→Style→Generation→Validation
│   ├── deterministic_checks.py     # 确定性检查
│   └── helpers.py
└── ui/
    └── cli.py                      # 1199行, Rich CLI交互界面
```
(LLM 客户端不在本模块内 — 共用全局 `src/llm/client.py` 的 `LLMClient`)

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| 验证主入口 | `main.py` | 4阶段流水线 |
| 数据模型定义 | `core/models.py` | 所有Pydantic BaseModel |
| 配置常量 | `config.py` | DOC_PARTS, TEST_PREFIX, 置信阈值 |
| 渐进式证据提取 | `evidence/extractor.py` | 自评分: 0.9→0.8→...→0.5 |
| 示例匹配 | `generator/get_example.py` | 多策略匹配 (精确→相似→describe) |
| CLI交互 | `ui/cli.py` | 1199行, Rich/questionary |
| 一致性验证 | `validator/consistency.py` | ConsistencyChecker |

## CONVENTIONS
- **TEST_PREFIX**: `test_` 识别测试/示例文件
- **ACLNN_PREFIX**: `aclnn` 用于算子名匹配
- **唯一有 `__init__.py`** 的模块 — 使用相对import + `__all__` re-export facade
- **Pydantic 数据模型**: 所有数据结构使用 BaseModel (非 dict)
- **4阶段流水线**: EvidenceStage → StyleStage → GenerationStage → ValidationStage

## ANTI-PATTERNS
- **无测试框架** - 虽有 TEST_PREFIX 但无 pytest/unittest
- **示例文件依赖** - 依赖外部 C++ 示例文件
- **ui/__init__.py 为空** - 死模块边界 (0 bytes)
- **CLI过重** - cli.py 1199行, run_full_pipeline 单函数500行

## NOTES
- 数据驱动：证据位于 `data/evidences/example/`
- 风格数据：位于 `resources/style/`
- 流水线入口结构清晰，适合独立运行
