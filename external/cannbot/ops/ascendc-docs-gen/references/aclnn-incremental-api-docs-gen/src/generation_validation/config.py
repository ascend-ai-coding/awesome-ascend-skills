# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Configuration module - centralized configuration variables for the project."""

from pathlib import Path

# -------------------------------
# 路径类
# -------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EVIDENCE_DIR = DATA_DIR / "evidences"
# 从文档中抽取的证据的路径
EVIDENCE_EXAMPLE_DIR = EVIDENCE_DIR / "example"
# 文档写作风格的路径
RESOURCES_DIR = PROJECT_ROOT / "resources"
# "芯片型号对应表"的路径
CHIP_MAP_YAML = RESOURCES_DIR / "chip_map.yaml"
# "文档模板配置"的路径
TEMPLATES_YAML = RESOURCES_DIR / "templates.yaml"

OUTPUT_DIR = DATA_DIR / "outputs"
LOG_DIR = DATA_DIR / "logs"

# -------------------------------
# 生成 + 校验的可拓展内容及threshold
# -------------------------------

# 校验中 EvidenceBatchManager 支持最大证据 token 长度
MAX_TOKENS = 20000
CHUNK_OVERLAP = 400

TEST_PREFIX = "test_"
ACLNN_PREFIX = "aclnn"

EXAMPLE_PREFIX = "## 调用示例\n\n示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。\n"

DOC_PARTS = [
    "support",
    "summary",
    "function_proto",
    "sub_function_hash",
    "constraint",
    "example",
]
DOC_PART_PATTERN = (
    r"^(support|summary|function_proto|sub_function_hash|constraint)\.json$"
)

DOC_PART_NAMES_ZH = {
    "support": "## 产品支持情况",
    "summary": "## 功能说明",
    "function_proto": "## 函数原型",
    "sub_function_hash": "## 各接口函数说明",
    "constraint": "## 约束说明",
    "example": "## 调用示例",
}


def format_doc_part(name: str) -> str:
    return DOC_PART_NAMES_ZH.get(name, name)


def format_doc_parts(parts: list[str]) -> str:
    return ", ".join(format_doc_part(p) for p in parts)


# doc-part:"调用示例"的路径名 通常为“abs/examples/”
EXAMPLES_DIR = "examples"

# 校验失败后的重生成次数
MAX_REGENERATION_RETRIES = 3
EVIDENCE_CONFIDENCE_THRESHOLD = 0.8

# 一致性校验通过阈值：LLM 对生成文档打出的 5 维加权均分（factual_accuracy * 0.3
# + formula_correctness * 0.2 + numerical_accuracy * 0.2 + description_consistency * 0.15
# + logical_coherence * 0.15）达到此值即判 consistency_passed=True。
# issues 列表仅作为诊断信息保留，不参与 pass 决策；unresolvable issue 在上层
# retry 循环中以 evidence_conflict 形式提前阻断，与本阈值正交。
CONSISTENCY_PASS_THRESHOLD = 0.8

# ModificationMetric 中"高修改"判定的 edit_distance_ratio 阈值。
# edit_distance_ratio = 1 - difflib.SequenceMatcher.ratio()，范围 [0, 1]：
#   0.0 = 一字未改，1.0 = 完全不同。
# 超过此值时，需要 avg_confidence >= METRIC_HIGH_CONFIDENCE_THRESHOLD 才允许通过，
# 否则 metric_passed="FAIL"。
METRIC_HIGH_MODIFICATION_THRESHOLD = 0.5

# ModificationMetric 中"高置信度"判定的 avg_confidence 阈值。
# avg_confidence 是 evidences_extracted 中所有 evidence 的 confidence 均值。
# 当 edit_distance_ratio > METRIC_HIGH_MODIFICATION_THRESHOLD 时，
# 必须 avg_confidence >= 此值 metric 才判 PASS（大改动要有强证据支撑）。
METRIC_HIGH_CONFIDENCE_THRESHOLD = 0.7

# 后置分析（_step_post_analyze）的"无实际变更"判定阈值。
# 当某个 doc_part 的所有 validation 的 edit_distance_ratio 都严格低于此值时，
# 视为 LLM 生成未真正改动文档（假阳性），阻止回填并标记为失败。
# 1% 约等于空白字符级别的差异，对实际文档而言等于没改。
# 注：缺省值（edit_distance_ratio 缺失）按 1.0 处理，相当于"视为已改过"不触发阻断。
POST_ANALYZE_NO_CHANGE_THRESHOLD = 0.01

# 预校验开关：是否在 full 模式下先对现有文档做一致性验证
ENABLE_PRE_VALIDATION = False

# 证据文件过滤开关：是否只保留 modified_files 相关的 anchor（与预校验解耦）
# 过滤策略：should_filter = ENABLE_EVIDENCE_FILE_FILTER or pre_val_passed_map[doc_part]
#   ENABLE_EVIDENCE_FILE_FILTER=True  + 任意预校验结果      → 按 modified_files 过滤
#   ENABLE_EVIDENCE_FILE_FILTER=False + 预校验通过          → 按 modified_files 过滤
#   ENABLE_EVIDENCE_FILE_FILTER=False + 预校验未通过/未运行 → 保留全量证据
ENABLE_EVIDENCE_FILE_FILTER = True

# 并行化参数
EXTRACT_MAX_WORKERS = 4
VALIDATION_MAX_WORKERS = 4
GENERATION_MAX_WORKERS = 4
GEN_VAL_MAX_WORKERS = 4
SUPPORT_MAX_WORKERS = 4

# -------------------------------
# 证据阶段
# -------------------------------

EVIDENCE_STAGES = ["file", "func", "note", "section"]
