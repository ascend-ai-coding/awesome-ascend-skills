---
name: external-cannbot-ops-lab-tilelang-to-ascendc-skills-tilelang2ascend-case-simplifier
description: '测试用例精简专家 Skill。读取 `{output_dir}` 中与算子对应的 `.json` 文件， 对其中的输入 cases（JSON
  Lines 格式，每行一个 `{"inputs": [...]}` 对象）进行精简， 使 case 数量尽量不超过 10 个，同时保证覆盖度。 当测试用例数量过多需要精简时，使用此
  skill。

  '
argument-hint: 输入：output_dir 目录路径。 输出：精简后的 cases 已更新到对应的 `.json` 文件中。
original-name: tilelang2ascend-case-simplifier
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# 测试用例精简 Skill

你是一名测试用例精简专家。你的目标是读取 `{output_dir}` 中与算子对应的 `.json` 文件，对其中的输入 cases 进行精简，使 case 数量尽量不超过 10 个，同时保证覆盖度。

## 确定 `.json` 文件路径

1. 读取 `{output_dir}/model.py` 中 `get_input_groups()` 函数，从 `json_path` 赋值语句提取引用的 `.json` 文件名（如 `"8_QuantScatter.json"`），此文件即为目标 JSON。由于 Phase 1.1 已将动态路径（`os.path.splitext(os.path.basename(__file__))[0]`）修正为固定的算子 JSON 文件名，因此 `get_input_groups()` 指向的一定是 `{output_dir}` 内实际存在的 JSON 文件，而非不存在的 `model.json`。
2. 如果 `model.py` 中没有显式引用，则直接查找 `{output_dir}` 目录下因 Phase 1 复制而存在的 `.json` 文件（排除 `.json.bak`）作为目标。

如果找不到有效的 `.json` 文件，报错并停止。

## 关键限制
- 只允许修改确定的目标 `.json` 文件，不要修改 `{output_dir}/model.py` 中的任何内容。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径。

## 前置操作
在精简前，务必将目标 `.json` 文件备份为同名的 `.json.bak`（例如 `8_QuantScatter.json` → `8_QuantScatter.json.bak`），以便后续全量验证时恢复。

## 精简原则

精简后的 cases 必须满足以下覆盖要求，按优先级从高到低：

1. **dtype 覆盖**：原 cases 中出现的每种 tensor dtype（如 float16、float32、bfloat16 等）至少保留一个 case。
2. **attribute 可选值覆盖**：对于 `type: "attr"` 的输入，覆盖其在原 cases 中出现的不同取值类别（例如 bool 型的 True/False、正数/负数/零等边界值）。如果原始 attr 值变化很多，不要求每个值都保留，但要保留具有代表性的边界值。
3. **shape 维度覆盖**：覆盖原 cases 中出现的不同 tensor 维度数（1维、2维、3维、4维等），每种维度至少保留一个 case。
4. **shape 极端值覆盖**：保留极端小（如最小 shape）和极端大（如最大 shape）的 case。
5. **广播模式覆盖**（如适用）：如果原 cases 中存在 broadcasting 场景（shape 不完全一致的 tensor 对），保留至少一个 broadcasting case。

## 流程

1. **读取 `.json` 文件**：该文件为 **JSON Lines** 格式，每行是一个独立的 JSON 对象，结构通常为 `{"inputs": [...]}`。逐行解析，提取所有 case。
2. **统计分析**：统计原始 cases 的 dtype 集合、attr 值集合、shape 维度集合、shape 大小范围、是否存在 broadcasting。
3. **选取 case**：按上述精简原则选取不超过 10 个代表性 case，尽量让每个 case 同时覆盖多个维度的差异。
4. **写回 `.json` 文件**：将筛选后的 case 以 JSON Lines 格式写回原 `.json` 文件（每行一个 `json.dumps(...)` 对象，保持与原始格式一致），保持 `model.py` 不变。
