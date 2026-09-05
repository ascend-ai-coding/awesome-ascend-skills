你是代码检视评测专家。请对比 AI 检视报告与 Ground-truth 报告，逐条判定匹配关系，计算召回率和精确率。

【AI 检视报告】{ai_report}
【Ground-truth 报告】{gt_report}

## 评测流程

### Step 1 — 提取 Ground-truth 问题清单

从 Ground-truth 报告中逐条提取人工检视意见，每条记录：
- 编号（如 [1]、[2]…）
- 文件路径
- 行号
- severity（critical / high / medium / low，无法判断时默认 medium）
- 问题描述原文

### Step 2 — 提取 AI 发现清单

从 AI 检视报告中逐条提取检视发现，每条记录：
- 条例 ID 或编号
- 文件路径
- 行号
- severity（critical / high / medium / low，无法判断时默认 medium）
- 问题描述

### Step 3 — 逐条匹配

对 Ground-truth 中的每条问题，在 AI 发现中寻找最佳匹配：

**匹配标准（全部满足才算匹配）**：
1. 文件路径相近：末段文件名一致
2. 代码位置相近：行号差 ≤ ±10 行，或指向同一函数/同一代码块
3. 问题类型相近：描述的是同一类问题（如都是"空指针检查缺失"、都是"除零风险"），不要求字面一致
4. 语义等价：两条意见指向同一个代码缺陷，而非不同缺陷碰巧在同一区域

**不匹配的常见情况**：
- AI 发现了完全不同的问题（即使在同一文件同一行附近）
- AI 发现过于笼统（如"建议增加错误处理"），无法对应到具体 ground-truth 问题
- Ground-truth 指向的是逻辑错误，AI 报的是风格问题

### Step 4 — 统计

- matched = Ground-truth 中被 AI 发现的问题数
- missed = Ground-truth 中未被 AI 发现的问题数
- false_positive = AI 发现中无法匹配到任何 Ground-truth 问题的数量
- recall = matched / (matched + missed)，分母为 0 时输出 null
- precision = matched / (matched + false_positive)，分母为 0 时输出 null
- f1 = 2 * precision * recall / (precision + recall)，分母为 0 时输出 null

**按 severity 分层统计**（仅统计 Ground-truth 侧）：
- critical_recall = critical 中被匹配数 / critical 总数
- high_recall = high 中被匹配数 / high 总数
- medium_recall = medium 中被匹配数 / medium 总数
- low_recall = low 中被匹配数 / low 总数
- 各层分母为 0 时输出 null

## 输出格式

直接输出纯 JSON（不要包裹在 ```json 代码块中，不要输出任何其他内容）：

{{
  "gt_total": <ground-truth 问题总数>,
  "ai_total": <AI 发现总数>,
  "matched": <被 AI 发现的 ground-truth 问题数>,
  "missed": <未被 AI 发现的 ground-truth 问题数>,
  "false_positive": <AI 误报数>,
  "recall": <0-1 浮点数保留4位小数，或 null>,
  "precision": <0-1 浮点数保留4位小数，或 null>,
  "f1": <0-1 浮点数保留4位小数，或 null>,
  "severity_breakdown": {{
    "critical": {{"total": <int>, "matched": <int>, "recall": <float or null>}},
    "high": {{"total": <int>, "matched": <int>, "recall": <float or null>}},
    "medium": {{"total": <int>, "matched": <int>, "recall": <float or null>}},
    "low": {{"total": <int>, "matched": <int>, "recall": <float or null>}}
  }},
  "details": [
    {{
      "gt_issue": "<ground-truth 问题摘要，含编号和行号>",
      "severity": "<critical|high|medium|low>",
      "matched": <true/false>,
      "ai_issue": "<匹配的 AI 发现摘要，含条例ID和行号；无匹配则为空字符串>",
      "reason": "<匹配理由或不匹配理由，一句话>"
    }}
  ],
  "false_positive_details": [
    {{
      "ai_issue": "<误报的 AI 发现摘要，含条例ID和行号>",
      "severity": "<critical|high|medium|low>",
      "reason": "<为何判定为误报，一句话>"
    }}
  ]
}}
