# Kernel 侧 API 文档预研

## 定位

API 预研报告是检视子 Agent 查阅 API 物理约束的主要来源，覆盖代码中核心 API 的对齐要求、参数限制、配对规则和精度约束。

file-review 和 pr-review 共用。Phase 0 与 code-summarize + clause-routing **并行**执行。

仅当 code-summarize 或 clause-routing 判定侧别包含 Kernel 时触发；纯 Tiling 侧跳过。

## 派发

```
Agent({
  subagent_type: "general",
  model: "opus",
  description: "Kernel API 文档预研",
  prompt: "Kernel 侧 API 文档预研

【输入】
- 代码文件：{file_input}（Kernel 侧文件列表）
- 概要输出路径：{api_prestudy_path}

【执行流程】

Step 1 — 提取代码中使用的 API

1.1 Read 代码文件，提取所有 AscendC:: 命名空间下的函数调用。
1.2 按以下核心 API 清单筛选需要预研的 API（仅保留代码中实际使用的）：

| 类别 | API | 学习重点 |
|------|-----|---------|
| 数据搬运 | DataCopy, DataCopyPad | 32 字节对齐要求、同步机制（EnQue/DeQue 配对） |
| 内存管理 | InitBuffer, AllocTensor, FreeTensor, EnQue, DeQue | 配对要求、UB 容量限制、生命周期 |
| 向量计算 | Add, Sub, Mul, Div, Cast | 参数限制（repeatTimes≤255）、RoundMode 正确性 |
| 归约操作 | ReduceSum, ReduceMax | FP32 中间精度保护、累加器 dtype |

1.3 若代码中出现核心清单外的 API（如 Compare, Exp, Sqrt），也一并记录。

Step 2 — 查阅官方文档

对 Step 1 提取的每个 API，使用 /ascendc-docs-search skill 查阅最新官方文档，提取：
- 函数签名（参数类型、返回值）
- 对齐要求（字节数）
- 参数限制（repeatTimes、dtype 约束、shape 约束）
- 配对/同步要求（哪些 API 必须成对使用）
- 精度约束（中间精度、累加器要求）
- RoundMode 选项及默认值

Step 3 — 输出预研报告

将预研结果写入 {api_prestudy_path}，格式如下：

# API 预研报告

## 数据搬运

### DataCopy
- 对齐要求：src/dst 地址必须 32 字节对齐
- 同步机制：需配合 EnQue/DeQue 使用
- 参数限制：...
- 代码中的使用位置：{file}:{line}

### DataCopyPad
- ...

## 内存管理

### InitBuffer
- ...

（每个 API 一个子章节）

## 未匹配 API（代码中使用但不在核心清单中的）

- {API名}: {一行描述}

禁止生成报告文件以外的输出。"
})
```

## 输出

- 预研报告路径：`./operators/{operator_name}/api_prestudy.md`
- 后续 Phase 1 的 clause-review sub-agent 在检视 API-\*、PERF-\*、PREC-\* 条款时，Read 此文件获取 API 约束上下文，减少重复查阅
