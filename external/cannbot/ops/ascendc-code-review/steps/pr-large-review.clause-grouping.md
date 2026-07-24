# 条例分组 — 小组策略 + 负载感知波次规划（大型 PR 检视）

派发为子 Agent 执行。基于 global-pre-scan 产出的 per-group matched_rules，按小组策略分组，再按负载感知算法构建波次。

## 派发

```
Agent({
  subagent_type: "general",
  model: "haiku",
  description: "条例分组 + 波次规划",
  prompt: "条例分组 + 负载感知波次规划

【输入】
- per-group matched_rules: {matched_rules}
- 各组文件数: {group_file_counts}

【执行要求】
严格按本文件「执行流程」中定义的步骤执行，产出全局波次规划表。禁止生成报告文件。"
})
```

---

## 执行流程（子 Agent 执行指南）

## 前置输入

global-pre-scan 产出的 per-group matched_rules，每组已知：file_group 名、文件数、激活的规则文件、匹配的条例ID 列表。

## 执行流程

### Step 1 — 条例归类

将每个 file_group 的 matched_rules 按类别归类：

| 条例ID 前缀 | 类别 | 优先级 |
|------------|------|--------|
| SEC（cpp-secure）中的数值安全/内存安全/输入验证 | 高危安全 | 1 |
| TOPK-8 等 TOPK 安全类 | 高危安全 | 1 |
| API（ascendc-api） | API使用 | 2 |
| SEC（cpp-secure）中的资源管理/并发安全/类型安全 | 一般安全 | 2 |
| GEN（cpp-general） | 通用规范 | 4 |
| SIMT（simt-api-analysis） | 领域规则 | 3 |
| MC2（mc2-specific） | 领域规则 | 3 |
| PERF / PREC / TIL（ascendc-perf） | 性能 | 4 |
| CMP（compile-secure） | 编译 | 5 |
| PY（python-secure） | Python | 5 |

### Step 2 — 小组策略打组

按类别分组，每组上限：

| 类别 | 每组合上限 |
|------|----------|
| 高危安全 | **2 条** |
| API使用 / 一般安全 | **2 条** |
| 领域规则 | **2 条** |
| 通用规范 / 性能 / 编译 / Python | **3 条** |

每组的文件列表维持 file_group 的原始文件（≤5 文件，由 file-split 保证）。

每组打标：
```
{
  group_id: "kernel_G1_安全_01",
  file_group: "kernel_G1",
  file_list: ["file1.cpp", "file2.h", ...],  // ≤5 文件
  rule_ids: ["SEC-2.1", "SEC-2.3"],          // 2-3 条
  priority: 1,
  file_count: 5,
  rule_count: 2,
  estimated_load: 10                           // file_count × rule_count
}
```

### Step 3 — 负载感知波次构建

```
1. 将所有 rule_group 按 priority 升序排列（priority 1 先）

2. 同 priority 内按 estimated_load 降序排列
   （重负载组先派发，避免轻负载组全跑完了重负载还在等）

3. 贪心构建波次:
   wave = []
   for group in sorted_groups:
     if len(wave) < 10:
       wave.append(group)
     else:
       开始新 wave

4. 组间均衡检查（每波构建完成后）:
   统计本波中各 file_group 的占比
   若某 file_group 在单波中 >5 组（即 >50%）:
     将其超出 5 组的 group 推迟到下一波
     （避免某文件组独占一波，其他组的检视饿死）

5. 输出波次规划表
```

### 输出格式

```
全局波次规划:

Wave 1（10组，优先级1-2）:
  kernel_G1_安全_01: [SEC-2.1, SEC-2.3] | 5文件 | load=10
  kernel_G2_安全_01: [SEC-2.1, SEC-3.1] | 3文件 | load=6
  host_G1_安全_01:   [SEC-2.1, SEC-4.1] | 6文件 | load=12
  kernel_G1_API_01:  [API-3, API-7]      | 5文件 | load=10
  ...

Wave 2（8组，优先级2-3）:
  ...

Wave 3（7组，优先级3-4）:
  ...

共 {G} 组，{W} 波
```

## 约束

- 主 Agent 直接执行，不派发子 Agent（纯计算，无需读代码）
- 每文件组单波占比 ≤50%（均衡硬约束）
- 不生成报告文件，波次规划表直接用于 Stage 2
