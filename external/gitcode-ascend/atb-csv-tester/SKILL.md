---
name: external-gitcode-ascend-atb-csv-tester
description: '运行 ATB (Ascend Transformer Boost) CSV 测试。当用户需要执行 CSV 格式的 ATB 测试用例、 验证算子正确性、或运行任何ATB下的
  CSV 测试文件时调用此技能。 需配合 CANN 环境和已编译的 ATB 测试框架使用。

  '
keywords:
- atb
- csv
- test
- csv-test
- ascend-transformer-boost
- 昇腾
- 算子测试
metadata:
  author: ascend-transformer-boost-team + Claude Code + deepseek-v4-pro
  version: 1.6.0
  created: '2026-04-17'
  updated: '2026-04-29'
  skill-type: test
  allowed-tools: Bash(docker|bash|cd|ls|grep|find|echo|cat) Read(*)
  gates:
  - id: gate-5
    description: 全量回归门控
    trigger: Phase 5b 完成后，所有相关 CSV 必须全部通过，方可进入 Phase 6 交付
hooks:
  PreToolUse:
  - matcher: Write|Edit|Bash
    hooks:
    - type: command
      command: ([ -z "$CANN_PATH" ] || [ ! -f "$CANN_PATH/set_env.sh" ]) && echo '[PATH
        CHECK] CANN_PATH 未设置或无效，请先向用户获取 CANN 路径' >&2; ([ -z "$ATB_REPO_PATH" ] ||
        [ ! -d "$ATB_REPO_PATH" ]) && echo '[PATH CHECK] ATB_REPO_PATH 未设置或无效，请先向用户获取
        ATB 路径' >&2 || true
  PostToolUse:
  - matcher: Write|Edit
    hooks:
    - type: command
      command: echo '[REMINDER] 若 910B 测试全部通过，请提示用户在 Ascend950 设备上手动运行 CSV 测试。'
  - matcher: Bash
    hooks:
    - type: command
      command: echo '[PHASE 5A→5B REMINDER] 若 Phase 5a 设计用例已全部通过，请输出全量回归 CSV 列表，提示用户进入
        Phase 5b。全量回归未完成前，禁止交付。'
    - type: command
      command: 'echo ''[GATE 5 REMINDER] 若刚完成 CSV 测试，请确认: (1) 所有相关 CSV 已列出? (2) 是否已全部运行并通过?
        未完成全量回归前，禁止交付。'''
  Stop:
  - hooks:
    - type: command
      command: echo '[GATE 5 CHECK] 提醒：所有相关 CSV 测试文件是否已全部运行并通过？若未全部通过，严禁进入 Phase 6
        交付。'
original-name: atb-csv-tester
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ATB CSV Tester

## 功能概述

运行 ATB CSV 测试用例，验证算子正确性。需配合 CANN 环境和已编译的 ATB 测试框架。

## 路径约束

**必须从用户处获取以下路径**：
- `<CANN_PATH>`: CANN 安装路径（如 `/usr/local/Ascend/ascend-toolkit/latest`）
- `<ATB_REPO_PATH>`: ATB 仓库路径

**若路径无效，立即停止并报错。**

## 调用时机

- 运行 ATB CSV 测试（如 `linear.csv`）
- 验证已迁移算子的正确性
- CANN 环境下的算子级别功能测试

---

## 设备支持说明

**当前测试环境仅支持 Ascend910B 设备。**

### 运行规则

- 执行测试时**仅运行 SocVersion 包含 Ascend910B 的用例**（测试框架自动匹配当前设备 SocVersion）
- CSV 测试用例生成时，已根据设备差异**分离设计**（参见 `atb-csv-testcase-generator` 的多设备测试设计原则）

### 910B/950 差异用例分离要求

**⚠️ 当 910B 和 950 在以下任何方面存在差异时，必须拆分为独立用例、各自指定 SocVersion：**

| 差异维度 | 拆分要求 |
|---------|---------|
| dtype 支持 | 各自支持的 dtype 分别设计正例和反例 |
| format 支持 | 各自合法的 format 分别测试 |
| 参数约束 | 各自合法/非法参数值分别验证 |
| 维度限制 | 各自边界值/UB 上限分别设计 |
| ExpectedError | 同一输入可能触发不同错误码，分别设计反例 |
| 性能基准 | 性能测试分别标记各自 SocVersion |

**严禁将存在功能差异的用例合并为 `Ascend910B;Ascend950`**——合并 SocVersion 仅限两个设备完全一致的场景。

### 910B 测试完成后的提示

**Ascend910B 测试全部通过后，必须提示用户：**

> "910B 测试已全部完成。CSV 测试用例中已包含针对 Ascend950 的独立用例（SocVersion 不含 910B），请在 Ascend950 设备上手动运行 CSV 测试以验证 950 兼容性："
> ```bash
> cd <ATB_REPO_PATH>/tests/framework/python/CsvOpsTestTool
> python3 atb_csv_ops_test.py -i <CSV_FILE_PATH>
> ```

- 950 设备测试命令相同，用户仅需将测试环境切换至 Ascend950 机器后重新执行
- 950 测试中若有失败，应对比 910B 结果分析是否为设备差异导致的预期行为

---

## Gate 5: 全量回归门控

**强制规则：进入 Phase 6 交付前，所有与目标算子相关的 CSV 文件必须全部运行并通过。**

### 相关 CSV 枚举

以目标算子名称（如 `layer_norm`）为基准，枚举所有相关 CSV：

```bash
find <ATB_REPO_PATH>/tests/apitest/opstest/csv/ -name '{op_name}*.csv' -type f | sort
```

| 匹配规则 | 说明 | 示例 (layer_norm) |
|---------|------|-------------------|
| 精确匹配 `{op_name}.csv` | 主算子 CSV | `layer_norm.csv` |
| 前缀匹配 `{op_name}_*.csv` | 变体/扩展 CSV | `layer_norm_test.csv` |

### 门控检查流程

```
[Phase 5b 开始]
    │
    ▼
[步骤 1: 枚举相关 CSV] → find csv/ -name '{op}*.csv'
    │
    ▼
[步骤 2: 逐个运行] → python3 atb_csv_ops_test.py -i <每个 CSV>
    │
    ▼
[步骤 3: 汇总结果] → 记录各 CSV 通过/失败/跳过
    │
    ▼
[Gate 5 检查]
    ├─ 全部通过 → Gate 5 通过 → 可进入 Phase 6
    └─ 任一失败 → Gate 5 阻断 → 返回修复
```

### 门控阻断时的话术

```markdown
## Gate 5 阻断 — 全量回归未通过

| CSV 文件 | 总用例 | 通过 | 失败 | 状态 |
|---------|--------|------|------|------|
| {op}_test.csv | 31 | 31 | 0 | ✅ |
| {op}.csv | 90 | - | - | 未执行 |

**在 Gate 5 通过前，禁止进入 Phase 6 交付。**

下一步：运行未执行的 CSV 文件，修复失败用例，重新执行全量回归。
```

---

## Phase 5a 通过后：自动提示进入全量回归

**当 Phase 5a 设计用例全部通过后，Agent 必须自动输出以下提示：**

```markdown
## Phase 5a 全部通过 — 请进入 Phase 5b 全量回归

Phase 5a 设计用例已全部通过（N/N）。

### Phase 5b 需执行的全量回归 CSV 列表：

| # | CSV 文件 | 路径 | 
|---|---------|------|
| 1 | `{op}_test.csv` | `tests/apitest/opstest/csv/{op}_test.csv` |
| 2 | `{op}.csv` | `tests/apitest/opstest/csv/{op}.csv` |

### 执行命令：
```bash
find <ATB_REPO_PATH>/tests/apitest/opstest/csv/ -name '{op}*.csv' -type f | while read f; do
    echo "=== Running: $f ==="
    python3 atb_csv_ops_test.py -i "$f"
done
```

**在全量回归全部通过（Gate 5b 通过）之前，禁止进入 Phase 6 交付。**
```

### Phase 5a/5b 测试策略对照

| 阶段 | 测试范围 | CSV | 执行方式 | 迭代数 |
|------|----------|-----|---------|--------|
| Phase 5a Step 1 | 设计用例（正例+反例） | `_func.csv` | **前台**，等待完成 | 默认 1 |
| Phase 5a Step 2 | 设计用例（性能） | `_perf.csv` | **后台 subagent**，逐个 `-n N:N` | 默认 400 |
| Phase 5b 全量回归 | 所有相关 CSV | `_func.csv` + `_perf.csv`（或 `_test.csv`） | 前台逐个 | 默认 1 / 400 |

---

## 详细内容

<details>
<summary>🔍 查看详细内容</summary>

### 前置条件

- CANN 环境已安装并 source
- ATB 测试框架已编译（`libatb_test_framework.so` 存在）
- ATB 源码仓库已克隆

### 执行步骤

#### 第1步：进入 Docker 容器环境

```bash
docker exec -it <DOCKER_NAME> bash
```

#### 第2步：Source CANN 环境

```bash
source <CANN_PATH>/set_env.sh
```

#### 第3步：Source ATB 输出环境

```bash
source <ATB_REPO_PATH>/output/atb/set_env.sh
```

#### 第4步：验证测试框架

```bash
ls -lh $ATB_HOME_PATH/lib/libatb_test_framework.so
```

**若文件不存在**，先执行编译：

```bash
cd <ATB_REPO_PATH>
bash scripts/build.sh testframework
source output/atb/set_env.sh
```

#### 第5步：运行 CSV 测试

**方式一：使用技能脚本**

```bash
bash <SKILLS_DIR>/atb-csv-tester/scripts/run-csv-tests.sh \
    --atb-repo-path <ATB_REPO_PATH> \
    --csv-file <CSV_FILE_PATH> \
    --case-range <CASE_RANGE>
```

**方式二：手动执行**

```bash
cd <ATB_REPO_PATH>/tests/framework/python/CsvOpsTestTool
python3 atb_csv_ops_test.py -i <CSV_FILE_PATH>
```

## 测试运行策略

**⚠️ 正例/反例与性能测试需区分对待，`-t` 参数行为不同。**

### `-t` 参数实际行为

`atb_csv_ops_test.py` 根据 `TestType` 列自动决定运行次数：

```python
if TestType == "Performance":
    实际次数 = 400 if args.times < 400 else args.times + 1
else:
    实际次数 = args.times  # 默认 1
```

- 性能测试前 200 次迭代跳过精度统计（`SKIP_PERFORMANCE_CASE_NUM = 200`），仅从第 201 次开始累积

### 按用例类型推荐策略

| 用例类型 | 推荐 `-t` | 实际运行次数 | 用途 |
|---------|----------|------------|------|
| **正例** (ExpectedError=NO_ERROR, TestType≠Performance) | 不指定（默认 `-t 1`） | 1 | 功能正确性验证 |
| **反例** (ExpectedError≠NO_ERROR) | 不指定（默认 `-t 1`） | 1 | 错误分支覆盖验证 |
| **性能开发验证** (TestType=Performance) | 不指定 | 400（200 预热 + 200 统计） | 开发期快速验证性能和精度 |
| **性能交付采集** (TestType=Performance) | 不指定 | 400 | CI/交付前完整采集 |

> **注意**：`-t 50` 对 Performance 用例**不生效**（50 < 400 → 回退为 400）。若确实需要减少性能迭代数，必须将 CSV 中对应用例的 `TestType` 从 `Performance` 改为空或 `Function`，但这会跳过性能阈值检查。

### Phase 5a 测试策略（双文件 + 两步流程）

**⚠️ Phase 5a 必须分两步执行：前台功能验证 → 后台性能测试。**

#### Step 1: 前台运行功能 CSV（正例+反例）

```bash
cd <ATB_REPO_PATH>/tests/framework/python/CsvOpsTestTool
python3 atb_csv_ops_test.py -i <CSV_DIR>/{op}_func.csv
```

- 不指定 `-t`，默认 1 次，秒级完成
- 确认所有正例通过、反例返回预期错误码
- 必须等待完成后再进入下一步

#### Step 2: 后台 Subagent 运行性能 CSV

**使用 subagent 后台执行**，避免阻塞主流程和 OOM：

```python
Agent(
    description="运行 {op} 性能测试",
    prompt="""
逐个运行 {op}_perf.csv 中的性能测试用例，使用 -n 避免 OOM：

cd {ATB_REPO_PATH}/tests/framework/python/CsvOpsTestTool
source {CANN_PATH}/set_env.sh
source {ATB_REPO_PATH}/output/atb/set_env.sh

# 获取性能用例总数
lines=$(tail -n +2 {ATB_REPO_PATH}/tests/apitest/opstest/csv/{op}_perf.csv | wc -l)

# 逐个运行每个性能 case
for i in $(seq 1 $lines); do
    echo "=== 运行性能 case $i/$lines ==="
    python3 atb_csv_ops_test.py -i {ATB_REPO_PATH}/tests/apitest/opstest/csv/{op}_perf.csv -n $i:$i
done

完成后汇总：通过/失败/耗时。
""",
    subagent_type="general-purpose",
    run_in_background=True
)
```

#### 流程时序

```
Phase 5a 开始
    │
    ├─ Step 1: 前台跑 _func.csv（~30秒）
    │     └─ 全部通过 ✓
    │
    ├─ Step 2: 启动后台 subagent 跑 _perf.csv（~10分钟）
    │     └─ 逐个 -n N:N 避免 OOM
    │
    └─ 前台继续: Phase 5b 全量回归 → Phase 6 交付
          │
          └─ subagent 完成 → 自动通知 → 汇总性能结果
```

#### 性能逐个运行详解

使用 `-n N:N` 指定单个 case 运行，避免多个性能 case 同时加载导致 OOM：

```bash
# _perf.csv 含 3 个性能 case (行号 1,2,3)
python3 atb_csv_ops_test.py -i {op}_perf.csv -n 1:1  # case 1（400次迭代）
python3 atb_csv_ops_test.py -i {op}_perf.csv -n 2:2  # case 2（400次迭代）
python3 atb_csv_ops_test.py -i {op}_perf.csv -n 3:3  # case 3（400次迭代）
```

| 方式 | 显存占用 | 风险 |
|------|---------|------|
| 整个文件一次跑 | 3 case × 400 迭代同时加载 | OOM (exit 137) |
| 逐个 `-n N:N` | 每次只有 1 个 case | 安全 |

**规则**：
- Phase 5a 正例/反例**不指定 `-t`**（默认 1 次），跑 `_func.csv`，前台快速验证
- Phase 5a 性能测试跑 `_perf.csv`，**后台 subagent 逐个 `-n N:N`**，主流程不阻塞
- Phase 5b 全量回归跑 `_func.csv` + `_perf.csv`（或合并 `_test.csv`），使用默认迭代数
- 后台 subagent 完成后自动通知，Agent 需将性能结果汇入交付报告

### CSV 文件格式

CSV 测试文件使用管道符（`|`）作为分隔符：

| 列号 | 名称 | 说明 |
|---|---|---|
| 1 | CaseNum | 测试用例编号 |
| 3 | OpName | 算子名称（如 `LinearOperation`） |
| 4 | OpParam | 算子参数（JSON 格式） |
| 21 | ExpectedError | 期望错误类型（如 `NO_ERROR`） |

</details>

---

## 常用 CSV 测试文件

| 文件名 | 路径 | 用途 |
|---|---|---|
| `linear.csv` | `<ATB_REPO_PATH>/tests/apitest/opstest/csv/linear.csv` | Linear 算子基础测试 |
| `linear_unals.csv` | `<ATB_REPO_PATH>/tests/apitest/opstest/csv/linear_unals.csv` | Linear Unals 格式测试 |

---

## 常见问题

<details>
<summary>❓ libatb_test_framework.so 不存在</summary>

执行 `bash scripts/build.sh testframework`

</details>

<details>
<summary>❓ ATB_HOME_PATH is null</summary>

执行 `source output/atb/set_env.sh`

</details>

<details>
<summary>❓ 测试用例全部跳过</summary>

确认目标 SoC 版本是否在 CSV 文件的 `SocVersion` 列中

</details>

---

## 相关文档

- [ATB Skills 索引](../SKILL.md)
- [ATB 测试框架编译](../atb-testframework-build/SKILL.md)
