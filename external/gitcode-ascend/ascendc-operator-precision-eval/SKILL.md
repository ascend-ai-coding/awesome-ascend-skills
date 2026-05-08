---
name: external-gitcode-ascend-ascendc-operator-precision-eval
description: AscendC算子精度评估。对已编译安装的算子生成全面的精度测试用例集（≥30例），运行并生成精度验证报告。关键词：精度测试、precision
  evaluation、精度报告、accuracy、误差分析。执行完成后 MUST 在当前对话中展示总览、失败摘要与关键发现，不得仅附报告路径。
original-name: ascendc-operator-precision-eval
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# AscendC 算子精度评估

**Skill 类型**：评估型（测试生成 + 执行 + 报告输出）

本 skill 对已编译安装的 AscendC 算子进行系统化精度评估。测试用例由「常规 shape 测试」和「边界值测试」两部分组成，每个用例遍历算子支持的全部 dtype，运行后输出结构化精度报告。

## 前置条件

- 算子已编译安装（`ascend_kernel` 包可 import 或 `.so` 文件存在）
- 算子已通过基本功能测试（`tests/test_<op_name>.py` 存在且通过）
- 已知算子名称、PyTorch 调用方式、输入域约束、支持的全部 dtype
- **`csrc/ops/<op_name>/test/<op_name>-test-cases.md`** 已由 `ascendc-operator-testcase-gen` 生成（包含 SUPPORTED_DTYPES、TEST_SHAPES、BOUNDARY_VALUES、算子标杆）

## 核心流程

```
Phase 1: 加载用例文档 + 信息收集 → Phase 2: 用例适配 → Phase 3: 测试脚本生成 → Phase 4: 执行 → Phase 5: 报告生成
```

---

## Phase 1：加载用例文档 + 信息收集

### Step 1.1：加载 testcase-gen 用例文档（MANDATORY）

**MUST** 首先读取 `csrc/ops/<op_name>/test/<op_name>-test-cases.md`，从中提取：

| 提取项 | 在用例文档中的位置 | 用途 |
|--------|-------------------|------|
| SUPPORTED_DTYPES | §测试配置 | 精度测试遍历的 dtype 列表 |
| TEST_SHAPES | §测试配置 | 常规 shape 测试的 shape 列表 |
| BOUNDARY_VALUES | §测试配置 | 边界值测试的标量值列表 |
| NPU 调用方式 | §算子标杆 | NPU_CALL 表达式 |
| CPU 参考实现 | §算子标杆 | CPU_REF 表达式 |

> **若 `<op_name>-test-cases.md` 不存在**：回退为自行设计用例（按原 Phase 2 流程），但需在报告中注明"用例为自行设计，非 testcase-gen 产出"。

### Step 1.2：从代码补充信息

从现有代码中补充提取以下信息：

| 信息 | 来源 | 示例 |
|------|------|------|
| 算子名称 | 用户输入 / op_host 文件名 | `acosh` |
| NPU 调用方式 | `register.cpp` 中的 `m.def`（与用例文档交叉验证） | `torch.ops.npu.acosh(x)` |
| CPU 参考实现 | PyTorch 标准库（与用例文档交叉验证） | `torch.acosh(x.cpu().float()).to(dtype)` |
| 输入域约束 | 数学定义 / design.md | `x >= 1.0` |
| **支持的全部 dtype** | `op_host` 中的 TORCH_CHECK（与用例文档交叉验证） | `[torch.float16, torch.float32]` |
| **支持的维度/shape 约束** | design.md / op_host 中的逻辑 | elementwise 支持任意维度 |
| 精度阈值 | 生态算子开源精度标准（见下表） | 按 dtype 查 Threshold |

### 精度标准（生态算子开源精度标准）

采用 **MERE**（平均相对误差）和 **MARE**（最大相对误差）两个指标判定：

```
相对误差 = abs(actual - golden) / (abs(golden) + 1e-7)
MERE = mean(相对误差)
MARE = max(相对误差)
```

> 分母引入 `1e-7` 避免 golden 为零时除零。

**通过标准**：MERE < Threshold **且** MARE < 10 × Threshold。

| dtype | Threshold | MERE 上限 | MARE 上限 (10×) |
|-------|-----------|----------|----------------|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | 7.81e-3 | 7.81e-2 |
| float32 | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 |

> 完整 dtype 列表（含 HiFLOAT32、FLOAT8 E4M3/E5M2）见 [`references/OPS_PRECISION_STANDARDS.md`](references/OPS_PRECISION_STANDARDS.md)。

---

## Phase 2：用例适配

> **优先复用 testcase-gen 产出**：若 Phase 1 成功加载了 `<op_name>-test-cases.md`，则 TEST_SHAPES 和 BOUNDARY_VALUES **直接使用**用例文档中的定义，无需重新设计。仅在用例文档不存在时才自行设计。

### 设计原则

1. **全 dtype 覆盖**：每个 shape / 每个边界值都遍历算子支持的全部 dtype
2. **shape 由算子决定**：根据算子支持的维度选择合适的 shape，不要写固定维度
3. **shape 不要过大**：单个用例元素数控制在合理范围，避免不必要的大 tensor
4. **用例总数 = (len(TEST_SHAPES) + len(BOUNDARY_VALUES)) × len(SUPPORTED_DTYPES) ≥ 30**

### Part A: 常规 Shape 测试（TEST_SHAPES）

根据算子支持的维度，从以下维度池中选取适合的 shape，组成 `TEST_SHAPES` 列表。

**MUST** 根据算子实际支持的维度来选，不支持的维度不要选。

#### shape 选择参考池

| 维度 | 推荐 shape | 适用算子类型 |
|------|-----------|-------------|
| 1D | (128,), (1024,), (4096,), (8192,) | elementwise, reduction |
| 2D | (32, 512), (64, 768), (128, 1024) | elementwise, matmul, linear |
| 3D | (8, 16, 64), (4, 128, 256) | elementwise, attention, conv1d |
| 4D | (4, 8, 32, 16), (2, 64, 32, 32) | conv2d, elementwise |
| 5D | (2, 3, 4, 5, 6) | conv3d, elementwise |

> **shape 不要过大**：推荐单个 shape 元素数 ≤ 200K。

#### TEST_SHAPES 格式

```python
TEST_SHAPES = [
    ("category_name", "description", (dim0, dim1, ...)),
    # ...
]
```

Category 名称自定义，建议按维度或场景命名。示例：

```python
# elementwise 算子（支持任意维度）
TEST_SHAPES = [
    ("1D",         "128 elements",         (128,)),
    ("1D",         "1024 elements",        (1024,)),
    ("1D",         "4096 elements",        (4096,)),
    ("1D",         "8192 elements",        (8192,)),
    ("2D",         "batch*hidden 32x512",  (32, 512)),
    ("2D",         "BERT-base 64x768",     (64, 768)),
    ("2D",         "BERT-large 128x1024",  (128, 1024)),
    ("3D",         "8x16x64",             (8, 16, 64)),
    ("3D",         "4x128x256",           (4, 128, 256)),
    ("4D",         "4x8x32x16",           (4, 8, 32, 16)),
    ("Production", "ViT 8x197x768",       (8, 197, 768)),
    ("Production", "single sample 1x512", (1, 512)),
]
```

### Part B: 边界值测试（BOUNDARY_VALUES）

针对算子的输入域，选取关键边界点和典型值。使用固定小 shape `(1024,)` 进行测试。

**MUST** 根据算子的数学定义确定边界值，不同算子差异很大。

#### 边界值设计指导

| 算子类型 | 推荐边界值 |
|----------|-----------|
| acosh (x≥1) | x=1.0, x=1.001, x=10.0, x=1000.0 |
| log (x>0) | x=0.001, x=1.0, x=100.0, x=10000.0 |
| sigmoid (全域) | x=0.0, x=-5.0, x=5.0, x=-20.0, x=20.0 |
| sqrt (x≥0) | x=0.0, x=0.001, x=1.0, x=10000.0 |
| 无域限制 | x=0.0, x=1.0, x=-1.0, x=100.0 |

#### BOUNDARY_VALUES 格式

```python
BOUNDARY_VALUES = [
    ("description", scalar_value),
    # ...
]
```

示例（acosh）：

```python
BOUNDARY_VALUES = [
    ("domain lower bound x=1.0",  1.0),
    ("near boundary x=1.001",     1.001),
    ("moderate value x=10.0",     10.0),
    ("large value x=1000.0",      1000.0),
]
```

---

## Phase 3：测试脚本生成

**MUST** 先读取 `templates/` 目录下的模板文件，替换占位符后生成到算子目录的 `test/` 子目录。

### 输出目录

```
csrc/ops/<op_name>/test/
├── test_<op_name>_precision.py           ← pytest 测试
├── run_<op_name>_precision_report.py     ← 报告生成器
├── <op_name>_precision_report.json       ← JSON 报告（执行后生成）
└── <op_name>_precision_report.md         ← Markdown 报告（执行后生成）
```

### 模板文件

| 模板 | 路径 | 生成目标 |
|------|------|---------|
| pytest 测试 | `templates/test_op_precision_template.py` | `csrc/ops/<op_name>/test/test_<op_name>_precision.py` |
| 报告生成器 | `templates/run_precision_report_template.py` | `csrc/ops/<op_name>/test/run_<op_name>_precision_report.py` |

### 占位符替换表

| 占位符 | 说明 | 示例（acosh） |
|--------|------|--------------|
| `{{OP_NAME}}` | 算子名称 | `acosh` |
| `{{NPU_CALL}}` | NPU 调用表达式（使用变量 `x`） | `torch.ops.npu.acosh(x)` |
| `{{CPU_REF}}` | CPU 参考基线（使用变量 `x`, `dtype`） | `torch.acosh(x.cpu().float()).to(dtype)` |
| `{{SUPPORTED_DTYPES}}` | 支持的全部 dtype 列表 | `[torch.float16, torch.float32]` |
| `{{INPUT_LOW}}` | 随机输入的域下界 | `1.0` |
| `{{INPUT_HIGH}}` | 随机输入的域上界 | `11.0` |
| `{{TEST_SHAPES}}` | 常规 shape 列表（Phase 2 Part A 的输出） | 见上方示例 |
| `{{BOUNDARY_VALUES}}` | 边界值列表（Phase 2 Part B 的输出） | 见上方示例 |

### 必须采集的精度指标

**判定指标**（用于通过/失败判定）：

| 指标 | 计算方式 | 通过条件 |
|------|---------|---------|
| MERE | `((npu - ref).abs() / (ref.abs() + 1e-7)).mean()` | < Threshold |
| MARE | `((npu - ref).abs() / (ref.abs() + 1e-7)).max()` | < 10 × Threshold |

**辅助指标**（用于分析，不作为判定依据）：

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| MaxAbsErr | `(npu - ref).abs().max()` | 最大绝对误差 |
| MeanAbsErr | `(npu - ref).abs().mean()` | 平均绝对误差 |
| CosineSim | `F.cosine_similarity(npu.flatten(), ref.flatten())` | 余弦相似度 |

---

## Phase 4：执行测试

### 4.1 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/root/miniconda3/envs/py310/bin:$PATH
```

> **MUST** 在每个 Shell 调用前 source 环境。

### 4.2 执行 pytest

```bash
cd <project_root>
python3 -m pytest csrc/ops/<op_name>/test/test_<op_name>_precision.py -v --tb=short
```

### 4.3 生成报告

```bash
python3 csrc/ops/<op_name>/test/run_<op_name>_precision_report.py
```

### 4.4 失败处理

| 失败类型 | 排查方向 |
|----------|---------|
| RuntimeError (NPU kernel) | 输入数据超出定义域 / NPU 不支持该 dtype |
| AssertionError (精度) | 检查 MERE/MARE 是否略超 Threshold，分析是否为边界值导致 |
| 个别 dtype 用例 FAIL | 确认该 dtype 的 Threshold 是否匹配，检查 MARE 是否集中在少数异常点 |
| 大量 FAIL | 检查算子 Compute 逻辑是否有 bug |

### 4.5 精度问题深度排查

**当出现精度失败（allclose 不通过、输出偏差过大、输出全零/NaN）且简单阈值调整无法解决时**，**MUST** 读取并按 **`ascendc-operator-precision-debug`** skill 流程进行系统化根因定位：

1. 读取 `ascendc-operator-precision-debug` SKILL.md
2. 按其五阶段流程执行：误差分析 → 代码审查 → 实验隔离 → 插桩定位 → 修复验证
3. 修复后重新运行本 skill 的完整精度测试，确认全部通过

> **注意**：仅在精度问题无法通过调整阈值解决时才调用。个别 dtype 因硬件精度特性略超阈值的情况，优先通过放宽阈值（并在报告中说明）解决。

---

## Phase 5：报告生成

### 5.1 Markdown 报告

**MUST** 生成 `csrc/ops/<op_name>/test/<op_name>_precision_report.md`，参考 `templates/precision_report_template.md`。

报告包含：

1. **总览表**：总用例/通过/失败/通过率
2. **精度阈值标准表**
3. **常规 Shape 测试结果表**（按 category 分组）
4. **边界值测试结果表**
5. **按 dtype 汇总统计**
6. **关键发现**（≥3 条结论）

### 5.2 完成提示（文件 + 对话）

1. **文件**：**MUST** 生成 `csrc/ops/<op_name>/test/<op_name>_precision_report.md`（及同目录 `*_precision_report.json` 若脚本输出），并向用户给出**完整路径**：

```
精度验证报告已生成：
  csrc/ops/<op_name>/test/<op_name>_precision_report.md
  csrc/ops/<op_name>/test/<op_name>_precision_report.json   # 若存在
```

2. **当前对话**：**MUST** 同时遵守下节「对话内展示结果」，不得仅输出路径。

---

## 对话内展示结果（MANDATORY）

pytest 与报告脚本执行完毕且已生成 Markdown/JSON 后，助手在**当前对话**的回复中 **MUST**：

1. **粘贴可读结论**（用户无需打开文件即可掌握结果）：
   - **总览**：总用例数、通过数、失败数、**通过率**（百分比）。
   - **若有失败**：列出失败用例标识（case 名 / shape / dtype / 类别），以及主要误差指标（如 MaxAbsErr）或 pytest 摘要行。
   - **若全部通过**：明确写出「全部通过」及总用例数。
   - **关键发现 ≥3 条**：可与报告内「关键发现」一致或提炼自报告（dtype 差异、边界值表现、阈值是否收紧/放宽等）。
   - **可选**：按 dtype 汇总的通过情况表（摘录，case 多时可只列汇总行）。
2. **口径**：一两句话说明使用的精度标准（MERE/MARE，生态算子开源精度标准）及各 dtype 的 Threshold 值。
3. **路径殿后**：在展示完上述内容后，再附 **`<op_name>_precision_report.md`**（及 JSON）的完整路径。

**NEVER**：仅回复「报告已生成」和路径；NEVER 用「请自行打开 Markdown」替代在对话中展示通过率与失败摘要。

---

## 经验总结

### 输入生成

- **定义域**：务必查阅算子数学定义，确保输入合法
- **fp16 范围**：fp16 最大约 65504，输入不要超过此值
- **shape 大小**：推荐单个 shape 元素数 ≤ 200K，避免测试时间过长

### 精度指标

- **MERE / MARE**：判定指标，分母为 `abs(golden) + 1e-7`（非 clamp），与生态算子开源精度标准对齐
- **MaxAbsErr / MeanAbsErr**：辅助分析，帮助判断偏差量级
- **CosineSim**：全零输出时为 0 或 NaN，需标注说明而非判定失败

### 阈值说明

- 阈值来源：生态算子开源精度标准（`references/OPS_PRECISION_STANDARDS.md`）
- 通过条件：MERE < Threshold 且 MARE < 10 × Threshold
- 不建议随意放宽阈值；若确需放宽，**MUST** 在报告中说明原因

---

## 反模式（NEVER）

- NEVER 只生成报告文件而不在对话中展示总览与结论
- NEVER 隐瞒失败用例数量，仅报告路径

---

## 检查清单

- [ ] 已读取 `csrc/ops/<op_name>/test/<op_name>-test-cases.md`（若存在）
- [ ] 信息收集完成（算子名、调用方式、输入域、支持的全部 dtype、支持的维度）
- [ ] TEST_SHAPES 优先来自 testcase-gen 用例文档，shape 不过大
- [ ] BOUNDARY_VALUES 根据算子输入域设计
- [ ] 用例总数 = (shapes + boundary) × dtypes ≥ 30
- [ ] 算子支持的每种 dtype 都已测试
- [ ] pytest 全部通过
- [ ] JSON + Markdown 报告已生成
- [ ] 关键发现 ≥ 3 条
- [ ] 已向用户提示报告与 JSON 路径（若生成）
- [ ] **已在当前对话中展示**总览（通过率）、失败摘要（若有）及 ≥3 条关键发现，不仅附路径
