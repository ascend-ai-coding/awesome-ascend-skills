---
name: external-cannbot-ops-pypto-golden-generate
description: 当需要生成 golden 参考实现时使用此 skill。基于算子规格信息，生成 torch + torch_npu NPU 参考实现 `{op}_golden.py`，导出
  `{op}_golden()` 函数，作为精度验证基准。计算在 NPU 上执行；torch_npu 未安装时直接报错引导安装，仅无 NPU 硬件时回退 CPU。触发词：生成
  golden、生成参考实现、写 golden 函数、golden script、golden reference、reference implementation、generate
  golden、torch 参考、验证基准、baseline implementation、写验证代码、'帮我写 golden'、golden.py、参考代码。
original-name: pypto-golden-generate
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO Golden 参考实现生成（NPU）

基于算子规格信息，自动生成 torch + torch_npu NPU golden 参考实现及完整验证代码。计算的 golden 脚本用于开发阶段快速验证算子实现的正确性，可以作为独立模块被 `test_{op}.py` 等其他脚本导入调用。基于固定模板 [templates/golden-template.py](templates/golden-template.py) 生成（在§9 生成文件结构阶段读取该模板）。使用 torch 标准操作在 NPU 上执行；torch_npu 未安装时直接报错引导安装，仅无 NPU 硬件（`device_count() == 0`）时回退 CPU。禁止引入 pypto。性能采集不写入 golden 文件，统一由通用脚本 [scripts/profile_golden.py](scripts/profile_golden.py) 执行。

1. 从用户输入提取算子名称、公式、输入输出规格等必要信息
2. 如果信息不足，向用户逐步提问补充
3. 按工作流执行 golden 函数生成和验证
4. 输出 `{op}_golden.py` 到当前目录或用户指定位置

## 1. 所需信息

| 项目 | 说明 |
|------|------|
| **输入** | 算子规格信息（如结构化规格内容、自然语言描述等） |
| **输出** | `{op}_golden.py`，导出 `{op}_golden()` 函数，路径由调用者决定 |

---

## 2. 算子信息获取

从输入中提取算子名称，或由调用者指定。如果信息不足，向用户逐步提问补充。

---

## 3. 规格字段检查

> **脚本化（确定性，零思考 token）**：字段校验与文件骨架生成由 [`scripts/gen_golden_scaffold.py`](scripts/gen_golden_scaffold.py) 一步完成，不要靠对话逐字段推断：
>
> ```bash
> python3 scripts/gen_golden_scaffold.py --spec custom/<op>/SPEC.md \
>     --template templates/golden-template.py --out custom/<op>/<op>_golden.py
> ```
>
> 脚本读取 SPEC.md front matter（`op_name` / `p0_shapes` / `default_params` …）与 §5 输入/参数表，缺少必须字段时非零退出（吸收下表检查），并生成已填好**函数签名、`_make_inputs` 张量构造（按 p0_shapes + 参数表 concrete shape）、`_validate` 骨架**的 `<op>_golden.py`；仅在数式本体、受约束输入、算子固有属性检查处留 `# TODO`。LLM 只需填这些 TODO。

脚本退出码与下表分类一致；如需人工核对，按以下分类检查字段完整性：

### 必须字段（缺失则报错退出）

| 字段 | 位置 | 用途 |
|------|------|------|
| 算子名称 | §1 基础信息 | 文件名、函数名 |
| 数学公式 | §1 基础信息 | 生成 PyTorch 实现 |
| 输入规格 | §4 数据规格 | 函数参数、验证 shape |
| 输出规格 | §4 数据规格 | 返回类型、验证 shape |

### 建议字段（缺失时引导补充）

| 字段 | 位置 | 用途 | 缺失时处理 |
|------|------|------|------------|
| 典型配置 | §11 应用场景 | 典型 case 验证 | 引导用户补充到规格信息中 |
| 动态轴范围 | §7 动态轴说明 | 泛化 case 采样 | 使用默认范围 (1-1024) |

### 典型配置缺失时的处理

```
检查规格信息中是否包含典型配置
    │
    ├── 有 → 直接使用
    │
    └── 无 → 引导用户提供
            │
            ├── 用户提供 → 补充到规格信息中，继续生成
            │
            └── 用户跳过 → 根据动态轴范围生成默认配置
                          │
                          ├── 有动态轴范围 → 按范围推荐
                          │   └── 补充到规格信息中，继续生成
                          └── 无动态轴范围 → 使用通用默认值
                              └── 补充到规格信息中，继续生成
```

典型配置采用 7 列格式：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|

---

## 4. Golden 函数生成规范

### 实现方式

使用 **PyTorch + torch_npu** 实现 golden 函数，计算在 NPU 上执行。优先使用 PyTorch 内置 API，在没有直接对应 API 时由 LLM 基于公式生成实现。输入自动转移到 NPU device，返回 NPU tensor。torch_npu 未安装时直接报错引导安装；仅无 NPU 硬件（`device_count() == 0`）时回退 CPU。

**设备卡号（device ID）**：设备卡号统一通过环境变量 `TILE_FWK_DEVICE_ID` 读取（运行前 `export TILE_FWK_DEVICE_ID=<id>` 设置），未设置时默认使用卡 0。模板中 `_get_device()` 已内置该机制，生成 golden 时不要修改设备读取逻辑，不要硬编码为 `torch.device("npu")` 或 `torch.device("npu:0")`。

### 函数签名

```python
# 单输入
def silu_golden(x: torch.Tensor) -> torch.Tensor:

# 多输入
def swiglu_golden(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:

# 带可选参数
def layer_norm_golden(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
```

**规则**：
- 函数名：`{算子名}_golden`
- 参数顺序：必要张量参数在前，可选参数在后
- 所有 spec 中定义的参数都要实现，包括可选参数
- dtype 不作为参数，计算精度跟随输入张量
- **matmul 精度对齐**：所有 `torch.matmul` 输入必须先 `.float()`，与 `pypto.matmul` 在 NPU Cube L0C 上的 FP32 累加路径对齐。

  | 写法 | 累加精度 | 正确 |
  |------|---------|------|
  | `torch.matmul(a_bf16, b_bf16).float()` | BF16 | ❌ |
  | `torch.matmul(a.float(), b.float())` | FP32 | ✅ |
  | `torch.matmul(a.float(), b.float()).to(torch.bfloat16)` | FP32 累加 + BF16 输出 | ✅ |

### 边界条件映射

根据规格信息中的边界条件定义生成对应代码逻辑：

```python
# spec 定义：零值返回 1
def safe_div_golden(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = x / y
    result[y == 0] = 1.0  # 映射边界条件
    return result
```

### `_make_inputs()` 输入工厂函数（强制导出）

每个 golden 文件**必须**导出 `_make_inputs(device)` 函数，构造 P0 典型输入。该函数供 `_validate()` 验证和 `profile_golden.py --factory` 性能采集**共用**，确保两处使用完全一致的输入。

**函数签名与返回值**：

SPEC.md 中每个性能 P0 shape 都必须有对应的 case。返回值格式：

```python
# 单 P0 shape（向后兼容）
def _make_inputs(device):
    """构造 P0 典型输入。

    Returns:
        (args_list, kwargs_dict):
          - args_list: 位置参数列表（tensor，按 golden 函数签名顺序）
          - kwargs_dict: 关键字参数字典（scalar / 非 tensor 参数）
    """
    x = torch.randn(8, 1024, dtype=torch.bfloat16, device=device)
    return [x], {}

# 多 P0 shape（每个性能 P0 配置一组）
def _make_inputs(device):
    """构造所有 P0 典型输入。

    Returns:
        [(case_name, args_list, kwargs_dict), ...]:
          - case_name: 配置名称（对应 SPEC.md 典型配置表中的名称）
          - args_list: 位置参数列表（tensor，按 golden 函数签名顺序）
          - kwargs_dict: 关键字参数字典（scalar / 非 tensor 参数）
    """
    cases = []
    # 性能_P0 case 1: [2, 8, 512, 64]
    q = torch.randn(2, 8, 512, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 8, 512, 64, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 8, 512, 64, dtype=torch.bfloat16, device=device)
    cases.append(("perf_p0_small", [q, k, v], {}))
    # 性能_P0 case 2: [4, 16, 1024, 128]
    q = torch.randn(4, 16, 1024, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(4, 16, 1024, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(4, 16, 1024, 128, dtype=torch.bfloat16, device=device)
    cases.append(("perf_p0_large", [q, k, v], {}))
    return cases
```

`profile_golden.py` 自动检测返回格式：`list[tuple[str, list, dict]]` 视为多 case，`tuple[list, dict]` 视为单 case。

**语义约束分类与处理**：

根据算子输入对随机值的容忍度，分为三类：

| 类别 | 特征 | 示例 | `_make_inputs()` 要求 |
|------|------|------|----------------------|
| **无约束** | 所有 tensor 接受任意随机值 | silu, gelu, matmul, softmax | `torch.randn` 即可 |
| **值域约束** | 部分 tensor 需要特定值域 | log（正数）、div（非零）、block_table（非负索引） | 对受限 tensor 使用 `torch.rand` / `torch.abs` / `torch.randint(low=0, ...)` |
| **结构约束** | tensor 之间存在依赖关系，或需要特定初始化 | 状态缓存（需合法 block 索引）、位置编码（需匹配序列长度）、压缩比参数（需整除关系） | 必须手动构造所有关联 tensor，确保结构一致性 |

**关键规则**：

- 所有 tensor 必须带 `device=device` 创建
- `_validate()` 内部调用 `_make_inputs(device)` 获取输入，不重复构造
- scalar 参数（如 `eps`、`d`、`ratio`）放在 `kwargs_dict` 中
- 使用 SPEC.md 的 `p0_shapes` 和 `default_params` 确定 shape 和参数值
- **SPEC.md 中有多个性能 P0 shape 时，必须为每个 shape 生成一组 case**，case_name 对应典型配置表中的名称
- 状态类 tensor（如 `kv_state`、`score_state`）使用 `torch.zeros` 初始化

---

## 5. 置信度系统

根据实现方式评估置信度，影响最终输出的标注和提示强度：

```
1. 是否有等效 PyTorch API？
   ├─ 有 → 直接调用 → ⭐⭐⭐⭐⭐
   │
   └─ 无 → 2. 是否为已知论文算子？
            ├─ 是 → 搜索第三方实现 → ⭐⭐⭐⭐
            │
            └─ 否 → 3. LLM 智能转换
                     ├─ 简单公式转换 → ⭐⭐⭐⭐
                     └─ 复杂公式转换 → ⭐⭐⭐
```

**低置信度加强提示**（⭐⭐⭐ 及以下）：

在输出文件的 docstring 中添加醒目警告，提醒人工审查代码逻辑、与论文对比验证、使用多组数据验证边界情况。

置信度和验证是独立机制——置信度只影响标注和提示强度，验证标准对所有算子一致。自动修复后根据最终代码更新置信度。

---

## 6. 验证机制

### 验证执行方式

生成文件后，**必须通过直接执行脚本完成验证**（在 NPU 上运行）：

```bash
python3 {op}_golden.py
```

脚本内含 `if __name__ == "__main__": _validate()` 入口，会自动运行全部检查项（典型 case、泛化 case、值域、数值稳定性等）并输出验证报告。验证在 NPU 上执行；torch_npu 未安装时报错引导安装，仅无 NPU 硬件时回退 CPU。

**禁止**使用以下方式替代直接执行：
- `exec(open(...).read())` — 绕过脚本的独立执行环境
- 手动构造测试数据单独调用 golden 函数 — 与脚本内置验证逻辑重复且不完整

**门禁判定依据**：以 `python3 {op}_golden.py` 的 exit code（0 = 通过）和验证报告输出为准。

**device 约定**：golden 函数内部对输入执行 `.to(device)` 做设备迁移。为避免跨设备 mismatch，验证代码创建张量时必须直接指定 `device=device`（如 `torch.randn(..., device=device)`），不要依赖 golden 内部的隐式 `.to()`。

**device ID 指定**：若用户明确要求在特定 NPU 卡号上运行验证，必须通过环境变量 `TILE_FWK_DEVICE_ID=<id>` 传入（运行前 `export`）。禁止在代码中硬编码 `torch.device("npu:<id>")`。

### 验证 shape 来源

**两层来源**：

```
├── 典型 case（来自算子规格中的典型配置）
│   ├── 性能类型：按优先级验证，P0 必须通过
│   └── 功能类型：按优先级验证，P0 必须通过
│
└── 泛化 case（来自算子规格中的动态轴取值范围）
    └── 每个动态轴采样：最小值、中间值、最大值
        如 b: 1-128 → [1, 64, 128]
           s: 64-2048 → [64, 1024, 2048]
        组合生成 shape
```

### 验证顺序（按重要性）

1. 性能类型 P0 配置（最高优先级，必须通过）
2. 性能类型 P1 配置
3. 功能类型 P0 配置（必须通过）
4. 功能类型 P1 配置
5. 泛化 case（边界+中间采样）
6. 其他低优先级配置

### 验证检查项

| 检查项 | 说明 | 级别 |
|--------|------|------|
| 语法检查 | 代码能正常 import | 🔴 严重 |
| 形状一致性 | 输出 shape 与 spec 定义一致 | 🔴 严重 |
| 函数签名 | 参数与 spec 定义匹配 | 🔴 严重 |
| 值域检查 | 从公式推导值域约束 | 🟡 警告 |
| 特殊点验证 | 边界值、零值等 | 🟡 警告 |
| 数值稳定性 | 无 NaN/Inf | 🟡 警告 |
| 数学属性 | 单调性、对称性、守恒量等 | 🟡 警告 |
| API 对比 | 与 PyTorch API 对比（如适用） | 🟢 信息 |

### 从公式推导的验证属性

**值域约束**：根据公式特性推导输出值域。例如 sigmoid 的输出在 (0, 1)、ReLU 的输出 >= 0、softmax 的输出在 (0, 1) 且沿归约轴和为 1。

**数学属性**：检查奇偶性（tanh 是奇函数）、单调性（sigmoid 单调增）、守恒量（softmax 和为 1）等。

**特殊点验证**：验证关键输入值的输出，如 sigmoid(0) = 0.5、tanh(0) = 0、relu(0) = 0。

---

## 7. 自动修复策略

验证失败时，按优先级自动尝试修复：

1. **使用 PyTorch 稳定 API** — 将手写实现替换为 PyTorch 内置 API
   - `x / (1 + torch.exp(-x))` → `torch.sigmoid(x)`
   - `torch.exp(x) / torch.exp(x).sum()` → `torch.softmax(x, dim=-1)`

2. **添加数值稳定性保护** — 防止除零、log 负数等
   - `a / b` → `a / (b + 1e-8)`
   - `torch.log(x)` → `torch.log(torch.clamp(x, min=1e-8))`

3. **使用更稳定的等价形式** — 数学等价但数值更稳定的替代

**限制**：最多 3 次修复尝试。修复后根据最终代码更新置信度。

---

## 8. 错误分级与输出控制

| 级别 | 含义 | 修复失败后处理 |
|------|------|----------------|
| 🔴 严重 | 语法错误、shape 不匹配、签名错误 | **阻止输出**，报告错误原因 |
| 🟡 警告 | 值域检查失败、数值不稳定 | 输出文件 + 在 docstring 中标注警告 |
| 🟢 通过 | 所有检查通过 | 正常输出 |

---

## 9. 生成文件结构

文件骨架由 [`scripts/gen_golden_scaffold.py`](scripts/gen_golden_scaffold.py)（见 §3）**确定性生成**：脚本读取固定模板 [`templates/golden-template.py`](templates/golden-template.py)，机械替换 `{op}` 占位符、写入函数签名与 `_make_inputs` 张量构造，**不需要 LLM 手工拼骨架**。LLM 只填脚本留下的 `# TODO`（数式本体、受约束输入、固有属性检查）。生成文件包含以下结构：

- `import torch` + `import torch_npu` + NPU 设备初始化
- 文件级 docstring（算子名、公式、置信度）
- `{op}_golden()` 函数：torch + torch_npu 参考实现，计算在 NPU 上执行（含示例注释）
- `_make_inputs(device)` 函数：构造 P0 典型输入，返回 `(args_list, kwargs_dict)`，供验证和性能采集共用（详见 §4 `_make_inputs()` 节）
- `_validate()` 函数：自动验证（典型 case、泛化 case、值域检查、数值稳定性、API 对比），内部调用 `_make_inputs()` 获取输入
- `if __name__ == "__main__": _validate()` 入口

性能采集相关代码不进入 `{op}_golden.py`，统一使用 [scripts/profile_golden.py](scripts/profile_golden.py)，保证 golden 文件只包含可在 NPU 上运行的 PyTorch 参考实现和验证逻辑。

---

## 10. 用户交互

全自动生成与验证，仅在以下场景需要用户参与：

| 场景 | 交互方式 |
|------|----------|
| 生成代码、验证、修复 | 全自动，无需用户参与 |
| 典型配置缺失 | 引导用户提供或确认推荐配置 |
| 覆盖已有文件 | 通过 `AskUserQuestion` 询问确认 |

---

## 11. 异常处理

| 场景 | 处理方式 |
|------|----------|
| 缺少算子规格信息 | 报错退出，提示先补齐需求信息 |
| 规格信息解析失败 | 报错退出，提示输入格式不正确 |
| 规格信息缺少必须字段 | 列出缺失字段，引导用户补充 |
| 多个算子未指定 | 列出所有可用算子，要求用户指定 |
| golden 文件已存在 | 通过 `AskUserQuestion` 询问是否覆盖 |

---

## 12. 验证报告（必须执行）

文件生成完成后，向用户展示验证结果，示例：

```
✅ Golden 参考实现已生成:
  • {name}_golden.py

============================================================
attention_golden 验证报告
============================================================

[典型 case 验证]
  性能_P0: b=2,h=8,s=512,d=64,w=128 ... ✓ PASS
  功能_P0: b=1,h=4,s=256,d=64,w=128 ... ✓ PASS

[泛化 case 验证]
  b=1,h=4,s=128,d=32,w=64 ... ✓ PASS
  b=64,h=8,s=1024,d=64,w=128 ... ✓ PASS

[值域检查]
  检查 softmax 归一化 ... ✓ PASS

[数值稳定性检查]
  大值输入 (x=100) ... ✓ PASS
  小窗口 (w=4) ... ✓ PASS

[功能正确性检查]
  验证窗口边界 ... ✓ PASS

============================================================
✅ 所有验证通过
============================================================
```

---

## 13. 既有参考的 PyPTO 友好化（Reference Normalization Path）

> **适用场景**：用户提供了已有的 PyTorch / NumPy 参考实现，需要将其规范化为
> PyPTO 友好的 golden。当用户没有提供参考实现而是直接通过规格信息生成 golden
> 时，走 §1-§12 的从规格生成路径即可，本节不适用。

### 执行指引（必读）

> **⚠️ 执行 §13 前必须先阅读** [references/reference-normalization.md](references/reference-normalization.md)。该文档包含强制规则、L0/L1 路径判断逻辑、Full vs Tiled 策略选择、PyPTO 不友好模式审计清单等**不可跳过的操作步骤**。

> **机械审计（确定性）**：`<op>_golden.py` 写完/编辑后，立即自查两条零误报硬规则——`import pypto`（禁止）、`.T` / `.t()`（禁止，须用 `torch.transpose`）。命中即视为不合格，必须当场改掉，不要用「这个 golden 是否 PyPTO 友好」的对话推断代替这两条检查。需要 dtype/shape 数据流的规则（§4 matmul `.float()`、隐式 broadcast、shape 注释）不在此断言，仍由人工审阅与 NPU 精度验证负责。

---

## 14. 升级路径（仅在 pre-check 触发约束时只读查阅）

当 golden 在 pre-check 阶段触发归约（reduction）对齐或 matmul 相关约束时，只读查阅
`pypto-general-debug` 中 `references/DEBUG_GUIDEBOOK.md` 的归约对齐与 matmul 章节即可，
**不要** fork 到 debug 子流程。本 skill 只生成纯 torch 的 golden，调试流程归 debugger 拥有。

---

## 15. NPU 性能 Profiling（验证通过后必须执行）

> **⛔ 强制步骤**：验证通过后，**必须**使用通用脚本 `scripts/profile_golden.py` 采集 NPU 性能数据。**`GOLDEN_PERF_REPORT.md` 是 Stage 2 的强制交付物**，未生成不得进入 Stage 3。SPEC.md 中有多个性能 P0 shape 时，必须对每个 shape 分别 profiling 并在报告中注明对应关系。

### 执行指引（必读）

> **⚠️ 执行 profiling 前必须先阅读** [references/profiling.md](references/profiling.md)。该文档包含输入模式决策树、多 case `_make_inputs()` 返回格式、参数构造步骤、约束类型表（6 种崩溃场景）、E2E 双路径提取逻辑、故障排查决策树等**不可跳过的操作步骤**。
