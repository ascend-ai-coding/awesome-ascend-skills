---
name: external-cannbot-ops-pypto-op-design
description: 当需要设计 PyPTO 算子实现方案时使用。通过迭代式约束收敛，生成 DESIGN.md（含 API 映射、精度路由、Tiling 推导、Loop
  结构设计）。触发词：生成设计方案、生成 design、设计方案、写 DESIGN.md、算子设计、API 映射、Tiling 策略、tiling 推导、Loop
  结构、数据流设计、精度路由。
original-name: pypto-op-design
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-06-14'
synced-commit: f2b0db65abdced502b1cd1956e2e48e6e0197add
license: UNKNOWN
---

# PyPTO 算子方案设计

通过迭代式问题驱动，生成可直接翻译为代码的 DESIGN.md。

**核心原则**：
- 设计文档不是复述 SPEC，而是回答"怎么实现"的决策记录
- 每个决策必须包含**结论 + 推导过程 + 排除的替代方案**
- 伪代码是核心产出，必须标注每个 tensor 的 shape、dtype、以及变量是否为 SymbolicScalar
- 只有运行时才确定大小的轴标 `pypto.DYNAMIC`，编译期已知的轴不标

---

## 1. 输入与输出

| 来源 | 必须 | 用途 |
|------|------|------|
| 算子规格 | 是 | 公式、shape、dtype、动态轴、典型配置 |
| API 探索报告 | 否 | API 可用性（缺失时在第 1 轮自行查 `pypto/docs/zh/`） |
| Golden 参考实现 | 否 | 辅助理解计算逻辑 |

**输出**：`DESIGN.md`，基于模板 [templates/design-template.md](templates/design-template.md)

---

## 2. 迭代设计流程

设计是**问题驱动**的迭代，不是线性填表。每一轮聚焦一个核心问题，发现矛盾时回溯修正前序决策。

### 第 1 轮：计算图与精度路由

**核心问题**：数学公式的每一步用哪个 PyPTO API？dtype 怎么流转？哪里必须 cast？

**步骤**：
1. 拆分公式为原子操作
2. 查 `pypto/docs/zh/api/` 找到对应 API 及其 dtype 限制
3. 标注每步的输入/输出 dtype，识别必须的 cast 点
4. 写出带 shape/dtype 注释的计算伪代码
5. 记录被排除的替代 API 及原因

**收敛标志**：每步都有确定的 API 和 dtype，无类型冲突。

**可能发现的问题**（触发回退）：
- `pypto.sum` 要求 FP32 但输入是 BF16 → 插入 cast → 检查后续是否需要 cast 回
- 操作无对应 API → 拆解为组合操作 → 中间 tensor 数量增加（影响第 2 轮）

**产出**：DESIGN.md §1（计算图与精度路由）

---

### 第 2 轮：Tiling 推导

**核心问题**：第 1 轮确定的 tensor 能同时放进 UB 吗？tile 应该取多大？

**前置依赖**：第 1 轮确定的 API 序列和 tensor 清单。

**步骤**：
1. **分类**：含 matmul → cube + vec 混合；纯 vector → 仅 vec
2. **列出同时驻留的 tensor**（输入、输出、所有中间结果）及其 dtype
3. **尾轴对齐**：bf16/fp16 → 16 元素，fp32 → 8 元素
4. **容量估算**：`tile_size × dtype_bytes × tensor_count ≤ UB 容量`
5. **展开检查**：`(shape / tile) × tensor_count < 18000`
6. **混合算子**：不同计算阶段可能需要不同的 tiling 配置

**回溯条件**：
- tile 算出来过大（UB 放不下）→ 回第 1 轮减少中间 tensor 或调整精度路由
- tile 过小（展开爆炸）→ 调整计算分块方式

**产出**：DESIGN.md §3（Tiling 策略）

---

### 第 3 轮：Loop、数据流与 SymbolicScalar 分析

**核心问题**：哪些轴需要 loop？数据怎么搬运？完整计算流可行吗？

**前置依赖**：第 1 轮的 API 序列 + 第 2 轮的 tile 配置。

#### 3.1 动态轴分析

逐轴判定：
- **编译期已知且单 tile 可覆盖** → **不标 DYNAMIC**，不需要 loop
- **编译期已知但超出 tile** → **不标 DYNAMIC**，用 Python for 或编译器自动切分
- **运行时才确定大小** → **标 `pypto.DYNAMIC`**，用 `pypto.loop`

#### SymbolicScalar 约束

动态轴的 `tensor.shape[i]` 和 `pypto.loop` 返回的索引都是 SymbolicScalar，**不是 Python int**：

| 禁止操作 | 报错示例 | 正确替代 |
|----------|---------|----------|
| `sym ** n` | 不支持幂运算 | 静态值用 `math.sqrt`；动态值用 `pypto.Element` |
| `sym % n` | 不支持取模 | `sym - (sym // n) * n` |
| `list[sym]` | 不能做下标 | `pypto.view(tensor, shape, [sym, ...])` |
| `if sym > x:` | 不能做 Python 条件 | `pypto.cond(sym > x)` |
| `min(sym, x)` | 不能用 Python min | `sym.min(x)` |
| `range(sym)` | 不能用 Python range | `pypto.loop(sym)` |

#### 3.2 完整伪代码

伪代码是设计的核心产出，必须可直接"翻译"为实现代码。要求：

1. **每个 tensor 标注 shape 和 dtype**（作为行尾注释）
2. **标注 tiling 配置的位置**
3. **标注哪些变量是 SymbolicScalar**
4. **标注 view/assemble 的 offset 计算**
5. **标注累加器/状态变量的初始化位置和更新方式**

示例：

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 128], pypto.DT_FP32),    # [B, D]
    out: pypto.Tensor([pypto.DYNAMIC, 128], pypto.DT_FP32),  # [B, D]
):
    B = x.shape[0]   # SymbolicScalar（动态轴）
    D = 128           # Python int（静态值，不标 DYNAMIC）

    pypto.set_vec_tile_shapes(1, D)  # tile = [1, 128]

    for b in pypto.loop(B, name="batch"):  # B 是 SymbolicScalar → 必须用 pypto.loop
        x_tile = pypto.view(x, [1, D], [b, 0])       # [1, 128], FP32
        x_max = pypto.amax(x_tile, dim=-1, keepdim=True)  # [1, 1], FP32
        x_shifted = pypto.sub(x_tile, x_max)          # [1, 128], FP32
        x_exp = pypto.exp(x_shifted)                   # [1, 128], FP32
        x_sum = pypto.sum(x_exp, dim=-1, keepdim=True) # [1, 1], FP32（sum 仅支持 FP32）
        result = pypto.div(x_exp, x_sum)               # [1, 128], FP32
        pypto.assemble(result, [b, 0], out)            # 写回 out[b, :]
```

#### 3.3 伪代码可行性验证

写完伪代码后，逐行检查以下约束。这是设计阶段最关键的环节——提前发现的约束冲突可以在设计阶段修正，否则在实现阶段会以编译错误或精度异常出现。

**A. API dtype 约束** — 逐个 API 检查输入 dtype 是否满足要求：

| API | 支持的 dtype | 常见陷阱 |
|-----|-------------|---------|
| `sum` | 仅 FP32 | BF16 输入必须先 cast |
| `softmax`/`sin`/`cos` | 仅 FP32 | — |
| `matmul` | 两侧 dtype 一致 | 一侧 cast 后忘记另一侧 |
| `add`/`sub`/`mul`/`div` | 两侧 dtype 一致，2-4 维 | 不存在隐式类型提升 |
| `where` | condition 必须 BOOL | — |
| `amax`/`amin` | FP16/BF16/FP32，2-4 维 | — |
| `exp`/`log` | FP16/BF16/FP32 | 精度敏感建议用 FP32 |

**B. 广播与 Shape 兼容检查** — PyPTO 仅支持单轴广播（一个维度为 1 与另一个对齐），不支持多轴同时广播。检查每个二元操作的两个输入 shape 是否兼容。

**C. 值类型检查** — 标注伪代码中每个变量的类型（SymbolicScalar / Python 标量 / Tensor / Element），检查是否使用了不支持的操作：

| 禁忌写法 | 原因 | 正确写法 |
|----------|------|---------|
| `D ** (-0.5)` 其中 D 是 SymbolicScalar | `**` 不支持 | `pypto.Element(DT_FP32, 1/math.sqrt(D_static))` |
| `list[loop_idx]` | SymbolicScalar 不能做下标 | `pypto.view(tensor, shape, [loop_idx, ...])` |
| `if B > 0:` 其中 B 是 SymbolicScalar | Python `if` 不支持 | `pypto.cond(B > 0)` |
| `min(remaining, tile)` | Python `min` 不接受 | `remaining.min(tile)` |
| `range(N)` 其中 N 是 SymbolicScalar | Python `range` 不接受 | `pypto.loop(N)` |
| `output = result` | 只重绑 Python 变量，不写回 | `output[:] = result` |

**D. Tiling 配置时序检查**：
- `set_vec_tile_shapes` 必须在首个向量操作或 `zeros`/`full` 之前调用
- `set_cube_tile_shapes` 必须在 `matmul` 之前调用
- TileShape 维度数必须等于操作涉及的 tensor 维度数

**E. 数据搬运约束**：
- 同一 tensor 不能在同一 JIT 图中既被 `view` 读又被 `assemble` 写（DAG 环路）
- `assemble` 无返回值，直接修改目标 tensor
- loop 内不应分配新 tensor（应在 loop 外初始化）

**产出**：DESIGN.md §4（Loop 与数据流）+ 完整伪代码

---

### 第 4 轮：约束交叉验证

逐项检查，不通过的回溯到对应轮次修正：

**API 层（→ 第 1 轮）**
- [ ] 所有 `sum` 输入为 FP32
- [ ] 所有 `matmul` 左右 dtype 一致
- [ ] 每个 API 的 dtype 在 docs 中有记录

**Tiling 层（→ 第 2 轮）**
- [ ] TileShape 维度数 = tensor 维度数
- [ ] 尾轴 ≥ 对齐要求
- [ ] 同阶段驻留 buffer ≤ UB 容量
- [ ] 表达式展开 < 18000

**Loop 层（→ 第 3 轮）**
- [ ] 输出写回用 `[:]` / `.move()` / `assemble()`
- [ ] 无 view + assemble 环路
- [ ] 动态轴标了 `pypto.DYNAMIC`，静态轴未标
- [ ] 动态 loop 有 `unroll_list`
- [ ] 跨迭代依赖 → `submit_before_loop=True`
- [ ] 尾块 → `valid_shape`

**SymbolicScalar 检查**
- 伪代码中无 `sym ** n`、`list[sym]`、`if sym:` 等禁止操作
- 动态轴相关计算使用 `.min()` / `.max()` 而非 Python `min()` / `max()`
- 动态维度不直接用于 matmul 的 M/K/N 维度

---

## 3. DESIGN.md 输出结构与参考

**输出模板**：[templates/design-template.md](templates/design-template.md)

| 章节 | 对应迭代 | 必须内容 |
|------|---------|----------|
| §1 计算图与精度路由 | 第 1 轮 | API 序列、dtype 流、cast 点、备选方案 |
| §2 数据规格 | SPEC + 第 1 轮 | kernel 签名、动态轴标注、值类型分析 |
| §3 Tiling 策略 | 第 2 轮 | tile 参数 + 推导过程 + 约束验证 |
| §4 Loop 与数据流 | 第 3 轮 | **完整伪代码** + 数据搬运 + 尾块处理 |
| §5 约束检查与开放问题 | 第 4 轮 | 检查清单 + SymbolicScalar 审查 |
| §6 验证计划 | SPEC | 测试配置 + 精度容差 |

**参考资源**：
- [references/quick_ref.md](references/quick_ref.md) — 约束速查与冲突表
- `pypto/docs/zh/api/` — API 签名与 dtype 约束（最高优先级）
- `pypto/docs/zh/tutorials/` — 使用模式
- `models/`（排除 experimental）— 真实算子实现参考

---

## 4. 完成报告

```text
设计状态：{已收敛 / 有待确认项}

迭代过程：
  第 1 轮：API 调用链 {N} 步，cast {M} 处
  第 2 轮：Tiling {vec/cube/混合}，tile = {参数}
  第 3 轮：Loop {N} 层，动态轴 {列表}，跨迭代依赖 {有/无}
  第 4 轮：约束检查 {通过}/{总数}

{如有回退}
回退记录：
  第 X 轮 → 第 Y 轮：{原因}

{如有未决问题}
开放问题：
  · {问题} — {影响范围}
```
