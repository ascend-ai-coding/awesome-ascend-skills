---
name: external-cannbot-ops-pypto-op-design
description: 当需要设计 PyPTO 算子实现方案时使用。通过迭代式约束收敛，生成 DESIGN.md（含 API 映射、精度路由、Tiling 推导、Loop
  结构设计）。触发词：生成设计方案、生成 design、设计方案、写 DESIGN.md、算子设计、API 映射、Tiling 策略、tiling 推导、Loop
  结构、数据流设计、精度路由。
original-name: pypto-op-design
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO 算子方案设计

通过迭代式问题驱动，生成可直接翻译为代码的 DESIGN.md。

> 资料获取统一使用 skill `pypto-docs-search`：按需搜索算子 API 文档、参考实现与 golden 等文件/目录/内容。

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
| API 探索报告 | 否 | API 可用性（缺失时在第 1 轮自行查 `docs/`） |
| Golden 参考实现 | 否 | 辅助理解计算逻辑 |

**输出**：`DESIGN.md`，以 [templates/design-template.md](templates/design-template.md) 为结构参考，**直接 Write 到新路径一次成文**（保留模板原标题）；禁止 `cp` 模板再编辑——`cp` 在工具层外预先建文件，会触发 Read-before-Write 拒绝并导致整篇文档重复生成。

---

## 2. 迭代设计流程

设计是**问题驱动**的迭代，不是线性填表。每一轮聚焦一个核心问题，发现矛盾时回溯修正前序决策。

### 第 0 轮：复杂度评估与 module_count 决策（mandatory，先于其它轮）

**核心问题**：这个算子要拆成几个 module？是否需要走 Stage 4 的多 module 流水线？

> 拆分阈值不是"复杂度门槛"而是"复杂度单位刻度"——每个 module 的目标厚度 ≈ 1 个标准 FlashAttention forward 的工作量。FlashAttention 本身正好是 1 个 module 的标准厚度，因此**不拆**。比 FA 复杂的算子拆出的每个 module 也应该 ≈ 1 FA 工作量，而不是把 FA 切成多个比 FA 简单的小 module。

#### 0.1 复杂度单位（complexity unit）

**1 复杂度单位 = 1 个标准 FlashAttention forward 的工作量**：
- ~25-30 行有效 golden 代码
- ~2 次 matmul
- ~1 个跨 tile reduce（含 online state，例如 softmax 整套算 1 个，不切 reduce / normalize）
- ~1 组 loop-carry state
- 可独立产生一个语义清晰、可命名、可验证的输出张量

#### 0.2-0.3 总复杂度与 module_count（脚本计算）

只需人工统计两个信号：S（loop-carry 状态组数，FA 的 m/l/o 算 1 组，gated_delta_rule 的 state + decay 算 2 组）与跨 tile reduce 次数（softmax 整套算 1 次）。其余信号（L、matmul 数）、`total_complexity = max(L,S,O)`、1.3 阈值与行数封顶公式全部由脚本计算：

```
python skills/pypto-op-design/scripts/derive_design_params.py \
    custom/<op>/<op>_golden.py --state-groups <S> --cross-tile-reduce <R> --json
```

直接采用输出 JSON 的 `total_complexity` / `module_count` / `path`；公式与参考算子验算表内置于脚本（`--self-test` 复核）。

#### 0.4 Heavy / Light op 分类（按跨 tile 通信）

后续 Stage 4 拆分时会用到，这里在 DESIGN.md 提前约定：

| 类别 | 算子 | 说明 |
|---|---|---|
| **Heavy ops**（构成 module 主要工作量）| `pypto.matmul` | Cube 输出跨 tile |
| | 跨 tile reduce（reduce 轴 > 单 tile 大小）| 需要 tile 间结果合并 |
| | Online softmax / softmax 整套 | 跨 tile 状态合并，整体算 1 个 heavy op |
| | Scan / recurrence step（state 跨 tile 传递）| 跨 tile state 通信 |
| | Outer product（结果跨 tile）| 跨 tile 写出 |
| **Light ops**（依附到相邻 heavy op 所在 module）| Elementwise（`add/mul/exp/sigmoid/tanh/sqrt/...`）| Tile 内计算 |
| | `cast` / dtype 转换 | Tile 内 |
| | Tile 内 reduce（reduce 轴 ≤ 单 tile 大小）| 无跨 tile 通信 |
| | **`pypto.view`** | **kernel 入口固定操作，强制合并到首 module** |
| | **`pypto.assemble`** | **kernel 出口固定操作，强制合并到尾 module** |
| | 简单 reshape / transpose（不改 stride 语义）| 结构调整 |

**关键约定**：Heavy op 不是 module 的**边界**，而是 module 的**骨架**——一个 module 可以含多个 heavy ops。

#### 0.5 数据流断点（仅当 module_count ≥ 2 时）

需要在 golden 的数据流上**自主**找出 `module_count - 1` 个"语义清晰、可命名、可独立验证"的中间张量作为模块边界。每段约等于 1 复杂度单位。

不给定固定清单——根据数据流的自然 stage 划分。每个 module 内含若干 heavy ops + 相关 light ops + view/assemble（首尾 module）。

#### 0.6 算子分类参考表（已内置脚本）

参考算子（GELU/Softmax/FA fwd/bwd/gated_delta_rule/mamba_ssm 等）的验算表内置于 `derive_design_params.py --self-test`，不再手算。

**产出**：DESIGN.md §0（Decomposition Decision）—— 包含 §0.1-§0.5 全部字段。**Stage 4 (designer / pypto-op-construct skill) 必须先读 §0.3 的 `module_count`，决定走 L0 单 module 路径或 L1 多 module 路径**。

**收敛标志**：`total_complexity`、`module_count`、（若 ≥2）数据流断点列表全部填写完毕（如人为偏离脚本结果需在 §0.5 注明原因——但不增加 architect override 字段，按公式走）。

---

### 第 1 轮：计算图与精度路由

**核心问题**：数学公式的每一步用哪个 PyPTO API？dtype 怎么流转？哪里必须 cast？

**步骤**：
1. 拆分公式为原子操作
2. 用 skill `pypto-docs-search` 搜索 `<op>` 的 API 文档，找到对应 API 及其 dtype 限制
3. 标注每步的输入/输出 dtype，识别必须的 cast 点
4. 写出带 shape/dtype 注释的计算伪代码
5. 记录被排除的替代 API 及原因

**收敛标志**：每步都有确定的 API 和 dtype，无类型冲突。

**可能发现的问题**（触发回退）：
- `pypto.sum` 要求 FP32 但输入是 BF16 → 插入 cast → 检查后续是否需要 cast 回
- 操作无对应 API → 拆解为组合操作 → 中间 tensor 数量增加（影响第 2 轮）

**产出**：DESIGN.md §1（计算图与精度路由）

---

### 第 2 轮：Tiling 推导（Stage 7 前 Tile shape 基线）

**核心问题**：第 1 轮确定的 tensor 能否按 Stage 7 前 Tile shape 基线稳定编译和首跑？

**前置依赖**：第 1 轮确定的 API 序列和 tensor 清单。

**步骤**：
1. **分类**：含 matmul → cube + vec 混合；纯 vector → 仅 vec

1.5. **Shape analysis** (思考可视化, mandatory):
   - 若含 matmul：确定 M / N / K 的静态/动态/范围
   - 确定尾轴 (last axis) 与 dtype → 算出 alignment 单位
   - 列举并行化候选轴 (batch / sequence 等)
   - **列举所有 vec op 的 operand shape 并按 shape class 分组**：
     例：`[1,1,BT,K]` class, `[1,1,BT,BT]` class, `[1,1,K,V]` class
     → 若有 2 个以上 shape class 则需要 **per-stage `set_vec_tile_shapes`** (与 cube tile 的 per-stage 规则对称)
   - **列举所有 matmul 的 operand shape** (用于判定是否需要 per-matmul 的 cube tile 设置)

1.6. **Tile design (mandatory，基线由脚本给出)**：
   - 运行 `derive_design_params.py`（§0.2 同一命令）即得 Stage 7 前基线：cube `[128,128]`、vec 每轴 ∈ `[16,64]`（首选偏小值）、尾轴 alignment、UB 估算。
   - 偏离基线只允许两种情况，且都写入 §3.2.2 rationale：API/shape 硬约束（取最近合法值）、UB 超出（轴值降至 <16）。
   - 混合算子仍需 per-stage 显式切换 vec/cube tile（切换作用域，不动基线值）；user-specified tile 须先验基线，例外标注「user-specified exception」。

1.7. **Alternatives considered** (optional)：
   - 推荐但不强制：列举 1-3 个备选方案 + 否决理由
   - 性能权衡是 Stage 7 的工作，architect 阶段不需要穷举

2. **列出同时驻留的 tensor**（输入、输出、所有中间结果）及其 dtype
3-5. **尾轴对齐 / UB 估算 / 展开检查**：由 `derive_design_params.py` 输出（`tail_alignment`、`ub_fits_worst_op`、`expansion_limit`）。原理备忘：per-op 估算（unary=2 / binary=3 / reduce·expand=4 个 tensor），tile 足够小则框架自然组 subgraph。

6. **混合算子**：不同计算阶段可能需要不同的 tiling 配置（per-stage tile）

7. **PyPTO syntax compliance check** (machine-verifiable, mandatory):
   填齐 DESIGN.md §3.2.4 的 syntax checklist 所有 checkbox（清单正本在 design-template §3.2.4）。机器可判项（alignment / UB / 展开 / 基线 / tile 静态性）直接采用脚本 `checks` 输出；其余（tile 维数、cube dim 关系、broadcast 1 轴、per-stage 设置）人工勾选。全部 ☑ 后才能进入第 3 轮。

**回溯条件**：
- 基线 tile 算出来过大（UB/L1 放不下）→ 回第 1 轮减少中间 tensor、调整精度路由，或记录 API/shape 硬约束例外
- tile 过小（展开爆炸 > 18000）→ vec 适度调大轴的 tile 值（仍在 [16, 64] 内）；cube 回到 Stage 7 前 128 基线，不得使用 16/32 等过小 cube tile 作为首跑方案
- syntax compliance fail → 回到步骤 1.6 重新设计 tile

**产出**：DESIGN.md §3 (Tiling 策略), 特别是 §3.2.1〜§3.2.5 所有 sub-section 都填齐

---

### 第 3 轮：Loop、数据流与 SymbolicScalar 分析

**核心问题**：哪些轴需要 loop？数据怎么搬运？完整计算流可行吗？

**前置依赖**：第 1 轮的 API 序列 + 第 2 轮的 tile 配置。

#### 3.0 数据流原则（必须先确认）

以下五个原则是数据流设计的前提，违反任一项都会在 impl 落地时出问题或在精度验证时露馅。在进入 3.1 之前先按这五点过一遍设计。

**Recurrent state（递归状态）切片规则**

对于在 loop 间持有的 buffer / 状态张量：

- 只切**语义上正确**的那一部分（按 step / chunk / module 自然边界）。
- **不要**仅仅因为 PyTorch 允许，就把高 rank tensor `view` 成自己写起来方便的 rank。view 的形状必须能在 golden 中说出含义。

**Broadcast（广播）规则**

如果设计中需要 broadcast：

- 优先**一次只 broadcast 一个轴**的形状。
- 避免**隐式的双轴 expand**（PyTorch 会做，PyPTO 经常会错或难以追踪）。
- 行 / 列方向的 scale tensor 要**显式标注形状**（例如 `[B, 1, H]` vs `[B, S, 1]`），不要靠 implicit broadcasting。

**Host 侧 reshape / 批次维折叠规则**

当算子的数学形式是「带前导批次轴的 2D contraction」（前导若干 batch 轴 + 一次 matmul，如 `out[..., m, n] = Σ_k a[..., m, k] * w[k, n]`）时：

- 在 host wrapper（Layer K）里用 **torch** `reshape` / `squeeze` 把前导 batch 轴折叠掉，让 kernel 里的 `pypto.matmul` 保持 2D `[M,K]@[K,N]`，输出再 `reshape` 回用户 rank。
- **不要**把另一个 operand `unsqueeze` 升到和输入一样的高 rank 去「凑维度」——那会逼出退化的 `[1, 1, ...]` 高 rank matmul 和多层嵌套动态 loop，得不偿失。
- 行主序连续张量折叠前导维是 no-copy view，输出 reshape 是其逆操作，语义可逆。

**大轴 scan / 超 UB reduction 的分块 + carry 规则**

仅适用于「沿一个大轴做 native scan/cumulative（`pypto.cumsum` / `pypto.cumprod` 等），或沿大轴做的 native reduction，其**单 op UB（数据 + 内部 workspace）超过 UB 预算**」的场景（例：`pypto.cumsum` 对 4000 长的轴需要 ~256 KB > 192 KB）。此时：

- 把该大轴切成长度 T 的 **block**，T 选得让「该 op 的 per-block UB 落在预算内」（例：cumsum 取 T=1000 → 64 KB < 192 KB）。
- 用一个**少量迭代**的 block loop：`pypto.view` 切出 `[.., T]` 的**实 block** → 对 block 调 native op → 跨 block 传播 carry / accumulator → `pypto.assemble` 写回。block 间有数据依赖时，在 block loop 上设 `submit_before_loop=True`（迭代数小，无 task 爆炸风险）。
- **禁止**两种退化写法：① 对整个大轴一次性调 native op（UB 超过 → OOM）；② 用 `view([1, 1])` 在大轴上逐元素 loop（正确但产生 `axis_len × batch` 量级的 task，导致 host crash / timeout）。

> **注意**：这是 loop / view 的**结构**规则，与 vec tile 尺寸设计是两件事。**block 长 T ≠ vec tile 尺寸**：上面 T=1000 时 vec tile 仍可保持小尺寸（如 (16, 16)），正确性由分块结构保证，不由 tile 值决定。**不要**因此改动既有 vec tile 规则（每轴 ∈ [16, 64]、rank 匹配、single-op UB fit）。

**Layer K Host wrapper：output buffer 必须用 `torch.*` 预分配后再传入 JIT kernel**

JIT kernel 不在 host wrapper 内部 allocate output——output buffer 必须由 Layer K（host wrapper）用 **torch.\*** 预先开好，再作为参数传给 `@pypto.frontend.jit` 入口。

- `pypto.zeros / pypto.empty / pypto.ones / pypto.full` 是 **JIT-context API**（只在 `@pypto.frontend.jit` 函数体内部合法），在 host wrapper 里调用会 runtime crash：
  - `pypto.zeros((B, N), dtype=..., device=x.device)` → `TypeError: pypto.zeros() got an unexpected keyword argument 'device'`
  - `pypto.zeros((B, N), dtype=pypto.DT_FP32)` → `RuntimeError: ASSERT FAILED: ErrCode: F21003! FeError::INVALID_TYPE`
- host wrapper 内**只能**用 torch 等价物：`torch.empty / torch.zeros / torch.ones / torch.full / torch.empty_like / torch.zeros_like`，**显式带 `dtype=` 与 `device=`**，再传给 JIT 入口。

DESIGN.md §4（Loop + data flow / Layer K wrapper section）必须把这一步**显式写出来**——把 output allocation 当成 wrapper 责任的一部分，不要丢给 kernel 自己处理。

```python
# ❌ 错误（Matmul_Mish_Mish 实测 bug：host wrapper 内用 pypto.zeros）
def matmul_mish_mish_wrapper(x, w, b):
    out = pypto.zeros((x.shape[0], 20), dtype=pypto.DT_FP32, device=x.device)
    matmul_mish_mish_kernel_npu(x, w, b, out)   # ← runtime crash before reaching kernel
    return out

# ✅ 正确：host 侧 torch 预分配 → 传入 JIT
def matmul_mish_mish_wrapper(x, w, b):
    out = torch.empty(x.shape[0], 20, dtype=torch.float32, device=x.device)
    matmul_mish_mish_kernel_npu(x, w, b, out)
    return out
```

> 与 §4 的 layer 责任对应：Layer K = host-only torch 操作（reshape / allocate / 调 kernel 一次）；JIT 入口收到的 output 必须是已分配的 buffer。`pypto.zeros` 等 creation API 只在 Layer H/I（JIT 图内）才能用（例如临时 workspace 张量）。**禁止**在 Layer K wrapper 内调用 `pypto.zeros / empty / ones / full`；传给 JIT 入口的 output 参数必须先用 `torch.*` 预分配。
>
> 这些约束不是性能问题，而是 PyTorch 与 PyPTO lowering 之间的语义差异。第 1 轮已经选好 API 后，**进入第 3 轮的 loop 设计前必须把这五点钉清楚**。

#### 3.1 动态轴分析

逐轴判定：
- **编译期已知且单 tile 可覆盖** → **不标 DYNAMIC**，不需要 loop
- **编译期已知但超出 tile** → **不标 DYNAMIC**，用 Python for 或编译器自动切分
- **运行时才确定大小** → **标 `pypto.DYNAMIC`**，用 `pypto.loop`

**`for ... in range(...)` 的使用条件**：JIT 图代码内的循环应优先使用 `pypto.loop` / `pypto.loop_unroll`。仅当循环变量需要作为 **Python 具体整数** 参与以下操作时，才使用 `for ... in range(...)`（编译期全展开）：

1. **Python 数据结构索引** — 用作 list 下标、dict 键、`.append()` 等（SymInt 不可索引 Python 容器）
2. **Python 级条件分支** — `if i == 0` 区分初始化与累加、选择不同命名变量等（SymInt 不可做 Python `==` 判断）
3. **嵌套在 `pypto.loop_unroll` 内部** — 外层已生成动态循环，内层需具体 int 计算 offset 或索引 block table

#### Production kernel 动态轴 4 要素（canonical，必须同时具备）

接受任意 shape 的 production kernel 必须同时具备以下 4 要素，**缺一就 production
unusable**：

| # | 要素 | 示例 | 缺失时的后果 |
|---|------|------|-------------|
| 1 | Annotation 中标注 `pypto.DYNAMIC` literal | `pypto.Tensor([pypto.DYNAMIC, 16, 64], dtype)` | 下游无法识别动态轴 |
| 2 | Tile config | `pypto.set_vec_tile_shapes(1, 16, 64)` | 编译失败 |
| 3 | 沿动态轴的 loop | `for b in pypto.loop(B, name=..., unroll_list=[1])` | 运行时无法遍历动态轴 |
| 4 | loop 内 view 切 concrete tile | `pypto.view(x, [1, 16, 64], [b, 0, 0])` | workspace estimator overflow / OOM |

**Canonical 模板**（可直接复用）：

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 16, 64], pypto.DT_FP32),    # ① DYNAMIC + 2 静态轴
    y: pypto.Tensor([pypto.DYNAMIC, 16, 64], pypto.DT_FP32),
    ...
):
    pypto.set_vec_tile_shapes(1, 16, 64)                           # ② tile (3D 匹配 view 维数；batch=1 是 loop-collapsed，末两轴 ∈ [16, 64])
    B = x.shape[0]                                                  # SymbolicScalar
    for b in pypto.loop(B, name="batch", unroll_list=[1]):         # ③ loop (Stage 6 之前单一值，默认 [1])
        x_tile = pypto.view(x, [1, 16, 64], [b, 0, 0])              # ④ view (1×16×64 = 1024 elem = 4 KB FP32)
        # 在 concrete tile 上进行 compute
        ...
        pypto.assemble(result, [b, 0, 0], y)
```

> **`unroll_list` 在 Stage 6 之前只能含单一值（默认 `[1]`）**：从 DESIGN 到 Stage 6，
> 每个动态 loop 的 `unroll_list` **只能写一个值**。默认用 `[1]`（关闭循环展开）；
> 若有依据（如已知静态边界的某个约数）也可用其它**单一**值，但必须在 DESIGN.md §4
> 记录所选值与理由。**禁止在 Stage 6 之前写多值 `unroll_list`**（如 `[16, 8, 4, 2, 1]`）——
> 多值会为每个迭代次数生成一条编译路径，导致编译路径爆炸、显著拖慢编译并使开发流程
> 超时。多值展开调优**仅允许在 Stage 7 optimization** 进行。
>
> **嵌套 loop 时的 `unroll_list` 归属**：若按 ③ 行的 batch loop 内再嵌套一层 `pypto.loop`（例如末轴更大时进一步切分），必须把 `unroll_list` 从外层 batch loop **移到最内层 `pypto.loop`**。外层 loop 不能带 `unroll_list`——否则触发编译路径爆炸或寄存器拷贝 pass 引起的精度异常。

#### 尾块处理：`valid_shape` 必须 ≤ `shape`（用 `.min(TILE)` clamp）

沿动态轴 `B` 用固定 `TILE_B` 切 tile 时，若 `B` 不能整除 `TILE_B`，最后一个反复
会不足 `TILE_B` 行。这种 tail 通过 `pypto.view(..., valid_shape=[...])` 表达
「declared shape 中实际有效的部分」，约定上必须满足：

> **`valid_shape[i] ≤ shape[i]` 对每个轴 `i` 都成立。**

注意 `rem = B - offset` 这种 SymbolicScalar 在**非末尾反复中会大于** `TILE_B`
（例：`B=128`, `TILE_B=16` 时，第 0 次迭代 `rem=128`, 远大于 `TILE_B=16`），直接传给
`valid_shape` 违反约束。必须用 `.min(TILE_B)` clamp，使运行时始终 `rem ≤ TILE_B`：

```python
# ❌ 错误（实测 bug，源自 MSELoss）：rem 未 clamp
rem = B - offset                           # SymbolicScalar，可能 ≫ TILE_B
p_tile = pypto.view(x, [TILE_B, D], [offset, 0],
                    valid_shape=[rem, D])  # valid_shape[0] > shape[0]，违法

# ✅ 正确：用 .min(TILE_B) clamp
rem = (B - offset).min(TILE_B)             # SymbolicScalar.min(int) → SymbolicScalar
p_tile = pypto.view(x, [TILE_B, D], [offset, 0],
                    valid_shape=[rem, D])  # 保证 valid_shape[0] ≤ TILE_B
```

**Production 参考**（用 skill `pypto-docs-search` 搜索递归/gated 类算子范本，如 gated_delta_rule 之类）的标准写法：

```python
actual_l = (s - s_idx).min(l)              # ← .min(l) 把上限固定在 declared shape
query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0],
                        valid_shape=[actual_l, 1, d])
```

**简化优先**：若 P0 形状中动态轴**整除** `TILE`（例 `B=128`, `TILE_B=16` →
`128/16=8` 无尾），直接用 `pypto.loop(B // TILE_B, ...)` 配
`pypto.view(..., [TILE_B, D], [b*TILE_B, 0])`（**不带 `valid_shape`**），无需
tail 处理。仅当 `B` 范围可能不整除 `TILE_B` 时才使用 `valid_shape + .min(TILE_B)`
模式。

> 这是 SymbolicScalar 在 `valid_shape` 上的**语义**约束。rank 一致只是
> 形式要求；**形式 rank 一致 ≠ 值 ≤ shape**，两者必须同时满足。

#### Anti-pattern（必须避免）

❌ **空 `pypto.Tensor()` / `pypto.Tensor([])` 注解**（即 "per-shape compile" 逃避路线）

```python
# WRONG — 看似能编译通过，production 必崩
def kernel(
    x: pypto.Tensor([], pypto.DT_FP32),     # 无 DYNAMIC literal
    ...
):
    pypto.set_vec_tile_shapes(16, 16, 64)
    sum_x = x.sum(-1, keepdim=True)         # 全 tensor 直接 sum，无 loop/view
```

为什么禁止：
1. **任意 shape 下无法工作** — 仅 P0 shape 能 PASS；其他 shape 立即 crash
2. **绕过动态轴声明** — 需将 DESIGN.md 写成 `dynamic_axes: []` 才能闭环，违反 SPEC 中的动态轴声明
3. **workspace estimator 不稳定** — implicit tiling 在大 input（例如 B≥16）会失败
4. **下游 KernelVerifier 多形状测试时暴露** — verifier 一旦用非 P0 shape 测就报错

❌ **省略 `pypto.view`，将 DYNAMIC tensor 直接喂给 compute API**

```python
# WRONG — workspace estimator overflow / OOM
def kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 16, 64], pypto.DT_FP32),
    ...
):
    pypto.set_vec_tile_shapes(1, 16, 64)
    sum_x = x.sum(-1, keepdim=True)   # 全 tensor 直接 sum，无 loop/view
                                       # → workspace estimator 看到 DYNAMIC 不能 bound：
                                       #   大 B 时 INT32 overflow / OOM
```

→ 必须先用 `pypto.view` 切出 concrete tile，再做 compute（详见 §9.13）。

❌ **将 `pypto.is_loop_begin` / `pypto.is_loop_end` 切到非 JIT 辅助函数内**

```python
# WRONG — 编译期 F00002 Not concrete value，报错栈不指向具体行
def _kernel_impl(...):
    for idx in pypto.loop(N):
        if pypto.is_loop_begin(idx):
            ...

@pypto.frontend.jit(...)
def kernel_npu(...):
    _kernel_impl(...)
```

设计 Layer I / Layer H 切分时，如果辅助函数 body 需要用 `pypto.is_loop_begin` / `pypto.is_loop_end`，**Coder 必须把这部分逻辑 inline 到 `@pypto.frontend.jit` body**，或在辅助函数上加 `@pypto.frontend.function` 装饰器（仅支持 tensor 参数）。Designer 在 DESIGN.md §4 写 Layer I 伪代码时，**避免**把 loop begin/end 分支逻辑写在独立辅助函数里。详见 `pypto-op-develop/SKILL.md` 实现注意点。

#### Cross-reference

- `pypto-general-debug/references/jit-signature.md` §9.13 — INT32_MAX overflow 的正确解决方式
- `pypto-op-design/references/quick_ref.md` §2.4 — 动态轴 quick reference

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
- [ ] `derive_design_params.py` 的 `checks` 全 PASS（基线 / alignment / UB / 展开 / tile 静态性）
- [ ] 人工项：tile 维数 = tensor 维数；cube dim 关系 (mL0 ≤ mL1)；broadcast 1 轴 rule；per-stage vec tile（多 shape class 场合）；tile dims ≠ tensor.shape；tile_size × dtype_bytes ∈ 16–64 KB（recurrent 可豁免）
- [ ] DESIGN.md §3.2.2 rationale（1-3 句）、§3.2.3 alternatives（1-3 个）、§3.2.4 全部 ☑

**Loop 层（→ 第 3 轮）**
- [ ] 输出写回用 `[:]` / `.move()` / `assemble()`
- [ ] 无 view + assemble 环路
- [ ] 动态轴标了 `pypto.DYNAMIC`，静态轴未标
- [ ] 动态 loop 有 `unroll_list`（**嵌套场景下仅最内层 `pypto.loop`** 携带；外层 loop 不能加）
- [ ] `unroll_list` 在 Stage 6 之前**只含单一值**（默认 `[1]`；有依据时可用其它单值并在 §4 记录理由；禁止多值；多值调优留到 Stage 7）
- [ ] 跨迭代依赖 → `submit_before_loop=True`
- [ ] 尾块 → 在 `pypto.view` / `pypto.reshape` 处传 `valid_shape=...`（**不是** `pypto.assemble` 的参数；assemble 不接受 `valid_shape`）

**SymbolicScalar 检查**
- 伪代码中无 `sym ** n`、`list[sym]`、`if sym:` 等禁止操作
- 动态轴相关计算使用 `.min()` / `.max()` 而非 Python `min()` / `max()`
- 动态维度不直接用于 matmul 的 M/K/N 维度

---

## 3. DESIGN.md 输出结构与参考

**输出模板**：[templates/design-template.md](templates/design-template.md)

| 章节 | 对应迭代 | 必须内容 |
|------|---------|----------|
| §0 Decomposition Decision | 第 0 轮 | 复杂度信号采集、total_complexity、module_count、（≥2 时）数据流断点列表 |
| §1 计算图与精度路由 | 第 1 轮 | API 序列、dtype 流、cast 点、备选方案 |
| §2 数据规格 | SPEC + 第 1 轮 | kernel 签名、动态轴标注、值类型分析 |
| §3 Tiling 策略 | 第 2 轮 | tile 参数 + 推导过程 + 约束验证 |
| §4 Loop 与数据流 | 第 3 轮 | **完整伪代码** + 数据搬运 + 尾块处理 |
| §5 约束检查与开放问题 | 第 4 轮 | 检查清单 + SymbolicScalar 审查 |
| §6 验证计划 | SPEC | 测试配置 + 精度容差 |

**参考资源**：
- [references/quick_ref.md](references/quick_ref.md) — 约束速查与冲突表
- 用 `pypto-docs-search` 搜索 API 文档（`docs/api/`）— API 签名与 dtype 约束（最高优先级）
- 用 `pypto-docs-search` 搜索教程（`docs/tutorials/`）— 使用模式
- 用 `pypto-docs-search` 搜索算子参考实现— 真实算子实现参考

---

## 4. 完成报告

```text
设计状态：{已收敛 / 有待确认项}

迭代过程：
  第 0 轮：total_complexity = {value}，module_count = {1 | N}（{L0|L1}）
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
