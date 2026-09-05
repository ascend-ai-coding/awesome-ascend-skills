---
name: external-cannbot-ops-pypto-op-perf-tune-tune-frontend
description: PyPTO 算子开箱性能调优技能。主要关注代码级的调优、前端写法不同导致的性能差异，包括 loop 写法优化、TileShape 设置优化、数据操作优化等。当用户需要进行算子初始开发性能优化、开箱性能调优时使用此技能。触发词：开箱性能调优、代码级优化、loop
  优化、TileShape 设置、前端优化。
original-name: tune-frontend
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO 算子开箱性能调优

## 概述

开箱性能调优主要关注代码级的调优、前端写法不同导致的性能差异。在算子初始编写过程中直接得到较好的开箱性能。

> 资料获取统一使用 skill `pypto-docs-search`：按需搜索算子参考实现与 tile 配置等文件/目录/内容。

## ⛔ 调优三阶段流程（编排器强制执行）

开箱调优必须严格按以下三阶段顺序执行，编排器会逐阶段核查输出制品：

```
阶段A: 全局分析 (F-1~F-10) ──→ 阶段B: 局部分析 (F-11~F-15) ──→ 阶段C: 逐项优化
  │                               │                               │
  │ 产出:                         │ 产出:                         │ 产出:
  │ · A1 Loop结构分析表           │ · B1 数据操作分析表            │ · 逐项优化记录
  │ · A2 常量依赖关系图           │                               │
  │ · A3 Reshape全局分析表        │                               │
  │ · A4 基本块(TileShape)审查表  │                               │
  │                               │                               │
  └──── 编排器核查 ───────────────┘└──── 编排器核查 ───────────────┘└─ 按分析结果逐项优化
       制品完整性                     制品完整性                     每项验证精度+性能
```

**⛔ 禁止：未完成阶段A+阶段B的分析就直接进入阶段C逐项优化。**
**⛔ 禁止：凭直觉选择优化点跳过分析环节。**

## 阶段A: 全局分析（对应优化点 F-1~F-10）

**目标**：从算子整体结构出发，理解循环组织、常量依赖、Reshape 分布、基本块(TileShape)配置。对应阶段C中的"全局性能优化"（F-1~F-10）。

### A1. Loop 结构分析（对应优化点 F-1~F-3, F-5~F-8）

调 unroll / stitch 前，先用 `pypto-docs-search` 搜索算子参考实现中现有 production kernel 的 loop 写法（如 `loop_unroll` / `unroll_list` / `stitch`）作为初始候选，不要从零猜。

逐行扫描算子 kernel 代码中所有 `pypto.loop` 和 Python `for`/`range` 调用，填写下表：

| # | 循环名 | 代码行 | 类型(pypto.loop/Python for) | 轴性质(静态/动态) | 循环次数 | 循环体主要操作 | 最内层? |
|---|--------|-------|---------------------------|------------------|---------|-------------|--------|
| 1 | LOOP_n2 | L272 | pypto.loop | 动态(num_kv_tiles) | 4 | Q@K^T matmul, P@V matmul, online softmax | 否(外层) |
| 2 | range(num_kv_heads) | L272 | Python for | 静态(8) | - | - | - |

**关键检查项**：
- [ ] 静态轴是否使用了 `pypto.loop`？（应改为 Python for）
- [ ] 最内层循环次数是否 > 100？（应切块）
- [ ] 最内层循环体计算量是否太小？（应 unroll 或增大切块）
- [ ] 是否有可合并的独立 loop？
- [ ] 外层循环体内的常量/配置是否与内层循环变量无关联？（可外提）

### A2. 常量与参数依赖分析（对应优化点 F-11）

扫描算子中所有硬编码常量（如 BLOCK_SIZE、s_tile、g_tile 等），以及 `jit` 装饰器中的 `runtime_options`、`pass_options`：

| # | 常量名 | 值 | 代码行 | 被引用位置(行号) | 引用的TileShape/基本块 | 影响范围 |
|---|--------|-----|-------|-----------------|---------------------|---------|
| 1 | s_tile | 512 | L146 | L282,L292,L298,L304,L308,L318 | vec_tile(s_tile, kv_size) | 循环次数、view shape、vec_tile |

**关键检查项**：
- [ ] 常量值变更后，所有引用该常量的 `set_vec_tile_shapes` / `set_cube_tile_shapes` 是否需要同步调整？
- [ ] `runtime_options` / `pass_options` 的每个参数是否有明确的作用说明？

### A3. Reshape 全局分析（对应优化点 F-4）

对算子中**每一个** `pypto.reshape` 调用，逐个分析：

**分析方法**：
1. 使用 `grep -nE "pypto\.(reshape|squeeze|unsqueeze)" <算子文件>` 获取所有 reshape/squeeze/unsqueeze 调用位置
2. 逐行分析每个操作的输入来源、目标 shape、是否在 loop 内

| # | 代码行 | 输入 Tensor | 源 Shape | 目标 Shape | 在loop内? | 输入类型(原始/中间/输出) | inplace? | 冗余? | 问题与建议 |
|---|--------|------------|---------|-----------|----------|----------------------|----------|------|-----------|
| 1 | L183 | input_ln_weight | [4096] | [1,4096] | 否 | 原始输入 | ✅ | 否 | 无问题 |
| 2 | L246 | k_embed | [8,128] | [8,128] | 否 | 中间结果 | ❌ | **✅冗余** | 源shape==目标shape，删除 |

**关键检查项**：
- [ ] 源 Shape == 目标 Shape 的冗余 reshape？（应删除）
- [ ] 原始输入的 reshape 是否在 loop 外？（应外提）
- [ ] 原始输入的 reshape 是否使用了 `inplace=True`？（必须用）
- [ ] 原始输入的 squeeze/unsqueeze 是否在 loop 内？（⚠️ squeeze/unsqueeze 不支持 inplace，应替换为等价的 `pypto.reshape(..., inplace=True)` 外提）
- [ ] 中间结果或输出 tensor 是否使用了 `inplace=True`？（禁止用）
- [ ] loop 内的 reshape 是否依赖循环变量？（不依赖则外提）
- [ ] reshape 前后是否有对应的 `set_vec_tile_shapes` 变更？

### A4. 基本块(TileShape)审查（对应优化点 F-9, F-10）

调 TileShape 前，先用 `pypto-docs-search` 搜索算子参考实现中现有 production kernel 的 tile 配置（如 `set_vec_tile_shapes` / `set_cube_tile_shapes`）作为初始候选，不要从零猜。

对算子中**每一个 operation**，逐行审查其 shape 与 TileShape 设置：

**分析方法**：
1. 使用 `grep -n "set_cube_tile_shapes\|set_vec_tile_shapes\|matmul\|cast\|mul\|add\|sum\|exp\|div\|assemble\|view" <算子文件>` 获取所有 operation 和 TileShape 调用
2. 逐行配对：每个 operation 前最近的 TileShape 设置是否匹配该 operation 的实际 tensor shape

| # | 代码行 | Operation | 输入 Shape | 输出 Shape | 前置TileShape | 合理? | 问题与建议 |
|---|--------|-----------|-----------|-----------|-------------|------|-----------|
| 1 | L292 | view(key_cache) | [max_kv_len, kv_size] | [s_tile, kv_size] | vec(s_tile=2048, kv_size=1024) | ⚠️ | 数据量4MB远超UB |

**关键检查项**：
- [ ] `vec_tile_shapes` 每维是否 ≤ 对应 tensor 实际维度？
- [ ] `vec_tile_shapes` 数据量是否在 16~64KB 范围内？
- [ ] `cube_tile_shapes` 的 L1 是否超过实际轴长？
- [ ] 多个不同 shape 的 matmul 是否各自独立设置了 `cube_tile_shapes`？
- [ ] Decode 场景 M=1 的 matmul 是否使用了 K 轴三维配置 `[kL0, kAL1, kBL1]`？
- [ ] reshape 前的 vec_tile 是否按源 shape 设？reshape 后是否按目标 shape 重设？
- [ ] assemble 前的 vec_tile 是否匹配目标 shape？

**冗余操作一并检查**：

| # | 检查类型 | 代码行 | 具体操作 | 是否冗余 | 建议 |
|---|---------|--------|---------|---------|------|
| 1 | 重复TileShape | L94, L109 | 两次 `set_vec_tile_shapes(1, 4096)` | 可能冗余 | 同shape连续设置可合并 |
| 2 | 冗余reshape | L246 | reshape 到相同 shape | 冗余 | 删除 |

**基本块全局分析：op 前后 TileShape 边界审查**：

检查每个 op 前后的 TileShape 设置一致性，识别基本块（子图）拆分边界。基本块的拆分由 TileShape 设置决定——当相邻 op 的 TileShape 不兼容（维度不匹配、切分粒度不统一）时，它们会被分配到不同子图，形成多对一/一对多/多对多的基本块拼接模式，增加 GM 搬运和调度开销。

**基本块拼接模式说明**（由 TileShape 边界决定，非数据依赖）：

| 模式 | 形成原因（TileShape 视角） | 示意图 | 性能影响 |
|------|--------------------------|--------|---------|
| 一对一 | op A 的输出 TileShape 与 op B 的输入需求兼容 → 合并为同一子图 | `A→B` | 最优 |
| 一对多 | op A 的 TileShape 与多个下游 op 的 TileShape 均不兼容 → 每个下游自成一子图 | `A→{B,C}` | 重复搬运相同数据 |
| 多对一 | 多个上游 op 的 TileShape 各不同，到 op B 处需汇聚 → GM 落地后再合并 | `{A,B}→C` | 中间结果写回 GM |
| 多对多 | 同时存在前后 TileShape 多方向不兼容 | `{A,B}→{C,D}` | 最复杂，搬运叠加 |

**分析方法**：
1. 逐行定位每个 op 前最近的 `set_cube_tile_shapes` / `set_vec_tile_shapes` 调用
2. 检查相邻 op 的 TileShape 是否兼容：前序 op 的输出 shape 的 tile 切分，是否能被后续 op 的 TileShape 直接消费
3. 判断每个 op 边界的拼接模式

**审查表格**：

| # | 拼接模式 | op位置 | 前序TileShape | 该opTileShape | 后序TileShape | 边界问题 | 优化建议 |
|---|---------|--------|-------------|--------------|-------------|---------|---------|
| 1 | 一对多 | cast(L80) | vec(8,128) | vec(8,128) | sub(L85):vec(8,128), add(L86):vec(8,128) | 无，TileShape兼容 | 可合图 |
| 2 | 多对一 | cast(L87) | sub(L85):vec(8,128), add(L86):vec(8,64) | vec(8,128) | - | sub和add的尾轴TileShape不同(128 vs 64) | 统一add的vec_tile为(8,128)使多对一变为一对一 |
| 3 | 一对多 | gathermask(L72) | vec(256,128) | vec(32,128) | cast×6(L79-L84):各vec(32,128) | gathermask切出6个tensor，每个独立子图 | 增大vec_tile减少gathermask次数 |

**关键检查项**：
- [ ] 相邻 op 的 TileShape 是否兼容？（输出 shape 的 tile 切分能否被下游直接消费）
- [ ] 是否存在**一对多 fan-out**导致重复搬运？（优化：统一下游 TileShape 使兼容，或使用合图消除 GM 落地）
- [ ] 是否存在**多对一 fan-in**导致 GM 中间结果？（优化：调整上游 TileShape 使兼容，拆解为多个一对一链）
- [ ] 是否存在**多对多**复杂结构？（优化：先拆分基本块为多对一+一对多的组合，再逐级优化）
- [ ] fan-out/fan-in 是否在**最内层热循环**中？（优先级最高，调度开销叠加最明显）
- [ ] 共享输出是否可通过调整 TileShape 减小数据量，从而降低搬运开销？

### A5. 全局分析输出要求

阶段A完成后，**必须产出**：
1. **Loop结构分析表**（按 A1 模板填写）
2. **常量依赖关系图**（按 A2 模板填写）
3. **Reshape全局分析表**（按 A3 模板填写，每个 reshape 一行）
4. **基本块(TileShape)审查表**（按 A4 模板填写，每个 operation 一行）
5. **基本块 TileShape 边界分析表**（按 A4 新增模板填写，标注 op 前后 TileShape 拼接模式）
6. **基于分析的优化建议清单**（标注优先级 P0/P1/P2）

编排器核查：以上 6 项制品齐全 → 允许进入阶段B。

---

## 阶段B: 局部分析（对应优化点 F-11~F-15）

**目标**：对算子中的数据操作逐行审查，识别局部优化机会（NZ格式、Transpose融合、冗余搬运消除、尾轴 broadcast 合轴）。对应阶段C中的"局部性能优化"（F-11~F-15，其中 F-11 常量分析已在阶段A2完成）。

### B1. 数据操作分析（对应优化点 F-12~F-15）

对算子中**所有涉及数据格式和搬运的操作**，逐行分析：

**分析方法**：
1. 使用 `grep -n "transpose\|concat\|assemble" <算子文件>` 获取相关调用
2. 检查权重矩阵的 shape 和访问模式（F-12）
3. 检查是否有 transpose + matmul 模式（F-13）
4. 检查 concat 是否可替换为 assemble（F-14）
5. 扫描所有 tensor shape，查找尾轴为 1 的 broadcast 二元运算（F-15）

| # | 检查类型 | 优化点 | 代码行 | 具体操作 | 适用? | 问题与建议 |
|---|---------|--------|--------|---------|------|-----------|
| 1 | 输入矩阵格式 | F-12 | - | 权重矩阵 shape 检查 | ❌/✅ | Shape 较大时可尝试 NZ 格式 |
| 2 | Transpose+Matmul | F-13 | - | transpose 后紧跟 matmul | ❌/✅ | 可通过 a_trans/b_trans 融合 |
| 3 | 冗余搬运 | F-14 | - | concat → assemble 替换 | ❌/✅ | 可替换 concat 为 assemble |
| 4 | 尾轴 broadcast | F-15 | - | `[M,1]*[M,N]` 等尾轴1 broadcast | ❌/✅ | 可添加 `combine_axis=True` 内联 brcb |

**关键检查项**：
- [ ] 权重矩阵 Shape 较大（>1024）？（F-12：可尝试 NZ 格式）
- [ ] 有 transpose 后紧跟 matmul 的模式？（F-13：可用 a_trans/b_trans 融合）
- [ ] 有 concat 数据搬运操作？（F-14：可替换为 assemble）
- [ ] 存在尾轴为 1 的 tensor 参与 broadcast 二元运算？（F-15：可尝试 `combine_axis=True`）

### B2. 局部分析输出要求

阶段B完成后，**必须产出**：
1. **数据操作分析表**（按 B1 模板填写，覆盖 F-12~F-15）
2. **最终优化点排序清单**（基于 A+B 分析结果，按优先级排序，引用 optimization_catalog.md 编号）

编排器核查：以上 2 项制品齐全 → 允许进入阶段C（逐项优化）。

---

## 阶段C: 逐项优化

**前提**：阶段A和阶段B的分析已完成，优化点排序清单已生成。

**执行方式**：按优化点排序清单，由编排器的迭代循环（ITER_START → ITER_MODIFY → ITER_VERIFY → ITER_MEASURE → ITER_RECORD → ITER_JUDGE）逐项执行。

**每项优化前必须确认**：
1. 该优化点来源于阶段A或阶段B的分析结论（有明确的分析表格行号引用）
2. 修改只涉及一个参数
3. 修改后该参数的依赖项是否需要同步调整（参考 A2 常量依赖关系图）
4. 如果本项优化涉及代码结构变更（loop合并/拆分、reshape移动、循环切块），必须在修改后重新检查受影响的 A1/A3/A4 表格行，更新过期的分析结论

**以下各章节为具体的优化操作指南，供 ITER_MODIFY 阶段参考。**

---

## 💡 关键启发

### 1. loop_unroll 的本质
- **目的**：增加并行度，让多个循环任务并行执行
- **unroll_list 数值含义**：代表并行块的大小（如[8,4,2,1]表示优先并行 8 个块）
- **关键**：数值代表并行块的大小，不是简单的循环展开次数

### 2. 优化循环结构的思路
- **合并循环** - 减少嵌套层级，增加单层迭代次数
- **外层动态轴** → 使用切块（静态切分）
- **内层动态轴** → 使用loop_unroll（动态展开）
- **黄金组合** → 外层切块 + 内层 unroll = 最优并行度

### 3. 性能优化三要素
1. **任务粒度** - 每个任务的计算量（越大越好）
2. **并行度** - 可以并行执行的任务数（越多越好）
3. **调度开销** - 任务切换的时间成本（越小越好）

**优化目标**：在保证任务粒度的前提下，最大化并行度，最小化调度开销。

## 调优方向（阶段C 参考指南）

以下各章节为阶段C逐项优化时的具体操作指南。每个章节对应 optimization_catalog.md 中的优化点编号。

**阶段C 优化顺序规则**：
1. **先全局后局部**：P0（F-1~F-4）→ P1（F-5~F-8）→ P2（F-9~F-10）→ P3（F-11~F-15）
2. **每次只执行一个优化点**，按 ITER 循环执行
3. **优先执行阶段A/B分析中发现的问题**，而非盲目按编号顺序

### 全局性能优化（P0: F-1~F-4, P1: F-5~F-8, P2: F-9~F-10）

#### 1. Loop 写法优化（F-1~F-3, F-5~F-8）

**增加 root function 的大小，减少它们的个数**
由于不同 root function 之间的子图不能合并，而子图合并是 PyPTO 优化性能的关键手段。

##### 1.1 静态轴使用 Python for 循环

`pypto.loop` 方法会按当前轴循环展开成不同的 root function。因此静态轴上的循环应使用 Python 的 for 循环。

```python
# ✅ 推荐：静态轴使用 Python for
for i in range(batch_size):
    result[i] = process(data[i])

# ❌ 避免：静态轴使用 PyPTO loop
for i in pypto.loop(batch_size, name="LOOP_1", idx_name="i"):
    result[i] = process(data[i])
```

##### 1.2 减少循环次数，增加并行度
###### 1.2.1 如果外层的动态轴范围很大，使用切块处理（高优先级）

算子的循环轴的 dim 数值范围往往较广，往往需要对其进行静态切分，否则循环次数太大。

示例：
```python
# 推荐：动态轴使用 loop 切块，对b_loop轴按128进行切分，提高并行度
bsz, h = x.shape
b = 128
b_loop = (bsz + b - 1) // b
for b_idx in pypto.loop(b_loop, name="LOOP_1", idx_name="b_idx"):
    b_valid = (bsz - b_idx * b).min(b)
    x_view = pypto.view(x, [b, h], [b_idx * b, 0], valid_shape=[b_valid, h])
    # Matmul
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    y = pypto.matmul(x_view, W)
```

###### 1.2.2 如果内层的动态轴范围很大，调整切块大小或使用loop_unroll展开，增加并行度
   ```python
   for idx in pypto.loop(A.shape[0] // 64, unroll_list=[128, 64, 8, 1], name="A", idx_name='b'):
       offset = idx * s2_tile
   ```

   **参数说明：**
   - `unroll_list=[8, 4, 2, 1]`: 展开因子列表，按优先级从高到低排列，数值代表并行块大小，优先尝试并行 8 个块
   - `[8, 4, 2, 1]`: 常用配置，适应性强
   - `[128, 64, 16, 8, 1]`: 适用于更大循环
   - `name`: 循环名称（用于调试）
   - `idx_name`: 循环索引变量名

    **⚠️ 重要原则：**
    - **loop_unroll 必须放在最内层循环！不要在外层循环使用 unroll_list**
    - 数值代表**并行块大小**，不是简单的展开次数
    - 目的是**增加并行度**，让多个任务可以并行执行
    - **unroll_list 的最大值，不要超过循环次数**
    - 改多值 `unroll_list` 后若精度回归，立即回退该 loop

###### 1.2.3 切块优化策略：外层切块 + 内层unroll

**核心思路**：对循环轴切块，减少循环次数，增大任务粒度，然后在最内层使用unroll增加并行度。

**适用场景**：当最内层循环次数过少（<8）无法有效unroll时，应考虑对外层轴切块。

**🔥 典型案例：Flash Attention（最常见错误）**

**❌ 错误实现（任务粒度过小）**：
```python
for q_idx in pypto.loop(SEQ_LEN_Q, name="LOOP_Q"):  # 64 次
    q_vec = pypto.view(query, [1, HEAD_DIM], [q_idx, 0])  # ❌ M 轴 = 1
    for kv_idx in pypto.loop(num_kv_blocks, name="LOOP_KV"):  # 1 次
        scores = pypto.matmul(q_vec, k_block, ...)  # ❌ [1, 128]
```

**问题**：
- Matmul M 轴只有 1，浪费 Cube 计算能力
- 循环 64 次，调度开销极大
- 每个任务计算量太小，任务粒度过细

**✅ 正确实现（Query 切块）**：
```python
Q_BLOCK_SIZE = 16
num_q_blocks = SEQ_LEN_Q // Q_BLOCK_SIZE  # 4 次

for q_block_idx in pypto.loop(num_q_blocks, name="LOOP_Q"):  # 4 次
    q_start = q_block_idx * Q_BLOCK_SIZE
    cur_q_size = pypto.min(Q_BLOCK_SIZE, SEQ_LEN_Q - q_start)

    # ✅ 批量获取 16 个 query
    q_block = pypto.view(query, [Q_BLOCK_SIZE, HEAD_DIM], [q_start, 0],
                        valid_shape=[cur_q_size, HEAD_DIM])

    for kv_block_idx in pypto.loop(num_kv_blocks, unroll_list=[4,2,1], name="LOOP_KV"):  # ✅ 可 unroll
        # ✅ Matmul M 轴 = 16
        scores = pypto.matmul(q_block, k_block, ...)  # ✅ [16, 128]
```

**收益来源**：
- ✅ 任务数: 64 → 4（减少 16 倍）
- ✅ Matmul M 轴: 1 → 16（计算量增大 16 倍）

**⚠️ 切块大小建议**:
- 从较大值开始尝试（如 64, 32, 16），但切开的值不应该超过shape中该维度的大小
- 平衡任务粒度和内存占用
- 调整中间 tensor 的 shape

##### 1.3 尽可能合并 loop

检查算子代码是否有可以合并的 loop 块：

```python
# ❌ 不推荐：两个独立的 loop
bsz = x1.shape[0]
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
    out_1 = Operation1(x1[b_idx, :], y)
for b_idx in pypto.loop(bsz, name="LOOP_2", idx_name="b_idx"):
    out_2 = Operation2(x2[b_idx, :], y)

# ✅ 推荐：合并 loop
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
    out_1 = Operation1(x1[b_idx, :], y)
    out_2 = Operation2(x2[b_idx, :], y)
```

#### 2. Reshape 全局优化

> **⛔ 执行 F-4 Reshape 优化时，必须加载 [Reshape 全局优化](references/reshape-global-optimization.md) 获取完整操作指南。**

#### 3. 基本块优化

> **⛔ 执行 F-9/F-10 TileShape 优化时，必须加载 [基本块优化](references/basic-block-optimization.md) 获取完整规范。**

### 局部性能优化（P3: F-11~F-15）

#### 局部§1. 输入矩阵格式优化（对应优化点 F-12）

检查输入矩阵、尤其是 Shape 较大的权重矩阵是否可以提前以 NZ 格式存储。

**NZ 格式的数据搬运到 L1 的带宽更高。**

**调优方法**：
- 算子入口处对权重矩阵调用 `tensor.to_format(pypto.NZ)` 转换格式
- 要求权重矩阵在算子调用前已按 NZ 格式存储在 HBM 中
- NZ 格式的数据搬运到 L1 的带宽更高，适合只读一次的大权重矩阵

#### 局部§2. Transpose 优化（对应优化点 F-13）

矩阵乘前后有 transpose 时，可以尝试更换左右矩阵并使用左右矩阵转置的配置。

当 M 轴较大、N 轴较小时，使得左右矩阵有更大的尾轴，提升搬运带宽。

**⚠️ 重要原则**
- `transpose + matmul` 的结构，可以通过 matmul 的 `a_trans` 及 `b_trans` 参数进行配置，完成 op 融合。好处是，matmul 运算时，可以随路 transpose

**代码示例**：

```python
# ❌ 不推荐：先 transpose 再 matmul
b_t = pypto.transpose(b, [1, 0])
y = pypto.matmul(a, b_t)

# ✅ 推荐：matmul 参数直接带转置
y = pypto.matmul(a, b, b_trans=True)

# 当 M 轴大于 N 轴时，交换左右矩阵
# A: [M, K], B: [N, K]
# y = matmul(A, B, b_trans=True)  →  Matmul shape: [M, N]
# 等价于 y^T = matmul(B, A, a_trans=True)  →  Matmul shape: [N, M]
```

#### 局部§3. 冗余搬运优化（对应优化点 F-14）

检查是否有不合理数据操作导致的冗余搬运：

**优化方法**：
- 更换 concat 为 assemble：当目标仅是将数据拼接到已有 tensor 的指定位置时，`pypto.assemble` 比 `pypto.concat` 开销更低
- 识别 concat 拼接后不再修改的场景，替换为 assemble 直接写入

**判断依据**：
- `concat`：需要分配新内存 + 数据搬运，开销高
- `assemble`：直接写入目标位置，零额外内存分配

#### 局部§4. 尾轴 Broadcast 合轴优化（对应优化点 F-15）

##### 局部§4.1 问题诊断

当算子中存在形如 `[M, 1] * [M, N]` 的尾轴 1 broadcast 二元运算时，默认编译策略会先将 `[M, 1]` 展开（broadcast）为 `[M, N]` 再执行计算，多一次数据搬运。

**典型来源**：online softmax 中的 `sum_update`/`max_update`（`pypto.sum(keepdim=True)` / `pypto.amax(keepdim=True)` 产生），形如 `[g_tile, 1]`。

**诊断步骤**：
1. 扫描算子所有 tensor shape，标记 shape 尾轴为 1 的 tensor
2. 追踪该 tensor 的参与的所有二元运算（mul/add/sub/div），检查另一侧 tensor 尾轴是否 >1
3. 确认尾轴 1 的 tensor 是否由前序 reduce 操作（`sum`/`amax` 等 + `keepdim=True`）产生（保证 GM 连续）

##### 局部§4.2 优化方法

在 JIT 函数体首行添加：

```python
pypto.experimental.set_operation_options(combine_axis=True)
```

编译器会将 `[32,1] + [32,128]` 优化为：`[32,1]` 通过 brcb 指令扩展到 `[32,8]`，再做 `[32,8] + [32,128]`，省去完整 broadcast 的数据搬运。

##### 局部§4.3 约束条件

- 尾轴 broadcast 输入尾轴**必须连续**，否则功能失效
- `pypto.sum(keepdim=True)` / `pypto.amax(keepdim=True)` 输出保证连续，符合条件
- 若前序是 COPY_IN，需在前端保证 GM 连续
- 设置是**局部**的，只影响当前 jit/loop 作用域

##### 局部§4.4 收益预期

- ✅ **Vector 密集算子**（纯 softmax、layer norm、激活函数）：预期有明显收益
- ✅ **尾轴 1 broadcast 位于内层热循环**且执行次数多：预期有收益
- ⚠️ **Cube 密集型算子**（大 matmul 占 >80% 时间）：收益有限，vector 操作非瓶颈

##### 局部§4.5 案例

详见 [尾轴 Broadcast 合轴优化案例](cases/combine-axis-broadcast.md)

## 参考资料

- [性能调优文档](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/performance.md)
- [GDR 算子案例](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/performance_case_GDR.md)
- [Matmul 高性能编程](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/matmul_performance_guide.md)
- [典型案例库](cases/README.md)
