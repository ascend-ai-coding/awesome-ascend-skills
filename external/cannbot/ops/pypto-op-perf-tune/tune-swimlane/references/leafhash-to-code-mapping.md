# leafHash → 前端代码映射方法

## 概述

`analyze_aiv_dep_chains.py` 脚本基于 `dyn_topo.txt` 输出的依赖链和优化建议使用 leafHash 标识节点。本文档描述如何将 leafHash 映射回前端 Python 代码中的具体位置，以便确定 `sg_set_scope` 的插入点。

## 自动映射工具

使用 `leafhash_to_code.py` 脚本自动完成映射：

```bash
# 查看所有 leafHash 的代码映射
python3 ../scripts/leafhash_to_code.py <output_dir>

# 查看指定 leafHash
python3 ../scripts/leafhash_to_code.py <output_dir> --leafhash 3907163356593077760

# 输出 JSON
python3 ../scripts/leafhash_to_code.py <output_dir> --json result.json
```

**输入文件**（`output_dir` 中）：
- `program.json` — 编译期函数信息，含 operations 的 `file`/`line` 字段（必需）
- `dyn_topo.txt` — 运行时任务拓扑，含 rootIndex、psgId、执行次数（可选）

**输出内容**：
- 每个 leafHash 对应的前端代码文件和行号范围
- 每行涉及的具体操作指令
- rootIndex、psgId、执行次数等运行时信息

## 手动映射步骤

### Step 1: 从依赖链建议中提取候选 leafHash

运行 `analyze_aiv_dep_chains.py`，从 Part 2 "sg_set_scope 优化建议" 中收集所有 leafHash。

### Step 2: 在 program.json 中查找函数

对每个 leafHash，在 `program.json` 的 `functions` 数组中查找 `hash` 字段匹配的函数。

### Step 3: 从 operations 中提取 file/line

函数的 `operations` 数组中每个 operation 包含 `file` 和 `line` 字段，直接指向前端源代码位置：

```python
for op in func.get('operations', []):
    f_val = op.get('file', '')   # 源文件路径
    l_val = op.get('line', '')   # 源代码行号
    opcode = op.get('opcode', '')
    if f_val and l_val:
        print(f"L{l_val}: {opcode}")
```

这是最直接的映射方式——编译器在编译时记录了每个 operation 对应的源代码行号。

### Step 4: 通过 rootIndex 确认循环层级

在 `dyn_topo.txt` 中，同一 leafHash 的所有行具有相同的 `rootIndex`。不同 rootIndex 意味着不同 `pypto.loop` 的循环体。

如果两个 leaf 的 rootIndex 不同，**跨 loop 不能用 sg_set_scope 合并**。

### Step 5: 综合判断可合并性

对脚本建议的每个链段，按以下检查清单逐项验证：

| 检查项 | 验证方法 | 通过标准 |
|:---|:---|:---|
| 数据依赖 | dyn_topo 中存在 VEC→VEC successors 边 | 有直接数据依赖 |
| 同循环层级 | rootIndex 比对 | 所有节点 rootIndex 相同 |
| 纯 vector 操作 | program.json ops 中无 A_MUL_B/A_MULACC_B | 无 cube 指令 |
| 无 cube 后置依赖 | dyn_topo successors 中无 coreType=1 | 后继不含 matmul |
| ✂ cube边界节点排除 | 脚本标记的截断点 | 有 cube 后继的节点不参与合并 |
| 代码行连续性 | file/line 映射验证 | scope 内只有被合并的操作，无夹杂 |

**只有全部通过的链段才是可合并的。**

### Step 6: 确认代码连续性，必要时调整代码顺序

通过 `file`/`line` 映射确定每个 leafHash 对应的前端代码行后，检查这些代码行在源文件中是否**紧密相邻且中间没有其他不相关的操作**。

`sg_set_scope` 会将包裹范围内的所有操作交给编译器处理。如果范围内夹带了不属于合并段的操作（如其他变量的 view、reshape 等），会导致不相关的操作也被合并进同一子图。

**如果代码不连续**，需要在前端代码中将待合并的操作调整到相邻位置：

1. 分析依赖关系，确认被移动的操作不影响其他计算逻辑
2. 将不相关的操作移到 sg_set_scope 包裹范围之外
3. 确保移动后的代码语义不变（PyPTO 是声明式构图，只要数据依赖不变，代码顺序可以调整）

**调整原则**：
- 只移动与合并段无数据依赖的操作
- 移动后的代码不能跨越 `pypto.loop` 边界
- 调整后必须重新运行精度验证

## 常见陷阱

### 1. CAST 不一定是 pypto.cast()

`pypto.view` 做 reshape 时，编译器会生成 CAST 指令做格式/layout 转换。必须通过 `file`/`line` 字段确认实际对应的前端代码行。

### 2. 一个前端操作可能被拆成多个 leaf

编译器会按子图划分策略将一个前端操作拆分到不同 leaf function 中。必须通过 `file`/`line` 映射确认覆盖范围，而非仅凭 opcode 推测。

### 3. rootIndex 跨循环边界

两个 leaf 即使存在 dyn_topo 中的 VEC→VEC 依赖边，如果 rootIndex 不同，则跨 loop 边界，不能用 sg_set_scope 合并。

### 4. ✂ cube边界节点不参与合并

标记 `✂ cube边界` 的节点其后继包含 cube 操作，不应裹进 sg_set_scope。

### 5. sg_set_scope 必须覆盖参与合并的每个 leaf 的全部操作

如果某个 leaf 的部分操作在 scope 外，该 leaf 不会完整合并。scope 必须覆盖所有参与合并的 leaf 对应的全部代码行，不能只包裹其中一部分。

### 6. sg_set_scope 范围内不能夹杂不相关操作

`sg_set_scope` 包裹的是前端代码的一段连续区域。如果两个待合并 leaf 对应的代码行之间夹着其他操作（如另一个变量的 view），直接包裹会把不相关操作也卷入。此时需要调整代码顺序。

## 完整示例

以 flash_attention_score_grad Level 1 为例：

```
脚本建议 1: psgId 1 → 2 → 1, 3 个节点
  3907163356593077760 [psg=1]: CAST+CAST
  2360323566658746396 [psg=2]: MUL+CAST+CAST+ROWSUM_SINGLE
  2768731787098226973 [psg=1]: MULS+SUB+EXP+DIV+SUB+MUL+CAST+CAST [✂ cube边界]
```

**运行 leafhash_to_code.py 获取映射**：

```
3907163356593077760: impl.py:110-111, rootIndex=0
  L110: COPY_IN (dy_i view)
  L111: COPY_IN (ao_i view)

2360323566658746396: impl.py:120-121, rootIndex=0
  L120: MUL, CAST
  L121: ROWSUM_SINGLE

2768731787098226973: impl.py:33-142, rootIndex=16
  L43: MULS, L44: SUB/EXP, L45: DIV, L48: SUB/MUL
  L141: CAST, L142: CAST
```

**验证**：
- 3907 → 2360: ✅ 依赖、✅ 同 rootIndex=0、✅ 纯 vec、✅ 无 cube 后继 → **可合并**
- 2360 → 27687: ✅ 依赖、❌ rootIndex 0→16 跨 loop → **不可合并**
- 27687: ❌ 有 cube 后继 → **不可合并**

**发现代码不连续**：

3907 对应 L110-111，2360 对应 L120-121，但 L113-117 是 smax_i/ssum_i 的 view 操作，不属于合并段。如果直接在 L110 前开启 scope、L121 后关闭 scope，会把 smax/ssum 的 view 也裹进去。

**调整代码顺序**：

将 smax_i/ssum_i 的 view（L113-117）移到 dy_i/ao_i view 之前，使合并段的代码紧密相邻：

```python
# 调整前:
#   L108-111: q_i, dy_i, ao_i view     ← leaf 3907 的部分
#   L113-117: smax_i, ssum_i view       ← 不相关，夹在中间
#   L119-121: D_i = sum(cast(mul(...))) ← leaf 2360

# 调整后:
sm_i_8 = pypto.view(sm_2d, [S_TILE, 8], ...)
ss_i_8 = pypto.view(ss_2d, [S_TILE, 8], ...)
smax_i = pypto.view(sm_i_8, [S_TILE, 1], ...)
ssum_i = pypto.view(ss_i_8, [S_TILE, 1], ...)

pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
q_i  = pypto.view(q_2d, ...)
dy_i = pypto.view(dy_2d, ...)
ao_i = pypto.view(ao_2d, ...)

pypto.set_pass_options(sg_set_scope=1)                 # 开启
dy_ao_fp32 = pypto.cast(pypto.mul(dy_i, ao_i), pypto.DT_FP32)
D_i = pypto.sum(dy_ao_fp32, -1, keepdim=True)
pypto.set_pass_options(sg_set_scope=-1)                # 关闭
```

**验证**：调整后 Level 1 和 Level 3 精度均通过。
