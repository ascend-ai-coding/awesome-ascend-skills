### 合图调优

合图是指将计算图中多个逻辑上独立的 Task 合并为一个逻辑子图。

**⛔ ⛔ ⛔ ⛔ 合图前强制三步分析（禁止跳过任何一步）：⛔ ⛔ ⛔ ⛔**

1. **第 1 步：核使用率分析**（analyze_core_usage.py）— 判定每个 leafHash FULL/NOT FULL。核未满必须优先通过 TileShape 填核，禁止跳过直接合图。
2. **第 2 步：泳道图分析**（analyze_swimlane.py --outer-loops N）— 获取每个 hashOrder 的 subGraphCount、t/iter，输出 Merge Tuning Guide。**必须基于当前最新性能数据的泳道图运行此脚本**，禁止使用旧数据或直接猜测 hashOrder。
3. **第 3 步：根据 Merge Tuning Guide 的 hashOrder 和粒度建议，设置特定参数**。禁止不用分析结果盲试全局参数（如 `{-1: N}`）。

**⚠️⚠️⚠️ 关键原则：**
1. **⛔ 合图前必须先完成核使用率分析（第 3 节）**：核未满时优先用满核，核满后再合图。盲目合图会导致核未满时性能退化。
2. pass_options key 有两种格式（**同一 dict 内禁止混用**）：
   - **整数键格式**：`{-1: N}`（全局通配所有子图），`vec_nbuffer_setting` 中必须加 `-2: 1` 作为 merge enable 标志
   - **字符串键格式**：`{"DEFAULT": M, "func{magic}_{order}": N}`（精细控制特定 hashOrder）
   - **场景选择**：需要所有子图均合图时用整数键 `-1`；仅需合图某几个 hashOrder 时用字符串键
   - hashOrder 来源：`merged_swimlane.json` 中每个事件的 `args.hashOrder-hint` 字段，对应三种合图类型：
     * `l1ReuseInfo hashOrder` → `cube_l1_reuse_setting` 的 key
     * `cubeMergeInfo hashOrder` → `cube_nbuffer_setting` 的 key
     * `vecMergeInfo hashOrder` → `vec_nbuffer_setting` 的 key
3. Value (N) 是合并粒度，每 N 个同构子图合并为一个。设为 1 表示不合并。
4. 合并粒度应由 **subGraphCount**（hashOrder-hint 中的同构子图数量）和核心数决定，常用值为 1/2/4/8/16。
5. **⛔ outer-loops 必须手动计算**：先阅读 kernel 代码确定 loop 嵌套结构，手动计算 outer-loops，再传入 `--outer-loops` 参数。脚本默认 `outer-loops=1`（auto 值不可靠），必须使用正确的计算值。

**合图调优标准流程**：

```bash
# Step 0: 核使用率分析（⛔ 强制前置，详见第 3 节）
# 对每个 AIC/AIV leafHash，统计其分布在多少个 core 上
# 判定规则：
#   - cores < total_cores 且可通过 TileShape 增加 → 先用满核，再回来
#   - cores == total_cores 或 TileShape 已充分尝试仍无法增核 → 进入 Step 1

# Step 1: 根据 loop 次数计算外层循环次数 outer-loops（必填），再用 analyze_swimlane.py 分析泳道图数据
python3 scripts/analyze_swimlane.py \
    output/output_<最新目录> --outer-loops xxx

# Step 2: 从输出确定：
#   - hashOrder 列 → 即为合图的 key（set_pass_options 的 key）
#   - core 列 → AIC 对应 cube 配置，AIV 对应 vec 配置
#   - t/iter 列 → 单个 root function 中子图数量，合图粒度参考值

# Step 3: 根据分析结果设置配置

```

#### 1. 确定合图粒度

`subGraphCount` 是 `hashOrder-hint` 中提供的同构子图总数（跨所有外层循环迭代）。`t/iter = subGraphCount / outer_loops`，表示单个 root function 中该 hashOrder 的子图数量，直接指导合图粒度。

**获取方式**：从 `merged_swimlane.json` 中每个事件的 `args.hashOrder-hint` 字段解析，格式为：

```

l1ReuseInfo hashOrder: func15_1, subGraphCount: 90
cubeMergeInfo hashOrder: func15_1, subGraphCount: 90
vecMergeInfo hashOrder: func5_4, subGraphCount: 40

```

**自动分析**：运行 `analyze_swimlane.py` 后，Merge Tuning Guide 部分会自动输出每个 hashOrder 的 subGraphCount、t/iter 和建议粒度。

#### 确定 outer_loops（⛔ 必须手动计算，脚本默认 outer-loops=1）

`outer_loops` 是外层循环的总迭代次数，`t/iter = subGraphCount / outer_loops`。

**⛔ 禁止依赖默认值 1**：脚本默认 outer-loops=1 仅为保证脚本不中断运行。默认值会导致 t/iter 错误（偏大），合图粒度建议也全部错误。
**必须人工阅读 kernel 代码计算 outer_loops 后传入 `--outer-loops`。**

**手动计算方法**：分析实现代码的 loop 嵌套和 tile 切块：

```python
# 示例：flash_attention_score_grad
# B=2, N=8, S=256, S_TILE=128, s_loop=S//S_TILE=2
# loop 嵌套: b(2) × n(8) × s1(s_loop=2) × s2(s_loop=2)
# 外层循环（s2 之外）: b × n × s1 = 2 × 8 × 2 = 32
for b_idx in pypto.loop(b, ...):
    for n_idx in pypto.loop(N, ...):
        for s1_idx in pypto.loop(s_loop, ...):
            for s2_idx in pypto.loop(s_loop, ...):   # 最内层
                ...

```

分析方法：
1. 找到 kernel 函数中所有 `pypto.loop()` 调用，确定嵌套层级
2. 最内层循环（通常是带 `unroll_list` 的那个）不参与 outer_loops 计算
3. `outer_loops = 各外层循环次数的乘积`

可用 `--outer-loops` 参数手动指定精确值：

```bash
python3 scripts/analyze_swimlane.py output/output_xxx --outer-loops 32

```

#### t/iter 对合图粒度的指导

| t/iter | 含义 | 核状态 | 合图粒度建议 |
|--------|------|--------|-------------|
| 1 | 单个 root function 中只有 1 个子图 | — | 粒度 1（不合并），或跨 root function 尝试 2/4 |
| 2 | 单个 root function 中有 2 个子图 | 未满 | cube 类粒度 1（不合并）；vec_nbuffer 可尝试 `{-2: 1, -1: 2}` 或 `{"DEFAULT": 1, "func5_4": 4}` |
| 2 | 同上 | 已满 | 优先试 2，再试 4 |
| 4+ | 单个 root function 中有多个子图 | 未满 | 优先试 vec_nbuffer，cube 类通常退化 |
| 4+ | 同上 | 已满 | 可试 2/4/8，逐步增大 |

**⚠️ 核未满时的合图策略**：
- `vec_nbuffer_setting`：可尝试，用整数键 `{-2: 1, -1: N}` 或字符串键 `{"DEFAULT": 1, "func5_4": N}`，从 N=4 开始逐步试 8/16
- `cube_l1_reuse_setting` / `cube_nbuffer_setting`：通常退化，不建议设置；核未满时合图会进一步减少并行度

**⚠️ 粒度过大的风险**：
- avg<10us 的短耗时子图，合图粒度不宜超过 8（实测 N=16 时退化）
- 合图粒度可以大于 t/iter（跨 root function 合并），但不宜过大以免 L1/UB 内存溢出
- 每次调整后必须实测验证端到端耗时，禁止凭推测判定

#### 1.1 输出解读示例

```
#   leafHash                 min(us)   max(us)   avg(us)  total(us) core hashOrder    subGCnt t/iter root_name                                  compute_ops
1   17445...                   13.74     42.30     28.83    2537.46  AIC func15_1          90   11.2 ...        L1_TO_L0A+...
2   14789...                   42.14     46.12     44.67     938.10  AIC func15_2          22    2.8 ...        L1_TO_L0A+...
3   10766...                   28.76     31.54     29.91     598.16  AIC func5_0           20    2.5 ...        L1_TO_L0A+...
4   10531...                   25.16     28.64     26.91     565.18  AIC func15_0          22    2.8 ...        L1_TO_L0A+...
5   85886...                    1.12      3.72      1.69     108.20  AIV func11_0           1    0.1 ...        MULS+BAR.V+...
6   16650...                    3.80      4.26      4.09      12.28  AIV func5_4           40    5.0 ...        (pure copy)

outer_loops=8 (user-specified)

================================================================================
Merge Tuning Guide (hashOrder = merge key)
================================================================================

[AIC] cube_l1_reuse_setting:
  hashOrder=func15_1: subGraphCount=90, t/iter=11, avg=28.49us
    -> integer key: {-1: 2/4/8/16} (global)
    -> func key:    {"DEFAULT": 2/4, "func15_1": 2/4/8/16} (specific)
  hashOrder=func15_2: subGraphCount=22, t/iter=3, avg=44.08us
    -> integer key: {-1: 2/4/8} (global)
    -> func key:    {"DEFAULT": 2/4, "func15_2": 2/4/8} (specific)

[AIC] cube_nbuffer_setting:
  hashOrder=func15_1: subGraphCount=90, t/iter=11, avg=28.49us
    -> integer key: {-1: 2/4/8/16} (global)
    -> func key:    {"DEFAULT": 2/4, "func15_1": 2/4/8/16} (specific)
  hashOrder=func5_0: subGraphCount=20, t/iter=2, avg=29.91us
    -> integer key: {-1: 2/4/8} (global)
    -> func key:    {"DEFAULT": 2/4, "func5_0": 2/4/8} (specific)

[AIV] vec_nbuffer_setting:
  hashOrder=func5_4: subGraphCount=40, t/iter=5, avg=3.78us
    -> integer key: {-2: 1, -1: 2/4/8/16} (global)
    -> func key:    {"DEFAULT": 1, "func5_4": 2/4/8/16} (specific)

**解读**：
- hashOrder=func15_1 的 AIC 子图 subGCnt=90、t/iter=11（单个 root function 有 11 个子图），所有子图均合图时用整数键 `{-1: 4}`，仅合此 hashOrder 时用 `{"DEFAULT": 1, "func15_1": 4}`
- hashOrder=func5_0 的 AIC 子图 subGCnt=20、t/iter=2、avg=29.91us，所有子图均合图时用 `{-1: 2}`，仅合此 hashOrder 时用 `{"DEFAULT": 1, "func5_0": 2}`
- hashOrder=func5_4 的 AIV 子图 subGCnt=40、t/iter=5，所有子图均合图时用 `{-2: 1, -1: 4}`，仅合此 hashOrder 时用 `{"DEFAULT": 1, "func5_4": 4}`

#### 2. Vector 合图

**⛔ 重要原则**：`vec_nbuffer_setting` 中**必须**添加 `-2: 1` 配置，以规避部分合图不生效的问题。无论后续如何调优粒度，此配置不可省略。

##### 2.1 自动合图方案（vec_nbuffer_setting）

```python
# 整数键格式（全局通配所有子图）：
@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-2: 1, -1: 8}
    }
)

# 字符串键格式（精细控制特定 hashOrder）：
@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {"DEFAULT": 1, "func5_4": 8}
    }
)

```

**适用场景**：自动切图的vector task之间有直接依赖关系，且每一个task耗时很短（<10us）

**参数说明**
- 整数键格式：`-1:N` 代表所有 vector 子图按 N 的粒度合图；`-2:1` 必须添加作为 merge enable 标志
- 字符串键格式：`"DEFAULT":1` 是必需的 merge enable 标志；`"func5_4":N` 仅对 hashOrder=func5_4 的子图生效
- **禁止混用**：同一个 dict 内不能同时包含整数键和字符串键

**调优方法**：
1. 运行 [analyze_swimlane.py](../scripts/analyze_swimlane.py)，查看 `[AIV] vec_nbuffer_setting` 部分的输出
2. 根据 `hashOrder` 确定合图 key，根据 `t/iter` 确定粒度参考值
3. t/iter=1 的组先设为 1，t/iter≥2 的组设为对应值或更小
4. 需所有子图均合图时用整数键 `{-2: 1, -1: N}`，需精细控制特定 hashOrder 时用字符串键

**参考资料**
- [vec_nbuffer_setting 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_pass_options.md)

##### 2.2 手动合图方案（sg_set_scope）

通过 `sg_set_scope` 将有数据依赖的连续 Vector 操作强制合并到同一子图，减少子图间调度开销和数据搬运。

```python
pypto.set_pass_options(sg_set_scope=1)
# ... 连续的 Vector 操作（有直接数据依赖、同循环层级、无 Cube 夹杂）...
pypto.set_pass_options(sg_set_scope=-1)

```

**约束**：
- 仅对有直接上下游数据依赖的 Vector 操作生效
- 不要包裹 Cube（matmul）操作，也不要包裹其后继含 Cube 的节点
- 跨 `pypto.loop` 边界不能合并
- 每个 scope 使用不同的正整数 ID

###### 2.2.1 依赖链分析工具

使用 `analyze_aiv_dep_chains.py` 从 `dyn_topo.txt` 中提取 AIV 任务之间的依赖链路，自动识别 cube 边界并给出 sg_set_scope 合并建议。

**用法**：

```bash
python3 scripts/analyze_aiv_dep_chains.py <output_dir>
python3 scripts/analyze_aiv_dep_chains.py <output_dir> --json result.json

```

**输入文件**（`output_dir` 中）：
- `dyn_topo.txt` — 任务动态拓扑（含 successors 依赖，必需）
- `program.json` — 程序编译数据（可选，用于标注操作类型）

**输出分两部分**：

**Part 1: 原始依赖链**（完整链路，不截断）

```

链路A（16次）
3907163356593077760
  │
  ▼
2360323566658746396
  │
  ▼
2768731787098226973
  3907163356593077760: op=10001, psg=1, [vec] CAST+CAST
  2360323566658746396: op=10002, psg=2, [vec] MUL+CAST+CAST+ROWSUM_SINGLE
  2768731787098226973: op=10001, psg=1, [vec] MULS+SUB+EXP+DIV+SUB+MUL+CAST+CAST

```

**Part 2: sg_set_scope 优化建议**（在 cube 边界截断）

脚本自动检测每个 AIV 节点的后继是否包含 cube（matmul）任务。截断规则：
- 遇到后继含 cube 的 AIV 节点时，**保留该节点但不继续展开后继**
- 截断后 ≥2 节点且 psgId 有变化的链段，建议用 `sg_set_scope` 合并

```

sg_set_scope 优化建议

  建议 1: 截断后 3 个节点, 16 次, psgId 变化: 1 → 2 → 1
    3907163356593077760
      │
      ▼
    2360323566658746396
      │
      ▼
    2768731787098226973
    3907163356593077760: psg=1, [vec] CAST+CAST
    2360323566658746396: psg=2, [vec] MUL+CAST+CAST+ROWSUM_SINGLE
    2768731787098226973: psg=1, [vec] MULS+SUB+EXP+DIV+SUB+MUL+CAST+CAST [✂ cube边界]
    → 建议: 用 sg_set_scope 包裹 psgId 1 → 2 → 1 的 vector 操作
    ✂ 截断点 (后继含 cube): ['2768731787098226973']

```

**输出字段说明**：
- **leafHash**：叶子函数哈希，通过 `program.json` 的 `hash` 字段可映射到具体函数
- **opmagic**：操作类型标识
- **psgId**：当前所属子图 ID
- **[vec]/[cube]**：操作核心类型（基于 opcode 自动判断，含 `A_MUL_B/A_MULACC_B` 为 cube，否则为 vec）
- **opcode 序列**：过滤掉框架指令后的实际计算指令
- **✂ cube边界**：该 AIV 节点的后继包含 cube 任务

**⚠️ 重要：脚本建议是候选，必须经过 4.1.2.2 映射验证后才能实施。**

###### 2.2.2 从建议到实施的验证流程

脚本的优化建议是基于 `dyn_topo.txt` 的自动分析，不能直接用于修改前端代码。必须通过 `program.json` 的 `file`/`line` 字段将 leafHash 映射到前端代码，验证可合并性，并确认代码连续性。

详细的映射方法和自动映射工具参见 [leafHash → 前端代码映射方法](leafhash-to-code-mapping.md)。

**自动映射工具**：

```bash
# 查看指定 leafHash 的代码位置
python3 scripts/leafhash_to_code.py <output_dir> --leafhash <hash>

# 查看所有 leafHash
python3 scripts/leafhash_to_code.py <output_dir>

```

**验证检查清单**（对建议中的每个链段逐项检查）：

| 检查项 | 验证方法 | 通过标准 |
|:---|:---|:---|
| 数据依赖 | dyn_topo 中存在 VEC→VEC successors 边 | 有直接数据依赖 |
| 同循环层级 | dyn_topo 的 rootIndex 比对 | 所有节点 rootIndex 相同 |
| 纯 vector 操作 | program.json ops 中无 A_MUL_B/A_MULACC_B | 无 cube 指令 |
| 无 cube 后置依赖 | dyn_topo successors 中无 coreType=1 | 后继不含 matmul |
| 代码行连续性 | file/line 映射，确认中间无夹杂 | scope 内只有被合并的操作 |
| ✂ cube边界节点排除 | 脚本标记的截断点 | 有 cube 后继的节点不参与合并 |

**只有全部通过的链段才是可合并的。**

**代码连续性检查与调整**：

通过 `leafhash_to_code.py` 确认每个 leafHash 对应的前端代码行后，检查待合并的代码行之间是否夹带不相关操作。如果两个 leaf 对应的代码行之间有其他操作（如无关变量的 view），直接包裹 sg_set_scope 会把这些操作也卷入合并。

此时需要调整前端代码顺序，将不相关的操作移到 sg_set_scope 包裹范围之外，使待合并的操作紧密相邻。PyPTO 是声明式构图，只要数据依赖关系不变，代码顺序可以调整。

**调整原则**：
- 只移动与合并段无数据依赖的操作
- 移动后的代码不能跨越 `pypto.loop` 边界
- scope 必须覆盖所有参与合并的 leaf 的全部代码行，不能只包裹部分操作
- 调整后必须重新运行精度验证

**完整工作流程**：
1. 运行测试用例采集泳道数据（需 `debug_options={"runtime_debug_mode": 1}`）
2. 运行 `analyze_aiv_dep_chains.py` 分析依赖链，获取 Part 2 优化建议
3. 运行 `leafhash_to_code.py` 将 leafHash 映射到前端代码行
4. 用验证检查清单过滤，排除不可合并的段
5. 检查代码连续性：合并段对应的代码行之间是否有不相关操作
6. 如有夹杂，调整代码顺序使合并段紧密相邻
7. 在连续的代码段位置插入 `sg_set_scope`
8. 验证精度和性能

**参考资料**
- [leafHash → 前端代码映射方法](leafhash-to-code-mapping.md)
- [sg_set_scope 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_pass_options.md)


#### 3. Cube 合图

##### 3.1 L1Reuse 策略（默认开启，用于合并具有 L1 重复搬运的子图）

**适用场景**：matmul 的 M 或 N 轴进行了切分，存在重复搬运

```python
# 整数键格式（全局通配所有子图）：
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {-1: 2}}
)

# 字符串键格式（精细控制特定 hashOrder）：
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {"DEFAULT": 2, "func15_1": 8}}
)

```

**调优方法**：
1. 运行 [analyze_swimlane.py](../scripts/analyze_swimlane.py)，查看 `[AIC] cube_l1_reuse_setting` 部分的输出
2. 根据 `hashOrder` 确定合图 key，优先对 total 耗时大且有重复搬运的子图调优
3. `t/iter` 越大（单个 root function 中子图越多），L1 复用收益越高，可设更大粒度
4. ⚠️ **核未满时通常退化**：如果 `analyze_core_usage.py` 显示核未满，cube_l1_reuse 通常导致性能退化，不建议设置

**参考资料**
- [cube_l1_reuse_setting 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_pass_options.md)

##### 3.2 CubeNBuffer 策略（用于合并同构的子图）

**适用场景**：
- 同构子图数量很多，且每一个task的执行耗时很短（10us以下）
- K 轴很长且没有切 K

```python
# 整数键格式（全局通配所有子图）：
@pypto.frontend.jit(
    pass_options={"cube_nbuffer_setting": {-1: 2}}
)

# 字符串键格式（精细控制特定 hashOrder）：
@pypto.frontend.jit(
    pass_options={"cube_nbuffer_setting": {"DEFAULT": 2, "func15_1": 4}}
)

```

**调优方法**：
1. 运行 [analyze_swimlane.py](../scripts/analyze_swimlane.py)，查看 `[AIC] cube_nbuffer_setting` 部分的输出
2. 根据 `hashOrder` 确定合图 key，根据 `t/iter` 和 avg 耗时确定粒度
3. avg<10us 且 t/iter≥2 的组优先设置，如 `cube_nbuffer_setting: {-1: 2}` 或 `cube_nbuffer_setting: {"DEFAULT": 1, "func5_0": 2}`
4. ⚠️ **核未满时通常退化**：如果 `analyze_core_usage.py` 显示核未满，cube_nbuffer 通常导致性能退化，不建议设置

**参考资料**
- [cube_nbuffer_setting 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_pass_options.md)

##### 3.3 L1Reuse 与 CubeNBuffer 的协同使用

**⚠️ 重要：两者作用维度不同，需协同配置，不宜同时过大。**

| 参数 | 合并维度 | 核心目的 |
|------|---------|---------|
| cube_l1_reuse_setting | 消除 GM 数据重复搬运 | 多个子图复用同一份 L1 数据，减少搬运开销 |
| cube_nbuffer_setting | 合并结构相同的 AIC 子图 | 减少调度开销，提升核心利用率 |

**协同原则**：
1. **两者不宜同时设置过大**：cube_l1_reuse 通过消除重复搬运带来收益，合并力度越大 L1 复用越好；cube_nbuffer 通过合并同构子图减少调度开销。但两者同时过大会导致单个子图过大，占用过多 L1/UB 内存，反而引发性能退化。
2. **优先调 cube_l1_reuse_setting**：先确定 L1 数据复用的合并力度（消除重复搬运是更直接的收益），再调整 cube_nbuffer_setting。
3. **观察 Task Count 变化**：合图后 Total Task Count 应适度下降。如果 Task Count 不降反升（例如 1664→6400），说明合图配置过度，应回退。
4. **用 analyze_swimlane.py 确定参数**：hashOrder 即合图 key，t/iter 指导粒度。

**反面案例**（flash_attention_score_grad 实测）：

```python
# baseline: 728us, Task=1664
"cube_l1_reuse_setting": {-1: 8},
"cube_nbuffer_setting": {-1: 8}
# → 711us ✅ (Task=1664, 利用率 58.8%→61.4%)

# 过度合图: 734us ❌ 性能退化
"cube_l1_reuse_setting": {-1: 8},
"cube_nbuffer_setting": {-1: 16}

```

##### 3.4 自动合图模式（空字典 `{}`）的风险

**⚠️ 风险提示：自动模式可能过度合图导致性能严重退化，不建议直接使用。**

当设置为空字典 `{}` 时，Pass 会根据硬件核心数自动计算合并粒度。但自动模式不了解算子的实际数据流特征，可能将不应合并的子图强行合并，导致：
- Task 数暴增（如 1664→6400）
- 子图过大，L1/UB 内存争用
- 核心利用率大幅下降（如 58%→48%）

**反面案例**（flash_attention_score_grad 实测）：

```python
# 自动模式: 1038us ❌ 性能退化 42%
"cube_l1_reuse_setting": {},
"cube_nbuffer_setting": {}
# Task Count: 1664 → 6400, 利用率: 58.8% → 48.7%

```

**建议**：始终使用 [analyze_swimlane.py](../scripts/analyze_swimlane.py) 分析泳道图获取 hashOrder 和 t/iter 后手动精确配置，避免使用空字典 `{}` 自动模式。
