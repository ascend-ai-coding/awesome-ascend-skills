### 合图调优

合图是指将计算图中多个逻辑上独立的 Task 合并为一个逻辑子图。

**⛔ ⛔ ⛔ ⛔ 合图前强制三步分析（禁止跳过任何一步）：⛔ ⛔ ⛔ ⛔**

1. **第 1 步：核使用率分析**（analyze_core_usage.py）— 判定每个 leafHash FULL/NOT FULL。核未满必须优先通过 TileShape 填核，禁止跳过直接合图。
2. **第 2 步：泳道图分析**（analyze_swimlane.py --outer-loops N）— 获取每个 hashOrder 的 subGraphCount、t/iter，输出 Merge Tuning Guide。**必须基于当前最新性能数据的泳道图运行此脚本**，禁止使用旧数据或直接猜测 hashOrder。
3. **第 3 步：根据 Merge Tuning Guide 的 hashOrder 和粒度建议，设置特定参数**。禁止不用分析结果盲试全局参数（如 `{-1: N}`）。

**⚠️⚠️⚠️ 关键原则：**

1. **⛔ 合图前必须先完成核使用率分析（第 3 节）**：核未满时优先用满核，核满后再合图。盲目合图会导致核未满时性能退化。
2. pass_options key 有两种格式（**同一 dict 内禁止混用**）：
   - **整数键格式**：`{-1: N}`（全局通配所有子图）
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

| t/iter | 含义                               | 核状态 | 合图粒度建议                                                                                      |
| ------ | ---------------------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| 1      | 单个 root function 中只有 1 个子图 | —     | 粒度 1（不合并），或跨 root function 尝试 2/4                                                     |
| 2      | 单个 root function 中有 2 个子图   | 未满   | cube 类粒度 1（不合并）；vec_nbuffer 可尝试`{-1: 2}` 或 `{"DEFAULT": 1, "func5_4": 4}` |
| 2      | 同上                               | 已满   | 优先试 2，再试 4                                                                                  |
| 4+     | 单个 root function 中有多个子图    | 未满   | 优先试 vec_nbuffer，cube 类通常退化                                                               |
| 4+     | 同上                               | 已满   | 可试 2/4/8，逐步增大                                                                              |

**⚠️ 核未满时的合图策略**：

- `vec_nbuffer_setting`：可尝试，用整数键 `{-1: N}` 或字符串键 `{"DEFAULT": 1, "func5_4": N}`，从 N=4 开始逐步试 8/16
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
    -> integer key: {-1: 2/4/8/16} (global)
    -> func key:    {"DEFAULT": 1, "func5_4": 2/4/8/16} (specific)

**解读**：
- hashOrder=func15_1 的 AIC 子图 subGCnt=90、t/iter=11（单个 root function 有 11 个子图），所有子图均合图时用整数键 `{-1: 4}`，仅合此 hashOrder 时用 `{"DEFAULT": 1, "func15_1": 4}`
- hashOrder=func5_0 的 AIC 子图 subGCnt=20、t/iter=2、avg=29.91us，所有子图均合图时用 `{-1: 2}`，仅合此 hashOrder 时用 `{"DEFAULT": 1, "func5_0": 2}`
- hashOrder=func5_4 的 AIV 子图 subGCnt=40、t/iter=5，所有子图均合图时用 `{-1: 4}`，仅合此 hashOrder 时用 `{"DEFAULT": 1, "func5_4": 4}`

#### 2. Vector 合图

##### 2.1 自动合图方案（vec_nbuffer_setting）

```python
# 整数键格式（全局通配所有子图）：
@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 8}
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

- 整数键格式：`-1:N` 代表所有 vector 子图按 N 的粒度合图
- 字符串键格式：`"DEFAULT":1` 是必需的 merge enable 标志；`"func5_4":N` 仅对 hashOrder=func5_4 的子图生效
- **禁止混用**：同一个 dict 内不能同时包含整数键和字符串键

**调优方法**：

1. 运行 [analyze_swimlane.py](../scripts/analyze_swimlane.py)，查看 `[AIV] vec_nbuffer_setting` 部分的输出
2. 根据 `hashOrder` 确定合图 key，根据 `t/iter` 确定粒度参考值
3. t/iter=1 的组先设为 1，t/iter≥2 的组设为对应值或更小
4. 需所有子图均合图时用整数键 `{-1: N}`，需精细控制特定 hashOrder 时用字符串键

**参考资料**

- [vec_nbuffer_setting 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_pass_options.md)

##### 2.2 手动合图方案（sg_set_scope）

通过 `sg_set_scope` 将有数据依赖的连续 Vector 操作强制合并到同一子图，减少子图间调度开销和数据搬运。**普通合图只包裹 Vector 段（不含 Cube）**；包裹 Cube+Vector 段的合图称为 Mix合图（见 §4，仅 A5 平台支持）。

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

| 检查项              | 验证方法                                 | 通过标准                     |
| :------------------ | :--------------------------------------- | :--------------------------- |
| 数据依赖            | dyn_topo 中存在 VEC→VEC successors 边   | 有直接数据依赖               |
| 同循环层级          | dyn_topo 的 rootIndex 比对               | 所有节点 rootIndex 相同      |
| 纯 vector 操作      | program.json ops 中无 A_MUL_B/A_MULACC_B | 无 cube 指令                 |
| 无 cube 后置依赖    | dyn_topo successors 中无 coreType=1      | 后继不含 matmul              |
| 代码行连续性        | file/line 映射，确认中间无夹杂           | scope 内只有被合并的操作     |
| ✂ cube边界节点排除 | 脚本标记的截断点                         | 有 cube 后继的节点不参与合并 |

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

| 参数                  | 合并维度                | 核心目的                                 |
| --------------------- | ----------------------- | ---------------------------------------- |
| cube_l1_reuse_setting | 消除 GM 数据重复搬运    | 多个子图复用同一份 L1 数据，减少搬运开销 |
| cube_nbuffer_setting  | 合并结构相同的 AIC 子图 | 减少调度开销，提升核心利用率             |

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

#### 4. A5 Mix合图（CV 融合，消除 CV 间搬运）

> ⚠️ **平台限制**：仅 A5 平台（`DAV_3510`）支持 mixed-CV 合图。mixed-CV 的 L0C→UB、UB→L1 直连识别只在 DAV_3510 上启用。非 A5 平台使用 `sg_set_scope` 同时包裹 AIC 和 AIV 算子会直接编译失败。DAV_3003/DAV_3113 等 Lite 平台有独立的 LiteNPU 图形模式推断路径，不依赖 mixed-CV scope 机制，支持范围和识别图形不同，不可等同于 DAV_3510 mixed-CV 规则。所有 Mix合图代码必须用 `pypto.platform.npuarch == 'DAV_3510'` 条件包裹。

**原理**：传统合图中 Cube（AIC）与 Vector（AIV）之间通过 GM 搬运传递数据。Mix合图让 CV 之间走 **CV 通路**（片上直连）替代搬运，消除 CV 间数据搬运开销。CV 配比 **1:N**（1 个 Cube 配 N 个 Vector），其中 N 为整数，**N > 1 才能用满核**（N=1 时 V 核利用率不足），**N 为偶数性能更优**，**N=2 是最优配比**（A5 硬件物理核为 32C + 64V = 1:2，由芯片物理连接决定）。Mix合图由编译器自动按 1:2 配比对 CV 进行划分融合，具体划分方式因算子的 CV 交替结构不同而变化。**Mix合图功能只在 A5 平台上才有**；只包裹 Vector 段的合图称为普通合图（见 §2.2），在所有平台均可用。

##### 4.1 使用方式

**⛔⛔⛔ 核心认知：自动合图和手动合图是同一个 Mix合图功能的两种开关方式，功能完全一致，只是 scope 范围的确定方式不同。**

| | 自动合图（方式 A） | 手动合图（方式 B） |
|---|---|---|
| 开关 | `pypto.experimental.auto_mix_partition(1)` | `sg_set_scope(正整数)` / `sg_set_scope(-1)` |
| scope 范围 | 编译器自动决定 CV 段的包裹范围 | 用户手动圈定 CV 段 |
| 功能 | 完全一致：消除 CV 间 DDR 搬运，走 CV 通路 | 完全一致 |
| 配套参数 | 完全一致：nbuffer、TileShape、unroll 等 | 完全一致 |
| 编译超时处理 | 完全一致（见关键路径 Step 2） | 完全一致 |
| 退化处理 | 完全一致（见关键路径 Step 3→4） | 完全一致 |

**⛔ 调优互斥原则**：性能调优时不能同时使用两种方式——要么使用自动合图，要么使用手动合图。切换时必须移除旧配置（重置 `pypto.experimental.auto_mix_partition(0)` 或移除所有 `sg_set_scope`），同时使用会导致调优变量混淆。

**方式 A：自动合图（优先尝试）**

在 jit kernel 函数体内调用 `pypto.experimental.auto_mix_partition(1)`，编译器底层自动对 CV 算子做合图包裹，无需手动设置 `sg_set_scope`：

```python
@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 1},
        "cube_nbuffer_setting": {-1: 1},
    },
)
def kernel(...):
    pypto.experimental.auto_mix_partition(1)
    # ... kernel 计算代码（无需手动 sg_set_scope 包裹）...
```

**方式 B：手动合图**

用 `sg_set_scope` 将待融合的 kernel 代码段包裹起来（开始 正整数，结束 -1），必须带 A5 平台判断：

```python
# 开始 Mix合图包裹
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=5001)

# ... kernel 计算代码（Cube + Vector 交替段）...

# 结束 Mix合图包裹
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_scope=-1)
```

**参考资料**

- [sg_set_scope 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_pass_options.md)

---

#### ⛔ Mix合图调优关键路径（从头到尾按顺序走，不可跳步）

> **铁律：Mix合图一旦开启，不论编译超时还是性能退化，都禁止直接移除 Mix合图配置转向非 Mix 优化（如 stitch/sched_mode/NONE_CACHEABLE/vf_options）。必须走完下方全部 Step 后才允许退出。**

> **⛔ 两条独立调试线原则：自动合图（`pypto.experimental.auto_mix_partition(1)`）和手动合图（`sg_set_scope`）是两条独立的调试线，各自必须完整走 Step 0→1→2→3→4→5→6。先完整走完自动合图线（Step 0→6），再切换走手动合图线（Step 0→6，Step 0 数据流分析可复用）。禁止从自动合图的某个中间步骤直接跳入手动合图的某个中间步骤。两条线全部走完仍未达到预期目标性能才允许退出 Mix合图。**

```
Step 0: 数据流分析（进入 Mix合图前的强制准备）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⛔ 禁止跳过此步骤直接配置 Mix合图。
  ⛔ 分析结果必须记录到调优日志，后续 Step 1-4 的决策必须引用本步骤的分析结论。

  0a. 识别算子内所有 CV 数据流

      逐行扫描 kernel 代码，填写 CV 数据流表：

      | # | 生产者 | 类型 | 输出 shape | 消费者 | 类型 | 中间有无其他Cube | shape单调 | 跨迭代依赖 | 数据走向 |
      |---|--------|------|-----------|--------|------|-----------------|----------|-----------|---------|
      | 1 | ? | Vec/Cube | ? | ? | Vec/Cube | ? | ? | ? | ? |

      对每条数据流，判断：
      - 生产者和消费者之间是否有其他 Cube 操作？（有则不能走 CV 通路）
      - 生产者的输出 shape 和消费者的输入 shape 是否单调？（交叉则不能走 CV 通路）
      - 消费者是否在 is_loop_begin/end 内操作跨迭代依赖 tensor？（是则该段有跨迭代依赖）

  0b. 确定 scope 范围（基于数据流分析）

      规则1: 如果 V 段的输出直接喂 C 段（V→C），且中间无其他 Cube，
             则 V 和 C 应在同一 scope 内，走 CV 通路（UB→L1）。
      规则2: 如果 C 段的输出直接喂 V 段（C→V），且中间无其他 Cube，
             则 C 和 V 应在同一 scope 内，走 CV 通路（L0C→UB）。
      规则3: 如果 V 段在 is_loop_begin/end 内操作跨迭代依赖 tensor
             （在 loop 外声明、loop 内读写的 tensor），该 V 段有跨迭代依赖，
             应从 Mix scope 中放出，**并立即用独立 sg_set_scope 做普通合图**（对应 S-21）。
             ⛔ 禁止只放出不做普通合图——放出的 V 段内部多个 Vector 子图间仍有调度开销。
              多 scope 布局（详见 S-21 两种策略）：
             ```
             sg_set_scope=20001  # Mix scope (大 ID)
             ... V0→C1→V1→C2 (CV 链) ...
             sg_set_scope=-1
             sg_set_scope=1      # 普通合图 scope (小 ID)
             ... V2 (online softmax update, 纯 Vector) ...
             sg_set_scope=-1
             ```
      规则4: gather_in_ub 的输出在 UB 中，如果该输出是后续 C 段的输入，
             则 gather 段应纳入 scope，否则 gather 结果走 DDR 到 C 段。

      根据规则1-4，画出 scope 布局方案：
      ```
      scope=XXXX (Mix合图): [列出纳入的段]
      scope=-1
      scope=YYYY (普通合图): [列出放出的段，如 V2 update]
      scope=-1
      ```

  0c. 确定配套参数（基于算子特征）

      分析算子特征，填写配套参数表：

      | 参数 | 分析依据 | 推荐值 |
      |------|---------|--------|
      | loop tile (s2_tile等) | 计算 bn_per_batch = cur_seq // loop_tile，目标让 bn_per_batch 在 1~8 范围（太少并行度不足，太多调度开销大） | ? |
      | unroll_list | 最内层 loop 预期迭代次数 | ? |
      | cube L1 | matmul 的 K 轴大小？K 大则 L1 调大 | ? |
      | vec tile 分段方案 | 逐行扫描每个 V 段，标记所有 tensor shape 变化点，每个变化点设置匹配的 vec tile（详见 Step 3c vec tile 调优方法） | ? |
      | gather/view 宽度 | 取实际有效数据宽度，不要用对齐填充宽度 | ? |
      | ooo_sched_mode | 有连续 CV 交替结构则试 GAPMIN/HLF（推荐先试 GAPMIN） | ? |
      | max_workspace_kb | 从 NPU 编译输出 "Recommended: set max_workspace_kb near XXX KB" 提取推荐值 | ? |
      | host_options | compile_monitor_enable: 0 减少编译监控开销 | {"compile_monitor_enable": 0} |
      | nbuffer | 从 1:1 开始 | {-1: 1} |

      ⛔ 所有参数必须基于算子自身特征推导

  0d. 输出分析结论

      必须输出以下内容，后续 Step 1 的原子优化点必须基于此结论：
      ```
      ### Mix合图数据流分析
      - CV 数据流：[列出所有 CV 数据流]
      - scope 布局方案：[scope=XXXX 包裹哪些段 / scope=YYYY 包裹哪些段]
      - 配套参数：
        - loop tile = [值]（依据：bn_per_batch = [计算值]）
        - unroll_list = [值]（依据：最内层 loop 预期迭代 [值] 次）
        - cube L1 = [值]（依据：K轴 = [值]）
        - vec tile 分段方案：[每个 V 段每个 shape 变化点的 vec tile 值]
        - gather/view 宽度 = [值]（依据：实际有效数据宽度）
        - ooo_sched_mode = [值或无]
        - max_workspace_kb = [值或无]
        - nbuffer = 1
      ```

Step 1: 开启 Mix合图 + 配套参数（原子优化点）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  操作：基于 Step 0 的分析结论，一次性设置全套配套配置：
    - Mix合图开关：按 Step 0b 的 scope 布局方案配置
      （自动合图 pypto.experimental.auto_mix_partition(1)，或手动合图 sg_set_scope）
    - 配套参数：按 Step 0c 的推荐值配置
      （cube tile / unroll_list / ooo_sched_mode / max_workspace_kb）
    - nbuffer：按 Step 0c 的推荐值配置（如无特殊推荐则默认 1:1）
    - ⚠️ 先不带 debug_options，确认编译通过且精度正确后再加 debug_options 采集性能
  实测性能。
    ↓
  ✅ 达标 → 完成，退出关键路径
  ❌ 编译超时 → 进入 Step 2
  ✅ 性能提升(E2E下降)但未达标 → 进入 Step 3c 调参（不需要走 Step 3a/3b 退化分析，但 3c 必须完整走完）
  ❌ 性能退化(E2E上升) → 进入 Step 3（必须走完 3a→3b→3c 全部子步骤）

Step 2: 编译超时处理（⛔ 禁止移除 Mix合图）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⛔ 编译超时的本质是编译器处理的子图规模过大。
  ⛔ 本步骤只处理编译超时：不切换开关方式（Step 5 职责）、不调整 scope 框架（Step 4 职责）。

  按以下优先级逐个尝试，每次修改后重新编译实测：

  2a. 移除 debug_options + 添加 host_options（零成本，最常见原因）
      - debug_options 会插入性能采集指令，显著增加编译时间
      - 先不带 debug_options 验证编译是否通过
      - 添加 host_options={"compile_monitor_enable": 0} 关闭编译监控
      → 重试。仍超时 → 继续 2b

  2b. 回退 unroll（编译超时最直接原因：展开过多独立代码路径）
      - unroll_list 降档（如 [8,4,2,1]→[4,2,1]→[2,1]→[1]）
      - 记录被回退的 unroll 值，后续 Step 3c 不再尝试该值（避免死循环）
      → 重试。仍超时 → 继续 2c

  2c. 回退 nbuffer（⛔ 必须检查，nbuffer 调大会导致子图膨胀，编译变慢）
      - 检查当前 nbuffer 值是否 > 1（包括 Step 3c 调参过程中调大的）
      - 如果 > 1 → 回退到 {-1:1} 或 {"DEFAULT": 1}
      - nbuffer 调大允许编译器跨迭代重叠计算，但同时增加子图规模，编译时间显著增加
      → 重试。仍超时 → 继续 2d

  2d. 调小核内 TileShape / loop tile / vec tile
      - 调小 cube L0（如 [128,128]→[64,64]）
      - 调小 loop tile（如 2048→1024→512）
      - 调小 vec tile（减少编译器处理的 vector 操作粒度）
      → 重试。仍超时 → 继续调小，直到编译通过或所有手段均无效

  编译通过后 → 进入 Step 3c 调参
  ⚠️ Step 3c 调参时如果再次调大 nbuffer 导致编译超时，回到 Step 2c 回退 nbuffer 后重试
  ⚠️ 如果 unroll 在 Step 2b 被回退，Step 3c 应优先尝试调优 vec tile / cube tile 等其他参数弥补并行度损失，不再尝试已被回退的 unroll 值
  所有手段均无效 → 记录原因 → 跳转 Step 5（切换开关方式，重新开始调合图）

Step 3: 性能退化处理 — 阶段 A（在当前 scope 范围内充分调参）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⛔⛔⛔ 铁律：退化后禁止直接回退！必须先分析退化原因，再在当前 scope 范围内调参。
  ⛔⛔⛔ 禁止跳过分析直接回退到非 Mix 优化（如 stitch/sched_mode/NONE_CACHEABLE）。
  ⛔ scope 范围调整是框架级操作，属于 Step 4，不在本步骤内。

  3a. 分析退化原因（诊断性操作，不改变参数）

      必须回答以下问题，记录到调优日志：

      Q1: CV 通路是否生效？
          - 解析 program.json，追踪 CV 间数据流向
          - 走 CV 通路（L0C_COPY_UB / UB_COPY_L1）→ ✅
          - 走 DDR 中转（COPY_OUT→COPY_IN 无 CV_SYNC）→ ❌
          - 如果部分走 CV 部分走 DDR → 记录哪些数据走了 DDR

      Q2: spill 数量是否过多？
          - 泳道图中 WorkspaceGm 数量 > 20 → spill 过多
          - ≤ 20 → 正常

      Q3: 退化幅度与哪个因素最相关？
          - 如果 CV 通路大部分走 DDR → §4.2 硬限制问题，需要在当前 scope 内修复
          - 如果 spill 过多 → TileShape/nbuffer 过大，需要在当前 scope 内调小
          - 如果 CV 通路正常且无 spill → 可能是 Mix 串行化减少并行度，
            需要在当前 scope 内调小 loop tile/L0 增加 task 数
          - 如果以上均正常 → 当前 scope 框架可能不合理，记录后进入 Step 4

  3b. 根据原因在当前 scope 范围内调参

      原因1: CV 通路未生效（DDR 中转）
        → 排查 §4.2 硬限制：UB 248KB / shape 单调 / 衔接形态 / shape-tile 对齐
        → 在当前 scope 内修复（调小 tile_shape / 调整 tile 值）后重新实测

      原因2: spill 过多
        → 调小 TileShape 或调小 nbuffer（减少 UB 并发占用）

      原因3: Mix 串行化减少并行度
        → 调小 loop tile 增加 task 数
        → 调小 L0 增加核内并行度

      原因4: 以上均正常
        → 先完成 3c 充分调参，仍退化则进入 Step 4 调整 scope 框架

  3c. 配套参数逐个调优（⛔ 必须全部尝试，每项标记 ✅已尝试/❌已失败，禁止跳过）

      ┌─────────────────────────────────────────────────────────────┐
      │ ⛔ 3c 调参清单（逐个尝试，每次只改一个参数，实测后标记）     │
      │                                                             │
       │ □ nbuffer: vec_nbuffer 1→2→4→8→16→32… 逐值实测              │
       │   （1/2/4/8是常用范围，8有收益继续增大，劣化则回退不再增大）   │
      │ □ loop tile: 尝试至少 2 个值（如 s2_tile 1024→512→2048）   │
      │ □ cube L0: [128,128]→[64,64]→[256,128] 逐个尝试            │
      │ □ cube L1: [128,128]→[128,256]→[256,256] 逐个尝试          │
      │ □ vec tile: 按 V 段 shape 变化点分段设置（详见下方方法）     │
      │ □ unroll_list: [8,4,2,1]→[4,2,1]→[2,1] 至少 1 次降档       │
      │ □ ooo_sched_mode: GAPMIN → HLF 逐个尝试                    │
      │ □ max_workspace_kb: 如果 spill 多则调大                     │
      │                                                             │
      │ ⛔ 以上 8 项全部标记为 ✅或❌后才允许退出 3c               │
      │ ⛔ 禁止只试了其中几项就判定"3c 已充分调优"                  │
      └─────────────────────────────────────────────────────────────┘

      **vec tile 调优方法**：

      核心原则：**每次 tensor shape 发生变化时，必须重新设置 vec tile 匹配当前 shape**。
      一个 V 段内可能有多次 shape 变化（reshape/view/cast/reduce），每次变化都需要独立的 vec tile 设置。

      调优步骤：
      1. 逐行扫描每个 V 段，标记所有 tensor shape 变化点
      2. 在每个 shape 变化点前设置 vec tile，匹配该段操作的 tensor 实际 shape
      3. vec tile 两维的选择：
         - 行方向（第一维）：匹配 tensor 的行数，reduce 操作取小值
         - 列方向（第二维）：匹配 tensor 的列数，尽量取满（不超过实际列数）
      4. 特别注意同一 V 段内标量运算（如 [M,1]）和矩阵运算（如 [M,N]）交替的场景：
         - 标量运算段：行方向取大值（如 128），列方向取小值（如 128）
         - 矩阵运算段：行方向取小值（如 32），列方向取大值匹配数据宽度（如 512）
         - 必须在标量运算和矩阵运算之间切换 vec tile

      **vec tile 联动调优**：
      - vec tile 调大后可能导致 UB 248KB 超限（运行时错误或 CV 通路降级到 DDR）
      - 如果某个 V 段的 vec tile 调大后运行时错误，不要直接放弃，应尝试同时调小其他 tile（如 cube L0 或 loop tile）释放 UB 空间
      - UB 空间是所有 V 段共享的，多个 V 段的 vec tile 之和不超 248KB

      **gather/view 宽度优化**：
      - gather 和 view 的宽度应取实际有效数据宽度，不要用对齐填充宽度
      - 对齐填充会多搬运无效数据，增加内存带宽开销
      - 例如 packed_dim=656 时用 656 而非 672（32B 对齐填充），每行少搬运 16 字节

      **dequant reshape 对齐**：
      - dequant 操作中 reshape 应利用 block_size 对齐，使 scale broadcast 高效
      - 例如将 [s2, dn] reshape 为 [s2 * (dn/block_size), block_size]，scale 广播正好对齐

      **loop 嵌套结构**：
      - 如果算子有 n_kv / group 等静态维度（值为 1 时只迭代 1 次），保留这些循环层不要省略
      - 多出的循环层虽然只迭代 1 次，但会影响编译器的 root function 划分和 task 调度

  ⛔⛔⛔ 3c 退出条件（修复版——禁止从 3c 直接跳 Step 5）：

  3c 全部 8 项参数调完后，按以下路径退出：
  ├─ 达标 → ✅ 完成，退出关键路径
  ├─ 仍退化(E2E 仍高于非合图基线) → 进入 Step 4（调整 scope 框架）
  └─ 有提升但未达标(E2E 下降但未到目标) → 进入 Step 4（调整 scope 框架）

  ⛔ 禁止从 3c 直接跳到 Step 5！
  ⛔ 无论"仍退化"还是"有提升但未达标"，3c 调完后都必须先走 Step 4（调整 scope 框架）。
  ⛔ 只有 Step 4 也走完仍无收益，才允许进入 Step 5（切换开关方式）。

Step 4: 性能退化/未达标处理 — 阶段 B（调整 scope 框架）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⛔ 3c 全部参数调完后，无论"仍退化"还是"有提升但未达标"，都进入此步骤。
  ⛔ scope 范围是 Mix合图的框架，调整 scope 意味着重新定义哪些 CV 段参与合图。

  4a. 调整 scope 框架

      回到 Step 0b 的四条规则，重新审视 scope 布局：

      方向1: CV 全合 → CV 不全合
        - 当前 scope 包裹了所有 CV 段（全合），性能退化
         - 按规则3放出涉及跨迭代依赖的 V 段（is_loop_begin/end 内操作跨迭代依赖 tensor 的段）
        - 放出的段用独立 sg_set_scope 做普通合图或局部 Mix合图

      方向2: CV 不全合 → 扩大 scope
        - 当前 scope 只包裹了部分 CV 段（不全合），性能退化
        - 按规则1/2/4检查：是否有 V→C 或 C→V 的数据流对被拆在不同 scope？
        - 特别注意：V0(gather+dequant+assemble) 如果不在 scope 内，
          gather 结果走 DDR 到 C1，应按规则4纳入 scope 走 CV 通路

      方向3: 切换不同的不全合分段方案
        - 尝试不同的 scope 拆分方式
        - 分析 scope 布局是否合理

  4b. 在新 scope 框架下重新走阶段 A（3a→3b→3c）

  阶段 B 走完仍无收益（既未达标也无提升趋势）→ 进入 Step 5

Step 5: 切换另一种开关方式（自动↔手动）
━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⛔ 只有当前开关方式下 Step 3（阶段A：3a→3b→3c）+ Step 4（阶段B：调整scope框架）全部走完仍无收益（既未达标也无提升趋势）才进入此步骤。
  ⛔ 禁止从 Step 3c 直接跳到 Step 5——必须先走 Step 4。

  - 当前是自动合图 → 切换到手动合图（重置 pypto.experimental.auto_mix_partition(0)，添加 sg_set_scope）
  - 当前是手动合图 → 切换到自动合图（移除 sg_set_scope，调用 pypto.experimental.auto_mix_partition(1)）
    ⚠️ 如果自动合图在 Step 2 中已编译超时，跳过此方向，直接进入 Step 6

  ⛔ **切换后等同于 Mix合图重新开始调优**：
  - nbuffer 重置为 1（`vec_nbuffer:{"DEFAULT":1}` + `cube_nbuffer:{-1:1}` + `cube_l1_reuse:{-1:1}`）
  - 必须重新走 Step 3（阶段A：分析原因→调参，包括 3c 的 nbuffer 逐步调大）→ Step 4（阶段B：调整scope框架）
  - ⛔ 禁止切换后直接判断退化就回退——刚切换时退化是因为缺配套参数，不是开关方式本身的问题
  - 切换前的 Step 0 数据流分析结论仍然有效（scope 布局方案不变），但配套参数需在新开关方式下重新调优

  两种开关方式全部走完仍无收益 → 进入 Step 6

Step 6: 退出 Mix合图
━━━━━━━━━━━━━━━━━━━━
  ⛔ 只有 Step 1→2→3→4→5 全部走完仍无收益，才允许退出 Mix合图。
  移除 Mix合图配置，转向其他优化（S-4 普通合图 / S-9 Stitch / S-10 调度等）。
  记录退出原因和已尝试的全部配置到调优日志。
```

**⛔ 关键路径执行检查表**（每步完成后打勾，未打勾禁止跳到下一步）：

```
□ Step 0: 数据流分析（⛔ 禁止跳过）
  □ 0a 识别 CV 数据流（填写 CV 数据流表）
  □ 0b 确定 scope 布局（基于规则1-4：V→C 同 scope / C→V 同 scope / 跨迭代依赖放出 / gather 纳入）
  □ 0c 确定配套参数（loop tile / unroll / cube L1 / vec tile分段 / gather宽度 / ooo_sched / workspace / nbuffer=1）
  □ 0d 输出分析结论（scope 布局方案 + 每项配套参数的推荐值和依据）

□ Step 1: Mix合图 + 配套参数已提交实测（基于 Step 0 结论，先不带 debug_options）
  □ 达标？→ 完成
  □ 编译超时？→ Step 2
  □ 提升但未达标？→ Step 3c 调参
  □ 退化？→ Step 3

□ Step 2: 编译超时处理（禁止移除 Mix合图）
  □ 2a 移除 debug_options + 添加 host_options → 重试
  □ 2b 回退 unroll（记录被回退的值）→ 重试
  □ 2c 回退 nbuffer（⛔必须检查，包括Step 3c调参过程中调大的）→ 重试
  □ 2d 调小 TileShape/loop tile/vec tile → 重试
  □ 编译通过 → Step 3c（不再试已回退的 unroll 值，nbuffer调大后超时回到2c）
  □ 所有手段无效 → Step 5

□ Step 3: 阶段 A — 当前 scope 内分析原因并调参（⛔ 禁止跳过分析直接回退）
  □ 3a 分析退化原因（必须回答 Q1-Q3 并记录）
    □ Q1: CV 通路是否生效？（program.json）
    □ Q2: spill 数量是否过多？（WorkspaceGm ≤ 20？）
    □ Q3: 退化幅度与哪个因素最相关？
  □ 3b 根据原因在当前 scope 内调参
    □ 原因1(CV未生效) → 排查硬限制，在当前 scope 内修复
    □ 原因2(spill过多) → 调小 TileShape/nbuffer
    □ 原因3(串行化减并行度) → 调小 loop tile/L0
    □ 原因4(均正常) → 3c 充分调参，仍退化则 Step 4
  □ 3c 配套参数逐个调优（⛔ 8 项全部标记 ✅或❌ 后才允许退出）
    □ nbuffer: vec_nbuffer 1→2→4→8→16→32… 逐值实测（劣化则回退）
    □ loop tile: 至少 3 个值
    □ cube L0: [128,128]→[64,64]→[256,128]
    □ cube L1: [128,128]→[128,256]→[256,256]
    □ vec tile: 按 V 段 shape 变化点分段设置
    □ unroll_list: 至少 1 次降档
    □ ooo_sched_mode: GAPMIN → HLF
    □ max_workspace_kb: 如果 spill 多则调大
  □ 达标？→ 完成
  □ ⛔ 3c 调完后无论"仍退化"还是"有提升但未达标" → 都进入 Step 4（禁止直接跳 Step 5）

□ Step 4: 阶段 B — 调整 scope 框架（⛔ 3c 调完后必须走此步骤，禁止跳过到 Step 5）
  □ 4a 回到 Step 0b 四条规则重新审视 scope 布局
    □ 方向1: 全合→不全合（按规则3放出跨迭代依赖 V 段）
    □ 方向2: 不全合→扩大 scope（按规则1/2/4检查是否有数据流对被拆开，如 V0 gather）
    □ 方向3: 切换不同的不全合分段方案
  □ 4b 新 scope 框架下重新走阶段 A（3a→3b→3c）
  □ 达标？→ 完成
  □ 仍无收益 → Step 5

□ Step 5: 切换另一种开关方式（自动↔手动）
  □ ⛔ 切换后等同于Mix合图重新开始：nbuffer重置为1，必须重走Step 3→4完整流程
  □ ⛔ 禁止切换后直接判断退化就回退——刚切换时退化是缺配套参数，不是开关方式问题
  □ 如果自动合图在 Step 2 中已编译超时 → 跳过自动方向，直接 Step 6
  □ 切换后重新走 Step 3（阶段A）→ Step 4（阶段B）
  □ 达标？→ 完成
  □ 仍无收益 → Step 6

□ Step 6: 退出 Mix合图（记录原因和已尝试配置）
```

> **注意**：`pypto.experimental.auto_mix_partition` 默认关闭（0）。自动合图还受代价和收益判断影响，可能跳过部分段，因此不作为确定性保证。需要精确控制合图段时用手动 `sg_set_scope`。

**说明**：

- `sg_set_scope` 后面的数字（如 1、5001）只是唯一标志，无功能差异，在一个 kernel 内不重复即可。**相同数字表示同一段合图，不同数字表示不同段合图**。scope ID 本身没有"必须大于 5000"之类的特殊语义。
- `sg_set_scope` 作用于设置之后创建的算子；段尾必须恢复为 -1，否则后续算子会继续带上该 scope。同一 scope ID 的 merge flags 必须一致；scope=-1 时两个 merge flags 必须都是 False。
- 包裹范围应覆盖有 CV 数据依赖的连续计算段（Cube↔Vector 交替），Mix合图与普通合图的区别就在于包裹范围内是否包含 Cube 段
- 一个 loop 迭代内可包裹一段或多段 CV 交替计算
- **sg_set_scope 三元组**：`sg_set_scope` 除正整数 ID 外还支持三元组形式 `(id, allowParallelMerge, allowCrossScopeMerge)`，控制段间合并行为。实测参考配置 `(1, True, False)`（允许段内并行合并，禁止跨段合并）。仅单段简单合图用正整数即可，多段合图需协调合并行为时用三元组。
- **Lite 平台边界**：DAV_3113/3003 等 Lite 平台不依赖 `sg_set_scope`，靠 LiteNPU 图形模式自动推断。本节 mixed-CV scope 方法仅适用于 DAV_3510（A5），Lite 平台无需配置。
- **Mix合图默认 nbuffer 配置**：`vec_nbuffer_setting={-1: 1}` 和 `cube_nbuffer_setting={-1: 1}` 是 Mix合图的默认配置（粒度 1，不额外合并，交由 Mix合图接管融合）。若 `cube_nbuffer_setting` 无效，可尝试 `cube_l1_reuse_setting={-1: 1}` 替代。此处 nbuffer=1:1 是**软件级缓冲配置**，与硬件 CV 配比 1:2 是不同层面的概念——硬件配比 1:2 决定物理核如何连接，软件 nbuffer=1:1 决定编译器是否额外合并子图，两者不矛盾。nbuffer 调优见 §4.2.5 第 2 条
- **cube_nbuffer_setting 与 Mix合图的配合**：Mix合图消除 CV 间搬运后，仍可通过 `cube_nbuffer_setting` 进一步合并同构 Cube 子图减少调度开销。两者互补——Mix消除 CV 间 DDR 搬运，cube_nbuffer 减少 Cube 子图间调度开销。建议 Mix合图生效后在默认 `cube_nbuffer_setting={-1: 1}` 基础上从 1 开始按 2 的幂次递增逐值实测（1→2→4→8→16→32…，劣化则回退），与 `cube_l1_reuse_setting` 配合使用。⚠️ 两者的 hashOrder 分析和粒度设置应基于 analyze_swimlane.py 的输出，不宜同时过大（见 §3.3 协同原则）

##### 4.2 硬性限制条件（⛔ 不满足则 Mix合图不生效）

1. **⛔ CV 间数据传递与消费关系**（走 CV 通路的必要条件）：

   - **方向限制**：CV 间传递数据的发送方与接收方数量关系必须为 1:N（1 个发送给 N 个）或 N:1（N 个发给 1 个），不支持 M:N（多对多交叉）。这是 A5 芯片 CV 通路物理连接的硬性限制。若存在 M:N 数据传递，需调整代码结构拆分为多个 1:N/N:1，否则该数据走 DDR。（注：1:N 是**相同需求**广播给多个核，属正常；禁用的是不同需求的多个消费者并行读取同一 matmul 结果。）
   - **消费者扇出限制**：一个 matmul 结果只允许喂一条 vector 计算链。若多个 vector 消费者对同一 matmul 结果有**不同需求**（不同计算路径），须复制数据拆分为独立链，否则编译报错或数据走 DDR。
   - **消费者同核约束**：同一 `L0C_COPY_UB` 的所有 vector 消费者必须在同一 AIV 核上，否则编译报错。
2. **CV 前后不允许交叉大小的不规则 shape**：CV 之间传递数据的 shape 必须保持单调变化（要么小搬大，要么大搬小）。禁止交叉大小变化，例如 `[64, 128] → [128, 64]` 这种行列互换的 shape 不能走 CV 通路，需调整代码顺序或 reshape 使其单调。
3. **⛔ UB 使用上限 248KB**（关键！常被忽略）：A5 芯片 UB 总容量约 248KB（经验值，实测留余量建议按 < 240KB 规划）。单个 tensor 的 ND+NZ 总大小须 < 248KB，一旦超限，编译器会将该 tensor 从 CV 通路降级为 DDR 搬运（COPY_OUT/COPY_IN），Mix合图在该 tensor 上失效。

   - **判断方法**：检查单个 tensor 的 ND+NZ 总大小是否 < 248KB。
   - **对策**：减小 tile_shape（如 gather_vec_tile_shape 第一维减半）使单 tensor 的 ND+NZ 总大小 < 248KB。
4. **⛔ 算子衔接形态**（CV 交替段的结构约束）：

   - matmul 必须直接接 vector（matmul 结果走 L0C→UB 给 vector），或 vector 输出直接喂 matmul（vector 结果走 UB→L1 给 matmul）。中间不得插入其他 Cube 操作。
   - **判断依据是最终 TileGraph 而非是否手写**：CV 通路是否形成取决于展开、切块后的最终 TileGraph 是否匹配直连识别模式，而非用户是否手写了 view/reshape/assemble。框架可在中间自动插入 View、Assemble 等节点；显式拼接场景也可合法手写这些算子（如 `vector → assemble → view → matmul`）。
   - **DAV_3510 识别的两类 producer-consumer 关系**：
     - 小块→大块：通过 ASSEMBLE 汇聚
     - 大块→小块：通过 VIEW 拆分
   - **应避免**：在 cube 和 vector 之间插入不受支持的计算算子；通过复杂 reshape 改变切块关系导致两端 tile 无法建立整倍数映射；同一维度关系不一致（如一维放大、另一维缩小）。
5. **⛔ shape/tile 硬数值约束**（衔接处的对齐与容量限制）：

   - **衔接 tensor 必须 2D**：CV 间传递的 tensor 必须为 2 维，3D 及以上须先 reshape。
   - **L0C→UB vec tile 16 对齐**：Cube→Vector 方向，vec tile 的两维须 16 对齐（CheckUBTileShape 校验）。
   - **cube tile 与 vec tile 衔接轴相等或整数倍**（IsDimMultiple 校验）：CV 衔接轴的 cube tile 与 vec tile 须相等或为整数倍关系。
   - **UB→L1 内轴切分 32B 对齐**（C0 对齐）：Vector→Cube 方向，UB→L1 的内轴切分大小须为 32B/dtype 字节数的倍数。
    - **带 assemble 场景输出 ≤ UB×0.35**：含 assemble 的合图段，输出 tensor 大小须 ≤ UB 容量的 35%，否则走 DDR。

##### 4.2.5 调优注意事项（非限制条件，但影响性能选型）

以下内容不是"不满足则 Mix合图不生效"的硬性限制，而是调优过程中的关键决策点：

1. **scope ID 只是唯一标记，无功能差异**：`sg_set_scope` 的数字（如 5001、20001）只是一个唯一标识符，在一个 kernel 代码中不重复使用即可，对 Mix合图行为无影响。
2. **nbuffer 配比须实测对比**：真正的性能差异来自 **nbuffer 配比**：nbuffer 调大允许编译器跨迭代重叠计算与搬运，但同时增加 UB 并发占用，可能引发 spill（溢出 workspace GM，性能下降）。实测中大 nbuffer 也可能导致部分 tensor 退回 DDR 中转，须逐值验证 CV 通路是否仍生效（解析 program.json，见§4.5）。须逐值实测取优。

   - **建议**：从 1 开始按 2 的幂次递增逐值实测性能（1→2→4→8→16→32…，1/2/4/8 是常用范围不是上限，8 有收益继续增大），取最优配置——若调大后性能劣化则回退至上一个最优值，不再继续增大。
3. **spill 数量参考阈值 ≤ 20**（性能指标，非硬性限制）：Mix合图场景下，spill 指编译器因 UB/L1 寄存器不足将数据溢出到 workspace GM 的次数，在泳道图中体现为 `WorkspaceGm`。复杂算子 spill ≤ 20 属正常（经验阈值，非绝对标准）；**超过 20 说明性能还有优化空间**，一般调小 TileShape或调小 nbuffer 会减少 spill（减小并发 tensor 占用，释放寄存器/UB 空间）。spill 与 UB 并发使用（§4.2 第 3 条）相关但不同：UB 超限导致数据走 DDR 中转（CV 通路失效），spill 超限导致数据溢出到 workspace GM（性能下降但 CV 通路可能仍生效）。
4. **多次尝试，不可一次劣化即回退**：A5 场景下 Mix合图是核心性能调优手段，其核心收益是消除 CV 间 DDR 搬运开销，最终一般都能带来优化。首次配置后若性能劣化，**不可直接回退放弃**——需结合本节其他条目多次尝试：同步调小 TileShape（第 9 条）、对比 nbuffer 配比（第 2 条）、调整 unroll 策略（第 8 条）、排查 DDR 回退（§4.5/§4.6），确认所有合理组合都试过仍无收益才可回退。

    **⛔ Mix合图调优必须按 §4.1 关键路径（Step 0→1→2→3→4→5→6）执行**。关键路径包含完整的数据流分析（Step 0）、编译超时处理（Step 2）、退化分析与调参（Step 3/4）、开关切换（Step 5）等全部流程。禁止跳过关键路径中的任何步骤。

   **Mix合图原子优化点配套参数候选值表**：首次开启 Mix合图时，使用以下"推荐起始值"作为原子优化点一次性提交。退化后按"候选值"列逐个替换（每次只改一个），在 Mix 框架内迭代：

   | 参数 | 推荐起始值 | 候选值（逐个尝试） | 说明 |
   |------|-----------|-------------------|------|
   | cube L0（mL0, nL0） | [128, 128] | [128,128] / [64,128] / [256,128] | 128 附近通常最优，勿用极小值（如 16，会破坏 CV 通路 shape-tile 衔接轴整数倍约束） |
   | cube L1（kAL1, kBL1, mL1, nL1） | [128, 128] | [128,128] / [128,256] / [256,256] | 与 L0 相近或更大 |
   | s2_tile（loop tile） | 1024 | 1024 / 512 / 2048 | Mix 串行化减少 task 数，调小 loop tile 增加任务数弥补并行度损失 |
    | vec tile | [128, 128] | [128,128] / [128,512] / [128,256] / [64,128] / [32,512] | 根据实际 tensor 维度设置，优先用满尾轴；与 cube L0 同方向 |
    | nbuffer（vec/cube） | {-1: 1} | 1→2→4→8→16→32… 逐值实测（劣化则回退） | 从 1:1 开始，按 2 的幂次递增 |
    | unroll_list | [8,4,2,1] | [8,4,2,1] / [4,2,1] / [4] / [2,1] / [1] | 全展开最优；编译超时则降档（按第 7 条对策）。Mix合图场景下推荐多级 unroll_list（如 [8,4,2,1]），多级展开减少循环调度开销，单值（如 [1]）仅在多级编译失败时使用 |

    ⛔ 首次开启 Mix 时，使用"推荐起始值"列的全部值作为原子优化点一次性提交。退化后，按候选值表逐个替换参数（每次只改一个），在 Mix 框架内迭代。具体配套参数由 Step 0c 分析确定。

    **前置：scope 范围分析**：进入 Mix合图前必须先完成 §4.1 关键路径 Step 0 数据流分析（0a: CV 数据流表 → 0b: scope 布局方案 → 0c: 配套参数推荐值 → 0d: 分析结论）。

5. **CV 全合 vs 不全合策略**：

    **⛔ 核心原则：CV 全合和 CV 不全合都是合法的调优路径，不存在谁对谁错。** 选定一条路径后，必须在该 scope 范围内充分调参（nbuffer、TileShape、unroll、spill 等），所有参数都调完仍退化才考虑切换 scope 范围。scope 范围是调优的框架，不是调优的参数——先在一个框架内把参数调到位，框架不行再换框架。

    Mix合图默认尝试将所有 CV 交替段合入同一段 mix合图（CV 全合）。CV 全合的最优结果须同时满足以下四个条件：
   - **(1) 用满核无气泡**：从泳道图看，每列并发的任务数要占满所有物理核（A5 为 32 个 C 核 + 64 个 V 核），无显著等待气泡
   - **(2) 全走 CV 通路**：所有 CV 间数据走 CV 通路（片上直连），无 DDR 中转（解析 program.json 验证，见 §4.5）
   - **(3) 无 spill**：编译器无数据溢出到 workspace GM（泳道图中 `WorkspaceGm` 数量为 0 或极少 ≤ 20）
   - **(4) 核内计算流水排布紧密无空闲**：CV 交替段内计算与搬运流水化，核上无显著空闲间隙

   **⚠️ 全合与核占满的内在矛盾**：CV 全合将多段交替合为一段，单段子图规模放大；若切分粒度不变，任务数随之减少，可能占不满核（违反条件 1）。**调小 TileShape 可能有收益**——调小核内 TileShape 不仅提高核内并行度，还增加任务数以填满核，同时减少 UB 并发占用降低 spill，是逼近最优结果的首要手段。当算子总计算量小，或全合后子图过大导致即便调小 TileShape 任务数仍 < 物理核数时，全合反而劣化，应改 CV 不全合：放出部分段缩小单段子图规模，增加并发任务数。先以 CV 全合多次尝试（配合第 9 条配套 TileShape 调整、对比 nbuffer、排查 DDR 回退），**满足以下任一条件即改尝试 CV 不全合**：
   - **AICore E2E Time 无收益**：CV 全合（含配套 TileShape/nbuffer 调优）多次尝试后 AICore E2E Time 未下降
   - **四个最优条件中有明确违反且无法通过调优修复**：如全合后 task 数 < 物理核数且调小 TileShape 仍无法增加任务数（违反条件 1）；或 spill > 20 且调小 TileShape/调小 nbuffer 均无法降低（违反条件 3）

   无需四个条件全部验证不通过才改不全合——四个条件是判断"是否达到理论最优"的标准，而非判断"是否应改不全合"的门槛。只要性能无收益或某条件明确违反且不可修复，即可改不全合。**⚠️ 自动合图（`pypto.experimental.auto_mix_partition(1)`）模式下，全合/不全合由编译器自动决策，用户无需手动放出段**——以下手动放出段策略仅适用于手动 `sg_set_scope` 模式。

   **⛔ CV 全合最低尝试要求**：CV 全合首次开启后，"多次尝试"不是模糊的——必须完成以下全部尝试维度后才允许判定"无收益"并切换到 CV 不全合：

   | # | 尝试维度 | 最低尝试次数 | 具体操作 | 说明 |
   |---|---------|------------|---------|------|
   | 1 | 原子优化点 | 1 次 | CV 全合 + 推荐起始值（见候选值表） | 首次实测 |
    | 2 | nbuffer 调整 | 至少 2 个值 | 1→2→4→8→16→32…（逐值实测，劣化则回退不再增大） | 从 1:1 开始按 2 的幂次递增 |
   | 3 | TileShape/loop tile 调整 | 至少 1 次 | s2_tile 或 cube tile 调整（按候选值表） | 在 Mix 框架内调小 |
   | 4 | unroll 策略 | 至少 1 次 | unroll_list 降档（如 [8,4,2,1]→[4,2,1]） | 编译超时按第 7 条对策 |

   ⚠️ 编译超时不计入"尝试失败"，按第 7 条对策处理（调小 TileShape/降 unroll）后重试。
   ⚠️ 以上 4 个维度全部完成后 E2E 仍退化（至少 5 轮）→ 当前 scope 范围内参数已充分调优 → 按诊断决策树进入阶段 B，切换 scope 范围（如 CV 全合→CV 不全合），在新范围内重新充分调参。

   **Mix合图内部调优的轮次计算**：Mix合图（S-14）内部调优轮次按以下规则计入 SWIMLANE 阶段的"连续无提升"计数器：

   | 操作 | 是否计入轮次 | 说明 |
   |------|------------|------|
   | CV 全合首次开启（原子优化点） | ✅ 1 轮 | 含配套 TileShape 的首次实测 |
   | CV 全合→CV 不全合切换 | ✅ 1 轮 | scope 范围调整后重新实测 |
    | nbuffer 调整（1→2→4→8→16→32…） | ✅ 每个值 1 轮 | 逐值实测，劣化则回退 |
   | TileShape 调整（配套参数） | ✅ 每个参数 1 轮 | 每次只改一个 |
   | unroll_list 降档 | ✅ 每档 1 轮 | 编译超时按第 7 条对策后重试 |
   | program.json 诊断（不改变参数） | ❌ 不计入 | 诊断性操作 |
   | DDR 回退排查（不改变参数） | ❌ 不计入 | 诊断性操作 |

   ⚠️ Mix合图内部的"连续无提升"独立计数。当 Mix 内部连续无提升达到 5 轮时，必须完成退化诊断决策树的全部 4 步。4 步全部排查完且仍无收益，才允许退出 Mix 合图转向其他优化（如 S-4 普通合图/S-9 Stitch 等）。禁止在 Mix 退化时直接跳到 ooo_sched_mode/vf_options 等非 Mix 优化。

   **⛔ CV 不全合的判断标准与操作方法**：

   **判断哪些 V 段应该从 Mix合图中放出**：存在**跨迭代依赖**的 V 段应放出。判断标准：
   - 该 V 段在 `is_loop_begin` / `is_loop_end` 条件分支内
   - 且写入在 loop 外声明、loop 内读写的 tensor（构成跨迭代依赖）
   - 这类 V 段放进 Mix合图会导致 UB 并发占用过大（跨迭代依赖 tensor 生命周期跨越整个 loop），引发 spill 或 DDR 回退

   **放出后的处理**：放出的段**必须**用 `sg_set_scope` 单独包裹做局部合图——没有被 mix合图包裹的独立段，用 scope 单独包起来一般也会有性能提升（减少段内子图间调度开销和数据搬运开销）。具体方式：
   - 若放出的段仍含 CV 交替结构 → 做局部 Mix合图（A5 平台，scope 用正整数）
   - 若只含 V 段 → 做普通合图（scope 用正整数，不包裹 Cube 操作）
   - 为每个独立合图段设置不同的正整数值（从 1 开始，不要重复），不同 scope 值的段分属不同 stitch group，组间通过 DDR 传递数据

   **典型 scope 布局示例**（attention 类算子）：
   ```
   scope=5001 (Mix合图): C1(matmul) + V1(softmax) + C2(matmul)  → 消除这3段间CV DDR搬运
   scope=-1              → 关闭Mix合图
   scope=1 (普通合图):   V2 else分支(flash update: exp/mul/add/div/cast/assemble) → 减少V2内部调度开销
   scope=-1              → 关闭普通合图
   scope=2 (普通合图):   V2 loop_begin分支(flash init: div/reshape/cast/assemble) → 减少V2内部调度开销
   scope=-1              → 关闭普通合图
   ```
   即 Mix合图只包裹 C1+V1+C2（消除 CV 间搬运），V2（flash update，存在跨迭代依赖）从 Mix合图中放出，用独立的 scope 做普通合图减少 V2 内部子图间调度开销。
6. **⚠️ Mix合图对核上 compute 的影响须以 AICore E2E Time 为准**：Mix合图通过 `sg_set_scope` 改变 CV 调度方式，可能减少 CV 间 DDR 搬运（降低 wall time / device time），但**同时可能增加核上等待时间**（CV 交替段的同步开销、scope 内子图串行化），导致 **AICore End-to-End Time（核上 compute）上升**。实测案例：某 attention 算子 Mix合图后 device time 从 1727us 降至 667us，但 AICore E2E Time 从 102us 升至 167us——核上计算反而变慢。**⛔ Mix合图的优化效果必须以 AICore E2E Time 下降为准**，不能仅看 wall time 或 device time。若 Mix合图后 AICore E2E Time 上升，即使 wall time 下降也属错误优化方向，应回退或调整 scope 策略（如 split scope 替代全合 scope）。
7. **Mix合图后子图膨胀可能引发编译超时**：Mix合图本身会将 CV 交替段合并为大子图，叠加其他会进一步放大子图规模的调优（如调大 `unroll_list` 展开更多迭代、调大 nbuffer 增加并发缓冲、增大 loop tile 等）可能导致编译器 pass 阶段复杂度爆炸，表现为编译卡死数分钟无进展。**⛔ 编译时间门槛 = 20 分钟**：Mix合图（及叠加其他放大子图规模的调优）单次编译耗时超过 20min 即判定为不可应用——即使有性能收益，编译时间过长无法应用到实际整网中，直接回退或改尝试缩小子图规模的方案。**对策**（按优先级）：① 改小或回退当次会放大子图的调整；② 减小 Mix合图包裹段——按第 5 条「CV 全合 vs 不全合」将部分段从 mix合图中放出，缩小单段子图规模后重试；③ **调小核内 TileShape（L0）和 loop tile**——TileShape 过大是编译路径爆炸的常见根因，配合第 9 条的配套 TileShape 调整（L0 调小 + L1 调大 + loop tile 调小）可显著减少编译产物规模。
 8. **Loop unroll 减少调度开销**：可尝试对最内层 loop unroll 更多子图或全 unroll，以减少循环迭代次数和 host task 调度开销（`unroll_list` 包含所有可能的尾块次数，或对静态轴用 `range` 展开）。**Mix合图场景下推荐多级 unroll_list**（如 `[8,4,2,1]`），多级展开让编译器根据循环剩余次数选择最优展开粒度，比单值（如 `[1]`）减少更多循环调度开销。单值仅在多级编译失败或迭代间数据依赖严重时使用。**⚠️ 全 unroll 副作用**：全 unroll 将每个循环迭代编译为独立代码路径，若迭代间存在跨迭代依赖（如 online softmax 的 mi/li/oi 累积状态），编译器无法跨代码路径保持状态，被迫走 DDR 传递（见 §4.6）——属算法固有依赖，数据量小时性能影响可忽略。
9. **⛔ Cube/Vector 核内 TileShape 必须同步调整**（Mix合图后强制步骤）：设置 Mix合图后，**必须同步调整核内 TileShape**（通过 `set_cube_tile_shapes` / `set_vec_tile_shapes` 设置），这是 Mix合图调优中不可跳过的关键步骤。

   **合图前 vs 合图后 TileShape 调优方向对比**：

   | 维度 | 合图前（传统合图） | 合图后（Mix合图） |
   |------|-----------------|-----------------|
   | CV 间数据传递 | 走 DDR 搬运（耗时） | 走 CV 通路（片上直连，几乎无开销） |
   | 性能瓶颈 | 搬运开销主导 | 计算并行度 + 任务紧凑性主导 |
   | TileShape 方向 | **调大**（一次搬运更多数据，减少搬运次数） | **分方向调整**（见下方） |
   | 调小收益 | — | ① 增加任务数填满物理核（提升核内并行度）；② 减小单 task 粒度使 CV 交替更紧凑（提升核内计算任务紧凑性，减少核间等待） |
   | 调小风险 | 搬运次数增多，性能退化 | 无搬运开销，调小一般无负面影响（仅极端过小可能增加调度开销） |

   **⛔ `set_cube_tile_shapes` 的 L0 与 L1 调优方向相反**：

   `set_cube_tile_shapes` 接受三组参数 `[mL0, nL0], [kAL1, kBL1], [mL1, nL1]`（具体顺序以 API 文档为准），其中 L0 和 L1 在 Mix合图后的调优方向不同：

   | 参数 | Mix合图后方向 | 原因 |
   |------|-------------|------|
   | L0 tile（如 `[mL0, nL0]`） | **调小**（如 256→128） | Mix合图消除 CV 搬运后，核内并行度成为瓶颈，小 L0 增加核内并行 |
   | L1 tile（如 `[kAL1, kBL1, mL1, nL1]`） | **调大**（如 64→128） | 增大 L1 复用范围，减少重复搬运，配合 `cube_l1_reuse_setting` |
   | `set_vec_tile_shapes` | **调小** | 与 Cube L0 同方向，减小单 task 粒度使 CV 交替更紧凑 |

   **调优方法**：Mix合图生效后，将 Cube L0 tile 和 vec tile 从合图前的大值（如 256）调小至 128 量级，同时将 Cube L1 tile 从小值（如 64）调大至 128 量级。**⚠️ 注意区分 loop tile**：此处调的是核内 TileShape，不是控制循环次数的 loop tile（如 s2_tile）。Mix合图将 CV 段串行化会减少 task 数，**调小 loop tile**（如 2048→1024）可增加 task 数弥补并行度损失。三者（L0 调小 + L1 调大 + loop tile 调小）作为一组同时调整。

   **⚠️ "调小/调大"是相对方向，128 附近通常最优**：上述 L0 调小 / L1 调大是方向性指导，非绝对值。L0 和 L1 的最优值通常在 128 附近（如 [128,128] / [128,256]），而非极端值（如 L0=16 或 L1=1024）。从 128 开始尝试，按候选值表逐步调整。

   **⛔ 合图前后 TileShape 衔接规则**：FRONTEND 阶段为无合图场景调优的 TileShape，在进入 Mix合图后需要重新评估。无合图时为特殊场景调小的非标准值（如 decode 小 M 场景 mL0=16），在合图后需要回到标准值（128 附近），因为 Mix合图需要标准对齐值建立 CV 通路，非标准值（如 mL0=16）会破坏 §4.2 第 5 条 shape/tile 衔接轴整数倍约束。Mix合图首次开启时，将 FRONTEND 阶段调优的非标准 TileShape 回退到标准值，作为"原子优化点"的一部分一次性提交。

   | 场景 | 无合图时（FRONTEND 阶段） | Mix合图后 | 原因 |
   |------|-------------------------|----------|------|
   | decode 小 M（M<128） | mL0 可调小到 M 值（如 16） | mL0 回到 128 | Mix合图需要标准对齐值建立 CV 通路，mL0=16 破坏衔接轴整数倍约束 |
   | 标准 M（M≥128） | mL0=128 | mL0=128 | 无变化 |
   | loop tile（s2_tile） | 可较大（2048） | 调小（1024） | Mix 串行化减少 task 数，调小 loop tile 增加 task 数弥补 |

   **⛔ Mix合图配置的组合依赖**：Mix合图的性能收益依赖于核内 TileShape（L0 调小 + L1 调大）和 loop tile（调小）的配合，三者存在组合依赖——单独开启 Mix合图或单独调任一 TileShape 维度通常退化（Mix 串行化减少 task 数，但不调 TileShape 无法弥补并行度损失）。因此 Mix合图调优时**不按单参数原则逐个试**，而是**一次性设置全套配套配置**（Mix合图开启 + L0 调小 + L1 调大 + loop tile 调小 + nbuffer 默认 1:1），在配套配置基础上再逐个微调。
10. **CV 通路未生效时不要直接回退**：开始调试 Mix合图时，若解析 program.json 发现 CV 间数据没走 CV 通路（或只部分走），**不要直接回退放弃 Mix合图**——CV 合图在满足必要条件下肯定能走 CV 通路，没走或没全走说明某项必要条件未满足。须按 §4.2 硬限制逐项排查（方向/扇出/同核、shape 单调、UB 248KB、算子衔接形态、shape/tile 对齐），结合 §4.5 验证方法定位回退原因，多次尝试调整后再下结论。

##### 4.3 代码示例（参考 sparse_flash_attention_quant_d_950）

**方式 A：自动合图（优先尝试）**

```python
@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 1},      # v_n 配比 1
        "cube_nbuffer_setting": {-1: 1},     # c_n 配比 1（无效则改 cube_l1_reuse_setting={-1:1}）
    },
    runtime_options={...}
)
def kernel(...):
    pypto.experimental.auto_mix_partition(1)   # 自动合图，在 kernel 函数体内调用，无需手动 sg_set_scope
    pypto.experimental.set_operation_options(combine_axis=True)

    for s2_idx in pypto.loop(0, bn_per_batch, 1,
            name="LOOP_L4", idx_name="s2_idx", unroll_list={16, 1}):
        # 无需手动 sg_set_scope 包裹，编译器自动识别 CV 交替段
        pypto.set_semantic_label("Sa_V0")
        # ... Vector: gather kn/kr ...
        pypto.set_semantic_label("Sa_C1")
        # ... Cube: matmul(qi, kj) → sij ...
        pypto.set_semantic_label("Sa_V1")
        # ... Vector: softmax(sij) ...
        pypto.set_semantic_label("Sa_C2")
        # ... Cube: matmul(tilda_pij, vj) → q1 ...
        pypto.set_semantic_label("Sa_V2/Sa_UpdateVec2")
        # ... Vector: flash update ...
```

**方式 B：手动合图**

```python
@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 1},      # v_n 配比 1
        "cube_nbuffer_setting": {-1: 1},     # c_n 配比 1（无效则改 cube_l1_reuse_setting={-1:1}）
    },
    runtime_options={...}
)
def kernel(...):
    pypto.experimental.set_operation_options(combine_axis=True)

    # loop 间存在数据依赖（如 online softmax 跨迭代更新）→ 全 unroll
    for s2_idx in pypto.loop(0, bn_per_batch, 1,
            name="LOOP_L4", idx_name="s2_idx", unroll_list={16, 1}):
        # 开始 Mix合图
        if pypto.platform.npuarch == 'DAV_3510':
            pypto.set_pass_options(sg_set_scope=5001)

        pypto.set_semantic_label("Sa_V0")
        # ... Vector: gather kn/kr ...
        pypto.set_semantic_label("Sa_C1")
        # ... Cube: matmul(qi, kj) → sij ...
        pypto.set_semantic_label("Sa_V1")
        # ... Vector: softmax(sij) ...
        pypto.set_semantic_label("Sa_C2")
        # ... Cube: matmul(tilda_pij, vj) → q1 ...
        pypto.set_semantic_label("Sa_V2/Sa_UpdateVec2")
        # ... Vector: flash update ...

        # 结束 Mix合图
        if pypto.platform.npuarch == 'DAV_3510':
            pypto.set_pass_options(sg_set_scope=-1)
```

##### 4.4 调优检查清单

- [ ] 确认运行在 A5 平台（`pypto.platform.npuarch == 'DAV_3510'`）
- [ ] ⛔ 进入 Mix合图前必须先完成 Step 0 数据流分析（0a: CV 数据流表 → 0b: scope 布局方案 → 0c: 配套参数推荐值 → 0d: 分析结论），Step 1 的原子优化点必须基于 Step 0 结论
- [ ] 按 §4.1 关键路径 Step 1 开启 Mix合图 + 配套参数（基于 Step 0 结论的原子优化点），实测性能
- [ ] ⛔ 编译超时 → 按关键路径 Step 2（2a→2d 顺序）处理，**禁止移除 Mix合图**
- [ ] ⛔ 退化 → **禁止跳过分析直接回退到非 Mix 优化**，按关键路径 Step 3 阶段 A 先分析退化原因（3a: Q1 CV通路 / Q2 spill / Q3 退化因素 → 3b: 在当前 scope 内调参 → 3c: 配套参数调优），阶段 A 仍退化 → Step 4 阶段 B 调整 scope 框架（全合↔不全合）
- [ ] 当前开关方式全部走完仍无收益 → 按关键路径 Step 5 切换另一种开关方式（自动↔手动）
- [ ] 两种开关方式全部走完仍无收益 → 按关键路径 Step 6 退出 Mix合图
- [ ] ⛔ 检查 CV 间数据传递是否为 1:N 或 N:1（不支持 M:N 多对多，否则走 DDR）
- [ ] ⛔ 检查消费者约束：一个 matmul 结果是否只喂一条 vector 链（不同需求消费者须拆分）；同一 L0C_COPY_UB 的 vector 消费者是否在同一 AIV 核
- [ ] 检查 CV 间所有数据传递的 shape 变化是否单调（无交叉大小，如 [64,128]→[128,64] 禁止）
- [ ] ⛔ 检查算子衔接形态：matmul 是否直接接 vector（中间无其他 Cube）；最终 TileGraph 是否匹配直连识别模式（小块→大块 ASSEMBLE 汇聚 / 大块→小块 VIEW 拆分）；框架自动插入或合法手写的 assemble/view 均可
- [ ] ⛔ 检查 shape/tile 硬数值约束：衔接 tensor 是否 2D；L0C→UB vec tile 两维是否 16 对齐；cube tile 与 vec tile 衔接轴是否相等或整数倍；UB→L1 内轴切分是否 32B 对齐；assemble 场景输出是否 ≤ UB×0.35
- [ ] 配置 `vec_nbuffer_setting={-1: 1}` + `cube_nbuffer_setting={-1: 1}`（后者无效则试 `cube_l1_reuse_setting={-1: 1}`）
- [ ] 是否尝试 unroll 更多子图/全 unroll 减少调度开销
- [ ] Mix合图生效后同步调小 TileShape（如调至 128 量级）提升核内并行
- [ ] 检查泳道图中 spill（WorkspaceGm）数量是否 ≤ 20（经验阈值），超过则调小 TileShape或调小 nbuffer 减少 spill
- [ ] 每次修改后重新验证精度 + 测性能（⛔ 上板实测）
- [ ] ⛔ 检查 UB 使用：单个 tensor 的 ND+NZ 总大小是否 < 248KB
- [ ] 检查 CV 间传递的 tensor 生命周期是否过长（跨越多个计算阶段），尝试缩短
- [ ] ⛔ 从 1 开始按 2 的幂次递增逐值实测性能（1→2→4→8→16→32…，1/2/4/8 是常用范围不是上限，8 有收益继续增大），取最优配置（若调大后劣化则回退至上一个最优值，不再继续增大）
- [ ] 首次配置后若性能劣化，不可直接回退——多次尝试（同步调小 TileShape、对比 nbuffer、调整 unroll、排查 DDR 回退）确认无收益才可回退
- [ ] CV 通路未生效（program.json 无 CV 通路 opcode 或仅部分走）时不可直接回退——按 §4.2 硬限制逐项排查，多次尝试调整后再下结论
- [ ] ⛔ CV 全合和 CV 不全合都是合法调优路径。选定一个 scope 范围后，必须先在该范围内充分调参（配合调小 TileShape、对比 nbuffer、排查 DDR 回退、调整 unroll），所有参数都调完仍退化才切换 scope 范围（如 CV 全合→CV 不全合），在新范围内重新充分调参

##### 4.5 CV 通路生效验证方法（⛔ 调优必做）

Mix合图是否真正生效，**不能靠配置推断，必须通过泳道图/program.json 验证**。步骤：

1. **找到 program.json**：在 `output/output_*/` 目录下，或编译产物目录中。
2. **统计 opcode 判断数据流向**：

   | opcode                             | 含义                      | 方向         | 是否走 CV 通路        |
   | ---------------------------------- | ------------------------- | ------------ | --------------------- |
   | `L0C_COPY_UB`                    | Cube 结果从 L0C 拷贝到 UB | Cube→Vector | ✅ 是                 |
   | `UB_COPY_L1` / `UB_COPY_ND2NZ` | UB 数据拷贝到 L1          | Vector→Cube | ✅ 是                 |
   | `CV_SYNC_SRC` / `CV_SYNC_DST`  | CV 通路同步信号           | 双向         | ✅ 是（伴随上述操作） |
   | `COPY_OUT`                       | 数据写回 DDR              | 任一→DDR    | ❌ 否（走了 DDR）     |
   | `COPY_IN`                        | 从 DDR 读入               | DDR→任一    | ❌ 否（走了 DDR）     |
3. **验证方法**（⛔ 不能只看 opcode 是否存在，须追踪 CV 间数据流向）：

   - **目标**：Mix合图目的是让 CV 之间所有数据都走 CV 通路（片上直连），不走 DDR 中转。须验证**所有** CV 间数据是否都走 CV 通路，而非仅部分
   - **CV 通路 opcode**（走片上直连 ✅）：`L0C_COPY_UB`（Cube→Vector）、`UB_COPY_L1`/`UB_COPY_ND2NZ`（Vector→Cube）、`CV_SYNC_SRC`/`CV_SYNC_DST`（同步信号）
   - **DDR 中转**（走 DDR ❌）：`COPY_OUT` 后紧接 `COPY_IN` 且无 CV_SYNC 邻居，说明该数据走了 DDR 中转而非 CV 通路
   - **正常的 DDR 搬运（非 CV 间，不影响 Mix合图生效）**：算子最终输出写回 DDR、原始输入从 DDR 读入、跨迭代依赖数据（如 online softmax 的 mi/li/oi）走 DDR
   - **验证步骤**：
     1. 识别 CV 间应传递的数据（通过 semantic_label 和代码位置定位，如 Cube 的 matmul 结果传给 Vector 做 softmax、Vector 的 softmax 结果传给 Cube 做 matmul）
     2. 对每个 CV 间数据，追踪其在 program.json 中的传递路径：走 CV 通路 opcode（✅）还是 DDR 中转（❌）
     3. 若有 CV 间数据走了 DDR 中转，说明 Mix合图对该数据未生效，需排查 UB 超限 / shape 非单调 / tensor 生命周期过长
   - 用 `analyze_swimlane.py` 提取 leafHash 统计，或直接解析 program.json 的 operations 列表
4. **UB 超限排查**：对走 DDR 中转的 tensor，检查其单个 tensor 的 ND+NZ 总大小是否超过 248KB。
5. **spill 数量检查**（性能指标）：在泳道图中检查 `WorkspaceGm` 数量（即 spill 次数）。spill ≤ 20 属正常；超过 20 说明性能还有优化空间，一般调小 TileShape或调小 nbuffer 减少 spill（详见 §4.2.5 第 3 条）。

##### 4.6 常见 DDR 回退场景与对策

| 场景                              | 根因                                                    | 对策                                                                        |
| --------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------- |
| gather_in_ub 结果走 DDR           | gather 输出 tensor 的 ND+NZ 总大小超 248KB              | 减小 gather tile_shape；或缩短 tensor 生命周期（如复制后立即释放原 tensor） |
| online softmax 的 mi/li/oi 走 DDR | 全 unroll 将迭代编译为独立路径，跨路径无法保持状态      | 算法固有依赖，数据量小时性能影响可忽略，无需消除                            |
| 大 tensor（>128KB）走 DDR         | 单个 tensor 接近 UB 上限，ND+NZ 总大小超 248KB          | 减小 tile_shape 使单 tensor ND+NZ 总大小 < 248KB                            |
| CV 间 shape 交叉变化走 DDR        | shape 非单调变化（如 [64,128]→[128,64]）违反限制条件 2 | 调整代码顺序或 reshape 使 shape 单调                                        |
| bias/scale 强制走 DDR             | `copy_in_mode==0`，bias/scale 参数不走 CV 通路          | 属框架默认行为，无需消除                                                    |
| 动态 valid_shape 走 DDR           | L0C→L1 方向 valid_shape 非立即数时编译器无法静态推断    | 尽量用静态 shape；UB→L1 方向允许动态 valid_shape，L0C→L1 方向不支持          |

##### 4.7 Mix合图失败诊断子流程（⛔ 性能退化/无收益时强制执行）

> **触发条件**：Mix合图配置后性能退化（AICore E2E Time 上升）或无收益，且已按 §4.2.5 第 4 条多次尝试（调小 TileShape、对比 nbuffer、调整 unroll）仍无改善。

> **目的**：从 DDR 回退现象追溯到代码具体行，区分「不可修复的架构约束」与「可修复的代码结构」，给出具体改造方案。

**诊断流程（按顺序逐步执行）**：

```
Step 1: 采集 program.json（须带 debug_options 运行）
  └→ 定位所有 DDR 中转数据（COPY_OUT 后紧接 COPY_IN 无 CV_SYNC 邻居）

Step 2: 对每个 DDR 中转数据，追溯代码位置
  └→ 用 leafhash_to_code.py 将 DDR 数据的 leafHash 映射到前端代码行
  └→ 确认该 DDR 数据是哪个 gather / assemble / 中间 tensor 产生的

Step 3: 分类 DDR 断点（关键！）
  ├─ 不可修复（架构约束）：
  │   ├─ UB 超限：单 tensor ND+NZ > 248KB 且 tile_shape 已最小（§4.2 第 3 条）
  │   └─ M:N 数据依赖：多个不同需求消费者读同一 matmul 结果（§4.2 第 1 条）
  │      → 标记为「架构约束断点」，记录后继续诊断其他断点
  │
  └─ 可修复（代码结构）：
      ├─ 独立 gather 打断 CV 通路：CV 序列中间插入独立 gather_in_ub/gather_in_l1
      │   → 该 gather 的数据本可通过 view 复用已有数据，无需独立搬运
      ├─ 中间 assemble 到 DDR：CV 段内显式 assemble 写回 GM
      │   → 该数据应在 UB 内通过 view/reshape 传递，不写回 GM
      └─ tensor 生命周期过长：数据在 UB 中跨越多个计算阶段未释放
          → 缩短生命周期或复制后立即释放原 tensor

Step 4: 对「可修复断点」制定改造方案
  └→ 参考下方「代码结构修复模式表」选择对应修复模式
  └→ 每次只改一个断点，改后重跑精度+性能验证

Step 5: 所有可修复断点修复后，重新评估 Mix合图
  └→ 若 CV 通路全部闭合 → Mix合图生效，继续 nbuffer/TileShape 调优
  └→ 若仍有不可修复断点 → Mix合图在该断点处无法闭合，评估是否拆分 scope
      （将该断点处的 CV 段拆为独立 scope，其余段仍做 Mix合图）
```

**代码结构修复模式表**：

| 修复模式 | 诊断方法 | 修改方式 | 典型场景 |
|----------|---------|---------|---------|
| **view 复用消除独立 gather** | CV 序列中间有 `gather_in_ub`/`gather_in_l1` 取的数据，与已有数据存在重叠（如同一 source 的不同段、或已 gather 的 tensor 的子段） | 用 `pypto.view(已有tensor, [shape], [offset])` 切出所需数据，替代独立 gather | V2 阶段 vj 独立 gather kn 的部分段 → `view(kn)` 复用已 gather 的 kn |
| **合并 gather 减少搬运次数** | CV 序列中有多个 gather 取同一 source tensor 的不同列段 | 在算子入口将 source 拼接为 `[rows, col1+col2]`，一次 gather 取全部列段 | kn (512维) + kr (64维) 两次 gather → 拼接为 key_2d (576维) 一次 gather |
| **消除 CV 段内 assemble** | CV 合图段内有显式 `pypto.assemble` 将中间结果写回 GM | 改为在 UB 内通过 `view`/`reshape` 传递，仅在 scope 结束后 assemble 最终结果 | C1 输出到 GM 再 gather 进 C2 → C1 结果在 UB 内 view 给 C2 |
| **缩短 tensor 生命周期** | DDR 回退因 tensor 在 UB 占用过大导致后续 tensor 无法驻留 UB | 复制 tensor 后立即释放原 tensor（或缩小 tile_shape 减小单 tensor 占用） | 大 kn tensor 持续占用 UB → 用完后释放或缩小 gather tile |

**⛔ 诊断输出要求**：诊断完成后必须记录以下内容到调优日志：
```
### Mix合图失败诊断记录
- DDR 断点数量：X
- 不可修复断点：[列出每个断点的根因分类]
- 可修复断点：[列出每个断点的根因分类 + 修复模式]
- 修复后 CV 通路状态：[全部闭合 / 部分闭合（哪些段闭合）/ 仍不闭合]
- 最终决策：[继续 Mix合图 / 拆分 scope / 放弃 Mix合图]
```

**典型案例（sparse_flash_attention_quant bf16 路径）**：

> 原始实现：V0 阶段分两次 gather（kn 512 维 + kr 64 维），C2 阶段第三次 gather 取 vj。
>
> 诊断结果：
> - DDR 断点 1（可修复）：kn 和 kr 两次独立 gather 打断 CV 通路连续性 → 修复模式：合并 gather（拼接 key_2d=[kn,kr] 一次取 576 维）
> - DDR 断点 2（可修复）：vj 独立 gather kn 的部分数据，与已 gather 的 kn 重叠 → 修复模式：view 复用（`view(kn, [s2_tile, dn])` 切出 vj）
> - 无不可修复断点（matmul 输出 FP32 是 V1 softmax 的输入，可走 CV 通路）
>
> 修复后：CV 通路全部闭合，Mix合图 scope=5001 生效，性能从 1019us 降至 611us（-40%）

---

## 5. VF 融合（编排原则与增强旋钮）

### 5.0 四者关系澄清

VF 融合相关概念容易混淆，先明确边界：

| 层面         | 机制                                                                                  | 触发条件                       | 范围                                             |
| ------------ | ------------------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------ |
| 默认功能     | VF 融合                                                                               | A5 平台自动开启，无需任何开关  | 通用（A5）                                       |
| 代码编排优化 | 本节三条编排原则                                                                      | 写 vec 算子代码时遵循          | 通用（A5），独立于 mix合图                       |
| 编译选项优化 | `vf_options`（见 tune-swimlane SKILL.md §8 / [S-16]）                              | 显式配置`codegen_options`    | 通用（A5，含 Vector 计算的算子），独立于 mix合图 |
| 调优模式旋钮 | `sg_set_tunevf_mode`（见 [§5.4](#54-sg_set_tunevf_mode-旋钮vf-调优-pass-行为模式)） | 显式配置 VF 调优 Pass 行为模式 | 通用（A5，含 Vector 计算的算子），独立于 mix合图 |
| 效果增强旋钮 | `sg_set_ooo_scope`（见 [§5.3](#53-sg_set_ooo_scope-旋钮vf-融合效果增强)）           | 包裹需融合的 vec 算子段        | 仅 mix 子图                                      |

**关键**：VF 融合在 A5 上**默认开启**，无需 `vf_options` 触发。本节三条原则是让默认开启的 VF 融合**效果更好**的代码编排指导（通用，独立于 mix合图）；`sg_set_tunevf_mode` 控制 VF 调优 Pass 的行为模式（通用，独立于 mix合图）；`sg_set_ooo_scope` 是显式增强融合效果的开关旋钮（⛔ 仅对 mix 子图生效，须在 mix合图段内使用）；`vf_options` 是编译优化选项（A5 平台对所有含 Vector 计算的算子通用）。编排原则、`vf_options`、`sg_set_tunevf_mode` 三者可独立使用，`sg_set_ooo_scope` 须配合 mix合图。

### 5.1 适用场景

写连续 vec 算子序列时，遵循这三条原则可让编译器把更多 vec 指令合并到同一融合段，提升执行效率。典型场景：attention 类算子的 softmax 段、elementwise 连续计算段、reduce→expand 组合段等。

### 5.2 三条编排原则

#### 原则 1：相同 Shape OP（Reduce/expand 只看 dst Shape）

VF 融合把多条 vec 指令合并到同一循环流水线，要求工作 shape 一致。对会改变 shape 的 Reduce（缩小）或 expand（放大），判断"是否同 shape"**只看目标输出 dst shape**，不看输入——因为 dst 是融合循环的实际工作 shape。

- dst shape 一致的 OP 可并入同一融合段
- dst shape 不一致的 OP 无法并入同一融合段，会断开融合

#### 原则 2：区间约束（同区间或无 overlap）

融合段内所有 tensor 的内存区间关系只允许两种：

1. **完全重叠**（inplace 复用同一 buffer）
2. **完全不重叠**（独立 buffer）

**禁止部分重叠**——部分重叠会让编译器的读写时序/生命周期追踪混乱，破坏融合调度。

伪代码示例：

```
T1 = OP1(T0)
T2 = OP2(T3)
```

`T0 - T1 - T2 - T3` 这四个 tensor 的区间关系要么是同一块区间（inplace 复用），要么是没有 overlap 的独立区间，不允许部分重叠。

`set-flag` / `wait-flag` 是不携带数据的纯时序同步原语，不引入数据依赖，故可在融合段前后自由插入，不影响融合生效。

#### 原则 3：Reduce 尽量最后，expand 放到最前面

编排融合序列时：

- **expand（放大）放到最前面**：让后续 OP 都在大 shape 上运行
- **Reduce（缩小）尽量最后**：让前面 OP 都在大 shape 上融合

本质是"让尽可能多的 OP 落在相同的大 shape 上"，最大化融合收益。反过来若 Reduce 夹中间，会提前缩小 shape，割断后续 OP 的融合。

### 5.3 sg_set_ooo_scope 旋钮（vf 融合效果增强）

VF 融合在 A5 上默认开启，`sg_set_ooo_scope` 是显式增强默认开启 vf 融合效果的开关旋钮：把需要融合的连续 vec 算子指令包起来，让编译器不要被搬运或同步指令打断这些指令的融合调度。

**使用方式**（与 `sg_set_scope` 一致）：

```python
# 开始 vf 融合包裹
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_ooo_scope=1)  # 正整数标识一个 ooo scope

# ... vec 算子指令（需融合的连续 vector 计算）...

# 结束 vf 融合包裹
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_ooo_scope=-1)
```

**限制**：

- scope ID 为正整数，在同一个 kernel 内不重复使用即可（与 `sg_set_scope` 的 ID 体系独立）
- ⛔ **仅对 mix 子图生效**：必须在 `sg_set_scope` 包裹的 Mix合图段内使用，独立使用无效

**典型用法**：在 softmax 计算段用 `sg_set_ooo_scope=1/-1` 包裹，在 flash update 的 else 分支用 `sg_set_ooo_scope=2/-1` 包裹。

**参考资料**

- [sg_set_ooo_scope 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_pass_options.md)

### 5.4 sg_set_tunevf_mode 旋钮（VF 调优 Pass 行为模式）

`sg_set_tunevf_mode` 控制 VF 调优 Pass 的行为模式，决定在 OoO Pass 输出的 op 序列基础上是否调整 op 顺序、以及调整时偏向流水还是融合。A5 平台对所有含 Vector 计算的算子通用，独立于 mix合图。

**取值说明**（默认值 `0`，取值范围 `{0, 1, 2}`）：

| 取值  | 模式             | 行为                                                                                             |
| ----- | ---------------- | ------------------------------------------------------------------------------------------------ |
| `0` | 均衡模式         | 在 OoO Pass 输出的 op 序列基础上自动调整 op 顺序，自动平衡 Pipeline 流水与 VF 融合的整体性能收益 |
| `1` | 指令流水优先模式 | 不改变 OoO 排好的 op 执行序                                                                      |
| `2` | vf 融合优先模式  | 不考虑性能建模的收益评估，尽量调整 op 顺序以保证更大范围的 VF 融合                               |

**使用方式**：

```python
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_pass_options(sg_set_tunevf_mode=2)  # vf 融合优先
```

**调优建议**：

- 默认 `0`（均衡）已对多数场景较优，先以默认值采集基线
- 若编排原则（[S-17]）已就位但融合收益仍未达预期，尝试 `2`（vf 融合优先）扩大融合范围
- 若融合优先导致流水退化（AICore E2E Time 上升），回退 `1`（流水优先）保留 OoO 执行序
- 须以 AICore E2E Time 不恶化为前提，每次只改一个取值并实测

### 5.5 配合其他层面的建议

- 若同时使用 `vf_options`（[S-16]）：编排原则让融合候选更多，`vf_options` 让融合生成代码更优，两者互补可叠加收益
- 若同时使用 `sg_set_tunevf_mode`（[S-18]）：编排原则（[S-17]）扩大融合候选，`sg_set_tunevf_mode` 控制 Pass 调整 op 序列的激进程度，两者协同——编排原则就位后再用 `sg_set_tunevf_mode=2` 收益更明显
- 若同时使用 mix合图（[S-14]）+ `sg_set_ooo_scope`（[§5.3](#53-sg_set_ooo_scope-旋钮vf-融合效果增强)）：mix合图段内的 vec 子图同样适用本节编排原则，编排合理时 `sg_set_ooo_scope` 包裹的融合段效果更好
- ⚠️ 多项叠加（编排原则 + `vf_options` + `sg_set_tunevf_mode` + nbuffer=1 + 全 unroll + Mix合图）时编译时间会显著增长，详见 [§4.2.5](#425-调优注意事项非限制条件但影响性能选型)
