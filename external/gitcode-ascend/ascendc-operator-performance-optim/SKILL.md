---
name: external-gitcode-ascend-ascendc-operator-performance-optim
description: 排查并优化 Ascend C 算子性能。当用户开发、审查或优化 Ascend C kernel 算子时使用，或当用户提及 Ascend C
  性能优化、算子优化、tiling、流水、搬运、 内存优化、NPU/昇腾等关键词时触发。
original-name: ascendc-operator-performance-optim
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---

# Ascend C 算子性能优化（排查 → 修改 → 验证 闭环）

本 skill 不仅排查性能问题，还负责 **修改代码并验证优化效果**。完整流程为：

```
Phase 1: 排查 — 审查代码 + 学习设计文档，发现优化点
Phase 2: 基线 — 保存当前性能测试结果（自定义算子 vs 标杆）
Phase 3: 优化 — 学习 code-gen 知识后修改算子代码
Phase 4: 精度 — 精度验证（确保优化后功能正确）
Phase 5: 性能 — 同 case 性能对比（优化后 vs 标杆）
Phase 6: 迭代 — 未提升则继续优化，最多 3 轮
```

---

## Phase 1: 排查 — 发现优化点

### 1.1 学习算子设计文档

**MANDATORY — 排查前必须先理解算子设计**：

1. 读取 `ascend-kernel/csrc/ops/<op_name>/design.md`（若存在），提取：
   - 算子类型（elementwise / 行处理 / Cube）
   - Tiling 策略（核间切分 / 核内切分）
   - UB 空间分配方案
   - 计算逻辑与数据流
2. 读取 `op_host/<op_name>.cpp` 和 `op_kernel/<op_name>.cpp` 全部源码

### 1.2 逐阶段排查

按以下顺序逐阶段审查算子代码。对每个阶段，加载对应的 reference 文件，逐项
对照代码检查。

```
- [ ] 1. Tiling    — 数据在多核与 L2Cache 间的切分策略
- [ ] 2. 搬运      — DataCopy 的带宽利用率
- [ ] 3. API 使用  — Ascend C API 的高效用法
- [ ] 4. 内存      — 数据在存储层级中的放置策略
- [ ] 5. 流水      — CopyIn / Compute / CopyOut 的重叠执行
```

每个阶段有独立的 reference 文件，排查时**仅加载当前阶段**的文件：

- 阶段 1：[references/tiling-prof.md](references/tiling-prof.md)
- 阶段 2：[references/data-copy-prof.md](references/data-copy-prof.md)
- 阶段 3：[references/api-usage-prof.md](references/api-usage-prof.md)
- 阶段 4：[references/memory-prof.md](references/memory-prof.md)
- 阶段 5：[references/pipeline-prof.md](references/pipeline-prof.md)

#### 1. Tiling

> 详细示例：[references/tiling-prof.md](references/tiling-prof.md)

排查项：

- [ ] **1.1 多核切分**：`blockDim` 是否设为硬件核数？
  - 耦合架构：`GetCoreNumAiv()` 或 `GetCoreNumAic()`
  - 分离架构 Vector 算子：AIV 核数（如 40）
  - 分离架构 Cube 算子：AIC 核数（如 20）
  - 分离架构 MIX 算子：物理核组数（如 20 = 40 AIV / 2），不可超过物理核数
- [ ] **1.2 L2Cache 切分**：当 `输入 + 输出 > L2Cache 容量` 时，是否将数据
  按 L2Cache 大小分块，所有核协同处理同一块后再切换下一块？
- [ ] **1.3 核间负载均衡**：L2Cache 切分后，尾块是否在各 pass 间交替分配，
  避免固定某些核始终拖尾？

#### 2. 搬运

> 详细示例：[references/data-copy-prof.md](references/data-copy-prof.md)

排查项：

- [ ] **2.1 单次搬运量 >= 16 KB**：每次 `DataCopy` 是否搬运至少 16 KB？
  小于此值带宽利用率显著下降。
- [ ] **2.2 GM 地址 512B 对齐**：GM 起始地址是否 512 字节对齐？
  （Atlas A2 系列上，32B 对齐比 512B 对齐带宽最多低 30%。）
- [ ] **2.3 stride 参数代替 for 循环**：间隔搬运是否使用 `DataCopyParams`
  （blockCount/blockLen/srcStride/dstStride）一次下发，而非用 for 循环逐行搬运？

#### 3. API 使用

> 详细示例：[references/api-usage-prof.md](references/api-usage-prof.md)

排查项：

- [ ] **3.1 TPipe 在 kernel 类外创建**：`TPipe` 是否在 kernel 入口函数中创建
  并以指针传入类？（类内 TPipe 会阻止 Scalar 常量折叠，增加约 17% scalar_time。）
- [ ] **3.2 纯搬运算子使用 TQueBind**：无 Vector 计算的算子是否用
  `TQueBind<VECIN, VECOUT>` 替代了分离的 `TQue<VECIN>` + `TQue<VECOUT>`？
  （消除冗余的 LocalTensor 间 DataCopy，`aiv_vec_time` 降至约 0。）
- [ ] **3.3 Counter 模式（SetMaskCount）**：Vector 指令是否使用 Counter 模式，
  而非 Normal 模式手动计算主块/尾块 mask？
- [ ] **3.4 Matmul AtomicAdd**：Matmul 结果 C 需要与 GM 矩阵 D 相加时，
  是否在 `IterateAll`/`GetTensorC` 中设置 `enAtomic=1` 融合累加？
  （可减少约 12% cycle。）
- [ ] **3.5 归约指令组合**：连续 buffer 归约到标量时，是否使用
  `BlockReduceSum` + `WholeReduceSum` 组合，而非多次相同归约指令？

#### 4. 内存

> 详细示例：[references/memory-prof.md](references/memory-prof.md)

排查项：

- [ ] **4.1 UB Buffer 融合**：连续 Vector 运算（如 Exp → Abs）的中间结果
  是否留在 UB 内，而非经 GM 往返？
- [ ] **4.2 L0C 累加矩阵乘**：`A1*B1 + A2*B2 + ...` 场景下，Mmad 结果是否
  在 CO1（L0C）中原地累加，而非逐次写 GM 再在 UB 求和？
- [ ] **4.3 小矩阵长驻 L1**：当 L1 无法同时容纳左右矩阵时，较小矩阵是否
  一次加载后常驻 L1，仅循环搬运较大矩阵？
- [ ] **4.4 BT Buffer 存放 bias**（分离架构）：bias 是否存入 BT Buffer（C2）
  并通过 `Mmad` 一步融合，而非在 UB 中单独做 Add？
- [ ] **4.5 FP Buffer 存放量化参数**（分离架构）：量化参数是否存入
  FP Buffer（C2PIPE2GM）并通过 `Fixpipe` 随路量化，而非在 UB 中单独计算？

#### 5. 流水

> 详细示例：[references/pipeline-prof.md](references/pipeline-prof.md)

排查项：

- [ ] **5.1 CopyIn/Compute/CopyOut 范式**：算子是否划分为三级流水，
  使用 `TQue` 进行级间同步？
- [ ] **5.2 Double Buffer**：`InitBuffer` 的 buffer 个数是否设为 2，
  使 CopyIn/CopyOut 与 Compute 重叠执行？
  （前提：循环次数 >= 2，且搬运时间相对计算时间不可忽略。）
- [ ] **5.3 异步 Iterate（MIX 模式）**：Matmul MIX 场景下，是否使用
  `Iterate<false>()`/`IterateAll<false>()` 避免每次迭代的 AIC/AIV 同步开销？

### 1.3 输出排查报告

排查完所有阶段后，按以下格式输出汇总：

```
## 优化排查报告

### 发现的问题（按预期收益排序）
1. [阶段 X.Y] <问题描述> — <预期收益>
2. [阶段 X.Y] <问题描述> — <预期收益>
...

### 已确认无问题
- [阶段 X.Y] <检查项描述>
...

### 优化计划
按预期收益从大到小排列，确定本轮优化的目标项。
```

---

## Phase 2: 基线 — 保存当前性能测试结果

优化前必须保存性能基线，以便优化后精确对比。

### 2.1 检查现有性能验证结果

检查 `csrc/ops/<op_name>/test/` 下是否已存在：
- `<op_name>_perf_cases.jsonl` — 性能测试用例
- `<op_name>_torch_npu_profiler_report.md` — 性能对比报告

### 2.2 无结果时执行性能评估

若上述文件不存在或结果已过时（如代码已更新但报告未重新生成），**MUST** 调用
**`ascendc-operator-performance-eval`** skill 完成完整性能评估：

1. 读取 `ascendc-operator-performance-eval` SKILL.md
2. 按其流程生成性能用例（JSONL）、运行 profiler、生成对比报告
3. 确保报告包含**自定义算子 vs 标杆**的完整对比数据

### 2.3 保存基线快照

将当前性能报告备份为基线文件，命名为 `<op_name>_baseline_report.md`，
保存在同一 `test/` 目录下。该文件后续用于对比优化效果。

```
csrc/ops/<op_name>/test/
├── <op_name>_perf_cases.jsonl                 ← 性能用例（优化前后共用）
├── <op_name>_torch_npu_profiler_report.md     ← 当前报告（会被覆盖）
└── <op_name>_baseline_report.md               ← 基线快照（优化前的性能数据）
```

---

## Phase 3: 优化 — 学习知识后修改代码

### 3.1 学习算子开发知识（MANDATORY）

**修改代码前 MUST 加载 `ascendc-operator-code-gen` skill 的 reference 文件**，
确保对 AscendC API、数据搬运、同步控制等有准确理解。

按需加载以下 reference（位于 `ascendc-operator-code-gen/references/`）：

| Reference 文件 | 用途 |
|---------------|------|
| `GUIDE.md` | 总览：模板选择、代码生成流程 |
| `data-copy-api.md` | DataCopy/DataCopyPad API 详解 |
| `vector-compute-api.md` | Vector 计算 API 详解 |
| `sync-control-api.md` | TQue/Pipe 同步控制 |
| `resource-management-api.md` | TPipe/TBuf 资源管理 |
| `basic-data-structures-api.md` | LocalTensor/GlobalTensor 等基础结构 |
| `kernel-constraints.md` | Kernel 编程约束与常见陷阱 |

根据 Phase 1 发现的优化点，选择性加载相关 reference。例如：
- 优化搬运 → 加载 `data-copy-api.md`
- 优化流水 → 加载 `sync-control-api.md` + `resource-management-api.md`
- 优化计算 → 加载 `vector-compute-api.md`

### 3.2 制定修改方案

针对 Phase 1 排查报告中的每个优化点，制定具体的代码修改方案：

```
优化点 [X.Y]: <问题描述>
├── 修改文件: op_host / op_kernel / 两者
├── 修改内容: <具体代码变更描述>
├── 预期效果: <量化预期（如搬运时间减少 30%）>
└── 风险评估: <是否可能影响精度/是否需要修改 tiling>
```

### 3.3 执行代码修改

按照修改方案逐一修改代码。修改时遵守以下规则：

**MUST 遵守 code-gen 反模式清单**：
- **NEVER** 让 FP16/BF16 直接参与复杂数学计算，必须先 Cast 到 FP32
- **NEVER** 在 EXEC_KERNEL_CMD 中传右值
- **NEVER** 对 GM↔UB 搬运使用 DataCopy，必须用 DataCopyPad
- **NEVER** 在 ReduceSum/ReduceMax 后直接复用源 tensor
- **NEVER** 在 kernel 中使用 `std::min/max/abs/sqrt/exp` 等标准库函数
- **NEVER** 向高维切分 API 传入 repeatTime > 255
- **NEVER** 修改 `cmake/` 或 `csrc/utils/` 下的文件
- **NEVER** 硬编码核数或 UB 大小

### 3.4 编译安装

修改完成后必须重新编译安装：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cd task/ascend-kernel
bash build.sh
pip install output/ascend_kernel*.whl --force-reinstall --no-deps
```

编译失败时进入排错循环（最多 3 次）。

---

## Phase 4: 精度验证 — 确保优化后功能正确

**MANDATORY — 优化后必须先通过精度验证再进行性能对比。**

### 4.1 调用精度评估 skill

读取并执行 **`ascendc-operator-precision-eval`** SKILL.md 的完整流程：

1. 生成精度测试用例（≥30 例，覆盖全部 dtype）
2. 运行 pytest 精度测试
3. 生成精度报告（Markdown + JSON）
4. **在当前对话中展示**总览、失败摘要与关键发现

### 4.2 精度判定

| 结果 | 处理 |
|------|------|
| **全部通过** | 进入 Phase 5 性能验证 |
| **部分失败** | 分析失败原因，回退或修复代码，重新进入 Phase 3 |
| **大量失败** | 回退本轮所有修改，重新分析优化方案 |

---

## Phase 5: 性能验证 — 确认优化效果

### 5.1 运行同 case 性能测试

使用 **Phase 2 中相同的性能用例**（`<op_name>_perf_cases.jsonl`），调用
**`ascendc-operator-performance-eval`** skill 重新执行性能评估。

关键要求：
- **MUST** 使用与基线完全相同的 perf_cases.jsonl（不能增删用例）
- **MUST** 生成新的 `<op_name>_torch_npu_profiler_report.md`
- **MUST** 在当前对话中展示对比表、汇总与结论

### 5.2 对比分析

将优化后的性能数据与 Phase 2 保存的基线进行对比：

```
## 优化效果对比

| Case | Shape | dtype | 基线 per-step(us) | 优化后 per-step(us) | 提升比 | 标杆 per-step(us) | vs 标杆 |
|------|-------|-------|-------------------|--------------------|---------|--------------------|---------|
| ...  | ...   | ...   | ...               | ...                | ...     | ...                | ...     |

### 汇总
- 平均提升: X%
- 最大提升: X%（Case Y）
- vs 标杆平均比值: 优化前 A → 优化后 B
```

### 5.3 性能判定

| 结果 | 处理 |
|------|------|
| **性能提升**（大部分 case 优化后更快） | 优化成功，输出最终报告 |
| **性能未提升或回退** | 进入 Phase 6 迭代优化 |

---

## Phase 6: 迭代优化（最多 3 轮）

若 Phase 5 判定性能未提升，进入迭代：

```
当前轮次: N (N ∈ {1, 2, 3})

├── N < 3: 回到 Phase 1，选择下一优先级优化点或调整方案
│   ├── 重新排查，分析上一轮修改为何未生效
│   ├── 选择新的优化点或调整上一轮的方案
│   └── 重复 Phase 3 → Phase 4 → Phase 5
│
└── N = 3: 停止迭代，输出最终报告（含所有轮次记录）
```

### 迭代记录

每轮迭代必须记录：

```
### 第 N 轮优化
- 优化目标: [阶段 X.Y] <描述>
- 修改内容: <代码变更摘要>
- 精度结果: 通过 / 失败
- 性能结果: 提升 X% / 未提升 / 回退 Y%
- 决策: 保留本轮修改 / 回退 / 继续下一轮
```

---

## 最终输出

所有轮次完成后（成功提升或达到 3 轮上限），输出最终汇总报告。

### 在当前对话中展示（MANDATORY）

**MUST** 在对话中展示以下内容，**NEVER** 仅输出文件路径：

1. **优化排查总结**：发现的所有问题及处理状态
2. **性能对比总表**：基线 → 优化后 → 标杆的三方对比
3. **迭代历史摘要**：每轮的优化目标、结果、决策
4. **≥3 条关键结论**：主要瓶颈、优化收益分布、剩余优化空间等
5. **文件路径殿后**：报告与代码文件路径

### 文件产物

```
csrc/ops/<op_name>/test/
├── <op_name>_perf_cases.jsonl                 ← 性能用例
├── <op_name>_baseline_report.md               ← 优化前基线
├── <op_name>_torch_npu_profiler_report.md     ← 优化后最终性能报告
├── <op_name>_precision_report.md              ← 精度验证报告
└── <op_name>_optim_summary.md                 ← 优化迭代汇总报告（新增）

csrc/ops/<op_name>/
├── op_host/<op_name>.cpp                      ← 优化后的 host 代码
└── op_kernel/<op_name>.cpp                    ← 优化后的 kernel 代码
```

### 优化迭代汇总报告结构

`<op_name>_optim_summary.md` 必须包含：

```markdown
# <op_name> 性能优化报告

## 排查发现
（Phase 1 的排查报告内容）

## 优化前基线
（Phase 2 的性能数据摘要）

## 迭代历史

### 第 1 轮
- 优化目标: ...
- 代码修改: ...
- 精度结果: ...
- 性能结果: ...

### 第 N 轮
...

## 最终性能对比
（优化前 vs 优化后 vs 标杆 三方对比表）

## 结论
（≥3 条关键发现）
```

---

## 检查清单（助手自检）

### Phase 1: 排查
- [ ] 已读取算子设计文档（design.md）
- [ ] 已读取 op_host + op_kernel 完整源码
- [ ] 已逐阶段加载 reference 并逐项排查
- [ ] 已输出排查报告，优化点按预期收益排序

### Phase 2: 基线
- [ ] 已确认或生成性能测试用例（JSONL）
- [ ] 已确认或运行性能评估（自定义 vs 标杆）
- [ ] 已保存基线快照（`_baseline_report.md`）

### Phase 3: 优化
- [ ] 已加载 code-gen reference（修改前必读）
- [ ] 代码修改遵守反模式清单
- [ ] 编译安装成功

### Phase 4: 精度
- [ ] 已按 `ascendc-operator-precision-eval` 流程完成精度验证
- [ ] 精度验证通过（全部或大部分用例 PASS）

### Phase 5: 性能
- [ ] 使用与基线相同的 perf_cases.jsonl
- [ ] 已在对话中展示性能对比数据
- [ ] 已判定是否提升

### Phase 6: 迭代
- [ ] 迭代不超过 3 轮
- [ ] 每轮均有记录（目标、修改、精度、性能、决策）
- [ ] 已输出最终汇总报告（`_optim_summary.md`）

### 输出
- [ ] **已在当前对话中展示**排查总结、性能对比、迭代历史、≥3 条结论
- [ ] **NEVER** 仅输出文件路径
