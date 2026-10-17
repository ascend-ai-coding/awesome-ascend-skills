# Ascend C 算子 Tiling 通用设计方法论

> 当 spec.yaml 的 `op.paradigms` 在本 skill 的 `references/paradigm-refs.yaml` 中
> 没有匹配的范式路由时，使用本文档作为 Tiling 设计的兜底方法论。命中具体范式时，
> 优先按对应的 `references/<paradigm>/patterns.md` 进行设计；本文档仅作为通用基线。

---

## 一、通用设计要素（所有算子类别必须完成）

### 1. 多核切分策略

**核心问题**：任务如何分配给多个 AI Core？

**设计要点**：
- **负载均衡**：每个核处理的任务量尽量相等；不均匀负载会让快核空等慢核。
- **数据局部性**：相邻数据尽量分配给同一核，减少跨核同步与重复搬运。
- **粒度适中**：tile 不能太小（调度开销大），也不能太大（并行度低）。
- **尾块处理**：总任务量不被核数整除时，需要明确尾块归属（最后一个核多吃、轮转分配等）。

**输出清单**：
- [ ] 总任务切分方式（按哪个维度切、切多少份）
- [ ] 每个 AI Core 处理的任务量（含尾块策略）
- [ ] 使用的 AI Core 数量（与目标芯片的 vector / cube 核数对齐）

### 2. UB 切分策略

**核心问题**：单次能在 UB 中处理多少数据？

**设计要点**：
- **UB 容量限制**：DAV_2201 ≈ 192 KB、DAV_3510 ≈ 248 KB。具体可用容量需扣除 stack、
  workspace、double buffer 等开销，按目标芯片实测预算编排。
- **单次处理数据量**：根据 UB 预算反推每次 `DataCopy` / 计算批量的元素数。
- **是否分 chunk**：单核任务量超出 UB 时需要切 chunk 串行处理；chunk 大小决定循环次数与
  pipeline 深度。
- **dtype 转换余量**：若 Compute 阶段需要升精度（fp16 → fp32），UB 预算要预留临时 buffer。

**输出清单**：
- [ ] 单次处理的数据量（元素数 + 字节数）
- [ ] 是否需要分 chunk、chunk 数量与剩余尾 chunk 处理
- [ ] chunk 大小计算公式（含 UB 预算、dtype 字节数、buffer 数量等变量）

### 3. Buffer 规划

**核心问题**：需要哪些 buffer？各多大？总 UB 占用是否超限？

**设计要点**：
- **输入 buffer（inQueue）**：每路输入一个 inQueue；多输入算子可共用 inQueue 或独立 inQueue。
- **输出 buffer（outQueue）**：每路输出一个 outQueue。
- **中间计算 buffer（tmpBuf / workBuf）**：保存中间结果、升精度中转、reduce 树等。
- **Double Buffer**：搬运/计算重叠优化，启用时 inQueue / outQueue 容量翻倍。
- **总和校验**：所有 buffer 字节数之和必须 ≤ 该芯片可用 UB（含 double buffer 倍数）。

**输出清单**：
- [ ] Buffer 列表及用途（名称 + 类型 + 容量 + 是否启用 double buffer）
- [ ] 各 buffer 大小计算公式（含 dtype、对齐、double buffer 倍数）
- [ ] 总 UB 使用量及与芯片预算的差额

### 4. 分支场景覆盖

**核心问题**：算子需要处理哪些不同场景？每种场景的策略分别是什么？

**常见分支维度**：
- **数据类型**：FP32 / FP16 / BF16 / INT8 / INT32 等；不同 dtype 的字节数、API 支持、
  累加器选择都可能不同。
- **Shape 大小**：大 shape（按通用 chunk 路径）、小 shape（fullload 路径），切分公式分支。
- **数据对齐**：32 字节对齐（DataCopy 直通）/ 非对齐（需要 padding 或非对齐搬运变体）。
- **边界情况**：reduce 轴长度为 1、空 tensor、rank=0 标量、最大/最小 shape、特殊数值
  （NaN / Inf / 极值）。
- **架构差异**：DAV_2201 / DAV_3510 等不同芯片的 UB 容量、API 可用性、并行度差异。

**输出清单**：
- [ ] 分支决策条件（按上述分支维度逐项列出）
- [ ] 各分支的处理策略（差异点 + 复用点）
- [ ] 边界测试用例（对应到 spec.yaml 的 `boundary_conditions` / `extreme_inputs`）

---

## 二、TilingKey 与 Tiling 字段

**核心问题**：上述分支决策如何在 host 侧编码为 `TilingKey`？哪些参数作为 Tiling 字段
随 TilingData 下发到 kernel？

**设计要点**：
- 每个独立分支路径分配唯一 TilingKey；TilingKey 的位域规划由架构师决定（通常把 dtype
  / shape 级别 / 对齐情况编码进高位）。
- Tiling 字段最小化原则：能在 kernel 内推导的不要放进 TilingData。
- Tiling 字段命名与单位（元素数 / 字节数 / 块数）必须在 DESIGN.md 与 `op_def.cpp` 中保持
  一致。

**输出清单**：
- [ ] TilingKey 取值集合及对应分支
- [ ] Tiling 字段表（名称 + 类型 + 单位 + 取值范围）
- [ ] Tiling 函数关键计算公式（多核切分、UB 切分、buffer 大小）
