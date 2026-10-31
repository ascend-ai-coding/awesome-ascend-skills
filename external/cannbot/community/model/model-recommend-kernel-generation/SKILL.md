---
name: external-cannbot-community-model-model-recommend-kernel-generation
description: AscendC Kernel 生成（工作流条目 11）。为融合规划选定的候选生成可在本机（Ascend 910B 系列）运行的标准 AscendC
  kernel：msopgen 工程、Matmul 高阶 API、精度/性能验证、GE 集成。触发：kernel 生成、AscendC 开发、自定义算子、算子性能优化、Matmul
  高阶 API、算子工程、kernel 对接 GE。
original-name: model-recommend-kernel-generation
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# AscendC Kernel 生成

## Prerequisites

- [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) 选定的融合候选（算法规格：数学约束、输入输出 dtype/shape 范围、bound 判定、理论收益区间、来源标记）——**唯一合法入口，见约束项**
- 目标交付形态（独立算子工程 / 并入模型自定义算子包）
- 硬件规格获取原则见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) Step 2（`PlatformAscendC` 运行时获取，禁止硬编码）

## 目标

为融合候选生成可在本机（Ascend 910B 系列）运行的**标准 AscendC** kernel，并完成精度/性能验证。**每个 kernel 必须带测试与 benchmark**（实现 → 注册 → 测试 → benchmark → 构建 → 验证）。

> 当前环境不支持 cannbot-dsl，本 skill 全程使用标准 AscendC（msopgen 自定义算子工程）；950 的 Blaze/tensor_api 路径不适用本机。

## 输入 / 输出契约

**输入**：条目 10 融合规划排序清单中的开发项；目标交付形态。

**输出**：AscendC 算子工程 + 精度测试用例 + benchmark（vs 原算子串加速比报告，对照条目 10 理论收益区间复核）+ GE 集成产物。

## 约束项：开发入口门禁

- **只接受 [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) 排序清单中的开发项**：收到"直接开发某算子"的请求（含来自适配失败项、热点分析结论、用户口头指定的），必须先回流条目 10 完成 bound 判定/收益估计/排序，凭清单条目开发
- 开发项须携带规划阶段给出的：理论收益区间（验收时对照实测）、bound 判定（性能目标推导依据）、来源标记（hot-chain / adaptation-failure）
- **例外（可不经条目 10）**：纯 bug 修复、既有算子的调用修正、精度问题修复——属条目 7/4 范畴；图跑不起来的功能补齐属 [custom-op-development](../model-recommend-custom-op-development/SKILL.md)

## 工作流步骤

### Step 1: 需求理解 → 设计文档

API 映射（算子名/输入输出/属性）、精度路由（dtype 分支）、Tiling 推导（多 shape 档位）、循环结构、数据流设计；输出 `design.md`（强制交付件，放 `docs/{model_name}/kernel-generation/{op_name}/`）。

### Step 2: 路径决策（910B，标准 AscendC）

| 算子类型 | 路径 | 备注 |
|---------|------|------|
| 纯 Vector 逐元素/归约/查表 | AscendC 常规 kernel（Vector 侧） | 从已验证工程模板出发改造，不裸写 |
| Matmul 族（GEMM/BMM/matmul+bias） | **Matmul 高阶 API**（AscendC lib/matmul 接口，Cube 侧） | 910B 支持高阶 API；Blaze/tensor_api 为 950 专属路径，本机不可用 |
| Cube+Vector 融合 | AIC kernel（Matmul）+ epilogue 或 AIC/AIV 协同 | 按标准 AscendC 编程模型组织 |
| 多段融合（查表+计算） | 单 kernel 内分段（DataCopy 与计算重叠） | 访存优化联动 [memory-access-optimize](../model-recommend-memory-access-optimize/SKILL.md) |

### Step 3: 工程实现

1. `msopgen` 生成自定义算子工程骨架（`--soc-version` 以本机为准，Ascend 910B 系列；项目结构含 op_host（shape/tiling 推导）/ op_kernel（AscendC 实现）/ 测试目录）
2. 编码实现：kernel 逻辑 + host 侧 infer shape/dtype + tiling 策略
3. 工程编译（产出自定义算子 run 包）

### Step 4: 精度验证

- golden 真源：numpy / CPU onnxruntime 参考实现（与算子数学定义逐行对齐）
- 判定标准按 dtype：浮点 `allclose`（atol/rtol 按 golden 与 dtype 定，默认量级 `1e-3`；fp16 场景放宽须说明）；整型/bool 精确相等
- 尺寸覆盖：batch 1/2/N、动态维边界、典型 shape 档位（对照模型实际用到的 shape 集合）
- 全量通过后才进性能阶段

### Step 5: 性能测试

- msprof 采集 vs 原算子串（条目 10 候选链路的原始 kernel 序列）标杆对比
- 输出加速比报告（geomean/mean/min/max，按 shape 档位分列）
- **对照条目 10 理论收益区间复核**：超出区间（好/坏）均须归因

### Step 6: 交付集成（双门禁终验，强制）

按 [custom-op-development](../model-recommend-custom-op-development/SKILL.md) 的 GE 接入路径完成部署与图接入（自定义算子包 + onnx 节点 + 模型环境声明）；集成后必须过双门禁才算交付：

1. **精度门禁**：整图串行 `run.sh <id> --verify` 全 pass（算子级用例通过不足以替代模型级验证）
2. **性能门禁**：profiling 确认融合 kernel 替换了原算子串（真实生效）+ **整图端到端主指标不劣化超过噪声带**——算子级加速比必须以端到端复核（融合 kernel 引入的调度/布局变化可能抵消算子收益；端到端回退时归因并回退或重设计）
3. **落账**：写 perf_ledger（算子级 + 端到端两级数据）

## 昇腾 NPU 专属要点（Tiling 与约束）

- **硬件参数运行时经 `PlatformAscendC` 获取，禁止硬编码**（核心数/UB/L1/L0A/L0B/L0C 等）
- 910B 无 950 的 CV 直通通路/BufferID 同步/CCU 通算等新编程模型特性，按 910B 标准 AscendC 编程模型设计同步（set/wait）与数据流
- `msopgen` 生成的 op_host 侧 tiling 推导须覆盖模型实际 shape 档位；动态 shape 场景评估 tiling 退化风险
- **调试保障**：精度问题走 DumpTensor 分段定位（输入/中间/输出逐段比对）；卡死/崩溃/内存越界走 plog 解析 + memcheck，转 [runtime-debug](../model-recommend-runtime-debug/SKILL.md)

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **生成的算子** → `ops/{op_name}/`：AscendC 工程源码（op_host/op_kernel/测试）、golden 参考实现、benchmark、编译产物说明（按算子名组织，与模型无关）
- **文档类交付件** → `docs/{model_name}/kernel-generation/{op_name}/`：设计文档（design.md）、精度/性能验收报告
- **中间产物** → `tmp/{model_name}/kernel-generation/{op_name}/`：编译产物（build 目录、.o）、DumpTensor 中间数据、msprof 原始 trace
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 精度用例全过（atol/rtol 达标，尺寸覆盖齐全）
- 算子级性能对比报告产出（vs 原算子串加速比 + geomean/mean/min/max），与条目 10 理论区间对照有归因
- **双门禁通过**：整图 verify 全 pass；整图端到端主指标不劣化超过噪声带（算子级提升已用端到端复核）；profiling 确认融合 kernel 替换了原算子串；已写 perf_ledger
- design.md 与测试/benchmark 交付齐全

## 依据

【实测】本机 `msopgen`/`atc`/`msprof` 可用（CANN 9.1.0）；cannbot-dsl 不可用（标准 AscendC 唯一路径）。
【框架】CANN 自带 AscendC 文档（编程模型、Matmul 高阶 API、msopgen 工程结构、DumpTensor）。
【技能库经验】kernel 必须带测试与 benchmark、模板改造不裸写、唯一入口纪律。
