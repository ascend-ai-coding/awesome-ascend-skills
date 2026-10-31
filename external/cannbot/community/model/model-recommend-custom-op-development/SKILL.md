---
name: external-cannbot-community-model-model-recommend-custom-op-development
description: 自定义算子开发与 GE 接入（工作流条目 12，功能补齐向）。补齐 GE parser 无法解析或无原生实现的算子：标准 AscendC（msopgen
  工程）实现 → 自定义算子包部署 → onnx 节点入图 → 精度/性能验证。触发：自定义算子、算子不支持、非标 domain、无原生算子、算子接入 GE、算子包部署、msopgen、算子语义对齐。
original-name: model-recommend-custom-op-development
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 自定义算子开发与 GE 接入

## 流水位置

承接 [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) 未适配项中的**功能补齐型**（GE 无原生算子 / 非标 domain 无法折叠 / 语义不匹配，图跑不起来或语义无法对齐）：先尝试 onnx 等价改写/现有算子组合（v1 验证语义），不满足再下沉标准 AscendC 自定义算子。**性能替换型必须走 [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) → [kernel-generation](../model-recommend-kernel-generation/SKILL.md)**。

> 当前环境不支持 cannbot-dsl：本 skill 全程使用标准 AscendC（msopgen 自定义算子工程），无 DSL 路径。

## Prerequisites

- [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) 已输出明确的未适配原因（无算子/硬约束/语义不匹配）
- `msopgen` 可用（CANN 9.1.0 自带）
- 目标算子的数学定义与参考实现来源明确（原框架代码/CPU 参考实现）

## 工作流步骤

### Step 1: 方案设计

1. **算法规格**：输入/输出 shape+dtype、计算公式、FLOPs 与访存量分析
2. **分解策略**：拆为 Cube 段（GEMM）与 Vector 段（elementwise/reduce/查表/搬运），识别融合机会
3. **输出设计文档 DESIGN.md**（强制交付件，`ops/{op_name}/DESIGN.md`）

### Step 2: 实现路径选择（先组合后下沉）

| 路径 | 适用 | 优势 | 劣势 |
|------|------|------|------|
| onnx 等价改写 / 现有算子组合（v1 原型） | 语义可由现有 onnx 算子组合表达 | 零编译开销、快速验证语义 | 多 kernel launch、无融合 |
| 标准 AscendC 自定义算子工程（v2 下沉） | 组合不满足性能 / 无对应算子 | 单 kernel、可控 tiling | 开发与调试成本高 |

**v1 → v2 渐进策略**：先组合验证语义与精度基线 → 性能不达标再下沉 AscendC（经条目 10 排序进条目 11 或在本条目完成，功能补齐型可在本条目直接下沉）。

### Step 3: AscendC 实现（标准路径）

1. `msopgen` 生成工程（`--soc-version` 以本机为准，Ascend 910B 系列）
2. op_host：infer shape/dtype、tiling 推导（与 onnx 节点语义严格对齐）
3. op_kernel：AscendC 实现（纯 Vector 常规路径；GEMM 段用 Matmul 高阶 API；硬件参数 `PlatformAscendC` 运行时获取，禁止硬编码）
4. 工程编译为自定义算子 run 包

### Step 4: GE 接入（关键链路）

1. **算子包部署**：run 包部署路径经 `ASCEND_CUSTOM_OPP_PATH` 环境变量或标准 OPP 目录注入
2. **模型环境声明**：部署与依赖配置写入 `models/M<id>/M<id>_env.sh`（`test.sh` 每次运行自动 source，模型间隔离）
3. **onnx 节点对齐**：onnx 图中对应节点（Custom domain / 特定 op_type）的 domain 与 op_type 命名、输入输出序/属性**必须与算子包注册信息严格一致**，否则 GE parser 解析失败（排查转 [runtime-debug](../model-recommend-runtime-debug/SKILL.md)）
4. **infer shape/dtype 校验**：`prepare_model.py` 能正确解析含自定义节点的 onnx（shape 推导不报错）；运行后 `output_shapes.json` 与预期一致

### Step 5: 精度验证（强制门禁）

1. **golden 真源**：CPU 参考实现（numpy / onnxruntime，与算子数学定义逐行对齐）；onnxruntime 不可用时（自定义 domain）用 numpy 重算
2. **测试用例构造**（pytest 驱动，强制交付件）：
   - 尺寸覆盖：batch 1/2/N、动态维边界、模型实际 shape 档位
   - 边界回归：空输入、极值、索引越界防护（如适用）
   - 判定标准：浮点 `allclose`（atol/rtol 按 dtype）；整型/bool 精确相等；逐元素 maxdiff 报告
3. **交叉验证**：独立 golden 脚本与 pytest 互相印证
4. **集成验证（精度门禁的模型级终验）**：模型级 `run.sh <id> --verify` 全 pass——算子级用例通过不足以替代模型级验证
5. 全量通过后才进性能阶段；任一不通过 → 回退并修复，禁止带病进入性能阶段

### Step 6: 性能验证（强制门禁）

1. **基准方法**：warmup ≥5 + timed ≥30，msprof/计时包夹，取中位数（μs）
2. **对照基准**：v1 组合实现 + GEMM-only/访存-only 估算（按 Roofline 折算）
3. **性能目标约束**：
   - **底线**：不高于 v1 等价算子串总耗时，即 `speedup = v1_time / custom_time ≥ 1.0`（下沉的目的至少是消除 launch 开销与中间张量物化）
   - **达标**：按 Roofline 判定——compute bound：有效 TFLOPS / 本机实测峰值 ≥ **0.8**；memory bound：有效带宽 / 本机实测带宽 ≥ **0.8**（有效值 = 实际 FLOPs/字节数 ÷ 实测耗时；峰值取 `data/env-baseline/` 锚定值）
   - **性能报告格式**（强制交付）：

     ```
     | 指标 | 值 | 本机实测峰值 | 利用率 | 判定 |
     |------|-----|---------|--------|------|
     | 有效 TFLOPS | xxx | xxx | xx% | compute bound 达标/未达标 |
     | 有效带宽 (GB/s) | xxx | xxx | xx% | memory bound 达标/未达标 |
     | vs v1 组合 | 加速比 | — | — | 底线达标/未达标 |
     ```

   - 未达标必须输出**瓶颈分析**（哪一段是瓶颈、差多少、改进方向），禁止标记"已完成"

### Step 6.5: 整图双门禁终验（强制）

算子级门禁通过后，在整图中完成双门禁：

1. **精度门禁**：含自定义算子的模型串行 `run.sh <id> --verify` 全 pass
2. **性能门禁**：整图端到端主指标不劣化超过噪声带（vs 改动前配置同口径对比——功能补齐型无"改动前可跑"基线时，对照 v1 组合实现的整图数据）
3. **落账**：写 perf_ledger（算子级 + 整图两级数据）；任一不通过 → 修复或回退

### Step 7: 交付件清单（强制）

| 交付件 | 路径 | 必须 |
|--------|------|------|
| 算子工程源码 | `ops/{op_name}/`（op_host / op_kernel / 编译配置） | ✅ |
| 设计文档 | `ops/{op_name}/DESIGN.md` | ✅ |
| 精度测试用例 | `ops/{op_name}/test_{op_name}.py`（pytest） | ✅ |
| 性能基准 | `ops/{op_name}/bench_{op_name}.py` | 推荐 |
| GE 接入说明 | `ops/{op_name}/INTEGRATION.md`（部署路径、M<id>_env.sh 配置、onnx 节点对齐说明） | ✅ |

## 昇腾 NPU 专属要点

- **soc_version 以本机为准**（Ascend 910B 系列，`msopgen --soc-version` 参数取本机支持值）
- onnx 节点 domain/op_type 与算子包注册信息不一致是接入失败首因；命名对齐后在 INTEGRATION.md 中固化映射关系
- bool dtype 输入注意：NPU 部分 bool 场景无 engine，参考「bool→int32 无损 Cast」思路评估是否在算子内以 int32 处理更稳
- 索引类输入 int64 语义与 host 侧 tiling 推导对齐；动态 shape 下 tiling 退化风险须评估
- 算子包部署路径变更后必须更新 `M<id>_env.sh` 并重新验证（环境隔离由框架保证）

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **算子工程 + 设计文档 + 测试 + GE 接入说明** → `ops/{op_name}/`（长期保留）
- **文档类交付件** → `docs/{model_name}/custom-op-development/`：方案与验收报告
- **profiling 归档** → `prof/{model_name}/custom-ops/{op_name}/`
- **中间产物** → `tmp/{model_name}/custom-op-development/`：编译产物、调试日志、试验性 onnx
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- **精度**：全部测试用例 PASSED（浮点 allclose 达标 / 整型精确）+ 模型级 verify 全 pass
- **测试覆盖**：batch/动态维/边界，最少 5 用例
- **性能（强制门禁）**：底线（vs v1 组合 ≥1.0）+ 达标（compute/memory bound 利用率 ≥0.8 至少一项）；未达标附瓶颈分析，禁止标记完成
- **整图双门禁通过**：整图 verify 全 pass 且端到端主指标不劣化（vs 改动前或 v1 组合口径，同口径对比）；已写 perf_ledger
- **交付件**：工程源码 + DESIGN.md + test_*.py + INTEGRATION.md 齐全
- **GE 接入**：含自定义算子的模型完整跑通，`M<id>_env.sh` 配置可复现

## 依据

【实测】msopgen 可用、cannbot-dsl 不可用（标准 AscendC 唯一路径）；本机算力/带宽锚定数据。
【框架】`M<id>_env.sh` 环境注入机制、onnx 解析失败排查项（§12）、verify 口径。
【框架】CANN 自带 AscendC 文档与 msopgen 工程模板（msopgen 工程结构、部署路径、注册信息）。
【技能库经验】先组合后下沉、性能门禁（底线 + Roofline 达标 + 瓶颈分析）、测试与 benchmark 强制。
