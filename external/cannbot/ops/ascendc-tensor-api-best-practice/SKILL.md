---
name: external-cannbot-ops-ascendc-tensor-api-best-practice
description: 在 Ascend 950 / DAV_3510 上以 Tensor API 为主、C API 为辅开发用户明确指定 Tensor API
  的算子时使用。覆盖路线边界、事实调查、DESIGN/PLAN 合同、Layout 与内存预算、构建、精度、性能、验收和排障。不适用于 Matmul 类 Blaze
  覆盖场景或 Attention 原生 AscendC 路线。
original-name: ascendc-tensor-api-best-practice
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# AscendC Tensor API 算子开发指南

## 定位与路径合同

本 Skill 是 Ascend C 直调工作流中的 Tensor API 实现路线权威源，不是独立工作流，也不新增 PM、Agent、workflow 模板或 harness 状态。用户明确要求 Tensor API 时，需求清单记录用户原话与路线来源，并固定 implementation_route=tensor_api_custom；该路线为硬约束，不由 Blaze 优先选型改写。

后续只验证该路线是否可行；不得静默切换到 Blaze 或原生 AscendC。若 API、ABI、Layout、内存或同步无法闭合，输出 unsupported 与证据，由用户决定下一步。

路径约定：

    project_root: <project-root>
    operator_root: <project-root>/operators/<operator_name>/
    design_path: operators/<operator_name>/docs/DESIGN.md
    plan_path: operators/<operator_name>/docs/PLAN.md
    execution_record_path: operators/<operator_name>/docs/EXECUTION_RECORD.md

本地手册或官方样例缺失时运行 [ensure_docs.py](scripts/ensure_docs.py)：主工作区 `~/asc-devkit` 在 master 上时自动 clone 或 fast-forward；若被其他分支占用（如 vendor 头锁定的 feature 分支）或与 master 历史分叉，脚本自动维护独立的 master 浅镜像 `~/asc-devkit-master` 提供手册与样例，不修改主工作区。输出 JSON 的 `selected_root` 即当前事实源根目录。仓库 dirty 或 origin 不匹配时阻断，不强制覆盖。

## 高频排障结论速查

以下三类联调/评测高频问题的既有结论，先记结论、细节再查对应 reference。

### bf16 Cube matmul 精度超标怎么定位

先看误差量级区分两类问题：实现 bug 的 MERE 通常是 1e-1~1e1 量级，硬件舍入极限是 1e-6~1e-5 量级，先看量级再选排障路径。再按证据强度排障：① 官方 kernel 对照实验（官方 torch_npu matmul 跑同数据同判定，再 bit 级对比输出，一致即数值等价）；② 控制变量变体实验（从现有数据切 K/M/N 派生真子集，注意验证切片后 shape 真的变了）；③ 拟合 σ(段数) 模型（σ≈2.6e-6×sqrt(K段数)，K段数=ceil(K/128)，mmad k=256 会被拆成 2×128，K 链长度物理不可减）；④ 位置基数效应（超标数=位置数×超标率，大 M×N case 更容易撞逐点门限）；⑤ 纯 Cube 修复死路表（树状累加受 L0C 单 tile 限制、split-K GM 中转走 hf32 丢 5 位尾数引入更大误差、拆双链 σ 不减等，勿重复尝试）。详见 [precision-model.md](references/precision-model.md)。

### L1 超容量（B 槽位 6→8 后全挂 + 300ms 超时）

这是 L1 容量超限（A5 L1 总预算 512KB，8 槽 × 列片尺寸可能达到 576KB 超出）；失败模式是全挂 + 超时而非明确报错，极难定位。预防：构建期容量预算门禁（CMake 检查 槽数×列片尺寸×dtype ≤ 预算×安全利用率，实证 459KB/87.6% 可行），超限硬失败。另注意 B 槽位跨 expert 场景必须区分 expert 维度否则数值错配。详见 [design-guidance.md](references/kernel-design/design-guidance.md) §4、[performance-patterns.md](references/performance-patterns.md) §4。

### 无 torch_npu 环境的编译面验证策略

分层验证：`build.sh --run-mode=sim` 覆盖全部内核代码编译；`cmake -D<OP>_DEMO_ONLY=ON -DCMAKE_ASC_RUN_MODE=npu` 覆盖 kernel+host 侧 ASC 编译面（绕开 torch 依赖）；torch extension 的 C++ 部分设计为纯 Python.h 入口壳，版本敏感代码收敛在 kernel 侧；远程真机完成完整链路验证；补充环境矩阵意识——torch_npu 只能在系统 CANN 环境跑。详见 [verification-and-debug.md](references/verification-and-debug.md)。

## 按请求目的路由

| 请求目的 | 执行 |
|---|---|
| 明确要求从零完整开发算子 | Step 1 → Step 2 → Step 3 → Step 4 |
| 明确要求设计或方案分析，不实施 | Step 1 → Step 2 → Step 3 |
| 咨询、解释、评审、排障、能力查询 | 只读取相关 references |

不要按调用者身份增加模式。发现已有 DESIGN/PLAN 时，不得仅因此进入 Step 4；direct invoke Developer 直接消费相同 DESIGN/PLAN，本 Skill 的 Step 4 只属于完整四步开发流程。

## 计算执行原则

算子的全部计算步骤必须在 device 侧单一 Kernel 中完成。Host 侧只负责数据准备、物理布局转换、Tiling 计算、Kernel launch 和结果搬运。

不得通过把非核心计算步骤排除到 Host 侧来缩小 API 覆盖性判定范围。若单一 Kernel、数据驻留或同步形态是用户硬约束，Step 2/3 不得降低该约束。

## 四步流程

### Step 1: Project Setup

运行 scripts/ensure_docs.py 确保 Tensor API 手册与样例可用（输出 JSON 的 `selected_root` 为事实源根目录）；需要固定文档版本时传入 --asc-devkit-ref。随后创建 operator_root。不预先实现公式、Tiling、Golden 或固定工程。文档或样例缺失且无法自动获取时保持 Step 1 blocked。

### Step 2: Tensor API Investigation

按需求语义只读调查 Tensor API：候选 API、Pattern、Copy/Mmad、C API 辅助能力、物理数据与 ABI 事实。只生成 Investigation，不修改源码、不启动构建。

### Step 3: Kernel Design

依据需求和 Investigation 生成 DESIGN 与 PLAN，覆盖逻辑接口、API 分工、数据流、Layout、内存预算、尾块、同步 owner、ABI crosswalk、构建合同、Golden 合同、性能基线和验证 checkpoint。若一次补充调查后仍缺决定性事实，保持 blocked，不生成最终 DESIGN/PLAN。

### Step 4: Implementation

只在完整四步流程中执行当前 operator_root 的 DESIGN/PLAN。实施期间持续更新 PLAN 的执行记录；不得重新选择 API 路线、改变接口/ABI、切换备选或扩大支持域。

## 知识路由

| 问题域 | 文档 |
|---|---|
| API 分工、数据流、Layout、内存预算 | references/api-dataflow-layout.md |
| NPU 执行模型、存储层级与 Tensor API 概念 | references/npu-execution-model.md |
| Tensor API 公开 API、Pattern、Copy/Mmad 与 Routing | references/tensor-api-reference.md |
| Kernel 端到端骨架（三形态、地址空间、最小可编译 skeleton） | references/kernel-skeleton.md |
| C API 辅助能力签名（初始化/同步/锁/原子/缓存） | references/c-api-primitives.md |
| 官方样例路由与检索规则 | references/sample-router.md |
| Experimental Reg Vector 与 move/move_mask/mem_bar | references/experimental-reg-vector.md |
| 同步、事件、生命周期与输出 writer | references/sync-and-lifecycle.md |
| 构建、工具链、自带头文件与 CMake | references/build-and-toolchain.md |
| 文档与样例自动获取 | scripts/ensure_docs.py |
| DESIGN 合同模板 | references/kernel-design/design-template.md |
| DESIGN 模板填写指导 | references/kernel-design/design-guidance.md |
| PLAN 与实施合同模板 | references/kernel-design/plan-template.md |
| Launcher、逻辑物理转换与证据隔离 | references/launcher/launcher-development.md |
| Golden 生成与精度对比（参考实现、ABI 解码、误差门 vs bit 级） | references/golden-generation.md |
| Cube 精度模型、排障、边界语义与低精度/量化选型 | references/precision-model.md |
| Cube/Matmul 性能瓶颈地图、递进优化序列与优化模式 | references/performance-patterns.md |
| 寄存器驻留 Vector 高性能、VF 优化与 Scalar 优化 | references/vector-performance-patterns.md |
| 融合算子设计（AIC/AIV 混合、CV 融合 UB 直通） | references/fusion-design.md |
| Tiling 决策树与多核切分 | references/tiling-strategy.md |
| 性能基线、峰值参数、瓶颈定位、达标线与优化建议 | references/performance-analysis.md |
| 参考算子阶梯路由（金标准四算子） | references/reference-operator-map.md |
| 无真机验证、环境矩阵、调试与排障 | references/verification-and-debug.md |
| 功能精度、性能与代码检视清单 | references/review-checklist.md |

## 问题、恢复与证据记录

每个开发问题保留原始问题，而非只保留最终经验。EXECUTION_RECORD 或等价记录逐项写明：

    issue_id
    original_problem
    observed_signal_and_command
    first_failed_boundary: build | abi | layout | sync | precision | performance
    classification: actual_failure | preflight_risk | environment_blocked
    root_cause_and_evidence
    fix_or_recovery
    positive_regression
    remaining_boundary
    skill_promotion: reusable | operator_local | none

编译、静态检查和 CPU Golden 只能证明对应层级。设备结论必须来自真实设备路径，并记录 sandbox/device_visible、设备节点、SoC/架构、命令、返回码和输出。没有同合同 baseline 时性能写 NOT_EVALUATED。

## 可复用完成门禁

1. Golden 独立于被测 kernel，使用最终实际 H2D 物理 bytes；量化输入先按 ABI 解码。
2. 冻结输入 dtype、累加 dtype、转换顺序、公式顺序、舍入、clamp/saturation 和非有限值处理。
3. 明确 mismatch_count 语义：bitwise 差异数与超过误差门的元素数不得混用。
4. 覆盖会改变失败边界的 shape：尾块、非对齐、跨 tile、多核、连续重复 launch。
5. 任何 device error、越界或 crash 为阻断项。
6. 闭合任务类型、同步 owner、输出唯一 writer 和资源生命周期。
7. 有界执行：设备 case 设置 timeout，保留已完成记录；挂起单 case 重跑通过不能抹掉原始挂起。
8. 逻辑到物理转换必须写入 abi_crosswalk，并在 H2D 前做 shape、size 和逐字节一致性检查。
9. 区分公开输出与诊断中间量。

## 与其他技能协作

- Matmul 类 Blaze 场景：组件选型与实现以 ascendc-blaze-best-practice 为权威源；本 Skill 只提供 Tensor API 底层事实。
- 远程真机：按环境提供的真机执行能力执行。
