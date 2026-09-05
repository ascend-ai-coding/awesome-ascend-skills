---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-testcase-gen
description: 为基于 SHMEM 的算子生成正确性测试计划、case matrix、golden/checker 和测试脚本。触发：用户需要设计 SHMEM
  算子测试用例、case matrix、golden、checker 或测试脚本时使用。关键词：基于SHMEM测试用例、case matrix、golden、checker、gen_data、check_result。
original-name: shmem-ops-testcase-gen
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 算子测试用例生成

**Skill类型**：文档与脚本生成型（读取设计文档，输出测试计划和验证脚本）

> **中文写作要求**：测试计划和 case matrix 文档必须使用中文撰写。仅 Case ID、代码片段、API 名称等技术术语保留英文原文。

将 `design.md` 中的 correctness contract 和 invariants 转化为可执行的测试计划、数据生成脚本和精度验证脚本。测试计划是实现约束，必须在代码生成前完成。

## 必读资料

| 文件 | 用途 |
| --- | --- |
| [references/correctness.md](references/correctness.md) | 测试规则——golden 构造策略、case matrix 类别、invariant→test 映射 |
| [references/precision-standard.md](references/precision-standard.md) | 精度标准——OpTypes 分类、rtol/atol 取值表、双统计判定、双级兜底 |
| [references/test-structure-template.md](references/test-structure-template.md) | 代码模板——gen_data.py / check_result.py / `scripts/run.sh`（含 timeout）/ main.cpp 边界规则 |
| [references/testcase-scale-standard.md](references/testcase-scale-standard.md) | 规模分档（XS/S/M/L）、边界条件、dtype 覆盖、case matrix 模板、最小 case 数 ≥ 20 |
| [references/GUIDE.md](references/GUIDE.md) | 本 skill 参考索引 |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO`（对照仓内 examples 构造 golden 前） |

## 输入

- `design.md` 的 `correctness` 字段：oracle、rtol/atol、invariants、case_matrix
- `design.md` 的 `interface` 字段：inputs/outputs/dtype/shape
- `design.md` 的 `semantics` 字段：local_compute、communication、finalize

## 工作流

```
步骤 1  提取 correctness contract
步骤 2  生成 case matrix
步骤 3  生成数据生成脚本 (gen_data.py)
步骤 4  生成精度验证脚本 (check_result.py)
步骤 5  生成运行脚本 (`scripts/run.sh`)
```

---

## 步骤 1：提取 correctness contract

从 `design.md` 中提取：
- oracle 方案（rank pattern / CPU numpy / PyTorch / file-based）
- tolerance（rtol/atol，按 dtype 区分）
- invariants 列表
- case matrix 规模要求

---

## 步骤 2：生成 case matrix

必须按 [references/testcase-scale-standard.md](references/testcase-scale-standard.md) 的规模分档生成，覆盖 XS/S/M/L 四档 + 边界 + 压力，总 case 数 ≥ 20。类别要求详见 [references/correctness.md](references/correctness.md) §3。

---

## 步骤 3：生成 gen_data.py

职责：
- 固定随机种子（`np.random.seed(42)`），确保可复现
- 为每个 PE 生成独立输入文件（`input_pe{rank}.bin`）
- 计算 golden 输出（使用高精度 float32）
- 保存 golden 文件（`golden_pe{rank}.bin` 或 `golden.bin`）
- 保存配置文件（shape、dtype、PE 数等）

Golden 构造方法：
- **通信算子**：rank pattern — 直接按通信语义构造期望输出；可选 CPU numpy/PyTorch 参考实现作兜底
- **通算融合算子**：PyTorch / CPU numpy 参考实现 — 模拟 local compute → communication → finalize 全流程（见 [correctness.md §4.3](references/correctness.md)）

---

## 步骤 4：生成 check_result.py

职责：
- 读取算子输出和 golden 文件
- **按 `meta.op_kind` 和 dtype 选择 tolerance**（不能一律使用 `1e-5`）；`op_kind` → `OpTypes` 映射见 [precision-standard.md §5.1](references/precision-standard.md)
- 每个 PE 计算并打印 MaxAE、MaxRE、MeanRE 和 tolerance 阈值
- 打印详细 mismatch（PE、index、actual、expected、error）
- 返回明确退出码（0=通过，非0=失败）

Tolerance 选择规则详见 [references/precision-standard.md](references/precision-standard.md)。`scripts/run.sh` 调用 check_result.py 时必须显式传 `--rtol` 和 `--atol`，不依赖脚本内默认值。

---

## 步骤 5：生成 `scripts/run.sh`

按 [test-structure-template.md](references/test-structure-template.md) §4 生成算子内 `scripts/run.sh`。

**交付入口（custom-ops 默认）**：skill/README **MUST** 以 [custom-ops-entrypoints.md](../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §2 为首选运行入口，算子内 `scripts/run.sh` 为被转发实现。build 失败提示 **MUST** 为同文件 §1 编译。

多 PE 启动脚本，每个 PE 一个独立进程。

**环境设置 MUST** 内联 [env-setup.snippet.md](../shmem-ops-compile-debug/references/env-setup.snippet.md) 中的 `setup_shmem_runtime_env`，或在模板中等价内联：先激活 SHMEM install 环境脚本，再追加 `build/lib` 和 `${ASCEND_HOME_PATH}/lib64`。禁止只手动设 `LD_LIBRARY_PATH=build/lib`。

---

## 输出产物

```
op_name/
├── scripts/
│   ├── gen_data.py
│   ├── check_result.py
│   ├── `scripts/run.sh`
│   └── run_case_matrix.py
└── data/              (gen_data.py 生成)
    ├── input_pe0.bin
    ├── input_pe1.bin
    ├── golden.bin
    └── config.json
```

case matrix 执行报告输出到 `docs/case_matrix_report.md`。

---

## MUST检查

- [ ] case matrix 覆盖 XS/S/M/L 四档规模（见 testcase-scale-standard.md §1.1 或 §1.2）
- [ ] case matrix 覆盖所有类别（functional + boundary + stress + **performance**）
- [ ] **performance 类别 MUST 含 S 档 + L 档各 1**（`perf_s_*`、`perf_l_*`，8PE 推荐）
- [ ] 总 case 数 ≥ 20
- [ ] gen_data.py 可独立运行，输出确定性数据
- [ ] check_result.py 返回明确退出码
- [ ] `scripts/run.sh` 支持多 PE 启动
- [ ] `scripts/run.sh` 调用 `setup_shmem_runtime_env`（或等价完整 LD_LIBRARY_PATH 链）
- [ ] `scripts/run.sh` 通过 `setup_shmem_dynamic_endpoints` 避免固定 `IPPORT`/`SHMEM_UID_SESSION_ID` 冲突
- [ ] 包含中等规模 case（M 档）和大规模 case（L 档），或标记未满足原因
- [ ] golden 逻辑与 design 语义一致

---

## 反模式（NEVER DO THESE）

- ❌ golden 逻辑放在 main.cpp 中
- ❌ 精度验证放在 main.cpp 中
- ❌ 只有 smoke case 没有 medium scale
- ❌ performance case 缺少 S 档或 L 档（Phase 6 须双档）
- ❌ 只覆盖单一规模档次（必须覆盖 XS/S/M/L 四档）
- ❌ 总 case 数不足 20
- ❌ golden 用低精度计算导致误差累积
- ❌ 测试脚本依赖未声明的环境变量
- ❌ `scripts/run.sh` 只设 `LD_LIBRARY_PATH=build/lib`，不 `source `${SHMEM_REPO}/install/set_env.sh`
- ❌ `scripts/run.sh` 写死 `IPPORT=tcp://127.0.0.1:27010` 等全局默认端口（与 Torch 测试或其它 run 并行时 shmem init 失败）
- ❌ `result_compare.py` / checker 的 `dtype` 与 golden 生成 dtype 不一致（如 int8 数据用 float16 读取）
- ❌ transport/纯搬运算子使用 relaxed tolerance（应为 bitwise exact）
- ❌ check_result.py 只打印 PASS/FAIL 不输出 MaxAE/MaxRE/MeanRE
