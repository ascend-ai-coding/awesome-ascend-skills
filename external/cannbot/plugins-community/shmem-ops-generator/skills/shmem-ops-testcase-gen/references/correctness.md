# SHMEM 正确性验证规则

> **仓内路径**：下文 `examples/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

本文定义 SHMEM 算子正确性验证的规则和策略：测什么、怎么构造 golden、精度容差如何选取、invariant 如何映射到测试。代码模板（gen_data.py / check_result.py / `scripts/run.sh` / main.cpp 边界）见 [test-structure-template.md](test-structure-template.md)。

## 1. 核心原则

**职责分离**：算子 `main.cpp` 只负责单 PE Host 编排、kernel 调用和输出文件写入。golden 生成和精度验证由独立的 Python 脚本完成。复杂 Host 计划逻辑拆到独立 `.cpp/.h`。

**单进程单 PE**：`main.cpp` 一次启动只对应一个 PE。多 PE 测试由 `scripts/run.sh` 启动多个独立进程。

**算子必须在 Device 侧执行**：即使是 correctness-first 实现，也必须使用 Device kernel。禁止 Host RMA 作为主通信路径。可以先用简单的 Device kernel（如单 block 串行）实现正确性。

**标准流程**：
1. Python 脚本生成输入数据和 golden 输出
2. Device kernel 执行算子，Host 侧收集结果写入文件
3. Python 脚本读取输出文件，与 golden 比较精度

## 2. 输入来源

生成测试前从 `design.md` 读取：

| 字段 | 用途 |
| --- | --- |
| `interface` | 输入输出、dtype、shape、输出 visibility |
| `semantics` | local compute、communication、finalize/result |
| `topology` | PE/team/subgroup、rank 映射、scope |
| `memory` | symmetric buffer、signal/state、输出 ownership |
| `schedule` | core_partition、tiling、phases |
| `correctness` | oracle、tolerance、invariants、case_matrix（含 stress case） |

如果缺少 oracle、tolerance、PE 范围或输出可见性，停止生成测试并要求修订设计。

## 3. 测试计划

实现代码前先产出 case matrix，避免代码写完后才补正确性目标。

测试计划至少包含：

| 类别 | 最小要求 |
| --- | --- |
| smoke | 2 PE、小 shape、主 dtype、单轮运行 |
| rank pattern | 每个 PE 输入含 rank 特征，验证通信错位 |
| contract cases | 覆盖 design 指定 dtype、shape、PE count、scope |
| tail/chunk | 小于 chunk、等于 chunk、大于 chunk 且有 remainder |
| repeats | 多轮复用 signal/state，验证 epoch/magic 或清零逻辑 |
| gap cases | gap analysis 中新增能力或 engine-specific path |
| visibility | replicated、sharded 或 owner-only 输出分别检查 |

### 4.1 Rank Pattern（通信算子推荐）

为每个 PE 生成含 rank 特征的输入（如 `pe_id * 1000000 + token * 1000 + h`），golden 按通信语义直接构造。适合 allgather、shuffle、put/get 等纯搬运算子。

优点：输出可直接看出数据来自哪个 PE，容易定位通信错位，适合 exact compare。

参考：`examples/allgather/`

### 4.2 CPU/Numpy Reference（通信算子 + 浮点计算）

Python 脚本使用 numpy 按通信语义计算 golden。浮点累加使用 float32 中间结果避免精度损失。

参考：`examples/allgather/scripts/data_gen.py`

### 4.3 PyTorch Reference

Python 脚本使用 PyTorch 模拟完整语义（local compute → communication → finalize）。计算使用 float32，最终按 design 的 dtype/cast 规则转换。

参考：`examples/allgather/scripts/gen_data.py`、`examples/kv_shuffle/`

### 4.4 Golden 构造通用要求

- 固定随机种子（`np.random.seed(42)` 或 `torch.manual_seed(42)`）
- 浮点累加优先用 float32 中间结果
- 输入/golden 按 PE 分文件，文件名包含 pe_id
- 保存 config.json（shape、dtype、n_pes、expected_elems 等）

## 5. 精度容差规则

从 `design.md` 的 `correctness.tolerance` 读取阈值。设计未给阈值时先补设计，不在 checker 中临时硬编码。

所有容差值、OpTypes 分类、双统计判定和双级兜底规则见 [precision-standard.md](precision-standard.md)。本节仅保留与容差无关的 invariant 测试映射（§6）。

### 5.1 核心要点
- rtol / atol 按 [precision-standard.md §3](precision-standard.md) 的 OpTypes × dtype × compute_times 三要素选取
- atol 统一等于 rtol
- check_result.py 同时计算 `precision_percent`（逐元素通过率）和 `eb`（平均偏置），通过标准见 [precision-standard.md §4](precision-standard.md)
- 支持可选的 `--torch-output` 参数启用 reference pass 兜底，判定见 [precision-standard.md §5](precision-standard.md)
- 比较时转 float64，检查 nan/inf
- `scripts/run.sh` **MUST** 显式传 `--rtol` 和 `--atol`

## 6. Invariant → 测试映射

把 correctness invariants 转成可执行检查：

| Invariant | 测试方式 |
| --- | --- |
| symmetric allocation 顺序和大小一致 | Host 侧记录 layout，所有 PE 参数一致；运行前后检查 malloc 结果非空 |
| remote write before consume | rank pattern + repeats；signal/wait 后再读取 |
| phase k 完成后 phase k+1 消费 | 每 phase 写入可识别 marker 或保留中间 output |
| output visibility 匹配 | replicated 检查所有 PE；sharded 检查每 PE slice；owner-only 只检查 owner |
| dtype/cast/accumulation 与 oracle 一致 | CPU reference 使用相同 cast/finalize 规则 |
| team/subgroup PE 映射一致 | 只让指定 PE 参与，并检查非参与 PE 输出保持预期 |
| tail/chunk 正确 | 专门构造边界 shape 和不整除 PE 数 |

测试脚本要把未验证的不变量显式列出，不能静默省略。
