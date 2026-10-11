> 平台：Ascend950。本文件及其同目录资源仅用于此分支；相对脚本路径以本文件所在目录为基准。

# 当前仓库系统测试

通过一个技能入口，提供可信证据，证明算子在目标环境中满足其契约。可长期维护的资产是具有独立判定基准并能追溯到契约的可执行 pytest；与单次运行绑定的产物是验收报告。

可信度由三个缺一不可的部分组成：

```text
可信用例 × 完备的适用覆盖 × 成功的执行证据
```

完备覆盖能够减少未经测试的行为，从而提高可信度；但它无法弥补凭空编造的契约、依赖被测实现的判定基准、非法输入或薄弱断言。统计覆盖时只计算可信的验证结论。

使用随技能提供的脚本前，将 `ST_SKILL_DIR` 设置为本 `instructions.md` 所在目录。这是 `tilelang-op-test-design/Ascend950/` 的真实目录，不是顶层 Skill 目录。

## 选择一种业务场景

- **从零交付**：从用户需求、算子原型、接口文档或设计开始，产出契约清单、独立判定基准、完备的适用覆盖设计、可执行 pytest 和验收证据。开始设计时不要求已经存在 kernel 或测试文件。
- **存量治理**：检查仓库代码和测试，判断现有测试是否可信，找出缺失维度，并补充或修复最少但有效的 pytest 用例。
- **交付验收**：将 `PASS`、`FAIL` 或 `NOT_VERIFIED` 结论绑定到固定 commit/worktree、指定后端与设备、可信覆盖和已记录的执行结果。

从零交付或版本发布时，先阅读[交付工作流](references/delivery-workflows.md)。遵守用户指定的较小范围：只要求审查或设计时，只产出审查材料而不修改测试；要求新增、补全或交付 ST 时，视为已授权实现并验证 pytest。

## 必须执行的工作流

### 1. 定位或定义公开接口与测试入口

对于存量算子，记录公开函数、分发封装、各后端 kernel、独立 reference、现有测试和目标设备/后端。对于从零交付的算子，在编写 reference 或测试前，先依据需求或设计确定已约定的公开签名和支持范围。

按当前仓库实际文件识别 pytest 入口：

- TK 的 `tests/` 中的算子测试；
- TL 的 `testing/ascend/` 中的框架测试、`examples/ascend/test_*.py` 及各示例子目录中的配套测试；
- 用户指定的测试、算子产物目录中的 `test_*.py`，以及示例文件中内嵌的 `test_*` 函数；内嵌测试通常须显式指定文件或 nodeid。

目标尚不明确时，在导入 TileLang 或 torch 前运行静态发现工具：

```bash
python "$ST_SKILL_DIR/scripts/discover_tests.py" \
  --repo-root . --symbol <public_function> --format markdown
```

收集或运行测试前阅读[仓库测试执行](references/repository-testing.md)，尤其是涉及源码内嵌测试、NPU 执行、xdist 或测试级别时。

### 2. 先确立契约，再判断覆盖

建立简短的需求及来源清单，按以下优先级取证：

1. 用户确认的需求，或权威的接口/设计文档；
2. 公开 API 签名和 docstring；
3. 独立 PyTorch/CPU reference 或明确的数学定义；
4. 成熟的同等 CUDA/旧实现行为，作为佐证；
5. 现有测试和实现检查，作为当前行为的证据。

通过实现代码识别分发路径和边界路径。不得只凭被测实现证明预期语义。不同来源相互冲突时，应明确指出冲突；必需行为没有可靠来源时，将其标为 `CONTRACT_GAP`，不要凭空编写通过/失败断言。

审查或创建正确性断言时，阅读[测试可信度](references/test-credibility.md)。

### 3. 先审查可信度，再统计覆盖

逐个检查相关测试：

- 正向用例的输入对于所验证场景是否合法；负向用例是否有明确的异常拒绝契约；
- 判定基准对当前验证结论是否足够独立；
- 断言是否检查了所有实质输出、梯度或状态变化；
- 用例是否确实进入所声称的后端和路径；
- 测试是否被 pytest 收集并实际执行，而不只是存在于源码中；
- 测试结果和实际使用的随机种子能否复现。

只检查 shape 的断言只能证明形状，不能证明数值正确。仅与 CUDA 做差分对比只能作为辅助证据；主要语义结论必须有规格、独立 reference、明确的期望值或有充分理由的变形关系作为依据。

对每项验证结论分别标记为 `TRUSTED`、`PARTIAL`、`UNTRUSTED` 或 `UNKNOWN`。同一个测试对输出形状可能可信，对数值却可能只有部分可信度。

### 4. 评估十个覆盖与验收维度

每个维度都要记录 `covered`、`partial`、`missing`、`unknown` 或 `not_applicable`，关联相应用例 ID 并说明理由。`not_applicable` 必须是经过审查得出的结论，不能只是留空。

| 序号 | 维度 | 适用时必须检查的内容 |
|---:|---|---|
| 1 | 功能 | 公式、mask、广播、可选行为、多输出 |
| 2 | 精度 | 精确比较，或按输出/dtype 给出有依据的容差 |
| 3 | 边界 | 最小值、空输入、合法极值，以及真实的 tile/core 尾块 |
| 4 | 梯度 | 所有必需的反向结果均与支持 autograd 的 reference 比较 |
| 5 | 状态修改 | 应发生的变化，以及不得变化的输入、区域、padding 或存储 |
| 6 | 异常拒绝 | 文档定义的非法输入、准确的异常类型和稳定的报错原因 |
| 7 | 布局与接口 | shape、dtype、device、stride、连续性、别名关系、元数据 |
| 8 | 后端与分支 | 必需的 CUDA/Ascend 分发路径和关键算法路径 |
| 9 | 随机性 | 随机种子语义、可复现性、不变量，以及随机算子的分布 |
| 10 | 执行证据 | 测试收集、nodeid、目标运行时、结果、环境和随机种子策略 |

前九项描述算子与测试覆盖；执行证据说明设计出的覆盖是否真正运行。以下附加分类轴应分别记录：

- 仓库测试级别：`0` 核心、`1` 默认、`2` 全量；
- 测试种类：普通、边界、负向、梯度、有状态、benchmark；
- 契约维度：dtype、rank/shape、属性、可选参数依赖、layout/stride、dispatch/backend、数值范围、副作用、输出和梯度。

根据公开合法域以及实际分发和 tiling 代码推导边界候选。只有公开契约允许该输入，并且最终运行时维度确实产生余数时，`B+1` 才是有效的正向尾块用例。添加边界、负向、梯度或有状态用例前阅读[覆盖设计](references/coverage-design.md)。

不要以固定用例数为目标。优先选择一组精简用例，区分有意义的路径及其交互，同时避免不必要的 JIT 变体和内存成本。

### 5. 实现可长期维护的 ST 资产

- 从零交付时，如果相应的独立测试文件和可审查的 PyTorch/CPU reference 尚不存在，应创建它们。测试必须调用约定的公开 API，不得在测试中实现生产 kernel。
- 存量治理时，若已有测试文件具备合适的 fixture 和 conftest 行为，应在该文件中扩展。
- `get_device()`、`get_test_level()`、生成器、reference、`make_param_id` 和数值辅助函数的语义匹配时，应复用它们。
- 为新增参数化用例提供稳定且能说明目的的 ID。
- 除非权威契约证明需要修改，否则保留现有 reference 逻辑和容差。
- 比较所有实质输出。对于原地 kernel，应分别克隆 DUT 和 reference 输入，并在适用时检查受保护存储或未修改区域。
- 负向用例必须检查明确的预期异常；导入、设备、编译或 OOM 等无关异常不能算作成功的拒绝测试。
- 合法边界产生错误结果时，pytest 必须失败，不能降级成警告。

从零交付和交付验收的测试设计使用[用例规格](references/case-specification.md)中定义的 JSON 格式记录。只有对小型存量算子进行修改且契约映射十分明确时，才可以省略。使用以下命令校验：

```bash
python "$ST_SKILL_DIR/scripts/check_st_spec.py" \
  path/to/spec.json --repo-root .
```

最终验收时增加 `--require-complete-coverage`；此时存在 `partial`、`missing` 或 `unknown` 维度都会阻止 `PASS`。

### 6. 收集、运行并保存执行证据

先在与正式执行相同的后端和设备环境中收集准确目标。静态 `node_hint` 只能用于发现测试函数；`is_ascend()` 等依赖后端的生成器可能产生不同的参数化 nodeid。应从目标环境的收集结果中选择具体 nodeid，批量运行前再用非 xdist 方式验证这些 nodeid。收集数量为零或所有相关测试均被跳过时，结论是 `NOT_VERIFIED`，绝不能判为通过。先运行新增或修改的用例，再运行已确定的完整目标用例清单。需要扩大覆盖时，按当前项目实际测试配置执行扩展测试。

调用时接收目标后端 PTO 或 AscendC。运行测试收集、新增用例及最终完整精度验收时，在命令前显式指定对应后端：PTO 使用 `TILELANG_DEFAULT_TARGET=pto`，AscendC 使用 `TILELANG_DEFAULT_TARGET=ascend`；下方执行证据脚本的命令也使用相同前缀。修复后在选定后端重跑最终验收。

常规独立测试的命令参数和设备并发遵循本分支的 [测试执行说明](references/repository-testing.md)；ST 最终验收的测试级别以上一段为准。vLLM 源码内嵌测试应显式指定并单独运行。除非用户要求，否则不要刷新 benchmark 基线或内存画像。

记录 commit/diff、环境及实际导入的依赖路径、目标后端/设备、完整命令、收集到的 nodeid、结果数量、失败/跳过信息和实际随机种子策略。明确区分测试失败、收集错误、依赖错误、编译错误、设备资源错误、中断和超时。

需要持久化 JSON 记录时，使用执行证据记录工具：

```bash
python "$ST_SKILL_DIR/scripts/pytest_st_evidence.py" \
  --repo-root . --output <artifact-dir>/pytest-evidence.json \
  --device "Ascend NPU" --backend "PTO/Ascend" -- \
  tests/<domain>/test_<op>.py -m "not benchmark" -q
```

该工具只能证明测试收集和执行事实。必须同时完成契约校验与可信度审查，才能给出 `PASS`。

### 7. 给出交付结论

先说明正确性结论及其适用范围，并包含以下内容：

| 项目 | 必须包含的内容 |
|---|---|
| 契约 | 需求、来源、后端、确定程度 |
| 可信度 | 可信、部分可信、不可信、未知的验证结论及原因 |
| 覆盖 | 已被可信覆盖的维度和剩余缺口 |
| 修改 | 新增用例，以及每个用例能检出哪类独立缺陷 |
| 执行 | 收集、执行、通过、失败、跳过、未运行的数量 |
| 复现 | 准确命令、nodeid、环境、随机种子 |

不能因为少量用例通过，就声称算子已被广泛证明正确。设计完成、已经实现、仅被收集或被跳过的用例，都不能算作实际执行覆盖。

- `PASS`：所有强制且适用的维度都在指定目标上获得可信的已执行证据，并且不存在尚未解决的必需契约缺口。
- `FAIL`：违反任何强制的正确性、异常拒绝、状态、梯度、编译/运行或其他交付要求。
- `NOT_VERIFIED`：缺少必需证据、测试未被收集或执行、测试被跳过、受到环境阻塞，或契约尚不明确。

## 架构清单

- **一个入口**：`tilelang-op-test-design`。
- **三种业务场景**：从零交付、存量治理、交付验收。
- **两种 pytest 组织形式**：独立测试和 vLLM 源码内嵌测试。
- **四份核心资料**：[交付工作流](references/delivery-workflows.md)、[测试可信度](references/test-credibility.md)、[覆盖设计](references/coverage-design.md)和[仓库测试执行](references/repository-testing.md)。
- **三个工具**：用于静态发现的 `discover_tests.py`、用于审查契约/用例的 `check_st_spec.py`、用于记录收集/执行证据的 `pytest_st_evidence.py`。
- **三个试点**：`engram_hash`、SwiGLU 和 `bmp_to_patches`。
- **十个维度**：第 4 步中的覆盖与证据检查表。

[用例规格](references/case-specification.md)提供结构化格式，[试点证据](references/pilot-evidence.md)记录示例；二者是四份核心资料的支撑材料，不增加新的工作流类别。

[泛化证据](references/generalization-evidence.md)记录试点之后的应用情况。这些应用在不改变架构清单的前提下，发现或修复了真实覆盖缺口。

## 试点路由

处理 `engram_hash`、SwiGLU 或 `bmp_to_patches` 时阅读[试点证据](references/pilot-evidence.md)。这些示例分别打通了整数精确输出、可选参数约束和 Ascend 源码内嵌边界测试的首批路径。处理其他算子时，应使用通用工作流，不要照搬试点特有的 shape 或容差。
