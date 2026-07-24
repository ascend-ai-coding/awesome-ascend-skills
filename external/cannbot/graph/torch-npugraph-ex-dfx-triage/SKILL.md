---
name: external-cannbot-graph-torch-npugraph-ex-dfx-triage
description: PyTorch 昇腾 NPU npugraph_ex DFX 问题分诊入口。统一执行首轮全量日志收集与最少闭环信息核对，按报错栈和现象将问题路由到
  compile-error / runtime-error / accuracy / performance 四个专科 sub-skill。本 skill 不输出最终诊断结论，只完成「采集
  + 分类 + 加载下游 skill」。触发：当用户报告 npugraph_ex 相关报错、断图、精度差异或性能回退、需要 debug/dump 定位时加载。关键词：问题定位、报错、断图、精度、性能、debug、dump、aot_eager。
original-name: torch-npugraph-ex-dfx-triage
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# npugraph_ex DFX 分诊

> 本 skill 是 npugraph_ex 全部 DFX 问题的统一入口。完成首轮信息收集和分类后，**必须**加载对应的 sub-skill 继续诊断；本 skill 自身不直接给修复结论。
>
> 🔴 **加载本 skill 后的第一动作**：评估用户已提供的信息是否满足「5-step 分层产物完整性」要求（见下方工作流）。  
> - 若用户已提供完整的 5-step 产物目录，**直接进入「第二步：分诊路由」**，不重复运行。  
> - 否则，**必须先引导用户执行「分层日志采集工作流」**，日志完整后再分诊。  
> **任何情况下，禁止在日志信息足够前给出根因判断或修复建议。**

## 第一步：首轮全量信息收集

首轮必须收集以下最少闭环信息；若缺失，**先补齐信息再判断**，不要直接给修复代码：

- 最小可复现脚本，或至少核心函数 / 编译调用片段
- PyTorch / `torch_npu` / `torchair` / CANN 版本
- 输入 shape 是否固定
- 是否涉及多流 / 自定义算子 / In-place
- 现象是「报错崩溃」「结果不一致」还是「性能不达预期」
- 完整报错栈或关键日志片段——**必须按下方「分层日志采集工作流」执行**，并在回复中注明日志文件名和关键行号

### 分层日志采集工作流

不要直接拿一份 npugraph_ex 报错就开始猜根因。按下表的 5 个 step **顺序执行**，每步只跑当前 backend，**遇到第一个失败的 step 立即停止**，失败 step 的日志即为分诊主证据。哪一步首次失败，问题就出现在那一层。

| step | 形式 | 调试日志开关 |
|------|------|--------------|
| 1 | 不带 `torch.compile`，纯 eager | `TORCH_LOGS="+all"` |
| 2 | `torch.compile(backend="eager")` | `TORCH_LOGS="+all"` |
| 3 | `torch.compile(backend="aot_eager")` | `TORCH_LOGS="+all"` |
| 4 | `torch.compile(backend="npugraph_ex", options={"force_eager": True})` | `TORCH_COMPILE_DEBUG=1` + `TORCH_COMPILE_DEBUG_DIR` |
| 5 | `torch.compile(backend="npugraph_ex")` | `TORCH_COMPILE_DEBUG=1` + `TORCH_COMPILE_DEBUG_DIR` |

> step 1–3 的 `TORCH_LOGS` 输出会进 stderr，自动被合并打屏文件捕获，不再额外产 `torch_compile_debug/`；step 4–5 是 npugraph_ex 自家链路，开 `TORCH_COMPILE_DEBUG=1` 把 FX/AOT graph 等结构化产物落到 `torch_compile_debug/`。
> Ascend 日志相关环境变量参考 [Ascend 官方环境变量说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/envref/envref_07_0119.html)。

#### 目录结构（强制）

所有产物落到当前工作目录下，**根目录** `torch-npugraph-ex-triage-logs/`，**下一层**子目录名带时间戳 + shell PID 避免多次分诊互相覆盖：

```
torch-npugraph-ex-triage-logs/
└── <YYYYMMDD-HHMMSS>-pid<SHELL_PID>/
    ├── env.txt                              # 版本与关键环境变量快照
    ├── first_failure.txt                    # 首个失败 step + exit code；全通过则不存在
    ├── warnings.txt                         # 缺失/空日志产物告警；没有告警则不存在
    ├── scripts/
    │   ├── original.py                      # 用户原始复现脚本备份
    │   ├── step1-eager.py
    │   ├── step2-compile-eager.py
    │   ├── step3-compile-aot_eager.py
    │   ├── step4-npugraph_ex-force_eager.py
    │   └── step5-npugraph_ex.py
    ├── step1-eager/
    │   ├── stdout_stderr.log                # stdout + stderr 合并
    │   └── ascend_plog/                     # ASCEND_PROCESS_LOG_PATH 直接落盘
    ├── step2-compile-eager/
    │   ├── stdout_stderr.log
    │   └── ascend_plog/
    ├── step3-compile-aot_eager/
    │   ├── stdout_stderr.log
    │   └── ascend_plog/
    ├── step4-npugraph_ex-force_eager/
    │   ├── stdout_stderr.log
    │   ├── torch_compile_debug/             # TORCH_COMPILE_DEBUG_DIR 指定
    │   └── ascend_plog/
    └── step5-npugraph_ex/
        ├── stdout_stderr.log
        ├── torch_compile_debug/
        └── ascend_plog/
```

实际只会包含跑过的 step 目录；`scripts/` 里应保留本次分层验证实际运行的脚本副本。成功的 step 建议额外落一个 `SUCCESS` 标记文件；失败时在根目录写 `first_failure.txt`，不要只靠目录是否存在来反推首个失败 step。任一 step 缺 `stdout_stderr.log` 或 `ascend_plog/`，视为信息不全、必须补跑。

#### agent 最小化改造原则

不要要求用户手改入口脚本、手加 `TRIAGE_MODE`，或为了分诊手写 5 份脚本。**由 agent 基于用户现有复现脚本做最小化修改**：

1. 先把用户原始复现脚本备份到 `$ROOT/scripts/original.py`。
2. 再基于原脚本生成 `step1` 到 `step5` 的副本；**只改复现路径上的 `torch.compile(...)` 调用及其紧邻赋值**，不要改模型定义、输入构造、数据路径、随机种子、训练循环、环境初始化等无关逻辑。
3. 如果原脚本里有多个 `torch.compile` 调用，只改和当前复现路径直接相关的那一个；不要做全局重构。
4. 如果原脚本把 `torch.compile(...)` 封装在 helper / factory 里，优先在 helper 内做最小修改，而不是重写整份入口脚本。

#### compile 参数继承规则（强制）

- **step1**：去掉外层 `torch.compile(...)`，直接运行其第一个位置参数（原 model / callable）；其余业务逻辑保持不变。
- **step2 / step3**：只把 `backend` 改成 `eager` / `aot_eager`；默认**继承**原调用里的非 backend-specific kwargs，例如 `fullgraph`、`dynamic`、`mode`、`disable` 等。原脚本没有显式传的参数，不要擅自新增。
- **step2 / step3 对 `options` 的处理**：默认把 `options` 视为 backend-specific，不要把明显属于 `npugraph_ex` / NPU 的选项原样带到 `eager` / `aot_eager`；但如果 agent 能明确确认某个选项是跨 backend 的通用编译选项，可以保留，并在交接里注明保留了哪些键。
- **step4**：在原 `torch.compile(...)` 调用基础上，保留用户原有的 `fullgraph`、`dynamic` 等 kwargs；若原本有 `options={...}`，按**浅合并**处理，例如 `{**user_options, "force_eager": True}`，发生冲突时以 `force_eager=True` 为准。
- **step5**：尽量保持用户原始 `torch.compile(...)` 配置不变，包含用户已有的 `options`、`fullgraph`、`dynamic` 和其它 compile kwargs；这是和用户真实问题最接近的一步。

例如，若用户原脚本本来是：

```python
compiled = torch.compile(
    model,
    backend="npugraph_ex",
    fullgraph=True,
    dynamic=False,
    options={"foo": 1, "bar": 2},
)
```

则 5 个 step 的最小化修改应接近：

```python
# step1
compiled = model

# step2
compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)

# step3
compiled = torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=False)

# step4
compiled = torch.compile(
    model,
    backend="npugraph_ex",
    fullgraph=True,
    dynamic=False,
    options={"foo": 1, "bar": 2, "force_eager": True},
)

# step5
compiled = torch.compile(
    model,
    backend="npugraph_ex",
    fullgraph=True,
    dynamic=False,
    options={"foo": 1, "bar": 2},
)
```

#### 运行命令：调用 run_triage.sh

**目录建立、env 快照、5-step 顺序执行、失败即停、日志完整性检查、SUCCESS / first_failure / warnings 标记** 全部由本 skill 自带的 `scripts/run_triage.sh` 完成。agent 不要再现写 bash 模板，只负责：

1. 选择 ROOT 路径并 `mkdir -p "$ROOT/scripts"`。
2. 把用户原始脚本复制到 `$ROOT/scripts/original.py`。
3. 按上文「agent 最小化改造原则」+「compile 参数继承规则」生成 5 个 step 脚本（文件名固定，缺哪个就跳过哪个）：
   - `step1-eager.py`
   - `step2-compile-eager.py`
   - `step3-compile-aot_eager.py`
   - `step4-npugraph_ex-force_eager.py`
   - `step5-npugraph_ex.py`
4. 调用 `run_triage.sh`，传入 ROOT 和**原始启动命令模板**（用 `{SCRIPT}` 占位符替代脚本路径）。

```bash
SKILL_DIR=<本 skill 所在目录>   # 即 .../graph/torch-npugraph-ex-dfx-triage
ROOT=torch-npugraph-ex-triage-logs/$(date +%Y%m%d-%H%M%S)-pid$$
mkdir -p "$ROOT/scripts"

cp <用户脚本> "$ROOT/scripts/original.py"
# agent 在这里写出 step1-eager.py ... step5-npugraph_ex.py 到 "$ROOT/scripts/"

# 单卡:
bash "$SKILL_DIR/scripts/run_triage.sh" "$ROOT" "python {SCRIPT}"

# 带启动参数:
bash "$SKILL_DIR/scripts/run_triage.sh" "$ROOT" "python {SCRIPT} --batch 32 --device npu:0"

# torchrun 多卡:
bash "$SKILL_DIR/scripts/run_triage.sh" "$ROOT" "torchrun --nproc_per_node=8 {SCRIPT} --config cfg.yaml"
```

`run_triage.sh` 退出码：`0` 表示全通过，`1-5` 表示首个失败 step 编号，`64` 表示参数错误。agent 拿到退出码后直接读 `$ROOT/first_failure.txt` 和对应 step 目录的 `stdout_stderr.log` / `torch_compile_debug/` / `ascend_plog/` 做分诊，不需要重复跑。

> 设计意图：5-step 执行链路是**确定性**工作，不应每次让 LLM 现写 bash。脚本化后行为可复现、失败语义统一、agent 上下文聚焦在「脚本最小化改造」「致因报错识别」「分诊路由」这三件真正需要推理的事上。

#### 边界提示

- 多卡 / 多进程：`ASCEND_PROCESS_LOG_PATH` 共享同一目录即可，plog 文件名自带 PID 区分。
- 5 个 step 全部通过但仍现象异常（精度 / 性能） → 走 `torch-npugraph-ex-accuracy-diagnosis` 或 `torch-npugraph-ex-performance-diagnosis`。

## 总原则：先找首个致因报错

看任何日志时，优先锚定**首个致因报错**：也就是最早出现、且足以解释后续连锁失败的那条错误。

- 不要机械地取“第一行红字”或最后一行包装异常
- 不要被后续级联报错、summary、retry 失败、replay/task/stream 二次失败带偏
- 分诊时优先用这条致因报错做 compile-error / runtime-error 分类

## 第二步：分诊路由

**预归类规则（按分层工作流首次失败的 step 判断，优先于关键字规则）：**

- step1 / step2 / step3 / step4 失败 → 走 `torch-npugraph-ex-compile-error-diagnosis`。其中 **step1 失败** 需要在交接里额外提示：「eager 即崩，问题大概率不在 npugraph_ex，sub-skill 应优先排查模型 / 数据 / 环境，再回到编译链路」。
- step5 失败 → 进入下方关键字规则继续判断；若 **step4 成功但 step5 失败**，交接里要额外明确这是只在 `force_eager=False` 时暴露的问题，优先怀疑 full compile / optimization path。

**关键字规则**：按以下顺序判断，命中即停止并加载对应 sub-skill；**判断依据优先取首个致因报错**：

1. **报错栈出现以下任一关键字 → `torch-npugraph-ex-compile-error-diagnosis`**
   - `torch._dynamo`、`Unsupported`、`BackendCompilerFailed`、`graph break`
   - `Meta`、`FakeTensor`、AOTAutograd 相关栈帧
   - `npu_fx_compiler`、ACL graph capture 失败
   - 用户复现脚本「第一次调用编译后模型时就崩」，且首个致因报错**不含** `aclnnXxx` / `ACL` / `HCCL` / `stream` / `event` 等运行时关键字
2. **报错栈出现以下任一关键字 → `torch-npugraph-ex-runtime-error-diagnosis`**
   - `aclnnXxx`、`ACL`、`acl error`、`CANN` plog、`HCCL`、`hccl error`
   - `stream`、`event`、`device side assert`、`segfault`、`OOM`
   - 用户脚本「训练 / 推理跑了 N 步后才崩」
3. **无报错栈但图模式与 Eager 结果不一致 → `torch-npugraph-ex-accuracy-diagnosis`**
4. **无报错栈但性能不达预期 / 慢于基线 → `torch-npugraph-ex-performance-diagnosis`**

> 边界情况（少数报错横跨编译与运行时，例如 capture 阶段未报错、replay 时算子才 fail）→ 默认归 **runtime-error**，并在分诊回复中提示「若证据不足，可同时参考 compile-error skill」。

## 第三步：交接

在回复中明确告诉用户：

- 已收集到哪些日志（文件名 + 关键行号）
- 命中了哪条分诊规则、依据是哪一行报错或现象
- 首个失败的 step 是哪一个；若 `step4` 成功但 `step5` 失败，要明确写出这是 `force_eager=False` 才出现的问题
- 若即将加载 `torch-npugraph-ex-performance-diagnosis`，必须一并交接 `step5-npugraph_ex/torch_compile_debug/run_<ts>-pid_<pid>/` 的精确路径；不要只写“step5 success”
- 若即将加载 `torch-npugraph-ex-accuracy-diagnosis`，优先一并交接 `step4-npugraph_ex-force_eager/torch_compile_debug/run_<ts>-pid_<pid>/` 与 `step5-npugraph_ex/torch_compile_debug/run_<ts>-pid_<pid>/` 的精确路径（若存在）
- 即将加载的 sub-skill 名称
- 还缺哪些信息需要用户补充（若有）

加载 sub-skill 后，由 sub-skill 接管后续诊断流程。本 skill 不再继续输出根因或修复建议。
