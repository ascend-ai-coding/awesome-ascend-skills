# NPU Profiling 问题排查指南

按"先看产物，后看运行时错误"的顺序排查。绝大多数"profiler 跑完了但没用"的情况，都是产物问题而非崩溃。

---

## 问题零：框架内置采集（enable_profiler）产物只有 trace_view.json、所有 CSV 缺失

### 症状

`enable_profiler: True` 跑完，`log_0.log` 里一串：

```
[ERROR] profiler.py: Task [KernelViewParser] run failed.
[ERROR] profiler.py: Task [OperatorViewParser] run failed.
[ERROR] profiler.py: Task [TraceViewParser] run failed.
... (CANNTimelineParser / TraceStepTimeParser / DbParser 等全 run failed)
```

采的每个阶段（`prof/prefill`、`prof/decode`）的 `ASCEND_PROFILER_OUTPUT/` 里只剩一个 `trace_view.json`（而且常被**截断**、不是合法 JSON），没有 `kernel_details.csv` / `op_statistic.csv` / `api_statistic.csv`。

### 根因：profiling 输出目录所在的文件系统/盘（不是框架/权限/版本）

框架 `enable_profiler` 的离线解析由 CANN 的 `msprof` 工具完成。出现上述症状时，**最常见的根因是 profiling 输出目录所在的文件系统/盘**——某些挂载（数据盘 / 网络盘 / 特定容器 bind-mount）会让 `msprof` 的离线分析产不出 CSV，而把输出换到另一个文件系统（尤其本地盘）就完全正常。

排除项（实测，遇到此故障**不要**往这些方向查）：
- **不是**框架代码 / `ProfilerManager` 用法（逐特征复刻全成功）；
- **不是**目录权限（`os.access` 全 `R/W/X` True、`strace` 显示对数据的文件操作全成功、零 `EACCES`，ACL 也授权了）；
- **不是** torch_npu/CANN 版本号、**不是**算子、**不是** exe_mode；
- 同一份采集数据：输出在「坏盘」→ 全部 parser run failed、无 CSV；输出在「好盘」→ 框架原样产出 `kernel_details.csv` ~47 列 + 全套 CSV、零 run failed。

### 解决（按顺序）

1. 确认是不是这个故障：`grep -c "run failed" <res>/log_0.log` 非 0、且 `kernel_details.csv` 缺失。
2. **首选：把 profiling 输出换到另一个目录/文件系统**（优先本地盘），重跑后重新验收。profiler 输出路径 = `WORK_DIR/RES_PATH/prof/...`（`execution_engine.py:83`）；但 `function.sh` 会把 `WORK_DIR` 重写成 `cann-recipes-infer/models/<model>`，**直接 `export WORK_DIR` 会被覆盖**。可靠做法是把 `res` 软链到本地盘：`ln -sfn /tmp/recipes_res cann-recipes-infer/models/<model>/res`（res 已是非空真实目录则先移走），再跑 `infer.sh`。多换几个盘定位哪个文件系统可用。
3. 快速坐实是不是输出目录的问题：把那份失败的 `*_ascend_pt` 原始数据 `cp -r` 到另一个盘，再 `python3 -c "import torch_npu; torch_npu.profiler.profiler.analyse('<新路径>')"` —— 能解析出 CSV，就确定是原输出目录所在盘的问题。
4. **换了几个盘都不行**，再回退到注入采集（SKILL.md 路径 B / `scripts/profile_template.py`）兜底。

> 具体为什么某些盘上 CANN `msprof` 离线分析会失败（`msprof` 把错误吞了、文件权限/锁/空间都正常），属 CANN 原生层，未进一步定位；但"换到本地盘就好"可稳定复现、可用。

---

## 问题零点五：框架 prefill 采集是单步（active=1）

框架 prefill 用 `active=1, skip_first=0`，每个 request 只采 1 步 prefill（prefill 本就只有 1 步）。**实测：只要输出目录没问题（见问题零），prefill 这份能正常产出 47 列 CSV + 合法 trace_view**，不必担心"单步不完整"。如需多步 prefill / 自定义 prefill 窗口，用注入路径「模式 B」（`active=1` 后补一次收尾 `prof.step()`）。

---

## 问题一：`kernel_details.csv` 只有 9 列，缺 shape、stream、aic/aiv 指标

### 症状

```bash
$ head -1 ASCEND_PROFILER_OUTPUT/kernel_details.csv | tr ',' '\n' | wc -l
9
```

或 `op_statistic.csv` / `api_statistic.csv` 整个文件不生成。

### 原因

没有传 `ExperimentalConfig`，或者 `profiler_level` 设成了 `Level0`。默认行为只收最基本的 kernel trace，流水线利用率、算子 shape、stream 归属这些字段统统不会记录。

### 解决

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
)

profiler = torch_npu.profiler.profile(
    ...,
    experimental_config=experimental_config,   # ← 必须传
)
```

### 验证

期望 47 列，必含这些：

- `Stream ID`、`Input Shapes`、`Input Data Types`、`Output Shapes`、`Output Data Types`
- `aic_mac_time(us)`、`aic_mac_ratio`、`aic_mte1_ratio`、`aic_mte2_ratio`、`aic_fixpipe_ratio`
- `aiv_vec_time(us)`、`aiv_vec_ratio`、`aiv_scalar_ratio`、`aiv_mte2_ratio`、`aiv_mte3_ratio`
- `cube_utilization(%)`

---

## 问题二：`trace_view.json` 不以 `]` 结尾 / JSON 截断

### 症状

```
[WARNING] Incorrect schedule: Stop profiler while current state is RECORD
         which may result in incomplete parsed data.
```

以及（**仅在旧版 torch_npu 上**）：

```
[ERROR] Task [CANNTimelineParser] run failed.
[ERROR] Task [RelationParser] run failed.
[ERROR] Task [TraceViewParser] run failed.
...
```

### 版本差异

| torch_npu 版本 | warning 行为 | 实际 JSON 产物 |
|---|---|---|
| ≤ 2.7.x | warning 触发时大概率 JSON 截断、parser 全 fail | 无法用 |
| ≥ 2.8.0.post2（已实测） | warning 仍然打，但 CANN 会补齐 JSON，parsers 照常成功 | 多数情况下可用，但行为未文档化、不要依赖 |

无论哪个版本，看到 `Incorrect schedule` warning 都应视为配置错误要修掉。

### 根本原因

schedule 的转换发生在 `prof.step()` 被调用时。如果循环结束时 profiler 还停在 RECORD 状态（即 `prof.step()` 的总调用次数 ≤ `skip_first + warmup + active`），profiler 没机会走"结束 active → 触发 trace 收尾"这条转换，`__exit__` 里只能强制停止。

### 正确规则

**`prof.step()` 总调用次数 ≥ `skip_first + warmup + active + 1`**（精确边界是 `skip_first + warmup + active`，+1 为收尾余量）。下面例子都用 `skip_first=0`，故化简成 `warmup + active + 1`；框架 decode 是 `skip_first=3, active=3`，最小就是 `3+0+3=6`、留余量取 7。

两种等价实现：

**A. 把 `MAX_STEPS` 调够大**

```python
WARMUP, ACTIVE = 2, 30
MAX_STEPS = 33   # ≥ warmup + active + 1
```

**B. 循环结束后补一次 `prof.step()`**

```python
WARMUP, ACTIVE, MAX_STEPS = 2, 30, 30

with profiler as prof:
    for _ in range(MAX_STEPS):
        run_one_step()
        prof.step()
    prof.step()    # ← 收尾：让 schedule 跨出 active 窗口
```

推荐 B：保证 profile 窗口就是你关心的那 N 步，收尾步只推进 schedule、不采新数据。

### 为什么老文档里写的是 `MAX_STEPS < warmup + active`

这是一条被误传的规则。按它配，profiler 在循环结束时一定停在 active 窗口中间、状态为 RECORD —— 正是 warning 想警告的状态。新版 CANN 的兼容处理让这条错误规则"看上去"有效，但它从来都没真正让 profiler 干净收尾过。

### 验证 JSON 完整性

```bash
tail -c 50 ASCEND_PROFILER_OUTPUT/trace_view.json
# 正确结尾示例： ..."cat": "async_npu"}]

python3 -c "import json; json.load(open('ASCEND_PROFILER_OUTPUT/trace_view.json')); print('OK')"
```

---

## 问题三：采到的是编译时间，不是推理时间

### 症状

`kernel_details.csv` 里出现大量跟你的模型无关的编译/图构建 op（`aclnn*` 大量出现在前几百毫秒，之后骤减）；`trace_view` 前几秒全是 host 侧的 aclnn 绑定、图下发，真正的 kernel 非常稀疏；第一个 decode step 显示的耗时远大于非 profile 跑的稳态耗时。

### 原因 —— 是**图编译**，不是 prefill

常见误解："prefill 太慢，把 prefill 放窗口外就行"。
真正污染采样的是 **第一次 `torch.compile` / `torchair.get_npu_backend` / JIT** 触发的图构建，不管它发生在 prefill 还是 decode。Prefill 只是"第一次 forward"，在 eager 模式下并没有编译成本；decode 第一次调用才会触发 torchair / `torch.compile` 的图构建。

### 解决

真正要放 profiler 外的是**第一次**触发框架层面编译/trace 的调用：

```python
# 1) 跑一次 prefill（也顺带暖 prefill 需要的任何 shape 特化）
run_prefill(...)
torch.npu.synchronize()

# 2) 跑一次 decode（torchair/torch.compile 在这里真正编图）
warm_one_decode(...)
torch.npu.synchronize()

# 3) 这之后才进 profiler —— 此时进 profiler 的 prefill 也是真实 kernel 时间
with create_profiler(...) as prof:
    # 想分别采 prefill / decode，见 SKILL.md「关键 3」的四种模式
    ...
    prof.step()  # 收尾
```

### 快速判定有没有踩到

看 `step_trace_time.csv` 第一行 `Computing` 是不是显著大于后面几行（差 10× 以上）。如果是，说明第一次 forward 的编译时间被记进去了——把那次 forward 挪到 profiler 外再跑一遍即可。

---

## 问题四：`ASCEND_PROFILER_OUTPUT` 整个目录为空

### 排查顺序

1. **没调 `prof.step()`**
   ```python
   with profiler as prof:
       for step in range(MAX_STEPS):
           run()
           prof.step()   # ← 漏这行就什么都采不到
   ```

2. **步数少于 `warmup`**：schedule 还没进 RECORD 循环就退出了。保证 `MAX_STEPS > warmup`。

3. **权限问题**：`save_path` 没写权限。换个路径试。

4. **`on_trace_ready` 没配**：必须传 `tensorboard_trace_handler(save_path)`，否则默认不落盘。

---

## 问题五：`communication.json` / `communication_matrix.json` 没生成

### 不是 bug。

这两个文件只在 **多卡分布式** 场景会生成（需要有 HCCL 通信算子被捕获）。单卡跑 `world_size=1` 永远不会有这两个文件，不用排查。如果参考产物有、你这边没有，**先确认两边 `world_size` 是否一致**。

---

## 问题六：tensorboard 打开看不到 Profile 面板

### 症状

```bash
$ tensorboard --logdir=./prof
# 只显示 No scalar data was found
```

### 原因

TensorBoard 默认显示 scalar，NPU profile 数据要装 `torch_tb_profiler` 插件并切换到 "PyTorch Profiler" 面板。

### 解决

```bash
pip install torch_tb_profiler
# 重启 tensorboard，切 PyTorch Profiler tab
```

或者不用 tensorboard，直接在 Chrome 打开 `chrome://tracing` 加载 `trace_view.json`。

---

## Schedule 状态机参考

配置 `wait=0, warmup=2, active=30, repeat=1`，`in_cycle` 每次 `prof.step()` 后 +1：

| `prof.step()` 第几次被调 | `in_cycle` | 状态 |
|---|---|---|
| 1 | 0 | WARMUP |
| 2 | 1 | WARMUP |
| 3 | 2 | RECORD（开始采）|
| ... | ... | RECORD |
| 32 | 31 | RECORD（仍在采）|
| **33** | 32 → 0（新 cycle）或 NONE（repeat=1 已满）| ← 这一步触发收尾回调 |

想让 profiler 干净退出，就必须让第 33 次调用发生（即 `MAX_STEPS >= skip_first + warmup + active + 1`，本表 `skip_first=0` 故 = 33，或者 loop 外补一次 `prof.step()`）。

---

## 最佳实践清单

1. `ExperimentalConfig(Level1 + PipeUtilization)`，别省。
2. prefill / 图编译放 profiler 外。
3. loop 后补一次 `prof.step()`（或 `MAX_STEPS >= skip_first + warmup + active + 1`）。
4. 采完先跑「产物自检清单」（见 SKILL.md），列数不对就不用下一步分析了。
5. 用相同 config 复跑一遍对比基线，避免看个例波动。

---

## 环境信息

本文档基于以下环境验证：

| 组件 | 新文档验证版本 | 旧文档原版本 |
|---|---|---|
| torch_npu | 2.8.0.post2 | 2.7.1.post2.dev20251226 |
| CANN | 8.x/9.x | 9.0.0 |
| 平台 | Atlas A2 / A3 | Ascend910B |

新旧版本主要行为差异已在"问题二"表中列出。
