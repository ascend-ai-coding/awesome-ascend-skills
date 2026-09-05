---
name: external-cannbot-graph-torch-npugraph-ex-performance-diagnosis
description: PyTorch 昇腾 NPU npugraph_ex 性能诊断（FX 图静态审计，聚焦 reinplace 未命中导致的冗余 tensor
  move）。处理「5 step 全部通过、但推理慢/Device 利用率低」阶段：基于 TORCH_COMPILE_DEBUG=1 产出的 FX 图序列与 debug.log，定位图里冗余的
  tensor move——重点是 reinplace_inplaceable_ops_pass（out-of-place→in-place）与 reinplace_input_mutated_ops（折叠输入侧
  copy_）回填未完成留下的 copy_/clone/out-of-place 冗余，按成因分类区分输入侧 copy_ epilogue、auto_functionalized
  物化 clone、多流校验失败三类。本 skill 由 dfx-triage 路由进入；**不**处理 graph break / recompile / guard
  failure（归 compile-error-diagnosis）。触发：当用户报告 npugraph_ex 推理慢、性能回退、怀疑图里有多余搬运/拷贝、reinplace
  没生效时加载。关键词：性能、慢、tensormove、tensor move、reinplace、functionalize、copy_、clone、auto_functionalized、multi_stream、can_inplace、missed
  reinplacing、inplace_pass、input_inplace_pass、clone_input、TORCH_COMPILE_DEBUG。
original-name: torch-npugraph-ex-performance-diagnosis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# npugraph_ex 性能诊断（reinplace 回填未完成所致的冗余 tensor move）

> 本 skill 处理「5 step 全部通过、但性能不达预期」阶段：基于 `TORCH_COMPILE_DEBUG=1` 产出的 FX 图序列与 `debug.log`，**找出图里冗余的 tensor move**，重点是 reinplace 未命中留下的搬运。纯静态审计，不替代 Profiling。由 `torch-npugraph-ex-dfx-triage` 路由进入；若尚未做首轮日志收集，先回 triage skill 完成 5 step 采集。若用户已直接提供独立 `torch_compile_debug/run_*` 产物，也可按等价前置条件进入。

## 适用范围

- 推理延迟高于 Eager 基线 / 业务预期，怀疑图里有"不该有"的搬运
- 图里残留 `aten.copy_`或 `aten.clone`
- out-of-place / functional 算子本可转 in-place 却没转

不属于本 skill 的场景：
- 任何报错 / 崩溃 / 异常 → 回 dfx-triage 重新分诊
- graph break、recompile、guard failure → `torch-npugraph-ex-compile-error-diagnosis`
- 图模式与 Eager 结果不一致 → `torch-npugraph-ex-accuracy-diagnosis`

## 总原则

1. **别只盯 `copy_`**：输入侧 `copy_` 是 functionalize 的写回 epilogue（正常产物、是判据而非缺陷）；reinplace 回填未完成才会留下三类冗余 tensor move 信号——输入侧未折叠的 `copy_`、`auto_functionalized` 物化出的 `clone`(+写回 `copy_`)、原样保留的 out-of-place 算子（无 move 节点）。审计时三类信号都要看：末图里同时 grep `copy_` **和** `clone`，再配合 `debug.log` 的 `missed opportunities` 日志定位无 move 节点的那类。详见「reinplace 与 functionalize 的真实关系」。
2. **先完成成因分类与定位**：先按「冗余 tensor move 的成因分类与定位」把成因一/二/三走完。
3. **证据分两类附带**：日志 / FX 图结论必须附 `文件名` + 关键片段（`debug.log` 行或 FX txt 片段）；源码归因必须附 repo-relative path。找不到证据就标"待确认"，不猜。
4. **碰到分类外信号立刻转走**：在 `torchdynamo/debug.log` 看到 `graph break` / `Restarting analysis` / `guard failure` → 停止，转 `torch-npugraph-ex-compile-error-diagnosis`。
5. **不主动执行用户脚本**：只给 `TORCH_COMPILE_DEBUG=1` 命令模板让用户自跑，再回到本 skill 做静态审计。

## 源码路径约定

下文若简写 `graph_pass.py` / `utils.py` / `graph_utils.py`，统一指 npugraph_ex 的以下实现路径，而不是 workspace 里其它同名文件：

- `graph_pass.py` → `torchair/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py`
- `utils.py` → `torchair/npugraph_ex/npugraph_ex/_acl_concrete_graph/utils.py`
- `graph_utils.py` → `torchair/npugraph_ex/npugraph_ex/_utils/graph_utils.py`

## tensor move 定义

**tensor move** 指张量数据的搬运 / 拷贝（device 上 tensor→tensor 的 copy），等价于 `aten.clone` 的拷贝语义，且不做 dtype/format 转换（那是 Cast / TransData 的职责）。

## reinplace 与 functionalize 的真实关系

Functionalize 把 in-place 算子改写成 out-of-place SSA。两种结果不同：

- **中间张量** 上的 in-place（如临时张量 `a.add_(2)`）→ functionalize 后**不产生 `copy_`**，只是多出一个 out-of-place 算子（多一次 buffer 分配 + HBM 写）。
- **图输入 / 输出** 的 mutation（caller 要观察到，如 KV Cache 写回）→ functionalize 追加**一个 `aten.copy_(input, new)` epilogue**。

npugraph_ex 相关的三条 pass（前两条做 reinplace 回填，第三条展开 auto_functionalized 并物化未命中的 clone）：

| pass | 配置名 | 干什么 | 图中可见信号 |
|------|--------|--------|-------------|
| `reinplace_inplaceable_ops_pass`（核心 `reinplace_with_multi_stream_check`，见上文 `graph_pass.py` 路径约定） | `inplace_pass` | out-of-place `b=foo(a)` → in-place `a.foo_()`（如 `add`→`add_`）；另处理 view_scatter、npu 多原地算子。**不碰 auto_functionalized** | **out-of-place / functional 算子原样留着**（多一次 buffer 分配 + HBM 写，**无 move 节点**） |
| `reinplace_input_mutated_ops`（`_mutated_input_reinplace`，见上文 `graph_pass.py` 路径约定） | `input_inplace_pass` | 折叠 `placeholder`/`get_attr` 上的 `copy_` epilogue；**做 `auto_functionalized` 的 base reinplace 决策** | **图输入上残留 `copy_`**；`auto_functionalized` 的 base 没命中时决策记在 meta，later 由 `decompose_auto_functionalized` 物化成 **`clone`** |
| `decompose_auto_functionalized`（见上文 `graph_pass.py` 路径约定） | — | 把 `auto_functionalized` 展开成实际 mutable 算子；对未命中 reinplace 的 base 物化 `clone` + 写回 `copy_` | **`clone` + 非同源 `copy_`**（即成因二） |

> **关键：`copy_` 只是 functionalize 的一种产物，不是性能问题的唯一（甚至不是主要）信号。** `aten.copy_` 本身不是模型里写的算子，而是 functionalize 为"图输入/输出的可观察 mutation"补的写回 epilogue；reinplace 的目标就是把它（以及 out-of-place 形态）回填掉——**输入侧 `copy_` 是判据，未折叠才是问题**。所以**冗余信号对应三种成因，别只盯 `copy_`**：
>
> | 图中可见信号 | 来源 | 对应 | 是不是 tensor move |
> |---|---|---|---|
> | 输入侧 `copy_(placeholder/get_attr, src)` | functionalize 写回 epilogue 没被 `reinplace_input_mutated_ops` 折叠 | **成因一** | 是（一次写回） |
> | `clone` + 非同源 `copy_` | `auto_functionalized` 的 base 没 reinplace，`decompose_auto_functionalized` 物化出来 | **成因二** | 是（clone 一次分配+读，copy_ 一次写回） |
> | out-of-place / `*_functional` 算子原样保留（`add` 没变 `add_`） | `reinplace_inplaceable_ops_pass` 没转 in-place | **成因三** | **否**：没有 `copy_`/`clone` 节点，只是多一次 buffer 分配 + HBM 写 |
>
> 推论：① 只数 `copy_` 会**漏掉成因二的 `clone`**（auto_functionalized 失败的主信号）和**成因三**（根本没有 move 节点，要看 `debug.log` 的 `missed opportunities` 日志）；② `copy_` 计数大致只覆盖成因一（+成因二写回的那半）。审计时三类信号都要看：末图里 grep `copy_` **和** `clone`，再配合 `missed opportunities` 日志定位成因三。

## 前置条件

1. 已按 dfx-triage 跑完 5 step，**step5 success**；若未走 triage，至少已直接产出等价的 standalone `torch_compile_debug/run_<ts>-pid_<pid>/`
2. 已有以下任一布局的 debug 产物，且目录内至少含 `npugraph_ex/debug.log` 与 `npugraph_ex/model__*/forward/` 下的 FX 图序列：
   - `torch-npugraph-ex-triage-logs/<YYYYMMDD-HHMMSS>-pid<SHELL_PID>/step5-npugraph_ex/torch_compile_debug/run_<ts>-pid_<pid>/`
   - `torch_compile_debug/run_<ts>-pid_<pid>/`

任一不满足 → 不展开，回 dfx-triage 补齐。

## 产物路径地图

```
RUN_DIR = torch-npugraph-ex-triage-logs/<YYYYMMDD-HHMMSS>-pid<SHELL_PID>/step5-npugraph_ex/torch_compile_debug/run_<ts>-pid_<pid>/
# 或 standalone: torch_compile_debug/run_<ts>-pid_<pid>/

RUN_DIR/
├── npugraph_ex/
│   ├── debug.log                                          # ★ reinplace 判定主证据
│   ├── model__N/forward/
│   │   ├── output_code.py                                 # 编译后图结构
│   │   ├── 000_aot_forward_graph.txt                      # functionalize 后、优化前
│   │   ├── 001_aot_forward_graph_after_<pass>.txt         # 每个 pass 之后
│   │   └── ...
│   └── ...
└── torchdynamo/debug.log                                  # 仅用于检测"应转走"关键字
```

与 tensor move 相关的 `<pass>`（按调度顺序）：

```
reinplace_inplaceable_ops_pass      ← out-of-place → in-place（add→add_）
reinplace_input_mutated_ops         ← 折叠输入侧 copy_ epilogue
decompose_auto_functionalized       ← auto_functionalized 展开成 clone + mutable op
eliminate_self_copy                 ← 清扫 copy_(x, x) self-copy
```

## 通用诊断步骤（只走一次）

1. **定位**：在最后一个 `model__N/forward/*_after_*.txt` 里 grep 残留 tensor move：
   ```bash
   grep -n "aten\.copy_\.default(\|aten\.clone\.default(" *_after_eliminate_self_copy.txt
   ```
2. **看产物特征**，据此归入下列某一成因：
   - `copy_` 第一参是 `placeholder` / `get_attr` → **成因一**
   - `*_after_decompose_auto_functionalized.txt` 里出现 `as_strided→clone→as_strided→自定义 in-place 算子`，且末尾有非同源的 `copy_(input, 结果)` → **成因二**
   - out-of-place / `*_functional` 算子在 `*_after_reinplace_inplaceable_ops_pass.txt` 里原样保留、且其输出后续没被用 → **成因三**
3. **确认**：拿成因结论去 `npugraph_ex/debug.log` grep 对应关键字（见各成因）坐实。
4. **指向**：给出源码 repo-relative path + 下一步最小动作。证据必须附 `文件名` 或源码 path，找不到就标"待确认"，不猜。

> 在 `torchdynamo/debug.log` 看到 `graph break` / `Restarting analysis` / `guard failure` → 立即停，转 `torch-npugraph-ex-compile-error-diagnosis`。

## 冗余 tensor move 的成因分类与定位

reinplace 回填未完成有三种成因，分别对应三类产物特征；每种成因再判到具体根因。

### 成因一：输入侧 copy_ 写回 epilogue 未折叠（reinplace_input_mutated_ops）

**产物特征**：末图里 `aten.copy_.default(<placeholder|get_attr>, src)`，第一参是图入参或 `get_attr` 取出的 buffer（典型：KV Cache）。
**触发机制**：`reinplace_input_mutated_ops` 没把这个回写 epilogue 折叠掉。
**日志判据**（grep `debug.log`）：

| grep 关键字 | 含义 | 下一步 |
|------------|------|--------|
| `[_reinplace_input_mutated_ops] mutated input replace candidates:` **未出现** | 没识别到候选 | 查 KV Cache 结构是否真有 `copy_` 回写到入参 buffer（见上文 `graph_pass.py` 路径约定） |
| 出现，但 `can_inplace return False, will skip reinplacing for node:` | 输入的某个 view 在 `copy_` 之前被读，**或多流校验失败**（多流校验两条 pass 共用，见「多流校验」节） | 通常是模型代码结构问题（见上文 `graph_pass.py` 路径约定） |
| `cannot find an inplace op for node` | 该算子无 `_` in-place 变体 | 无 in-place 形式；考虑自定义算子或社区报点（见上文 `graph_pass.py` 路径约定） |
| `reinplace failed, all mutated args(get_item) must have copy epilogues:` | 多原地白名单算子的 epilogue 不齐 | 检查该算子所有 mutated arg 是否都有 `copy_` 回写（见上文 `graph_pass.py` 路径约定） |

### 成因二：auto_functionalized base 未原地化（decompose_auto_functionalized 物化 clone）

**产物特征（失败 pattern，实测 `issue-314.py` 的 `custom_scatter_update`）**：`*_after_decompose_auto_functionalized.txt` 里出现这一串——

```
as_strided(input, ...)              # 把待原地的输入 view 成目标形状
clone(as_strided_...)               # ← ① clone：复制一份输入（因为不能原地）
as_strided(clone_..., ...)          # 再把副本 view 回原形状
custom.<op>.default(as_strided_clone, ...)   # 在副本上做 in-place
...                                 # （可能还有读副本的算子，如 add）
copy_(input, as_strided_clone)      # ← ② copy_：把改完的副本写回原输入
```

即 **`as_strided → clone → as_strided → 自定义 in-place 算子 + 末尾 copy_(input, 结果)`**。一次失败留下**两个 tensor move**：`clone`（多一次分配 + 读）和 `copy_`（多一次写回输入）。

**触发机制**：`auto_functionalized_v2` 的某个 base 没能 reinplace，functionalize 形成的"clone 输入 → 在副本上 mutate → copy_ 写回输入"三件套**没有被折叠**成"直接原地 mutate 输入"（决定记在 `node.meta["only_clone_these_tensors"]`，见上文 `graph_pass.py` 路径约定）。
**关键**：这里的 `copy_` **第一参是原输入、第二参是 clone 出来的副本，源与目标不同源 → 不是 self-copy**，`eliminate_self_copy` **不会**清掉它。所以末图（`*_after_eliminate_self_copy.txt`）里 `clone` 和 `copy_` **都还在**。
**日志判据**：根因多为 **base 或其 view 在算子之后仍被使用（used-later）**，或多流校验失败（多流校验前缀为 `multi_stream_auto_functionalize`，见「多流校验」节）→ 不能原地。
**log 确认**（grep `debug.log`）：
```
possible missed reinplacing opportunities
Total size of missed opportunities          # 含字节级损失估计（见上文 `graph_pass.py` 路径约定）
```
**命中对照**：命中时**没有 clone**——自定义算子直接在输入 buffer 上原地写；末尾的写回退化成 `copy_(input, input 自己的 view)` 这种**同源 self-copy**，被 `eliminate_self_copy` 清掉。所以末图里**看到 `clone`、或看到非同源的 `copy_(input, clone_...)` 残留 = 真没命中**；二者皆无（或只剩被清掉的 self-copy）= 命中。
**处置建议**：定位那个 used-later 的读点（多为模型代码把 base 的另一个 view 留到后面用），改写或拆开。`can_inplace` 逻辑与 `reinplace_and_refine_tensors_to_clone` 都见上文 `graph_pass.py` 路径约定。

### 成因三：中间 out-of-place 算子未转 in-place（reinplace_inplaceable_ops_pass）

**产物特征**：`*_after_reinplace_inplaceable_ops_pass.txt` 里某 out-of-place 或 `*_functional` 算子原样保留，**且其输出后续并未被使用**（本该可转 in-place）。
**触发机制**：`reinplace_inplaceable_ops_pass` 跳过了转换。二选一定根因：

1. **多流校验失败**（最常见的"明明能转却没转"）
   - grep `debug.log`：
     ```
     check stream safety failed for reinplace
     a cross-stream dependency exists without event protection
     ```
     多流校验是两条 pass 都会做的校验（见「多流校验」节）；**本 pass 命中的前缀只有** `[multi_stream_single_reinplace]`（单输入算子）/ `[multi_stream_multi_reinplace]`（npu 多原地白名单）——见上文 `utils.py` 与 `graph_pass.py` 路径约定。
   - 典型：mutated arg 在 `npu_stream_switch` 的另一条流里被使用，且跨流依赖没有 event 保护。
   - 处置建议：给跨流依赖加 event 保护，或确认该多流划分是否必要。
2. **metadata / 别名**（语义上就不能转）
   - dtype/size 不一致或内存重叠（见上文 `graph_pass.py` 路径约定）；`mul_(a, a)` 这类重复 self 参；self 别名了图输入。
   - 处置建议：这类是正确行为，一般无需改；如为 dtype/size 误判可核对模型写法。

### npu 多原地白名单（成因二/三都可能涉及）

下列算子注册为多原地（multi-inplace），走专门路径、**跳过 used-later 检查**（见上文 `graph_pass.py` 路径约定）：

| 算子 | mutated arg | 来源 |
|------|-------------|------|
| `npu_kv_rmsnorm_rope_cache_v2` | `[5, 6]` (k_cache, ckv_cache) | `torchair/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py` |
| `npu_mla_prolog_v3` | `[9, 10]` (kv_cache, kr_cache) | `torchair/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py` |
| `npu_add_rms_norm_v2` | `[0, 1]` (x1, x2) | `torchair/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py` |

白名单算子的残留多因 epilogue 不齐（成因一第 4 行）或多流校验失败（见「多流校验」节），不因 used-later。


## 两条 reinplace pass 的日志逐条释义

> 上面的成因分类按"产物形态"入手；本节按"日志逐条"入手，把两条 pass 在 `debug.log` 里能打出的**每一条**日志，连同触发条件（根因）和性能影响列全。
>
> **多流校验在两条 pass 中都会执行**，单列在「多流校验」节；「reinplace_inplaceable_ops_pass 日志」「reinplace_input_mutated_ops 日志」只列各自特有的日志，遇到多流相关日志回看「多流校验」节。

### 多流校验（两条 pass 都会做的安全校验）

两条 pass 都要先过多流安全校验才允许 reinplace —— `reinplace_inplaceable_ops_pass` 把它作为**第一步安全校验**（不过即跳过本节点）；`reinplace_input_mutated_ops` 把它与 `can_inplace` 一起作为门控（`can_inplace(...) and check_multi_stream_for_*(...)`）。两者共用同一条日志子链：

```
check_reinplace_streams (见上文 `utils.py` 路径约定)
   → verify_cross_stream_event_protected (见上文 `graph_utils.py` 路径约定)   # 算 happens-before/after
   → _verify_and_log (见上文 `utils.py` 路径约定)                             # 打 success / failed
   → 上层 Node[X] check multi-stream is True/False (见上文 `graph_pass.py` 路径约定)
```

校验的核心问题：**被 mutate 的 buffer 的所有 user 里，有没有"跨流且无 event 保护"的读者**。有 → 拦下（保守，避免跨流竞争）。

| 日志（源码文件） | 触发条件 / 根因 | 结果 | 性能影响 |
|---|---|---|---|
| `[Reinplace check_reinplace_streams]Check Node:X with inplace args indices:[i], which are [A] has all_inputs_users:[[...]]` (utils.py) | 进入某候选节点的多流校验；`all_inputs_users` 是**被 mutate 输入 A 的 storage 级 user 列表**（不是节点 X 自己的 user） | — | 信息行；user 列表驱动跨流判定 |
| `[event_protected] No cross-stream users of X, event protection is not required.` (graph_utils.py) | A 的所有 user 都与 inplace 节点同流（`stream_label` 相同） | **通过**(True) | 利好：放行 |
| `[event_protected] All cross-stream users of X are protected by event synchronization.` (graph_utils.py) | 有跨流 user，但都被 `tagged_event_record/wait` 的 happens-before/after 覆盖 | **通过**(True) | 利好：放行 |
| `[event_protected] Cross-stream user Y (stream=S, pos=P) is NOT protected by events for inplace node N.` (graph_utils.py) | 存在跨流 user Y，BFS 后既不在 happens-after 也不在 happens-before | **失败**(False) | **未命中（漏原地化）**：拦下 reinplace。真阳性=真跨流竞争；假阳性=如 `stream_label=None` 的 functional wrapper 被当成"另一条流" |
| `[event_protected] No event record/wait pairs found in graph, cannot verify protection.` (graph_utils.py) | 有跨流 user，但全图没有任何 event record/wait 对可供验证顺序 | **失败**(False) | **未命中**：保守拦下 |
| `[multi_stream_single_reinplace\|multi_stream_multi_reinplace\|multi_stream_auto_functionalize]Current node: X ... check stream safety success ... All the users are [...]` (utils.py) | `verify_*` 返回 True | check_stream=True | 利好：放行进入后续检查 |
| `... check stream safety failed for reinplace ... a cross-stream dependency exists without event protection ...` (utils.py) | `verify_*` 返回 False | check_stream=False | **未命中**：本节点不 reinplace |
| `Node[X] check multi-stream is True/False` (graph_pass.py) | 多流校验最终结果 | True 继续 / False 跳过 | False = **未命中**（最常见的"明明能转却没转"） |

> 三个前缀对应三类候选：`single_reinplace`（单输入原地算子，两条 pass 都有）/ `multi_reinplace`（npu 多原地白名单，两条 pass 都有）/ `auto_functionalize`（auto_functionalized_v2 的 base 决策，仅 `reinplace_input_mutated_ops`）。判定逻辑同一套。

### reinplace_inplaceable_ops_pass 日志（核心 `reinplace_with_multi_stream_check`）

**单个候选节点的判定流水线（执行顺序，任一步不过即 `continue`，本节点不 reinplace）：**

```
① 多流校验 check_stream      ── False → 跳过      （见「多流校验」节）
② metadata 校验（size/dtype/overlap）── 不过 → 跳过
③ repeated self 参（mul_(a,a)）       ── 命中 → 跳过
④ self 是 program input               ── 命中 → 跳过，交给「reinplace_input_mutated_ops 日志」节
⑤ used-later 校验 can_reinplace       ── False → 跳过
⑥ 找到 in-place 变体 → 真正改写 + 更新别名表
```

①（多流校验）的全部日志见「多流校验」节；以下是本 pass 特有日志：

| 日志（源码文件） | 触发条件 / 根因 | 结果 | 性能影响 |
|---|---|---|---|
| `[_reinplace_inplaceable_ops_pass]processing ... for graph: <id>` (graph_pass.py) | pass 开始（`inplace_pass=True`） | — | 无，仅边界标记 |
| `Skipped fx_pass _reinplace_inplaceable_ops_pass for unsupported fx graph <id>` (graph_pass.py) | `torch<2.5` 或内部 reinplace 工具不可用 / pass 内抛异常 | **整个 pass 跳过** | 最严重：**全图零 reinplace**，所有 out-of-place 原样保留 |
| `Node[X] has self_has_wrong_metadata, skip reinplace` (graph_pass.py) | ② self 与输出 size/numel 或 dtype 不一致，或内存重叠（如 `expand().add_()`） | 跳过 | **正确跳过**（语义必需，非性能问题） |
| `Node[X] has repeated self argument, skip reinplace` (graph_pass.py) | ③ 同一 tensor 在 args 里出现 >1 次（`mul_(a,a)`） | 跳过 | **正确跳过**（安全） |
| `Node[X] arg_index[i] mutate program inputs, skip reinplace` (graph_pass.py) | ④ self 的 storage 是 program input（图入参） | 跳过 | **不是性能损失**：图输入的原地化交给「reinplace_input_mutated_ops 日志」节的 `reinplace_input_mutated_ops` |
| `Node[X] arg_index[i] check reinplace is True/False` (graph_pass.py) | ⑤ `can_reinplace = (该 buffer 的后续真实读者集 - view_inverse) == 0` | True 继续 / False 跳过 | False = **未命中（漏原地化，used-later）**：buffer 后面还有人读，原地化会破坏数据 |
| `Node[X] can reinplace, start to update alias maps` (graph_pass.py) | ①~⑤ 全过，找到 in-place 变体 | **reinplace 命中** | 利好：少一次 buffer 分配 + 一次 HBM 写；输入 storage 与输出 union，后续 use 改写到输入 |

### reinplace_input_mutated_ops 日志（`_mutated_input_reinplace`）

处理**图输入 / get_attr buffer 上的 `copy_` 回写 epilogue**（典型 KV Cache），把 "out-of-place + copy_(input, new)" 折叠回 in-place，消掉输入侧 `copy_`。门控 = `can_inplace(...)` **且** 多流校验（见「多流校验」节）。
**这条 pass 同时负责 `auto_functionalized_v2` 的 base reinplace 决策**（与「reinplace_inplaceable_ops_pass 日志」节的 `inplaceable_ops_pass` 无关）：没命中的 base 不在这里物化，而是 later 由 `decompose_auto_functionalized` 展开成 `clone` + 写回 `copy_`（见成因二）。

| 日志（源码文件） | 触发条件 / 根因 | 结果 | 性能影响 |
|---|---|---|---|
| `[_reinplace_input_mutated_ops]processing ... for graph: <id>` (graph_pass.py) | pass 开始（`input_inplace_pass=True`） | — | 边界标记 |
| `[_reinplace_input_mutated_ops] mutated input replace candidates: to_replace_targets={...}, copy_args_to_copy_nodes={...}` (graph_pass.py) | 扫描全图找 `aten.copy_(placeholder\|get_attr, src)`：`to_replace_targets`=被回写进输入的 src，`copy_args_to_copy_nodes`=`(dst,src)→copy 节点` | 两者**为空**=无输入回写候选 | 空 = 没有可折叠的输入 `copy_`（模型本就无 KV 写回时正常） |
| `reinplace failed, all mutated args(get_item) must have copy epilogues: <target>` (graph_pass.py) | 多原地白名单算子的 mutated args **没有全部**带 `copy_` epilogue | 跳过折叠 | **未命中（漏原地化）**：输入侧 `copy_` 残留 |
| `cannot find an inplace op for node <target>` (graph_pass.py) | `to_replace_targets` 里的算子没有 `_` in-place 变体（`_maybe_get_inplace_op` 返回 None） | 跳过 | **未命中**：该算子无原地形式，`copy_` 残留 |
| `can_inplace return False, will skip reinplacing for node: <target>` (graph_pass.py) | `can_inplace` False 或 `extra_check` / 多流校验（见「多流校验」节）False | 跳过 | **未命中**：输入侧 `copy_` 残留 |
| `For node X, attempted to reinplace <A>. We were unable to reinplace <B>; <C> (if non-empty) are possible missed reinplacing opportunities ... : <N> bytes.` (graph_pass.py, `log_inplace_results`) | `auto_functionalized_v2` 路径对各 base 尝试 reinplace 后的汇总 | B/C 非空=有 base 没命中 | **未命中量化**：`C` 非空 + `N>0` = 这些 base 会在 `decompose_auto_functionalized` 阶段被 `clone`（成因二）；`unable to reinplace []` + `0 bytes` = 全部命中 |

**`can_inplace` 返回 False 的根因细分（graph_pass.py）：**

| 根因 | 说明 |
|---|---|
| `get_node_storage(mutated_arg) is None` | 拿不到 storage（如非 tensor），不能原地 |
| list/tuple 里多个 mutated arg 互相 alias | 同组里两个张量共享存储，无法同时原地 |
| placeholder/get_attr 但**没有 copy_ 回写**（`copy_node is None`） | 语义上程序根本没 mutate 这个输入，不能原地 |
| `any_use_of_views_after_node` True | 输入 buffer 的某个 view 在算子之后、copy epilogue 之前**仍被读**（used-later） |
| mutated arg 是某个图输入的 view | 需要更复杂算法，当前直接不允许 |

### 命中 / 未命中 速判（一句话）

- **命中**：`Node[X] can reinplace, start to update alias maps`，或汇总行 `unable to reinplace []` + `0 bytes`。
- **未命中且值得查**：`check multi-stream is False` / `check stream safety failed`（见「多流校验」节）/ `check reinplace is False` / `missed opportunities ... N bytes`(N>0) / 输入侧 `can_inplace return False`。
- **未命中但正确（无需改）**：`mutate program inputs`（交给「reinplace_input_mutated_ops 日志」节）、`self_has_wrong_metadata`、`repeated self argument`。

## 历史定位案例（按需加载）

历史案例只作为当前证据的补充，不替代对本次 Profiling、FX 图、日志和版本信息的核对。

匹配规则：

1. 先确认实际后端，再匹配算子名、FX 结构和日志关键词。
2. 同一分支中的命中条件需同时满足；表中用“或”分隔的内容表示可独立命中的不同分支。
3. 只读取命中的案例，并说明当前问题与历史案例的相同点和差异。
4. 未命中时继续执行通用诊断流程，不强行选择相近案例。

| CASE | 适用范围 | 最小命中条件 | 定位主题 |
|---|---|---|---|
| [CASE-001](references/case-001-pad-noncontiguous-sliceaicore.md) | ACLNN / npugraph_ex Profiling | `aclnnConstantPadNd` 内出现 `SliceAiCore`；Pad 输入非连续 | Pad 内部隐式连续化 |
| [CASE-002](references/case-002-deepseek-v4-view-cast-format.md) | npugraph_ex | `auto_functionalized_v2` 的 mutable base 是跨 dtype View；展开后出现 `clone + copy_` | 跨 dtype Cache View 阻断 Reinplace |
| [CASE-003](references/case-003-mla-prolog-v3-cache-tensormove.md) | GE / Ascend IR 后端边界案例 | 实际后端为 GE/Ascend IR；`MlaPrologV3` 前存在 Cache `TensorMove` | Legacy Functional Converter 搬运 |
| [CASE-004](references/case-004-kimi-k3-state-writeback-clone.md) | Kimi K3 / npugraph_ex | KDA State 尾部批量 `copy_`，或 ShortConv 冻结权重重复产生 `clone` | State 写回与显式连续化 |

## 兜底文档

- `torch-npugraph-ex-knowledge` 中「性能优化」「调试定位」小节
