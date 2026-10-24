# Lint 门禁规则速查（pypto-op-lint）

post-edit / stop 门禁报 `[OLxx][Sx]` 时在此查规则语义；严重级别 S0/S1 会 block，S2/S3 为提醒。

## OL21 测试级别命名（最容易踩的坑）

test 文件必须同时存在 Level 0 与 Level 1 两级测试函数。判定只看**顶层测试函数名**是否包含以下子串：

- level0：`_l0` 或 `level0`
- level1：`_l1` 或 `level1`
- 备选：源码中出现 `功能_P0` / `func_p0` / `test_p0` 视为 level0，`性能_P0` / `perf_p0` 视为 level1

```python
# ✅ 合规
def test_add_l0_basic(): ...
def test_add_l1_basic(): ...
def test_add_level0_basic(): ...   # level0 字样同样认可

# ❌ 不合规（OL21 FAIL：缺少测试级别）
def test_add_basic(): ...          # 无级别标记
def test_add_p0(): ...             # p0 不等于 level0
```

## supported_dtypes 语义（OL30）

`SPEC.md` front matter 的 `supported_dtypes` = **P0 输入/输出 tensor 的 dtype 集合**，应与 `REQUIRE.md` 声明一致：

- 权重 buffer（如量化权重 int8）与中间计算 dtype（如中间累加 float32）**不写入** `supported_dtypes`；
- Stage 5/6 的 OL30 要求其中每个 dtype 都在测试文件中实际覆盖；
- Stage≥2 后 SPEC.md 冻结，但**仅减少 `supported_dtypes` 的 Edit 会被放行**（冻结豁免），无需回退 Stage 1。

## 桥接文件豁免（{op}_pypto_impl.py）

benchmark 桥接文件 `{op}_pypto_impl.py`（纯 torch adapter，含 `ModelNew`）不套用 kernel impl 规则（OL01/OL07/OL08 等）：文件名匹配 `*_pypto_impl.py` 且 AST 判定未导入顶层 `pypto` 包时自动豁免。不要在桥接文件中补 `import pypto` 来“修复” OL07——那反而违反桥接规约。

## D1 框架约束合规

| 规则 | 级别 | 适用阶段 | 语义 |
|------|------|----------|------|
| OL01 | S0 | 5,6 | kernel 函数必须有且仅有一个 @pypto.frontend.jit 装饰器 |
| OL02 | S1 | 5,6 | 输出写回必须用 [:]/move()/assemble()，禁止 out = expr |
| OL03 | S1 | 5,6 | kernel 函数不能有 return 语句 |
| OL04 | S1 | 5,6 | @pypto.frontend.jit 入口及其同文件可达的 Layer I/H helper 中必须调用 set_vec_tile_shapes 或 set_cube_tile_shapes；允许 JIT 入口保持一行委托 |
| OL05 | S1 | 5,6 | kernel 张量参数必须有 pypto.Tensor 类型注解 |
| OL06 | S1 | 5,6 | kernel 内禁用 Python 原生 min()/max() |
| OL07 | S0 | 5,6 | impl 文件必须且只能使用正规 `import pypto` 作为 PyPTO 导入；禁止 alias、import pypto.frontend as ... 与 from-import |
| OL08 | S1 | 5,6 | wrapper 函数必须导出且以 _wrapper 结尾 |
| OL23 | S2 | 5,6 | 所有算子的 impl 应显式说明是否需要 loop；若未检测到 loop 相关结构则给出提醒 |
| OL25 | S1 | 5,6 | Tensor 注解禁止空注解：pypto.Tensor() / pypto.Tensor([], dtype) 一律 FAIL；只缺 dtype 时 WARN |
| OL26 | S1 | 5,6 | JIT 函数中张量参数必须在非张量参数之前 |
| OL28 | S2 | 5,6 | sigmoid/softmax/sin/cos 仅支持 DT_FP32，非 FP32 dtype 时警告 |
| OL29 | S2 | 5,6 | Tensor 注解的 shape 中应声明 pypto.DYNAMIC/pypto.DYN 维度 |
| OL37 | S3 | 5,6 | design 与 impl 的关键中间变量命名应具备可追溯性（重合较少时给出信息提示） |
| OL45 | S0 | 5,6 | Layer K (host wrapper / host_wrapper / *_wrapper) must NOT contain `for ... in range(...)` calling the kernel per chunk; chunking belongs inside _kernel_impl as pypto.loop(N) + pypto.view offsets |
| OL46 | S2 | 5,6 | pypto.loop(1) wrapper is only valid when no other pypto.loop(N) exists in the same scope; redundant pypto.loop(1) around an inner pypto.loop(N) is forbidden |
| OL47 | S3 | 5,6 | When _kernel_impl calls 2+ pypto_* sub-kernels, prefer setting tile shapes inside each sub-kernel (per-stage local tiles) rather than once at the top of _kernel_impl |
| OL48 | S0 | 5,6 | set_vec_tile_shapes / set_cube_tile_shapes 的所有 tile 参数（含 list 元素）必须是编译期可知的 Python int 字面量，或解析到字面量的模块级 / 函数局部常量 Assign；禁止使用 kernel 入参、tensor.shape[i]、SymbolicScalar、运行时计算等动态值。此外 set_cube_tile_shapes 的 m/k/n 每轴必须是 2 元素 list [L0, L1]（不能是单元素 [L0] 或 3+ 元素），且 L0/L1 为字面量时须满足 0 < L0 <= L1 且 L1 % L0 == 0（见 docs/zh/api/config/pypto-set_cube_tile_shapes.md） |
| OL49 | S1 | 5,6 | unroll_list 只能出现在最内层 pypto.loop（即其 body 内不再嵌套其他 pypto.loop）；外层 pypto.loop 携带 unroll_list 会触发编译路径爆炸或寄存器拷贝 pass 引起的精度异常 |
| OL52 | S1 | 5,6 | pypto.view(t, shape=[...], offsets=[...], valid_shape=[...]) 的 shape/offsets/valid_shape 必须同 rank（list 长度一致）。pypto.view 是同 rank 的 sub-view 抽取 API，不是 reshape |
| OL55 | S0 | 4,5,6 | 禁止使用 PyPTO 中不存在的 `pypto.<attr>` — 通过对比 AST 中的属性访问与 `dir(pypto)` 校验, 在 DESIGN.md (代码块内) 与 `<op>_impl.py` / `modules/<op>_module*_impl.py` 上 post-edit 即时拒绝, 防止 typo (如 `pypto.empty` / `pypto.empty_like`) 走到 runtime |
| OL56 | S0 | 4,5,6 | Stage 6 之前 pypto.loop 的 unroll_list 只能含单一值（默认 [1]，有依据时可用其它单值）；含 2 个及以上值会触发编译路径爆炸、拖慢编译并使开发流程超时。多值展开调优仅允许在 Stage 7 optimization。在 DESIGN.md（```python``` 代码块）与 <op>_impl.py / modules/<op>_module*_impl.py 上 post-edit 即时拒绝 |
| OL57 | S0 | 5,6 | @pypto.frontend.jit 图代码（kernel 本体 + 其调用到的所有函数 / 含 pypto 算子的函数）内允许 pypto.loop / pypto.loop_unroll / for...in range(...) 循环；禁止 while 和非 range 的 Python for（及含 pypto 算子的推导式）。迭代可用 pypto.loop（迭代间有依赖时加 submit_before_loop=True）或 for...in range(...)（编译期全展开）。Layer K host wrapper 的 kernel 驱动循环由 OL45 管辖 |
| OL58 | S0 | 5,6 | Layer K host wrapper 内禁止调用 pypto.zeros / pypto.empty / pypto.ones / pypto.full —— 这些是 JIT-context creation API, 在 host 上下文调用会 runtime crash (`device=` kwarg 不接受, 或 `F21003 INVALID_TYPE`)。output buffer 必须用 torch.empty / torch.zeros / torch.empty_like 等 torch allocation API (显式 dtype= 与 device=) 预先分配, 再作为参数传给 @pypto.frontend.jit 入口。每个传入 JIT 入口的 Name 参数必须解析到 wrapper 参数, 或 torch.* allocation, 或对 wrapper 参数的 torch 变换 (例 .reshape / .contiguous) |

## D2 工件完整性与流程合规

| 规则 | 级别 | 适用阶段 | 语义 |
|------|------|----------|------|
| OL09 | S1 | 1 | SPEC.md 必须通过结构化章节校验（数学公式、输入输出规格、精度要求）和 front matter schema 校验（p0_shapes/supported_dtypes/tolerance 格式） |
| OL10 | S1 | 1 | API_REPORT.md 必须通过结构化章节校验（API 映射、约束、Tiling）；Stage 1 产出，Stage 3 设计阶段读取 |
| OL11 | S2 | 2 | Stage 2 完成时校验 {op}_golden.py 可导入（作为 Stage 3/4/5 精度基线） |
| OL12 | S1 | 3,4 | DESIGN.md 结构化章节校验（计算图、Tiling、验证方案），Stage 3 完成时即校验，Stage 4 复查兜底 |
| OL13 | S1 | 5 | 集成成品三件套完整: {op}_impl.py, test_{op}.py, README.md (Stage 5 cleanup) |
| OL14 | S1 | 6 | 进入 Stage 6（结构验证）需 Stage 5（含 cleanup）已 completed |
| OL24 | S1 | 1,2,3,4,5,6,7 | .orchestrator_state.json 结构合法 |
| OL39 | S1 | 5,6 | strict 模式下文档必须包含 front matter |
| OL40 | S1 | 5,6 | strict 模式下 front matter 必填字段必须完整 |
| OL41 | S1 | 5,6 | 代码工件禁止包含 lint/门禁输出文本污染；覆盖顶层产物和 modules/<op>_module*_impl.py / _golden.py / test_*.py |
| OL44 | S1 | 5 | Stage 5 active Phase M_k requires modules/<op>_module<suffix_k>_impl.py + _golden.py + test_*.py three-set |
| OL54 | S1 | 5 | complete_phase 时，MEMORY.md 中必须有 `## Phase M_k self-review` 章节，且 6 个必须项（signature 一致 / output 写出 / view rank / inventory 更新 / 无 for-range / JIT exactly once）全部标记为 `- [x]` |
| OL59 | S1 | 2 | Stage 2 完成时 GOLDEN_PERF_REPORT.md 必须存在且包含 Op Performance section（由 pypto-golden-generate/scripts/profile_golden.py §15 生成） |
| OL61 | S1 | 5,6 | Experience Preflight 门禁（Stage 5/6，Coder 写 impl 前创建并自检；Stage 1-4 不触碰 MEMORY.md）: (1) MEMORY.md `## Experience Preflight` section 存在且非占位符; (2) 格式合规（markdown checklist 非表格，≤20 条，[-] 项须有 ⚠️ 待验证 注释）; (3) 所有 [-] 待验证项必须消除（改为 [x] 或标注 ✅ 已知风险）; (4) 对 impl.py 做 AST 扫描检测 4 类反模式（F4 非法 cast 路径/F2 Element 双重包装/F1 scalar 首参/F8 zeros dtype 位置错误） |

## D3 三文件分离

| 规则 | 级别 | 适用阶段 | 语义 |
|------|------|----------|------|
| OL15 | S1 | 2,3,4,5,6 | golden 文件须为纯 torch 规范化实现：禁止 import pypto，禁止 `.T` / `.t()`（须用 torch.transpose） |
| OL16 | S1 | 5,6 | impl 文件不应导入 golden 模块 |
| OL17 | S1 | 5,6 | test 文件不应包含 kernel 实现代码 |
| OL18 | S1 | 5,6 | test 文件必须从 impl 和 golden 分别导入 |

## D4 测试规范

| 规则 | 级别 | 适用阶段 | 语义 |
|------|------|----------|------|
| OL19 | S1 | 5,6 | test 必须使用 assert_allclose 或 detailed_tensor_compare 做精度比对，禁止手写 assert max_diff |
| OL20 | S1 | 5,6 | test 必须处理 TILE_FWK_DEVICE_ID 环境变量并调用 set_device |
| OL21 | S2 | 5,6 | test 必须有 Level 0 和 Level 1 两级测试函数 |
| OL22 | S2 | 5,6 | test 应设置 torch.manual_seed 保证可复现 |
| OL42 | S1 | 5,6 | NPU 环境可用时，test 禁止硬编码 sim 模式（default='sim' 或 run_mode='sim'） |
| OL60 | S0 | 5,6,7 | test_<op>.py 从 *_impl 模块 import 并实际调用的入口函数, 必须在同文件可达调用链中能到达至少一个 @pypto.frontend.jit 函数。否则测试跳过 PyPTO 内核 (flash_kda 类失败模式: test 调用纯 PyTorch 入口, 绕过 @jit)。 |

## D5 跨文件一致性

| 规则 | 级别 | 适用阶段 | 语义 |
|------|------|----------|------|
| OL30 | S2 | 5,6 | spec.md 声明支持的 dtype 必须在测试文件中覆盖 |
| OL31 | S1 | 5,6 | design.md 动态轴声明须与 impl Tensor 注解中的 DYNAMIC 一致 |
| OL32 | S2 | 5,6 | spec.md 精度容差(atol/rtol)须与 test 文件一致 |
| OL33 | S2 | 5,6 | golden 与 wrapper 的必需参数数量应一致（避免调用接口不兼容） |
| OL34 | S1 | 5,6 | spec.md P0 测试配置 shape 应在 test 文件中覆盖 |
| OL43 | S1 | 5,6 | DESIGN.md 声明动态轴时，每个相关 impl 的 jit 函数必须包含遍历动态轴的真实 pypto.loop；该规则为硬性门禁，不得因 NPU 运行通过而跳过 |
| OL50 | S1 | 5,6 | Layer K wrapper 的显式参数必须与 eval/module_interfaces.yaml 的 primary_inputs 声明顺序（以及对应 module 的 inputs[*].name source=primary）完全一致；生产 ABI 不暴露 runtime/debug 参数，调试扩展走 **kwargs 或 _debug/ 产物 |
| OL51 | S1 | 5,6,7 | OL51.a 数量层: 对 eval/module_interfaces.yaml 声明的输出数 N，impl 内至少存在 N 个真实写出点 (pypto.assemble / out.move / out[:]=)。OL51.b 非平凡写入层: JIT 写出的 output 不得是平凡 pass-through (直接复制输入 / pypto.zeros / 浅层 reshape)，须由真实 pypto compute op 产生。 |
| OL53 | S2 | 5,6 | MEMORY.md → Golden function inventory 章节的所有行都必须标记为 Status ✅（仅在 complete_stage 时严格判定；complete_phase 允许未着手 module 行为 ❌） |
| OL62 | S0 | 5,6,7 | <op>_impl.py 内 torch 仅可用于 layout/alloc/cast/reshape；任何 torch 张量算术 (torch.matmul / .exp / .sum / `@` / F.* 等) 即 FAIL。算子数值计算必须在 @pypto.frontend.jit 图内用 pypto.* op 实现，host (wrapper / 非 JIT helper) 只允许打包/分配/布局/dtype 转换。封堵 dummy-JIT 全谱 (dead JIT+host torch / 小 JIT+torch 主计算 / torch 旁路再计算)。AST 检测，忽略注释与 __main__/_validate 等自测块 |
| OL63 | S1 | 7 | Stage 7 完成门禁（仅 PANKO 路径）：路径取自 .orchestrator_state.json 的 stage7_tuning_mode, 由 start_stage(stage=7, tuning_mode=...) 记录。"stepwise" 或未设置时 SKIP, 分步骤调优照常 complete_stage(7)。"panko" 时 complete_stage(7) 前 optimization/search_state.json 的 progress.stop_reason 必须非空——即 panko_harness.py 的 stop() 已判定真正停止。编排器是薄 relay, 不得自行判断"够好 / 差不多"提前结束优化。search_state.json 存在却未登记为 panko 时同样 FAIL |
