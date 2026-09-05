---
name: external-cannbot-ops-pypto-op-develop
description: PyPTO 算子 impl 编码手册。coder agent 收到调度时使用，先 per-Phase 累计构建 `{op}_module{k}_impl.py`，最后一个
  Phase 通过验证后 cleanup 整理出 `{op}_impl.py` + `README.md`。基于 Layer A–L 设计规范，配合 `impl_template.py`
  模板生成符合规范的 PyPTO 实现代码。触发关键词：实现算子、写 kernel、编写实现、写 impl、算子编码、code the op、op develop、kernel
  实现。
original-name: pypto-op-develop
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO 算子 impl 实现

基于 Layer A–L 设计规范，生成 PyPTO kernel 实现文件。**仅负责 impl 部分**（Layer G–K）；golden 与 test 由 mathematician / verifier 在另外的 skill 中生成。

> 资料获取统一使用 skill `pypto-docs-search`：按需搜索算子 API 文档、参考实现与 golden 等文件/目录/内容。

> **新方式（per-Phase 阶段构建 + cleanup）**：每次 Phase 调度只生成**一个** `<op>_module<k>_impl.py`，然后停止。所有 Phase M_k 通过验证后，coder 再被调度一次执行 cleanup，把累计 impl 整理成 `<op>_impl.py` 并写 `README.md`。test 文件由 pypto-op-verifier 在 scaffolding step C 和 cleanup 调度时生成。

---

## 所需输入

### 需求规格信息（来自 SPEC.md）

| 字段 | 用途 |
|------|------|
| 算子名称 `<op>` | 文件命名、函数命名 |
| 数学公式 | 理解计算逻辑 |
| 输入 / 输出规格（shape、dtype） | tensor 描述符与 wrapper 数据准备 |
| 支持的数据类型 | impl 类型处理 |
| 精度要求 | 与 verifier 共享，记录到 MEMORY |
| 服务器类型 | 环境兼容性确认 |

### 设计方案信息（来自 DESIGN.md + module_interfaces.yaml）

| 字段 | 用途 |
|------|------|
| API 映射设计 | impl 核心实现 |
| Tiling 策略 | impl tiling 配置（per-stage local，详见 design-format §11c） |
| Loop 结构设计 | impl kernel 逻辑（Layer I 内的 `pypto.loop`） |
| Layers A–L 划分 | 每个模块文件的层级结构 |
| Module 契约（`module_interfaces.yaml`） | 当前 module 的输入 / 输出形状、tile 约定、组合方式 |
| 数值稳定性档案 | 决定每个模块的精度路由（bf16 / fp32），影响 cast |

### 参考实现信息（来自 `<op>_golden.py`）

| 信息 | 用途 |
|------|------|
| stage 标签（`# ===== (A) ... =====`） | 与 impl 中 `pypto_*` 子内核名一一对应 |
| 输入 / 输出契约 | impl wrapper 的 layout 适配 |

如以上信息不足，向 orchestrator 反馈缺失项；不要自行猜测。

---

## 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|----------|
| [templates/impl_template.py](templates/impl_template.py) | impl 文件骨架（Layer G–K），含反模式注释 | 生成每个 `<op>_module<k>_impl.py` 与集成 `<op>_impl.py` 前必读 |
| [references/pypto-kernel-design-format.md](references/pypto-kernel-design-format.md) | Layer A–L 完整设计规范（含 §11 shape 标注、§11b loop(1) 用法、§11c tile 作用域） | 编码前通读，编码中反复对照 |
| [references/execution-constraints.md](references/execution-constraints.md) | PyPTO 框架级约束清单（动态轴、TileShape、Element、loop / cond、回环读写） | 进入实现阶段前必读；编码与自检时反复对照 |
| [references/error-code-troubleshooting.md](references/error-code-troubleshooting.md) | 错误码排查流程与常见错误码速查 | 验证失败时按流程排查 |
| [scripts/environment_prepare.sh](scripts/environment_prepare.sh) | 环境初始化脚本 | 环境准备阶段按需执行 |
| [scripts/list_idle_chip_ids.sh](scripts/list_idle_chip_ids.sh) | 输出当前可用 chip id 列表（兼容 910B / 910C） | 设置 `TILE_FWK_DEVICE_ID` 前执行 |

> golden 模板和 test 模板由 pypto-op-verify 拥有，coder 不直接读取这些模板。

---

## 开发阶段

### 阶段一：环境准备

1. **检查 CANN 是否安装**

```bash
echo ${PATH} | grep cann-9.0.0
```

2. **检查 pto-isa 源码是否获取**

```bash
echo ${PTO_TILE_LIB_CODE_PATH}
```

- 如果路径不存在，执行 `scripts/environment_prepare.sh` 进行环境初始化
- 如果仍不成功，参考 `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/install/prepare_environment.md` 获取 pto-isa 源码并设置环境变量

3. **准备开发资料**：用 skill `pypto-docs-search` 搜索算子的 API 文档、参考实现与 golden 用法。

4. **设置 device_id**

```bash
# 查找空闲 chip id
bash scripts/list_idle_chip_ids.sh

# 根据输出结果设置环境变量（例如输出为 "0 1 3 ..." 时，先选 chip 0）
export TILE_FWK_DEVICE_ID=$(bash scripts/list_idle_chip_ids.sh | awk '{print $1}')
export PTO_TILE_LIB_CODE_PATH=./pto_isa/pto-isa/
```

- 如果脚本无输出，说明当前所有 chip 都被进程占用，不能随意设置 `TILE_FWK_DEVICE_ID`

⚠️ 未设置 `TILE_FWK_DEVICE_ID` 会导致运行时报错："If no NPU environment is available"

如果以上检查未通过，参考 `docs/install` 中的资料完成环境准备。

---

### 阶段二：代码生成（Layer A–L 方式）

**目标**：每次 orchestrator 调度，生成**一个** impl 文件。

| 调度场景 | 输出文件 | 触发条件 |
|---------|---------|---------|
| Per-Phase M_k 调度 | `custom/<op>/modules/<op>_module<k>_impl.py` | orchestrator 设置 `active_module: M_k` |
| Cleanup 调度 | `custom/<op>/<op>_impl.py` + `custom/<op>/README.md` | 所有 Phase M_k 已通过 verifier 验证 |

**禁止**：单次调度内生成多个 impl 文件、提前生成下一个模块、生成测试代码。

**准备工作**（并行读取，同一条消息中发起所有 Read 调用）：
- `templates/impl_template.py` — impl 骨架与反模式注释
- `references/pypto-kernel-design-format.md` — Layer A–L 完整规范
- `references/execution-constraints.md` — 框架约束清单
- `custom/<op>/DESIGN.md` — 设计方案
- `custom/<op>/module_interfaces.yaml` — 模块契约
- `custom/<op>/MEMORY.md` — 共享叙事，确认 active_module 与历史失败原因

**生成顺序（per-Phase 调度）**：

1. 读取 `module_interfaces.yaml` 中 `active_module: M_k` 的契约：输入 / 输出形状、tile 设定、与上游模块的组合方式。
2. 【检查点】基于 `references/execution-constraints.md` 输出**本模块适用约束项**，格式如：

   ```
   【本模块适用约束项】
   - M_k 有 B、S、N 三个动态轴 → 第5节：必须采用 2D + loop 模式
   - M_k 使用 pypto.concat → 第4.8节：仅支持 2-4D
   - M_k 需要 cast → 第4.10节：显式指定 CastMode
   ```

3. 基于 `impl_template.py` 的 Layer G–K 骨架与 `pypto-kernel-design-format.md` 的 Layer A–L 规范，生成 `<op>_module<k>_impl.py`：
   - **Layer G**（cache / bridge）：torch-only，准备 NPU 兼容的 layout
   - **Layer H**（PyPTO 子内核）：每个 `pypto_*` 函数做单一一件事；本地设置 tile 形状（详见 §11c）
   - **Layer I**（kernel 实现）：所有 `pypto.loop` 调用所在层；不放 `set_*_tile_shapes` 全局调用
   - **Layer J**（JIT 入口）：`@pypto.frontend.jit`，纯类型签名 + `runtime_options`，body 一行委托 Layer I
   - **Layer K**（host wrapper）：仅 4 个职责（layout / 分配输出 / 单次调用 JIT / reshape 还原），**禁止 Python `for ... in range(...)` 循环驱动 kernel**
4. 完成后停止；不要继续生成 M_{k+1} 或 test 文件。

**生成顺序（cleanup 调度）**：

1. 读取 `<op>_module<N>_impl.py`（最后一个累积模块）作为基础。
2. **重命名 / 清理 imports / 整合 layers**，使其作为独立的生产级 kernel 阅读：
   - 文件名 `<op>_impl.py`（不带 `_module<k>` 后缀）
   - 删除阶段调试代码、临时 marker
   - import 仅保留必要项
   - Layer A–L 完整性自检
3. 生成 `README.md`（中文）：
   - 算子概述与公式
   - 目录结构
   - 运行方式
   - 验证入口
   - 已知限制
   - 性能指标（来自 DESIGN.md）

> Cleanup 调度中 coder 不生成 `test_<op>.py`，那是 verifier 的职责（用 `test_template.py` + `test_cases.json` 生成）。

---

### 阶段三：测试验证（与 verifier 协作）

coder **不直接运行 test**，由 verifier 在 Phase Stop / Stage Stop 时调用 `detailed_tensor_compare` 进行裁决。

**coder 在编码完成后的自检**：

1. **静态检查**（手动 grep，提交前确认）：
   - `grep -c '@pypto.frontend.jit' <文件>` 应为 1
   - `grep -nE 'for [a-z]+ in range' <文件>` 在 JIT body 内应为 0（Layer K 的 host wrapper 同样禁止）
   - 所有 vec-op 操作数 rank 2–4
   - 没有 bf16/fp16/autocast/half/bfloat16 残留（如该算子规定全 fp32）

2. **环境检查**（一次性，每个 session）：

```bash
python3 build_ci.py -f python3 --disable_auto_execute  # 如未安装 pypto
echo $TILE_FWK_DEVICE_ID  # 必须有值
```

3. **失败时**：将 stderr 中包含 `Errcode: F` / `ErrCode: F` 的错误码原文记入 MEMORY，并按 [references/error-code-troubleshooting.md](references/error-code-troubleshooting.md) 的流程排查。**不要自行修复架构 / 算法层级问题**，那是 architect / debugger 的职责。

⚠️ 有 NPU 卡的情况下，禁止用 `run_mode=sim` 跑验证。

---

## 实现注意点

1. **PyPTO tensor 创建后是未初始化随机值**：使用前先初始化，或者保证先写后读；不要把 `pypto.tensor(...)` 当成已初始化张量使用。
2. **禁止无中生有 op**：实现时只能使用 PyPTO 已支持的 API，遇到缺失能力应回退到 API 探索或设计阶段重新确认。
3. **优先使用 `@pypto.frontend.jit` 写法**：选择最新的非 wrapper 包装写法，参考 `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-frontend-jit.md`，与现有示例和文档保持一致。
4. **golden / impl / test 必须职责分离**：不要把 golden 逻辑、实现逻辑和测试逻辑混写到同一个文件中（golden 与 test 均不得 `import pypto`）。
5. **动态数据范围使用 valid_shape**：当最后一块数据量可能小于固定块大小时，`pypto.view` / `pypto.reshape` 中必须指定 `valid_shape`。
6. **动态循环边界使用 unroll_list**：当循环次数为动态值时，需要使用 `unroll_list`；多层循环嵌套时，最内层使用 `unroll_list`。**Stage 6 之前 `unroll_list` 只能含单一值**（默认 `[1]`）——照搬 DESIGN.md §4 中 Designer 选定的单值，禁止自行扩成多值（如 `[16, 8, 4, 2, 1]`）；多值会触发编译路径爆炸、拖慢编译并使开发流程超时，多值展开调优仅允许在 Stage 7 optimization。
7. **matmul / cube 场景**：必须确认 `set_cube_tile_shapes(...)` 已正确配置，并优先放在使用它的 `pypto_*` 子内核内部（详见 design-format §11c）。具体 tile 值见 DESIGN.md §3.2.5。
8. **输出写回必须显式完成**：使用 `output[:] = ...`、`output.move(...)` 或 `pypto.assemble(..., output)`；不要写 `output = ...`。
9. **动态轴必须显式标注**：所有动态 shape 输入和输出都必须在 Tensor 注解中标成 `pypto.DYNAMIC` / `pypto.DYN`。**禁止** `pypto.Tensor()` / `pypto.Tensor([], dtype)` 这类空注解写法；静态轴写常量整数，动态轴写 `pypto.DYNAMIC`，不可混淆。
10. **声明动态轴时 kernel 必须含真实 `pypto.loop`**：DESIGN.md `dynamic_axes` 非空时，JIT 函数内必须存在遍历动态轴的 `pypto.loop(...)` 调用，trip count 必须来自动态轴（`tensor.shape[i]`、函数参数或其符号表达式）；**禁止**用 `pypto.loop(1)`、`pypto.loop(常量)` 等空循环，或只在注释里写 `pypto.loop` 来充数。
11. **Element 用于固定标量 dtype**：当标量参与计算且 dtype 不能依赖隐式映射时，显式使用 `pypto.Element(dtype, value)`。
12. **避免同图内回环读写**：同一 Tensor 不要在同一图里既 `view` 读取又 `assemble` 回写。
13. **设计方案优先**：如果设计方案中已有 tiling / loop 约束，编码时优先遵循设计方案；不得在 Stage 5 引入性能调优型 tile 分支。
14. **Layer K 严禁 Python loop 驱动 kernel**：chunk 迭代必须放进 Layer I 的 `pypto.loop(NT)` + `pypto.view(..., offsets=[...])`，**不要**在 Layer K 里 `for chunk in range(NT): kernel_npu(...)`。
15. **`pypto.loop(1)` 是 layout-check 逃生口而非默认包装**（详见 design-format §11b）：仅当内核没有其他 `pypto.loop` 且 vector pipe 简单 op 需要满足布局检查时使用；如果已有 `pypto.loop(N)`，禁止再外加 `pypto.loop(1)`。
16. **Tile shape 必须编译期静态**：`set_vec_tile_shapes(...)` 与 `set_cube_tile_shapes([...], [...], [...])` 的每个参数（含 list 元素）必须是 Python int 字面量，或解析到字面量的局部 / 模块级 Assign（如 `D = 128` 后写 `set_vec_tile_shapes(1, D)` 可接受）。**禁止**用 kernel 入参、`tensor.shape[i]`、SymbolicScalar（含 `B = x.shape[0]` 间接绑定）、运行时计算、`Call` 结果等动态值。违反该约束属致命错误。
17. **Layer K wrapper 中 output 必须 `torch.*` 预分配后再传给 JIT**：host wrapper（Layer K）调 JIT 入口前，每个 output buffer 必须用 **torch** 等价物预先开好——`torch.empty / torch.zeros / torch.ones / torch.full / torch.empty_like / torch.zeros_like`，**显式带 `dtype=` 和 `device=`**——再作为参数传给 `@pypto.frontend.jit` 入口。**禁止**在 Layer K 内调用 `pypto.zeros / pypto.empty / pypto.ones / pypto.full`——这些是 JIT-context API，host 调用会 runtime crash（`pypto.zeros((B,N), device=x.device)` → `TypeError: unexpected keyword 'device'`；去掉 `device=` → `F21003 INVALID_TYPE`）。BAD/GOOD 示例：

    ```python
    # ❌ 错误（Matmul_Mish_Mish 实测 bug）
    def matmul_mish_mish_wrapper(x, w, b):
        out = pypto.zeros((x.shape[0], 20), dtype=pypto.DT_FP32, device=x.device)
        matmul_mish_mish_kernel_npu(x, w, b, out)
        return out

    # ✅ 正确
    def matmul_mish_mish_wrapper(x, w, b):
        out = torch.empty(x.shape[0], 20, dtype=torch.float32, device=x.device)
        matmul_mish_mish_kernel_npu(x, w, b, out)
        return out
    ```

    `pypto.zeros` 等 creation API 只能在 Layer H/I（JIT 图内，例如临时 workspace 张量）使用。debugger 修补 host wrapper allocation 报错时，**第一反应应当是改用 `torch.*`**，不要尝试调整 `pypto.zeros` 的关键字参数（这条路不通）。
18. **`pypto.is_loop_begin` / `pypto.is_loop_end` 必须直接写在 `@pypto.frontend.jit` body 内**：包含 `pypto.is_loop_begin(idx)` 或 `pypto.is_loop_end(idx)` 的逻辑必须直接出现在 `@pypto.frontend.jit` 装饰的函数体里。**禁止**把这类逻辑放进辅助函数（例如 Layer I 的 `_<op>_kernel_impl(...)`），再由 JIT body 调用——parser 会在编译期抛出 **`F00002, ValueError: Not concrete value`**，且报错栈不会指向具体行，定位困难。BAD/GOOD：

    ```python
    # ❌ 错误：is_loop_begin 在辅助函数内 → 编译期 F00002，无源行信息
    def _my_kernel_impl(x, y):
        for idx in pypto.loop(N):
            if pypto.is_loop_begin(idx):
                ...

    @pypto.frontend.jit(...)
    def my_kernel_npu(x, y):
        _my_kernel_impl(x, y)

    # ✅ 正确：直接写在 JIT body
    @pypto.frontend.jit(...)
    def my_kernel_npu(x, y):
        for idx in pypto.loop(N):
            if pypto.is_loop_begin(idx):
                ...

    # ✅ 替代方案：辅助函数加 @pypto.frontend.function（仅支持 tensor 参数）
    @pypto.frontend.function
    def _my_kernel_impl(x, y):
        for idx in pypto.loop(N):
            if pypto.is_loop_begin(idx):
                ...

    @pypto.frontend.jit(...)
    def my_kernel_npu(x, y):
        _my_kernel_impl(x, y)
    ```

    **设计含义**：当 DESIGN.md 把 Layer I 设计为独立辅助函数（如 `_<op>_kernel_impl`），如果该 body 含 `pypto.is_loop_begin` / `pypto.is_loop_end`，Coder 必须把整个 body inline 到 Layer J 的 `@pypto.frontend.jit` 函数里，或在 Layer I 上加 `@pypto.frontend.function`。**这是模板 `impl_template.py` 默认 Layer I/J 切分的已知陷阱**。
19. **直接采用 DESIGN.md tile**（Stage 5 默认）：第一次写 `<op>_module<k>_impl.py` 或集成 kernel 时，按 DESIGN.md §3.2.5 的 tile shape 原样落码。**禁止在 coder 阶段擅自引入训练/decode/核利用率等 cube-tile 分支**——性能调优是 Stage 7 的工作。若 DESIGN.md §3.2.5 未填好，交回 pypto-op-orchestrator 而不要猜。

---

## 常见问题与解决方案

### 常见的 7 类错误

1. **BFloat16 转 NumPy 失败**：必须先 `.float()` 再 `.numpy()`
2. **环境变量未设置**：先运行 `bash scripts/list_idle_chip_ids.sh` 确认可用 chip id，再设置 `export TILE_FWK_DEVICE_ID=<空闲 chip id>`
3. **动态轴定义位置错误**：必须在 jit 函数外部定义
4. **Tile Shape 未设置或过小**：matmul 前必须调用 `set_cube_tile_shapes`；vec 操作前需要 `set_vec_tile_shapes`；具体值按 DESIGN.md §3.2.5
5. **精度标准不合理**：bfloat16 使用 `atol=0.0001, rtol=0.0078125`
6. **使用 PyTorch 作为 Golden**：使用 NumPy 实现 golden 函数时，bfloat16 数据类型转换不够准确；golden 必须独立在 `{op}_golden.py`，使用纯 torch 实现
7. **SymbolicScalar 用作 list 索引报错**：`TypeError: list indices must be integers or slices, not SymbolicScalar`。原因：`pypto.loop` 返回的是编译时符号值，不是 Python runtime 对象。解决方法：使用 tensor slice 或 `pypto.view`/`pypto.assemble` 构建数据流。

### 多动态轴算子的关键陷阱（8-13）

8. **4D 多 DYN 轴直接 matmul 报错**：当 tensor 有 2 个及以上 DYN 维度时，matmul 编译期无法确定 shape（显示为 -1）。**必须**采用 "2D reshape + 嵌套 loop + concrete tile" 模式，参考实现用 skill `pypto-docs-search` 搜索 attention 类算子范本（如 glm_attention 之类）
9. **pypto.view 的 shape 参数传入 SymbolicScalar**：`shape` 参数只接受 Python int，SymbolicScalar 只能用在 `offsets` 和 `valid_shape` 中。Python 切片 `tensor[sym:sym+1]` 同样不可用
10. **inplace=True 用在函数输出参数上**：`pypto.reshape(..., inplace=True)` 的输出不能是 kernel 的输出参数，否则产生静默 NaN 错误。在 wrapper 层提前 reshape，kernel 内直接引用
11. **漏设 set_vec_tile_shapes**：每次 matmul / mul / cast / sum 前都必须设置 tile shapes，维度数必须匹配操作数。在多 loop 嵌套中极易遗漏
12. **梯度算子跨维度累加**：当多个输出在不同维度累加时（如 dQ 沿 S2 累加、dK/dV 沿 S1 累加），使用**两趟分离计算**，避免跨 loop 的读写依赖
13. **pto-isa 版本与 CANN 不匹配**：如遇头文件找不到（如 `TROWARGMAX`），切换到源码版 pto-isa

### 错误处理

| 场景 | 处理方式 |
|------|----------|
| 模板占位符替换不完整 | 检查生成文件中是否残留 `{op}` / `<op>` / `<suffix_k>` 字面量，定位并修正 |
| import 失败（找不到 impl/golden） | 确认文件已生成且在同一目录，wrapper 名为 `<op>_module<k>_wrapper` |
| 编译或执行超过 10 分钟且卡住 | 中断并杀掉相关进程，通知 verifier，记入 MEMORY 并升级 debugger |

---

## Checklist

提交一个 `<op>_module<k>_impl.py` 或 `<op>_impl.py` 之前，确认 ALL of the following：

**结构与模板**
1. 文件以 `impl_template.py` 为骨架，包含 Layer G / H / I / J / K（按需省略时，在 MEMORY 中说明）。
2. wrapper 名为 `<op>_module<k>_wrapper`（per-Phase 调度）或 `<op>_wrapper`（cleanup 调度）。
3. 仅有 1 个 `@pypto.frontend.jit` 装饰器（项目惯例：每个 .py 文件 1 个 JIT）。

**Layer K（host wrapper）**
4. 没有 `for ... in range(...)` 循环。
5. 仅做 4 件事：layout 适配 / 分配输出 / 调一次 JIT / reshape 还原。

**Layer I + Layer H**
6. 所有 `pypto.loop(...)` 调用都在 Layer I；Layer K 内零 `pypto.loop`。
7. Tile shape 设置遵循 design-format §11c：单 stage 用全局，多 stage 各 stage 局部设置；具体值按 DESIGN.md §3.2.5。
8. 没有冗余的 `pypto.loop(1)` 包裹真实循环。

**类型与签名**
9. 所有 tensor 参数有 `pypto.Tensor[...]` 注解，动态轴显式 `pypto.DYNAMIC`。
10. tensor 参数排在非 tensor 参数之前。
11. JIT body 内零 `return`。
12. 输出写回用 `[:]` / `move()` / `assemble()`，不用 `out = expr`。

**约束清单自检**
13. 已对照 `references/execution-constraints.md` 输出本模块约束项清单，并记入 MEMORY。
14. 没有 bf16 / fp16 / autocast / `.half()` / `.bfloat16()`（如该算子是 fp32 算子）。

**与设计 / 上游产物的一致性**
15. `module_interfaces.yaml` 中 `active_module: M_k` 的契约（输入 / 输出形状、tile）已严格对应。
16. Layer H 中的 stage 名（`pypto_stage_alpha` 等）与 `<op>_golden.py` 的 stage 标签一致。

提交后由 verifier 用 `detailed_tensor_compare` + prefix-eval `--up-to-module k` + layout 检查裁决；FAIL 时将 failure_category 记入 MEMORY，等待 debugger 给出补丁建议。
