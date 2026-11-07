# 算子接入与档案推导

本文合并引擎契约、兼容性审计、算子档案推导和接入门禁。技能不预置算子档案；每次都针对用户当前算子仓推导或复核。

## 引擎固定契约

### 输入与用例提取

- 使用 TTK 新版 CSV 列集合；一个 CSV 只能包含一个 `op_name`。
- 引擎只解析 CSV 列；输入槽位、可选输入、M/N/K、转置属性、格式、bias、batch 和模式由当前算子的识别适配器解释。
- `derive_facts(ctx)` 必须返回 `M/N/K`、`x1_dtype/x2_dtype/scale_dtype`、`bias`、`trans_a/trans_b`、`x1_format/x2_format`、`case_class/batch_count`、`quant_mode`（当前档案的计算或量化模式名，用于路由）。无 scale 时返回 `scale_dtype=None`，无 bias 时返回 `bias="none"`。`x1/x2` 表示参与矩阵乘的 A/B，而非固定槽位。
- 输出类型从 `output_dtypes` 解析；识别适配器可返回额外事实，折叠、广播或量化相关事实由实际源码决定。
- 不强制通用 `attributes.dtype` 或某个量化模式的输入槽位规则；这些检查写入 `validate_case`。

两输入算子直接使用真实输入和属性：在 `input_rules.slot_order` 记录接口顺序，由 `derive_facts` 根据源码解释输入角色与转置属性；没有 scale 或 bias 时返回上述空值。六输入量化算子同样由适配器解释 scale、bias 及其余可选输入。

`validate_case` 应按当前算子接口核对 `input_shapes/input_dtypes/input_formats` 的槽位数、可选性和模式约束。`slot_order` 是适配器可读取的接口声明；引擎将原始列传入适配器，事实推导和校验都使用同一份 `input_rules`。

### 切分参数与注入

- 基线和锚点保留当前布局声明的全部字段，不要求固定字段名，不给缺失字段补默认值。
- 可注入字段由档案中的 `TUNE_*` 声明和赋值映射决定，候选必须完整设置目标模板的宏字段。真实字段语义及无效字段由当前源码确认。
- 现有容量健康度模型仍使用 `baseM/baseN/baseK/kAL1/kBL1/scaleKL1/nBufferNum/dbL0C`；缺少这些字段时记录无法估算，不能据此丢弃实际切分数据。
- kernel type 的位移和宽度由 `references/profiles/<op>/profile.json` 的 `tiling_key` 声明。不同模式可复用数值；模式内按 `kernel_type_map` 映射，若同模式在不同用例类别中含义不同，声明 `kernel_type_map_by_case_class`。当前模式与类别内必须唯一；未知 key 或上下文不足时保留 `unknown`，不得按槽位数或字段值猜模板。旧档案也必须显式补齐位段声明。
- 支持强制 vendor/工作树和普通 vendor 的安装侧打桩两种执行方式。
- kernel 参数通过 `TUNE_*` 宏注入并依赖 TTK JIT；`blockDim`、workspace 等启动侧字段不能通过此机制修改。
- 未知字段、只改无效字段、宏未命中或重复命中都必须失败。
- 安装侧打桩不能强制主机侧切换模板，只能扫描基线实际命中且可注入的模板。
- 构建成功不等于算子可发现。`package_gate.required_files` 必须声明动态 Python、注册头、共享库等关键产物。
- vendor 配置、安装侧内核、编译缓存和设备是共享状态，基线、锚点和扫描必须串行。

### 算子档案适配器接口

- `references/profiles/<op>/recognition.py`：`derive_facts(ctx) -> dict`、`validate_case(ctx, input_rules) -> str | None`。
- 若需调度指标，在 `profile.json.scheduling` 声明 `model/evidence`，在 `recognition.exports` 添加并实现 `derive_schedule(ctx) -> dict`。上下文含 `params/tiling_data/template/aic_num/soc`；返回当前模型适用的任务数、K 循环或核利用率，不得覆盖容量指标。
- `references/profiles/<op>/constraints.py`：`check_candidate(candidate, params, anchor, ctx) -> (bool, str)`、`validate_rules_against_anchors(anchors, params, ctx) -> dict[str, list[str]]`。
- 识别上下文包含原始 `row`，解析后的 `input_shapes/input_dtypes/input_formats/attributes/y_dtype`，以及档案的 `input_rules/fold_rule/quant_modes/case_classes`。`validate_case` 还接收已推导的 `params` 事实。
- 两个适配器必须只使用 Python 标准库、自包含，不读取引擎内部状态。

锚点自检返回 `{"errors": [...], "warnings": [...]}`：`errors` 是规则与锚点矛盾等阻断问题；`warnings` 是规则不适用、跳过某项检查等提示。例如 afull 锚点不满足全载条件而跳过相关规则，应写入 `warnings`。`collect_anchor_probe(strict=True)` 打印提示，只因错误、缺模板或缺少 blockDim/layout 而阻断。兼容旧适配器返回 `list[str]`，但列表里的每一项都视为错误，不得混入提示。

## 兼容性审计

逐项写入 `<RUN_DIR>/compatibility/compatibility_report.json`，状态只能是 `compatible`、`needs_engine_change`、`not_applicable`，并给出理由和源码位置：

1. 输入数、槽位和属性能否由现有 CSV 解析器解释？
2. M/N/K、batch、broadcast、transpose 与 format 能否映射到现有 `params`？
3. 待调字段能否由档案声明映射到唯一宏注入点，语义是否有当前源码证据？
4. 主机侧能否产出可解析锚点，布局、kernel type 和 key 位段能否声明？
5. 内核是否有稳定、唯一的宏注入点，修改后能否经 JIT 生效？
6. 强制 vendor 或安装侧打桩是否适用，能否用实际 tiling key 核对模板？
7. 基线、精度插件、结果 CSV 与排名逻辑能否直接复用？
8. SoC 常量是否来自目标平台？

任一项为 `needs_engine_change` 时，记录缺口、建议改动点和所需测试，停止本轮调优。不生成可执行档案、不构建、不跑基线、不修改安装目录。

## 创建算子档案

仅在没有有效档案或版本对账失败时执行：

```bash
python3 <SKILL>/scripts/create_profile_scaffold.py \
  --op <op_name> --repo-rel <repo_relative_op_dir> \
  --repo-commit <full_op_repo_commit>
```

命令生成 `references/profiles/<op_name>/` 四件套：

| 文件 | 内容 |
|---|---|
| `profile.json` | 身份、版本、SoC、输入、布局、模式、模板、注入点和打包闸门 |
| `recognition.py` | 输入、模式、batch 和用例分类 |
| `constraints.py` | 硬约束和锚点自检 |
| `constraints.md` | 硬约束证据、候选方向和探索点 |

生成物带 `_scaffold: true`，执行器会拒绝加载。只有下述推导、证据和测试全部完成后才能删除该字段。

## 源码推导清单

### 1. 输入与维度

从算子定义、校验器、主机侧 shape 解析和 CSV 样例确认槽位顺序、可选性、数据类型、格式、M/N/K、batch/broadcast 和模式识别分支顺序。字段写入 `references/profiles/<op>/profile.json` 的 `input_rules`、`fold_rule`，识别逻辑写入同目录的 `recognition.py`。

### 2. tiling key、分类与布局

从 tiling key 定义、主机策略、kernel dispatch 和全部 tiling data 结构体确认位段、路径矩阵、字段偏移、`byte_size`、`u32_slots` 和路径切换条件。写入 `tiling_key`、`case_classes`、`struct_layouts` 和 `quant_modes`。

必须满足 `byte_size == u32_slots × 4`。槽位数不是结构体身份：同长度布局通过 `struct_layouts.<name>.when` 声明选择条件，例如 `{"quant_mode": ["mode_a"], "case_class": ["plain"], "kernel_type": [2]}`。条件按 AND 匹配；值可以是单值或列表。可使用识别适配器产出的事实，以及解码后的 `kernel_type/template`。

从结构体定义、host 写入路径和 kernel 分发条件推导字段偏移及选择条件，不能从旧布局长度套用。没有条件的声明只适用于能唯一判定的布局。零匹配或多匹配都阻断解析；真实结果用 `verify_layout_coverage()` 核对实际布局选择和解包。

### 3. 模板与注入链路

从主机侧 capability/选择门禁、内核文件、强制返回点或安装侧稳定锚点确认 `mode × case × template` 是否可达、是否可注入、宏与字段是否一一对应、哪些字段无效。写入 `quant_modes.*.kernel_files/dead_fields` 和 `templates.*.tuning_blocks`。

多策略算子还须核对注册顺序或 priority、前序策略的 `IsCapable` 和实际选择入口。目标策略 `IsCapable=true` 只代表它可参与选择；`GetKernelType/GetTilingKey` 影响已选策略的内核路由，不能证明 host 一定选择目标策略。档案的强制补丁应覆盖实际选择点，或限定到能命中目标策略且经探针验证的用例范围。需要的选择层补丁无法通过现有声明表达时，兼容性审计应标记 `needs_engine_change` 并停止本轮。

探针命中竞争策略、与已声明的强制模板不符时，应复核路由补丁或兼容范围。不能将这个补丁失败改写为 `not_applicable` 来绕过缺失锚点门禁；真正的模板不适用须有当前源码和用例语义依据。

`dead_fields` 可以是字段名列表，也可以是 `{字段名: 原因}` 字典；两者都按字段名过滤，死字段不参与宏更新和扫参注入。

`kernel_files` 的字符串路径或字典中的 `install` 支持绝对路径和包内相对路径（如 `op_impl/ai_core/tbe/.../kernel.h`）。相对路径从当前 vendor 安装根目录下的包目录解析：worktree 模式使用模板对应的 `vendor_install_dir`，install-stub 模式使用 normal 包 `custom_normal_nn`。扫参的包选择与路径解析一致；`repo_rel` 仍是源码仓内路径。

install-stub 将 `tune_macro_define_block` 放在文件开头、所有条件编译和 include 之前，确保宏在各输入类型分支中可见；调参代码块仍按 `sed_anchor/insert_mode` 注入。宏定义块应只声明当前档案的调参宏，注入后的可见性须覆盖该模板支持的条件分支。

### 4. 打包、发现与注册

从构建脚本、CMake/manifest、动态 Python、注册头和共享库确认 TTK 发现算子所需产物。写入 `package_gate.required_files`，并用 `validate_vendor_package_gate()` 验收真实安装包；不得手工补安装文件让门禁变绿。

`required_files[].contains` 默认按 `contains_match="whitespace"` 检查：换行、缩进等连续空白合并为一个空格，反斜杠续行按逻辑行拼接；标识符、数值和符号仍须一致。尽量选注册名等稳定片段。确实需要逐字符检查的文本，显式声明 `contains_match="literal"`。这是内容存在性检查，不证明代码语义或运行路由正确。

显式构建会重新编译全部目标包，不记录或校验构建指纹。`.run` 文件名只包含 vendor 名，而 `--ops=<op_name>` 会过滤包内容，因此已有 `.run` 和安装目录不能证明包含当前算子。`start_build()`、`run_build_with_retry()` 和 `run_build_all(build=True)` 都会先删除旧 `.run` 再编译；只有显式选择 `build=False` 才准备工作树并复用安装目录，之后仍须通过安装包门禁。编译与安装检查退出码、产物及完成标记，失败不会因旧安装目录存在而被视为成功。

skill 工作流构建时传入 `run_dir`，先由 `prepare_run_environment()` 备份原环境并分配本轮临时工作树，收尾调用 `cleanup_run_environment()` 恢复。Python 旧接口未传 `run_dir` 时保留原有路径行为，不具备本轮归属和自动回收能力；不得用该接口执行需要自动恢复的技能流程。

### 5. 约束、SoC 与预算

将规则分为三类：

- **硬约束**：源码机制可证必然越界、漏块、容量超限、对齐非法或不可注入，写入 `references/profiles/<op>/constraints.py`。
- **性能方向**：只表示可能更快或更慢，写入 `references/profiles/<op>/constraints.md`，仅影响候选预算。
- **探索点**：证据不足但潜力高，保留少量实测配额。

SoC 容量、对齐和数据尺寸必须来自目标平台。未知 dtype 的尺寸不能默认按 1 字节计算：根据当前类型定义填写 `dtype_sizeof`（每元素字节数，支持半字节）及对应的 `dtype_evidence`。`mx_group_size` 不是通用必填项；现有容量模型在 scale 深度大于 0 时才使用它，适用时必须有分组语义依据。核数在运行时查询，`soc.core_query` 必须声明 `core_type/spec_key/evidence`；从目标 host 的 AIC/AIV 使用方式及当前 CANN 平台接口确认口径，不默认 `AiCore + CORE_NUM` 等于所需计算核数。查询值是目标 SoC 的平台编译规格，不是实时空闲核数；调度适配器还需核对实际启动与分配规则。

CANN Python 路径优先使用 `get_aic_num(soc, cann_path=...)` 显式路径，其次当前 `ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME`，最后 Python 当前可发现的唯一 TBE 包。环境变量冲突或已导入另一安装的 TBE 时停止并对齐环境，不从目录排序挑“最新版”。调度模型从任务分配、batch 合并、K 分片与归约等源码推导，写入 `scheduling` 与 `recognition.derive_schedule`；不要默认任务数为 M/N 网格乘 batch，也不要从某个字段非零判断 StreamK。无证据时省略 `scheduling`，引擎保留无法估算的说明。新建档案默认要求每个模板至少生成 20 条候选；如需调整，候选下限必须有覆盖要求和时间预算证据。

## 接入门禁

全部通过后才能发布性能推荐：

1. 兼容性审计无 `needs_engine_change`。
2. 算子档案可加载，识别和约束适配器正反例通过。
3. 安装包发现与注册链路验收通过。
4. 真实基线的布局与 kernel type 覆盖无缺口。
5. 取得全部可注入模板锚点，且 layout、shape、`blockDim` 和实际模板一致。
6. 硬约束不无解释拒绝主机侧锚点。
7. 每个可扫模板至少一个基线回放候选通过唯一命中、写后回读和实际 tiling key 校验。
8. 至少一个代表用例完成基线、锚点、候选、精度门禁和排名闭环，且安装侧修改已恢复。

支持级别属于当次生成的算子档案及其版本锚点，不属于算子名。目录名包含 matmul 不等于已经支持。
