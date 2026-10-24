# 算子接入与档案推导

本文合并引擎契约、兼容性审计、算子档案推导和接入门禁。技能不预置算子档案；每次都针对用户当前算子仓推导或复核。

## 引擎固定契约

### 输入与用例提取

- 使用 TTK 新版 CSV 列集合；一个 CSV 只能包含一个 `op_name`。
- `extract_shape_params()` 按 QBMM 风格读取 `x1/x2/scale/offset/bias/pertoken_scale` 六个输入槽位，并从前两个输入的末两维提取 M/N/K。
- transpose、format、bias、输出类型和量化事实必须能组成现有 `params` 词汇表。
- 识别适配器可改变识别逻辑，不能改变引擎传入的原始上下文和返回键。

### 切分参数与注入

- 通用字段为 `baseM/baseN/baseK/kAL1/kBL1/scaleKL1/nBufferNum/dbL0C/weightMustHitL2`，可选 `singleCoreK/kL1`。
- 算子档案可声明布局、无效字段与宏映射，不能新增通用字段或改变字段语义。
- kernel type 的位移和宽度由 `references/profiles/<op>/profile.json` 的 `tiling_key` 声明。同一档案内的一个 kernel type 数值只能对应一个归一化模板名。
- 支持强制 vendor/工作树和普通 vendor 的安装侧打桩两种执行方式。
- kernel 参数通过 `TUNE_*` 宏注入并依赖 TTK JIT；`blockDim`、workspace 等启动侧字段不能通过此机制修改。
- 未知字段、只改无效字段、宏未命中或重复命中都必须失败。
- 安装侧打桩不能强制主机侧切换模板，只能扫描基线实际命中且可注入的模板。
- 构建成功不等于算子可发现。`package_gate.required_files` 必须声明动态 Python、注册头、共享库等关键产物。
- vendor 配置、安装侧内核、编译缓存和设备是共享状态，基线、锚点和扫描必须串行。

### 算子档案适配器接口

- `references/profiles/<op>/recognition.py`：`derive_facts(ctx) -> dict`、`validate_case(ctx, input_rules) -> str | None`。
- `references/profiles/<op>/constraints.py`：`check_candidate(candidate, params, anchor, ctx) -> (bool, str)`、`validate_rules_against_anchors(anchors, params, ctx) -> list[str]`。
- 两个适配器必须只使用 Python 标准库、自包含，不读取引擎内部状态。

## 兼容性审计

逐项写入 `<RUN_DIR>/compatibility/compatibility_report.json`，状态只能是 `compatible`、`needs_engine_change`、`not_applicable`，并给出理由和源码位置：

1. 输入数、槽位和属性能否由现有 CSV 解析器解释？
2. M/N/K、batch、broadcast、transpose 与 format 能否映射到现有 `params`？
3. 待调字段是否属于通用字段集，语义是否一致？
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

必须满足 `byte_size == u32_slots × 4`；不同可解析布局不得共用槽位数；真实结果用 `verify_layout_coverage()` 验证。

### 3. 模板与注入链路

从主机侧 capability/选择门禁、内核文件、强制返回点或安装侧稳定锚点确认 `mode × case × template` 是否可达、是否可注入、宏与字段是否一一对应、哪些字段无效。写入 `quant_modes.*.kernel_files/dead_fields` 和 `templates.*.tuning_blocks`。

### 4. 打包、发现与注册

从构建脚本、CMake/manifest、动态 Python、注册头和共享库确认 TTK 发现算子所需产物。写入 `package_gate.required_files`，并用 `validate_vendor_package_gate()` 验收真实安装包；不得手工补安装文件让门禁变绿。

### 5. 约束、SoC 与预算

将规则分为三类：

- **硬约束**：源码机制可证必然越界、漏块、容量超限、对齐非法或不可注入，写入 `references/profiles/<op>/constraints.py`。
- **性能方向**：只表示可能更快或更慢，写入 `references/profiles/<op>/constraints.md`，仅影响候选预算。
- **探索点**：证据不足但潜力高，保留少量实测配额。

SoC 容量、对齐和数据尺寸必须来自目标平台，核心数继续运行时探测。候选下限必须有覆盖要求和时间预算证据。

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
