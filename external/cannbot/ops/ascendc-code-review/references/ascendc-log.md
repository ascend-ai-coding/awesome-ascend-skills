# Ascend C Host 侧日志与错误码规范

```
<适用>
语言: C++
侧别: Host
默认启用: true

领域: true
触发: 必须触发
</适用>
```


## 快速索引

**适用边界**：本文件按 Host C++ 必检。文件级命中后，LOG-1 和 LOG-2 分别核验实际日志或日志包装调用，变更没有这类调用时标注不适用。LOG-6 仍需核验变更的失败出口。Host 调用下层失败时，按 LOG-5 和 LOG-6 核验下层首报责任；设备侧的上报能力及宏用法按对应边界的契约核验。

**验证边界**：静态检视分别说明尚未验证的编译、负向日志和运行时首错唯一性。

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| LOG-1 | 语义可映射时核验正式错误上报入口 | 错误码规范 | 中 | [适用: Host] |
| LOG-2 | 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 | 日志规范 | 中 | [适用: Host] |
| LOG-3 | 禁止新增治理废弃错误码及旧宏 | 错误码规范 | 高 | [适用: Host] |
| LOG-4 | 错误码语义、目标版本注册与 Arglist 必须匹配 | 错误码规范 | 高 | [适用: Host] |
| LOG-5 | 根因层一次首报，包装层仅日志并透传 | 错误码规范 | 中 | [适用: Host] |
| LOG-6 | 失败返回路径需有明确首报责任和可操作诊断 | 错误码规范 | 中 | [适用: Host] |

---

## LOG-1: 语义可映射时核验正式错误上报入口

**严重级别**：中

### 问题描述

本条的唯一错误码基准为 `{skill_base}/../cann-error-code-standard/SKILL.md` 及其 `references/development-standard.md`、`references/documented-error-codes.json`。废弃码、注册契约和首报判据分别由 LOG-3~6 定义，不再在日志规则中维护第二份宏码映射表。

### 错误示例

```cpp
// ❌ 错误：参数错误仍使用已被治理禁用的按 input index 上报码。
OP_LOGE_WITH_INVALID_INPUT_SHAPE(context->GetNodeName(), 0, actualShape, expectedShape);
```

### 正确示例

```cpp
// ✅ 正确：目标版本已确认支持该语义宏及其 EZ0008 注册项；使用稳定参数名。
OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "x", actualShape, expectedShape);
```

### 注意事项

1. 分工与去重：LOG-1 核验日志调用选用的正式上报入口，LOG-2 检查算子名来源。废弃码、语义与注册契约、重复首报和缺失首报分别归 LOG-3 至 LOG-6。同一调用点同一根因由对应 LOG 条款输出意见。已知下层首报后，外层仅日志并透传——过程信息不能无证据认定为正式上报缺陷
2. 反臆测：不能把所有裸 `OP_LOGE` 都判为 EZ9999，也不能把所有带后缀的 `OP_LOGE_WITH_*` 都判为合规——实际宏来源可能是 op_common、nnopbase 或项目包装，须先解析目标 include 链，再判断最终上报叶子

### 专属检视方法

1. 定位被检范围的日志与包装宏，读取上述基准规范，追溯实际宏定义文件、include 链和最终 ErrorManager 入口。源码入口见 [CANN opbase 日志宏](https://gitcode.com/cann/opbase/blob/master/include/op_common/log/log.h)。
2. 语义可映射且目标注册项、宏签名和参数可得性已核验，才建议具体替换；缺少目标版本证据时标记 SUSPICIOUS 并列出缺口，不依据宏名臆测。
3. 禁止推荐基准 `governance_deprecated` 中的 `EZ0001、EZ0002、EZ0003、EZ0005、EZ0006、EZ0007` 及其对应旧宏。
4. 沿用通用假设检验方法。意见给出原调用、真实叶子、根因类别、资料 URL、目标注册/宏路径、Arglist 和建议写法；保持公开返回值兼容。
5. 同一调用点同根因已由 LOG-3~6 报告时合并为一条意见；LOG-2 仍单独检查算子名来源。

---

## LOG-2: 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串

**严重级别**：中

### 问题描述

Host 侧代码打印日志时需要算子名称的，必须从宿主上下文（TilingContext 等）获取，**禁止在代码中自定义一个算子名称的字符串**。算子名获取的三级层级：

| 层级 | 用法 | 判定 |
|------|------|------|
| 优先 | 日志宏直接传 `context`（如 `OP_LOGE(context, ...)`）——框架自动携带算子标识，无需额外获取 | PASS |
| 合规 | `context->GetNodeName()` 显式获取；tiling 类成员在构造/Init 时从 context 获取后复用（如 `opName_`） | PASS，不算报错 |
| 违规 | 自定义算子名称的字符串——出现在第一参数、格式串前缀、或预定义宏 entityName | FAIL |

> **例外：context 判空分支豁免**。当获取到的 context 为空（代码处于 `context == nullptr` / `!context` 判空保护分支内）时，无 context 可传、也无法调用 `GetNodeName()`——该分支内的日志**允许使用算子名字符串**（建议与注册名一致），不判违规。豁免**仅限判空分支体内**；分支外的日志仍须从 context 获取；若存在 context 之外的算子名来源（如 aclnn 实现的入参 `opName`），仍优先使用该来源。

**危害**：

- 算子改名/代码复制到新算子时，硬编码的算子名与实际算子不符，误导问题定位
- 同一算子多文件日志口径不一（有的手写、有的从 context 获取）
- 调试阶段遗留的假算子名（如 `"xxx-test"`）混入生产日志

### 错误示例

```cpp
// ❌ 错误1: 第一参数硬编码算子名（ops-math PR#2195 同类）
OP_LOGE("Pow", "tiling failed, powNum is %ld.", powNum);                     // ❌ 自定义算子名字符串

// ❌ 错误2: 消息里手写算子名前缀——context 已自动携带算子标识（ops-math PR#1767 同类）
OP_LOGE(context, "AddCustom tiling failed, totalNum is %ld.", totalNum);     // ❌ 手动前缀冗余

// ❌ 错误3: 调试遗留假名（ops-transformer PR#4089 同类）
OP_LOGI(context, "alltoallmatmul-test enter, tilingKey = %lu.", tilingKey);  // ❌ 调试遗留

// ❌ 错误4: 预定义宏 entityName 传字符串字面量
OP_LOGE_FOR_INVALID_SHAPEDIM("Pow", "x", "3D", "4D");                        // ❌ entityName 硬编码
```

### 正确示例

```cpp
// ✅ 正确1【优先】: 直接传 context，框架自动携带算子标识
OP_LOGE(context, "tiling failed, totalNum is %ld.", totalNum);

// ✅ 正确2【合规】: 显式 GetNodeName()，不算报错
OP_LOGE(context->GetNodeName(), "tiling failed, totalNum is %ld.", totalNum);

// ✅ 正确3【合规】: 预定义宏 entityName 从 context 获取（宏参数为 const char*，无法直接传 context）
OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", "3D", "4D");

// ✅ 正确4【合规】: tiling 类成员初始化时从 context 获取后复用
// Init 中: opName_(context->GetNodeName());
OP_LOGE(opName_, "fail to get platform info");

// ✅ 正确5【豁免】: context 判空分支内无 context 可传，允许使用算子名字符串
if (context == nullptr) {
    OP_LOGE("Xxx", "TilingContext is null, cannot get node name.");   // 豁免：判空分支内（建议与注册名一致）
    return ge::GRAPH_FAILED;
}
OP_LOGE(context, "tiling failed, totalNum is %ld.", totalNum);        // 分支外仍须从 context 获取
```

### 注意事项

1. 优先级：直接传 context > `context->GetNodeName()` > 硬编码字符串（本条报错对象仅为硬编码）
2. LOG-1 管宏选择（用哪个宏）、LOG-2 管算子名来源（名字从哪来），两者独立判定——裸 OP_LOGE 且算子名硬编码可同时报两条
3. 预定义宏（`OP_LOGE_WITH_*`/`FOR_*`）的 entityName 参数为 `const char*`，从 `context->GetNodeName()` 获取
4. 融合算子/多算子共用 tiling 基类场景，可通过基类虚函数返回算子名（每个继承类重写自己的实现），禁止各文件各自硬编码
5. context 判空分支豁免的边界：仅限 `context == nullptr` / `!context` 保护分支体内，不得以"context 可能为空"为由在正常路径硬编码；存在 context 之外的算子名来源（如 aclnn 入参 `opName`）时仍优先使用该来源；判空分支内使用的字符串建议与算子注册名一致

### 专属检视方法

**步骤1 — 定位日志调用**：检索 `\bOP_LOG[A-Z]*\s*\(`（含 `OP_LOGE_WITH_*` / `OP_LOGE_FOR_*` 预定义宏与 `OP_LOGI` / `OP_LOGW` / `OP_LOGD`）。

**步骤2 — 算子名佐证**：从概要/代码中提取本算子注册名——`IMPL_OP(XXX)`、`REG_OP(XXX)`、文件路径 `op_host/{op}_tiling.cpp`、aclnn 接口名。

**步骤3 — 逐调用判定**：

| 检查对象 | 判定 |
|---------|------|
| 第一参数为 `context` / `context->GetNodeName()` / 初始化自 context 的成员变量（Grep 其声明与初始化链） | PASS |
| 第一参数、格式串前缀或实参中存在算子名风格字符串字面量（与注册名/路径吻合的大驼峰、`*-test` 调试遗留） | FAIL |
| 无 context 的静态工具函数中的日志（纯工具函数无宿主上下文） | 豁免，不报 |
| 日志调用位于 context 判空保护分支内（向上追溯所在 `if (context == nullptr)` / `if (!context)` 分支体） | 豁免，不报（无 context 可传；建议字符串与注册名一致） |
| 疑似算子名但无法确认（字面量与注册名不吻合） | SUSPICIOUS |

**步骤4 — 意见输出**：建议改为**直接传 context（优先）**或 `context->GetNodeName()`；理由引用：算子名变更/复制代码到新算子后日志不匹配、多文件口径不一致。

---

## LOG-3: 禁止新增治理废弃错误码及旧宏

**严重级别**：高

### 问题描述

检查新增或修改的错误码上报是否仍使用治理废弃码。直接上报、对应旧宏和别名最终落到同一废弃码时均适用。内部码语义与首报层次分别由 LOG-4、LOG-5 检查。

| 治理废弃码 | 扫描器识别的旧宏候选 |
|---------|---------|
| `EZ0001` | `OP_LOGE_WITH_INVALID_INPUT_SHAPE` |
| `EZ0002` | `OP_LOGE_WITH_INVALID_ATTR` |
| `EZ0003` | `OP_LOGE_WITH_INVALID_ATTR_SIZE` |
| `EZ0005` | `OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE` |
| `EZ0006` | `OP_LOGE_WITH_INVALID_INPUT_FORMAT` |
| `EZ0007` | `OP_LOGE_WITH_INVALID_INPUT_DTYPE` |

表中列出废弃码的旧宏候选，实际生效的宏定义以被检目标版本为准。

### 错误示例

```cpp
// ❌ 新增正式上报仍使用治理废弃码。
REPORT_PREDEFINED_ERR_MSG("EZ0001", msgKey, msgValue);
```

### 正确示例

```cpp
// ✅ 单个输入 shape 约束。目标头文件和注册项核验通过后使用对应语义宏。
OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "x", actualShape, expectedShape);
```

### 注意事项

- 禁用集合以 `cann-error-code-standard/references/documented-error-codes.json` 的 `governance_deprecated` 为准。替换入口按单参数、跨参数以及 expected/reason 的语义选择，保留公开返回值和 errno。
- 与 LOG-1 命中同一处、同一根因时保留一条错误码意见。

### 专属检视方法

1. 读取并在结论中逐一列出唯一基准的三份文件，即 `cann-error-code-standard/SKILL.md`、`cann-error-code-standard/references/development-standard.md` 和 `cann-error-code-standard/references/documented-error-codes.json`。用户询问拦截范围时，逐对列出上表六个废弃码及其对应旧宏。在检视范围查找废弃编号、旧宏及其别名，去掉注释和说明文本。
2. 可用既有 `scripts/clause.check_error_codes.py` 辅助定位。文件检视用 `--root {repo_path} --file {相对文件路径} --format json`。PR 已确认 base/head 对象时用 `--base {merge_base_sha} --head {source_sha}` 限定变更；否则按输入 diff 人工定位。退出码 0 为无词法命中，1 为有候选，2 为扫描未完成。其它 LOG 条款仍须逐条执行。
3. 对宏候选展开到实际生效的上报定义，核对 PR 新侧落点。源码入口见 [CANN opbase 日志宏](https://gitcode.com/cann/opbase/blob/master/include/op_common/log/log.h)，正式上报接口见 [CANN ReportPredefinedErrMsg 与封装宏](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/API/runtimeapi/cann_base_api_0005.html)。以被检目标版本的头文件、实际 include 链和注册项为准。确认新增废弃码后按通用证据要求提交 FAIL，在 `problem_desc` 标记 `[P1] 合入阻塞`，修复建议说明语义入口和目标版本可见性。保留未验证的定义、注册或编译证据缺口。

---

## LOG-4: 错误码语义、目标版本注册与 Arglist 必须匹配

**严重级别**：高

### 问题描述

检查上报叶子与失败根因是否匹配。用户可纠正的 shape、rank、dtype、format、属性值及必选入参错误须使用相应的正式参数诊断。裸 `OP_LOGE`、`OP_CHECK_NULL_WITH_CONTEXT` 及包装宏可能在调用点隐藏 `EZ9999` 等 `E*9***` 内部码。宏名、失败返回值和日志文本只用于定位，根因与宏叶子决定结论。

### 错误示例

来自 [ops-nn PR #12850](https://gitcode.com/cann/ops-nn/pull/12850/discuss) 的 `optim/lamb_next_m_v/op_host/lamb_next_m_v_infershape.cpp` 和 `pooling/max_pool_ext2/op_host/max_pool_ext2_infershape.cpp`。

```cpp
// ❌ 输入 rank 超上限属于参数错误。此处裸 OP_LOGE 经实际 log/log.h 上报内部码。
OP_LOGE(context->GetNodeName(), "input %zu rank %zu exceeds max supported rank %zu",
        i, inputRank, maxRank);

// ❌ 将 ksize 参数缺失的正式诊断替换为隐式上报内部码的判空宏。
OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
```

### 正确示例

```cpp
// ✅ 核对目标注册项及宏签名后提供参数名、实际 rank 和约束。
OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), inputName,
                                        actualRankText, "rank exceeds the supported maximum");
return ge::GRAPH_FAILED;
```

### 注意事项

- 真实内部不变量须结合调用契约和变量来源证明。平台对象、输出描述符、工作区和 tiling buffer 判空应分别确认责任边界。宏展开至内部码本身仅是候选。
- 同名 `OP_LOGE` 可能按 opName/context 或 errno 转发。修复保留公开返回值和 errno，按目标版本的注册名、Arglist 和实际宏签名给建议。需要新增正式码时说明类别和字段，由对应模块分配编号。

### 专属检视方法

1. **定位候选**。在 PR 新增和修改的 C/C++ 行，或文件检视范围内，逐点找 `OP_LOGE(`、`OP_CHECK_NULL*`、`REPORT_INNER_ERR_MSG`、`REPORT_PREDEFINED_ERR_MSG` 和其它上报入口。展开 `OP_CHECK_IF` 的日志参数及跨行调用，沿本地别名和包装函数追到上报点。对 PR 删除的正式上报逐个检查同一失败分支的新入口。
2. **追溯叶子和根因**。按调用点实际 include、构建路径和生效分支读取定义。正式和内部上报入口分别见 [ReportPredefinedErrMsg](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/API/runtimeapi/cann_base_api_0005.html) 和 [ReportInnerErrMsg](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/cann_base_api_0004.html)。典型链为 `OP_CHECK_NULL_WITH_CONTEXT → OP_LOGE → REPORT_INNER_ERR_MSG("EZ9999", ...)`，以目标头文件为准。逐点记录失败条件、被校验对象的来源、实际宏链及叶子，区分用户参数、框架契约、真实内部不变量和下层透传。仅日志替换后的首报缺失归 LOG-6。
3. **核验并提交**。读 `cann-error-code-standard` 的 `SKILL.md`、`references/development-standard.md`、`references/documented-error-codes.json`，按源码索引核对目标版本注册项。直接上报逐字核对 Arglist 的名称、数量和顺序。确认根因与码义不符、正式语义退化为内部码或注册契约不匹配时，按现有 YAML 提交高严重级别 FAIL，`problem_desc` 标记 `[P1] 合入阻塞`。用变更的新侧宏调用作落点。同名宏、未解析 include 或未证实的根因按通用置信度处理，写明「错误码检视未完成」及缺少的证据。建议中逐项标注未验证的注册、编译和负向日志。

---

## LOG-5: 根因层一次首报，包装层仅日志并透传

**严重级别**：中

### 问题描述

检查同一条失败路径的下层首报与外层包装。下层已经在实际返回失败的分支上报根因时，外层只写上下文日志并保留已有状态映射。外层内部码和外层正式码都可能重复首报。

### 错误示例

来自 [ops-nn PR #12850](https://gitcode.com/cann/ops-nn/pull/12850/discuss) 的 `pooling/max_pool_ext2/op_host/arch35/max_pool_ext2_tiling.cpp`。

```cpp
// ❌ ValidateFormat 的格式拒绝分支已正式上报 x/y format。外层再次正式上报。
OP_CHECK_IF(ValidateFormat(context) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "format", "invalid", "ValidateFormat error"),
            return ge::GRAPH_FAILED);
```

### 正确示例

```cpp
auto rc = ValidateFormat(context);
if (rc != ge::GRAPH_SUCCESS) {
    // ✅ 保留 callee、原始 rc 和公开返回行为。
    OP_LOGE_WITHOUT_REPORT(opName_, "ValidateFormat failed, rc=%d", static_cast<int>(rc));
    return ge::GRAPH_FAILED;
}
```

### 注意事项

- 以 `cann-error-code-standard/references/development-standard.md` 的首报责任为准。`OP_LOGE_WITHOUT_REPORT`、`OPS_LOG_E`、`D_OP_LOGE` 等按实际叶子区分 PLOG 和正式上报。
- Kernel/Runtime/Driver 的首报责任按对应边界核验。保留已有公开状态映射，诊断中记录下层原始 rc。

### 专属检视方法

1. **定位包装**。找新增或修改的 `OP_CHECK_IF(Callee(...) != SUCCESS, ...)`、`rc = Callee(...)` 后的失败分支，以及同类 `if` 早退。检查下层首报有改动、外层仍上报的路径。
2. **验证同一失败**。从外层条件反查实际 callee 和对应失败分支，分别列出两个宏叶子、位置和返回值。下层其它分支有上报、或只有返回失败，均不足以证明当前路径重复。核对与 LOG-4 的归属，优先在本条报告同一根因的外层重复上报。
3. **提交发现**。确认两层对同一根因均正式上报后，按既有 YAML 提交 FAIL。优先定位变更新侧的外层上报行，附下层首报证据，建议仅保留 PLOG 与已有返回行为。静态证据用于判定该源码路径，未执行负向日志和运行时首错唯一性验证时明确标注。

---

## LOG-6: 失败返回路径需有明确首报责任和可操作诊断

**严重级别**：中

### 问题描述

检查新增或修改的失败出口，以及 PR 删除正式上报后保留的失败出口。当前层拥有根因事实时须给出正式上报和可操作诊断。下层已首报时保留上下文和返回行为。正常空张量早退、返回布尔谓词和证明确实不可达的防线按各自契约排除。

### 错误示例

```cpp
if (actualBytes < requiredBytes) {
    // ❌ 当前分支负责容量校验，直接失败会丢失所需和实际容量。
    return ge::GRAPH_FAILED;
}
```

### 正确示例

```cpp
if (rc != ge::GRAPH_SUCCESS) {
    // ✅ 已证实 allocator/Runtime 负责首报。记录可定位的上下文。
    OP_LOGE_WITHOUT_REPORT(opName, "Allocate failed, required=%zu, available=%zu, rc=%d",
                          requiredBytes, availableBytes, static_cast<int>(rc));
    return rc;
}
```

### 注意事项

- `GRAPH_FAILED`、失败语义的 `nullptr`、`false` 和下层 rc 传播都需核对首报归属。正式码语义错误归 LOG-4，重复首报归 LOG-5。
- 诊断字段依场景选参数名、actual、expected/reason、required/available bytes 或原始 rc。保留公开返回值和 errno。

### 专属检视方法

1. **枚举出口**。以变更失败条件、返回语句、提前返回宏及删除的上报为入口。对照 base/head 同一分支，确认正式上报是否被删除或替换为仅日志，即使失败返回行未改也追查该路径。
2. **定位责任**。按 `cann-error-code-standard/references/development-standard.md` 核对实际 callee、宏叶子和调用契约。记录何处失败、什么约束被违反、能否从日志确定修复方向，并区分下层已首报与证据未取得。
3. **提交发现**。确认责任分支缺失首报后按既有 YAML 提交 FAIL，`problem_desc` 标记 `[P1] 合入阻塞`；根因码与首报均正确而仅缺诊断字段时按本条中严重级别报告。PR 用变更的新侧条件或返回锚定。无法确认下层首报时按通用置信度记录缺口，逐项注明未验证的编译、负向日志及首错唯一性。
