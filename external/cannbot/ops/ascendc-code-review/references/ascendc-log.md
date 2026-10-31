# Ascend C Host 侧日志上报规范

<适用>
语言: C++
侧别: Host
领域: true
触发: OP_LOGE, OP_LOGI, OP_LOGW, OP_LOGD
默认启用: true
适用场景: op_host/**/*.cpp, op_graph/**/*.cpp, op_api/**/*.cpp（覆盖 tiling、InferShape、InferDataType、aclnn 实现）
介绍: LOG-1 按 cann-error-code-standard 核验日志的实际上报入口和正式语义，禁止推荐治理废弃宏；LOG-2 检查日志中的算子名来源。
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 1
</检视负载>

## 快速索引

| 规范编号 | 规范名称 | 类别 | 侧别 | 严重级别 |
|---------|---------|------|------|---------|
| LOG-1 | 语义可映射时使用目标版本正式入口，不推荐治理废弃宏 | 错误码规范 | Host | 中 |
| LOG-2 | 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 | 日志规范 | Host | 中 |

## LOG-1: 语义可映射时核验正式错误上报入口

**严重级别**：中

### 问题描述

本条的唯一错误码基准为 `{skill_base}/../cann-error-code-standard/SKILL.md` 及其 `references/development-standard.md`、`references/documented-error-codes.json`。废弃码、注册契约和首报判据分别由 ERR-1~4 定义，不再在日志规则中维护第二份宏码映射表。

不能把所有裸 `OP_LOGE` 都判为 EZ9999，也不能把所有带后缀的 `OP_LOGE_WITH_*` 都判为合规。实际宏来源可能是 op_common、nnopbase 或项目包装；先解析目标 include 链，再判断最终上报叶子。已知下层首报后，外层仅日志并透传；过程信息不能无证据认定为正式上报缺陷。

### 错误示例

```cpp
// 参数错误仍使用已被治理禁用的按 input index 上报码。
OP_LOGE_WITH_INVALID_INPUT_SHAPE(context->GetNodeName(), 0, actualShape, expectedShape);
```

### 正确示例

```cpp
// 目标版本已确认支持该语义宏及其 EZ0008 注册项；使用稳定参数名。
OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "x", actualShape, expectedShape);
```

### 专属检视方法

1. 定位被检范围的日志与包装宏，读取上述基准规范，追溯实际宏定义文件、include 链和最终 ErrorManager 入口。
2. 语义可映射且目标注册项、宏签名和参数可得性已核验，才建议具体替换；缺少目标版本证据时标记 SUSPICIOUS 并列出缺口，不依据宏名臆测。
3. 禁止推荐基准 `governance_deprecated` 中的 `EZ0001、EZ0002、EZ0003、EZ0005、EZ0006、EZ0007` 及其对应旧宏。
4. 沿用通用假设检验方法。意见给出原调用、真实叶子、根因类别、资料 URL、目标注册/宏路径、Arglist 和建议写法；保持公开返回值兼容。
5. 同一调用点同根因已由 ERR-* 报告时合并为一条意见；LOG-2 仍单独检查算子名来源。

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

### 真实检视案例

案例1 — 第一参数硬编码算子名（ops-math PR#2195）：

> 评论：「第一个参数用了硬编码字符串 "Pow"，其他所有文件统一用的是 context_->GetNodeName()。建议改为 context_->GetNodeName() 保持一致，也避免算子名变更后日志不匹配。」

案例2 — OP_LOGI 调试遗留假名（ops-transformer PR#4089）：

> 评论：「这行 OP_LOGI 用了硬编码的 "alltoallmatmul-test" 作为算子名，看起来是调试阶段留下的日志。生产代码中不应该用 test 后缀的标识符做日志tag，要么删掉这行，要么改成使用实际的 opName 参数。」

案例3 — context 已携带算子标识又手动加前缀（ops-math PR#1767 / #1994）：

> 评论：「context->GetNodeName() 这个已经获取了算子名称，后面打印可以不需要再添加。」
> 作者采纳后回复：「日志输出中不再手动添加算子名前缀，依赖 context->GetNodeName() 或 context 自动携带的算子标识。」

### 反例

```cpp
// 反例1: 第一参数硬编码算子名（ops-math PR#2195 同类）
OP_LOGE("Pow", "tiling failed, powNum is %ld.", powNum);                     // ❌ 自定义算子名字符串

// 反例2: 消息里手写算子名前缀——context 已自动携带算子标识（ops-math PR#1767 同类）
OP_LOGE(context, "AddCustom tiling failed, totalNum is %ld.", totalNum);     // ❌ 手动前缀冗余

// 反例3: 调试遗留假名（ops-transformer PR#4089 同类）
OP_LOGI(context, "alltoallmatmul-test enter, tilingKey = %lu.", tilingKey);  // ❌ 调试遗留

// 反例4: 预定义宏 entityName 传字符串字面量
OP_LOGE_FOR_INVALID_SHAPEDIM("Pow", "x", "3D", "4D");                        // ❌ entityName 硬编码
```

### 正例

```cpp
// 正例1【优先】: 直接传 context，框架自动携带算子标识
OP_LOGE(context, "tiling failed, totalNum is %ld.", totalNum);

// 正例2【合规】: 显式 GetNodeName()，不算报错
OP_LOGE(context->GetNodeName(), "tiling failed, totalNum is %ld.", totalNum);

// 正例3【合规】: 预定义宏 entityName 从 context 获取（宏参数为 const char*，无法直接传 context）
OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", "3D", "4D");

// 正例4【合规】: tiling 类成员初始化时从 context 获取后复用
// Init 中: opName_(context->GetNodeName());
OP_LOGE(opName_, "fail to get platform info");

// 正例5【豁免】: context 判空分支内无 context 可传，允许使用算子名字符串
if (context == nullptr) {
    OP_LOGE("Xxx", "TilingContext is null, cannot get node name.");   // 豁免：判空分支内（建议与注册名一致）
    return ge::GRAPH_FAILED;
}
OP_LOGE(context, "tiling failed, totalNum is %ld.", totalNum);        // 分支外仍须从 context 获取
```

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

### 注意事项

1. 优先级：直接传 context > `context->GetNodeName()` > 硬编码字符串（本条报错对象仅为硬编码）
2. LOG-1 管宏选择（用哪个宏）、LOG-2 管算子名来源（名字从哪来），两者独立判定——裸 OP_LOGE 且算子名硬编码可同时报两条
3. 预定义宏（`OP_LOGE_WITH_*`/`FOR_*`）的 entityName 参数为 `const char*`，从 `context->GetNodeName()` 获取
4. 融合算子/多算子共用 tiling 基类场景，可通过基类虚函数返回算子名（每个继承类重写自己的实现），禁止各文件各自硬编码
5. context 判空分支豁免的边界：仅限 `context == nullptr` / `!context` 保护分支体内，不得以"context 可能为空"为由在正常路径硬编码；存在 context 之外的算子名来源（如 aclnn 入参 `opName`）时仍优先使用该来源；判空分支内使用的字符串建议与算子注册名一致
