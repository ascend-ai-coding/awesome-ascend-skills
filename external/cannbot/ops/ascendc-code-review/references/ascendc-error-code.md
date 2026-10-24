# CANN算子错误码检测

<适用>
语言: C++
侧别: All
领域: true
触发: 必须触发
默认启用: true
介绍: 基于 cann-error-code-standard 检测废弃错误码、根因语义和注册契约、首报唯一性与失败返回路径；无错误码字符串的失败出口也需检查。
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 2
</检视负载>

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| ERR-1 | 禁止新增治理废弃错误码及旧宏 | 错误码规范 | 高 | [适用: All] |
| ERR-2 | 错误码语义、目标版本注册与 Arglist 必须匹配 | 错误码规范 | 高 | [适用: All] |
| ERR-3 | 根因层一次首报，包装层仅日志并透传 | 错误码规范 | 高 | [适用: All] |
| ERR-4 | 失败返回路径需有明确首报责任和可操作诊断 | 错误码规范 | 中 | [适用: All] |

## ERR-1: 禁止新增治理废弃错误码及旧宏

**严重级别**：高

### 问题描述

先读取 `{skill_base}/../cann-error-code-standard/SKILL.md` 及其 `references/source-and-maintenance.md`，从 `references/documented-error-codes.json` 的 `governance_deprecated` 读取治理禁用集合。要拦截的错误码为 `EZ0001、EZ0002、EZ0003、EZ0005、EZ0006、EZ0007`。对应旧宏也属于候选；不能因为宏名以 `OP_LOGE_WITH_` 开头就认为合规。

### 错误示例

```cpp
REPORT_PREDEFINED_ERR_MSG("EZ0001", args);
OP_LOGE_WITH_INVALID_INPUT_SHAPE(context->GetNodeName(), 0, actual, expected);
```

### 正确示例

```cpp
// 单参数 shape 约束；须先确认目标版本的宏存在、签名及 EZ0008 注册项。
OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "x", actual, expected);
```

### 专属检视方法

1. 完整文件检视可运行 `python3 {skill_base}/scripts/clause.check_error_codes.py --root {repo_path} --file {相对文件路径} --format json`，多个文件重复 `--file`。PR 检视使用 `--base {merge_base_sha} --head {source_sha}`，仅将新增/修改行的候选归属本 PR，历史存量单列。
2. 脚本返回 `0` 表示所选词法范围无候选，`1` 表示命中，`2` 表示未完成扫描；不能把缺失基准、解码或 Git 错误当作通过。脚本默认排除 `.git`、三方依赖和构建产物；不排除 `experimental`。其他范围如需检视，人工逐点核验并说明边界。
3. 脚本跳过 C/C++ 注释，识别直接编号、对应旧宏、本文件简单宏别名及相邻字符串拼接；不替代跨头文件宏展开、条件编译、动态字符串和实际 include 链分析。对扫描候选及包装入口继续追溯真实定义。基准或定义缺失时标为待确认，不能静默跳过错误码检测。
4. 使用通用假设检验流程。直接编号命中且位于本次代码范围时可确认治理违规；旧宏或别名须核验目标定义仍上报废弃码后再判 FAIL。证据不足标记 SUSPICIOUS，并列出待核验的定义/include。
5. 修复建议回到基准的 `development-standard.md` 按单参数/跨参数、expected/reason 选路由，不机械替换编号，不修改公开返回值。与 LOG-1 同一位置同根因只保留一条意见。

## ERR-2: 错误码语义、目标版本注册与 Arglist 必须匹配

**严重级别**：高

### 问题描述

本条判据只从 `cann-error-code-standard/references/development-standard.md` 和码表取得；码表没有合适语义时按其 `component-error-code-source-index.md` 查归属组件。资料收录不等于目标版本已注册、宏可见或实际可调用。不能把发生阶段当作根因，也不能为用户参数错误推荐泛化 `E*9***` 内部码。

### 错误示例

```cpp
// 用户 shape 错误被包装成内部异常，缺少参数名及 actual/expected。
if (actualRank != expectedRank) {
    REPORT_INNER_ERR_MSG("EZ9999", "tiling failed");
    return ge::GRAPH_FAILED;
}
```

### 正确示例

```cpp
// 核验实际 include 的 opbase/log.h、正式注册项及参数签名后，选择语义入口。
OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", actualRankText, expectedRankText);
return ge::GRAPH_FAILED;
```

### 专属检视方法

读取基准开发规范，逐点记录根因、资料 URL、目标注册文件/版本、实际宏定义/include 链和 Arglist。优先真实存在的正式语义宏；直接上报须逐字核对字段名、数量和顺序。`OP_LOGE` 必须区分 opName/context 型与 errno 型的真实叶子，不能仅凭同名宏认定 EZ9999。正式码缺失时提出语义类别和字段，不能编造编号。无法解析目标构建分支时只报告证据缺口，不宣称已通过注册和编译验证；真实内部不变量等例外须给出证明与诊断上下文。

## ERR-3: 根因层一次首报，包装层仅日志并透传

**严重级别**：高

### 问题描述

读取基准 `development-standard.md` 的首报和层次责任规范，检查同一根因是否被外层通用码覆盖或重复上报。

### 错误示例

```cpp
auto rc = LowerLayer();  // 已经上报根因正式码。
if (rc != SUCCESS) {
    REPORT_INNER_ERR_MSG("EZ9999", "LowerLayer failed");
    return GENERIC_FAILURE;
}
```

### 正确示例

```cpp
auto rc = LowerLayer();
if (rc != SUCCESS) {
    OP_LOGE_WITHOUT_REPORT(opName, "LowerLayer failed, rc=%d", rc);
    return rc;
}
```

### 专属检视方法

沿调用链核验下层是否实际首报，不能把“下层返回失败”当成“下层已首报”。仅日志宏不计为正式上报。外层应保留 callee、原始 rc 和关联信息；公开 errno 与 ErrorManager 上报码分开判断，未有兼容性评审不能要求改公开返回行为。Kernel/Runtime/Driver 异常首报归属相应边界，不机械在设备侧补 Host 宏。负向日志缺失时注明首错唯一性未验证；仅做静态咨询、未执行编译和负向用例时，明确注明本次未验证编译、负向日志及首错唯一性。

## ERR-4: 失败返回路径需有明确首报责任和可操作诊断

**严重级别**：中

### 问题描述

没有错误码字符串的 `return GRAPH_FAILED`、nullptr 和下层 rc 传播也可能丢失诊断；只扫描编号不能证明错误路径完备。

### 错误示例

```cpp
if (actualBytes < requiredBytes) {
    return ge::GRAPH_FAILED;  // 此根因尚无上报者或容量诊断。
}
```

### 正确示例

```cpp
if (rc != SUCCESS) {
    // 已验证 allocator/Runtime 首报资源不足；本层保留诊断与兼容返回值。
    OP_LOGE_WITHOUT_REPORT(opName, "Allocate failed, required=%zu, actual=%zu, rc=%d",
                          requiredBytes, actualBytes, rc);
    return rc;
}
```

### 专属检视方法

枚举被检范围的失败出口，按基准输出契约记录 What/Where/Why/What Next、首报责任层、实际上报入口和仍缺少的证据。字段按场景提供 parameter、actual、expected/reason、required/available bytes 或原始 rc。不可达防线不要求补码，应先证明可达性。静态检视不修改源码、不执行设备测试，不把缺少编译/负向测试证据说成“错误码已生效”。
