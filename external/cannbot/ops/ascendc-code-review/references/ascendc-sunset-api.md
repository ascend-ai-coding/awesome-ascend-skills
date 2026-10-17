# Ascend C 日落（废弃）API 检查规范

<适用>
语言: C++
侧别: All
领域: true
触发: aclnn, aclrt, aclop
默认启用: true
适用场景: 跨侧别日落（废弃）API/头文件/库检查（SUNSET-1/SUNSET-2），Host 侧 aclnn 代码与 Kernel 侧均生效
介绍: CANN 日落（废弃）API 与头文件/库的兼容性检查，清单由 clause.get_sunset_api.py 动态解析官方废弃文档生成，检视策略为预研报告驱动的确定性符号匹配（不走假设检验）
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 2
</检视负载>

> **机制联动**：api-prestudy Step 2.5 运行 `{skill_base}/scripts/clause.get_sunset_api.py` 比对代码 API 清单，产出预研报告「## 日落 API」章节；本文件条款优先消费该章节，无章节时 fallback 脚本自查（检视策略见各条款）。
>
> **yaml 提交约定**：本文件条款不走假设检验，FAIL/SUSPICIOUS 的 yaml 使用 `category: sunset`，不含 evidence/confidence 字段（collector 已对此类条款豁免校验；problem_desc/fix_suggestion/code_snippet 仍必填）。

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| SUNSET-1 | 禁止使用日落（废弃）API | API兼容性 | 高 | [适用: All] |
| SUNSET-2 | 禁止引用日落头文件/库 | API兼容性 | 高 | [适用: All] |

---

## SUNSET-1: 禁止使用日落（废弃）API

**严重级别**：高  **适用范围**：[适用: All]

### 问题描述

CANN 每版本废弃（日落）一批接口并提供替代，废弃接口在删除期限后无法编译。覆盖 Runtime API(C/Python) 的 `aclrt*`/`acl.op.*` 与算子库 `aclnn*`。清单随版本动态变化，由 `{skill_base}/scripts/clause.get_sunset_api.py` 动态解析官方废弃文档生成。

### 错误示例

```cpp
aclrtGetVersion(&version);          // ❌ 已废弃 → aclsysGetVersionStr
aclnnGroupedMatmulV2(...);          // ❌ 删除期限 2027.3.30 → aclnnGroupedMatmulV5
```

### 正确示例

```cpp
aclsysGetVersionStr(versionStr);    // ✅ 替代接口
aclnnGroupedMatmulV5(...);          // ✅ 最新版
```

### 检视策略 — 预研报告驱动（不走假设检验）

确定性符号匹配，命中即违规，不收集证据分值。

1. Read `{api_prestudy_path}` 的「## 日落 API」章节（api-prestudy Step 2.5 比对产出）。
2. 无命中 → PASS。有命中 → Grep 命中行核实是否真使用（排除注释/字符串/死代码），且是日落符号本身而非 `V2`/`V3` 等替代品。真使用 → FAIL(HIGH) + 替代接口 + 删除期限；仅注释提及 → PASS + 清理提示。
3. 预研报告无此章节 → fallback：`python3 {skill_base}/scripts/clause.get_sunset_api.py` 得清单（格式 `sym -> rep`，**只取箭头左侧的日落符号**，右侧是替代品勿混入），grep 代码 `acl(rt|nn)[A-Za-z0-9_]+` / `acl\.(op|rt)\.` 比对（注意词法边界，不误匹配替代品）。无法获取 → SUSPICIOUS + 标注。

---

## SUNSET-2: 禁止引用日落头文件/库

**严重级别**：高  **适用范围**：[适用: All]

### 问题描述

CANN 同样日落头文件路径与链接库，废弃项在删除期限后不再存在导致编译/链接失败。典型：`op_proto/inc/*.h` → `op_graph/inc/${ops_project}_ops_proto.h`；`libopapi.so` → `libopapi_${ops_project}.so`（期限 2026.12.30）。

### 错误示例

```cpp
#include "op_proto/inc/ops_proto.h"       // ❌ → op_graph/inc/${ops_project}_ops_proto.h
target_link_libraries(... libopapi.so)    // ❌ → libopapi_${ops_project}.so
```

### 检视策略 — 预研报告驱动（不走假设检验）

1. Read `{api_prestudy_path}`「## 日落 API」章节（含头文件/库命中）。
2. 无命中 → PASS。有命中 → Grep 确认是真实 `#include`/链接配置（非注释），且是日落路径本身（`op_proto/inc` 日落，`op_graph/inc` 替代）。真引用 → FAIL(HIGH) + 替代 + 期限；仅注释 → PASS + 清理提示。
3. 预研报告无此章节 → fallback：`{skill_base}/scripts/clause.get_sunset_api.py` 得「算子库(头文件/库)」清单，grep `#include.*op_proto` / `libopapi\.so`。无法获取 → SUSPICIOUS。
