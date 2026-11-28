# Task A 流程总览

## 定位

Task A 是一个 Route Evaluator 系统：LLM 将源码分发逻辑翻译为可执行 evaluator，脚本枚举 route case objects 并自动执行 tiling/kernel 分发，最终生成 path graph 和高质量 route seed。

核心原则：

- 路径事实由脚本执行得到，不由 LLM 静态枚举。
- route case 是输入，path graph 是输出。
- `case_values` 是所有 route case 共用的固定字段集合，只包含可直接控制、参与或控制 tiling 侧路径判断且表达独立构造自由度的代表字段。
- 每个 route case 使用同一套 `case_values` 字段集合；当前目标分支未直接约束的字段也必须在合法域内取值。
- Task A 在覆盖 tiling/kernel route 分支的同时，必须提高 `case_values` 中 tiling 分支变量的合法取值多样性。
- Case generator 不得把每个分支退化为固定 witness 值表，也不得用固定候选池代替分支约束采样；应根据 Step 1 route dispatch model 中的 `branch_conditions` / `branch_value_constraints` / `branch_priority` 生成分支约束代码，在约束域内随机采样 `case_values`，并通过 evaluator 复验 route 稳定性。
- `case_values` 多样化不得通过加入内部推导变量、等价变量或不具备独立构造自由度的字段实现。
- 完整输入输出构造、shape/rank 多样性和接口层 case 组装由后续 Mapper 承接。
- 同一 tiling_key 下多个 kernel 实例分支只在实际可达时生成 path。
- 后续阶段消费 `S2P1_path_list.json`，不回读源码。
- Task A 产物中的自然语言说明、规则描述和人工可读信息统一使用中文；代码标识符、字段名、枚举值、路径、宏名、平台名及其他机器消费内容保持原文。

## 三步流程

| Step | 职责 | 输出 |
|---|---|---|
| Step 1 | 建立 evaluator 指导书 | `S2P1_route_dispatch_model.md` |
| Step 2 | 生成 evaluator 脚本 + case generator | `S2P1_route_extract.py` + `S2P1_case_generator.py` |
| Step 3 | 运行 case generator 生成 cases，执行 evaluator | `S2P1_route_cases.json` + `S2P1_path_list.json` |

## 文件读取顺序

1. `00-overview.md`
2. `01-step1-dispatch-model.md`
3. `02-step2-generate-extractor.md`
4. `03-step3-generate-cases.md`

## 输入范围

Task A 只允许读取：

1. `S2P0_source_scope.md`
2. source scope 中标记的 tiling P0 入口源码
3. source scope 中标记的 kernel P0 dispatch 入口源码

禁止读取 scout、P1/P2 深层文件、kernel 非 dispatch 实现、Mapper 产物、schema 文件（`S2P1_path_list_schema.json` / `S2P1_route_cases_schema.json` / `S2P1_cases_schema.json`）。

读取纪律：每步只读取该步骤输入表/输入范围中列出的文件，禁止读取未列出的文件。

## 产物关系

```text
S2P1_route_dispatch_model.md  (Step 1 指导书)
S2P1_tiling_glossary.md       (Step 1 变量语义表)
  ↓
S2P1_route_extract.py         (Step 2 evaluator 脚本)
S2P1_case_generator.py        (Step 2 case 生成器)
  ↓
S2P1_route_cases.json         (Step 3 case generator 输出)
  ↓
S2P1_path_list.json           (Step 3 evaluator 输出，path graph)
S2P1_cases.json               (Step 3 evaluator 附带输出，flat array 用例文件)
```

## Schema 文件

- `S2P1_path_list_schema.json`
- `S2P1_route_cases_schema.json`
- `S2P1_cases_schema.json`

定位：仅为输出字段定义的设计参考，无任何运行时消费方（模板以硬编码 `SCHEMA_VERSION` 完成校验）。Task A 执行期间禁止读取；输出字段要求以步骤文档和模板静态区为准。

## 执行约束

- 必须按 `01 -> 02 -> 03` 顺序执行。
- Step 2 不重新设计分发语义。
- Step 3 的 route cases 由 `S2P1_case_generator.py` 自动生成并回调 evaluator 验证。
