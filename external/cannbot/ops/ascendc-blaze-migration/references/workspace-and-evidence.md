# 工作区与证据

## 目录

1. 工程布局
2. 代码仓隔离
3. 环境状态文件
4. 迁移记录与交接
5. 产物所有权
6. 制品清单
7. 完整性规则

## 1. 工程布局

在 Agent CLI 启动目录创建：

```text
blaze-migration-<task-id>/
├── repo/
│   ├── original/{ops-nn|ops-transformer,ops-tensor-if-needed}/
│   └── blaze/{ops-nn|ops-transformer,ops-tensor}/
├── reports/
│   ├── migration-record.md
│   ├── environment-state.json
│   ├── migration-design.md
│   ├── migration-validation.md
│   └── migration-review.md
├── packages/{original,blaze}/
├── validation/
│   ├── runner/
│   ├── cases/
│   ├── inputs/
│   ├── results/{original,blaze}/
│   ├── msprof/{original,blaze}/
│   └── logs/
└── SHA256SUMS
```

`packages/original/` 和 `packages/blaze/` 保存 G3 从冻结代码身份构建的独立正式 OPP。二者不是仓库副本，不得共享可变构建目录、安装根或加载环境。

## 2. 代码仓隔离

- 使用 G0 已获取的独立 checkout；`repo/original/` 的基线 `master` 不修改，`repo/blaze/` 的迁移代码直接提交到 `master`；源码获取、ops-tensor submodule 完整性和初始 SHA 规则以[串行工作流](serial-workflow.md)的 G0 章节为准；
- G2 只修改 `repo/blaze/`，`repo/original/` 始终保持冻结；
- G3 分别从 `repo/original/` 和 `repo/blaze/` 构建正式制品，G4 只加载 original，G5 只加载 Blaze；
- 两侧具有独立 checkout、构建目录、依赖目录和 OPP 根；`repo/original/` 使用基线 `master`，`repo/blaze/` 使用提交迁移代码后的 `master`；
- `validation/` 只保存看护工程和数据，不复制产品代码仓；
- 不通过共享未固定缓存或系统已安装自定义 OPP 隐式串包。

## 3. 环境状态文件

### 单一事实源

G0 调用 `ascendc-env-check` 后必须生成 `reports/environment-state.json`。后续门禁只读该文件，不再次调用环境检查，也不以单一工具结果覆盖它。

以下结构是必须满足的最小结构：

```json
{
  "schema_version": 1,
  "revision": 1,
  "checked_at": "",
  "checker": {"skill": "ascendc-env-check", "status": "completed"},
  "toolchain": {
    "cann": {"state": "unknown", "version": "", "home_path": ""},
    "opp": {"state": "unknown", "path": ""},
    "compiler": {"state": "unknown", "path": ""},
    "msprof": {"state": "unknown", "path": ""},
    "simulator": {"state": "unknown", "platforms": []}
  },
  "device": {
    "state": "unknown",
    "device_ids": [],
    "soc_versions": [],
    "npu_arch": [],
    "health": "unknown",
    "evidence_sources": []
  },
  "capabilities": {
    "build_opp": "unknown",
    "run_device_tests": "unknown",
    "collect_msprof": "unknown"
  },
  "limitations": [],
  "evidence": []
}
```

环境字段状态只用 `available`、`unavailable`、`unknown`。环境文件记录工具链、设备和能力事实，不记录门禁状态、支持域、迁移范围或测试结论。不得仅因 `npu-smi` 不存在把设备写为不可用；按 `ascendc-env-check` 的完整探测和回退证据归一化。

G0 必须使用 JSON 解析器检查必需字段、状态枚举和 `checker.skill`，计算文件 SHA256，并将 revision/hash 写入 record。缺文件、结构无效、字段缺失或哈希不一致时，G0 不能关闭。

正常任务只生成一次。只有文件损坏、用户明确声明环境切换，或 CANN、驱动、设备映射、工具链发生可证明变化时才允许重新探测。重新探测前在 record 登记原因；增加 `revision`，保留旧修订摘要与旧哈希，并按影响矩阵使证据失效。设备临时繁忙或单次命令失败先由当前阶段诊断，不自动重探测。

## 4. 迁移记录与交接

`migration-record.md` 建议结构：

```markdown
# 迁移记录

## 任务身份
- task_id:
- startup_cwd:
- target_operator:
- platform:
- source_repo_url:
- source_requested_branch: master
- source_resolved_sha:
- ops_tensor_repo_url:
- ops_tensor_requested_branch: master
- ops_tensor_resolved_sha:
- ops_tensor_submodule_shas:
- ops_tensor_checkout_complete:

## 当前进度
- current_gate:
- current_phase:
- status: unknown | verified | blocked
- updated_at:

## 环境状态
- path: reports/environment-state.json
- revision:
- sha256:

## 冻结身份
- original_repo_sha:
- blaze_repo_sha:
- original_ops_tensor_sha:
- blaze_ops_tensor_sha:
- g1_migration_design_section_sha256:
- g3_validation_design_section_sha256:
- final_design_file_sha256:
- execution_assets_sha256:

## 门禁交接
| 门禁 | 状态 | 固定输入 | 固定输出 | 证据 | 失效原因 |

## 证据索引
| 证据 | 路径 | SHA256 | 生产门禁 | 消费门禁 |

## 决策与下一步
```

record 只维护状态、交接和索引，不复制完整环境、设计、性能或审查事实。门禁仅在动作、产物、集合和证据齐备时标为 `verified`。

## 5. 产物所有权

| 资产 | owner | 后续规则 |
|---|---|---|
| `environment-state.json` | G0 | 后续只读，明确失效时生成新修订 |
| `migration-design.md` 的迁移合同、行为模型、范围、owner、内部合同和验证义务 | G1 | G2-G6 不得改写；事实错误返回 G1 |
| Blaze 代码、组件证明、反模式/CMCT 扫描和开发反馈编译结果 | G2 | G3 构建正式制品；实现修复返回 G2 |
| `migration-design.md` 的逐 case 验证章节、runner、cases、inputs、两侧正式 OPP 和 manifest | G3 | G4/G5 只读；验证资产或构建问题返回 G3 |
| original 结果、msprof 和验证报告原始章节 | G4 | G5/G6 只读；不得在 G5 临时重建 |
| Blaze 结果、msprof、对比和 review | G5 | G6 只读；实现问题返回 G2 |
| 本地提交、依赖和复现记录 | G6 | 必须匹配 G5；不创建 PR、不推送 |

一个事实只由 owner 维护，其他文档引用 owner 和哈希。`migration-design.md` 使用 `<!-- G1_MIGRATION_DESIGN_END -->` 标记分隔 owner；G1 固定标记之前的分段 SHA256，G3 只能在标记之后追加验证章节，并同时固定 G3 分段和最终整文件 SHA256。G5 可向验证报告追加 Blaze 章节，但不能改写 G4 原始章节。

## 6. 制品清单

G3 为两套正式制品生成同构 manifest：

- role：`original` 或 `blaze`；
- 来源仓、ops-tensor 和 ops-tensor submodule SHA；
- G1 迁移设计和 G3 验证设计 SHA256；
- 环境文件 revision 与 SHA256；
- CANN、编译器、SoC、release 模式和完整构建命令；
- package 路径及 SHA256；
- OPP 根、vendor、Kernel symbol 与 SHA256；
- runner、用例和输入资产 SHA256；
- 最小加载检查和日志索引。

每个 runner 结果和 profiling 结果索引都必须记录环境 revision/hash、两段 design SHA、runner SHA、case/input SHA、package manifest 和 role。生成的结果 JSON、manifest 和汇总是运行证据，不是第二份人工维护的支持合同。

两侧必须使用独立 OPP 根或 vendor，在独立进程加载。时间戳和目录名不能作为身份结论。

## 7. 完整性规则

- 根 `SHA256SUMS` 只覆盖长期保留资产，不纳入继续变化的缓存；
- 报告结论必须追溯到代码、制品、用例、输入、runner、环境和原始结果；
- 日志保留命令、退出码和输出，摘要不能替代原始日志；
- G1 分段 SHA256 在 G2-G6 必须保持不变；G3 合法追加不能成为改写 G1 迁移结论的通道；
- G1 覆盖义务集合必须与 G3 覆盖映射逐项闭合；没有依据的遗漏、合并或排除直接失败；
- G3 具体用例表、输入明细表、case 资产、runner 注册和 G4/G5 执行清单的 `case_id` 必须完全相等且唯一；
- runner 结果和覆盖汇总必须绑定 obligation_id、最终 Shape、Shape 类别、对齐状态、shape seed、生成器版本、重抽记录、输入 SHA256 和实际路由；
- 覆盖统计必须能追溯到 G1 验证义务和 G3 具体 case，不能只记录人工填写的总覆盖率；
- `PASS`、`FAIL`、`NOT_RUN` 只能由 runner 生成的逐 case 结果产生，汇总不得只统计通过项或由人工覆盖；
- G4 功能 `PASS` 必须包含 original 真实执行、完整输出和重复执行逐字节稳定性；不要求 CPU golden。G5 功能 `PASS` 必须包含 original/Blaze 有效输出、必要动态元数据和 inplace after-state 的逐字节比较；close 只能作为诊断字段；
- runner 必须验证目标 Kernel 实际执行、执行次数大于零、输出存在、可控检查完整和 role/package/runner/design/environment/input 身份一致；否则只能生成 `FAIL` 或 `NOT_RUN`；
- 不在 `/tmp` 保存唯一证据，不把任务资产散落到工程外；
- 产品仓提交不得包含 reports、validation、packages 或迁移工程路径；
- 冻结资产失效时先登记旧身份和原因，再生成新证据，不静默覆盖。
