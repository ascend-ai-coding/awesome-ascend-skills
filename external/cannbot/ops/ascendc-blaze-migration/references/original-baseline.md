# G3 正式制品准备与 G4 原始实现基线

本文件规定迁移开发完成后如何构建两套可比正式制品，以及如何只运行 original 建立 G5 唯一可用的功能和性能基线。

## 目录

1. G3 正式构建前置条件
2. G3 Original 与 Blaze 正式构建
3. G3 制品冻结
4. G4.1 基线输入与身份核对
5. G4.2 原始功能与稳定性验证
6. G4.3 原始性能验证
7. G4.4 原始基线冻结
8. 失败处理与关闭条件

## 1. G3 正式构建前置条件

开始正式构建前确认：

- G1 迁移合同、行为模型、迁移范围和验证义务未失效；
- G2 Blaze 代码、组件证明、内部合同、反模式和开发反馈编译均已冻结；
- G3 逐 case 用例表、输入明细、覆盖映射、runner 源码和输入资产已冻结；
- G1 每条义务已绑定具体 case 或审计排除项；
- 正式功能性能用例总数为 10~30，每个正式用例同时定义功能和性能要求；
- design 表、case 资产、runner 注册和 G4/G5 执行清单完全相等且唯一；
- `environment-state.json` 结构和哈希有效。

从环境文件读取 `capabilities.build_opp`。只有 `available` 时执行两侧正式构建；为 `unavailable` 或 `unknown` 时 G3 正式构建保持 `unknown`，不重复环境探测，也不得用 G2 开发态编译产物替代。

G3 不运行正式功能或性能测试。不得通过先试跑 Blaze 再删改用例、输入或预期行为。

## 2. G3 Original 与 Blaze 正式构建

Blaze/ops-nn 的本地 `ops-tensor` include staging、CMake 生成和 `asc_opc` 逐任务编译必须遵循[Blaze OPP 编译指导](blaze-opp-build.md)。本节只规定 G3 两侧制品的身份、隔离和可比性，不重复维护具体命令。

从 G0/G2 冻结身份分别执行干净 release 构建：

```text
repo/original -> packages/original/opp/
repo/blaze    -> packages/blaze/opp/
```

两侧必须：

- 使用相同 CANN、编译器、SoC、release 模式和公共构建选项；
- 使用独立构建目录、依赖目录、自定义 OPP 根和 vendor/加载环境；
- 保存完整命令、日志、package、Kernel 和 SHA256；
- 证明 Kernel 来自对应冻结代码，而不是系统同名包、旧安装包或另一侧 OPP；
- 从同一 G3 runner 源码生成可执行文件，参数和执行路径只通过 role/OPP 根切换；
- 不把 G2 开发态增量包升级为正式制品。

若 original 确实依赖 ops-tensor，使用 G0 已冻结在 `repo/original/` 的 checkout；否则不为目录对称引入无用副本。

## 3. G3 制品冻结

两套 manifest 使用同构字段：

- role、来源仓、ops-tensor、submodule 和最终代码 SHA；
- G1 迁移设计、G3 验证设计、runner、case 和输入 SHA256；
- 环境 revision、CANN、编译器、SoC、release 模式和构建命令；
- package、OPP 根、vendor、Kernel symbol 和 SHA256；
- 依赖、最小加载检查和日志索引。

Blaze manifest 形成后必须核对其活动 Kernel 及依赖闭包属于 G2 已扫描集合，编排层反模式和两组 CMCT/CGMCT 实现命中仍为零；不一致时返回 G2，不得在 G3 用排除项掩盖新增活动依赖。

两套制品全部成功且身份、公共构建协议可比后，G3 才可关闭。仅一侧成功、加载来源不明、公共选项不一致或 runner 不同均不能进入 G4。

## 4. G4.1 基线输入与身份核对

G4 只使用 original 正式制品。运行前核对：

- G1/G2/G3 身份与哈希未变化；
- original package、Kernel、runner、输入、环境和实际加载路径一致；
- 具体用例表、覆盖映射、case 资产、runner 注册和 G4 执行清单完全相等；
- 最终 Shape、seed、生成器版本、输入 SHA256、预期路由和性能协议已冻结；
- 进程不会加载 Blaze OPP 或系统同名自定义 OPP。

## 5. G4.2 原始功能与稳定性验证

只有 `capabilities.run_device_tests=available` 时开始；否则 G4.2 保持 `unknown`，不重复环境探测，也不得跳过进入 G5。

按 G3 清单逐 case 运行：

1. 正式功能性能用例保存全部输出 bin、动态元数据和必要 inplace after-state；
2. fallback 保存功能结果和实际路由；
3. 非法辅助用例保存拒绝阶段、行为和诊断类别；
4. 每轮检查 runner 可控制的 buffer canary、输出初始化、文件长度和重复稳定性；
5. 保存环境、package、runner、case、input、Shape、seed、对齐状态、实际路由和执行次数身份。

结果状态只允许 `PASS`、`FAIL`、`NOT_RUN`。退出码为零但执行次数为零、目标 Kernel 未运行、路由错误、输出缺失、动态元数据缺失或可控检查异常均不能 `PASS`。

G4 的目标是建立 original 稳定二进制基线，不引入 CPU golden、NumPy reference 或其他独立参考实现。每个有效功能输出必须完整保存且重复执行逐字节稳定；close 或平均误差不能替代稳定性。

## 6. G4.3 原始性能验证

只有 `capabilities.collect_msprof=available` 时开始；否则 G4.3 保持 `unknown`，不以 wall-clock 或其他计时替代。

对全部正式功能性能用例调用 `ops-profiling`，明确使用 msprof，并提供 original runner、隔离 OPP、case、输入、device、warm-up、repeat、输出目录和目标 Kernel 身份。

保存 `validation/msprof/original/` 下的原始采集文件、字段解析、单次样本、逐例统计和必要 AIC metrics。命令和指标知识由 `ops-profiling` 负责，本 skill 只维护身份、可比性和门限。

## 7. G4.4 原始基线冻结

在 `migration-validation.md` 的 original 章节记录：

- original 代码、依赖、环境、package、OPP、Kernel 和 runner 身份；
- G1 义务、G3 设计集合与 G4 实际集合核对；
- 每个 case 的输入、路由、输出、状态、重复稳定性和日志；
- 正式、fallback 和非法辅助结论；
- 每个正式性能 case 的 msprof 数据、统计量和路由；
- 原始实现偏离文档或设计行为预期的确认事实；
- 资产路径和 SHA256。

冻结 original manifest、结果和报告章节后，G5 只读。失效时追加失效记录并返回 G4，不能在 G5 临时重跑创建旁路基线。

## 8. 失败处理与关闭条件

文档声明支持但 original 用例失败时，保留完整失败证据，依次检查 runner/输入、package/加载身份、路由/执行次数、环境和原始实现。具体用例或 runner 错误返回 G3；G1 合同或源行为理解错误返回 G1；原始实现与公开合同真实冲突且无法在任务内解决时标记 `blocked`。

不得删除、替换、缩小或重分类用例，不得用 Blaze 结果反向定义 original 合同。

G4 关闭必须满足：

- 全部 G3 用例有真实结果，无 `FAIL`、`NOT_RUN` 或缺失；
- 用例集合、冻结字段和路由一致；
- 正式、fallback 和非法辅助合同有明确结论；
- 全部正式功能性能用例完成 msprof；
- manifest、结果、报告原始章节和证据索引已固定。

满足后 G4 为 `verified`，自动进入 G5。
