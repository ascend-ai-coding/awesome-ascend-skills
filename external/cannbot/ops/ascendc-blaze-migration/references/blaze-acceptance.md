# G5 Blaze 迁移验收

G5 的用例集合、Shape 行为分类、对齐/合法非对齐状态和覆盖义务遵循[用例覆盖设计](case-coverage-design.md)，本阶段不得重新选择或随机生成 Shape。

## 目录

1. 前置条件
2. G5.1 验收输入核对
3. G5.2 Blaze 功能验证
4. G5.3 Blaze 性能验证
5. G5.4 原始基线对比
6. G5.5 最终代码审查
7. G5.6 验收归档
8. 问题闭环
9. 关闭条件

## 1. 前置条件

G5 只在 G4 `verified` 后启动。G5 不重新构建正式 OPP，不修改 G1/G3 冻结设计，不重新运行或改写 G4 原始基线。

只读 `environment-state.json`。`run_device_tests` 或 `collect_msprof` 不是 `available` 时，相应子阶段保持 `unknown`，不得重复调用环境检查或用其他计时替代。

## 2. G5.1 验收输入核对

核对：

1. G1 迁移合同和验证义务、G3 具体用例、runner、输入、逐字节规则、性能门限和清单哈希未变化；G1 义务到 G3 case 的映射闭合，design 表、case 资产、runner 注册集合和 G5 执行清单完全相等且唯一；
2. G4 original manifest、全部结果、msprof 和报告原始章节完整；
3. G2 Blaze 代码/扫描证据与 G3 Blaze manifest、package、Kernel 和 runner 一致；
4. 两套制品使用可比的 CANN、编译器、SoC、release 模式和性能协议；
5. 环境 revision 与有效基线一致；
6. Blaze OPP 根不会加载旧包或原始 OPP。

不匹配时返回 owner 门禁，不使用旧证据继续。

## 3. G5.2 Blaze 功能验证

只有 `capabilities.run_device_tests=available` 时开始。按 G3 冻结执行清单只运行 Blaze 实现：

1. 读取与 G4 相同的输入、最终 Shape、seed 和覆盖义务；不得重新随机生成 Shape；
2. 保存全部输出、动态元数据、inplace after-state、runner 可控制的检查结果和重复稳定性；
3. 核对环境 revision、Blaze package、runner、最终 Shape、Shape 类别、对齐状态、实际路由和执行次数；
4. 读取 G4 对应 `case_id` 的原始结果并执行两侧逐字节比较；同时比较必要的动态元数据和 inplace after-state；
5. fallback 比较功能与路由，非法输入比较拒绝阶段和行为。

每条设计用例必须存在 G4 与 G5 对应结果。只有有效输出、必要动态元数据和 inplace after-state 全部逐字节一致，且身份、路由、执行次数和 runner 可控制的检查通过时才能为 `PASS`。`FAIL`、`NOT_RUN`、错误路由、输出缺失、零执行次数、动态元数据不一致或检查失败均不能通过。

不得使用 `close`、`rtol`、`atol`、平均误差或少量样例通过替代逐字节比较，不得归一化 `+0/-0`、NaN payload 或 Inf。close 结果如需保留，只能作为失败诊断字段；不能改变 case 状态或 G5 门禁结论。

功能和性能模式必须共享 G3 冻结的 Tensor、属性、输入、路由和身份检查。性能模式不能通过省略功能模式检查来隐藏参数、路由、执行次数或 guard 错误。

## 4. G5.3 Blaze 性能验证

只有 `capabilities.collect_msprof=available` 时开始。调用 `ops-profiling`，使用 msprof 采集全部 Blaze 正式功能性能用例：

- 使用与 G4 相同的 device、CANN、release 配置、输入、路由、warm-up 和统计协议；
- 输出到 `validation/msprof/blaze/`；
- 保存原始数据、单次样本、逐例统计和必要 AIC metrics；
- 读取 G4 冻结统计量，逐例报告 Blaze/原始实现；
- 使用仓内正式门限，缺失时使用 G1 冻结门限和 G3 case 绑定；
- 不用跨用例平均值掩盖单例回退。

本 skill 不维护 msprof 命令、字段含义、AIC metrics 解析或瓶颈判断；这些内容全部由 `ops-profiling` skill 提供。迁移 skill 只维护 case、身份、可比性、门限和结果绑定。

发生性能回退时，必须先完成完整取证，再形成结论：

```text
case
→ ops-profiling 原始数据和逐例统计
→ Kernel/package/runner 身份
→ 代码位置和 behavior_id
→ 内部执行合同
→ 差异分类
→ 性能结论和修复要求
```

不得只凭平均耗时、单一指标、功能通过或少量样本形成性能结论；不得停止取证等待普通批准。

## 5. G5.4 原始基线对比

执行三类完整性检查：

1. G1 验证义务 = G3 覆盖映射；G3 冻结执行集合 = G4 原始执行集合 = G5 Blaze 执行集合；集合必须完全相等，不允许删除、跳过、替换、重复或额外执行；
2. 每个 `case_id` 的冻结字段、输入、预期路由、基线比较规则、逐字节规则和性能门限相同；
3. 每个功能、fallback、非法和性能结论都绑定两侧身份与原始证据。

名称相同但 transpose、storage、属性或输入发生变化仍是设计违背。缺失、失效或身份不明的原始结果必须返回 G4，不能在 G5 即时重跑创建旁路基线。

## 6. G5.5 最终代码审查

审查输入包括 G1 迁移设计、G2 Blaze 开发证据、G3 两侧制品和验证设计、G4 原始基线、G5 全部结果及固定 Convention。不能只查看 diff、编译日志或性能摘要。

执行双向映射：

1. 原始实现到 Blaze：每个源活动行为映射到目标位置，或以 `INACTIVE_REMOVAL` 证明不可达；
2. Blaze 到原始实现：每个目标行为标明源依据或允许差异分类。

两侧覆盖率必须为 100%。强制审查：

- Tiling、ABI、workspace、blockDim、支持域、数学语义和任务覆盖；
- 地址、offset、layout、stride、gap、packing、tail；
- Buffer 分组、数量、选择、生产消费、复用和生命周期；
- VF、舍入、饱和、累加、转换顺序和同步；
- 参数到最终使用点的路径；
- G1 owner 矩阵和中文迁移状态；
- 编排层反模式为零，运行时边界只承担允许用途；
- 组件内部低阶实现有注册、范围和契约测试；
- 两组 CMCT/CGMCT 实现命中为零；
- 外部参考策略均有目标源证据；
- Convention 逐条结论。

Convention 状态只用 `compliant`、`violation`、`N/A with reason`、`unknown`。硬性违规不能由功能或性能通过豁免。

## 7. G5.6 验收归档

`migration-validation.md` 的 Blaze 章节记录全部 `case_id`、身份、路由、输出比较、性能比值、失败修复和证据路径，不修改 G4 原始章节。

`migration-review.md` 记录最终代码身份、双向映射、差异、内部合同、owner、反模式、CMCT/CGMCT、设计保真性、Convention、findings、修复和结论。record 只登记状态和证据索引。

## 8. 问题闭环

- Blaze 代码或 ops-tensor 问题：返回 G2，修复后由 G3 重新构建 Blaze 制品，再重跑受影响 G5；
- 原始身份或基线问题：返回 G4；若正式制品或 runner 身份有误则返回 G3；
- 具体用例、runner、输入、路由或执行资产问题：返回 G3；支持域、源行为、验证义务或判定合同问题返回 G1；
- 环境事实变化：按恢复规则更新环境修订并计算失效范围；
- 仅证据索引缺失且事实未变：补充索引并重做完整性检查。

普通 finding 不等待批准。无法在冻结合同内修复时保留证据并将 G5 标为 `blocked`。

## 9. 关闭条件

- G1/G2/G3/G4 身份和证据未失效；
- 全部冻结用例一一对应，无 `FAIL`、`NOT_RUN` 或缺失；
- 正向、fallback、非法合同全部满足；
- 全部正式功能性能用例完成 Blaze msprof 并满足逐例门限；
- 双向审查、反模式、Convention、内部合同和 CMCT/CGMCT 检查通过；
- 代码、package、runner、结果和报告一致。

满足后 G5 为 `verified`，完成交接并自动进入 G6。
