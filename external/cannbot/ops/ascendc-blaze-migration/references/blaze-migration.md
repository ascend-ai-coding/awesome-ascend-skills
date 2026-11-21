# G2 Blaze 迁移开发与静态验收

## 目录

1. 前置条件与隔离
2. G2.1 交接输入核对
3. G2.2 Blaze 迁移开发
4. Blaze owner 与差异分类
5. 内部执行合同证明
6. G2.3 组件与内部合同检查
7. G2.4 迁移范围与反模式检查
8. G2.5 开发反馈编译与代码冻结
9. 关闭条件

## 1. 前置条件与隔离

G2 只有在 G1 `verified` 后启动。只使用 G0 获取并冻结身份的 `repo/blaze/` 来源仓和完整 `ops-tensor` checkout；不重新获取源码、不跟踪移动分支，不在 `repo/original/` 开发，不复用其可变构建目录、依赖缓存或安装根。

G2 只读 `environment-state.json` 中的工具链、目标架构和路径，不调用环境检查。正式 OPP 构建属于 G3，设备功能和性能验证属于 G4/G5。

## 2. G2.1 交接输入核对

开始开发前核对：

- G1 migration design、验证义务、逐字节规则、性能门限、构建和 review 计划哈希未变化；
- G0 原始实现、Blaze checkout、依赖和环境 revision 未失效；
- `repo/blaze/` 仍绑定计划中的初始 SHA 和依赖；
- G1 owner 矩阵覆盖全部活动行为。

任何不匹配返回其 owner 门禁，不带着不完整输入开始开发。

## 3. G2.2 Blaze 迁移开发

按以下顺序实施：

1. 建立 Policy、模板约束和入口分发，确保分发发生在禁止对象构造之前；
2. 在 Scheduler/Kernel 无损表达任务遍历、坐标、边界和尾块；
3. 将 Tensor、Buffer、layout、Copy 和 VF 迁移到 G1 指定 owner；
4. 将主计算放入设计 owner；源使用高阶 Matmul 时完整展开隐含数据通路；
5. 在 Kernel 组装原控制流、AIC/AIV 分支和同步协议；
6. 保持 TilingData 到 Params 的机械映射并证明最终使用点；
7. 完成组件测试、正负实例化、Host Tiling 和开发反馈编译检查。

发现 G1 遗漏的活动行为时返回 G1，不得在编排层临时保留低阶 API。

## 4. Blaze owner 与差异分类

| 行为 | 合理 owner | 禁止位置 |
|---|---|---|
| 编译期能力与合法组合 | Policy/traits | 来源仓随意分支 |
| 任务遍历、坐标和边界 | Scheduler/Kernel | Tile/Copy atom |
| ABI、启动、AIC/AIV 编排 | Kernel | Tile/Storage |
| Buffer 分配、槽位和生命周期 | Storage/BufferManager/Block | 来源仓 Adapter |
| GM/L1/L0/UB 搬运 | Tensor API、注册 Copy atom/Tile | Kernel/Block/Scheduler 低阶调用 |
| Cube Matmul 数据通路 | BlockMmad 或等价组件 | AscendC 高阶 Matmul 直接调用 |
| 无状态 VF/转换原子 | Tile/Utility | 全局 Scheduler |
| 主计算后处理 | Epilogue/对应组件 | 来源仓入口 |
| TilingData 到 Params 映射 | 来源仓 Adapter | 公共组件读取私有 Tiling |

主计算前处理保留在 Kernel 文件内的合理 Blaze 编排层级；Matmul 场景即 MMAD 前处理，不为单个算子创建私有 Prologue 层级。公共组件必须声明支持/拒绝范围、参数化轴、资源和同步约束，并提供正负实例化或契约测试。不支持的组合必须通过编译期约束或 `static_assert` 拒绝，不能等到设备运行时才暴露。

差异分类只用：

- `STRUCTURAL_MOVE`
- `API_MECHANICAL`
- `TENSOR_API_REFACTOR`
- `BLAZE_COMPONENT_EXTENSION`
- `RUNTIME_BOUNDARY`
- `MATMUL_EXPANSION`
- `FRAMEWORK_GLUE`
- `INACTIVE_REMOVAL`
- `IMPLEMENTATION_DRIFT`

减少资源、合并独立资源、改变轮转、降低并发或改变同步均为 `IMPLEMENTATION_DRIFT`，默认恢复源行为。确需改变时记录必要性及对应功能、性能验证义务。

## 5. 内部执行合同证明

至少证明：

- 每个源资源组有明确 Blaze 对应，独立资源未无依据合并；
- 数量来源和合法域保持，未静默常量化；
- 生产者与消费者在各阶段选择同一预期实例；
- 参数到达实际地址、索引、循环、资源选择或同步位置；
- 地址、容量、stride、offset、gap、packing 和对齐无越界、重叠或单位混用；
- 首次使用、稳定复用和结束处理生命周期闭合；
- 任务遍历、tail、VF、舍入、饱和和同步保持；
- 所有计划差异均绑定目标源证据。

编译成功、单一功能样例和参考实现相似不能替代证明。外部算子、历史实现和样例只用于确认能力或接口用法；其资源模型、Buffer 数、调度、layout 和同步必须重新对照目标源合同。

保持源 Set/Wait 协议，不得用 LOCK 机械替代源同步。`asc_lock/asc_unlock` 只能存在于已注册、具有公开 owner、支持范围和契约测试的 Buffer 管理组件内部，不能出现在 Kernel、Block 或 Scheduler 编排层。

### 高阶计算 API 展开

源活动路径使用高阶 Matmul 时恢复 GM 到 L1、L1 到 L0A/L0B/BT、MMAD flags、L0C/Fixpipe、transpose、layout、padding、tail、bias、转换、舍入、饱和、multi-buffer 和完成条件。其他高阶计算 API 从其源码和活动调用恢复，不套用 Matmul 模板。

## 6. G2.3 组件与内部合同检查

逐项核对组件支持/拒绝范围、正负实例化、资源与同步约束，以及第 5 节内部执行合同。组件证据、参数最终使用点和源行为映射存在缺口时不得进入反模式扫描。

## 7. G2.4 迁移范围与反模式检查

扫描目标活动路径、实际依赖闭包和改动实现文件。

### Kernel、Block、Scheduler 禁止

- `DataCopy`、`DataCopyPad`、相关 ExtParams 和直接 AscendC Copy；
- `TPipe`、`TBuf`、`TQue` 等手工 Buffer 管理；
- `LocalTensor`、`GlobalTensor` 作为直接搬运对象；
- 未封装的 `asc_copy_*` 或同类低阶 intrinsic；
- AscendC Matmul 高阶 API；
- 未经登记的 `asc_lock`、`asc_unlock` 或 LOCK 同步替代；
- 绕过 owner 的手工 packed 地址、layout 和资源管理；
- 未登记的运行时原语或低阶同步替代。

Tile、Copy atom、Storage 或 Tensor API 内部可使用必要低阶实现，但必须已注册、有公开入口、明确支持范围和契约测试，且实现不泄漏到编排层。运行时边界必须绑定 G1 范围条目、owner 和具体用途。每个组件的正向实例化、负向实例化或编译期拒绝结果必须写入 G2 证明索引。

所有 DAV_3510 迁移分别扫描目标活动依赖闭包和相对 G0 冻结身份的改动实现文件，两组 `CMCT`/`CGMCT` 实现命中必须为零。包装、别名和编译开关排除项需保存文件、行、用途和不可达/非实现证据；G3 正式构建后再核对 manifest 中的活动 Kernel 与该扫描集合一致。

## 8. G2.5 开发反馈编译与代码冻结

根据变更范围执行必要的增量编译、翻译单元编译、组件实例化和 Host Tiling 编译：

- 使用环境文件绑定的 CANN、编译器和目标 SoC；
- 保存编译目标、命令、日志、退出码和代码 SHA256；
- 证明开发反馈覆盖实际改动的模板和活动分支；
- 将迁移代码直接提交到 `repo/blaze/` 的 `master`，并记录 ops-nn 与 ops-tensor 的提交 SHA；
- 将通过检查的 Blaze 代码身份冻结并交给 G3。

本阶段的编译只解决 API、模板、类型和依赖问题，不建立正式 package 身份。不得把开发态增量包放入 `packages/blaze/`，不得运行正式功能/性能用例，不得以编译通过声明迁移等价。G3 必须从冻结身份重新执行正式干净构建。

### 代码冻结记录

G2 代码冻结记录至少包含：

- 来源仓、ops-tensor、G1 输入和最终代码身份；
- 环境 revision、CANN、编译器、SoC 和开发反馈编译命令；
- 组件测试、实例化、Host Tiling 和合同证明索引；
- 分层 API 扫描集合、命中、排除和结果；
- CMCT/CGMCT 两组扫描集合、排除和结果。

## 9. 关闭条件

- G1 交接输入完整且未失效；
- 每个迁移范围条目实现到正确 owner，无 `unknown`；
- 内部执行合同到最终使用点的证明完整；
- 组件支持/拒绝范围和契约测试完整；
- 编排层反模式命中为零，运行时边界均有登记；
- 两组 CMCT/CGMCT 实现命中为零；
- 开发反馈编译和静态检查绑定最终代码；
- 没有生成或冒充正式 OPP、runner、功能或性能结果。

满足后 G2 为 `verified`，完成交接并自动进入 G3。
