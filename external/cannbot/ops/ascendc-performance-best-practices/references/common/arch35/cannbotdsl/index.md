# CANNBot-DSL 公共实现知识（arch35）

本目录沉淀**跨算子族**的 CANNBot-DSL 实现知识（arch35 / A5）：五段链方法论、DSL 编程模型要点与证据纪律。各算子族的 DSL 实现知识（族语义、实例与模块）放在 `<族>/arch35/cannbot-dsl/`（如 [FlashAttention 族](../../../flash_attention/arch35/cannbot-dsl/index.md)），由族级 `guide.md` 索引；本目录只收公共部分，不重复族内内容。

## 五段链方法论（优化手段 → DSL 代码）

| 层 | 回答的问题 | 承载位置 |
|---|---|---|
| 1. **数学语义** | 算子算什么，哪些数值语义必须保留 | 族库 `<族>/arch35/cannbot-dsl/optimize/1-math-semantics.md` |
| 2. **流水建模** | 数据在 Cube/Vector 两侧如何流动、节拍如何编排 | 族库 `optimize/2-pipeline-modeling.md` |
| 3. **容量方程** | tile 形状、缓冲深度、滞后如何从台账解出 | 族库 `optimize/3-capacity-equations.md` |
| 4. **积木模块** | 按类边界切分、可整体投放的模块（胶水随类走） | 族库 `assemble/modules/`（逐模块独立文件，便于复用/替换） |
| 5. **胶水模块** | 把模块粘成 kernel 的表达层规则与拼接件 | 族库 `assemble/glue-idioms.md` / `assemble/glue-fragments.md` |

通用规则：**先查算子范式 → 再查该范式的公共优化经验 → 再查特定语言的知识**；同构计算链可按族库"替换路径"逐模块复用，异构链按五段链逐层重建。

## DSL 编程模型总结（官方文档导读）

以下主题的完整定义以官方 [编程模型](https://cannbot-dsl.gitcode.com/programming-model/) / [API](https://cannbot-dsl.gitcode.com/api/)（0.7.0）为准，此处只总结与**高性能用法**相关的要点；样例依据见 [samples/test](https://gitcode.com/cann/cannbot-dsl/tree/85b619cd9be3/samples)（`master@85b619cd9be3`，cannbot-dsl 仓暂无 tag，官方打 tag 后替换）。不在此复制版本相关细节，避免随框架版本过期。

- [performance](https://cannbot-dsl.gitcode.com/programming-model/performance)：性能口诀与优化入口。核心是让 Cube/Vector/MTE 各 pipe 占满且互相藏匿；落地见各族库的流水建模与容量方程。
- [data-and-layout](https://cannbot-dsl.gitcode.com/programming-model/data-and-layout)：数据布局与视图。NZ 分形、physical stride 与逻辑 shape 的区别直接影响拆块拷贝与 L0 装配（族库胶水惯用法 §1）。
- [onchip-memory](https://cannbot-dsl.gitcode.com/programming-model/onchip-memory)：Buffer/Channel 的容量、depth 与存活期语义。台账按物理 capacity 计费，不是软件调度器（族库容量方程 §0）。
- [vector-registers](https://cannbot-dsl.gitcode.com/programming-model/vector-registers)：寄存器数据流与 VF 链。量化/scale/cast 热链应驻留寄存器链，减少 UB 往返；寄存器压力与 ICache 成本需在完整调用中核验。
- [compute](https://cannbot-dsl.gitcode.com/programming-model/compute)：Cube/Vector 计算表达。MM 的 shape/dtype/layout 与随路操作决定指令选择（族库积木模块）。
- [data-types](https://cannbot-dsl.gitcode.com/programming-model/data-types)：dtype 与转换规则。降低输入/累加精度不是默认授权手段，舍入点/饱和/FTZ 逐项核验。
- [synchronization](https://cannbot-dsl.gitcode.com/programming-model/synchronization)：同步语义。编译器结合 Tensor 访问生成同步；不能删除仍负责可见性或生命周期的同步（族库流水建模通用规则）。

## 跨族专题经验（暂记，未沉淀为库）

量化/寄存器热链——统计量、scale、cast/pack、累加 dtype 的驻留链评估，来源 flash_attn_fp8_fullquant、quant_*_indexer 等样例；metadata 生命周期——AI CPU metadata 的生成/复用边界与完整调用口径，来源 qsa_indexer、quant_sparse_lightning_indexer_metadata_dsl。两者待对应算子族建库时按五段链展开。

## 证据纪律

- 每条实践注明：family/arch/版本、触发证据、修改机制、正确性约束、编译观察、反例、验证状态与来源。`documented` 只表示文档/源码有依据；`compiled`、`precision_verified`、`performance_verified` 必须引用独立真实产物及精确 case/环境范围。
- 源码修改后先确认参数进入实际编译变体（host/Kernel 构造配置、TensorSpec/Dim/stride、分支、目标和数值模式）；构图期 Python 已完成的计算不消耗设备 scalar，不能把 C++ 成员局部化/常量外提的收益套到 Python。
- 每项修改有预期生成变化（展开/硬件循环、指令与同步、UB 中转、加载/存储、常量或 launch 拓扑）；查生成 AscendC/MLIR 与真实设备行为，无变化时区分未选中路径、配置未生效、lowering 与缓存/身份问题。
- 诊断 dump/手改生成代码/设备插桩的结果不参与正式终选。减少专门化可改善编译/冷启动成本，但可能增加动态分支并降低稳态性能，两类成本分别报告。
- 性能提升不从 README 图表或 AscendC 历史收益继承；实例结论迁移到其他算子时按族库数学语义的同构判定重新取证。

## 建库规则（新算子族）

1. 在对应代际目录第一层下建语言子目录 `<族>/arch35/cannbot-dsl/`，按五段链自包含建库，由该族代际 `guide.md` 索引（SKILL.md「扩展新算子族」第 4 条）。
2. 公共方法（本目录）不改；族内新共同机制修改族级原文并链接语言实例，语言专有机制进族库，不维护相同算法的平行手册。
