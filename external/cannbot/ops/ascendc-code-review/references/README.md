# 规则集总览

> **声明**：本表**仅供人读**，用于快速了解全部规则文件的分类与定位。机器路由（规则加载、侧别过滤、触发匹配）依赖各规则文件自身的「适用」声明头，**不依赖本表**。新增或修改规则文件时须同步更新本表（见 `docs/architecture.md` 5.1 节）。
>
> 最后更新：2026-09-28（KERNEL 条款精简：删除 ascendc-kernel-rules.md 10 条——原 KERNEL-1（禁用 GlobalTensor::SetValue/GetValue）/2（禁用 std:: 计算函数）/4（Compare 256 字节对齐）/6（禁止动态内存分配）/8（Cast RoundMode）/9（blockLen 单位差异）/11（循环内逐元素）/13（PipeBarrier）/14（流水线同步）/15（FP16 中间精度），重排编号消除空洞：旧 3→1、5→2、7→3、10→4、12→5、16→6、17→7、18→8、19→9、20→10、21→11，条款数 21→11，触发词删 GetValue/SetValue（22→20 个），检视检查清单移至快速索引之后并同步精简；cpp-style.md 中 KERNEL-18 引用同步改为 KERNEL-8；KERNEL-1 语义替换：由「DataCopy/DataCopyPad 对齐要求」改为「推荐使用 DataCopyPad，不推荐使用 DataCopy」（严重级别 高→中，措辞建议性、存量可保留，32 字节对齐与 blockLen 单位机理保留为不推荐依据，非对齐场景意见措辞升级为正确性缺陷；plan-design Step 5 筛查示例同步改称「DataCopy 不推荐」；KERNEL-2 检视方法重设计：由 grep 文本计数比对改为 5 步法（定位/范围过滤/队列分组核对/路径完备性检查/意见输出），并补充提前 return 漏 Free 错误/正确示例；KERNEL-4 拆分：核内 SetFlag/WaitFlag 配对检查独立成条（新增 KERNEL-10，含 HardEvent 模板参数/eventID 一致、双向路径完备、eventID 独占、禁连续 SetFlag 四类核心原则与 PR#4021/PR#8802 真实陷阱），KERNEL-4 全名化消除裸名歧义并重写检视方法为 4 步式，原 KERNEL-10/11 顺延为 11/12，kernel-rules 触发词补 SetFlag/WaitFlag（20→22 个），KERNEL-9 增加与 KERNEL-10 的去重边界；KERNEL-3 经 asc-devkit 官方文档核实：repeatTime≤255 无误（高维切分文档「由于硬件限制，repeatTime不能超过255」），修正受影响 API 错误举例（DataCopy 的 blockCount 实为 uint16_t、取值范围 [0,4095]，不受 255 限制），参数名 repeatTimes→repeatTime，新增连续计算（count 形态）优先推荐与步骤式检视方法；KERNEL-5 重设计：硬件参数定义表化（核数/UB 大小/Block 粒度/片上缓存四类，区分代际可变与架构不变量），修正正确示例（GetBlockDim→GetBlockNum 等系统变量 API，注明 GetRuntimeUBSize 仅 950 系列支持），检视方法改为 4 步三重过滤（名称+数值+常量清单候选定位 / npu-arch 代际值表比对 / 用途与来源判定），API 查阅表与检视检查清单补 KERNEL-5 条目；新增 KERNEL-11：禁止使用 PipeBarrier<PIPE_ALL>（核内同步，严重级别中，依据官方 PipeBarrier 文档性能约束 + 复用已删 KERNEL-13 的 PR#0309/PR#3019 真实案例，含细粒度替代方案表），原 KERNEL-11/12 顺延为 12/13；KERNEL-13 增加例外：`__simt_vf__`（线程级）VF 及其 callee 豁免——SIMT 按线程独立执行标量控制流，线程级分支守卫（边界保护）是编程模型常规形态，禁止范围收窄为向量级 `__simd_vf__` 及其 callee；LOG-2 增加例外：context 判空保护分支（`context == nullptr` / `!context`）内的日志无 context 可传，允许使用算子名字符串不判违规——豁免仅限判空分支体内，分支外仍须从 context 获取，存在其他算子名来源（如 aclnn 入参 opName）时仍优先使用；报告模板精简：文件/PR 检视报告改为「检视概览（文件数/代码行数/问题总数+高中低计数）+ 检视意见单一大表（条例编号/严重级别/问题描述/代码位置/修改建议）」，严重级别取条款声明（clause-review yaml 增 severity 字段、collector 校验），删证据表/代码片段/三态统计/旧概览占位符，style 与范围外并入大表计数，超长过滤阈值 5000→2000；设计一致性检查（docs-detect/design-check/doc-style.md/S1-S7+D8）与结果缺失条款报告标注停用删除（波次完整性校验与三级补派保留、静默执行），code-fetch 输出补变更文件数/变更行数）；2026-09-16（KERNEL 条款重排补洞：旧 KERNEL-6~10 → 5~9、旧 12 → 10、旧 13~17 → 11~15、旧 18~20 → 16~18，消除已删除条款原 KERNEL-5/11 留下的编号空洞，历史引用按此映射换算；GEN-4.2（魔鬼数字）增补专属检视方法：候选定位 grep 策略 + 豁免排除 + 概要常量清单交叉核对，条款数不变；新增 LOG-2：日志中算子名必须从 TilingContext 获取（优先直接传 context，GetNodeName() 合规，硬编码报错），LOG 文件触发词扩为 OP_LOGE/OP_LOGI/OP_LOGW/OP_LOGD 四宏；新增 KERNEL-19：核内同步推荐 Lock/Unlock、SetFlag/WaitFlag 不推荐使用（仅 950 代际报出，FAIL+HIGH 但措辞为建议迁移非强制，严重级别中）；新增 KERNEL-20：禁止 __VEC_SCOPE__，VF 任务须以具名 VF 函数（__simd_vf__/__simt_vf__）+ asc_vf_call 实现（严重级别高，kernel-rules 触发词补 __VEC_SCOPE__/asc_vf_call/__simd_vf__/__simt_vf__）；新增 KERNEL-21：VF 函数实现内部（含 callee 子函数）禁运行时分支 if/switch/三目（豁免 if constexpr 与 #if，机理=cmp 指令+无法生成硬件循环，严重级别高）；新增 HOST-3：获取的 platform 参数（coreNum/ubSize/blockSize 等）须校验取值合法性（GetPlatformInfo 判空、有符号 <=0 / 无符号 ==0 类型匹配拦截，host-rules 触发词补 GetPlatformInfo/GetCoreNumAic/GetCoreNumAiv/GetUbBlockSize）；新增 HOST-4：platform 参数须在 Tiling 方法中直接获取，禁止 TilingParse 经 compileInfo 传递给 Tiling（写端/读端任一命中即 FAIL，host-rules 触发词补 TilingParse））；2026-09-14（API 条款迁移：ascendc-api.md 拆分为 ascendc-kernel-rules.md（条款前缀已由 API 改为 KERNEL）+ ascendc-sunset-api.md（SUNSET-1/2）；ascendc-perf.md 5 条并入 kernel-rules 后删除；ascendc-op-conventions.md 4 条拆分迁入 kernel-rules 与 host-rules（HOST-2）后删除；高危修复：数字编号 5 文件补前缀 SEC/GEN/STYLE/RED/TOPK，共 13 个文件）

## 规则文件总表

| # | 规则文件 | 分类 | 条款前缀 | 语言 | 适用侧别 | 启用/触发机制 | 派发方式 | 条款数 | 内容定位 |
|---|---------|------|---------|------|---------|-------------|---------|--------|---------|
| 1 | cpp-secure.md | C++安全编码 | SEC | C++ | All | 必须触发（无条件启用，不参与内容筛查） | 通用分组（容量10） | 24 | 内存/指针安全、输入校验、资源管理、安全函数、LOG API 安全，红线级规范 |
| 2 | cpp-general.md | C++通用编码 | GEN | C++ | All | 语言+侧别匹配即启用 | 通用分组（容量10） | 39 | 代码设计、头文件、数据类型、常量、表达式、函数使用等通用编码规范 |
| 3 | cpp-style.md | 代码风格 | STYLE | C++ | All | 默认启用 | 专项子 agent 独立托管（组名 style） | 22 | 命名/格式/注释风格，结果以 yaml category: style 输出，违反计入报告问题数（severity 取条款声明） |
| 4 | ascendc-sunset-api.md | AscendC 日落检查 | SUNSET | C++ | All | 关键词触发（aclnn、aclrt、aclop） | 通用分组（容量2） | 2 | 跨侧别日落（废弃）API/头文件/库检查，预研报告驱动 + clause.get_sunset_api.py fallback |
| 5 | ascendc-host-rules.md | AscendC工程约定 | HOST | C++ | Host | 关键词触发（InferDataType、IMPL_OP、REG_OP、IMPL_OP_INFERSHAPE、GetPlatformInfo、GetCoreNumAic/Aiv、GetUbBlockSize、TilingParse） | 通用分组（容量2） | 4 | HOST-1 InferDataType 须放 op_graph 目录（禁 op_host 链式写法）；HOST-2 REG_OP 算子必须实现 InferDataType（互斥分工）；HOST-3 platform 参数须校验取值合法性（判空+类型匹配零值拦截）；HOST-4 platform 参数须在 Tiling 方法中直接获取（禁 TilingParse→compileInfo→Tiling 传递） |
| 6 | ascendc-log.md | AscendC日志 | LOG | C++ | Host | 关键词触发（OP_LOGE、OP_LOGI、OP_LOGW、OP_LOGD） | 通用分组（容量1） | 2 | LOG-1 裸 OP_LOGE 语义可映射时须替换为 log.h 预定义宏（EZ 码），意见须给出宏名+文件路径；LOG-2 日志中算子名须从 TilingContext 获取（优先直接传 context，GetNodeName() 亦合规），禁止硬编码算子名字符串，context 判空分支内豁免 |
| 7 | ascendc-red-line.md | 高频问题清单 | RED | C++ | All | 必须触发（无条件启用，不参与内容筛查） | 通用分组（容量3），**最高优先级** | 10 | 生产环境总结的红线问题（除零/越界/溢出/未初始化/数据竞争等） |
| 8 | ascendc-topk.md | 高频问题清单 | TOPK | C++ | All | 必须触发（无条件启用，不参与内容筛查） | 通用分组（容量3），**最高优先级** | 14 | 检视实践中高频风险点（野指针/特殊值/返回值校验/atomic 清零等） |
| 9 | mc2-specific.md | 领域专项 | MC2 | C++ | Host, Kernel | 关键词触发（hccl_、expert、dispatch 等 26 个） | 通用分组（容量5） | 19 | MC² 通算融合算子：通信同步/MoE 专家路由/量化精度/CCU 硬件约束 |
| 10 | simt-api-analysis.md | 领域专项 | SIMT | C++ | Kernel | 关键词触发（Simt::、simt_api/asc_simt.h 等） | 通用分组（容量5） | 16 | SIMT API 转换/保留规则（GetThreadNum→blockDim.x、UintDiv 保留等） |
| 11 | simt-constraints.md | 知识参考 | — | C++ | Kernel | — | 知识参考（无适用声明头，不参与自动路由） | — | SIMT 禁止事项清单（参数传递/启动方式/UB/TilingData/Dim3），由 load-balance 编排引用 |
| 12 | ascendc-kernel-rules.md | AscendC Kernel编码 | KERNEL | C++ | Kernel | 关键词触发（AscendC::、DataCopy、EnQue、TILING_KEY_IS 等 22 个） | 通用分组（容量5） | 13 | Kernel 侧编码规范（KERNEL-1~5 API 使用与硬件约定 + 6~8 TilingKey注册/派发/命名 + 9 核内同步推荐 Lock/Unlock + 10 SetFlag/WaitFlag 配对 + 11 禁 PipeBarrier<PIPE_ALL> + 12 禁 __VEC_SCOPE__ 用具名 VF 函数+asc_vf_call + 13 VF 内禁 if 分支），由 api+perf+op-conventions 合并 |

## 特殊机制注记

| 文件 | 特殊机制 |
|------|---------|
| ascendc-red-line.md / ascendc-topk.md | 最高优先级：独立成组、不参与跨条款合并、必进第一波（`steps/common.plan-design.md` Step 6.2/7） |
| cpp-style.md | 22 条单独成 1 组（组名 style），全文 Read，结果以 yaml `category: style` 字段输出（无文本前缀），不走假设检验，违反计入报告问题数（severity 取条款声明） |
| simt-constraints.md | 无适用/检视负载声明头，不进入 references 自动扫描；作为 SIMT 检视的知识补充被 `core/review-load-balance.md` 引用 |
| cpp-secure.md / ascendc-red-line.md / ascendc-topk.md | `触发: 必须触发` = 不依赖代码命中关键词，无条件激活且不参与 Step 5 内容筛查 |
| mc2-specific.md | 有 `排除场景` 声明（纯通信中转/纯计算无集合通信时排除），plan-design Step 3 读取应用 |

## 维护说明

- **新增规则文件**：放入 `references/` + 写适用/检视负载声明头即自动注册；同时在本表追加一行（按分类归位排序）
- **修改规则文件**（条款增删、触发词变化、侧别调整）：同步更新本表对应行的条款数/触发机制/侧别等列
- **删除规则文件**：同步删除本表对应行
- 各列取值须与文件实际声明头和快速索引一致，禁止凭记忆填写
