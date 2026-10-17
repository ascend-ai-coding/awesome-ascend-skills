# 工具链编排：上游分工、退出码与降级规则（P0/P1/P2/P5）

> **使用时机**：Phase 0 环境与输入检查、Phase 1 收集准备、Phase 2 判读 msaicerr 退出码、Phase 5 结论码判定前必读本文件。本文件回答三个问题：哪类工具知识到哪个上游技能取、msaicerr 退出码如何判读、收集要素缺失时如何降级。
>
> **引用约定**：本文 path:line 引用：oam-tools 相关路径相对于 CANN 社区仓 cann/oam-tools 根目录；`sd_doc/...` 为内部架构文档工作区路径（仅溯源标注，不在 oam-tools 仓内）。行号为源文件真实行号，可疑时按行号回读源码核实。文中 constant.py、aicore_error_parser.py、collection.py 均指 `oam-tools/src/msaicerr/ms_interface/` 下的同名文件，msaicerr.py 指 `oam-tools/src/msaicerr/msaicerr.py`。

---

## 一、上游工具技能分工路由表

本技能是诊断工作流，不复制工具手册。工具用法的真源是 tools/ 域两个 toolkit 技能（知识依赖单向性：修改工具用法只在真源处更新，本技能与本文只引用不复制），执行到对应环节时按表取用：

| 知识类别 | 真源技能 | 覆盖内容 |
| --- | --- | --- |
| msaicerr CLI 六参数总表、三种运行模式（-p/-d/-e）、参数依赖规则（-out 须配 -p/-d、-dev 须配 -p/-e、-dtype 须配 -d）、分派优先级 | msaicerr-toolkit | 命令速查、参数总览、约束 |
| 环境准备三步（装 Toolkit、source set_env.sh、cd 工具目录）、ASCEND_OPP_PATH、python 版本、仅本地分析、RC 形态约束 | msaicerr-toolkit | 环境准备、硬约束 |
| 环境自检（-e，golden op 标杆算子）命令与输出判读 | msaicerr-toolkit | environment-check |
| 收集目录三要素排查（dfx/data-dump、dfx/ops、dfx/log/host/cann） | msaicerr-toolkit | 使用要点 |
| 8 个暂不支持分析的算子清单（MatmulAllReduce 类、AllGatherMatmul、MemSet 等） | msaicerr-toolkit | functions-and-restrictions |
| dump 文件解析（-d）、-dtype 数据类型转换、bfloat16ext 依赖 | msaicerr-toolkit | dump-parsing、dump-dtype-conversion |
| 多个 AI Core Error 只解析第一个（按日志时间取首次出现） | msaicerr-toolkit | 使用要点 |
| asys collect / asys analyze 用法与参数（task_dir、tar、output、-r、--path） | asys-toolkit | collect-and-launch、analyze-files |
| asys 自动收集的 6 个环境变量一致性清单（ASCEND_PROCESS_LOG_PATH、NPU_COLLECT_PATH、DUMP_GRAPH_PATH、ASCEND_WORK_PATH、ASCEND_CACHE_PATH、ASCEND_CUSTOM_OPP_PATH） | asys-toolkit | constraints |
| asys_output 收集目录树全景（software_info.txt、dfx 各子目录） | asys-toolkit | collect-and-launch 产物结构 |

## 二、msaicerr 退出码全表

退出码定义见 constant.py:58-78。0 表示成功；1-11 为工具状态码（工具自身执行状态）；101-107 为分析结论码（对故障根因类别的判定），是诊断报告的核心输入。

### 2.1 工具状态码（0-11）

| 退出码 | 常量名 | 含义 |
| --- | --- | --- |
| 0 | MS_AICERR_NONE_ERROR | 执行成功 |
| 1 | MS_AICERR_INVALID_PARAM_ERROR | 参数错误 |
| 2 | MS_AICERR_INVALID_PATH_ERROR | 路径错误（含 ASCEND_OPP_PATH 缺失、目录不可写、-out 与 -p 嵌套） |
| 3 | MS_AICERR_CONNECT_ERROR | 连接错误 |
| 4 | MS_AICERR_INVALID_DUMP_DATA_ERROR | dump 数据非法或缺失 |
| 5 | MS_AICERR_OPEN_FILE_ERROR | 打开文件失败 |
| 6 | MS_AICERR_EXECUTE_COMMAND_ERROR | 执行命令失败 |
| 7 | MS_AICERR_INVALID_CONFIG_DATA_ERROR | 配置数据非法 |
| 8 | MS_AICERR_INVALID_SLOG_DATA_ERROR | 日志数据不可用（无法从日志提取 AI Core Error 信息时输出空报告） |
| 9 | MS_AICERR_FIND_DATA_ERROR | 找不到数据 |
| 10 | MS_AICERR_GET_DRIVER_AICORE_NUMBER_ERROR | 获取驱动 AI Core 数量失败 |
| 11 | MS_AICERR_GET_RUNTIME_BLOCKDIM_ERROR | 获取 runtime blockDim 失败 |

常见触发场景（均已由代码或测试核实）：

- 退出码 2：ASCEND_OPP_PATH 未设置（msaicerr.py:290-293）；当前目录或 debug_info.txt 不可写（msaicerr.py:301-307）；当前目录或 -out 位于 -p 目录内（msaicerr.py:120-123）。
- 退出码 1：无参数（msaicerr.py:295-298）；未识别参数且无有效路径（msaicerr.py:316-318）。
- 退出码 103：`-e` 环境自检失败，无论样例算子运行失败还是过程抛异常，统一返回退出码 103（msaicerr.py:195-207）。
- 退出码 8：收集失败或日志中无法提取 AI Core Error 信息时，工具在 `aicerror` 目录写出空结论报告后返回退出码 8（aicore_error_parser.py:1819-1824）。
- 退出码 4：`-d` 解析路径校验失败或解析异常（msaicerr.py:161-171）。

### 2.2 分析结论码（101-107）

结论码语义为 constant.py 内中文注释原文（constant.py:72-78）：

| 结论码 | 常量名 | 语义 |
| --- | --- | --- |
| 101 | MS_AICERR_SINGLE_OP_ERR | 检查到单算子运行错误 |
| 102 | MS_AICERR_MEMORY_ALLOCATION_ERR | 检查到内存分配错误 |
| 103 | MS_AICERR_HARDWARE_ERR | 检查到硬件错误 |
| 104 | MS_AICERR_OPERATOR_INPUT_DATA_ERR | 检查到算子输入数据错误 |
| 105 | MS_AICERR_FRAMEWORK_MEMSET_MISSING | 检查到框架未执行 memset 清零 |
| 106 | MS_AICERR_OPERATOR_ARGS_OVERWRITTEN | 检查到算子 args 被踩 |
| 107 | MS_AICERR_ATOMIC_OPERATOR_OVERFLOW | 检查到 atomic 算子遇到溢出数据导致错误 |

## 三、结论码优先级链（get_return_code）

get_return_code 是一条 elif 链，顺序即优先级，先命中先返回（aicore_error_parser.py:1963-1980）。实际触发顺序为 **105→107→104→106→102→101→103→0**，逐级条件如下：

| 级序 | 结论码 | 触发条件（源码条件转述） | 源码锚点 |
| --- | --- | --- | --- |
| 1 | 105 | atomic_clean_check 为假：图中算子前未插入 memset/atomic_clean | aicore_error_parser.py:1964-1965 |
| 2 | 107 | atomic_add_err 为真：错误码为 0x800000 且 plog 含 dha status 1 | aicore_error_parser.py:1966-1967（求值条件见 1878-1882） |
| 3 | 104 | dump_info 含 "data invalid" 子串：dump 解析发现输入 NaN/INF | aicore_error_parser.py:1968-1969 |
| 4 | 106 | check_args_result 为假：args 执行前后不一致 | aicore_error_parser.py:1970-1971 |
| 5 | 102 | data_dump_result 为假：dump 落盘失败 | aicore_error_parser.py:1972-1973 |
| 6 | 101 | single_op_test_result 不等于 RetCode.SUCCESS（FAILED 与 NOT_RUN 均命中） | aicore_error_parser.py:1974-1976 |
| 7 | 103 | env_available 为假：标杆算子自检失败 | aicore_error_parser.py:1977-1978 |
| 8 | 0 | 以上全部不成立，兜底返回无错误 | aicore_error_parser.py:1979-1980 |

优先级链有 ST 测试锚定：test_get_return_code 先把全部判定字段同时置为命中态断言返回 105，再按各级顺序逐级翻转判定字段并断言返回码随之变化（oam-tools/test/st/msaicerr/testcase/test_aicore_error_parse_st.py:464-509）。

四个判读细节：

1. **107 在返回码层面实际不可达**：atomic_add_err 仅当错误码恰为 0x800000 且 atomic_clean_check 为假时才求值（aicore_error_parser.py:1878-1882），而 atomic_clean_check 为假会先命中 105，两个条件互斥（sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/error_code.md:99-106）。判读报告时不要期望见到返回码 107。
2. **NOT_RUN 同样命中 101**：源码注释明确 NOT_RUN 也返回非 0，用例没跑起来时不能认为没有发现问题（aicore_error_parser.py:1974）。
3. **flag_check、AI Core 数量不足、current_pc 为 0x0 没有专属返回码**：这三种情形只进结论文案，返回码会落到后续分支或 0，依赖返回码做告警会漏掉它们。
4. **结论文案链与返回码链排序不同**：人工读文案、脚本读返回码时预期到这个差异（详见 fix-suggestions.md 第二节）。

## 四、收集降级标记规则

诊断流程中，收集目录要素缺失时按以下规则标记降级并在报告中如实呈现，禁止隐瞒降级或臆断结论：

| 缺失项 | 工具侧行为（证据） | 诊断流程标记与后续动作 |
| --- | --- | --- |
| 缺 dump 文件 | Collection 校验报错并抛异常，收集失败返回 False（oam-tools/src/msaicerr/ms_interface/collection.py:340-358、399-408）；随后解析器写出空结论报告并返回退出码 8（aicore_error_parser.py:1819-1824） | 标记"缺 dump"；报错文案中的两种可能原因（dump 缺失、日志非报错时日志）需逐项向用户核实；后续复现阶段改走构造数据路径（见 reproduction-guide.md），不得基于无 dump 证据下 104 结论 |
| 缺 .o/.json 编译文件 | 定位链缺少反编译输入；info.txt 中 `cce file` 等字段为空（模板见 oam-tools/src/msaicerr/ms_interface/aic_error_info.py:180-182） | 标记"定位降级"；报错行定位能力受限，报告须注明定位结论置信度降低 |
| 缺 plog（dfx/log/host/cann 无日志） | 无法使用 msaicerr 提取 AI Core Error 信息（oam-tools/docs/zh/msaicerr/AI_Core_error_analysis.md:19）；收集阶段找不到 `[Dump][Exception]` 关键字即报错（collection.py:206-211） | 停止分析，要求用户重新收集或补齐日志后再进入后续阶段 |

补充两种相关场景：

- **空报告统一退出码 8**：收集失败或日志无法提取 AI Core Error 信息时，工具在 `aicerror` 目录写出空结论报告并返回退出码 8（aicore_error_parser.py:1819-1824）。见到退出码 8 即按"输入信息不可用"处理，检查三要素与日志时间窗。
- **SK 场景**：收集阶段在日志中命中 `Begin to dump callback exception` 即判定为 SK 场景，该场景只生成 host.o，没有 device .o/.json/.cce（collection.py:221-226），同样触发定位降级；SK 场景的判读与处理动作见 error-line-location.md。

降级标记必须写入最终诊断报告的"场景与定位"段，作为证据链的一部分；任何降级状态下给出的结论都必须标注置信度并附反证说明。
