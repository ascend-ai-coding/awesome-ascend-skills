# 修复建议与证据链汇聚（P5）

> **使用时机**：诊断流程Phase 5（证据链汇聚与修复建议）执行前必读本文件。前四个阶段产出的报错行、错误位、dump（异常数据落盘文件）判读与复现结果在本阶段汇聚为结论码与修复建议；本文件给出四证据链汇聚框架、结论码优先级链、修复建议映射表、复现失败升级路径与证据链格式诊断报告模板。
>
> **引用约定**：本文引用均为`path:line`形式：oam-tools 相关路径相对于 CANN 社区仓 cann/oam-tools 根目录；`sd_doc/...` 为内部架构文档工作区路径（仅溯源标注，不在 oam-tools 仓内）。行号为源文件真实行号，可疑时按行号回读源码核实。正文中aicore_error_parser.py、aic_error_info.py、constant.py、dump_data_parser.py均为`oam-tools/src/msaicerr/ms_interface/`目录下同名文件的简写，msaicerr.py为`oam-tools/src/msaicerr/msaicerr.py`的简写。

---

## 一、四证据链框架

msaicerr把AI Core Error定位拆成四条相互独立的证据链，每条链单独能跑、单独能失败，最后汇聚成一个结论码。四链独立性是工具最重要的设计属性：任一条链因数据缺失而失效，其余三条照常给出结论，而不是整体失败（sd_doc/msaicerr-arch-doc/msaicerr-doc/README.md:10-19）。四链权威定义如下（sd_doc/msaicerr-arch-doc/msaicerr-doc/README.md:12-17）：

| 证据链 | 输入 | 回答的问题 | 对应结论码 |
| --- | --- | --- | --- |
| 寄存器解码 | plog（进程日志）中的error code | 硬件报了哪一类异常（176个错误位） | 103 HARDWARE_ERR |
| 反编译 | 算子.o与PC（Program Counter，程序计数器）偏移 | 出错在算子源码的哪一行 | 101 SINGLE_OP_ERR |
| dump校验 | exception dump二进制 | 输入数据本身是否已经是脏的 | 104 OPERATOR_INPUT_DATA_ERR |
| 单算子复现 | 算子.o/.json与tiling（算子切分参数） | 脱离模型能否稳定复现，args（参数）有没有被踩 | 105/106/107 |

其中寄存器解码链的103结论辅以golden op（标杆算子）环境自检：判定字段env_available由解析流程的Step 10在故障环境现场编译运行标杆算子填充（aicore_error_parser.py:1925-1935），自检失败时结论链输出"Failed to execute the built-in sample operator. Check the environment."（aic_error_info.py:253-254）。

### 1.1 各链证据来源与报告落点

工具解析完成后产出"结论加六节"的info.txt报告（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/conclusion/conclusion.md:254-289）。每条证据链的事实落在固定报告节，判定字段与结论码的对应关系如下：

| 证据链 | 判定字段（由谁填充） | 报告落点 | 命中结论码 |
| --- | --- | --- | --- |
| 寄存器解码 | error_code/error_code_all（plog提取后按错误位字典解码） | 第2节AI Core DFX Register（aic_error_info.py:186-188） | 103（判定字段env_available由Step 10标杆算子填充，aicore_error_parser.py:1925-1935） |
| 反编译 | 报错行与出错指令上下文 | 第3节Operator Error Line Number（aic_error_info.py:190-191） | 101（经单算子复现落地）；args执行前后比对落在第4节（aic_error_info.py:195-196），支撑106 |
| dump校验 | data_dump_result（Step 7落盘校验，aicore_error_parser.py:1892、1896）；dump_info中的"data invalid"子串（dump_data_parser.py:191-194） | 第5节Operator Dump File Parsing（aic_error_info.py:198-200） | 104（data invalid子串）、102（落盘失败） |
| 单算子复现 | single_op_test_result（Step 9）；atomic_clean_check（Step 6，aicore_error_parser.py:1877）；atomic_add_err（aicore_error_parser.py:1878-1882） | 第6节Execution Result of the Single-Operator Test Case（aic_error_info.py:202-205） | 101、105、107 |

### 1.2 汇总表填写方法

汇聚阶段对每条链各填一行：链名、链上事实、判定字段取值、是否支撑某结论码。填写规则：

1. 事实列只写工具实际产出的内容（报告节原文、错误位名、日志关键行），不写推测。
2. 判定列写结论码编号与常量名（见第二节），并注明该码在优先级链中的位置。
3. 某条链数据缺失时明确写"本链失效（原因）"，不允许留空或臆造。
4. 四链汇总完成后，按第二节优先级链取第一个命中的结论码作为最终结论。

## 二、结论码优先级链（get_return_code）

结论码101-107不是工具错误，而是分析结论：工具跑成功了，101-107就是诊断结果；1-11段才是工具自身执行状态（sd_doc/msaicerr-arch-doc/msaicerr-doc/README.md:79-81）。结论码语义（constant.py:72-78）：

| 结论码 | 常量名 | 语义 |
| --- | --- | --- |
| 101 | MS_AICERR_SINGLE_OP_ERR | 检查到单算子运行错误 |
| 102 | MS_AICERR_MEMORY_ALLOCATION_ERR | 检查到内存分配错误 |
| 103 | MS_AICERR_HARDWARE_ERR | 检查到硬件错误 |
| 104 | MS_AICERR_OPERATOR_INPUT_DATA_ERR | 检查到算子输入数据错误 |
| 105 | MS_AICERR_FRAMEWORK_MEMSET_MISSING | 检查到框架未执行memset（内存清零） |
| 106 | MS_AICERR_OPERATOR_ARGS_OVERWRITTEN | 检查到算子args被踩 |
| 107 | MS_AICERR_ATOMIC_OPERATOR_OVERFLOW | 检查到atomic（原子）算子遇到溢出数据导致错误 |

get_return_code是一条elif链，顺序即优先级，先命中先返回（aicore_error_parser.py:1962-1980）。实际触发顺序为105→107→104→106→102→101→103→0，逐级条件如下：

| 级序 | 结论码 | 触发条件（源码条件转述） | 源码锚点 |
| --- | --- | --- | --- |
| 1 | 105 | atomic_clean_check为假：图中算子前未插入memset/atomic_clean | aicore_error_parser.py:1964-1965 |
| 2 | 107 | atomic_add_err为真：错误码为0x800000且plog含dha status 1 | aicore_error_parser.py:1966-1967（求值条件见1878-1882） |
| 3 | 104 | dump_info含"data invalid"子串：dump解析发现输入NaN/INF | aicore_error_parser.py:1968-1969 |
| 4 | 106 | check_args_result为假：args执行前后不一致 | aicore_error_parser.py:1970-1971 |
| 5 | 102 | data_dump_result为假：dump落盘失败 | aicore_error_parser.py:1972-1973 |
| 6 | 101 | single_op_test_result不等于RetCode.SUCCESS（FAILED与NOT_RUN均命中） | aicore_error_parser.py:1974-1976 |
| 7 | 103 | env_available为假：标杆算子自检失败 | aicore_error_parser.py:1977-1978 |
| 8 | 0 | 以上全部不成立，兜底返回无错误 | aicore_error_parser.py:1979-1980 |

优先级链有ST测试锚定：test_get_return_code先把全部判定字段同时置为命中态断言返回105，再按各级顺序逐级翻转判定字段并断言返回码随之变化（oam-tools/test/st/msaicerr/testcase/test_aicore_error_parse_st.py:464-509）。

排布逻辑是"从可直接修的原因排到最难改的原因"：105/107可归因于框架或数据，一旦命中用户有直接可执行动作；104输入脏数据问题在上游算子，同样可归因到具体位置；103硬件错误排最后，只有其他可能都排除后才归因硬件，避免一有异常就报硬件故障（sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/error_code.md:72-80）。

### 2.1 四个必须知道的判读细节

1. **107在返回码层面实际不可达**：atomic_add_err仅当错误码恰为0x800000且atomic_clean_check为假时才求值（aicore_error_parser.py:1878-1882），而atomic_clean_check为假会先命中105，两个条件互斥（sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/error_code.md:99-106）。结论文案同样先命中memset缺失分支（aic_error_info.py:212-216在前，227-231在后）。判读报告时不要期望见到返回码107。
2. **NOT_RUN同样命中101**：源码注释明确NOT_RUN也返回非0，用例没跑起来时不能认为没有发现问题（aicore_error_parser.py:1974）。SK（SuperKernel，融合算子）场景下单算子未执行，字段保持NOT_RUN，只要前五级判据未命中，返回码就是101，把"没跑"报成了"单算子运行错误"；报告文本无此混淆（sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/error_code.md:127）。
3. **flag_check、AI Core数量不足、current_pc为0x0没有专属返回码**：这三种情形只进结论文案（aic_error_info.py:217-226），返回码会落到后续分支或0，依赖返回码做告警会漏掉它们（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/conclusion/conclusion.md:155）。
4. **结论文案链与返回码链排序不同**：典型组合是args被踩（check_args_result为假）同时复现成功（single_op_test_result为FAILED），此时结论文案链先命中单算子分支说算子逻辑可能有误，返回码链先命中106返回args被踩（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/conclusion/conclusion.md:148-153）。人工读文案、脚本读返回码时预期到这个差异。

## 三、修复建议映射表

映射表每行为一个触发条件到修复动作的四元组：触发条件（结论码或错误位）、根因方向、修复建议、验证方法。使用约束：

1. 错误位名称以constant.py:155-369（Stars形态，ASCEND_910B芯片）与constant.py:378-1123（David形态，ASCEND_950芯片）为唯一名称源（芯片形态划分见constant.py:41-49），表中名称逐字取自字典。
2. 根因方向区分两类：带path:line的是代码或文档依据；标注"经验"的是排查方向建议，无代码直接依据，不构成确定性根因。
3. 结论码触发与错误位触发同时存在时，以结论码行为主、错误位行为根因细化。
4. 复现失败场景只给排查方向，不给确定性根因（见第四节）。

| 触发条件（结论码/错误位） | 根因方向 | 修复建议 | 验证方法 |
| --- | --- | --- | --- |
| 结论码105 | 图中算子前未插入memset/atomic_clean，算子读到未清零内存（aicore_error_parser.py:1872-1877；结论链文案aic_error_info.py:212-216） | 排查GE（Graph Engine，图引擎）侧清零算子未插入的原因（如图优化移除，经验），必要时由框架侧在算子前补插清零算子 | 复跑后plog能检索到AtomicLaunchKernelWithFlag_<node_name>（aicore_error_parser.py:1466-1474的判定命令），且AI Core Error不再复现 |
| 结论码107（返回码层面实际不可达，见2.1节第1条） | atomic add累加精度溢出：错误码0x800000且plog含dha status 1（aicore_error_parser.py:1878-1882、795-805；结论链文案aic_error_info.py:227-231） | 检查算子精度与累加数值范围；任务在NPU上并发执行时可能是误报（aic_error_info.py:230） | 修复后复跑，plog不再出现dha status 1（aicore_error_parser.py:795-805的检索条件） |
| 结论码104 | dump解析发现输入张量含NaN/INF，dump_info出现"data invalid"（dump_data_parser.py:191-194） | 沿数据流向上游追溯产出该输入的算子或数据处理脚本，修正脏数据来源 | 修复上游后复跑业务，或用msaicerr -d复查该dump输入，摘要不再出现NaN/INF提示（判据同dump_data_parser.py:191-194） |
| 结论码106 | 算子args执行前后不一致，可能存在内存越界踩踏（aicore_error_parser.py:93-103；结论链文案aic_error_info.py:242-247） | 按结论链建议使用内存错误检测模型定位越界（aic_error_info.py:245-247），可用msSanitizer或op_debug_config=oom（见第四节升级路径） | 修复越界后复跑，报告第4节args before与after两列表一致（aic_error_info.py:195-196） |
| 结论码102 | exception dump落盘失败，输入输出内存可能非法（判据为D2H failed或地址可疑日志，aicore_error_parser.py:1377-1395、1400-1424） | 检查算子输入输出内存是否被提前释放或越界改写；dump落盘本身失败时联系GE侧处理（aicore_error_parser.py:1370-1372） | plog不再出现[Dump][Exception] D2H failed（aicore_error_parser.py:1377-1390的检索条件），复跑不再报错 |
| 结论码101（含NOT_RUN，见2.1节第2条） | 单算子用例复现出AI Core Error，算子逻辑可能有误（aic_error_info.py:237-241）；NOT_RUN表示用例未跑起来（aic_error_info.py:164-165） | 复现成功时结合报告第3节报错行修改算子代码；NOT_RUN时先核对.o与.json文件完整且匹配，再查单算子日志失败原因（aic_error_info.py:255-260） | 修复后重跑单算子用例，日志不再出现错误串与kernel名共现（判定方法aicore_error_parser.py:1548-1560） |
| 结论码103 | 标杆算子环境自检失败，环境异常（aicore_error_parser.py:1925-1935；msaicerr.py:196-199） | 按自检日志排查芯片不兼容、驱动问题、依赖缺失（msaicerr.py:197-198列举）；换卡复测排除单卡硬件故障（经验） | python3 msaicerr.py -e -dev <device_id>自检通过，输出环境正常（msaicerr.py:192-194） |
| 结论码0（兜底） | 全部判据未命中，0不代表故障不存在（sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/error_code.md:80） | 逐节核对六节报告证据，证据不足时按第四节升级路径处理，不臆断根因 | 补充收集故障信息后重跑msaicerr -p，观察结论码变化 |
| 错误位biu_l2_read_oob（Stars位0，constant.py:156） | 总线读访问错误，字典建议检查L2侧代码（constant.py:156） | 核对算子L2相关读地址计算与边界（经验） | 修复后重跑msaicerr -p，报告第2节该错误位不再出现（经验） |
| 错误位biu_l2_write_oob（Stars位1，constant.py:157） | 总线写访问错误，字典建议检查L2侧代码（constant.py:157） | 核对算子L2相关写地址计算与边界（经验） | 重跑后报告第2节该错误位不再出现（经验） |
| 错误位vec_ub_wrap_around（Stars位62，constant.py:228） | VEC（向量单元）指令读写UB（Unified Buffer，片上统一缓冲区）地址越界绕回（constant.py:228） | 检查UB空间分配与单次搬运块大小、循环次数是否匹配（经验） | 重跑后错误位消失，且单算子复现验证通过 |
| 错误位vec_ub_addr_wrap_around（Stars位154，constant.py:335）与VEC_UB_WRAP_AROUND（David位341，constant.py:1026-1029） | UB访问地址越出范围（constant.py:335、1026-1029） | 检查tiling切分后单次处理数据量是否超出UB容量（经验） | 重跑后错误位消失 |
| 错误位mte_gdma_read_overflow（Stars位32，constant.py:195）与MTE_GDMA_READ_OVERFLOW（David位81，constant.py:542-545） | MTE（数据搬运单元）读片上buffer地址越界（constant.py:195、542-545） | 核对DataCopy类搬运的源地址、shape（形状）与stride（步长）计算（经验） | 重跑后错误位消失 |
| 错误位mte_gdma_write_overflow（Stars位33，constant.py:196）与MTE_GDMA_WRITE_OVERFLOW（David位82，constant.py:546-549） | MTE写片上buffer地址越界（constant.py:196、546-549） | 核对搬运目的地址与目的空间大小（经验） | 重跑后错误位消失 |
| 错误位mte_ub_wr_ovflw（Stars位124，constant.py:299）与mte_ub_rd_ovflw（Stars位125，constant.py:300） | MTE写/读UB越界（constant.py:299-300） | 检查搬运目的/源UB偏移与单块长度（经验） | 重跑后错误位消失 |
| 错误位ccu_ub_overflow_err（Stars位135，constant.py:313） | scalar（标量）读写UB地址越界（constant.py:313） | 检查scalar侧UB指针推进逻辑（经验） | 重跑后错误位消失 |
| 错误位fixp_err_read_ub_ovflw（Stars位104，constant.py:278）与fixp_err_write_ub_ovflw（Stars位106，constant.py:280） | FIXP读写UB越界（constant.py:278、280） | 检查FIXP指令的UB地址参数（经验） | 重跑后错误位消失 |
| 错误位CUBE_FM_ADDR_OVERFLOW（David位35，constant.py:461-464） | matmul左矩阵或卷积feature map超出L1（constant.py:461-464） | 减小单次matmul/conv（卷积）的切分规模，使左矩阵适配L1（经验） | 重跑后错误位消失 |
| 错误位cube_l0a_wrap_around（Stars位12，constant.py:170）与CUBE_L0A_WRAP_AROUND（David位5，constant.py:395-398） | L0A操作地址超出L0A最大范围（constant.py:170、395-398） | 核对matmul分块后的L0A占用（经验） | 重跑后错误位消失 |
| 错误位vec_inf_nan（Stars位54，constant.py:220）、vec_idata_inf_nan（Stars位156，constant.py:337）与VEC_ERR_IDATA_INF_NAN_T0（David位336，constant.py:1006-1009） | VEC指令输入数据为INF或NAN（constant.py:220、337、1006-1009） | 同结论码104沿数据流追上游脏数据；并检查算子内部是否自行产生NaN，如0除0、无穷大相减（经验） | 输入修复后复现验证不再报错 |
| 错误位ccu_inf_nan（Stars位72，constant.py:238）与SU_CCU_INF_NAN_T0（David位262，constant.py:840-843） | scalar浮点指令输入为nan/inf（constant.py:238、840-843） | 检查scalar计算分支的除零与无穷大参与运算（经验） | 重跑后错误位消失 |
| 错误位cube_invld_input（Stars位9，constant.py:167）与CUBE_INVLD_INPUT（David位4，constant.py:391-394） | L0a/L0b回读数据为INF或NAN（constant.py:167、391-394） | 检查matmul/conv输入矩阵数据有效性（经验） | 输入修复后复现验证不再报错 |
| 错误位ccu_lsu_atomic_err（Stars位139，constant.py:318）与SU_CCU_LSU_ATOMIC_ERR_T0（David位275，constant.py:887-891） | scalar原子指令访问已被修改但未回写的GM（constant.py:318-319、887-891） | 检查原子操作地址合法性与并发写回顺序（经验） | 重跑后错误位消失 |
| 错误位mte_atm_addr_misalg（Stars位74，constant.py:241）与MTE_ATM_ADD_ADDR_MISALIGN（David位79，constant.py:534-537） | MTE原子指令地址未按数据类型位宽对齐（constant.py:241-242、534-537） | 调整原子搬运起始地址至位宽对齐（经验） | 重跑后错误位消失 |
| 错误位vec_div0（Stars位52，constant.py:218）与ccu_div0（Stars位3，constant.py:159） | VEC倒数除零、scalar除零（constant.py:218、159） | 检查分母来源数据并补充保护逻辑（经验） | 重跑后错误位消失 |
| 错误位ccu_neg_sqrt（Stars位7，constant.py:165）与vec_neg_sqrt（Stars位58，constant.py:224） | 开方或倒数开方输入为负数（constant.py:165、224） | 检查输入数据范围或增加clamp（截断）保护（经验） | 重跑后错误位消失 |
| 错误位cube_l0c_ecc（Stars位16，constant.py:174）、mte_l1_ecc（Stars位40，constant.py:203）、vec_ub_ecc（Stars位60，constant.py:226） | 多位ECC（Error Checking and Correcting，错误校验纠正）错误，字典原文指引见RAS告警处理（constant.py:174、203、226） | 按RAS告警流程处理；同算子同输入复测，区分偶发翻转与持续硬件故障（经验） | 复测不再出现ECC位视为偶发；持续复现则按硬件问题上报（经验） |
| 错误位ifu_bus_err（Stars位20，constant.py:178） | 取指地址非法：应用提前卸载算子二进制或栈破坏（constant.py:178-179） | 检查算子生命周期（.o是否被提前卸载）与调用栈完整性（经验） | 重跑后错误位消失 |

完整错误位字典位置：constant.py:155-369（Stars形态176位）与constant.py:378-1123（David形态187条），分类解读另见sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/register_error_dict.md。本表只收高频错误位，映射未覆盖的错误位先查字典含义，再按本表已覆盖的根因方向（越界读写、UB越界绕回、原子溢出、输入数据非法、地址未对齐、ECC硬件类）就近归类。

## 四、复现失败升级路径

单算子复现结果为RetCode.SUCCESS（用例执行完成但未复现出AI Core Error）时什么都没证明，不能据此认定算子无问题（aic_error_info.py:160-162；sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:276-279）。此时工具结论链固定给出三条排查建议（aic_error_info.py:261-271），结合SK场景补第四条。复现失败场景下只给以下排查方向，不给确定性根因：

1. **msSanitizer复查**：用msSanitizer工具再次检查算子（aic_error_info.py:266建议(1)原文）。
2. **开内存错误检测复跑**：若其他算子存在越界访问，开启op_debug_config=oom内存错误检测后检查算子（aic_error_info.py:267-269建议(2)原文）。
3. **SK场景关融合重跑**：融合（SuperKernel）场景定位不清时，关闭算子融合后重跑，让故障以单算子形式复现，四条证据链即可全部生效（sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:353；问题指引见sd_doc/msaicerr-arch-doc/msaicerr-doc/README.md:70）。
4. **框架侧上报**：怀疑框架问题时联系技术支持（aic_error_info.py:270建议(3)原文）。

## 五、证据链格式诊断报告模板

每个诊断结论必须包含事实、对比、判定、置信度四要素，并附反证（counter_evidence）。格式参照oam-tools/skills/cann-npu-perfanalysis/SKILL.md:29的证据链判定要求（每个诊断必须包含事实指标、阈值对比、判定理由、置信度）与oam-tools/skills/cann-npu-perfanalysis/SKILL.md:407-413的counter_evidence字段样式。模板如下：

```markdown
## AI Core Error诊断结论：<一句话概述>

**最终判定**：<结论码与常量名>（置信度：<高/中/低>）

### 证据链汇聚
| 证据链 | 事实 | 对比（判据与出处） | 判定 |
| --- | --- | --- | --- |
| 反编译 | <报错行/出错指令，出处报告第3节> | <对照源码行语义> | <是否支撑101> |
| 寄存器解码 | <错误位名，出处报告第2节> | <对照constant.py字典含义> | <是否支撑103> |
| dump校验 | <张量统计结论，出处报告第5节> | <NaN/INF即data invalid，dump_data_parser.py:191-194> | <是否支撑104> |
| 单算子复现 | <三变体结果，出处报告第6节> | <对照第二节优先级链> | <是否支撑105-107> |

### 反证（counter_evidence）
<列出与主判定矛盾或缺失的证据，例如：dump缺失导致104链失效；任务并发执行时atomic告警可能是误报（aic_error_info.py:230）>

### 修复建议
<按第三节映射表给出，逐条标注依据来源与"经验"项>

### 验证方法
<按映射表验证列执行，写明复查命令与判读要点>
```

置信度口径（本skill定义）：

- **高**：结论码由工具判定字段直接触发，且至少两条证据链相互印证，无未解释的反证。
- **中**：仅一条证据链支撑结论，或存在可解释的反证（如并发误报风险、SK场景精度受限）。
- **低**：结论码为0兜底或证据链大面积缺失（如无dump可用、复现NOT_RUN），只输出排查方向。

填写要求：事实列只写工具产出；对比列必须给出判据出处（path:line或字典行号）；判定列写结论码；反证列不得省略，无反证时写"无"。
