# 报错行定位：场景判定与错误行定位（P2）

> **使用时机**：诊断流程Phase 2（场景判定与报错行定位）执行任何判读前，必须先读本文件。本文件回答三个问题：当前故障属于哪类场景（L0/L1/ffts/SK）、如何执行`msaicerr -p`并判读`info.txt`、报错行定位链条（PC修正→符号化→反汇编→源码行）如何走通与如何回退。
>
> **引用约定**：本文 path:line 引用：oam-tools 相关路径相对于 CANN 社区仓 cann/oam-tools 根目录；`sd_doc/...` 为内部架构文档工作区路径（仅溯源标注，不在 oam-tools 仓内）。行号为源文件真实行号，可疑时按行号回读源码核实。工具行为均以源码为准；标注"技能动作"的条目是本诊断技能的编排约定，不是工具内置行为。

---

## 一、定位思路总览

msaicerr的报错行定位本质是一条"偏移量换算链"：plog中的`start_pc`与`current_pc`相减得到kernel内偏移，偏移在反汇编文本中对应到指令，再经`_loc.json`映射回cce与TBE源码行号（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/decompile/decompile.md:5）。这条链上每一步都可能断（PC不准、`.o`缺失、符号化工具缺失、`_loc.json`不存在），因此本文件的判读规则都给出断链时的回退动作。

工具在`msaicerr -p`执行时内部自动完成11步解析（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1802-1960），其中与P2直接相关的判读点：场景判定在全部步骤之前（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1809-1811），算子信息与PC修正对应Step 1（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1813-1817、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:540-543），反编译定位对应Step 5（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1859-1870），dump失败判定对应Step 7（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1884-1906）。最终结论写入`info.txt`并打印`Analysis info is saved in`提示（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1354-1358）。

## 二、场景判定（四场景关键词表）

### 2.1判定时机与原理

场景判定发生在解析流程最前面：`parse()`先执行`add_objdump_to_path()`，随后调用`check_plog_info()`用三条grep确定解析级别与场景标志（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1806-1811）。三条grep分别探测L1标记、ffts标记、SK标记（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:331-365），各自设置标志位`parse_level`/`ffts_flag`/`is_sk`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:84-88初始化，全流程分支都读这三个开关）。判定只依赖日志文案命中与否，不依赖芯片型号（sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:20）。

### 2.2场景判定表

| 场景 | grep关键词原文 | 命中后果 | plog证据样例 |
| --- | --- | --- | --- |
| L1 | `[AIC_INFO] dev_func:` | 命中则`parse_level = 1`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:348-349） | `[AIC_INFO] dev_func:te_gatherv2_657cb48fa1743a43209d7bc779fe8c294760a5b09b3079a3323fdf18376fc408_1__kernel0`（oam-tools/test/st/msaicerr/testcase/conftest.py:43） |
| L0 | `[AIC_INFO] dev_func:`（未命中） | 未命中则`parse_level = 0`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:350-351） | L0场景plog无dev_func打印，错误信息形态如`ProcessStarsCoreErrorInfo:...pc start: 0x12c042d73754, current: 0x12c042d75b18, vec error info: 0x99000000a2,...`（oam-tools/test/st/msaicerr/testcase/conftest.py:56） |
| ffts | `fftsplus task execute failed` | 命中则`ffts_flag = True`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:346-347） | `fftsplus task execute failed, device_id=0, stream_id=2, report_stream_id=2, task_id=6, flip_num=0, fault kernel_name=2_0_11_GatherV2, program id=1.`（oam-tools/test/st/msaicerr/testcase/conftest.py:42） |
| SK（SuperKernel） | `Begin to dump callback exception` | 命中则`is_sk = True`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:364-365） | `[ERROR] RUNTIME...PrintInfo:[Dump][Exception] Begin to dump callback exception. coreType=0, coreId=1, argAddr=0x12c200000000, argSize=64, binHandle=0x12c200000100, extraTensorNum=2, kernelName=Add_sk_kernel_900016000.`（oam-tools/test/st/msaicerr/testcase/conftest.py:71-75） |

判定命令（技能动作，与工具内部grep一致，作用域为收集目录）：

```bash
grep -rn "\[AIC_INFO\] dev_func:" <收集目录>
grep -rn "fftsplus task execute failed" <收集目录>
grep -rn "Begin to dump callback exception" <收集目录>
```

判读规则：第一条有输出为L1，无输出为L0；后两条有输出即置对应标志。三个标志相互独立、可同时为真（sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:18）。

### 2.3各场景对后续阶段的影响

**L0的影响**：算子名取自`Aicore kernel execute failed`行的`fault kernel_name=`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:250-282）；tiling取自plog的`tilingKey =`/`blockDim=`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:839-859）；只给汇编偏移，不做cce/TBE行号映射（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1116-1117）；二级指针解析执行（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1913-1914）；dump结论走`check_dump_result(dfx_message)`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1894-1896）。

**L1的影响**：算子名取自`[AIC_INFO] dev_func:`字段（架构对照见sd_doc/msaicerr-arch-doc/msaicerr-doc/architecture.md:279）；tiling取自`[AIC_INFO] tiling_key:`/`block_dim:`打印（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:820-837）；执行cce/TBE行号映射与模板算子基址修正（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1106-1115）；二级指针解析不做，工具会告警"当前为L1异常、含指针张量的算子不支持"（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1913-1920）；dump结论走`_get_data_dump_result()`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1891-1892）。**关键影响：L1场景若无有效dump数据（dump失败或未落盘），P4单算子复现不再使用dump输入，转构造数据路径（路径B）**——工具侧行为是`data_dump_result`为假时device单算子用例被跳过（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1765），此时P4按reproduction-guide.md的路径B（shape/dtype推断构造数据）继续。

**ffts的影响**：kernel名改从`fftsplus task execute failed`行提取（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:283-306）；L0下tiling_key改从`fftsplus aivector error`行的kernel名尾段提取（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:860-874）；单算子复现时args需前置`ffts_addrs_num`个c2c控制地址（sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:265-273）。**警惕D2H失败**：ffts场景dump判定与其他场景共用同一组失败特征，若plog含`[Dump][Exception] D2H failed`，则dump结论为失败、结论码102（详见本文件第十节）。

**SK的影响**：SK（SuperKernel）为多算子融合编译，只产出`host.o`，没有device侧`.o`/`.json`/`.cce`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:353）；算子名以SK标志打印内的`kernelName=`为准并覆盖常规解析结果（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:240-243、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:308-313）；反编译只复制`host.o`、跳过cce/TBE行号映射（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1099-1102）；device单算子测试跳过、`hash_id`无来源置空（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1762-1763；sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:93）。**对后续阶段的动作：给出关闭融合重跑建议**——SK场景定位不清时，建议用户关闭算子融合后重跑，让故障以单算子形式复现，四条证据链即可全部生效（sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:353；sd_doc/msaicerr-arch-doc/msaicerr-doc/README.md:70）。

注意：每个场景判定各依赖单条日志文案，若上游版本改了文案，标志会静默保持假值，表现为大量"文件找不到"告警而非明确报错；排查时应优先确认这三条打印是否存在（sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:46、sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:371-372）。

### 2.4 L0与L1的能力差异速查

| 环节 | L0 | L1 |
| --- | --- | --- |
| cce/TBE行号映射 | 不做，只给汇编偏移 | 做，经`_loc.json`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1106-1117） |
| 模板算子err pc基址修正 | 不做 | 做（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1107-1109） |
| 二级指针解析 | 做 | 不做，打印不支持告警（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1913-1920） |
| dump结论来源 | `check_dump_result(dfx_message)` | `_get_data_dump_result()`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1891-1896） |
| workspace大小 | 恒0 | 从workspace的npy计算（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:422-433） |
| 报错指令上下文（10条） | 不做 | 做（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1110-1113） |

（完整差异表另见sd_doc/msaicerr-arch-doc/msaicerr-doc/architecture.md:277-287。）

## 三、msaicerr -p执行规范

命令格式（oam-tools/docs/zh/msaicerr/AI_Core_error_analysis.md:23-25）：

```bash
python3 msaicerr.py -p <收集目录> -out <结果目录> -dev <Device ID>
```

参数约束：

- `-p`必选，指定故障信息收集目录；**不得进入`-p`指定目录或其子目录内执行msaicerr**，否则工具解析卡住或失败（oam-tools/docs/zh/msaicerr/AI_Core_error_analysis.md:29；源码禁令oam-tools/src/msaicerr/msaicerr.py:120-123：`Do not run msaicerr in the directory specified by -p or its subdirectory. Make sure -out specifies a different directory (including its subdirectory) from -p.`）。
- `-out`可选，不指定时默认用当前目录（oam-tools/src/msaicerr/msaicerr.py:114-116）；不得为`-p`目录或其子目录（oam-tools/src/msaicerr/msaicerr.py:120-123）。
- `-dev`可选，默认0，用于内置算子样例的环境自检（oam-tools/docs/zh/msaicerr/AI_Core_error_analysis.md:31）。

执行规范（技能动作）：

1. 在msaicerr.py所在目录执行，先完成P0/P1的环境与输入检查（见 SKILL.md Phase 0/Phase 1，工具准备详见 msaicerr-toolkit 技能）。
2. 预检查三要素：`dfx/data-dump`下有dump文件、有算子编译产物（`.o`与`.json`）、`dfx/log/host/cann`下有日志；缺任一则工具无法提取AI Core Error信息（oam-tools/docs/zh/msaicerr/AI_Core_error_analysis.md:19）。
3. 执行后跟随终端提示`Analysis info is saved in <info.txt路径>`定位结果文件（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1358）；工具执行日志在同级的`debug_info.txt`（oam-tools/docs/zh/msaicerr/AI_Core_error_analysis.md:41）。
4. 结果目录形态：`<out>/info_{执行时间戳}/aicerror_0_{错误时间戳}/info.txt`（输出目录拼接oam-tools/src/msaicerr/msaicerr.py:127，错误子目录拼接oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1829-1830）。

**多错误只解析第一个**：若故障信息中存在多个AI Core Error问题，msaicerr按日志时间解析第一次出现的问题（oam-tools/docs/zh/msaicerr/AI_Core_error_analysis.md:39）。实现上，错误记录按`err_time`排序（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:567-569），再与dump侧线程号匹配取对应记录；tid不一致时报`Dump data tid is not the same with rts tid`并终止解析（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:575-597）。若plog中常规错误正则（`error info:`行加`AICORE_ERR_OCCUR`）无命中，工具改试outstanding双PC正则`AICORE_ERR_OCCUR_OST`，仍无命中则报`Aicore error exception does not match`终止（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:554-565；正则定义oam-tools/src/msaicerr/ms_interface/constant.py:1195-1210）。

**info提取失败分支**：Step 1拿不到算子信息（或采集上报失败）时，工具直接写一份空结论的`info.txt`到`aicerror/`目录并返回8（`MS_AICERR_INVALID_SLOG_DATA_ERROR`），不执行后续步骤（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1820-1824）。此时按P1的输入完整性检查回退（三要素排查详见 msaicerr-toolkit 技能），不要重复执行`-p`。

## 四、info.txt判读

### 4.1六段结构

`info.txt`由`analyse()`渲染，固定六段（oam-tools/src/msaicerr/ms_interface/aic_error_info.py:169-206）：①基本信息；②AI Core DFX Register；③Operator Error Line Number；④Operator Input/Output Memory；⑤Operator Dump File Parsing；⑥Execution Result of the Single-Operator Test Case。P2阶段重点判读第③段，第②段转error-type-decode.md，第⑤⑥段转reproduction-guide.md。

### 4.2第③段逐行判读

第③段由`_get_pc_str()`渲染，先给原始信息块（oam-tools/src/msaicerr/ms_interface/aic_error_info.py:290-297）：

```text
Original Info:
start pc          : 0x...
current pc        : 0x...
Error occurred most likely at line: <汇编偏移>
```

判读规则：

- `Error occurred most likely at line:`后跟的是**估算的汇编偏移**（十六进制，无`0x`前缀），由`current_pc`与`start_pc`末5位差值结合子寄存器Error PC位修正得出，不是源码行号（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1146、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1119-1147）。工具自己声明这是估算值（"most likely"，sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/decompile/decompile.md:433）。
- 偏移行之后若跟随`<反汇编文件名>:<err_pc>`与cce行号、TBE源码`文件:行号`，说明L1行号映射链走通（映射链见本文件第七节）。
- 若跟随`related instructions (error occurred before the mark *):`段，为报错指令上下文，判读见第十一节。

### 4.3 Corrected Info块

当任一修正引擎产出修正PC时，第③段追加修正信息块（oam-tools/src/msaicerr/ms_interface/aic_error_info.py:298-311）：

```text
Corrected Info:
start pc          : 0x...
current pc        : 0x...
Error occurred most likely at line: <修正后偏移>
<函数名> at <源文件>:<行>:<列>

The corrected PC is fixed by the AI Core error registers and is more
credible than the original PC. Locate the fault by the corrected info first.
```

判读规则：**优先按Corrected Info定位**——修正PC由AI Core错误寄存器回填得出，工具明确声明其比原始PC更可信（oam-tools/src/msaicerr/ms_interface/aic_error_info.py:308-311）。`Error occurred most likely at line:`后是修正后偏移的十六进制（无`0x`前缀），随后一行为llvm-symbolizer符号化结果（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1037-1040）。无修正结果时整块不出现，只有Original Info（oam-tools/test/ut/msaicerr/testcase/test_pc_corrector_ut.py:441-446）。

### 4.4 "??" 与行号缺失提示的判读

- `info.txt`中修正块出现`Unable to calculate the corrected line number, check the logs for more details.`：表示修正PC已得到，但符号化失败（未装llvm-symbolizer、`.o`不存在、执行失败或无调试信息），行号一行被替换为该提示（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1030-1036）。此时修正PC仍然有效，转手动回退（P2.4，见本文件第六节）。
- 手动或自动符号化输出中出现`??`或`??:0:0`：`??`是llvm-symbolizer的未知位置标记（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:101-102），任何一行以`??`开头即判定该偏移在`.o`中**无调试信息、符号化失败**，工具告警后放弃行号（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:266-273）。判读动作同上：转手动回退（P2.4），并确认`.o`是否为带调试信息的编译产物。

## 五、PC修正两引擎（原理简述）

**为什么要修正**：AI Core是流水线架构，异常上报时`current_pc`可能已越过真正出错指令；子寄存器中的Error PC字段是异常瞬间锁存的，更准，但只有部分位（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/decompile/decompile.md:427）。

修正由`get_corrected_pc()`统一调度，**引擎一优先、引擎二兜底**（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:322-343）。

### 5.1引擎一：adump已修正值优先

plog中若存在adump打印的`[Dump][Exception] Error PC information. coreId=..., originalStartPC=..., fixedStartPC=..., originalCurrentPC=..., fixedCurrentPC=..., fixedPCOffset=...`，直接采用其中已修正好的`fixedStartPC`/`fixedCurrentPC`/`fixedPCOffset`（grep与正则见oam-tools/src/msaicerr/ms_interface/pc_corrector.py:277-304；正则`ADUMP_FIXED_PC`定义oam-tools/src/msaicerr/ms_interface/constant.py:1213-1218）。按`core_id`匹配本核记录；无本核记录时不借用他核的修正值，跳过修正（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:288-298）。命中时输出`from_dump=True`（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:299-304；优先级验证oam-tools/test/ut/msaicerr/testcase/test_pc_corrector_ut.py:142-148）。

### 5.2引擎二：本地掩码表重算

无adump修正值时，工具打印`No adump Error PC information in plog, fix pc by error registers locally.`后本地重算（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:341-356）。原理简述：**用错误寄存器的指定位段回填PC的对应位段**。寄存器来源两种：adump的逐寄存器打印（grep`Error register information`，oam-tools/src/msaicerr/ms_interface/pc_corrector.py:307-319；正则`ADUMP_ERR_REGS`见oam-tools/src/msaicerr/ms_interface/constant.py:1224-1227），或从plog的`error info:`行合成V100寄存器组（FIXP寄存器取自`The extend info:`行，oam-tools/src/msaicerr/ms_interface/pc_corrector.py:137-154）。

掩码表按芯片分三套，均为"源寄存器名、源位段、目标PC位段"三元组形式的条目表（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:42-89）：V100表`V100_ENTRIES`按CUBE/CCU/MTE/VEC/FIXP模块分组；V200表`V200_GROUPS`与V300表`V300_GROUPS`以`*_ERROR_T0_*`触发寄存器为准（仅adump日志里有），V300复用V200寄存器布局但掩码不同。选表逻辑：芯片判为ASCEND_950用V300表，否则V200表，均未命中再回落V100模块匹配（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:219-246）。回填实现`_replace_pc_bits`：源位段与目标位段位宽相等才回填，不等则告警并保持原值（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:109-117）。芯片类型由调用方按plog dump info字段判定后传入（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:328-330）。

**如何使用其输出**：两引擎成功时返回`start_pc`/`current_pc`/`offset`三元组，渲染为info.txt第③段的Corrected Info块（见4.3）；`offset`（修正后PC与start pc的差）供llvm-symbolizer符号化（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1030-1040）。

### 5.3放弃修正的三种情形

以下情形`get_corrected_pc()`返回空，info.txt只报原始PC（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:357-368）：

1. 无任何位段被回填（修正前后PC相同）——避免报出与原始完全相同的"修正值"误导判读（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:357-362）；
2. 修正后`fixedCurrentPC < startPC`——结果不合理，跳过（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:363-368）；
3. 原始PC本身非法（解析为负）——提前跳过（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:344-348）。

判读动作：出现这三种情形时以Original Info的估算偏移走第七节反汇编链，偏移本身仍可用于定位（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/decompile/decompile.md:299）。

## 六、llvm-symbolizer自动流程与手动回退（P2.4）

### 6.1自动流程

修正PC得到后，工具立即用llvm-symbolizer把`offset`符号化为"`函数名 at 文件:行:列`"（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1022-1040）。流程四步：

1. 选`.o`文件：kernel的`.o`优先；不存在时退到按去后缀算子名glob的`<算子名>*_host.o`（mix算子名需先去掉`_mix_aic`等后缀才能命中）；都拿不到返回空串交由告警（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:999-1020）。
2. 探测工具：`shutil.which("llvm-symbolizer")`，工具名常量`SYMBOLIZER_FILE`（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:250-251；oam-tools/src/msaicerr/ms_interface/constant.py:132）。
3. 执行命令，格式为`llvm-symbolizer -obj=<o文件> <十六进制偏移>`（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:258）。
4. 解读输出：默认两行——函数名、`源文件:行:列`，拼接为"`函数名 at 源文件:行:列`"写入修正块（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:263-274；输出样例oam-tools/test/ut/msaicerr/testcase/test_pc_corrector_ut.py:391-404，如`my_kernel at /path/to/kernel.cce:88:3`）。

### 6.2失败四态与判读

| 失败态 | 工具输出 | 判读与动作 |
| --- | --- | --- |
| 未安装 | `llvm-symbolizer is not installed.`（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:251-254） | 转手动回退（P2.4），先按6.3找到工具 |
| `.o`不存在 | `The *.o file <路径> does not exist, skip symbolize.`（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:255-257） | 回P1检查编译产物收集，确认`collection/compile`下是否有对应`.o` |
| 执行失败 | `Failed to symbolize <偏移> in <o文件>.`（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:259-262） | 看debug_info.txt中具体报错，多为`.o`损坏或格式不支持 |
| 无调试信息 | 输出含`??`（如`??`、`??:0:0`），告警`the *.o may have no debug info`（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:266-273） | 符号化失败：该偏移无调试信息，转手动回退（P2.4）并用第七节反汇编链定位 |

注意：修正PC的得出与llvm-symbolizer无关，符号化失败时Corrected Info块照常输出修正PC，仅行号一行替换为`Unable to calculate the corrected line number...`提示（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1024-1029、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1034-1036；行为验证oam-tools/test/ut/msaicerr/testcase/test_pc_corrector_ut.py:407-417）。

### 6.3手动回退

工具查找路径说明：`shutil.which`只查`PATH`；工具启动时`add_objdump_to_path()`会把找到的objdump所在目录前置加入`PATH`——优先当前目录下`tools/cce-objdump[_aarch64]`，其次`which`命中项，最后猜测CANN安装目录`/usr/local/Ascend/latest/<aarch64-linux|x86_64-linux>/ccec_compiler/bin/`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1644-1700）。若llvm-symbolizer与objdump同置于该CANN工具链目录，自动流程即可命中。

手动回退步骤（技能动作）：

1. 找工具：`which llvm-symbolizer`；未命中则到CANN工具链目录查找（msaicerr对objdump的猜测路径为`/usr/local/Ascend/latest/<arch>-linux/ccec_compiler/bin/`，oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1679-1688；llvm-symbolizer的确切目录以现场CANN安装为准，此为排查建议）。
2. 取输入：从info.txt第③段Corrected Info拿`offset`（`Error occurred most likely at line:`后的十六进制值）与kernel的`.o`路径（第①段`kernel file`字段）。
3. 执行与自动流程同形的命令：

```bash
llvm-symbolizer -obj=<kernel的.o文件> 0x<修正后偏移>
```

4. 判读：输出两行为符号与`文件:行:列`；出现`??`或`??:0:0`即该`.o`无调试信息（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:101-102、oam-tools/src/msaicerr/ms_interface/pc_corrector.py:266-273），放弃符号化、改走第七节objdump反汇编链。

## 七、objdump回退链（偏移到TBE源码行）

### 7.1自动反汇编

反汇编工具二选一：`cce-objdump`优先（常量`OBJ_DUMP_FILE`），`llvm-objdump`兜底（常量`NEW_DUMP_FILE`），用`shutil.which`探测（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1042-1050；常量oam-tools/src/msaicerr/ms_interface/constant.py:130-131）。命令形态：

```bash
cce-objdump -d --line-numbers <kernel.o> > <kernel.o>.txt
```

`--line-numbers`让输出在汇编指令旁附`cce:<行号>`注释，是行号定位的基础（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1051-1052；sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/decompile/decompile.md:254）。两者都不可用时报`No cce-objdump or llvm-objdump found.`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1049）；反汇编执行失败时工具给出可操作替代方案——把`cce-objdump`与`.o`复制到其他机器执行`cce-objdump -d <kernel_name>.o > <kernel_name>.o.txt`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1092-1097；sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/decompile/decompile.md:255）。

### 7.2五步定位链

反汇编成功后，L1场景按以下链条把偏移映射到TBE源码行（总入口`_get_cce_tbe_code_number`，oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1106-1115、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1178-1213）：

1. **算差值与修正偏移**：`current_pc`与`start_pc`各取末5位（高位是加载基址，对定位无用），朴素差值为`diff_str`，结合子寄存器Error PC位（`find_extra_pc`，从出错模块寄存器如`vec error info`的[7:0]或MTE的[39:32]+[7:0]取位段，oam-tools/src/msaicerr/ms_interface/aic_error_info.py:480-524）修正为`err_pc`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:975-997、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1119-1147）。
2. **模板算子加基址**：模板类算子多`.o`合成，先grep反汇编文本中`<kernel_name>_<tiling_key>$local`符号取基址加到`err_pc`；找不到则不调整（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1055-1070、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1107-1109）。
3. **定位指令行**：逐行扫反汇编文本，命中`<err_pc>:`（偏移行）即停（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1160-1162）。
4. **前溯cce行号**：偏移行之前最近一次出现的`cce:(\d+)`注释行即该指令对应的cce行号（`--line-numbers`把`cce:<行号>`插在对应指令之前，持续记住最近值即可，oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1155-1159；sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/decompile/decompile.md:355）。
5. **映射TBE源码行**：读`<kernel_name>_loc.json`的`cce_line2loc`列表，找`cce_line`等于上一步行号且`loc[0]`非空的条目，输出`loc[0]:loc[1]`即TBE源码`文件:行号`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1170-1176）。

`info.txt`第③段逐级追加三级定位信息：反汇编`文件名:err_pc`、cce行号、TBE源码`文件:行号`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1161、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1167、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1176）。

### 7.3断链点与降级

| 断链点 | 工具行为 | 判读 |
| --- | --- | --- |
| `.o`与`.json`都不存在 | 报错返回，该步失败（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1083-1087） | 回P1检查编译产物；SK场景属正常（只有`host.o`） |
| 反汇编文本中无cce行号 | 告警`The cce code num is not exist in decompile file`，返回假（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1188-1192） | 保留偏移信息，转手动回退（7.4） |
| `_loc.json`不存在或为空 | 告警后返回真，**保留cce行号这一级结果**（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1195-1199） | L0场景通常无`_loc.json`，属正常降级（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/decompile/decompile.md:356） |
| `.cce`文件不存在 | 报错返回假（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1210-1212） | PIPE_ALL检查（第八节）无法进行，其余结果保留 |
| L0场景 | 到偏移为止，不做2-5步（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1116-1117） | 只给汇编偏移，直接走7.4手动链 |

### 7.4手动回退

技能动作，按工具同款逻辑手工执行（适用于工具侧断链或需人工核验时）：

```bash
# 1. 反汇编（与工具命令同形）
cce-objdump -d --line-numbers <kernel.o> > <kernel.o>.txt
# 2. 定位报错偏移行（<err_pc> 为 info.txt 第③段偏移，十六进制无 0x 前缀）
grep -n "<err_pc>:" <kernel.o>.txt
# 3. 从该行向上找最近的 cce:<行号> 注释
grep -n "cce:" <kernel.o>.txt
# 4. 用 cce 行号查 _loc.json 的 cce_line2loc，取 loc[0]:loc[1] 即 TBE 源码位置
grep -n "\"cce_line\": <cce行号>" <kernel_name>_loc.json
```

第4步命中后按`loc`字段读出`[源文件, 行号]`（结构见oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1174-1176）。SK场景只反汇编`host.o`且不做行号映射（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1099-1102），手动链到第2步为止。

## 八、PIPE_ALL提示（set_flag/wait_flag配对检查）

L1行号定位到cce行号后，工具顺带读`.cce`文件第`cce_code_num - 1`行（即0起数对应行），该行含`PIPE_ALL`时置提示（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1203-1209）：

```text
Please check the set_flag/wait_flag is match or not!!!.
```

判读：报错行做了全流水同步（PIPE_ALL）往往意味着`set_flag`/`wait_flag`使用不当。该提示进入结论链后，info.txt根因段给出`The set_flag and wait_flag instructions are not used together in the operator code.`（oam-tools/src/msaicerr/ms_interface/aic_error_info.py:217-218）。动作：检查算子代码中set_flag与wait_flag是否配对使用（该结论在根因链中优先级仅次于memset缺失，sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/decompile/decompile.md:460）。

## 九、SK场景处理与关融合重跑建议

SK场景下的证据链退化：寄存器解码与dump解析两条链完好，反编译链只余`host.o`反汇编（无cce/TBE行号映射），单算子复现链不可用（无device `.o`/`.json`，sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:343）。

工具侧行为（判读时知悉，不需要人工干预）：

- 算子名以SK标志打印内`kernelName=`为准，覆盖常规解析结果（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:308-313）；
- 编译文件定位改走glob`<算子名>*_host.o`，`json_file`/`cce_file`为空串（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1099-1102）；
- device单算子测试跳过，只跑`host_single_op`用例；`hash_id`无来源置空，host/device一致性哈希比对失去参照（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1762-1763；sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:93-95、sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:349）。

**技能动作（关融合重跑建议）**：SK场景定位不到具体出错算子时，向用户给出建议——关闭算子融合后重跑业务，让故障以单算子形式复现，四条证据链即可全部生效（sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:353；sd_doc/msaicerr-arch-doc/msaicerr-doc/README.md:70）。融合开关的设置方法属框架侧配置，不在本技能范围内，按用户所用框架版本查询。

## 十、多错误与dump失败分类

### 10.1多错误只解析第一个

按日志时间取第一次出现的AI Core Error（oam-tools/docs/zh/msaicerr/AI_Core_error_analysis.md:39；排序实现oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:567-569）。AI Core Error往往引发连锁故障，最早的那条才是根因现场（sd_doc/msaicerr-arch-doc/msaicerr-doc/architecture.md:121）。若修复首个错误后仍有报错，重新收集日志再走一轮P1-P5。

### 10.2 dump失败三类特征与102结论

dump结论失败时，`info.txt`第⑤段出现`Failed to get dump data of error op!`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1905-1906），且工具返回结论码102（`MS_AICERR_MEMORY_ALLOCATION_ERR`，内存分配错误方向；oam-tools/src/msaicerr/ms_interface/constant.py:73；触发点oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1972-1973），device单算子用例同时被跳过（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1765）。三类失败特征（L1走`_get_data_dump_result`，L0走`check_dump_result`，特征串相同）：

| 失败特征（grep原文） | 工具提示 | 方向 |
| --- | --- | --- |
| `exception_dumper.cc.*Dump exception.*failed` | `Data dump failed in exception dump. Please contact GE to resolve it!`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1362-1373、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1401-1412） | GE侧异常dump失败，转框架侧排查 |
| `[Dump][Exception] D2H failed` | L1：`Data dump failed. Maybe memory is invalid. Search 'D2H failed' in plog!`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1377-1395）；L0：`Data dump failed. Copy data from device to host fail!`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1413-1424） | 数据从device拷贝到host失败，常与内存无效相伴 |
| `[Exception] the address maybe invalid` | `Data dump failed. Maybe memory is invalid!`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1383-1395、oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1425-1429） | 异常地址本身无效，指向内存分配/越界方向 |

手动判定命令（技能动作）：

```bash
grep -rn "Dump exception.*failed" <收集目录>
grep -rn "D2H failed" <收集目录>
grep -rn "the address maybe invalid" <收集目录>
```

判读规则：任一特征命中即dump失败，结论码102（内存分配错误方向），P4复现无dump输入可用；此时结合P3的寄存器解码判断是否越界/地址类错误位，并按fix-suggestions.md的102条目给修复方向。ffts场景特别警惕`D2H failed`：命中后`data_dump_result`为假，单算子复现连args前置c2c地址都无从验证（sd_doc/msaicerr-arch-doc/msaicerr-doc/features/superkernel_adapt.md:312）。

## 十一、报错点10条指令上下文

L1场景下行号映射后，工具在`info.txt`第③段追加报错指令上下文段（仅L1执行，oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1110-1113）：

```text
related instructions (error occurred before the mark *):
<指令行 1>
...
*<指令行 N>

For complete instructions, please view <kernel.o>.txt
```

生成规则：用朴素差值`diff_str`（非修正偏移）在反汇编文本中定位指令行，取**该行及其之前至多9条指令（合计至多10条，不足9条时从文本头部开始）**，`*`标记报错指令行（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1322-1352；起始行计算oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1341，标记拼接oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1347-1349）。指令行形态如`504:    04c20000    ST.b64         X1, [X0], #0`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1323注释样例）。

判读规则：

- 看`*`标记行及其前2-3条指令的语义：访存指令（LD/ST类）看地址与边界，同步指令看配对，向量指令看操作数范围；
- 上下文只是定位辅助，完整反汇编以`For complete instructions`提示的`<kernel.o>.txt`为准（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1350）；
- `diff_str`定位失败时工具告警`Get fault instruction failed`且该段不出现（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1335-1339），此时直接打开反汇编文件按偏移人工查看。

---

**相关文件**：场景与收集输入判定见 SKILL.md 参考文件路由表（工具用法详见 msaicerr-toolkit、asys-toolkit 技能）；错误码与寄存器解码见`error-type-decode.md`；单算子复现见`reproduction-guide.md`；结论汇聚与修复建议见`fix-suggestions.md`。
