# 错误类型识别（P3知识）

本文件是 tool-exception-dump 技能的P3阶段知识，覆盖芯片形态判定、错误位解读、六模块子域分析、五类特殊场景、AI Core数量与blockDim检查、结论码101-107全表。对应`msaicerr -p`内部解析管线的寄存器解码环节，其产出即分析报告第2段"AI Core DFX Register"（`oam-tools/src/msaicerr/ms_interface/aic_error_info.py:186-188`，段名样例见`oam-tools/test/st/msaicerr/testcase/test_aicore_error_parse_st.py:448`）。

路径约定：本文 path:line 引用：oam-tools 相关路径（如`oam-tools/src/...`）相对于 CANN 社区仓 cann/oam-tools 根目录；`sd_doc/...` 为内部架构文档工作区路径（仅溯源标注，不在 oam-tools 仓内）。文中简写的`constant.py`、`aic_error_info.py`、`aicore_error_parser.py`、`utils.py`均指`oam-tools/src/msaicerr/ms_interface/`下的同名文件。术语约定：plog指收集目录下`collection/plog`内的进程运行日志；UB（Unified Buffer，统一缓冲区）、ECC（Error Correcting Code，错误校验）为硬件术语。

前置依赖：已按`error-line-location.md`完成场景判定与报错行定位；本文件结论供P4单算子复现与P5证据链汇聚使用。

## 芯片形态判定（两法）

Stars与David两种芯片形态打印错误码的方言不同，解码前必须先判型。判定有两法，方法二更可靠，两法结论冲突时以方法二为准。

### 方法一：按错误码形态判定

实现于`detect_chip_type`（`aic_error_info.py:25-40`）。判定方向固定为：0x形态指向Stars，逗号分隔的十进制指向David，不得写反。

| 错误码形态 | 判定结果 | 后续解码路径 |
| --- | --- | --- |
| 以0x开头的十六进制（如`0x40000`） | Stars（ASCEND_910B） | 位扫描：十六进制值展开为置位位号，查Stars字典 |
| 裸`0` | Stars（兜底） | 两种芯片打印0同形，只能落到Stars路径，无位置位时走trap_or_timeout兜底 |
| 十进制位号（单个或逗号分隔列表，如`257`或`257, 341`） | David（ASCEND_950） | 位号即绝对位号，直接查David字典 |
| 其他形态 | Stars（兜底） | 按Stars路径处理 |

机制背景（`constant.py:41-49`的ChipType说明与`aic_error_info.py:28-31`）：910B是Stars架构（errorMapInfo_），错误码按%#PRIx64打印原始aicError[0]寄存器值，非零值必带0x前缀；950是David架构（g_davidErrorMapInfo），打印runtime已解析好的十进制位号逗号列表。主正则的error_code捕获组同时容许两种形态（`constant.py:1199`）：

```
(?P<error_code>0x[0-9a-fA-F]+|\d+(?:,\s*\d+)*)
```

裸`0`特例：950以裸`0`作无错误标记，但910B打印0时同样不带前缀，两芯片同形，`detect_chip_type`只能将其落到910B路径，由既有的trap_or_timeout兜底逻辑处理（`aic_error_info.py:31-33`）。

### 方法二：按dump info寄存器键判定

实现于`detect_chip_type_by_dump_info`（`aic_error_info.py:43-58`），依据是两种芯片dump info行打印的寄存器集互不重叠（`constant.py:1167-1170`）。

| dump info行包含的键 | 判定结果 |
| --- | --- |
| `sc error info:`或`su error info:`或`l1 error info:` | David（950） |
| `ifu error info:`或`ccu error info:`或`biu error info:` | Stars（910B） |
| 两者都没有 | 判不出，返回None，由调用方决定回退口径 |

该方法比方法一可靠：错误码为裸`0`时方法一只能落到Stars，而寄存器键不会同形。

判型时序硬约束（`aicore_error_parser.py:491-496`）：判型必须在`_get_extra_info`之前执行。`_get_extra_info`只抽取Stars的六个寄存器字段，David独有的sc/su/l1 error info会被丢弃，之后再判型则无法判定。判出的chip_type同时传给PC修正引擎（`aicore_error_parser.py:538-540`），PC修正细节见`error-line-location.md`。

### 两套错误码的解码差异

Stars解码（`aic_error_info.py:437-477`）：

- `parse_stars_error_bits`把十六进制错误码转为整数后按二进制展开置位位号（`utils.hexstr_to_list_bin`，`oam-tools/src/msaicerr/ms_interface/utils.py:403-414`）；空串、不可解析或值为0时返回空列表（`aic_error_info.py:61-70`）。
- 位号列表为空时输出兜底文案`NO_ERROR_BIT_INFO`，即`trap_or_timeout, timeout or trap error`（`constant.py:143-146`，`aic_error_info.py:441-443`）。注意这不是位0：位0是`biu_l2_read_oob`（`constant.py:156`）。看到trap_or_timeout不能直接读成"确认超时"，寄存器证据链给不出方向时应依赖其余证据链（反编译、dump、单算子复现）。
- 每个置位位号查Stars字典`AIC_ERROR_INFO_DICT`（`constant.py:155-369`，共176项，键为位0-175），得到"错误名, 英文释义"格式的串。
- 错误名首个下划线前的前缀（`name.split('_')[0]`转小写）识别功能单元，分派到六模块子域分析（见下节）。

David解码（`aic_error_info.py:419-435`）：

- runtime已完成位扫描与六个错误寄存器的掩码过滤，error_code就是绝对位号列表，且保留runtime打印顺序（`aic_error_info.py:73-84`）。
- 位号查David字典`AIC_ERROR_INFO_DICT_DAVID`（`constant.py:378-1123`，共187条）。
- 位号在字典中查不到时输出`unknown error bit <位号>`（runtime同样跳过这些位，`aic_error_info.py:427-431`）。
- 没有六模块子寄存器解析：950日志的extra_info中没有`*_ERR_INFO`寄存器（`aic_error_info.py:485-488`）。但David错误名自带模块前缀（CUBE_、MTE_、L1_、SC_、SU_、VEC_），可直接读出模块。
- 位号按模块分段（`constant.py:362-369`）：CUBE段从0起、MTE段从64起、L1段从128起（L1扩展段从160起）、SC段从192起、SU段从256起、VEC段从320起（VEC扩展段从352起）。读位号时先按区段定位模块，再查字典，例如位341落在VEC段（320-351），位275落在SU段（256-319）。

版本差异提示：`sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/register_error_dict.md`基于旧版源码快照，其中"位0为trap_or_timeout兜底"的描述与现行代码不一致（现行代码位0是`biu_l2_read_oob`，无位置位的兜底是独立常量`NO_ERROR_BIT_INFO`）。两处冲突时以`constant.py`现行为准。

## 高频错误位表

本节收录诊断中最常命中的错误位，共30条数据行（Stars 15条、David 15条）。选位规则（确定性，扩展本表时必须沿用）：

1. 优先选取名称含obound、oob、wrap、atomic、atm、ub、overflow、ovflw（越界、绕回、原子、溢出语义）的条目。
2. 覆盖8类根因方向，每类至少1位：越界读、越界写、UB越界绕回、原子溢出、输入数据非法、memset缺失、参数改写、硬件与内存分配异常。
3. Stars与David各至少8位。
4. 每行标注constant.py出处行号，错误名与释义以`constant.py`为唯一名称源。

"常见根因方向"列的标注口径：未标注者为代码或文档依据（方向来自constant.py条目释义或sd_doc架构文档）；标注"经验方向"者为诊断经验归纳的症状位映射，不是字典明示的因果。memset缺失与参数改写两类在错误位字典中没有直接对应位：memset缺失映射到atomic检查的候选错误码`0x800000`（即Stars位23置位，见场景1），参数改写映射到"指令被改写"类释义位（Stars位77、David位11）。内存分配失败（结论码102方向）同样无错误位对应，其判定走dump落盘结果，见`reproduction-guide.md`。

| 适用芯片 | 位号 | 名称 | 含义 | 常见根因方向 | constant.py出处 |
| --- | --- | --- | --- | --- | --- |
| Stars（910B） | 0 | `biu_l2_read_oob` | 总线读访问错误，建议检查L2代码 | 越界读（总线与L2侧） | `constant.py:156` |
| Stars（910B） | 1 | `biu_l2_write_oob` | 总线写访问错误，建议检查L2代码 | 越界写（总线与L2侧） | `constant.py:157` |
| Stars（910B） | 8 | `ccu_ub_ecc` | CCU读写UB时发生多位ECC错误，转RAS告警处理 | 硬件异常（ECC） | `constant.py:166` |
| Stars（910B） | 9 | `cube_invld_input` | L0A与L0B读回的数据为INF或NAN | 输入数据非法（INF/NAN） | `constant.py:167` |
| Stars（910B） | 22 | `mte_bas_raddr_obound` | mte load3d指令基地址越界 | 越界读（load3d基地址） | `constant.py:181` |
| Stars（910B） | 23 | `mte_biu_rdwr_resp` | MTE访问无效GM地址或跨device访存超时 | atomic溢出检查的候选错误码0x800000即此位置位；memset缺失场景伴随位（经验方向，见场景1） | `constant.py:182` |
| Stars（910B） | 32 | `mte_gdma_read_overflow` | MTE指令读片上buffer的地址越界 | 越界读（片上buffer） | `constant.py:195` |
| Stars（910B） | 33 | `mte_gdma_write_overflow` | MTE指令写片上buffer的地址越界 | 越界写（片上buffer） | `constant.py:196` |
| Stars（910B） | 48 | `mte_write_overflow` | mte load2d指令写地址超过目的最大地址 | 越界写（load2d目的地址） | `constant.py:213` |
| Stars（910B） | 62 | `vec_ub_wrap_around` | VEC指令读写UB的地址越界（绕回） | UB越界绕回 | `constant.py:228` |
| Stars（910B） | 77 | `vec_instr_undef` | VEC指令异常：参数违反指令约束、二进制版本不匹配或指令被改写 | 参数改写类场景的症状位（经验方向） | `constant.py:245` |
| Stars（910B） | 124 | `mte_ub_wr_ovflw` | MTE写UB的地址越界 | UB越界（MTE写UB） | `constant.py:299` |
| Stars（910B） | 125 | `mte_ub_rd_ovflw` | MTE读UB的地址越界 | UB越界（MTE读UB） | `constant.py:300` |
| Stars（910B） | 139 | `ccu_lsu_atomic_err` | scalar原子指令访问被scalar修改但未写回的GM | 原子操作异常 | `constant.py:318` |
| Stars（910B） | 156 | `vec_idata_inf_nan` | 指令运算的输入数据为INF/NAN | 输入数据非法（INF/NAN） | `constant.py:337` |
| David（950） | 4 | `CUBE_INVLD_INPUT` | L0A与L0B读回的数据为INF或NAN | 输入数据非法（INF/NAN） | `constant.py:391` |
| David（950） | 5 | `CUBE_L0A_WRAP_AROUND` | CUBE操作L0A的地址越界 | 越界绕回（L0A） | `constant.py:395` |
| David（950） | 11 | `CUBE_ILLEGAL_INSTR` | CUBE指令异常：参数违反约束、二进制版本不匹配或指令被改写 | 参数改写类场景的症状位（经验方向） | `constant.py:419` |
| David（950） | 95 | `MTE_BIU_RDWR_RESP` | MTE指令访问无效GM地址或跨device访存超时 | 无效GM访问症状位；memset缺失类场景候选（经验方向，见场景1） | `constant.py:592` |
| David（950） | 73 | `MTE_READ_OVERFLOW` | MTE 2D指令读L1的地址越界 | 越界读（2D读L1） | `constant.py:517` |
| David（950） | 74 | `MTE_WRITE_OVERFLOW` | MTE 2D指令写L1/L0A/L0B的地址越界 | 越界写（2D写） | `constant.py:521` |
| David（950） | 79 | `MTE_ATM_ADD_ADDR_MISALIGN` | MTE原子指令地址未对齐 | 原子操作异常（地址未对齐） | `constant.py:534` |
| David（950） | 81 | `MTE_GDMA_READ_OVERFLOW` | MTE2读UB、MTE3读L1/UB的地址越界 | 越界读（UB与L1） | `constant.py:542` |
| David（950） | 82 | `MTE_GDMA_WRITE_OVERFLOW` | MTE2写UB、MTE3写L1/UB的地址越界 | 越界写（UB与L1） | `constant.py:546` |
| David（950） | 93 | `MTE_UB_ECC` | MTE读UB时发生多位ECC错误，转RAS告警处理 | 硬件异常（ECC） | `constant.py:584` |
| David（950） | 153 | `L1_BAS_RADDR_OBOUND` | LOAD3D指令指定的初始地址超出L1 3D size范围 | 越界读（LOAD3D初始地址） | `constant.py:690` |
| David（950） | 174 | `L1_UB_WR_OVFLW` | MTE从L1搬运到UB的地址越界 | UB越界（L1到UB搬运） | `constant.py:766` |
| David（950） | 275 | `SU_CCU_LSU_ATOMIC_ERR_T0` | scalar原子指令访问被scalar修改但未写回的GM | 原子操作异常 | `constant.py:887` |
| David（950） | 336 | `VEC_ERR_IDATA_INF_NAN_T0` | 指令运算的输入数据为INF/NAN | 输入数据非法（INF/NAN） | `constant.py:1006` |
| David（950） | 341 | `VEC_UB_WRAP_AROUND` | VEC访问UB的地址越界 | UB越界绕回 | `constant.py:1026` |

完整字典位置指引：

- Stars全量字典（176位）：`oam-tools/src/msaicerr/ms_interface/constant.py:155-369`（`AIC_ERROR_INFO_DICT`，位0-175；360-369行为David位段偏移常量`DAVID_OFFSET_*`）。
- David全量字典（187条）：`constant.py:378-1123`（`AIC_ERROR_INFO_DICT_DAVID`）。
- 字典解读补充（错误性质分类、子字典引用关系、"Lite Only"位说明）：`sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/register_error_dict.md`。注意该文档基于旧版源码快照，位0描述与现行代码不一致，以`constant.py`为准。
- 本表未收录的错误位一律回`constant.py`查原文释义，禁止凭记忆补写含义。

## 六模块子域分析

Stars错误码展开为置位位号后，报告第2段对每个命中模块输出子寄存器解读。机制是两级翻译（`aic_error_info.py:437-477`）：

- 第一级：位号查Stars字典得"错误名, 英文释义"，错误名前缀识别功能单元。
- 第二级：每个单元首次命中时，从extra_info中用`单元键=(\S+)`提取该单元子寄存器串，按该单元位段解析出错误类型与出错地址；同一单元去重（子寄存器只解析一次），但每个位的错误名与释义逐位输出。
- 子寄存器串取不到时输出`No <单元>_ERR_INFO found`，不报错（plog可能只记录部分单元）。

extra_info由`_get_extra_info`用六个正则从plog原文提取并重排（`aicore_error_parser.py:367-387`），六个键定义于`constant.py:1160-1165`：

- `IFU_ERR_INFO`取自plog原文`ifu error info:`（正则`ifu error info:\s(\S+),`）。
- `CCU_ERR_INFO`取自`ccu error info:`，`BIU_ERR_INFO`取自`biu error info:`。
- `CUBE_ERR_INFO`取自`cube error info:`，`MTE_ERR_INFO`取自`mte error info:`，`VEC_ERR_INFO`取自`vec error info:`（正则同形）。

六个模块的位段定义（`bit[high:low]`为闭区间；`aic_error_info.py:526-661`，位段汇总另见`sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/register_parse/register_parse.md:230-239`）：

| 模块 | 错误类型位段 | 类型释义字典 | 地址位段 | 地址语义 | 补零近似位数 | 额外位段 |
| --- | --- | --- | --- | --- | --- | --- |
| IFU | `bit[50:48]` | SOC_ERR_INFO_DICT | `bit[47:2]` | IFU Error Address[47:2] | 2 | 无 |
| MTE | `bit[26:24]` | 按错误位动态选择（见下表） | `bit[22:8]` | MTE Error Address[19:5] | 5 | 无 |
| CCU | 无 | 无 | `bit[22:8]` | CCU Error Address[17:3] | 3 | 无 |
| BIU | 无 | 无 | `bit[24:0]` | 完整25位地址 | 0（直接转十六进制） | 无 |
| CUBE | 无 | 无 | `bit[16:8]` | CUBE Error Address[17:9] | 9 | 无 |
| VEC | 无 | 无 | `bit[28:16]` | VEC Error Address[17:5] | 5 | `bit[15:8]`重复计数，转十进制 |

MTE是唯一按错误位动态选子字典的模块（`aic_error_info.py:562-573`）：

| 错误位err_bit | 使用子字典 | 语义 |
| --- | --- | --- |
| 46（`mte_unzip`） | UNZIP_ERR_INFO_DICT | 解压错误 |
| 34（`mte_comp`） | FMC_ERR_INFO_DICT | FMC错误 |
| 25（`mte_decomp`） | FMD_ERR_INFO_DICT | FMD错误 |
| 23（`mte_biu_rdwr_resp`） | SOC_ERR_INFO_DICT | SoC级读写错误 |
| 21（`mte_aipp_illegal_param`） | AIPP_ERR_INFO_DICT | AIPP错误 |
| 其他 | 空字典 | 类型码释义显示NA |

五张子字典内容（`constant.py:1124-1158`）：

- SOC_ERR_INFO_DICT（7项）：`000`读脏数据（read poison）；`001`读越界（read oob，L2作buffer模式时读地址超过配置的L2虚拟地址size）；`010`读总线错误（read bus error，含安全与非安全地址交叉访问、收不到response、atomic运算异常）；`011`读译码错误（read decode error，读地址不在各模块地址空间内，即越界）；`101`写越界（write oob）；`110`写总线错误（write bus error）；`111`写译码错误（write decode error）。`100`未定义，属硬件保留编码（`sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/register_error_dict.md:136`）。
- FMC_ERR_INFO_DICT（2项）：`000`为`fmc_read_over_turn_err`，`001`为`fmc_blk_num_zero_err`。
- FMD_ERR_INFO_DICT（5项）：`000`到`100`依次为`fmd_write_over_turn_err`、`fmd_blk_num_zero_err`、`fmd_blk_num_noequal_err`、`fmd_header_err`、`fmd_decompress_err`。
- UNZIP_ERR_INFO_DICT（5项）：`000`到`100`依次为`uzp_write_over_turn_err`、`uzp_blk_num_zero_err`、`uzp_index_noenough_err`、`uzp_index_err`、`uzp_decompress_err`。
- AIPP_ERR_INFO_DICT（3项）：`000`访问外部存储绕回（`aipp_mte_ex_round`）；`001`访问L1 buffer绕回（`aipp_mte_l1_round`）；`010`配置AIPP SPR相关fp16为INF或NAN（`aipp_mte_inerr`）。

解读要点：

- `approximate`是补零近似值，不是精确地址。寄存器只存地址高位（如MTE存[19:5]，低5位被硬件丢弃），输出补零后的十六进制仅供定位参考（`aic_error_info.py:581-590`）。
- Stars位23（`mte_biu_rdwr_resp`）的`mte_err_addr`记录的是搬运相反方向那一端的地址，搬运方向为输出到L1时，读错误记录的是L1地址，直接照寄存器值查会找错方向（`sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/register_error_dict.md:103`，文档依据）。
- VEC的`vec_err_rcnt`（`bit[15:8]`）转十进制输出`repeats`，表示错误重复次数，可区分偶发与持续错误（`aic_error_info.py:652-660`）。
- IFU与MTE的错误类型码会写入`ifu_err_type`与`mte_err_type`字段供后续判断使用（`aic_error_info.py:534-535`、`559-560`）。
- fixp、sc、cnt前缀的错误位没有对应的子寄存器键与解析分支（分派仅覆盖六个单元，`aic_error_info.py:450-473`），只能靠主字典释义定位。
- David错误码不走子域分析（见芯片形态判定节），其模块信息直接读错误名前缀。

`find_extra_pc`（`aic_error_info.py:480-524`）从子寄存器中取Error PC的[9:2]位片，供反编译模块修正出错指令地址：遇CCU类错误直接返回空串（CCU寄存器无Error PC信息）；MTE取`bit[39:32]`与`bit[7:0]`拼接共16位，其余单元取`bit[7:0]`共8位；David（950）无`*_ERR_INFO`寄存器，恒返回空串。该位片如何参与PC修正见`error-line-location.md`。

## 特殊场景判读

五类特殊场景在错误码与日志形态上有专门判定逻辑，解读时优先套用。

### 场景1：atomic溢出（0x800000、dha status 1与memset检查）

触发链条（`aicore_error_parser.py:1872-1882`，解析Step 6）：

1. `_need_atomic_clean`判定算子是否需要框架清零：算子json含compileInfo或compile_info（动态shape算子）且parameters非空即需要（`aicore_error_parser.py:1438-1459`）。
2. `_check_atomic_clean`对需要清零的算子，在收集目录grep`AtomicLaunchKernelWithFlag_<node_name>`；找不到即atomic_clean_check为False，表示图中该算子前未插入memset或atomic_clean（`aicore_error_parser.py:1461-1479`）。
3. 仅当错误码恰为`0x800000`且atomic_clean_check为False时，`_get_atomic_err_log`在收集目录grep`dha status 1`，命中即atomic_add_err为True（`aicore_error_parser.py:795-805`，门控条件见1879-1881）。

判读要点：

- `0x800000`即Stars位23置位（`mte_biu_rdwr_resp`，`constant.py:182`）；工具将该码视为atomic溢出候选，只有此码才触发dha status检查。
- 结论文案为`Atomic add has a precision overflow. Check the operator precision. Note that if tasks are concurrently executed on the NPU, a false warning may be reported.`（`aic_error_info.py:227-231`），并发执行时该结论可能误报，需结合场景4。
- 优先级与互斥：`get_return_code`先判105（memset缺失）再判107（atomic溢出），而atomic_add_err的求值前提本身包含atomic_clean_check为False，两条件在elif链上互斥，107在当前实现下不会成为最终返回码（`aicore_error_parser.py:1963-1980`；`sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/error_code.md:99-106`专门记录了这一点）。三证据（0x800000、dha status 1、memset缺失）并存时结论按105优先，atomic溢出作为伴随解释。

### 场景2：args前后改写

- 提取：解析Step 4以`[AIC_INFO] args before execute`与`[AIC_INFO] args.*after execute`为关键字分别提取执行前与执行后的args（`aicore_error_parser.py:1849-1857`、`1515-1521`）。
- 比对：`_check_args`将执行后首个参数与执行前参数列表逐一比对，命中任意一个即通过，全部不命中即check_args_result为False；args_before全为0时跳过检查（GE可能未打印，`aicore_error_parser.py:93-103`）。
- 报告呈现：第4段逐行输出args before execution与args after execution（`aic_error_info.py:195-196`）。
- 结论：args前后不一致即算子参数在执行期间被改写（被踩），结论为`If the arguments are inconsistent before and after operator execution, memory access may be out of bounds. You are advised to use the memory error detection model to locate the fault.`（`aic_error_info.py:242-247`），对应结论码106（`constant.py:77`）。常见根因是本算子或并发算子越界写踩到args区域，下一步走P4复现或内存错误检测模式（升级路径见`fix-suggestions.md`）。

### 场景3：outstanding双PC

- plog主正则AICORE_ERR_OCCUR无命中时，`get_op_info`降级使用outstanding变体AICORE_ERR_OCCUR_OST（`aicore_error_parser.py:554-565`），该变体把start_pc改由`first pc start:`捕获，并额外捕获`second pc start:`（s_start_pc，`constant.py:1203-1210`）。
- 一条异常日志同时记录两个PC起点。`update_dumpinfo_for_outstanding`汇总并发信息后，若判定错误属于第二个任务（is_second_error为True），`get_op_info`把start_pc替换为s_start_pc（`aicore_error_parser.py:583-588`、`2126-2149`）。
- 判读要点：双PC场景选错起点会导致报错行定位整体偏移；报告第3段的PC若来自outstanding记录，应确认使用的是与出错任务对应的那个PC。报错行定位流程见`error-line-location.md`。

### 场景4：并发执行isconcurrentexe

- `get_is_concurrentexe_value`以`isconcurrentexe`为关键字在plog中检索，正则提取first与second两组task id、stream id及isconcurrentexe值（`aicore_error_parser.py:2020-2076`）。
- 去重后任务数超过2（不同的task id与stream id组合多于2组）时放弃自动判定，提示需要人工分析（`aicore_error_parser.py:2049-2059`）。
- is_concurrentexe为1且`is_scalar_register_err`成立（该线程错误码落在[256, 320)区间，即SU寄存器错误段，正是David字典SU段256-319，`constant.py:367`）时，认定错误属于第二个任务：dump取second侧，并置is_second_error供场景3替换PC（`aicore_error_parser.py:1999-2017`、`2132-2138`）。
- 判读要点：并发场景下atomic精度溢出结论可能误报（见场景1结论文案）；错误码落在SU段且日志含isconcurrentexe时，优先核对任务归属再下结论。

### 场景5：错误码0x0的v300三元组重读

- `set_info`在error_code为`0`或`0x0`时调用`_get_v300_error_code`重读扩展错误码（`aicore_error_parser.py:535-536`）。
- `_get_v300_error_code`在collection/plog下grep`The extend info: errcode:`，正则提取三元组（`code0`、`code1`、`code2`），拼接为一个大整数：code0占低64位，code1左移64位，code2位重组后左移128位，返回十六进制串（`aicore_error_parser.py:389-403`）。
- 报告第2段AIC_ERROR行优先显示与error_code匹配的三元组原文error_code_all（`aicore_error_parser.py:483-490`；`aic_error_info.py:187`），如`(0x200000000, 0, 0)`（断言样例见`oam-tools/test/st/msaicerr/testcase/test_aicore_error_parse_st.py:446`）。
- 判读要点：错误码为0不代表无异常，可能是寄存器未直接给出置位值，需从extend info三元组重建；三元组也重建不出时错误码仍为`0x0`，无位置位，按trap_or_timeout兜底处理（`constant.py:143-146`）。

## AI Core数量与blockDim检查

- driver_aicore_num：`collect_driver_aicore_number`定位runtime动态库后以ctypes调用`rtGetAiCoreCount`获取（`aicore_error_parser.py:405-420`）；获取失败报工具状态码10（`constant.py:69`）。
- rts_block_dim：在plog中检索`RUNTIME`行，按`blockDim=(\d+)`提取并取最大值（`aicore_error_parser.py:599-611`）。
- 判定：两者均有效（非-1）且rts_block_dim大于driver_aicore_num的2倍时，结论为`The number of AI Cores in the environment is less than that required by the operator.`，即环境AI Core数量不满足算子并发需求（`aic_error_info.py:219-224`）。
- 报告Basic information段直接展示rts_block_dim与driver_aicore_num（`aic_error_info.py:183-184`），核对时直接对照这两个值。

## 结论码101-107全表

结论码是msaicerr的分析结论，与工具状态码（0、1-11）分工不同：返回非0不一定是工具出错，101-107表示工具正常完成并给出了明确的故障结论（`sd_doc/msaicerr-arch-doc/msaicerr-doc/appendix/error_code.md:7-14`）。全表如下（常量定义与语义见`constant.py:72-78`）：

| 结论码 | 常量名 | 语义 | 触发判定（代码依据） |
| --- | --- | --- | --- |
| 101 | MS_AICERR_SINGLE_OP_ERR | 检查到单算子运行错误 | single_op_test_result不为SUCCESS（FAILED与NOT_RUN均命中；`aicore_error_parser.py:1974-1976`） |
| 102 | MS_AICERR_MEMORY_ALLOCATION_ERR | 检查到内存分配错误 | data_dump_result为False，即dump落盘失败（`aicore_error_parser.py:1972-1973`） |
| 103 | MS_AICERR_HARDWARE_ERR | 检查到硬件错误 | env_available为False，即标杆算子环境自检失败（`aicore_error_parser.py:1977-1978`） |
| 104 | MS_AICERR_OPERATOR_INPUT_DATA_ERR | 检查到算子输入数据错误 | dump_info含`data invalid`（`aicore_error_parser.py:1968-1969`） |
| 105 | MS_AICERR_FRAMEWORK_MEMSET_MISSING | 检查到框架未执行memset清零 | atomic_clean_check为False（见场景1） |
| 106 | MS_AICERR_OPERATOR_ARGS_OVERWRITTEN | 检查到算子args被踩 | check_args_result为False（见场景2） |
| 107 | MS_AICERR_ATOMIC_OPERATOR_OVERFLOW | 检查到atomic算子遇到溢出数据导致错误 | atomic_add_err为True（见场景1；与105互斥，实际不作为返回码返回） |

优先级链：`get_return_code`按105→107→104→106→102→101→103→0逐级判定，先命中先返回（`aicore_error_parser.py:1963-1980`；优先级链测试见`oam-tools/test/st/msaicerr/testcase/test_aicore_error_parse_st.py:464-509`）。完整判定链解读与修复建议映射归P5，见`fix-suggestions.md`。

## 与其他参考文件的关系

- 场景判定与报错行定位（PC修正、llvm-symbolizer、objdump回退）：`error-line-location.md`。
- dump解析与单算子复现（104输入数据证据链、INF/NAN判读）：`reproduction-guide.md`。
- 证据链汇聚、优先级链与修复建议映射：`fix-suggestions.md`。
- 工具用法、环境约束与收集流程：msaicerr-toolkit 与 asys-toolkit 技能（分工见 SKILL.md 参考文件路由表）。
