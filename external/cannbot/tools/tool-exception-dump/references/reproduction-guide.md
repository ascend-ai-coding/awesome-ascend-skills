# 单算子复现指南（P4阶段知识）

> **使用时机**：诊断流程Phase 4（单算子复现）开始时必读本文件。本文件覆盖复现路径选择决策、路径A（基于dump数据的标准复现）、路径B（构造数据复现）、内存越界探测机制与复现结果三态判读，是SKILL.md步骤P4.1至P4.6的知识底座。
>
> **引用约定**：本文 path:line 引用：oam-tools 相关路径相对于 CANN 社区仓 cann/oam-tools 根目录；`sd_doc/...` 为内部架构文档工作区路径（仅溯源标注，不在 oam-tools 仓内）。行号为源文件真实行号，可疑时按行号回读源码核实。

---

## 一、复现在诊断管线中的位置

单算子复现是四条证据链中唯一主动执行的一条：把故障算子从原始业务中剥离出来，用故障现场的数据（或构造的数据）单独跑一遍，观察AI Core Error能否复现（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:5-9）。能复现说明问题在算子自身，与调度、并发、上下文无关；不能复现则说明问题依赖运行环境或时序，需要往框架、并发方向排查。

`msaicerr -p`的自动分析在Step 9自动执行单算子验证（入口调用oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1922-1923，实现在:1733-1800）。本文件的知识既用于判读自动执行的结果，也用于指导手工复现。

## 二、复现路径选择决策

| 判定条件 | 选择路径 | 依据 |
| --- | --- | --- |
| dump解析产出了输入数据（info.input_list非空），或收集目录dfx/data-dump下存在exception dump文件 | 路径A：基于dump数据的标准复现 | oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1888-1901 |
| L1场景（P2场景判定已识别为L1，无exception dump数据） | 路径B：构造数据复现 | single_op变体仅在info.data_dump_result为True时执行（aicore_error_parser.py:1765）；L1场景判定见error-line-location.md |
| dump收集失败（plog含Dump exception failed、D2H failed、address maybe invalid等关键字） | 路径B，并按P2的dump失败分类处理 | oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1360-1398 |

判定细则：

1. **有效dump的判定**：自动流程Step 7解析收集目录`collection/dump`下的dump文件（aicore_error_parser.py:1884-1890）；info.data_dump_result为True时，解析产出的输入、输出、workspace文件列表填入info.input_list/output_list/workspace_list（:1897-1901），这些列表正是复现config中各文件列表键的来源。dump失败时info.dump_file记为`Failed to get dump data of error op!`（:1905-1906）。
2. **L1场景**：L1日志不含device侧exception dump数据，自动复现拿不到真实输入。注意L1场景下info.data_dump_result按plog中是否出现dump失败关键字判定（:1891-1892，_get_data_dump_result实现在:1360-1398），即使为True，dump目录中无匹配文件时解析结果仍为空（oam-tools/src/msaicerr/ms_interface/dump_data_parser.py:645-648），复现依旧缺真实输入，走路径B。
3. **人工快速核对**：拿不准时直接查看收集目录`dfx/data-dump/<device_id>/`下是否存在`exception_info.*`开头的文件（收集目录结构见 asys-toolkit 技能）。存在则优先路径A。

## 三、路径A：基于有效dump的标准复现

### 3.1 dump数据解析（msaicerr -d）

解析dump文件的命令（在msaicerr.py所在目录执行，oam-tools/docs/zh/msaicerr/Dump_files_parsing.md:9-11）：

```bash
python3 msaicerr.py -d <dump文件或目录路径> [-out <结果输出目录>]
```

参数约束（oam-tools/src/msaicerr/msaicerr.py:266-288）：`-out`仅可与`-p`或`-d`搭配；`-dtype`仅可与`-d`搭配（RequireOtherArgs机制，违反时报`must be used with`错误）。命令入口在msaicerr.py:309-310，经convert_dump_data（:151-171）进入DumpDataParser。

**输入形态三分支**（oam-tools/src/msaicerr/ms_interface/dump_data_parser.py:594-627）：

| 输入形态 | 行为 | 依据 |
| --- | --- | --- |
| `.npy`文件 | 拒绝解析，报`The dump file cannot be an npy file.` | dump_data_parser.py:600-602；UT断言见oam-tools/test/ut/msaicerr/testcase/test_msaicerr_ut.py:166-173 |
| `.bin`文件 | 走bin转npy流程，必须给`-dtype`，否则报`Need to specify the dtype when convert a bin file.` | dump_data_parser.py:603-605、:489-494 |
| 目录或无npy/bin后缀的dump文件 | 目录按node_name匹配文件名后逐个解析；单文件直接解析 | dump_data_parser.py:612-643 |

**输出命名规则**：解析产出的tensor文件命名为`{kernel_name或dump文件名}.{input|output|workspace}.{序号}[.{dtype}].{npy|bin}`，numpy可表示的dtype存`.npy`，不可表示的存`.bin`（dump_data_parser.py:382-390）；dump头中的`space`类型落盘时改名为`workspace`（:470-471）。bin转npy时输出`{原文件名去dtype段}.{目标dtype}.npy`（:520-524），例如`exception_info.2.1.20250611171538370.input.0.bin -dtype int8`产出`...input.0.int8.npy`（oam-tools/docs/zh/msaicerr/Dump_files_data_types_conversion.md:26-33）。

**input_type==7即tiling**：dump文件二进制头部中，input列表里`input_type`为7（Constant.TILING_TYPE，oam-tools/src/msaicerr/ms_interface/constant.py:1173）的条目就是tiling数据，解析时被单独取出存入tiling_data（dump_data_parser.py:769-773），随后写入`{kernel_name}_tiling.bin`（见3.3节）。

**-dtype合法值（双源口径）**：官方文档列举13种常用值（`float32`、`float16`、`float64`、`int8`、`int16`、`int32`、`int64`、`uint8`、`uint16`、`uint32`、`uint64`、`bool`、`bfloat16`，oam-tools/docs/zh/msaicerr/Dump_files_data_types_conversion.md:21）；代码实际校验集`VALID_DTYPES`为`DATA_TYPE_TO_DTYPE_MAP`全部43个值（dump_data_parser.py:45-93，13种之外另含complex64、complex128、int4、hifloat8等），传入校验集外的值报`Invalid dest_dtype`并打印全集（:496-497）。文档13种为主流用法，判读以代码校验集为准。实际可保存为npy的是numpy可表示类型：原生14种`float32`、`float16`、`float64`、`int8`、`int16`、`int32`、`int64`、`uint8`、`uint16`、`uint32`、`uint64`、`bool`、`complex64`、`complex128`（NUMPY_NATIVE_DTYPES，dump_data_parser.py:95-110），加`bfloat16`（需bfloat16ext注册，:112、:173-175）；其余28个校验集值虽过-dtype校验但numpy不可表示，bin转npy时报`Failed to convert bin to npy`（:541-552）。指定的dtype与bin文件名中的原dtype不一致时仅给Warning并按指定dtype解析（:510-515）。

**bfloat16依赖bfloat16ext**：numpy本身不认识bfloat16，`astype("bfloat16")`前必须导入第三方库bfloat16ext完成dtype注册，否则报TypeError（dump_data_parser.py:149-162）；bin转npy路径同样先检查该库，未安装时报`Can not convert to bfloat16: the bfloat16ext module is not installed.`（:526-532）。工具日志若提示`Can not read with dtype bfloat16`，需用户自行安装bfloat16ext（oam-tools/docs/zh/msaicerr/Dump_files_parsing.md:33）。

### 3.2 输入数据有效性判读（NaN/Inf与0.9倍值域，关联104）

dump解析时对每个可转numpy的tensor做两级有效性检查（`_check_tensor_data`，dump_data_parser.py:184-218），检查结果写入info.dump_info，是结论码104（算子输入数据错误，constant.py:75）的直接证据链：

1. **NaN/Inf检查**：数组中存在NaN或Inf（源码判据`np.isinf(array).any() or np.isnan(array).any()`，:191）时，报`{parse_type}[{index}] NaN/INF. Input data invalid. Please check!`（parse_type为input/output/workspace类型、index为张量序号，:192-195）。bfloat16无finfo，按float32视角检查值域（:188-189）。
2. **0.9倍值域检查**：数值型dtype（整型int16/int32/int64、无符号uint16/uint32/uint64、浮点float16/float32/float64）在`np.max(array) > 0.9 * dtype_max`或`np.min(array) < 0.9 * dtype_min`时（0.9规则在:212），报`{parse_type}[{index}] max {最大值} or min {最小值}. Input data maybe invalid. Please check!`（实际日志中{最大值}/{最小值}为具体数值，:213-217）。非数值型（如bool）不检查直接通过（:210-211）。

**到104的链路**：上述提示串中的`Input data invalid`与`Input data maybe invalid`均含子串`data invalid`，随解析结果写入aic_info.dump_info后，get_return_code判定`elif "data invalid" in aic_info.dump_info: return 104`（aicore_error_parser.py:1968-1969）。

**判读规则**：

| 检查命中 | 证据强度 | 判读 |
| --- | --- | --- |
| NaN/Inf | 强证据 | 输入数据非法，优先核对数据生成与预处理环节，关联104结论 |
| 0.9倍值域 | 弱证据（源码措辞为maybe invalid） | 数值贴近类型边界，可能是正常大值也可能是溢出前兆，须结合业务值域人工确认后再关联104 |

注意：dump文件头op属性user_tag若含子串`data invalid`会被改写为`data_invalid`，避免用户自定义标签污染104判定（dump_data_parser.py:420-422）。

### 3.3 tiling数据重建（三个来源）

tiling数据统一由`write_tiling_data_to_file`落盘为`{kernel_path}/{kernel_name}_tiling.bin`（aicore_error_parser.py:926-940），优先级从高到低三个来源：

| 优先级 | 来源 | 机制 | 依据 |
| --- | --- | --- | --- |
| 1 | dump内tiling（L0） | BigDumpDataParser解析时把input_type==7的条目存入info.tiling_data_bytes（dump_data_parser.py:569、:769-773），落盘时优先使用 | aicore_error_parser.py:929-930 |
| 2 | plog十六进制args（L0） | TilingDataParser从plog中grep `after execute:`的args十六进制串（oam-tools/src/msaicerr/ms_interface/tiling_data_parser.py:30-55），借助`exception info dump args data`地址与`para base`推算tiling指针在args中的下标（:56-127），再把下标之后的每个64位hex串去0x、补齐16位、按8字节逆序拼接后`bytes.fromhex`还原字节流（:130-146） | aicore_error_parser.py:918-921 |
| 3 | [AIC_INFO] tiling_data（L1） | 正则`\[AIC_INFO\]\stiling_data:(.*)`提取；值以`0x`开头则去0x后`bytes.fromhex`解码，失败时降级为`0000`两字节（:892-904）；否则按utf-8字符串处理（:905-909） | aicore_error_parser.py:884-910 |

三个来源都拿不到时打印`No tiling data is found in dump files and logs.`（:939-940）。复现运行时，tiling_data以`.bin`结尾则读文件为bytes，否则按utf-8编码字符串处理（oam-tools/src/msaicerr/ms_interface/single_op_test_frame/single_op_case.py:215-219），随后拷贝到HBM并拼入kernel args（oam-tools/src/msaicerr/ms_interface/single_op_test_frame/common/ascend_tbe_op.py:577-591）。

### 3.4 复现config的16个键

自动复现的config由`SingleOpCase.generate_config()`生成15键（single_op_case.py:77-95），执行前再由`_test_single_op`补充第16键`compile_temp_dir`（aicore_error_parser.py:1237-1238），合计16键。逐键说明：

| 键名 | 来源 | 构造方法与作用 |
| --- | --- | --- |
| `cce_file` | 编译产物目录 | 先取info.cce_file，不存在则回退`{kernel_path}/{kernel_name}_{tiling_key}.cce`（single_op_case.py:122-134）。自动流程中被注释掉（见3.5节），仅手工运行时用于ccec重编译 |
| `bin_path` | info.bin_file | kernel二进制`.o`路径，注册到device执行（single_op_case.py:80） |
| `json_path` | info.json_file | kernel元信息json路径，解析blockDim、kernelName、parameters、opParaSize、workspace等（ascend_tbe_op.py:98-122） |
| `tiling_data` | info.tiling_data | `{kernel_name}_tiling.bin`路径（三源重建见3.3节；single_op_case.py:82） |
| `tiling_key` | info.tiling_key | 注册stub函数名后缀，非kernel0结尾的kernel按`{stub_func_name}_{tiling_key}`注册（ascend_tbe_op.py:775-782） |
| `block_dim` | info.block_dim | launch的block数；config未给时用kernel json中的blockDim兜底（ascend_tbe_op.py:898-902） |
| `input_file_list` | dump解析产出（info.input_list） | 输入npy列表，L1路径按输入语义填充args（aicore_error_parser.py:1898；single_op_case.py:85） |
| `output_file_list` | dump解析产出（info.output_list） | 输出npy列表，运行时读各npy的size/dtype/shape用于分配输出显存（aicore_error_parser.py:1899；single_op_case.py:187-199；dtype为`\|V2`时视作float16并告警） |
| `workspace_file_list` | dump解析产出（info.workspace_list） | workspace文件列表（aicore_error_parser.py:1900；single_op_case.py:87） |
| `bin_file_list` | dump解析产出（info.bin_list） | L0路径专用：exception dump按args顺序导出的整串buffer，按序整体填回，不区分输入输出（aicore_error_parser.py:1901；single_op_case.py:88；ascend_tbe_op.py:857-865） |
| `kernel_name` | info.kernel_name | 日志检索与重编译命名的依据（single_op_case.py:89） |
| `device_id` | info.run_device_id | 目标device；非整数时告警并回退0（single_op_case.py:90、:223-228） |
| `sub_ptr_addrs` | Step 8解析二级指针产出 | 记录二级指针张量的args_list与dynamic_tensor_count，运行时重建两层指针布局（aicore_error_parser.py:1911-1914；single_op_case.py:91；ascend_tbe_op.py:717-761） |
| `ffts_addrs_num` | Step 7产出get_ffts_addrs_num | args前部需补充的c2c控制地址个数（aicore_error_parser.py:1893、:942-949；single_op_case.py:92；ascend_tbe_op.py:893-897） |
| `workspace` | get_workspace_info | kernel json未声明workspace大小时的兜底值（aicore_error_parser.py:1902-1904；single_op_case.py:93） |
| `compile_temp_dir` | Step 9生成的`temp_{时间戳}`目录 | `_test_single_op`在15键之外补充（aicore_error_parser.py:1237-1238；目录创建:1734-1735），作为dirty_ub的编译输出目录（oam-tools/src/msaicerr/ms_interface/run_dirty_ub.py:35-36），结束后删除（aicore_error_parser.py:1798-1800） |

`bin_file_list`与`input/output/workspace_file_list`互斥：前者是L0路径（按args顺序整体填入），后者是L1路径（按输入、输出、workspace语义分别构造），由exec_single_case中`if ascend_op_param.bin_list:`分支决定（ascend_tbe_op.py:857-886）。

**样例文件差异说明**：oam-tools/src/msaicerr/test_single_op.py:27-46的config样例仅展示其中12键（cce_file、bin_path、json_path、tiling_data、tiling_key、block_dim、device_id、ffts_addrs_num、input_file_list、output_file_list、kernel_name、compile_temp_dir），未包含`workspace_file_list`、`bin_file_list`、`sub_ptr_addrs`、`workspace`这4键。这4键在generate_config（single_op_case.py:77-95）与ST断言（oam-tools/test/st/msaicerr/testcase/test_single_op_case_st.py:96-112）中均存在，缺省时按空列表/空字典/0处理，手工构造config时可按需省略。

### 3.5 复现用例的生成与运行规范

自动流程中每个变体生成独立用例文件并以子进程执行（`_test_single_op`，aicore_error_parser.py:1233-1299）：

1. **生成用例文件**：`__generate_case`把config序列化后写出`test_{op_test}.py`（如test_single_op.py、test_host_single_op.py），内容为4行：import SingleOpCase、config字典、OP_TEST赋值、`SingleOpCase.run(config, OP_TEST)`（aicore_error_parser.py:1216-1230）。
2. **注释cce_file键**：`comment_cce_in_case`把用例中的`"cce_file"`改写为注释（:1301-1307），使run_kernel取不到cce_file从而跳过ccec重编译，直接使用现场`.o`。这样做是因为复现必须用现场那份二进制，重编译可能因编译器版本差异产出不同机器码（设计取向说明见sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:100-102）。
3. **子进程环境变量**（aicore_error_parser.py:1245-1251）：
   - `ASCEND_SLOG_PRINT_TO_STDOUT=0`：必须为0，置1时日志走stdout不落盘，日志检索会失效（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:90）；
   - `ASCEND_PROCESS_LOG_PATH={case_path}/{op_test}_{时间戳}`：单算子自身的plog与故障现场plog物理隔离（:1241-1242）；
   - `PYTHONPATH`追加msaicerr根目录（即msaicerr.py所在目录：aicore_error_parser.py位于其ms_interface子目录下，取该子目录的上一级加入，:1244-1251）。
4. **执行**：`subprocess.run([sys.executable, case_file], ...)`以绝对路径python运行（:1254-1261）。
5. **SingleOpCase.run内部顺序**：先探测soc_version（DSMI，失败回退cce文件正则，两者都失败则中止并报错，single_op_case.py:246-265），再执行run_dirty_ub（:273-274），最后run_kernel（:275-277）。
6. **run_kernel装配**：读tiling为bytes、np.load各输入npy、读输出npy的shape/dtype/size，构造AscendOpKernel与运行参数后执行（single_op_case.py:203-244）。

**手动复现命令**：复现失败或未运行时，工具打印提示（`print_single_op_result`，aicore_error_parser.py:1310-1319）：

```bash
export PYTHONPATH=<msaicerr根目录>:$PYTHONPATH;cd <msaicerr根目录>;python3 <用例文件路径>
```

手动运行未改写的用例文件时`cce_file`键有效，run_kernel会走`update_kernel_by_cce`用ccec重编译一份`{kernel_name}_new.o`（single_op_case.py:203-213；重编译实现:137-171：从cce文件头`//`注释行提取ccec命令，ccec不在PATH时按`/usr/local/Ascend/latest/{aarch64-linux|x86_64-linux}/ccec_compiler/bin/ccec`猜测路径）。判读手动复现结果时优先相信未重编译（自动流程）的结果。

### 3.6 三变体执行顺序与判读

Step 9按固定顺序最多执行三个变体（run_single_operator，aicore_error_parser.py:1733-1800）。执行前提：kernel文件存在（SK场景仅要求bin_file，非SK要求bin_file与json_file，:1737-1747），否则跳过并打印`Skip exec single op case because the kernel file dose not exist.`（:1744-1747）。

| 变体（op_test） | 使用的kernel | 目的 | 输入 | 判读要点 |
| --- | --- | --- | --- | --- |
| `host_single_op` | `{kernel_name}*_host.o`（_find_sk_host_o按glob匹配，:106-115、:1750-1752） | 校验host侧加载的kernel与device上执行的是否同一份 | 复用info，仅bin_file换成host.o（deepcopy的host_info，:1749-1752） | check_hash_id比对plog的hash_id与单算子日志中`hash=(\d+)`（:1702-1719，grep串`Aicore kernel execute failed\|AI Core kernel execution failed`）；不一致时报error`the kernel load on the host is different from the device`（:1756-1760），指向框架加载错误 |
| `single_op` | info.bin_file（device侧`.o`） | 标准复现：用dump出的真实输入重跑故障算子 | dump解析产出的input/output/workspace/bin列表（仅info.data_dump_result为True时执行，:1763-1773） | 结果记入info.single_op_test_result/single_op_mem_monitor/single_op_log_path（:1767-1773）；三态判读见第六节 |
| `error_single_op` | 同single_op | 反向验证：标准复现未复现时，故意制造必错场景，验证检测链路本身有效 | 同single_op，但执行时`knl_args[-1] = 0`末参置零（oam-tools/src/msaicerr/ms_interface/single_op_test_frame/common/ascend_tbe_op.py:904-905） | 仅在single_op结果为SUCCESS（未复现）时执行（:1774-1780）；若连人为置零地址都不报aicore error，说明plog落地或关键字匹配链路有问题，而非算子没错（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:50-52） |

SK场景没有device侧`.o`，跳过single_op与error_single_op（:1762-1763）。三个变体共用同一个compile_temp_dir（:1735），全部结束后删除（:1798-1800）。

## 四、路径B：无dump数据的构造复现

### 4.1 最小复现原则：shape与dtype对齐

路径B的目标是最小复现而非全量还原：优先对齐shape、dtype、block_dim、tiling_key等执行要素，输入数据只需覆盖触发故障的最小集合。能复现即可锁定算子自身问题；不必追求与业务完全一致的数据分布。

### 4.2 数据构造与用例编写

1. **推断执行要素**：从kernel元信息json读取blockDim、kernelName、parameters、opParaSize（tiling大小）、workspace（ascend_tbe_op.py:106-122）；shape与dtype可从算子定义、业务代码或plog中的shape信息推断。
2. **构造输入输出**：用numpy按推断的shape/dtype生成输入npy（如`np.zeros`、`np.full`、`np.random`），输出npy只需形状与dtype正确（运行时仅读其size/dtype/shape分配显存，single_op_case.py:187-199）。tiling_data无dump与plog来源时可传空串或手工构造的字符串（非`.bin`后缀按utf-8编码，single_op_case.py:215-219）。
3. **编写用例**：参照oam-tools/src/msaicerr/test_single_op.py:27-48的模板填config（12键起步，按需补3.4节其余键），执行`SingleOpCase.run(config, OP_TEST)`，OP_TEST取`single_op`。
4. **运行命令**：与3.5节手动复现命令相同（PYTHONPATH指向msaicerr根目录后python3执行用例文件）。
5. **可选仿真模式**：AscendOpKernelRunner支持simulator_mode（pv/ca/tm三种形态，simulator_lib_path通常为Toolkit安装目录下的tools/simulator，ascend_tbe_op.py:373-393；rts_api.py:66-71、:82-108）。msaicerr主流程默认真机，构造复现拿不到真机时可评估仿真形态，行为差异须另行确认。

判读路径B结果时同样使用第六节的三态规则。

### 4.3 脏UB预写（run_dirty_ub）

无论路径A还是路径B，`SingleOpCase.run`在每次run_kernel之前都无条件先执行一次脏UB预写（single_op_case.py:273-274）：用一个专门kernel把整个Unified Buffer填满极大浮点值`1.7976931348623157e+30`（oam-tools/src/msaicerr/ms_interface/run_dirty_ub.py:44-46）。目的：UB不会在kernel间自动清零，依赖UB初值为0的隐性缺陷在干净环境可能跑对，预写脏后必然出错，把偶发问题变成确定问题（sd_doc/msaicerr-arch-doc/msaicerr-doc/features/dirty_ub.md:5-10）。

机制与边界（run_dirty_ub.py:26-74）：

- 分派：先按soc_version匹配已注册的芯片Handler，未命中回退TIK实现run_dirty_ub_tik（:69-74）。
- TIK实现：`cce.get_soc_spec("UB_SIZE")`取UB大小，`vec_dup`循环填充，kernel名为DirtyCustom（constant.py:26），BuildCCE编译后运行（:26-66）。
- 失败不阻断：tbe导入失败、编译产物缺失等均告警后返回，不影响后续kernel执行（:31-34、:51-53；特性说明sd_doc/msaicerr-arch-doc/msaicerr-doc/features/dirty_ub.md:120-122）。预写脏是否生效只能从子进程日志的告警判断。
- 该值非NaN非Inf但远超正常中间结果：参与运算立刻溢出为Inf/NaN，比随机脏值更易观察（features/dirty_ub.md:95-101）。

设备侧显存同样不保证清零：运行框架的malloc在分配后立即用`secrets.token_bytes`随机字节填充（oam-tools/src/msaicerr/ms_interface/single_op_test_frame/runtime/rts_api.py:637-638），构造复现不应假设任何设备内存初值为0。

## 五、内存越界探测：tail与magic双模式

`AscendOpKernelRunner.run`对同一用例连跑两遍，分别用两种显存布局策略（ascend_tbe_op.py:919-932）：先tail模式、后magic模式。

### 5.1 tail模式：页对齐抓向后越界

malloc按页大小（PageMemorySize=0x200000，即2MB）向上对齐后，把张量数据顶到页尾（:240-251）。任何向后越界访问都会踩到下一页，若该页未映射，硬件直接抛访存异常，sync_ret非0。粒度是页级（2MB）但确定性强（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:147-167）。

### 5.2 magic模式：0x55哨兵

常量定义（ascend_tbe_op.py:63-67）：`MagicMemorySize = 0x80`（前后各128字节）、`MagicData = 0x55`、`ForwardDestroy = 1`（向前越界）、`BackwardDestroy = 2`（向后越界）。

填充逻辑（:252-271）：按32字节对齐张量大小，前后各加128字节哨兵区后malloc；先把整块（含数据区）全部写0x55（`np.ones(adjust_size, dtype=np.int8) * 0x55`）并拷贝到device，再把数据指针前移0x80。执行结束后`_check_magic_memory`逐张量检查两端哨兵（:825-838）：head哨兵（origin_pointer处）被改写返回ForwardDestroy=1，tail哨兵（magic_pointer处）被改写返回BackwardDestroy=2，均未被改写返回0（比对逻辑:815-823）。

两遍结果合并（:942-953）：launch_ret或sync_ret非0时按error输出launch/sync/magic三个返回值；都为0时输出`exec single op case success.`，magic_ret非0再追加memory status check结果，即kernel跑通但内存被越界改写的组合会被单独标出。

| 模式 | 检测机制 | 向前越界 | 向后越界 | 粒度 |
| --- | --- | --- | --- | --- |
| tail | MMU页保护 | 否 | 是（跨页时） | 2MB页 |
| magic | 0x55哨兵字节比对 | 是（ForwardDestroy） | 是（BackwardDestroy） | 128字节 |

（对照表依据sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:199-209；magic模式只能抓写越界，读越界不改变哨兵，且越界距离超过128字节可能跳过哨兵区。）

### 5.3 越界判读要点

- **`Access Memory without OverBoundary`关键字**：`_test_single_op`从子进程stdout中`SingleOpCase.run Execute Info`标记之后检索该串，命中则single_op_mem_monitor记录`Out-of-bounds memory access.`（aicore_error_parser.py:1263-1267）。注意：按架构文档注记，这两个关键字在当前msaicerr代码内没有产出方，mem_monitor字段实际常为空串，属于待接通的钩子（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:302-310）；运行时侧输出的该关键字仍可作越界佐证。
- **越界证据的可靠来源**：magic_ret（1=向前越界、2=向后越界）与tail遍的sync_ret，以`memory status check result`、`execute result`形式出现在用例输出中（ascend_tbe_op.py:942-951）。
- 判读顺序：先看三态（第六节）确定是否复现，再看magic_ret与sync_ret判断是否伴随越界，最后结合P3寄存器解码的错误位（如越界类错误位）交叉印证。

## 六、复现结果三态与下一步动作

**复现成功的判定标准**：`search_aicerr_log`在`{op_test}_{时间戳}`日志目录下遍历`.log`文件（aicore_error_parser.py:1548-1560），先等日志落盘稳定（:1537-1545），再按`_check_file_content`判定：日志内容命中以下五种错误串之一，**且同一日志文件内同时出现目标kernel名**（:1524-1535，两条件缺一不可；kernel名先剥`_mix_aic`/`_mix_aiv`后缀，:1549）：

1. `there is an aivec error exception`
2. `there is an aicore error exception`
3. `there is an exception of aivec error`
4. `there is an exception of aicore error`
5. `aicore exception`

注意`RetCode`语义反转：它描述的是单算子用例本身跑没跑成功，FAILED才是复现成功（constant.py:29-32定义SUCCESS=0/FAILED=1/NOT_RUN=2；语义说明见sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:264-279）。

| 三态 | 判定条件 | 工具输出 | 下一步动作 |
| --- | --- | --- | --- |
| 复现成功（RetCode.FAILED=1） | search_aicerr_log命中（五种错误串+kernel名共现） | `Successfully reproduced the AI Core exception by running the single-operator test case.`（aicore_error_parser.py:1782-1785） | 问题锁定在算子自身：结合P2报错行定位与P3寄存器解码收敛根因方向，进入P5修复建议；返回码层面走101（MS_AICERR_SINGLE_OP_ERR，get_return_code优先级链:1975-1976，前提是无更优先证据命中） |
| 未复现（RetCode.SUCCESS=0） | 用例正常跑完且日志未命中错误串 | `Failed to reproduce the AI Core exception by running the single-operator test case.`（:1794-1797） | 此时error_single_op已执行（末参置零反向验证）；问题大概率依赖运行环境或时序，走P5复现失败升级路径（msSanitizer、op_debug_config=oom、关融合重跑、框架侧上报），不臆断根因 |
| 未运行（RetCode.NOT_RUN=2） | 两种触发：①子进程退出码非0且日志无错误串（:1271-1279，提示`exited with code ... and no aicore error was found`）；②stdout含`Execute single op case failed`（NOT_RUN_EXEC_FAILED，constant.py:37）或`exec single op case failed`（NOT_RUN_LAUNCH_FAILED，constant.py:38）（:1284-1297） | `The single-operator test case was not executed successfully. Whether the AI Core exception can be reproduced is unknown.`（:1786-1793） | 检查用例文件、依赖（numpy/tbe/ccec/驱动）、日志路径与权限后重试；NOT_RUN同样计入非SUCCESS，返回码走101（:1974-1976，用例没跑起来不能视为没有问题） |

结果字段去向：single_op变体的三态与日志路径记入info.single_op_test_result/single_op_mem_monitor/single_op_log_path（:1767-1773）；host_single_op与error_single_op仅打日志不进info（:1752-1760、:1774-1780）。ST用例以`single op case success`子串断言运行成功（oam-tools/test/st/msaicerr/testcase/test_single_op_case_st.py:224，源码输出`exec single op case success.`，ascend_tbe_op.py:949）。

## 七、注意事项与常见误区

1. **不要通过修改ms_interface源码来复现**：复现框架（single_op_test_frame）是诊断工具的一部分，改源码复现会污染证据链；数据或环境不满足时走路径B构造或升级路径。
2. **kernel名共现防误判**：仅凭日志含aicore error关键字不能认定复现，必须同一日志文件内出现目标kernel名（:1532-1534），避免把其他算子的错误算到目标算子头上。
3. **日志等待无超时上限**：_wait_for_log_stabilization以0.2秒间隔轮询文件大小直到两次相同（:1537-1545），plog异步落盘时属正常等待；若有其他进程持续写同一目录理论上会一直等。
4. **路径B不做全量还原承诺**：构造数据复现成功可锁定算子自身问题；构造复现未复现不能反推业务场景无问题（数据分布、并发、时序差异都可能影响）。
5. **环境依赖自查**：单算子复现依赖NPU设备与驱动、tbe/te（脏UB）、CANN toolkit、DSMI、numpy，各依赖缺失时的行为见环境依赖表（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/single_op/single_op.md:416-425）；复现前先确认`-e`环境自检通过（环境自检详见 msaicerr-toolkit 技能）。
6. **复现成功≠根因确认**：复现只证明问题在算子自身可复现，根因方向仍需P2（报错行）、P3（错误位）、dump校验（104证据链）共同收敛，最终按P5四证据链框架出结论。
