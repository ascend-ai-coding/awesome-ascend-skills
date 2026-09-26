---
name: external-cannbot-tools-tool-exception-dump
description: '昇腾 NPU AI Core Error 系统化诊断工作流：环境与输入检查、故障信息收集、场景判定与报错行定位、错误类型识别、单算子复现、证据链汇聚与修复建议，基于
  msaicerr/asys 官方工具管线执行与判读，输出六段式证据链诊断报告。

  Use when（满足任一强信号即触发）：当用户提到 aicore error、aivec error、AI Core Error、AI Core 异常、核错误、算子报错；越界、野指针、内存踩踏、地址非法；exception
  dump 解析、plog、故障日志收集、asys collect、msaicerr -p/-d/-e；报错行定位、错误码定位、寄存器解码、结论码 101-107；单算子复现、最小复现、根因分析；Kernel
  报错调试。即使只是询问某报错是否为 AI Core Error 也应使用本技能。

  Do NOT use：仅询问 msaicerr/asys 工具用法与参数细节（转 msaicerr-toolkit、asys-toolkit 技能）；GE/图编译层问题（转
  graph/ 域技能）；算子精度不达标（转 ascendc-precision-debug）；AI CPU Error、HCCL 等其他错误类型（不在覆盖范围，说明后指路）。'
original-name: tool-exception-dump
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-26'
synced-commit: edcd4b94390c2a6b664f0935fbdd8a65e3644e9f
license: UNKNOWN
---

# tool-exception-dump：AI Core Error 系统化诊断工作流

> **执行纪律**：本技能运行于故障发生环境本地；按 Phase 0→Phase 5 顺序推进 34 个编号步骤，每步四要素齐全（动作、命令、预期输出与判读、失败分支）；不可跳步，失败分支指明的回退动作优先；穷尽细节按第二节路由表读取 references/ 文件或上游 toolkit 技能后再操作；全程只读分析，不修改任何产品代码。

## 一、身份与边界

本技能是 AI Core Error 诊断工作流，面向昇腾 NPU 上发生的 AI Core Error（含 aivec error 同链路判读），提供从环境检查、故障信息收集、场景判定与报错行定位、错误类型识别、单算子复现到证据链汇聚与修复建议的完整诊断管线。管线与 msaicerr 工具内置流程逐步对齐：工具自动完成的部分判读其产出，工具缺位的部分（手动回退、构造复现、证据汇聚）由本技能补位；全部行为性依据带 path:line 引用或路由至 references。

- **运行位置**：故障发生环境本地。msaicerr 仅支持部署工具的环境与日志所在环境为同一环境，不支持异地分析（oam-tools/docs/zh/msaicerr/msaicerr_functions_and_restrictions.md:5）。
- **工具底座**：msaicerr 与 asys（CANN 软件包内置）。msaicerr 承载三种运行模式：`-p` 完整分析、`-d` dump 解析、`-e` 环境自检；asys 是配套的故障信息收集工具。
- **术语约定**：plog 指业务进程运行日志（位于收集目录 dfx/log/host/cann 内）；dump 指异常发生时的数据落盘文件；SK 指 SuperKernel（多算子融合编译形态）。
- **覆盖范围**：仅 AI Core Error；AI CPU Error、HCCL 等其他错误类型不在本技能范围内。
- **指路规则**：仅询问 msaicerr/asys 工具用法、CLI 参数、环境准备、三要素排查等纯工具细节时转 `msaicerr-toolkit`、`asys-toolkit` 技能（工具用法真源，本技能不复制工具手册）；GE/图编译层问题（图编译报错、GE 进程崩溃等）转 graph/ 域技能；算子精度问题（rtol/atol 不达标、数值偏差）转 `ascendc-precision-debug`；不确定时标注不确定、不得编造，查证不了的写"待确认（需查 <文件>）"。
- **操作边界**：只读分析加新增结果目录；不做硬件维修判断，只给方向与升级路径；所有结论必须带证据链（见第五节 NEVER 清单）。

## 二、参考文件路由表

穷尽细节不在本文件展开；执行到对应阶段或步骤时，先读下表指向的文件再操作。工具用法层知识（CLI 参数、环境准备、三要素排查、asys collect 用法、多错误取第一等）一律路由到上游 toolkit 技能，本技能不复制：

| 阶段/步骤 | 参考文件 | 覆盖内容 |
| --- | --- | --- |
| P0/P1/P2/P5 退出码与降级 | [toolchain-orchestration.md](references/toolchain-orchestration.md) | 上游 toolkit 分工路由表、msaicerr 退出码全表（0/1-11/101-107）、get_return_code 优先级链、收集降级标记规则 |
| P2 场景判定与报错行定位 | [error-line-location.md](references/error-line-location.md) | 四场景关键词表、info.txt 判读、PC 修正两引擎、llvm-symbolizer 与 objdump 回退链、多错误与 dump 失败分类 |
| P3 错误类型识别 | [error-type-decode.md](references/error-type-decode.md) | 芯片形态判定两法、高频错误位表、六模块子域分析、五类特殊场景、结论码 101-107 全表 |
| P4 单算子复现 | [reproduction-guide.md](references/reproduction-guide.md) | 复现路径决策、路径 A/B、tiling 三源重建、16 键 config、三变体、越界探测双模式、三态判读 |
| P5 证据链汇聚与修复建议 | [fix-suggestions.md](references/fix-suggestions.md) | 四证据链框架、优先级链逐级触发条件、修复建议映射表、升级路径、证据链格式报告模板 |
| 评测自查 | [eval-cases.md](references/eval-cases.md) | 14 个 RED-GREEN 用例，GREEN 断言覆盖全部 34 步 |

上游 toolkit 技能分工（知识依赖单向性，修改只在真源处更新）：msaicerr CLI 参数与命令速查、环境准备、`-e` 环境自检、收集目录三要素排查、8 个暂不支持算子清单、dump 解析与 `-dtype` 转换 → `msaicerr-toolkit`；asys collect/analyze 用法与参数、asys_output 目录树全景、自动收集的 6 个环境变量一致性清单 → `asys-toolkit`。

## 三、诊断剧本（Phase 0-5，34 个编号步骤）

步骤编号规则：P<阶段>.<序号>，P0×7、P1×5、P2×6、P3×5、P4×6、P5×5，共 34 步。命令中的 `<占位符>` 为需替换项，命令均在 msaicerr.py 所在目录执行；关键命令内联呈现，判读的穷尽细节路由至 references/ 与上游 toolkit 技能。

### Phase 0：环境与输入检查（P0.1-P0.7）

#### P0.1 确认运行位置与本地分析前提

**动作**：确认本技能运行在故障发生环境本地（工具与日志必须同一环境），且运行形态受支持。
**命令**：

```bash
python3 --version    # 需不低于 3.7.5
```

**判读**：`Python 3.7.5` 及以上即满足依赖；同时确认三条硬约束：仅本地分析（不支持异地）、不支持在 Ascend RC 形态下使用、python 依赖 3.7.5 及以上（oam-tools/docs/zh/msaicerr/msaicerr_functions_and_restrictions.md:5-7）。
**失败分支**：环境为异地、RC 形态或 python 版本过低时停止分析并如实告知用户，不得进入 P0.2。

#### P0.2 设置 CANN 环境变量并进入工具目录

**动作**：source 环境变量脚本，进入 msaicerr.py 所在目录，校验 ASCEND_OPP_PATH 已设置。
**命令**：

```bash
source <CANN安装路径>/set_env.sh      # <CANN安装路径>替换为实际路径，root 用户默认 /usr/local/Ascend/cann
cd <CANN安装路径>/tools/msaicerr     # 必须在 msaicerr.py 所在目录执行
echo "$ASCEND_OPP_PATH"             # 必须输出非空
```

**判读**：echo 输出非空路径即通过；ASCEND_OPP_PATH 缺失时工具在解析参数前即报 `Environment variable not set after the CANN software is installed.` 并返回退出码 2（oam-tools/src/msaicerr/msaicerr.py:290-293）。环境准备三步详见 msaicerr-toolkit 技能。
**失败分支**：输出为空则重新 source 或核查 CANN 安装（环境准备详见 msaicerr-toolkit 技能）；仍为空则停止，不得继续 P0.3。

#### P0.3 环境自检（golden op 标杆算子）

**动作**：运行内置算子样例自检软硬件环境，同时为后续 103 结论的 env_available 判定建立基线。
**命令**：

```bash
python3 msaicerr.py -e -dev <device_id>    # <device_id> 默认 0
```

**判读**：正常输出 5 行 `[INFO]`，末行为 `The built-in sample operator runs successfully, The environment is normal.` 且退出码 0（oam-tools/src/msaicerr/msaicerr.py:192-194）；自检失败时退出码 103 并提示 `See the detailed error logs above`（oam-tools/src/msaicerr/msaicerr.py:196-199）。
**失败分支**：自检失败按上方日志排查芯片不兼容、驱动问题、依赖缺失；该失败同时构成 103（硬件错误）方向的证据，记录后转 P5.1 汇聚，环境恢复前不执行 P2.2 的 -p 分析。

#### P0.4 清点输入四要素并按需求助

**动作**：清点故障日志、收集目录、Device ID、算子名四项输入；缺项时逐项询问用户。
**命令**：

```bash
ls <用户提供的路径>    # 对用户给出的每个路径做存在性核验
```

**判读**：形成输入清单：收集目录（或 tar.gz 包）、故障日志位置、`<device_id>`、`<算子名>`（Device ID 默认 0，oam-tools/src/msaicerr/msaicerr.py:281-282）；四要素齐备或降级口径明确后方可进入 P1。
**失败分支**：关键输入缺失时向用户逐项询问（收集目录路径、故障日志位置、Device ID、报错算子名，已获得的项明确告知来源），得到补充前不启动分析；plog 始终无法获得时按停止条件终止。

#### P0.5 拦截 8 个暂不支持算子

**动作**：若已知算子名，先与 8 个暂不支持分析的算子清单比对拦截。
**命令**：

```bash
echo "<算子名>" | grep -iE "MatmulAllReduce|AllGatherMatmul|MatmulReduceScatter|GroupedMatmulAllReduce|^MemSet$|NonMaxSuppressionBucketize"
```

**判读**：无输出即不命中，可继续；有输出即命中暂不支持算子。完整 8 项清单：MatmulAllReduce 类算子、MatmulAllReduceAddRmsNorm、MatmulAllReduceInplaceAddRmsNorm、AllGatherMatmul、MatmulReduceScatter、GroupedMatmulAllReduce、MemSet、NonMaxSuppressionBucketize（oam-tools/docs/zh/msaicerr/msaicerr_functions_and_restrictions.md:8-16，清单与约束详见 msaicerr-toolkit 技能）。
**失败分支**：命中即停止分析并如实告知用户该算子暂不支持，不得强行继续；否则转 P0.6。

#### P0.6 预检执行位置与可写性约束

**动作**：检查当前目录与 debug_info.txt 可写、规划结果目录，确认不在 -p 目录及其子目录内执行、-out 与 -p 不嵌套。
**命令**：

```bash
pwd                                            # 不得位于 <收集目录> 或其子目录内
touch debug_info.txt && rm -f debug_info.txt  # 当前目录可写且已存在的 debug_info.txt 可写
mkdir -p <结果目录>                            # <结果目录> 不得为 <收集目录> 或其子目录
```

**判读**：三条均成功即通过；工具侧对应禁令文案为 `Do not run msaicerr in the directory specified by -p or its subdirectory. Make sure -out specifies a different directory (including its subdirectory) from -p.`，违反时退出码 2（oam-tools/src/msaicerr/msaicerr.py:120-123）；当前目录或 debug_info.txt 不可写报 `The current directory or debug_info.txt is immutable, Please check.` 退出码 2（oam-tools/src/msaicerr/msaicerr.py:301-307）。
**失败分支**：预检不过则更换执行目录或结果目录后重试；仍无法满足时停止（硬约束详见 msaicerr-toolkit 技能），不得带病进入 P1。

#### P0.7 汇总 Phase 0 结论并决定入口

**动作**：汇总环境与输入检查结果，写入诊断报告"环境与输入"段，并确定 P1 的入口分支。
**命令**：

```bash
ls debug_info.txt    # msaicerr 每次执行后生成的过程日志，作为 Phase 0 工具执行痕迹核验
```

**判读**：列出 debug_info.txt 即 P0.3 自检执行痕迹在案（oam-tools/docs/zh/msaicerr/environment_check.md:34）；环境自检通过、无不支持算子、位置约束满足、输入形态明确（已有目录、tar.gz、无收集三者居其一）即允许进入 P1.1；降级项（如缺算子名）逐一列入报告。
**失败分支**：任一硬约束失败已在其所属步骤处置，此处仅汇总；环境自检失败（P0.3）时与用户确认是否仅输出 103 方向结论后结束，否则转 P1.1。

### Phase 1：收集准备（P1.1-P1.5）

阶段说明：本阶段是收集准备（输入形态判定、三要素校验、asys collect、tar.gz 处理、降级标记）。注意与工具内部行为区分：`msaicerr -p` 执行时内部自动完成的 6 步收集（拷贝日志→定位 dump 文件→提取算子名→按算子名取编译文件→收集编译产物→收集 GE 图，oam-tools/src/msaicerr/ms_interface/collection.py:383-429）发生在 P2.2 执行 -p 的时刻，不是本阶段步骤，两者不得混淆。

#### P1.1 输入形态判定（三分支）

**动作**：判定手头输入属于"已有收集目录、tar.gz 压缩包、无任何收集"哪种形态，选择对应分支。
**命令**：

```bash
ls -d <输入路径>         # 存在性核验
ls -d <输入路径>/dfx     # 含 dfx 子目录即为收集目录形态
```

**判读**：输入为 asys_output_* 目录或同构目录（含 dfx）走分支 A（转 P1.3）；输入为 `*.tar.gz` 走分支 B（转 P1.2）；只有报错现象、无目录无压缩包走分支 C（转 P1.4）。
**失败分支**：路径不存在或形态不明时向用户确认，不得猜测路径继续；确认后重新判定。

#### P1.2 tar.gz 解压处理（分支 B）

**动作**：将压缩包手动解压到独立目录，并核验解压结果恰好包含一个子目录。
**命令**：

```bash
mkdir -p <解压目录> && tar -xzf <压缩包路径> -C <解压目录>
ls <解压目录>    # 必须恰好一个 asys_output_* 子目录
```

**判读**：ls 恰好列出一个 asys_output_* 子目录即成功，以该子目录为收集目录转 P1.3；工具源码中虽有 tar 自动解压分支（oam-tools/src/msaicerr/msaicerr.py:130-134），但其依赖的 `--tar_file` 参数未在 main() 中注册，当前版本该分支不可达，因此必须手动解压（sd_doc/msaicerr-arch-doc/msaicerr-doc/modules/entry/entry.md:213）；多于一个子目录时工具侧 get_select_dir 会抛 ValueError（oam-tools/src/msaicerr/msaicerr.py:84-89）。
**失败分支**：解压失败检查压缩包完整性与磁盘空间；目录形态不符时向用户确认哪个是有效收集目录，否则不进入 P1.3。

#### P1.3 收集目录三要素校验（分支 A）

**动作**：对收集目录逐项校验 dump 文件、算子编译信息、日志文件三要素。
**命令**：

```bash
ls <收集目录>/dfx/data-dump/        # dump 文件（L0 场景 exception dump）
ls <收集目录>/dfx/ops/              # 算子编译信息（*.o 与 *.json）
ls <收集目录>/dfx/log/host/cann/    # host 侧 cann 日志（含 plog）
```

**判读**：三个目录均非空即三要素齐备（oam-tools/docs/zh/msaicerr/AI_Core_error_analysis.md:19）；任一为空转 P1.5 降级标记。三要素排查口径与 asys_output 目录树全景详见 msaicerr-toolkit 与 asys-toolkit 技能。
**失败分支**：缺 plog（dfx/log/host/cann 无日志）时停止分析、要求用户重新收集（转 P1.5 与停止条件）；缺 dump 或缺 .o/.json 带降级标记继续，转 P1.5。

#### P1.4 执行故障信息收集（分支 C）

**动作**：无任何收集时执行 asys collect 收集故障信息，或用 asys analyze 一键入口。
**命令**：

```bash
asys collect --task_dir=<编译缓存目录> --tar="True" --output=<输出前缀>    # 入口 1：先收集，产出目录后走分支 A
asys analyze -r=aicore_error    # 入口 2：不带 --path，自动收集后内部直接调 msaicerr -p
```

**判读**：入口 1 产出 `{output}/asys_output_时间戳` 目录（不带 output 时存放在命令执行目录）；执行 asys 前必须核对 6 个环境变量（ASCEND_PROCESS_LOG_PATH、NPU_COLLECT_PATH、DUMP_GRAPH_PATH、ASCEND_WORK_PATH、ASCEND_CACHE_PATH、ASCEND_CUSTOM_OPP_PATH）与业务运行时一致（oam-tools/docs/zh/asys/fault_information_collection.md:17）；为保证解析数据准确性，复现前建议先清理日志（oam-tools/docs/zh/asys/AI_Core_error_analysis.md:9）。参数细节与目录树详见 asys-toolkit 技能。
**失败分支**：环境变量不一致时先警示用户并要求确认，不得静默继续；收集失败检查 task_dir 与 output 参数合法性、写权限，否则转 P1.5 记录降级。

#### P1.5 降级标记

**动作**：按三要素缺失情况与 SK 场景打降级标记，写入报告"场景与定位"段。
**命令**：

```bash
ls <收集目录>/dfx/ops/*.o <收集目录>/dfx/ops/*.json 2>/dev/null    # 编译产物核对
grep -rln "Begin to dump callback exception" <收集目录>            # SK 场景同样触发定位降级
```

**判读**：缺 dump 标记"缺 dump"，P4 改走构造数据路径且不得基于无 dump 证据下 104 结论；缺 .o/.json 标记"定位降级"，报错行定位置信度降低；SK 场景只生成 host.o，同样定位降级；工具侧收集失败或日志无法提取 AI Core Error 信息时写出空结论报告并返回退出码 8（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1819-1824），见到退出码 8 即按"输入信息不可用"处理。标记规则全表见 references/toolchain-orchestration.md「收集降级标记规则」。
**失败分支**：无 plog 即停止（停止条件见第四节），不进入 P2；否则携带降级标记转 P2.1。

### Phase 2：场景判定与报错行定位（P2.1-P2.6）

#### P2.1 场景判定（L0/L1/ffts/SK）

**动作**：对收集目录执行三条 grep，判定解析级别与场景标志，并记录对后续阶段的影响。
**命令**：

```bash
grep -rn "\[AIC_INFO\] dev_func:" <收集目录>           # 命中=L1，未命中=L0
grep -rn "fftsplus task execute failed" <收集目录>     # 命中=ffts
grep -rn "Begin to dump callback exception" <收集目录>  # 命中=SK
```

**判读**：三个标志相互独立、可同时为真，判定只依赖日志文案命中与否（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:331-365）；关键影响：L1 无有效 dump 时 P4 转构造数据路径，SK 只产出 host.o、单算子复现链不可用（定位不清时建议关融合重跑），ffts 场景警惕 `D2H failed`。关键词表与影响明细见 references/error-line-location.md「二、场景判定」与「2.3 各场景对后续阶段的影响」。
**失败分支**：三条均无输出且收集目录中无 AI Core Error 关键日志时，回 P1.3 复核三要素（可能被分析日志不是报错时日志），确认后仍无则停止；否则记录场景结论转 P2.2。

#### P2.2 执行 msaicerr -p 完整分析

**动作**：在 msaicerr.py 所在目录执行完整根因分析，跟随终端提示定位 info.txt。
**命令**：

```bash
python3 msaicerr.py -p <收集目录> -out <结果目录> -dev <device_id>
```

**判读**：成功时终端打印 `Analysis info is saved in <info.txt路径>`（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1354-1358），结果目录形态为 `<out>/info_{执行时间戳}/aicerror_0_{错误时间戳}/info.txt`；退出码 0 或 101-107 为正常完成（101-107 即分析结论码），1-11 为工具状态码（全表见 references/toolchain-orchestration.md）。本步执行时工具内部自动完成 6 步收集（oam-tools/src/msaicerr/ms_interface/collection.py:383-429）与 11 步解析（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1802-1960），解析产出对应后续各判读点：错误行提取与 PC 修正→P2、寄存器解码→P3、dump 解析与单算子复现→P4、结论与报告→P5。
**失败分支**：退出码 8（空报告）表示收集失败或日志无法提取信息，回 P1.3 复核三要素与日志时间窗，不要盲目重复执行 -p；退出码 2 回 P0.6 检查路径约束；退出码 1 检查参数依赖规则（详见 msaicerr-toolkit 技能）；否则转 P2.3 判读 info.txt。

#### P2.3 判读 info.txt 第③段（报错行定位）

**动作**：读取 info.txt，按六段结构定位第③段 Operator Error Line Number，优先按 Corrected Info 判读。
**命令**：

```bash
cat <info.txt路径>
```

**判读**：第③段先给 Original Info（`start pc`、`current pc`、`Error occurred most likely at line: <汇编偏移>`，偏移为十六进制无 0x 前缀，是估算值而非源码行号）；出现 Corrected Info 块时优先按其定位（修正 PC 由 AI Core 错误寄存器回填，比原始 PC 更可信）；偏移行之后跟随 `<反汇编文件名>:<err_pc>`、cce 行号、TBE 源码 `文件:行号` 三级信息（L1 才有后两级）；修正块出现 `Unable to calculate the corrected line number, check the logs for more details.` 表示符号化失败。六段结构与逐行判读见 references/error-line-location.md「四、info.txt 判读」与「4.3 Corrected Info 块」。
**失败分支**：无 Corrected Info 块时以 Original Info 估算偏移转 P2.5 反汇编链；行号缺失提示出现则转 P2.4 手动回退。

#### P2.4 llvm-symbolizer 手动回退

**动作**：自动符号化失败（未安装、.o 不存在、执行失败、无调试信息）时，手动定位 llvm-symbolizer 并符号化修正后偏移。
**命令**：

```bash
which llvm-symbolizer    # 未命中则到 CANN 工具链目录查找，如 /usr/local/Ascend/latest/<arch>-linux/ccec_compiler/bin/
llvm-symbolizer -obj=<kernel的.o文件> 0x<修正后偏移>
```

**判读**：正常输出两行，即函数名与 `源文件:行:列`（如 `my_kernel at /path/to/kernel.cce:88:3`）；输出 `??` 或 `??:0:0` 即该 .o 无调试信息、符号化失败（oam-tools/src/msaicerr/ms_interface/pc_corrector.py:266-273）；未安装态工具文案为 `llvm-symbolizer is not installed.`（pc_corrector.py:251-254）。输入取值：偏移来自 info.txt 第③段 Corrected Info，.o 路径来自第①段 kernel file 字段。失败四态与手动回退全流程见 references/error-line-location.md「六、llvm-symbolizer 自动流程与手动回退（P2.4）」。
**失败分支**：手动仍输出 `??` 则放弃符号化，转 P2.5 objdump 反汇编链；.o 文件不存在则回 P1 检查编译产物收集（dfx/ops），补齐后重试本步。

#### P2.5 objdump 反汇编回退链

**动作**：反汇编 kernel 的 .o 文件，按五步定位链把偏移映射到 cce/TBE 源码行，并判读 PIPE_ALL 配对提示。
**命令**：

```bash
cce-objdump -d --line-numbers <kernel.o> > <kernel.o>.txt     # 反汇编（llvm-objdump 为备选）
grep -n "<err_pc>:" <kernel.o>.txt                            # <err_pc> 为第③段偏移（十六进制无 0x 前缀）
grep -n "cce:" <kernel.o>.txt                                 # 从报错行向上找最近 cce:<行号>
grep -n "\"cce_line\": <cce行号>" <kernel_name>_loc.json      # 映射 TBE 源码 文件:行号
```

**判读**：五步链走通得到 TBE 源码 `文件:行号`；反汇编文本报错行含 PIPE_ALL 时工具给出 set_flag/wait_flag 配对检查提示（提示原文见 references/error-line-location.md「八、PIPE_ALL 提示」），动作是检查算子代码中 set_flag 与 wait_flag 是否配对使用；断链点（.o 与 .json 都不存在、无 cce 行号、_loc.json 缺失、L0 到偏移为止）均为正常降级，保留已有级结果。五步链与断链点全表见 references/error-line-location.md「七、objdump 回退链」。
**失败分支**：cce-objdump 与 llvm-objdump 均不可用（`No cce-objdump or llvm-objdump found.`）时，按工具建议把 cce-objdump 与 .o 复制到其他机器执行 `cce-objdump -d <kernel_name>.o > <kernel_name>.o.txt`；L0 场景只给汇编偏移属正常，围绕偏移与 10 条指令上下文判读（references/error-line-location.md「十一、报错点 10 条指令上下文」）；完成后转 P3.1。

#### P2.6 dump 失败分类与多错误判读

**动作**：判读 info.txt 第⑤段 dump 结论，必要时 grep 三类 dump 失败特征；处置 SK 场景与多错误场景。
**命令**：

```bash
grep -rn "Dump exception.*failed" <收集目录>       # GE 侧异常 dump 失败
grep -rn "D2H failed" <收集目录>                  # device 到 host 拷贝失败
grep -rn "the address maybe invalid" <收集目录>    # 异常地址无效
```

**判读**：第⑤段出现 `Failed to get dump data of error op!` 或任一特征命中即 dump 失败、结论码 102（内存分配错误方向，oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1972-1973），P4 无 dump 输入可用；SK 场景（P2.1 判定）证据链退化，定位不清时向用户给出关闭算子融合后重跑的建议（四链即可全部生效）；故障信息含多个 AI Core Error 时工具按日志时间只解析第一个（多错误行为详见 msaicerr-toolkit 技能），修复首个错误后仍有报错则重新收集走一轮 P1-P5。三类特征与 102 判读见 references/error-line-location.md「十、多错误与 dump 失败分类」，SK 处置见「九、SK 场景处理与关融合重跑建议」。
**失败分支**：dump 失败则结合 P3 寄存器解码判断是否越界、地址类错误位，P4 改走路径 B，转 P3.1；dump 有效则直接转 P3.1。

### Phase 3：错误类型识别（P3.1-P3.5）

#### P3.1 芯片形态判定（Stars/David 两法）

**动作**：判读 info.txt 第②段 AI Core DFX Register，先判芯片形态（Stars 或 David）再选择解码路径。
**命令**：无独立命令（判读动作：错误码形态如 `0x40000` 或 `257, 341` 来自第②段与 plog 原文）。
**判读**：方法一按错误码形态：以 0x 开头的十六进制指向 Stars（ASCEND_910B），逗号分隔的十进制位号指向 David（ASCEND_950），方向不得写反（oam-tools/src/msaicerr/ms_interface/aic_error_info.py:25-40）；裸 `0` 两芯片同形，只能落到 Stars 路径走 trap_or_timeout 兜底。方法二按 dump info 寄存器键：含 `sc error info:`、`su error info:`、`l1 error info:` 指向 David，含 `ifu error info:`、`ccu error info:`、`biu error info:` 指向 Stars；两法冲突以方法二为准（aic_error_info.py:43-58）。两法详解见 references/error-type-decode.md「芯片形态判定（两法）」。
**失败分支**：两法都判不出（返回 None）时按 Stars 路径兜底处理并在报告注明判定置信度降低；否则按判型转 P3.2。

#### P3.2 错误位解码

**动作**：按判型把错误码解码为错误位名与释义，查高频错误位表与字典。
**命令**：

```bash
grep -n "<错误位名>" oam-tools/src/msaicerr/ms_interface/constant.py    # 表未收录的错误位回字典查原文
```

**判读**：Stars 把十六进制错误码按二进制展开为置位位号，查 Stars 字典（constant.py:155-369，共 176 位）；David 错误码即绝对位号列表，直接查 David 字典（constant.py:378-1123，共 187 条），位号按模块分段（CUBE 从 0、MTE 从 64、L1 从 128、SC 从 192、SU 从 256、VEC 从 320 起）；无位置位时输出兜底文案 `trap_or_timeout, timeout or trap error`，这不是位 0，不能直接读成"确认超时"。高频 30 条错误位表（含越界读、越界写、UB 越界绕回、原子溢出、输入数据非法等根因方向与 constant.py 出处行号）见 references/error-type-decode.md「高频错误位表」。
**失败分支**：位号在 David 字典查不到时输出 `unknown error bit <位号>`，如实记录不臆断含义；错误码为 `0x0` 时转 P3.4 场景 5 三元组重读；否则转 P3.3。

#### P3.3 六模块子域分析

**动作**：对 Stars 命中的模块提取并解读子寄存器串（IFU、MTE、CCU、BIU、CUBE、VEC 六模块）。
**命令**：无独立命令（判读动作：子寄存器串来自 plog 的 `vec error info:` 等六个键）。
**判读**：每个单元首次命中时提取 `<单元键>=(\S+)` 解析错误类型与出错地址；MTE 按错误位动态选子字典（SOC、FMC、FMD、UNZIP、AIPP）；approximate 地址是补零近似值仅供参考；VEC 的 repeats 字段表示错误重复次数；David 不走子域分析，模块信息直接读错误名前缀（CUBE_、MTE_、L1_、SC_、SU_、VEC_）。位段表与五张子字典见 references/error-type-decode.md「六模块子域分析」。
**失败分支**：子寄存器串取不到时输出 `No <单元>_ERR_INFO found` 属正常（plog 可能只记录部分单元），靠主字典释义继续，转 P3.4。

#### P3.4 特殊场景判读（五项）

**动作**：套用五类特殊场景的专门判定逻辑，优先于一般解读。
**命令**：

```bash
grep -rn "dha status 1" <收集目录>                 # atomic 溢出链（仅错误码恰为 0x800000 时）
grep -rn "isconcurrentexe" <收集目录>              # 并发执行判定
grep -rn "The extend info: errcode:" <收集目录>    # 错误码 0x0 的 v300 三元组
```

**判读**：场景 1 atomic 溢出，等于错误码恰 `0x800000` 加 `dha status 1` 命中加 memset 检查缺失（AtomicLaunchKernelWithFlag 未找到），三证据并存时结论按 105 优先；场景 2 args 前后改写，即第④段 args before 与 after 不一致，指向 106 方向；场景 3 outstanding 双 PC，主正则无命中时降级 AICORE_ERR_OCCUR_OST，需确认使用与出错任务对应的 PC；场景 4 并发 isconcurrentexe，去重后任务数超过 2 时放弃自动判定需人工分析，并发时 atomic 结论可能误报；场景 5 错误码 `0x0` 不等于无异常，从 extend info 三元组重建错误码，重建不出按 trap_or_timeout 兜底。五项详解见 references/error-type-decode.md「特殊场景判读」。
**失败分支**：错误码 0x0 且三元组重建失败时按无位置位兜底处理并注明置信度低；否则转 P3.5。

#### P3.5 AI Core 数量与 blockDim 检查及结论码对照

**动作**：核对报告基本信息段的 rts_block_dim 与 driver_aicore_num，并对照结论码语义表。
**命令**：无独立命令（判读动作：直接读报告 Basic information 段的 `rts_block_dim` 与 `driver_aicore_num`）。
**判读**：两者均有效且 rts_block_dim 大于 driver_aicore_num 的 2 倍时，结论为环境 AI Core 数量不满足算子并发需求（oam-tools/src/msaicerr/ms_interface/aic_error_info.py:219-224）；结论码语义：101 单算子运行错误、102 内存分配错误、103 硬件错误、104 算子输入数据错误、105 框架未执行 memset 清零、106 算子 args 被踩、107 atomic 算子溢出（oam-tools/src/msaicerr/ms_interface/constant.py:72-78），全表见 references/error-type-decode.md「结论码 101-107 全表」。P3 产出的错误位与特殊场景结论供 P5 证据链汇聚使用。
**失败分支**：driver_aicore_num 获取失败对应工具状态码 10，检查驱动与 runtime 动态库后重试；否则转 P4.1。

### Phase 4：单算子复现（P4.1-P4.6）

#### P4.1 复现路径选择决策

**动作**：按 dump 有效性选择复现路径 A（基于 dump 的标准复现）或路径 B（构造数据复现）。
**命令**：

```bash
ls <收集目录>/dfx/data-dump/<device_id>/    # 人工快速核对是否存在 exception_info.* 文件
```

**判读**：dump 解析产出输入数据（info.input_list 非空）或 dfx/data-dump 下存在 exception dump 文件则走路径 A（转 P4.2）；L1 场景无 exception dump 数据或 dump 收集失败则走路径 B（转 P4.4 起构造）。决策表见 references/reproduction-guide.md「二、复现路径选择决策」。
**失败分支**：拿不准时直接查看 dfx/data-dump/<device_id>/ 下 exception_info.* 开头文件，存在则路径 A，否则路径 B；不得因无 dump 而停止复现（无 dump 只影响复现输入来源）。

#### P4.2 路径 A：dump 数据解析

**动作**：用 msaicerr -d 把 dump 文件解析为 npy 或 bin 张量文件。
**命令**：

```bash
python3 msaicerr.py -d <dump文件或目录路径> -out <结果目录> -dtype <目标dtype>    # .bin 文件必须给 -dtype
```

**判读**：产出 tensor 文件命名 `{kernel名}.{input|output|workspace}.{序号}[.{dtype}].{npy|bin}`；`.npy` 文件被拒绝（报 `The dump file cannot be an npy file.`）；-dtype 官方文档列举 13 种常用值，代码实际校验集 VALID_DTYPES 为 DATA_TYPE_TO_DTYPE_MAP 全部 43 个值，判读以代码校验集为准（依据：docs/zh/msaicerr/Dump_files_data_types_conversion.md:21；ms_interface/dump_data_parser.py:93、:496-497）；bfloat16 依赖第三方库 bfloat16ext；dump 文件头 input_type==7 的条目即 tiling 数据。输入形态三分支与命名规则见 references/reproduction-guide.md「3.1 dump 数据解析（msaicerr -d）」，-dtype 与 bfloat16ext 详见 msaicerr-toolkit 技能。
**失败分支**：报 `Can not read with dtype bfloat16` 时安装 bfloat16ext 后重试；退出码 4 为 -d 路径校验失败或解析异常，核查路径形态后重试；否则转 P4.3。

#### P4.3 输入数据有效性判读（104 证据链）

**动作**：判读 dump 解析输出的数据有效性提示，形成 104 证据链。
**命令**：

```bash
grep -n "Input data invalid\|Input data maybe invalid" <解析输出日志>
```

**判读**：NaN/Inf 命中（`{类型}[{序号}] NaN/INF. Input data invalid. Please check!`）为强证据，输入数据非法、关联 104；0.9 倍值域命中（`{类型}[{序号}] max <最大值> or min <最小值>. Input data maybe invalid. Please check!`）为弱证据（源码措辞为 maybe），须结合业务值域人工确认后再关联 104；两提示串均含子串 `data invalid`，是结论码 104 的直接判定依据（oam-tools/src/msaicerr/ms_interface/dump_data_parser.py:184-218，0.9 规则在 :212）。判读表见 references/reproduction-guide.md「3.2 输入数据有效性判读（NaN/Inf 与 0.9 倍值域，关联 104）」。
**失败分支**：数据无效只关联 104，不得臆断其他结论（见第五节 NEVER 清单）；无命中则数据有效，转 P4.4。

#### P4.4 tiling 重建与 16 键 config 构造

**动作**：重建 tiling 数据并构造单算子复现 config（路径 A 与路径 B 共用）。
**命令**：无统一命令（构造动作：tiling 落盘为 `{kernel_name}_tiling.bin`，config 写入 `test_{op_test}.py` 用例文件）。
**判读**：tiling 三源按优先级：一是 dump 内 tiling（input_type==7 条目），二是 plog 十六进制 args（从 `after execute:` 串推算），三是 `[AIC_INFO] tiling_data`（L1）；三源都无时打印 `No tiling data is found in dump files and logs.`。config 共 16 键，即 SingleOpCase.generate_config 的 15 键（oam-tools/src/msaicerr/ms_interface/single_op_test_frame/single_op_case.py:77-95）加第 16 键 compile_temp_dir（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1237-1238）；样例文件 test_single_op.py:27-48 仅展示其中 12 键（缺 workspace_file_list、bin_file_list、sub_ptr_addrs、workspace 这 4 键，缺省按空值处理）。逐键说明见 references/reproduction-guide.md「3.3 tiling 数据重建（三个来源）」与「3.4 复现 config 的 16 个键」。
**失败分支**：tiling 三源都拿不到时如实记录；路径 B 可传空串或手工构造字符串继续（非 .bin 后缀按 utf-8 编码处理）；否则转 P4.5。

#### P4.5 执行单算子复现（三变体顺序）

**动作**：按固定顺序执行三个复现变体，复现失败或未运行时按工具提示手动复现。
**命令**：

```bash
export PYTHONPATH=<msaicerr根目录>:$PYTHONPATH;cd <msaicerr根目录>;python3 <用例文件路径>    # 手动复现命令，工具在复现失败或未运行时打印（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1310-1319）
```

**判读**：三变体固定顺序 host_single_op（校验 host 与 device 加载同一份 kernel，比对 hash_id）→single_op（用 dump 真实输入重跑故障算子，仅 dump 有效时执行）→error_single_op（仅 single_op 未复现时执行，末参置零反向验证检测链路）；运行规范：ASCEND_SLOG_PRINT_TO_STDOUT=0、ASCEND_PROCESS_LOG_PATH 与故障 plog 物理隔离、PYTHONPATH 指向 msaicerr 根目录；每次 run_kernel 前自动执行脏 UB 预写（run_dirty_ub，用极大浮点值填满 UB，把依赖 UB 初值为 0 的隐性缺陷变成确定问题）。变体表见 references/reproduction-guide.md「3.6 三变体执行顺序与判读」，运行规范见「3.5 复现用例的生成与运行规范」，路径 B 构造见「四、路径 B：无 dump 数据的构造复现」。
**失败分支**：SK 场景跳过 single_op 与 error_single_op（无 device 侧 .o）属正常；用例未跑起来（NOT_RUN）时检查用例文件、依赖（numpy、tbe、ccec、驱动）、日志路径与权限后重试；执行后转 P4.6 判读结果。

#### P4.6 复现结果三态判读与越界探测

**动作**：按三态判读复现结果，结合 magic 与 tail 两遍结果判读越界。
**命令**：

```bash
grep -n "aicore error\|aivec error\|aicore exception" <单算子日志目录>/*.log
```

**判读**：三态判读：复现成功等于日志命中五种错误串之一（如 `there is an aicore error exception`）且同一日志文件内出现目标 kernel 名（RetCode.FAILED，注意语义反转，FAILED 才是复现成功），问题锁定算子自身；未复现（RetCode.SUCCESS）什么都没证明，走升级路径；未运行（NOT_RUN）检查环境后重试，NOT_RUN 同样计入 101 方向。越界判读以 magic_ret（1 为向前越界、2 为向后越界）与 tail 遍 sync_ret 为主；`Access Memory without OverBoundary` 关键字在当前 msaicerr 代码内没有产出方（mem_monitor 字段实际常为空串，属待接通钩子），运行时侧若输出该关键字仍可作越界佐证，判读时如实注明此口径。三态表见 references/reproduction-guide.md「六、复现结果三态与下一步动作」，双模式见「五、内存越界探测：tail 与 magic 双模式」。
**失败分支**：未复现不臆断根因，转 P5.4 升级路径；复现成功或 NOT_RUN 转 P5.1 汇聚。

### Phase 5：证据链汇聚与修复建议（P5.1-P5.5）

#### P5.1 四证据链汇聚

**动作**：把 P2 至 P4 的产出按四证据链框架逐链填写事实与判定，形成汇总表。
**命令**：无（汇聚动作：按 `证据链→事实→判定字段→是否支撑结论码` 四列逐链填写）。
**判读**：四链权威定义：反编译→101（出错在算子源码哪一行）；寄存器解码→103（硬件报了哪一类异常，辅以 golden op 环境自检）；dump 校验→104（输入数据本身是否已脏）；单算子复现→105-107（脱离模型能否稳定复现、args 有没有被踩）（sd_doc/msaicerr-arch-doc/msaicerr-doc/README.md:12-17）。某条链数据缺失时明确写"本链失效（原因）"，不允许留空或臆造。各链证据来源与报告落点见 references/fix-suggestions.md「一、四证据链框架」。
**失败分支**：链大面积缺失（无 dump、复现 NOT_RUN）时结论置信度降级为低，只输出排查方向；否则转 P5.2。

#### P5.2 结论码优先级链判定

**动作**：按 get_return_code 优先级链取最终结论码。
**命令**：无（判定动作：优先级链为 `get_return_code` 的固定 elif 逻辑）。
**判读**：优先级链 105→107→104→106→102→101→103→0，先命中先返回（oam-tools/src/msaicerr/ms_interface/aicore_error_parser.py:1963-1980）；四个判读细节：一是 107 在返回码层面实际不可达（atomic_add_err 求值前提含 atomic_clean_check 为假，会先命中 105，两条件互斥）；二是 NOT_RUN 同样命中 101（用例没跑起来不能认为没有发现问题）；三是 flag_check、AI Core 数量不足、current_pc 为 0x0 没有专属返回码，只进结论文案；四是结论文案链与返回码链排序不同，人工读文案、脚本读返回码时预期到差异。逐级触发条件见 references/fix-suggestions.md「二、结论码优先级链（get_return_code）」与 references/toolchain-orchestration.md。
**失败分支**：全部判据未命中返回 0（兜底），0 不代表故障不存在，逐节核对六节报告证据，证据不足转 P5.4 升级路径；否则转 P5.3。

#### P5.3 修复建议映射

**动作**：按结论码与错误位查修复建议映射表，输出修复建议与验证方法。
**命令**：无（查表动作：以 `<结论码或错误位>` 为键查映射表）。
**判读**：映射表每行为四元组（触发条件、根因方向、修复建议、验证方法），约 25 条；代表性示例：结论码 105（图中算子前未插入 memset/atomic_clean）→排查 GE 侧清零算子未插入原因→验证方法为复跑后 plog 能检索到 `AtomicLaunchKernelWithFlag_<node_name>` 且 AI Core Error 不再复现；错误位 vec_ub_wrap_around（UB 越界绕回）→检查 UB 空间分配与单次搬运块大小、循环次数是否匹配。结论码触发与错误位触发同时存在时以结论码行为主、错误位行为根因细化；错误位名称以 constant.py:155-1123 为唯一名称源。完整映射表见 references/fix-suggestions.md「三、修复建议映射表」。
**失败分支**：映射表未覆盖且无法按已覆盖根因方向（越界读写、UB 越界绕回、原子溢出、输入数据非法、地址未对齐、ECC 硬件类）就近归类时，只给排查方向不给确定性根因，并注明依据不足；否则转 P5.4。

#### P5.4 复现失败升级路径

**动作**：复现未复现或证据不足时，输出升级排查路径。
**命令**：无（建议动作：含 `msSanitizer` 与 `op_debug_config=oom` 等四条路径）。
**判读**：四条路径：一是用 msSanitizer 工具再次检查算子；二是开启 op_debug_config=oom 内存错误检测后复跑检查；三是 SK 场景关闭算子融合后重跑，让故障以单算子形式复现（四链全部生效）；四是怀疑框架问题时联系技术支持上报。复现失败场景只给排查方向，不给确定性根因（未复现什么都没证明）。详见 references/fix-suggestions.md「四、复现失败升级路径」。
**失败分支**：升级路径执行后仍未定位时按停止条件结束本轮，如实汇报已排除项与剩余未知；否则转 P5.5。

#### P5.5 输出六段式诊断报告

**动作**：按第四节输出契约组装诊断报告，每个结论附证据链格式与反证。
**命令**：无（输出动作：报告模板含 `counter_evidence` 反证字段）。
**判读**：六段齐备（环境与输入、场景与定位、错误类型、复现结果、证据链汇聚判定、修复建议与升级路径）；每个诊断结论包含事实、对比（判据出处）、判定、置信度四要素并附反证；置信度口径：高为结论码由判定字段直接触发且至少两条证据链相互印证，中为仅一条链支撑或存在可解释反证，低为 0 兜底或证据链大面积缺失只输出排查方向。模板见 references/fix-suggestions.md「五、证据链格式诊断报告模板」。
**失败分支**：任一段证据不足时如实写"证据不足"与缺失原因，不得编造；报告输出后对照停止条件判定结束或进入新一轮收集。

## 四、六段式诊断报告输出契约

诊断完成后按以下固定顺序输出六段式报告，段落名不得增删改：

1. **环境与输入**：运行环境结论（本地分析确认、python 版本、ASCEND_OPP_PATH、环境自检结果）、输入四要素、收集目录形态、降级标记清单。
2. **场景与定位**：场景判定结论（L0/L1/ffts/SK 及证据行）、报错行定位结论（Corrected Info 优先，偏移、cce 行号、TBE 源码行三级）、定位降级说明。
3. **错误类型**：芯片形态判定结论（两法）、错误位名与释义、模块子域解读、特殊场景判读结论。
4. **复现结果**：复现路径（A 或 B）、三变体执行情况、三态结论（复现成功、未复现、未运行）、越界探测判读（以 magic_ret 与 sync_ret 为主）。
5. **证据链汇聚判定**：四链汇总表（链名、事实、判定字段、是否支撑结论码）、优先级链命中的最终结论码、置信度与反证。
6. **修复建议与升级路径**：按映射表给出修复建议与验证方法；复现失败或证据不足时给出升级路径。

写入规则：事实列只写工具实际产出；每个判定附判据出处（path:line 或字典行号）；降级与证据不足如实标注；反证段不得省略，无反证时写"无"。

**停止条件**（出现以下任一情形即停止或结束本轮诊断）：

1. 环境不支持：异地环境、Ascend RC 形态、python 低于 3.7.5，停止并如实告知用户。
2. 算子命中 8 个暂不支持清单，停止并告知。
3. 收集目录无 plog，停止，要求用户重新收集或补齐日志后再进入后续阶段。
4. 用户无法补齐关键输入（求助发出后无有效响应），结束并说明已完成项与缺失项。
5. 修复首个 AI Core Error 后业务仍有报错：不算失败，重新收集故障信息，从 P1 开始新一轮（工具每次只解析日志时间上第一个错误）。
6. P5.5 报告输出完毕且用户无新输入，正常结束。
7. 升级路径（P5.4）执行后仍未定位：结束本轮，如实汇报已排除项与剩余未知，不臆断根因。

## 五、执行纪律（NEVER 清单）

1. NEVER 在 `-p` 指定的目录或其子目录内执行 msaicerr，也绝不把 `-p` 目录或其子目录用作 `-out`（oam-tools/src/msaicerr/msaicerr.py:120-123，违反时工具解析卡住或失败）。
2. NEVER 伪造定位或复现结果：报错行、错误位、复现三态只能来自工具实际产出（info.txt 与日志原文），查证不了就写"待确认"，禁止编造。
3. NEVER 在 dump 数据无效时关联 104 以外的结论：NaN/Inf 与 0.9 倍值域命中只支撑 104 证据链，不得据此臆断 101、103 等其他结论。
4. NEVER 在复现失败（未复现）时臆断根因：未复现什么都没证明，必须走 P5.4 升级路径（msSanitizer、op_debug_config=oom、关融合重跑、框架侧上报）。
5. NEVER 对 8 个暂不支持算子强行分析（MatmulAllReduce 类、MemSet 等，oam-tools/docs/zh/msaicerr/msaicerr_functions_and_restrictions.md:8-16），命中即如实告知用户并停止。
6. NEVER 修改 msaicerr 或 oam-tools 产品代码来配合诊断或复现：复现框架是诊断工具的一部分，改源码会污染证据链。
7. NEVER 在 asys 环境变量与业务运行时不一致时静默继续收集：6 个环境变量必须先核对，不一致先警示用户并要求确认。
8. NEVER 在无 plog（dfx/log/host/cann 无日志）时继续分析：无法提取 AI Core Error 信息，停止并要求重新收集。
9. NEVER 输出不带证据链的报告结论：每个判定必须包含事实、对比（判据出处）、判定、置信度并附反证。
10. NEVER 把退出码 101-107 当作工具报错（它们是分析结论码），也不把退出码 0 当作无故障（0 是兜底，不代表故障不存在）。
11. NEVER 仅凭日志含 aicore error 关键字就认定复现成功：必须同一日志文件内出现目标 kernel 名，防止把其他算子的错误算到目标算子头上。
12. NEVER 假设设备内存或 UB 初值为 0（malloc 后填随机字节、UB 不自动清零），也不在异地环境分析故障日志（仅本地分析）。

## 六、信息来源

本技能知识来自 CANN 社区仓 [cann/oam-tools](https://gitcode.com/cann/oam-tools) 源码与官方文档（docs/zh/msaicerr、docs/zh/asys）：工具用法层（CLI 参数、环境准备、三要素排查、asys collect 用法、多错误取第一）以 `msaicerr-toolkit` 与 `asys-toolkit` 技能为真源，本技能只引用不复制；退出码全表、结论码优先级链与收集降级规则等本技能独有知识见 references/toolchain-orchestration.md。本文所有 path:line 引用相对于 cann/oam-tools 仓根目录，行号为源文件真实行号，可疑时按行号回读源码核实；不确定的行为不推断，查证不了的写"待确认（需查 <文件>）"。部分架构设计细节引用内部架构文档（sd_doc 工作区路径，仅溯源标注）。
