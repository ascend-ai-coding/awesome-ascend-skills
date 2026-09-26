# msobjdump 元数据说明

用于解释 dump 输出，不是所有版本或所有输入必含字段的清单。按 kernel 名称归组，保留工具打印的字段原名和数值。重复索引不代表 kernel 名称相同，缺少字段也不代表对应功能关闭。

## 二进制级信息

| 字段 | 含义与解释边界 |
| --- | --- |
| `.ascend.meta META INFO` | 全局元数据信息的打印标题 |
| `VERSION` | 产物元数据中的版本值；不是 CANN 安装版本或 msobjdump 软件版本 |
| `DEBUG` | `debugBufSize` 是调试缓冲区大小，`debugOptions` 是调试选项；按对应版本定义解释，不能从它判断本次运行已产生调试数据 |
| `DYNAMIC_PARAM` | `dynamicParamMode` 表示动态参数模式；文档定义 0 关闭、1 开启 |
| `OPTIONAL_PARAM` | `optionalInputMode` / `optionalOutputMode` 表示可选输入/输出是否占位；文档定义 0 不占位、1 占位 |
| `RUNTIME_IMPLICIT_INFO` | 工具按版本映射隐式信息，如 Printf、Hardware Sync、L2Cache Hint 标志；未知编号保留原值，不套用其他版本的映射 |

文档列出的 `debugOptions` 值为 0（关闭）、1（DumpTensor/printf）、2（assert）、4（时间戳打点）、8（内存越界检测）。遇到其他值或组合，先核对当前版本定义，不能自行猜测。

## kernel 级信息

| 字段 | 含义与解释边界 |
| --- | --- |
| `.ascend.meta. [索引]: 名称` | kernel 名称，可能为 C++ 修饰后的符号；需要可读名称时可另外使用可用的 `c++filt`，同时保留原符号 |
| `KERNEL_TYPE` | 产物声明的 core 类型；常见 AIC、AIV、MIX_AIC_MAIN、MIX_AIV_MAIN，分别指 Cube、Vector 及相应主核的混合类型；其他枚举按工具原值呈现 |
| `CROSS_CORE_SYNC` | 硬同步配置，`USE_SYNC` / `NO_USE_SYNC`；适用平台需结合当前 CANN 文档，不能扩展为所有芯片均支持 |
| `MIX_TASK_RATION` | Cube/Vector 任务配比，例如 `[1:2]`；保留工具原拼写，不解读为耗时或利用率比例 |
| `DETERMINISTIC_INFO` | 文档定义 0 非确定性、1 确定性计算；表示编译产物配置，不代替运行测试 |
| `FUNCTION_ENTRY` | 工具文档定义为 TilingKey；保留完整整数，不与符号地址、入口地址或真实执行过的分支混淆 |
| `BLOCK_NUM` | 当前源码打印固定占位值 `0xFFFFFFFF`；不能转换成实际使用 4294967295 个核 |
| `ENABLE_EARLY_START` | 当前源码识别并打印该字段整数；仅凭值不能断言 Early-Start 已在运行中生效或取得加速 |

例如输出 `KERNEL_TYPE: MIX_AIC_MAIN` 与 `MIX_TASK_RATION: [1:2]`，可以说明产物采用相应的混合核配置，不能由此推算算子耗时。`mix_aic` / `mix_aiv` 两个符号可以是同一混合算子的两部分，不能直接当作两个独立算子。

## verbose 与输入格式

verbose 额外提供 ELF header、section、program header 和 symbol 等信息，是结构信息而不是指令反汇编；device 指令反汇编由 `--sass` 完成。分析融合编译产物时，详细信息可能来自提取后的 device ELF，而不是外层 host ELF；解释架构和入口时标明对象。

| 内容特征 | dump/list/extract 处理方式 | `--sass` 处理方式 |
| --- | --- | --- |
| `.ascend.meta.*` | 直接读取 kernel 元数据；单算子 ELF 没有内嵌文件时 list/extract 提示无内容 | 按元数据定位后再反汇编 |
| 独立 device ELF（machine 为 `hiipu` 等） | 直接解析 | 直接反汇编 |
| `.ascend.kernel.*` | 按内部描述读取 kernel 二进制并列出或提取 | 提取各 kernel 后逐个反汇编 |
| `_o_start` / `_json_start` 等符号 | 识别包含二进制及 JSON 的打包产物，支持相应文件恢复 | 恢复 `.o` 负载后反汇编 |
| `.aicore_binary` | 使用 llvm-objcopy 提取 device 段，再读取元数据；列表名形如 `demo.aicore.o` | llvm-objcopy 提取后反汇编，来源标注 `demo[.aicore_binary]` |
| `.a` 归档 | FileAction 预处理按首成员 `ar x` 提取，可能落盘当前目录；不代表任意成员均支持 | 遍历全部成员（`ar p` 写入临时目录，不污染当前目录）；thin archive 明确不支持 |

这些是内容识别线索，不是扩展名白名单。无法识别时保留实际输出，不能统一归因为文件损坏。`--sass` 对嵌套容器最多向下遍历 4 层。

## sass 反汇编输出

`--sass` 调用所选 CANN 的 llvm-objdump 反汇编 device ELF，输出形如：

```text
===== [SASS] /path/demo[.aicore_binary] =====

/tmp/.../aicore_xxx.o:    file format elf64-hiipu

Disassembly of section .text:

0000000000000000 <__init_kfc_workspace_addr.cube>:
       0:                07027fff    MOV         X1, #32767
       4:                02050880    MOV        X2, COREID
```

每行为“地址、十六进制编码、指令助记符与操作数”。`<函数名>` 块按符号划分，不等于 kernel 边界。`file format` 行指明反汇编对象架构；融合产物的指令来自提取后的 device ELF，不是外层 host ELF。

`<not available>` 表示当前 llvm-objdump 没有该架构的反汇编后端，不是文件损坏；msobjdump 会将其判定为失败并报错。指令语义、调度与性能不能凭静态反汇编直接推断，需结合架构手册与实测。

## 依据

以下路径位于 asc-tools 仓库；本页已保留使用所需含义，独立分发 Skill 时不依赖仓库在固定本机路径存在。

- `docs/03_msobjdump.md`：命令格式和字段定义。
- `examples/04_msobjdump/README.md`：融合编译产物的输出示例，含 `--sass` 用法。
- `utils/msobjdump/msobjdump/msobjdump_main.py`：`B_TYPE_MAP`、`F_TYPE_MAP`、`K_TYPE_MAP`、`_show_ascend_meta_tlv`、`_show_ascend_meta_op_tlv`、`_detect_obj_type`、`_classify_sass_file`、`_collect_sass_images`、`_disassemble_sass_image`。
- `scripts/package/asc-tools/`：`npu-objdump` 主命令与 `msobjdump` 兼容软链的安装与卸载脚本。
