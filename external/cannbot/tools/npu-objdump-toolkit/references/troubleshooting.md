# msobjdump 故障排查

先看用户实际命令、退出码和输出，再检查相关依赖。不要把生成 ELF 的编译依赖与解析已有 ELF 的使用依赖混为一谈。

## 命令或 Python 包不可用

确认用户所选 CANN 安装目录包含 `set_env.sh`，在同一 shell 中 source 后重试 `msobjdump -h`。新 shell 不自动保留此前加载的环境。

多版本或虚拟环境混用时检查：

```bash
command -v python3 msobjdump readelf
python3 -c 'import msobjdump; print(msobjdump.__file__)'
```

命令来自一套 CANN 而 Python 模块来自另一套安装时，先调整当前 shell 环境。`-V` 是 verbose，CLI 未提供通用的 `--version` 参数；版本来源应使用安装记录或包信息，dump 输出的 `VERSION` 不能代替软件版本。

缺 `readelf` 或 `ar` 时可提示检查系统 binutils；缺 `llvm-objcopy` 或 `llvm-objdump` 时检查对应 CANN 编译工具目录是否在 PATH。不是所有输入都需要 ar/llvm-objcopy，llvm-objdump 仅 `--sass` 使用，不应因此要求安装完整驱动、Ops 包或启动 NPU。未经用户要求不自动安装。

## 路径带空格或特殊字符

Shell 包装脚本若以未引用的 `$@` 调用 Python，外层引用仍会在脚本内部被拆词或展开，表现为 `unrecognized arguments` 或“文件不存在”。**当前已安装的 CANN 包（如 9.2.0）入口仍是未引用 `$@`；asc-tools 新版入口已修复为 `"$@"`，随后续安装包发布。** 核对命令入口与模块来源后，在旧入口上改用同一包的模块 CLI：

```bash
python3 -m msobjdump --dump-elf "$input_file"
python3 -m msobjdump --extract-elf "$input_file" --out-dir "$output_dir"
python3 -m msobjdump --sass "$input_file"
```

该方式绕过 Shell 包装，不绕过 Python 解析器，也不能修复解析器本身的问题。不要为了消除报错重命名原文件。

某些旧实现按路径中是否含 `.a` 判断静态库，可能将 `demo.aicore.o` 误送入归档预处理。当前仓库 `FileAction`（dump/list/extract 路径）使用 `.endswith('.a')`；`--sass` 使用独立的 `SassFileAction`，不做该预处理。具体安装版本须另行核对。仅在确认误分类后，才能在独立目录中建立普通名称的副本以辅助诊断，保留原路径和来源，不把改名当成对工具的修复。

## 格式与解析输出

| 现象 | 下一步 |
| --- | --- |
| `The kernel meta information cannot be found.` | 检查文件确实存在、可读，readelf 是否正常，输入是否包含受支持的 Ascend 元数据；不能仅据此宣称文件损坏或成功解析 |
| `nothing to list in single op elf file` | 输入是已独立的单算子 ELF，没有可列出的内嵌文件；需要元数据时使用 dump |
| `nothing to extra in single op elf file` | 没有可提取内容，不应声称生成了新 ELF |
| `llvm-objcopy is not available` | 融合产物需要 device 段提取；检查所选 CANN 的 PATH |
| `no device ELF found in input` | `--sass` 在输入中未找到 device 镜像：确认输入含 device 内容（可先 dump/list 查看），host-only ELF 属于预期失败，不是文件损坏 |
| `thin archive is not supported` | `--sass` 不支持 thin archive；改用普通归档或先解出成员再反汇编 |
| `llvm-objdump produced no valid instruction lines` / 输出全为 `<not available>` | 当前 PATH 选中的 llvm-objdump 缺 Ascend device 架构反汇编后端（同版本号也可能有差异）；换用带该后端的环境（如 CANN 的 bisheng_compiler 工具链）或更新 CANN，不重命名输入 |
| `llvm-objdump is not available, please source the matching CANN environment` | `--sass` 需要 llvm-objdump；source 匹配的 CANN 后重试，不要安装系统 llvm 替代 |
| `--sass cannot be combined with ...`（退出码 2） | 参数互斥属预期行为；dump/list/extract/sass 分开调用，保存结果用 Shell 重定向 |
| 提取失败、结果为空或异常回溯 | 保存 stderr 和实际文件检查结果；确认工具版本与产物格式，不能自动用其他格式解释 |
| 重复全局元数据或未知字段 | 保留原输出；当前源码只在部分输入路径去重。重复行不等于重复 kernel，未知字段不猜义 |
| 静态库没有匹配成员 | ar 可用并不意味着任意归档都支持；记录归档结构和实际输出，说明工具限制 |

需要读取结构时，使用已确认的输入绝对路径：

```bash
readelf -SW "$input_file"
readelf -sW "$input_file"
```

对于无输出或只打印错误的情况，不能只检查 msobjdump 的退出码。新版已传播多数错误（运行错误退出 1、参数错误退出 2），但部分外部命令错误仍未向上传递，需要直接观察 readelf 的退出码和 stderr。

## 工作目录与提取结果

`--out-dir` 默认当前目录，指定目录需要预先存在。list/dump 同样创建临时目录（位于 out-dir 下），dump/list/extract 的静态库预处理可能向 cwd 提取成员，因此先选独立可写 cwd，再传入绝对输入路径。`--sass` 对静态库使用 `ar p` 写入临时目录，不污染 cwd；其结果只写 stdout，落盘文件由用户重定向产生。

extract 不一定打印成功提示。检查目标目录新增的文件及其大小，必要时再次解析提取文件；不同输入路径下的输出可能有重复项差异，不能只按全文相等判断元数据一致。

工具可能覆盖同名文件。提取前选择空目录，或使用用户同意的覆盖策略。异常后只处理本次创建的临时目录，不递归清空用户输出目录，也不清理其他进程的 `objdump_*`。

## 依据

- asc-tools `utils/msobjdump/msobjdump.sh`：Shell 参数转发（新版已引用 `"$@"`）。
- `utils/msobjdump/msobjdump/__main__.py`：模块 CLI 入口。
- `utils/msobjdump/msobjdump/msobjdump_main.py`：`FileAction`、`SassFileAction`、`_set_out_dir`、`_extract_aicore_binary`、`_classify_sass_file`、`_collect_sass_images`、`_disassemble_sass_image`、`_sass_process`、`_list_elf`、`_extra_elf`、`_clean`。
- `utils/msobjdump/msobjdump/utils.py`：readelf/ar/llvm-objcopy/llvm-objdump 调用与文件复制。
- `scripts/package/asc-tools/`：`npu-objdump` 主命令与 `msobjdump` 兼容软链的创建与清理。

问题定位以当前安装实现为准，不将某次安装的行为扩展为全部 CANN 版本。
