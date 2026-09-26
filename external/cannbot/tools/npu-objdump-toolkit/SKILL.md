---
name: external-cannbot-tools-npu-objdump-toolkit
description: 使用 npu-objdump 查看 Ascend C 算子编译产物的 kernel 元数据、完整 ELF 信息，列出或提取内嵌 device
  ELF，以及反汇编 device 侧 Ascend AICore 指令。当用户询问 npu-objdump 用法、分析 Ascend 算子 ELF、查询核类型或
  TilingKey、提取 device 文件、反汇编算子 AICore 指令时使用。不用于 host 侧（x86/AArch64 等）普通 ELF 反汇编或 NPU
  运行性能采集。
original-name: npu-objdump-toolkit
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-26'
synced-commit: edcd4b94390c2a6b664f0935fbdd8a65e3644e9f
license: UNKNOWN
---

# npu-objdump 功能与用法

`npu-objdump` 用于分析已有的 Ascend 算子编译产物，不执行输入程序。它支持查看元数据和 ELF 结构、列出或提取内嵌 device ELF，以及反汇编 device 侧 Ascend AICore 指令。

## 功能与命令

| 功能 | 命令 |
| --- | --- |
| 查看 kernel 名称、核类型、TilingKey 等元数据 | `npu-objdump --dump-elf "$input_file"` |
| 查看 ELF header、section、symbol 等完整信息 | `npu-objdump --dump-elf "$input_file" --verbose` |
| 列出内嵌 device ELF | `npu-objdump --list-elf "$input_file"` |
| 提取内嵌文件 | `npu-objdump --extract-elf "$input_file" --out-dir "$output_dir"` |
| 反汇编 device 侧 Ascend AICore 指令 | `npu-objdump --sass "$input_file"` |

`input_file` 可以是独立 device ELF、包含 `.aicore_binary` 的融合编译 host ELF、普通 `.a` 静态库，以及工具支持的 `.ascend.kernel.*`、`_o/_json` 封装产物。工具按文件内容识别格式，不能只凭扩展名判断是否支持。

## 使用说明

- 每次只使用 `--dump-elf`、`--list-elf`、`--extract-elf`、`--sass` 中的一种主操作；需要多个结果时分别调用。
- `--verbose`（或 `-V`）是 `--dump-elf` 的详细模式，不是版本查询选项。
- 必须指定主操作，不能只写 `npu-objdump "$input_file"`。
- 提取前创建 `output_dir`；命令结束后检查目录中的实际文件，不能只按退出码或终端输出判断是否提取成功。
- `--sass` 输出到标准输出。保存结果时使用 `npu-objdump --sass "$input_file" > "$output_file"`，不要与 `--out-dir` 组合。
- `--sass` 依赖带 Ascend device 架构反汇编后端的 `llvm-objdump`；融合编译产物还需要 `llvm-objcopy`，静态库需要 `ar`。thin archive 不支持。
- `--sass` 中的 SASS 指 Ascend AICore 指令，不是 NVIDIA SASS。host 侧 x86/AArch64 指令应使用对应的 GNU/LLVM objdump。

## 结果说明

- `--dump-elf` 展示的是编译产物元数据，不能据此判断真实运行核数、精度或性能。
- `--list-elf` 对单算子 ELF 提示 `nothing to list`，表示没有内嵌文件，不代表输入损坏。
- `--extract-elf` 返回后，以输出目录中实际生成的非空文件为准。
- `--sass` 对多个 device 镜像分别输出来源和反汇编内容；没有 device ELF 或反汇编依赖不可用时，不应声称反汇编成功。

## 参考资料

- [元数据说明](references/metadata.md)：字段含义和解释边界。
- [故障排查](references/troubleshooting.md)：依赖、格式和调用问题。
- [自检与更新规则](references/update-policy.md)：来源或 CLI 发生变化时读取。
