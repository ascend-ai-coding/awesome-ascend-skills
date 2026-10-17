---
name: external-cannbot-tools-npu-objdump-toolkit
description: 使用 npu-objdump/msobjdump 分析 Ascend 算子编译产物。触发：查看元数据、提取 device ELF、反汇编
  AIC/AIV 指令，或结合报错日志中的异常 PC 定位指令时。
original-name: npu-objdump-toolkit
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-17'
synced-commit: fe65c77485d42a93c7e45f3e67ab96d95504b9dc
license: UNKNOWN
---

# npu-objdump/msobjdump 功能与用法

本技能分析已有的 Ascend 算子编译产物，不执行输入程序。安装入口可能是 `npu-objdump` 或兼容命令 `msobjdump`，使用前以接口自检结果为准。它支持查看元数据和 ELF 结构、列出或提取内嵌 device ELF、反汇编 device 侧 Ascend AICore 指令，以及按异常 PC 截取精确指令窗口。

host 侧普通 ELF 反汇编、AI Core Error 根因诊断和性能采集不在本技能范围内。

```bash
OBJDUMP_CMD="$(command -v npu-objdump 2>/dev/null || command -v msobjdump 2>/dev/null)"
test -n "$OBJDUMP_CMD"
python3 scripts/check_interface.py --tool "$OBJDUMP_CMD"
```

探测结果缺少 `--sass` 时仍可使用 dump/list/extract；不要向旧版 `msobjdump` 传入不存在的选项。

## 功能与命令

| 功能 | 命令 |
| --- | --- |
| 查看 kernel 名称、核类型、TilingKey 等元数据 | `"$OBJDUMP_CMD" --dump-elf "$input_file"` |
| 查看 ELF header、section、symbol 等完整信息 | `"$OBJDUMP_CMD" --dump-elf "$input_file" --verbose` |
| 列出内嵌 device ELF | `"$OBJDUMP_CMD" --list-elf "$input_file"` |
| 提取内嵌文件 | `"$OBJDUMP_CMD" --extract-elf "$input_file" --out-dir "$output_dir"` |
| 反汇编 device 侧 Ascend AICore 指令 | `"$OBJDUMP_CMD" --sass "$input_file"`（仅探测确认支持时） |

`input_file` 可以是独立 device ELF、包含 `.aicore_binary` 的融合编译 host ELF、普通 `.a` 静态库，以及工具支持的 `.ascend.kernel.*`、`_o/_json` 封装产物。工具按文件内容识别格式，不能只凭扩展名判断是否支持。

## 使用说明

- 每次只使用 `--dump-elf`、`--list-elf`、`--extract-elf`、`--sass` 中的一种主操作；需要多个结果时分别调用。
- `--verbose`（或 `-V`）是 `--dump-elf` 的详细模式，不是版本查询选项。
- 必须指定主操作，不能只写 `"$OBJDUMP_CMD" "$input_file"`。
- 提取前创建 `output_dir`；命令结束后检查目录中的实际文件，不能只按退出码或终端输出判断是否提取成功。
- `--sass` 输出到标准输出。保存结果时使用 `"$OBJDUMP_CMD" --sass "$input_file" > "$output_file"`，不要与 `--out-dir` 组合。
- `--sass` 依赖带 Ascend device 架构反汇编后端的 `llvm-objdump`；融合编译产物还需要 `llvm-objcopy`，静态库需要 `ar`。thin archive 不支持。
- `--sass` 中的 SASS 指 Ascend AICore 指令，不是 NVIDIA SASS。host 侧 x86/AArch64 指令应使用对应的 GNU/LLVM objdump。
- 需要从 OPP 中筛选实际对象、区分 MIX AIC/AIV 物理符号，或按 ERR_INFO/对象 PC 定位故障指令时，读取[异常 PC 精确反汇编](references/fault-pc-disassembly.md)。该流程直接调用匹配 CANN 的 `llvm-objdump`，并不假定当前安装一定提供 `npu-objdump` 命令或 `--sass`。

## 异常 PC 工作流

1. **确认环境与输入**：记录实际 `npu-objdump`/`msobjdump`、`llvm-objdump` 路径；容器输入先提取并确认独立 device ELF。
2. **锁定对象**：优先使用已确认的 device ELF；搜索 OPP 时按算子名及已知 SoC 筛选，TilingKey 为可选过滤条件，再以运行时路径或报告中的 SHA256 核对候选身份。哈希不一致时停止映射。
3. **归一化 PC**：确认 EXCP DS 对应单元、寄存器格式、字节/指令单位和地址域。缺少可靠基址时保留证据缺口，不把相对偏移当对象 VMA。
4. **选择并验证指令**：MIX 对象按异常 core type 选择 `_mix_aic` 或 `_mix_aiv`，用 CANN `llvm-objdump` 截取窗口；目标 PC 未产生指令或符号不覆盖该地址时返回上一步核对。

完成时应给出对象路径与 SHA256、物理符号、原始及归一化 PC、换算公式、实际 `llvm-objdump` 路径和包含目标指令的窗口。反汇编结果只证明所选对象内容，不能单独证明运行时加载身份或错误根因。

## 结果说明

- `--dump-elf` 展示的是编译产物元数据，不能据此判断真实运行核数、精度或性能。
- `--list-elf` 对单算子 ELF 提示 `nothing to list`，表示没有内嵌文件，不代表输入损坏。
- `--extract-elf` 返回后，以输出目录中实际生成的非空文件为准。
- `--sass` 对多个 device 镜像分别输出来源和反汇编内容；没有 device ELF 或反汇编依赖不可用时，不应声称反汇编成功。
- `--extract-elf` 和 Shell 重定向会写文件；使用独立输出目录，检查同名覆盖，并保留输入路径和哈希。其他步骤只读，不执行待分析二进制。

## 修改验证

改动本 Skill 或脚本后，至少运行 Python 语法检查、Skill 结构/内容检查和 `git diff --check`。ERR_INFO 算术回归应确认 CCU `0x32b510506c8000e9` 与 MTE `0xd9000000e9` 都得到 `0x367a4`；真实对象回归还要确认 AIC/AIV 物理符号分别可反汇编，窄窗口包含目标对象 PC，错误符号和越界偏移返回非零。

```bash
python3 -m py_compile tools/npu-objdump-toolkit/scripts/*.py
python3 tools/npu-objdump-toolkit/scripts/decode_err_info_pc.py \
  --error-type MTE --err-info 0xd9000000e9 \
  --pc-base 0x36000 --direct-offset 0x7a4
INCREMENTAL_SKILLS=npu-objdump-toolkit bash tests/unit/skills/test-structure.sh
INCREMENTAL_SKILLS=npu-objdump-toolkit bash tests/unit/skills/test-content.sh
python3 tools/npu-objdump-toolkit/scripts/gen_interface.py \
  --source-root /path/to/asc-tools --check
git diff --check
```

验证结果应同时覆盖：接口快照与 asc-tools 源码一致、`decodedPcBytes=0x367a4`、同域比较得到 `equal=true`、真实 MIX 对象的 AIC/AIV 符号可分别输出指令，以及非法参数返回非零。没有真实对象时明确标记该项未验证，不能以语法检查代替反汇编回归。

## 参考资料

- [元数据说明](references/metadata.md)：字段含义和解释边界。
- [异常 PC 精确反汇编](references/fault-pc-disassembly.md)：OPP 对象筛选、ERR_INFO PC 核验、MIX AIC/AIV 物理符号选择和窄指令窗口。
- [故障排查](references/troubleshooting.md)：依赖、格式和调用问题。
- [自检与更新规则](references/update-policy.md)：来源或 CLI 发生变化时读取。
