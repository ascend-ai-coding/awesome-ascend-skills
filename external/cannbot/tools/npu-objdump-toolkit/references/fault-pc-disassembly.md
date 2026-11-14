# 异常 PC 精确反汇编

本流程把运行时 kernel 身份、AIC/AIV 类型和异常 PC 映射到实际 OPP 对象、物理符号及具体指令。它只建立二进制证据，不判断 AI Core Error 根因。

## 1. 确认工具和输入

加载生成目标对象的 CANN 环境，再记录实际解析到的命令：

```bash
source /path/to/cann/set_env.sh
META_OBJDUMP="$(command -v npu-objdump 2>/dev/null || command -v msobjdump 2>/dev/null)"
LLVM_OBJDUMP="$(command -v llvm-objdump)"
test -n "$META_OBJDUMP" && test -n "$LLVM_OBJDUMP"
printf 'metadata tool: %s\ninstruction disassembler: %s\n' \
  "$META_OBJDUMP" "$LLVM_OBJDUMP"
"$LLVM_OBJDUMP" --version
```

`META_OBJDUMP` 用于元数据、ELF 列表和提取；确认后的独立 device ELF 由同一套 CANN 中的 `llvm-objdump` 精确反汇编。不要用缺少 Ascend 后端的系统 GNU `objdump`。`npu-objdump` 是新版 asc-tools 的安装入口，但 CANN 9.2.0 安装中可能只有 `msobjdump`，且该版本没有 `--sass`；必须以 `scripts/check_interface.py` 的探测结果为准。

容器、融合 ELF 或静态库先按本 Skill 主文档列出并提取真实 device ELF。下面的精确窗口脚本只接受已经确认的独立 device ELF。

## 2. 从 OPP 筛选对象

已有故障报告中的对象或运行缓存时优先使用。已确认独立 device ELF 时可直接进入符号选择和反汇编；只有逻辑算子名时才搜索 OPP：

```bash
python3 scripts/locate_kernel.py \
  --operator FlashAttentionScoreGrad \
  --soc ascend910b \
  > kernel-candidates.json
```

`--soc` 和 `--tiling-key` 均为可选过滤条件。已知故障现场的 TilingKey 时可追加 `--tiling-key 123956`；未知时省略该参数，保留匹配对象的全部 `kernelList` 入口，再通过运行时对象路径、kernelName、算子包版本或报告中的对象哈希消歧。TilingKey 只能缩小候选范围，不能单独确认实际加载对象。

脚本默认搜索 `ASCEND_OPP_PATH` 中存在且去重后的目录；变量缺失时回退到 `$ASCEND_HOME_PATH/opp`。也可以显式传 `--opp-root` 或 `--cann-root`。

结果包含 metadata/object 路径、对象 SHA256、metadata SHA256 及匹配结果、TilingKey、core type 和物理符号候选。返回多个候选时，继续用运行时 kernelName、SoC、算子包版本、报告对象或 SHA256 消歧。metadata 声明哈希却与对象不一致时停止映射；文件名匹配只能生成候选，不能确认实际加载身份。

## 3. 选择 MIX AIC/AIV 物理符号

保存完整 ELF 元数据：

```bash
"$META_OBJDUMP" --dump-elf exact-kernel.o --verbose > elf.txt
```

MIX kernel 可能同时包含 `_mix_aic` 和 `_mix_aiv` 物理符号。根据同一次异常记录中的 core type 选择一侧，并在 `elf.txt` 或符号表中确认符号真实存在。

反汇编整个物理符号：

```bash
python3 scripts/disassemble_pc.py \
  --obj exact-kernel.o \
  --symbol ExactKernelSymbol \
  > symbol.sass
```

已知该符号内字节偏移时截取窄窗口：

```bash
python3 scripts/disassemble_pc.py \
  --obj exact-kernel.o \
  --symbol ExactKernelSymbol --symbol-offset 0x7a4 \
  --before 64 --after 128 \
  > symbol-window.sass
```

脚本先读取函数符号表，并拒绝不存在的物理符号、越过符号范围或未按 4 字节对齐的 PC。

## 4. 从 ERR_INFO 核验 PC

只有已经根据 EXCP DS 或 exception bit 确认报错单元，并确认寄存器格式符合目标 SoC/CANN 时，才使用下表。不同单元的位域不能互换；故障工具已经给出修正 PC 时，以修正结果为主，手工公式仅用于交叉核验。

| 单元 | 16 位指令地址 | 字节 PC |
| --- | --- | --- |
| CCU | `ERR_INFO[30:23] << 8 \| ERR_INFO[7:0]` | `instruction_address * 4` |
| MTE | `ERR_INFO[39:32] << 8 \| ERR_INFO[7:0]` | `instruction_address * 4` |
| IFU | 不拼接 | 整个 IFU_ERR_INFO 已是指令 PC |
| CUBE | `ERR_INFO[31:24] << 8 \| ERR_INFO[7:0]` | `instruction_address * 4` |
| VEC | `ERR_INFO[39:32] << 8 \| ERR_INFO[7:0]` | `instruction_address * 4` |

CCU 示例：

```text
CCU_ERR_INFO = 0x32b510506c8000e9
high = (ERR_INFO >> 23) & 0xff = 0xd9
low  = ERR_INFO & 0xff = 0xe9
instruction_address = 0xd9e9
decoded_pc = 0xd9e9 * 4 = 0x367a4
```

MTE 示例 `0xd9000000e9` 同样得到 `0xd9e9 * 4 = 0x367a4`。用脚本保留公式和中间值：

```bash
python3 scripts/decode_err_info_pc.py \
  --error-type CCU --err-info 0x32b510506c8000e9
```

某些 CANN/SoC 会对 `current_pc/start_pc` 做局部修正，packed IFU 寄存器也可能需要先提取地址字段。这类工具修正与“从整个 ERR_INFO 还原 16 位地址”不是同一计算目的。格式不能确认时只报告算术结果和证据缺口，不把结果作为故障 PC。

## 5. 与直接相减结果比较

位域解码值和直接相减值只有在单位和地址域一致时才能比较：

```text
decoded_offset = decoded_pc - pc_base
direct_offset  = current_pc - start_pc
equal          = decoded_offset == direct_offset
```

例如 `decoded_pc=0x367a4`、同域 `pc_base=0x36000` 时，解码偏移为 `0x7a4`，应与直接相减得到的 `0x7a4` 比较。不能直接比较 `0x367a4` 和 `0x7a4`。

```bash
python3 scripts/decode_err_info_pc.py \
  --error-type MTE --err-info 0xd9000000e9 \
  --pc-base 0x36000 --direct-offset 0x7a4

python3 scripts/decode_err_info_pc.py \
  --error-type MTE --err-info 0xd9000000e9 \
  --current-pc 0x367a4 --start-pc 0x36000
```

向 `--direct-offset` 传相对偏移时必须同时提供同域 `--pc-base`。只有确认解码结果本身就是对象相对偏移时，才显式使用 `--pc-base 0`。缺少 `current_pc`、`start_pc`、基址或位宽语义时，不能断言两种算法相等。

## 6. 归一化为对象 PC 并截取窗口

先确认 PC 类型：

- 对象内 VMA：直接作为 object PC；
- 符号相对偏移：使用 `--symbol` 和 `--symbol-offset`；
- 运行时绝对地址：仅在加载基址可靠时计算 `object_pc = runtime_pc - load_base`。

```bash
python3 scripts/disassemble_pc.py \
  --obj exact-kernel.o \
  --pc 0x12345678 --load-base 0x12340000 \
  --before 64 --after 128 \
  > pc-window.sass
```

窗口模式输出物理符号、符号起点、符号内偏移和对象 PC，并验证目标地址确实对应一条指令。若对象身份、core type、地址域或加载基址未确认，只能报告静态反汇编，不能声称已经映射到故障现场。

## 7. 保存证据

至少记录以下内容：

- CANN 环境路径、`META_OBJDUMP` 与 `LLVM_OBJDUMP` 的绝对路径；
- metadata、device ELF 路径及 SHA256；
- runtime kernelName、SoC 和 AIC/AIV core type；TilingKey 已知时记录，缺失时标记为未知；
- 原始 PC/ERR_INFO、位域公式、基址和归一化公式；
- 物理符号、对象 PC、符号内偏移及上下文指令窗口。

反汇编只证明所选对象中的指令。判断非法访问时还需要寄存器值、访问长度和分配范围，不能只凭一条可疑 load/store 指令确认根因。
