# PC → 源码行映射

**定位**：将 trace 中的 PC / instruction address 映射回 AscendC 源码行，回答"阻塞指令来自哪段代码"。

**使用时机**：空泡分析已定位到具体 pipeline / 指令，但仍无法判断对应 AscendC 源码；trace 事件、`args` 或事件名中包含 PC / address 字段。

---

## 前置条件

满足以下条件时使用：

1. 拥有与这次仿真**逐字节一致**、以 `-g` 保留 debug/line info 且未 strip 的 ELF（先抽取 `.aicore_binary` 内嵌 ELF）。
2. 已确定可靠 runtime load base。

如果没有可靠 base，或二进制与运行产物不一致，优先重新采集 / 重编译；不要静默猜测。

## 产物准备

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 保留 debug/line info | 编译时加 `-g`，不执行 strip；高优化级别下行号可能指向内联或被移动后的位置 |
| 2 | 抽取设备 ELF | `.aicore_binary` / fat object 不能直接送 `llvm-symbolizer`，先执行 `msobjdump --extract-elf <aicore_binary> --out-dir /tmp/extracted` |
| 3 | 确认 ELF | 使用抽取出的 `.aicore.o` / 设备 ELF，而不是容器文件或事后重编译的新产物 |
| 4 | 确定 base | 优先取仿真器 / trace 明确记录的 load address；其次 map file、linker script、ELF LOAD 段；仅在可验证页对齐场景使用 `--infer-base` |
| 5 | 记录 hash | 输出中的 `binary_sha256` 用于确认映射对象与运行产物一致 |

换算公式：`object_offset = runtime_pc - load_base`。

## 运行映射脚本

```bash
# 先抽取设备 ELF
msobjdump --extract-elf aicore_binary --out-dir /tmp/extracted

# 从 trace 抽取 PC 并映射
python3 {skill_path}/scripts/map_pc_to_source.py \
  --trace ./report/trace_core0.json \
  --binary /tmp/extracted/*.aicore.o \
  --base 0x1000 \
  --source-root /absolute/path/to/kernel/project \
  --output pc_source_map.json

# 只映射手工给出的地址
python3 {skill_path}/scripts/map_pc_to_source.py \
  --binary device.elf --address 0x10234 --address 0x10567 --base 0x1000

# 复用映射结果（按 binary hash + object offset 缓存）
python3 {skill_path}/scripts/map_pc_to_source.py \
  --trace ./report/ --binary device.elf --base 0x1000 \
  --cache pc-source-cache.json
```

脚本会从 Chrome Trace 事件的以下位置提取 PC：

| 位置 | 字段 / 规则 |
|------|-------------|
| `args` 等嵌套 mapping | `pc`、`address`、`addr`、`instruction_addr`、`instruction_pc`、`pc_address` |
| 事件文本字段 | `name`、`detail`、`message` 中 `PC=0x...`、`PC: 0x...`、`ADDRESS: 0x...` 等模式 |

PC 提取字段基于常见 trace 结构实现；不同 npusim 版本字段可能不同。若输出 `unresolved_count` 为 0 且 `addresses` 为空，应先检查 trace 原始 JSON 中实际 PC 字段名，再手动通过 `--address` 传入。

## 输出解读

| 字段 | 含义 |
|------|------|
| `binary_sha256` | 映射二进制指纹；与运行产物不一致时旧地址可能全部失效 |
| `base` / `base_mode` | base 值及来源：`explicit`、`inferred`、`offset` |
| `addresses[].offset` | `runtime_pc - base` 后传给 symbolizer 的对象内偏移 |
| `addresses[].frames` | 全部符号化帧，包含 inline 帧 |
| `addresses[].source` | 优先选择 source root 下的第一个真实源码帧 |
| `addresses[].contexts` | 该 PC 在 trace 中的事件上下文（事件名、pid/tid、ts/dur） |
| `addresses[].resolved` | 是否找到源码位置；false 时必须保留 unresolved，不得伪造匹配 |
| `diagnostics` | symbolizer 缺失、base 异常、trace 解析失败等诊断 |

默认输出是完整 JSON。`--strict` 将 unresolved 或 diagnostics 转为非零退出码，适合 CI；不带 `--strict` 时，映射失败是诊断元数据，不应导致外围性能报告失败。
