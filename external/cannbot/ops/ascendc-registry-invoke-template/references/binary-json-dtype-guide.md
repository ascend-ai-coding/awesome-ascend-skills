# binary.json dtype 组合推导指南

> **适用范围**：ops-math / ops-nn / ops-cv / ops-transformer 仓库（使用 `add_modules_sources` + `build.sh` + OPC 编译流程）。

## 1. 规则

`op_host/config/{soc}/{op}_binary.json` 中每个 `op_list` 条目的 inputs/outputs dtype 组合，必须与 `_def.cpp` 中该 SoC 对应的 DataType 列表**按索引位置一一对应**。不一致会导致 OPC 编译静默跳过（无 kernel binary 产出）。

## 2. 推导算法

1. 读取 `_def.cpp`，找到默认 config（`this->Input("x").DataType({...})` 等，无 `OpAICoreConfig` 包装）
2. 查找是否有 per-SoC `OpAICoreConfig`（如 `config950`），通过 `this->AICore().AddConfig("ascend950", config950)` 注册
3. 对每个 SoC，确定其使用的 DataType 列表：
   - 有 per-SoC config → 使用该 config 的 DataType 列表
   - 无 per-SoC config → 使用默认 DataType 列表
4. 按索引位置枚举 dtype 组合：同一索引 i 的所有 Input/Output DataType 列表取第 i 个元素，构成一个 binary.json 条目
5. 条目编号从 0 开始（`bin_filename` 为 `OpName_0`, `OpName_1`, ...）

## 3. ge::DT_* 到 JSON dtype 映射

| `_def.cpp` (ge:: namespace) | `binary.json` (string) | 说明 |
|-----------------------------|------------------------|------|
| `ge::DT_FLOAT` | `"float32"` | 单精度浮点 |
| `ge::DT_FLOAT16` | `"float16"` | 半精度浮点 |
| `ge::DT_BF16` | `"bfloat16"` | BFloat16 |
| `ge::DT_INT8` | `"int8"` | 8位整数 |
| `ge::DT_INT32` | `"int32"` | 32位整数 |
| `ge::DT_UINT8` | `"uint8"` | 8位无符号整数 |
| `ge::DT_INT16` | `"int16"` | 16位整数 |
| `ge::DT_UINT16` | `"uint16"` | 16位无符号整数 |
| `ge::DT_UINT32` | `"uint32"` | 32位无符号整数 |
| `ge::DT_INT64` | `"int64"` | 64位整数 |
| `ge::DT_BOOL` | `"bool"` | 布尔 |

> **注意**：不能用 `"float"` 代替 `"float32"`，不能用 `"half"` 代替 `"float16"`。

## 4. 示例：同 dtype 算子（ascend910b / ascend910_93 典型）

当所有 Input/Output 使用相同 DataType 列表时（默认 config），每个 dtype 只出现一次：

```cpp
// _def.cpp — 默认 config，适用于无 per-SoC config 的平台
this->Input("x")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
this->Input("cos")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
this->Input("sin")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
this->Output("x")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
```

推导出 **3 个组合**（索引 0, 1, 2）：

| 条目 | x | cos | sin | output |
|------|---|-----|------|--------|
| `OpName_0` | float16 | float16 | float16 | float16 |
| `OpName_1` | float32 | float32 | float32 | float32 |
| `OpName_2` | bfloat16 | bfloat16 | bfloat16 | bfloat16 |

对应 JSON：

```json
{
  "op_type": "OpName",
  "op_list": [
    {
      "bin_filename": "OpName_0",
      "inputs": [
        {"name": "x", "index": 0, "dtype": "float16", "format": "ND", ...},
        {"name": "cos", "index": 1, "dtype": "float16", "format": "ND", ...},
        {"name": "sin", "index": 2, "dtype": "float16", "format": "ND", ...}
      ],
      "outputs": [
        {"name": "x", "index": 0, "dtype": "float16", "format": "ND", ...}
      ],
      "attrs": [...]
    },
    {
      "bin_filename": "OpName_1",
      "inputs": [
        {"name": "x", "index": 0, "dtype": "float32", "format": "ND", ...},
        ...
      ],
      ...
    },
    {
      "bin_filename": "OpName_2",
      "inputs": [
        {"name": "x", "index": 0, "dtype": "bfloat16", "format": "ND", ...},
        ...
      ],
      ...
    }
  ]
}
```

## 5. 示例：混合 dtype 算子（ascend950 config950）

当某个 SoC 有独立的 `OpAICoreConfig`，且其 DataType 列表与默认 config 不同时，不同 Input/Output 的同一索引可能对应不同 dtype：

```cpp
// _def.cpp — config950（仅 ascend950 使用）
config950.Input("x")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
config950.Input("cos")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
config950.Input("sin")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
config950.Output("x")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
this->AICore().AddConfig("ascend950", config950);
```

推导出 **5 个组合**（索引 0-4）：

| 条目 | x | cos | sin | output | 说明 |
|------|---|-----|------|--------|------|
| `OpName_0` | float16 | float16 | float16 | float16 | 同 dtype |
| `OpName_1` | float32 | float32 | float32 | float32 | 同 dtype |
| `OpName_2` | bfloat16 | bfloat16 | bfloat16 | bfloat16 | 同 dtype |
| `OpName_3` | float16 | float32 | float32 | float16 | 混合 dtype |
| `OpName_4` | bfloat16 | float32 | float32 | bfloat16 | 混合 dtype |

> ascend950 regbase kernel 经常使用混合 dtype（cos/sin 提升到 fp32 计算后 Cast 回原始 dtype）。

**关键**：ascend910b 和 ascend910_93 使用默认 config（3 条），ascend950 使用 config950（5 条）——**不同 SoC 的 binary.json 条目数和内容可能不同**。

## 6. 常见错误

| 错误 | 描述 | 严重性 |
|------|------|--------|
| **重复条目** | 把同 dtype 组合写多遍（如 _0 和 _3 都是 float16 同 dtype） | Fatal — OPC 找不到匹配 dispatch |
| **遗漏混合 dtype** | ascend950 有 config950 时只写 3 条同 dtype，遗漏索引 3、4 | Fatal — 部分 dtype 组合无 kernel |
| **dtype 名称错误** | 用 `"float"` 代替 `"float32"`，或 `"half"` 代替 `"float16"` | Fatal — OPC dtype 匹配失败 |
| **不同 SoC JSON 相同** | 910b/910_93/950 三个 JSON 文件内容完全一样 | Fatal — 950 应有 5 条而非 3 条 |

## 7. 验证方法

### 7.1 提取 binary.json dtype 组合

```bash
python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for i, entry in enumerate(data['op_list']):
    dtypes = [(inp['name'], inp['dtype']) for inp in entry['inputs']]
    dtypes += [(out['name'], out['dtype']) for out in entry['outputs']]
    print(f'Entry {i}: {dtypes}')
" op_host/config/ascend910b/xxx_binary.json
```

### 7.2 提取 _def.cpp DataType 列表

```bash
grep 'DataType' op_host/xxx_def.cpp
```

### 7.3 对比验证

binary.json 条目数 = DataType 列表长度（最大长度），每个条目的 dtype 组合必须与 _def.cpp 按索引一一对应。

## 8. 目录结构参考

```
op_host/config/
├── ascend910_93/
│   └── {op}_binary.json          # 910_93 的 dtype 组合（通常与 910b 相同）
├── ascend910b/
│   └── {op}_binary.json          # 910b 的 dtype 组合
└── ascend950/
    └── {op}_binary.json          # 950 的 dtype 组合（可能有混合 dtype）
```

每个 SoC 目录下还有 `_simplified_key.ini`，格式为：

```ini
[ClassName]
default=0
```

> `ClassName` 必须与 `OP_ADD(ClassName)` 一致（PascalCase）。