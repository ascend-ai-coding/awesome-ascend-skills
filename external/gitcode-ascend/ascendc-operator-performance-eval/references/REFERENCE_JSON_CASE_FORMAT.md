# 性能用例 JSONL 完整格式

> 与同级 **`SKILL.md`** 正文章节 **「性能用例 JSONL 完整规范」** 保持一致；任一处更新时请同步另一处。

本文件定义 **ascend-kernel** 算子 `csrc/ops/<op>/test/<op>_perf_cases.jsonl` 中每条用例的 JSON 结构。基准代码实现 **`load_cases` / `build_inputs`** 时须与本规范一致；**用例文件仅使用 `.jsonl`**，不维护并行的 `.json` 数组文件。

---

## 1. 文件形态

| 形态 | 说明 |
|------|------|
| **JSONL** | 每行 **一个** JSON 对象，行尾换行；空行忽略。扩展名 **`.jsonl`**。 |

脚本侧 **`load_cases` 仅接受 `.jsonl` 路径**（扩展名不符则报错）。

---

## 2. 单条用例顶层结构

每个用例对象 **必须** 包含键 **`"inputs"`**，值为 **数组**。

**Layer Norm 示例**（其它算子替换字段，结构相同）：

```json
{
  "inputs": [
    { "name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [8, 128] },
    { "name": "normalized_shape", "type": "attr", "required": true, "dtype": "int", "value": [128] },
    { "name": "use_affine", "type": "attr", "required": false, "dtype": "bool", "value": true },
    { "name": "eps", "type": "attr", "required": false, "dtype": "float", "value": 1e-05 }
  ]
}
```

- `normalized_shape` 的 `value` 为 **整数列表**（与精度测试 JSONL 约定一致，`dtype` 字段写 `"int"` 表示「形状列表」类属性，由脚本解析）。
- **`inputs` 内 `name` 在同一用例内唯一**。

---

## 3. 输入项：`type` 与字段

### 3.1 `type: "tensor"`（浮点 / 一般张量）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 参数名，与脚本中读取键一致。 |
| `type` | string | 是 | 固定 `"tensor"`。 |
| `required` | boolean | 是 | 性能用例一般为 `true`。 |
| `dtype` | string | 是 | 见 §5。 |
| `shape` | number[] | 是 | 各维为正整数。 |

### 3.2 `type: "tensor"`（整数张量）

若 `dtype` 为 **`int32`** 或 **`int64`**，**必须** 增加 **`range`**：`[low, high]`，供 `randint` 使用。

### 3.3 `type: "tensor_list"`

| 字段 | 类型 | 必填 |
|------|------|------|
| `shapes` | number[][] | 是 |
| `dim` | number | 是 |

### 3.4 `type: "attr"`

| 字段 | 类型 | 必填 |
|------|------|------|
| `dtype` | string | 是（`float` / `int` / `bool` / `str` 等） |
| `value` | 任意 | 按算子约定 |

---

## 4. 属性 `dtype`（`type: "attr"` 时）

| `dtype` | `value` JSON 类型 |
|---------|-------------------|
| `float` | number |
| `int` | number 或 **整数列表**（如 `normalized_shape`） |
| `bool` | boolean |
| `str` | string |

---

## 5. 张量 `dtype` 常用取值

`float32` / `float16` / `bfloat16` / `int32` / `int64` / `bool` 等，以算子实现为准。报告按 **主张量**（如 `x`）的 `dtype` 分节。

---

## 6. 性能用例设计要点

1. **dtype 覆盖**：覆盖算子支持的主要类型，便于 Markdown 分节。
2. **形状**：典型 LLM / 视觉形状 + **非对齐维**。
3. **与精度测试对齐**：可与 `tests/cases/` 或同算子精度 JSONL 对照。
4. **命名**：仅 **`<op>_perf_cases.jsonl`**，与算子目录名一致；**不生成** `<op>_perf_cases.json`。

---

## 7. 完整示例（JSONL 两行，Layer Norm）

```json
{"inputs":[{"name":"x","type":"tensor","required":true,"dtype":"float16","shape":[2,128]},{"name":"normalized_shape","type":"attr","required":true,"dtype":"int","value":[128]},{"name":"use_affine","type":"attr","required":false,"dtype":"bool","value":true},{"name":"eps","type":"attr","required":false,"dtype":"float","value":1e-05}]}
{"inputs":[{"name":"x","type":"tensor","required":true,"dtype":"float16","shape":[4,256]},{"name":"normalized_shape","type":"attr","required":true,"dtype":"int","value":[256]},{"name":"use_affine","type":"attr","required":false,"dtype":"bool","value":false},{"name":"eps","type":"attr","required":false,"dtype":"float","value":1e-05}]}
```
