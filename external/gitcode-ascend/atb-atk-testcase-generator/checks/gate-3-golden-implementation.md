# Gate 3 Golden 实现

## 输入

- Gate 2 通过结果
- `ATB_<OpName>_gen.yaml`
- `generator_<op>.py`
- Golden 候选来源（data_generation/apitest/用户说明）

## 命令

```bash
# 查找 Golden 参考实现
rg "class <OpName>" "$ATB_REPO_PATH/tests/framework/python/CsvOpsTestTool/data_generation.py"
rg "golden_calc|<op_keyword>" "$ATB_REPO_PATH/tests/apitest/opstest/python/operations"
```

## 通过标准

### Golden 来源优先级

1. `data_generation.py`
2. `tests/apitest/opstest/python/operations/`
3. 用户提供算法说明

## 实现要求

- 生成 `execute_<op>.py`
- 兼容 `in_formats` 的 string/list 两种形态
- 兼容 `op_param` 的 dict/string 两种形态
- 避免依赖全局状态缓存

- [ ] 来源明确
- [ ] `execute_*.py` 生成完成
- [ ] `@register` 与 YAML `api_type` 一致
- [ ] 用户确认 Gate 3 通过

## 失败回流

- `@register` 与 `api_type` 不一致：修复命名后重试 Gate 3
- 执行器依赖全局状态：改为独立实现后重试 Gate 3
- 用户未确认：停止在 Gate 3，等待确认后再进入 Gate 4

## 与 data_generation 的关系及解耦

- **对齐优先级**：在 ATK 侧构造 CPU Golden 时，应优先与仓库 **`data_generation.py`**（或 `apitest` 下同名算子）中 **已有 ref / golden 数学路径一致**，否则易出现「CPU 与 ATB 对比失败」而算子本身可运行。
- **内联解耦**：若业务上不希望运行时 **`import data_generation`**（体积、导入副作用、环境耦合），可在 **`execute_<op>.py` 内联**与当前 YAML 覆盖范围一致的子集（例如固定 layout、maskType、quantType），保持 **numpy softmax / GQA matmul / 张量 gather** 等与 ref 等价；扩展配置时再增加分支或恢复引用。
- **PagedAttention 级别复盘**（精度失败、block_table 与 contextLen、generator ranges 陷阱）：见 [`../references/pagedattention-case-study.md`](../references/pagedattention-case-study.md)。

## ATK 输入规范化（建议在 init_by_input_data）

使 CPU Golden 与 ATB **读取同一套输入**，避免泛化 JSON 越界导致 ref 崩溃或不对称：

- **物理块 id**：对 **`blockTables`** 按可用块数 **clamp**（典型上界为 **`min(key_cache.dim0, value_cache.dim0) - 1`**）。
- **序列长度**：若 ref 按 **`block_table[j // block_size]`** 索引行块表，需保证 **`context_len` 不会使该索引越过行宽**；常见做法是限制 **`contextLens`** 并同步 **`op_param`** 中的 **`contextLens`**（详见 **`../references/common-faq.md`** Q7）。

## Generator 与 ranges

不建议在 **`generator_*.py`** 中随意改写 **`ranges.valid.values`** 等嵌套类型以规避越界（可能与 **`InputCaseConfig.__hash__`** 或边界数据生成冲突）；优先采用上文 **executor 侧规范化**。
