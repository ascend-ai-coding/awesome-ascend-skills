# PagedAttention ATK 精度案例复盘

面向 ATK Golden 与 `execute_<op>.py` 实现者；与 CSV DataGen 三件套场景不同，见 **atb-golden-developer** 技能说明。

本文记录在 ATK `accuracy` 任务下 PagedAttention 类算子从「精度不达标」到「全量通过」的典型路径与陷阱；实现细节以各仓库内 `execute_paged_attention.py`、`generator_paged_attention.py`、`run_pa.sh` 为准（路径因 fork 而异）。

## 背景与对比链

ATK 精度任务同时跑 CPU 侧 **Golden（`execute_<op>.py`）** 与 ATB OPP，再按 YAML 中 `standard.acc`（如 `single_bm`）对比输出。Golden 与 ATB **数学路径不一致**时，会出现 ATB 执行成功但 **`acc_pass_result:Failed`**、通过率 0%。

## 现象

- `atk task --task accuracy` 报告 CPU 侧不达标。
- 日志或报告中 **`accuracy` 列为 False**，或与 ATB 对比 **max diff** 异常。

## 根因一：Golden 实现与参考不一致

早期在 `execute_paged_attention.py` 中用 **手写 `F.softmax` + 简化循环** 模拟分页注意力；而 ATB/CSV 侧标杆在 `data_generation.py` 中通过 **`ref_single_query_cached_kv_attention`** 实现：按序列位置 **`j`** 使用 **`block_table[j // block_size]`** 取物理块与块内偏移，再经 **`ref_masked_attention`**（**numpy softmax** + **`group_mm_torch`** GQA 分组 matmul）。两条路径数值不一致会导致系统性失败。

**结论**：Golden 应对齐 **`data_generation` 中与 maskType/quant 条件匹配的 ref 分支**，或在内联实现中保持相同运算顺序与 softmax 方式。

## 根因二：泛化输入与 ref 前置条件不一致

即便数学路径对齐，**JSON/生成器产出的张量**仍可能违反 ref 的隐含假设。

### 物理块索引

`blockTables` 中物理块 id 须落在 **`[0, num_blocks)`**，且 **K/V cache 的 `num_blocks`（dim0）须一致可索引**；实践上对 **`blockTables` 使用 `clamp`，上界取 `min(key_cache.dim0, value_cache.dim0) - 1`**，保证 CPU Golden 与 ATB 输入同一视图。

### contextLen 与 block_table 行宽

ref 在内层循环对每个位置 **`j`** 访问 **`block_table[j // block_size]`**（每行一张 block table）。若 **`context_len` 过大**，会使 **`j // block_size`** 超出 **该行长度**（第二维），报错形态常为 **`index k is out of bounds for dimension 0 with size k`**——易被误判为 **key_cache 的 num_blocks**；实为 **block_table 行**越界。

**缓解**：在 executor 的 **`init_by_input_data`** 中将每条 **`contextLens`** 限制为不超过 **`blockTables.shape[1] * block_size`**（与 tiling 中 block_size 一致），并 **同步写回 `op_param`/`contextLens`**，避免仅改 tensor 未改字符串参数。

## 误区：在 Generator 里改 `ranges.valid.values`

为收紧 block id 范围，曾尝试设置 **`ranges.valid.values = [[0, max_block]]`**：

- **嵌套 list** 参与 **`InputCaseConfig.__hash__`** 时触发 **`TypeError: unhashable type: 'list'`**。
- 改为 **tuple** 后又与 ATK **边界数据生成**（如 **`<=` 与标量比较**）冲突，出现 **`'<=' not supported between int and list`**。

**结论**：**不要随意改写 ATK case 配置里嵌套 list 的形态**；优先在 **`execute_*.py` 规范化输入**（与上文 clamp/cap 一致）。

## 解耦：去掉运行时 `import data_generation`

`data_generation` 模块体积大、副作用多。可在 **`execute_paged_attention.py` 内联**与当前 ATK 配置一致的子集（如 ND、**maskType=0**、**quantType=0**）：**`_softmax_numpy`、GQA `group_mm_torch`、逐 token  gather + `ref_masked_attention` 等价路径**，从而 **不再运行时依赖 `atb/common/data_generation.py`**。扩展 mask/量化时再按需增加分支或恢复对齐调用。

## 验证命令与产物

- 快速：**`bash run_pa.sh debug`**（典型为生成少量用例 + 精度前若干条）。
- 全量：**`bash run_pa.sh accu`** / **`accu-all`**（视脚本定义）。
- 报告：通常在算子目录下 **`atk_output/**/report/*.xlsx`**；关注 **`acc_pass_result`**、通过率。

## 协作与环境提示

- 严格按任务描述交付；复杂任务可用 planning 三文件或宿主仓库约定的 **`working_files`** 归纳。
- CANN/ATB 日志：宿主环境常通过 **`set_env.sh`** 设置 **`ASCEND_PROCESS_LOG_PATH`**、日志级别等；与 ATK Python 侧 **`atk_output/*/log`** 分工查看。

## 相关文档

- [PagedAttention 最佳实践摘要](pagedattention-best-practice.md)
- [ATK 常见问题（FAQ）](common-faq.md)
- [Gate 3 Golden 实现](../checks/gate-3-golden-implementation.md)
