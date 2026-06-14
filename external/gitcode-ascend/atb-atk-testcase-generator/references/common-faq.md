# 常见问题 FAQ

## Q1: `AttributeError: 'list' object has no attribute 'split'`

`in_formats` 被解析成 list。需要在 `execute_*.py` 中同时兼容 string 与 list。

## Q2: `mat1 and mat2 shapes cannot be multiplied`

通常是 cache shape 未对齐。优先检查 generator 是否强制了所有相关维度。

## Q3: 性能测试 OOM（exit 137）

优先使用单进程模式（如 `-sp`），并降低并发压力。

## Q4: CSV 用例被跳过

检查 `SocVersion` 是否与当前设备匹配。

## Q5: 任务跑了但感觉没有走 ATB / 没验证到 ATB 路径

检查 `node.yaml` 与 `node_perf.yaml`：ATK 只有声明为 `backend: atb` 的节点才会进入 ATB 侧精度与性能逻辑。若仅配置了 `backend: cpu` 等，不会覆盖 ATB。须至少各保留一条 `backend: atb` 的节点（可与 CPU 对照节点共存）。详见 `checks/gate-2-yaml-generator-design.md` 中「node.yaml / node_perf.yaml 后端约束」。

## Q6: `TypeError: unhashable type: 'list'`（case 生成阶段）或改写 ranges 后边界生成报错

部分 ATK 版本对 **`InputCaseConfig`** 做哈希；若在 generator 里把 **`ranges.valid.values`** 设为 **`[[min, max]]`** 等嵌套 list，可能触发不可哈希错误。改为 tuple 等形式又可能与 **`gen_boundary_tensor_data`** 等路径不兼容（如 **`<=` 在 int 与 list 之间**）。

**建议**：不要随意改写 case 配置里嵌套结构的类型；对 **block id / contextLen** 等与 ref 一致性相关的约束，优先在 **`execute_<op>.py` 的 `init_by_input_data`** 中与 ATB 对齐后再跑 Golden。

## Q7: `index k is out of bounds for dimension 0 with size k`（PagedAttention / 分页类 Golden）

既可能是 **key_cache 物理块维**越界，也可能是 **`blockTables` 单行长度不足**：ref 按位置 **`j`** 访问 **`block_table[j // block_size]`**，若 **`context_len` 过大**会先踩 **block_table 行宽**。请核对 **`context_len` 与 `blockTables` 第二维、`block_size` 的关系**（典型上限：**`blockTables.shape[1] * block_size`**），并保证 **`op_param` 内 contextLens** 与 tensor 一致。

详见 **`references/pagedattention-case-study.md`**。

## Q8: Golden 是否要 `import data_generation`？

**对齐路径**：优先与 **`data_generation`**（或 apitest）中同名 ref **同一数学路径**，精度才可与 CSV/ATB 对齐。

**解耦路径**：若仅需固定子集（如 ND、maskType=0、quantType=0），可将等价逻辑 **内联到 `execute_<op>.py`**，避免运行时依赖大体量模块与副作用；扩展场景再加分支或恢复导入。与 **`checks/gate-3-golden-implementation.md`** 中「与 data_generation 的关系」一节一致。

## Q9: `performance_device` 下 `NPUBackend.custom_data_computation` 报 `TypeError` / `KeyError`

- **先看栈**：出现 **`npu_backend`** **不等于**走错 Backend（**`AtbBackend` 继承 `NPUBackend`**）。详见 **atb-debug-guide** [`atk-atb-perf-backend-stack.md`](../../atb-debug-guide/references/atk-atb-perf-backend-stack.md)。
- **`perf` 常为 `PerformanceConfig`**（Pydantic）；Profiling 不完整时 **`custom_data` / `cube_computation` / `api_statistic` / `power_temp_info`** 等可能为 **`None`**，合并逻辑下标会得到 **`TypeError`**；**`power_temp_info`** 若为空 **`{}`** 又可能缺 **`power`** 触发 **`KeyError`**。
- **缓解**：在 ATK 插件基类（业务仓常见命名如 **`atb_base_api.py`**）中对 **`NPUBackend.custom_data_computation`** 做入口包装，在进入原生实现前为 **`PerformanceConfig`** 填写与类型注解一致的占位值（具体以本仓库实现为准）。
- **调试**：设置 **`ATK_PERF_DEBUG=1`**（或 **`true`**/**`yes`**），在 **`atk.log`** 搜索 **`[ATK_PERF_DEBUG]`**。
