# Sort / Select 族 Tiling 算法路由


> Sort / Select 族（TopK、Sort、ArgSort、KthValue）Tiling 入口。TopK 必须按
> `outer / D / inner / rows / k / dtype / UB / coreNum` 做 case 级路由，不能只按 dtype 选择算法。

---

## 1. 已注册算法

| 优先级 | 算法 | 选择条件 | 入口 |
|--------|------|---------|------|
| 100 | DAV_3510 TopK 多模板复现 | Ascend 950、输出有序、`1 <= k <= min(D, 2048)` | [topk.md](topk.md) |
| 0（兜底） | 通用 M-way Merge Sort | Sort 或 TopK 专用路径不适用 | [fallback/](fallback/) |

> 命中 DAV_3510 TopK 后必须完整读取 `topk.md`。该文档定义一级路由顺序、common 内部模板
> 顺序、UB/workspace 预算、launch 契约和回归门禁；只看本表无法复现性能。

TopK Host Tiling 的完整复原对象由两部分组成：

- [topk_host_tiling.md](topk_host_tiling.md)：CANN 接线、精确 ABI、SDK 查询点与 C++ 复原步骤；
- [script/topk_tiling.py](script/topk_tiling.py)：可执行的 case 路由、切分、UB 和 workspace 参考模型。

两者必须一起使用。Python 输出中的 `sdk_queries` 必须在目标 CANN 中完成，不能用探索估计生成生产代码。

---

## 2. 复现原则

1. 保留 `small-axis -> fallback/transpose-fallback -> common` 的一级优先级。
2. 保留 common 内 `non-last mode 8 -> small merge -> single block -> fp32 merge -> single core -> optimized multi-core -> general radix -> SortAndTopK` 的优先级。
3. Host tiling 按复原规格与 Python 模型生成；TilingData、kernel 和 launch 必须整体核对，禁止只拷 kernel
   后重写简化 tiling。
4. 阈值只能在相邻边界全量 A/B 后调整；不能从单个 case 外推。

---

## 3. 贡献新算法

继续放在 `sort/` 下并更新本文件；TopK 不新建算子族目录。新算法必须提供可判定门控、UB 预算、
正确性约束、失败回退和逐 case A/B 数据。
