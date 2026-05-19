# RegBase API 索引

该子树是 RegBase 直接调用工作的 API 入口，用于组织 API 家族、VF 边界、数据搬运和同步。这里不是 SDK 全量手册；它用于缩小检查范围，然后回到 SDK header、官方文档或真实参考实现确认最终签名。

## 使用规则

- 当问题主要是 API family、VF 边界、数据搬运或同步时，先读本索引。
- 针对一个子问题，先按 `purpose`、`read_when`、`keywords` 选最小文档集合，最多展开 5 篇叶子文档。
- 如果 5 篇 API 文档仍不能关闭问题，把问题拆成 API、数据流、精度或同步子问题；不要一次加载整个子树。
- API 名、模板参数、枚举名、namespace 和代码标识保持英文原样；中文只解释语义。

## 路由表

| 文件 | 用途 | 何时阅读 | 不适用场景 | 关键词 |
|---|---|---|---|---|
| [RegBase API 白名单](./regbase_api_whitelist.md) | 验证 API 是否属于可用家族，以及普通 kernel API 与 VF-safe API 的边界 | 编码或审查前必须确认某个 API family 是否允许 | 性能调优 | whitelist, VF, Reg, MicroAPI |
| [RegBase API 参考](./regbase_api_reference.md) | API 可用性缩小后，提供核心类型、作用域、代表 family 和数据搬运 family 的签名地图 | 需要做 family-level 调用提醒 | 替代 SDK header 做穷尽签名验证 | signatures, RegTensor, LoadDist |
| [MemBase 与 RegBase 计算 API 对照](./compute_api_membase_vs_regbase.md) | 对比 `LocalTensor` compute 与 `RegTensor` compute 的对象、入口和 API 形态 | 迁移 MemBase 计算链或审查 API 层级混用 | 只判断 GM/UB copy 对齐 | MemBase, LocalTensor, RegTensor |
| [RegBase API 同步](./regbase_api_sync.md) | 说明 VF 内部 `LocalMemBar` 的声明、`MemType` 组合和真实使用边界 | 问题涉及 VF 内同步、本地 scratch 复用或错误 barrier | 顶层方案路线选择 | sync, LocalMemBar |
| [RegBase DataCopy 分层](./datacopy_best_practices.md) | 区分 GM/UB staging 与 UB/Register VF 搬运 | 设计 `CopyIn -> Compute -> CopyOut` 或审查 `DataCopyPad` / `Reg::LoadAlign` 层级 | 纯 MemBase 算子实现 | DataCopyPad, LoadAlign, StoreAlign |
| [Pipeline 与 Buffer](./pipeline_and_buffer.md) | 解释 `TQue` / `TBuf` / queue handoff 作为兼容材料时的含义和限制 | 需要比较 queue-based fallback 或 UB pipeline | 默认 RegBase shell 语义 | pipeline, TQue, buffer |
| [算术与归约](./arithmetic_and_reduce.md) | 选择 scalar、broadcast、reduce、compare 等 API family，并记录 repeat/alignment 限制 | 数学已经明确，但 helper family 不清楚 | 构建或打包问题 | arithmetic, reduce, repeat |

## 常见路径

- API 可用性、VF-function 边界、签名提醒：[[regbase_api_whitelist]] + [[regbase_api_reference]]
- MemBase 与 RegBase 计算 API 对照：[[compute_api_membase_vs_regbase]]
- GM/UB staging 与 UB/Register 搬运分层：[[datacopy_best_practices]]
- buffering 和 queue 兼容说明：[[pipeline_and_buffer]] + [[regbase_api_sync]]
- scalar、broadcast、reduce、compare 约束：[[arithmetic_and_reduce]]
- 精度和 cast 策略：[[../pitfalls/precision_guide]]；Cast / Truncate 签名：[[regbase_api_reference]]

## 相关文档

- [[../regbase_development_guide]]
- [[../pitfalls/index]]
- [[../dev-experience/index]]
- [[../reference-ops/open_source_operator_table]]
