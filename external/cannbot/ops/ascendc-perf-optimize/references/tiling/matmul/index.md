# MatMul 族 Tiling 算法路由

> MatMul 族（matmul / matmul_a16w16 / matmul_mxfp4 / batch_matmul / group_matmul）Tiling 算法入口。
>
> 先按目标代际路由。`arch22`（A2/A3）读取目标专用知识；`arch35`（A5）才进入本目录的 SWAT / FullLoad / StreamK fallback。

---

## 1. 路由规则

```
给定：目标 SoC/架构 + 算子类型 + Shape + dtype

先匹配目标代际：
  ├─ arch22（DAV_2201 / 220x）→ arch22 MatMul 指南
  ├─ arch35（DAV_3510）       → 遍历已注册算法
  └─ 未知目标                 → 返回 architecture_reference_gap

arch35 内按优先级匹配：
  ├─ 命中扩展算法 → 路由到该算法
  └─ 未命中       → 回退到 fallback 兜底算法
```

---

## 2. 已注册算法

| 优先级 | 算法/知识入口 | 选择条件 | 目录 |
|--------|------|---------|------|
| 100 | arch22 MatMul 目标模型 | DAV_2201 / 220x；A2/A3 | [A2/A3 MatMul 指南](../../../../ascendc-performance-best-practices/references/matmul/arch22/guide.md) |
| 50 | 实测求解工具包 | 需要扫描候选配置并由真机实测裁决时（非纯理论建模） | [solver.md](solver.md) |
| 0（兜底） | arch35 官方参考 | DAV_3510 且变体/能力匹配 | [fallback/](fallback/) |

arch22 路线继续读取 [流水、预取与大包搬运](../../../../ascendc-performance-best-practices/references/matmul/arch22/pipeline_design.md) 和 [复用、切分与缓存](../../../../ascendc-performance-best-practices/references/matmul/arch22/reuse_and_split_design.md)。A2 样例参数和性能不能直接写成 A3 结论；A3 必须重新代入核数、L0/L1/L2、事件/API 能力并逐 case 验证。

`fallback/` 的 SWAT / FullLoad / StreamK、MX scale 和固定阈值是 arch35 资料，A2/A3 不得把它作为默认兜底。

---

## 3. 贡献新算法

在 `matmul/` 下创建新目录，在此文件的路由表中注册选择条件和优先级：

```
matmul/
├── index.md              # 本文件（算法路由）
├── fallback/             # arch35 官方参考算法
│   ├── fallback-algorithm.md
│   ├── tiling-flow.md
│   ├── tiling-fields.md
│   └── tiling-variants.md
├── solver.md             # 实测求解工具包入口（候选扫描 + 真机实测裁决）
├── solver/               # 自包含求解器（scripts/ + references/ + 运行时生成的 profiles/）
└── <算法名>/             # 社区贡献
    └── algorithm.md
```

选择条件必须先包含适用架构，再组合 dtype、shape 范围、核数等可判定字段。
