# Convolution 族 Tiling 算法路由

> Conv2D / DepthwiseConv Tiling 算法入口。按条件路由到具体算法，默认使用 fallback 兜底算法。

---

## 1. 路由规则

```
给定：算子类型 + Shape + dtype

遍历已注册算法（按优先级降序），匹配选择条件：
  ├─ 命中 → 路由到该算法
  └─ 未命中任何扩展算法 → 回退到 fallback 兜底算法
```

---

## 2. 已注册算法

| 优先级 | 算法 | 选择条件 | 目录 |
|--------|------|---------|------|
| 0（兜底） | FORMULAS（Mmode + HWmode） | 无条件（回退保障） | [fallback/](fallback/) |

---

## 3. 贡献新算法

在 `convolution/` 下创建新目录，在此文件的路由表中注册选择条件和优先级：

```
convolution/
├── index.md              # 本文件（算法路由）
├── fallback/             # 官方参考算法
│   ├── fallback-algorithm.md
│   ├── tiling-flow.md
│   ├── tiling-fields.md
│   └── demo/
└── <算法名>/             # 社区贡献
    └── algorithm.md
```

选择条件可以是 dtype、shape 范围、核数等任意可判定字段的组合。
