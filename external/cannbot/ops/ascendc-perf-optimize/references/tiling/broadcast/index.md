# Broadcast 族 Tiling 算法路由

> Broadcast 族（Add, Mul, Sub 等广播类算子）Tiling 算法入口。默认使用 fallback 兜底算法。

---

## 1. 已注册算法

| 优先级 | 算法 | 选择条件 | 目录 |
|--------|------|---------|------|
| 0（兜底） | 官方参考 | 无条件（回退保障） | [fallback/](fallback/) |

---

## 2. 贡献新算法

在 `broadcast/` 下创建新目录，在此文件的路由表中注册。
