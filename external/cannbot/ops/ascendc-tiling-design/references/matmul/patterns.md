# MatMul 类算子场景路由

> 本文档仅用于 MatMul 类算子的**平台判定**与**开发路径路由**。确定平台与路径后，请进入对应文档继续设计；本文档不展开具体 tiling 方案、后融合实现细节或 Blaze 组件设计。

---

## 平台路由

| 目标平台 | NpuArch | 开发路径 | 查阅入口 |
|---------|---------|----------|----------|
| Atlas A2 / A3 | `DAV_2201` | Ascend C 高阶 API | `ascendc-api-matmul-tiling.md` / `ascendc-api-gmm-tiling.md` |
| Ascend 950 | `DAV_3510` | Blaze / tensor_api | `/ascendc-blaze-best-practice` → `references/tiling/tiling-selection.md` |

---

## A2 / A3 路径（DAV_2201）

A2 / A3 平台上的 MatMul 类算子采用 Ascend C 高阶 API 路径：

- 通用 MatMul / BatchMatMul：查阅 `ascendc-api-matmul-tiling.md`
- GroupedMatmul：查阅 `ascendc-api-gmm-tiling.md`

除上述场景外，其他 MatMul 形态建议结合 Ascend C 官方文档与示例进一步确认。

---

## A5 路径（DAV_3510）

Ascend 950 / DAV_3510 平台上的 MatMul 类算子统一采用 Blaze / tensor_api 路径，**不使用** Ascend C `MatmulImpl` / `MatmulApiTiling`。

- **正确路径**：`/ascendc-blaze-best-practice` → `references/tiling/tiling-selection.md`
- **禁止路径**：`ascendc-api-matmul-tiling.md`、`ascendc-api-gmm-tiling.md`
