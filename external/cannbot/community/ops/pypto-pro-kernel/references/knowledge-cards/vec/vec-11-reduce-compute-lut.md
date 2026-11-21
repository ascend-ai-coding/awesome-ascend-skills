---
type: "PyPTO Performance Optimization Card"
title: "减少超越函数计算量（PyPTO-Pro 能力门控）"
description: "超越函数主导时先删除重复调用或使用公开融合接口，LUT 与近似方案保持能力和精度门控。"
status: "stable"
tags: ["pypto-pro", "vec", "transcendental", "capability-gate"]
item_id: "vec-11"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/vf.md#23-precision-reaches-ir-but-results-are-bitwise-identical"
      scope: "选择超越函数近似或 precision 候选前，核对参数是否改变当前后端的实际行为。"
      boundary: "IR 中存在 precision 不等于性能或数值发生变化；本关联不宣称存在 LUT、expandLevel 或未公开的 Python 参数。"
bound_hint: "VEC"
applicability: "超越函数主导热点，且存在以下机会之一：重复调用复用、公开融合接口或有当前 API、编译产物和精度证据的 LUT/近似方案"
target_api_gate: "仅限 Ascend 950PR 或 950DT；采用目标版本支持的公开超越函数或融合接口，所需 precision/LUT 能力未明确支持时记录 capability gap"
---
# 技术卡片 vec-11：减少超越函数计算量（PyPTO-Pro 能力门控）

- **适用 bound**：VEC（深度）
- **一句话**：超越函数主导时先消除重复调用、使用公开融合接口；LUT/插值只能在当前 PyPTO-Pro 源码或编译产物明确支持时启用。

## 何时用（诊断特征）

- `aiv_vec_ratio` 很高，exp/log/tanh 等超越函数是主要 VECTOR 事件。历史经验筛选线是 `aiv_vec_ratio >90%` 且超越函数事件占 VECTOR `>30%`；它们是候选阈值，不是 PyPTO-Pro 的绝对达标线。
- TileGroup、搬运、Cast、融合指令等调整已无明显收益，瓶颈在有效计算量。
- 同一输入重复求超越函数，或算法可用公开融合操作（如 softmax 的 `vf.exp_sub`）替代多条操作。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 超越函数不是主要 VECTOR 事件，或 Cast、搬运、融合和流水瓶颈尚未排除。
- 方案需要当前公开 PyPTO-Pro 不存在的 expandLevel、LUT 或近似参数。
- 减少调用或近似会破坏完整精度回归，或只能引用未在当前 case 复现的 30% 经验线。

## 原理

AscendC 的 `Exp<T, expandLevel, ...>` 支持 LUT/插值选择；不能据此推断 PyPTO-Pro 有等价参数。本卡使用的公开 `vf.exp(src, preg)` / `vf.exp_sub(src, max_val, preg)` 调用不含 `expandLevel`。优化原则是减少超越函数调用次数、跨后续计算复用结果、使用公开融合指令；任何 precision/LUT 能力都必须以目标版本的 API 文档和编译结果为准，未明确支持时不得采用。

## 怎么改（before / after）

以下 exp_sub Python 为嵌入片段；LUT 或近似路径仅为能力门控伪码。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def stable_exp_vf(src_tile, max_tile, out_tile, valid: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    x = vf.load_align(src_tile, 0)
    max_reg = vf.load_align(max_tile, 0, dist=pl.LoadDist.BRC_B32)

    # before 形态：shifted = vf.sub(x, max_reg, preg); out = vf.exp(shifted, preg)
    # after：使用公开融合接口；同一 out 后续复用，不重复 exp
    out = vf.exp_sub(x, max_reg, preg)
    vf.store_align(out_tile, out, preg)
```

若确需 LUT：先在当前 PyPTO-Pro checkout 的 `python/pypto_pro/language/_vf_api.py` 与对应 API 文档搜索目标接口；没有公开参数就把方案标为 capability gap/待实现，不写伪 API、不改生成的 CCE C++ 绕过 PyPTO-Pro。

## 性能与验证指标

比较 `Task Duration(us)` 与超越函数事件数，按复用、融合或近似路径核对计算量变化。**待实测**：“超越函数 >30% 时可能收益明显”只是待复现的历史假设。

## 技术限制与风险

- 不在 PyPTO-Pro 中写 `expandLevel`、AscendC `Exp<T,...>` 或臆造 `vf.exp(..., lut=True)`。
- 任何近似、LUT、减少 Taylor 项都属精度换性能，必须通过全量精度回归。
- 优先公开 `vf.*` 能力；源码未支持时明确记录能力缺口。

## 参考资料

- 方法边界：从算法层减少计算量；AscendC LUT 调用不能直接用作 PyPTO-Pro API。
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/Reg-API/basic_arithmetic/exp.md`；`Vf.exp_sub` 定义于 `_vf_api.py`。
