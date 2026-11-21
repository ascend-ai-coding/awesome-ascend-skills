---
type: "PyPTO Performance Optimization Card"
title: "权重 NZ 离线预打包（GM NZ 声明 + NZ→NZ 纯搬运）"
description: "权重固定的推理场景下，把权重在离线阶段打包为 NZ 物理排布，kernel 签名把该 GM 张量声明为 pl.NZ，GM→L1 的 pl.load 变为 NZ→NZ 纯数据搬运，消除随路 ND→NZ 格式转换开销。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "nz", "layout", "mte2"]
item_id: "cube-07"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/datamove.md#13-tile_view-three-shape-layers-and-a-shared-cursor"
      scope: "离线 NZ 打包后分别核对逻辑形状、物理容量和布局视图，GM 声明仅解释已存在的物理数据。"
      boundary: "视图契约不完成 ND→NZ 打包，不证明 NZ 转置可用，也不替代调用方接口变更、padding 清零和打包 value test。"
bound_hint: "MTE2"
applicability: "推理场景权重固定、可离线预处理；matmul 为 MTE2 bound 且权重搬运占比高；原始内轴不对齐时随路转换代价更高、收益更明显"
target_api_gate: "仅限 Ascend 950PR 或 950DT；GM Tensor 仅支持 ND/NZ 两种声明，NZ 要求调用方已按 NZ 物理排布 packing 且按对齐后容量分配；NZ 搬运不支持降序 order 转置"
---
# 技术卡片 cube-07：权重 NZ 离线预打包（GM NZ 声明 + NZ→NZ 纯搬运）

- **适用 bound**：MTE2（权重搬运主导，随路格式转换损耗带宽）
- **一句话**：权重离线打包成 NZ，kernel 按 NZ 声明，load 退化为纯搬运。

## 何时用（诊断特征）

- 推理场景，权重固定、可离线预处理；训练（权重每步更新）禁用。
- MTE2 bound 且权重搬运字节占比高；权重内轴非对齐时随路转换开销更大。
- 调用方（框架/上层图）能接受输入格式约定变化（NZ packing 外溢到接口）。

## 何时不适用

- 权重会更新或调用方只能传 ND：离线转换不可行。
- 权重小、搬运占比低：收益被接口复杂度吞掉。
- 需要转置搬运的 operand：NZ 声明的 GM 张量不支持降序 order 转置载入。

## 原理

GM 声明为 ND 时，`pl.load` 到 L1 的 NZ tile 需要随路完成 ND→NZ 格式转换；权重离线打包为 NZ 物理排布后，kernel 把该参数声明为 `pl.NZ`，`pl.load` 成为 NZ→NZ 纯数据搬运，带宽利用率更高。NZ 把逻辑 `[..., M, N]` 存为 `[..., ceil(N/C0), ceil(M/16), 16, C0]`（C0 随 dtype 变化），存储容量按对齐后尺寸分配；NZ 声明只解释内存排布，框架不做自动转换或扩容。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的 matmul kernel 对与离线打包函数（fp16，C0=16）；离线打包函数的正确性（块间序、块内序、补齐清零）须单独验证。

```python
import torch
import pypto_pro.language as pl

# host 侧（离线一次性，非 kernel 包装职责）：[K, N] ND -> NZ 物理 [ceil(N/C0), ceil(K/16), 16, C0]
def pack_nz(b_nd):
    K_, N_ = b_nd.shape
    Ka = (K_ + 15) // 16 * 16
    Na = (N_ + C0 - 1) // C0 * C0
    padded = torch.zeros(Ka, Na, dtype=b_nd.dtype, device=b_nd.device)
    padded[:K_, :N_] = b_nd
    return padded.view(Ka // 16, 16, Na // C0, C0).permute(2, 0, 1, 3).contiguous()

# before：权重声明为 ND（默认），pl.load 到 L1 随路完成 ND→NZ 转换
# def kernel(a: pl.Tensor[[M, K], pl.DT_FP16],
#            b: pl.Tensor[[K, N], pl.DT_FP16], ...):
#     pl.load(b_mat, b, [0, nt * NT])        # ND→NZ 随路转换

# after：权重声明为 NZ，pl.load 退化为 NZ→NZ 纯搬运
@pl.jit(auto_mutex=True)
def kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
           b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
           out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    with pl.section_cube():
        # cid = pl.get_block_idx(); ncore = pl.get_block_num()；tile group 声明与常规 matmul 相同
        pl.load(a_mat, a, [0, 0])            # A 侧不变（ND→NZ 随路转换）
        for nt in pl.range(cid, n_tiles, ncore):
            pl.load(b_mat, b, [0, nt * NT])  # NZ→NZ：纯搬运，无转换
            pl.move(a_left, a_mat)
            pl.move(b_right, b_mat)
            pl.matmul(acc, a_left, b_right)
            pl.store(out, acc, [0, nt * NT])

# 调用：打包后的物理 buffer 以 rank-2 [K, N] 视图传入（物理字节保持 NZ 序）
# b_nz = pack_nz(b_nd)
# kernel[stream, block_dim](a, b_nz.view(K, N), out)
```

调用侧自检：打包结果按块序还原（`b_nz.permute(1, 2, 0, 3).reshape(K, N)`）须与 ND 原矩阵逐元素一致，再进 kernel 端到端精度回归。

## 性能与验证指标

观察 MTE2 段耗时与 Task Duration；搬运字节数不变，变化的是搬运效率。正确性须双重核对：离线打包的 NZ 排布（与 ND 参考逐块比对）与 kernel 端到端精度。

## 技术限制与风险

- 格式约定外溢：接口语义从"传 ND 权重"变为"传 NZ 权重"，集成方必须感知；建议接口层加格式检查或双格式分支。
- GM 分配须按对齐后容量；补齐区不属于逻辑内容，打包时须清零。
- 传入绑定按声明的逻辑 shape 校验 rank：NZ 物理 buffer 须以 rank-2 逻辑视图传入（如 `b_nz.view(K, N)`），直接传 NZ 物理多维 shape 会被绑定拒绝。
- 训练/权重更新场景禁用；打包函数本身的块序错误会表现为全量精度错误。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/basic_data_structures/TensorLayout.md`（GM NZ 声明与物理排布）
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`（NZ→NZ 搬运约束）
