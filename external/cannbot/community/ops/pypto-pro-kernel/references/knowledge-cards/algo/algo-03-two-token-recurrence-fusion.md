---
type: "PyPTO Performance Optimization Card"
title: "两步线性递推合并：每两个 token 共享状态读写"
description: "状态按 S_t = S_{t-1} A_t + u_t v_t^T 递推、输出取更新后状态的线性投影时，把相邻两步的更新代数合并，用同一遍状态载入同时算两个 token 的投影，用同一遍状态写出覆盖两步更新。"
status: "stable"
tags: ["pypto-pro", "algorithm", "recurrence", "state-traffic"]
item_id: "algo-03"
bound_hint: "VEC"
applicability: "逐 token 的状态递推为线性映射加秩 1 更新（A_t 为对角阵），输出是更新后状态的线性投影；第二个 token 的更新系数只通过旧状态的标量/低维投影依赖前一步，因此可在不展开中间状态的情况下与第一步合并；相邻 token 可配对，状态形状规则且可按行/分块处理"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖已核验的 vf.* 逐元素运算、reduce_sum、广播与 store；配对会改变舍入顺序，须完整精度复验"
---

# 技术卡片 algo-03：两步线性递推合并（每两个 token 共享状态读写）

- **适用 bound**：VEC（状态读写/指令吞吐）
- **一句话**：把“每个 token 完整走一遍状态读改写”改成“每两个 token 共享一轮状态读改写”，用代数合并保持输出使用更新后状态的语义。

## 何时用（诊断特征）

- 状态递推可写成 `S_t = S_{t-1} A_t + u_t v_t^T`，其中 `A_t` 为对角阵，`u_t`、`v_t` 为向量。
- 输出是更新后状态的线性投影：`o_t = (S_t x_t) * scale`，或少量同类投影。
- 第二步的更新系数 `u_{t+1}` 只通过 `S_t` 的标量/低维投影依赖第一步，可写成旧状态的投影加低秩修正。
- 每个 token 的 state 访问是两遍（每遍可含多次载入/写出），且状态读写与归约在 trace 中占主要部分。
- token 数固定且可配对；状态形状规则，`D` 可按行或分块处理。

## 何时不适用

- 递推含跨 token 的非线性依赖（如 token 间 softmax、归一化或门控耦合），不能把两步更新代数合并。
- 要求与逐 token 实现完全 bitwise 一致；合并会改变乘积与求和的舍入顺序。
- 第二步的更新系数需要完整的中间状态，无法化为旧状态的标量/低维投影。
- 寄存器/片上 live set 放不下配对后的投影与修正量；或状态形状不规则、无法按行/分块处理。
- 单 token 计算量很小、状态访问不是瓶颈；合并增加的交叉标量与归约反而更贵。

## 原理

设 `S = S_{t-1}`，`A0=A_t`、`A1=A_{t+1}`，`u0=u_t`、`u1=u_{t+1}`，`v0=v_t`、`v1=v_{t+1}`。两步状态为：

```text
S_1 = S A0 + u0 v0^T
S_2 = S A0 A1 + u0 (A1 v0)^T + u1 v1^T
```

第二步的更新系数 `u1` 依赖 `S_1`，但只需 `S_1` 的标量/低维投影。把这些投影代入 `S_1 = S A0 + u0 v0^T`，可写成旧状态 `S` 的投影加低秩修正，不必展开或写回 `S_1`。输出同理：`o0` 必须用 `S_1`，`o1` 必须用 `S_2`；两者都写成旧状态投影（`S A0` / `S A0 A1`，含累计 transition）加秩 1 修正。

因此同一遍状态载入可同时服务两个 token 的全部投影，第二遍载入只用于写出合并后的 `S_2`。state 访问从每 token 两遍，降为每两 token 两遍；单遍内的载入/写出次数不变。

## 怎么改（before / after）

以下为**结构示意**；`vf.*` 形态取自已验证 kernel，`project_*`/`cross_scalars`/`update_coeff`/`outer` 为占位符，实际分块、mask、广播方式和精度门须按当前算子核验，片段不可直接编译。

**before（每 token 两遍 state）：**

```结构示意
for t in tokens:
    # pass 1：用 S_{t-1} 做投影
    proj = project(S, token[t])
    u = update_coeff(proj, token[t])
    # pass 2：更新 state，并用更新后的 state 算输出
    for row in rows(S):
        s = load(S, row)
        s = s * A_t + outer(u, v_t)
        store(S, row, s)
        o[t] += project(s, x_t) * scale
```

**after（每两个 token 共享 state）：**

```结构示意
for pair in pairs(tokens):
    t0, t1 = pair
    # pass 1：一次载入，按行算两个 token 的基础投影
    # project_u / project_o 都必须含累计 transition：
    # pu0 用 A0*v0、pu1 用 A0*A1*v1；po0 用 A0*x0、po1 用 A0*A1*x1
    # （A 为对角阵，等价于把 A 折进向量，不增加逐行开销）
    for row in rows(S):
        s = load(S, row)
        pu0[row] = project_u(s, A0 * v0)
        pu1[row] = project_u(s, A0 * A1 * v1)
        po0[row] = project_o(s, A0 * x0)
        po1[row] = project_o(s, A0 * A1 * x1)
    # 跨 token 标量：每个只归约一次，与行无关
    # c_o0 = v0^T x0；c_o1 = v0^T (A1 x1)；c_o1b = v1^T x1（纯秩 1 标量）
    # c_u1 = v0^T (A1 v1)：第一步更新对第二步投影的修正
    c_u1, c_o0, c_o1, c_o1b = cross_scalars(t0, t1)
    # 组合：u1/o0/o1 用秩 1 修正；o0 必须用更新后的状态
    u0 = update_coeff(pu0, t0)
    u1 = update_coeff(pu1 + u0 * c_u1, t1)
    o[t0] = (po0 + u0 * c_o0) * scale
    o[t1] = (po1 + u0 * c_o1 + u1 * c_o1b) * scale
    # pass 2：一次载入 + 一次写出，覆盖两步更新
    for row in rows(S):
        s = load(S, row)
        s = s * (A0 * A1) + outer(u0, A1 * v0) + outer(u1, v1)
        store(S, row, s)
```

**关键前提：**

- `u0/u1` 的旧状态投影也必须含累计 transition：`pu0` 用 `A0*v0`（= `S A0 v0`），`pu1` 用 `A0*A1*v1`（= `S A0 A1 v1`）；漏掉 `A0` 会让 `u1` 用错第一步的衰减。
- `o[t0]` 必须使用 `S_1`（更新后），不能直接用 pass 1 的 `S`：`po0` 已含累计 transition `A0`（`project_o(s, A0*x0)` = `(S A0) x0`），再叠加 `u0*c_o0` 的秩 1 修正；`po1` 同理含 `A0*A1`。
- `u1` 的修正项 `u0*c_u1` 不能漏；否则第二个 token 的投影错误。
- 两个 token 之间的交叉投影（如 `v0` 与第二个 token 的投影方向、`x1` 的内积）应在 pass 1 内用一次归约得到，不要在每行重复归约。
- `A0*A1`、`A1*v0` 等乘积顺序会影响 FP32 舍入；须按 golden 容差复验。

## 性能与验证指标

- 指标：state 的载入/写出字节与指令数、归约次数、`aiv_vec_time`、`Task Duration`；同时确认寄存器/spill 没有回退。
- 正确性：覆盖偶数配对、奇数尾 token、`D` 不能被向量宽度整除、不同动态范围；合并改变舍入顺序，不要求 bitwise 一致。
- 测量：在目标形态固定输入与采样协议（多次预热 + 多次采样取均值、冷 L2），对本优化做 A/B 对比；具体收益随核数与 batch 变化。

## 技术限制与风险

- 精度余量会变小：低精度输出（如 FP16/BF16）对舍入敏感；若合同更严或输入动态范围更大，须重新评估。
- 交叉投影增加 D 长归约；若 D 很小或状态访问不是瓶颈，增益会被抵消。
- 配对要求 token 数为偶数；奇数尾 token 需单独走原路径。
- 逐 token 的状态顺序更新语义不能被改变；如果递推有跨 token 的归一化/门控，先证明可分解再改。
- 与相邻卡片：`mem-06` 让状态片上常驻，消除 GM 往返；本卡在片上减少 state 读写次数，二者可叠加。`algo-01` 改写收缩所在的计算单元；本卡合并递推步。`pipe-09` 预取跨迭代输入；本卡合并迭代内相邻 token。

## 参考资料

- 相关卡片：`mem-06`（状态常驻）、`algo-01`（收缩等价改写）、`pipe-09`（跨迭代预取）。
- 验证案例：`ops/pro/kimi_k3/mega_recurrent_kda_replayssm`（`DESIGN.md` §5.2.3、`op_kernel/mega_recurrent_kda_replayssm.py` 的 `rec_pair_vf`）。该算子的实测误差为 `replay_u` 2.8e-3、`ws_gated` 2.6e-3、`out` 4.0e-4（契约 5e-3）；数字仅属于该实例，未在本卡泛化。
