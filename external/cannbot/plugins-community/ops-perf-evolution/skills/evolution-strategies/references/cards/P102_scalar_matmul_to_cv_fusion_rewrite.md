---
id: P102
bottlenecks: [scalar_compute, compute_bound]
op_families: [cv_fusion, matmul, attention, special]
complexity: L1
conflicts_with: []
synergizes_with: [P29, P46, P87, P92]
requires: []
has_preconditions: true
has_playbook: false
quantified_gain:
  - shape: "GDR B=1,T=64,Hk=4,Hv=4,fp16"
    baseline_us: 3650.0
    optimized_us: 261.0
    speedup: 14.0
    source: "gated_delta_rule_cv rewrite (scalar+VEC → MIX_AIC_1_1)"
  - shape: "GDR geomean over T=64..4096"
    baseline_us: 1.0
    optimized_us: 1.0
    speedup: 8.85
    source: "gated_delta_rule_cv vs PyTorch reference, wall-clock 20 iters"
---

# P102: 标量/向量模拟矩阵运算 → MIX CV 融合重写

## 核心思想

当 kernel 用**标量循环或纯 Vector 指令模拟矩阵运算**（dot product、matmul、三角求解、链式矩阵乘加）时，P94/P95 的向量化收益会封顶——Vector 单元本就不是为矩阵运算设计的。此时唯一数量级收益的路径是**架构级重写为 MIX CV 融合**：Cube（AIC）承担全部 matmul（Mmad），Vector（AIV）承担 elementwise/decay/mask/状态更新，中间结果经 GM workspace 交接，`SyncAll<false>()` 做全局 barrier。

实证（gated_delta_rule，910B2C）：scalar/VEC 实现优化到 3.3x 后仍比参考慢 17x（scalar 做矩阵运算是原罪）→ MIX_AIC_1_1 重写后 **geomean 8.85x vs PyTorch 参考**，单 stage 对比：AIV 标量三角求解 98µs/chunk → Cube Neumann 链 11 个 Mmad（见 P103）。

## 适用判定（何时选这张卡而不是 P94/P95）

```
kernel 内层循环是以下形态之一？
  ├─ for(i) for(j) acc += A[i][k] * B[k][j]        (标量 matmul/dot)
  ├─ for(i) for(j) for(k) ...                      (嵌套标量乘加，维度 ≥ 64)
  ├─ for(i) for(j<i) solve/linsolve 递推           (三角求解、前向替换)
  └─ Vector 指令拼出的"伪 matmul"（Mul+ReduceSum 逐行）
    │
    ├─ 是 → 矩阵维度 ≥ 64×64？
    │   ├─ 是 → **选 P102（CV 融合重写）**，P94/P95 只做收尾的 elementwise 部分
    │   └─ 否（< 64）→ P94/P95 向量化即可，Cube 启动开销不划算
    └─ 否（纯 elementwise/归约）→ P94/P95
```

## 四阶段重写工作流（每阶段有明确退出标准，不可跳步）

### Phase 0 — 纯 AIC 单路径 derisk（Cube 链路先对）
- 用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)` 写最小 matmul（一个 Mmad），验证 Nd2Nz → LoadData → Mmad → Fixpipe 的 shape/stride 配置。
- **退出标准**：单 matmul 数值与 torch 参考一致。
- 常见坑：Nd2Nz `srcDValue` 是 GM 行步长（**元素**单位）；Fixpipe NZ2ND（默认 CFG_ROW_MAJOR）`dstStride` 单位是**元素**，非 NZ 模式是 32B。

### Phase 1 — MIX 骨架（只证明"MIX 不挂"）
- 切 `KERNEL_TYPE_MIX_AIC_1_1`，加核类型守卫，AIV 路径留空，结尾一次 `SyncAll<false>()`。
- **退出标准**：不 hang 且 AIC 结果与 Phase 0 完全一致。
- 挂了 → 按 `references/mix-cv-fusion-guide.md` 的 R1~R4 对照检查（几乎都是守卫缺失）。

### Phase 2 — 配对与同步探针（证明"核对应关系"）
- AIV 路径写 `GetBlockIdx()/GetBlockNum()` 到 GM，host 读回打印。
- **退出标准**：确认 pairing（MIX_AIC_1_1 下 AIV idx == AIC idx）与实际核数。此后 AIC/AIV 可用相同的 work→core 分配逻辑各算各的。

### Phase 3 — 逐 stage 增量（一次一个 stage，各自对参考）
- 每个 stage：AIC 写中间矩阵到 GM workspace → `SyncAll<false>()` → AIV 处理 → `SyncAll<false>()` → ...
- **每个 stage 落地即验证**：host 临时 return workspace 张量，与 torch 参考逐 stage 对比。
- **必须测多 chunk / 多迭代（T≥128）**：单 chunk 会掩盖状态依赖路径的 bug（GDR 实例连踩两次：decay mask inclusive vs strict、漏乘 scale，单 chunk 均恰好正确）。
- **退出标准**：全 stage 对齐后，端到端精度 PASS。

## 守卫代码骨架（可直接套用）

```cpp
class KernelX {
public:
    __aicore__ inline void Init(...) {
        if ASCEND_IS_AIC {
            pipe_.InitBuffer(a1_, 1, SZ_A1);   // A1/B1/A2/B2/CO1 — AIC 专属
            pipe_.InitBuffer(b1_, 1, SZ_B1);
            pipe_.InitBuffer(a2_, 1, SZ_A2);
            pipe_.InitBuffer(b2_, 1, SZ_B2);
            pipe_.InitBuffer(c1_, 1, SZ_C1);
        } else {
            pipe_.InitBuffer(ub_, UB_SIZE);    // VECCALC — AIV 专属
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { ProcessAIC(); } else { ProcessAIV(); }
        // 两侧 SyncAll<false>() 调用次数/顺序严格一致
    }
};
extern "C" __global__ __aicore__ void kernel_x(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);   // 函数体内
    KernelX k; k.Init(...); k.Process();
}
```

**五条铁律**（违反即 hang，无报错只有 100% AICore 空转）：
1. **AIV 永不触碰 A1/B1/A2/B2/CO1；AIC 永不触碰 UB**——包括 InitBuffer。
2. **两侧 `SyncAll<false>()` 计数一致**——每改一侧，数另一侧。无参模板默认 `isAIVOnly=true` 只同步 AIV，MIX 全局 barrier 必须显式 `<false>`。
3. **所有核循环次数一致**——`slots = ceil(work/cores)` padded loop，越界核只走 barrier 不做工（`valid` 守卫工作体）。
4. **每个 Mmad 操作数完整走队列**：Alloc→Load→EnQue→DeQue→Mmad→Free；Mmad→Fixpipe、Fixpipe→下次 GM 读之间 `PipeBarrier<PIPE_ALL>`。
5. **中间结果走 GM workspace 交换**（AIC Fixpipe 写 → barrier → AIV DataCopy 读，或反向），不要试图共享 L0/L1/UB。

API 规范、hang 根因分类（R1-R6）与四阶段开发流程详见本 skill 的 `references/mix-cv-fusion-guide.md`。

## 常见陷阱

⚠️ **改了两样东西（launch type + 参数）时，hang 不一定归因于最后改的那个**——AIC_ONLY 编译运行正常 ⇒ kernel 逻辑无问题，launch 模式是病根，不要回退参数空转
⚠️ **build 缓存**：setup.py/构建脚本检测到 .so 存在会跳过 cmake——**改代码必须 `rm -rf build`**，否则"改了没生效"造成假诊断
⚠️ **带病 NPU**：被前一次 hang kernel 污染的设备会让 cross-core barrier 失效，后续输出非确定性 NaN（位置逐 run 变化）——同 binary 换一块干净 NPU PASS ⇒ 卡问题，与代码无关
⚠️ **单 chunk 不算验证**：状态依赖路径（S 更新、跨 chunk scale）在单 chunk 下恰好正确，必须 T≥128 多 chunk 用例
⚠️ **诊断前先 torch 算法仿真**：纯 torch 复刻 kernel 的分解公式再对参考，不花编译时间就能抓公式错误

## 代码搜索关键词

```bash
# 检测标量/向量模拟矩阵运算（P102 适用信号）
grep -nE "for\s*\(.*\+\+.*\)\s*\{?\s*for\s*\(" op_kernel/*.cpp          # 嵌套标量循环
grep -nE "reinterpret_cast.*GetPhyAddr|\.GetPhyAddr\(\)" op_kernel/*.cpp  # raw pointer 标量计算
grep -nE "Mul\(.*\).*ReduceSum|ReduceSum" op_kernel/*.cpp                  # Vector 拼伪 matmul

# 检测是否已是 MIX 架构（已重写则不适用）
grep -nE "KERNEL_TYPE_MIX|ASCEND_IS_AIC|Mmad" op_kernel/*.cpp
```

## 来源

- gated_delta_rule CV 融合重写（a5_ops_slim/workspace/gated_delta_rule_cv）：scalar/VEC 3.3x 且比参考慢 17x → MIX_AIC_1_1 重写 geomean **8.85x**（T=64 时 14.0x），精度全 shape PASS（MERE < 9.766e-04）。方法论见该目录 docs/MIX_DEBUG_WORKFLOW.md
