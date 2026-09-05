# Ascend CV 融合算子生成方法论 — CV 融合篇

> 基于 Sparse Flash Attention (SFA) 等 CV 融合算子优化实践总结
> 适用范围：所有需要 Cube-Vector 异构核间数据流水线的 Triton-Ascend 算子

---

## 1. 核心思想：数据流驱动的模块化拆分

CV 融合算子的本质是 **Cube-Vector 异构核心间的数据流水线**。生成方法论的第一步不是写代码，而是画出数据依赖图：

- **Cube 生产者** → **Vector 消费者**：`al.fixpipe`（L0C → UB）
- **Vector 生产者** → **Cube 消费者**：`al.copy`（UB → L1，需 NZ 格式转换）
- **全局内存回退**：任何一方都不走 GM 时，才是真正的融合

### 1.1 拆分原则：按生产者-消费者边界切分

以 FlashAttention 类算子为例，天然的数据流边界为：

```
[Vector] Load KV  →  [Cube] QK Matmul  →  [Vector] Softmax
     ↓                                               ↓
   GM→L1                                         UB→L1
     ↓                                               ↓
[Cube] PV Matmul  ←  [Vector] Accum Output  ←  P Copy
```

**每个箭头就是一个独立的 CV 通道优化点。**

对任意 CV 融合算子，先回答：

1. 哪些计算放在 Cube？
2. 哪些计算放在 Vector？
3. 中间结果通过哪个 On-Chip buffer 握手？
4. 有没有任何阶段被迫回退到 GM？

### 1.2 独立验证：先分后合

把融合过程拆成若干阶段，每阶段单独验证精度：

| 阶段 | 验证方式 | 输出目标 |
|------|---------|---------|
| Phase 1 (Vector Load) | 加载后 `tl.store` 到 GM Buffer | `Load_Buffer` |
| Phase 2 (Cube Compute1) | 读取 Load_Buffer，计算，`tl.store` 到 GM | `C1_Buffer` |
| Phase 3 (Vector Process) | 读取 C1_Buffer，后处理，`tl.store` 到 GM | `V2_Buffer` |
| Phase 4 (Cube Compute2) | 读取 V2_Buffer + Load_Buffer，计算，`tl.store` 到 GM | `C2_Buffer` |
| Phase 5 (Vector Final) | 读取 C2_Buffer，在线累加/输出 | `Out`, `LSE` |

**只有当前阶段精度验证通过（diff < 1e-1）后，才进入下一阶段。** 这是避免问题累积的关键。

---

## 2. CV 通道优化：从 GM 回退到 On-Chip

### 2.1 Cube → Vector：`al.fixpipe` + `ROW_SPLIT`

**场景**：Cube 计算完结果（L0C）→ Vector 需要读取进行后处理（UB）

```python
# Cube scope
result_full = tl.dot(a_full, tl.trans(b_load))  # [BLOCK_M, BLOCK_N] in L0C
al.fixpipe(result_full, result_ub,
           al.FixpipeDMAMode.NZ2ND,
           al.FixpipeDualDstMode.ROW_SPLIT)

# Vector scope
result_sub = bl.to_tensor(result_ub)  # 每个 sub-core 自动拿到 HALF_BLOCK_M 行
```

**关键要点**：
- `ROW_SPLIT` 自动将 `BLOCK_M` 行均分给两个 Vector sub-core
- UB buffer 只需分配 `[HALF_BLOCK_M, BLOCK_N]`，而非 `[BLOCK_M, BLOCK_N]`
- 必须通过 `bl.to_tensor(result_ub)` 读取，不能当作普通 tensor 直接访问

### 2.2 Vector → Cube：`al.copy` + NZ 格式 + `bl.subview`

**场景**：Vector 计算完结果（UB）→ Cube 需要读取进行下一步 Matmul（L1）

```python
# 1. UB ND 格式 → NZ 格式转换（3D permute 风格）
r_cast = r_sub.to(tl.bfloat16)                          # [HALF_BLOCK_M, BLOCK_N]
r_nz_tmp = tl.permute(
    r_cast.reshape(HALF_BLOCK_M, BLOCK_N // 16, 16),    # [M, W//16, 16]
    (1, 0, 2)                                           # → [W//16, M, 16]
)
r_nz = r_nz_tmp.reshape(
    BLOCK_N // 16, HALF_BLOCK_M // 16, 16, 16           # → [W//16, M//16, 16, 16]
)

# 2. 通过 subview 按 sub_vec_id 写入 L1 的不同区域
r_l1_sub = bl.subview(
    r_l1,
    [0, sub_vec_id * (HALF_BLOCK_M // 16), 0, 0],
    [BLOCK_N // 16, HALF_BLOCK_M // 16, 16, 16],
    [1, 1, 1, 1],
)
al.copy(bl.to_buffer(r_nz, al.ascend_address_space.UB), r_l1_sub)

# 3. Cube 侧通过 target_shape 读回 ND 格式
r_tensor = bl.to_tensor(r_l1, target_shape=[BLOCK_M, BLOCK_N])
```

**关键要点**：
- NZ 格式是 Ascend Cube 的 native layout，必须转换，不能直接 copy ND
- `bl.subview` 的 offset 按 `sub_vec_id * (HALF_BLOCK_M // 16)` 计算，确保双核不覆盖
- 两个 sub-core 合写满整个 L1 buffer
- Cube 读回时使用 `target_shape=[BLOCK_M, BLOCK_N]`，编译器自动做 NZ→ND 解析

### 2.3 同样的 NZ 逻辑适用于 Load 数据（如 K/V 共享或分离）

```python
kv_nz_tmp = tl.permute(
    kv.reshape(HALF_BLOCK_N, HEAD_DIM // 16, 16),   # [N, D//16, 16]
    (1, 0, 2)                                        # → [D//16, N, 16]
)
kv_nz = kv_nz_tmp.reshape(
    HEAD_DIM // 16, HALF_BLOCK_N // 16, 16, 16       # → [D//16, N//16, 16, 16]
)
```

---

## 3. 同步设计：细粒度信号与 PIPE 绑定

### 3.1 信号分配策略

每个 CV 通道需要一对 sync 信号（Vector set → Cube wait，Cube set → Vector wait）：

| 信号 | 方向 | 含义 | 推荐 PIPE |
|------|------|------|----------|
| 0 | V→C | 数据已 copy 到 L1，Cube 可开始 Compute1 | MTE3→MTE1 |
| 1 | C→V | Compute1 fixpipe 到 UB 完成，Vector 可后处理 | FIX→V |
| 2 | V→C | 后处理结果已 copy 到 L1，Cube 可开始 Compute2 | MTE3→MTE1 |
| 3 | C→V | Compute2 fixpipe 到 UB 完成，Vector 可累加/输出 | FIX→V |

```python
# Vector 侧
al.sync_block_set("vector", "cube", 0, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
al.sync_block_wait("cube", "vector", 1, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
al.sync_block_wait("cube", "vector", 3, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

# Cube 侧
al.sync_block_wait("vector", "cube", 0, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
al.sync_block_set("cube", "vector", 1, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
al.sync_block_set("cube", "vector", 3, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
```

### 3.2 是否需要进一步流水线？

- 仅对单个 buffer 做简单 ping-pong（如 Signal 4/5）在当前配置下通常无收益，反而增加同步开销。
- 若需要进一步隐藏 Load/Compute 延迟，推荐直接采用 **Batch 流水线（PIPE_STAGES=2）**，见 `references/operators/cv-fusion-pingpong.md`。

| 场景 | 建议 |
|------|------|
| 单 step / 内存紧张 | 使用 Signal 0/1/2/3，不做流水线 |
| 多 step / 内存充裕 | Batch 流水线（PIPE_STAGES=2），见 `references/operators/cv-fusion-pingpong.md` |

### 3.3 PIPE 参数选择规则

- **MTE3→MTE1**：Vector 完成 L1 copy（MTE3 写 L1）→ Cube 开始读取 L1（MTE1 读 L1）
- **FIX→V**：Cube 完成 fixpipe（FIX pipe 写 UB）→ Vector 开始读取 UB（V pipe 读 UB）

**不要混用 PIPE**，否则会导致死锁或数据竞争。

---

## 4. Scope 合并：从交替到聚合

### 4.1 原始交替模式（N 个 scope）

```python
for k in range(num_steps):
    with al.scope(core_mode="vector"):  # Load
    with al.scope(core_mode="cube"):    # Compute1
    with al.scope(core_mode="vector"):  # Process
    with al.scope(core_mode="cube"):    # Compute2
    with al.scope(core_mode="vector"):  # Accum
```

**问题**：每轮循环多次 scope 切换，编译器插入大量隐式同步，性能极差。

### 4.2 合并为 2 个 scope（推荐）

```python
with al.scope(core_mode="vector"):
    for outer_tile in range(0, OUTER_DIM, BLOCK_OUTER):
        for k in range(num_steps):
            # Load → Signal 0
            # Wait 1 → Process → Signal 2
            # Wait 3 → Accum
        # Final output

with al.scope(core_mode="cube"):
    for outer_tile in range(0, OUTER_DIM, BLOCK_OUTER):
        a_full = tl.load(...)
        for k in range(num_steps):
            # Wait 0 → Compute1 → Signal 1
            # Wait 2 → Compute2 → Signal 3
```

**关键要点**：
- 相同 core_mode 的代码合并到同一个 `with al.scope()` 块内
- 循环可以嵌套在 scope 内部
- 外层循环放 scope 内 vs 放 scope 外（通过子函数）对性能影响极小，但放 scope 内代码更紧凑

### 4.3 合并后的性能对比（参考 SFA）

| 版本 | 延迟 | 说明 |
|------|------|------|
| 原始多 scope 交替 | ~5.0 ms | 大量隐式同步 |
| 2 scope 合并 | ~1.95 ms | **1.52x 加速** |

---

## 5. 精度调试 checklist

### 5.1 分阶段验证

每引入一个新的 CV 通道，必须先跑通独立验证：

1. **只改 Cube 侧**：从 GM Buffer 改为 fixpipe → 验证 Vector 侧读到的 buffer 正确
2. **只改 Vector 侧**：从 GM Buffer 改为 copy 到 L1 → 验证 Cube 侧读到的 buffer 正确
3. **组合验证**：所有通道同时启用 → 最终 diff 应该与分阶段一致

### 5.2 常见精度陷阱

| 现象 | 根因 | 修复 |
|------|------|------|
| 输出 diff 较大 | Vector→Cube 的 copy 未转 NZ 格式 | 严格按 3D permute + reshape 转换 |
| 输出 diff 巨大 | BLOCK 尺寸过大导致 NZ subview 越界 | 检查 subview offset/size 与 BLOCK 的匹配 |
| 输出 nan / inf | `tl.where` mask 错误，导致全 -inf | 检查 mask 生成逻辑 |
| 部分输出对，部分错 | 某一 CV 通道的 fixpipe / readback 有问题 | 单独验证该通道 |
| 性能反而下降 | 同一数据 copy 两次（如共享数据却分配了独立 buffer）| 共享时保持单 buffer |

### 5.3 参考实现对比法

找到已验证通过的参考实现，逐条列出差异点：

1. 是否存在额外信号
2. 共享数据是否分离
3. NZ permute 是 3D 还是 4D
4. 外层循环位置（scope 内/外）
5. BLOCK 取值
6. `vf_merge_level` 是否启用
7. Scope 顺序（Vector 先 / Cube 先）

**每次只修改一个差异点，验证精度和性能，记录影响。** 这是避免多变量干扰的最佳实践。

---

## 6. 性能调优经验

### 6.1 数据通道优先级

优化收益排序（从高到低）：

1. **Cube→Vector fixpipe**：消除中间结果 GM 回退 → 最大收益
2. **Vector→Cube copy**：消除中间结果 GM 回退
3. **Load copy**：消除输入分别读 GM
4. **其他 Cube→Vector fixpipe**：消除输出 GM 回退

### 6.2 ROW_SPLIT 的收益条件

- `BLOCK_M // 2` 必须能被 16 整除（用于 NZ 的 subview offset）
- 若 `BLOCK_M = 32`，则 `HALF_BLOCK_M = 16`，完美匹配
- 若 `BLOCK_M = 16`，则 `HALF_BLOCK_M = 8`，NZ subview 的 `// 16 = 0`，需要特殊处理

### 6.3 常见优化收益参考

| 优化点 | 效果 | 结论 |
|--------|------|------|
| 共享数据分配独立 L1 buffer | -2% 性能 | 共享时保持单 buffer |
| Batch 流水线（PIPE_STAGES=2） | **1.11x ~ 1.17x** | 多 step 场景有效，见 `references/operators/cv-fusion-pingpong.md` |
| `vf_merge_level=1` | -3% 性能 | 部分算子反而劣化 |
| BLOCK 过大（如 128） | 编译器断言失败 / UB overflow | 需验证平台支持 |

---

## 7. 代码规范 checklist

- [ ] `disable_auto_inject_block_sync=True` 已设置
- [ ] 所有 `al.sync_block_set/wait` 都带上了正确的 PIPE 参数
- [ ] UB buffer 分配使用 `bl.alloc(dtype, [HALF_BLOCK_M, ...], al.ascend_address_space.UB)`
- [ ] L1 buffer 分配使用 `bl.alloc(dtype, [W//16, M//16, 16, 16], al.ascend_address_space.L1)`
- [ ] NZ 转换遵循 `reshape → permute → reshape` 三步
- [ ] `bl.subview` 的 offset 使用 `sub_vec_id * (HALF_BLOCK_X // 16)`
- [ ] Cube 读 L1 使用 `bl.to_tensor(l1_buf, target_shape=[M, W])`
- [ ] `tl.dot` 的输入 dtype 一致（必要时 `.to(dtype)`）
- [ ] 最终输出 diff 验证通过（< 1e-1）

---

## 8. 快速迭代流程图

```
原始算子 (GM-only)
    │
    ▼
拆分为 N 个阶段 (Cube/Vector 交替)
    │
    ├──► 阶段 1: 改 Vector Load → 验证 Load_Buffer
    ├──► 阶段 2: 改 Cube Compute1 fixpipe → 验证 C1_Buffer
    ├──► 阶段 3: 改 Vector Process copy L1 → 验证 V2_Buffer
    ├──► 阶段 4: 改 Cube Compute2 fixpipe → 验证 C2_Buffer
    └──► 阶段 5: 改 Vector Accum → 验证 Out/LSE
    │
    ▼
合并相同 scope (N → 2)
    │
    ▼
对比参考实现，列出差异点
    │
    ▼
逐一修改差异点 + 精度/性能测试
    │
    ▼
输出最终代码 + benchmark 报告
    │
    ▼
（可选）Batch 流水线：进一步隐藏 Load-Compute 延迟
```

*注：Batch 流水线（PIPE_STAGES=2）在 SFA 前向已验证有效，见 `references/operators/cv-fusion-pingpong.md`。*
