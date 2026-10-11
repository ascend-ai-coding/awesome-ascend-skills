# MACHINE 组件经验

> 对应错误码范围：F7-F8XXXX

---

## 1. F7B007: Device 感知 / `set_device` 缺失

**触发场景**：

| 触发条件 | 错误关键词 |
|---------|----------|
| wrapper 硬编码 `device = 'npu:0'`，kernel 在 `TILE_FWK_DEVICE_ID` 指定设备执行 | `F7B007 RT_CAPTURE_FAILED 107003: get capture info failed: 107003` |
| JIT 编译前未调用 `torch.npu.set_device()` | `F7B007 RT_CAPTURE_FAILED 107003`（kernel 首次调用时） |

**解决方案**：
```python
import os
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))
device = f'npu:{device_id}'
torch.npu.set_device(device_id)  # 必须在所有 JIT 调用前执行
```

---

## 2. AICPU error: `pypto.Tensor()` buffer 声明在 loop 内/外创建

**根因**：`pypto.Tensor()` / `pypto.tensor()` 是 buffer 声明（一次性分配 workspace），不是计算操作。与 `pypto.full()`（计算操作，每次执行产生新 tensor）语义不同。

**触发场景**：`pypto.loop` 内部调用 `pypto.Tensor([...], dtype, "name")` 声明 buffer，per-iteration 分配与设备内存映射冲突。编译无错，运行时崩溃。

**错误关键词**：`AICPU error: errorCode=0x2a, taskid: 1, retcode: 507018` / `torch.npu.synchronize()` 崩溃

**解决方案**：用 `pypto.full()`（计算操作）替代 `pypto.Tensor()`（buffer 声明），确保 per-iteration 独立内存。

**嵌套循环注意事项**：对于嵌套循环，buffer **不可在 outer loop 外声明**。`faq.md` §"submit_before_loop" 说明该参数仅将循环前任务先提交到调度队列，不序列化外层迭代。`faq.md` §"隐式 loop(1)" 示例表明 loop 外创建的 tensor 被框架置于独立的隐式 `loop(1)` 中，与主 loop 分离。outer loop 外创建的 buffer 被所有并行 outer 迭代共享同一内存槽，导致精度失败（B=1 PASS 但 B>1 FAIL，值在 batch 位置间互换）。

```python
# ❌ 嵌套循环：outer loop 外创建 named buffer → 并行 outer 迭代共享内存
h_buf = pypto.Tensor([1, N, N], pypto.DT_FP32, name="h_buf")  # 共享内存！
for b in pypto.loop(B, name="batch", submit_before_loop=True):  # 不序列化！
    h_buf[:] = h_init
    for _ in pypto.loop(19, ...):
        ...
        h_buf[:] = h_new

# ✅ outer loop 内用 pypto.full 创建匿名 buffer → per-iteration 独立内存
for b in pypto.loop(B, name="batch"):
    h_buf = pypto.full([1, N, N], 0.0, pypto.DT_FP32)  # 每次迭代独立
    h_buf[:] = h_init
    for _ in pypto.loop(19, ...):
        ...
        h_buf[:] = h_new
```

---

## 3. AC110005: matmul 后的 vec 操作需要匹配 matmul 输出维度的 tile shape

**触发场景**：`set_vec_tile_shapes(1, H)` 为 matmul 前的操作设置，matmul 输出维度是 D。matmul 核心 tiling 使用 cube tile（`GetCubeTile()`），batch matmul 构造代码内部会 save/restore vec tile，但 matmul **后续的 vec 操作**（cast/mul/add 等）仍使用旧的 `(1, H)` tile，对 `[1, D]` 输出做 tiling 时内存布局错误导致读写出界。编译无报错。

**错误关键词**：`AC110005 (aicore error) / Aborted (core dumped)`

**解决方案**：matmul 后、vec 操作前，重新设置匹配 matmul 输出维度的 tile：
```python
# ❌ matmul 后 vec tile 仍是 (1, H=512)，但输出是 [1, D=128]
pypto.set_vec_tile_shapes(1, H)
o = pypto.matmul(q[1, H], k[H, D], ...)
y = pypto.mul(o, factor)  # vec op 用 (1, H) tile 操作 [1, D] tensor → crash

# ✅ matmul 后重设 tile 匹配输出
pypto.set_vec_tile_shapes(1, H)
o = pypto.matmul(q[1, H], k[H, D], ...)
pypto.set_vec_tile_shapes(1, D)  # ← matmul 后必须重设
y = pypto.mul(o, factor)
```

---

## 4. Python for 展开重循环体导致 aicore error

**触发场景**：Python `for h in range(N)` 将计算图展开 N 倍（每次迭代生成独立 IR 副本），当循环体较重时（如含 matmul + 多个 vec 操作），设备资源不足导致 AiCore crash。

**错误关键词**：`aicore error (507018 / 0x2a)` — 重循环体用 Python `for` 展开而非 `pypto.loop`

**解决方案**：将循环改为 `pypto.loop`，避免 IR 展开爆炸。

**何时用 Python `for` 而非 `pypto.loop`**：

本规则适用于**循环变量仅用于索引**的场景（如 `q[h]`）。如果循环变量需要作为 `pypto.view` 的 shape/offset 参数，则必须用 Python `for`（静态边界）：

```python
# ✅ 场景 A：循环变量仅用于索引 → pypto.loop
for h in pypto.loop(num_heads):
    q_h = q[h]  # h 只用于索引
    result = pypto.matmul(q_h, k_h)

# ✅ 场景 B：循环变量用于 shape/offset → Python for（静态边界）
for i in range(8):  # 编译期已知
    buf_view = pypto.view(buf, [i, col_num], [0, 0])  # i 用于 shape
```

**判断规则**：
- 循环边界**动态**（运行时才知道）→ 必须用 `pypto.loop`
- 循环边界**静态** + 循环变量仅用于 offset/索引/计算 → 优先 `pypto.loop`（避免 IR 展开爆炸）

---

## 5. Host SIGSEGV: bare `[:]` 赋值后再次读取源 tensor

**根因**：`B[:] = A`（bare `[:]`，无显式起止索引）触发 move 语义（`_is_empty_slice` 判断后调用 `self.move(value)` → C++ `Move`），A 的 `storage_` shared_ptr 被置空。后续读取 A 时解引用空指针导致 SIGSEGV。

**重要区分**：仅 bare `[:]` 是 move。显式切片如 `B[0:, 0:] = A` 走 `assemble`（copy 语义），A 保持有效。

**错误关键词**：`Segmentation fault (core dumped)` / `SIGSEGV` — 无 PyPTO 错误码；gdb 堆栈指向 `LogicalTensor::Datatype() const+0`（空指针解引用）

**诊断方法**：定位 bare `[:]` 赋值操作与对该源 tensor 的后续读取操作之间的顺序关系。若 `[:]` 在读取之前执行，即为根因。

**解决方案**（三选一）：

1. **交换操作顺序**：确保对 A 的所有读取在 `B[:] = A` 之前完成
2. **改用显式切片**：`B[0:, 0:] = A`（copy 语义，A 不失效）
3. **改用 assemble**：`pypto.assemble(A, [0, 0], B)`（copy 语义）

```python
# ❌ bare [:] = move → A 失效 → SIGSEGV
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
m_update[:] = tilda_mij
tsub = pypto.sub(sij_scale, tilda_mij)  # 读取已失效的 tilda_mij

# ✅ 方案 1：先读取，后赋值
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
tsub = pypto.sub(sij_scale, tilda_mij)
m_update[:] = tilda_mij

# ✅ 方案 2：显式切片 = copy 语义
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
m_update[0:, 0:] = tilda_mij  # copy, tilda_mij 不失效
tsub = pypto.sub(sij_scale, tilda_mij)
```

| 触发条件 | 错误关键词 |
|---------|----------|
| bare `[:]` 赋值后再次读取同一源 tensor | `Segmentation fault (core dumped)` / `SIGSEGV` / `LogicalTensor::Datatype() const+0` |
| loop 内先写后读同一 intermediate tensor | 同上（仅第一次迭代触发，因 tensor 在第一次 `[:]` 后即失效） |

---

## 6. Workspace overlap in PagedAttention / online-softmax

**触发场景**：PagedAttention / Flash Attention / online-softmax 手工实现中，使用共享 workspace 存储 partial statistics（partial_max、partial_sum、KV_cache 切片）。若两个 live 中间 tensor 的 allocate 范围发生重叠（共享 workspace 内地址复用），会导致非确定性输出或 SIGABRT。

**错误关键词**：`SIGABRT` / `non-deterministic output` / `workspace overlap` / `output mismatch runs 1 and 2`

**诊断方法**：
1. 对比同一输入两次运行结果是否不同（非确定性）
2. 检查 workspace tensor 的 allocate 顺序：所有 persistent 中间 tensor 应在 loop 外预先 allocate
3. 确认 loop 内不创建新 tensor 分配

**解决方案**：
1. **（推荐）** 所有 workspace tensor 在 loop 外一次性分配，loop 内仅做 `[:]` 就地更新
2. 使用官方内存重叠检测工具定位具体重叠位置：`python3 tools/schema/schema_memory_check.py -d <device_log_dir> -t <dyn_topo_file>`
3. **（最后手段，项目级临时规避）** 将 attention 计算降级为 host 侧 torch 实现（当 NPU workspace 管理无法可靠隔离时）— 来自 glm_v4_5_attention_fusion 的实际决策

| 触发条件 | 错误关键词 |
|---------|----------|
| PagedAttention / online-softmax 在线累积 | `workspace_overlap` / `non-deterministic output` / `output mismatch` |
| loop 内 allocate 新 tensor 覆盖已存活的 workspace 块 | 同上 + `SIGABRT` |

---

## 7. AiCore CCU instruction address check error (retcode 507015)

**触发场景**：NPU 运行时 CCU（Computational Control Unit）检测到非法内存访问或指令
地址越界。编译可通过（Pass_27/PASS 无报错），但 Launch 阶段 CCU 硬件报错。与
F4/F5 类 PASS 错误不同：编译期不抓、仅在运行时触发。

**错误关键词**：`CCU instruction address check error` / `retcode 507015` /
`LaunchKernelTorch → _execute_kernel`

**诊断方法**（按优先级依次排查）：

1. **算法缺失**：对比 golden.py 与 impl.py 的计算步骤 → 检查是否漏掉了某个操作
   （如 softmax 的 scale 因子、exp 步、sum/div 归约）。在 sparse_attention_grad_tnd
   案例中，缺失 `scale` 和 `softmax` 子步骤直接导致 CCU crash，补齐后通过。
   ```
   correct:  amax → sub → exp → sum → div
   missing:  sub → exp        # 缺 3 个步骤 → CCU crash
   ```

2. **Cube tile K 维度与 matmul 实际 K 不匹配**：检查 cube tile 的 kL0 是否 > 矩阵 K 维度。
   当 kL0 超出时，CCU 尝试读取超出 buffer 边界的数据 → crash。
   ```
   matmul(M×K, K×N) where K=576
   cube_tile used kL0=768      # 768 > 576 → CCU crash
   fix: kL0=16 (≤ K and 16-aligned)
   ```

3. **K=1 边缘 case**：当 K=1 时，cube tile [16,16] 的 kL0=16 > 1 → CCU crash；
   cube tile [1,1] 的 kL0=1 不满足 16 对齐的 FC4001。根本性不兼容，必须用 vec 操作
（`pypto.mul` + `pypto.sum`）替代 matmul。见 `experience_classified/matmul.md` §5。

| 触发条件 | 错误关键词 |
|---------|----------|
| 算法缺失（漏掉 scale/exp/sum 步骤） | `CCU error 507015` + 编译通过 |
| cube tile kL0 > 实际 K 维度 | 同上 |
| K=1 matmul 用 cube tile | 同上 + `FC4001` |

---

## 8. SIGABRT: pypto.index_add / index_add_ inside pypto.loop body

**触发场景**：在 `pypto.loop` 循环体内调用 `pypto.index_add` 或 `pypto.index_add_`。
JIT 编译器的 IR 生成阶段，index_add 的控制流与 loop 的迭代展开发生冲突 → 编译期
内部断言触发 SIGABRT（signal 6）。

**错误关键词**：`SIGABRT` / `Aborted (core dumped)` / `signal 6`

**解决方案**：将 kernel 侧 tiling 提升到 host 侧。Kernel 使用静态 shape（如 `[64]` 或 `[64, 16]`），
host wrapper 用 Python for-loop 逐 tile 调用 kernel，tail 用零填充块（index=0 + update=0.0 作为 no-op）。

```python
# ❌ JIT 内 loop+batch index_add → SIGABRT
@pypto.frontend.jit
def kernel(tensor_in, indices, updates):
    for b in pypto.loop(B):
        pypto.index_add(tensor_in, indices, updates, axis=0)

# ✅ host 侧 tiling，kernel 内无 loop
@pypto.frontend.jit
def kernel_tile(tensor_tile, indices_tile, updates_tile):  # 静态 shape
    pypto.index_add(tensor_tile, indices_tile, updates_tile, axis=0)

for k_start in range(0, K, TILE_K):
    tile = tensor_in[k_start:k_start+TILE_K]
    kernel_tile(tile, indices_slice, updates_slice)
```

| 触发条件 | 错误关键词 |
|---------|----------|
| `pypto.index_add` 或 `index_add_` 在 `pypto.loop` 内 | `SIGABRT` / `signal 6` / `Aborted (core dumped)` |
| `pypto.scatter_add` 或 `index_add` 在动态轴上 | 同上 |

---

## 9. F71008: MAP_REG_ADDR_FAILED — 设备资源争用

**触发场景**：PyPTO 运行时在 kernel launch 阶段调用 `HalMemCtl` 映射设备寄存器地址时失败
（`runtime_agent.cpp` 中 `HalMemCtl` 调用，`MAP_REG_ADDR_FAILED` 错误日志）。该错误与 kernel 代码无关，**纯粹是外部环境问题**：目标 NPU 设备
已被其他进程占用，寄存器地址空间不可用。

**错误关键词**：`F71008` / `HostLauncherErr::MAP_REG_ADDR_FAILED` / `Map reg addr fail, maybe others are using current device`

**诊断方法**：
1. 运行 `npu-smi info` 查看所有设备的 HBM 使用量
2. 空闲设备的 HBM 使用量通常在 ~2800-3200 MB，被占用设备显著偏高（>4500 MB）
3. 确认 `TILE_FWK_DEVICE_ID` 指向的设备是否被占用

**解决方案**：切换到空闲设备重新运行：
```bash
# 查找空闲设备（HBM 使用量最低的设备）
npu-smi info

# 切换到空闲设备运行
TILE_FWK_DEVICE_ID=<idle_device_id> python your_impl.py
```

**注意**：此错误可能在多算子并行场景中频繁出现（多个进程
同时占用不同设备）。orchestrator 应在检测到 F71008 时自动切换设备重试，
而非视为 kernel 代码缺陷。

| 触发条件 | 错误关键词 |
|---------|----------|
| 目标设备被其他进程占用 | `F71008 MAP_REG_ADDR_FAILED` / `Map reg addr fail` |
| 多进程并行设备争用 | 同上 |

---

## 10. 跨 pypto.loop 的 assemble→view 数据不可靠

**根因**：`insert_sync.cpp` 的 DDR 依赖追踪按 leaf function 粒度工作，跨 `pypto.loop` 场景覆盖不充分。同时 workspace 按 `WorkspaceRecyclePeriod`（`stitch_function_max_num × MAX_UNROLL_TIMES`）周期回收而非按数据依赖，跨 loop 的 assemble→view 数据传递不可靠。官方示例（`flash_attention_mha_impl.py` 等）从不从 assembled buffer 跨 loop 回读。

**触发场景**：在两个独立的 `pypto.loop` 之间，通过 `pypto.assemble` 写入 buffer、再用 `pypto.view` 读取同一 buffer 传递数据。**数据不可靠**。

**错误关键词**：无特定错误码，表现为**精度失败**（`all_close: false`），且仅影响跨 loop 读取的数据

**诊断方法**：
1. 控制变量实验：同一 kernel 中，直接读取（不经过 buffer）PASS，经 buffer 读取 FAIL
2. 检查 `flash_attention_mha_impl.py` 等官方参考实现：**从不**从 assembled buffer 读取，Q 始终从输入 tensor 直接 view

**解决方案**：在第二个 loop 中**重新计算**所需数据，而非从 buffer 读取：

```python
# ❌ 跨 loop 通过 assemble→view 传递 Q
for b_idx in pypto.loop(M, name="preprocess"):
    q_final = compute_q(...)
    pypto.assemble(q_final, [b_idx, 0, 0], q_out_3d)  # 写入 buffer

for b_idx in pypto.loop(M, name="attention"):
    q_tok = pypto.view(q_out_3d, [1, N1, D], [b_idx, 0, 0])  # 不可靠！
    ifa(q_tok, ...)

# ✅ 在第二个 loop 中重新计算 Q
for b_idx in pypto.loop(M, name="preprocess"):
    k_final, v_final = compute_kv(...)  # 只计算 K/V
    scatter_kv(k_final, v_final)         # scatter 到 cache

for b_idx in pypto.loop(M, name="attention"):
    q_final = compute_q(...)  # 重新计算 Q（~14% 开销）
    ifa(q_final, ...)          # 直接使用，无 buffer 读取
```

| 触发条件 | 错误关键词 |
|---------|----------|
| 两个 `pypto.loop` 之间通过 assemble 写入、view 读取同一 buffer | 精度失败（无特定错误码） |
| 跨 loop 数据传递 | 同上 |

---

## 11. F76005: STITCH_HANDLE_INDEX_OUT_OF_RANGE — 跨 loop 中间 buffer stitch 编码器溢出

**触发场景**：两个 `pypto.loop` 通过中间 buffer 连接（第一个 loop 写入 buffer，第二个 loop 读取），stitch 编码器的 handle index 超出上限。PyPTO 文档未覆盖此错误码；`https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/trouble_shooting/machine.md` 仅在性能分析上下文中提及 `stitch_function_max_num` 和 workspace 内存。

**错误关键词**：`F76005 STITCH_HANDLE_INDEX_OUT_OF_RANGE at operationIndex=N` + SIGABRT

**解决方案**：合并两个 loop 为单个 loop，消除中间 buffer 依赖：

```python
# ❌ 两 loop 通过中间 buffer 连接 → stitch handle 溢出
for b in pypto.loop(B, name="preprocess"):
    q_had = compute_q(...)
    pypto.assemble(q_had, [b, 0, 0], q_had_buf)  # 写入中间 buffer

for b in pypto.loop(B, name="attention"):
    q = pypto.view(q_had_buf, [1, H, D], [b, 0, 0])  # F76005
    ifa(q, ...)

# ✅ 合并为单 loop
for b in pypto.loop(B, name="combined"):
    q_had = compute_q(...)
    ifa(q_had, ...)  # 直接使用，无中间 buffer
```

| 触发条件 | 错误关键词 |
|---------|----------|
| 两个 `pypto.loop` 通过中间 buffer 连接 | `F76005 STITCH_HANDLE_INDEX_OUT_OF_RANGE` + SIGABRT |
