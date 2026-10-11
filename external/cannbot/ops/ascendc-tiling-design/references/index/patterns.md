# Index 类算子场景路由

> 本文档用于**场景判定**和**策略选择**。确定场景后，按链接进入对应详细文档。
> 本文档与各 Pattern 文档面向 **A2/A3 代际**（910B / 910C，`arch22`，`dav_c220` / `DAV_2201`，
> 单核 UB 192KB），实测基线为 Ascend 910B3 / CANN 9.0.0。

---

## 场景判定流程

Index（索引类）：以索引为核心访问数据——index 决定"从输入哪里取数"或"数据写到输出哪里"，存在跨元素寻址，输出形状/写位置不独立于 index。典型算子：Gather、GatherElements、IndexSelect、Take、Scatter、ScatterElements、ArgMax/ArgMin（返回索引）。

```
给定: N 个输入 shape（value + index）+ M 个输出 shape

Step 1 — 数据访问方向判定（决定数据流形态）:
  index 决定从输入哪里取数（输出形状由 index 形状决定）？
    ├─ YES → 读索引型（Gather 系）→ Step 2
    └─ NO  → index 决定数据写到输出的哪里（index 形状 == 输出形状）？
        ├─ YES → 写索引型（Scatter 系）→ Step 3
        └─ NO  → 归约同时返回极值索引（ArgMax/ArgMin）？
            ├─ YES → 归约索引型 → [../reduction/with-index.md](../reduction/with-index.md)
            └─ NO  → 无 index 输入，输出位置由 repeats 解析确定（RepeatInterleave）？
                ├─ YES → P7 [repeat_interleave.md](repeat_interleave.md)
                └─ NO  → 待补充场景
```

```
Step 2 — 读索引型（Gather 系）:
  index 的形状与输出形状的关系？
    ├─ index 与输出同形（每行一份 index，需按行分解前导维坐标）→ P2 [gather.md](gather.md)
    ├─ index 是 1-D 全行共享（源行查找纯算术、输出行连续）→ P5 [index_select.md](index_select.md)
    ├─ index 展平为 1D 连续取数（Take / Gather 展平取数）→ P1
    └─ 其他取数形态 → 待补充
```

```
Step 3 — 写索引型（Scatter 系）:
  index 的形状？
    ├─ index 与输出同形（逐元素映射）→ 按是否冲突分流:
    │    ├─ 无冲突 / 后写覆盖（ScatterUpdate / ScatterElements）→ P3 [scatter.md](scatter.md)
    │    └─ 有冲突需归约合并（ScatterReduce）→ P4 [scatter.md](scatter.md)
    │   （P3/P4 共用同一分核与核内流程，仅写出语义不同）
    └─ index 沿轴 1-D（一行 index 影响整个 inner 块）+ 累加语义（IndexAdd）→ P6 [index_add.md](index_add.md)
```

---

## Pattern 清单

> 各 Pattern 的详细设计文档**逐个补充**，补充前状态为"待补充"。

| 编号 | 场景 | 典型算子 | 判定条件 | 状态 | 详细文档 |
|------|------|---------|---------|------|---------|
| P1 | 扁平索引取数 | Take, Gather(展平) | index 展平为 1D，按 offset 连续取数 | 待补充 | - |
| P2 | 沿轴索引取数 | Gather, GatherElements | index 与输出同形，沿指定轴取数、需按行分解前导维坐标 | 已补充（三档模式：mode0 批量 / mode1 双轴分片，indexAxis 无上限 / mode2 scalar） | [gather.md](gather.md) |
| P3 | 写索引覆盖 | ScatterUpdate, ScatterElements | index 形状与输出一致，写位置无归约冲突 | 已补充（按输出元素空间 64B 整块分核 + **UB 内 RMW 散写 + 批量回写**，快原生 3.2x） | [scatter.md](scatter.md) |
| P4 | 写索引归约 | ScatterReduce | 同一写位置多次命中，需归约合并 | 已补充（与 P3 共用分核：输出元素单属主，累加无需原子） | [scatter.md](scatter.md) |
| P5 | 沿轴单索引取数 | **IndexSelect** | index 为 1-D 全行共享，输出行连续、源行查找纯算术 | 已补充（mode0 纯搬运行/段拷贝 / mode1 Gather 向量化 + kSplit 防空核） | [index_select.md](index_select.md) |
| P6 | 沿轴索引累加 | **IndexAdd** | index 沿轴 1-D，一行 index 影响整个 inner 块，累加语义 | 已补充（t-tile slab RMW + 选择性 upd 行搬入 + **per-core 命中表**，实测 1.04~17.50x） | [index_add.md](index_add.md) |
| P7 | 沿轴重复展开 | **RepeatInterleave** | 无 index 输入，展开位置由 repeats 解析确定（前缀和） | 已补充（不物化 index → bit-exact；mode0 段拷贝 / mode1 展开下标 + Gather） | [repeat_interleave.md](repeat_interleave.md) |

---

## 通用规则

以下规则适用于所有 Index 分支。

- **UB 预算**：value 与 index 都需要搬入 UB，UB 预算需同时计入两者（index 通常为 int32）
- **index 与 value 的依赖**：取数/写数的目标位置由 index 值决定，无法像 EleWise 一样按地址连续性直接流水，切分策略需围绕 index 的布局设计
- **多核对齐 / 切分基准**：读索引型按行或按元素总量分核（输出与 index 一一对应，无写冲突）；
  写索引型按**输出元素空间**分核（见下条红线）
- **🛑 写索引型的多核切分基准是「输出元素空间」，不是 index 元素空间**：
  AI Core 标量写 GM 时，两个核并发写同一条 64B cache line 的不同 32B sector 会丢失其中一个
  sector（表现为某些元素完全未写入、且无错值）。读索引型（Gather）按 index 元素量分核是安全的
  （输出与 index 一一对应），但写索引型必须按输出空间按 64B 对齐整块分核、每核独占连续整块。
  详见 [scatter.md §3](scatter.md)
- **🛑 写索引型不要用标量直接写 GM，要在 UB 内散写后整片回写**：
  散射地址由 index 值决定、无法连续 `DataCopy` 搬出，标量写不可避免；但**标量写的目标应是 UB**
  （GM 标量写延迟约 55 cycles/迭代，UB 标量写快一个数量级）。流程：
  `DataCopyPad` 从**输入**整片读入基准片 → 向量化「负索引修正 + 片基址重定位」→ 标量 UB 散写 →
  `DataCopyPad` 整片回写输出。实测把 scatter 从 0.45x 提到 3.2x。详见 [scatter.md §2/§4](scatter.md)
- **🛑 不要用 `clone` 给输出做初值：kernel 直接「读 x 写 y」，torch 侧 `at::empty_like`**：
  需要保留输入原值再改写的 Pattern（如 IndexAdd、Scatter 覆盖写）若先 `var.clone()` 再原地改，
  会多一次全量设备拷贝（读+写），且 clone 是 torch 算子、会 enqueue 进延迟队列，
  对 kernel 可能不可见（竞态）。正确做法是 kernel 直接从 x 读基准、把结果写独立输出 y——
  前提是**工作单元划分恰好覆盖整个输出空间**：IndexAdd 的 `units = outer*tTiles*iTiles` 满足；
  Scatter 按输出元素空间 64B 对齐分核同样满足，但**核内必须遍历输出的全部行**——
  index 前部维小于输入时输出存在「无对应 index 行」的行（如 `x=(8,32,16)` /
  `index=(4,16,8)` / `dim=1`），这些行要按输入原值整片回写，只遍历 index 行会漏写
  （实测 Max diff 5.24，详见 [scatter.md §3/§4](scatter.md)）。
  纯读型（IndexSelect / RepeatInterleave）输出全新、本来就不需要 clone，用 `at::empty` 即可。
  详见 [index_add.md](index_add.md)
- **🛑 `stream(true)` 必须在「所有 torch 侧算子之后」调用**：
  `NPUStream::stream(true)` → `stream()` → `MakeSureQueueEmpty()`，会清空 torch_npu 的
  延迟任务队列、把此前 enqueue 的 torch 算子真正下发。若先取 stream 再执行 `fill_`/`clone`，
  这些算子会留在队列里没下发，kernel 读到 `at::empty` 的**未初始化内存** →
  表现为「**首调用随机错、重跑即好、加 synchronize 即好**」。
  诊断特征：失败集每次不同 + 对失败用例重复 5 次恒为「第 1 次失败后 4 次全过」。
  详见 [index_add.md](index_add.md)
- **⚠️ 单元内含「扫描全部 index」的 Pattern，成本 ∝ 单元数 × index 长度**：
  当 UB 放不下整行导致单元数激增（如 `tTiles` 大）且 index 长时，标量扫描会成为瓶颈，
  而非搬运（诊断：固定 shape 扫 index 长度，耗时线性增长即为扫描放大）。
  缓解：branchless compaction（`selp[nSel]=k; nSel += hit;`）+ 4 路展开；
  根治：每核只扫一次 index 并按 tile 分桶（per-core 命中表）。详见 [index_add.md §5](index_add.md)
- **🛑 index 类型必须 host / kernel 一致，统一按 int32**：
  host 侧 `index.to(at::kInt)`，kernel 侧 `GlobalTensor<int32_t>`；上游为 int64 时
  （PyTorch 默认）须转换并**核对索引上界 < 2^31**（int32 溢出会越界读）。
  UB 侧 `indexBuf` 容量也按 `sizeof(int32_t)` 计算（早期按 `sizeof(int64_t)` 的写法已废弃）
- **⚠️ `DataCopyExtParams::blockLen` 是 `uint16_t`**（单次搬运 ≤ 65535 字节），
  超过会报 `-Wc++11-narrowing` 编译错。片长（gather `chunkLen`、scatter `pieceCap`、
  repeat_interleave 段长）**必须由 host 侧约束**，不能指望 kernel 内自适应
- **⚠️ stride 单位**：`DataCopyExtParams::srcStride/dstStride` 在 **GM 侧为字节、UB 侧为 32B DataBlock**；
  `DataCopyParams::stride` 是**块间隔**语义（不是字节偏移）。连续数据 stride 填 0。
  最常见的写错：UB 侧 stride 用了字节（应除以 32）
- **⚠️ 同步纪律：单缓冲 + 方向明确的事件同步，热路径禁用 `PipeBarrier<PIPE_ALL>()`**：
  `PIPE_ALL` 会阻塞全部流水、把可并行的搬运与计算串行化。用 `SyncM2toV` / `SyncVtoM2` /
  `SyncM2toM3` / `SyncVtoM3` 等方向明确的替代。**每一次「分片搬入」之前必须有对应的
  `Sync?toM2`** —— 上一片的向量/标量计算仍在读该缓冲时被 MTE2 覆盖，会导致大 indexAxis
  用例**稳定错位**（固定位置出错、输出 0.0）。
  **单缓冲 + 大 chunk 优于双缓冲**：给 index/x 加双缓冲会挤占 chunk 长度、使同步轮数
  增大 30~70%，净亏（瓶颈是**同步次数与搬运延迟**，不是带宽——实测带宽利用率仅 0.7~1.5 GB/s）
- **⚠️ 不要忽视 host 固定开销**：小 shape 上 kernel 可能比 torch 快 2x 而端到端更慢，
  瓶颈是 `at::empty` + launch（kernel 在途时 allocator 需延迟回收）。用输出缓冲池复用显存，
  详见 [index_select.md](index_select.md) §Host 固定开销与输出缓冲池

---

## 常见问题排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 输出数据错位 | 缺少同步 | 搬入后补方向明确的事件同步（见各 Pattern 的同步链） |
| 部分数据为 0 / 随机值 | 异步操作未完成 | 同上；`printf` 能"解决"只是因为它有延迟，不是正确做法 |
| 行间数据串扰 | stride 计算错误 | 核对 srcStride/dstStride 单位（GM 字节 / UB 32B 块） |
| 多核结果错误 | 分核方式错 | 读索引型按行或元素总量；写索引型按输出空间 64B 整块 |
| 累加结果丢失/错误 | 多核写同一 cache line | **按输出元素空间 64B 对齐整块分核**，不要用 `SetAtomicAdd` |
| 小 shape 端到端比 torch 慢但 kernel 更快 | host 固定开销（alloc + launch） | 用输出缓冲池复用输出显存 |
| 大 `nidx` 的 mode0 index_add 慢 | 每单元扫描全部 index，放大 = 单元数 | 走 per-core 命中表 |
| 3D+ 非尾轴行映射错位（2D 正常） | 行号→坐标分解顺序错 | 最后一个前导维变化最快，必须**从最后一维往前**分解 |
| 大 shape 段错误 | 坐标分解顺序错 | 线性索引→多维坐标必须**从前到后**循环 |
| `DataCopy` 搬出越界 / OOM（长度非 32B 对齐） | `DataCopy` 要求 32B 对齐，会按对齐长度多搬 | 改用 `DataCopyPad`（按字节精确搬出），`blockLen = 元素数 * sizeof(T)` |
| 用了 `index.accessor<int64_t,1>()` 编译/运行失败，或 host 侧校验后耗时暴涨 | accessor 只支持 CPU 张量；`index.to(CPU)` 触发 **NPU→CPU 同步**（Host 从 ~20us 升到 9000+us） | **不要在 host 读 index 值**做数据级校验，只做 shape 级校验；范围校验放 kernel 端（见 [scatter.md](scatter.md) §7） |

---

## 验证与性能测量纪律

索引类算子的正确性判定与性能测量有统一口径，避免误判。

### 正确性口径

- **bit-exact（逐位一致，max diff = 0）**：索引类算子的输出是**纯搬运 / 选值**，
  不含浮点运算，因此默认要求逐位一致。
- **例外：IndexAdd 用 `allclose`**：index_add 是累加，浮点累加顺序不同会引入末位差异，
  用 `allclose`（而非 bit-exact）判定。
- **负索引专项**：`torch.gather` **不支持负索引**，因此负索引用例的 golden 需单独用
  `index % axis` 修正后再比对，不能直接用 `torch.gather` 当 golden。
- **用例分层**：小 shape（功能）/ 大 shape（性能）/ 双轴分片专项 / 负索引专项 / 空 tensor。

### 性能测量纪律

- **多轮取最小值**：`bench(fn, iters=50, warmup=20, rounds=5)`，**取 min-of-5**。
  每次采集前 `torch.npu.synchronize()`。
- **设备波动**：NPU 共用设备波动可达 ±30% 常态（实测出现离散度 729%、1008%）。
  离散度 > 10% 需复采；4 组取 min + 中位数交叉裁定。
  **单次平均值会误判优化方向** —— 曾把噪声误读为「UB 预算变大导致变差」，
  改用 min-of-5 后稳定提升（±1%）才显现。
- **跨进程漂移可达 2~5x，必须同进程交错对照**：同一 case 的 custom/torch 耗时在不同进程间
  漂移极大（index_select 实测 18us ↔ 49us），**跨进程比较会得出互相矛盾的结论**。
  做 A/B 对照（如命中表 hit vs nohit）时必须：**同进程内交错执行各方案 + 提高轮次取 min
  （rounds=25）**，否则无法归因。
- **C++ 侧消融开关不能用 `static` 缓存 `getenv`**：若把 `getenv("XXX")` 的结果缓存进
  `static` 变量，Python 侧运行中改 `os.environ` 后**不生效**，A/B 会全部跑到同一分支。
  消融开关必须每次调用都重新读 `getenv`（去掉 `static`）。
- **直调通路只验功能、不验性能**：直调 exe 的单次 launch 被**进程级冷启动**支配
  （微小 shape 输出 512 元素仍耗 8.6ms）。权威性能数据取 **PyTorch 通路**。
- **带宽口径**：`GB/s = (x.numel() + y.numel()) * element_size / 耗时`。
  910B 的 L2 = 192MB，流量小于 L2 时测得的是 **L2 带宽**而非 HBM 带宽。

---

## 参考实现

以下工程均已在 Ascend 910B3 上编译、上板验证：

- `/data/fr/operator/gather_custom` - gather（三档 mode0/1/2）
- `/data/fr/operator/index_custom` - index_select
- `/data/fr/operator/index_add_custom` - index_add
- `/data/fr/operator/scatter_custom` - scatter / scatter_add
- `/data/fr/operator/repeat_interleave_custom` - repeat_interleave

---

## 跨场景参考

| 主题 | 文档 |
|------|------|
| 归约索引型（ArgMax/ArgMin 返回极值索引） | [../reduction/with-index.md](../reduction/with-index.md) |
