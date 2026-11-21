# MIX (Cube+Vector 融合) 开发与调优指南

适用：`KERNEL_TYPE_MIX_AIC_1_1` / `MIX_AIC_1_2` 的 Cube+Vector 融合 kernel 开发与 hang 排查。
来源：gated_delta_rule CV 融合重写实证（scalar/VEC 3.3x → MIX_AIC_1_1 geomean 8.85x，`a5_ops_slim/workspace/gated_delta_rule_cv`）。
配套策略卡：P102（scalar→MIX 重写决策）、P103（Cube Neumann 三角求逆）、P1（双缓冲决策矩阵交叉引用）。

---

## 1. MIX 核心机制

| 机制 | 要点 |
|------|------|
| 单一二进制双核执行 | MIX 启动后，**同一份 `__global__` 代码在 AIC 和 AIV 上都执行**。不做核类型分支 = AIV 执行 AIC 专属指令 = **死锁** |
| `ASCEND_IS_AIC` / `ASCEND_IS_AIV` | 编译期宏，每个编译趟静态求值，`if ASCEND_IS_AIC {...}` 被干净裁剪（**宏自带括号，不要再加 `()`**） |
| `GetBlockIdx()` 配对 | MIX_AIC_1_1 下 **AIV 核 i 与 AIC 核 i 一一配对**；1:2 下 AIV idx = AIC idx × 2 + {0,1} |
| `SyncAll<false>()` | AIC+AIV **全局 barrier**（ffts 硬件握手）。无参模板默认 `isAIVOnly=true` 只同步 AIV——**必须显式 `<false>`** |
| Buffer 初始化 | `pipe_.InitBuffer(A1/B1/A2/B2/CO1 队列)` 只能在 AIC 趟执行；`TBuf<VECCALC>` 只能在 AIV 趟 |

**hang 的第一性原理**：MIX 模式本身可行且是必需架构。hang 几乎全部来自"错误类型的核执行了不该执行的指令"或"两侧 barrier/迭代次数不对齐"。

## 2. 核类型守卫与队列归属

| 资源 | 所属核 |
|------|--------|
| `TQue<TPosition::A1/B1/A2/B2/CO1, N>` | AIC |
| `TBuf<TPosition::VECCALC/VECIN/VECOUT>` | AIV |

**越界即死锁**（100% AICore 空转，无报错）：AIV 趟执行 AIC 队列的 InitBuffer/AllocTensor，或 AIC 趟触碰 UB buffer。
`InitBuffer` 必须与使用该 buffer 的代码**同守卫**，且**在循环外一次性调用**（循环内反复调用会重置队列状态）。

```cpp
class KernelX {
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
extern "C" __global__ __aicore__ void kernel_x(GM_ADDR ...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);   // 函数体内第一行
    KernelX k; k.Init(...); k.Process();
}
```

逻辑核号换算：
```cpp
int32_t coreIdx;
if ASCEND_IS_AIC { coreIdx = GetBlockIdx(); }
if ASCEND_IS_AIV { coreIdx = GetBlockIdx() / GetSubBlockNum(); }  // 1:2 时 /2，1:1 时 =1
```

## 3. 最小 Mmad 链（GM → L1 → L0 → Mmad → Fixpipe → GM）

```cpp
// GM(ND) → L1(NZ)：srcDValue 是 GM 行步长，单位【元素】
Nd2NzParams p; p.ndNum=1; p.nValue=m; p.dValue=n; p.srcNdMatrixStride=0;
p.srcDValue=ld; p.dstNzC0Stride=m; p.dstNzNStride=1; p.dstNzMatrixStride=0;
DataCopy(l1Tensor, gmTensor, p);

// L1(NZ) → L0A/L0B：LoadData3DParamsV2
LoadData3DParamsV2<half> lp;
lp.l1H=1; lp.l1W=colC0Stride; lp.channelSize=k; lp.kExtension=k; lp.mExtension=m;
lp.strideH=1; lp.strideW=1; lp.filterH=1; lp.filterW=1;
lp.dilationFilterH=1; lp.dilationFilterW=1;
LoadData(l0Tensor, l1Tensor, lp);

// Mmad：cmatrixInitVal 1=覆盖 L0C；0=在 L0C 上累加（链式乘加关键）
MmadParams mp{}; mp.m=m; mp.n=n; mp.k=k; mp.cmatrixInitVal=1;
Mmad(c1Tensor, a2Tensor, b2Tensor, mp);

// Fixpipe L0C → GM：NZ2ND 默认 CFG_ROW_MAJOR，dstStride 单位【元素】
// （非 NZ 模式 dstStride 单位是 32B，注意区分）；fp32→fp16 加 quantPre=F322F16
FixpipeParamsV220 fp; fp.nSize=n; fp.mSize=m; fp.srcStride=m; fp.dstStride=n;
fp.ndNum=1; fp.srcNdStride=0; fp.dstNdStride=0;
Fixpipe(gmOut, c1Tensor, fp);
```

**队列完整性**：每个 Mmad 操作数必须 `Alloc→Load→EnQue→DeQue→Mmad→Free` 完整走队列；Mmad→Fixpipe、Fixpipe→下次 GM 读之间 `PipeBarrier<PIPE_ALL>`。缺事件同步 = aicore exception（有报错，非 hang）。

## 4. Padded Loop + valid 守卫（循环内 SyncAll 的前提）

```cpp
int32_t slots = (totalWork + ncore - 1) / ncore;   // 全核相同
for (int32_t s = 0; s < slots; ++s) {
    int32_t work = s * ncore + core;
    bool valid = work < totalWork;
    if (valid) { /* 实际工作 */ }
    SyncAll<false>();   // 越界核也走到，只同步不做工
    if (valid) { /* 下一阶段 */ }
    SyncAll<false>();
}
```

## 5. GM Workspace 交换模式

AIC 与 AIV **不能共享 L0/L1/UB**，中间结果必须经 GM workspace 交接：

```
AIC: Mmad → Fixpipe 写 GM ws slot → SyncAll<false>()
AIV: DataCopy 读 GM ws slot → elementwise 处理 → DataCopy 写另一 slot → SyncAll<false>()
```

工程实践：wsF（float slot）/ wsH（half slot）两个 workspace，编译期常量定义 slot 偏移布局，host 与 kernel 两侧保持一致；host 预写固定内容（如 Neumann 链的 identity 矩阵 `at::eye`）；一次性 elementwise/布局预处理（transpose、cumsum、decay mask、fp16 缩放）放 **host 侧 torch** 完成。

## 6. Host 启动与流竞争陷阱

```cpp
// ✅ 正确：OpCommand::RunOpApi 入队，与 torch 预处理/fill 算子同流顺序
EXEC_KERNEL_CMD(kernel_name, blockDim, args...);

// ❌ 危险：裸 aclrtlaunch_* 与 torch 算子可能流竞争
// 实例：torch::zeros 分配的输出被 fill kernel 在 kernel 写完后覆盖为全 0
aclrtlaunch_kernel(blockDim, stream, ...);
```

建议：输出分配用 `torch::empty`（kernel 负责全量写出）；若必须裸 `aclrtlaunch_*`，启动前显式 `aclrtSynchronizeDevice()`。

**build 缓存陷阱**：构建脚本检测 .so 存在会跳过 cmake——**改代码必须 `rm -rf build`**，否则"改了没生效"造成假诊断。

---

## 7. Hang 根因分类（R1-R6）

| # | 根因 | 现象 | 检查方法 |
|---|------|------|---------|
| R1 | AIV 执行 AIC 队列/指令（无核类型守卫） | 100% AICore 空转，无输出 | 所有 `TPosition::A1/B1/A2/B2/CO1` 操作是否在 `ASCEND_IS_AIC` 内 |
| R2 | `InitBuffer` 在错误的核趟执行 | 同上，启动即挂 | InitBuffer 必须与使用该 buffer 的代码同守卫 |
| R3 | AIC/AIV 两侧 `SyncAll` 次数/顺序不一致 | 死锁（部分核先到，永远等待） | 数两侧的 barrier 数；逐段打印 |
| R4 | 各核迭代次数不同 + 循环内全局 barrier | 死锁 | 循环上界必须全核一致（padded loop + `valid` 守卫） |
| R5 | 自定义 Mmad 链缺管道事件同步 | aicore exception（非 hang，有报错） | L0A/L0B 加载走完整 EnQue/DeQue；Mmad→Fixpipe、Fixpipe→GM 读之间 `PipeBarrier<PIPE_ALL>` |
| R6 | **设备被前一次 hang 的 kernel 搞 wedge** | 后续运行在张量初始化就 hang；或输出 NaN 且**位置逐 run 变化** | 换一块干净 NPU 重试，不要在挂死的卡上诊断 |

## 8. 预防式开发流程（四阶段，各阶段退出标准）

### Phase 0 — 纯 AIC 单路径 derisk（Cube 链路先对）
- `KERNEL_TYPE_AIC_ONLY` 写最小 matmul（一个 Mmad），验证 Nd2Nz/LoadData/Mmad/Fixpipe 配置。
- **退出标准**：单 matmul 数值与 torch 参考一致。

### Phase 1 — MIX 骨架（只证明"MIX 不挂"）
- 切 `KERNEL_TYPE_MIX_AIC_1_1`，加守卫，AIV 路径留空，结尾一次 `SyncAll<false>()`。
- **退出标准**：不 hang 且 AIC 结果与 Phase 0 完全一致。挂了 → 按 R1~R4 对照检查。

### Phase 2 — 配对与同步探针
- AIV 路径写 `GetBlockIdx()/GetBlockNum()` 到 GM，host 读回。
- **退出标准**：确认 pairing 与实际核数。

### Phase 3 — 逐 stage 增量（一次一个 stage，各自对参考）
- 每个 stage：AIC 写中间矩阵到 GM ws → barrier → AIV 处理 → barrier → ...
- **每个 stage 落地即验证**：host 临时 return 工作区张量，与 torch 参考逐 stage 对比。
- **必须测多 chunk（T≥128）**：单 chunk 会掩盖状态依赖路径的 bug。
- **退出标准**：全 stage 对齐后，端到端精度 PASS。

### 误诊教训（实例）
1. **改了两样东西时，hang 不一定归因于最后改的那个**——AIC_ONLY 正常 ⇒ kernel 逻辑无问题，launch 模式是病根。
2. **单 chunk 掩盖状态依赖 bug**（decay mask inclusive vs strict、漏乘 scale，单 chunk 均恰好正确）。
3. **诊断前先 torch 算法仿真**：纯 torch 复刻 kernel 分解公式再对参考，不花编译时间抓公式错误。

## 9. 双缓冲决策矩阵（何时开 / 何时不开）

判断标准：**同类存储资源上是否存在连续迭代流**，使"第 i+1 次搬运"能与"第 i 次计算"重叠。不存在迭代流的 buffer 开 num=2 是纯浪费。

| Buffer 场景 | num | 理由 | 实证 |
|-------------|-----|------|------|
| 主循环 CopyIn 队列（VECIN/A1，逐 tile 迭代） | **2** | 经典场景：MTE2 与 Vector/Mmad 重叠 | flash_attention `queL0A_/queL0B_` num=2（matmul 内层 K-loop） |
| 主循环 CopyOut 队列（VECOUT/CO1，逐 tile 迭代） | **2** | MTE3 与下一次计算重叠 | 标准 elementwise 流水线 |
| **L0C 累加器** | **1（禁止 2）** | 链式 Mmad（`cmatrixInitVal=0`）依赖**同一块 L0C 驻留累加**；双缓冲破坏累加语义 | flash_attention `queL0C_` num=1 |
| 被全局 barrier 串行化的 stage 间队列 | **1** | `SyncAll` 冲刷整条流水，stage 之间不存在可重叠的迭代流 | gdr_cv AIC 侧 5 队列**全部 num=1** |
| 单次读写的 buffer（prologue/epilogue） | **1** | 无第 i+1 次搬运 | flash_attention `outputQue1_` num=1 |
| tile 极小（搬运延迟 < 事件开销） | **1** | EnQue/DeQue 开销超过隐藏的搬运延迟，负收益 | — |

### MIX 下双缓冲与 SyncAll 的交互

两个正交层级，先层 1 再层 2：
- **层 1：核间重叠** — AIC/AIV 分工 + GM workspace 交接（Cube 计算与 Vector 计算并行）
- **层 2：核内重叠** — 队列 num=2 双缓冲（同核内相邻迭代搬运/计算并行）

规则：`SyncAll<false>()` 冲刷整条流水，跨 stage 双缓冲无意义；双缓冲只在 barrier 内部的连续迭代流里有意义（如 matmul 内层 K-loop）；UB/L1 预算紧张时优先砍双缓冲（num 2→1）而不是砍 tile。

### TBuf 需要双缓冲时

TBuf 无内建双缓冲（无事件管理）。方案一：两个独立 TBuf + 手动奇偶轮转；方案二：单 TBuf 按 offset 切半。⚠️ 手动轮转的同步责任在使用者（需 PipeBarrier/SetFlag，TBuf 不会自动插入事件依赖）。

---

## 10. 调试工具箱与一页纸速查

| 工具 | 用法 | 解决什么 |
|------|------|---------|
| `timeout --signal=KILL N` | 包裹每次上板运行 | 防止 hang 住会话 |
| `npu-smi info` AICore% | hang 时查看：100% = kernel 空转（R1~R5）；0% = host 侧挂（查 R6 wedge） | 区分 device/host hang |
| 换 NPU 重试 | hang 后设备可能 wedge；同 binary 换卡 PASS ⇒ 卡问题 | 避免在坏卡上误诊 |
| GM 探针 | 核内写 blockIdx/中间值到 GM，host 读回 | 配对验证、数据流追踪 |
| host 临时 return 工作区 | op 暂时返回 wsF/wsH 而非正式输出 | 逐 stage 数值二分 |
| torch 算法仿真 | 纯 torch 复刻 kernel 分解公式再对参考 | 不花编译时间抓公式错误 |

```
MIX hang?  → ① 守卫（AIC/AIV 各碰各的） ② barrier 两侧同数 ③ 循环次数全核一致
           → ④ Mmad 链完整队列事件 ⑤ 换块干净 NPU 再下结论
开发顺序   → AIC_ONLY 单 matmul 验证 → MIX 骨架 → 配对探针 → 逐 stage 增量（多 chunk!）
验证纪律   → 单 chunk 不算验证；stage 中间张量逐个对参考；改码必 rm -rf build
NaN 判别   → 位置逐 run 变化 = 设备 sync 污染（换卡）；固定位置系统性错误 = kernel bug
性能杠杆   → 串行链上的标量循环先上 Cube（Neumann/分块 matmul），再砍 barrier，再排流水
```
