# put-all-to-all-quant-matmul 开发指导

> **场景 ID**：`put-all-to-all-quant-matmul`（PUT 模式，AllToAll + QuantMatmul 融合，K 轴切分，UDMA 直调）。
> 本文档在 Step 3 设计冻结后编译进项目级 PLAN：`implementation_route=apace_custom`、`selected_scenario=put-all-to-all-quant-matmul`，生成 `operators/<operator_name>/docs/PLAN.md`（模板见 [`plan-template.md`](../../operator-design/plan-template.md)）。
> 官方参考：`all_to_all_quant_matmul`（CANN 内置 apace `kernel/all_to_all_quant_matmul/`），经 CMake `-I` 直引，零复制。

## 1. 有序开发动作（Step 4 只执行 PLAN）

Step 4 不重新设计、不切换候选、不扩大支持域，仅按 PLAN §9.3 `ordered_actions` 顺序执行：

| 顺序 | 动作 | 要点 | 验证 |
|:---|:---|:---|:---|
| A1 | 复用官方参考的 Scheduler/Kernel owner 基座 | 以 `all_to_all_quant_matmul` 的 `BlockScheduler`/`BlockMmad` 编排与 `{op}_udma_impl.h`（`Init`/`Run`/`RunAllToAll`/`RunMatmul`/`SetupParams`）为锚点起手，禁止从零写文件 | 静态走查，锚点逐一核对 |
| A2 | 适配通信对象 `AllToAllCommPutImpl` | 通信对象数 = 2（A data + A scale）共享同一 channel；scale 对象 winOffset = `rankSize × rankDataBytes`；UB 双 commBuf（各 512B）+ barrierBuf；AIV 先 → AIC 后 | 对照 [`communication.md`](../../fundamentals/communication.md) 四段式契约 |
| A3 | 适配计算内核（`QuantMatmulMxKernel` 或 FragmentTensor） | PUT 默认 vendor 复用 `QuantMatmulMxKernel`（Blaze `BlockMmad` + `BlockScheduler`）；改 FragmentTensor 为例外须 SCALAR 论证（[`fusion.md`](../../fundamentals/fusion.md) §6.2.2）；K 轴切分：`splitKNum`、`MakeLayoutB{}(rankSize*K, N)`、ProblemShape N = 完整 axisN | kernel 编译通过 |
| A4 | 在具体 `__mix__` 入口补全 C-ready/V-done/slot/final drain | 入口含 `KERNEL_TYPE_MIX_AIC_1_1`（禁 `__schedmode__(1)`）；CrossCore flag idx 配对（C-ready Set ↔ V-done Wait）；slot 环形偏移与 `tileMaxByteSize_` 同源；`DoWait` 末轮 final drain（self rank 跳过 Drain 但仍执行 barrier） | 冒烟：单 rank 输出非全 0 |

失败回退：实现层问题 Step 4 内修复；需改通信方向/Win/Flag 语义/支持域 → 记录 `design_issue` 回 Step 3。

## 2. 文件级契约（[MODIFY]/[REUSE]）

| 文件 | 动作 | 契约 |
|:---|:---|:---|
| `kernel/{op}_tiling_data.h` | 新建 | tiling 结构 + `CommContext`（`CommUdmaContext`+`CommUbmemContext`）；含 `localMatmul` 字段；`#pragma pack(push, 8)`+`alignas(8)`，字段顺序即 host-device 契约 |
| `kernel/{op}_udma_impl.h` | 新建（改自参考 impl） | A1/A2 落地处；`localMatmul==1` 时 `RunLocalMatmul()` → `PipeBarrier<PIPE_ALL>()` → `RunMatmul()`（推荐修复，防 aclError:507015） |
| `kernel/{op}_quant_matmul_mx_kernel.h` | 新建或直引 | A3 落地处；`LocalParams` 含 `localMatmul`/`matmulMode`/`splitKNum` |
| `src/kernel_launcher.h` | 新建 | 4 个 dtype 变体入口（E4M3E4M3/E5M2E5M2/E4M3E5M2/E5M2E4M3），每个含 `KERNEL_TYPE_MIX_AIC_1_1` |
| `src/main.cpp` | 新建（改自参考 ST） | 运行期 dtype dispatch（本 skill 新增红线/生产教训——⚠️ 官方参考 ST 本身**硬编码只 launch E4M3E4M3 入口**，4 变体定义未使用，照抄官方会被 R3 判 FAIL，须自行补 dispatch）；显式 `localMatmul` 赋值；`splitKNum` 规则：localMatmul=1 → `rankSize-1`，0/2 → `rankSize`；host 前置校验在 fork/建链前 |
| `src/root_info_exchanger.h`、`src/utils.h` | copy_and_adapt | 从参考算子 ST 复制 |
| `scripts/gen_data.py` / `verify_result.py` | 新建 | golden 语义先行（输入 K 轴切分：每卡全 M 行 × 本卡 K 段 `[M_total, Ka]`；输出 M 轴分布 `[m, N]`，见 design.md §6）；golden float32 高精度路径；固定种子；host 约束双侧校验 |
| `CMakeLists.txt` / `cases.csv` / `run.sh` | 新建 | `APACE_ROOT` 指 CANN 内置 apace；`hccl_fwk` 用 `--no-as-needed` 包裹；cases.csv 覆盖 T=1/T>1、小 rank、tail 非整除、tile 粒度扫描 |
| apace 共享层（`block/` `tiling/` `basic/` `utils/`） | [REUSE] 只读 | 禁止复制/篡改，CMake `-I` 直引；算子目录出现共享层副本即 FAIL |

## 3. 验证矩阵（PUT AllToAll 专属）

| 维度 | 要求 | 达标条件 |
|:---|:---|:---|
| 精度标准 | `verify_result.py` | `rtol=atol=1e-2`（all_to_all 系容差），多 rank × 多 shape 全 PASS |
| localMatmul 三态 | 0 / 1 / 2 分支 | localMatmul=1 含 PipeBarrier；三态精度均过 |
| dtype 覆盖 | 4 变体 | E4M3/E5M2 四组合各自 PASS（dispatch 错误特征：matched_ratio=0% + 误差 36-59%） |
| 通信对象 | data + scale 双对象 | winOffset 同源；self rank 跳过 Drain 但 barrier 保留 |
| 单轮 PUT 大小 | ⚠️ 风险提示（非硬上限） | bring-up 期先用小单轮跑通；放大后按 case 复核（无官方上限，边界未定） |
| tileCnt 策略 | 精度调试期 tileCnt=1 串行基线 → 性能期扫 {1,2,4,8,16} | 切换 tileCnt 必须重调 `GetTilingData`；档位上限受 R9 口径约束（flagId=tid 时轮次索引 < 16 且避开 SyncAll 保留区 14 → 安全上界 13，见 design.md §3.3） |
| 性能门槛（R15） | 真实大 shape × R=2/4 双档 × 三路径对标归档 | 仅 toy shape 基线 = FAIL |
| 性能采集 | msprof + 每轮 L2 flush 实接线 | flush kernel 记录数 == 轮数 |

## 4. 常见 FAIL 信号与定位

| 现象 | 根因 | 修复方向 |
|:---|:---|:---|
| 死锁 aclError:507015 | localMatmul=1 缺 PipeBarrier，或误加 schedmode | 补 `PipeBarrier<PIPE_ALL>()`；删调度属性 |
| E5M2 变体 matched_ratio=0% | dispatch 硬编码 E4M3 入口，字节流被错误模板解释 | 4 变体入口 + host 运行期 dispatch 宏 |
| 精度不收敛、golden 全错 | golden 切分轴/每卡语义写错（输入 K 轴切分、输出 M 轴分布——勿把输出当完整 C） | 先修 gen_data 再怀疑 kernel |
| 精度"假通过"、大 shape 紊乱 | PUT 数据覆盖 Win 区元数据/barrier 区 | 数据/元数据偏移同源（R14） |
| 通信时间 ≈ R × 单 target | 通信被臆造约束退化为通信对象 totalJobs=1 串行 PUT | 通信对象改 totalJobs=rankSize（R13 前半） |
| 运行期间歇 FAIL | 单轮 PUT 字节过大（经验阈值 ~512KB 量级，非硬上限） | 按 R14 风险提示复核单轮大小 + 调小 tile 粒度 |

## 5. 合规映射（本场景重点红线）

| 红线 | 本场景落点 |
|:---|:---|
| R1/R2 | 禁 `__schedmode__(1)`/`core_ratio`；每个入口含 `KERNEL_TYPE_MIX_AIC_1_1` |
| R3 | PUT = 4 dtype 变体入口 + host 运行期 dispatch |
| R4 | `block/`/`tiling/` 与 CANN 内置版本 diff 一致，零复制 |
| R5 | C-ready/V-done flag idx 跨核配对（含计数式） |
| R6 | UDMA 模式入口含 `__gm__ CommContext*` |
| R7/R8 | 禁 `AscendC::Matmul`、禁 `Hccl::*` 高阶 API |
| R11 | host 前置校验（整除/对齐/核数下限/Win 容量/flag 峰值），main.cpp 与 gen_data.py 双侧 |
| R12 | commBuf/barrierBuf 与 TPipe buffer 物理隔离（静态偏移） |
| R13 | 通信对象 `totalJobs=rankSize` 多核并行 PUT；TeamBarrier `totalJobs` 官方为 `rankSize`（`teamBarrier_.Init(buf, ctx, rankSize, GetBlockIdx())`，前 R 核映射下 CrossDevice 由 `Wait<BARRIER_DEVICE>` 内建触发）。compute-first 后 R 核映射的自研编排可采用 TeamBarrier totalJobs=1 + 显式 CrossDevice 替代，两种映射不得混用 |
| R14 | Win 数据/元数据偏移 host/kernel 同源（硬红线）；单轮 PUT 大小按 case 复核（风险提示） |
| R15/R20 | 投产级性能门槛 + L2 flush 实接线（见 §3 验证矩阵） |

> 完整 R1-R20 与 FAIL 诊断见 [`review-checklist.md`](../../review-checklist.md)；改造食谱锚点见 [`development-guide.md`](../../operator-design/development-guide.md) §3.2/§3.3。
