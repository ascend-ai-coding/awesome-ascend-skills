# 参考工程（all_to_all_matmul/）改造食谱

本文档是 Agent 在开发阶段的实操指南：从基底工程 `references/foundations/blaze-shmem/all_to_all_matmul/` 复制起手，按 `[REUSE]` / `[MODIFY]` 标记定点改造。

读完本文档应能回答：
- 哪些文件不能动？
- 改不同种类的 MC2 算子（AllReduce+Matmul、AllGather+Matmul、不同 dtype），分别要改哪些文件？
- 每个文件改动的典型 diff 是什么？

---

## 1. 工程总览

```
references/foundations/blaze-shmem/all_to_all_matmul/
├── CMakeLists.txt              # [REUSE]   构建脚本，含 NPU_ARCH 校验、SHMEM/Blaze 链接
├── run.sh                      # [REUSE]   一键脚本：cmake + gen_data + 跑算子 + verify
├── cmake/
│   ├── ascend.cmake            # [REUSE]   定位 ASCEND_HOME_PATH、设 BISHENG 编译器
│   ├── shmem.cmake             # [REUSE]   clone third_party/shmem（v1.5.0）并构建
│   └── tensor_api.cmake        # [REUSE]   clone third_party/tensor_api（asc-devkit）
├── scripts/
│   ├── gen_data.py             # [MODIFY]  按 dtype/shape 改
│   └── verify_result.py        # [MODIFY]  按 dtype/容差改
├── src/
│   ├── all_to_all_matmul.cpp   # [REUSE 骨架 / MODIFY 入口] host 程序
│   └── utils.h                 # [REUSE]   ACL_CHECK / ReadFile / WriteFile 等宏
├── common/                     # [REUSE]   host/kernel 通用工具
│   ├── kernel_utils/
│   ├── host_utils/
│   └── ...
├── include/
│   ├── kernel/
│   │   ├── all_to_all_comm_udma.h      # [MODIFY]  通信层（改通信原语时动）
│   │   ├── all_to_all_matmul_impl.h    # [MODIFY]  通算融合主类（改流水编排时动）
│   │   ├── qbmm_mx_kernel.h            # [MODIFY]  Blaze Kernel 包装（改 Scale / dtype 时动）
│   │   └── heavy_kernels.h             # [REUSE]   性能模式专用（boost + cache_flush）
│   ├── tiling/
│   │   ├── all_to_all_matmul_tiling_data.h  # [MODIFY]  TilingData 字段
│   │   ├── quant_matmul_mx_tiling_swat.h    # [REUSE]  SWAT tiling 算法
│   │   ├── quant_matmul_tiling_data.h       # [REUSE]  基础 TilingData 结构
│   │   ├── quant_matmul_tiling_base.h       # [REUSE]  Tiling 基类
│   │   └── quant_matmul_tiling_common.h     # [REUSE]  公共常量
│   ├── block/                  # [REUSE]   Blaze block 级
│   ├── tile/                   # [REUSE]   Blaze tile 级
│   ├── policy/dispatch_policy.h # [REUSE]  Blaze dispatch policy
│   └── utils/constant.h        # [REUSE]   公共常量
└── third_party/                # 首次 cmake 配置时自动 git clone（不入版本控制）
    ├── shmem                   # gitcode.com/cann/shmem v1.5.0（cmake/shmem.cmake 拉取）
    └── tensor_api              # gitcode.com/cann/asc-devkit（cmake/tensor_api.cmake 拉取，与 blaze skill 共享来源）
```

**原则**：`[REUSE]` 文件常规不动；`[MODIFY]` 文件按需动。改动量越大，编译/精度风险越高。

---

## 2. 改造场景速查

不同 MC2 算子需要改动的文件清单：

| 场景 | 通信层 | 计算层 | Tiling | scripts | host 入口 |
|------|--------|--------|--------|---------|----------|
| **换 dtype**（如 MX FP8 → BF16） | 不动 | `qbmm_mx_kernel.h`（去 scale）、`all_to_all_matmul_impl.h`（类型参数） | `quant_matmul_tiling_data.h`（字段） | 改 dtype | 改类型实参 |
| **换 shape**（M/N/K） | 不动 | 不动 | 不动（host 动态算） | 改 gen_data | 不动 |
| **换通信原语**（AllToAll → AllReduce） | `all_to_all_comm_udma.h`（重写为 `AllReduceComm`） | `qbmm_mx_kernel.h`（reduce 逻辑） | `all_to_all_matmul_tiling_data.h`（字段名） | 改 verify | 改 kernel 调用名 |
| **加 bias** | 不动 | `qbmm_mx_kernel.h`（bias 地址）、`all_to_all_matmul_impl.h`（params） | 加 bias 字段 | 改 gen_data | 改 bias 参数 |
| **换卡数**（4 → 8） | 不动 | 不动 | 不动 | 改 gen_data | 第 4 参改 8 |
| **改流水深度**（bufferSize 4 → 8） | 不动 | 不动 | `all_to_all_matmul_tiling_data.h`（字段值） | 不动 | 改 `bufferSize` 字段 + SHMEM 空间 |
| **调通算并行**（tileCnt 1→N） | 不动 | 不动 | 不动 | 不动 | 改 `headMSize` 计算（详见 [`pipeline_tuning.md`](../../shared/pipeline_tuning.md)） |

---

## 3. 关键文件改造食谱

### 3.1 `src/all_to_all_matmul.cpp`（host 入口）

**典型改动**：

1. **重命名**（替换 `all_to_all_matmul` → `{op_name}`）：

```bash
# 全局替换（注意 CMakeLists 目标名也要同步）
sed -i 's/all_to_all_matmul/{op_name}/g' src/{op_name}.cpp CMakeLists.txt run.sh
```

2. **新增输入参数**（如加 bias）：

```cpp
// 修改 parseArguments 和 runAllToAllMatmul 的签名
int runAllToAllMatmul(int rankNum, int rankId, int m, int k, int n,
                     GM_ADDR bias,  // 新增
                     const std::string& mode);
```

3. **改 mode 分支**（如增加"dump 中间数据"模式）：

```cpp
if (mode == "precision") {
    // ...
} else if (mode == "perf") {
    // ...
} else if (mode == "debug") {
    // 新增：跑一次 + dump 所有 SHMEM 内容
}
```

4. **改 perf 主循环的 L2 flush 大小**（如果 B 矩阵比 256 MB 大得多）：

```cpp
constexpr int64_t CACHE_FLUSH_ELEM_COUNT = 128LL * 1024 * 1024;  // 256 MB
// 若 B 矩阵 > 256 MB，按 1.5x B 大小调大
```

### 3.2 `include/kernel/all_to_all_comm_udma.h`（通信层）

**典型改动**：

1. **重命名 class**（`AllToAllComm` → `AllReduceComm` 等）：

```cpp
template<typename XType>
class AllReduceComm {  // 原 AllToAllComm
public:
  // ...
};
```

2. **新增通信方法**：

```cpp
template<typename XType>
__aicore__ inline void AllReduceComm<XType>::ReduceBuffer(uint32_t bufferId) {
  // bufferId 处的数据 layout: [rankSize][reduceLen]
  // 用 AIV Vector 做 reduce（参考 AscendC::Add）
  // ...
}
```

3. **改 Put 策略**（如只 Put 给特定 rank，AllToAll 变 ReduceScatter）：

```cpp
template<typename XType>
__aicore__ inline void AllReduceComm<XType>::PutToTargetRank(
    uint32_t targetRank, uint64_t srcMOffset, uint64_t mSize, uint32_t bufferId) {
  if (AscendC::GetBlockIdx() == 0) {  // 只用一个 Block 发
    PutSegmentToRank(targetRank, srcMOffset, mSize, bufferId);
  }
  BarrierAll();
}
```

**注意事项**：
- 不要改 `aclshmemx_udma_put_nbi` 的调用方式（src/dst/size 语义固定）；
- `aclshmemx_udma_quiet(remoteRank)` 必须保留；
- `BarrierAll()` 用 `aclshmem_barrier_all()`，不要换成 HCCL 同步。

### 3.3 `include/kernel/all_to_all_matmul_impl.h`（通算融合主类）

**典型改动**：

1. **改流水深度**（bufferSize=4 → 8）：

```cpp
// host 侧
tilingData.commTilingData.tileCnt;
```

**注意事项**：
- 改完每条 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 都要核对 idx 配对；
- `MatmulProcess` 的 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx)` 始终在 kernel 外部调用，不要移到 kernel 内部（会破坏 1:1 配对）。

### 3.4 `include/kernel/qbmm_mx_kernel.h`（Blaze Kernel 包装）

**典型改动**：

1. **去 scale（非量化场景）**：

```cpp
// 原（含 scale）：
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>;
// ...
auto gmScaleA = Te::MakeTensor(...);
mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, ...);

// 改（去 scale）：
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<NONE_FULL_LOAD_MODE>;
// ...
mmadOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, ...);  // 去 scale 参数
```

同时换 BlockMmad 模板：`block_mmad_qbmm_mx.h` → `block_mmad.h`。

2. **加 bias**：

```cpp
// Init 中
if (isBias_) {
  biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
}

// ProcessSingleBatch 中
auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0L, nPos),
    Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));
mmadOp_(..., gmBlockBias, ...);
```

3. ** B 的切分维度**（按 N 而非按 K 切，或反之）：

```cpp
// ProcessSingleBatch 中 rank 循环里 gmB 的 Slice 偏移。
// 参考工程按 rank 切 K 轴段（每卡持自己那 K 段的 B）：
auto gmBlockB = gmB.Slice(Te::MakeCoord(rank * Get<MNK_K>(problemShape), nPos), ...);
// 若改按 N 轴切（如 AllGather+Matmul），改 nPos 偏移逻辑 + 调整 B 矩阵 layout。
```

### 3.5 `include/tiling/all_to_all_matmul_tiling_data.h`（TilingData 字段）

**典型改动**：

1. **加字段**：

```cpp
struct AllToAllCommTilingData {
    uint32_t tileCnt;
    uint32_t bufferSize;
};
```

2. **改字段类型**（如 bufferSize 改 uint64_t）：

```cpp
struct AllToAllCommTilingData {
    uint32_t tileCnt;
    uint64_t bufferSize;
};
```

**注意**：改字段后必须同步改 host 侧赋值和 device 侧读取。

---

## 4. CMakeLists 改造

新算子的 CMakeLists 通常只改两处：

```cmake
# 1. 项目名
project({op_name} LANGUAGES C CXX ASC)

# 2. 目标名 + 源文件
add_executable({op_name} src/{op_name}.cpp)
# 原：add_executable(all_to_all_matmul src/all_to_all_matmul.cpp)
```

**不要动**：
- `NPU_ARCH` 校验逻辑（dav-3510 限定）；
- `include()` 的 shmem.cmake / tensor_api.cmake；
- `target_include_directories` 的路径列表（tensor_api 由 `cann_samples::tensor_api` 提供，来自 asc-devkit clone）；
- `target_compile_options` 的 `-xasc --npu-arch=dav-3510`；
- `target_link_libraries` 的 `cann_samples::tensor_api` / `cann_samples::shmem`。

---

## 5. run.sh 改造

通常只改默认 shape（如果新算子的典型 shape 不同）：

```bash
# 原：
M="${1:-2048}"
K="${2:-8192}"
N="${3:-3584}"
RANK="${4:-4}"

# 改（如新算子典型 shape 是 4096x4096x4096）：
M="${1:-4096}"
K="${2:-4096}"
N="${3:-4096}"
RANK="${4:-8}"
```

---

## 6. 改造清单（Developer 工作流）

Developer 应按以下顺序工作：

```
1. cp -r references/foundations/blaze-shmem/all_to_all_matmul operators/{op_name}

2. 改名（src/all_to_all_matmul.cpp → src/{op_name}.cpp）
   全局替换 all_to_all_matmul → {op_name}

3. 按 DESIGN.md 的 [MODIFY] 清单定点改造：
   - 通信层（include/kernel/all_to_all_comm_udma.h）
   - 计算层（include/kernel/all_to_all_matmul_impl.h, qbmm_mx_kernel.h）
   - Tiling（include/tiling/）
   - scripts（gen_data.py, verify_result.py）

4. 冒烟编译：
   cmake -S . -B build -DNPU_ARCH=dav-3510
   cmake --build build -j

5. 冒烟测试（precision 模式）：
   bash run.sh 512 4096 3072 4  # 小 shape 冒烟

6. 全量精度测试：
   bash run.sh  # 默认 shape

7. 把改动写入 PLAN.md 的"实际改动清单"，供代码审查核对
```

**禁止**：
- 不复制基底工程、从零写文件；
- 改 `[REUSE]` 标记的文件（除非 Architect 在 DESIGN.md 显式说明）；
- 跳过冒烟直接上全量。

---

## 7. 代码审查清单

代码审查时应检查：

| 检查项 | 验收条件 |
|--------|---------|
| 改动文件清单与 DESIGN.md 一致 | 与基底工程对比，改动文件清单与设计文档一致 |
| `[REUSE]` 文件未被修改 | `[REUSE]` 文件不出现在 diff 中 |
| CMakeLists 目标名与文件名一致 | `add_executable` 目标名与算子名匹配 |
| run.sh 默认参数合理 | M/K/N/RANK 默认值正确 |
| kernel 入口函数名与 host 调用一致 | `__global__` 入口函数名与 main.cpp 中的 LaunchKernel 调用一致 |

---

## 8. 后续阅读

| 想了解 | 读 |
|--------|---|
| 各 Step 的具体动作 | `workflow_integration.md` |
| MC2 整体架构 | `mc2_architecture.md` |
| SHMEM/UDMA 细节 | `comm_shmem.md` |
| Blaze 细节 | `matmul_blaze.md` |
| 性能采集 | `profiling_mc2.md` |
