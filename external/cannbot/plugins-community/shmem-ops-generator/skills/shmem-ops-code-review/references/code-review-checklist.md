# SHMEM 算子代码走读检查项

> **仓内路径**：下文 `examples/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

本文列出 SHMEM 算子实现与设计一致性走读的分类检查项。每项包含描述、错误示例和正确做法。

## 1. CMake 与构建

### 1.1 构建模式匹配

【描述】`CMakeLists.txt` 的构建模式必须与 `design.md` 的 `compile_contract.build_mode` 一致。独立工程（默认模式）显式引用 `src/` 下源码并设置 include 目录；仓内 in-tree example 使用 `aclshmem_add_collective_example()` 宏。

【错误做法】独立工程中依赖仓内 example 宏查找扁平布局源码，导致 `src/` 下的文件不可编译。

【正确做法】独立工程 CMake 显式列出 `src/main.cpp`、`src/<op_name>_kernel.cpp` 并设置 `include_directories(src)` 和 `examples/utils` 路径；仓内 example 只写 `aclshmem_add_collective_example(<op_name>)`，由 `examples/CMakeLists.txt` 统一管理依赖。

### 1.2 Target 命名

【描述】编译产物名称应与 `design.md` 的 `op_name` 一致，便于 `scripts/run.sh` 和调试引用。

## 2. 通信数据面

### 2.1 跨 PE 搬运接口

【描述】跨 PE 数据搬运必须使用 `aclshmem_*` 或公开 `aclshmemx_*` 接口，禁止 Device kernel 中用 `DataCopy` 直接读写远端 PE 地址。

【错误做法】
```cpp
// kernel 中直接 DataCopy 远端地址
DataCopy(dst_gm, remote_gm_addr, count);
```

【正确做法】
```cpp
// 使用 SHMEM RMA 接口
aclshmemx_mte_put_nbi(remote_symm, local_gm, ub_buf, size, pe);
```

### 2.2 Transport API 与设计一致

【描述】实际使用的传输 API（MTE put/get、RDMA put/get、SDMA）必须与 `design.md` 的 `transport` 模块一致。不能设计写 RDMA 实际用 MTE，或反之。

### 2.3 通信方向正确

【描述】put 和 get 的源/目标不能搞反。put 是本地写远端，get 是远端读到本地。

## 3. 同步正确性

### 3.1 Signal/Wait 配对

【描述】每个 `signal` 必须有对应的 `wait` 或 `test`。signal 值必须与 wait 的期望值匹配（magic/epoch 策略需一致）。

【错误做法】
```cpp
// 发送端 signal(magic=1)，接收端 wait(magic=2)
aclshmemx_signal_op(remote_sig, 1, SIGNAL_SET, pe);
// ...
aclshmem_signal_wait_until(local_sig, ACLSHMEM_CMP_EQ, 2);  // 永远等不到
```

### 3.2 Phase 边界同步

【描述】相邻 phase 之间的同步方式必须与 `design.md` 的 `sync` 模块一致。设计写 signal/wait 的不能实现为 Host barrier；设计写 device barrier 的不能省略。

### 3.3 Quiet/Fence 使用

【描述】如果 design 要求 quiet 保证远端写入可见性，实现中不能省略。单引擎 MTE 路径用 `aclshmemx_mte_quiet`；多引擎混用或收尾用 `aclshmem_quiet`。`fence` 当前实现等价于全引擎 `quiet`。

### 3.4 Internal / Deprecated API

【描述】custom-ops 新算子 **禁止** 调用 `aclshmemi_*`（见 [internal-api-boundary.md](../../shmem-ops-code-gen/references/internal-api-boundary.md)）。**禁止** 新增 `aclshmemx_barrier_all_vec`（用 `aclshmem_barrier_all`）。核间同步用 `AscendC::SyncAll`，勿用 `aclshmemi_barrier_core_soft`。

【错误做法】
```cpp
aclshmemi_barrier_core_soft();
aclshmemx_barrier_all_vec();
```

【正确做法】
```cpp
AscendC::SyncAll<true>();
aclshmem_barrier_all();
aclshmemx_mte_quiet();
```

## 4. 分核与并发

### 4.1 Block Dim 匹配

【描述】`launch` 时的 `block_dim` 必须与 `design.md` 的 `schedule.core_partition` 一致。设计要求多 block 并发时，不能固定 `block_dim=1`。

【错误做法】设计要求 `2*n_pes` 核并发，但代码写死 `block_dim=1`。

【正确做法】
```cpp
uint32_t block_dim = (trans_size < threshold) ? BLOCK_NUM_SMALL : 2 * n_pes;
```

### 4.2 Compute/Comm 分核

【描述】如果设计有 compute core 和 comm core 分离的策略，kernel 中的 `GetBlockIdx()` 分支必须正确实现分核逻辑。

## 5. 内存与 Buffer

### 5.1 Symmetric 分配顺序

【描述】所有 PE 的 symmetric `aclshmem_malloc` 调用顺序和大小必须完全一致，否则地址偏移错乱。

### 5.2 Buffer 对齐

【描述】SHMEM 对称内存默认 16 字节对齐。如果 design 有更高对齐要求（如 512 字节用于 DMA），必须使用 `aclshmem_align`。

### 5.3 UB Buffer 溢出

【描述】kernel 中使用的 UB buffer 大小不能超过引擎配置的 UB 上限。allgather 等 example 常用 `190*1024` 作为 UB DMA 上限。

## 6. Tile/Chunk/Tail

### 6.1 Tail 处理

【描述】shape 不整除 tile/chunk 时，最后一个 chunk 的大小处理必须正确。不能假设所有 chunk 等大。

【错误做法】
```cpp
// 未处理 tail，最后一个 core 可能越界
int per_core = total / core_num;
```

【正确做法】
```cpp
int per_core = total / core_num;
if (core_idx == core_num - 1) {
    per_core = total - per_core * core_idx;  // tail 修正
}
```

### 6.2 Chunk 循环边界

【描述】搬运循环中的 `while (remaining >= chunk_size)` 后必须处理剩余部分。

## 7. Host 入口

### 7.1 Main.cpp 职责边界

【描述】`main.cpp` 只做初始化、数据拷贝、kernel launch、结果回收和清理。复杂的 tiling、route、packing 逻辑必须在独立 `.cpp/.h` 中。

### 7.2 单进程单 PE

【描述】`main.cpp` 不能 fork/spawn 多 PE。一个进程只绑定一个 device，初始化一个 `my_pe`。

## 8. 性能路径

### 8.1 统一 kernel 路径（默认）

【描述】默认 **single path** 一套 kernel + UB 内分块。仅当 design 明确记录 profiling ≥5% 收益时，才允许 `small`/`big` 或 `*_small`/`*_large` 并行路径。

【错误做法】未证明收益即维护 `if (0)` / `INT64_MAX` 禁用的 dead `large` 路径与 unified 路径长期并存。

【正确做法】合并为 unified；外层 tile 仅因 GVA/symm 容量保留。见 implementation-boundary、code-patterns §6.4。

### 8.2 不必要的 GM Scratch

【描述】如果可以做 UB streaming reduce，不应先写 GM 再读 GM。检查是否有多余的 GM 中转。
