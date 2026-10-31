# API 分工、数据流与 Layout

> 本参考合并原 tensor-api-api-contract 与 tensor-api-dataflow-layout 的领域知识，供 ops-direct-invoke 的 2.2 开发方案使用。

## 职责边界

| 能力 | 归属 | 典型入口 |
|---|---|---|
| Tensor 声明 | Tensor API | `make_tensor` |
| Layout 管理 | Tensor API | `make_frame_layout` |
| 切片 / squeeze | Tensor API | `slice` / `squeeze` |
| 数据搬运抽象 | Tensor API | `make_copy` / `copy` |
| 矩阵计算抽象 | Tensor API | `make_mmad` / `mmad` |
| 初始化 | C API | `asc_init` |
| 管道与事件同步 | C API | `asc_sync_pipe` / `asc_sync_notify` / `asc_sync_wait` |
| 锁 | C API | `asc_lock` / `asc_unlock` |
| 原子操作 | C API | atomic 系列 |
| 缓存控制 | C API | cache_ctrl 系列 |
| 系统变量与杂项 | C API | sys_var / misc 系列 |
| Scalar/Vector/Cube/Reg compute | C API 辅助路径 | 仅在官方授权或辅助控制时使用 |

禁止项：TPipe、TQue、TQueBind、TBufPool；用 C API 承担主体计算；缺失 `asc_init` 或最终 `asc_sync_pipe(PIPE_ALL)`；事件 ID 使用 6 或 7；凭命名猜测 API 语义。

## 数据流

| 路线 | 典型路径 |
|---|---|
| Vector | GM → UB → Vector → UB → GM |
| Cube | GM → L1 → L0A/L0B → Cube → L0C → FixPipe → GM |

每段搬运必须声明源/目标 Tensor、Layout、tile shape、元素数、dtype、读写次数、同步 owner 和尾块策略。

## Layout 与内存预算

| 存储 | 容量基线 | 对齐 |
|---|---:|---:|
| UB | 256KB | 32B |
| L1 | 512KB | 32B |
| L0A | 64KB | 512B |
| L0B | 64KB | 512B |
| L0C | 256KB | 64B；mmad C 需 1024B |

推荐分形：L1 NZ，L0A NZ，L0B ZN，L0C NZ。容量必须按当前芯片手册与编译选项复核。

DESIGN 必须包含逐级内存预算表，记录 buffer 数、元素数、dtype 字节宽度、offset 和安全利用率；任一级超限先改 tile、分块或流水份数，再编码。

## mmad 与同步硬门禁

1. A 在 L0A，B 在 L0B，C 在 L0C。
2. A/B 起始地址 512B 对齐，C 起始地址 1024B 对齐。
3. m/n/k 范围与 0 值语义以当前手册为准，不凭记忆猜测。
4. M/N/K 尾块显式处理，padding 不参与 golden。
5. 首轮 K 累加保证 L0C 初始状态正确。
6. 同步事件生产者/消费者配对且 ID 不冲突。

## 事实源

- Tensor API 手册与样例：`$ASC_DEVKIT_ROOT/docs/zh/api/SIMD-API/tensor_api/`
- 官方样例：`$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/07_tensor_api/`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时运行 `../scripts/ensure_docs.py` 自动获取。引用接口前必须核对手册或头文件；冲突时以当前 SDK 头文件为准，并在设计文档记录差异。

## Pattern 与物理排列

对原始 tensor (dim0, dim1)，NZ 物理排列为：

    (dim1/C0, dim0/16, 16, C0)

其中 C0 = 32 / sizeof(dtype)，fp4 例外。

| dtype | C0 |
|---|---:|
| fp16 / bf16 | 16 |
| fp32 | 8 |
| int8 / fp8 | 32 |
| fp4x2 | 64 |
| fp8_e8m0 scale | 2 |

A/B 的 trans 与 Pattern 示例：

| 矩阵 | trans | 逻辑 shape | Pattern | 物理 shape |
|---|---|---|---|---|
| A[M,K] | false | (M,K) | NDExtLayoutPtn | (M,K) |
| A[M,K] | false | (M,K) | NZLayoutPtn | (K/C0, M/16, 16, C0) |
| A[M,K] | true | (K,M) | DNExtLayoutPtn | (K,M) |
| A[M,K] | true | (M,K) | ZNLayoutPtn | (M/C0, K/16, 16, C0) |
| B[K,N] | false | (K,N) | NDExtLayoutPtn | (K,N) |
| B[K,N] | false | (K,N) | NZLayoutPtn | (N/C0, K/16, 16, C0) |
| B[K,N] | true | (N,K) | DNExtLayoutPtn | (N,K) |
| B[K,N] | true | (K,N) | ZNLayoutPtn | (K/C0, N/16, 16, C0) |

## 物理大小与对齐

NZ 物理 buffer 大小：

    dim1Blocks = ceil(dim1 / C0)
    dim0Blocks = ceil(dim0 / 16)
    bytes = dim1Blocks * dim0Blocks * 16 * C0 * sizeof(dtype)

对齐规则：

1. NZ 外轴按 16 对齐。
2. NZ 内轴按 C0 对齐。
3. fp16/bf16 的 C0 为 16；int8/fp8 的 C0 为 32。
4. GM 端构造 NZ layout 时必须显式传正确 C0，不能依赖默认 uint16_t trait。

## 常见错误

| 现象 | 根因 |
|---|---|
| NZ 输入全 FAIL | 物理 permute 顺序错误，应为 (2,0,1,3)，不是 (0,2,1,3) |
| int8/fp8 多 tile FAIL，单 tile 通过 | GM layout 使用默认 C0=16，实际需要 C0=32 |
| 输入文件大小不匹配 | Host size 按逻辑 shape 计算，未按 NZ 物理 shape 计算 |
| 非对齐 shape FAIL | tiling 未按外轴 16 或内轴 C0 对齐 |
| trans 后近似 100% mismatch | Pattern 未随 transA/transB 同步切换 |
| K≤16 通过，K>16 多 tile FAIL | Slice 破坏基于 fullK 的 NZ stride |
