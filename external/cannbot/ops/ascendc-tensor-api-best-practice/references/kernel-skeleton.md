# Kernel 端到端骨架（从 API 字典到能编译的 kernel）

本参考在 Step 2/3（API 调查、Kernel Design）与 Step 4（实施）节点按 implementation_route=tensor_api_custom 条件加载。它补齐 `tensor-api-reference.md` 只给签名、不给「怎么串成一个能跑的 kernel」的空隙。

> 实证源：`07_tensor_api/` 下 `custom_op_tensor_api`（Cube+自定义 Copy Operation）、`matmul_bias_fusion`（`__mix__(1,2)` 融合）、`experimental/reg_vector_compute/copy_vector`（Vector）。

## 0. 一个关键澄清：Tensor API 路线没有 `add_custom`

Tensor API 路线的 kernel 是**带修饰符的普通函数 + host 侧 `<<<>>>` 直接启动**，不是传统 AscendC 的 `add_custom` 注册算子：

```cpp
__global__ __cube__ void my_kernel(__gm__ half* a, __gm__ half* b, __gm__ half* c) { ... }

// host 侧 main()：
my_kernel<T...><<<num_blocks, 0, stream>>>(a_device, b_device, c_device);
```

`add_custom` 是传统 AscendC（`00_introduction` / `02_features`）的自定义算子注册入口，**不在本路线使用**。此前 `sample-router.md` 的「make_tensor/add_custom 封装」表述不准确，本路线应为「make_tensor/make_copy/make_mmad + `__global__ __cube__/__vector__` 直调」。

## 1. Kernel 三种形态（按执行单元）

| 修饰符 | 执行单元 | 适用 | 样例 |
|---|---|---|---|
| `__global__ __cube__` | AIC（Cube 单元） | 纯 Matmul/Cube 算子 | `custom_op_tensor_api`、`matmul_tensor_api` |
| `__global__ __vector__` | AIV（Vector 单元） | 纯 Vector 算子 | `reg_vector_compute/*` |
| `__global__ __mix__(a, b)` | AIC + AIV 混合（a 个 Cube + b 个 Vector） | Cube+Vector 融合 | `matmul_bias_fusion`（`__mix__(1, 2)`） |

- `__mix__(a,b)` 内用 `ASCEND_IS_AIC` 与 `ASCEND_IS_AIV` 两个独立 `if` 分支分别标记 AIC/AIV 代码路径（落地见 [fusion-design.md](fusion-design.md)）。
- 不要从函数名或相似工程猜测修饰符，以当前待仿样例为准。

## 2. 地址空间修饰符（片上 buffer 声明）

| 修饰符 | 存储 | 用途 |
|---|---|---|
| `__gm__` | Global Memory | kernel 入参指针 |
| `__ubuf__` | Unified Buffer (UB) | Vector 工作区 |
| `__cbuf__` | L1 Buffer | Cube 的 A/B tile 暂存 |
| `__ca__` | L0A Buffer | Cube 输入 A |
| `__cb__` | L0B Buffer | Cube 输入 B |
| `__cc__` | L0C Buffer | Cube 累加输出 |
| `__biasbuf__` | Bias buffer | Cube bias |

声明例（Cube 路径）：

```cpp
__cbuf__ half l1_a_buf[base_m * base_k];
__ca__   half l0_a_buf[base_m * base_k];
__cb__   half l0_b_buf[base_k * base_n];
__cc__   float l0_c_buf[base_m * base_n];
__biasbuf__ float bias_buf[base_n];
```

Vector 路径用 `__ubuf__`。（这些是编译器内建关键字，不是 API。）

## 3. `make_mem_ptr` 与 `location`

`make_mem_ptr` 把裸指针包装成可消费的 memory pointer，location 可显式指定或从 buffer 类型推断：

```cpp
namespace asc::te::location { struct gm; struct ub; struct l1; struct l0a; struct l0b;
                              struct l0c; struct bias; struct fixbuf; struct ssbuf; /* 等 */ }

// GM 端：显式指定 location（kernel 入参指针）
make_mem_ptr<location::gm>(a);
// 片上：从 buffer 修饰符自动推断，省略 location
make_mem_ptr(l1_a_buf);
```

`make_frame_layout` 两种形态（dtype 可显式或从 buffer 推断，GM 端构造分形 layout 时必须显式传正确 dtype 以得到正确 C0）：

```cpp
make_frame_layout<nd_ext_layout_ptn>(m, k);           // dtype 从 buffer 推断
make_frame_layout<nz_layout_ptn, half>(base_m, base_k); // 显式 dtype（GM 端 NZ 必需）
```

## 4. 最小可编译骨架（Cube 路径，单 tile、无尾块）

```cpp
#include <cstdint>
#include "acl/acl.h"           // host 侧 ACL（main 里用）
#include "c_api/asc_simd.h"    // C API：asc_init / asc_sync_pipe / asc_lock
#include "tensor_api/tensor.h" // Tensor API 唯一公共入口

template <typename T>
__global__ __cube__ void matmul_kernel(__gm__ T* a, __gm__ T* b, __gm__ float* c)
{
    asc_init();                    // 1. 必须最先调用
    using namespace asc::te;

    __cbuf__ T   l1_a[base_m * base_k];
    __cbuf__ T   l1_b[base_k * base_n];
    __ca__   T   l0_a[base_m * base_k];
    __cb__   T   l0_b[base_k * base_n];
    __cc__   float l0_c[base_m * base_n];

    auto gm_a = make_tensor(make_mem_ptr<location::gm>(a), make_frame_layout<nd_ext_layout_ptn>(m, k));
    auto gm_b = make_tensor(make_mem_ptr<location::gm>(b), make_frame_layout<nd_ext_layout_ptn>(k, n));
    auto gm_c = make_tensor(make_mem_ptr<location::gm>(c), make_frame_layout<nd_ext_layout_ptn>(m, n));
    auto l1_a_t = make_tensor(make_mem_ptr(l1_a), make_frame_layout<nz_layout_ptn, T>(base_m, base_k));
    auto l1_b_t = make_tensor(make_mem_ptr(l1_b), make_frame_layout<nz_layout_ptn, T>(base_k, base_n));
    auto l0_a_t = make_tensor(make_mem_ptr(l0_a), make_frame_layout<nz_layout_ptn, T>(base_m, base_k));
    auto l0_b_t = make_tensor(make_mem_ptr(l0_b), make_frame_layout<zn_layout_ptn, T>(base_k, base_n));
    auto l0_c_t = make_tensor(make_mem_ptr(l0_c), make_frame_layout<nz_layout_ptn>(base_m, base_n));

    const auto mmad_atom = make_mmad(make_mmad_operation{}, mmad_trait_default{});

    constexpr uint8_t mutex_id = 0;
    // 2. MTE2：GM -> L1（用 lock/unlock 串起流水时序）
    asc_lock(pipe_t::PIPE_MTE2, mutex_id);
    copy(l1_a_t, gm_a.slice(make_coord(0, 0), make_shape(base_m, base_k)));
    copy(l1_b_t, gm_b.slice(make_coord(0, 0), make_shape(base_k, base_n)));
    asc_unlock(pipe_t::PIPE_MTE2, mutex_id);

    // 3. MTE1：L1 -> L0A/L0B
    asc_lock(pipe_t::PIPE_MTE1, mutex_id);
    copy(l0_a_t, l1_a_t);
    copy(l0_b_t, l1_b_t);
    asc_unlock(pipe_t::PIPE_MTE1, mutex_id);

    // 4. Cube：mmad（首轮 is_first_k=true 初始化 L0C）
    asc_lock(pipe_t::PIPE_M, mutex_id);
    mmad(mmad_atom.with(mmad_params{base_m, base_n, base_k, unit_flag_mode::disable, true}), l0_c_t, l0_a_t, l0_b_t);
    asc_unlock(pipe_t::PIPE_M, mutex_id);

    // 5. Fixpipe：L0C -> GM
    asc_lock(pipe_t::PIPE_FIX, mutex_id);
    copy(gm_c.slice(make_coord(0, 0), make_shape(base_m, base_n)), l0_c_t);
    asc_unlock(pipe_t::PIPE_FIX, mutex_id);

    asc_sync_pipe(PIPE_ALL);       // 6. 必须最后调用，排空所有在途流水
}
```

> 骨架内的 `base_m/base_k` 等为编译期常量或模板参数，真实场景见 §5-6 的 Tiling 循环与尾块。

## 5. 关键要素逐条说明

1. **`asc_init()` 与 `asc_sync_pipe(PIPE_ALL)`**：一进一出，缺一即门禁失败（见 `api-dataflow-layout.md` 禁止项）。
2. **同步用 C API**：Tensor API 不接管事件同步。`asc_lock/unlock` 保护各流水（PIPE_MTE2/MTE1/M/FIX/V/MTE3）的 buffer 访问时序；跨块流水复用用 `asc_sync_notify/wait`（见 `sync-and-lifecycle.md`）。
3. **`mmad_params` 五参**：`{M, N, K, unit_flag_mode, is_first_k}`。`is_first_k=true` 表示首轮 K 累加（L0C 由 bias/零初始化），后续 tile 传 `false` 做累加。`unit_flag_mode` 是 UnitFlag 优化开关（见 `performance-patterns.md`）。
4. **`copy(dst, src)`**：`dst` 在前；`copy(gm_c.slice(...), l0_c_t)` 是 L0C→GM 方向（Fixpipe 写出）。
5. **`copy(atom, dst, src)`**：自定义 Copy Operation 或需要指定 trait 时用带 atom 的重载（见 `custom_op_tensor_api` 的 `copy_slice_gm_to_l1`）。
6. **多核偏移**：kernel 用 `block_idx` 算本核 GM 偏移（`const uint32_t m_idx = block_idx % m_iter;`），核数由 host 侧 `<<<num_blocks, 0, stream>>>` 给定。

## 6. 运行时 Tiling 与尾块

`custom_op_tensor_api` 用**编译期 constexpr**（shape 全部模板参数）；`matmul_bias_fusion` 用**运行时 Tiling**：kernel 从 GM 读 Tiling 结构体再算 tile 循环，尾块用 `(axis % base) ? axis % base : base` 显式处理：

```cpp
const uint32_t m_loop = (core_m + base_m - 1) / base_m;
const uint32_t tile_m  = (core_m % base_m != 0) ? core_m % base_m : base_m; // 尾块显式
```

Tiling 四场景选型见 `tiling-strategy.md`，mmad 尾块 padding 不参与 Golden（见 `api-dataflow-layout.md` mmad 门禁）。

## 7. Vector 路径骨架差异

```cpp
__global__ __vector__ void add_vector(__gm__ T* src0, __gm__ T* src1, __gm__ T* dst)
{
    asc_init();
    using namespace asc::te::experimental;   // Reg Vector 入口
    __ubuf__ T ub_buf[block_len];
    auto gm_s0 = make_tensor(make_mem_ptr<location::gm>(src0), make_frame_layout<nd_ext_layout_ptn>(len));
    auto ub_d  = make_tensor(make_mem_ptr(ub_buf), make_frame_layout<nd_ext_layout_ptn>(block_len));

    constexpr uint8_t lock_id = 0;
    asc_lock(pipe_t::PIPE_MTE2, lock_id);      // GM -> UB
    copy(ub_d, gm_s0);
    asc_unlock(pipe_t::PIPE_MTE2, lock_id);

    asc_lock(pipe_t::PIPE_V, lock_id);          // Vector 计算
    auto coord = make_coord(0);
    auto mask  = experimental::update_mask<T>(len);
    auto lhs   = experimental::load(ub_d, coord).with_mask(mask);
    auto rhs   = experimental::load(ub_d, coord).with_mask(mask);  // 示意
    experimental::store(ub_d, coord, lhs + rhs);
    asc_unlock(pipe_t::PIPE_V, lock_id);

    asc_sync_pipe(PIPE_ALL);
}
```

细节（`update_mask`/`load`/`store`/`mem_bar`/寄存器上限/VF 循环规范）见 `experimental-reg-vector.md` 与 `vector-performance-patterns.md`。

## 8. Host 侧启动与 ACL 生命周期（direct launch）

默认工程形态与官方 `07_tensor_api` 样例一致：kernel 和 host `main()` 位于同一
`<op_name>.asc`，CMake 以 `LANGUAGES ASC CXX` 构建单个 `demo` 可执行文件。只有
用户明确要求或目标工程已有约束时，才拆分 .cpp/host 入口。

host 侧 `main()` 的最小序列（完整见 `launcher-development.md`）：

```cpp
aclInit(nullptr);
aclrtSetDevice(0);
aclrtStream stream = nullptr; aclrtCreateStream(&stream);
aclrtMalloc(&a_dev, size, ACL_MEM_MALLOC_HUGE_FIRST);       // 分 input/output
aclrtMemcpy(a_dev, size, a_host, size, ACL_MEMCPY_HOST_TO_DEVICE);  // H2D
my_kernel<T...><<<num_blocks, 0, stream>>>(a_dev, b_dev, c_dev);     // launch
aclrtSynchronizeStream(stream);
aclrtMemcpy(c_host, size, c_dev, size, ACL_MEMCPY_DEVICE_TO_HOST);  // D2H
// ... 反序释放：free -> destroy stream -> reset device -> aclFinalize
```

## 9. Kernel 参数与核 ID 纪律

- kernel 入参指针本身不可改，输入参数指向内容不可改，输出参数指向内容可改；Tiling 数据指针与内容都不可改（见 `verification-and-debug.md` §4.4）。
- 核 ID 用 `block_idx`（纯 Vector/Cube/Mix(1,1)）；Mix(1,2) 的 Vector 核用 `block_idx * asc_get_sub_block_num() + asc_get_sub_block_id()`（见 `c_api/utils/sys_var.h` 注释）。核数函数为 `asc_get_block_num()`。
- `block_idx`/`block_num` 自 9.2.0 起标注 deprecated（建议 `asc_get_block_idx()`/`asc_get_block_num()`，POV 至 2027/09/07），以当前头文件为准。

## 门禁

1. `asc_init()` 首、`asc_sync_pipe(PIPE_ALL)` 尾，缺一即 fail。
2. 禁止直接 include `impl/` 或内部子目录头。
3. 片上 buffer 用 `__ubuf__/__cbuf__/__ca__/__cb__/__cc__/__biasbuf__` 声明，`make_mem_ptr` 包装后构造 tensor。
4. 首轮 mmad `is_first_k=true`，后续累加 `false`；尾块显式处理且 padding 不参与 Golden。
5. 每个新 API/修饰符用法必须对照当前头文件与官方样例，不凭记忆。

## 事实源

- `$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/07_tensor_api/{custom_op_tensor_api, matmul_bias_fusion, experimental/reg_vector_compute/copy_vector}/`
- `$ASC_DEVKIT_ROOT/include/tensor_api/`（tensor.h、tensor/pointer.h、atom/cube/mmad_atom.h、arch/cube/mmad_trait.h、utils/utils.h）
- `$ASC_DEVKIT_ROOT/include/c_api/`（asc_simd.h、sync/sync.h、utils/sys_init.h、utils/sys_var.h）

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时在 skill 根目录运行 `scripts/ensure_docs.py` 自动获取。
