# SHMEM Internal API 边界（custom-ops 必读）

> **API 分层**（custom-ops 可用）：
> - `aclshmem_*` — 标准公开 API
> - `aclshmemx_*` — **扩展公开 API**（MTE/SDMA/RDMA 数据面、`mte_quiet` 等，**允许且常用**）
> - `aclshmemi_*` / `shmemi_*` 实现符号 — **internal**，新算子 **禁止** 直接调用
>
> legacy `examples/` 可能仍调用 internal 接口；**禁止**照抄到 custom-ops。

## 1. 禁止直接调用的 internal 符号（新算子）

| 符号 | 位置 | 替代方案 |
| --- | --- | --- |
| `aclshmemi_barrier_core_soft` | `shmemi_device_cc.h` | `AscendC::SyncAll`（见 [core-allocation.md §6.4](../../shmem-ops-design/references/core-allocation.md)） |
| `aclshmemi_barrier_core` | internal | `AscendC::SyncAll`；勿依赖 MIX fallback |
| `aclshmemi_copy_ub2gm` / `aclshmemi_*` 数据面 | internal | 公开 `aclshmemx_mte_*` / `aclshmem_*` |
| `aclshmemi_get_state` 等 | internal | 勿在算子 kernel 中直接访问 |

**检测**：生成或走读时对 `custom-ops/` 执行 `rg 'aclshmemi_'` / `rg 'shmemi_'`（**排除**注释与 profiling 宏头文件）。

## 2. 已 deprecated 的接口（新算子勿用）

以下 **少数** `aclshmemx_*` 已标记 deprecated；其余 `aclshmemx_*` 扩展接口仍 **允许** 正常使用。

| deprecated | 替代 |
| --- | --- |
| `aclshmemx_barrier_all_vec` | `aclshmem_barrier_all` |
| `aclshmemx_barrier_vec` | `aclshmem_barrier` |

legacy example / 早期 custom-ops 可能仍保留 `*_barrier_all_vec`；新代码与 skill 模板 **MUST** 写 `aclshmem_barrier_all`。

## 3. quiet 选型（勿混用）

| API | 范围 |
| --- | --- |
| `aclshmem_quiet` | **全引擎**（MTE + SDMA + UDMA + RDMA） |
| `aclshmemx_mte_quiet` | 仅 MTE |
| `aclshmemx_sdma_quiet` | 仅 SDMA |
| `aclshmemx_udma_quiet(pe)` | 仅 UDMA |
| `aclshmemx_roce_quiet` | 仅 RDMA/RoCE |

纯 MTE put/get 路径 **MUST NOT** 用 `aclshmem_quiet` 代替 `aclshmemx_mte_quiet`（见 [core-allocation.md §4](../../shmem-ops-design/references/core-allocation.md)）。

## 4. 允许的 `shmemi_*` 头文件（例外）

| 头文件 | 用途 | 约束 |
| --- | --- | --- |
| `utils/prof/shmemi_prof.h` | Device 侧 cycle profiling 宏（`SHMEMI_PROF_*`） | **仅**性能调试；默认性能路径 **禁止**常驻 include；宏内部调用 `aclshmemi_get_state` 属 SDK 实现，算子 **禁止**手写等价逻辑 |

## 5. 走读检查

- [ ] `custom-ops/<op>/src/*_kernel.cpp` 无 `aclshmemi_` 调用（prof 宏除外）
- [ ] 无新增 `aclshmemx_barrier_all_vec`（用 `aclshmem_barrier_all`）
- [ ] MTE-only 路径用 `aclshmemx_mte_quiet`，非全局 `aclshmem_quiet`
- [ ] 核间同步用 `AscendC::SyncAll`，非 `aclshmemi_barrier_core_soft`
