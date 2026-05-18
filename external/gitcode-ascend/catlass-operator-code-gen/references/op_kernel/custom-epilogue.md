# 自定义 Epilogue（代码生成）

**前置**：[catlass-operator-design 的 custom-epilogue.md](../../catlass-operator-design/references/custom-epilogue.md)（先确认是否需自定义、设计文档已写清）。

**侧重点**：在设计文档已约定「自定义 Tile」的前提下，在 **算子工程** 中**非侵入**增加头文件、改 `op_kernel/CMakeLists.txt`、在 `*_impl.h` 中组装 `BlockEpilogue` + `MatmulEpilogue`。**算子 kernel 必须用 Device 调用**，禁止把 `DeviceGemm` 适配器写进 `op_kernel`（见 [kernel-rules.md](./kernel-rules.md)）。

---

## 1. 目录与文件

```
ops/<op_name>/op_kernel/
├── <op>_impl.h              # 含 using TileXxx、BlockEpilogue、MatmulKernel
├── custom_epilogue/
│   └── tile_<semantic>.hpp  # 自定义 struct，CATLASS_DEVICE operator()
└── CMakeLists.txt           # 无需额外配置，custom_epilogue 目录自动扫描
```

- 头文件 guard + `namespace Catlass::Epilogue::Tile { ... }` 与常见 catlass 习惯一致即可。
- `#include "catlass/catlass.hpp"`，在 `operator()` 内仅用 **AscendC** 向量 API（与 `computeLength`、UB 约束一致）。
- **编译选项**：`custom_epilogue/` 目录下的头文件会被自动扫描，无需在 `ascendc_config.json` 中额外配置 `-I`。

---

## 2. 自定义 Tile 骨架（固定写法）

```cpp
#pragma once
#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_>
struct TileMyCustom {
  using ArchTag = ArchTag_;
  using ElementCompute = typename ComputeType_::Element;
  static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

  CATLASS_DEVICE void operator()(
      AscendC::LocalTensor<ElementCompute> const &ubOut,
      AscendC::LocalTensor<ElementCompute> const &ubIn0,
      AscendC::LocalTensor<ElementCompute> const &ubIn1)
  {
    // AscendC::Add / Mul / ... 与 COMPUTE_LENGTH
  }
};

} // namespace Catlass::Epilogue::Tile
```

- **NoSource** policy 的 Tile 若签名不同，以 catlass 同 policy 的现有 Tile 为准对照。
- `*_impl.h`：`#include "tile_<semantic>.hpp"`，再 `using TileElemWiseEpilogue = TileMyCustom<ArchTag, ComputeType, computeLength>;` 等，与 **设计文档** 一致。

---

## 3. 与入口 `.cpp`

- 入口只 `GET_TILING_DATA` + `TILING_KEY_IS` + 调用 `*_impl.h` 中封装好的 Device 调用；**不**在此文件展开 Tile 定义。

---

## 4. 检查清单（codegen）

- [ ] 自定义头文件在 `op_kernel/custom_epilogue/`，未改 `catlass/` 上游源码
- [ ] `BlockEpilogue`、`MatmulEpilogue` 组装与 **catlass-operator-design** 设计文档一致
- [ ] 未在 `op_kernel` 使用 `DeviceGemm` / `DeviceGemm` 适配器路径
