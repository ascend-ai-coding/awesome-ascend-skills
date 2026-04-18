# op_kernel 生成

## Catlass 编译配置

编译选项配置见 [compile-options.md](../compile-options.md)。

### Catlass 源码位置

- **放哪克隆**：只在 **OPS_PROJECT_ROOT**（与 `build.sh`、`ops/` 同级）需要 `catlass/`。缺则：`git clone https://gitcode.com/cann/catlass.git catlass`。不要克隆进 `ops/<算子名>/`。

### dtype 与 L1 分块

示例多为 **half**；若采用与 example 不同的类型（如 float），可能需要缩小 L1/L0 TileShape，参考仓库内对应类型范例（如 `examples/15_gemm`）。

---

## 编程约束

### 仅使用 Catlass 提供的实现

- **禁止以 SIMT（单线程标量循环）视角编写算子代码**：AICore 架构与 GPU 不同，标量循环极慢，必须使用向量/块级接口。
- Kernel 中**只能**使用 Catlass 提供的 Block/Tile/Kernel 组合。
- **不得**自行实现：矩阵乘、逐元素加、逐元素乘、拷贝等计算逻辑。
- 可用组件示例：`BlockMmad`、`BasicMatmul`、`MatmulEpilogue`、`BlockEpilogue`、`TileElemWiseAdd`、`TileCopy` 等（以 catlass 仓内头文件为准）。

### 无现成组件时的处理

- 若设计文档要求的融合逻辑在 Catlass 中**无**现成组件，应先在**设计阶段**改为选用 Catlass 已有组件。
- 实在没办法时，参考 `custom-xxx.md` 手搓实现，但需向用户说明需**扩展 Catlass 库**后再生成代码。

---

## Kernel 实现结构

1. 导入所需头文件
   - catlass 系列头文件
   - lib/matmul_intf.h（固定）
2. 编写 kernel 模板函数
3. 编写 kernel 入口
   - `GET_TILING_DATA` 取 tiling 数据
   - 从 tiling 数据构造 kernel 入口所需的一些结构体和常量，如 `GemmCoord problemShape`
   - 在 `TILING_KEY_IS(key)` 分支中实例化 kernel
   - 例：`using Kernel=xxx; using KernelParams=Kernel::Params; Params params{...};`（通常每个 key 对应一种 dtype/转置组合，但不限于这些情形。任何可能导致模板参数不同的参数组合，都应当被视为新的TILING_KEY_IS分支，如Swizzle）
4. 对于多 `TILING_KEY_IS` 分支内使用相同模板的情形，可以将固定的模板拆分到一个 `.h` 文件中，`.cpp` 只对对应 KEY 实例化对应模板，以使 cpp 文件中的内容更加清晰。

详细可参考 `examples/catlass_basic_matmul.cpp`。

---

## Host 调用 vs Device 调用（关键区分）

Catlass example 中存在两种 Kernel 调用模式，op_kernel 必须使用 Device 调用：

| 模式 | 适用场景 | 关键类 | 调用方式 |
|------|---------|--------|---------|
| **Device 调用** | op_kernel（算子工程） | 直接使用 `Kernel::BasicMatmul` 等 Kernel 类 | 构造 `Kernel::Params`，调用 `Kernel(params)` 执行 |
| Host 调用 | example 测试/演示 | `Gemm::Device::DeviceGemm<Kernel>` 适配器 | `Initialize` → `operator(stream, coreNum)` 管理 workspace 等 |

**op_kernel 中的 Device 调用写法（必须遵循）：**

```cpp
// 在 TILING_KEY_IS 分支内
using Kernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
Kernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, workspace};
Kernel kernel;
kernel(params);
```

**错误写法 — 不要在 op_kernel 中使用 Host 适配器：**

```cpp
// ❌ DeviceGemm 是 Host 侧适配器，仅用于 example/main 中调测，不可出现在 op_kernel
using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
MatmulAdapter matmulOp;
matmulOp.Initialize(arguments, deviceWorkspace);
matmulOp(stream, aicCoreNum);
```

`catlass/examples` 中 `DeviceGemm` 适配器用于封装 Host 调用流程（管理 workspace 分配、stream 调度等），但算子工程的 op_kernel 运行在 AI Core 上，应直接实例化 Kernel 类并调用 `operator()`。Host 侧的 workspace 分配、tiling 数据准备由 op_host 的 Tiling 函数和框架调度完成。

---

## Workspace 固定写法（仅 op_kernel）

Host 侧 `GetWorkspaceSizes` 写法见 [tiling-rules.md](./tiling-rules.md)。

需把 GM workspace 交给 Catlass `Params` 时：

```cpp
#include "kernel_operator.h"
// ...
GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));
// Catlass ... Params{ ..., userWs, ... };
```

**禁止**：`SetSysWorkspaceForce(workspace)`。

---

## MatmulEpilogue 与独立输出缓冲

`MatmulEpilogue::ToUnderlyingArguments` 将 epilogue 的 X/D 设为同一指针；若需要 **Y = A@B − X** 且 **y 与 x3 为不同 GM**，须在 Device 侧**手动构造** `MatmulKernel::Params` / `BlockEpilogue::Params`（`ptrX=x3`，`ptrD=y`），勿照搬仅含 `Arguments` 的 Host 适配器路径。

---

## 易踩坑

- **`TILING_KEY_IS(...)`**：预编译阶段要求括号内为**数字常量**（如 `TILING_KEY_IS(0)`），不要用 `constexpr` 变量名。
- **`GetUserWorkspace`**：使用 **`AscendC::GetUserWorkspace(workspace)`**（与模板里无命名空间的全局形式区分）。
- **`MatmulActivation` + GM workspace**：`ToUnderlyingArguments(..., uint8_t *workspace)` 的指针为**非 gm**，**禁止**将 `GM_ADDR`（`__gm__`）`reinterpret_cast` 成 `uint8_t *`。应在 Device 侧按头文件中的字段顺序**手写** `typename MatmulKernel::Params{ ..., userWs, epilogueParams }`。
- **Epilogue 向量长与矩阵规模**：固定模板 `COMPUTE_LENGTH` 的 `TileElemWise*` 与过小的矩阵规模或尾块组合时，运行期可能报 UB 越界。测试形状宜取 L1 Tile M/N 的整数倍。

---

## 检查清单（op_kernel）

- [ ] `scripts/kernel/binary_config/ascendc_config.json` 中已添加算子配置（见 [compile-options.md](../compile-options.md)）
- [ ] TILING_KEY_IS 分支覆盖设计文档中所有 dtype/转置组合
- [ ] 各分支使用 Catlass 提供的 Kernel/Block/Tile 组合
- [ ] 使用 Device 调用：直接实例化 `Kernel` + `Kernel::Params`，未使用 `DeviceGemm` 适配器
- [ ] 未手写标量/逐元素计算
- [ ] 若使用 GM workspace：按上文写法；**未**对 `workspace` 调 `SetSysWorkspaceForce`

TilingKey 与 Host 侧说明见 [tiling-rules.md](./tiling-rules.md)；端到端顺序中 op_kernel 步骤见 [SKILL.md](../SKILL.md)「Step 5」。
