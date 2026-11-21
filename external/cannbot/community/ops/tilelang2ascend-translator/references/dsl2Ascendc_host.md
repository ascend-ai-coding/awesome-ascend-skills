## Host 侧准备详细参考

本文档只包含 Host 侧 tiling/pybind11 的实现细节与代码示例。
概览与判断规则见 `@references/dsl2Ascendc.md`。

---

## 第一章：Host 侧准备 `xxx_tiling.h` + `pybind11.cpp`

### 1. Tiling 参数一致性

确保所有 kernel 组件使用一致的 tiling 参数：

```cpp
// 在一处定义，到处使用
constexpr uint32_t baseM = 64;
constexpr uint32_t baseN = 64;
constexpr uint32_t baseK = 64;

// 多 Vector 核的子块
constexpr uint32_t subBlockM = baseM;  // 或 baseM / vec_num
constexpr uint32_t vecBlockN = baseN;  // 必须与 baseN 匹配！
```

**警告：** 参数不匹配（如 `vecBlockN = 256` 而 `baseN = 64`）会导致错误的内存访问模式。

---


### 2. Tiling Struct：在 Host 侧预计算运行时参数

避免在 kernel 的 `Process()` 中重复计算 `nTiles` / `nTilesPerH` 等派生量。推荐在 Host（pybind11.cpp）侧预先计算并写入 tiling struct，kernel 直接读取：

**tiling struct 推荐字段**：

```cpp
struct ReshapeMatmulQuantTiling {
    int32_t M, N, H_K;      // 基本形状
    int32_t baseM, baseN, baseK, K_L1;  // tile 大小
    int32_t nTiles;          // = N / baseN，避免 kernel 里除法
    int32_t nTilesPerH;      // = H_K / baseN，避免 kernel 里除法
};
```

**Host 侧填充（pybind11.cpp）**：

```cpp
tp->nTiles     = N   / DEFAULT_BASE_N;
tp->nTilesPerH = H_K / DEFAULT_BASE_N;
```

**Kernel 侧使用**：

```cpp
for (int by = 0; by < tiling_.nTiles; by++) {
    int groupId    = by / tiling_.nTilesPerH;
    int colInGroup = by % tiling_.nTilesPerH;
    ...
}
```

> 虽然 `N / baseN` 在 kernel 里做也能正确，但将派生量存入 tiling struct 是生产代码的标准做法，便于调试和验证。

---


### 3. 绑定层职责

`kernel/pybind11.cpp` 的职责是把 TileLang 对应的 AscendC kernel 包装成 Python 可调用扩展。

#### 3.1 模块名

模块名由

```cpp
PYBIND11_MODULE(<name>, m)
```

决定，`model_new_ascendc.py` 必须 import 同一个 `<name>`。

推荐格式：
- 任务目录：`<op_name>`
- 扩展模块：`_<op_name>_ext`
- Python 导入：`import _<op_name>_ext as _ext`

不要让扩展模块名与任务目录同名，否则 Python 可能先命中同名目录而不是扩展模块。

#### 3.2 绑定函数

绑定函数只接收 DSL 的显式输入张量，不接收输出张量和 workspace。

函数内部负责：
- 检查输入 shape 和 dtype
- 从输入 shape 推导运行时参数，如 `M/N/K`
- 分配输出张量
- 如有需要，分配 workspace
- 构造 tiling tensor
- 调用 `extern "C"` kernel launch 函数
- 返回输出张量

#### 3. Workspace

规则：只要 DSL 显式声明了 workspace 参数，或 `@tilelang.jit(...)` 显式给出了 `workspace_idx`，`pybind11.cpp` 就必须分配 workspace。

实践要求：
- workspace 的字节数必须和 DSL 中的 block 组织、累加 dtype、并行度一致
- workspace 在 `pybind11.cpp` 中可以分配为一维 `Byte` tensor，只要总字节数正确即可
