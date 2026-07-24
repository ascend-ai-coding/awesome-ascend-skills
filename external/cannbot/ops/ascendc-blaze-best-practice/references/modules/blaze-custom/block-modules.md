# Blaze Custom Block 层模块手册

> **定位**：blaze_custom 路径下 Block 层各模块的使用手册。Block 层负责单 block（baseM × baseN）的数据搬运（GM→L1→L0）与 MMAD 计算（L0A×L0B→L0C），以及最终输出（L0C→GM 或 L0C→UB）。

---

## §1 BlockMmad（SWAT）

| 项目 | 说明 |
|------|------|
| 头文件 | `block/matmul_block_mmad.h` |
| 命名空间 | `Block::BlockMmad` |
| 模板签名 | 8 参数（含 Enable） |

**完整模板签名**：

```cpp
template <
    class DispatchPolicy_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_, class Enable = void>
class BlockMmad;
```

SWAT 特化通过 SFINAE 激活：

```cpp
class BlockMmad<..., AscendC::Std::enable_if_t<
    AscendC::Std::is_base_of_v<
        MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>, DispatchPolicy_>>>;
```

**DispatchPolicy 绑定**：`MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>` 定义在 `policy/dispatch_policy.h`。本 skill 不提供 full-load 分支。

**Params / L1Params 字段**：

| 结构体 | 字段 | 类型 | 说明 |
|--------|------|------|------|
| `Params` | `aGmAddr` | `GM_ADDR` | A 矩阵 GM 地址 |
| `Params` | `bGmAddr` | `GM_ADDR` | B 矩阵 GM 地址 |
| `Params` | `cGmAddr` | `GM_ADDR` | C 矩阵 GM 地址 |
| `L1Params` | `kL1` | `uint64_t` | 单次 L1 流水覆盖的 K 长度（= baseK × stepK） |

**GM / UB 输出分支**：

BlockMmad 根据传入的 TensorC 的 `Location` 编译期分流：

| TensorC Location | 输出路径 | 适用场景 |
|------------------|---------|---------|
| `Location::GM` | `CopyL0C2GM`（Fixpipe 直写 GM） | 纯 matmul（MatmulKernel） |
| `Location::UB` | `CopyL0C2UB`（DUAL_DST_SPLIT_M Trait） | 融合场景（MatmulKernelFused） |

该机制不属于 DispatchPolicy，也不扩展 BlockMmad 模板参数。Kernel 层传 GM 还是 UB Tensor 即决定走哪条路径。

**Init 签名**：

```cpp
void Init(const TupleShape& problemShape,   // {m, n, k} 3D
          const BlockShape& l0TileShape,     // {baseM, baseN, baseK, 0}
          const L1Params& l1Params,
          bool enableL0cPingPong);
```

**operator() 签名**：

```cpp
template <typename TensorA, typename TensorB, typename TensorC>
void operator()(TensorA gmA, TensorB gmB, TensorC gmC, BlockShape singleShape);
```

---

## §2 公共约束

### BLOCK_CUBE 值表

| dtype | BLOCK_CUBE | 说明 |
|-------|-----------|------|
| `float` (fp32) | 8 | 32 / sizeof(float) = 8 |
| `bfloat16_t` / `half` (bf16/fp16) | 16 | 32 / 2 = 16 |
| `int8_t` / `fp8_e4m3_t` (int8/fp8) | 32 | 32 / 1 = 32 |
| `fp4` (packed) | 64 | 2 元素共享 1 字节 |

> **注意**：blaze_custom 中 `BLOCK_CUBE` 硬编码为 16，`BLOCK_CUBE_L0C` 恒为 16（dav-3510 上 L0C cube 边长与 dtype 无关）。不要写成 `32 / sizeof(L0CType)`。

### L0CType 选择规则

| 输入 dtype | L0CType | 说明 |
|-----------|---------|------|
| `int8_t` | `int32_t` | 硬件 MMAD 要求 int8→int32 累加 |
| 其他（fp32/bf16/fp16/fp8） | `float` | fp32 累加精度 |

通过 `conditional_t<is_same_v<AType, int8_t>, int32_t, float>` 实现。

### 事件标志分配规则

| 事件类型 | 槽位 | 用途 |
|---------|------|------|
| `MTE1_MTE2` slot 0/1 | L1 双缓冲 ping-pong | A/B GM→L1 搬运同步 |
| `MTE2_MTE1` slot 0/1 | L1 双缓冲反向 | L1 就绪后通知 MTE1 |
| `M_MTE1` slot 0 (ZERO_FLAG) | L0 ping-pong | L0A/L0B 加载同步 |
| `M_MTE1` slot 1 (FIRST_FLAG) | L0 ping-pong | L0A/L0B 加载同步 |
| CrossCore flag 5/8+ | AIC↔AIV | CV 同步（CvSync 常量） |

---

## §3 常见陷阱

| 陷阱 | 说明 |
|------|------|
| L1 偏移忘记乘 sizeof | `MakeMemPtr<Location::L1, T>(offset)` 接受**字节偏移**，必须显式乘 `sizeof(AType)`/`sizeof(BType)`，否则 ping-pong 半区物理重叠 |
| L0C 偏移忘记乘 sizeof | 同 L1，`HALF_L0C_SIZE` 已是字节，`l0cOffset` 直接使用即可 |
| BLOCK_CUBE_L0C 写错 | dav-3510 上 L0C cube 边长**恒为 16**，不要写成 `32 / sizeof(L0CType)`（fp32 时会算出 8），导致 fixpipe stride 减半 |
| cmatrixInitVal 语义 | `mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0)` 是 loop-local 语义，每次 operator() 调用都在 (0,0) 重置 L0C，与 A 是否驻留 L1 无关 |

---

## §4 与 blaze 库 BlockMmad 差异

| 差异点 | blaze_custom BlockMmad | blaze 库 BlockMmad |
|--------|----------------------|---------------------|
| 模板参数数量 | 8（含 Enable） | 9（多一个 `AccumulateType` 或类似参数） |
| SFINAE 机制 | `enable_if_t<is_base_of_v<MatmulMultiBlockPolicy<MODE>, DispatchPolicy_>>` | 直接模板匹配（无 SFINAE） |
| 输出路径控制 | 由传入的 TensorC Location 决定（GM/UB） | 由模板参数或 Policy 显式指定 |
| 命名空间 | `Block::BlockMmad` | `Blaze::Gemm::Block::BlockMmad` |
| DispatchPolicy | `MatmulMultiBlockPolicy<MODE>` | `MatmulMultiBlockBasic` / `MatmulWithScaleMx` 等 |
| L1 事件管理 | 手动 WaitFlag/SetFlag 编排 | 框架封装 |

---

## §5 命名空间说明

| 模块 | 命名空间 | 说明 |
|------|---------|------|
| blaze_custom Block | `Block::` | 如 `Block::BlockMmad`、`Block::MatmulBlockScheduler` |
| blaze 库 Block | `Blaze::Gemm::Block::` | 如 `Blaze::Gemm::Block::BlockMmad` |
| blaze_custom Policy | 全局作用域 | `MatmulMultiBlockPolicy<MODE>` 定义在全局 namespace |
| blaze_custom Scheduler 标签 | 全局作用域 | `MatmulSwatScheduler<MODE>` 定义在全局 namespace |

Block 层内部使用 `Block::` namespace，但 Policy 和 Scheduler 标签类定义在全局作用域，被 Block 层通过模板参数引用。
