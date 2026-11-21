---
type: reference
status: stable
platforms:
- '950'
sources:
- resource: asc-devkit master include/tensor_api
  role: primary
updated_at: '2026-10-29'
---

# Tensor API 参考摘要

> 事实源：`~/asc-devkit` 当前检出分支的 `include/tensor_api/`。当前摘要基于公开头文件整理；完整签名以当前头文件为准。

## 命名空间与入口

- 命名空间：`asc::te`。
- 基础入口：`#include "tensor_api/tensor.h"`。
- 实验性 Reg Vector 入口：`#include "tensor_api/experimental/vector_ops.h"`。
- 禁止直接 include `impl/` 或内部子目录头文件。

`tensor.h` 聚合 Tensor/Layout/Pointer/Utils/Arch/Atom/Copy/Mmad 与 legacy 接口；
`vector_ops.h` 聚合 Experimental Reg Vector 公共接口：reg_tensor、mask、
load/load_broadcast/load_gather、store/store_scatter、unalign_load/unalign_store、
算术/逻辑/复合/类型转换/归约/直方图、reorder/padding/index/gather 和 mem_bar。
Reg Vector 属于 Tensor API 路线，但使用前必须按 [experimental-reg-vector.md](experimental-reg-vector.md)
确认架构、dtype、mask 语义与 devkit commit。

## 核心流程

```text
make_mem_ptr<Location, Type>(address/offset)
make_frame_layout<Pattern, Trait>(args)
make_tensor(pointer, layout)
make_copy(operation[, trait])
copy(atom, dst, src[, ...])
make_mmad(operation[, trait])
mmad(atom, c, a, b[, bias])
```

## 公开 API

### Tensor / Pointer

```cpp
template <typename EngineT, typename LayoutT> struct local_tensor;

template <typename Iterator, typename... Args>
__aicore__ inline auto make_tensor(const Iterator& iter, const Args&... args);

template <size_t... SqueezeDims, typename Input, typename Enable>
__aicore__ inline auto squeeze(const Input& x);

template <typename Pattern, typename Input, typename Enable>
__aicore__ inline auto squeeze(const Input& x, const Pattern& pattern);

template <typename PtrPattern, typename DataType, typename Addr>
__aicore__ inline auto make_mem_ptr(Addr address);

template <typename PtrPattern, typename Iterator>
__aicore__ inline auto make_mem_ptr(Iterator iterator);

template <typename Iterator>
__aicore__ inline auto make_mem_ptr(Iterator iterator);
```

### Shape / Stride / Coord / Layout

```cpp
make_shape(...)
make_stride(...)
make_coord(...)
make_layout(shape)
make_layout(shape, stride)
make_pattern_layout<Pattern, Trait>(shape, stride)
make_frame_layout<Pattern, Trait>(args)
make_frame_layout<Pattern, C0Element>(args)
```

Layout 查询：

| API | 作用 |
|---|---|
| `get_shape(layout)` | 获取 shape |
| `get_stride(layout)` | 获取 stride |
| `rank(layout)` | 维度数 |
| `size(layout)` | 有效元素数 |
| `capacity(layout)` | 容量 |
| `coshape(layout)` | 物理 shape |
| `cosize(layout)` | 物理 size |
| `select(layout)` | 按维度选择信息 |
| `get(layout)` | 获取指定维度信息 |
| `slice(tensor, coord, info)` | 切片 |

### Copy

```cpp
template <typename Atom, typename DstTensor, typename SrcTensor>
__aicore__ inline void copy(
    const copy_atom<Atom>& atom,
    const DstTensor& dst,
    const SrcTensor& src);

template <typename Atom, typename DstTensor, typename SrcTensor, typename Quant>
__aicore__ inline void copy(
    const copy_atom<Atom>& atom,
    const DstTensor& dst,
    const SrcTensor& src,
    const Quant& quant);

template <typename DstTensor, typename SrcTensor>
__aicore__ inline void copy(const DstTensor& dst, const SrcTensor& src);

template <
    typename Atom, typename DstTensor, typename SrcTensor,
    typename DstCoord, typename SrcCoord, typename CopyShape>
__aicore__ inline void copy(
    const copy_atom<Atom>& atom,
    const DstTensor& dst,
    const SrcTensor& src,
    const DstCoord& dst_coord,
    const SrcCoord& src_coord,
    const CopyShape& copy_shape);

template <typename CopyOperation>
__aicore__ inline constexpr auto make_copy(const CopyOperation& operation);

template <typename CopyOperation, typename CopyTrait>
__aicore__ inline constexpr auto make_copy(
    const CopyOperation& operation,
    const CopyTrait& trait);
```

要点：

- `dst` 在前。
- `copy(atom, dst, src)` 需要显式 atom。
- `copy(dst, src)` 可根据 Tensor 属性推断默认 atom。
- 支持 quant 参数。
- 支持 region copy：destination/source 坐标和 copy shape。

### MMAD

```cpp
template <typename Atom, typename CTensor, typename ATensor, typename BTensor>
__aicore__ inline void mmad(
    const mmad_atom<Atom>& atom,
    const CTensor& c,
    const ATensor& a,
    const BTensor& b);

template <
    typename Atom, typename CTensor, typename ATensor, typename BTensor,
    typename BiasTensor>
__aicore__ inline void mmad(
    const mmad_atom<Atom>& atom,
    const CTensor& c,
    const ATensor& a,
    const BTensor& b,
    const BiasTensor& bias);

template <typename MmadOperation>
__aicore__ inline constexpr auto make_mmad(const MmadOperation& operation);

template <typename MmadOperation, typename MmadTrait>
__aicore__ inline constexpr auto make_mmad(
    const MmadOperation& operation,
    const MmadTrait& trait);
```

## Pattern 体系

常见 Pattern 包括：

- `nd_layout_ptn`
- `dn_layout_ptn`
- `nd_ext_layout_ptn`
- `dn_ext_layout_ptn`
- `nz_layout_ptn`
- `zn_layout_ptn`
- `zz_layout_ptn`
- `nn_layout_ptn`
- ScaleA/ScaleB 相关 Pattern

Pattern 是编译期路由标签。Routing 按目标/源位置、架构、Pattern、Trait 等条件派发；不在白名单会触发 `static_assert`。

## C0 与分形

C0 由 dtype、架构和 LayoutTrait 决定，必须从当前头文件读取。常见 3510 实证值：

| dtype | C0 |
|---|---:|
| fp16 / bf16 | 16 |
| fp32 | 8；L0C 仍按 16 处理 |
| int8 / fp8 | 32 |
| fp4x2 | 64 |
| fp8_e8m0 scale | 2 |

不要把 `32 / sizeof(T)` 写成跨架构常量。

## 架构边界

本 Skill 默认业务目标是 Ascend 950 / DAV_3510。当前 C API Reg Vector 入口条件包含 3510 和 9201；任何非 3510 目标都必须逐 API 查看头文件注释和实现特化，不得根据目录名或历史描述推断支持。

## 常见陷阱

| 现象 | 处理 |
|---|---|
| `Unsupported layout pattern` | 检查 Pattern 是否注册，源/目标位置组合是否合法 |
| `#warning internal header` | 改用公共入口，不 include `impl/` |
| offset 越界或重叠 | `make_mem_ptr` 的片上 offset 是字节偏移，必须乘 `sizeof(T)` |
| trans 后大量 mismatch | Pattern 未随 transA/transB 同步切换 |
| masked-out 值参与 golden | 明确 mask 语义，golden 忽略 inactive lane |
| 用旧 `AscendC::Te` / 大驼峰 API | master 使用 `asc::te` / 小驼峰 API |

## 官方样例

见 [sample-router.md](sample-router.md)。
