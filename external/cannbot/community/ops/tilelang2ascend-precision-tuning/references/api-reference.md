# DumpTensor API Reference

## 常见调用形态

```cpp
DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
```

```cpp
DumpTensor(globalTensor[offset], uint32_t desc, uint32_t dumpSize)
```

第一种是对 LocalTensor/UB 中的数据进行 dump，第二种是对 GM/global tensor 上某个偏移位置的数据进行 dump。复杂算子里经常直接 dump GM 或 Workspace 上的 ND 数据，便于和 CPU golden 一一对齐。

## Parameters

| Parameter  | Type | Description |
|------------|------|-------------|
| `tensor`   | LocalTensor 或 GM/global tensor 位置 | The tensor slice to dump |
| `desc`     | `uint32_t` | Unique identifier (use systematic numbering) |
| `dumpSize` | `uint32_t` | Number of elements to dump |

## Desc Numbering Convention

| Range   | Stage |
|---------|-------|
| 100-199 | C输入 |
| 200-299 | C中间 |
| 300-399 | C输出 |
| 400-499 | V输入 |
| 500-599 | V中间 |
| 600-699 | V输出 |

约定很简单：

- 没有 Cube 的简单算子，直接从 `400-699` 用起。
- 有 Cube 和 Vector 的算子，统一按 `C -> V` 六段走。
- 同一大段里需要多个点位时，用 `+10`、`+20` 这类子号区分，不要再创造新的阶段名。

## Best Practices

```cpp
// Control dump size - dump subset for large tensors
uint32_t dumpSize = std::min(tileLength, 32u);
DumpTensor(outputLocal, 600, dumpSize);

// Dump GM/global tensor directly when the ND layout there is easier to inspect
DumpTensor(wsSGm_[sOffset], 300 + slot, 32);

// Avoid dumping entire large tensors
DumpTensor(outputLocal, 600, 8192);  // ❌ Too much
DumpTensor(outputLocal, 600, 32);    // ✅ Better
```
