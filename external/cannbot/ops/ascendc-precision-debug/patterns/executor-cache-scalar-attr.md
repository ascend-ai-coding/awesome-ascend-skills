# 框架机制：DFX_IN 遗漏输入参数导致 ACLNN 缓存误命中

根因类别：**框架机制**（非数值精度）。

## 形态指纹

- **单独跑 PASS、批量跑 FAIL**（跨用例污染的核心特征）；
- 失败用例的结果里出现"前一个同 shape 用例"的属性值（如用了上一个 case 的 beta/threshold）；
- 换一套"每个用例唯一 shape"的用例集就全过（规避了 cache 命中）；
- 多发于：标量属性经 Host tiling 写入 TilingData、kernel 直接读取的算子。

## 根因机理

ACLNN L2 的 cache key = hash(**API 名 + op config + `DFX_IN` 参数 + `DFX_OUT` 参数**)（源码 `op_cache.h:GetFromCache`）。命中时直接返回已有 `aclOpExecutor`（含已计算的 TilingData），**跳过 Tiling 函数**；缓存按 LRU 管理，默认上限 10000 条（`K_CACHE_LIMIT`）。

**若 `DFX_IN` 遗漏了某个影响 TilingData 的输入参数（如 aclScalar* 类型的 beta/threshold，或 int64_t/float/bool/const char* 等普通 C++ 类型参数），该参数不参与 cache key hash**：同 shape 不同参数值的调用命中同一缓存项、复用旧 TilingData，拿旧属性参与计算 → 结果错误。

硬性约束：`OP_ATTR(...)` 中的每个 scalar 必须同时在 L2 的 `DFX_IN(...)` 中出现；`DFX_IN` 应覆盖 aclTensor* + aclScalar* + aclXXArray* + 普通 C++ 数据类型（int64_t/float/bool/const char* 等）的全部输入。

## 验证方法

1. 单跑失败用例（PASS）→ 批量跑（FAIL）→ 锁定跨用例维度。注意：若用例集每个用例 shape 不同，单跑/批跑的差异可能来自 shape 而非 cache，需下一步排除；
2. **首选隔离实验（scalar 扫描）**：同一进程内**固定 shape、逐次变化 scalar 属性值**调用算子——"首次调用正确、后续调用全错"即高度疑似 cache key 漏 scalar，且能直接指出遗漏的是哪个属性。比单跑/批跑更干净（排除 shape 干扰），比 `DISABLE_L2_CACHE` 定位粒度更细；
3. **快速确认**：`export DISABLE_L2_CACHE=1` 后批量重跑——失败消失即确认是 cache 相关（这是调试手段，不是修复）；
4. 审查该算子 L2 接口的 `DFX_IN(...)`：是否包含全部输入（aclTensor* + aclScalar* + aclXXArray* + 普通 C++ 类型），与 `OP_ATTR(...)` 的 scalar 一一对应，找出遗漏项；
5. （可选佐证）tiling 函数加 `OP_LOGE` 计数，确认同 shape 第二次调用时 tiling 未被执行。

## 修复 pattern

```cpp
// 正解：DFX_IN 补全所有影响 TilingData 的输入（aclTensor* + aclScalar* + aclXXArray* + 普通 C++ 类型）
// 反例：scalar 漏打包，beta/threshold 不参与 cache key
L2_DFX_PHASE_1(aclnnFoo, DFX_IN(x, y), DFX_OUT(z));

// 正例：全部输入进 DFX_IN，不同属性值产生不同 cache key
L2_DFX_PHASE_1(aclnnFoo, DFX_IN(x, y, beta, threshold), DFX_OUT(z));
```

- **首选修复是补全 `DFX_IN`**（保留 cache 的性能收益），不是关缓存；
- `AbandonCache(true)` 只是调试/临时规避手段，不作为最终修复；
- 同类排查：凡 `OP_ATTR` 传 scalar 的 L2 接口，以及签名中有普通 C++ 类型参数的 L2 接口，批量审查 `DFX_IN` 完整性。

## 反例（像但不是）

- 单跑也错 → 不是 cache 问题，回到单用例维度排查；
- `DISABLE_L2_CACHE=1` 后仍失败 → 与 L2 cache 无关，查他处；
- 结果完全随机无规律 → 更像 uninitialized-memory；
- 不同 shape 之间也互相影响 → 查全局内存越界，不是 tiling cache。
