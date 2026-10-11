# broadcast 范式 standard Tiling 切分策略

> UB 切分、多核切分、Workspace 分配的完整算法

依赖输入：broadcast-template-overview.md、broadcast-standard-dag-buffers.md、broadcast-standard-tiling-preprocess.md

## 1 UB 切分轴

**参考案例**: `example/examples-adam-design/adam_apply_one_assign_design.md` §5.2（UB 切分）
**参考源码**: `example/examples-adam-code/op_host/arch35/adam_apply_one_assign_tiling_arch35.cpp`（FindSplitAxis 函数）

> 详细描述ub切分逻辑

**约束**：
- 每份 UB buffer 大小需要 blocksize 对齐（`Ops::Base::GetUbBlockSize`，不要写 32）
- 所有 UB buffer 总和，不能超过实际 UB 大小
- 必须给出伪码实现
- 必须考虑空 shape 场景：任一输入/输出维度 `<= 0` 由 Host 侧 `HasNonPositiveDim` 在 PadAndSqueeze 前报错拦截（见 [broadcast-standard-tiling-preprocess.md](broadcast-standard-tiling-preprocess.md) §2），不下发 kernel
- 即便有空核路径，也严禁 SetBlockDim(0)；若使用核数为 0，需要 SetBlockDim(1)，kernel 侧通过 usedCoreNum=0 短路处理

### 1.1 切分原理

Broadcast 场景使用 **UB 单切分**——以 effective_shape 为坐标系选择单根切分轴。

被切分的轴 a = ubOuter × ubFactor。包含 ubFactor 在内的所有内侧轴为「UB 内轴」，一次加载进 UB；ubOuter 至最外侧轴为「UB 外轴」，是 UB 外的 for 循环。

```
UB 内轴 = 轴 k (贡献 ubFactor) + 轴 k+1..n-1 (全量)
UB 外轴 = 轴 0..k-1 (全量循环) × ubOuter (轴 k 的分段循环)

UB 内存占用 = ubFactor × ∏_{j>k} d_j
必须满足: ubFactor × inner ≤ perBufElems
```

选轴算法——从最内轴向外扫描，累积内侧轴连乘 inner。当 `d_k × inner > perBufElems` 时，该轴无法全量放入——在此切分。

```
inner = 1
for k from n-1 down to 0:
    if d_k × inner > perBufElems:
        split at k
        ubFactor = perBufElems / inner
        ubOuter = ceil(d_k / ubFactor)
        break
    inner *= d_k
```

永远至少有一根可切轴。

### 1.2 切分公式

```cpp
// FindSplitAxis — 以 effective_shape 为坐标系选择单根切分轴
// perBufElems = ((ubPerCore / physNodes) & ~(ubBlockSize - 1)) / dtypeSize，dtypeSize 按实际计算精度选取
// ubBlockSize 来自 Ops::Base::GetUbBlockSize(ctx_)（util/platform_util.h），不要写 32；
// Ops::Base::CeilDiv 需在 tiling 侧 #include "util/math_util.h"（位于 op_common/op_host 下）
void FindSplitAxis(const std::vector<int64_t>& maxBroShape,
                   int64_t dtypeSize, int64_t ubPerCore, int64_t physNodes, int64_t ubBlockSize,
                   SplitResult &out)
{
    int64_t perBufBytes = (ubPerCore / physNodes) & ~(ubBlockSize - 1);
    int64_t perBufElems = perBufBytes / dtypeSize;
    int64_t rank = static_cast<int64_t>(maxBroShape.size());
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (maxBroShape[k] * inner > perBufElems) {
            out.ubFactor = perBufElems / inner;
            out.ubOuter = Ops::Base::CeilDiv(maxBroShape[k], out.ubFactor);
            int64_t rem = maxBroShape[k] % out.ubFactor;
            out.ubTail = (rem == 0) ? out.ubFactor : rem;
            out.ubSplitIdx = k;
            return;
        }
        if (k == 0) {
            // 全量装得下
            out.ubSplitIdx = 0;
            out.ubFactor = maxBroShape[0];
            out.ubOuter = 1;
            out.ubTail = out.ubFactor;
            return;
        }
        inner *= maxBroShape[k];
    }
}
```

尾块：d_k 不能整除 ubFactor 时，最后一轮 ubTail = d_k % ubFactor（整除时为 ubFactor）。

### 1.3 特殊场景

- 末尾轴装不下：d_{n-1} > perBufElems → ubSplitIdx = n-1, ubFactor = perBufElems
- 全量装得下：total ≤ perBufElems → ubSplitIdx = 0, ubFactor = d_0, ubOuter = 1
- 空 shape（任一维 ≤ 0）：Host 侧 `HasNonPositiveDim` 报错拦截，不进入切分、不下发 kernel

## 2 多核切分

**参考源码**: `example/examples-adam-code/op_host/arch35/adam_apply_one_assign_tiling_arch35.cpp`（MultiCoreSplit 函数）

**约束**：
- 严禁 SetBlockDim(0)；空 shape 已由 Host 侧 HasNonPositiveDim 拦截，若仍有使用核数为 0 的路径，需要 SetBlockDim(1)，kernel 侧通过 usedCoreNum=0 短路处理
- 设置的 BlockDim 严禁超过实际的物理核数
- (范式融合约束) 使用核间均衡的切分策略，假设有如下字段：  
    - totalTiles 用于切多核的总的tile块数
    - coreNum 总的aiv核数
    - usedCoreNum 实际使用的aiv核数
    - mainTiles 主核分配的tile块数 (尾核分配的tile块数=mainTiles-1)
    - mainCoreNum 主核数 (尾核数=usedCoreNum-mainCoreNum)  
- 必须给出伪码实现

### 2.1 切分原理

多核切分以 UB 切分产出的 tile 为单元，将 `totalTiles = ubOuter × Π_{j<k} d_j` 平摊到多核。

### 2.2 切分公式

```cpp
// MultiCoreSplit — tile 均衡分配
void MultiCoreSplit(const std::vector<int64_t>& maxBroShape,
                    const SplitResult &ubSplit, int64_t maxCores, MultiCoreResult &out)
{
    int64_t k = ubSplit.ubSplitIdx;
    int64_t outerProd = 1;
    for (int64_t j = 0; j < k; j++) {
        outerProd *= maxBroShape[j];
    }
    out.totalTiles = outerProd * ubSplit.ubOuter;

    out.usedCoreNum  = (out.totalTiles < maxCores) ? out.totalTiles : maxCores;
    out.mainTiles  = Ops::Base::CeilDiv(out.totalTiles, out.usedCoreNum);
    out.mainCoreNum  = out.totalTiles - (out.mainTiles - 1) * out.usedCoreNum;
    // 前 mainCoreNum 个核各处理 mainTiles 块，其余核各处理 mainTiles-1 块（母模板核间均衡切分公式）
}
```

### 2.3 特殊场景

- 空 shape（任一维 ≤ 0）：Host 侧 `HasNonPositiveDim` 报错拦截，不进入多核切分、不下发 kernel

## 3 切分策略微调

> 本章节是出于性能考虑，为了平衡ub切分与多核切分做的优化调整，如不需要，填 不涉及  
> 不管使用ub优先还是多核优先的切分策略，都存在一定缺陷，可能核用的少、可能ub用的不够。

### 3.1 微调方式

不涉及。当前 UB 单切分以 effective_shape 为坐标系，功能全覆盖。部分场景非最优（多数入参远小于 effective_shape 时 padding 占比高），预留多切分扩展点。

## 4 整体切分伪码

**参考源码**: `example/examples-adam-code/op_host/arch35/adam_apply_one_assign_tiling_arch35.cpp`（底部 TilingPrepare/TilingFunc/DoTilingAndSet 三个函数）

> 切分策略是一个整体，ub切分、多核切分可能是耦合的，这里给出样例

```
DoTiling():
  1. GetPlatformInfo()                → coreNum, ubSize, ubBlockSize（tiling 时直接从系统获取，
                                        ubBlockSize 用 Ops::Base::GetUbBlockSize；零值守卫：== 0 时报错返回；不经 CompileInfo 缓存）
  2. GetOpParam()                    → inputShapes, outputShapes（指针用专用宏判空）
  3. HasNonPositiveDim()             → 空 shape 防御：任一维 <= 0 时 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
                                       报错返回（避免 maxDim=0 → totalTiles=0 → usedCoreNum=0
                                       → CeilDiv 除零 / SetBlockDim(0)）
  4. PadAndSqueeze()                 → maximumBroShape, normalInputShapes, normalOutputShapes
  5. rank_ = maximumBroShape.size()
  6. rank 上限校验                    → rank_ > RANK_8 时 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON 报错返回
                                       （防止 delta = R - rank_ 为负数导致数组负索引）
  7. CheckBroadcastShape()           → broadcast 兼容校验（输入同维非 1 一致 + 输出稠密）
  8. perBufBytes = (ubSize / P) & ~(ubBlockSize - 1)   # P 来自 DAG 分析结论
  9. perBufElems = perBufBytes / dtypeSize   # dtypeSize 按实际计算精度选取
  10. FindSplitAxis(maxBroShape, dtypeSize, ubSize, physNodes, ubBlockSize, split)
  11. MultiCoreSplit(maxBroShape, split, coreNum, multicore)
  12. CalTilingKey()                 → key = (rank <= RANK_4) ? RANK_4 : RANK_8
  13. WriteTilingData()              → 填充 BroadcastTilingData<RANK>
  14. SetWorkspace(0)                → Broadcast 不需要 workspace
```

## 5 切分后处理

### 5.1 UB 大小计算

> 简述是按buffer个数均匀分配，还是在 tiling 切分逻辑中实时计算(每个buffer大小不同)

按 buffer 个数均匀分配：`perBufBytes = (UB / P) & ~(ubBlockSize - 1)`（P 个 buffer 均分 UB；ubBlockSize 来自 `Ops::Base::GetUbBlockSize`，当前平台为 32B 对齐）。

### 5.2 Workspace 分配

```cpp
size_t *ws = context->GetWorkspaceSizes(1);
OP_CHECK_NULL_WITH_CONTEXT(context, ws);
ws[0] = 0;
```

**约束**:
- 必须设置，即使不需要使用workspace，也需要显示设置为0
- 如果要使用workspace，必须额外分配 16MB，即总大小为 `16MB + 实际使用` (ascendc要求)
- 如果涉及workspace使用，必须使用INFO级别日志打印分配大小。

### 5.3 TilingKey 设置

> 必须使用INFO级别日志打印 tilingkey 各模板参数

```cpp
int64_t mapped = (rank_ <= {OP_UPPER}_RANK_4) ? {OP_UPPER}_RANK_4 : {OP_UPPER}_RANK_8;
ctx_->SetTilingKey(GET_TPL_TILING_KEY(mapped));
```

> `RANK_4/RANK_8` 为 struct.h 中定义的命名常量，禁止裸数字 4/8。

### 5.4 TilingData 设置

> 必须使用INFO级别日志打印 tilingdata各字段

**Tiling 维测规范**：两条日志（外部参数 + TilingData 全量），shape/stride 数组用 `MakeShapeFromDims` 组装 `gert::Shape` 后经 `Ops::Base::ToString` 统一打印 `"[d0, d1, ...]"`（不要自写 Arr2String 之类的 ostringstream helper），从 `tiling->` struct 数组打印（R 维全量含 padding），禁止从中间 vector（rank_ 维）打印。TilingData 全量含逐 input/output slot 的 shape/stride 数组循环打印（`for i in numIn/numOut` 各一条 OP_LOGI）。

```cpp
// 用 dims 前 dimNum 个元素组装 gert::Shape，供 Ops::Base::ToString 统一打印 "[d0, d1, ...]"。
static gert::Shape MakeShapeFromDims(const int64_t *dims, int32_t dimNum)
{
    gert::Shape shape;
    for (int32_t i = 0; i < dimNum; ++i) {
        shape.AppendDim(dims[i]);
    }
    return shape;
}

// 打印示例（maxBroShape 含 R 维 padding 全量）
Ops::Base::ToString(MakeShapeFromDims(tiling->maxBroShape, static_cast<int32_t>(R))).c_str()
```

Broadcast 关键字段表（日志对照用）：

| 日志字段 | 常见 bug |
|---------|----------|
| `maxBroShape` | padding 方向/值（前补=1 vs 后补=0），broadcast 轴是否被 squeeze |
| `input[i].stride` | 播轴 stride=0，padding stride=0 |
| `split(ubSplitIdx, ubFactor, ubOuter)` | ubSplitIdx 是否因 padding 错位 |
| `rank→R` | 实际 rank vs 模板 RANK 映射 |

### 5.5 ScheduleMode 设置

**约束**:
- SyncAll() 场景必须设置 SetScheduleMode(1)

## 6 切分策略产出校验

> 重要！为了校验agent是否真正理解了切分策略，需要产出所有的切分可能性。"人"来校验。

**对范式开发者的要求**：
- 1. 给一个 case, 枚举所有切分可能性，作为 golden。可以人工枚举，可以人与agent交互产生，不管怎样，需要"人"保证正确性。
- 2. 删除 golden, agent读取本文档可以产出所有的切分可能，如果产出不了，需要不断改进范式质量，直到agent能产出与golden一致的结果
- 3. golden 随本章节上库

**golden 样例（Adam: fp32, P=5, UB=262144, coreNum=24, ubBlockSize=32）**：

输入 shape 全同 `(256, 128, 64)`，PadAndSqueeze 后 `maxBroShape=(256,128,64)`, rank=3 → RANK=4。

| 参数 | 值 | 计算 |
|------|-----|------|
| perBufBytes | 52416 | `(262144 / 5) & ~(32 - 1)` |
| perBufElems | 13104 | `52416 / 4`（FP32） |
| split.ubSplitIdx | 1 | d=2(64): 64*1=64 ≤ 13104 ✓; d=1(128): 128*64=8192 ≤ 13104 ✓; d=0(256): 256*8192 > 13104 ✗ → split at 0 |

修正：split.ubSplitIdx=0（d=0 无法全量放入），ubFactor=13104/8192=1（取整后=1），ubOuter=ceil(256/1)=256。

| 参数 | 值 |
|------|-----|
| split.ubSplitIdx | 0 |
| split.ubFactor | 1 |
| split.ubOuter | 256 |
| split.ubTail | 1 |
| totalTiles | 256 |
| usedCoreNum | 24 |
| mainTiles | 11 (ceil(256/24)=11) |
| mainCoreNum | 16 (256-(11-1)*24=16) |

前 16 个核各处理 11 个 tile，后 8 个核各处理 10 个 tile。

**Python 交叉验证**（4 块纯逻辑，各出 Python 对比，随机 case 100 次）：

| # | 验什么 | 关键检查点 |
|---|--------|----------|
| 1 | 计算流序列 | golden(原始公式) vs opt(融合+重排) vs kernel Process — 三者一致 |
| 2 | 循环 count | `count = ubBlockLength × Π maxBroShape[d>ubSplitIdx]`；split 轴 = 最内维时 inner=1 |
| 3 | 地址偏移 | 播轴 stride=0→offset 不贡献；padding stride=0→offset 不贡献；GM 偏移 = Σ coord×stride |
| 4 | NDDMA 参数 | loopSize = 目标 shape（非源 shape）；五字段全填；最多 5 维；前补 dim→最外维 loopSize=1 |

> **注意**：交叉验证的输入数据要和 kernel 看到的完全一致——Python 用 `maxBro`（rank_ 元素），kernel 用 `maxBroShape[RANK]`（R 元素含 padding）。差一个 padding 值就是 bug 盲区。

---

## 7 物理计算流验证 harness

> 对应 patterns.md §3.3「物理计算流验证」（非必须步骤）。原 `asset/templates/verify.py` 已内联至此。
> 用途：用 golden（spec 公式逐行翻译）与 opt（计算流伪码逐行翻译、独立实现）跑随机 case 对比，验证计算流设计正确性。

**使用流程**：

1. 写 `golden.py` — 导出 `golden(*args, **kwargs)` 和 `generate_test_cases()`
2. 写 `opt.py` — 导出 `opt(*args, **kwargs)`，签名与 golden 一致，独立实现
3. 把下面的 harness 存为 `verify.py`，取消 `from golden import ...` / `from opt import ...` 两行注释
4. `python3 verify.py`，exit 必须为 0
5. `cat verify_results.json` 确认 `total >= 180` 且 `failed == 0`

`generate_test_cases()` 返回 `list[dict]`，每项至少含 `"args"`（位置参数 list）、`"kwargs"`（关键字参数，至少含 `"dtype"`）、可选 `"tolerance"`（`{"rtol": float, "atol": float}`）。

```python
#!/usr/bin/env python3
"""算子物理计算流验证脚本（通用模板）。

约定：
    golden.py  — 导出 golden(*args, **kwargs) 和 generate_test_cases()
    opt.py     — 导出 opt(*args, **kwargs)，签名与 golden 一致，独立实现
"""

import json
import sys
import numpy as np

# ── 填入你的 golden / opt 模块 ──
# from golden import golden, generate_test_cases
# from opt import opt


def main():
    cases = generate_test_cases()
    passed = 0
    failed = 0
    failures = []

    for i, case in enumerate(cases):
        case_id = case.get("case_id", f"case_{i:04d}")
        args = case["args"]
        kwargs = {k: v for k, v in case.items() if k not in ("args", "tolerance", "case_id")}
        tol = case.get("tolerance", {"rtol": 1e-5, "atol": 1e-5})

        try:
            g = golden(*args, **kwargs)
            o = opt(*args, **kwargs)

            if isinstance(g, np.ndarray):
                ok = bool(np.allclose(g, o, rtol=tol["rtol"], atol=tol["atol"], equal_nan=True))
            elif isinstance(g, (tuple, list)):
                ok = all(
                    np.allclose(gi, oi, rtol=tol["rtol"], atol=tol["atol"], equal_nan=True)
                    for gi, oi in zip(g, o)
                )
            else:
                ok = bool(np.isclose(g, o, rtol=tol["rtol"], atol=tol["atol"]))

            if ok:
                passed += 1
            else:
                failed += 1
                failures.append({"case_id": case_id, "error": "tolerance exceeded"})

        except Exception as e:
            failed += 1
            failures.append({"case_id": case_id, "error": f"{type(e).__name__}: {e}"})

    total = len(cases)

    with open("verify_results.json", "w") as f:
        json.dump(
            {"total": total, "passed": passed, "failed": failed, "failures": failures[:20]},
            f, indent=2, default=str,
        )

    with open("verify_results.txt", "w") as f:
        f.write(f"Total: {total}  Passed: {passed}  Failed: {failed}\n")
        for r in failures[:10]:
            f.write(f"  FAIL {r['case_id']}: {r['error']}\n")
        f.write("ALL PASSED\n" if failed == 0 and total >= 180 else "INCOMPLETE OR FAILED\n")

    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    if total < 180:
        print(f"ERROR: total={total} < 180 — 试验次数不足!")
        sys.exit(2)
    if failed > 0:
        print("FAILED")
        sys.exit(1)
    print("ALL PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

## 8 Host 侧注册与 Tiling 整体结构

> Host 侧注册宏、CompileInfo、TilingPrepare/TilingFunc/DoTilingAndSet 三函数的完整写法

### 8.1 CompileInfo 结构体

CompileInfo 为空结构：Broadcast 算子无跨次编译缓存信息，平台参数在 Tiling 阶段直接从系统获取（见 §8.5），不经 CompileInfo 中转。后续若做 binary 复用需要缓存字段，再挂到 CompileInfo 并在 TilingParse 中解析：

```cpp
struct {OP}CompileInfo {};
```

### 8.2 TilingPrepare（TilingParse 回调）

空实现，恒返回成功，仅保留注册挂载点：

```cpp
// TilingParse 空实现：{OP}CompileInfo 为空结构（本算子无跨次编译缓存信息），
// 恒返回成功；后续若做 binary 复用，缓存字段挂 CompileInfo 并在此解析。
static ge::graphStatus TilingPrepareFor{OP}([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}
```

### 8.3 TilingFunc 入口函数

TilingFunc 是框架入口，创建 Tiling 类实例并调用 RunTiling，然后设置 workspace：

```cpp
static ge::graphStatus TilingFunc{OP}(gert::TilingContext *context)
{
    {OP}Tiling tiling(context);
    auto ret = tiling.RunTiling();
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;  // Broadcast 不需要 workspace
    return GRAPH_SUCCESS;
}
```

### 8.4 RunTiling 分发

RunTiling 按 RANK 分叉到 DoTilingAndSet<4> 或 DoTilingAndSet<8>，设置 TilingKey：

```cpp
ge::graphStatus {OP}Tiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    int64_t mapped = (rank_ <= {OP_UPPER}_RANK_4) ? {OP_UPPER}_RANK_4 : {OP_UPPER}_RANK_8;
    if (mapped == {OP_UPPER}_RANK_4) {
        ret = DoTilingAndSet<{OP_UPPER}_RANK_4>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY({OP_UPPER}_RANK_4));
    } else {
        ret = DoTilingAndSet<{OP_UPPER}_RANK_8>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY({OP_UPPER}_RANK_8));
    }
    return ret;
}
```

### 8.5 DoTilingAndSet 模板函数

按 RANK 模板实例化，填充 TilingData 全字段、设置 BlockDim、打印维测日志。完整伪码见 §4。平台参数（coreNum/ubSize/ubBlockSize）在本函数内通过 `ctx_->GetPlatformInfo()` + `platform_ascendc::PlatformAscendC` 现场获取并做零值守卫（`GetCoreNumAiv` / `GetCoreMemSize(UB)`，`ubBlockSize` 用 `Ops::Base::GetUbBlockSize(ctx_)` 获取，`perBufBytes = (ubPerCore / physNodes) & ~(ubBlockSize - 1)`），任一为 0 视为环境异常直接失败。

### 8.6 IMPL_OP_OPTILING 注册宏

一行注册三件事：Tiling 函数、TilingParse 回调、CompileInfo 类型：

```cpp
IMPL_OP_OPTILING({OP}).Tiling(TilingFunc{OP}).TilingParse<{OP}CompileInfo>(TilingPrepareFor{OP});
```

**约束**：
- `{OP}` = 算子名 CamelCase，与 `OP_ADD({OP})` 一致
- `TilingParse<{OP}CompileInfo>` 的模板参数须与 tiling 头文件中定义的 `{OP}CompileInfo`（空结构）一致
- TilingFunc 签名固定为 `static ge::graphStatus(gert::TilingContext*)`
