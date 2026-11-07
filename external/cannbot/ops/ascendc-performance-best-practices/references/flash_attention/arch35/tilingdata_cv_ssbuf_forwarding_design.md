# TilingData CV-SSBUF 转发优化设计

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

## 1. 优化目标

针对 `kv_quant_sparse_flash_attention` 在 A5 / arch35 混合 AIC+AIV kernel 启动阶段重复拷贝完整 TilingData 的问题，将 full TilingData 拷贝保留在 vector 侧，由 vector 抽取 cube/vector 共享字段并通过 `__ssbuf__` 转发给 cube，减少 cube 侧启动头开销。

## 2. 架构概览

### 优化前

```text
AIV: GET_TILING_DATA_WITH_STRUCT(full tiling) -> vector init
AIC: GET_TILING_DATA_WITH_STRUCT(full tiling) -> cube init
```

问题：cube 和 vector 只需要 TilingData 的不同子集，但两侧都执行 full tiling struct 拷贝。小 shape 或启动头占比较高时，cube 侧 tiling 初始化会进入关键路径。

### 优化后

```text
AIV:
  GET_TILING_DATA_WITH_STRUCT(full tiling)
  fill CVSharedParams
  write CVSharedParams to __ssbuf__
  CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15)

AIC:
  independent cube init
  CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15)
  read CVSharedParams from __ssbuf__
  continue compute
```

## 3. 关键参数 / 修改点

| 修改点 | 内容 | 收益来源 |
|--------|------|----------|
| kernel entry 分支 | A5 cube 分支不再执行 full `GET_TILING_DATA_WITH_STRUCT`，向 `Init` 传 `nullptr` | 去除 cube 冗余 tiling 拷贝 |
| vector shared params | vector 从 full tiling 中抽取 `CVSharedParams` | 共享字段只推导一次 |
| CV 转发 | vector subblock 0 将 `CVSharedParams` 按 `uint32_t` 写入 `__ssbuf__` | 用紧凑字段替代 full tiling struct |
| 同步 | vector 写完后 `CrossCoreSetFlag`，cube 读前 `CrossCoreWaitFlag` | 保证 cube 读取到完整共享参数 |
| cube init 顺序 | cube 先执行不依赖共享参数的初始化，再等待 flag | 隐藏部分 vector tiling 提取延迟 |

`CVSharedParams` 只放 cube 必需字段，例如 batch、seq、qSeq、group、sparse block count、used core num 和必要 scale 字段。结构体字段需保持 word-copy 友好。

## 4. 核心代码骨架

### kernel entry

```cpp
// A5 cube branch: skip full tiling copy
op.Init(query, key, value, sparseIndices, keyScale, valueScale,
        blocktable, actualSeqLengthsQuery, actualSeqLengthsKV,
        attentionOut, user, nullptr, &tPipe);

// A5 vector branch: keep full tiling copy
GET_TILING_DATA_WITH_STRUCT(KvQuantSparseFlashAttentionTilingDataMla,
                            tilingDataIn, tiling);
const auto *tilingData = &tilingDataIn;
op.Init(..., tilingData, &tPipe);
```

### vector 写入 / cube 读取

```cpp
auto ssbuf = reinterpret_cast<__ssbuf__ uint32_t *>(0);
auto src = reinterpret_cast<uint32_t *>(&sharedParams);
#pragma unroll
for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i) {
    ssbuf[i] = src[i];
}
CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
```

```cpp
CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
auto ssbuf = reinterpret_cast<__ssbuf__ uint32_t *>(0);
auto dst = reinterpret_cast<uint32_t *>(&sharedParams);
#pragma unroll
for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i) {
    dst[i] = ssbuf[i];
}
```

## 5. 效果与适用边界

| 场景 | 建议 |
|------|------|
| AIC/AIV 都 full copy TilingData，cube 只用少量字段 | 适用 |
| vector 不在启动关键路径，或 cube 可先做独立初始化 | 适用 |
| cube 后续需要大量 full tiling 字段 | 不适用，压缩收益不足 |
| `CVSharedParams` 过大或字段含复杂非 word-copy 类型 | 不适用 |
| `__ssbuf__` 地址或 flag id 已被其他通道使用 | 必须重新分配，不能复用冲突位置 |

效果：减少 cube 侧 full TilingData 拷贝和启动 Scalar 指令。该案例未提供具体量化数据，因此不编造收益数字；落地时以启动 timeline、cube 侧头开销和主/control shape 精度作为验收。
