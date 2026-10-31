# 技术卡片 mem-05：一次性数据绕过 L2

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：访存 / Cache
- **一句话**：确认某块数据后续不会被当前核或后续 wave 再读时，才用 L2 bypass 避免它污染可复用的 cache；有任何复用可能就保留正常 cache。

## 何时用（诊断特征）

- 大权重、scale 或输入块只被一个完整 problem/tile 消费一次，而热数据的 L2 命中受这些流式读挤压。
- 能在运行时由合法 shape/schedule 谓词证明“后续没有同一物理区域的读取”，而不是仅凭某个 benchmark case 猜测。
- Tensor 物理行跨度满足该 API 的 cache-line 对齐要求。

## 原理

流式一次性读填充 L2 会驱逐之后真正要复用的数据。对这样的读取设置 bypass，使数据不填入 L2；但若当前 problem 被拆成多个 M tile、多个 wave 或热点专家会被重复处理，L2 命中本身就是收益来源，bypass 会退化。

## 怎么改（before / after）

```cpp
// 伪代码：逻辑谓词由 host tiling 计算；SetL2CacheHint 的可用枚举以当前 CANN 头文件为准。
// before: 无条件沿用默认 L2 策略
gmWeight.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);

// after: 仅在可证明无后续复用且地址布局对齐时 bypass
bool noLaterReuse = coversWholeProblem && m <= tileM;
bool rowAligned = IsPhysicalRowCacheLineAligned(weightLayout);
bool canBypass = noLaterReuse && rowAligned;
gmWeight.SetL2CacheHint(canBypass ? CacheMode::CACHE_MODE_DISABLE
                                  : CacheMode::CACHE_MODE_NORMAL);
```

把 `noLaterReuse` 作为 tiling/调度导出的运行时谓词。例如 Wave 切分时，只有一个 wave 覆盖专家从首行到尾行、且同一 problem 只有一个 M tile，才允许绕过；权重和 scale 的物理 layout 不同，要分别判断对齐。对于通用 MatMul/QBMM，还应把用户开关、总工作集是否小于 L2、baseM 是否全载、滑窗 block 数和 L1 切分块对齐一起纳入谓词，不能只看单个矩阵的逻辑 shape。

## 性能数据

比较 L2 命中、DDR 流量和 `aiv_time`，至少覆盖“完整一次性读取”与“拆成多 tile/wave”的两类 case。后者必须验证回退到 normal cache 后不退化。

## 红线（Stop rules）

- **没有可证明的无复用谓词就不能开 bypass**；禁止按 case ID、固定 shape 列表或一次 profiling 结果硬路由。
- 对齐按物理布局而非逻辑 shape 判断。950 历史样本以 128B cache line 校验，ND/NZ 与 scale 的 stride 分别检查；实际 API 以当前 CANN 头文件为准。
- 同一数据被后续 M tile、其他 wave、其他核或重算路径读取时保持 normal cache。
- 该手段优化 cache 污染，不会减少 GM 读字节；带宽本已饱和且无热数据被驱逐时通常无收益。

## 关联

- 源：[ops-transformer `53d75d22`](https://gitcode.com/cann/ops-transformer/commit/53d75d22)（arch35 MegaMoE：仅完整专家且单 M tile 的 one-shot weight/scale 设置 L2 bypass，并按物理行跨度校验）、[ops-nn `485732746`](https://gitcode.com/cann/ops-nn/commit/485732746)（arch35 QBMMV3：由 host tiling 下发 `weightMustHitL2`，按 shape/dtype/broadcast/L2 容量判定）、[ops-nn `95133c68`](https://gitcode.com/cann/ops-nn/commit/95133c68)（arch35 MatMul：用户开关配合全载、单滑窗、128B 内轴和 L1 切分对齐条件选择 B/A L2 disable）。
- 相邻卡片：mem-04（数据有短距离复用时优先驻留 UB）、mem-02（以预取隐藏延迟）。
