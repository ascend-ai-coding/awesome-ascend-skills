---
id: D5
bottlenecks: [no_overlap]
op_families: [datapath]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# D5: BF16 & Platform-Specific Handling (BF16/多平台处理)

## 核心思想
专家实现针对不同的Ascend芯片（910B, 910_93, 910_95, 310P, kirinx90等）提供了特定的配置支持。这种多平台适配策略不仅体现在编译配置上，还深入到了Shape校验和平台特性检测层面。例如，在Ascend910_95上，算子支持2D shape (g, n)用于分组矩阵乘法场景，而在其他平台上只支持(1, n)或(n,)。这种差异化支持允许算子在不同的硬件能力和使用场景下都能获得最优性能。平台特性检测通过GetChipFeature函数实现，动态检测芯片是否支持特定功能（如Intrinsic_data_move_l12bt的bf16支持），从而实现运行时适配。

## 代码骨架

// === 改造前（基线）===
```cpp
this->AICore().AddConfig("ascend910b");
```

// === 改造后（专家模式）===
```cpp
this->AICore().AddConfig("ascend910_95");
this->AICore().AddConfig("ascend910_93");
this->AICore().AddConfig("ascend910b");
OpAICoreConfig config_kirin = GetKirinCoreConfig();
this->AICore().AddConfig("kirinx90", config_kirin);
```

## 关键修改点

1. 预期收益: 算子可移植性强，适配多种硬件; 算子可移植性强，支持多种华为Ascend芯片; 提升算子的可移植性和适用范围

## 常见陷阱

⚠️ 需要维护多个平台配置
⚠️ 需要额外的FP32 workspace内存和cast操作的开销
⚠️ 需要维护多套配置，增加测试负担

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
