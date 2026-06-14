# PatchCore 代码全面检查报告

> 生成时间：2025-07-09

---

## 1. 比赛要求

| 要求 | 状态 | 说明 |
|------|:----:|------|
| 昇腾 800I A2 NPU 适配 | ✅ | 全程 torch_npu，`inference.py` 默认 `--device auto` 自动检测 NPU/CPU |
| 端到端推理（训练+测试） | ✅ | `inference.py` 单脚本完成训练→推理→评估完整流程 |
| 环境自动检测与参数调优 | ✅ | 新增 `detect_npu()` / `detect_cpu_count()` / `detect_ram_gib()` / `detect_disk_gib()` / `auto_tune_params()`，容器感知 |
| 15 类 MVTec-AD 全覆盖 | ✅ | `ALL_CATEGORIES` 15 类，支持 `--categories` 子集指定 |
| 输出指标：ImgAUROC / PixAUROC / PRO | ✅ | 逐类打印 + 汇总平均，保存 CSV |
| 端到端耗时 | ✅ | 每个类别计时 + 总 wall time |
| 原论文精度对标 | ✅ | `docs/RESULTS.md` 给出与论文对比：ImgAUROC 0.990 vs 0.990 |

> 比赛提交入口就是 `inference.py`，参数清晰、输出规范。

---

## 2. 程序结构

```
Patchcore/
├── src/
│   ├── inference.py               ← 比赛提交入口（端到端训练+推理）
│   └── patchcore/
│   ├── __init__.py             # 模块导出
│   ├── patchcore.py            # PatchCore 核心类 (load/fit/predict) + PatchMaker
│   ├── baihu_nn.py             # 【NPU 适配核心】NPU 最近邻搜索引擎
│   ├── common.py               # FaissNN / 特征聚合 / NearestNeighbourScorer
│   ├── sampler.py              # 采样策略：GreedyCoreset / Approximate / Random
│   ├── backbones.py            # 骨干网络注册表（23种支持）
│   ├── metrics.py              # AUROC / PRO 评估指标
│   ├── utils.py                # 日志、绘图、CSV 结果存储
│   └── datasets/
│       └── mvtec.py            # MVTec-AD 数据集加载器
├── bin/
│   ├── run_patchcore.py        # 原始训练管道（click 链式命令）
│   ├── load_and_evaluate_patchcore.py  # 加载预训练模型评估
│   └── verify_precision.py     # BaihuNN vs Faiss 精度验证
├── test/
│   ├── test_patchcore.py       # 核心类单元测试
│   ├── test_sampler.py         # 采样器单元测试
│   ├── test_pro_metric.py      # PRO 指标单元测试
│   └── test_common.py          # 通用工具单元测试
├── test_npu.py                 # NPU 一键测试入口
└── docs/RESULTS.md             # 完整实验结果
```

**结构评价**：清晰、分层合理。`inference.py` 是比赛入口，核心算法在 `src/patchcore/` 下独立模块化。

---

## 3. 算法使用

**PatchCore（Roth et al. CVPR 2022）** 的完整实现链：

```
训练图片 → WideResNet50 骨干 → layer2+layer3 特征提取
  → PatchMaker 分块（patchsize=3, stride=1）
  → Preprocessing (MeanMapper 降维 1024)
  → Aggregator (adaptive_avg_pool 聚合)
  → GreedyCoresetSampler (p=0.1/0.01) 采样核心集
  → NearestNeighbourScorer 存入记忆库

测试图片 → 同上提取特征 → NN 搜索最近邻
  → 异常得分 = 最近邻 L2 距离均值
  → PatchMaker.unpatch_scores 恢复空间布局
  → RescaleSegmentor + Gaussian 平滑 → 分割图
```

**NPU 独特优化**──两阶段 BaihuNN 搜索：

- **第一阶段**：FP16 矩阵乘法粗筛（`‖q-x‖² = ‖q‖² + ‖x‖² - 2·q·xᵀ`），候选 top-(K×8)
- **第二阶段**：CPU FP32 精排恢复精度
- 实测距离 MRE < 1e-5，索引一致率 > 99%

---

## 4. 实现代码检查

### ✅ 正常的部分

| 模块 | 状态 | 评价 |
|------|:----:|------|
| `inference.py` | ✅ | 参数完整，异常处理到位（`try/except` 兜底），CSV 输出规范；**新增自动环境检测+参数调优** |
| `patchcore.py` | ✅ | 类设计清晰，`load/fit/predict` 三阶段，NPU/CPU 自动分支 |
| `baihu_nn.py` | ✅ | FP16+FP32 两阶段设计精巧，自动回退机制，内存管理合理 |
| `common.py` | ✅ | FaissNN/NearestNeighbourScorer/特征聚合，功能完整 |
| `metrics.py` | ✅ | AUROC、PRO 指标实现正确，有充分的单元测试覆盖 |
| `sampler.py` | ✅ | GreedyCoreset + ApproximateGreedyCoreset + Random，实现完整 |
| `datasets/mvtec.py` | ✅ | 标准 PyTorch Dataset，ImageNet 归一化，mask 加载正确 |

### ⚠️ 潜在问题 / 可改进点

| 问题 | 文件 | 说明 | 影响 |
|------|------|------|:----:|
| 🔴 `_spade_nn` 参数未使用 | `test_patchcore.py` L56 | `_standard_patchcore` 传入 `spade_nn=2` 但 `PatchCore.load()` 没有这个参数 | 只是测试代码的无用参数，不影响功能 |
| 🟡 `append` 在 `_predict` 中 | `patchcore.py` L208-214 | `_predict_dataloader` 使用 `scores.append()` / `masks.append()` 而非预分配 | 小数据集无影响 |
| 🟡 PRO 计算逐阈值循环 | `metrics.py` | 1000 个阈值每个都遍历所有图片 | 15 类时可能增加数十秒 |
| 🟡 默认 `--num_workers` | `inference.py` L99-104 | 默认 `num_workers=4`，NPU 下建议匹配物理核数 | 性能优化项 |
| 🟡 `pin_memory=True` 在 NPU | `inference.py` L224 | `pin_memory=True` 对 NPU 可能无效，但无害 | 不报错 |

---

## 5. 代码接口与关联模块

```
inference.py (比赛入口)
  ├── detect_npu() / detect_cpu_count() / detect_ram_gib()
  │     └── detect_npu_mem_gib() / detect_disk_gib() → 环境自动检测
  ├── auto_tune_params() → 根据检测到的资源自动调优 num_workers / batch_size / sampler_percentage
  ├── build_device() → Tuple[torch.device, bool] (支持 "auto" 自动检测 NPU/CPU)
  ├── get_dataloader() → DataLoader (调用 datasets.mvtec.MVTecDataset)
  ├── run_single_category()
  │     ├── backbones.load(args.backbone) → 骨干网络
  │     ├── BaihuNN() / None → 自动切换 NPU/CPU
  │     ├── PatchCore.load() → 模型构建
  │     ├── PatchCore.fit() → 训练（embed + sampler + scorer.fit）
  │     ├── PatchCore.predict() → 推理
  │     └── metrics.* → AUROC/PRO 评估
  └── 结果 CSV 写入

src/patchcore/
  ├── backbones.py           → _BACKBONES 字典，load(name) 返回模型
  ├── baihu_nn.py            → BaihuNN 类，fit/run/save/load 接口
  ├── common.py              → FaissNN / NearestNeighbourScorer / Aggregator / Preprocessing
  ├── sampler.py             → IdentitySampler / BaseSampler / GreedyCoresetSampler / RandomSampler
  ├── metrics.py             → compute_imagewise_retrieval_metrics / compute_pro_metric
  ├── utils.py               → plot_segmentation_images / write_results_to_csv
  ├── patchcore.py           → PatchCore (依赖 common/sampler/backbones/baihu_nn)
  └── datasets/mvtec.py      → MVTecDataset (ImageNet 归一化)
```

**接口一致性检查**：

- `BaihuNN.fit/run/save/load/reset_index` ←→ `FaissNN.fit/run/save/load/reset_index` ✅ 接口完全对齐
- `PatchCore.load_from_path()` → `nn_method` 注入 ✅
- `inference.py` 中 `device_context` 兼容 NPU 和 CPU ✅
- `sample_training.sh` / `sample_evaluation.sh` 给出完整 GPU 参考命令 ✅

---

## 📌 总结

| 维度 | 评分 | 说明 |
|------|:----:|------|
| 比赛要求匹配度 | 10/10 | 参数设计、输出格式、NPU 适配完全达标 |
| 结构清晰度 | 9/10 | 模块分离合理，入口明确 |
| 算法正确性 | 10/10 | 完整复现 PatchCore 论文流程 + NPU 优化 |
| 代码质量 | 8/10 | 类型标注完整，异常处理到位，少量可优化点 |
| 接口兼容性 | 10/10 | BaihuNN↔FaissNN 接口一致，自动切换无感 |

> 无功能性阻塞问题。正式运行前确认 `pip install -e .` 已安装即可。
