# Kirin9030 平台铁律：容器 omg、UINT4、拆分网格权重

9030 与 9020 的量化链路在底层有两处根本差异，违反任意一条都会在 omg 阶段报错。本文展开两条铁律的背景、落地命令与原理。

---

## 铁律 1：omg 必须在容器内跑

**原因**：9030 的 omg + AscendC 工具链需要 glibc ≥ 2.35（含新版 `GLIBCXX`），典型量化开发机（如 CentOS / 老 Ubuntu，glibc 2.17）直接跑会崩或报 glibc 缺符号。

**做法**：用一个 glibc ≥ 2.35 的容器（如基于 Ubuntu 22.04/24.04 的镜像），把工作目录挂进去，在容器内执行 omg。容器内必须：

```bash
# 1) 激活 AscendC 工具链（关键）
source <ddk>/tools/tools_ascendc/install.sh
source <ddk>/tools/tools_ascendc/set_ascendc_env.sh
which ccec          # 必须非空，否则 W4 MatMul 退回预编译 kernel → 报 type MatMul don't support

# 2) 视镜像情况补 libstdc++（确保 GLIBCXX_3.4.32 可用）
export LD_PRELOAD=<conda>/lib/libstdc++.so.6.0.34   # 按实际路径

# 3) 跑 omg（见 quantization-pipeline.md 阶段 5b）
```

**为什么 `which ccec` 不能空**：9030 的 W4 MatMul kernel 是 AscendC 即时编译的；工具链没激活时 omg 找不到 `ccec`，退回预编译 kernel，而预编译 kernel 不支持 W4，于是报 `type MatMul don't support`。

> 标点 punc 等模型若本地无合规容器，可把图 + quant_param_file 打包，拿到合规机器（Ubuntu 24.04，glibc ≥ 2.35）上转 omc 再回传。

---

## 铁律 2：9030 W4 内核只支持 UINT4（[0,15]），不支持 INT4

9020 的 W4 用 INT4（有符号，-1 合法）；9030 的 W4 kernel 只认 UINT4（无符号 [0,15]）。带来两个强制要求：

### 2.1 量化参数按 UINT4 打包

阶段 ② `generate_quant_params(...)` 必须 `quant_param_2=True`。`False` 会按 INT4 打包，9030 omg 报 `type MatMul don't support`。

### 2.2 权重必须落在 dopt 4bit 网格上

UINT4 只能表示 [0,15]，若权重是任意原始浮点（含负值），打包时报：

```
CompressInt4ToInt8 Value -1 out of range [0,15]
```

**保证落网格的两种途径**：

- **整网量化**：阶段 ③ 用 `s3_export.py` 加载 `fake_quant_weight.pth`（量化-反量化后的权重）再 torch 导出——权重天然落在 dopt 网格上。**不要**导入预先固定的浮点 onnx。
- **拆分子图**：见下节「拆分网格权重」。

---

## 拆分场景的网格权重处理

当把 encoder 拆成多段独立 omc、且复用整网的 `quant_param_file`（不对每段重新 PTQ）时，会遇到 9030 专属问题：

- 整网量化参数是按「dopt 量化到网格后的权重」标定的（整网 onnx 权重已是 [0,15] 的 UINT4 码）。
- 而拆分子图通常来自**原始浮点权重**（含负值），套同一份参数会量化出 -1 → UINT4 越界。

**解法（`s5_grid_weights.py` 思路）**：把整网已落网格的 MatMul 权重，按节点名搬进每个拆分子图，替换其原始浮点权重。这样拆分子图也落在 [0,15] 网格上，复用整网参数即可。

- 只有 4bit（UINT4）的 MatMul 需要搬网格权重。
- 8bit（`Quant_aigc_ptq` 非对称）的件（如 emd / ctc）容忍原始权重，不需替换。
- 9020 拆分用 INT4（-1 合法）所以没这问题；**这步是 9030 专属**。

> 拆分流水线全貌与打包见 `split-and-packing.md`。

---

## 9030 vs 9020 对照（避免混用）

| 维度 | Kirin9020 | Kirin9030 |
| --- | --- | --- |
| W4 编码 | INT4（有符号，-1 合法） | UINT4（[0,15]，-1 越界） |
| `quant_param_2` | False | **True** |
| 权重落网格 | 不强制（INT4 容忍负值） | **强制**（torch 导出 / 搬网格权重） |
| omg 运行环境 | 宿主可跑 | **必须容器**（glibc ≥ 2.35 + AscendC 工具链） |
| 拆分网格权重替换 | 不需要 | **需要** |
