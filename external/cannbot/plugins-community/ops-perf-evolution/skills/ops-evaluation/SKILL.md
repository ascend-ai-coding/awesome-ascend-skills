---
name: external-cannbot-plugins-community-ops-perf-evolution-skills-ops-evaluation
description: 从 ops-nn/cv/math/transformer 等算子仓库构建、部署并评估 AscendC 算子，比较基线与进化后的性能差异。当需要构建
  ops 仓库算子、运行正确性验证与性能评测、或对比基线版本与进化版本差异时使用。
original-name: ops-evaluation
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

## 功能说明

使用 `build.sh` 构建 ops 仓库算子，安装到指定路径，生成 PyBind 绑定，运行正确性验证与性能评测，并对比基线版本与进化版本的差异。

## 使用场景

适用于 ops 仓库算子进化优化场景，需要评估算子在优化前（基线）和优化后（进化）的性能差异时使用。

## 执行流程

**Python 环境**：优先使用 `.venv/bin/python3`，除非 `.venv` 不存在才回退到系统 `python3`。

**必须按顺序执行以下所有步骤，不允许跳过，也不允许自行编写评估脚本。**

### 步骤 1：构建并安装算子

调用 `build_ops.py` 从 ops 仓库构建并安装算子。

```bash
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/build_ops.py \
    --repo-root {REPO_ROOT} \
    --op-name {custom_op_name} \
    --soc {soc} \
    --install-path {absolute_install_path}
```

**注意**：`--install-path` 必须使用**绝对路径**，相对路径会导致安装失败。

脚本内部实际执行：
```bash
cd {REPO_ROOT}
rm -rf build/ build_out/
bash build.sh --pkg --vendor_name=custom --soc={soc} --ops={custom_op_name} -j$(nproc)
./build_out/cann-ops-*-custom-linux.*.run --install-path={install_path}
```

基线版本与进化版本需要分别构建：
1. 先构建基线版本（仓库原始代码）
2. 将修改后的代码应用到仓库
3. 构建进化版本，安装到另一个路径
4. 通过 `git checkout` 恢复原始代码

### 步骤 2：生成 PyBind

运行 `generate_pybind.py` 生成 PyBind 绑定并安装 whl。

```bash
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/generate_pybind.py {op_name} \
    --work-dir {install_path_dir}
```

PyBind 只需生成**一次**（基线和进化版本的 aclnn 接口相同）。

### 步骤 3：评估正确性与性能

调用 `evaluate_ops.py` 对比基线版本与进化版本。

```bash
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/evaluate_ops.py {op_name} \
    --baseline-path {baseline_install_path} \
    --evolved-path {evolved_install_path} \
    --reference-py {path_to_reference.py} \
    --custom-py {path_to_custom.py} \
    --device-id {device_id} \
    --task-type {vector|cube}
```

脚本使用**子进程隔离**分别评估两个版本（CANN runtime 加载 OPP 库后无法在运行时切换）：
- 子进程 1：`ASCEND_CUSTOM_OPP_PATH={baseline}/vendors/{vendor_subdir}` → 正确性 + 性能分析
- 子进程 2：`ASCEND_CUSTOM_OPP_PATH={evolved}/vendors/{vendor_subdir}` → 正确性 + 性能分析

可选参数：
- `--device-id`：NPU 设备 ID（默认 0）
- `--task-type`：用于性能瓶颈分析的算子类型（`vector`、`cube`、`cv-mix`、`unknown`）
- `--output`：评估结果 JSON 输出路径（默认在 evolved 路径下生成 `evaluation_results.json`）
- `--num-trials`：性能分析采样次数（默认 20）

### 步骤 4：对比结果

脚本会自动合并两个版本的性能分析数据，并生成对比报告 `evaluation_results.json`。

## 输出格式

`evaluation_results.json`：
```json
{
  "op_name": "ada_layer_norm",
  "repo_type": "nn",
  "soc": "ascend910b",
  "baseline": {
    "install_path": "output/.../baseline",
    "time_us": 456.5,
    "precision_passed": true,
    "profiling_dir": "output/.../baseline_profiling/",
    "pipeline": {"mte2_pct": 38.0, "vec_pct": 47.0, "scalar_pct": 12.0, "mte3_pct": 18.0},
    "bottleneck": "memory_bound"
  },
  "evolved": {
    "install_path": "output/.../evolved",
    "time_us": 233.8,
    "precision_passed": true,
    "profiling_dir": "output/.../evolved_profiling/",
    "pipeline": {"mte2_pct": 28.0, "vec_pct": 49.0, "scalar_pct": 8.0, "mte3_pct": 15.0},
    "bottleneck": "balanced"
  },
  "comparison": {
    "speedup": 1.95,
    "time_delta_us": -222.7,
    "bottleneck_change": "memory_bound -> balanced",
    "compilation_success": true,
    "precision_passed": true
  }
}
```

## 常见错误

- `ASCEND custom OPP directory not found`：检查 `--install-path` 是否为绝对路径
- 构建失败：检查 `ASCEND_HOME_PATH` 是否已设置；检查 `--ops=` 后的算子名称是否正确
- 仓库类型自动检测：通过 `.run` 文件名关键字（`ops-nn`/`ops-cv`/`ops-math`）或 `build.sh` 中的 `REPOSITORY_NAME` 自动识别
- `Cannot switch OPP path at runtime`：`evaluate_ops.py` 已通过子进程隔离处理，不要尝试在同一个进程中评估两个版本

## build.sh 注意事项

**关键**：在 docker + tmux 环境中，ops 仓库的 `build.sh` 应直接调用（不要 `source`），因为它设计为独立脚本，与自定义算子的 `build.sh` 不同。

## 目录结构

完整评估后的输出结构：
```
output/{op_name}_ops-evo_{timestamp}/
├── baseline/                        # 基线安装目录
│   └── vendors/custom_nn/...
├── evolved/                         # 进化安装目录
│   └── vendors/custom_nn/...
├── baseline_profiling/              # 基线性能分析数据
├── evolved_profiling/               # 进化性能分析数据
└── evaluation_results.json          # 对比报告
```
