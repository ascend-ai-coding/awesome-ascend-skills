# TTK 测试 Conv2D（Conv2DV2）使用说明

面向在本机用 TTK（ops-test-kit）做 `conv2dv2` Kernel 精度 + 性能测试时的操作手册。

---

## 1. 环境

```bash
source <conda_path>/etc/profile.d/conda.sh   # 按实际 conda 路径
conda activate <env_name>
source <cann_path>/bin/setenv.bash           # 按实际 CANN 安装路径
```

| 项 | 路径/值 |
|----|---------|
| CANN | 用户提供的 CANN 安装路径（含 `bin/setenv.bash`） |
| SoC | `ascend950` / `Ascend950PR_9599`（按实际设备） |
| TTK | 用户提供的 TTK 代码目录（`ops-test-kit`） |
| 算子源码 | 用户提供的 ops-nn 卷积目录（如 `ops-nn/conv/conv2d_v2`） |
| Golden 插件 | `<ops_nn_conv_dir>/tests/assets/golden.py` |
| Input 插件 | `<ops_nn_conv_dir>/tests/assets/input.py` |

检查设备：

```bash
cd <ttk_dir>
python3 -m ttk info
```

---

## 2. 编译与安装算子（可选但推荐）

按 ops-nn 文档编译自定义包（`--soc` 按实际设备）：

```bash
cd <ops_nn_root>   # ops-nn 仓库根目录
bash build.sh --pkg --soc=ascend950 --ops=conv2d_v2 --vendor_name=custom -j8
./build_out/cann-ops-nn-custom_linux-*.run --quiet
```

安装后一般落在：

```text
${ASCEND_HOME_PATH}/opp/vendors/custom_nn
```

`vendors/config.ini` 中应有：

```ini
load_priority=custom_nn,custom_math
```

注意：

- `--ops=conv2d_v2` 会连带编译依赖算子二进制，耗时很长。
- **不要单独强行** `export ASCEND_CUSTOM_OPP_PATH=.../custom_nn` 再跑 tiling：可能导致 `Operator conv2dv2 not found`。优先依赖 `load_priority`。
- 仅做 `-b=release` 测试时，也可直接用 CANN 内置 `ops_nn/conv2d_v2` 发布内核。

---

## 3. 用例 CSV

Kernel 模式字段参考 TTK 文档与 ST 样例：

关键点：

| 字段 | 说明 |
|------|------|
| `op_name` | 写 `conv2dv2` |
| `input_shapes` / `input_ori_shapes` | `(x, filter, bias, offset_w)`，可选输入用 `None` |
| `attributes` | **Python 字面量**，不要用 JSON `true/false`。例：`{'strides':[1,1,1,1], 'pads':[1,1,1,1], 'dilations':[1,1,1,1], 'groups':1, 'data_format':'NCHW', 'pad_mode':'SPECIFIC', 'enable_hf32':False, 'offset_x':0}` |
| `input_data_ranges` | fp16 慎用过大值域（如 `[-50,100]`），易溢出导致精度必挂 |
| `precision_tolerances` | 常用 `((0.001, 0.001),)` |

预览用例：

```bash
python3 -m ttk list -i <cases_csv>
```

---

## 4. 推荐测试命令（精度 + 性能）

**必须带 Golden 插件**，否则会出现 `BIN_GOLD: UNSUPPORTED`，精度失败、性能不可信。

```bash
cd <ttk_dir>

GOLDEN=<ops_nn_conv_dir>/tests/assets/golden.py
INPUT=<ops_nn_conv_dir>/tests/assets/input.py
CASES=<cases_csv>
OUT=<output_dir>/ttk_kernel_results.csv

python3 -m ttk kernel \
  -i "${CASES}" \
  -d=false -b=release \
  --dev 0 --pc=1 --run=3 \
  --compare close \
  --plugin "${GOLDEN},${INPUT}" \
  -o "${OUT}"
```

或使用本目录一键脚本（需设置环境变量）：

```bash
export CONDA_SH=<conda_path>/etc/profile.d/conda.sh
export CONDA_ENV=<env_name>
export CANN_SETENV=<cann_path>/bin/setenv.bash
export TTK_DIR=<ttk_dir>
export OPS_NN_CONV_DIR=<ops_nn_conv_dir>
export CASES_CSV=<cases_csv>
export OUT_DIR=<output_dir>

bash scripts/run_test.sh
```

### 参数含义

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `-d=false` | 关闭动态编译 | 只跑二进制，避免额外 dyn compile |
| `-b=release` | 发布内核 | 匹配已安装 `.o`/json |
| `--compare close` | 数值近似 | 对齐 CSV rtol/atol；默认 `stat_rel_err` 对 Conv2D 易过严 |
| `--plugin` | golden+input | 否则无 Golden |
| `--run=3` | 板端执行次数 | 性能取 profiling |
| `--pc=1` | 单进程 | 大 shape 更稳 |
| `-t <name>` | 可选 | 只跑单个 case |

### 调试单 case

```bash
python3 -m ttk kernel -i <cases_csv> -t <case_name> \
  -d=false -b=release --compare close \
  --plugin golden.py,input.py \
  --dump-on-fail --dump out,golden --dump-format npy
```

---

## 5. 结果字段怎么读

输出 CSV（`-o`）中与本任务相关的列：

| 列 | 含义 |
|----|------|
| `bin_perf_us` | 性能（微秒，Task Duration） |
| `precision_status` | 精度 PASS/FAIL |
| `perf_status` | 执行/profiling PASS/FAIL |
| `bin_tiling_key` | tiling key（十进制 + hex） |
| `bin_tiling_data` | tiling bytes 的 int32 流（uint64 拆成 lo/hi） |
| `bin_block_dim` | block dim |

本目录后处理工具：

```bash
# 汇总表
python3 scripts/summarize_results.py --cases <cases_csv> \
  --results <results_csv> --out <output_dir>/summary_table.csv

# 解码 tiling_data → 命名字段
python3 scripts/decode_tiling.py --results <results_csv> \
  --out <output_dir>/tiling_decoded.json
```

字段含义详见 `TILING_FIELDS.md`。常用关注：

- **L1**: `hoL1, woL1, kAL1, kBL1, nBL1`
- **L0**: `hoL0, woL0, kL0, nL0`
- **多核**: `batchDim, nDim, hoDim, woDim`
- **标志**: `pBufferFlag, hasBias, hf32Enable`

多数 case `woL1=woL0=0` 表示 **M-mode**（`Ho×Wo` 折到 M）。

---

## 6. 常见问题

### 6.1 `BIN_GOLD: UNSUPPORTED` / 精度全 FAIL

未加 `--plugin .../golden.py`。Kernel 模式不会自动找 torch.conv2d。

### 6.2 `attributes` 解析失败：`name 'false' is not defined`

CSV 里写了 JSON `false/true`，应改为 Python `False/True`。

### 6.3 `Operator conv2dv2 not found` / `OPTILING_FAILURE`

多为 `ASCEND_CUSTOM_OPP_PATH` 只指到 custom vendor 导致 host tiling 找不到。去掉该环境变量，依赖 `vendors/config.ini` 的 `load_priority`。

### 6.4 fp16 大值域精度极差（如 close≈0%）

输入过大导致 NPU 输出大量 `inf`，而 Golden 会做 fp16 截断。应缩小 `input_data_ranges`（shape/attr 可不变）。

### 6.5 默认比对过严

未指定 `--compare` 时可能走 `stat_rel_err`。Conv2D 建议显式 `--compare close`。

---

## 7. 本指南产物索引

| 文件 | 说明 |
|------|------|
| `TTK_Conv2D_Usage.md` | 本文件（使用说明） |
| `TILING_FIELDS.md` | tiling 字段释义 |
| `scripts/decode_tiling.py` | tiling_data 解码脚本 |
| `scripts/summarize_results.py` | 结果汇总脚本 |
| `scripts/run_test.sh` | 一键测试脚本（环境变量驱动） |

---

## 8. 最短复现清单

```bash
# 1) 环境
conda activate <env_name>
source <cann_path>/bin/setenv.bash

# 2) 跑测
export TTK_DIR=<ttk_dir>
export OPS_NN_CONV_DIR=<ops_nn_conv_dir>
export CASES_CSV=<cases_csv>
export OUT_DIR=<output_dir>
bash scripts/run_test.sh

# 3) 看表
less <output_dir>/summary_table.md
```
