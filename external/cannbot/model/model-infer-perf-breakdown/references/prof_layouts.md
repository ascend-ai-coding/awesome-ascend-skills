# CANN / Ascend 性能采集目录布局参考

本 skill 在 Step 1 消费两个文件：**算子粒度 csv**（每个 op 一行，含 `name` / 时间 / shape）+ **trace timeline json**（chrome trace 格式）。Ascend 生态有两条采集链产出这两个文件，目录布局与文件命名差别较大；agent 在写 `inputs.json` 前**必须先识别 prof 是哪一种**，再去对应位置取 csv / json。

约定俗成的说法："`kernel_details.csv`" 与 "`trace_view.json`" 是 torch_npu.profiler（A 类）的产物名；"`op_summary_*.csv`" 与 "`msprof_*.json`" 是原生 msprof（B 类）的产物名。两者可互相充当本 skill 的输入，团队 wrapper 也常把 B 类改名为 A 类的名字——以**字段内容**为准，不要被文件名误导。

---

## A 类：torch_npu.profiler（PyTorch 适配，最常见）

PyTorch 调 `torch_npu.profiler.profile(...)` + `tensorboard_trace_handler` 产生。每个 rank 一个独立 top-level 目录。

```
<output_root>/
├── <ts>_host_<pid>_rank0_ascend_pt/        # rank 0（一卡一目录）
│   ├── profiler_metadata.json              # 环境变量 / 自定义 metadata
│   ├── logs/                               # 解析日志
│   ├── ASCEND_PROFILER_OUTPUT/             # ← 本 skill 主消费这里
│   │   ├── trace_view.json                 # Step 1 / Step 4 用（chrome trace）
│   │   ├── kernel_details.csv              # Step 1 用（算子粒度，row≈一条算子）
│   │   ├── operator_details.csv            # Torch op 层（PyTorch 算子，非 NPU kernel）
│   │   ├── memory_record.csv
│   │   └── l2_cache.csv                    # 仅当 l2_cache=True 时存在
│   ├── FRAMEWORK/                          # 框架层原始数据（GE / ACL 等）
│   └── PROF_<id>_<ts>/                     # CANN 底层 raw 数据（B 类同款结构）
└── <ts>_host_<pid>_rank1_ascend_pt/        # rank 1
    └── ...（同上）
```

**多卡识别要点**：
- top-level 一卡一目录，目录名末尾 `_rank<N>_ascend_pt`
- agent 看到多个 `*_ascend_pt/` 时必须列出所有 rank 让用户挑（一般用户只关心特定 rank，跨 rank 算子序列也不可比）
- `kernel_details.csv` 已经按卡分开，不需要按 device 再切

**`inputs.json` 写法**：
```json
{
  "prof_dir": "/abs/.../<ts>_host_<pid>_rank0_ascend_pt",
  "kernel_csv": "/abs/.../<ts>_host_<pid>_rank0_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
  "trace_json": "/abs/.../<ts>_host_<pid>_rank0_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json",
  "model_script_paths": ["/abs/.../modeling_xxx.py", ...],
  "phase": "decode",
  "rank": 0
}
```

---

## B 类：原生 msprof / CANN 工具链

`msprof` CLI 或 AscendCL Profiling API 产出。一次采集产一个 `PROF_<id>_<ts>_<hash>/`。

```
PROF_<id>_<ts>_<hash>/
├── host/data/                              # host 侧 raw
├── device_<id>/data/                       # device 侧 raw，按卡分子目录
├── mindstudio_profiler_log/log/            # 解析日志
└── mindstudio_profiler_output/             # ← 本 skill 主消费这里（解析产物）
    ├── msprof_*.json                       # ← Step 1 / Step 4 用（trace timeline）
    ├── op_summary_*.csv                    # ← Step 1 用（算子粒度）
    ├── task_time_*.csv                     # Task Scheduler
    ├── op_statistic_*.csv                  # 算子级统计
    ├── api_statistic_*.csv                 # CANN API 统计
    └── README.txt
```

部分老版本 / 部分采集模式下 csv 落在 `device_<id>/summary/op_summary.csv`，json 落在 `device_<id>/timeline/msprof_*.json`；新版统一收敛到 `mindstudio_profiler_output/`。两处都有时优先用 `mindstudio_profiler_output/`。

**多卡识别要点**：
- 同一 PROF 下并列多个 `device_<N>/` → 单次采集多卡
- 同一 output 根下并列多个 `PROF_<id>_<ts>_<hash>/` → 多次采集（每次可能采不同 device）
- agent 看到任一种多卡形态都必须列 `(PROF, device)` 组合让用户挑

**`inputs.json` 写法**：
```json
{
  "prof_dir": "/abs/.../PROF_000001_20260522_100000_XXXX",
  "kernel_csv": "/abs/.../PROF_.../mindstudio_profiler_output/op_summary_20260522_100000.csv",
  "trace_json": "/abs/.../PROF_.../mindstudio_profiler_output/msprof_20260522_100000.json",
  "model_script_paths": [...],
  "phase": "decode",
  "rank": 0
}
```

---

## phase 与 device / rank 的关系

- **phase**（prefill / decode / encode / ...）**不在目录名里**，无论 A / B 类都得问用户。同一个采集任务可能只跑 prefill，也可能 prefill + decode 各采一次。
- **rank** 在 A 类目录名里直接给出；B 类需要看 `device_<id>/` 或用户告知。每 process 绑一张卡时 `rank == device_id`，分布式训练 rank 与 device 不一一对应时由用户给。
- 多卡 / 多次采集 / 多 phase 任一维度组合数 > 1 时，agent **必须**枚举所有组合让用户挑，每个选中的 `(phase, rank/device)` 各起一个 sub `<run_dir>` 独立跑完整 skill。

---

## 参考

- 《Profiling 工具使用指南》（CANN 商用版 8.x 系列），昇腾社区官方文档
  入口：https://www.hiascend.com/document/redirect/CannCommunityVersionList
- torch_npu profiler 输出说明（PyTorch 适配，昇腾社区）
- Ascend/pytorch 仓库 `profiler/` 子模块 README：https://gitee.com/ascend/mstt
- CANN 版本号会持续滚动（截至 2026-05 商用版主线 8.x），各版本目录命名细节略有差异——以**实地 `ls` 看到的结构**为准，本文档给的是当前主流形态。
