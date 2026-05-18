---
name: ascend-dmi
description: |
  当用户需要对华为昇腾 NPU 进行硬件层面的管理、测试或诊断时使用此 skill。典型场景：

  - 查看 NPU 卡的状态、温度、利用率
  - 测试内存带宽（h2d/d2h/d2d/p2p）
  - 跑算力/功耗基准测试（TFLOPS、TOPS）
  - 诊断 NPU 硬件故障或做健康检查
  - 对 NPU 卡做压力测试（aicore、内存）
  - 复位/恢复卡住或异常的 NPU 卡

  典型用户问题（即使不提 ascend-dmi 也应触发）：
  - "帮我测一下这台服务器的算力和带宽"
  - "看看 192.168.1.50 上 NPU 的温度和利用率"
  - "跑个 fp16 算力测试，看看 TFLOPS 多少"
  - "d2d 带宽有没有达标"
  - "3 号卡好像有问题，帮我诊断一下"
  - "NPU 卡住了，复位一下"
  - "对 8 张卡做个压力测试"
  - "跑个功耗测试，edp 模式"

  Use this skill whenever the user wants to measure, diagnose, or manage Ascend NPU hardware directly — including checking server NPU performance (算力/带宽/功耗), diagnosing hardware faults, or resetting NPU cards. Trigger even when the user only mentions "服务器" + "算力/带宽/NPU状态" without saying "ascend-dmi".

  Do NOT use for: writing NPU code, installing frameworks, debugging training issues, or Docker configuration.
---

# Ascend DMI 工具使用指南

Ascend DMI（Ascend Device Management Interface）是华为昇腾 NPU 设备的管理和诊断工具。

**你必须严格按照本文件的步骤顺序执行，不得跳步、不得自行发明替代方案。** 每一步都有明确的通过条件，未通过则不能进入下一步。这不是参考文档，是你必须遵守的操作规程。

## 快速决策：根据操作类型选择路径

```
用户请求
  │
  ├─ 信息查询（-v/-i/-c/-h）或眼图测试（--sq）
  │   → 【快速路径】检查 ascend-dmi → 确认参数 → 构建命令 → 直接执行 → 解读输出
  │
  ├─ 性能测试（--bw/-f/-p）或 healthCheck/performanceCheck
  │   → 【标准路径】检查 ascend-dmi + CANN → 确认参数 → 构建命令 → 告知风险并确认 → 执行 → 解读输出
  │
  └─ 压力测试（--dg -s）或 NPU 恢复（-r）
      → 【高风险路径】检查 ascend-dmi + CANN → 确认参数 → 构建命令 → 二次确认 → 检查设备占用 → 执行 → 解读输出
```

注意：标准路径和高风险路径都**必须完成 CANN 检查**才能执行命令。见第 2 步。

## 第 1 步：远程还是本地？

- **本地**（用户未提及远程 IP/主机名）：直接在当前服务器执行 `ascend-dmi` 命令
- **远程**（用户提到了 IP 地址、主机名或"远程服务器"）：**必须先调用 `remote-server-guide` skill 建立连接**，不要自己尝试 SSH。原因：直接 `ssh` 无法在非交互环境中输入密码，会认证失败。remote-server-guide 会通过 tmux 等方式安全处理密码输入。连接建立后，再在远程执行后续步骤。

  **禁止行为**：不要直接执行 `ssh user@host "command"`，这在 Claude 环境中一定会失败（无法输入密码）。

## 第 2 步：环境检查

环境检查分两部分，**必须按顺序完成两部分后才能执行命令**。

### 2a. 检查 ascend-dmi

```bash
which ascend-dmi && ascend-dmi -v
```

如果不可用：
```bash
source /usr/local/Ascend/toolbox/set_env.sh
which ascend-dmi
```
仍不可用则读取 [references/environment-setup.md] 按步骤安装 MindCluster ToolBox。

### 2b. 检查 CANN（性能测试、诊断、压力测试必须）

带宽测试（--bw）、算力测试（-f）、功耗测试（-p）、健康检查（--dg）、压力测试（-s）都依赖 CANN 环境。**不完成 CANN 检查就执行这些命令一定会失败，没有例外。**

**验证命令（整个流程中反复使用这个命令来判断 CANN 是否可用）：**
```bash
python3 -c "import acl; print(acl.get_soc_name())"
```
- 输出芯片型号（如 `Ascend910B`）= CANN 可用 ✓ → 进入第 3 步
- 任何报错（ModuleNotFoundError、ImportError 等）= CANN 不可用 ✗ → 执行下方修复流程

---

**CANN 不可用时的修复流程（必须按 1→2→3→4 顺序执行，不得跳步）：**

**1. 先问用户。** 向用户提问："CANN 安装在哪个路径？"然后等待用户回复。
- 用户给出路径 → `source <路径>/set_env.sh` → 运行验证命令 → 通过则进入第 3 步，失败则告知用户该路径不可用并继续步骤 2
- 用户说不知道 → 进入步骤 2

**2. 征得用户同意后自动查找。** 先问用户"是否允许我自动搜索 CANN 路径？"得到同意后，按以下优先级**逐个**尝试（CANN 8.5.0+ 目录名为 `cann`，旧版为 `ascend-toolkit`）：

| 优先级 | 尝试的命令 |
|-------|-----------|
| ② | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` → 验证 |
| ① | `source /usr/local/Ascend/cann/set_env.sh` → 验证 |
| ① | `source /home/miniconda3/Ascend/cann/set_env.sh` → 验证 |
| ③ | `find /home -path "*/Ascend/cann/set_env.sh" -o -path "*/Ascend/ascend-toolkit/set_env.sh" 2>/dev/null`，对找到的每个路径 source → 验证 |

每个优先级的处理方式完全相同：
```
source <路径>/set_env.sh → python3 -c "import acl; print(acl.get_soc_name())"
  ├─ 验证通过（输出芯片型号）→ 成功，进入第 3 步
  └─ 验证失败（任何报错）→ 放弃这个路径，尝试下一优先级
```

**"看到目录存在"不等于"CANN 可用"。** 目录存在、文件存在、set_env.sh 能 source，都不能代替验证命令。只有 `python3 -c "import acl; print(acl.get_soc_name())"` 成功才算 CANN 可用。验证失败直接放弃该路径。目录下不存在`set_env.sh`文件也直接放弃该路径。

**3. 全部失败 → 停下来。** 告知用户"自动查找 CANN 失败，请提供准确的 CANN 安装路径"，然后**停止操作，等待用户回复**。不要继续执行 ascend-dmi 命令。

---

**以下行为被禁止，因为它们会导致环境损坏或命令执行出错：**
- ✗ 跳过步骤 1 直接搜索（用户可能直接知道路径，搜索是浪费时间）
- ✗ 用 `export PATH=...`、`export PYTHONPATH=...`、`export LD_LIBRARY_PATH=...` 手动拼凑环境
- ✗ 验证失败后在同一个路径上反复尝试修复（换 python 版本、装 pip 包、改权限等）
- ✗ 看到 `/usr/local/Ascend/ascend-toolkit` 或 `/usr/local/Ascend/cann` 目录存在就认为找到了，不运行验证命令
- ✗ 所有路径都失败后仍然继续执行 ascend-dmi 测试命令

## 第 3 步：确认测试参数

在构建命令前，先向用户确认具体的测试参数。用户的请求中可能已包含部分参数，只需补全缺失的部分。**如果用户已在请求中明确指定了全部参数，则无需重复询问。**

先执行 `npu-smi info` 获取可用设备列表，以便向用户展示可选项。

### 各测试类型需确认的参数

**带宽测试**：
- 测试模式：h2d / d2h / d2d / p2p（可多选）
- 测试哪些卡（设备 ID），还是所有卡？
- 是否指定数据大小（`-s`）？注意：D2D 在 A2/A3 上不支持 `-s`
- P2P 是否指定源/目标设备（`-ds`/`-dd`）？不指定则测全部设备间矩阵

**算力测试**：
- 数据类型：fp16 / fp32 / int8 / bf16 / hf32（默认 fp16，不同产品支持不同，详见 [references/constraints.md]）
- 测试哪些卡（`-d`），还是整机（`--all`）？两者互斥
- 是否调整执行次数（`--et`，默认 60，单位十万）？

**功耗测试**：
- 测试类型：edp（估计设计功耗）还是 tdp（热设计功耗）？
- 测试时长（`--dur`，默认 600 秒）

**健康检查 / 性能规格**：
- 测试哪些卡，还是所有卡？

**压力测试**：
- 测试项目：aicore / hbm（片上内存）/ bandwidth（P2P）/ edp / tdp / aicpu？
- 测试哪些卡？注意：aicpu 必须单独执行，不能与其他项组合

**NPU 恢复**：
- 恢复哪张卡？（必须指定 `-d`）

**眼图测试**：
- 测试类型：pcie / hbm / roce / all？
- 测试哪些卡？

## 第 4 步：构建命令

根据用户确认的参数，按下表构建命令。

### 命令速查

| 操作 | 命令模板 | 风险 |
|-----|---------|------|
| 版本 | `ascend-dmi -v` | 无 |
| 设备状态 | `ascend-dmi -i [-b\|--dt]` | 无 |
| 兼容性检查 | `ascend-dmi -c` | 无 |
| 眼图测试 | `ascend-dmi --sq -t pcie\|hbm\|roce\|all [-d ID]` | 无 |
| 带宽测试 | `ascend-dmi --bw -t h2d\|d2h\|d2d\|p2p [-d ID] [-s SIZE] [-q]` | 中 |
| 算力测试 | `ascend-dmi -f -t fp16\|fp32\|int8\|bf16\|hf32 [-d ID] [--all] [-q]` | 中 |
| 功耗测试 | `ascend-dmi -p [-pt edp\|tdp] [--dur SEC] [-q]` | 中 |
| 健康检查 | `ascend-dmi --dg --se healthCheck [-d ID] [-q]` | 中 |
| 性能规格 | `ascend-dmi --dg --se performanceCheck [-d ID] [-q]` | 中 |
| 压力测试 | `ascend-dmi --dg --se stressTest [-d ID] [-q]` 或 `-s -i ITEM` | **高** |
| NPU 恢复 | `ascend-dmi -r -d ID [-q]` | **高** |

完整命令和参数组合见 [references/command-cheatsheet.md]。具体参数说明见 [references/parameters/] 下的各文件。

### 注意事项
- **`-q` 参数**：对中/高风险操作，在用户确认后添加 `-q` 跳过命令行交互提示

## 第 5 步：风险确认与执行

### 无风险操作（信息查询、眼图测试）

直接执行，无需额外确认。

### 中等风险操作（带宽/算力/功耗测试、健康检查、性能规格）

这些操作会独占 NPU 设备，中断正在运行的训练或推理任务。执行前：
1. 告知用户影响："该测试会独占 NPU，正在运行的训练/推理任务将被中断"
2. 获取用户确认后执行
3. 添加 `-q` 参数跳过命令行交互提示

### 高风险操作（压力测试、NPU 恢复）

这些操作可能导致长时间高负载、硬件过热或 NPU 暂时不可用。执行前：
1. 明确告知具体风险（如"压力测试会持续数小时高负载运行，可能触发硬件保护机制"）
2. 检查设备占用：`fuser -v /dev/davinci*`
3. 要求用户二次确认：确认已停止相关业务、了解风险
4. 添加 `-q` 参数

详细风险矩阵见 [references/risk-assessment.md]。

## 第 6 步：解读输出

执行完成后，提取关键指标并给出结论：

**带宽测试**：关注 Bandwidth(GB/s) 值。典型参考：H2D 20-28 GB/s, D2H 25-30 GB/s, D2D 760-800 GB/s, P2P 25-55 GB/s。注意：D2D 在 A2/A3 上传输大小由 AI Core 决定、不支持 `-s` 参数；P2P 不指定 `-ds`/`-dd` 时输出的是设备间矩阵。

**算力测试**：关注 TFLOPS/TOPS 值。注意区分测试模式：使用 `--all` 时，TFLOPS 是**整机所有卡的算力总和**，不是单卡算力；不带 `--all` 时才是单卡算力。判断是否达标时必须对应正确的规格（整机规格 = 单卡规格 × 卡数）。如显著低于产品规格，可能是温度过高触发降频。

**功耗测试**：关注 Power(W) 和 Temperature(°C)。温度应低于 90°C。

**诊断测试**：PASS = 正常，FAIL = 需处理，SKIP = 当前产品不支持，WARN = 关注警告级别。

详细结果解读方法见 [references/output-interpretation.md]。

## 资源索引

| 需要时 | 读取 |
|-------|------|
| 安装/配置 ascend-dmi 或 CANN | [references/environment-setup.md] |
| 查看完整命令和参数 | [references/command-cheatsheet.md] + [references/parameters/] |
| 详细风险评估 | [references/risk-assessment.md] |
| 深入解读测试结果 | [references/output-interpretation.md] |
| 使用约束和限制 | [references/constraints.md] |
| 命令模板示例 | [scripts/templates/] |
| 测试报告模板 | [assets/report-template.md] |
