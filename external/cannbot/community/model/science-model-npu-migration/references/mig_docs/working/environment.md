# 环境快照（environment.md）

> 过程记录，位于 **`mig_docs/working/`**。按 [environment-setup-objectives.md](../../environment-setup-objectives.md) 与 [part-03-environment.md](../../part-03-environment.md) **§4.0.1～4.0.3** 填写。**可行性预判**见 part-02 → `Mig_report` §2.2，**不替代**本节门禁判定。  
> **范围**：仅记录运行时/CANN/驱动与框架插件可见性；**不要求** ATC/OM 模型转换工具。

## 元信息

- **generated_at**: （ISO 或本地时间）
- **check_summary**: PASS=… FAIL=…
- **set_env**: `（set_env 脚本路径，未加载写「未加载」）`
- **requirements**: `（实际使用的 requirements 路径）`
- **venv_dir**: `（如 .venv）`
- **data_dir**: `（可选，与 Mig_Readme §3.1/§3.2 对齐时填写）`

## Ascend / CANN（set_env 后）

- **ASCEND_TOOLKIT_HOME**: `…`
- **ASCEND_INSTALL_PATH**: `…`

## Python

- **system_python**: `…`
- **venv_python**: `…`

## 设备可见性（原始输出，可含沙箱内/外对照）

```text
（粘贴 npu-smi info 或等价只读检测的完整输出；若做过沙箱外复检，分别标注）
```

## 框架 Ascend 插件（原始输出）

```text
（粘贴 torch_npu / MindSpore Ascend 等版本与 import 自检输出；不含 ATC/OM 转换工具检查）
```

## 依赖与 README/requirements 适配判定（4.0.2～4.0.3 填写）

- **readme_python_hint**: （README 声明的 Python / OS / CUDA 等要点）
- **requirements_notable_pins**: （关键包及版本约束摘录）
- **文档是否声明 CANN/驱动版本**: （有则摘录；无则写「未声明」）
- **与 environment.md 机器事实对比摘要**: （一两句话：一致 / 冲突点）
- **判定**: **AUTO** | **MANUAL_STOP** | **UNKNOWN**
- **手动需求清单**（仅当 MANUAL_STOP）：逐项列出需用户安装的驱动、CANN 运行时、框架插件版本等；**勿**要求 ATC/OM 离线转换作为代码级迁移前置
