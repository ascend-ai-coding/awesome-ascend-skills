# Skill 自检与更新规则

安装的 msobjdump 与本 Skill 的接口快照（[interface.json](interface.json)）可能不同步。机制分两层：运行期无状态探测（用户侧）与 PR 期快照再生（维护侧）。发现变化不等于新能力，一律先核验。

## 运行期探测（无状态）

`scripts/check_interface.py` 拿当前环境 `msobjdump -h` 与快照比对，不读不写任何缓存或仓库文件；同样的环境在任何机器、任何会话得到同样结论。双向漂移都可见：

| 状态 | 含义 | 处理 |
| --- | --- | --- |
| `ok` | 记载的必备参数齐备 | 正常使用；`optional_missing_options` 提示的能力按"安装早于该能力"说明限制 |
| `newer-install` | 安装有快照未记载的选项 | Skill 文档落后于安装；核对 asc-tools 变更或官方文档后再用新参数 |
| `missing-options` | 安装缺记载的必备参数 | 按安装版实际能力回答，不声称支持 |
| `unavailable` / `failed` | 命令缺失或 `-h` 失败 | source 匹配的 CANN 后重试 |

退出码：0 正常；3 需注意（`missing-options` / `newer-install`）；1 内部错误；2 保留给 argparse，与探测结果区分。

探测同时报告 npu-objdump / msobjdump 命令存在性和 Python 模块来源；命令与模块来自不同 CANN 时先修正当前 shell 环境。

## 快照更新（PR 期）

1. `python3 tools/npu-objdump-toolkit/scripts/gen_interface.py --source-root <asc-tools 根目录>` 重新生成 `interface.json`。
2. 新增 `add_argument` 选项必须先在 `gen_interface.py` 的 `OPTION_CLASS` 中分类，否则生成直接失败——这是刻意的守护点，强制评审者对每个新选项表态。
3. 语义变化（选项增删改、字段映射增删、跟踪文案改写）需同步修订 `SKILL.md`、references 和 eval；无关重构不改变快照（提取语义而非整文件哈希，误报率低）。
4. 快照 provenance 记录生成时的 commit，仅作溯源；HEAD 前移不算漂移，不参与 `--check` 比较。

## 合入前自查

改动 msobjdump 接口（参数、字段映射、跟踪文案）的 MR 应附带快照更新：`gen_interface.py --check` 比对源码提取与快照，不一致时给出再生成与评审指引。方向感知：本 checkout 的 msobjdump 早于快照记载的接口（skill PR 与接口 MR 分开评审的正常堆叠）时 `--check` 报 guard inactive 而非失败，两边 MR 可独立合入；接口侧合入后再执行 `--check` 即正常比对。

## 约束

- 运行期探测不写文件、不安装依赖、不修改 shell 启动文件、不提交或推送 Git。
- 持久更新只能走 PR 评审；没有"运行时接受基线"的路径，也没有本地缓存可被污染。
- 正常使用只执行与当前任务相关的探测；纯用法咨询且用户明确不执行命令时不运行。
- 更新后的资料保留来源路径、工具环境和验证边界，不依赖完整输出文本的固定排列。
