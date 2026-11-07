# 格式检查与汇总提交

本步骤由一个独立通用子 Agent执行。它只检查 `file_input` 明确包含的对象，将完整结果整理为一份汇总 `type: format` YAML，并通过 collector 提交；不得接触 `yaml_dir` 或修改待检查文件。

## 输入

- 检查对象：`{file_input}`
- collector端口：`{collector_port}`
- skill 资源目录：`{skill_base}`（主流程传入的绝对路径，仅用于调用自带脚本）

主 Agent将上述输入传给格式检查子 Agent，并要求其完整执行本文件。子 Agent完成后只返回提交状态和问题数量，不在文本回复中重复输出问题详情或diff。

## 执行流程

### 1. 固定检查范围并规范化路径

只处理 `file_input` 明确给出的文件；调用方已经展开目录或PR变更文件，不得自行扫描仓库或扩大范围。

从 `git rev-parse --show-toplevel` 获取仓库根目录。仓库内文件在汇总YAML中统一使用相对仓库根目录的路径，不得写入workspace绝对路径。用户粘贴内容需要运行工具时，在仓库外创建临时文件，并在结果中使用 `<pasted-python>` 或 `<pasted-cpp>` 等逻辑名称；结束后清理临时文件。

按扩展名分类：

| 类型 | 扩展名 | 处理方式 |
|---|---|---|
| Python | `.py`、`.pyi` | Ruff lint和format检查 |
| C/C++ | `.c`、`.cc`、`.cpp`、`.cxx`、`.h`、`.hpp`、`.hh`、`.icc` | clang-format检查 |
| Markdown | `.md` | 不运行格式工具，记录到 `skipped_files`，由 `references/doc-style.md` 检视 |
| 其他 | 其他扩展名 | 记录到 `skipped_files`并说明不适用 |

不存在、不是普通文件或无法读取的输入也写入 `skipped_files`，不得记为已检查。

### 2. 检测并按需安装工具

仅在存在对应类型文件时检查工具：

```bash
ruff --version
clang-format --version
```

工具缺失时允许自动安装：

```bash
# Ruff
python3 -m pip install "ruff==0.16.8"

# clang-format：根据当前环境选择一种可用方式
sudo apt-get install clang-format-18
# 或
brew install clang-format@18
# 或
python3 -m pip install clang-format==18.1.8
```

安装后必须重新解析可执行文件并再次检查版本。安装需要的命令、网络、权限或包管理器不可用时，不得反复尝试同一路径；把工具名、失败命令和实际错误写入 `execution_errors`，继续执行其他可用检查。不得把工具缺失或执行失败记为通过。

`tools` 必须记录本次涉及工具的版本和状态；没有对应输入的工具可记为 `not_applicable`。

### 3. 检查Python文件

从仓库根目录显式传入筛选后的Python文件：

```bash
bash "{skill_base}/scripts/check-python.sh" path/to/a.py path/to/b.pyi
```

解析脚本返回的JSON：

- `issues` 转为 `python.lint_issues`；
- `format_issues` 中的文件转为 `python.format_issues`；
- `checked_files` 中的路径写入 `python.files_checked`，不得用原始输入数或仅成功的 lint/format 单项数代替；
- `execution_errors` 合入汇总YAML的同名字段，`skipped_files` 合入汇总YAML的同名字段。某文件只有 lint 或 format 单项成功时，该文件不计入 `files_checked`，但保留已成功单项的问题；根据 `lint_files_checked` 和 `format_files_checked` 区分“部分成功”与“全部失败”，分别设置 `PARTIAL` 或 `ERROR`。

每个lint问题统一包含可获得的以下字段，不得虚构缺失信息：

```yaml
file_path: path/to/file.py
line: 12
column: 5
source: 原始问题行
code: F401
message: imported but unused
fix_suggestion: Ruff提供的修复信息或空
```

根据Ruff返回的位置读取对应源码行补充 `source`。需要格式化的文件执行：

```bash
ruff format --diff path/to/file.py
```

把完整diff和文件路径写入：

```yaml
file_path: path/to/file.py
message: 需要Ruff格式化
diff: 完整diff文本
```

不得运行 `ruff check --fix` 或会修改文件的 `ruff format <file>`。

### 4. 检查C/C++文件

从仓库根目录显式传入筛选后的C/C++文件：

```bash
bash "{skill_base}/scripts/check-cpp.sh" path/to/a.cc path/to/a.h
```

将脚本 `checked_files` 中实际成功检查的文件写入 `cpp.files_checked`，并合入脚本的 `execution_errors` 和 `skipped_files`。对脚本返回的每个问题文件执行：

```bash
clang-format --style=file path/to/file.cc | diff -u path/to/file.cc -
```

将文件路径、`需要clang-format格式化` 和完整diff写入 `cpp.format_issues`。不得使用 `clang-format -i`。

### 5. 确定汇总状态

按以下优先级设置顶层 `status`：

1. `ERROR`：存在适用的Python或C/C++文件，但所有适用格式检查都因工具或执行错误而未完成。
2. `PARTIAL`：只完成部分适用检查，存在未检查输入，或者没有可由Ruff/clang-format检查的文件；纯Markdown输入属于此状态。
3. `FAIL`：所有适用输入均完成检查，没有跳过或执行错误，但发现至少一个lint或格式问题。
4. `PASS`：所有适用输入均完成检查，没有跳过、执行错误、lint问题或格式问题。

已完成部分中发现的问题必须保留，即使最终状态是 `PARTIAL`。

Ruff因发现lint或格式差异而返回非零退出码，以及 `diff` 因存在差异而返回退出码1，均属于正常“发现问题”，不能写入 `execution_errors`。只有工具无法启动、输出无法解析或命令异常中断时才视为执行错误。

### 6. 生成并提交一份汇总YAML

使用安全的YAML序列化方式生成以下结构，保留完整问题和diff：

```yaml
type: format
check_id: file-format
status: PASS
tools:
  ruff:
    version: 0.12.0
    status: available
  clang-format:
    version: 18.1.8
    status: available
python:
  files_checked: []
  lint_issues: []
  format_issues: []
cpp:
  files_checked: []
  format_issues: []
markdown:
  files_checked: []
skipped_files: []
execution_errors: []
```

`markdown.files_checked` 保持为空；Markdown文件放入 `skipped_files`，理由注明由 `references/doc-style.md` 检视。

通过以下端点提交一次：

```text
http://127.0.0.1:{collector_port}/submit?type=format
```

使用 `curl -sS --noproxy 127.0.0.1,localhost --fail-with-body -X POST --data-binary` 提交完整YAML，并检查退出状态及collector响应。访问本地collector必须显式绕过环境中的HTTP或SOCKS代理，但不得清除影响外部工具下载的全局代理设置。HTTP 400表示YAML结构不符合schema，应根据实际错误修正后重新提交；HTTP 200后不得主动重交。网络状态不明确导致重复提交时，collector生成 `format_dupN.yaml`，所有重复文件都会进入报告和统计。

collector不限制请求体大小，format YAML可以保留完整结果；最终报告由assembler限制Ruff示例数和diff展示行数。

## 返回

成功后只返回：

```text
格式检查完成：已提交1份format YAML；状态{STATUS}；Ruff {N}项；Python格式 {P}个文件；C/C++格式 {C}个文件。
```

提交失败时返回实际HTTP或工具错误，不得声称已经完成，也不得在文本回复中粘贴完整YAML或diff。

## 约束

- 检查期间严禁修改项目文件，也不得调用任何 `fix-*.sh`。
- 不把工具缺失、命令失败、文件不存在、零文件命中或纯Markdown输入记为 `PASS`。
- 不读取或写入 `yaml_dir`；所有结果只通过collector提交。
- 本步骤不生成最终报告，不询问是否修复；修复只能在全部检视和报告完成后由用户明确授权。
