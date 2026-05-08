---
name: external-gitcode-ascend-security-code-review
description: 多语言安全代码审查 (Security Code Review)。对 Python、C++、Shell、Markdown 文件进行系统性安全漏洞检测与修复指导。覆盖
  OWASP Top 10、CWE Top 25、CERT 安全编码标准。当用户提及以下内容时，务必使用此技能：安全审查、安全代码审查、security review、code
  review 中的安全检查、漏洞扫描、安全合规检查（CWE/CERT/OWASP）、编写安全代码、检查代码安全性、推理服务安全审计、多模态 Token 安全校验、JSON
  嵌套深度攻击防护。即使用户没有明确说'安全审查'，只要涉及代码安全性评估、漏洞检测、安全最佳实践，都应触发此技能。
original-name: security-code-review
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# 多语言安全代码审查 (Security Code Review)

## 适用场景

- **安全审查**: 对 Python、C++、Shell、Markdown 文件进行安全代码审查
- **Code Review**: 在代码评审中检查安全漏洞
- **新代码编写**: 编写安全的代码，避免常见漏洞
- **合规检查**: 满足安全合规要求 (CWE, CERT, OWASP)
- **CI/CD 集成**: 在流水线中集成安全扫描工具

## 审查工作流程

执行安全审查时，按以下步骤进行：

1. **确定审查范围**：识别目标代码的语言类型（Python/C++/Shell/Markdown）
2. **加载对应参考**：根据语言读取 `references/` 下的详细审查指南
3. **逐项检查**：按照下方各语言的审查要点和检查清单执行
4. **查阅历史经验**：对于推理服务相关代码，读取 `references/lessons-learned.md` 排查已知问题模式
5. **生成审查报告**：按照本文件末尾的输出规范生成 CSV 报告

---

## Python 安全审查

> 详细代码示例和安全/不安全对比：读取 `references/python.md`

**审查覆盖的 10 个关键领域：**

| # | 领域 | 核心风险 | 关键搜索模式 |
|---|------|---------|-------------|
| 1 | 代码注入 | 任意代码执行 | `eval(`, `exec(`, `shell=True` |
| 2 | SQL 注入 | 数据泄露/篡改 | `f"SELECT`, `cursor.execute(f"` |
| 3 | 反序列化 | RCE | `pickle.loads`, `yaml.load(` (无 safe_load) |
| 4 | 路径遍历 | 任意文件读取 | `os.path.join(` + 用户输入, 无 `realpath` 校验 |
| 5 | 敏感信息泄露 | 密钥暴露 | 硬编码 `API_KEY=`, `PASSWORD=`, `logger.*(password` |
| 6 | assert 误用 | 权限绕过 | `assert user.is_admin`, `assert.*auth` |
| 7 | 临时文件 | 竞态/符号链接攻击 | `open("/tmp/` (未用 `tempfile`) |
| 8 | ReDoS | CPU 耗尽 | `(a+)+`, `(.*)*`, 嵌套量词 |
| 9 | JSON 嵌套深度 | 栈溢出/资源耗尽 | `json.loads(` 无深度限制 (CWE-674, CWE-400) |
| 10 | 特殊 Token 注入 | 进程崩溃/DoS | `np.where(np.equal(input_ids,` 无配对校验 (CWE-129, CWE-248) |

**Python 安全工具：**

| 工具 | 用途 | 命令 |
|------|------|------|
| **bandit** | 静态安全分析 | `bandit -r src/` |
| **pip-audit** | 依赖漏洞扫描 | `pip-audit` |
| **pylint** | 代码质量 + 部分安全规则 | `pylint src/` |
| **mypy** | 类型检查，防止类型混淆 | `mypy src/` |
| **semgrep** | 自定义安全规则 | `semgrep --config=p/python` |

---

## C++ 安全审查

> 详细代码示例和安全/不安全对比：读取 `references/cpp.md`

**审查覆盖的 11 个关键领域：**

| # | 领域 | 核心风险 | 关键搜索模式 |
|---|------|---------|-------------|
| 1 | 缓冲区溢出 | 代码执行 | `strcpy(`, `sprintf(`, `gets(` |
| 2 | 内存管理 | UAF/Double-free | `new`/`delete` (非智能指针), `raw pointer` |
| 3 | 整数溢出 | 缓冲区分配错误 | `new char[size]` 无范围检查 |
| 4 | 格式化字符串 | 内存读写 | `printf(user_input)`, `fprintf(stderr, var)` |
| 5 | 未初始化变量 | 未定义行为 | 声明后未赋值即使用 |
| 6 | RAII/资源泄漏 | 资源耗尽 | `fopen`/`fclose`, `mutex.lock()`/`.unlock()` |
| 7 | 线程安全 | 数据竞争 | 非 `atomic` 跨线程变量, 无锁共享数据 |
| 8 | 类型转换 | 类型混淆 | C 风格强转 `(Type*)ptr`, 尤其是向下转型 |
| 9 | JSON 嵌套深度 | 栈溢出 | `nlohmann::json::parse(` 无深度限制 (CWE-674) |
| 10 | 服务资源上限 | OOM | 配置参数组合峰值内存超物理内存 (CWE-400, CWE-770) |
| 11 | 特殊 Token 注入 | 段错误/崩溃 | Token 配对假设, 固定偏移访问 (CWE-129, CWE-248) |

**C++ 安全工具：**

| 工具 | 用途 | 命令/用法 |
|------|------|----------|
| **AddressSanitizer** | 内存错误检测 | `-fsanitize=address` |
| **ThreadSanitizer** | 数据竞争检测 | `-fsanitize=thread` |
| **UBSanitizer** | 未定义行为检测 | `-fsanitize=undefined` |
| **Valgrind** | 内存泄漏检测 | `valgrind --leak-check=full ./app` |
| **cppcheck** | 静态分析 | `cppcheck --enable=all src/` |
| **clang-tidy** | Linter + 安全规则 | `clang-tidy -checks='*' src/*.cpp` |
| **Coverity** | 企业级静态分析 | CI 集成 |

---

## Shell 安全审查

> 详细代码示例和安全/不安全对比：读取 `references/shell.md`

**审查覆盖的 8 个关键领域：**

| # | 领域 | 核心风险 | 关键搜索模式 |
|---|------|---------|-------------|
| 1 | 脚本头部 | 静默失败 | 缺少 `set -euo pipefail` |
| 2 | 变量引用 | 命令注入/误删 | `$var` (未用 `"$var"`) |
| 3 | 命令注入 | 任意命令执行 | `eval "$user_input"` |
| 4 | 临时文件 | 竞态条件 | `> /tmp/fixed_name` (未用 `mktemp`) |
| 5 | 权限 | 未授权访问 | `chmod 777`, `chmod 666` |
| 6 | PATH 安全 | 路径劫持 | 关键命令未用绝对路径 |
| 7 | 信号处理 | 资源泄漏 | 缺少 `trap cleanup EXIT` |
| 8 | 输入验证 | 路径遍历 | 未校验的 `$filename` 直接 `cat` |

**Shell 安全工具：**

| 工具 | 用途 | 命令 |
|------|------|------|
| **ShellCheck** | Shell 脚本静态分析 | `shellcheck script.sh` |
| **shfmt** | Shell 格式化 | `shfmt -d script.sh` |

---

## Markdown 安全审查

> 详细代码示例和安全/不安全对比：读取 `references/markdown.md`

**审查覆盖的 4 个关键领域：**

| # | 领域 | 核心风险 | 关键搜索模式 |
|---|------|---------|-------------|
| 1 | XSS 注入 | 脚本执行 | `<script>`, `<iframe>`, `onerror=`, `onmouseover=` |
| 2 | 链接安全 | XSS/钓鱼 | `javascript:`, `data:`, `vbscript:` |
| 3 | 敏感信息 | 密钥泄露 | `sk-`, `ghp_`, `AKIA`, 内部 IP/URL |
| 4 | 图片安全 | 追踪/DoS | 外部追踪像素, 超大图片 |

---

## 历史安全问题经验库

> 完整案例分析和排查清单：读取 `references/lessons-learned.md`

审查推理服务引擎代码时，重点排查以下已知问题模式：

| 编号 | 问题 | 严重级别 | 核心教训 |
|------|------|---------|---------|
| SEC-EXP-001 | 配置参数组合导致 OOM | CRITICAL | `峰值内存 = maxReqs × bodyLimit × JSON放大系数`，各参数独立合理但组合后超出物理内存 |
| SEC-EXP-002 | `<\|begin_of_image\|>` 无 `<\|end_of_image\|>` 导致 DoS | CRITICAL | 假设特殊 Token 成对出现，未校验即用硬索引访问 |
| SEC-EXP-003 | `<\|vision_start\|><\|video_pad\|><\|vision_end\|>` 打破格式假设 | CRITICAL | 假设 Token 序列遵循私有协议格式，用固定偏移取值导致越界 |

---

## 通用安全约束 (Constraints)

### 必须遵守 (MUST)

1. **输入验证**: 验证所有外部输入，永远不信任用户数据
2. **参数化查询**: 使用参数化查询防止注入攻击
3. **最小权限**: 进程、文件、用户使用最小必要权限
4. **安全默认值**: 变量初始化、错误处理使用安全的默认值
5. **依赖审计**: 定期扫描依赖中的已知漏洞
6. **敏感信息保护**: 密钥、密码、Token 通过环境变量管理
7. **日志脱敏**: 日志中不记录密码、密钥等敏感信息
8. **错误处理**: 错误消息不泄露内部实现细节
9. **JSON 嵌套深度校验**: 对所有外部 JSON 请求在解析前/解析时校验嵌套深度，防止栈溢出和资源耗尽攻击 (CWE-674, CWE-400)
10. **服务化请求资源上限校验**: 服务配置参数（最大并发数、请求体上限、请求头上限等）组合后的峰值内存必须小于部署环境可用内存；需考虑 JSON/XML 等反序列化库的内存放大系数 (CWE-400, CWE-770)
11. **特殊 Token 输入校验**: 多模态/多轮对话等场景中的特殊 Token（如 begin/end_of_image、vision_start/end 等）必须校验配对完整性和序列格式合法性；禁止对 Token 序列做隐式格式假设；框架层必须兜底捕获 IndexError/out_of_range 防止进程崩溃 (CWE-129, CWE-248, CWE-20)

### 禁止事项 (MUST NOT)

1. **禁止 eval 类函数**: Python `eval()`/`exec()`、Shell `eval`、C++ 无等价物但禁止动态代码生成
2. **禁止硬编码密钥**: 不在代码中硬编码密码、API Key、Token
3. **禁止提交敏感文件**: `.env`、私钥、证书不得提交到版本控制
4. **禁止忽略错误**: 不得静默吞掉异常或忽略返回值
5. **禁止过宽权限**: 不使用 `chmod 777`、`0.0.0.0` 无限制监听
6. **禁止使用已弃用的不安全函数**: `gets()`, `sprintf()`, `strcpy()` 等

---

## 安全审查检查清单 (Checklist)

### Python
```
- [ ] 无 eval()/exec() 使用不受信任的输入
- [ ] 无 pickle.loads() 加载不受信任的数据
- [ ] 使用 yaml.safe_load() 替代 yaml.load()
- [ ] subprocess 调用不使用 shell=True
- [ ] SQL 查询使用参数化方式
- [ ] 文件路径操作有路径遍历防护
- [ ] 无硬编码的密钥/密码
- [ ] 日志不记录敏感信息
- [ ] 未使用 assert 进行安全/权限相关检查
- [ ] 依赖已通过 pip-audit/bandit 扫描
- [ ] 正则表达式无 ReDoS 风险
- [ ] 临时文件使用 tempfile 模块
- [ ] JSON 请求解析入口有嵌套深度限制（建议 ≤ 32 层）
- [ ] 递归遍历 JSON 数据结构有深度保护
- [ ] 多模态特殊 Token（begin/end_of_image、vision_start/end 等）校验了配对完整性，不假设成对出现
- [ ] 推理请求处理路径有框架层 IndexError/ValueError/KeyError 兜底捕获，防止单请求异常导致进程崩溃
```

### C++
```
- [ ] 无缓冲区溢出风险 (strcpy → strncpy/std::string)
- [ ] 使用智能指针管理内存
- [ ] 无 use-after-free / double-free
- [ ] 整数运算有溢出检查
- [ ] printf 系列函数使用固定格式字符串
- [ ] 所有变量在使用前初始化
- [ ] 资源管理遵循 RAII 原则
- [ ] 多线程代码无数据竞争
- [ ] 使用 C++ 风格类型转换 (static_cast/dynamic_cast)
- [ ] 编译启用安全选项 (-Wall -Werror -fsanitize=address)
- [ ] JSON 请求解析入口有嵌套深度限制（建议 ≤ 32 层）
- [ ] 递归遍历 JSON 数据结构有深度参数并设上限
- [ ] 服务配置参数组合后峰值内存不超过部署环境可用内存（考虑 JSON 放大系数）
- [ ] 存在运行时内存水位监控或请求准入控制机制
- [ ] 多模态特殊 Token（boi/eoi、vision_start/end 等）校验了配对完整性和序列格式，不做隐式格式假设
- [ ] model forward / pre-processing 路径有框架层 std::out_of_range/std::invalid_argument 兜底 catch，防止进程崩溃
```

### Shell
```
- [ ] 脚本使用 set -euo pipefail
- [ ] 所有变量使用双引号包裹 ("$var")
- [ ] 无 eval 使用用户输入
- [ ] 临时文件使用 mktemp
- [ ] 文件权限不超过 755（脚本）/ 644（配置）
- [ ] 关键命令使用绝对路径
- [ ] 有 trap 清理机制
- [ ] 输入经过验证和过滤
- [ ] 通过 ShellCheck 无警告
- [ ] 不使用 . 或空目录在 PATH 中
```

### Markdown
```
- [ ] 无内嵌 <script>/<iframe> 标签
- [ ] 无 javascript:/data: 协议链接
- [ ] 无硬编码密钥/密码/Token
- [ ] 无内部 IP 地址或内部 URL 泄露
- [ ] 外部图片来源可信
- [ ] 无追踪像素
```

---

## 推荐工具汇总

| 语言 | 工具 | 类型 | 说明 |
|------|------|------|------|
| Python | **bandit** | 静态分析 | Python 安全漏洞检测 |
| Python | **pip-audit** | 依赖扫描 | Python 依赖漏洞检查 |
| Python | **semgrep** | 规则引擎 | 自定义安全规则匹配 |
| Python | **mypy** | 类型检查 | 类型安全，防止类型混淆 |
| C++ | **AddressSanitizer** | 运行时检测 | 内存错误检测 |
| C++ | **ThreadSanitizer** | 运行时检测 | 数据竞争检测 |
| C++ | **cppcheck** | 静态分析 | C/C++ 静态分析 |
| C++ | **clang-tidy** | Linter | 代码质量 + 安全规则 |
| C++ | **Valgrind** | 运行时检测 | 内存泄漏检测 |
| Shell | **ShellCheck** | 静态分析 | Shell 脚本安全分析 |
| 通用 | **git-secrets** | 预提交钩子 | 防止提交密钥 |
| 通用 | **trufflehog** | 密钥扫描 | 扫描代码中的密钥 |
| 通用 | **gitleaks** | 密钥扫描 | Git 仓库密钥泄露检测 |
| Markdown | **markdownlint** | Linter | Markdown 格式检查 |

---

## 参考标准

- [CWE Top 25](https://cwe.mitre.org/top25/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CERT C++ Secure Coding](https://wiki.sei.cmu.edu/confluence/display/cplusplus)
- [CERT C Secure Coding](https://wiki.sei.cmu.edu/confluence/display/c)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [ShellCheck Wiki](https://www.shellcheck.net/wiki/)

---

## 审查结果输出 (Output)

### 审查流程要求

当使用此 Skill 对代码仓库执行安全审查（即参数包含 `review` 关键字）时，完成所有审查后自动将结果保存为 CSV 文件。这是审查流程的最终必要步骤。

### CSV 输出规范

**文件名：** `security_code_review_report.csv`
**保存位置：** 被审查的代码仓库根目录下（即 `review <path>` 中的 `<path>` 下）
**编码：** UTF-8

**CSV 必须包含以下 9 列（表头固定）：**

```csv
编号,严重级别,漏洞类别,语言,文件路径,行号,问题描述,风险说明,建议修复方案
```

| 列名 | 说明 | 示例值 |
|------|------|--------|
| **编号** | 唯一编号，格式为 `严重级别首字母-序号` | `C-01`, `H-05`, `M-12`, `L-03` |
| **严重级别** | 四级：`CRITICAL` / `HIGH` / `MEDIUM` / `LOW` | `CRITICAL` |
| **漏洞类别** | 安全漏洞分类名称 | `命令注入`, `不安全反序列化`, `线程安全`, `路径遍历` |
| **语言** | 代码语言/文件类型 | `Python`, `C++`, `Shell`, `Markdown`, `Docker`, `Config` |
| **文件路径** | 相对于仓库根目录的文件路径 | `src/utils/file_utils.py` |
| **行号** | 问题代码所在行号，多行用逗号分隔 | `79`, `40-44, 220-232` |
| **问题描述** | 简明扼要描述发现的问题 | `pickle.loads() 反序列化来自共享内存的数据` |
| **风险说明** | 说明该问题可能导致的安全风险 | `攻击者可注入恶意 pickle payload 实现 RCE` |
| **建议修复方案** | 具体的修复建议和代码示例 | `用 json.loads() 替代 pickle.loads()` |

### CSV 格式要求

1. **逗号分隔**，含逗号的字段值用双引号包裹
2. 字段值内部的双引号用两个双引号转义（`""`）
3. 第一行为表头行，之后每行一条发现
4. 按严重级别排序：CRITICAL → HIGH → MEDIUM → LOW
5. 同级别内按编号顺序排列
6. 文件路径使用相对路径（相对于仓库根目录）

### 输出流程

审查完成后执行以下步骤：

1. **汇总所有发现**：收集所有审查代理/扫描的结果
2. **去重合并**：合并重复发现，确保每条记录唯一
3. **生成 CSV**：按上述规范生成 CSV 文件并写入目标路径
4. **验证 CSV**：用 Python csv 模块验证文件格式正确、行数与发现数一致
5. **输出摘要**：向用户报告文件位置和各严重级别的统计数量

### 示例输出

```csv
编号,严重级别,漏洞类别,语言,文件路径,行号,问题描述,风险说明,建议修复方案
C-01,CRITICAL,不安全反序列化,Python,src/utils/share_memory.py,79,"pickle.loads() 反序列化来自共享内存的数据","攻击者可注入恶意 pickle payload 实现 RCE","用 json.loads() 替代 pickle.loads()"
H-01,HIGH,命令注入,Shell,scripts/run.sh,52,"eval 执行含用户输入的命令字符串","模型路径含 shell 元字符时可注入任意命令","改用 bash 数组构建命令, 消除 eval"
M-01,MEDIUM,线程安全,C++,src/thread_pool.h,45,"m_shutdown 为非原子 bool 跨线程读写","数据竞争导致工作线程可能无法退出","改为 std::atomic<bool>"
L-01,LOW,临时文件安全,Python,tests/test_utils.py,30,"硬编码 /tmp 路径","共享 CI 环境中符号链接攻击风险","使用 tempfile.mkdtemp()"
```

### 注意事项

- 即使审查未发现任何问题，也应生成 CSV 文件（仅含表头行），并向用户说明"未发现安全问题"
- CSV 文件用于人工审核，**描述必须清晰具体，避免模糊表述**
- 如果审查仅针对特定文件而非整个仓库，CSV 保存到该文件所在目录
- 生成 CSV 后用 Python 脚本验证格式，确保可被 Excel/WPS 正确打开

---

## Metadata

- **Version**: 2.0.0
- **Last updated**: 2026-04-24
- **Languages**: Python, C++, Shell/Bash, Markdown
- **Tags**: `#security` `#code-review` `#python` `#cpp` `#shell` `#markdown` `#OWASP` `#CWE`
