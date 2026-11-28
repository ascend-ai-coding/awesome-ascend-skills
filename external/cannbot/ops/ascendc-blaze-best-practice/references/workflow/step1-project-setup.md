# Step 1: Project Setup

> **定位**：建立当前算子工程、根级 Blaze 源码版本、只读源码区和版本一致性记录。Step 1 按授权执行网络获取或更新，或只读校验并复制完整本地预置源；不调查 Blaze 组装方案、不选择路线、不创建场景实现。

## 1. 路径合同和目录结构

`<project-root>` 是最高层项目根，`<operator_name>` 是 `operators/` 下的当前算子目录。`ops-tensor/` 与 `operators/` 必须同级：

```text
project_root: <project-root>
operators_root: <project-root>/operators/
operator_name: <operator_name>
operator_root: <project-root>/operators/<operator_name>/
blaze_source_root: <project-root>/ops-tensor/
```

完整目录结构：

```text
<project-root>/
├── ops-tensor/                         # 共享 Blaze 上游源码，只读
└── operators/
    └── <operator_name>/
        ├── docs/
        │   ├── blaze/
        │   │   └── blaze-investigation-report.md # Step 2 输出
        │   ├── DESIGN.md               # Step 3 输出
        │   └── PLAN.md                 # Step 3 输出
        ├── op_kernel/
        │   ├── include/
        │   │   ├── blaze/              # 同源官方副本，只读
        │   │   ├── tensor_api/         # 同源官方副本，只读
        │   │   └── blaze_custom/       # 仅 blaze_custom 按 PLAN 使用
        │   │       ├── block/
        │   │       ├── kernel/
        │   │       ├── epilogue/
        │   │       ├── policy/
        │   │       └── utils/
        │   └── <Kernel/Wrapper 文件>   # 由 PLAN 冻结
        ├── op_tiling/
        │   └── <TilingData/Tiling 文件>
        ├── <Launcher 文件>
        ├── scripts/
        ├── data/
        │   ├── input/
        │   ├── golden/
        │   └── output/
        ├── CMakeLists.txt
        └── <其他构建文件>
```

创建当前算子根和最小空目录。不要预先生成固定 Kernel、Wrapper、Launcher、脚本或测试文件；DESIGN/PLAN 只由 Step 3 生成。

```bash
mkdir -p <project-root>/operators/<operator_name>/docs/blaze
mkdir -p <project-root>/operators/<operator_name>/op_kernel/include
mkdir -p <project-root>/operators/<operator_name>/op_tiling
```

## 2. Blaze 源码版本门禁

### 2.1 源码获取

按以下顺序判定来源，命中即停止。无论哪一级，最终事实根只能是 `<project-root>/ops-tensor/`，并保持只读。

| 级别 | 条件 | 动作 |
|---|---|---|
| L1 | `<project-root>/ops-tensor` 目录项已存在（含符号链接） | 先只读校验真实路径、Git 根与来源，再进入 2.2；断链、不可读或校验失败时停止，不降级换源 |
| L2 | 按下文顺序唯一选定完整本地源 | 先只读校验授权来源与完整性，再解引用复制到 `<project-root>/ops-tensor/` 并只读复核；将来源与一致性状态写入现有抽象版本记录 |
| L3 | 网络可达 | 从授权仓库递归 clone |
| L4 | 以上均不可用 | 记录 `source_unavailable`，停止 Step 1，不得进入 Blaze 能力判定 |

当前 install-helper 的 project 安装在 `<project-root>/ops-tensor` 建立项目链接，属于 L1。`<project-root>/.cannbot/ops-tensor` 仅兼容旧工作区布局，不作为当前安装器的输出约定。

**L2 选源顺序**：

1. 调用方明确指定的 `local_blaze_source`（绝对路径）。已指定但路径不存在、无法解析或校验不合格时停止，不静默换源。
2. 未指定 `local_blaze_source` 时，检查 `<project-root>/.cannbot/ops-tensor`。该目录项不存在才进入 L3；存在但断链、不可读或校验不合格时停止。

高优先级来源已选定时，不扫描低优先级目录来否定它。自动查找仅限当前项目内上述位置；显式 `local_blaze_source` 只访问调用方已授权的来源，不扩展为其它目录或全局搜索。复制目标仍只能是当前项目的 `blaze_source_root`。

**选定后先只读校验**：

- 确认来源是可读取的 Git 工作区；选定目录和 `git rev-parse --show-toplevel` 的输出都先解析为真实路径，再比较两者相等，不能直接比较符号链接路径与 Git 返回的路径字符串。L1 的项目链接同样适用；断链或任一解析失败时停止。按授权仓库身份核对 remote。相同仓库的 HTTPS、SSH 地址及末尾 `.git` 写法可以不同，例如 `https://gitcode.com/cann/ops-tensor.git` 与 `git@gitcode.com:cann/ops-tensor.git`。不能只比较原始 URL 字符串，也不任意折叠仓库路径大小写。
- 明确授权的镜像或本地预置来源按已有授权记录核对；不自动信任任意 fork，也不把 remote 名为 `upstream` 当作授权证明。缺少 remote 或无法确认授权时，按现有合同停止，不联网补证据，不增加外部 manifest 依赖。
- 核对当前布局中必要的 Blaze、tensor_api 源码目录非空；工作区、分支和版本状态满足 §2.2。所有递归 submodule 必须已初始化、与父仓 gitlink 一致且无冲突；不能用初始化、同步、更新或联网下载来补齐本地来源。

以下为 Linux shell 的只读校验示例；实际源码目录按 §3 检查，不能只凭目录名或一个命令退出成功就认定完整。其他平台采用等价的真实路径解析，不直接比较不同路径表示：

```bash
source_path='<selected-source>'  # L1 项目路径或 L2 选定来源
source_real=$(realpath -e -- "$source_path") || exit 1
test -d "$source_real" && test -r "$source_real" && test -x "$source_real" || exit 1
git_root=$(git -C "$source_real" rev-parse --show-toplevel) || exit 1
git_root_real=$(realpath -e -- "$git_root") || exit 1
test "$source_real" = "$git_root_real" || exit 1

git -C "$source_real" remote -v
git --no-optional-locks -C "$source_real" status --short --ignore-submodules=none
git -C "$source_real" submodule status --recursive
git -C "$source_real" ls-tree HEAD include/tensor_api
```

任一选源歧义或校验失败都是 Step 1 setup 阻塞，不得写成 Blaze 不支持该需求。全部通过后才复制：

```bash
mkdir -p <project-root>/ops-tensor
cp -aL <local-source>/. <project-root>/ops-tensor/
test ! -L <project-root>/ops-tensor
git --no-optional-locks -C <project-root>/ops-tensor status --short --ignore-submodules=none
git -C <project-root>/ops-tensor submodule status --recursive
```

`cp -aL` 必须解引用软链：`<local-source>` 可能是安装期建出的符号链接，`cp -a` 只复制链接本身，会让事实根仍指向安装目录并随其更新静默漂移；复制后必须确认目标是实体目录，并按相同只读检查复核 Git 工作区、源码目录与 submodule。缺失或不一致时停止，不修改来源或联网补齐。

L3 的授权仓库来源：

```bash
git clone --recurse-submodules \
  https://gitcode.com/cann/ops-tensor.git \
  <project-root>/ops-tensor
git -C <project-root>/ops-tensor submodule sync --recursive
git -C <project-root>/ops-tensor submodule update --init --recursive
```

获取目标只能是 `<project-root>/ops-tensor/`。不得拉取到 `operators/`、当前算子目录、其他项目或其他临时位置，也不得直接把 `<local-source>` 当作事实根。`source_unavailable` 只表示本次获取失败，不表示 Blaze 不支持该需求。

### 2.2 已存在源码

已有 `<project-root>/ops-tensor` 时先执行 §2.1 的只读路径与来源校验，不重复 clone。符号链接存在但目标缺失也属于已有目录项，必须阻塞，不能当作路径不存在。安装器链接指向的来源默认只读；更新授权必须覆盖该真实来源，不能因项目路径可写就更新共享 checkout。只在当前版本合同授权下更新，并检查 remote、branch、worktree、父仓 gitlink 和递归 submodule：

```bash
git -C <project-root>/ops-tensor remote -v
git --no-optional-locks -C <project-root>/ops-tensor status --short --ignore-submodules=none
git -C <project-root>/ops-tensor fetch origin master
git -C <project-root>/ops-tensor checkout master
git -C <project-root>/ops-tensor pull --ff-only origin master
git -C <project-root>/ops-tensor submodule sync --recursive
git -C <project-root>/ops-tensor submodule update --init --recursive
git -C <project-root>/ops-tensor submodule status --recursive
git -C <project-root>/ops-tensor rev-parse HEAD
git -C <project-root>/ops-tensor ls-tree HEAD include/tensor_api
```

命令输出仅用于建立抽象一致性状态。不要把提交 ID、manifest/hash、文件哈希或时间戳写入 Blaze Skill 合同。dirty、detached、ahead/diverged、remote 不匹配、父仓 gitlink 不一致、未初始化或不匹配的递归 submodule 都必须停止。

本节 `fetch`、`checkout`、`pull --ff-only`、`submodule sync` 和 `submodule update` 只适用于版本合同已授权的网络仓更新（L1 的原始 clone 或 L3）。经 §2.1 L2 从本地预置源复制来的 checkout 只做只读的 submodule、父仓 gitlink 和 worktree 一致性核对，不执行同步、初始化、更新或任何联网步骤，并沿用来源 checkout 的已核实授权身份与版本。remote 身份按 §2.1 核对；本地存在未提交修改时仍按 dirty 停止。

## 3. 同源只读区

从同一 `blaze_source_root` 复制或绑定编译所需的官方目录：

```bash
cp -r <project-root>/ops-tensor/include/blaze \
  <project-root>/operators/<operator_name>/op_kernel/include/
cp -r <project-root>/ops-tensor/include/tensor_api \
  <project-root>/operators/<operator_name>/op_kernel/include/
```

实际源目录以当前 checkout 为准；布局不一致时停止，不猜测替代路径。以下三个区域始终只读：

- `<project-root>/ops-tensor/`；
- `<project-root>/operators/<operator_name>/op_kernel/include/blaze/`；
- `<project-root>/operators/<operator_name>/op_kernel/include/tensor_api/`。

Skill Asset 原文件同样只读。后续编译 include path 显式指向项目副本，不通过修改只读源码绕过版本问题。

每次复制都必须显式区分 source 和 destination，并在命令返回后核对目标文件树；
source 与 destination 相同、目标目录落入只读区或复制返回非零时，首个失败边界是
Step 1 setup，必须先修复并重新核对，不能继续生成代码或把工具错误归因于编译/设备。

## 4. Custom 隔离区

Step 1 可以建立空目录：

```bash
mkdir -p <project-root>/operators/<operator_name>/op_kernel/include/blaze_custom/{block,kernel,epilogue,policy,utils}
```

该目录不表示已经选择 `blaze_custom`。只有同时满足以下条件，Step 4 才能复制并适配项目副本：

1. Step 3 冻结 `implementation_route=blaze_custom`；
2. 唯一场景合同授权对应层级；
3. PLAN 明确列出复制、修改和验证动作。

`blaze_native` 和 `unsupported` 不预复制 custom 文件。纯 MatMul 的官方能力缺口不能通过创建 custom 文件自动 fallback。Custom 符号使用独立 namespace，不依靠 include 顺序覆盖官方偏特化。

## 5. 最小工程边界

Step 1 只建立工程路径和明确需要的空目录，不实现公式、Tiling、Golden 或测试。具体文件名、数据格式、构建目标和脚本数量由 Step 3 PLAN 决定。

不要把固定 Basic/Grouped/MX 工程、固定 recipe、旧 Launcher 或静态组合表复制为默认工程。`assets/op_tiling/` 只有在 Step 3 逐字段证明兼容并写入 DESIGN/PLAN 后，才能由 Step 4 复制到项目内适配；Asset 原文件保持零 diff。

## 6. Handoff

Step 1 交接以下抽象字段：

```text
project_root
operators_root
operator_name
operator_root
blaze_source_root
source_version_status: current_checkout | same_investigation_source | consistent | source_unavailable
source_origin: network_clone | local_staged | unavailable
checkout_consistency: confirmed | blocking
submodule_status
read_only_source_regions
upstream_read_roots
```

`upstream_read_roots` 只能包含同一 `blaze_source_root`，不能引入第二源码根。Step 2/3/4 必须绑定同一次一致的 Blaze 源码版本；后续发现变化时返回 Step 1。

## 7. 完成门禁

- `project_root`、`operators_root`、`operator_root` 和 `blaze_source_root` 已按合同派生；
- `ops-tensor/` 与 `operators/` 同级；
- ops-tensor 来自授权仓库，remote/branch/worktree 状态可核验；
- 所有递归 submodule 已初始化且与父仓 gitlink 一致；
- 三个官方源码区和全部 Skill Asset 原文件只读；
- custom 空目录没有被当作路线选择或能力证据；
- 未预复制场景实现、固定工程或旧 recipe；
- 版本记录使用抽象字段，不包含实际提交节点。

完成后进入 [Step 2: Blaze Investigation](step2-blaze-investigation.md)。
