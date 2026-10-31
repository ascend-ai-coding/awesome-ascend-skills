# Step 1: Project Setup

> **定位**：建立当前算子工程、根级 Blaze 源码版本、只读源码区和版本一致性记录。Step 1 保留 clone 或本地预置源复制、递归 submodule 初始化和版本确认；不调查 Blaze 组装方案、不选择路线、不创建场景实现。

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
| L1 | `<project-root>/ops-tensor/` 已存在 | 不重新获取，直接进入 2.2 |
| L2 | `<project-root>/.cannbot/ops-tensor` 存在且完整，或调用方传入 `local_blaze_source` 指向完整 ops-tensor（绝对路径） | 解引用复制到 `<project-root>/ops-tensor/`，再核对 submodule；把来源路径和 HEAD 写入抽象版本记录 |
| L3 | 网络可达 | 从授权仓库递归 clone |
| L4 | 以上均不可用 | 记录 `source_unavailable`，停止 Step 1，不得进入 Blaze 能力判定 |

L2 的本地来源：

```bash
mkdir -p <project-root>/ops-tensor
cp -aL <local-source>/. <project-root>/ops-tensor/
test ! -L <project-root>/ops-tensor
git -C <project-root>/ops-tensor submodule sync --recursive
git -C <project-root>/ops-tensor submodule update --init --recursive
```

`cp -aL` 必须解引用软链：`<local-source>` 可能是安装期建出的符号链接，`cp -a` 只复制链接本身，会让事实根仍指向安装目录并随其更新静默漂移；复制后必须确认目标是实体目录。

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

已有 `<project-root>/ops-tensor/` 时不重复 clone。只在当前版本合同授权下更新，并检查 remote、branch、worktree、父仓 gitlink 和递归 submodule：

```bash
git -C <project-root>/ops-tensor remote -v
git -C <project-root>/ops-tensor status --short --ignore-submodules=none
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

本节 `fetch`、`checkout` 和 `pull --ff-only` 只适用于来源是授权网络仓的 checkout（L1 的原始 clone 或 L3）。经 §2.1 L2 从本地预置源复制来的 checkout 不执行任何联网步骤：只做 `submodule`、父仓 gitlink 和 worktree 一致性核对，并沿用来源 checkout 记录的 remote 与 HEAD；本地存在未提交修改时仍按 dirty 停止。

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
