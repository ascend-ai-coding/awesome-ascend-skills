# 范式开发指导

本文档指导如何为 `ascendc-regbase-best-practice` skill 新增一个算子范式（paradigm）。

## 目录结构

```
references/paradigms/
├── routes.yaml                    # 路由配置（所有范式共享）
├── general-methodology.md         # 通用方法论（未匹配范式时的 fallback）
├── common/                        # 公共知识库（与范式同级，跨范式通用）
├── <paradigm>/                    # 范式目录（如 broadcast/、reduction/）
│   ├── patterns.md                # [必需] 入口索引，场景路由 + 设计指南
│   ├── manifest.yaml              # [可选] 程序化校验契约
│   ├── validate_gates.py          # [可选] 自定义校验脚本
│   ├── references/                # [可选] 范式专属参考材料
│   │   ├── {paradigm}-template-overview.md  # 模板划分总览
│   │   ├── {paradigm}-{scene1}-template/    # 模板一
│   │   ├── {paradigm}-{scene2}-template/    # 模板二
│   │   └── ...
│   ├── example/                   # [可选] 参考实现
│   │   ├── examples-*-design/     # 参考设计文档
│   │   └── examples-*-code/       # 参考源码
│   └── asset/                     # [可选] 模板和工具
│       ├── templates/             # 代码模板（.templ）
│       └── requirements/          # 分阶段需求文档
```

## 基础范式 references 组织规范

**模板清单**：

| 文件 | 职责 |
|------|------|
| `{paradigm}-template-overview.md` | 模板 / TilingKey / TilingData 划分及对应关系 |
| `{paradigm}-tiling-preprocess.md` | 平台/算子信息获取、值依赖、shape 归一化 |
| `{paradigm}-kernel-entrance.md` | 入口注册、模板分发、TilingData 获取 |
| `{paradigm}-{scene}-dag-buffers.md` | 计算流、存活节点、VF 融合、buffer 划分 |
| `{paradigm}-{scene}-tiling.md` | UB 切分、多核切分、切分后处理 |
| `{paradigm}-{scene}-kernel.md` | 骨架、偏移、CopyIn、Compute、CopyOut、同步 |

**适用范围**：仅独立基础范式；`numerical/`、`stateful/` 等必须与其他范式结合的范式不适用本规则。

## 新增范式步骤

### 1. 创建范式目录

```bash
mkdir -p references/paradigms/<paradigm>
```

目录名使用小写，与 `spec.yaml` 中 `op.paradigms` 的值对应（如 `Elementwise` → `elewise`）。

### 2. 编写 patterns.md（必需）

`patterns.md` 是范式的入口索引，负责：
- **场景判定**：根据算子特征路由到具体设计策略
- **设计指南**：提供 Read→Write→Check 分步工作流

最小结构：

```markdown
# <Paradigm> 类算子场景路由

> 本文档用于**场景判定**和**策略选择**。确定场景后，按链接进入对应详细文档。

---

## 场景判定流程

\```
给定: <输入特征>

Step 1 — <判定条件>:
  ├─ YES → <策略 A> → [strategy-a.md]
  └─ NO  → <策略 B> → [strategy-b.md]
\```

---

## 通用规则

- <规则 1>
- <规则 2>

---

## 跨场景参考

| 主题 | 文档 |
|------|------|
| <主题> | [<文档名>](<路径>) |
```

### 3. 注册路由（必需）

编辑 `references/paradigms/routes.yaml`，在 `paradigms` 下添加路由：

```yaml
routes:
  paradigms:
    <Paradigm>:
      - references/paradigms/<paradigm>/patterns.md
```

`paradigms` 按 `op.paradigms` 列表匹配，结果自动去重。

### 4. 添加校验契约（可选）

`manifest.yaml` 定义 DESIGN.md 必须包含的章节和内容规则，被 `validate_completeness.py` 程序化校验：

```yaml
paradigm: <Paradigm>
version: 1

section_3_1:
  heading: "TilingKey 模板化"
  required: true

section_3_2:
  heading: "TilingData 字段定义"
  required: true
  required_patterns:
    - pattern: "struct\\s+\\w+TilingData"
      description: "缺少 TilingData 结构体定义"
```

### 5. 添加参考实现（可选）

`example/` 目录提供参考设计和源码，供 agent 学习范式特征：

```
example/
├── examples-<op>-design/
│   └── <op>_design.md          # 参考设计文档
└── examples-<op>-code/
    ├── op_host/                # Host 侧 Tiling 代码
    └── op_kernel/              # Kernel 侧计算代码
```

### 6. 添加代码模板（可选）

`asset/templates/` 存放代码模板，使用占位符供 agent 填充：

```
asset/templates/
├── apt.cpp.templ              # Kernel 入口模板
├── kernel.h.templ             # Kernel 类声明模板
├── tiling_data.h.templ        # TilingData 结构体模板
└── op_struct.h.templ          # TilingKey 结构体模板
```

模板使用 `{PLACEHOLDER}` 占位符，agent 在生成代码时替换为实际值。

## 验证

新增范式后，运行以下命令验证：

```bash
cd plugins-official/ops-registry-invoke/skills/spec-to-design

# 1. 验证路由解析
python3 -c "
import sys
sys.path.insert(0, 'scripts')
from pathlib import Path
from slice_design_inputs import load_paradigm_refs, resolve_paradigm_refs

skill_dir = Path('.').resolve()
refs_config = load_paradigm_refs(skill_dir)
cat, para, matched = resolve_paradigm_refs(refs_config, 'test', '<Paradigm>', ['<Paradigm>'], skill_dir)
print(f'Resolved: {cat}')
"

# 2. 运行测试
python3 -m pytest test/test_spec_to_design.py -q
```

## 参考实现

- **完整范式**：`broadcast/`（含 manifest.yaml、validate_gates.py、example/、asset/）
- **轻量范式**：`elewise/`（仅 patterns.md + tiling.md）
- **通用方法论**：`general-methodology.md`（未匹配范式时的 fallback）

## 注意事项

1. **目录名小写**：`references/paradigms/<paradigm>/` 使用小写，与 PascalCase 的 paradigm 名对应
2. **patterns.md 是入口**：其他文件通过 patterns.md 按需引用，不要直接暴露给 agent
3. **路由配置集中**：所有路由在 `routes.yaml` 中管理，不要分散到各范式目录
4. **校验契约可选**：`manifest.yaml` 用于程序化校验，初期可省略，待范式稳定后补充
