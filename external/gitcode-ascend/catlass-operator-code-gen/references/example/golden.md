# Golden 标杆：从本机 CANN 的 aclnn 接口选取或拼接

本文是给 **Agent** 用的简短工作流：在编写自定义算子前，先在本机 CANN 安装树中 **定位 aclnn 头文件与既有实现线索**，再为待实现算子选定 **Golden（数值真值来源）**——可直接复用单个 aclnn 算子，或用多个 aclnn/基础算子 **拼接** 出参考行为。

---

## 1. 约定路径

- **CANN 根目录**（可按用户环境替换）：`CANN_ROOT=/home/yuantao/Ascend/cann`
- **aclnn 对外头文件**（常见布局，以实际目录为准）：
  - `$(CANN_ROOT)/aarch64-linux/include/aclnnop/` — 聚合头与各类 `aclnn_*.h`
  - `$(CANN_ROOT)/aarch64-linux/include/aclnnop/level2/` — 大量 Level2 融合/专用算子声明
- **自定义/示例 op_api**（若有）：`$(CANN_ROOT)/opp/vendors/**/op_api/include/aclnn_*.h`

若本机为 `x86_64`，将 `aarch64-linux` 换成实际 triplet 目录（如 `x86_64-linux`）。

---

## 2. Agent 必做：先枚举再读声明

1. **按算子语义关键词搜索头文件名**（示例）：

   ```bash
   find "$CANN_ROOT" -path '*/include/*aclnnop*' -name 'aclnn_*.h' 2>/dev/null | head
   rg -l 'MatMul|matmul|Add|add' "$CANN_ROOT/aarch64-linux/include/aclnnop" 2>/dev/null | head -n 30
   ```

2. **打开命中的 `aclnn_<name>.h`**，记录：
   - 函数名（如 `aclnnXxxGetWorkspaceSize` / `aclnnXxx`）
   - 输入/输出 `aclTensor` 顺序与语义
   - 文档注释中的 **数学定义、约束、dtype 支持**

3. **若头文件不足以判断行为**：在同一 CANN 树内搜索 **同名或同前缀的 `.cpp`**（实现或样例），或查阅随包文档；仍不明确则在回复中 **列出待用户确认的点**，不要臆造公式。

---

## 3. 选取 Golden 的决策规则

| 情形 | 建议 Golden |
|------|----------------|
| CANN 已有与目标 **语义一致** 的 aclnn 接口 | 以该 aclnn 算子输出为 **主 Golden**（同 shape/dtype 下对比） |
| 仅有 **子步骤** 接口（如 matmul + add） | 用 **多算子拼接** 得到参考输出（注明拼接顺序与是否 inplace） |
| 算子为 **新融合**，无单接口对应 | 用 **数学等价分解**（如 PyTorch/NumPy 公式，或 aclnn 基础算子链）写清 **参考伪代码**；数值对比时注明 rtol/atol |
| 需要与 **PyTorch** 对齐 | 明确 `torch` 算子名与 `dtype/device`，并说明与 aclnn 拼接的差异（如 layout） |

Golden 文档中应固定：**输入张量规格、dtype、layout（若相关）、随机种子、对比指标（max abs / rtol+atol）**。

---

## 4. 输出模板（写入设计或测试说明即可）

```markdown
## Golden 来源

- **CANN 路径**：`<具体头文件路径>`
- **接口**：`<aclnn 函数名>`
- **语义摘要**：`<与目标算子一致的数学行为一句话>`
- **若拼接**：`<算子链：op1 -> op2 -> ...>`
- **对比方式**：`<单测中 golden 张量如何生成；容差>`
- **未覆盖 / 风险**：`<边界 shape、溢出、非对齐等>`
```

---

## 5. 自检

- [ ] 头文件路径来自用户本机 `CANN_ROOT`，非臆造。
- [ ] Golden 与目标算子 **I/O 契约** 一致（顺序、含义、是否覆盖输出）。
- [ ] 若使用拼接，**无未定义顺序**（如多输出依赖时应写清）。

---

**一句话**：先在 `CANN_ROOT` 下用 `find`/`rg` 锁定 `aclnn_*.h`，读清声明与注释，再决定 **单接口 Golden** 还是 **多接口拼接 Golden**，并把路径、函数名与容差写死，避免“口头 golden”。
