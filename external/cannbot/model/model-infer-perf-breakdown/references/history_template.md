# HISTORY.md 条目模板

`<run_dir>/HISTORY.md` 是 agent 维护的人类可读运行日志。每完成一次 `render.py`（Step 3 Phase 2 出 `runs/<label>/`）后必须 append 一段。append-only，按时间从老到新排，禁止回头改旧条目。

## 字段约定

| 字段 | 内容 |
|------|------|
| `<label>` | 本次 run 的 label（与 `runs/<label>/` 一致） |
| `<date>` | 写本条时的绝对日期 + 时分（不要用相对词如"今天"） |
| `prof` | `kernel_details.csv` 路径，相对或绝对均可，只要 reproducible |
| 复用 / 新建 | 每个产物文件单独一行；变更项写"变更点：…"或"rule 增减：…"等简短说明 |
| 模型结构 | 一两句话；与 `structure_spec.json` 一致即可，不重复展开 schema |
| sample 锁定 | 本次写入 `sample_ack.json` 的 stream sample 摘要；复用上一 label 时写 "同 `<prev_label>`" |
| cluster 规则要点 | 只写本 run 相对上一 label 的增删；首次跑写 "见 `<network>_spec.json`" |
| 渲染产物 | `<run_dir>/runs/<label>/index.html` 路径 |
| 关键观察 | 2–4 条 bullet：outlier 模式 / 与上一 label 的 Δ 摘要 / 后续待追的疑点 |

## 模板

```markdown
## <label> — YYYY-MM-DD HH:MM

**prof**: <kernel_details.csv 路径>
**复用 / 新建**:
- raw_ops: 新建 / 复用上一 label
- structure_spec: 复用 / 新建（变更点：…）
- structure_draft: 复用 / 新建（sample 变更：…）
- <network>_spec: 复用 / 新建（rule 增减：…）

**模型结构**: <一句话，如 "<network>，main N 层 [attn, moe]，mtp_head M 层 [mtp, moe]"。>

**sample 锁定**:
- <component> : Excel row <start>-<end>  head=<head_op>@<start>  tail=<tail_op>@<end>
- ...（每个 expected_component 一行；复用之前 sample 时写 "同 <prev_label>"）

**cluster 规则要点**: <本 run 相对上一 label 改了什么；首次跑写 "见 <network>_spec.json"。>

**渲染产物**: `<run_dir>/runs/<label>/index.html`

**关键观察**:
- <例：csa flash_attn 在 L3/L7/L11 比中位慢 18%，上一版无此抖动>
- <例：moe combine 中位降 0.42 ms，对 hcom-side 关注点确认>
```

## 撤回条目

若某次 label 被废弃（如用户说"上次跑错了"），**不要**修改旧条目内容，删整段并在新条目末尾加一行：

```
> 撤回 <bad_label>，原因：<一句话>。
```
