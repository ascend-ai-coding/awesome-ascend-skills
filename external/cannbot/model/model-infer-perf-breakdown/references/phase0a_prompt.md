# Phase 0a — 用户结构描述 prompt

Step 2 Phase 0a 的逐字 prompt 模板。跑本 skill 的 agent 把下面发给用户，不要改写，不要先于此发问。

---

描述一下这个模型在 prof 跑了什么，覆盖三件事（任意顺序，不会就写"不知道"，我会转 explore 模式帮你看候选）：

**① 主模型**：实际跑了多少层、用到哪几种 composition
- "component" = 一层里和 attention / ffn 平级的功能块：
  - attention 类（`self_attn` / `mla` / `swa` / `csa` / `hca` / `linear_attn` …）
  - ffn 类（`dense_mlp` / `moe`）
- 一层是一组 component 的组合（如 `[csa, moe]`），叫 **composition**
- 主模型用到的所有 composition 种类列出来即可，不必描述层之间怎么排
- prof 若被裁剪（config 写 61 但只跑 6），按实跑的写

**② 投机解码 / MTP / draft / speculative head**：有没有？几层？每层 component 是什么？

**③ 其他独立子模型**：encoder + decoder / vision_tower + lm 等？分别用什么名字、几层、什么 composition？

命名规则：同一种功能跨段用同一个 component 名（如 moe 在主干和 MTP 都出现就都叫 moe）；phase 名（main / mtp_head / encoder / decoder…）AI 自动起，你不用想。

---

## agent 后处理

解析答案后**自动起 phase 名**：
- ① → `main`
- ② → `mtp_head`
- ③ → 用户提到的类别名（`encoder` / `vision_tower` / ...）

写 `<run_dir>/structure_spec.json` 后**回显完整 JSON** 给用户 ack；用户改一句、agent 回贴一次，直到用户确认才能进 Phase 0b。
