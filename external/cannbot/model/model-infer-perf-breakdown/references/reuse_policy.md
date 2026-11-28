# 复用策略：根据 prof 差异选档位

进 `<run_dir>` 后 agent 必须做一次自动 diff，把"本次 prof vs 上一 run"分到三档之一，再按档位复用历史产物。判断完成后**复述一句结论给用户 ack**：

> 检测到 prof 相对上一 label `<X>`: 小改 / 中改 / 大改（原因：…）。打算 A / B / C 档复用，对吗？

用户不同意时按用户给的档位走，**不要**自动再判一次。

---

## Diff 信号源

跑新 prof 的 `analyze_kernels.py` 之前不能比对（没新 raw_ops.json）。做法是 **先跑 Step 1 出新 raw_ops.json**，但落盘到临时位置 `<run_dir>/raw_ops.next.json`，然后比：

| 信号 | 怎么算 | 用途 |
|------|--------|------|
| **kernel 集合 diff** | `set(new.normalized_name) Δ set(old.normalized_name)` | 出现新算子 → 至少中改 |
| **anchor count 偏差** | 选 1-2 个 anchor kernel（HISTORY 中 stream sample 的代表算子），比 new/old 总 count | 偏差大 = 层数 / phase 变了 |
| **总 op 数偏差** | `(|new| - |old|) / |old|` | > 30% 视为大改 |
| **sample 锚点试探** | 拿上一 run 的 stream sample op 名 + shape，在新 raw_ops 的相近 stream-local 位置试探命中 | 命中位置漂移度量 |
| **shape 分布** | 同名 anchor 的 input_shapes top-3 是否一致 | shape 漂移 = 中改 |

判断完后把 `raw_ops.next.json` rename 成 `raw_ops.json`（之前的覆盖前先看 HISTORY 有没有引用——`label_baseline` 这类初版别覆盖丢）。

---

## 三档判断 + 复用动作

### 小改 — A 档

**信号**（同时满足）：
- kernel 集合完全相同
- 每个 anchor kernel 的总 count 偏差 = 0
- 上一 sample 的 stream-local anchors 在新 raw_ops 中、距原位置 **±5 op** 内命中

**复用动作**：
- `structure_spec.json`：复用，**不用**问用户
- `structure_draft.json`：先把上一轮 stream sample 平移到命中位置并重写 `sample_ack.json`，再跑 Phase 1 `detect_structure.py` 重切
- `<network>_spec.json`：复用，**不用**对话
- 跳过 Phase 0a / 0b 对话；新 sample 显示给用户做一次 ack 即可

### 中改 — B 档

**信号**（任一即可）：
- 出现新 `normalized_name`（cluster 规则可能漏）
- anchor count 偏差 0 < δ ≤ 10%
- 上一 sample stream-local anchors 命中但偏移 > 5 op（层内逻辑动了，结构没变）
- 同名 anchor 的 input_shapes 分布前后不一致

**复用动作**：
- `structure_spec.json`：复用，回显问用户 "composition 还是 X / Y 吗？"
- `structure_draft.json`：**重走 Phase 0b**——sample 必须重出，原 sample 的偏移已经超 lookup 半径
- `<network>_spec.json`：复用 + 增量补 rule（针对新 `normalized_name`）；render 后看 `unmatched ops` 面板非空就回 Step 3 Phase 1 补规则

### 大改 — C 档

**信号**（任一即可）：
- anchor count 偏差 > 10%（层数变 / phase 增减 / sparse → dense 切换）
- 总 op 数偏差 > 30%
- 关键 anchor kernel 在新 prof 中**完全消失**（换了 attention 实现 / 去掉了某 phase）

**复用动作**：
- `structure_spec.json`：**重走 Phase 0a**——必须重新问用户模型结构
- `structure_draft.json`：重走 Phase 0b
- `<network>_spec.json`：保留作为人类可读参考，但**不要**直接 `-s` 喂进去；逐条 rule 审一遍，必要时改名 / 删除 / 新增

---

## 首次跑（无历史）

`<run_dir>` 里没有 `raw_ops.json` / `HISTORY.md` → 跳过 diff，按全量流程走 Step 1→4，最后 append 第一条 HISTORY。

## label 切换 ≠ prof 切换

用户给了新 `<label>` 但 prof 路径与上一 run 相同（如改了 sample 想再跑一次）→ diff 全 0，按 "无变化" 处理，所有历史产物直接复用，HISTORY 里只在"复用 / 新建"段标 "全部复用，仅改 label / sample"。
