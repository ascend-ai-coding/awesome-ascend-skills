# Phase 0b — stream sample ack 流程

Step 2 Phase 0b 的详细工作流。**核心契约**（SKILL.md 也写了）：每个 `expected_components[*]` 必须有用户 ack 过的 sample 范围，并转换成 `stream_sample_ack.v1`。每个 component 必须有且只有一个 `primary` stream；确认属于该 component 的旁路 / 辅助流写 `auxiliary`；无法确认的流不要硬塞进 component。

**`auxiliary` 纳入判据（什么样的流该并进 per-layer component）**：按**语义归属**判断，**不**按时间是否重叠判断。

- 流在语义上**属于某 component**（看它算的是哪个模块的什么）→ **写成该 component 的 `auxiliary`**，即使它的算子被调度到独立流、与本层时间脱节。
- 流**不属于任何 per-layer component**（全局通信 / 采样 / embedding / IO 等 side flow）→ 不写，留 `unmatched`。

**关键：时间脱节（overlap=0）不是"不写成 `auxiliary`"的理由。** 它只决定 `render.py` 要不要把这条流从 bubble 剔除（见下），**不决定归属**。一条语义上属于某 component 的脱节流，若因为"脱节"就扔进 `unmatched`，会丢失"它属于该 component"的语义、split 时也不再和该 component 关联——所以应写成 `auxiliary` 保留归属，而不是 unmatched。

**写成 `auxiliary` 后，时间脱节会被自动处理、不污染报告**：`detect_structure.py` 对脱节流发 hard warning `auxiliary_stream_temporally_displaced`（形态无关的并入膨胀体检，细节见 `detect_structure.py` / `sample_matching.py` docstring）；该流的 op **保留在原 component**（matched 不丢、partition 不破、Step 4 split 照切、逐层可查），`render.py` 自动把它从 cluster/bubble/TOTAL **剔除**并在报告标注（"已从 TOTAL/bubble 排除"）。所以默认就是 **① 写成 aux → 保留在 component + 度量剔除 + 标注**（信息最全、bubble 不虚高）；只有当你确认该流**不属于**这个 component 时，才 ② 留 `unmatched`。warning 只为知情，不强制。

输入路径分两档：

---

## A 档：用户直接给

最理想：用户给 `component:row_lo-row_hi`（Excel 行号）+ 入口/出口算子提示；或只给入口/出口算子提示，agent 去 raw_ops 定位典型窗口。

agent 工作步骤：

1. 把用户给的算子名对照 `raw_ops.json` 的 `normalized_name` 定位实际命中位置。用户给 row 区间时校验窗口开头/结尾附近的算子是否与用户提示一致。
2. 回贴给用户 ack（**所有行号用 Excel 行号** = `org_index + 2`，并列出邻近 op 辅助核对）：

   ```
   csa : excel 1210-1456  head=RmsNorm@1210  tail=TransposeBatchMatMul@1456
         邻近：RmsNorm@1208(prev), HcPre@1457(next)
   moe : excel 1456-1750  head=AddRmsNormDynamicQuant@1456  tail=MoeDistributeCombine@1750
         邻近：HcPost@1751(next)
   ```

3. 用户 ack（或单点纠正）后，agent 只读取这个 sample 小窗口，把窗口内 op 按 `stream_id` 拆成 `stream_samples[]`：
   - 默认把 op 数最多 / duration 最大、且包含主要 compute 路径的 stream 设为 `primary`。
   - 明确属于同一 component 的 index/cache/vector side path 设为 `auxiliary`。
   - 不确定的 stream 不写入 component；保留到 warning / unmatched，不要为了好看硬归属。
4. 写 `sample_ack.json`，进 Phase 1。

---

## B 档：用户给不出 → agent 读模型脚本推（仍必须用户 ack）

用户给不出 sample 范围时，agent **必须读 Step 0 收齐的 `model_script_paths`**。脚本缺失意味着 Step 0 没过关，回去补，不要绕过。

1. 读 `forward()` 找每个 component 的第一个 / 最后一个 compute 调用，推断对应 prof 端 kernel 名：
   - `nn.RMSNorm` → `RmsNorm`
   - `MoEDispatch` → `MoeDistributeDispatchV2`
   - `nn.Linear` / 量化矩乘 → `MatMulV3` / `QuantBatchMatmulV3`
2. （可选）跑 `python scripts/detect_structure.py --explore -r <run_dir>/raw_ops.json -o <run_dir>/explore.json` 看候选算子的频次是否符合预期（count = N × 该 component 在该 phase 出现次数；next_kernels_distinct 低的更适合做锚）。
3. **同样按 A 档第 2 步格式回贴用户 ack**——明确标注"以下 sample 窗口和入口/出口提示是 agent 读 `<脚本路径:行号>` 推出来的，请核对"。
4. 用户 ack 通过才写 `sample_ack.json`；用户单点纠正 → 回到 1 重推该项再 ack。

---

## `sample_ack.json` schema

```json
{
  "schema_version": "stream_sample_ack.v1",
  "acked_at": "2026-05-22T22:00:00+08:00",
  "components": {
    "csa": {
      "source": "user",
      "primary_stream_id": "189",
      "stream_samples": [
        {
          "stream_id": "189",
          "role": "primary",
          "head_op": "RmsNorm",
          "tail_op": "TransposeBatchMatMul",
          "op_indices": [1208, 1210, 1214]
        },
        {
          "stream_id": "203",
          "role": "auxiliary",
          "head_op": "Slice",
          "tail_op": "ScatterNdUpdate",
          "op_indices": [1209, 1212]
        }
      ],
      "agent_note": "stream 189 is the main compute path; stream 203 carries index/cache side ops"
    },
    "moe": {
      "source": "agent_inferred_from_script:modeling_xxx.py:412",
      "primary_stream_id": "203",
      "stream_samples": [
        {"stream_id": "203", "role": "primary", "op_indices": [1454, 1456, 1460]}
      ]
    }
  }
}
```

`source` 三种取值：
- `"user"` — 用户原话直接给
- `"agent_inferred_from_script:<path>:<line>"` — agent 读脚本推出来的
- `"user_corrected"` — agent 推完后用户改过一处

三种都必须经过用户 ack 才允许写入。`expected_components` 列表中任一缺失即视为 ack 未完成，`detect_structure.py` 启动时 exit 1。

若当前只有人工给出的 Excel 区间，agent 必须先读取对应小窗口，按 `stream_id` 拆成 `stream_samples[]`，再写入上面的 v1 schema（Step 2 输入须为此 v1 schema）。

---

## 行号 domain 速查

- **Excel 行号**（用户口头/书面交换，agent 用它读取小窗口并转换为 `sample_ack.json`）：1-based，header 占 row 1
- **`org_index`**（`raw_ops.json` 字段、Step 4 切 csv 用）：0-based 数据行号，与 Excel 关系 `excel_row = org_index + 2`（标准 csv 阅读器）
- **`index` / `op_indices`**（operators[] 内 sequential，AI_CPU 过滤后致密重编号）：`sample_ack.json`、`structure_draft.json`、`render.py`、`split_artifacts.py` 全部按这个 domain

AI_CPU drift：默认剔除 `accelerator_core ∈ {AI_CPU, AICPU}` 后 `index != org_index`，**手动减 2 后写裸 `op_indices` 必错**。agent 必须先按 Excel 行号反查对应 raw ops，再把确认后的致密 `index` 写入 `stream_samples[].op_indices`。
