# dump 与比对手段速查

约束：**不使用 ms 开头的官方工具（msopdump/msaccucmp 等），不使用 CPU 仿真器。**

## 手段选型

| 场景 | 手段 | 用途 |
| --- | --- | --- |
| om 输入/输出获取 | 自写 acl 执行脚本存 npy | acl.mdl 执行 om，输入输出 memcpy 回 host 存档 |
| kernel 内中间结果 | DumpTensor / printf | 打点输出 UB 中间值 |
| 算子逻辑最小闭环 | aclnn 直调脚本 | 绕过 ATC 链路，直接验证算子逻辑 |
| 误差比对与形态分析 | 自写比对脚本 / 项目验证工具 | 误差事实与形态判词（定位辅助，非验收） |
| 稳定性分流 | 重复运行比对 | 同一输入连跑多次，输出两两 diff |

## om 输入/输出 dump（自写 acl 脚本）

用 pyACL 完成最小执行闭环并存档：

1. `acl.mdl.load_from_file` 加载 om，按模型要求准备输入（动态 shape 模型先按档位设置）；
2. `acl.mdl.execute` 执行，输入 buffer 内容与每个输出 buffer 都 `acl.rt.memcpy` 回 host；
3. 存为 npy（命名带语义：`node_out_{name}_{shape}.npy`），交给机读比对（自写脚本或项目验证工具）。

整网 om 定界技巧：把可疑子图/节点**单独转 om 分段执行**，对比分段输入输出，把"整网错"收窄到具体节点区间。

## torch tensor 落盘比对

ST 框架产出 torch tensor 时，落盘为 `.pt` 或直接转 npy，交给机读比对（自写脚本或项目验证工具；容差从指定验证工具配置取）：

```python
torch.save(golden, "golden.pt"); torch.save(out, "npu_out.pt")
# 或直接转 npy：
np.save("golden.npy", golden.cpu().float().numpy())
np.save("npu_out.npy", out.cpu().float().numpy())
```

## 跨用例对比（单跑 vs 批跑）

疑似跨用例污染（单独跑 PASS、批量跑 FAIL）时：

1. 用项目 ST runner 单独跑失败用例，记录输出；
2. 按原批量顺序跑，记录同一用例输出；
3. 两次输出 diff：不一致 → 跨用例污染成立，且差值常能直接读出"污染来自哪个前置用例"（其属性值出现在错误结果里）；
4. 注意：**此类问题禁止缩复现到单用例**（会消除复现），最小化方向是"缩前置用例序列"。

## kernel 内打点（Ascend C）

```cpp
// 打印 UB 中间 tensor 的前若干个值
DumpTensor(tmpUb, 1, 64);        // (tensor, 序号, 元素个数)
// 或
printf("stage2 sum = %f\n", (float)tmpUb.GetValue(0));
```

打点原则：从输入端往输出端逐 stage 打，找到首个发散 stage 即停；不要一次性全量打（日志爆炸）。

## 三段对比法（怀疑编译链路时）

aclnn 直调结果 vs 单算子 om 结果 vs 整网 om 该节点结果（分段 om 获取），三者两两比对，定位差异引入环节（详见 patterns/atc-compile-chain-diff.md）。

## 稳定性分流（Step 2 的关键判断）

同一最小输入在板上**连续运行 3~5 次**，输出两两比对：

- 完全一致 → **逻辑类**（数值/维度/尾块方向）；
- 存在不一致 → **同步/内存类**（pipe 同步、double buffer、脏数据方向）。
