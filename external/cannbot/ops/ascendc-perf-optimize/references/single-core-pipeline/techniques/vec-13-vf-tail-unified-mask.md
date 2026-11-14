# 技术卡片 vec-13：VF 尾块统一 mask（折进最后一拍，不设独立尾块循环）

- **适用代际**：arch35（A5 / DAV_3510）独占；本卡依赖 RegBase / MicroAPI / VF，不用于 arch22 实现。
- **适用 bound**：VEC / Scalar
- **一句话**：尾块不另写 `tailTwoVL/tailOneVL` 循环，用「预算好的计数器 + `UpdateMask` 逐拍递减」让一个主循环的最后一拍自动窄化 mask 覆盖尾块。

## 何时用（诊断特征）

- VF 里对不整除 VL 的长度，用「`loopNum` 主循环 + 一到两个独立尾块循环」处理余数。
- 尾块循环把整段 load/compute/store 计算体**又抄一遍**（指令 cache 压力、维护易错），且尾块边界量（`tailNum/tailTwoVL/tailOneVL/tailLen`）计算繁琐、易越界。
- 一拍处理多个 VL 半区（如 interleave 一拍两个 VL）时尤为明显：尾块要区分"尾 > 1 VL"和"尾 ≤ 1 VL"两种。

## 原理

`UpdateMask<float>(cnt)` 每调用一次消耗一个 VL 宽度并**递减 `cnt`**：满拍返回全 mask，`cnt` 不足一个 VL 时自动返回窄化 mask。把总长按各 VL 半区**预算**成初值（尾块加到最后一拍所在的那份计数器上），单主循环内每拍 `UpdateMask` 递减，最后一拍的 mask 自然只覆盖尾部有效元素。**一个循环、一份计算体**覆盖主体 + 尾块。

## 怎么改（before / after）

```cpp
// before：主循环 + 两个独立尾块循环，计算体抄三遍
uint16_t loopNum = repeatTimes / 2;
for (i = 0; i < loopNum; i++)   { <load/compute/store 全宽 mask> }
for (i = 0; i < tailTwoVL; i++) { <同一段计算体，尾 mask> }   // 尾 > 1 VL
for (i = 0; i < tailOneVL; i++) { <又一遍计算体，尾 mask> }   // 尾 <= 1 VL

// after：预算 partN，单主循环，UpdateMask 逐拍递减，末拍自动窄化
uint32_t loopSize   = 2 * VL_FLOAT32_SIZE;             // 一拍两个 VL 半区
uint16_t dLoopCount = (dLen + loopSize - 1) / loopSize;// 单一循环次数，已含尾块
uint32_t part1Num   = (dLoopCount - 1) * VL_FLOAT32_SIZE;
uint32_t part2Num   = part1Num;
uint32_t tailNum    = dLen - part1Num - part2Num;
if (tailNum > VL_FLOAT32_SIZE) { part1Num += VL_FLOAT32_SIZE; part2Num += (tailNum - VL_FLOAT32_SIZE); }
else                          { part1Num += tailNum; }  // 尾 <= 1 VL：全落 part1，part2 末拍 mask 空
// 主循环：
uint32_t p1 = part1Num, p2 = part2Num;
for (uint16_t i = 0; i < dLoopCount; i++) {
    pregPart1 = Reg::UpdateMask<float>(p1);        // 逐拍递减，末拍窄化
    pregPart2 = Reg::UpdateMask<float>(p2);        // 逐拍递减，末拍窄化/清空
    <load/compute/store 均以 pregPart1 / pregPart2 为 mask，i*loopSize 为 offset>
}
```

## 性能数据

msprof `aiv_time` + I-Cache：计算体从 3 份收敛到 1 份，尾块无分支。**待实测**（预期：短向量 / 尾块占比高时，去重复计算体 + 省尾块标量分支收益明显）。

## 红线（Stop rules）

- 每个被处理的「行 / 段」进入主循环前**必须重置计数器**（`p1 = part1Num; p2 = part2Num;`），否则第二行起 mask 已被上一行递减耗尽。
- `part1Num/part2Num` 的预算须与 offset 步进（`i*loopSize`、`+VL`）严格对齐，算错会读越界或漏尾。
- 计数器类型用 `uint32_t`（`UpdateMask` 按引用递减），别用会溢出/截断的窄类型。

## 关联

- 源：RoPE arch35 `BatchInterleaveModeVF` 的统一 mask 写法（对照单行 `InterleaveModeVF` 的 `tailTwoVL/tailOneVL` 反面写法）；详见 `ascendc-performance-best-practices` 技能 rope 族 `vector_fused`（尾块小节）。
- 相邻卡片：vec-05（寄存器内直算，配套 VF 主体）、vec-10（小定长内循环全展开）、vec-09（外层 for 下沉进 VF）。
