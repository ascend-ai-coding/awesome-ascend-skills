# Alignment Guards for Vectorized Pooling Kernels

## 对齐约束总览

向量化 pooling kernel 有三层对齐约束：

| 约束 | 要求 | 原因 | 违规后果 |
|------|------|------|---------|
| C%8==0 | 无条件 | Add(C)/Muls(C) 操作 32B 对齐 | `UB address accessed by VEC instruction is not aligned` → 崩溃 |
| fp16 W*C%16==0 | 仅 fp16 | 输入 GM 行 DataCopy 32B 对齐 | 数据损坏（值变 512.0/nan） |
| fp16 OW*C%16==0 | 仅 fp16 | 输出 GM 行 DataCopy 32B 对齐 | 最后一个不完整输出行尾部损坏 |

> **⚠️ 方案相关（重要）**：上表假设「行粒度整行搬运」（`DataCopy` 以 `W*C` / `OW*C` 为单位）。若你的向量化槽在 **C 维**——即用 `DataCopyPad(blockCount=C, blockLen=sizeof(T))` 的 gather/scatter 方案（反向语义见 `ops/tilelang-op-design/references/pooling/backward-patterns.md`、实现见 [backward-implementation.md](backward-implementation.md)，另见 [grad-v2-lessons.md](grad-v2-lessons.md) §5）——fp16 守卫改为 **`C%16==0`（不是 `W*C%16`）**，且 **`C%16` 比 `W*C%16` 更严**（fp16 下 C=8 满足 `W*C%16` 但会被 `C%16` 拒绝）。写 host `TORCH_CHECK` 前先明确自己 kernel 的 blockLen/blockCount 布局（详见下文「方案相关的守卫」）。

## 为什么需要这些约束

### C%8==0: 向量计算对齐

AscendC V 核向量寄存器为 256-bit (32B)，一个 V 指令操作 256/32=8 个 fp32 元素或 256/16=16 个 fp16 元素。行粒度 kernel 使用 `Add(acc[ow*C], acc[ow*C], in[iw*C], C)` 这种 C-wide 向量操作，要求指针 32B 对齐。

C 非 8 倍数时，最后一个向量操作（C%8 元素）无法对齐，触发 UB 对齐 fault。

### fp16 W*C%16==0: 输入 DataCopy 对齐

`DataCopy(inLocal_, xGM_[rowBase], W*C)` 以 `W*C` 个 half 为单位搬运。GM→UB 的 DMA 要求源/目标地址 32B 对齐。W*C 个 half = W*C*2 bytes，需要 W*C*2 % 32 == 0，即 W*C%16==0。

不满足时，DMA 会损坏最后一个不对齐的 32B block 内的数据（非完整 block 的部分会被填充/截断）。

### fp16 OW*C%16==0: 输出 DataCopy 对齐

与输入对称：`DataCopy(yGM_[outRow*OW*C], outLocal_, OW*C)` 需要 OW*C 个 half 元素对齐。

不满足时，最后一个不完整 block 的尾部值损坏（实测值变为 512.0 或 nan）。

## 方案相关的守卫：向量化槽在 C 维时用 C%16，不是 W*C%16

上表三守卫对应「行粒度整行搬运」方案。但反向 gather/scatter（见 `ops/tilelang-op-design/references/pooling/backward-patterns.md`）和深度-only 快路径常用 `DataCopyPad` 把**每个 C 值 pad 到 32B 槽**（`blockCount=C, blockLen=sizeof(T)`），此时向量化槽在 **C 维**，守卫不同：

| 方案 | 向量化槽 | fp32 守卫 | fp16/bf16 守卫 |
|------|---------|----------|----------------|
| 行粒度整行搬运 | W×C 行 | C%8==0（计算对齐） | W*C%16==0 + OW*C%16==0（搬运对齐） |
| C 维向量化槽（gather/scatter / 深度快路径） | C | C%8==0 | **C%16==0** |

注意 **`C%16`（fp16）比 `W*C%16` 更严**：C=8 的 fp16 满足 `W*C%16`（W*8 为偶数即可）但会被 `C%16` 拒绝。规则：对齐守卫取决于**你的向量化槽在哪一维**，不是固定公式——槽在 C 维 → `C % (32/sizeof(T))`；槽在 W×C 行维 → `W*C % (32/sizeof(T))`。完整论证见 [grad-v2-lessons.md](grad-v2-lessons.md) §5（含「放宽可行性」评估：结构代价高于收益，默认保留 TORCH_CHECK 拒绝）。

## Host 侧守卫实现

```cpp
// 1. C 维度对齐（无条件，fp32 和 fp16 都需要）
TORCH_CHECK(self.size(1) % 8 == 0,
    "avg_pool3d: C must be a multiple of 8, got C=", self.size(1));

// 2. fp16 输入行对齐
if (self.scalar_type() == at::kHalf) {
    TORCH_CHECK((self.size(4) * self.size(1)) % 16 == 0,
        "avg_pool3d: fp16 requires W*C multiple of 16, got W*C=",
        self.size(4) * self.size(1));
}

// 3. fp16 输出行对齐（OW 是计算得出的！）
if (self.scalar_type() == at::kHalf) {
    TORCH_CHECK((OW * C) % 16 == 0,
        "avg_pool3d: fp16 requires OW*C multiple of 16, got OW*C=",
        OW * C);
}
```

**注意**: OW 由 pooling 参数计算得出，守卫 #3 必须在 OW/C 计算完成后设置，不能和 #1/#2 放在一起。

## 测试策略

### 对齐形状（应通过）
```
shape=(2,16,32,32,32)  C=16 → C%8=0 ✓, fp16 W*C=512 %16=0 ✓
shape=(2,32,16,32,32)  C=32 → C%8=0 ✓, fp16 W*C=1024 %16=0 ✓
shape=(2,64,8,16,64)   C=64 → C%8=0 ✓, fp16 W*C=4096 %16=0 ✓
```

### 不对齐形状（应报错）
```
shape=(2,4,32,32,32)   C=4 → TORCH_CHECK 拒绝 "C must be multiple of 8"
shape=(2,16,32,32,31)  fp16 W*C=496 → 496%16≠0 → TORCH_CHECK 拒绝
```

### 扩展 smoke 测试
标准 6 场景通常都已对齐（设计时避开了不对齐 case）。追加以下 smoke case 可发现守卫缺失：
- C=4（仅 fp32?）→ 预期报错
- fp16 + 奇数 W → 预期报错
- fp16 + 奇数 OW（通过 padding 制造）→ 预期报错

## 从崩溃到守卫的调试流程

```
向量化 kernel 崩溃
  │
  ├─ 错误信息: "UB address accessed by VEC instruction is not aligned"
  │   → 检查 C%8==0 守卫
  │
  ├─ 错误信息: 无崩溃，但输出值异常（512.0, nan）
  │   → 检查 W*C%16==0 (输入) 和 OW*C%16==0 (输出)
  │
  └─ 错误信息: 无
      → 扩展 smoke 测试用不对齐 shape 触发，确认守卫报 RuntimeError 而非崩溃
```

## 不应对齐的方式

```cpp
// ❌ 不要用 padding 或 s̲l̲i̲c̲i̲n̲g̲ 绕开对齐
// ❌ 不要静默丢尾（剩余元素的语义需保留）
// ✅ 必须 TORCH_CHECK 显式拒绝，给出清晰的 C/W/OW 值和修复建议
```
