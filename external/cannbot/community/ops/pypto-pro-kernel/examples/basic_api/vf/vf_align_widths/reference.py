# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""``dist=`` 的位宽绑定机制该产出什么：跨位宽的展开 / 广播 / 打包。

与 ``vf_align`` 的分布模式同属 ``dist=`` 参数，这里是**位宽绑定**的那一半：语义随
「源 dtype × dist 粒度」组合变化，用对了是零成本重排，用错了是静默垃圾。每条都是
2026-09-29 在 Ascend950PR 上实测反推的（bit-exact probe），机制细节的 owner 是
references/api/vf.md 二点十三 / 二点十四。
"""

import numpy as np

N = 128
BLOCK_BYTES = 32


def low_bytes(words, signed):
    """Bit extraction, including out-of-range words to distinguish saturation."""
    bits = (words.astype(np.int64) & 255).astype(np.uint8)
    return bits.view(np.int8) if signed else bits


def make_fp16_input(n=N):
    """``1..n`` 的 FP16（全部可精确表示），一眼能看出来自哪个下标。"""
    return np.arange(1, n + 1, dtype=np.float16).reshape(1, n)


def make_fp32_input(n=N):
    """``0.5, 1.0, 1.5, ...``：FP16 可精确表示，窄化不产生舍入歧义。"""
    return (np.arange(1, n + 1, dtype=np.float32) * 0.5).reshape(1, n)


def widen_unpk(x16):
    """UNPK + ``astype(FP32, ZERO)`` = 保序宽化：输出就是源值的 FP32，顺序不变。

    UNPK 把连续 64 个 FP16 展开进 64 个 b32 容器的低半（保源序、高半零填），
    ``astype(..., layout=ZERO)`` 恰好从容器低半做数值转换——两条指令等价于
    「FP32 源的一次 load」。这是 FP16 源做 FP32 计算的标准入口；普通 load 后直接
    宽化 astype 则是偶/奇下标**源子采样**，静默丢一半（vf.md 二点十三）。
    """
    return x16.astype(np.float32)


def load_e2b(x16, n=N):
    """E2B 的真实语义：**element-to-block 广播**——每个源元素复制填满一个 32B
    DataBlock（FP16 下 16 个 lane，一条寄存器装 8 个源元素的块）。

    官方 ``LoadDist.md`` 把它描述成「16位宽→32位宽扩展」，是错的（open-items
    上报候选）；``vf_align`` 第六节 FP32 下的既有观察（每元素重复 8 次 = 32B/4B）
    与本条（重复 16 次 = 32B/2B）互证：任何位宽下都是块广播。想要「16→32 展开」
    用 UNPK。
    """
    per_block = BLOCK_BYTES // 2
    out = np.empty((1, n), dtype=np.float16)
    for k in range(n // per_block):
        out[0, slice(k * per_block, (k + 1) * per_block)] = x16[0, k]
    return out


def narrow_pack_b32(x32):
    """``astype(FP16, ZERO)`` + store ``PACK_B32`` = 窄化直出稠密：输出 = fp16(源)。

    窄化结果落在每个 b32 容器的低 16 bit；``PACK_B32``（取每个 b32 元素的低半 bit
    连续写出）恰好把它们抽成稠密 FP16——写 **ND 目标**时替代「两 chunk
    de_interleave 拼接」少一条重排（vf.md 二点十）；喂 vsstb 游标散写（需要稠密
    128 宽寄存器作源）时仍走 de_interleave。
    """
    return x32.astype(np.float16)


def pack_coarse_trap_u16(x16, n=N):
    """**陷阱参考（钉住当前行为）**：粗名 ``PACK`` 按源 dtype 自动选粒度，FP16
    寄存器选到 PK_B16 =「取每个 **b16** 元素的低 **8 bit** 连续打包」——纯位
    截断，不是取元素。128 个低字节打包成 64 个 b16 槽（小端：元素 2k 的低字节
    在槽 k 的低 8 位、元素 2k+1 的在高 8 位），后 64 槽不被写。

    对 FP16 数据这是垃圾（1.0=0x3C00 的低 8 bit 是 0）——窄化直出必须**显式**
    ``PACK_B32``。返回 uint16 视角的期望（值域是位模式，不是数值）。
    """
    lo = (x16.view(np.uint16)[0] & 0xFF).astype(np.uint16)
    packed = lo[0::2] | (lo[1::2] << np.uint16(8))
    out = np.zeros(n, dtype=np.uint16)
    out[: packed.size] = packed
    return out.reshape(1, n)
