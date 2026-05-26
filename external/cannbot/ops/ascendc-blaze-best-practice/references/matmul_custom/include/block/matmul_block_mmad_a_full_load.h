/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_block_mmad_a_full_load.h
 * \brief A_FULL_LOAD_MODE 的 BlockMmad SFINAE 特化。
 *
 * 由 `matmul_block_mmad.h` 在 namespace Block 之外 #include，进入同一 namespace
 * 提供第二份 SFINAE 特化。两个特化通过 `MatmulMultiBlockPolicy<MODE>` 区分，
 * launcher 切 policy 即决定走哪条流水。
 *
 * 关键差异（vs NO_FULL_LOAD_MODE）：
 *   - L1 布局：A 在 offset=0 单缓冲常驻，B 紧跟其后双缓冲 ping-pong
 *   - A 仅在每个核首次 operator() 的 K 循环里搬入；后续 N-tile 直接复用
 *   - A 端用独立的 MTE1↔MTE2 event slot（A_FLAG=2），与 B 的 ping-pong (slot 0/1) 完全错开
 *   - 内层 L1→L0A 始终从持久 A 区按 kL1Offset 切片
 *   - mmadCmatrixInitVal 保持 loop-local 语义（iter0==0 && iter1==0），与 A 是否复用无关
 *
 * ⚠️ 本特化只支持 transA=false / transB=false（layoutA / layoutB = NDExtLayoutPtn）。
 * 启用 transB=true 需新增另一份特化或在本特化内分支 MakeLayoutBL1 的 pattern。
 */

#ifndef MATMUL_BLOCK_MMAD_A_FULL_LOAD_H
#define MATMUL_BLOCK_MMAD_A_FULL_LOAD_H

#ifndef MATMUL_BLOCK_MMAD_H
// 不允许独立 include；必须由 matmul_block_mmad.h 在它的 namespace 闭合后引入，
// 这样两份 SFINAE 特化才在同一 namespace Block 下。
#error "Include matmul_block_mmad.h instead of matmul_block_mmad_a_full_load.h directly."
#endif

namespace Block {

// A 端专用 MTE1↔MTE2 event slot：A_FLAG = 2，与 B 的双缓冲槽 (0/1) 完全错开。
constexpr uint16_t A_FULL_LOAD_A_FLAG = 2;

template <
    class DispatchPolicy_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_>
class BlockMmad<
    DispatchPolicy_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockPolicy<A_FULL_LOAD_MODE>, DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static_assert(!transA, "A_FULL_LOAD_MODE only supports transA=false");
    static_assert(!transB, "A_FULL_LOAD_MODE only supports transB=false");

    // A 端单缓冲（驻留 L1）；B 端继续 double-buffer ping-pong。
    static constexpr uint64_t L1_BUFFER_NUM = 2UL;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1UL;
    static constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    using L0CType = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AType, int8_t>, int32_t, float>;
    static constexpr uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t BLOCK_CUBE = 16UL;
    static constexpr uint64_t BLOCK_CUBE_L0C = 16UL;

    uint64_t m_{0UL};
    uint64_t n_{0UL};
    uint64_t k_{0UL};
    uint64_t kL1Iter_{0UL};
    uint64_t kL1_{0UL};
    uint64_t baseM_{0UL};
    uint64_t baseN_{0UL};
    uint64_t baseK_{0UL};
    // [KEY] abL1LoopCnt_ 跨 operator() 调用累加，初次满 kL1Iter_ 后 A 已就绪。
    uint64_t abL1LoopCnt_{0UL};
    uint64_t l0PingPong_{0UL};
    uint64_t l0cPingPong_{0UL};
    bool enableL0cPingPong_{false};

    // A 行主序 ND → AL1 = NZ；B 行主序 ND → BL1 = ZN。
    using MakeLayoutAL1 = AscendC::Te::FrameLayoutFormat<
        AscendC::Te::NZLayoutPtn, AscendC::Std::Int<BLOCK_CUBE>>;
    using MakeLayoutBL1 = AscendC::Te::FrameLayoutFormat<
        AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<BLOCK_CUBE>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    struct L1Params {
        // kL1 = baseK * stepK，等同 non-full-load 模式下的 L1 K 切片长度。
        uint64_t kL1;
    };

    __aicore__ inline BlockMmad()
    {
        // B 端 ping-pong 事件 slot 0/1：构造时预置 MTE1→MTE2，避免首迭代特判。
        #pragma unroll
        for (uint8_t i = 0; i < L1_BUFFER_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        // A 端专用 slot：构造时预置一次 Set；K 循环内 Wait/Set 对齐；析构 Wait 收尾。
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(A_FULL_LOAD_A_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        #pragma unroll
        for (uint8_t i = 0; i < L1_BUFFER_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        // [PITFALL] 若漏了这条 WaitFlag(A_FLAG)，event 计数器泄漏；下次 kernel 启动时
        // 同 event id 已占用，第二次调用立刻 hang。务必与构造的 SetFlag 一一成对。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(A_FULL_LOAD_A_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(false);
    }

public:
    __aicore__ inline void Init(
        const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params, bool enableL0cPingPong)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        enableL0cPingPong_ = enableL0cPingPong;
        // A 持久区 = Align(m,16) * Align(k,16) * sizeof(A)（字节）。
        // [PITFALL] MakeMemPtr<Location::L1, T>(offset) 接受字节偏移：aL1Total_ /
        // bL1OneBuffer_ 必须显式乘 sizeof()，否则 ping-pong 半区物理重叠。
        uint64_t mAlign = Align(m_, BLOCK_CUBE);
        uint64_t kAlign = Align(k_, BLOCK_CUBE);
        aL1Total_ = mAlign * kAlign * sizeof(AType);
        bL1OneBuffer_ = baseN_ * kL1_ * sizeof(BType);
        l1BufferAOffset_ = 0UL;
        // B 的两个 ping-pong buffer 紧跟 A 之后：B0 = aL1Total_，B1 = B0 + bL1OneBuffer_。
        #pragma unroll
        for (uint64_t bufferId = 0; bufferId < L1_BUFFER_NUM; ++bufferId) {
            l1BufferBOffset_[bufferId] = aL1Total_ + bufferId * bL1OneBuffer_;
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorC gmC, BlockShape singleShape)
    {
        auto curM = Get<IDX_M_TILEIDX>(singleShape);
        auto curN = Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NZLayoutPtn, AscendC::Std::Int<BLOCK_CUBE_L0C>>(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, L0CType>(l0cOffset), layoutL0C);

        // A 的持久 L1 视图：整块 A (curM × k_) 都驻留这里，跨 N-tile 共享。
        auto layoutAL1Full = MakeLayoutAL1{}(curM, k_);
        auto tensorAL1Full = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_),
            layoutAL1Full);

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = iter0 & L1_BUFFER_MASK;
            uint64_t kL1Offset = iter0 * kL1_;
            auto curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;

            // ---- A 端：只在前 kL1Iter_ 次 K-iter（每核第一次调度）搬 GM→L1 ----
            // [PITFALL] A_FLAG 的 Wait/Set 必须严格成对：每次进入守卫 1 Wait + 1 Set，
            // 配上构造时 1 Set、析构时 1 Wait，总计平衡。漏对一次就死锁。
            if (abL1LoopCnt_ < kL1Iter_) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(A_FULL_LOAD_A_FLAG);
                auto copyGM2L1A = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
                auto tensorBlockAL1 = tensorAL1Full.Slice(
                    AscendC::Te::MakeCoord(0, kL1Offset),
                    AscendC::Te::MakeShape(curM, curKL1));
                auto gmBlockA = gmA.Slice(
                    AscendC::Te::MakeCoord(0, kL1Offset),
                    AscendC::Te::MakeShape(curM, curKL1));
                AscendC::Te::Copy(copyGM2L1A, tensorBlockAL1, gmBlockA);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(A_FULL_LOAD_A_FLAG);
            }

            // ---- B 端：每次 K-iter 都搬 GM→L1（双缓冲）----
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]),
                layoutBL1);
            auto gmBlockB = gmB.Slice(
                AscendC::Te::MakeCoord(kL1Offset, 0), AscendC::Te::MakeShape(curKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            // A 当前 K 切片视图（始终从持久 L1.A 区按 kL1Offset 读）。
            auto tensorBlockAL1 = tensorAL1Full.Slice(
                AscendC::Te::MakeCoord(0, kL1Offset),
                AscendC::Te::MakeShape(curM, curKL1));

            // ---- 内层：按 baseK 切块，L1 → L0A/L0B → MMAD 累加 ----
            uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
            for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                auto kL0Offset = iter1 * baseK_;
                auto curKL0 = (kL0Offset + baseK_ > curKL1) ? (curKL1 - kL0Offset) : baseK_;
                uint64_t l0BufId = l0PingPong_ & 0x1;
                uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);

                auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
                auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
                auto layoutAL0 = AscendC::Te::MakeFrameLayout<
                    AscendC::Te::NZLayoutPtn, AscendC::Std::Int<BLOCK_CUBE>>(curM, curKL0);
                auto tensorAL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1Slice = tensorBlockAL1.Slice(
                    AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1Slice);

                auto layoutBL0 = AscendC::Te::MakeFrameLayout<
                    AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<BLOCK_CUBE>>(curKL0, curN);
                auto tensorBL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 = tensorBL1.Slice(
                    AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                uint8_t mmadUnitFlag =
                    (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                // [PITFALL] cmatrixInitVal 是 loop-local：每次 operator() 调用都在 (0,0) 重置
                // L0C 累加器，与 A 是否驻留 L1 无关。iter0/iter1 是函数局部变量，每次调用清零。
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
                AscendC::Te::MmadParams mmadParams{
                    static_cast<uint16_t>(curM),
                    static_cast<uint16_t>(curN),
                    static_cast<uint16_t>(curKL0),
                    mmadUnitFlag,
                    mmadCmatrixInitVal};
                AscendC::Te::Mmad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{}.with(mmadParams),
                    tensorL0C, tensorAL0, tensorBL0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong_++;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }

        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(CopyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    uint64_t aL1Total_ = 0UL;          // A 持久区字节数 = Align(m,16)*Align(k,16)*sizeof(A)
    uint64_t bL1OneBuffer_ = 0UL;      // B 单个 ping/pong 半区字节数 = baseN*kL1*sizeof(B)
    uint64_t l1BufferAOffset_ = 0UL;   // A 在 L1 中的字节偏移，固定 0
    uint64_t l1BufferBOffset_[L1_BUFFER_NUM] = {0UL};
};

} // namespace Block

#endif // MATMUL_BLOCK_MMAD_A_FULL_LOAD_H
