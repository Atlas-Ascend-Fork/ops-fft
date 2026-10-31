/*
 * PACO FFT — An I/O-Efficient Parallel FFT for Heterogeneous Architectures via a Single Global Exchange and Implementation
 *
 * Copyright (c) 2026 Shina Guo <fs20091003@163.com>, Yuan Tang <yuantang@fudan.edu.cn>
 * All rights reserved.
 *
 * This software is dual-licensed:
 *   1) Community license: use, modify, and redistribute freely under the
 *      GNU General Public License v3 or later (GPL-3.0-or-later);
 *   2) Commercial license: for closed-source or commercial use / redistribution,
 *      contact the copyright holder for a separate license.
 * You may choose either. Default terms are GPL-3.0-or-later.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you may redistribute it and/or modify it under
 * the terms of the GPL. It is distributed "as is", without warranty.
 * See COPYING-PACO (full GPL) and LICENSE-PACO (commercial terms) at the repo root.
 * Commercial licensing contact: yuantang@fudan.edu.cn
 *
 * -------------------------------------------------------------------------
 * 中文版 (Chinese):
 *
 * PACO FFT — An I/O-Efficient Parallel FFT for Heterogeneous Architectures via a Single Global Exchange and Implementation
 *
 * 本软件采用双重许可（dual licensing）：
 *   1) 社区许可：依据 GNU 通用公共许可证第 3 版或（你选择的）更新版本
 *      （GPL-3.0-or-later）自由使用、修改与再分发；
 *   2) 商业许可：闭源商用 / 再分发请与版权人联系获取单独授权。
 * 二者任选其一。默认条款以 GPL-3.0-or-later 为准。
 *
 * 本程序是自由软件，你可依 GPL 条款重新发布或修改；基于“原样”提供，无担保。
 * 详情见仓库根目录 COPYING-PACO（GPL 全文）与 LICENSE-PACO（商业条款）。
 * 商业授权联系：yuantang@fudan.edu.cn
 */

#ifndef RFFT1_D_R2C_SCHEDULE_COMMON_H
#define RFFT1_D_R2C_SCHEDULE_COMMON_H

#include "kernel/fft_common_kernel.h"
#include <stdint.h>

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#include "rfft1_d_r2c_unpack.h"
#include "rfft1_d_r2c_kernel_common.h"

namespace Rfft1DR2CCommonKernel {

struct VectorQueues {
    TQue<QuePosition::VECIN, 8> input;
    TQue<QuePosition::VECOUT, 2> output;

    __aicore__ inline explicit VectorQueues(TPipe &pipe) {
        pipe.InitBuffer(input, 8, 4096 * sizeof(float));
        pipe.InitBuffer(output, 2, 4096 * 2 * sizeof(float));
    }

    template<typename Twiddle>
    __aicore__ inline void Attach(Twiddle &opMul, UnpackCustom<float> &op) {
        opMul.Init2(input, output);
        op.Init2(input, output);
    }

    template<bool Distributed>
    __aicore__ inline void AttachTranspose(
        TransposeTwiddleMul<float, Distributed> &opMul, UnpackCustom<float> &op,
        GlobalTensor<float> &twiddleGlobal, GlobalTensor<float> &workGlobal,
        GlobalTensor<float> &workGlobal2, int32_t N, int halfBatchCeil)
    {
        Attach(opMul, op);
        opMul.SetMatrix(twiddleGlobal, twiddleGlobal[N], workGlobal, workGlobal[N * halfBatchCeil],
            workGlobal2, workGlobal2[N * halfBatchCeil]);
    }
};

__aicore__ inline void UnpackScheduledPair(
    UnpackCustom<float> &op, int i, int startIdx, int calcNum, int blockIdx, int batch,
    int blockN, int warmupIter, int blockNum, int pairCubeIdx, GlobalTensor<float> &workGlobal,
    int o, int32_t N, int halfBatchCeil, GlobalTensor<float> &yGlobal, int halfN,
    int &startIdxVec, int &calcNumVec)
{
    if (blockN == 0 || blockNum == 0) return;
    startIdxVec = i + startIdx + (blockIdx & 1) * (calcNum / 2);
    calcNumVec = calcNum / 2 + (blockIdx & 1) * (calcNum & 1);
    auto unpackSrc = workGlobal[(startIdxVec - i + o) * N];
    if (!(batch / 2 - i >= blockN || i / blockN < warmupIter + 2)) {
        auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * pairCubeIdx) * N];
        unpackSrc = workGlobalTmp[(startIdxVec - i - startIdx) * N];
    }
    UnpackPairRange(op, unpackSrc, unpackSrc[N * halfBatchCeil],
        yGlobal[startIdxVec * (halfN + 1) * 2],
        yGlobal[startIdxVec * (halfN + 1) * 2 + (halfN + 1) * 2 * halfBatchCeil], N, calcNumVec);
}

__aicore__ inline Fft3StageContext InitializeFft3Stages(
    CMatmulCustom<float> &opmm, TPipe &pipe, GlobalTensor<float> &twiddleGlobal,
    GlobalTensor<float> &xGlobal, GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal3,
    int32_t N, int32_t batch, int32_t n1, int32_t n2, int32_t n3, int halfBatchCeil,
    int blockNum, int blockIdx, int blockN, int warmupIter)
{
    twiddleGlobal = twiddleGlobal[N * 2];
    opmm.Init(&pipe);
    return {opmm, xGlobal, workGlobal, workGlobal3, twiddleGlobal,
        N, batch, n1, n2, n3, halfBatchCeil, blockNum, blockIdx, blockN, warmupIter};
}

template<bool Distributed>
__aicore__ inline void WarmupFft3Stages(Fft3StageContext &ctx) {
    int blockN = ctx.blockN, blockNum = ctx.blockNum;
    if (blockN == 0 || blockNum == 0 || ctx.warmupIter + 2 == 0) return;
    for (int i = 0; i < blockN * ctx.warmupIter; i += blockN) {
        int o = i / blockN % (ctx.warmupIter + 2) * blockN;
        int calcN = min(blockN, ctx.batch / 2 - i);
        if constexpr (Distributed) {
            ProcessFft3FirstStages<true>(ctx, i, o, calcN);
        } else {
            int averageNum = calcN / blockNum;
            int remainderNum = calcN % blockNum;
            int startIdx = ctx.blockIdx * averageNum + min(ctx.blockIdx, remainderNum);
            int calcNum = averageNum + (ctx.blockIdx < remainderNum);
            ProcessFft3FirstStages<false>(ctx, i, o, calcN, startIdx, calcNum);
        }
    }
}

struct Fft3KernelContext {
    KernelTiling<3> tiling;
    int blockN = 0, blockNum = 0, blockIdx = 0, pairCubeIdx = 0;
    GlobalTensor<float> xGlobal, yGlobal, workGlobal, workGlobal2, workGlobal3, twiddleGlobal;
    GlobalTensor<uint32_t> gatherGlobal;
    int warmupIter = 2, halfN = 0, halfBatchCeil = 0;

    __aicore__ inline Fft3KernelContext(
        __gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *work,
        __gm__ uint8_t *twiddle, __gm__ uint8_t *gather, __gm__ uint8_t *tilingGm)
        : tiling(tilingGm)
    {
        int32_t N = tiling.N;
        if (N == 0) return;
        blockN = 2621440 / N;
        blockNum = GetBlockNum();
        blockIdx = GetBlockIdx();
        pairCubeIdx = blockIdx >> 1;
        xGlobal = BindGlobal<float>(x);
        yGlobal = BindGlobal<float>(y);
        workGlobal2 = BindGlobal<float>(work);
        twiddleGlobal = BindGlobal<float>(twiddle);
        gatherGlobal = BindGlobal<uint32_t>(gather);
    }

    __aicore__ inline void InitSchedule() {
        int32_t N = tiling.N, batch = tiling.batch;
        if (blockN * 2 >= batch) warmupIter = 1;
        halfN = N / 2;
        halfBatchCeil = (batch + 1) / 2;
        workGlobal = xGlobal;
        workGlobal3 = workGlobal2[(batch + 1) * N];
    }
};

#ifdef __DAV_C220_VEC__
template<bool Distributed>
struct Fft3VectorState {
    VectorQueues queues;
    TransposeTwiddleMul<float, Distributed> opMul;
    UnpackCustom<float> op;

    __aicore__ inline Fft3VectorState(Fft3KernelContext &ctx, TPipe &pipe) : queues(pipe) {
        queues.AttachTranspose(opMul, op, ctx.twiddleGlobal, ctx.workGlobal, ctx.workGlobal2,
            ctx.tiling.N, ctx.halfBatchCeil);
    }
};
#endif

#ifdef __DAV_C220_CUBE__
struct Fft3CubeState {
    CMatmulCustom<float> opmm;
    Fft3StageContext stageContext;

    __aicore__ inline Fft3CubeState(Fft3KernelContext &ctx, TPipe &pipe)
        : stageContext(InitializeFft3Stages(opmm, pipe, ctx.twiddleGlobal, ctx.xGlobal,
            ctx.workGlobal, ctx.workGlobal3, ctx.tiling.N, ctx.tiling.batch, ctx.tiling.n1,
            ctx.tiling.n2, ctx.tiling.n3, ctx.halfBatchCeil, ctx.blockNum, ctx.blockIdx,
            ctx.blockN, ctx.warmupIter)) {}
};
#endif

template<typename Dispatch>
__aicore__ inline void RunFft3Kernel(
    __gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *work, __gm__ uint8_t *twiddle,
    __gm__ uint8_t *gather, __gm__ uint8_t *tilingGm, TPipe &pipe)
{
    Fft3KernelContext ctx(x, y, work, twiddle, gather, tilingGm);
    // tiling 经 KernelTiling 按 uint32 解读，N/batch 非法（负值/0）时静默返回，
    // 防止派生量 blockN、halfBatchCeil 等出现负值循环边界（issue #219）。
    if (ctx.tiling.N <= 0 || ctx.tiling.batch <= 0 || ctx.blockN == 0) return;
    ctx.InitSchedule();
    Dispatch::Run(ctx, pipe);
}

__aicore__ inline void ProcessFft3DistributedTwiddle(
    TransposeTwiddleMul<float, true> &opMul, int o, int32_t N, int32_t n1, int32_t n2, int32_t n3,
    int calcN, int blockNum, int blockIdx, int32_t batch, int blockN,
    GlobalTensor<float> &twiddleGlobal, GlobalTensor<float> &workGlobal,
    GlobalTensor<float> &workGlobal3, GlobalTensor<float> &workGlobal2, int halfBatchCeil)
{
    opMul.Process(o * N, o * N, n1, n2, n3, calcN, N, blockNum * 2, blockIdx);
    if (batch & 1 && calcN < blockN) {
        opMul.SetMatrix(twiddleGlobal, twiddleGlobal[N], workGlobal[(calcN + o) * N], workGlobal3,
            workGlobal2, workGlobal2[N * halfBatchCeil]);
        opMul.Process(0, (calcN + o) * N, n1, n2, n3, 1, N, blockNum * 2, blockIdx);
    }
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
    CrossCoreWaitFlag(0x3);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
}

struct Dft2Schedule {
    int n1, n2, l2Step, stepN, blockNum, blockIdx;
};

struct Dft2Layout4 {
    static constexpr bool wideOffsets = false;
    GlobalTensor<float> input, output, tailOutput;
    int offset, batch, N, n1, row, col;

    __aicore__ inline void SetMatrix(CMatmulCustom<float> &opmm, int i, int j, int n2, int stepN) {
        if (offset < batch - 1) {
            opmm.SetMatrix(input[i * n2 * stepN + j * row * stepN],
                input[i * n2 * stepN + j * row * stepN + N], output[i * col + j * stepN],
                output[i * col + j * stepN + N], n2, stepN, n1 * col, row * stepN, stepN);
        } else {
            opmm.SetMatrix(input[i * n2 * stepN + j * row * stepN],
                input[i * n2 * stepN + j * row * stepN + N], output[i * col + j * stepN],
                tailOutput[i * col + j * stepN], n2, stepN, n1 * col, row * stepN, stepN);
        }
    }
};

struct Dft2Layout5 {
    static constexpr bool wideOffsets = true;
    GlobalTensor<float> input, output;
    uint64_t row64, col64, halfN;
    int n1, col;

    __aicore__ inline void SetMatrix(CMatmulCustom<float> &opmm, int i, int j, int n2, int stepN) {
        uint64_t inputOffset = static_cast<uint64_t>(i) * static_cast<uint64_t>(n2) * static_cast<uint64_t>(stepN)
                             + static_cast<uint64_t>(j) * row64 * static_cast<uint64_t>(stepN);
        uint64_t outputOffset = static_cast<uint64_t>(i) * col64
                              + static_cast<uint64_t>(j) * static_cast<uint64_t>(stepN);
        opmm.SetMatrix(input[inputOffset], input[inputOffset + halfN], output[outputOffset],
            output[outputOffset + halfN], n2, stepN, n1 * col, row64 * static_cast<uint64_t>(stepN), stepN);
    }
};

template<typename Layout>
__aicore__ inline void ProcessDft2Tiles(CMatmulCustom<float> &opmm, GlobalTensor<float> &twiddle,
                                       const Dft2Schedule &schedule, Layout &layout)
{
    int n1 = schedule.n1, n2 = schedule.n2, l2Step = schedule.l2Step;
    int stepN = schedule.stepN, blockNum = schedule.blockNum, blockIdx = schedule.blockIdx;
    if (blockNum == 0 || l2Step == 0 || n2 == 0 || stepN == 0) return;
    int averageNum = l2Step * n1 / blockNum;
    int remainderNum = l2Step * n1 % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    int idx = 0;
    for (int i = startIdx / l2Step; idx < calcNum; ++i) {
        if constexpr (Layout::wideOffsets) {
            opmm.CopyInAReside(twiddle[static_cast<uint64_t>(i) * static_cast<uint64_t>(n2)
                * static_cast<uint64_t>(n2) * 3ULL], n2, n2);
        } else {
            opmm.CopyInAReside(twiddle[i * n2 * n2 * 3], n2, n2);
        }
        for (int j = !idx ? startIdx % l2Step : 0; j < l2Step; ) {
            int compBatch = min(l2Step - j, calcNum - idx);
            compBatch = min(compBatch, 8192 / n2 / stepN);
            if constexpr (Layout::wideOffsets) compBatch = max(1, compBatch);
            layout.SetMatrix(opmm, i, j, n2, stepN);
            opmm.Process(0, 0, 0, n2, stepN, n2, compBatch);
            idx += compBatch;
            j += compBatch;
            if (idx == calcNum) break;
        }
        opmm.FreeAReside();
    }
}

} // namespace Rfft1DR2CCommonKernel

#endif // RFFT1_D_R2C_SCHEDULE_COMMON_H
