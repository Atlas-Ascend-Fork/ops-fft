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
 * See COPYING (full GPL) and COMMERCIAL-LICENSE (commercial terms) at the repo root.
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
 * 详情见仓库根目录 COPYING（GPL 全文）与 COMMERCIAL-LICENSE（商业条款）。
 * 商业授权联系：yuantang@fudan.edu.cn
 */

#ifndef RFFT1_D_R2C_FFT3_1_KERNEL_H
#define RFFT1_D_R2C_FFT3_1_KERNEL_H

#include "../rfft1_d_r2c_schedule_common.h"

namespace Rfft1DR2CFft31Kernel {

using namespace::AscendC;

using Rfft1DR2CCommonKernel::CMatmulCustom;

template<typename T>
using MulCustom = Rfft1DR2CCommonKernel::TransposeTwiddleMul<T, false>;

using Rfft1DR2CCommonKernel::UnpackCustom;

#ifdef __DAV_C220_VEC__
__aicore__ inline void UnpackFft3_1Tail(
    int32_t batch, int i, int blockN, int warmupIter, UnpackCustom<float> &op,
    GlobalTensor<float> &workGlobal, int calcN, int o, int32_t N, GlobalTensor<float> &workGlobal3,
    GlobalTensor<float> &yGlobal, int halfN, int blockNum, int pairCubeIdx, int calcNum)
{
    int j = batch / 2;
    if (batch / 2 - i >= blockN || i / blockN < warmupIter + 2) {
        op.SetMatrix(workGlobal[(calcN + o) * N], workGlobal3, yGlobal[j * (halfN + 1) * 2], yGlobal[j * (halfN + 1) * 2]);
        op.UnPack2(N, 1, N, N + 2, 1, 0);
        op.SetValueBatch(workGlobal[(calcN + o) * N + halfN], yGlobal[j * (halfN + 1) * 2 + N], 1, N, N + 2);
    } else {
        auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * pairCubeIdx) * N];
        op.SetMatrix(workGlobalTmp[calcNum * N], workGlobal3, yGlobal[j * (halfN + 1) * 2], yGlobal[j * (halfN + 1) * 2]);
        op.UnPack2(N, 1, N, N + 2, 1, 0);
        op.SetValueBatch(workGlobalTmp[calcNum * N + halfN], yGlobal[j * (halfN + 1) * 2 + N], 1, N, N + 2);
    }
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void UnpackFft3_1Output(
    int i, int blockN, int warmupIter, int32_t batch, int blockNum, int pairCubeIdx,
    UnpackCustom<float> &op, GlobalTensor<uint32_t> &gatherGlobal, int blockIdx,
    GlobalTensor<float> &workGlobal, int32_t N, int halfBatchCeil, GlobalTensor<float> &yGlobal, int halfN,
    GlobalTensor<float> &workGlobal3)
{
    CrossCoreWaitFlag(0x0);
    int o = i / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - i);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = pairCubeIdx * averageNum + min(pairCubeIdx, remainderNum);
    int calcNum = averageNum + (pairCubeIdx < remainderNum);
    op.LoadGather(gatherGlobal);
    int startIdxVec = 0, calcNumVec = 0;
    Rfft1DR2CCommonKernel::UnpackScheduledPair(op, i, startIdx, calcNum, blockIdx, batch, blockN,
        warmupIter, blockNum, pairCubeIdx, workGlobal, o, N, halfBatchCeil, yGlobal, halfN,
        startIdxVec, calcNumVec);
    if (batch / 2 - i >= blockN || i / blockN < warmupIter + 2) {
        op.SetValueBatch(workGlobal[(startIdxVec - i + o) * N + halfN], yGlobal[startIdxVec * (halfN + 1) * 2 + N], calcNumVec, N, N + 2);
        op.SetValueBatch(workGlobal[(startIdxVec - i + o) * N + N * halfBatchCeil + halfN], yGlobal[startIdxVec * (halfN + 1) * 2 + (halfN + 1) * 2 * halfBatchCeil + N], calcNumVec, N, N + 2);
    } else {
        auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * pairCubeIdx) * N];
        op.SetValueBatch(workGlobalTmp[(startIdxVec - i - startIdx) * N + halfN], yGlobal[startIdxVec * (halfN + 1) * 2 + N], calcNumVec, N, N + 2);
        op.SetValueBatch(workGlobalTmp[(startIdxVec - i - startIdx) * N + N * halfBatchCeil + halfN], yGlobal[startIdxVec * (halfN + 1) * 2 + (halfN + 1) * 2 * halfBatchCeil + N], calcNumVec, N, N + 2);
    }
    if (batch &1 &&calcN < blockN &&blockIdx == blockNum * 2 - 1) {
        UnpackFft3_1Tail(batch, i, blockN, warmupIter, op, workGlobal, calcN, o, N, workGlobal3, yGlobal, halfN,
            blockNum, pairCubeIdx, calcNum);
    }
    op.FreeGather();
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void WarmupFft3_1Twiddle(
    int blockN, int32_t batch, int i, int blockNum, int pairCubeIdx, int blockIdx,
    MulCustom<float> &opMul, int o, int32_t N, int32_t n1, int32_t n2, int32_t n3,
    GlobalTensor<float> &twiddleGlobal, GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal3,
    GlobalTensor<float> &workGlobal2, int halfBatchCeil)
{
    CrossCoreWaitFlag(0x0);

    int calcN = min(blockN, batch / 2 - i);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = pairCubeIdx * averageNum + min(pairCubeIdx, remainderNum);
    int calcNum = averageNum + (pairCubeIdx < remainderNum);
    for (int j = i + startIdx + (blockIdx &1); j < i + startIdx + calcNum; j += 2) {
        opMul.Process((j - i + o) * N, (j - i + o) * N, n1, n2, n3);
    }

    if (batch &1 &&calcN < blockN &&blockIdx == blockNum * 2 - 1) {
        opMul.SetMatrix(twiddleGlobal, twiddleGlobal[N], workGlobal[(calcN + o) * N], workGlobal3, workGlobal2, workGlobal2[N * halfBatchCeil]);
        opMul.Process(0, (calcN + o) * N, n1, n2, n3);
    }

    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void AdvanceFft3_1Twiddle(
    int i, int blockN, int warmupIter, int32_t batch, int blockNum, int pairCubeIdx, int blockIdx,
    MulCustom<float> &opMul, int32_t N, int32_t n1, int32_t n2, int32_t n3, GlobalTensor<float> &workGlobal,
    GlobalTensor<float> &workGlobal2, GlobalTensor<float> &twiddleGlobal, int halfBatchCeil,
    GlobalTensor<float> &workGlobal3)
{
    CrossCoreWaitFlag(0x0);

    int nextI = i + blockN * warmupIter;
    int o = nextI / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - nextI);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = pairCubeIdx * averageNum + min(pairCubeIdx, remainderNum);
    int calcNum = averageNum + (pairCubeIdx < remainderNum);
    if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
        for (int j = nextI + startIdx + (blockIdx &1); j < nextI + startIdx + calcNum; j += 2) {
            opMul.Process((j - nextI + o) * N, (j - nextI + o) * N, n1, n2, n3);
        }
    } else {
        auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * pairCubeIdx) * N];
        auto workGlobal2Tmp = workGlobal2[o * N + (blockN / blockNum * pairCubeIdx) * N];
        opMul.SetMatrix(twiddleGlobal, twiddleGlobal[N], workGlobalTmp, workGlobalTmp[N * halfBatchCeil], workGlobal2Tmp, workGlobal2Tmp[N * halfBatchCeil]);
        for (int j = nextI + startIdx + (blockIdx &1); j < nextI + startIdx + calcNum; j += 2) {
            int t = j - nextI - startIdx;
            opMul.Process(t * N, t * N, n1, n2, n3);
        }
    }
    if (batch &1 &&calcN < blockN &&blockIdx == blockNum * 2 - 1) {
        if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
            opMul.SetMatrix(twiddleGlobal, twiddleGlobal[N], workGlobal[(calcN + o) * N], workGlobal3, workGlobal2, workGlobal2[N * halfBatchCeil]);
            opMul.Process(0, (calcN + o) * N, n1, n2, n3);
        } else {
            auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * pairCubeIdx) * N];
            auto workGlobal2Tmp = workGlobal2[o * N + (blockN / blockNum * pairCubeIdx) * N];
            opMul.SetMatrix(twiddleGlobal, twiddleGlobal[N], workGlobalTmp[calcNum * N], workGlobal3, workGlobal2Tmp, workGlobal2Tmp[N * halfBatchCeil]);
            opMul.Process(0, calcNum * N, n1, n2, n3);
        }
    }
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
}

#endif

__aicore__ inline void RunFft3_1_r2cVector(Rfft1DR2CCommonKernel::Fft3KernelContext &ctx, TPipe &pipe)
{
    #ifdef __DAV_C220_VEC__
        Rfft1DR2CCommonKernel::Fft3VectorState<false> state(ctx, pipe);
        for (int i = 0; i < ctx.blockN * ctx.warmupIter; i += ctx.blockN) {
            int o = i / ctx.blockN % (ctx.warmupIter + 2) * ctx.blockN;
            {
                WarmupFft3_1Twiddle(ctx.blockN, ctx.tiling.batch, i, ctx.blockNum, ctx.pairCubeIdx, ctx.blockIdx,
                    state.opMul, o, ctx.tiling.N, ctx.tiling.n1, ctx.tiling.n2, ctx.tiling.n3,
                    ctx.twiddleGlobal, ctx.workGlobal, ctx.workGlobal3, ctx.workGlobal2, ctx.halfBatchCeil);
            }
        }

        for (int i = 0; i < ctx.halfBatchCeil; i += ctx.blockN) {
            {
                UnpackFft3_1Output(i, ctx.blockN, ctx.warmupIter, ctx.tiling.batch, ctx.blockNum, ctx.pairCubeIdx,
                    state.op, ctx.gatherGlobal, ctx.blockIdx,
                    ctx.workGlobal, ctx.tiling.N, ctx.halfBatchCeil, ctx.yGlobal, ctx.halfN, ctx.workGlobal3);
            }

            if (i + ctx.blockN * ctx.warmupIter < ctx.halfBatchCeil) {
                AdvanceFft3_1Twiddle(i, ctx.blockN, ctx.warmupIter, ctx.tiling.batch, ctx.blockNum, ctx.pairCubeIdx,
                    ctx.blockIdx, state.opMul, ctx.tiling.N, ctx.tiling.n1,
                    ctx.tiling.n2, ctx.tiling.n3, ctx.workGlobal, ctx.workGlobal2, ctx.twiddleGlobal, ctx.halfBatchCeil, ctx.workGlobal3);
            }
        }

    #endif
}

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft3_1Dft3Tail(
    int32_t batch, int i, int blockN, int warmupIter, int _n2, int _n1, int32_t n3,
    CMatmulCustom<float> &opmm, GlobalTensor<float> &workGlobal2, int calcN, int o, int32_t N,
    int halfBatchCeil, GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal3, int blockNum,
    int blockIdx, int calcNum)
{
    if (batch / 2 - i >= blockN || i / blockN < warmupIter + 2) {
        int j = batch / 2;
        for (int i2 = 0; i2 < _n2; ) {
            int compBatch = min(_n2 - i2, 8192 / _n1 / n3);
            compBatch = max(compBatch, 1);
            opmm.SetMatrix(workGlobal2[(calcN + o) * N + i2 * _n1 * n3], workGlobal2[(calcN + o) * N + i2 * _n1 * n3 + N * halfBatchCeil], workGlobal[(calcN + o) * N + i2 * _n1], workGlobal3[i2 * _n1], n3, n3, _n1 * _n2, _n1 * n3, _n1);
            opmm.ProcessWithTranspose(0, 0, 0, n3, _n1, n3, compBatch);
            i2 += compBatch;
        }
    } else {
        auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
        auto workGlobal2Tmp = workGlobal2[o * N + (blockN / blockNum * blockIdx) * N];
        for (int i2 = 0; i2 < _n2; ) {
            int compBatch = min(_n2 - i2, 8192 / _n1 / n3);
            compBatch = max(compBatch, 1);
            opmm.SetMatrix(workGlobal2Tmp[calcNum * N + i2 * _n1 * n3], workGlobal2Tmp[calcNum * N + i2 * _n1 * n3 + N * halfBatchCeil], workGlobalTmp[calcNum * N + i2 * _n1], workGlobal3[i2 * _n1], n3, n3, _n1 * _n2, _n1 * n3, _n1);
            opmm.ProcessWithTranspose(0, 0, 0, n3, _n1, n3, compBatch);
            i2 += compBatch;
        }
    }
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft3_1Dft3(
    CMatmulCustom<float> &opmm, GlobalTensor<float> &twiddleGlobal, int32_t n1, int32_t n2, int32_t n3,
    int i, int startIdx, int calcNum, int32_t batch, int blockN, int warmupIter,
    GlobalTensor<float> &workGlobal2, int o, int32_t N, int halfBatchCeil, GlobalTensor<float> &workGlobal,
    int blockNum, int blockIdx, int calcN, GlobalTensor<float> &workGlobal3)
{
    opmm.CopyInAReside(twiddleGlobal[n1 * n1 * 3 + n1 * n2 * n2 * 3], n3, n3);
    int _n1 = 128, _n2 = n1 * n2 / 128;
    for (int j = i + startIdx; j < i + startIdx + calcNum; ++j) {
        for (int i2 = 0; i2 < _n2; ) {
            int compBatch = min(_n2 - i2, 8192 / _n1 / n3);
            compBatch = max(compBatch, 1);
            if (batch / 2 - i >= blockN || i / blockN < warmupIter + 2) {
                opmm.SetMatrix(workGlobal2[(j - i + o) * N + i2 * _n1 * n3], workGlobal2[(j - i + o) * N + i2 * _n1 * n3 + N * halfBatchCeil], workGlobal[(j - i + o) * N + i2 * _n1], workGlobal[(j - i + o) * N + i2 * _n1 + N * halfBatchCeil], n3, n3, _n1 * _n2, _n1 * n3, _n1);
            } else {
                auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
                auto workGlobal2Tmp = workGlobal2[o * N + (blockN / blockNum * blockIdx) * N];
                int t = j - i - startIdx;
                opmm.SetMatrix(workGlobal2Tmp[t * N + i2 * _n1 * n3], workGlobal2Tmp[t * N + i2 * _n1 * n3 + N * halfBatchCeil], workGlobalTmp[t * N + i2 * _n1], workGlobalTmp[t * N + i2 * _n1 + N * halfBatchCeil], n3, n3, _n1 * _n2, _n1 * n3, _n1);
            }
            opmm.ProcessWithTranspose(0, 0, 0, n3, _n1, n3, compBatch);
            i2 += compBatch;
        }
    }

    if (batch &1 &&calcN < blockN &&blockIdx == blockNum - 1) {
        ComputeFft3_1Dft3Tail(batch, i, blockN, warmupIter, _n2, _n1, n3, opmm, workGlobal2, calcN, o, N,
            halfBatchCeil, workGlobal, workGlobal3, blockNum, blockIdx, calcNum);
    }

    opmm.FreeAReside();
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void AdvanceFft3_1FirstStages(
    int i, int blockN, int warmupIter, int32_t batch, int blockNum, int blockIdx,
    Rfft1DR2CCommonKernel::Fft3StageContext &stageContext)
{
    int nextI = i + blockN * warmupIter;
    int o = nextI / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - nextI);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);

    Rfft1DR2CCommonKernel::ProcessFft3FirstStages<false>(stageContext, nextI, o, calcN, startIdx, calcNum);
}

#endif

__aicore__ inline void RunFft3_1_r2cCube(Rfft1DR2CCommonKernel::Fft3KernelContext &ctx, TPipe &pipe)
{
    #ifdef  __DAV_C220_CUBE__
        Rfft1DR2CCommonKernel::Fft3CubeState state(ctx, pipe);
        Rfft1DR2CCommonKernel::WarmupFft3Stages<false>(state.stageContext);

        for (int i = 0; i < ctx.halfBatchCeil; i += ctx.blockN) {
            int o = i / ctx.blockN % (ctx.warmupIter + 2) * ctx.blockN;
            int calcN = min(ctx.blockN, ctx.tiling.batch / 2 - i);
            int averageNum = calcN / ctx.blockNum;
            int remainderNum = calcN % ctx.blockNum;
            int startIdx = ctx.blockIdx * averageNum + min(ctx.blockIdx, remainderNum);
            int calcNum = averageNum + (ctx.blockIdx < remainderNum);

            CrossCoreWaitFlag(0x1);

            // dft 3
            {
                ComputeFft3_1Dft3(state.opmm, ctx.twiddleGlobal, ctx.tiling.n1, ctx.tiling.n2, ctx.tiling.n3, i,
                    startIdx, calcNum, ctx.tiling.batch, ctx.blockN, ctx.warmupIter,
                    ctx.workGlobal2, o, ctx.tiling.N, ctx.halfBatchCeil, ctx.workGlobal, ctx.blockNum, ctx.blockIdx, calcN, ctx.workGlobal3);
            }

            CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);

            if (i + ctx.blockN * ctx.warmupIter < ctx.halfBatchCeil) {
                AdvanceFft3_1FirstStages(i, ctx.blockN, ctx.warmupIter, ctx.tiling.batch, ctx.blockNum, ctx.blockIdx,
                    state.stageContext);
            }
        }

        #endif
}

struct Fft3_1Dispatch {
    __aicore__ inline static void Run(Rfft1DR2CCommonKernel::Fft3KernelContext &ctx, TPipe &pipe) {
        RunFft3_1_r2cVector(ctx, pipe);

        RunFft3_1_r2cCube(ctx, pipe);
    }
};

extern "C" __global__ __aicore__ void fft3_1_r2c(
    __gm__ uint8_t * __restrict__ X_gm,
    __gm__ uint8_t * __restrict__ Y_gm,
    __gm__ uint8_t * __restrict__ work_gm,
    __gm__ uint8_t * __restrict__ twiddle_gm,
    __gm__ uint8_t * __restrict__ gather_gm,
    __gm__ uint8_t * __restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    Rfft1DR2CCommonKernel::RunFft3Kernel<Fft3_1Dispatch>(
        X_gm, Y_gm, work_gm, twiddle_gm, gather_gm, tiling_para_gm, pipe);
}

}

#endif
