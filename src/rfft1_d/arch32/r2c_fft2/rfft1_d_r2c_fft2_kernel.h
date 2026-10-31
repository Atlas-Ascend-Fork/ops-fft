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

#ifndef RFFT1_D_R2C_FFT2_KERNEL_H
#define RFFT1_D_R2C_FFT2_KERNEL_H

#include "../rfft1_d_r2c_schedule_common.h"

namespace Rfft1DR2CFft2Kernel {

using namespace::AscendC;

using Rfft1DR2CCommonKernel::CMatmulCustom;

template<typename T>
using MulCustom = Rfft1DR2CCommonKernel::ResidentTwiddleMul<T>;

using Rfft1DR2CCommonKernel::UnpackCustom;

#ifdef __DAV_C220_VEC__
__aicore__ inline void UnpackFft2Tail(
    int32_t batch, int i, int blockN, int warmupIter, UnpackCustom<float> &op,
    GlobalTensor<float> &workGlobal, int calcN, int o, int32_t N, GlobalTensor<float> &workGlobal2,
    GlobalTensor<float> &yGlobal, int halfN, int blockNum, int pairCubeIdx, int calcNum)
{
    int j = batch / 2;
    if (batch / 2 - i >= blockN || i / blockN < warmupIter + 2) {
        op.SetMatrix(workGlobal[(calcN + o) * N], workGlobal2, yGlobal[j * (halfN + 1) * 2], yGlobal[j * (halfN + 1) * 2]);
        op.UnPack2(N, 1, N, N + 2, 1, 0);
        op.SetValueBatch(workGlobal[(calcN + o) * N + halfN], yGlobal[j * (halfN + 1) * 2 + N], 1, N, N + 2);
    } else {
        auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * pairCubeIdx) * N];
        op.SetMatrix(workGlobalTmp[calcNum * N], workGlobal2, yGlobal[j * (halfN + 1) * 2], yGlobal[j * (halfN + 1) * 2]);
        op.UnPack2(N, 1, N, N + 2, 1, 0);
        op.SetValueBatch(workGlobalTmp[calcNum * N + halfN], yGlobal[j * (halfN + 1) * 2 + N], 1, N, N + 2);
    }
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void UnpackFft2Output(
    UnpackCustom<float> &op, GlobalTensor<uint32_t> &gatherGlobal, int i, int blockN, int warmupIter,
    int32_t batch, int blockNum, int pairCubeIdx, int blockIdx, GlobalTensor<float> &workGlobal, int32_t N,
    int halfBatchCeil, GlobalTensor<float> &yGlobal, int halfN, GlobalTensor<float> &workGlobal2)
{
    op.LoadGather(gatherGlobal);
    CrossCoreWaitFlag(0x0);
    int o = i / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - i);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = pairCubeIdx * averageNum + min(pairCubeIdx, remainderNum);
    int calcNum = averageNum + (pairCubeIdx < remainderNum);
    int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    int startIdxVec = 0, calcNumVec = 0;
    Rfft1DR2CCommonKernel::UnpackScheduledPair(op, i, startIdx, calcNum, blockIdx, batch, blockN,
        warmupIter, blockNum, pairCubeIdx, workGlobal, o, N, halfBatchCeil, yGlobal, halfN,
        startIdxVec, calcNumVec);
    if (calcNumVec > 0) {
        if (batch / 2 - i >= blockN || i / blockN < warmupIter + 2) {
            op.SetValueBatch(workGlobal[(startIdxVec - i + o) * N + halfN], yGlobal[startIdxVec * (halfN + 1) * 2 + N], calcNumVec, N, N + 2);
            op.SetValueBatch(workGlobal[(startIdxVec - i + o) * N + N * halfBatchCeil + halfN], yGlobal[startIdxVec * (halfN + 1) * 2 + (halfN + 1) * 2 * halfBatchCeil + N], calcNumVec, N, N + 2);
        } else {
            auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * pairCubeIdx) * N];
            op.SetValueBatch(workGlobalTmp[(startIdxVec - i - startIdx) * N + halfN], yGlobal[startIdxVec * (halfN + 1) * 2 + N], calcNumVec, N, N + 2);
            op.SetValueBatch(workGlobalTmp[(startIdxVec - i - startIdx) * N + N * halfBatchCeil + halfN], yGlobal[startIdxVec * (halfN + 1) * 2 + (halfN + 1) * 2 * halfBatchCeil + N], calcNumVec, N, N + 2);
        }
    }
    if (batch &1 &&calcN < blockN &&blockIdx == blockNum * 2 - 1) {
        UnpackFft2Tail(batch, i, blockN, warmupIter, op, workGlobal, calcN, o, N, workGlobal2, yGlobal, halfN,
            blockNum, pairCubeIdx, calcNum);
    }
    op.FreeGather();
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void WarmupFft2Twiddle(
    int32_t N, MulCustom<float> &opMul, GlobalTensor<float> &twiddleGlobal, int i, int blockN,
    int warmupIter, int32_t batch, int blockNum, int pairCubeIdx, int blockIdx,
    GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal2)
{
    if (N <= 4096) opMul.LoadTwiddle(twiddleGlobal, twiddleGlobal[N], N);
    CrossCoreWaitFlag(0x0);

    int o = i / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - i);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = pairCubeIdx * averageNum + min(pairCubeIdx, remainderNum);
    int calcNum = averageNum + (pairCubeIdx < remainderNum);
    for (int offset = 0; offset < N; offset += 4096) {
        int calcLen = min(4096, (int)N - offset);
        if (N > 4096) opMul.LoadTwiddle(twiddleGlobal[offset], twiddleGlobal[offset + N], calcLen);
        for (int j = i + startIdx + (blockIdx &1); j < i + startIdx + calcNum; j += 2) {
            opMul.Process((j - i + o) * N + offset, (j - i + o) * N + offset, calcLen);
        }
        if (N > 4096) opMul.FreeTwiddle();
    }
    if (batch &1 &&calcN < blockN &&blockIdx == blockNum * 2 - 1) {
        opMul.SetMatrix(workGlobal, workGlobal2, workGlobal, workGlobal2);
        for (int offset = 0; offset < N; offset += 4096) {
            int calcLen = min(4096, (int)N - offset);
            if (N > 4096) opMul.LoadTwiddle(twiddleGlobal[offset], twiddleGlobal[offset + N], calcLen);
            opMul.Process((calcN + o) * N + offset, offset, calcLen);
            if (N > 4096) opMul.FreeTwiddle();
        }
    }
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
    if (N <= 4096) opMul.FreeTwiddle();
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void AdvanceFft2TwiddleTail(
    int32_t batch, int nextI, int blockN, int warmupIter, MulCustom<float> &opMul,
    GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal2, int32_t N,
    GlobalTensor<float> &twiddleGlobal, int calcN, int o, int blockNum, int pairCubeIdx, int calcNum)
{
    if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
        opMul.SetMatrix(workGlobal, workGlobal2, workGlobal, workGlobal2);
        for (int offset = 0; offset < N; offset += 4096) {
            int calcLen = min(4096, (int)N - offset);
            if (N > 4096) opMul.LoadTwiddle(twiddleGlobal[offset], twiddleGlobal[offset + N], calcLen);
            opMul.Process((calcN + o) * N + offset, offset, calcLen);
            if (N > 4096) opMul.FreeTwiddle();
        }
    } else {
        auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * pairCubeIdx) * N];
        opMul.SetMatrix(workGlobalTmp, workGlobal2, workGlobalTmp, workGlobal2);
        int j = batch / 2;
        for (int offset = 0; offset < N; offset += 4096) {
            int calcLen = min(4096, (int)N - offset);
            if (N > 4096) opMul.LoadTwiddle(twiddleGlobal[offset], twiddleGlobal[offset + N], calcLen);
            opMul.Process(calcNum * N + offset, offset, calcLen);
            if (N > 4096) opMul.FreeTwiddle();
        }
    }
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void AdvanceFft2Twiddle(
    int32_t N, MulCustom<float> &opMul, GlobalTensor<float> &twiddleGlobal, int i, int blockN,
    int warmupIter, int32_t batch, int blockNum, int pairCubeIdx, GlobalTensor<float> &workGlobal,
    int halfBatchCeil, int blockIdx, GlobalTensor<float> &workGlobal2)
{
    if (N <= 4096) opMul.LoadTwiddle(twiddleGlobal, twiddleGlobal[N], N);
    CrossCoreWaitFlag(0x0);
    int nextI = i + blockN * warmupIter;
    int o = nextI / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - nextI);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = pairCubeIdx * averageNum + min(pairCubeIdx, remainderNum);
    int calcNum = averageNum + (pairCubeIdx < remainderNum);
    if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
    } else {
        auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * pairCubeIdx) * N];
        opMul.SetMatrix(workGlobalTmp, workGlobalTmp[N * halfBatchCeil], workGlobalTmp, workGlobalTmp[N * halfBatchCeil]);
    }
    for (int offset = 0; offset < N; offset += 4096) {
        int calcLen = min(4096, (int)N - offset);
        if (N > 4096) opMul.LoadTwiddle(twiddleGlobal[offset], twiddleGlobal[offset + N], calcLen);
        for (int j = nextI + startIdx + (blockIdx &1); j < nextI + startIdx + calcNum; j += 2) {
            if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
                opMul.Process((j - nextI + o) * N + offset, (j - nextI + o) * N + offset, calcLen);
            } else {
                int t = j - nextI - startIdx;
                opMul.Process(t * N + offset, t * N + offset, calcLen);
            }
        }
        if (N > 4096) opMul.FreeTwiddle();
    }
    if (batch &1 &&calcN < blockN &&blockIdx == blockNum * 2 - 1) {
        AdvanceFft2TwiddleTail(batch, nextI, blockN, warmupIter, opMul, workGlobal, workGlobal2, N, twiddleGlobal,
            calcN, o, blockNum, pairCubeIdx, calcNum);
    }
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
    if (N <= 4096) opMul.FreeTwiddle();
}

#endif

__aicore__ inline void RunFft2_r2cVector(
    int32_t N, int32_t batch, int blockN, int blockNum, int blockIdx, int pairCubeIdx,
    GlobalTensor<float> &yGlobal, GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal2,
    GlobalTensor<float> &twiddleGlobal, GlobalTensor<uint32_t> &gatherGlobal, TPipe &pipe, int warmupIter,
    int halfN, int halfBatchCeil)
{
    #ifdef __DAV_C220_VEC__
        Rfft1DR2CCommonKernel::VectorQueues queues(pipe);
        MulCustom<float> opMul;
        UnpackCustom<float> op;
        queues.Attach(opMul, op);

        opMul.SetMatrix(workGlobal, workGlobal[N * halfBatchCeil], workGlobal, workGlobal[N * halfBatchCeil]);
        for (int i = 0; i < blockN * warmupIter; i += blockN) {
            {
                WarmupFft2Twiddle(N, opMul, twiddleGlobal, i, blockN, warmupIter, batch, blockNum, pairCubeIdx,
                    blockIdx, workGlobal, workGlobal2);
            }
        }

        for (int i = 0; i < halfBatchCeil; i += blockN) {
            {
                UnpackFft2Output(op, gatherGlobal, i, blockN, warmupIter, batch, blockNum, pairCubeIdx, blockIdx,
                    workGlobal, N, halfBatchCeil, yGlobal, halfN, workGlobal2);
            }


            if (i + blockN * warmupIter < halfBatchCeil) {
                AdvanceFft2Twiddle(N, opMul, twiddleGlobal, i, blockN, warmupIter, batch, blockNum, pairCubeIdx,
                    workGlobal, halfBatchCeil, blockIdx, workGlobal2);
            }
        }

    #endif
}

#ifdef __DAV_C220_CUBE__
__aicore__ inline void WarmupFft2Dft1(
    int i, int blockN, int warmupIter, int32_t batch, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobal, int32_t n1, int blockNum, int blockIdx, int32_t N,
    GlobalTensor<float> &xGlobal, int halfBatchCeil, GlobalTensor<float> &workGlobal, int32_t n2,
    GlobalTensor<float> &workGlobal2)
{
    int o = i / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - i);
    opmm.CopyInAReside(twiddleGlobal, n1, n1);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    for (int j = i + startIdx; j < i + startIdx + calcNum; ) {
        int compBatch = min(i + startIdx + calcNum - j, 8192 / (int)N);
        compBatch = max(compBatch, 1);
        opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N + N * halfBatchCeil], workGlobal[(j - i + o) * N], workGlobal[(j - i + o) * N + N * halfBatchCeil], n1, n2, n2, N, N);
        opmm.Process(0, 0, 0, n1, n2, n1, compBatch);
        j += compBatch;
    }
    if (batch &1 &&calcN < blockN &&blockIdx == blockNum - 1) {
        int j = batch / 2;
        opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N], workGlobal[(calcN + o) * N], workGlobal2, n1, n2, n2, N, N);
        opmm.Process(0, 0, 0, n1, n2, n1);
    }
    opmm.FreeAReside();
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft2Dft2(
    int i, int blockN, int warmupIter, int32_t batch, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobal, int32_t n1, int32_t n2, int blockNum, int blockIdx, int32_t N,
    GlobalTensor<float> &workGlobal, int halfBatchCeil, GlobalTensor<float> &workGlobal2)
{
    int o = i / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - i);
    CrossCoreWaitFlag(0x1);
    opmm.CopyInAReside(twiddleGlobal[n1 * n1 * 3], n2, n2);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    for (int j = i + startIdx; j < i + startIdx + calcNum; ) {
        int compBatch = min(i + startIdx + calcNum - j, 8192 / (int)N);
        compBatch = max(compBatch, 1);
        if (batch / 2 - i >= blockN || i / blockN < warmupIter + 2) {
            opmm.SetMatrix(workGlobal[(j - i + o) * N], workGlobal[(j - i + o) * N + N * halfBatchCeil], workGlobal[(j - i + o) * N], workGlobal[(j - i + o) * N + N * halfBatchCeil], n2, n2, n1, N, N);
        } else {
            auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
            int t = j - i - startIdx;
            opmm.SetMatrix(workGlobalTmp[t * N], workGlobalTmp[t * N + N * halfBatchCeil], workGlobalTmp[t * N], workGlobalTmp[t * N + N * halfBatchCeil], n2, n2, n1, N, N);
        }
        opmm.ProcessWithTransposeBatchMerged(0, 0, 0, n2, n1, n2, compBatch);
        j += compBatch;
    }
    if (batch &1 &&calcN < blockN &&blockIdx == blockNum - 1) {
        if (batch / 2 - i >= blockN || i / blockN < warmupIter + 2) {
            opmm.SetMatrix(workGlobal[(calcN + o) * N], workGlobal2, workGlobal[(calcN + o) * N], workGlobal2, n2, n2, n1, N, N);
        } else {
            auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
            opmm.SetMatrix(workGlobalTmp[calcNum * N], workGlobal2, workGlobalTmp[calcNum * N], workGlobal2, n2, n2, n1, N, N);
        }
        opmm.ProcessWithTransposeBatchMerged(0, 0, 0, n2, n1, n2);
    }
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
    opmm.FreeAReside();
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void AdvanceFft2Dft1(
    int i, int blockN, int warmupIter, int32_t batch, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobal, int32_t n1, int blockNum, int blockIdx, int32_t N,
    GlobalTensor<float> &xGlobal, int halfBatchCeil, GlobalTensor<float> &workGlobal, int32_t n2,
    GlobalTensor<float> &workGlobal2)
{
    int nextI = i + blockN * warmupIter;
    int o = nextI / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - nextI);
    opmm.CopyInAReside(twiddleGlobal, n1, n1);
    int averageNum = calcN / blockNum;
    int remainderNum = calcN % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    for (int j = nextI + startIdx; j < nextI + startIdx + calcNum; ) {
        int compBatch = min(nextI + startIdx + calcNum - j, 8192 / (int)N);
        compBatch = max(compBatch, 1);
        if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
            opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N + N * halfBatchCeil], workGlobal[(j - nextI + o) * N], workGlobal[(j - nextI + o) * N + N * halfBatchCeil], n1, n2, n2, N, N);
        } else {
            auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
            int t = j - nextI - startIdx;
            opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N + N * halfBatchCeil], workGlobalTmp[t * N], workGlobalTmp[t * N + N * halfBatchCeil], n1, n2, n2, N, N);
        }
        opmm.Process(0, 0, 0, n1, n2, n1, compBatch);
        j += compBatch;
    }
    if (batch &1 &&calcN < blockN &&blockIdx == blockNum - 1) {
        int j = batch / 2;
        if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
            opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N], workGlobal[(calcN + o) * N], workGlobal2, n1, n2, n2, N, N);
        } else {
            auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
            opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N], workGlobalTmp[calcNum * N], workGlobal2, n1, n2, n2, N, N);
        }
        opmm.Process(0, 0, 0, n1, n2, n1);
    }
    opmm.FreeAReside();
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
}

#endif

__aicore__ inline void RunFft2_r2cCube(
    int32_t N, int32_t batch, int32_t n1, int32_t n2, int blockN, int blockNum, int blockIdx,
    GlobalTensor<float> &xGlobal, GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal2,
    GlobalTensor<float> &twiddleGlobal, TPipe &pipe, int warmupIter, int halfBatchCeil)
{
    #ifdef  __DAV_C220_CUBE__
        twiddleGlobal = twiddleGlobal[N * 2];

        CMatmulCustom<float> opmm;
        opmm.Init(&pipe);

        for (int i = 0; i < blockN * warmupIter; i += blockN) {
            {
                WarmupFft2Dft1(i, blockN, warmupIter, batch, opmm, twiddleGlobal, n1, blockNum, blockIdx, N,
                    xGlobal, halfBatchCeil, workGlobal, n2, workGlobal2);
            }
        }

        for (int i = 0; i < halfBatchCeil; i += blockN) {
            {
                ComputeFft2Dft2(i, blockN, warmupIter, batch, opmm, twiddleGlobal, n1, n2, blockNum, blockIdx, N,
                    workGlobal, halfBatchCeil, workGlobal2);
            }

            if (i + blockN * warmupIter < halfBatchCeil) {
                AdvanceFft2Dft1(i, blockN, warmupIter, batch, opmm, twiddleGlobal, n1, blockNum, blockIdx, N,
                    xGlobal, halfBatchCeil, workGlobal, n2, workGlobal2);
            }
        }

        #endif
}

extern "C" __global__ __aicore__ void fft2_r2c(
    __gm__ uint8_t * __restrict__ X_gm,
    __gm__ uint8_t * __restrict__ Y_gm,
    __gm__ uint8_t * __restrict__ work_gm,
    __gm__ uint8_t * __restrict__ twiddle_gm,
    __gm__ uint8_t * __restrict__ gather_gm,
    __gm__ uint8_t * __restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Rfft1DR2CCommonKernel::KernelTiling<2> tiling(tiling_para_gm);
    int32_t N = tiling.N;
    int32_t batch = tiling.batch;
    int32_t n1 = tiling.n1;
    int32_t n2 = tiling.n2;
    if (N <= 0 || batch <= 0) {
        return;
    }

    int blockN;
    if (N <= 4096) {
        blockN = 1310720 / N;
    } else {
        blockN = 2621440 / N;
    }

    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int pairCubeIdx = blockIdx >> 1;

    GlobalTensor<float> xGlobal, yGlobal, workGlobal, workGlobal2, workGlobalLocal, twiddleGlobal;
    GlobalTensor<uint32_t> gatherGlobal;
    xGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(X_gm);
    yGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(Y_gm);
    workGlobal2 = Rfft1DR2CCommonKernel::BindGlobal<float>(work_gm);
    twiddleGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(twiddle_gm);
    gatherGlobal = Rfft1DR2CCommonKernel::BindGlobal<uint32_t>(gather_gm);

    workGlobal = xGlobal;

    TPipe pipe;
    int warmupIter = 2;
    if (blockN * 2 >= batch) warmupIter = 1;

    int halfN = N / 2;
    int halfBatchCeil = (batch + 1) / 2;

    RunFft2_r2cVector(N, batch, blockN, blockNum, blockIdx, pairCubeIdx, yGlobal, workGlobal, workGlobal2,
        twiddleGlobal, gatherGlobal, pipe, warmupIter, halfN, halfBatchCeil);

    RunFft2_r2cCube(N, batch, n1, n2, blockN, blockNum, blockIdx, xGlobal, workGlobal, workGlobal2, twiddleGlobal,
        pipe, warmupIter, halfBatchCeil);
}

}

#endif
