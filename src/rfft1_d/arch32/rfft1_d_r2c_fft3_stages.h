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

#ifndef RFFT1_D_R2C_FFT3_STAGES_H
#define RFFT1_D_R2C_FFT3_STAGES_H

#include "rfft1_d_r2c_cmatmul.h"

namespace Rfft1DR2CCommonKernel {

using namespace AscendC;

struct Fft3StageContext {
    CMatmulCustom<float> &opmm;
    GlobalTensor<float> xGlobal, workGlobal, workGlobal3, twiddleGlobal;
    int N, batch, n1, n2, n3, halfBatchCeil, blockNum, blockIdx, blockN, warmupIter;
};

__aicore__ inline void ProcessFft3DistributedDft1(
    int N, int n1, int calcN, int blockNum, int blockIdx, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobal, int nextI, GlobalTensor<float> &xGlobal, int halfBatchCeil,
    GlobalTensor<float> &workGlobal, int o, int n2, int n3, int batch, int blockN,
    GlobalTensor<float> &workGlobal3)
{
    if (n1 == 0 || blockNum == 0) {
        return;
    }
    int block = N / n1 / 128;
    if (block == 0) {
        return;
    }
    int averageNum = calcN * block / blockNum;
    int remainderNum = calcN * block % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    opmm.CopyInAReside(twiddleGlobal, n1, n1);
    int idx = 0;
    for (int j = nextI + startIdx / block; idx < calcNum; ++j) {
        int startB = !idx ? startIdx % block : 0;
        int compBatch = min(block - startB, calcNum - idx);
        opmm.SetMatrix(xGlobal[j * N + startB * 128], xGlobal[j * N + startB * 128 + N * halfBatchCeil], workGlobal[(j - nextI + o) * N + startB * 128], workGlobal[(j - nextI + o) * N + startB * 128 + N * halfBatchCeil], n1, n2 * n3, n2 * n3);
        opmm.Process(0, 0, 0, n1, compBatch * 128, n1);
        idx += compBatch;
    }
    if (batch & 1 && calcN < blockN) {
        int j = batch / 2;
        int averageNum = block / blockNum;
        int remainderNum = block % blockNum;
        int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);
        {
            opmm.SetMatrix(xGlobal[j * N + startIdx * 128], xGlobal[j * N + startIdx * 128], workGlobal[(calcN + o) * N + startIdx * 128], workGlobal3[startIdx * 128], n1, n2 * n3, n2 * n3);
            opmm.Process(0, 0, 0, n1, calcNum * 128, n1);
        }
    }
    opmm.FreeAReside();
}

__aicore__ inline void ProcessFft3DistributedDft2(
    int calcN, int n1, int blockNum, int blockIdx, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobal, int n2, int nextI, int n3, GlobalTensor<float> &workGlobal, int o,
    int N, int halfBatchCeil, int batch, int blockN, GlobalTensor<float> &workGlobal3)
{
    if (blockNum == 0 || calcN == 0 || n2 == 0 || n3 == 0) {
        return;
    }
    int averageNum = calcN * n1 / blockNum;
    int remainderNum = calcN * n1 % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    int idx = 0;
    for (int i1 = startIdx / calcN; idx < calcNum; ++i1) {
        opmm.CopyInAReside(twiddleGlobal[n1 * n1 * 3 + i1 * n2 * n2 * 3], n2, n2);
        for (int j = nextI + (!idx ? startIdx % calcN : 0); j < nextI + calcN; ) {
            int compBatch = min(nextI + calcN - j, 8192 / n2 / n3);
            compBatch = min(compBatch, calcNum - idx);
            compBatch = max(compBatch, 1);
            opmm.SetMatrix(workGlobal[(j - nextI + o) * N + i1 * n2 * n3], workGlobal[(j - nextI + o) * N + i1 * n2 * n3 + N * halfBatchCeil], workGlobal[(j - nextI + o) * N + i1 * n2 * n3], workGlobal[(j - nextI + o) * N + i1 * n2 * n3 + N * halfBatchCeil], n2, n3, n3, N, N);
            opmm.Process(0, 0, 0, n2, n3, n2, compBatch);
            j += compBatch;
            idx += compBatch;
            if (idx == calcNum) break;
        }
        opmm.FreeAReside();
    }
    if (batch & 1 && calcN < blockN) {
        int j = batch / 2;
        int averageNum = n1 / blockNum;
        int remainderNum = n1 % blockNum;
        int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);
        for (int i1 = startIdx; i1 < startIdx + calcNum; ++i1) {
            opmm.CopyInAReside(twiddleGlobal[n1 * n1 * 3 + i1 * n2 * n2 * 3], n2, n2);
            {
                opmm.SetMatrix(workGlobal[(calcN + o) * N + i1 * n2 * n3], workGlobal3[i1 * n2 * n3], workGlobal[(calcN + o) * N + i1 * n2 * n3], workGlobal3[i1 * n2 * n3], n2, n3, n3, N, N);
                opmm.Process(0, 0, 0, n2, n3, n2);
            }
            opmm.FreeAReside();
        }
    }
}

__aicore__ inline void ProcessFft3ResidentDft1(
    CMatmulCustom<float> &opmm, GlobalTensor<float> &twiddleGlobal, int n1, int nextI, int startIdx,
    int calcNum, int batch, int blockN, int warmupIter, GlobalTensor<float> &xGlobal, int N, int halfBatchCeil,
    GlobalTensor<float> &workGlobal, int o, int n2, int n3, int blockNum, int blockIdx, int calcN,
    GlobalTensor<float> &workGlobal3)
{
    if (blockN == 0 || blockNum == 0) {
        return;
    }
    opmm.CopyInAReside(twiddleGlobal, n1, n1);
    for (int j = nextI + startIdx; j < nextI + startIdx + calcNum; ++j) {
        if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
            opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N + N * halfBatchCeil], workGlobal[(j - nextI + o) * N], workGlobal[(j - nextI + o) * N + N * halfBatchCeil], n1, n2 * n3, n2 * n3);
        } else {
            auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
            int t = j - nextI - startIdx;
            opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N + N * halfBatchCeil], workGlobalTmp[t * N], workGlobalTmp[t * N + N * halfBatchCeil], n1, n2 * n3, n2 * n3);
        }
        opmm.Process(0, 0, 0, n1, n2 * n3, n1);
    }
    if (batch & 1 && calcN < blockN && blockIdx == blockNum - 1) {
        int j = batch / 2;
        if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
            opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N], workGlobal[(calcN + o) * N], workGlobal3, n1, n2 * n3, n2 * n3);
        } else {
            auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
            opmm.SetMatrix(xGlobal[j * N], xGlobal[j * N], workGlobalTmp[calcNum * N], workGlobal3, n1, n2 * n3, n2 * n3);
        }
        opmm.Process(0, 0, 0, n1, n2 * n3, n1);
    }
    opmm.FreeAReside();
}

__aicore__ inline void ProcessFft3ResidentDft2Tail(
    int batch, int nextI, int blockN, int warmupIter, int n1, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobal, int n2, GlobalTensor<float> &workGlobal, int calcN, int o, int N,
    int n3, GlobalTensor<float> &workGlobal3, int blockNum, int blockIdx, int calcNum)
{
    if (blockN == 0 || blockNum == 0) {
        return;
    }
    if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
        for (int i1 = 0; i1 < n1; ++i1) {
            opmm.CopyInAReside(twiddleGlobal[n1 * n1 * 3 + i1 * n2 * n2 * 3], n2, n2);
            {
                opmm.SetMatrix(workGlobal[(calcN + o) * N + i1 * n2 * n3], workGlobal3[i1 * n2 * n3], workGlobal[(calcN + o) * N + i1 * n2 * n3], workGlobal3[i1 * n2 * n3], n2, n3, n3, N, N);
                opmm.Process(0, 0, 0, n2, n3, n2, 1);
            }
            opmm.FreeAReside();
        }
    } else {
        auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
        for (int i1 = 0; i1 < n1; ++i1) {
            opmm.CopyInAReside(twiddleGlobal[n1 * n1 * 3 + i1 * n2 * n2 * 3], n2, n2);
            {
                opmm.SetMatrix(workGlobalTmp[calcNum * N + i1 * n2 * n3], workGlobal3[i1 * n2 * n3], workGlobalTmp[calcNum * N + i1 * n2 * n3], workGlobal3[i1 * n2 * n3], n2, n3, n3, N, N);
                opmm.Process(0, 0, 0, n2, n3, n2, 1);
            }
            opmm.FreeAReside();
        }
    }
}

__aicore__ inline void ProcessFft3ResidentDft2(
    int n1, CMatmulCustom<float> &opmm, GlobalTensor<float> &twiddleGlobal, int n2, int nextI, int startIdx,
    int calcNum, int n3, int batch, int blockN, int warmupIter, GlobalTensor<float> &workGlobal, int o, int N,
    int halfBatchCeil, int blockNum, int blockIdx, int calcN, GlobalTensor<float> &workGlobal3)
{
    if (n2 == 0 || n3 == 0 || blockN == 0 || blockNum == 0) {
        return;
    }
    for (int i1 = 0; i1 < n1; ++i1) {
        opmm.CopyInAReside(twiddleGlobal[n1 * n1 * 3 + i1 * n2 * n2 * 3], n2, n2);
        for (int j = nextI + startIdx; j < nextI + startIdx + calcNum; ) {
            int compBatch = min(nextI + startIdx + calcNum - j, 8192 / n2 / n3);
            compBatch = max(compBatch, 1);
            if (batch / 2 - nextI >= blockN || nextI / blockN < warmupIter + 2) {
                opmm.SetMatrix(workGlobal[(j - nextI + o) * N + i1 * n2 * n3], workGlobal[(j - nextI + o) * N + i1 * n2 * n3 + N * halfBatchCeil], workGlobal[(j - nextI + o) * N + i1 * n2 * n3], workGlobal[(j - nextI + o) * N + i1 * n2 * n3 + N * halfBatchCeil], n2, n3, n3, N, N);
            } else {
                auto workGlobalTmp = workGlobal[o * N + (blockN / blockNum * blockIdx) * N];
                int t = j - nextI - startIdx;
                opmm.SetMatrix(workGlobalTmp[t * N + i1 * n2 * n3], workGlobalTmp[t * N + i1 * n2 * n3 + N * halfBatchCeil], workGlobalTmp[t * N + i1 * n2 * n3], workGlobalTmp[t * N + i1 * n2 * n3 + N * halfBatchCeil], n2, n3, n3, N, N);
            }
            opmm.Process(0, 0, 0, n2, n3, n2, compBatch);
            j += compBatch;
        }
        opmm.FreeAReside();
    }
    if (batch & 1 && calcN < blockN && blockIdx == blockNum - 1) {
        ProcessFft3ResidentDft2Tail(batch, nextI, blockN, warmupIter, n1, opmm, twiddleGlobal, n2, workGlobal,
            calcN, o, N, n3, workGlobal3, blockNum, blockIdx, calcNum);
    }
}

template<bool Distributed>
__aicore__ inline void ProcessFft3FirstStages(Fft3StageContext &ctx, int nextI, int o,
                                            int calcN, int startIdx = 0, int calcNum = 0) {
    auto &opmm = ctx.opmm;
    auto xGlobal = ctx.xGlobal;
    auto workGlobal = ctx.workGlobal;
    auto workGlobal3 = ctx.workGlobal3;
    auto twiddleGlobal = ctx.twiddleGlobal;
    int N = ctx.N, batch = ctx.batch, n1 = ctx.n1, n2 = ctx.n2, n3 = ctx.n3;
    int halfBatchCeil = ctx.halfBatchCeil, blockNum = ctx.blockNum, blockIdx = ctx.blockIdx;
    int blockN = ctx.blockN, warmupIter = ctx.warmupIter;
    if constexpr (Distributed) {
        // dft 1
        {
            ProcessFft3DistributedDft1(N, n1, calcN, blockNum, blockIdx, opmm, twiddleGlobal, nextI, xGlobal,
                halfBatchCeil, workGlobal, o, n2, n3, batch, blockN, workGlobal3);
        }

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x2);
        CrossCoreWaitFlag(0x2);

        // dft 2
        {
            ProcessFft3DistributedDft2(calcN, n1, blockNum, blockIdx, opmm, twiddleGlobal, n2, nextI, n3,
                workGlobal, o, N, halfBatchCeil, batch, blockN, workGlobal3);
        }

        CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
    } else {
        // dft 1
        {
            ProcessFft3ResidentDft1(opmm, twiddleGlobal, n1, nextI, startIdx, calcNum, batch, blockN, warmupIter,
                xGlobal, N, halfBatchCeil, workGlobal, o, n2, n3, blockNum, blockIdx, calcN, workGlobal3);
        }

        int32_t eventIDFIXToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);

        // dft 2
        {
            ProcessFft3ResidentDft2(n1, opmm, twiddleGlobal, n2, nextI, startIdx, calcNum, n3, batch, blockN,
                warmupIter, workGlobal, o, N, halfBatchCeil, blockNum, blockIdx, calcN, workGlobal3);
        }

        CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
    }
}

} // namespace Rfft1DR2CCommonKernel

#endif // RFFT1_D_R2C_FFT3_STAGES_H
