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

#ifndef RFFT1_D_R2C_FFT4_KERNEL_H
#define RFFT1_D_R2C_FFT4_KERNEL_H

#include "../rfft1_d_r2c_schedule_common.h"

namespace Rfft1DR2CFft4Kernel {

using namespace::AscendC;

using Rfft1DR2CCommonKernel::CMatmulCustom;

template<typename T>
using MulCustom = Rfft1DR2CCommonKernel::RecurrenceTwiddleMul<T, false>;

using Rfft1DR2CCommonKernel::UnpackCustom;

#ifdef __DAV_C220_VEC__
__aicore__ inline void ApplyFft4Twiddle(
    int row, int col, GlobalTensor<float> &xGlobal, int offset, int32_t N, GlobalTensor<float> &workGlobal,
    int32_t batch, MulCustom<float> &opMul, GlobalTensor<float> &twiddleGlobal, int blockNum, int blockIdx,
    GlobalTensor<float> &workGlobal3)
{
    int stepN = 128;
    int l2Step = min(row / stepN, 4194304 / 2 / col / stepN);
    int l2Blocks = row / stepN / l2Step;

    auto xGlobalTmp = xGlobal[offset * N];
    auto workGlobalTmp = workGlobal[offset * N];
    if (offset < batch - 1) {
        opMul.SetMatrix(xGlobalTmp, xGlobalTmp[N], twiddleGlobal, twiddleGlobal[col * (blockNum * l2Blocks + 1)], workGlobalTmp, workGlobalTmp[N]);
        opMul.Process(row, col, stepN, l2Step, blockNum << 1, blockIdx);
    } else {
        opMul.SetMatrix(xGlobalTmp, workGlobal3, twiddleGlobal, twiddleGlobal[col * (blockNum * l2Blocks + 1)], workGlobalTmp, workGlobal3);
        opMul.Process(row, col, stepN, l2Step, blockNum << 1, blockIdx);
    }
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void UnpackFft4Output(
    GlobalTensor<float> &workGlobal2, int offset, int32_t N, GlobalTensor<float> &yGlobal, int32_t batch,
    UnpackCustom<float> &op, int blockNum, int blockIdx, int halfN)
{
    CrossCoreWaitFlag(0x0);

    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
    CrossCoreWaitFlag(0x3);
    auto workGlobal2Tmp = workGlobal2[offset * N];
    auto yGlobalTmp = yGlobal[offset * (N + 2)];
    if (offset < batch - 1) {
        op.SetMatrix(workGlobal2Tmp, workGlobal2Tmp[N], yGlobalTmp, yGlobalTmp[N + 2]);
    } else {
        op.SetMatrix(workGlobal2Tmp, workGlobal2Tmp[N], yGlobalTmp, yGlobalTmp);
    }
    op.UnPack2(N, 1, N, N + 2, blockNum << 1, blockIdx);
    if (!blockIdx) {
        op.SetValueBatch(workGlobal2Tmp[halfN], yGlobalTmp[N], 1, N, N + 2);
        if (offset < batch - 1) {
            op.SetValueBatch(workGlobal2Tmp[halfN + N], yGlobalTmp[N * 2 + 2], 1, N, N + 2);
        }
    }
}

#endif

__aicore__ inline void RunFft4_r2cVector(
    int32_t N, int32_t batch, int blockNum, int blockIdx, GlobalTensor<float> &xGlobal,
    GlobalTensor<float> &yGlobal, GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal2,
    GlobalTensor<float> &workGlobal3, GlobalTensor<float> &twiddleGlobal,
    GlobalTensor<uint32_t> &gatherGlobal, TPipe &pipe, int halfN, int row, int col, int halfBatchCeil)
{
    #ifdef __DAV_C220_VEC__
        int coreIdx = blockIdx >> 1;

        MulCustom<float> opMul;
        UnpackCustom<float> op;

        for (int offset = 0; offset < halfBatchCeil * 2; offset += 2) {

            CrossCoreWaitFlag(0x0);

            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
            CrossCoreWaitFlag(0x3);

            opMul.Init(&pipe);
            {
                ApplyFft4Twiddle(row, col, xGlobal, offset, N, workGlobal, batch, opMul, twiddleGlobal, blockNum,
                    blockIdx, workGlobal3);
            }
            pipe.Reset();

            op.Init(&pipe);
            op.LoadGather(gatherGlobal);
            {
                UnpackFft4Output(workGlobal2, offset, N, yGlobal, batch, op, blockNum, blockIdx, halfN);
            }
            op.FreeGather();
            pipe.Reset();

        }

    #endif
}

#ifdef __DAV_C220_CUBE__
__aicore__ inline void CopyFft4RealInput(
    CMatmulCustom<float> &opmm, GlobalTensor<float> &eyeGlobal, int row, int blockNum, int blockIdx,
    int l2Idx, int l2Step, GlobalTensor<float> &xGlobal, int stepN, int offset, int32_t N,
    GlobalTensor<float> &workGlobal2, int col, int32_t batch)
{
    opmm.CopyInEye(eyeGlobal, 32, 32);
    int totalNum = row / 32;
    int averageNum = totalNum / blockNum;
    int remainderNum = totalNum % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);

    for (int l2Idx2 = l2Idx; l2Idx2 < l2Idx + l2Step; ++l2Idx2) {
        auto xGlobalTmp = xGlobal[l2Idx2 * stepN + offset * N];
        auto workGlobal2Tmp = workGlobal2[(l2Idx2 - l2Idx) * row * stepN + offset * N];
        for (int i = startIdx; i < startIdx + calcNum; ) {
            int compBatch = min(8192 / 32 / stepN, startIdx + calcNum - i);
            opmm.SetMatrix(xGlobalTmp[i * col * 32], workGlobal2Tmp[i * stepN * 32], 32, col, stepN, col * 32, stepN * 32);
            opmm.ProcessReal(0, 0, 0, 32, stepN, 32, compBatch);
            i += compBatch;
        }
    }

    for (int l2Idx2 = l2Idx; l2Idx2 < l2Idx + l2Step; ++l2Idx2) {
        auto xGlobalTmp = xGlobal[l2Idx2 * stepN + offset * N];
        auto workGlobal2Tmp = workGlobal2[(l2Idx2 - l2Idx) * row * stepN + offset * N];
        for (int i = startIdx; i < startIdx + calcNum; ) {
            int compBatch = min(8192 / 32 / stepN, startIdx + calcNum - i);
            if (offset < batch - 1) {
                opmm.SetMatrix(xGlobalTmp[i * col * 32 + N], workGlobal2Tmp[i * stepN * 32 + N], 32, col, stepN, col * 32, stepN * 32);
            } else {
                opmm.SetMatrix(xGlobalTmp[i * col * 32], workGlobal2Tmp[i * stepN * 32 + N], 32, col, stepN, col * 32, stepN * 32);
            }
            opmm.ProcessReal(0, 0, 0, 32, stepN, 32, compBatch);
            i += compBatch;
        }
    }

    opmm.FreeEye();
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft4Dft1(
    CMatmulCustom<float> &opmm, GlobalTensor<float> &twiddleGlobalTmp, int32_t n1,
    GlobalTensor<float> &workGlobal2, int offset, int32_t N, int l2Step, int32_t n2, int blockNum,
    int blockIdx, int row, int stepN)
{
    opmm.CopyInAReside(twiddleGlobalTmp, n1, n1);
    auto workGlobal2Tmp = workGlobal2[offset * N];

    int totalNum = l2Step * n2;
    int averageNum = totalNum / blockNum;
    int remainderNum = totalNum % blockNum;
    int startIdx = (blockIdx * averageNum + min(blockIdx, remainderNum));
    int calcNum = (averageNum + (blockIdx < remainderNum));

    int idx = 0;
    for (int i = startIdx / n2; idx < calcNum; ++i) {
        int startJ = !idx ? startIdx % n2 : 0;
        int compBatch = min(n2 - startJ, calcNum - idx);

        opmm.SetMatrix(workGlobal2Tmp[i * row * stepN + startJ * stepN], workGlobal2Tmp[i * row * stepN + startJ * stepN + N], workGlobal2Tmp[i * row * stepN + startJ * stepN], workGlobal2Tmp[i * row * stepN + startJ * stepN + N], n1, n2 * stepN, n2 * stepN);
        opmm.Process(0, 0, 0, n1, compBatch * stepN, n1);
        idx += compBatch;
    }
    opmm.FreeAReside();
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft4Dft2(
    GlobalTensor<float> &twiddleGlobalTmp, int32_t n1, GlobalTensor<float> &xGlobal, int l2Idx, int stepN,
    int offset, int32_t N, GlobalTensor<float> &workGlobal3, GlobalTensor<float> &workGlobal2, int l2Step,
    int blockNum, int blockIdx, CMatmulCustom<float> &opmm, int32_t n2, int32_t batch, int row, int col)
{
    auto twiddleGlobalTmp2 = twiddleGlobalTmp[n1 * n1 * 3];
    auto xGlobalTmp = xGlobal[l2Idx * stepN + offset * N];
    auto workGlobal3Tmp = workGlobal3[l2Idx * stepN];
    auto workGlobal2Tmp = workGlobal2[offset * N];
    Rfft1DR2CCommonKernel::Dft2Layout4 layout{workGlobal2Tmp, xGlobalTmp, workGlobal3Tmp,
        offset, batch, N, n1, row, col};
    Rfft1DR2CCommonKernel::Dft2Schedule schedule{n1, n2, l2Step, stepN, blockNum, blockIdx};
    Rfft1DR2CCommonKernel::ProcessDft2Tiles(opmm, twiddleGlobalTmp2, schedule, layout);
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft4Dft3(
    GlobalTensor<float> &workGlobal, int l2Idx, int l2Step, int stepN, int col, int offset, int32_t N,
    GlobalTensor<float> &workGlobal3, int blockNum, int blockIdx, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobalTmp, int32_t n3, int32_t batch, int32_t n4)
{
    auto workGlobalTmp = workGlobal[(l2Idx % (l2Step * 2)) * stepN * col + offset * N];
    auto workGlobal3Tmp = workGlobal3[(l2Idx % (l2Step * 2)) * stepN * col];
    int averageNum = l2Step * stepN / blockNum;
    int remainderNum = l2Step * stepN % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);

    opmm.CopyInAReside(twiddleGlobalTmp, n3, n3);
    for (int i = startIdx; i < startIdx + calcNum; ) {
        int compBatch = min(startIdx + calcNum - i, 8192 / col);
        if (offset < batch - 1) {
            opmm.SetMatrix(workGlobalTmp[i * col], workGlobalTmp[i * col + N], workGlobalTmp[i * col], workGlobalTmp[i * col + N], n3, n4, n4, col, col);
        } else {
            opmm.SetMatrix(workGlobalTmp[i * col], workGlobal3Tmp[i * col], workGlobalTmp[i * col], workGlobal3Tmp[i * col], n3, n4, n4, col, col);
        }
        opmm.Process(0, 0, 0, n3, n4, n3, compBatch);
        i += compBatch;
    }
    opmm.FreeAReside();
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft4Dft4(
    GlobalTensor<float> &twiddleGlobalTmp, int32_t n3, GlobalTensor<float> &workGlobal, int l2Idx,
    int l2Step, int stepN, int col, int offset, int32_t N, GlobalTensor<float> &workGlobal2,
    GlobalTensor<float> &workGlobal3, int blockNum, int blockIdx, CMatmulCustom<float> &opmm, int32_t n4,
    int32_t batch, int row)
{
    auto twiddleGlobalTmp2 = twiddleGlobalTmp[n3 * n3 * 3];
    auto workGlobalTmp = workGlobal[(l2Idx % (l2Step * 2)) * stepN * col + offset * N];
    auto workGlobal2Tmp = workGlobal2[l2Idx * stepN + offset * N];
    auto workGlobal3Tmp = workGlobal3[(l2Idx % (l2Step * 2)) * stepN * col];
    int averageNum = n3 * l2Step / blockNum;
    int remainderNum = n3 * l2Step % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    int idx = 0;
    for (int j = startIdx / l2Step; idx < calcNum; ++j) {
        opmm.CopyInAReside(twiddleGlobalTmp2[j * n4 * n4 * 3], n4, n4);
        for (int k = !idx ? startIdx % l2Step : 0; k < l2Step; ) {
            int compBatch = min(l2Step - k, calcNum - idx);
            compBatch = min(compBatch, 8192 / stepN / n4);
            if (offset < batch - 1) {
                opmm.SetMatrix(workGlobalTmp[k * stepN * col + j * n4], workGlobalTmp[k * stepN * col + j * n4 + N], workGlobal2Tmp[k * stepN + j * row], workGlobal2Tmp[k * stepN + j * row + N], n4, col, row * n3, stepN * col, stepN);
            } else {
                opmm.SetMatrix(workGlobalTmp[k * stepN * col + j * n4], workGlobal3Tmp[k * stepN * col + j * n4], workGlobal2Tmp[k * stepN + j * row], workGlobal2Tmp[k * stepN + j * row + N], n4, col, row * n3, stepN * col, stepN);
            }
            opmm.ProcessWithTransposeBatchMerged(0, 0, 0, n4, stepN, n4, compBatch);
            k += compBatch;
            idx += compBatch;
            if (idx == calcNum) break;
        }
        opmm.FreeAReside();
    }
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft4FirstStages(
    int col, int row, CMatmulCustom<float> &opmm, GlobalTensor<float> &eyeGlobal, int blockNum,
    int blockIdx, GlobalTensor<float> &xGlobal, int offset, int32_t N, GlobalTensor<float> &workGlobal2,
    int32_t batch, GlobalTensor<float> &twiddleGlobalTmp, int32_t n1, int32_t n2,
    GlobalTensor<float> &workGlobal3)
{
    int stepN = 128;
    int l2Step = min(col / stepN, 4194304 / row / stepN);

    for (int l2Idx = 0; l2Idx < col / stepN; l2Idx += l2Step) {

        {
            CopyFft4RealInput(opmm, eyeGlobal, row, blockNum, blockIdx, l2Idx, l2Step, xGlobal, stepN, offset, N,
                workGlobal2, col, batch);
        }

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);

        {
            ComputeFft4Dft1(opmm, twiddleGlobalTmp, n1, workGlobal2, offset, N, l2Step, n2, blockNum, blockIdx, row,
                stepN);
        }

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);

        {
            ComputeFft4Dft2(twiddleGlobalTmp, n1, xGlobal, l2Idx, stepN, offset, N, workGlobal3, workGlobal2,
                l2Step, blockNum, blockIdx, opmm, n2, batch, row, col);
        }

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);
    }
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft4LastStages(
    int row, int col, GlobalTensor<float> &workGlobal, int offset, int32_t N,
    GlobalTensor<float> &workGlobal3, int blockNum, int blockIdx, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobalTmp, int32_t n3, int32_t batch, int32_t n4,
    GlobalTensor<float> &workGlobal2)
{
    int stepN = 128;
    int l2Step = min(row / stepN, 4194304 / 2 / col / stepN);

    for (int l2Idx = 0; l2Idx < row / stepN; l2Idx += l2Step) {

        CrossCoreWaitFlag(0x1);

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);

        {
            ComputeFft4Dft3(workGlobal, l2Idx, l2Step, stepN, col, offset, N, workGlobal3, blockNum, blockIdx, opmm,
                twiddleGlobalTmp, n3, batch, n4);
        }

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);

        {
            ComputeFft4Dft4(twiddleGlobalTmp, n3, workGlobal, l2Idx, l2Step, stepN, col, offset, N, workGlobal2,
                workGlobal3, blockNum, blockIdx, opmm, n4, batch, row);
        }

        if (l2Idx + l2Step * 2 < row / stepN) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
        }
    }
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
}

#endif

__aicore__ inline void RunFft4_r2cCube(
    int32_t N, int32_t batch, int32_t n1, int32_t n2, int32_t n3, int32_t n4, int blockNum, int blockIdx,
    GlobalTensor<float> &xGlobal, GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal2,
    GlobalTensor<float> &workGlobal3, GlobalTensor<float> &twiddleGlobal, GlobalTensor<float> &eyeGlobal,
    TPipe &pipe, int row, int col)
{
    #ifdef  __DAV_C220_CUBE__
        {
            int stepN = 128;
            int l2Step = min(row / stepN, 4194304 / 2 / col / stepN);
            int l2Blocks = row / stepN / l2Step;
            twiddleGlobal = twiddleGlobal[col * 2 * (blockNum * l2Blocks + 1)];
        }
        CMatmulCustom<float> opmm;
        opmm.Init(&pipe);

        for (int offset = 0; offset < batch; offset += 2) {
            auto twiddleGlobalTmp = twiddleGlobal;

            {
                ComputeFft4FirstStages(col, row, opmm, eyeGlobal, blockNum, blockIdx, xGlobal, offset, N,
                    workGlobal2, batch, twiddleGlobalTmp, n1, n2, workGlobal3);
            }

            twiddleGlobalTmp = twiddleGlobalTmp[n1 * n1 * 3];

            CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);

            twiddleGlobalTmp = twiddleGlobalTmp[n2 * n2 * n1 * 3];

            {
                ComputeFft4LastStages(row, col, workGlobal, offset, N, workGlobal3, blockNum, blockIdx, opmm,
                    twiddleGlobalTmp, n3, batch, n4, workGlobal2);
            }

        }
    #endif
}

extern "C" __global__ __aicore__ void fft4_r2c(
    __gm__ uint8_t * __restrict__ X_gm,
    __gm__ uint8_t * __restrict__ Y_gm,
    __gm__ uint8_t * __restrict__ work_gm,
    __gm__ uint8_t * __restrict__ twiddle_gm,
    __gm__ uint8_t * __restrict__ gather_gm,
    __gm__ uint8_t * __restrict__ eye_gm,
    __gm__ uint8_t * __restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Rfft1DR2CCommonKernel::KernelTiling<4> tiling(tiling_para_gm);
    int32_t N = tiling.N;
    int32_t batch = tiling.batch;
    int32_t n1 = tiling.n1;
    int32_t n2 = tiling.n2;
    int32_t n3 = tiling.n3;
    int32_t n4 = tiling.n4;
    if (N <= 0 || batch <= 0) {
        return;
    }

    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();

    GlobalTensor<float> xGlobal, yGlobal, workGlobal, workGlobal2, workGlobal3, twiddleGlobal, unpackTwiddleGlobal, eyeGlobal;
    GlobalTensor<uint32_t> gatherGlobal;
    xGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(X_gm);
    yGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(Y_gm);
    workGlobal2 = Rfft1DR2CCommonKernel::BindGlobal<float>(work_gm);

    twiddleGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(twiddle_gm);
    gatherGlobal = Rfft1DR2CCommonKernel::BindGlobal<uint32_t>(gather_gm);
    eyeGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(eye_gm);

    workGlobal = xGlobal;

    int halfN = N / 2;
    int row = n1 * n2;
    int col = n3 * n4;
    int halfBatchCeil = (batch + 1) / 2;
    workGlobal3 = workGlobal2[(batch + 1) * N];

    TPipe pipe;
    RunFft4_r2cVector(N, batch, blockNum, blockIdx, xGlobal, yGlobal, workGlobal, workGlobal2, workGlobal3,
        twiddleGlobal, gatherGlobal, pipe, halfN, row, col, halfBatchCeil);

    RunFft4_r2cCube(N, batch, n1, n2, n3, n4, blockNum, blockIdx, xGlobal, workGlobal, workGlobal2, workGlobal3,
        twiddleGlobal, eyeGlobal, pipe, row, col);
}

}

#endif
