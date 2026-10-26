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

#ifndef RFFT1_D_R2C_FFT3_2_KERNEL_H
#define RFFT1_D_R2C_FFT3_2_KERNEL_H

#include "../rfft1_d_r2c_schedule_common.h"

namespace Rfft1DR2CFft32Kernel {

using namespace::AscendC;

using Rfft1DR2CCommonKernel::CMatmulCustom;

template<typename T>
using MulCustom = Rfft1DR2CCommonKernel::TransposeTwiddleMul<T, true>;

using Rfft1DR2CCommonKernel::UnpackCustom;


#ifdef __DAV_C220_VEC__
__aicore__ inline void UnpackFft3_2Output(
    int i, int blockN, int warmupIter, int32_t batch, UnpackCustom<float> &op,
    GlobalTensor<uint32_t> &gatherGlobal, GlobalTensor<float> &workGlobal, int32_t N, int halfBatchCeil,
    GlobalTensor<float> &yGlobal, int halfN, int blockNum, int blockIdx, GlobalTensor<float> &workGlobal3)
{
    CrossCoreWaitFlag(0x0);

    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
    CrossCoreWaitFlag(0x3);

    int o = i / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - i);
    op.LoadGather(gatherGlobal);
    op.SetMatrix(workGlobal[o * N], workGlobal[o * N + N * halfBatchCeil], yGlobal[i * (N + 2)], yGlobal[i * (N + 2) + (halfN + 1) * 2 * halfBatchCeil]);
    op.UnPack2(N, calcN, N, N + 2, blockNum * 2, blockIdx);
    int averageNum = calcN / (blockNum * 2);
    int remainderNum = calcN % (blockNum * 2);
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    if (calcNum > 0) {
        op.SetValueBatch(workGlobal[(startIdx + o) * N + halfN], yGlobal[(i + startIdx) * (halfN + 1) * 2 + N], calcNum, N, N + 2);
        op.SetValueBatch(workGlobal[(startIdx + o) * N + N * halfBatchCeil + halfN], yGlobal[(i + startIdx) * (halfN + 1) * 2 + (halfN + 1) * 2 * halfBatchCeil + N], 1, N, N + 2);
    }
    if (batch &1 &&calcN < blockN) {
        int j = batch / 2;
        op.SetMatrix(workGlobal[(calcN + o) * N], workGlobal3, yGlobal[j * (N + 2)], yGlobal[j * (N + 2)]);
        op.UnPack2(N, 1, N, N + 2, blockNum * 2, blockIdx);
        if (!blockIdx) op.SetValueBatch(workGlobal[(calcN + o) * N + halfN], yGlobal[j * (N + 2) + N], 1, N, N + 2);
    }
    op.FreeGather();
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void WarmupFft3_2Twiddle(
    int blockN, int32_t batch, int i, MulCustom<float> &opMul, int o, int32_t N, int32_t n1, int32_t n2,
    int32_t n3, int blockNum, int blockIdx, GlobalTensor<float> &twiddleGlobal,
    GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal3, GlobalTensor<float> &workGlobal2,
    int halfBatchCeil)
{
    CrossCoreWaitFlag(0x0);

    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
    CrossCoreWaitFlag(0x3);

    int calcN = min(blockN, batch / 2 - i);
    Rfft1DR2CCommonKernel::ProcessFft3DistributedTwiddle(opMul, o, N, n1, n2, n3, calcN,
        blockNum, blockIdx, batch, blockN, twiddleGlobal, workGlobal, workGlobal3, workGlobal2,
        halfBatchCeil);
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void AdvanceFft3_2Twiddle(
    int i, int blockN, int warmupIter, int32_t batch, MulCustom<float> &opMul, int32_t N, int32_t n1,
    int32_t n2, int32_t n3, int blockNum, int blockIdx, GlobalTensor<float> &twiddleGlobal,
    GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal3, GlobalTensor<float> &workGlobal2,
    int halfBatchCeil)
{
    CrossCoreWaitFlag(0x0);

    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
    CrossCoreWaitFlag(0x3);

    int nextI = i + blockN * warmupIter;
    int o = nextI / blockN % (warmupIter + 2) * blockN;
    int calcN = min(blockN, batch / 2 - nextI);
    Rfft1DR2CCommonKernel::ProcessFft3DistributedTwiddle(opMul, o, N, n1, n2, n3, calcN,
        blockNum, blockIdx, batch, blockN, twiddleGlobal, workGlobal, workGlobal3, workGlobal2,
        halfBatchCeil);
}

#endif

__aicore__ inline void RunFft3_2_r2cVector(Rfft1DR2CCommonKernel::Fft3KernelContext &ctx, TPipe &pipe)
{
    #ifdef __DAV_C220_VEC__
        Rfft1DR2CCommonKernel::Fft3VectorState<true> state(ctx, pipe);
        for (int i = 0; i < ctx.blockN * ctx.warmupIter; i += ctx.blockN) {
            int o = i / ctx.blockN % (ctx.warmupIter + 2) * ctx.blockN;
            {
                WarmupFft3_2Twiddle(ctx.blockN, ctx.tiling.batch, i, state.opMul, o, ctx.tiling.N, ctx.tiling.n1,
                    ctx.tiling.n2, ctx.tiling.n3, ctx.blockNum, ctx.blockIdx, ctx.twiddleGlobal,
                    ctx.workGlobal, ctx.workGlobal3, ctx.workGlobal2, ctx.halfBatchCeil);
            }
        }

        for (int i = 0; i < ctx.halfBatchCeil; i += ctx.blockN) {
            {
                UnpackFft3_2Output(i, ctx.blockN, ctx.warmupIter, ctx.tiling.batch, state.op, ctx.gatherGlobal,
                    ctx.workGlobal, ctx.tiling.N, ctx.halfBatchCeil,
                    ctx.yGlobal, ctx.halfN, ctx.blockNum, ctx.blockIdx, ctx.workGlobal3);
            }

            if (i + ctx.blockN * ctx.warmupIter < ctx.halfBatchCeil) {
                AdvanceFft3_2Twiddle(i, ctx.blockN, ctx.warmupIter, ctx.tiling.batch, state.opMul, ctx.tiling.N,
                    ctx.tiling.n1, ctx.tiling.n2, ctx.tiling.n3, ctx.blockNum, ctx.blockIdx,
                    ctx.twiddleGlobal, ctx.workGlobal, ctx.workGlobal3, ctx.workGlobal2, ctx.halfBatchCeil);
            }
        }

    #endif
}

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft3_2Dft3Tail(
    int32_t batch, int _n2, int blockNum, int blockIdx, int _n1, int32_t n3, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &workGlobal2, int calcN, int o, int32_t N, int halfBatchCeil,
    GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal3)
{
    int j = batch / 2;
    int averageNum = _n2 / blockNum;
    int remainderNum = _n2 % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    for (int i2 = startIdx; i2 < startIdx + calcNum; ) {
        int compBatch = min(_n2 - i2, 8192 / _n1 / n3);
        compBatch = max(compBatch, 1);
        opmm.SetMatrix(workGlobal2[(calcN + o) * N + i2 * _n1 * n3], workGlobal2[(calcN + o) * N + i2 * _n1 * n3 + N * halfBatchCeil], workGlobal[(calcN + o) * N + i2 * _n1], workGlobal3[i2 * _n1], n3, n3, _n1 * _n2, _n1 * n3, _n1);
        opmm.ProcessWithTranspose(0, 0, 0, n3, _n1, n3, compBatch);
        i2 += compBatch;
    }
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft3_2Dft3(
    int32_t n1, int32_t n2, int calcN, int blockNum, int blockIdx, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobal, int32_t n3, int i, GlobalTensor<float> &workGlobal2, int o,
    int32_t N, int halfBatchCeil, GlobalTensor<float> &workGlobal, int32_t batch, int blockN,
    GlobalTensor<float> &workGlobal3)
{
    int _n1 = 128, _n2 = n1 * n2 / 128;
    int averageNum = calcN * _n2 / blockNum;
    int remainderNum = calcN * _n2 % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);
    opmm.CopyInAReside(twiddleGlobal[n1 * n1 * 3 + n1 * n2 * n2 * 3], n3, n3);
    int idx = 0;
    for (int j = i + startIdx / _n2; idx < calcNum; ++j) {
        for (int i2 = (!idx ? startIdx % _n2 : 0); i2 < _n2; ) {
            int compBatch = min(_n2 - i2, 8192 / _n1 / n3);
            compBatch = min(compBatch, calcNum - idx);
            compBatch = max(compBatch, 1);
            opmm.SetMatrix(workGlobal2[(j - i + o) * N + i2 * _n1 * n3], workGlobal2[(j - i + o) * N + i2 * _n1 * n3 + N * halfBatchCeil], workGlobal[(j - i + o) * N + i2 * _n1], workGlobal[(j - i + o) * N + i2 * _n1 + N * halfBatchCeil], n3, n3, _n1 * _n2, _n1 * n3, _n1);
            opmm.ProcessWithTranspose(0, 0, 0, n3, _n1, n3, compBatch);
            i2 += compBatch;
            idx += compBatch;
            if (idx == calcNum) break;
        }
    }
    if (batch &1 &&calcN < blockN) {
        ComputeFft3_2Dft3Tail(batch, _n2, blockNum, blockIdx, _n1, n3, opmm, workGlobal2, calcN, o, N,
            halfBatchCeil, workGlobal, workGlobal3);
    }
    opmm.FreeAReside();
}

#endif

__aicore__ inline void RunFft3_2_r2cCube(Rfft1DR2CCommonKernel::Fft3KernelContext &ctx, TPipe &pipe)
{
    #ifdef  __DAV_C220_CUBE__
        Rfft1DR2CCommonKernel::Fft3CubeState state(ctx, pipe);
        Rfft1DR2CCommonKernel::WarmupFft3Stages<true>(state.stageContext);

        for (int i = 0; i < ctx.halfBatchCeil; i += ctx.blockN) {
            int o = i / ctx.blockN % (ctx.warmupIter + 2) * ctx.blockN;
            int calcN = min(ctx.blockN, ctx.tiling.batch / 2 - i);

            CrossCoreWaitFlag(0x1);

            // dft 3
            {
                ComputeFft3_2Dft3(ctx.tiling.n1, ctx.tiling.n2, calcN, ctx.blockNum, ctx.blockIdx, state.opmm,
                    ctx.twiddleGlobal, ctx.tiling.n3, i, ctx.workGlobal2, o, ctx.tiling.N,
                    ctx.halfBatchCeil, ctx.workGlobal, ctx.tiling.batch, ctx.blockN, ctx.workGlobal3);
            }

            CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);

            if (i + ctx.blockN * ctx.warmupIter < ctx.halfBatchCeil) {
                int nextI = i + ctx.blockN * ctx.warmupIter;
                int o = nextI / ctx.blockN % (ctx.warmupIter + 2) * ctx.blockN;
                int calcN = min(ctx.blockN, ctx.tiling.batch / 2 - nextI);

                Rfft1DR2CCommonKernel::ProcessFft3FirstStages<true>(state.stageContext, nextI, o, calcN);
            }
        }

        #endif
}

struct Fft3_2Dispatch {
    __aicore__ inline static void Run(Rfft1DR2CCommonKernel::Fft3KernelContext &ctx, TPipe &pipe) {
        RunFft3_2_r2cVector(ctx, pipe);

        RunFft3_2_r2cCube(ctx, pipe);
    }
};

extern "C" __global__ __aicore__ void fft3_2_r2c(
    __gm__ uint8_t * __restrict__ X_gm,
    __gm__ uint8_t * __restrict__ Y_gm,
    __gm__ uint8_t * __restrict__ work_gm,
    __gm__ uint8_t * __restrict__ twiddle_gm,
    __gm__ uint8_t * __restrict__ gather_gm,
    __gm__ uint8_t * __restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    Rfft1DR2CCommonKernel::RunFft3Kernel<Fft3_2Dispatch>(
        X_gm, Y_gm, work_gm, twiddle_gm, gather_gm, tiling_para_gm, pipe);
}

}

#endif
