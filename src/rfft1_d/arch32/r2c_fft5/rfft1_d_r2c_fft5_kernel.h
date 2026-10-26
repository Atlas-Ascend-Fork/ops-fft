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

#ifndef RFFT1_D_R2C_FFT5_KERNEL_H
#define RFFT1_D_R2C_FFT5_KERNEL_H

#include "../rfft1_d_r2c_schedule_common.h"

namespace Rfft1DR2CFft5Kernel {

using namespace::AscendC;

using Rfft1DR2CCommonKernel::CMatmulCustom;

template<typename T>
class CopyCustom {
    TQue<QuePosition::VECIN, 2> inQueueSrc;
    TQue<QuePosition::VECOUT, 2> outQueueDst;
    int elementsPerBlock;
    static constexpr int TILE_LENGTH = 8192;
public:
    __aicore__ inline CopyCustom() {}
    __aicore__ inline void Init(TPipe *pipe) {
        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(inQueueSrc, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueDst, 4, TILE_LENGTH * sizeof(T));
    }
    __aicore__ inline void RowSwap2(GlobalTensor<T> srcGlobal, GlobalTensor<T> dstGlobal1, GlobalTensor<T> dstGlobal2, int m, int n, int batch, int srcStride, int dstStride, int srcBatchStride, int dstBatchStride, int blockIdx, int blockNum) {
        int averageNum = batch * m / blockNum;
        int remainderNum = batch * m % blockNum;
        int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        srcStride *= 2;
        srcBatchStride *= 2;

        int idx = 0;
        for (int i = startIdx / m; idx < calcNum; ++i) {
            for (int j = !idx ? startIdx % m : 0; j < m; ) {
                int compBatch = min(m - j, calcNum - idx);
                compBatch = min(compBatch, TILE_LENGTH / n / 2);
                DataCopyParams CopyInParams {
                    static_cast<uint16_t>(compBatch),
                    static_cast<uint16_t>(n * 2 / elementsPerBlock),
                    static_cast<uint16_t>((srcStride - n * 2) / elementsPerBlock),
                    0
                };
                DataCopyParams CopyOutParams {
                    static_cast<uint16_t>(compBatch),
                    static_cast<uint16_t>(n / elementsPerBlock),
                    0,
                    static_cast<uint16_t>((dstStride - n) / elementsPerBlock)
                };
                {
                    LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
                    DataCopy(srcLocal, srcGlobal[i * srcBatchStride + j * srcStride], CopyInParams);
                    inQueueSrc.EnQue(srcLocal);

                    srcLocal = inQueueSrc.DeQue<T>();
                    LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
                    uint64_t rsvdCnt;
                    GatherMask(dstLocal, srcLocal, 1, false, 0, {1, static_cast<uint16_t>((n * 2 * compBatch * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
                    GatherMask(dstLocal[TILE_LENGTH / 2], srcLocal, 2, false, 0, {1, static_cast<uint16_t>((n * 2 * compBatch * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
                    inQueueSrc.FreeTensor(srcLocal);
                    outQueueDst.EnQue(dstLocal);

                    dstLocal = outQueueDst.DeQue<T>();
                    DataCopy(dstGlobal1[i * dstBatchStride + j * dstStride], dstLocal, CopyOutParams);
                    DataCopy(dstGlobal2[i * dstBatchStride + j * dstStride], dstLocal[TILE_LENGTH / 2], CopyOutParams);
                    outQueueDst.FreeTensor(dstLocal);
                }
                idx += compBatch;
                j += compBatch;
                if (idx == calcNum) break;
            }
        }
    }
};

template<typename T>
using MulCustom = Rfft1DR2CCommonKernel::RecurrenceTwiddleMul<T, true>;

using Rfft1DR2CCommonKernel::UnpackCustom;

#ifdef __DAV_C220_VEC__
__aicore__ inline void CopyFft5Input(
    int col, uint64_t row64, int row, GlobalTensor<float> &xGlobal, int offset, uint64_t totalN,
    GlobalTensor<float> &workGlobal2, CopyCustom<float> &opCopy, uint64_t halfN, int blockIdx, int blockNum)
{
    int stepN = 128;
    int l2Step = min(col / stepN, static_cast<int>(4194304ULL / row64 / static_cast<uint64_t>(stepN)));
    int l2Blocks = row / stepN / l2Step;
    for (int l2Idx = 0; l2Idx < col / stepN; l2Idx += l2Step) {
        auto xGlobalTmp = xGlobal[static_cast<uint64_t>(l2Idx) * static_cast<uint64_t>(stepN) * 2ULL + static_cast<uint64_t>(offset) * totalN];
        auto workGlobal2Tmp = workGlobal2;
        if (l2Idx >= l2Step) CrossCoreWaitFlag(0x0);
        opCopy.RowSwap2(xGlobalTmp, workGlobal2Tmp, workGlobal2Tmp[halfN], row, stepN, l2Step, col, stepN, stepN, row64 * static_cast<uint64_t>(stepN), blockIdx, blockNum << 1);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
    }
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void ApplyFft5Twiddle(
    int row, GlobalTensor<float> &xGlobal, int offset, uint64_t totalN, GlobalTensor<float> &workGlobal,
    MulCustom<float> &opMul, uint64_t halfN, GlobalTensor<float> &twiddleGlobal, uint64_t col64,
    int blockNum, int col, int blockIdx)
{
    int stepN = 128;
    int l2Step = 1;
    int l2Blocks = row / stepN / l2Step;

    auto xGlobalTmp = xGlobal[static_cast<uint64_t>(offset) * totalN];
    auto workGlobalTmp = workGlobal;
    opMul.SetMatrix(workGlobalTmp, workGlobalTmp[halfN], twiddleGlobal, twiddleGlobal[col64 * static_cast<uint64_t>(blockNum * l2Blocks + 1)], xGlobalTmp, xGlobalTmp[halfN]);
    opMul.Process(row, col, stepN, l2Step, blockNum << 1, blockIdx);
}

#endif

#ifdef __DAV_C220_VEC__
__aicore__ inline void UnpackFft5Output(
    GlobalTensor<float> &workGlobal2, GlobalTensor<float> &yGlobal, int offset, uint64_t outputN,
    UnpackCustom<float> &op, uint64_t halfN, int blockNum, int blockIdx)
{
    CrossCoreWaitFlag(0x0);

    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
    CrossCoreWaitFlag(0x3);
    auto workGlobal2Tmp = workGlobal2;
    auto yGlobalTmp = yGlobal[static_cast<uint64_t>(offset) * outputN];
    op.SetMatrix(workGlobal2Tmp, workGlobal2Tmp[halfN], yGlobalTmp, yGlobalTmp[outputN]);
    op.UnPack(halfN, blockNum * 2, blockIdx);
}

#endif

__aicore__ inline void RunFft5_r2cVector(
    int32_t N, int32_t batch, int blockNum, int blockIdx, GlobalTensor<float> &xGlobal,
    GlobalTensor<float> &yGlobal, GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal2,
    GlobalTensor<float> &twiddleGlobal, GlobalTensor<float> &unpackTwiddleGlobal,
    GlobalTensor<float> &sTwiddleGlobal, GlobalTensor<uint32_t> &gatherGlobal, TPipe &pipe, uint64_t totalN,
    uint64_t halfN, uint64_t outputN, uint64_t row64, uint64_t col64, int row, int col)
{
    #ifdef __DAV_C220_VEC__
        int coreIdx = blockIdx >> 1;

        CopyCustom<float> opCopy;
        MulCustom<float> opMul;
        UnpackCustom<float> op;

        for (int offset = 0; offset < batch; ++offset) {
            opCopy.Init(&pipe);
            {
                CopyFft5Input(col, row64, row, xGlobal, offset, totalN, workGlobal2, opCopy, halfN, blockIdx,
                    blockNum);
            }
            pipe.Reset();

            CrossCoreWaitFlag(0x0);

            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
            CrossCoreWaitFlag(0x3);

            opMul.Init(&pipe);
            {
                ApplyFft5Twiddle(row, xGlobal, offset, totalN, workGlobal, opMul, halfN, twiddleGlobal, col64,
                    blockNum, col, blockIdx);
            }
            pipe.Reset();

            op.Init(&pipe);
            op.LoadGather(gatherGlobal);
            op.LoadTwiddle(unpackTwiddleGlobal, unpackTwiddleGlobal[static_cast<uint64_t>(blockNum * 2) * 4096ULL], sTwiddleGlobal, sTwiddleGlobal[static_cast<uint64_t>(N) / 16384ULL], blockIdx);
            {
                UnpackFft5Output(workGlobal2, yGlobal, offset, outputN, op, halfN, blockNum, blockIdx);
            }
            op.FreeGather();
            op.FreeTwiddle();
            pipe.Reset();

            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
            CrossCoreWaitFlag(0x3);
        }

    #endif
}

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft5Dft1(
    CMatmulCustom<float> &opmm, GlobalTensor<float> &twiddleGlobalTmp, int32_t n1,
    GlobalTensor<float> &workGlobal2, int l2Step, int32_t n2, int blockNum, int blockIdx, uint64_t row64,
    int stepN, uint64_t halfN)
{
    opmm.CopyInAReside(twiddleGlobalTmp, n1, n1);
    auto workGlobal2Tmp = workGlobal2;

    int totalNum = l2Step * n2;
    int averageNum = totalNum / blockNum;
    int remainderNum = totalNum % blockNum;
    int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);

    int idx = 0;
    for (int i = startIdx / n2; idx < calcNum; ++i) {
        int startJ = !idx ? startIdx % n2 : 0;
        int compBatch = min(n2 - startJ, calcNum - idx);

        uint64_t baseOffset = static_cast<uint64_t>(i) * row64 * static_cast<uint64_t>(stepN)
                            + static_cast<uint64_t>(startJ) * static_cast<uint64_t>(stepN);
        opmm.SetMatrix(workGlobal2Tmp[baseOffset], workGlobal2Tmp[baseOffset + halfN], workGlobal2Tmp[baseOffset], workGlobal2Tmp[baseOffset + halfN], n1, n2 * stepN, n2 * stepN);
        opmm.Process(0, 0, 0, n1, compBatch * stepN, n1);
        idx += compBatch;
    }
    opmm.FreeAReside();
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft5Dft2(
    GlobalTensor<float> &twiddleGlobalTmp, int32_t n1, GlobalTensor<float> &workGlobal2,
    GlobalTensor<float> &workGlobal, int l2Idx, int stepN, int l2Step, int blockNum, int blockIdx,
    CMatmulCustom<float> &opmm, int32_t n2, uint64_t row64, uint64_t col64, uint64_t halfN, int col)
{
    auto twiddleGlobalTmp2 = twiddleGlobalTmp[static_cast<uint64_t>(n1) * static_cast<uint64_t>(n1) * 3ULL];
    auto workGlobal2Tmp = workGlobal2;
    auto workGlobalTmp = workGlobal[static_cast<uint64_t>(l2Idx) * static_cast<uint64_t>(stepN)];
    Rfft1DR2CCommonKernel::Dft2Layout5 layout{workGlobal2Tmp, workGlobalTmp, row64, col64,
        halfN, n1, col};
    Rfft1DR2CCommonKernel::Dft2Schedule schedule{n1, n2, l2Step, stepN, blockNum, blockIdx};
    Rfft1DR2CCommonKernel::ProcessDft2Tiles(opmm, twiddleGlobalTmp2, schedule, layout);
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft5Dft3(
    GlobalTensor<float> &xGlobal, int l2Idx, int l2Step, int stepN, uint64_t col64, int offset,
    uint64_t totalN, int32_t n5, int32_t n4, int blockNum, int blockIdx, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobalTmp, int32_t n3, uint64_t halfN)
{
    auto xGlobalTmp = xGlobal[static_cast<uint64_t>(l2Idx % (l2Step * 2)) * static_cast<uint64_t>(stepN) * col64 + static_cast<uint64_t>(offset) * totalN];
    int ceilTo = 128 / n5;
    int averageNum = l2Step * stepN * n4 / ceilTo / blockNum;
    int remainderNum = l2Step * stepN * n4 / ceilTo % blockNum;
    int startIdx = (blockIdx * averageNum + min(blockIdx, remainderNum)) * ceilTo;
    int calcNum = (averageNum + (blockIdx < remainderNum)) * ceilTo;

    opmm.CopyInAReside(twiddleGlobalTmp, n3, n3);
    int idx = 0;
    for (int i = startIdx / n4; idx < calcNum; ++i) {
        int startI4 = !idx ? startIdx % n4 : 0;
        int compBatch = min(n4 - startI4, calcNum - idx);
        uint64_t matrixOffset = static_cast<uint64_t>(i) * col64
                              + static_cast<uint64_t>(startI4) * static_cast<uint64_t>(n5);
        opmm.SetMatrix(xGlobalTmp[matrixOffset], xGlobalTmp[matrixOffset + halfN], xGlobalTmp[matrixOffset], xGlobalTmp[matrixOffset + halfN], n3, n4 * n5, n4 * n5);
        opmm.Process(0, 0, 0, n3, compBatch * n5, n3);
        idx += compBatch;
    }
    opmm.FreeAReside();
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft5Dft4(
    GlobalTensor<float> &twiddleGlobalTmp, int32_t n3, GlobalTensor<float> &workGlobal, int l2Idx, int stepN,
    uint64_t col64, GlobalTensor<float> &xGlobal, int l2Step, int offset, uint64_t totalN, int32_t n5,
    int blockNum, int blockIdx, CMatmulCustom<float> &opmm, int32_t n4, uint64_t halfN)
{
    auto twiddleGlobalTmp2 = twiddleGlobalTmp[static_cast<uint64_t>(n3) * static_cast<uint64_t>(n3) * 3ULL];
    auto workGlobalTmp = workGlobal[static_cast<uint64_t>(l2Idx) * static_cast<uint64_t>(stepN) * col64];
    auto xGlobalTmp = xGlobal[static_cast<uint64_t>(l2Idx % (l2Step * 2)) * static_cast<uint64_t>(stepN) * col64 + static_cast<uint64_t>(offset) * totalN];
    int ceilTo = 128 / n5;
    int averageNum = n3 * stepN / ceilTo / blockNum;
    int remainderNum = n3 * stepN / ceilTo % blockNum;
    int startIdx = (blockIdx * averageNum + min(blockIdx, remainderNum)) * ceilTo;
    int calcNum = (averageNum + (blockIdx < remainderNum)) * ceilTo;
    int idx = 0;
    for (int i3 = startIdx / stepN; idx < calcNum; ++i3) {
        opmm.CopyInAReside(twiddleGlobalTmp2[static_cast<uint64_t>(i3) * static_cast<uint64_t>(n4) * static_cast<uint64_t>(n4) * 3ULL], n4, n4);
        for (int i1 = !idx ? startIdx % stepN : 0; i1 < stepN; ) {
            int compBatch = min(stepN - i1, calcNum - idx);
            compBatch = min(compBatch, 8192 / n4 / n5);
            uint64_t inputOffset = static_cast<uint64_t>(i1) * col64
                                 + static_cast<uint64_t>(i3) * static_cast<uint64_t>(n4) * static_cast<uint64_t>(n5);
            uint64_t outputOffset = (static_cast<uint64_t>(i3) * static_cast<uint64_t>(n4) * static_cast<uint64_t>(stepN)
                                   + static_cast<uint64_t>(i1)) * static_cast<uint64_t>(n5);
            opmm.SetMatrix(xGlobalTmp[inputOffset], xGlobalTmp[inputOffset + halfN], workGlobalTmp[outputOffset], workGlobalTmp[outputOffset + halfN], n4, n5, stepN * n5, n3 * n4 * n5, n5);
            opmm.Process(0, 0, 0, n4, n5, n4, compBatch);
            i1 += compBatch;
            idx += compBatch;
            if (idx == calcNum) break;
        }
        opmm.FreeAReside();
    }
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft5Dft5(
    GlobalTensor<float> &twiddleGlobalTmp, int32_t n3, int32_t n4, GlobalTensor<float> &workGlobal2,
    int l2Idx, int stepN, GlobalTensor<float> &workGlobal, uint64_t col64, int32_t n5, int blockNum,
    int blockIdx, CMatmulCustom<float> &opmm, uint64_t row64, uint64_t halfN, int32_t n1, int32_t n2)
{
    auto twiddleGlobalTmp2 = twiddleGlobalTmp[(static_cast<uint64_t>(n3) * static_cast<uint64_t>(n3) + static_cast<uint64_t>(n3) * static_cast<uint64_t>(n4) * static_cast<uint64_t>(n4)) * 3ULL];
    auto workGlobal2Tmp = workGlobal2[static_cast<uint64_t>(l2Idx) * static_cast<uint64_t>(stepN)];
    auto workGlobalTmp = workGlobal[static_cast<uint64_t>(l2Idx) * static_cast<uint64_t>(stepN) * col64];
    int ceilTo = 128 / n5;
    int averageNum = n3 * n4 / ceilTo / blockNum;
    int remainderNum = n3 * n4 / ceilTo % blockNum;
    int startIdx = (blockIdx * averageNum + min(blockIdx, remainderNum)) * ceilTo;
    int calcNum = (averageNum + (blockIdx < remainderNum)) * ceilTo;
    int idx = 0;
    opmm.SetFMatrix(n5, n5);
    for (int i3 = startIdx / n4; idx < calcNum; ++i3) {
        for (int i4 = !idx ? startIdx % n4 : 0; i4 < n4; ) {
            int compBatch = min(n4 - i4, calcNum - idx);
            compBatch = min(compBatch, 8192 / n5 / stepN);

            uint64_t transformIndex = static_cast<uint64_t>(i3) * static_cast<uint64_t>(n4) + static_cast<uint64_t>(i4);
            uint64_t matrixSize = static_cast<uint64_t>(n5) * static_cast<uint64_t>(n5);
            auto twiddleGlobalTmp3 = twiddleGlobalTmp2[transformIndex * matrixSize * 3ULL];
            uint64_t inputOffset = transformIndex * static_cast<uint64_t>(stepN) * static_cast<uint64_t>(n5);
            uint64_t outputOffset = (static_cast<uint64_t>(i4) * static_cast<uint64_t>(n3) + static_cast<uint64_t>(i3)) * row64;
            opmm.SetMatrix(twiddleGlobalTmp3, twiddleGlobalTmp3[matrixSize], twiddleGlobalTmp3[matrixSize * 2ULL], workGlobalTmp[inputOffset], workGlobalTmp[inputOffset + halfN], workGlobal2Tmp[outputOffset], workGlobal2Tmp[outputOffset + halfN], n5, n5, n1 * n2 * n3 * n4, n5 * stepN, n1 * n2 * n3);
            opmm.ProcessWithTranspose2(0, 0, 0, n5, stepN, n5, compBatch);
            i4 += compBatch;
            idx += compBatch;
            if (idx == calcNum) break;
        }
    }
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft5FirstStages(
    int col, uint64_t row64, CMatmulCustom<float> &opmm, GlobalTensor<float> &twiddleGlobalTmp, int32_t n1,
    GlobalTensor<float> &workGlobal2, int32_t n2, int blockNum, int blockIdx, uint64_t halfN,
    GlobalTensor<float> &workGlobal, uint64_t col64)
{
    int stepN = 128;
    int l2Step = min(col / stepN, static_cast<int>(4194304ULL / row64 / static_cast<uint64_t>(stepN)));

    for (int l2Idx = 0; l2Idx < col / stepN; l2Idx += l2Step) {

        CrossCoreWaitFlag(0x1);
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);

        {
            ComputeFft5Dft1(opmm, twiddleGlobalTmp, n1, workGlobal2, l2Step, n2, blockNum, blockIdx, row64, stepN,
                halfN);
        }

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);

        {
            ComputeFft5Dft2(twiddleGlobalTmp, n1, workGlobal2, workGlobal, l2Idx, stepN, l2Step, blockNum, blockIdx,
                opmm, n2, row64, col64, halfN, col);
        }

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);
        if (l2Idx + l2Step < col / stepN) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
        }
    }
}

#endif

#ifdef __DAV_C220_CUBE__
__aicore__ inline void ComputeFft5LastStages(
    int row, GlobalTensor<float> &xGlobal, uint64_t col64, int offset, uint64_t totalN, int32_t n5,
    int32_t n4, int blockNum, int blockIdx, CMatmulCustom<float> &opmm,
    GlobalTensor<float> &twiddleGlobalTmp, int32_t n3, uint64_t halfN, GlobalTensor<float> &workGlobal,
    GlobalTensor<float> &workGlobal2, uint64_t row64, int32_t n1, int32_t n2)
{
    int stepN = 128;
    int l2Step = 1;

    for (int l2Idx = 0; l2Idx < row / stepN; l2Idx += l2Step) {

        CrossCoreWaitFlag(0x1);

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);

        {
            ComputeFft5Dft3(xGlobal, l2Idx, l2Step, stepN, col64, offset, totalN, n5, n4, blockNum, blockIdx, opmm,
                twiddleGlobalTmp, n3, halfN);
        }

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);

        {
            ComputeFft5Dft4(twiddleGlobalTmp, n3, workGlobal, l2Idx, stepN, col64, xGlobal, l2Step, offset, totalN,
                n5, blockNum, blockIdx, opmm, n4, halfN);
        }

        CrossCoreSetFlag<0x0, PIPE_FIX>(0x7);
        CrossCoreWaitFlag(0x7);

        {
            ComputeFft5Dft5(twiddleGlobalTmp, n3, n4, workGlobal2, l2Idx, stepN, workGlobal, col64, n5, blockNum,
                blockIdx, opmm, row64, halfN, n1, n2);
        }

        if (static_cast<uint64_t>(l2Idx) + static_cast<uint64_t>(l2Step) * (1ULL + (col64 * static_cast<uint64_t>(stepN) <= 4194304ULL)) < row64 / static_cast<uint64_t>(stepN)) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
        }
    }
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
}

#endif

__aicore__ inline void RunFft5_r2cCube(
    int32_t batch, int32_t n1, int32_t n2, int32_t n3, int32_t n4, int32_t n5, int blockNum, int blockIdx,
    GlobalTensor<float> &xGlobal, GlobalTensor<float> &workGlobal, GlobalTensor<float> &workGlobal2,
    GlobalTensor<float> &twiddleGlobal, TPipe &pipe, uint64_t totalN, uint64_t halfN, uint64_t row64,
    uint64_t col64, int row, int col)
{
    #ifdef  __DAV_C220_CUBE__
        {
            int stepN = 128;
            int l2Step = 1;
            int l2Blocks = row / stepN / l2Step;
            twiddleGlobal = twiddleGlobal[col64 * 2ULL * static_cast<uint64_t>(blockNum * l2Blocks + 1) + (blockNum * 2) * 4096 * 2];
        }
        CMatmulCustom<float> opmm;
        opmm.Init(&pipe);

        for (int offset = 0; offset < batch; ++offset) {
            auto twiddleGlobalTmp = twiddleGlobal;

            {
                ComputeFft5FirstStages(col, row64, opmm, twiddleGlobalTmp, n1, workGlobal2, n2, blockNum, blockIdx,
                    halfN, workGlobal, col64);
            }

            twiddleGlobalTmp = twiddleGlobalTmp[static_cast<uint64_t>(n1) * static_cast<uint64_t>(n1) * 3ULL];

            CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);

            twiddleGlobalTmp = twiddleGlobalTmp[static_cast<uint64_t>(n2) * static_cast<uint64_t>(n2) * static_cast<uint64_t>(n1) * 3ULL];

            {
                ComputeFft5LastStages(row, xGlobal, col64, offset, totalN, n5, n4, blockNum, blockIdx, opmm,
                    twiddleGlobalTmp, n3, halfN, workGlobal, workGlobal2, row64, n1, n2);
            }
        }
    #endif
}

__aicore__ inline void InitializeFft5Twiddle(
    int row, GlobalTensor<float> &unpackTwiddleGlobal, GlobalTensor<float> &twiddleGlobal,
    uint64_t col64, int blockNum, int32_t n1, int32_t n2, int32_t n3, int32_t n4, int32_t n5,
    GlobalTensor<float> &sTwiddleGlobal)
{
    int stepN = 128;
    int l2Step = 1;
    int l2Blocks = row / stepN / l2Step;
    unpackTwiddleGlobal = twiddleGlobal[col64 * 2ULL * static_cast<uint64_t>(blockNum * l2Blocks + 1)];

    uint64_t matFloats = (static_cast<uint64_t>(n1) * n1 + static_cast<uint64_t>(n2) * n2 * n1 +
    static_cast<uint64_t>(n3) * n3 + static_cast<uint64_t>(n4) * n4 * n3 +
    static_cast<uint64_t>(n5) * n5 * n4 * n3) * 3ULL;
    uint64_t sBase = col64 * 2ULL * static_cast<uint64_t>(blockNum * l2Blocks + 1) +
    static_cast<uint64_t>(blockNum * 2) * 4096ULL * 2ULL + matFloats;
    sTwiddleGlobal = twiddleGlobal[sBase];
}

extern "C" __global__ __aicore__ void fft5_r2c(
    __gm__ uint8_t * __restrict__ X_gm,
    __gm__ uint8_t * __restrict__ Y_gm,
    __gm__ uint8_t * __restrict__ work_gm,
    __gm__ uint8_t * __restrict__ twiddle_gm,
    __gm__ uint8_t * __restrict__ gather_gm,
    __gm__ uint8_t * __restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Rfft1DR2CCommonKernel::KernelTiling<5> tiling(tiling_para_gm);
    int32_t N = tiling.N;
    int32_t batch = tiling.batch;
    int32_t n1 = tiling.n1;
    int32_t n2 = tiling.n2;
    int32_t n3 = tiling.n3;
    int32_t n4 = tiling.n4;
    int32_t n5 = tiling.n5;

    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();

    GlobalTensor<float> xGlobal, yGlobal, workGlobal, workGlobal2, twiddleGlobal, unpackTwiddleGlobal, eyeGlobal, sTwiddleGlobal;
    GlobalTensor<uint32_t> gatherGlobal;
    xGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(X_gm);
    yGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(Y_gm);
    workGlobal2 = Rfft1DR2CCommonKernel::BindGlobal<float>(work_gm);
    twiddleGlobal = Rfft1DR2CCommonKernel::BindGlobal<float>(twiddle_gm);
    gatherGlobal = Rfft1DR2CCommonKernel::BindGlobal<uint32_t>(gather_gm);

    uint64_t totalN = static_cast<uint64_t>(N);
    uint64_t halfN = totalN / 2;
    uint64_t outputN = totalN + 2;
    uint64_t row64 = static_cast<uint64_t>(n1) * static_cast<uint64_t>(n2);
    uint64_t col64 = static_cast<uint64_t>(n3) * static_cast<uint64_t>(n4) * static_cast<uint64_t>(n5);
    int row = static_cast<int>(row64);
    int col = static_cast<int>(col64);

    {
        InitializeFft5Twiddle(row, unpackTwiddleGlobal, twiddleGlobal, col64, blockNum, n1, n2, n3, n4, n5,
            sTwiddleGlobal);
    }
    workGlobal = workGlobal2[totalN];

    TPipe pipe;
    RunFft5_r2cVector(N, batch, blockNum, blockIdx, xGlobal, yGlobal, workGlobal, workGlobal2, twiddleGlobal,
        unpackTwiddleGlobal, sTwiddleGlobal, gatherGlobal, pipe, totalN, halfN, outputN, row64, col64, row, col);

    RunFft5_r2cCube(batch, n1, n2, n3, n4, n5, blockNum, blockIdx, xGlobal, workGlobal, workGlobal2, twiddleGlobal,
        pipe, totalN, halfN, row64, col64, row, col);
}

}

#endif
