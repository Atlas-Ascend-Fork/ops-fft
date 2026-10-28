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


#ifndef RFFT1_D_R2C_FFT1_KERNEL_H
#define RFFT1_D_R2C_FFT1_KERNEL_H

#include "kernel/fft_common_kernel.h"
#include <stdint.h>

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

namespace Rfft1DR2CFft1Kernel {

using namespace::AscendC;

template<typename T>
class MatmulCustom {
    GlobalTensor<T> aGlobal, bGlobal, cGlobal;
    TQue<QuePosition::A1, 2> inQueueA1;
    TQue<QuePosition::A2, 2> inQueueA2;
    TQue<QuePosition::B1, 2> inQueueB1;
    TQue<QuePosition::B2, 2> inQueueB2;
    TBuf<TPosition::CO1> outBufCO1;
    int stepK;
    int strideA, strideB, strideC;
    int baseM, baseN, baseK;
    int realM, realN, realK;
    int alignedK;
public:
    __aicore__ inline MatmulCustom() {}
    __aicore__ inline void Init(TPipe *pipe) {
        this->stepK = 4;

        pipe->InitBuffer(inQueueA1, 2, 131072);
        pipe->InitBuffer(inQueueB1, 2, 131072);
        pipe->InitBuffer(inQueueA2, 2, 32768);
        pipe->InitBuffer(inQueueB2, 2, 32768);
        pipe->InitBuffer(outBufCO1, 131072);
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aGlobal, GlobalTensor<T> bGlobal, GlobalTensor<T> cGlobal, int strideA, int strideB, int strideC) {
        this->aGlobal = aGlobal;
        this->bGlobal = bGlobal;
        this->cGlobal = cGlobal;
        this->strideA = strideA;
        this->strideB = strideB;
        this->strideC = strideC;
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int realM, int realN, int realK) {
        this->realM = realM;
        this->realN = realN;
        this->realK = realK;

        int blockM = (realM + 127) / 128 * 128;
        int blockN = (realN - 1) * 2;
        int blockK = realK;
        
        // default: (256, 128, 32)
        baseN = min(blockN, 256);
        baseK = min(blockK, 32);
        baseM = min(blockM, 128);
        baseM = min(baseM, min(32768 / baseN, 8192 / baseK));

        alignedK = max(baseK, 8);

        int mBlocks = (blockM + baseM - 1) / baseM;
        int nBlocks = (blockN + baseN - 1) / baseN;
        int kBlocks = (blockK + baseK - 1) / baseK;
        static constexpr uint8_t padList[]{0, 0, 0, 0};

        int cnt = min(stepK, kBlocks);

        CopyInB(offsetB, cnt, realN);

        LocalTensor<T> b1Local = inQueueB1.DeQue<T>();

        for (int i = 0; i < nBlocks; ++i) {
            int curN = min(realN - i * baseN, baseN);
            int curNAligned = (curN + 15) / 16 * 16;
            for (int j = 0; j < mBlocks; ++j) {
                int curM = min(realM - j * baseM, baseM);
                int curMAligned = (curM + 15) / 16 * 16;
                int offsetC_1 = offsetC + j * baseM * strideC + i * baseN;
                LocalTensor<T> c1Local = outBufCO1.Get<T>();
                Load3DSetFMatrixCal(1, curMAligned, padList);

                for (int k = 0; k < kBlocks; k += stepK) {
                    cnt = min(kBlocks - k, stepK);
                    Load3DSetFMatrixBCal(1, alignedK * cnt, padList);

                    int offsetA_1 = offsetA + j * baseM * strideA + k * baseK;
                    int offsetB_1 = offsetB + k * baseK * strideB + i * baseN;

                    CopyInA(offsetA_1, cnt, curM);
                    
                    LocalTensor<T> a1Local = inQueueA1.DeQue<T>();

                    // C1 += A1B1
                    for (int x = 0; x < cnt; ++x) {
                        SplitB(b1Local, x, curN);
                        SplitA(a1Local, x, curM);
                        Compute(c1Local, !k && !x, k + x == kBlocks - 1, curM, curN);
                    }
                    inQueueA1.FreeTensor(a1Local);
                }
                CopyOut(c1Local, offsetC_1, curM, curN);
            }
        }
        inQueueB1.FreeTensor(b1Local);
    }
private:
    __aicore__ inline void CopyInA(int offsetA_1, int cnt, int curM) {
        int curMAligned = (curM + 15) / 16 * 16;
        LocalTensor<T> a1Local = inQueueA1.AllocTensor<T>();
        Nd2NzParams intriParamsMatA {
            1,
            static_cast<uint16_t>(curM),
            static_cast<uint16_t>(alignedK * cnt),
            0,
            static_cast<uint16_t>(strideA),
            static_cast<uint16_t>(curMAligned),
            1,
            1
        };
        DataCopy(a1Local, aGlobal[offsetA_1], intriParamsMatA);
        inQueueA1.EnQue(a1Local);
    }

    __aicore__ inline void CopyInB(int offsetB_1, int cnt, int curN) {
        int curNAligned = (curN + 15) / 16 * 16;
        LocalTensor<T> b1Local = inQueueB1.AllocTensor<T>();
        Nd2NzParams intriParamsMatB {
            1,
            static_cast<uint16_t>(alignedK * cnt),
            static_cast<uint16_t>(curNAligned),
            0,
            static_cast<uint16_t>(strideB),
            static_cast<uint16_t>(alignedK * cnt),
            1,
            1
        };
        DataCopy(b1Local, bGlobal[offsetB_1], intriParamsMatB);
        inQueueB1.EnQue(b1Local);
    }

    __aicore__ inline void SplitA(const LocalTensor<T> &a1Local, int splitIdx, int curM) {
        int curMAligned = (curM + 15) / 16 * 16;
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        LoadData3DParamsV2Pro loadData3DV2;
        uint16_t mPos = 0, sAL1NOffset_ = 0;
        loadData3DV2.channelSize = alignedK;
        loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) |
                                ((uint64_t)curMAligned << 16) | (uint64_t)alignedK;
        LoadData<T>(a2Local, a1Local[splitIdx * curMAligned * alignedK], loadData3DV2);
        inQueueA2.EnQue(a2Local);
    }

    __aicore__ inline void SplitB(const LocalTensor<T> &b1Local, int splitIdx, int curN) {
        int curNAligned = (curN + 15) / 16 * 16;
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
        LoadData3DParamsV2Pro loadData3DV2;
        uint16_t mPos = 0, sBL1NOffset_ = 0;
        loadData3DV2.channelSize = curNAligned;
        loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sBL1NOffset_ << 32) |
                                ((uint64_t)alignedK << 16) | (uint64_t)curNAligned;
        loadData3DV2.fMatrixCtrl = true;
        LoadData<T>(b2Local, b1Local[splitIdx * alignedK * 8], loadData3DV2);
        inQueueB2.EnQue(b2Local);
    }

    __aicore__ inline void Compute(const LocalTensor<T> &c1Local, int first, const int unitFlag, int curM, int curN) {
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        LocalTensor<T> b2Local = inQueueB2.DeQue<T>();
        int curMAligned = (curM + 15) / 16 * 16;
        int curNAligned = (curN + 15) / 16 * 16;
        MmadParams mmadParams;
        mmadParams.m = curMAligned;
        mmadParams.n = curNAligned;
        mmadParams.k = baseK;
        mmadParams.unitFlag = unitFlag ? 3 : 2;
        if (first) {
            Mmad(c1Local, a2Local, b2Local, mmadParams);
        } else {
            mmadParams.cmatrixInitVal = false;
            Mmad(c1Local, a2Local, b2Local, c1Local, mmadParams);
        }
        if (baseM * baseN < 2560) 
            PipeBarrier<PIPE_M>();
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOut(const LocalTensor<T> &c1Local, int offsetC_1, int curM, int curN) {
        int curMAligned = (curM + 15) / 16 * 16;
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = realN;
        fixpipeParams.mSize = curM;
        fixpipeParams.srcStride = curMAligned;
        fixpipeParams.dstStride = strideC;

        fixpipeParams.unitFlag = 3;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        Fixpipe(cGlobal[offsetC_1], c1Local, fixpipeParams);
    }
};

template<typename T>
class FFTCustom {
    GlobalTensor<T> xGlobal;
    GlobalTensor<T> yGlobal;
    TQue<QuePosition::VECIN, 4> inQueueSrc;
    TQue<QuePosition::VECOUT, 2> outQueueDst;
    LocalTensor<uint32_t> gatherLocal, gatherLocal2;
    int elementsPerBlock;
    static constexpr int TILE_LENGTH = 4096;
public:
    __aicore__ inline FFTCustom() {}
    __aicore__ inline void Init(TPipe *pipe) {
        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(inQueueSrc, 4, TILE_LENGTH * 2 * sizeof(T));
        pipe->InitBuffer(outQueueDst, 2, TILE_LENGTH * 2 * sizeof(T));
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> xGlobal, GlobalTensor<T> yGlobal) {
        this->xGlobal = xGlobal;
        this->yGlobal = yGlobal;
    }
    __aicore__ inline void LoadGather(GlobalTensor<uint32_t> gatherGlobal) {
        gatherLocal = inQueueSrc.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, gatherGlobal, TILE_LENGTH * 2);
        inQueueSrc.EnQue(gatherLocal);
        gatherLocal = inQueueSrc.DeQue<uint32_t>();

        gatherLocal2 = inQueueSrc.AllocTensor<uint32_t>();
        DataCopy(gatherLocal2, gatherGlobal[TILE_LENGTH * 2], TILE_LENGTH * 2);
        inQueueSrc.EnQue(gatherLocal2);
        gatherLocal2 = inQueueSrc.DeQue<uint32_t>();
    }
    __aicore__ inline void FreeGather() {
        inQueueSrc.FreeTensor(gatherLocal);
        inQueueSrc.FreeTensor(gatherLocal2);
    }
    __aicore__ inline void Process(int m, int n, int blockNum, int blockIdx) {
        int averageNum = m / blockNum;
        int remainderNum = m % blockNum;
        int startIdx = (blockIdx * averageNum + min(blockIdx, remainderNum));
        int calcNum = (averageNum + (blockIdx < remainderNum));
        int compM = TILE_LENGTH / n;
        for (int i = startIdx; i < startIdx + calcNum; ) {
            int calcM = min(startIdx + calcNum - i, TILE_LENGTH / n);
            LocalTensor<T> xLocal = inQueueSrc.AllocTensor<T>();
            int calcMCeil = (calcM + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
            DataCopy(xLocal, xGlobal[i * n], calcMCeil * n);
            inQueueSrc.EnQue(xLocal);
            
            xLocal = inQueueSrc.DeQue<T>();
            LocalTensor<T> yLocal = outQueueDst.AllocTensor<T>();
            LocalTensor<T> tmpLocal = outQueueDst.AllocTensor<T>();
            
            if (n == 2) {
                Gather(yLocal, xLocal, gatherLocal, 0u, TILE_LENGTH);
                inQueueSrc.FreeTensor(xLocal);
                Add(tmpLocal, yLocal, yLocal[compM], compM);
                Sub(tmpLocal[compM], yLocal, yLocal[compM], compM);
                Duplicate(tmpLocal[compM * 2], 0.f, compM * 2);
            } else {
                Gather(tmpLocal, xLocal, gatherLocal, 0u, TILE_LENGTH);
                inQueueSrc.FreeTensor(xLocal);
                Add(yLocal, tmpLocal, tmpLocal[compM * 2], compM * 2); // a = x0 + x2, c = x1 + x3
                Sub(tmpLocal[compM * 4], tmpLocal[compM * 3], tmpLocal[compM], compM); // d = x3 - x1, X1 = b + 1j * d
                Sub(tmpLocal[compM], tmpLocal, tmpLocal[compM * 2], compM); // b = x0 - x2, X1 = b + 1j * d
                Add(tmpLocal, yLocal, yLocal[compM], compM); // X0 = a + c
                Sub(tmpLocal[compM * 2], yLocal, yLocal[compM], compM); // X2 = a - c
                Duplicate(tmpLocal[compM * 3], 0.f, compM);
                Duplicate(tmpLocal[compM * 5], 0.f, compM);
            }

            Gather(yLocal, tmpLocal, gatherLocal2, 0u, calcM * (n + 2));
            outQueueDst.FreeTensor(tmpLocal);
            outQueueDst.EnQue(yLocal);

            yLocal = outQueueDst.DeQue<T>();
            DataCopyExtParams CopyOutParams {
                1,
                static_cast<uint32_t>(calcM * (n + 2) * sizeof(T)),
                0,
                0,
                0
            };
            DataCopyPad(yGlobal[i * (n + 2)], yLocal, CopyOutParams);
            outQueueDst.FreeTensor(yLocal);
            i += calcM;
        }
    }
};


extern "C" __global__ __aicore__ void fft1_r2c(
    __gm__ uint8_t * __restrict__ X_gm,
    __gm__ uint8_t * __restrict__ Y_gm,
    __gm__ uint8_t * __restrict__ twiddle_gm,
    __gm__ uint8_t * __restrict__ gather_gm,
    __gm__ uint8_t * __restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // tiling 由 host 以 uint32 写入，读取端按同类型解读后再转 int32（issue #219）；
    // N/batch 非法则静默返回，防止 blockN = 常量/N 等有符号运算产生负值循环边界。
    auto tiling_para_u = reinterpret_cast<__gm__ uint32_t *>(tiling_para_gm);
    int32_t N = static_cast<int32_t>(tiling_para_u[0]);
    int32_t batch = static_cast<int32_t>(tiling_para_u[1]);
    if (N <= 0 || batch <= 0) {
        return;
    }

    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();

    GlobalTensor<float> xGlobal, yGlobal, twiddleGlobal;
    GlobalTensor<uint32_t> gatherGlobal;
    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(X_gm));
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Y_gm));
    twiddleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddle_gm));
    gatherGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather_gm));
    
    TPipe pipe;
    int halfN = N / 2;
    
    #ifdef __DAV_C220_VEC__
        if (N < 8) {
            FFTCustom<float> op;
            op.Init(&pipe);
            op.SetMatrix(xGlobal, yGlobal);
            op.LoadGather(gatherGlobal);
            op.Process(batch, N, blockNum << 1, blockIdx);
            op.FreeGather();
        }
    #endif

    #ifdef  __DAV_C220_CUBE__
        if (N >= 8) {
            MatmulCustom<float> opmm;
            opmm.Init(&pipe);

            int averageNum = batch / blockNum;
            int remainderNum = batch % blockNum;
            int startIdx = (blockIdx * averageNum + min(blockIdx, remainderNum));
            int calcNum = (averageNum + (blockIdx < remainderNum));

            opmm.SetMatrix(xGlobal[startIdx * N], twiddleGlobal, yGlobal[startIdx * (N + 2)], N, N * 2, N + 2);
            opmm.Process(0, 0, 0, calcNum, N + 1, N);
        }
    #endif
}

}

#endif
