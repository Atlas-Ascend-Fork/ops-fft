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

#ifndef RFFT1_D_R2C_CMATMUL_COMPUTE_H
#define RFFT1_D_R2C_CMATMUL_COMPUTE_H

#include "rfft1_d_r2c_cmatmul.h"

namespace Rfft1DR2CCommonKernel {

template<typename T>
__aicore__ inline void CMatmulCustom<T>::ProcessReal(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int batch) {
    int mBlocks, nBlocks, kBlocks;
    if (!InitBlockTiling(blockM, blockN, blockK, batch, mBlocks, nBlocks, kBlocks)) return;

    int cnt;

    for (int i = 0; i < nBlocks; ++i) {
        tailN = min(singleCoreN - i * baseN, baseN);
        for (int j = 0; j < mBlocks; ++j) {
            tailM = min(singleCoreM - j * baseM, baseM);
            int offsetC_1 = offsetC + j * baseM * strideC + i * baseN;
            LocalTensor<T> c1RealLocal = outBufCO1Real.Get<T>();

            for (int k = 0; k < kBlocks; k += stepK) {
                tailK = baseK;
                cnt = min(kBlocks - k, stepK);
                int offsetA_1 = offsetA + j * baseM * strideA + k * baseK;
                int offsetB_1 = offsetB + k * baseK * strideB + i * baseN;

                CopyInB<ComplexPart::Real>(offsetB_1, cnt, batch);

                LocalTensor<T> b1RealLocal = inQueueB1.DeQue<T>();
                // C1 += A1B1
                for (int x = 0; x < cnt; ++x) {
                    tailK = min(singleCoreK - (k + x) * baseK, baseK);
                    SplitA(a1Reside[3], x);
                    SplitB(b1RealLocal, x, batch);
                    LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
                    LocalTensor<T> b2Local = inQueueB2.DeQue<T>();
                    Compute(a2Local, b2Local, c1RealLocal, !k && !x, k + x == kBlocks - 1, batch);
                    inQueueA2.FreeTensor(a2Local);
                    inQueueB2.FreeTensor(b2Local);
                }
                inQueueB1.FreeTensor(b1RealLocal);
            }
            CopyOut<ComplexPart::Real>(c1RealLocal, offsetC_1, batch);
        }
    }
}

template<typename T>
template<typename CMatmulCustom<T>::ProcessMode mode>
__aicore__ inline void CMatmulCustom<T>::AccumulateRealB(LocalTensor<T> &a1RealLocal, LocalTensor<T> &a1ImagLocal,
                                      LocalTensor<T> &b1RealLocal, LocalTensor<T> &c1RealLocal,
                                      LocalTensor<T> &c1ImagLocal, int k, int cnt, int batch) {
    for (int x = 0; x < cnt; ++x) {
        tailK = min(singleCoreK - (k + x) * baseK, baseK);
        SplitAForMode<mode>(a1RealLocal, x, batch);
        SplitBForMode<mode>(b1RealLocal, x, batch);
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        LocalTensor<T> b2Local = inQueueB2.DeQue<T>();
        ComputeForMode<mode>(a2Local, b2Local, c1RealLocal, !k && !x, 0, batch);
        inQueueA2.FreeTensor(a2Local);
        SplitAForMode<mode>(a1ImagLocal, x, batch);
        a2Local = inQueueA2.DeQue<T>();
        ComputeForMode<mode>(a2Local, b2Local, c1ImagLocal, !k && !x, 0, batch);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
}

template<typename T>
template<typename CMatmulCustom<T>::ProcessMode mode>
__aicore__ inline void CMatmulCustom<T>::AccumulateImaginaryB(LocalTensor<T> &a1RealLocal, LocalTensor<T> &c1RealLocal,
                                           LocalTensor<T> &c1ImagLocal, int k, int cnt,
                                           int kBlocks, int offsetC_1, int batch) {
    LocalTensor<T> a1ImagNegLocal;
    if constexpr (mode == ProcessMode::BatchedA) a1ImagNegLocal = inQueueA1.DeQue<T>();
    else a1ImagNegLocal = a1Reside[2];
    LocalTensor<T> b1ImagLocal = inQueueB1.DeQue<T>();
    // C1 -= A2B2
    LocalTensor<T> a2Local, b2Local;
    for (int x = 0; x < cnt; ++x) {
        tailK = min(singleCoreK - (k + x) * baseK, baseK);
        SplitAForMode<mode>(a1ImagNegLocal, x, batch);
        SplitBForMode<mode>(b1ImagLocal, x, batch);
        a2Local = inQueueA2.DeQue<T>();
        b2Local = inQueueB2.DeQue<T>();
        ComputeForMode<mode>(a2Local, b2Local, c1RealLocal, 0, k + x == kBlocks - 1, batch);
        inQueueA2.FreeTensor(a2Local);
        if (x + 1 != cnt) inQueueB2.FreeTensor(b2Local);
    }
    if constexpr (mode == ProcessMode::BatchedA) inQueueA1.FreeTensor(a1ImagNegLocal);
    if (k + stepK >= kBlocks) {
        CopyOut<ComplexPart::Real>(c1RealLocal, offsetC_1, batch);
    }
    // C2 += A1B2
    for (int x = cnt - 1; x >= 0; --x) {
        SplitAForMode<mode>(a1RealLocal, x, batch);
        if (x + 1 != cnt) SplitBForMode<mode>(b1ImagLocal, x, batch);
        a2Local = inQueueA2.DeQue<T>();
        if (x + 1 != cnt) b2Local = inQueueB2.DeQue<T>();
        ComputeForMode<mode>(a2Local, b2Local, c1ImagLocal, 0, k + cnt - 1 - x == kBlocks - 1, batch);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    if constexpr (mode == ProcessMode::BatchedA) inQueueA1.FreeTensor(a1RealLocal);
    inQueueB1.FreeTensor(b1ImagLocal);
}

template<typename T>
template<typename CMatmulCustom<T>::ProcessMode mode>
__aicore__ inline void CMatmulCustom<T>::ProcessComplex(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int batch) {
    int mBlocks, nBlocks, kBlocks;
    if (!InitBlockTiling(blockM, blockN, blockK, batch, mBlocks, nBlocks, kBlocks)) return;

    int cnt;

    for (int i = 0; i < nBlocks; ++i) {
        tailN = min(singleCoreN - i * baseN, baseN);
        for (int j = 0; j < mBlocks; ++j) {
            tailM = min(singleCoreM - j * baseM, baseM);
            int offsetC_1 = offsetC + j * baseM * strideC + i * baseN;
            LocalTensor<T> c1RealLocal = outBufCO1Real.Get<T>();
            LocalTensor<T> c1ImagLocal = outBufCO1Imag.Get<T>();

            for (int k = 0; k < kBlocks; k += stepK) {
                tailK = baseK;
                cnt = min(kBlocks - k, stepK);
                int offsetA_1 = offsetA + j * baseM * strideA + k * baseK;
                int offsetB_1 = offsetB + k * baseK * strideB + i * baseN;

                if constexpr (mode == ProcessMode::BatchedA) {
                    CopyInA<ComplexPart::Real>(offsetA_1, cnt, batch);
                    CopyInA<ComplexPart::Imag>(offsetA_1, cnt, batch);
                    CopyInA<ComplexPart::ImagNeg>(offsetA_1, cnt, batch);
                }
                CopyInBForMode<mode, ComplexPart::Real>(offsetB_1, cnt, batch);
                CopyInBForMode<mode, ComplexPart::Imag>(offsetB_1, cnt, batch);

                LocalTensor<T> b1RealLocal = inQueueB1.DeQue<T>();
                LocalTensor<T> a1RealLocal, a1ImagLocal;
                if constexpr (mode == ProcessMode::BatchedA) {
                    a1RealLocal = inQueueA1.DeQue<T>();
                    a1ImagLocal = inQueueA1.DeQue<T>();
                } else {
                    a1RealLocal = a1Reside[0];
                    a1ImagLocal = a1Reside[1];
                }
                // C1 += A1B1
                // C2 += A2B1
                AccumulateRealB<mode>(a1RealLocal, a1ImagLocal, b1RealLocal, c1RealLocal,
                                      c1ImagLocal, k, cnt, batch);
                if constexpr (mode == ProcessMode::BatchedA) inQueueA1.FreeTensor(a1ImagLocal);
                inQueueB1.FreeTensor(b1RealLocal);

                AccumulateImaginaryB<mode>(a1RealLocal, c1RealLocal, c1ImagLocal, k, cnt,
                                           kBlocks, offsetC_1, batch);
            }
            CopyOut<ComplexPart::Imag>(c1ImagLocal, offsetC_1, batch);
        }
    }
}

template<typename T>
template<DFTType dt>
__aicore__ inline void CMatmulCustom<T>::Compute(const LocalTensor<T> &a2Local, const LocalTensor<T> &b2Local, const LocalTensor<T> &c1Local, int first, const int unitFlag, int batch) {
    MmadParams mmadParams;
    if constexpr (dt == DFTType::Col) {
        mmadParams.m = tailM;
        mmadParams.n = tailN * batch;
    } else {
        mmadParams.m = tailM * batch;
        mmadParams.n = tailN;
    }
    mmadParams.k = tailK;
    mmadParams.unitFlag = unitFlag ? 3 : 2;
    if (first) {
        Mmad(c1Local, a2Local, b2Local, mmadParams);
    } else {
        mmadParams.cmatrixInitVal = false;
        Mmad(c1Local, a2Local, b2Local, c1Local, mmadParams);
    }
    if (tailM * tailN * batch >> 8 < 10)
        PipeBarrier<PIPE_M>();
}

template<typename T>
template<bool BatchedA>
__aicore__ inline void CMatmulCustom<T>::ComputeBatched(const LocalTensor<T> &a2Local, const LocalTensor<T> &b2Local, const LocalTensor<T> &c1Local, int first, const int unitFlag, int batch) {
    MmadParams mmadParams;
    mmadParams.m = tailM;
    mmadParams.n = tailN;
    mmadParams.k = tailK;
    mmadParams.unitFlag = unitFlag ? 3 : 2;
    for (int i = 0; i < batch; ++i) {
        if (first) {
            Mmad(c1Local[i * tailM * tailN], a2Local[BatchedA ? i * tailM * tailK : 0], b2Local[i * tailK * tailN], mmadParams);
        } else {
            mmadParams.cmatrixInitVal = false;
            Mmad(c1Local[i * tailM * tailN], a2Local[BatchedA ? i * tailM * tailK : 0], b2Local[i * tailK * tailN], c1Local, mmadParams);
        }
        if (tailM * tailN >> 8 < 10)
            PipeBarrier<PIPE_M>();
    }
}

template<typename T>
__aicore__ inline void CMatmulCustom<T>::Compute2(const LocalTensor<T> &a, const LocalTensor<T> &b,
                                 const LocalTensor<T> &c, int first, int unitFlag, int batch) {
    ComputeBatched<true>(a, b, c, first, unitFlag, batch);
}

template<typename T>
__aicore__ inline void CMatmulCustom<T>::Compute3(const LocalTensor<T> &a, const LocalTensor<T> &b,
                                 const LocalTensor<T> &c, int first, int unitFlag, int batch) {
    ComputeBatched<false>(a, b, c, first, unitFlag, batch);
}

} // namespace Rfft1DR2CCommonKernel

#endif // RFFT1_D_R2C_CMATMUL_COMPUTE_H
