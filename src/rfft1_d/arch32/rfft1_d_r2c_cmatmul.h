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

#ifndef RFFT1_D_R2C_CMATMUL_H
#define RFFT1_D_R2C_CMATMUL_H

#include "kernel/fft_common_kernel.h"
#include <stdint.h>

namespace Rfft1DR2CCommonKernel {

using namespace::AscendC;

enum class ComplexPart { Real, Imag, ImagNeg };
enum class DFTType { Row, Col };

template<typename T>
class CMatmulCustom {
    GlobalTensor<T> aRealGlobal, aImagGlobal, aImagNegGlobal;
    GlobalTensor<T> bRealGlobal, bImagGlobal;
    GlobalTensor<T> cRealGlobal, cImagGlobal;
    TQue<QuePosition::A1, 4> inQueueA1;
    TQue<QuePosition::A2, 2> inQueueA2;
    TQue<QuePosition::B1, 4> inQueueB1;
    TQue<QuePosition::B2, 2> inQueueB2;
    TBuf<TPosition::CO1> outBufCO1Real;
    TBuf<TPosition::CO1> outBufCO1Imag;
    LocalTensor<T> a1Reside[4];
    int strideA, strideB, strideC;
    int strideBatchB, strideBatchC;
    int singleCoreM, singleCoreN, singleCoreK;
    int baseM, baseN, baseK;
    int tailM, tailN, tailK;
    int stepM, stepN;
    int aligned_n;
    int alignToN;
    int stepK;
    int c0Size_;
public:
    __aicore__ inline CMatmulCustom() {}
    __aicore__ inline void Init(TPipe *pipe) {
        this->c0Size_ = GetC0Size<T>();
        this->stepK = 2;
        pipe->InitBuffer(inQueueA1, 4, 65536);
        pipe->InitBuffer(inQueueB1, 4, 65536);
        pipe->InitBuffer(inQueueA2, 2, 32768);
        pipe->InitBuffer(inQueueB2, 2, 32768);
        pipe->InitBuffer(outBufCO1Imag, 65536);
        pipe->InitBuffer(outBufCO1Real, 65536);
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> bRealGlobal, GlobalTensor<T> cRealGlobal, int strideA, int strideB, int strideC, int strideBatchB = 0, int strideBatchC = 0) {
        this->strideA = strideA;
        this->strideB = strideB;
        this->strideC = strideC;
        this->strideBatchB = strideBatchB;
        this->strideBatchC = strideBatchC;
        this->bRealGlobal = bRealGlobal;
        this->cRealGlobal = cRealGlobal;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> bRealGlobal, GlobalTensor<T> bImagGlobal, GlobalTensor<T> cRealGlobal, GlobalTensor<T> cImagGlobal, int strideA, int strideB, int strideC, int strideBatchB = 0, int strideBatchC = 0) {
        this->strideA = strideA;
        this->strideB = strideB;
        this->strideC = strideC;
        this->strideBatchB = strideBatchB;
        this->strideBatchC = strideBatchC;
        this->bRealGlobal = bRealGlobal;
        this->bImagGlobal = bImagGlobal;
        this->cRealGlobal = cRealGlobal;
        this->cImagGlobal = cImagGlobal;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aRealGlobal, GlobalTensor<T> aImagGlobal, GlobalTensor<T> aImagNegGlobal, GlobalTensor<T> bRealGlobal, GlobalTensor<T> bImagGlobal, GlobalTensor<T> cRealGlobal, GlobalTensor<T> cImagGlobal, int strideA, int strideB, int strideC, int strideBatchB = 0, int strideBatchC = 0) {
        this->strideA = strideA;
        this->strideB = strideB;
        this->strideC = strideC;
        this->strideBatchB = strideBatchB;
        this->strideBatchC = strideBatchC;
        this->aRealGlobal = aRealGlobal;
        this->aImagGlobal = aImagGlobal;
        this->aImagNegGlobal = aImagNegGlobal;
        this->bRealGlobal = bRealGlobal;
        this->bImagGlobal = bImagGlobal;
        this->cRealGlobal = cRealGlobal;
        this->cImagGlobal = cImagGlobal;
    }
    __aicore__ inline void ProcessReal(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int batch = 1);
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int batch = 1) {
        ProcessComplex<ProcessMode::Normal>(offsetA, offsetB, offsetC, blockM, blockN, blockK, batch);
    }
    __aicore__ inline void ProcessWithTransposeBatchMerged(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int batch = 1) {
        ProcessComplex<ProcessMode::BatchMerged>(offsetA, offsetB, offsetC, blockM, blockN, blockK, batch);
    }
    __aicore__ inline void ProcessWithTranspose(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int batch = 1) {
        ProcessComplex<ProcessMode::Transposed>(offsetA, offsetB, offsetC, blockM, blockN, blockK, batch);
    }
    __aicore__ inline void ProcessWithTranspose2(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int batch = 1) {
        ProcessComplex<ProcessMode::BatchedA>(offsetA, offsetB, offsetC, blockM, blockN, blockK, batch);
    }
private:
    __aicore__ inline Nd2NzParams MakeResidentAParams(int m, int k) {
        return {1, static_cast<uint16_t>(m), static_cast<uint16_t>(k), 0,
                static_cast<uint16_t>(k), static_cast<uint16_t>(m), 1, 1};
    }

    enum class ProcessMode { Normal, BatchMerged, Transposed, BatchedA };

    __aicore__ inline bool InitBlockTiling(int blockM, int blockN, int blockK, int batch,
                                              int &mBlocks, int &nBlocks, int &kBlocks) {
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;

        // default: (128, 128, 64)
        baseM = min(singleCoreM, 128);
        baseK = min(singleCoreK, 64);
        if (baseM == 0) {
            return false;
        }
        if (baseK == 0) {
            return false;
        }
        if (baseM <= 0 || baseK <= 0 || batch <= 0) {
            return false;
        }
        baseN = min(singleCoreN, min(16384 / baseM, 8192 / baseK));
        if (baseN == 0) {
            return false;
        }
        if (baseN <= 0) {
            return false;
        }

        mBlocks = (singleCoreM + baseM - 1) / baseM;
        nBlocks = (singleCoreN + baseN - 1) / baseN;
        kBlocks = (singleCoreK + baseK - 1) / baseK;
        return true;
    }

    template<ProcessMode mode, ComplexPart cp>
    __aicore__ inline void CopyInBForMode(int offset, int cnt, int batch) {
        if constexpr (mode == ProcessMode::Normal) CopyInB<cp>(offset, cnt, batch);
        else if constexpr (mode == ProcessMode::BatchMerged) CopyInBWithTransposeBatchMerged<cp>(offset, cnt, batch);
        else CopyInBWithTranspose<cp>(offset, cnt, batch);
    }

    template<ProcessMode mode>
    __aicore__ inline void SplitAForMode(const LocalTensor<T> &a, int idx, int batch) {
        if constexpr (mode == ProcessMode::BatchedA) SplitA2(a, idx, batch);
        else SplitA(a, idx);
    }

    template<ProcessMode mode>
    __aicore__ inline void SplitBForMode(const LocalTensor<T> &b, int idx, int batch) {
        if constexpr (mode == ProcessMode::Normal) SplitB(b, idx, batch);
        else if constexpr (mode == ProcessMode::BatchMerged) SplitBWithTransposeBatchMerged(b, idx, batch);
        else SplitBWithTranspose(b, idx, batch);
    }

    template<ProcessMode mode>
    __aicore__ inline void ComputeForMode(const LocalTensor<T> &a, const LocalTensor<T> &b,
                                        const LocalTensor<T> &c, int init, int final, int batch) {
        if constexpr (mode == ProcessMode::BatchedA) Compute2(a, b, c, init, final, batch);
        else if constexpr (mode == ProcessMode::Transposed) Compute3(a, b, c, init, final, batch);
        else Compute(a, b, c, init, final, batch);
    }

    template<ProcessMode mode>
    __aicore__ inline void AccumulateRealB(LocalTensor<T> &a1RealLocal, LocalTensor<T> &a1ImagLocal,
                                          LocalTensor<T> &b1RealLocal, LocalTensor<T> &c1RealLocal,
                                          LocalTensor<T> &c1ImagLocal, int k, int cnt, int batch);

    template<ProcessMode mode>
    __aicore__ inline void AccumulateImaginaryB(LocalTensor<T> &a1RealLocal, LocalTensor<T> &c1RealLocal,
                                               LocalTensor<T> &c1ImagLocal, int k, int cnt,
                                               int kBlocks, int offsetC_1, int batch);

    template<ProcessMode mode>
    __aicore__ inline void ProcessComplex(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int batch = 1);
public:
    __aicore__ inline void CopyInEye(GlobalTensor<T> aGlobal, int m, int k) {
        Nd2NzParams intriParamsMatA = MakeResidentAParams(m, k);
        a1Reside[3]= inQueueA1.AllocTensor<T>();
        DataCopy(a1Reside[3], aGlobal, intriParamsMatA);
        inQueueA1.EnQue(a1Reside[3]);
        a1Reside[3] = inQueueA1.DeQue<T>();
        SetFMatrix(m, k);
    }
    __aicore__ inline void CopyInAReside(GlobalTensor<T> aGlobal, int m, int k) {
        static constexpr int cnt = 3;
        Nd2NzParams intriParamsMatA = MakeResidentAParams(m, k);
        for (int i = 0; i < cnt; ++i) {
            a1Reside[i]= inQueueA1.AllocTensor<T>();
            DataCopy(a1Reside[i], aGlobal[m * k * i], intriParamsMatA);
            inQueueA1.EnQue(a1Reside[i]);
        }
        for (int i = 0; i < cnt; ++i) {
            a1Reside[i] = inQueueA1.DeQue<T>();
        }
        SetFMatrix(m, k);
    }
    __aicore__ inline void SetFMatrix(int m, int k) {
        static constexpr uint8_t padList[]{0, 0, 0, 0};
        Load3DSetFMatrixCal(1, m, padList);
        Load3DSetFMatrixBCal(1, k, padList);
    }
    __aicore__ inline void FreeEye() {
        inQueueA1.FreeTensor(a1Reside[3]);
    }
    __aicore__ inline void FreeAReside() {
        static constexpr int cnt = 3;
        for (int i = 0; i < cnt; ++i) {
            inQueueA1.FreeTensor(a1Reside[i]);
        }
    }
private:
    template<ComplexPart cp, DFTType dt = DFTType::Col>
    __aicore__ inline void CopyInA(int offsetA_1, int cnt, int batch) {
        LocalTensor<T> a1Local = inQueueA1.AllocTensor<T>();
        if constexpr (dt == DFTType::Col) {
            Nd2NzParams intriParamsMatA {
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(tailM),
                static_cast<uint16_t>(tailK * cnt),
                static_cast<uint16_t>(strideA * strideA * 3),
                static_cast<uint16_t>(strideA),
                static_cast<uint16_t>(tailM),
                1,
                static_cast<uint16_t>(tailM * tailK * cnt)
            };
            if constexpr (cp == ComplexPart::Real)
                DataCopy(a1Local, aRealGlobal[offsetA_1], intriParamsMatA);
            else if constexpr (cp == ComplexPart::Imag)
                DataCopy(a1Local, aImagGlobal[offsetA_1], intriParamsMatA);
            else
                DataCopy(a1Local, aImagNegGlobal[offsetA_1], intriParamsMatA);
        } else {
            Nd2NzParams intriParamsMatA {
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(tailM),
                static_cast<uint16_t>(tailK * cnt),
                static_cast<uint16_t>(strideBatchB),
                static_cast<uint16_t>(strideA),
                static_cast<uint16_t>(tailM),
                1,
                static_cast<uint16_t>(tailM * tailK * cnt)
            };
            if constexpr (cp == ComplexPart::Real)
                DataCopy(a1Local, bRealGlobal[offsetA_1], intriParamsMatA);
            else
                DataCopy(a1Local, bImagGlobal[offsetA_1], intriParamsMatA);
        }
        inQueueA1.EnQue(a1Local);
    }
    template<ComplexPart cp, DFTType dt = DFTType::Col>
    __aicore__ inline void CopyInB(int offsetB_1, int cnt, int batch) {
        if constexpr (dt == DFTType::Col) {
            CopyInBLayout<cp, false>(offsetB_1, cnt, batch);
            return;
        } else {
            LocalTensor<T> b1Local = inQueueB1.AllocTensor<T>();
            Nd2NzParams intriParamsMatB {
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(tailK * cnt),
                static_cast<uint16_t>(tailN),
                static_cast<uint16_t>(strideB * strideB * 3),
                static_cast<uint16_t>(strideB),
                static_cast<uint16_t>(tailK * cnt),
                1,
                static_cast<uint16_t>(tailN * tailK * cnt)
            };
            if constexpr (cp == ComplexPart::Real)
                DataCopy(b1Local, aRealGlobal[offsetB_1], intriParamsMatB);
            else if constexpr (cp == ComplexPart::Imag)
                DataCopy(b1Local, aImagGlobal[offsetB_1], intriParamsMatB);
            else
                DataCopy(b1Local, aImagNegGlobal[offsetB_1], intriParamsMatB);
            inQueueB1.EnQue(b1Local);
        }
    }
    template<ComplexPart cp, bool Transposed, bool BatchMerged = false>
    __aicore__ inline void CopyInBLayout(int offsetB, int cnt, int batch) {
        LocalTensor<T> b1Local = inQueueB1.AllocTensor<T>();
        GlobalTensor<T> src = cp == ComplexPart::Real ? bRealGlobal : bImagGlobal;
        int n = Transposed ? tailN : tailK * cnt;
        int d = Transposed ? tailK * cnt : tailN;
        int dstNzStride = BatchMerged ? batch * tailN : n;
        int matrixStride = BatchMerged ? tailN * 8 : tailN * tailK * cnt;
        if (strideBatchB < UINT16_MAX) {
            Nd2NzParams params {
                static_cast<uint16_t>(batch), static_cast<uint16_t>(n), static_cast<uint16_t>(d),
                static_cast<uint16_t>(strideBatchB), static_cast<uint16_t>(strideB),
                static_cast<uint16_t>(dstNzStride), 1, static_cast<uint16_t>(matrixStride)
            };
            DataCopy(b1Local, src[offsetB], params);
        } else {
            Nd2NzParams params {
                1, static_cast<uint16_t>(n), static_cast<uint16_t>(d), 0,
                static_cast<uint16_t>(strideB), static_cast<uint16_t>(dstNzStride), 1,
                static_cast<uint16_t>(Transposed ? 0 : 1)
            };
            for (int i = 0; i < batch; ++i) {
                DataCopy(b1Local[i * matrixStride], src[offsetB + i * strideBatchB], params);
            }
        }
        inQueueB1.EnQue(b1Local);
    }

    template<ComplexPart cp>
    __aicore__ inline void CopyInBWithTranspose(int offsetB, int cnt, int batch) {
        CopyInBLayout<cp, true>(offsetB, cnt, batch);
    }

    template<ComplexPart cp>
    __aicore__ inline void CopyInBWithTransposeBatchMerged(int offsetB, int cnt, int batch) {
        CopyInBLayout<cp, true, true>(offsetB, cnt, batch);
    }
    __aicore__ inline void SplitA(const LocalTensor<T> &a1Local, int splitIdx) {
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        LoadData3DParamsV2Pro loadData3DV2;
        static constexpr uint16_t mPos = 0, sAL1NOffset_ = 0;
        loadData3DV2.channelSize = tailK;
        loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) |
                                ((uint64_t)tailM << 16) | (uint64_t)tailK;
        LoadData<T>(a2Local, a1Local[splitIdx * baseK * tailM], loadData3DV2);
        inQueueA2.EnQue(a2Local);
    }
    __aicore__ inline void SplitA2(const LocalTensor<T> &a1Local, int splitIdx, int batch = 1) {
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        LoadData3DParamsV2Pro loadData3DV2;
        static constexpr uint16_t mPos = 0, sAL1NOffset_ = 0;
        loadData3DV2.channelSize = tailK;
        loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) |
                                ((uint64_t)tailM << 16) | (uint64_t)tailK;
        for (int i = 0; i < batch; ++i) {
            LoadData<T>(a2Local[i * baseK * tailM], a1Local[splitIdx * baseK * tailM + i * baseK * tailM], loadData3DV2);
        }
        inQueueA2.EnQue(a2Local);
    }
    __aicore__ inline void SplitB(const LocalTensor<T> &b1Local, int splitIdx, int batch = 1) {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
        LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = tailN * batch;
        loadData3DV2.extConfig = ((uint64_t)splitIdx * tailK << 48) | ((uint64_t)0 << 32) |
                                ((uint64_t)tailK << 16) | (uint64_t)tailN * batch;
        loadData3DV2.fMatrixCtrl = true;
        LoadData<T>(b2Local, b1Local, loadData3DV2);
        inQueueB2.EnQue(b2Local);
    }
    __aicore__ inline void SplitBWithTranspose(const LocalTensor<T> &b1Local, int splitIdx, int batch = 1) {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();

        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = baseK * tailN / 128;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = 0;
        for (int i = 0; i < batch; ++i)
            LoadData<T>(b2Local[i * baseK * tailN], b1Local[splitIdx * baseK * tailN * batch + i * baseK * tailN], loadDataParams);
        inQueueB2.EnQue(b2Local);
    }
    __aicore__ inline void SplitBWithTransposeBatchMerged(const LocalTensor<T> &b1Local, int splitIdx, int batch = 1) {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();

        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = baseK * tailN * batch / 128;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = 0;
        LoadData<T>(b2Local, b1Local[splitIdx * baseK * tailN * batch], loadDataParams);
        inQueueB2.EnQue(b2Local);
    }
    template<DFTType dt = DFTType::Col>
    __aicore__ inline void Compute(const LocalTensor<T> &a2Local, const LocalTensor<T> &b2Local, const LocalTensor<T> &c1Local, int first, const int unitFlag, int batch = 1);
    template<bool BatchedA>
    __aicore__ inline void ComputeBatched(const LocalTensor<T> &a2Local, const LocalTensor<T> &b2Local, const LocalTensor<T> &c1Local, int first, const int unitFlag, int batch = 1);
    __aicore__ inline void Compute2(const LocalTensor<T> &a, const LocalTensor<T> &b,
                                     const LocalTensor<T> &c, int first, int unitFlag, int batch = 1);
    __aicore__ inline void Compute3(const LocalTensor<T> &a, const LocalTensor<T> &b,
                                     const LocalTensor<T> &c, int first, int unitFlag, int batch = 1);
    template<ComplexPart cp>
    __aicore__ inline void CopyOut(const LocalTensor<T> &c1Local, int offsetC_1, int batch = 1) {
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = tailN;
        fixpipeParams.mSize = tailM;
        fixpipeParams.srcStride = tailM;
        fixpipeParams.dstStride = strideC;

        fixpipeParams.unitFlag = 3;
        if (strideBatchC == tailN) {
            fixpipeParams.ndNum = 1;
            fixpipeParams.srcNdStride = 0;
            fixpipeParams.dstNdStride = 0;
            fixpipeParams.nSize = tailN * batch;
            if constexpr (cp == ComplexPart::Real)
                Fixpipe(cRealGlobal[offsetC_1], c1Local, fixpipeParams);
            else
                Fixpipe(cImagGlobal[offsetC_1], c1Local, fixpipeParams);
            return;
        }
        if (strideBatchC < UINT16_MAX) {
            fixpipeParams.ndNum = batch;
            fixpipeParams.srcNdStride = tailM * tailN / 256;
            fixpipeParams.dstNdStride = strideBatchC;
            if constexpr (cp == ComplexPart::Real)
                Fixpipe(cRealGlobal[offsetC_1], c1Local, fixpipeParams);
            else
                Fixpipe(cImagGlobal[offsetC_1], c1Local, fixpipeParams);
        } else {
            fixpipeParams.ndNum = 1;
            fixpipeParams.srcNdStride = 0;
            fixpipeParams.dstNdStride = 0;
            for (int i = 0; i < batch; ++i) {
                if constexpr (cp == ComplexPart::Real)
                    Fixpipe(cRealGlobal[offsetC_1 + i * strideBatchC], c1Local[i * tailM * tailN], fixpipeParams);
                else
                    Fixpipe(cImagGlobal[offsetC_1 + i * strideBatchC], c1Local[i * tailM * tailN], fixpipeParams);
            }
        }
    }
};

} // namespace Rfft1DR2CCommonKernel

#include "rfft1_d_r2c_cmatmul_compute.h"

#endif // RFFT1_D_R2C_CMATMUL_H
