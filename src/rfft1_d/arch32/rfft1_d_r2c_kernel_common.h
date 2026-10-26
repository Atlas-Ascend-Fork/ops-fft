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

#ifndef RFFT1_D_R2C_KERNEL_COMMON_H
#define RFFT1_D_R2C_KERNEL_COMMON_H

#include "rfft1_d_r2c_cmatmul.h"
#include "rfft1_d_r2c_fft3_stages.h"
#include "rfft1_d_r2c_unpack.h"

namespace Rfft1DR2CCommonKernel {
using namespace AscendC;

// FFT2 keeps its resident twiddle; the FFT3 path loads a twiddle tile per task.
template<typename T>
class ResidentTwiddleMul {
    TQue<QuePosition::VECIN, 8> inQueueSrc;
    TQue<QuePosition::VECOUT, 2> outQueueDst;
    GlobalTensor<T> srcGlobalReal, srcGlobalImag;
    GlobalTensor<T> dstGlobalReal, dstGlobalImag;
    LocalTensor<T> wkReal, wkImag, wkImagNeg;
    int elementsPerBlock;
public:
    static constexpr int TILE_LENGTH = 4096;
    __aicore__ inline ResidentTwiddleMul() {}
    __aicore__ inline void Init(TPipe *pipe) {
        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(inQueueSrc, 8, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueDst, 2, 2 * TILE_LENGTH * sizeof(T));
    }
    __aicore__ inline void Init2(TQue<QuePosition::VECIN, 8> inQueueSrc, TQue<QuePosition::VECOUT, 2> outQueueDst) {
        this->inQueueSrc = inQueueSrc;
        this->outQueueDst = outQueueDst;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> srcGlobalReal, GlobalTensor<T> srcGlobalImag, GlobalTensor<T> dstGlobalReal, GlobalTensor<T> dstGlobalImag) {
        this->srcGlobalReal = srcGlobalReal;
        this->srcGlobalImag = srcGlobalImag;
        this->dstGlobalReal = dstGlobalReal;
        this->dstGlobalImag = dstGlobalImag;
    }
    __aicore__ inline void LoadTwiddle(GlobalTensor<T> twiddleGlobalReal, GlobalTensor<T> twiddleGlobalImag, int n) {
        wkReal = inQueueSrc.AllocTensor<T>();
        wkImag = inQueueSrc.AllocTensor<T>();
        DataCopy(wkReal, twiddleGlobalReal, n);
        DataCopy(wkImag, twiddleGlobalImag, n);
        inQueueSrc.EnQue(wkReal);
        inQueueSrc.EnQue(wkImag);
        wkReal = inQueueSrc.DeQue<T>();
        wkImag = inQueueSrc.DeQue<T>();
        wkImagNeg = inQueueSrc.AllocTensor<T>();
        Muls(wkImagNeg, wkImag, -1.f, n);
    }
    __aicore__ inline void FreeTwiddle() {
        inQueueSrc.FreeTensor(wkReal);
        inQueueSrc.FreeTensor(wkImag);
        inQueueSrc.FreeTensor(wkImagNeg);
    }
    __aicore__ inline void Process(int realOffset, int imagOffset, int n) {
        LocalTensor<T> srcLocalReal = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> srcLocalImag = inQueueSrc.AllocTensor<T>();

        DataCopy(srcLocalReal, srcGlobalReal[realOffset], n);
        DataCopy(srcLocalImag, srcGlobalImag[imagOffset], n);

        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);

        LocalTensor<T> dstLocalReal = outQueueDst.AllocTensor<T>();
        LocalTensor<T> dstLocalImag = dstLocalReal[TILE_LENGTH];

        Mul(dstLocalReal, srcLocalReal, wkReal, n);
        MulAddDst(dstLocalReal, srcLocalImag, wkImagNeg, n);
        Mul(dstLocalImag, srcLocalReal, wkImag, n);
        MulAddDst(dstLocalImag, srcLocalImag, wkReal, n);

        inQueueSrc.FreeTensor(srcLocalReal);
        inQueueSrc.FreeTensor(srcLocalImag);

        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopy(dstGlobalReal[realOffset], dstLocalReal, n);
        DataCopy(dstGlobalImag[imagOffset], dstLocalImag, n);

        outQueueDst.FreeTensor(dstLocalReal);
    }
private:
};

template<typename T, bool Distributed>
class TransposeTwiddleMul {
    TQue<QuePosition::VECIN, 8> inQueueSrc;
    TQue<QuePosition::VECOUT, 2> outQueueDst;
    GlobalTensor<T> twiddleGlobalReal, twiddleGlobalImag;
    GlobalTensor<T> srcGlobalReal, srcGlobalImag;
    GlobalTensor<T> dstGlobalReal, dstGlobalImag;
    LocalTensor<T> wkReal, wkImag, wkImagNeg;
    int elementsPerBlock;
    static constexpr int TILE_LENGTH = 4096;
public:
    __aicore__ inline TransposeTwiddleMul() {}
    __aicore__ inline void Init(TPipe *pipe) {
        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(inQueueSrc, 8, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueDst, 2, 2 * TILE_LENGTH * sizeof(T));
    }
    __aicore__ inline void Init2(TQue<QuePosition::VECIN, 8> inQueueSrc, TQue<QuePosition::VECOUT, 2> outQueueDst) {
        elementsPerBlock = 32 / sizeof(T);

        this->inQueueSrc = inQueueSrc;
        this->outQueueDst = outQueueDst;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> twiddleGlobalReal, GlobalTensor<T> twiddleGlobalImag, GlobalTensor<T> srcGlobalReal, GlobalTensor<T> srcGlobalImag, GlobalTensor<T> dstGlobalReal, GlobalTensor<T> dstGlobalImag) {
        this->twiddleGlobalReal = twiddleGlobalReal;
        this->twiddleGlobalImag = twiddleGlobalImag;
        this->srcGlobalReal = srcGlobalReal;
        this->srcGlobalImag = srcGlobalImag;
        this->dstGlobalReal = dstGlobalReal;
        this->dstGlobalImag = dstGlobalImag;
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int n1, int n2, int n3) {
        if (n3 == 0) {
            return;
        }
        int b = __builtin_ctz(TILE_LENGTH / n3);
        int step2 = 1 << ((b + 1) / 2);
        int step1 = 1 << (b / 2);

        for (int i1 = 0; i1 < n1; i1 += step1) {
            for (int i2 = 0; i2 < n2; i2 += step2) {
                ProcessTile(srcOffset, dstOffset, n1, n2, n3, step1, step2, i1, i2);

            }
        }
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int n1, int n2, int n3, int batch, int batchStride, int blockNum, int blockIdx) {
        if (n3 == 0) {
            return;
        }
        if (blockNum == 0) {
            return;
        }
        int b = __builtin_ctz(TILE_LENGTH / n3);
        int step2 = 1 << ((b + 1) / 2);
        int step1 = 1 << (b / 2);

        int tileNum1 = n1 / step1;
        int tileNum2 = n2 / step2;
        int tilesPerBatch = tileNum1 * tileNum2;

        int totalNum = batch * tilesPerBatch;
        int averageNum = totalNum / blockNum;
        int remainderNum = totalNum % blockNum;
        int startIdx = averageNum * blockIdx + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        for (int idx = 0; idx < calcNum; ++idx) {
            int taskIdx = startIdx + idx;
            int batchIdx = taskIdx / tilesPerBatch;
            int tileIdx = taskIdx % tilesPerBatch;

            int i1 = tileIdx / tileNum2 * step1;
            int i2 = tileIdx % tileNum2 * step2;

            int srcBatchOffset = srcOffset + batchIdx * batchStride;
            int dstBatchOffset = dstOffset + batchIdx * batchStride;

            ProcessTile(srcBatchOffset, dstBatchOffset, n1, n2, n3, step1, step2, i1, i2);

        }
    }
private:
    __aicore__ inline void ComputeTransposeTile(
        int step1, int n3, LocalTensor<T> &dstLocalReal, LocalTensor<T> &srcLocalReal, int step2,
        LocalTensor<T> &wkReal, LocalTensor<T> &dstLocalImag, LocalTensor<T> &wkImag, LocalTensor<T> &srcLocalImag)
    {
        if constexpr (Distributed) {
            for (int j1 = 0; j1 < step1; ++j1) {
                for (int k = 0; k < n3; k += 64) {
                    int calcN = min(64, n3 - k);
                    Mul(dstLocalReal[j1 * n3 + k], srcLocalReal[j1 * step2 * n3 + k], wkReal[j1 * n3 + k], calcN, step2, {1, 1, 1, static_cast<uint8_t>(step1 * n3 / elementsPerBlock), static_cast<uint8_t>(n3 / elementsPerBlock), static_cast<uint8_t>(step1 * n3 / elementsPerBlock)});
                    Mul(dstLocalImag[j1 * n3 + k], srcLocalReal[j1 * step2 * n3 + k], wkImag[j1 * n3 + k], calcN, step2, {1, 1, 1, static_cast<uint8_t>(step1 * n3 / elementsPerBlock), static_cast<uint8_t>(n3 / elementsPerBlock), static_cast<uint8_t>(step1 * n3 / elementsPerBlock)});
                }
            }
            Muls(wkImag, wkImag, -1.f, TILE_LENGTH);
            for (int j1 = 0; j1 < step1; ++j1) {
                for (int k = 0; k < n3; k += 64) {
                    int calcN = min(64, n3 - k);
                    MulAddDst(dstLocalReal[j1 * n3 + k], srcLocalImag[j1 * step2 * n3 + k], wkImag[j1 * n3 + k], calcN, step2, {1, 1, 1, static_cast<uint8_t>(step1 * n3 / elementsPerBlock), static_cast<uint8_t>(n3 / elementsPerBlock), static_cast<uint8_t>(step1 * n3 / elementsPerBlock)});
                    MulAddDst(dstLocalImag[j1 * n3 + k], srcLocalImag[j1 * step2 * n3 + k], wkReal[j1 * n3 + k], calcN, step2, {1, 1, 1, static_cast<uint8_t>(step1 * n3 / elementsPerBlock), static_cast<uint8_t>(n3 / elementsPerBlock), static_cast<uint8_t>(step1 * n3 / elementsPerBlock)});
                }
            }
        } else {
            for (int j1 = 0; j1 < step1; ++j1) {
                Mul(dstLocalReal[j1 * n3], srcLocalReal[j1 * step2 * n3], wkReal[j1 * n3], n3, step2, {1, 1, 1, static_cast<uint8_t>(step1 * n3 / elementsPerBlock), static_cast<uint8_t>(n3 / elementsPerBlock), static_cast<uint8_t>(step1 * n3 / elementsPerBlock)});
                Mul(dstLocalImag[j1 * n3], srcLocalReal[j1 * step2 * n3], wkImag[j1 * n3], n3, step2, {1, 1, 1, static_cast<uint8_t>(step1 * n3 / elementsPerBlock), static_cast<uint8_t>(n3 / elementsPerBlock), static_cast<uint8_t>(step1 * n3 / elementsPerBlock)});
            }
            Muls(wkImag, wkImag, -1.f, TILE_LENGTH);
            for (int j1 = 0; j1 < step1; ++j1) {
                MulAddDst(dstLocalReal[j1 * n3], srcLocalImag[j1 * step2 * n3], wkImag[j1 * n3], n3, step2, {1, 1, 1, static_cast<uint8_t>(step1 * n3 / elementsPerBlock), static_cast<uint8_t>(n3 / elementsPerBlock), static_cast<uint8_t>(step1 * n3 / elementsPerBlock)});
                MulAddDst(dstLocalImag[j1 * n3], srcLocalImag[j1 * step2 * n3], wkReal[j1 * n3], n3, step2, {1, 1, 1, static_cast<uint8_t>(step1 * n3 / elementsPerBlock), static_cast<uint8_t>(n3 / elementsPerBlock), static_cast<uint8_t>(step1 * n3 / elementsPerBlock)});
            }
        }
    }

    __aicore__ inline void LoadTransposeTwiddle(
        int step2, int step1, int n3, int n1, LocalTensor<T> &wkReal, int i2, int i1, LocalTensor<T> &wkImag)
    {
        DataCopyParams CopyInParams2 {
            static_cast<uint16_t>(step2),
            static_cast<uint16_t>(step1 * n3 / elementsPerBlock),
            static_cast<uint16_t>((n1 * n3 - step1 * n3) / elementsPerBlock),
            0
        };

        DataCopy(wkReal, twiddleGlobalReal[i2 * n1 * n3 + i1 * n3], CopyInParams2);
        DataCopy(wkImag, twiddleGlobalImag[i2 * n1 * n3 + i1 * n3], CopyInParams2);

        inQueueSrc.EnQue(wkReal);
        inQueueSrc.EnQue(wkImag);
        wkReal = inQueueSrc.DeQue<T>();
        wkImag = inQueueSrc.DeQue<T>();
    }

    __aicore__ inline void ProcessTile(int srcOffset, int dstOffset, int n1, int n2, int n3,
                                      int step1, int step2, int i1, int i2) {
        LocalTensor<T> srcLocalReal = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> srcLocalImag = inQueueSrc.AllocTensor<T>();

        DataCopyParams CopyInParams {
            static_cast<uint16_t>(step1),
            static_cast<uint16_t>(step2 * n3 / elementsPerBlock),
            static_cast<uint16_t>((n2 * n3 - step2 * n3) / elementsPerBlock),
            0
        };

        DataCopy(srcLocalReal, srcGlobalReal[srcOffset + i1 * n2 * n3 + i2 * n3], CopyInParams);
        DataCopy(srcLocalImag, srcGlobalImag[srcOffset + i1 * n2 * n3 + i2 * n3], CopyInParams);

        inQueueSrc.EnQue(srcLocalReal);
        inQueueSrc.EnQue(srcLocalImag);
        srcLocalReal = inQueueSrc.DeQue<T>();
        srcLocalImag = inQueueSrc.DeQue<T>();

        LocalTensor<T> wkReal = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> wkImag = inQueueSrc.AllocTensor<T>();

        LoadTransposeTwiddle(step2, step1, n3, n1, wkReal, i2, i1, wkImag);

        LocalTensor<T> dstLocalReal = outQueueDst.AllocTensor<T>();
        LocalTensor<T> dstLocalImag = dstLocalReal[TILE_LENGTH];

        ComputeTransposeTile(step1, n3, dstLocalReal, srcLocalReal, step2, wkReal, dstLocalImag, wkImag,
            srcLocalImag);

        inQueueSrc.FreeTensor(srcLocalReal);
        inQueueSrc.FreeTensor(srcLocalImag);
        inQueueSrc.FreeTensor(wkReal);
        inQueueSrc.FreeTensor(wkImag);

        outQueueDst.EnQue(dstLocalReal);
        dstLocalReal = outQueueDst.DeQue<T>();
        if constexpr (Distributed) dstLocalImag = dstLocalReal[TILE_LENGTH];

        DataCopyParams CopyOutParams {
            static_cast<uint16_t>(step2),
            static_cast<uint16_t>(step1 * n3 / elementsPerBlock),
            0,
            static_cast<uint16_t>((n1 * n3 - step1 * n3) / elementsPerBlock)
        };

        DataCopy(dstGlobalReal[dstOffset + i2 * n1 * n3 + i1 * n3], dstLocalReal, CopyOutParams);
        DataCopy(dstGlobalImag[dstOffset + i2 * n1 * n3 + i1 * n3], dstLocalImag, CopyOutParams);

        outQueueDst.FreeTensor(dstLocalReal);
    }
};

// FFT4 and FFT5 use different producer lead distances; keep the wait policy explicit.
template<typename T, bool AdaptiveWait>
class RecurrenceTwiddleMul {
    GlobalTensor<T> xGlobalReal, xGlobalImag;
    GlobalTensor<T> yGlobalReal, yGlobalImag;
    GlobalTensor<T> zGlobalReal, zGlobalImag;
    TQue<QuePosition::VECIN, 5> inQueueSrc;
    TQue<QuePosition::VECOUT, 2> outQueueDst;
    int elementsPerBlock;
    static constexpr int TILE_LENGTH = 8192;
    static constexpr int TILE_LENGTH2 = 4096;
public:
    __aicore__ inline RecurrenceTwiddleMul() {}
    __aicore__ inline void Init(TPipe *pipe) {
        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(inQueueSrc, 5, TILE_LENGTH2 * 2 * sizeof(T));
        pipe->InitBuffer(outQueueDst, 2, TILE_LENGTH2 * sizeof(T));
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> xGlobalReal, GlobalTensor<T> xGlobalImag, GlobalTensor<T> yGlobalReal, GlobalTensor<T> yGlobalImag, GlobalTensor<T> zGlobalReal, GlobalTensor<T> zGlobalImag) {
        this->xGlobalReal = xGlobalReal;
        this->xGlobalImag = xGlobalImag;
        this->yGlobalReal = yGlobalReal;
        this->yGlobalImag = yGlobalImag;
        this->zGlobalReal = zGlobalReal;
        this->zGlobalImag = zGlobalImag;
    }
private:
    __aicore__ inline void ComputeRecurrenceBatch(
        int batch, LocalTensor<T> &zLocalReal, int calcN, LocalTensor<T> &xLocalImag, LocalTensor<T> &wkImag,
        LocalTensor<T> &xLocalReal, LocalTensor<T> &wkReal, LocalTensor<T> &zLocalImag, LocalTensor<T> &w1Real,
        LocalTensor<T> &w1Imag, LocalTensor<T> &w1ImagNeg)
    {
        for (int b = 0; b < batch; ++b) {
            Mul(zLocalReal[b * calcN], xLocalImag[b * calcN], wkImag, calcN);
            Muls(zLocalReal[b * calcN], zLocalReal[b * calcN], -1.f, calcN);
            MulAddDst(zLocalReal[b * calcN], xLocalReal[b * calcN], wkReal, calcN);
            Mul(zLocalImag[b * calcN], xLocalImag[b * calcN], wkReal, calcN);
            MulAddDst(zLocalImag[b * calcN], xLocalReal[b * calcN], wkImag, calcN);
            {
                LocalTensor<T> tmpReal = inQueueSrc.AllocTensor<T>();
                LocalTensor<T> tmpImag = tmpReal[TILE_LENGTH2];

                Mul(tmpReal, wkReal, w1Real, calcN);
                Mul(tmpImag, wkReal, w1Imag, calcN);
                MulAddDst(tmpImag, wkImag, w1Real, calcN);
                MulAddDst(tmpReal, wkImag, w1ImagNeg, calcN);
                inQueueSrc.FreeTensor(wkReal);
                wkReal = tmpReal;
                wkImag = tmpImag;
            }
        }
    }

    __aicore__ inline void ProcessRecurrenceRows(
        int startIdx, int calcNum, int calcN, int n, int l2Idx, int stepN, int i, LocalTensor<T> &wkImag,
        LocalTensor<T> &wkReal, LocalTensor<T> &w1Real, LocalTensor<T> &w1Imag, LocalTensor<T> &w1ImagNeg,
        int l2Step)
    {
        if (calcN == 0 || l2Step == 0) {
            return;
        }
        for (int j = startIdx; j < startIdx + calcNum; ) {
            int batch = min(startIdx + calcNum - j, TILE_LENGTH2 / calcN);

            LocalTensor<T> xLocalReal = inQueueSrc.AllocTensor<T>();
            LocalTensor<T> xLocalImag = xLocalReal[TILE_LENGTH2];
            LocalTensor<T> zLocalReal = outQueueDst.AllocTensor<T>();
            LocalTensor<T> zLocalImag = outQueueDst.AllocTensor<T>();

            DataCopyParams CopyInParams {
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(calcN / elementsPerBlock),
                static_cast<uint16_t>((n - calcN) / elementsPerBlock),
                0
            };

            DataCopy(xLocalReal, xGlobalReal[(j + l2Idx * stepN) * n + i], CopyInParams);
            DataCopy(xLocalImag, xGlobalImag[(j + l2Idx * stepN) * n + i], CopyInParams);

            inQueueSrc.EnQue(xLocalReal);

            xLocalReal = inQueueSrc.DeQue<T>();

            ComputeRecurrenceBatch(batch, zLocalReal, calcN, xLocalImag, wkImag, xLocalReal, wkReal, zLocalImag,
                w1Real, w1Imag, w1ImagNeg);

            inQueueSrc.FreeTensor(xLocalReal);
            outQueueDst.EnQue(zLocalReal);
            outQueueDst.EnQue(zLocalImag);

            DataCopyParams CopyOutParams {
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(calcN / elementsPerBlock),
                0,
                static_cast<uint16_t>((n - calcN) / elementsPerBlock)
            };

            zLocalReal = outQueueDst.DeQue<T>();
            DataCopy(zGlobalReal[(j + (l2Idx % (l2Step * 2)) * stepN) * n + i], zLocalReal, CopyOutParams);
            outQueueDst.FreeTensor(zLocalReal);
            zLocalImag = outQueueDst.DeQue<T>();
            DataCopy(zGlobalImag[(j + (l2Idx % (l2Step * 2)) * stepN) * n + i], zLocalImag, CopyOutParams);
            outQueueDst.FreeTensor(zLocalImag);

            j += batch;
        }
    }

    __aicore__ inline void ProcessRecurrenceColumn(
        LocalTensor<T> &w1Real, LocalTensor<T> &w1Imag, int i, int n, int coreNum, int l2Blocks, int calcN,
        int coreIdx, int l2Idx, int l2Step, int startIdx, int calcNum, int stepN)
    {
        if (l2Step == 0) {
            return;
        }
        w1Real = inQueueSrc.AllocTensor<T>();
        w1Imag = w1Real[TILE_LENGTH2];

        DataCopy(w1Imag, yGlobalImag[i + n * coreNum * l2Blocks], calcN);
        DataCopy(w1Real, yGlobalReal[i + n * coreNum * l2Blocks], calcN);

        inQueueSrc.EnQue(w1Real);
        w1Real = inQueueSrc.DeQue<T>();

        LocalTensor<T> w1ImagNeg = inQueueSrc.AllocTensor<T>();
        Muls(w1ImagNeg, w1Imag, T(-1), calcN);

        LocalTensor<T> wkReal = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> wkImag = wkReal[TILE_LENGTH2];

        DataCopy(wkImag, yGlobalImag[i + coreIdx * n + n * (l2Idx / l2Step) * coreNum], calcN);
        DataCopy(wkReal, yGlobalReal[i + coreIdx * n + n * (l2Idx / l2Step) * coreNum], calcN);

        inQueueSrc.EnQue(wkReal);

        wkReal = inQueueSrc.DeQue<T>();

        ProcessRecurrenceRows(startIdx, calcNum, calcN, n, l2Idx, stepN, i, wkImag, wkReal, w1Real, w1Imag,
            w1ImagNeg, l2Step);

        inQueueSrc.FreeTensor(w1Real);
        inQueueSrc.FreeTensor(w1ImagNeg);
        inQueueSrc.FreeTensor(wkReal);
    }

public:
    __aicore__ inline void Process(int m, int n, int stepN, int l2Step, int blockNum, int blockIdx) {
        if (stepN == 0) {
            return;
        }
        if (l2Step == 0) {
            return;
        }
        if (blockNum == 0) {
            return;
        }
        int coreIdx = blockIdx >> 1;
        int coreNum = blockNum >> 1;
        if (coreNum == 0) {
            return;
        }

        int calcN = min(n / 2, TILE_LENGTH2);
        if (calcN == 0) {
            return;
        }
        int averageNum = stepN * l2Step / coreNum;
        int remainderNum = stepN * l2Step % coreNum;
        int startIdx = coreIdx * averageNum + min(coreIdx, remainderNum);
        int calcNum = averageNum + (coreIdx < remainderNum);

        int l2Blocks = m / stepN / l2Step;

        LocalTensor<T> w1Imag, w1Real;
        for (int l2Idx = 0; l2Idx < m / stepN; l2Idx += l2Step) {
            if (l2Idx / l2Step >= (AdaptiveWait ? (1 + (n * stepN <= 4194304)) : 2)) {
                CrossCoreWaitFlag(0x0);

                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
                CrossCoreWaitFlag(0x3);
            }
            for (int i = (blockIdx & 1) * calcN; i < n; i += calcN * 2) {
                ProcessRecurrenceColumn(w1Real, w1Imag, i, n, coreNum, l2Blocks, calcN, coreIdx, l2Idx, l2Step,
                    startIdx, calcNum, stepN);
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
        }
    }
};

// Read only the radix fields present in the selected kernel's tiling ABI.
template<int Radices>
struct KernelTiling {
    int32_t N, batch, n1 = 0, n2 = 0, n3 = 0, n4 = 0, n5 = 0;
    __aicore__ inline explicit KernelTiling(__gm__ uint8_t *gm) {
        auto data = reinterpret_cast<__gm__ int32_t *>(gm);
        N = data[0];
        batch = data[1];
        if constexpr (Radices >= 1) n1 = data[2];
        if constexpr (Radices >= 2) n2 = data[3];
        if constexpr (Radices >= 3) n3 = data[4];
        if constexpr (Radices >= 4) n4 = data[5];
        if constexpr (Radices >= 5) n5 = data[6];
    }
};

template<typename T>
__aicore__ inline GlobalTensor<T> BindGlobal(__gm__ uint8_t *gm) {
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gm));
    return tensor;
}

__aicore__ inline void UnpackPairRange(UnpackCustom<float> &op,
                                      GlobalTensor<float> srcReal, GlobalTensor<float> srcImag,
                                      GlobalTensor<float> dstReal, GlobalTensor<float> dstImag,
                                      int n, int count) {
    for (int j = 0; j < count; ++j) {
        op.SetMatrix(srcReal[j * n], srcImag[j * n], dstReal[j * (n + 2)], dstImag[j * (n + 2)]);
        op.UnPack2(n, 1, n, n + 2, 1, 0);
    }
}

} // namespace Rfft1DR2CCommonKernel

#endif
