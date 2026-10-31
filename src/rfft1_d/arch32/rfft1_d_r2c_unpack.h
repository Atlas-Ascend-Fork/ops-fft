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

#ifndef RFFT1_D_R2C_UNPACK_H
#define RFFT1_D_R2C_UNPACK_H

#include "kernel/fft_common_kernel.h"
#include <stdint.h>

namespace Rfft1DR2CCommonKernel {

using namespace::AscendC;

template<typename T>
class UnpackCustom {
    TQue<QuePosition::VECIN, 8> inQueueSrc;
    TQue<QuePosition::VECOUT, 2> outQueueDst;
    GlobalTensor<uint32_t> gatherGlobal;
    GlobalTensor<T> srcGlobalReal, srcGlobalImag;
    GlobalTensor<T> dstGlobal, dstGlobal2;
    GlobalTensor<T> sTwiddleReal, sTwiddleImag;
    LocalTensor<uint32_t> gatherLocal, gatherLocal2;
    LocalTensor<T> wkReal, wkImag;
    T w1Real, w1Imag;
    int m, n;
    int elementsPerBlock = 32 / sizeof(T);
public:
    static constexpr int TILE_LENGTH = 4096;
    __aicore__ inline UnpackCustom() {}
    __aicore__ inline void Init(TPipe *pipe) {
        pipe->InitBuffer(inQueueSrc, 8, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueDst, 2, 2 * TILE_LENGTH * sizeof(T));
    }
    __aicore__ inline void Init2(TQue<QuePosition::VECIN, 8> inQueueSrc, TQue<QuePosition::VECOUT, 2> outQueueDst) {
        this->inQueueSrc = inQueueSrc;
        this->outQueueDst = outQueueDst;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> srcGlobalReal, GlobalTensor<T> srcGlobalImag, GlobalTensor<T> dstGlobal, GlobalTensor<T> dstGlobal2) {
        this->srcGlobalReal = srcGlobalReal;
        this->srcGlobalImag = srcGlobalImag;
        this->dstGlobal = dstGlobal;
        this->dstGlobal2 = dstGlobal2;
    }
    __aicore__ inline void SetValue(GlobalTensor<T> dstGlobal, T value) {
        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
        dstLocal.SetValue(0, value);
        outQueueDst.EnQue(dstLocal);
        dstLocal = outQueueDst.DeQue<T>();
        DataCopyExtParams CopyOutParams {
            1,
            static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPad(dstGlobal, dstLocal, CopyOutParams);
        outQueueDst.FreeTensor(dstLocal);
    }
    __aicore__ inline void SetValueBatch(GlobalTensor<T> srcGlobal, GlobalTensor<T> dstGlobal, int batch, int srcStride, int dstStride) {
        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
        for (int i = 0; i < batch; ++i) {
            dstLocal.SetValue(i * elementsPerBlock, srcGlobal.GetValue(i * srcStride));
        }
        outQueueDst.EnQue(dstLocal);
        dstLocal = outQueueDst.DeQue<T>();
        DataCopyExtParams CopyOutParams {
            static_cast<uint16_t>(batch),
            static_cast<uint32_t>(sizeof(T)),
            0,
            static_cast<uint32_t>((dstStride - 1) * sizeof(T)),
            0
        };
        DataCopyPad(dstGlobal, dstLocal, CopyOutParams);
        outQueueDst.FreeTensor(dstLocal);
    }
    __aicore__ inline void LoadGather(GlobalTensor<uint32_t> gatherGlobal) {
        gatherLocal = inQueueSrc.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, gatherGlobal, TILE_LENGTH);
        inQueueSrc.EnQue(gatherLocal);
        gatherLocal = inQueueSrc.DeQue<uint32_t>();

        gatherLocal2 = inQueueSrc.AllocTensor<uint32_t>();
        DataCopy(gatherLocal2, gatherGlobal[TILE_LENGTH], TILE_LENGTH);
        inQueueSrc.EnQue(gatherLocal2);
        gatherLocal2 = inQueueSrc.DeQue<uint32_t>();
    }
    __aicore__ inline void FreeGather() {
        inQueueSrc.FreeTensor(gatherLocal);
        inQueueSrc.FreeTensor(gatherLocal2);
    }
    __aicore__ inline void LoadTwiddle(GlobalTensor<T> twiddleGlobalReal, GlobalTensor<T> twiddleGlobalImag, int blockIdx) {
        wkReal = inQueueSrc.AllocTensor<T>();
        wkImag = inQueueSrc.AllocTensor<T>();

        DataCopy(wkReal, twiddleGlobalReal[blockIdx * TILE_LENGTH], TILE_LENGTH);
        DataCopy(wkImag, twiddleGlobalImag[blockIdx * TILE_LENGTH], TILE_LENGTH);

        inQueueSrc.EnQue(wkReal);
        inQueueSrc.EnQue(wkImag);

        wkReal = inQueueSrc.DeQue<T>();
        wkImag = inQueueSrc.DeQue<T>();

        w1Real = twiddleGlobalReal.GetValue(TILE_LENGTH / 2) * 2;
        w1Imag = twiddleGlobalImag.GetValue(TILE_LENGTH / 2) * 2;

        {
            T tmp = w1Real * w1Real - w1Imag * w1Imag;
            w1Imag = 2 * w1Real * w1Imag;
            w1Real = tmp;
        }
    }
    __aicore__ inline void LoadTwiddle(GlobalTensor<T> twiddleGlobalReal, GlobalTensor<T> twiddleGlobalImag, GlobalTensor<T> sTwiddleReal, GlobalTensor<T> sTwiddleImag, int blockIdx) {
        this->sTwiddleReal = sTwiddleReal;
        this->sTwiddleImag = sTwiddleImag;

        wkReal = inQueueSrc.AllocTensor<T>();
        wkImag = inQueueSrc.AllocTensor<T>();

        DataCopy(wkReal, twiddleGlobalReal[blockIdx * TILE_LENGTH], TILE_LENGTH);
        DataCopy(wkImag, twiddleGlobalImag[blockIdx * TILE_LENGTH], TILE_LENGTH);

        inQueueSrc.EnQue(wkReal);
        inQueueSrc.EnQue(wkImag);

        wkReal = inQueueSrc.DeQue<T>();
        wkImag = inQueueSrc.DeQue<T>();
    }
    __aicore__ inline void FreeTwiddle() {
        inQueueSrc.FreeTensor(wkReal);
        inQueueSrc.FreeTensor(wkImag);
    }
private:
    __aicore__ inline void RotateUnpackTile(
        int &jTile, LocalTensor<T> &zoLocalReal, int calcN, LocalTensor<T> &zoLocalImag)
    {
        T sre = sTwiddleReal.GetValue(jTile);
        T sim = sTwiddleImag.GetValue(jTile);
        LocalTensor<T> scratch = inQueueSrc.AllocTensor<T>();
        Muls(scratch, zoLocalReal, sim, calcN);
        Muls(zoLocalReal, zoLocalReal, sre, calcN);
        Axpy(zoLocalReal, zoLocalImag, sim, calcN);
        Muls(zoLocalImag, zoLocalImag, sre, calcN);
        Sub(zoLocalImag, zoLocalImag, scratch, calcN);
        inQueueSrc.FreeTensor(scratch);
        jTile++;
    }

    __aicore__ inline void StoreUnpackTile(
        LocalTensor<T> &zoLocalReal, LocalTensor<T> &zeLocalReal, int calcN, LocalTensor<T> &zoLocalImag,
        LocalTensor<T> &zeLocalImag, LocalTensor<T> &zeLocalRealTmp, int i, int n)
    {
        Gather(zoLocalReal, zeLocalReal, gatherLocal, 0, calcN);
        Gather(zoLocalImag, zeLocalImag, gatherLocal, 0, calcN);

        Gather(zeLocalReal, zeLocalRealTmp, gatherLocal2, 0u, calcN);
        Gather(zeLocalImag, zeLocalRealTmp[calcN / 2], gatherLocal2, 0u, calcN);

        Muls(zeLocalRealTmp, zeLocalReal, 1.f, calcN);
        Muls(zeLocalRealTmp[calcN], zeLocalImag, 1.f, calcN);

        Gather(zeLocalReal, zoLocalReal, gatherLocal2, 0u, calcN);
        Gather(zeLocalImag, zoLocalReal[calcN / 2], gatherLocal2, 0u, calcN);

        Muls(zoLocalReal, zeLocalReal, 1.f, calcN);
        Muls(zoLocalReal[calcN], zeLocalImag, 1.f, calcN);
        inQueueSrc.FreeTensor(zeLocalReal);
        inQueueSrc.FreeTensor(zeLocalImag);

        outQueueDst.EnQue(zeLocalRealTmp);
        zeLocalRealTmp = outQueueDst.DeQue<T>();
        DataCopy(dstGlobal[i * 2], zeLocalRealTmp, calcN * 2);
        outQueueDst.FreeTensor(zeLocalRealTmp);

        outQueueDst.EnQue(zoLocalReal);
        zoLocalReal = outQueueDst.DeQue<T>();
        DataCopy(dstGlobal[(n - i - calcN + 1) * 2], zoLocalReal, calcN * 2);
        outQueueDst.FreeTensor(zoLocalReal);
    }

    __aicore__ inline void ComputeUnpackTile(
        LocalTensor<T> &yLocalReal, int calcN, LocalTensor<T> &zoLocalReal, LocalTensor<T> &yLocalImag, int i,
        LocalTensor<T> &zoLocalImag, int &jTile, int n)
    {
        LocalTensor<T> zeLocalReal = inQueueSrc.AllocTensor<T>();
        Gather(zeLocalReal, yLocalReal, gatherLocal, 0, calcN);
        inQueueSrc.FreeTensor(yLocalReal);

        LocalTensor<T> zeLocalImag = inQueueSrc.AllocTensor<T>();
        Gather(zoLocalReal, yLocalImag, gatherLocal, 0, calcN);  // zoLocalImag → zoLocalReal
        inQueueSrc.FreeTensor(yLocalImag);

        LocalTensor<T> xLocalReal = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> xLocalImag = inQueueSrc.AllocTensor<T>();

        DataCopy(xLocalReal, srcGlobalReal[i], calcN);
        DataCopy(xLocalImag, srcGlobalImag[i], calcN);

        inQueueSrc.EnQue(xLocalReal);
        inQueueSrc.EnQue(xLocalImag);

        xLocalReal = inQueueSrc.DeQue<T>();
        xLocalImag = inQueueSrc.DeQue<T>();

        Sub(zoLocalImag, xLocalReal, zeLocalReal, calcN);       // zoLocalReal → zoLocalImag
        Sub(zeLocalImag, xLocalImag, zoLocalReal, calcN);       // zoLocalImag → zoLocalReal (注意这里是第二个参数)
        Add(zeLocalReal, xLocalReal, zeLocalReal, calcN);
        Add(zoLocalReal, xLocalImag, zoLocalReal, calcN);       // zoLocalImag → zoLocalReal

        inQueueSrc.FreeTensor(xLocalReal);
        inQueueSrc.FreeTensor(xLocalImag);

        Muls(zeLocalImag, zeLocalImag, 0.5f, calcN);
        Muls(zeLocalReal, zeLocalReal, 0.5f, calcN);

        {
            RotateUnpackTile(jTile, zoLocalReal, calcN, zoLocalImag);
        }

        LocalTensor<T> tmpReal = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> tmpImag = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> zeLocalRealTmp = outQueueDst.AllocTensor<T>();
        LocalTensor<T> zeLocalImagTmp = zeLocalRealTmp[TILE_LENGTH];

        Mul(tmpReal, zoLocalReal, wkReal, calcN);
        MulAddDst(tmpReal, zoLocalImag, wkImag, calcN);
        Add(zeLocalRealTmp, tmpReal, zeLocalReal, calcN);

        Mul(tmpImag, zoLocalReal, wkImag, calcN);
        Mul(zeLocalImagTmp, zoLocalImag, wkReal, calcN);
        Sub(tmpImag, tmpImag, zeLocalImagTmp, calcN);
        Add(zeLocalImagTmp, tmpImag, zeLocalImag, calcN);

        Sub(zeLocalReal, zeLocalReal, tmpReal, calcN);
        Sub(zeLocalImag, tmpImag, zeLocalImag, calcN);

        inQueueSrc.FreeTensor(tmpReal);
        inQueueSrc.FreeTensor(tmpImag);

        StoreUnpackTile(zoLocalReal, zeLocalReal, calcN, zoLocalImag, zeLocalImag, zeLocalRealTmp, i, n);
    }

    __aicore__ inline void ProcessUnpackTile(
        int n, int i, int calcN, int &jTile)
    {
        LocalTensor<T> yLocalReal = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> yLocalImag = inQueueSrc.AllocTensor<T>();

        {

            DataCopy(yLocalReal, srcGlobalReal[n - i - calcN + 1], calcN);
            DataCopy(yLocalImag, srcGlobalImag[n - i - calcN + 1], calcN);

            inQueueSrc.EnQue(yLocalReal);
            inQueueSrc.EnQue(yLocalImag);

            yLocalReal = inQueueSrc.DeQue<T>();
            yLocalImag = inQueueSrc.DeQue<T>();
        }

        LocalTensor<T> zoLocalReal = outQueueDst.AllocTensor<T>();
        LocalTensor<T> zoLocalImag = zoLocalReal[TILE_LENGTH];

        {
            ComputeUnpackTile(yLocalReal, calcN, zoLocalReal, yLocalImag, i, zoLocalImag, jTile, n);
        }
    }

    __aicore__ inline int GetUnpackTileLength(int n, int blockNum, int blockIdx) {
        if (n < 2 || blockNum <= 0 || blockIdx < 0 || blockIdx >= blockNum) {
            return 0;
        }
        return min(n / 2, TILE_LENGTH);
    }

public:
    __aicore__ inline void UnPack(int n, int blockNum, int blockIdx) {
        int calcN = GetUnpackTileLength(n, blockNum, blockIdx);
        if (calcN == 0 || blockNum == 0) {
            return;
        }
        int totalNum = n / 2 / calcN;
        int averageNum = totalNum / blockNum;
        int remainderNum = totalNum % blockNum;
        int startIdx = (averageNum * blockIdx + min(blockIdx, remainderNum)) * calcN;
        int calcNum = (averageNum + (blockIdx < remainderNum)) * calcN;

        T edgeZ0Re = 0, edgeZ0Im = 0, edgeHalfRe = 0, edgeHalfIm = 0;
        if (!blockIdx) {
            edgeZ0Re = srcGlobalReal.GetValue(0);
            edgeZ0Im = srcGlobalImag.GetValue(0);
            edgeHalfRe = srcGlobalReal.GetValue(n / 2);
            edgeHalfIm = srcGlobalImag.GetValue(n / 2);
        }

        int jTile = 0;
        for (int i = startIdx; i < startIdx + calcNum; i += calcN) {
            ProcessUnpackTile(n, i, calcN, jTile);
        }

        if (!blockIdx) {
            SetValue(dstGlobal, edgeZ0Re + edgeZ0Im);
            SetValue(dstGlobal[1], 0.f);
            SetValue(dstGlobal[n], edgeHalfRe);
            SetValue(dstGlobal[n + 1], -edgeHalfIm);
            SetValue(dstGlobal[n * 2], edgeZ0Re - edgeZ0Im);
            SetValue(dstGlobal[n * 2 + 1], 0.f);
        }
    }
private:
    __aicore__ inline void StoreUnpackPairTile(
        LocalTensor<T> &xLocalReal, LocalTensor<T> &zLocalReal, int calcN, LocalTensor<T> &xLocalImag,
        LocalTensor<T> &zLocalRealTmp, LocalTensor<T> &yLocalReal, LocalTensor<T> &yLocalImag, int dstBatchOffset,
        int i)
    {
        Gather(xLocalReal, zLocalReal, gatherLocal2, 0u, calcN);
        Gather(xLocalImag, zLocalReal[calcN / 2], gatherLocal2, 0u, calcN);

        Muls(zLocalReal, xLocalReal, 1.f, calcN);
        Muls(zLocalReal[calcN], xLocalImag, 1.f, calcN);

        Gather(xLocalReal, zLocalRealTmp, gatherLocal2, 0u, calcN);
        Gather(xLocalImag, zLocalRealTmp[calcN / 2], gatherLocal2, 0u, calcN);

        Muls(zLocalRealTmp, xLocalReal, 1.f, calcN);
        Muls(zLocalRealTmp[calcN], xLocalImag, 1.f, calcN);

        inQueueSrc.FreeTensor(xLocalReal);
        inQueueSrc.FreeTensor(xLocalImag);
        inQueueSrc.FreeTensor(yLocalReal);
        inQueueSrc.FreeTensor(yLocalImag);

        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopy(dstGlobal[dstBatchOffset + i * 2], zLocalReal, calcN * 2);

        outQueueDst.FreeTensor(zLocalReal);

        DataCopy(dstGlobal2[dstBatchOffset + i * 2], zLocalRealTmp, calcN * 2);

        outQueueDst.FreeTensor(zLocalRealTmp);
    }

    __aicore__ inline void ComputeUnpackPairTile(
        LocalTensor<T> &yLocalReal, int calcN, LocalTensor<T> &yLocalImag, LocalTensor<T> &xLocalImag,
        LocalTensor<T> &xLocalReal, int i, int dstBatchOffset)
    {
        LocalTensor<T> zLocalReal = outQueueDst.AllocTensor<T>();
        Gather(zLocalReal, yLocalReal, gatherLocal, 0, calcN);

        LocalTensor<T> zLocalImag = zLocalReal[TILE_LENGTH];
        Gather(zLocalImag, yLocalImag, gatherLocal, 0, calcN);

        LocalTensor<T> zLocalRealTmp = outQueueDst.AllocTensor<T>();
        LocalTensor<T> zLocalImagTmp = zLocalRealTmp[TILE_LENGTH];

        Add(zLocalRealTmp, xLocalImag, zLocalImag, calcN);
        Sub(zLocalImagTmp, zLocalReal, xLocalReal, calcN);

        Add(zLocalReal, xLocalReal, zLocalReal, calcN);
        Sub(zLocalImag, xLocalImag, zLocalImag, calcN);

        Muls(zLocalReal, zLocalReal, 0.5f, calcN);
        Muls(zLocalImag, zLocalImag, 0.5f, calcN);

        Muls(zLocalRealTmp, zLocalRealTmp, 0.5f, calcN);
        Muls(zLocalImagTmp, zLocalImagTmp, 0.5f, calcN);

        if (!i) {
            Adds(zLocalReal, xLocalReal, 0.f, 1);
            Adds(zLocalRealTmp, xLocalImag, 0.f, 1);
            Duplicate(zLocalImag, 0.f, 1);
            Duplicate(zLocalImagTmp, 0.f, 1);
        }

        StoreUnpackPairTile(xLocalReal, zLocalReal, calcN, xLocalImag, zLocalRealTmp, yLocalReal, yLocalImag,
            dstBatchOffset, i);
    }

    __aicore__ inline void ProcessUnpackPairTile(
        int idx, int tileNum, int calcN, int srcBatchStride, int dstBatchStride, int n)
    {
        if (tileNum == 0) {
            return;
        }
        int batchIdx = idx / tileNum;
        int tileIdx = idx % tileNum;
        int i = tileIdx * calcN;

        int srcBatchOffset = batchIdx * srcBatchStride;
        int dstBatchOffset = batchIdx * dstBatchStride;

        LocalTensor<T> xLocalReal = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> xLocalImag = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> yLocalReal = inQueueSrc.AllocTensor<T>();
        LocalTensor<T> yLocalImag = inQueueSrc.AllocTensor<T>();

        {
            DataCopy(xLocalReal, srcGlobalReal[srcBatchOffset + i], calcN);
            DataCopy(xLocalImag, srcGlobalImag[srcBatchOffset + i], calcN);

            DataCopy(yLocalReal, srcGlobalReal[srcBatchOffset + n - i - calcN + 1], calcN);
            DataCopy(yLocalImag, srcGlobalImag[srcBatchOffset + n - i - calcN + 1], calcN);

            int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        }

        {
            ComputeUnpackPairTile(yLocalReal, calcN, yLocalImag, xLocalImag, xLocalReal, i, dstBatchOffset);
        }
    }

public:
    __aicore__ inline void UnPack2(int n, int batch, int srcBatchStride, int dstBatchStride, int blockNum, int blockIdx) {
        if (batch <= 0) {
            return;
        }
        int calcN = GetUnpackTileLength(n, blockNum, blockIdx);
        if (calcN == 0 || blockNum == 0) {
            return;
        }
        int tileNum = n / 2 / calcN;
        if (tileNum <= 0) {
            return;
        }
        int totalNum = batch * tileNum;
        int averageNum = totalNum / blockNum;
        int remainderNum = totalNum % blockNum;
        int startIdx = averageNum * blockIdx + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        for (int idx = startIdx; idx < startIdx + calcNum; ++idx) {
            ProcessUnpackPairTile(idx, tileNum, calcN, srcBatchStride, dstBatchStride, n);
        }
    }
};

} // namespace Rfft1DR2CCommonKernel

#endif // RFFT1_D_R2C_UNPACK_H
