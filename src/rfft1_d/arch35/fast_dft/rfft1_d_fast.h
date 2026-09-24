/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rfft1_d_fast.h
 * \brief
 */

#ifndef RFFT1_D_FAST_H
#define RFFT1_D_FAST_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"
#include "common_func.h"

#define CUBE_DFT
using DTYPE_X = float;
using DTYPE_Y = float;

using namespace matmul;

constexpr uint32_t BLOCK_LEN_FP32 = 8;
constexpr uint32_t MAX_VEC_ELEMS_PER_REP = 64;
// 工作索引分母：原值 2 导致仅一半 block 参与批次计算，batches 超过核数一半时尾部批次
// 输出为未初始化内存；改回 1 恢复全部 block 参与（issue #98，需 950 环境回归验证）
constexpr uint32_t CORE_IDX_DIV = 1;
constexpr uint32_t MAX_FACTORS_LEN = 3;
constexpr uint32_t RFFT_HALF = 2;
constexpr uint32_t COMPLEX = 2;

constexpr uint32_t NZ_BORDER = 8;
constexpr uint32_t NZ_BLOCK = 16;

constexpr uint32_t TILING_FAST_BASE_M = 128;
constexpr uint32_t TILING_FAST_BASE_N = 128;
constexpr uint32_t TILING_FAST_BASE_K = 64;
constexpr uint32_t TILING_FAST_BASE_DEFAULT = 16;
constexpr uint32_t TILING_FAST_DEPTH_A = 8;
constexpr uint32_t TILING_FAST_DEPTH_B = 7;

class KernelRfftFastDFT {
private:
    const uint32_t fftLength;
    // norm 成员已删除：全程无读取，归一化实际在 host 生成 DFT 矩阵时烤入
    // （Rfft1DDftGen 的 normParam，issue #121）
    const uint32_t batchesPerCore;
    const uint32_t leftOverBatches;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_X> dftGm;

    uint32_t dftOverallSize;
    const __gm__ float* dftMatrix;

public:
    TPipe pipe;
    using inputAType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X>;
    using inputBType = MatmulType<TPosition::GM, CubeFormat::NZ, DTYPE_X>;
    using outputCType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_Y>;
    Matmul<inputAType, inputAType, outputCType> matmulObj;
    Matmul<inputAType, inputBType, outputCType> matmulObjNZ;

    uint32_t batches;
    uint32_t advancedBatches;
    uint32_t totalBatches;
    uint32_t modeLength;
    uint32_t cores;
    uint32_t batchesPerCoreCeil;

public:
    // 构造参数 factors 已删除：原参数接收后从未存储、从未使用（issue #99）
    __aicore__ inline KernelRfftFastDFT(
        const uint32_t& length, const uint32_t& batchesPerCore, const uint32_t& leftOverBatches,
        const uint32_t& dftOverallSize)
        : fftLength(length),
          batchesPerCore(batchesPerCore),
          leftOverBatches(leftOverBatches),
          dftOverallSize(dftOverallSize)
    {
        modeLength = (fftLength / RFFT_HALF + 1) * COMPLEX;

        cores = GetBlockNum();
        ASSERT(cores != 0 && "block dim can not be zero!");

        batches = batchesPerCore * cores + leftOverBatches;
        advancedBatches = leftOverBatches == 0 ? 0 : cores - leftOverBatches;
        totalBatches = batches + advancedBatches;
        batchesPerCoreCeil = (batches + advancedBatches) / GetBlockNum();
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dftMatrix, GM_ADDR y)
    {
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalBatches * fftLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, batches * modeLength);

        dftGm.SetGlobalBuffer((__gm__ DTYPE_X*)dftMatrix, dftOverallSize);
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    // Copies the input from GM and splits the input; normalization is baked
    // into the DFT matrix on host (Rfft1DDftGen normParam), not done here (issue #121)
    __aicore__ inline void CopyIn()
    {
        if (g_coreType != AIV or GetSubBlockIdx() != 0) {
            return;
        }
    }

    __aicore__ void CubeDftMul(
        GlobalTensor<DTYPE_X>& a1, GlobalTensor<DTYPE_X>& b1, GlobalTensor<DTYPE_Y>& c1, bool accumulate = false)
    {
        matmulObj.SetTensorA(a1);
        matmulObj.SetTensorB(b1);
        matmulObj.IterateAll(c1, accumulate);

        matmulObj.End();
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void CubeDftMul1(
        GlobalTensor<DTYPE_X>& a1, GlobalTensor<DTYPE_X>& b1, GlobalTensor<DTYPE_Y>& c1, bool accumulate = false)
    {
        matmulObjNZ.SetTensorA(a1);
        matmulObjNZ.SetTensorB(b1);
        matmulObjNZ.IterateAll(c1, accumulate);

        matmulObjNZ.End();
        PipeBarrier<PIPE_ALL>();
    }

    // Count DFT directly (the entire input is multiplied on 1 DFT matrix)
    __aicore__ inline void CountWholeDft()
    {
        if (g_coreType == AIV and GetSubBlockIdx() != 0) {
            return;
        }
        int64_t offset;
        if (batches <= GetBlockNum()) {
            if (GetBlockIdx() / CORE_IDX_DIV < batches) {
                offset = totalBatches / cores * (GetBlockIdx() / CORE_IDX_DIV);
            } else {
                return;
            }
        } else if (GetBlockIdx() / CORE_IDX_DIV < batches * GetBlockNum() / totalBatches + 1) {
            offset = (GetBlockIdx() / CORE_IDX_DIV != batches * GetBlockNum() / totalBatches) ?
                         totalBatches / cores * (GetBlockIdx() / CORE_IDX_DIV) :
                         totalBatches / cores * (GetBlockIdx() / CORE_IDX_DIV - 1) + batches % batchesPerCoreCeil;
        } else {
            return;
        }
        int64_t offsetX = fftLength * offset;
        int64_t offsetY = modeLength * offset;

        GlobalTensor<DTYPE_X> inputX = xGm[offsetX];
        GlobalTensor<DTYPE_X> inputDFT = dftGm;
        GlobalTensor<DTYPE_Y> outputY = yGm[offsetY];
        if (fftLength % NZ_BLOCK && fftLength % NZ_BLOCK <= NZ_BORDER) {
            CubeDftMul(inputX, inputDFT, outputY);
        } else {
            CubeDftMul1(inputX, inputDFT, outputY);
        }
    }

    // Depending on the input length perform an FFT computation
    __aicore__ inline void Compute()
    {
        CountWholeDft();
    }

    // Merges real and imaginary parts into complex representation and moves to GM
    __aicore__ inline void CopyOut()
    {
        if (g_coreType != AIV or GetSubBlockIdx() != 0) {
            return;
        }
    }
};

__aicore__ inline TCubeTiling PrepareTiling(unsigned int m, unsigned int n, unsigned int k)
{
    TCubeTiling tiling;

    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    if (k % NZ_BLOCK > 0 && k % NZ_BLOCK <= NZ_BORDER) {
        tiling.Kb = k;
    } else {
        tiling.Kb = k + (BLOCK_LEN_FP32 - k % BLOCK_LEN_FP32);
    }
    tiling.singleCoreM = tiling.M;
    tiling.singleCoreN = tiling.N;
    tiling.singleCoreK = tiling.Ka;
    bool good_tiling = (k > MAX_VEC_ELEMS_PER_REP) || ((k & (k - 1)) == 0);
    tiling.baseM = good_tiling ? TILING_FAST_BASE_M : TILING_FAST_BASE_DEFAULT;
    tiling.baseN = good_tiling ? TILING_FAST_BASE_N : TILING_FAST_BASE_DEFAULT;
    tiling.baseK = good_tiling ? TILING_FAST_BASE_K : TILING_FAST_BASE_DEFAULT;
    tiling.depthA1 = TILING_FAST_DEPTH_A;
    tiling.depthB1 = TILING_FAST_DEPTH_B;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 0;
    tiling.transLength = 1;
    tiling.iterateOrder = 1;
    tiling.shareMode = 0;
    tiling.shareL1Size = 0;
    tiling.shareL0CSize = 0;
    tiling.shareUbSize = 0;
    tiling.batchM = 0;
    tiling.batchN = 0;
    tiling.singleBatchM = 0;
    tiling.singleBatchN = 0;

    return tiling;
}

#endif // RFFT1_D_FAST_H