/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <numeric>
#include <vector>

#include "rfft1_d_tilingdata.h"
#include "rfft1_d_fast.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "acl/acl.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"
#include "../../rfft1_d.h"
#include "../../rfft1_d_size_utils.h"

static const uint32_t LAST_FACTOR = 64;
static const uint32_t COMPLEX_PART = 2;
static const double INIT_VALUE = 0.;
static const uint32_t MATMUL_SIZE_MULTIPLIER = 24;
static const uint32_t SIZE_PER_BATCH_MULTIPLIER = 4;
static const uint32_t BYTES_ALIGN = 8;
static const uint32_t ROW_PAD = 16;
static const uint32_t COL_PAD = 8;
static const uint32_t TWIDDLE_MATRICES_AMOUNT = 2;
static const uint32_t DFT_BORDER_VALUE = 4096;
static const uint32_t DFT_OFFSETS_COUNT = 3;
static const uint32_t RFFT_SYMMETRY_DIVISOR = 2;
// 归一化值域与 rfft1_d.h 文档对齐：0=BACKWARD（不缩放）、1=ORTHO（1/sqrt(n)）、2=FORWARD（1/n）（issue #82）
static const uint32_t BACKWARD = 0;
static const uint32_t FORWARD = 2;

// ACL 错误检查宏：失败时返回真实 aclError 而非魔数 1（issue #93，与 fft_common_core.h 同款）
#define CHECK_ACL(call)                                                                              \
    do {                                                                                             \
        aclError err = (call);                                                                       \
        if (err != ACL_SUCCESS) {                                                                    \
            std::cerr << "ACL error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return err;                                                                              \
        }                                                                                            \
    } while (0)

struct AclrtFreeDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

// CalcCooleyTukeyFactors 已删除：其仅在 n==524288 或 n>4096 时写入 factors，而入口
// 拒绝 n>4096（DFT_BORDER_VALUE），两个分支均不可达，在全部可达输入下为纯空操作
// （issue #99）；factors 由调用方初始化为 {1,1,1} 后直接进入 Bluestein 计算

static void CalcBluesteinFactors(uint32_t factors[], std::vector<uint32_t> availableFactors, const uint32_t pow2)
{
    std::vector<uint32_t> factorsTmpBluestein;
    if (pow2 == LAST_FACTOR * LAST_FACTOR * LAST_FACTOR * COMPLEX_PART) {
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = LAST_FACTOR;
        }
        factors[0] = LAST_FACTOR * COMPLEX_PART;
    } else {
        uint32_t tmpN = pow2;
        int curFactorIndex = availableFactors.size() - 1;

        while (curFactorIndex >= 0) {
            uint32_t curFactor = availableFactors[curFactorIndex];
            while (tmpN % curFactor == 0) {
                tmpN /= curFactor;
                factorsTmpBluestein.emplace_back(curFactor);
            }
            curFactorIndex -= 1;
        }

        while (factorsTmpBluestein.size() < MAX_FACTORS_LEN) {
            factorsTmpBluestein.emplace_back(1);
        }

        for (size_t i = 0; i < MAX_FACTORS_LEN; ++i) {
            factors[i] = factorsTmpBluestein[i];
        }
    }
}

static void CalcDftSizes(Rfft1DTilingData& tiling, const uint32_t factors[], const bool isBluestein, const uint32_t n)
{
    uint32_t dftRealOverallSize = 0;
    uint32_t dftImagOverallSize = 0;
    uint32_t twiddleOverallSize = 0;

    uint32_t dftRealOffsets[DFT_OFFSETS_COUNT] = {0, 1, 1};
    uint32_t dftImagOffsets[DFT_OFFSETS_COUNT] = {0, 1, 1};
    uint32_t twiddleOffsets[DFT_OFFSETS_COUNT] = {0, 0, 1};

    size_t prevFactors = 1;

    for (size_t curIndex = 0; curIndex < MAX_FACTORS_LEN; ++curIndex) {
        size_t curFactor = factors[curIndex];
        size_t rowsNum = curFactor * (1 + static_cast<size_t>(curIndex == 0 && isBluestein));
        size_t colsNum = curFactor * (COMPLEX_PART - static_cast<size_t>(curIndex != 0));
        size_t dftCurSize = rowsNum * colsNum;
        size_t twiddleCurSize = rowsNum * prevFactors * COMPLEX_PART;

        if (curIndex != 0) {
            dftRealOffsets[curIndex] = dftRealOverallSize;
            dftImagOffsets[curIndex] = dftImagOverallSize;
            if (curIndex != 1) {
                twiddleOffsets[curIndex] = twiddleOverallSize;
            }
        }

        dftRealOverallSize += dftCurSize;
        if (curIndex != 0) {
            dftImagOverallSize += dftCurSize;
            twiddleOverallSize += twiddleCurSize;
        }

        prevFactors *= curFactor;
    }
    uint32_t fftPadRow = n + (COL_PAD - n % COL_PAD);
    uint32_t fftPadCol = ((n % ROW_PAD) != 0) ? n + (ROW_PAD - n % ROW_PAD) : n;
    dftRealOverallSize = n <= DFT_BORDER_VALUE ? fftPadRow * fftPadCol : dftRealOverallSize;

    tiling.dftRealOverallSize = dftRealOverallSize;
    tiling.dftImagOverallSize = dftImagOverallSize;
    tiling.twiddleOverallSize = twiddleOverallSize;
    tiling.fftMatrOverallSize = dftRealOverallSize + dftImagOverallSize + TWIDDLE_MATRICES_AMOUNT * twiddleOverallSize;

    for (size_t i = 0; i < DFT_OFFSETS_COUNT; i++) {
        tiling.dftRealOffsets[i] = dftRealOffsets[i];
        tiling.dftImagOffsets[i] = dftImagOffsets[i];
        tiling.twiddleOffsets[i] = twiddleOffsets[i];
    }
}

static int SetTilingData(Rfft1DTilingData& tiling, uint32_t n, int32_t norm, uint32_t coreNum, uint32_t batches)
{
    uint32_t factors[MAX_FACTORS_LEN] = {1, 1, 1};
    uint32_t prevRadices[MAX_FACTORS_LEN] = {1, 1, 1};
    uint32_t nextRadices[MAX_FACTORS_LEN] = {n / factors[0], 1, 1};
    uint8_t prevRadicesAlign[MAX_FACTORS_LEN] = {0, 1, 1};

    std::vector<uint32_t> availableFactors(LAST_FACTOR - 1);
    std::iota(availableFactors.begin(), availableFactors.end(), COMPLEX_PART);

    const bool isBluestein = (n % LAST_FACTOR != 0) || (factors[0] * factors[1] * factors[2] != n);
    const uint32_t pow2 = COMPLEX_PART * uint32_t(std::pow(2, std::ceil(std::log2(double(n)))));
    const uint32_t lengthPad = isBluestein ? pow2 : n;

    if (isBluestein) {
        CalcBluesteinFactors(factors, availableFactors, pow2);
    }

    for (uint8_t i = 1; i < MAX_FACTORS_LEN; ++i) {
        prevRadices[i] = prevRadices[i - 1] * factors[i - 1];
        nextRadices[i] = nextRadices[i - 1] / factors[i];
        prevRadicesAlign[i] = (COMPLEX_PART * prevRadices[i] % BYTES_ALIGN) == 0;
    }

    auto roundUpBlock = [](const uint32_t& src, const uint32_t blockLen) {
        return src != 0 ? src + (blockLen - src % (blockLen ? blockLen : 1)) % (blockLen ? blockLen : 1) : blockLen;
    };

    const uint32_t tailSize =
        COMPLEX_PART * (((n / RFFT_SYMMETRY_DIVISOR) + 1) - (factors[2] / COMPLEX_PART) * (n / factors[2]));
    const uint32_t tmpLenPerBatch = 3 * roundUpBlock(
                                            COMPLEX_PART * (isBluestein ? lengthPad : n) + factors[2] * tailSize + 1,
                                            BYTES_ALIGN * SIZE_PER_BATCH_MULTIPLIER);

    tiling.length = n;
    tiling.isBluestein = isBluestein;
    tiling.lengthPad = lengthPad;
    tiling.outLength = (n / RFFT_SYMMETRY_DIVISOR) + 1;
    tiling.tailSize = tailSize;
    tiling.batchesPerCore = batches / (coreNum ? coreNum : 1);
    tiling.leftOverBatches = batches % (coreNum ? coreNum : 1);
    tiling.tmpLenPerBatch = tmpLenPerBatch;
    tiling.tmpSizePerBatch = tmpLenPerBatch * SIZE_PER_BATCH_MULTIPLIER;
    tiling.normal = norm;
    tiling.matmulTmpsLen = MATMUL_SIZE_MULTIPLIER * tmpLenPerBatch;
    tiling.matmulTmpsSize = MATMUL_SIZE_MULTIPLIER * tmpLenPerBatch * sizeof(float);

    for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
        tiling.factors[i] = factors[i];
        tiling.prevRadices[i] = prevRadices[i];
        tiling.nextRadices[i] = nextRadices[i];
        tiling.prevRadicesAlign[i] = prevRadicesAlign[i];
    }
    CalcDftSizes(tiling, factors, isBluestein, n);

    return ACL_SUCCESS;
}

static std::vector<float> Rfft1DDftGen(int64_t fftLength, int64_t norm)
{
    std::vector<float> dftNz;
    double normParam = 1.;

    if (norm != BACKWARD) {
        normParam = norm == FORWARD ? normParam / double(fftLength) : normParam / sqrt(double(fftLength));
    }

    size_t fftLenPad = fftLength + (NZ_BORDER - fftLength % NZ_BORDER);
    size_t fftLenPadRow = fftLength + (NZ_BLOCK - fftLength % NZ_BLOCK);

    size_t fftOut = fftLength / RFFT_SYMMETRY_DIVISOR + 1;
    dftNz.reserve(fftLenPadRow * fftLenPadRow);

    std::vector<double> dftNd(fftLenPad * fftLenPadRow, INIT_VALUE);
    size_t k = 0;

    if ((fftLength % NZ_BLOCK != 0) && (fftLength % NZ_BLOCK <= NZ_BORDER)) {
        for (int i = 0; i < fftLength; ++i) {
            for (size_t j = 0; j < fftOut; ++j) {
                double param = -2. * M_PI * i * j / double(fftLength);
                dftNd[k] = normParam * cos(param);
                ++k;
                dftNd[k] = normParam * sin(param);
                ++k;
            }
        }
        dftNz.assign(dftNd.begin(), dftNd.end());
        return dftNz;
    } else {
        for (int i = 0; i < fftLength; ++i) {
            for (size_t j = 0; j < fftOut; ++j) {
                double param = -2. * M_PI * i * j / double(fftLength);
                dftNd[k] = normParam * cos(param);
                ++k;
                dftNd[k] = normParam * sin(param);
                ++k;
            }
            for (size_t j = 0; j + COMPLEX_PART * fftOut < fftLenPad; ++j) {
                dftNd[k] = INIT_VALUE;
                ++k;
            }
        }
        for (size_t n = 0; n < fftLenPad / NZ_BORDER; ++n) {
            for (size_t i = 0; i < fftLenPadRow; ++i) {
                for (size_t blockIdx = 0; blockIdx < NZ_BORDER; ++blockIdx) {
                    dftNz.emplace_back(static_cast<float>(dftNd[i * fftLenPad + NZ_BORDER * n + blockIdx]));
                }
            }
        }
        return dftNz;
    }
}

extern "C" __global__ __aicore__ void rfft1_d_fast_dft(
    GM_ADDR x, GM_ADDR dft, GM_ADDR y, GM_ADDR workspace, Rfft1DTilingData tilingData)
{
    if (tilingData.length <= DFT_BORDER_VALUE) {
        // norm 死传参已删除：归一化在 host 生成 DFT 矩阵时烤入（issue #121）
        KernelRfftFastDFT op(
            tilingData.length, tilingData.batchesPerCore, tilingData.leftOverBatches, tilingData.dftRealOverallSize);

        uint32_t modeLength = COMPLEX_PART * (tilingData.length / RFFT_SYMMETRY_DIVISOR + 1);
        auto t1 = PrepareTiling((op.batches + op.advancedBatches) / GetBlockNum(), op.modeLength, tilingData.length);
        REGIST_MATMUL_OBJ(&op.pipe, workspace, op.matmulObj, (void*)&t1, op.matmulObjNZ, (void*)&t1);

        op.Init(x, dft, y);
        op.Process();
    }
    return;
}

extern "C" aclError aclfftRfft1D(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, void* stream)
{
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftRfft1D: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    if (n == 0) {
        std::cerr << "[ops-fft] aclfftRfft1D: n must be positive, got 0" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    if (n > DFT_BORDER_VALUE) {
        std::cerr << "[ops-fft] aclfftRfft1D: n=" << n << " exceeds FastDFT limit (" << DFT_BORDER_VALUE
                  << "), not implemented" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // norm 合法值校验（issue #118）：Rfft1DDftGen 对 norm != BACKWARD 落入
    // else 分支按 1/sqrt(n)（ORTHO 口径）静默缩放，非法取值（如 5、-1）会
    // 产出错误缩放结果；与 #89/#90 的 norm 校验口径对齐
    if (norm < 0 || norm > 2) {
        std::cerr << "[ops-fft] aclfftRfft1D: norm=" << norm << " is invalid (expected 0=BACKWARD, 1=ORTHO, 2=FORWARD)"
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();
    // 平台信息查询失败时 GetCoreNumAic 可能返回 0（同仓其余 13 处入口均已兜底），
    // 以 0 block 启动内核行为未定义，D2H 会把未写入的缓冲当结果返回（issue #112）
    if (coreNum == 0)
    {
        coreNum = 1;
    }

    auto dft = Rfft1DDftGen(n, norm);

    uint32_t sysWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();

    const size_t inputSize = rfft1_d::InputSize(n, batches);
    const size_t dftSize = dft.size() * sizeof(float);
    const size_t outputSize = rfft1_d::OutputSize(n, batches);

    void* dev_x = nullptr;
    void* dev_dft = nullptr;
    void* dev_y = nullptr;
    void* workspace_ptr = nullptr;

    CHECK_ACL(aclrtMalloc(&dev_x, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_x_guard(dev_x);
    CHECK_ACL(aclrtMalloc(&dev_dft, dftSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_dft_guard(dev_dft);
    CHECK_ACL(aclrtMalloc(&dev_y, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_y_guard(dev_y);
    CHECK_ACL(aclrtMalloc(&workspace_ptr, sysWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> workspace_ptr_guard(workspace_ptr);

    CHECK_ACL(aclrtMemcpy(dev_x, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_dft, dftSize, dft.data(), dftSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Rfft1DTilingData tilingData = {};
    aclError ret = SetTilingData(tilingData, n, norm, coreNum, batches);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    rfft1_d_fast_dft<<<coreNum, nullptr, stream>>>(
        (__gm__ uint8_t*)dev_x, (__gm__ uint8_t*)dev_dft, (__gm__ uint8_t*)dev_y, (__gm__ uint8_t*)workspace_ptr,
        tilingData);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_y, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
