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
 * \file fft1_d_c2c.cpp
 * \brief Ascend950 C2C Stockham mixed-radix FFT host code
 * Handles N > 256 with mixed radix (2,3,5,7,11,13,17,19)
 */

#include "../../fft1_d.h"
#include "fft_common_core.h"
#include "fft1_d_c2c_tilingdata.h"
#include "fft1_d_c2c_kernel.h"

static int32_t FindRadix(int64_t n)
{
    if (n % 2 == 0)
        return 2;
    if (n % 3 == 0)
        return 3;
    if (n % 5 == 0)
        return 5;
    if (n % 7 == 0)
        return 7;
    if (n % 11 == 0)
        return 11;
    if (n % 13 == 0)
        return 13;
    if (n % 17 == 0)
        return 17;
    if (n % 19 == 0)
        return 19;
    return 0;
}

static std::vector<float> GenerateDftMatrix(int32_t radix, int32_t isInverse)
{
    int64_t rows = radix;
    int64_t cols = radix;
    std::vector<float> dft(2 * rows * cols, 0.0f);
    double sign = (isInverse == 0) ? -1.0 : 1.0;
    constexpr double K_2PI = 2.0 * K_PI;

    for (int64_t u = 0; u < rows; u++) {
        for (int64_t v = 0; v < cols; v++) {
            double angle = sign * K_2PI * u * v / (radix ? radix : 1);
            dft[(2 * u) * radix + v] = static_cast<float>(cos(angle));
            dft[(2 * u + 1) * radix + v] = static_cast<float>(sin(angle));
        }
    }

    return dft;
}

static std::vector<float> GenerateTwiddleFactors(int32_t radix, int64_t M, int32_t isInverse)
{
    int64_t totalElements = 2 * radix * M;
    std::vector<float> tw(totalElements, 0.0f);
    double sign = (isInverse == 0) ? -1.0 : 1.0;
    constexpr double K_2PI = 2.0 * K_PI;
    int64_t tempN = static_cast<int64_t>(radix) * M;

    for (int64_t u = 0; u < radix; u++) {
        for (int64_t n2 = 0; n2 < M; n2++) {
            double angle = sign * K_2PI * u * n2 / tempN;
            tw[(2 * u) * M + n2] = static_cast<float>(cos(angle));
            tw[(2 * u + 1) * M + n2] = static_cast<float>(sin(angle));
        }
    }

    return tw;
}

static int SetTilingData(
    Fft1DC2CTilingData& tiling, int64_t fftN, int32_t isInverse, uint32_t coreNum, uint32_t batches)
{
    tiling.batchSize = batches;
    tiling.fftN = fftN;
    tiling.isInverse = isInverse;

    int64_t tempN = fftN;
    int64_t tempBatch = batches;
    int64_t dftOff = 0;
    int64_t twOff = 0;
    int32_t stageCount = 0;

    while (tempN > 1 && stageCount < MAX_FFT_STAGES_TILING) {
        int32_t radix = FindRadix(tempN);
        if (radix == 0) {
            std::cerr << "fft1_d_c2c: cannot factorize N=" << tempN << std::endl;
            // 质因子超出 2..19 支持域属能力不支持，返回参数错误供 exec 层翻译为
            // NOT_IMPLEMENTED（issue #97/#132，与 aclfftFft1DC2CMix 口径一致）
            return ACL_ERROR_INVALID_PARAM;
        }

        tiling.radix_arr[stageCount] = radix;
        tiling.M_arr[stageCount] = tempN / radix;
        tiling.currentBatch_arr[stageCount] = tempBatch;
        tiling.dft_offset_arr[stageCount] = dftOff;
        tiling.tw_offset_arr[stageCount] = twOff;

        tempBatch *= radix;
        dftOff += 2 * radix * radix;
        twOff += 2 * radix * tiling.M_arr[stageCount];
        tempN = tiling.M_arr[stageCount];
        stageCount++;
    }

    tiling.radixListLen = stageCount;

    int64_t singleBufferSize = batches * fftN * 2 * sizeof(float);
    int64_t workspaceSize = 2 * singleBufferSize;

    tiling.workspaceOffsets[0] = 0;
    tiling.workspaceOffsets[1] = singleBufferSize;
    tiling.workspaceOffsets[2] = 0;
    tiling.workspaceOffsets[3] = 0;
    tiling.workspaceOffsets[4] = 0;

    return 0;
}

extern "C" aclError aclfftFft1DC2C(
    float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream)
{
    // 防护: 本函数经 ACLFFT_API 导出可被外部直接调用，需校验输入输出指针（issue #81）
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftFft1DC2C: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // 当前版本仅支持 norm=0（BACKWARD，无缩放）；norm!=0 显式报错而非静默忽略（issue #90）
    if (norm != 0) {
        std::cerr << "[ops-fft] aclfftFft1DC2C: norm=" << norm << " is not supported (only 0=BACKWARD in this version)"
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // fftSize 上界校验（issue #130）：文档约束 n 不超过 2^27；超界时 twiddle 等
    // 宿主分配达 4n 字节量级（n=2^28 约 4GB），失败即 bad_alloc 穿透 extern "C"，
    // 且 uint32 上传尺寸回绕为 0 产生静默错误结果
    if (n == 0 || n > (1u << 27)) {
        std::cerr << "[ops-fft] aclfftFft1DC2C: n=" << n << " out of supported range (0, 2^27]" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    int32_t isInverse = 1 - isForward;
    int64_t fftN = static_cast<int64_t>(n);

    Fft1DC2CTilingData tilingData;
    if (SetTilingData(tilingData, fftN, isInverse, coreNum, batches) != 0) {
        // 透传 SetTilingData 的 ACL_ERROR_INVALID_PARAM（issue #132），不再返回魔法值 1
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<float> allDftMatrices;
    std::vector<float> allTwiddleFactors;
    int64_t tempN = fftN;
    int64_t dftOff = 0;
    int64_t twOff = 0;

    for (int32_t s = 0; s < tilingData.radixListLen; s++) {
        int32_t radix = tilingData.radix_arr[s];
        int64_t M = tilingData.M_arr[s];

        auto dftMat = GenerateDftMatrix(radix, isInverse);
        auto twMat = GenerateTwiddleFactors(radix, M, isInverse);

        allDftMatrices.insert(allDftMatrices.end(), dftMat.begin(), dftMat.end());
        allTwiddleFactors.insert(allTwiddleFactors.end(), twMat.begin(), twMat.end());

        tempN = M;
    }

    // 用 size_t 计算，避免 n*batches 等中间结果在 uint32_t 域溢出截断（issue #74）
    const size_t inputSize = static_cast<size_t>(n) * batches * sizeof(float) * 2;
    const size_t outputSize = inputSize;
    // 上传尺寸 size_t 口径（issue #130 防御性修复，#74 同款）：n=2^27 时
    // twSize 达 2^30，uint32 虽未回绕但放宽上限即回绕为 0
    const size_t dftMatrixSize = allDftMatrices.size() * sizeof(float);
    const size_t twSize = allTwiddleFactors.size() * sizeof(float);
    const size_t radixListSize = tilingData.radixListLen * sizeof(float);
    const size_t workspaceSize = 2 * static_cast<size_t>(batches) * n * sizeof(float) * 2;
    const uint32_t tilingSize = sizeof(Fft1DC2CTilingData);

    void* dev_input = nullptr;
    void* dev_output = nullptr;
    void* dev_dft_matrix = nullptr;
    void* dev_tw_matrix = nullptr;
    void* dev_radix_list = nullptr;
    void* dev_workspace = nullptr;
    void* dev_tiling = nullptr;

    CHECK_ACL(aclrtMalloc(&dev_input, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_input_guard(dev_input);
    CHECK_ACL(aclrtMalloc(&dev_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_output_guard(dev_output);
    CHECK_ACL(aclrtMalloc(&dev_dft_matrix, dftMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_dft_guard(dev_dft_matrix);
    CHECK_ACL(aclrtMalloc(&dev_tw_matrix, twSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_tw_guard(dev_tw_matrix);
    CHECK_ACL(aclrtMalloc(&dev_radix_list, radixListSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_radix_guard(dev_radix_list);
    CHECK_ACL(aclrtMalloc(&dev_workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_ws_guard(dev_workspace);
    CHECK_ACL(aclrtMalloc(&dev_tiling, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_tiling_guard(dev_tiling);

    CHECK_ACL(aclrtMemcpy(dev_input, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(
        aclrtMemcpy(dev_dft_matrix, dftMatrixSize, allDftMatrices.data(), dftMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_tw_matrix, twSize, allTwiddleFactors.data(), twSize, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<float> radixListHost(tilingData.radixListLen);
    for (int32_t s = 0; s < tilingData.radixListLen; s++) {
        radixListHost[s] = static_cast<float>(tilingData.radix_arr[s]);
    }
    CHECK_ACL(
        aclrtMemcpy(dev_radix_list, radixListSize, radixListHost.data(), radixListSize, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMemcpy(dev_tiling, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    fft_c2c_arch35_multi_core<<<coreNum, nullptr, stream>>>(
        (__gm__ float*)dev_input, (__gm__ float*)dev_dft_matrix, (__gm__ float*)dev_tw_matrix,
        (__gm__ float*)dev_radix_list, (__gm__ float*)dev_output, (__gm__ float*)dev_workspace,
        (__gm__ uint8_t*)dev_tiling);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}