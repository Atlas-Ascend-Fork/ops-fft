/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../rfft1_d.h"
#include "fft_common_core.h"
#include "rfft1_d_fft_tilingdata.h"
#include "rfft1_d_fft_kernel.h"
#include "../../rfft1_d_size_utils.h"

static int32_t FindRadixHost(int64_t n)
{
    if (n % 2 == 0)
        return 2;
    if (n % 3 == 0)
        return 3;
    if (n % 5 == 0)
        return 5;
    if (n % 7 == 0)
        return 7;
    return 0;
}

static std::vector<float> GenerateDftMatrixHost(int32_t radix)
{
    int64_t rows = radix;
    int64_t cols = radix;
    std::vector<float> dft(2 * rows * cols, 0.0f);
    constexpr double K_2PI = 2.0 * K_PI;
    double sign = -1.0;

    for (int64_t u = 0; u < rows; u++) {
        for (int64_t v = 0; v < cols; v++) {
            double angle = sign * K_2PI * u * v / (radix ? radix : 1);
            dft[(2 * u) * radix + v] = static_cast<float>(cos(angle));
            dft[(2 * u + 1) * radix + v] = static_cast<float>(sin(angle));
        }
    }

    return dft;
}

static std::vector<float> GenerateTwiddleFactorsHost(int32_t radix, int64_t M)
{
    int64_t totalElements = 2 * radix * M;
    std::vector<float> tw(totalElements, 0.0f);
    constexpr double K_2PI = 2.0 * K_PI;
    double sign = -1.0;
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

static std::vector<float> GenerateTwPostProcess(int64_t fftN)
{
    int64_t D = fftN / 2 + 1;
    std::vector<float> twPost(2 * D, 0.0f);
    constexpr double K_2PI = 2.0 * K_PI;
    double sign = -1.0;

    for (int64_t k = 0; k < D; k++) {
        double angle = sign * K_2PI * k / (fftN ? fftN : 1);
        twPost[k * 2] = static_cast<float>(cos(angle));
        twPost[k * 2 + 1] = static_cast<float>(sin(angle));
    }

    return twPost;
}

static int SetTilingData(
    Rfft1DFftTilingData& tiling, int64_t fftN, int32_t isInverse, uint32_t coreNum, uint32_t batches)
{
    tiling.batchSize = batches;
    tiling.fftN = fftN;
    tiling.isInverse = isInverse;
    tiling.isOddN = (fftN % 2 != 0) ? 1 : 0;

    bool isOddN = (fftN % 2 != 0);
    int64_t tempN = isOddN ? fftN : (fftN / 2);
    int64_t currentBatch = 1;
    int64_t dftOffset = 0;
    int64_t twOffset = 0;
    int32_t stageCount = 0;

    while (tempN > 1 && stageCount < MAX_FFT_STAGES_TILING) {
        int32_t radix = FindRadixHost(tempN);
        if (radix == 0) {
            std::cerr << "rfft1_d_fft: cannot factorize N=" << tempN << std::endl;
            // 质因子超出 {2,3,5,7} 支持域，返回参数错误供 exec 层翻译为 NOT_IMPLEMENTED（issue #97）
            return ACL_ERROR_INVALID_PARAM;
        }

        int64_t M = tempN / radix;

        tiling.radix_arr[stageCount] = radix;
        tiling.M_arr[stageCount] = M;
        tiling.currentBatch_arr[stageCount] = currentBatch;
        tiling.dft_offset_arr[stageCount] = dftOffset;
        tiling.tw_offset_arr[stageCount] = twOffset;

        dftOffset += 2 * radix * radix;
        twOffset += 2 * radix * M;

        currentBatch *= radix;
        tempN = M;
        stageCount++;
    }

    tiling.radixListLen = stageCount;

    int64_t fftPointCount = isOddN ? fftN : (fftN / 2);
    int64_t perBatchComplexBytes = fftPointCount * sizeof(float) * 2;
    int64_t workspaceSize = 2 * perBatchComplexBytes * batches;

    tiling.workspaceOffsets[0] = 0;
    tiling.workspaceOffsets[1] = perBatchComplexBytes * batches;
    tiling.workspaceOffsets[2] = 0;
    tiling.workspaceOffsets[3] = 0;
    tiling.workspaceOffsets[4] = 0;

    return 0;
}

extern "C" aclError aclfftRfft1DFft(
    float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream)
{
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftRfft1DFft: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // 当前版本仅支持 norm=0（BACKWARD，无缩放）；norm!=0 显式报错而非静默忽略（issue #89）
    if (norm != 0) {
        std::cerr << "[ops-fft] aclfftRfft1DFft: norm=" << norm << " is not supported (only 0=BACKWARD in this version)"
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // isForward 域校验（issue #147 附注）：合法取值仅 0(正向)/1(反向)
    if (isForward != 0 && isForward != 1) {
        std::cerr << "[ops-fft] aclfftRfft1DFft: isForward=" << isForward << " invalid (expected 0 or 1)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // n 域校验（issue #140）：与 #130 修复的 aclfftFft1DC2C 口径一致，n ∈ (0, 2^27]。
    // 上界防止 O(n) 宿主 twiddle 分配失败抛 std::bad_alloc 穿透 extern "C"；
    // 下界防止 n=0 直通 tiling 推导启动内核。
    if (n == 0 || n > (1u << 27)) {
        std::cerr << "[ops-fft] aclfftRfft1DFft: n=" << n << " out of range (0, " << (1u << 27) << "]" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // batches==0 拒绝（issue #226，补齐 arch35 对 arch32 #188 同款守卫的遗漏）：
    // batches=0 时输入尺寸为 0，0 字节拷贝与空内核启动返回"成功"但输出缓冲未写入
    if (batches == 0) {
        std::cerr << "[ops-fft] aclfftRfft1DFft: batches=0 is not supported" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    int32_t isInverse = 1 - isForward;
    int64_t fftN = static_cast<int64_t>(n);

    Rfft1DFftTilingData tilingData = {};
    if (SetTilingData(tilingData, fftN, isInverse, coreNum, batches) != 0) {
        // 质因子不支持：按能力不支持返回参数错误，exec 层翻译为 NOT_IMPLEMENTED（issue #97）
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<float> allDftMatrices;
    std::vector<float> allTwiddleFactors;

    bool isOddN = (fftN % 2 != 0);
    int64_t tempN = isOddN ? fftN : (fftN / 2);

    for (int32_t s = 0; s < tilingData.radixListLen; s++) {
        int32_t radix = tilingData.radix_arr[s];
        int64_t M = tilingData.M_arr[s];

        auto dftMat = GenerateDftMatrixHost(radix);
        auto twMat = GenerateTwiddleFactorsHost(radix, M);

        allDftMatrices.insert(allDftMatrices.end(), dftMat.begin(), dftMat.end());
        allTwiddleFactors.insert(allTwiddleFactors.end(), twMat.begin(), twMat.end());

        tempN = M;
    }

    std::vector<float> twPostProcess;
    if (!isOddN) {
        twPostProcess = GenerateTwPostProcess(fftN);
    }

    const size_t inputSize = rfft1_d::InputSize(n, batches);
    const size_t outputSize = rfft1_d::OutputSize(n, batches);
    const size_t dftMatrixSize = allDftMatrices.size() * sizeof(float);
    const size_t twSize = allTwiddleFactors.size() * sizeof(float);
    const size_t twPostSize = twPostProcess.size() * sizeof(float);

    const size_t workspaceSize = rfft1_d::FftWorkspaceSize(n, batches);
    const size_t tilingSize = sizeof(Rfft1DFftTilingData);

    void* dev_input = nullptr;
    void* dev_output = nullptr;
    void* dev_dft_matrix = nullptr;
    void* dev_tw_matrix = nullptr;
    void* dev_tw_post = nullptr;
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
    CHECK_ACL(aclrtMalloc(&dev_tw_post, twPostSize > 0 ? twPostSize : 32, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_tw_post_guard(dev_tw_post);
    CHECK_ACL(aclrtMalloc(&dev_workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_ws_guard(dev_workspace);
    CHECK_ACL(aclrtMalloc(&dev_tiling, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_tiling_guard(dev_tiling);

    CHECK_ACL(aclrtMemcpy(dev_input, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(
        aclrtMemcpy(dev_dft_matrix, dftMatrixSize, allDftMatrices.data(), dftMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_tw_matrix, twSize, allTwiddleFactors.data(), twSize, ACL_MEMCPY_HOST_TO_DEVICE));
    if (twPostSize > 0) {
        CHECK_ACL(aclrtMemcpy(dev_tw_post, twPostSize, twPostProcess.data(), twPostSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // radix_list 死代码移除（issue #149）：内核仅从 tiling_buf 读取 radix 信息，
    // radix_list 参数从未被解引用，不再分配/构表/H2D 拷贝，传 nullptr 占位
    CHECK_ACL(aclrtMemcpy(dev_tiling, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    fft_r2c_multi_core<<<coreNum, nullptr, stream>>>(
        (__gm__ float *)dev_input,
        (__gm__ float *)dev_dft_matrix,
        (__gm__ float *)dev_tw_matrix,
        (__gm__ float *)dev_tw_post,
        nullptr,  // radix_list：内核不消费（issue #149），传 nullptr 占位
        (__gm__ float *)dev_output,
        (__gm__ float *)dev_workspace,
        (__gm__ uint8_t *)dev_tiling
    );

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
