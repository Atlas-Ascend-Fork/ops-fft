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
 * \file fft1_d_c2c_mix.cpp
 * \brief Ascend950 C2C mixed-radix Stockham FFT host code (ported from fft_c2c_arch35_mix_multi_core).
 * Handles N>256 radix-mix (factors in {2,3,5,7,11,13,17,19}). dft/tw layout matches original exactly.
 */

#include "../../fft1_d.h"
#include "fft_common_core.h"
#include "fft1_d_c2c_mix_tilingdata.h"
#include "fft1_d_c2c_mix_kernel.h"

static constexpr int64_t MIX_ALLOWED_RADICES[] = {2, 3, 5, 7, 11, 13, 17, 19};

static int SetMixTilingData(
    FftAllMixTilingData& t, int64_t fftN, int32_t isInverse, uint32_t batches, std::vector<int32_t>& radixList)
{
    t.batchSize = static_cast<int64_t>(batches);
    t.fftN = fftN;
    t.isInverse = isInverse;
    t.outer = 1;
    t.scaleOut = 0;
    t.transpose = 0;
    t.isOddN = 0;
    for (int i = 0; i < 5; i++) {
        t.workspaceOffsets[i] = 0;
    }
    for (int i = 0; i < MAX_FFT_STAGES_TILING_MIX; i++) {
        t.radix_arr[i] = 0;
        t.M_arr[i] = 0;
        t.dft_offset_arr[i] = 0;
        t.tw_offset_arr[i] = 0;
        t.currentBatch_arr[i] = 0;
    }

    radixList.clear();
    int64_t tempN = fftN;
    // tempN > 1 前置（issue #129）：n=0 直调时 0 % r == 0 恒成立且 0/r 仍为 0，
    // 原循环永不退出，调用线程宿主死循环挂死
    for (size_t i = 0; i < sizeof(MIX_ALLOWED_RADICES) / sizeof(MIX_ALLOWED_RADICES[0]); i++) {
        int64_t r = MIX_ALLOWED_RADICES[i];
        while (tempN > 1 && tempN % r == 0) {
            tempN /= r;
            radixList.push_back(static_cast<int32_t>(r));
        }
    }
    if (tempN != 1) {
        std::cerr << "fft1_d_c2c_mix: unsupported factor remains: " << tempN << std::endl;
        // 质因子超出支持域属能力不支持，返回参数错误供 exec 层翻译为 NOT_IMPLEMENTED（issue #97）
        return ACL_ERROR_INVALID_PARAM;
    }
    if (radixList.empty()) {
        std::cerr << "fft1_d_c2c_mix: empty radix list for N=" << fftN << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    t.radixListLen = static_cast<int32_t>(radixList.size());
    return 0;
}

// original-exact dft layout: per stage 2*r*r floats, [re,im] at (q*r+p)*2
static std::vector<float> GenerateMixDft(const std::vector<int32_t>& radixList, int32_t isInverse)
{
    double sign = (isInverse == 0) ? -1.0 : 1.0;
    size_t total = 0;
    for (auto r : radixList) {
        total += static_cast<size_t>(r) * r * 2;
    }
    std::vector<float> dft(total, 0.0f);
    size_t offset = 0;
    for (auto r : radixList) {
        for (int64_t q = 0; q < r; q++) {
            for (int64_t p = 0; p < r; p++) {
                double angle = sign * K_2PI * p * q / r;
                dft[(offset + q * r + p) * 2] = static_cast<float>(std::cos(angle));
                dft[(offset + q * r + p) * 2 + 1] = static_cast<float>(std::sin(angle));
            }
        }
        offset += static_cast<size_t>(r) * r;
    }
    return dft;
}

// original-exact twiddle layout: per stage 2*r*prev floats, [re,im] at (p*prev+j)*2
static std::vector<float> GenerateMixTwiddle(const std::vector<int32_t>& radixList, int32_t isInverse)
{
    double sign = (isInverse == 0) ? -1.0 : 1.0;
    size_t total = 0;
    int64_t prev = 1;
    int64_t len = 1;
    for (auto r : radixList) {
        len *= r;
        total += static_cast<size_t>(r) * prev * 2;
        prev = len;
    }
    std::vector<float> tw(total, 0.0f);
    size_t offset = 0;
    prev = 1;
    len = 1;
    for (auto r : radixList) {
        len *= r;
        for (int64_t p = 0; p < r; p++) {
            for (int64_t j = 0; j < prev; j++) {
                double angle = sign * K_2PI * p * j / len;
                tw[(offset + p * prev + j) * 2] = static_cast<float>(std::cos(angle));
                tw[(offset + p * prev + j) * 2 + 1] = static_cast<float>(std::sin(angle));
            }
        }
        offset += static_cast<size_t>(r) * prev;
        prev = len;
    }
    return tw;
}

extern "C" aclError aclfftFft1DC2CMix(
    float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream)
{
    // 防护: 本函数经 ACLFFT_API 导出可被外部直接调用，需校验输入输出指针（issue #52）
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftFft1DC2CMix: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // 当前版本仅支持 norm=0（BACKWARD，无缩放）；norm!=0 显式报错而非静默忽略（issue #90）
    if (norm != 0) {
        std::cerr << "[ops-fft] aclfftFft1DC2CMix: norm=" << norm
                  << " is not supported (only 0=BACKWARD in this version)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    (void)norm;
    // fftSize 上界校验（issue #130，同 aclfftFft1DC2C）：文档约束 n 不超过 2^27
    if (n == 0 || n > (1u << 27)) {
        std::cerr << "[ops-fft] aclfftFft1DC2CMix: n=" << n << " out of supported range (0, 2^27]" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    int32_t isInverse = 1 - isForward;
    int64_t fftN = static_cast<int64_t>(n);

    FftAllMixTilingData tilingData;
    std::vector<int32_t> radixListHost;
    if (SetMixTilingData(tilingData, fftN, isInverse, batches, radixListHost) != 0) {
        // 质因子不支持：按能力不支持返回参数错误，exec 层翻译为 NOT_IMPLEMENTED（issue #97）
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<float> allDftMatrices = GenerateMixDft(radixListHost, isInverse);
    std::vector<float> allTwiddleFactors = GenerateMixTwiddle(radixListHost, isInverse);

    // 用 size_t 计算，避免 n*batches 等中间结果在 uint32_t 域溢出截断（issue #74）
    const size_t inputSize = static_cast<size_t>(n) * batches * sizeof(float) * 2;
    const size_t outputSize = inputSize;
    // 上传尺寸 size_t 口径（issue #130 防御性修复，#74 同款）
    const size_t dftMatrixSize = allDftMatrices.size() * sizeof(float);
    const size_t twSize = allTwiddleFactors.size() * sizeof(float);
    const size_t radixListSize = radixListHost.size() * sizeof(int32_t);
    const size_t workspaceSize = 2 * static_cast<size_t>(batches) * n * sizeof(float) * 2;
    const uint32_t tilingSize = sizeof(FftAllMixTilingData);

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
    CHECK_ACL(
        aclrtMemcpy(dev_radix_list, radixListSize, radixListHost.data(), radixListSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_tiling, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    fft_c2c_arch35_mix_multi_core<<<coreNum, nullptr, stream>>>(
        (__gm__ float*)dev_input, (__gm__ float*)dev_dft_matrix, (__gm__ float*)dev_tw_matrix,
        (__gm__ int32_t*)dev_radix_list, (__gm__ float*)dev_output, (__gm__ float*)dev_workspace,
        (__gm__ uint8_t*)dev_tiling);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
