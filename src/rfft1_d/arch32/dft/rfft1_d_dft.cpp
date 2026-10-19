/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>

#include "../rfft1_d.h"
#include "fft_common_core.h"
#include "rfft1_d_dft_tilingdata.h"
#include "rfft1_d_dft_kernel.h"

static constexpr int MUL_DIV_TWO = 2;
static constexpr int64_t DFT_R2C_WORKSPACE_SIZE = 32;
static constexpr int CUBE_DATA_COUNT_PER_LOOP_SMALL = 128;
static constexpr int CUBE_DATA_COUNT_PER_LOOP_LARGE = 64;
static constexpr int THRESHOLD_K = 32768;

static std::vector<float> GenerateDftMatrixR2C(int64_t fftN)
{
    int64_t M = fftN;
    int64_t N = (fftN / 2 + 1) * MUL_DIV_TWO;
    std::vector<float> dftMatrix(M * N, 0.0f);

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < fftN / 2 + 1; j++) {
            double angle = -2.0 * K_PI * i * j / (fftN ? fftN : 1);
            dftMatrix[i * N + 2 * j] = static_cast<float>(cos(angle));
            dftMatrix[i * N + 2 * j + 1] = static_cast<float>(sin(angle));
        }
    }

    return dftMatrix;
}

extern "C" aclError aclfftRfft1DDft(
    float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream)
{
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftRfft1DDft: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // 当前版本仅支持 norm=0（BACKWARD，无缩放）；norm!=0 显式报错而非静默忽略（issue #89）
    if (norm != 0) {
        std::cerr << "[ops-fft] aclfftRfft1DDft: norm=" << norm << " is not supported (only 0=BACKWARD in this version)"
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // n 范围校验（issue #131，与 #115 对 aclfftFft1DDft 口径一致）：文档约束
    // 1<=n<=1024；超界时 n^2 量级宿主 DFT 矩阵分配失败即 bad_alloc 穿透 C 边界
    if (n == 0 || n > 1024) {
        std::cerr << "[ops-fft] aclfftRfft1DDft: n=" << n << " out of range [1, 1024]" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // batches 上界防护（issue #162，对齐 fft1_d_n.cpp #119 口径）：uint32_t 的
    // batches 直接赋给 int32_t 时 >INT32_MAX 会截断为负值写入 tilingData.m，
    // 内核循环推导为负、循环体不执行，D2H 拷回未初始化输出仍返回成功
    // batches==0 拒绝（issue #188，与 n==0 同口径）：batches=0 时 inputSize=0
    // 触发 0 字节 aclrtMalloc 与空内核启动，返回"成功"但输出缓冲未写入
    if (batches == 0) {
        std::cerr << "[ops-fft] aclfftRfft1DDft: batches=0 is not supported" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    if (batches > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        std::cerr << "[ops-fft] aclfftRfft1DDft: batches=" << batches << " exceeds INT32_MAX" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = 1;
    }

    int64_t fftN = static_cast<int64_t>(n);
    int32_t M = batches;
    int32_t N = (fftN / 2 + 1) * MUL_DIV_TWO;
    int32_t K = fftN;

    Rfft1DDftTilingData tilingData;
    tilingData.batchSize = 1;
    tilingData.m = M;
    tilingData.n = N;
    tilingData.k = K;
    tilingData.transA = 0;
    tilingData.transB = 0;
    tilingData.m0 = CUBE_DATA_COUNT_PER_LOOP_SMALL;
    tilingData.n0 = CUBE_DATA_COUNT_PER_LOOP_SMALL;
    tilingData.k0 = CUBE_DATA_COUNT_PER_LOOP_SMALL;
    if (K > THRESHOLD_K) {
        tilingData.m0 = CUBE_DATA_COUNT_PER_LOOP_LARGE;
        tilingData.n0 = CUBE_DATA_COUNT_PER_LOOP_LARGE;
        tilingData.k0 = CUBE_DATA_COUNT_PER_LOOP_LARGE;
    }

    std::vector<float> dftMatrix = GenerateDftMatrixR2C(fftN);

    // 用 size_t 计算，避免 batches*fftN 等中间结果在 uint32_t 域溢出截断（issue #56）
    size_t inputSize = static_cast<size_t>(batches) * static_cast<size_t>(fftN) * sizeof(float);
    size_t dftMatrixSize = dftMatrix.size() * sizeof(float);
    size_t outputSize = static_cast<size_t>(batches) * (fftN / 2 + 1) * sizeof(float) * 2;
    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t tilingSize = sizeof(Rfft1DDftTilingData);

    void* dev_input = nullptr;
    void* dev_dft = nullptr;
    void* dev_output = nullptr;
    void* dev_workspace = nullptr;
    void* dev_tiling = nullptr;

    CHECK_ACL(aclrtMalloc(&dev_input, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_input_guard(dev_input);
    CHECK_ACL(aclrtMalloc(&dev_dft, dftMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_dft_guard(dev_dft);
    CHECK_ACL(aclrtMalloc(&dev_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_output_guard(dev_output);
    CHECK_ACL(aclrtMalloc(&dev_workspace, sysWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_ws_guard(dev_workspace);
    CHECK_ACL(aclrtMalloc(&dev_tiling, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_tiling_guard(dev_tiling);

    CHECK_ACL(aclrtMemcpy(dev_input, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_dft, dftMatrixSize, dftMatrix.data(), dftMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_tiling, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Rfft1DDftKernel::dft_r2c<<<coreNum, nullptr, stream>>>(
        (__gm__ float*)dev_input, (__gm__ float*)dev_dft, (__gm__ float*)dev_output, (__gm__ uint8_t*)dev_workspace,
        (__gm__ uint8_t*)dev_tiling);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
