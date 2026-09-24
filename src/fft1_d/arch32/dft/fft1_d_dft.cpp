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
 * \file fft1_d.cpp
 * \brief
 */

#include "../../fft1_d.h"
#include "fft_common_core.h"
#include "fft1_d_dft_tilingdata.h"
#include "fft1_d_dft_kernel.h"

static std::vector<float> InitRotationMatrix(int64_t nDoing)
{
    int64_t fftN = static_cast<int64_t>(nDoing);
    int64_t inSize = 2 * fftN;
    int64_t outSize = 2 * fftN;
    std::vector<float> rotationMatrixHost(outSize * inSize, INIT_VALUE);
    std::vector<float> cosTable(fftN);
    std::vector<float> sinTable(fftN);
    for (int64_t i = 0; i < fftN; i++) {
        cosTable[i] = cos(K_2PI * i / fftN);
        sinTable[i] = sin(K_2PI * i / fftN);
    }
    for (int64_t i = 0; i < fftN; i++) {
        for (int64_t j = 0; j < fftN; j++) {
            rotationMatrixHost[(2 * i) * (2 * fftN) + 2 * j] = cosTable[(i * j) % fftN];
            rotationMatrixHost[(2 * i) * (2 * fftN) + 2 * j + 1] = -sinTable[(i * j) % fftN];
            rotationMatrixHost[(2 * i + 1) * (2 * fftN) + 2 * j] = sinTable[(i * j) % fftN];
            rotationMatrixHost[(2 * i + 1) * (2 * fftN) + 2 * j + 1] = cosTable[(i * j) % fftN];
        }
    }
    return rotationMatrixHost;
}

static int SetTilingData(
    Fft1DDFTTilingData& tiling, uint32_t fftN, int32_t norm, int isInverse, uint32_t coreNum, uint32_t batches)
{
    const int thresholdK = 32768;
    const int cubeDataCountPerLoopSmall = 128;
    const int cubeDataCountPerLoopLarge = 64;

    tiling.batchSize = 1;
    tiling.m = batches;
    tiling.n = fftN * DFT_SIZE_MULTIPLIER;
    tiling.k = fftN * DFT_SIZE_MULTIPLIER;
    tiling.transA = 0;
    tiling.transB = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND910B) {
        if (static_cast<bool>(isInverse)) {
            tiling.transB = 1;
        }
    }
    tiling.m0 = cubeDataCountPerLoopSmall;
    tiling.n0 = cubeDataCountPerLoopSmall;
    tiling.k0 = cubeDataCountPerLoopSmall;
    if (tiling.k > thresholdK) {
        tiling.m0 = cubeDataCountPerLoopLarge;
        tiling.n0 = cubeDataCountPerLoopLarge;
        tiling.k0 = cubeDataCountPerLoopLarge;
    }
    return ACL_SUCCESS;
}

extern "C" aclError aclfftFft1DDft(
    float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream)
{
    // 防护: 本函数经 ACLFFT_API 导出可被外部直接调用，需校验输入输出指针（issue #76）
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftFft1DDft: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // 当前版本仅支持 norm=0（BACKWARD，无缩放）；norm!=0 显式报错而非静默忽略（issue #90）
    if (norm != 0) {
        std::cerr << "[ops-fft] aclfftFft1DDft: norm=" << norm << " is not supported (only 0=BACKWARD in this version)"
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // n 范围校验（issue #115）：n=0 会静默 no-op 返回成功误导调用方；超大 n 触发
    // (2n)^2 host 旋转矩阵分配，失败即 bad_alloc 穿透 extern "C" 终止进程。
    // 上限 1024 与 dispatch 路径的 DFT 内核支持域一致
    if (n == 0 || n > 1024) {
        std::cerr << "[ops-fft] aclfftFft1DDft: n=" << n << " out of range [1, 1024]" << std::endl;
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
    auto matrix = InitRotationMatrix(n);
    uint32_t sysWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();

    // 用 size_t 计算，避免 n*batches*8 在 uint32_t 域溢出回绕（大 batch 4GiB 场景，
    // 与 #74/#77 对 fft1_d_c2c.cpp 等的修复同款）（issue #87）
    const size_t inputSize = static_cast<size_t>(n) * batches * sizeof(float) * 2;
    const size_t matrixSize = matrix.size() * sizeof(float);
    const size_t outputSize = inputSize;

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    void* workspace_ptr = nullptr;

    // Allocate device memory for inputs and output
    // 守卫与 malloc 交错（issue #120）：任一分配失败时前序已分配内存自动释放
    // 智能指针指定删除器，在unique_ptr析构时自动调用指定的函数释放资源
    CHECK_ACL(aclrtMalloc(&dev_a, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_a_guard(dev_a);
    CHECK_ACL(aclrtMalloc(&dev_b, matrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_b_guard(dev_b);
    CHECK_ACL(aclrtMalloc(&dev_c, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_c_guard(dev_c);
    CHECK_ACL(aclrtMalloc(&workspace_ptr, sysWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_workspace_guard(workspace_ptr);

    // Copy inputs to device
    CHECK_ACL(aclrtMemcpy(dev_a, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_b, matrixSize, matrix.data(), matrixSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Fft1DDFTTilingData tilingData;

    // set tiling data
    int isInverse = 1 - isForward; // 正向传播则不需要inverse
    aclError ret = SetTilingData(tilingData, n, norm, isInverse, coreNum, batches);

    // Call host-side launcher (must be implemented to actually launch kernel)

    dft<<<coreNum, nullptr, stream>>>(
        (__gm__ float*)dev_a, (__gm__ float*)dev_b, (__gm__ float*)dev_c, (__gm__ uint8_t*)workspace_ptr, tilingData);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Copy device output back to host
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_c, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
