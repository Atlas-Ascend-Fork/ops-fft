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
 * \file fft1_d_stride.cpp
 * \brief
 */

#include "../../fft1_d.h"
#include "fft_common_core.h"
#include "fft1_d_stride_tilingdata.h"
#include "fft1_d_stride_kernel.h"

static uint32_t ComputeS0(uint32_t fftN, uint32_t strideSize)
{
    uint32_t s0 = 128;
    if (fftN == 32768 && strideSize >= 4096) {
        s0 = 256;
    } else if (fftN == 16384 && strideSize >= 8192) {
        s0 = 512;
    } else if (fftN >= 2048 && fftN <= 8192) {
        if (strideSize >= 512) {
            s0 = 512;
        } else if (strideSize >= 256) {
            s0 = 256;
        }
    } else {
        if (strideSize >= 1024) {
            s0 = 1024;
        } else if (strideSize >= 512) {
            s0 = 512;
        } else if (strideSize >= 256) {
            s0 = 256;
        }
    }
    return s0;
}

aclError aclfftFft1DStride(
    float* x, float* y, uint32_t n, uint32_t stride, uint32_t batches, int isForward, void* stream)
{
    // 防护: 本函数经 ACLFFT_API 导出可被外部直接调用，需校验输入输出指针（issue #76）
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftFft1DStride: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // 内核仅处理单个 n*stride 矩阵、host 缓冲也按单矩阵分配：batches>=2 时其余矩阵
    // 被静默跳过却返回成功，显式拒绝而非静默算错（issue #86）。
    // batches==0 同样拒绝（issue #188 补齐，PR #93 检视发现 #86 的 batches>1 检查
    // 未覆盖 0 值）：0 batch 时 kernel 不写入任何数据却返回成功
    if (batches != 1) {
        std::cerr << "[ops-fft] aclfftFft1DStride: batches=" << batches
                  << " is not supported (only single matrix batches=1 in this version)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = 1;
    }

    // 1、InitRadix()
    std::vector<uint32_t> radixVec;
    switch (n) {
        case 256:
            radixVec = {16, 16};
            break;
        case 512:
            radixVec = {16, 32};
            break;
        case 1024:
            radixVec = {16, 64};
            break;
        case 2048:
            radixVec = {32, 64};
            break;
        case 4096:
            radixVec = {64, 64};
            break;
        case 8192:
            radixVec = {16, 16, 32};
            break;
        case 16384:
            radixVec = {16, 32, 32};
            break;
        case 32768:
            radixVec = {32, 32, 32};
            break;
        case 65536:
            radixVec = {32, 32, 64};
            break;
        case 131072:
            radixVec = {32, 64, 64};
            break;
        case 262144:
            radixVec = {64, 64, 64};
            break;
        default:
            // 不再 throw：异常会穿透 extern "C" 边界导致 C 调用方进程终止（issue #72）
            std::cerr << "[ops-fft] aclfftFft1DStride: n=" << n << " not supported (must be power of 2 in [2^8, 2^18])"
                      << std::endl;
            return ACL_ERROR_INVALID_PARAM;
    }

    // 2、 InitSMatrix
    uint32_t totalSMatrixSize = 0;
    uint32_t pre = 1;
    for (auto radix : radixVec) {
        totalSMatrixSize += pre * radix * radix * 4;
        pre *= radix;
    }

    std::vector<float> sMatrixHost(totalSMatrixSize, 0.0f);
    uint32_t offset = 0;
    pre = 1;

    for (size_t it = 0; it < radixVec.size(); it++) {
        uint32_t radix = radixVec[it];

        for (uint32_t i = 0; i < pre; i++) {
            for (uint32_t m = 0; m < radix * radix; m++) {
                uint32_t j = m / radix;
                uint32_t k = m % radix;

                double angle = -K_2PI * (i + j * pre) * k / (pre * radix);
                float cos_val = cos(angle);
                float sin_val = sin(angle);

                uint32_t idx = offset + i * 4 * radix * radix + 2 * j * 2 * radix + k;
                sMatrixHost[idx] = cos_val;
                sMatrixHost[idx + radix] = sin_val * (isForward ? (-1.0) : (1.0));

                idx = offset + i * 4 * radix * radix + (2 * j + 1) * 2 * radix + k;
                sMatrixHost[idx] = sin_val * (isForward ? (1.0) : (-1.0));
                sMatrixHost[idx + radix] = cos_val;
            }
        }

        offset += pre * radix * radix * 4;
        pre *= radix;
    }

    uint32_t s0 = ComputeS0(n, stride);

    size_t inputSize = static_cast<size_t>(n) * stride * sizeof(float) * 2;
    uint32_t sMatrixSize = sMatrixHost.size() * sizeof(float);
    size_t outputSize = inputSize;
    size_t kernelWorkspaceSize = static_cast<size_t>(n) * s0 * sizeof(float) * 4;
    uint32_t tilingSize = sizeof(Fft1DStrideTilingData);
    uint32_t sysWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();
    size_t totalWorkspaceSize = kernelWorkspaceSize + sysWorkspaceSize;

    void* dev_input = nullptr;
    void* dev_s_matrix = nullptr;
    void* dev_output = nullptr;
    void* dev_workspace = nullptr;
    void* dev_tiling = nullptr;

    CHECK_ACL(aclrtMalloc(&dev_input, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_input_guard(dev_input);
    CHECK_ACL(aclrtMalloc(&dev_s_matrix, sMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_s_matrix_guard(dev_s_matrix);
    CHECK_ACL(aclrtMalloc(&dev_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_output_guard(dev_output);
    CHECK_ACL(aclrtMalloc(&dev_workspace, totalWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_workspace_guard(dev_workspace);
    CHECK_ACL(aclrtMalloc(&dev_tiling, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_tiling_guard(dev_tiling);

    CHECK_ACL(aclrtMemcpy(dev_input, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_s_matrix, sMatrixSize, sMatrixHost.data(), sMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Fft1DStrideTilingData tilingData;
    tilingData.batchSize = batches;
    tilingData.fftN = n;
    tilingData.fftStride = stride;
    tilingData.fftDirection = isForward ? -1 : 1;
    tilingData.iterCount = static_cast<uint32_t>(radixVec.size());
    for (size_t i = 0; i < radixVec.size() && i < 5; i++) {
        tilingData.radixVec[i] = radixVec[i];
    }
    tilingData.s0 = s0;

    CHECK_ACL(aclrtMemcpy(dev_tiling, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* ffts_addr_ptr = static_cast<uint8_t*>(dev_workspace) + kernelWorkspaceSize;

    uint8_t* sync = nullptr;
    CHECK_ACL(aclrtGetHardwareSyncAddr((void**)&sync));

    FFT1DStrideKernel::fft_stride<<<coreNum, nullptr, stream>>>(
        (__gm__ uint8_t*)sync, (__gm__ float*)dev_input, (__gm__ float*)dev_s_matrix, (__gm__ float*)dev_output,
        (__gm__ float*)dev_workspace, (__gm__ uint8_t*)dev_tiling);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
