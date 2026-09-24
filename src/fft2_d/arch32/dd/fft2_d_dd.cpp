/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../fft2_d.h"
#include "fft_common_core.h"
#include "fft2_d_dd_tilingdata.h"
#include "fft2_d_dd_kernel.h"

static int32_t CalcBatchNumsPerLoop(int32_t x, int32_t y)
{
    if (x == 128 && y == 128) return 1;
    if (x == 128 && y == 64) return 1;
    if (x == 128 && y == 32) return 2;
    if (x == 64 && y == 128) return 1;
    if (x == 64 && y == 64) return 2;
    if (x == 64 && y == 32) return 4;
    if (x == 32 && y == 128) return 2;
    if (x == 32 && y == 64) return 4;
    if (x == 32 && y == 32) return 8;
    return -1;
}

static std::vector<float> InitPQMatrix(int64_t fftN, bool forward, bool isP)
{
    int64_t inSize = 2 * fftN;
    int64_t outSize = 2 * fftN;

    std::vector<double> cosTable(fftN);
    std::vector<double> sinTable(fftN);
    for (int i = 0; i < fftN; i++) {
        cosTable[i] = cos(K_2PI * i / (fftN ? fftN : 1));
        sinTable[i] = sin(K_2PI * i / (fftN ? fftN : 1));
    }

    std::vector<float> pqMatrix(outSize * inSize, 0.0f);

    for (int i = 0; i < fftN; i++) {
        for (int j = 0; j < fftN; j++) {
            if (isP) {
                pqMatrix[(2 * i) * (2 * fftN) + j] = cosTable[(i * j) % (fftN ? fftN : 1)];
                pqMatrix[(2 * i + 1) * (2 * fftN) + j] = (forward ? -1.0f : 1.0f) * sinTable[(i * j) % (fftN ? fftN : 1)];
                pqMatrix[(2 * i) * (2 * fftN) + j + fftN] = (forward ? 1.0f : -1.0f) * sinTable[(i * j) % (fftN ? fftN : 1)];
                pqMatrix[(2 * i + 1) * (2 * fftN) + j + fftN] = cosTable[(i * j) % (fftN ? fftN : 1)];
            } else {
                pqMatrix[j * (2 * fftN) + (2 * i)] = cosTable[(i * j) % (fftN ? fftN : 1)];
                pqMatrix[j * (2 * fftN) + (2 * i + 1)] = (forward ? -1.0f : 1.0f) * sinTable[(i * j) % (fftN ? fftN : 1)];
                pqMatrix[(j + fftN) * (2 * fftN) + (2 * i)] = (forward ? 1.0f : -1.0f) * sinTable[(i * j) % (fftN ? fftN : 1)];
                pqMatrix[(j + fftN) * (2 * fftN) + (2 * i + 1)] = cosTable[(i * j) % (fftN ? fftN : 1)];
            }
        }
    }

    return pqMatrix;
}

extern "C" aclError aclfftFft2DDd(float *x, float *y, uint32_t fftX, uint32_t fftY,
                                   uint32_t batches, int isForward, void *stream) {
    // 防护: 本函数经 ACLFFT_API 导出可被外部直接调用，需校验输入输出指针（issue #53）
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftFft2DDd: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = 1;
    }

    int32_t batchNumsPerLoop = CalcBatchNumsPerLoop(fftX, fftY);
    if (batchNumsPerLoop <= 0) {
        std::cerr << "Unsupported fft sizes: " << fftX << "x" << fftY << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    size_t inputSize = static_cast<size_t>(batches) * fftX * fftY * sizeof(float) * 2;
    size_t outputSize = inputSize;

    std::vector<float> pMatrixHost = InitPQMatrix(fftX, isForward != 0, true);
    std::vector<float> qMatrixHost = InitPQMatrix(fftY, isForward != 0, false);
    uint32_t pMatrixSize = pMatrixHost.size() * sizeof(float);
    uint32_t qMatrixSize = qMatrixHost.size() * sizeof(float);

    uint64_t workspaceSize = static_cast<uint64_t>(batches) * fftX * fftY * sizeof(float) * 2;

    void *dev_input = nullptr;
    void *dev_output = nullptr;
    void *dev_pMatrix = nullptr;
    void *dev_qMatrix = nullptr;
    void *dev_workspace = nullptr;
    void *dev_tiling = nullptr;

    CHECK_ACL(aclrtMalloc(&dev_input, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_input_guard(dev_input);
    CHECK_ACL(aclrtMalloc(&dev_output, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_output_guard(dev_output);
    CHECK_ACL(aclrtMalloc(&dev_pMatrix, pMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_pMatrix_guard(dev_pMatrix);
    CHECK_ACL(aclrtMalloc(&dev_qMatrix, qMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_qMatrix_guard(dev_qMatrix);
    CHECK_ACL(aclrtMalloc(&dev_workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_workspace_guard(dev_workspace);
    CHECK_ACL(aclrtMalloc(&dev_tiling, sizeof(Fft2DTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_tiling_guard(dev_tiling);

    CHECK_ACL(aclrtMemcpy(dev_input, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_pMatrix, pMatrixSize, pMatrixHost.data(), pMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_qMatrix, qMatrixSize, qMatrixHost.data(), qMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Fft2DTilingData tilingData;
    tilingData.batchSize = batches;
    tilingData.fftX = fftX;
    tilingData.fftY = fftY;
    tilingData.batchNumsPerLoop = batchNumsPerLoop;

    CHECK_ACL(aclrtMemcpy(dev_tiling, sizeof(Fft2DTilingData), &tilingData, sizeof(Fft2DTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *sync = nullptr;
    CHECK_ACL(aclrtGetHardwareSyncAddr((void **)&sync));
    FFT2DKernel::dd<<<coreNum, nullptr, stream>>>(
        (__gm__ uint8_t *)sync,
        (__gm__ float *)dev_input,
        (__gm__ float *)dev_pMatrix,
        (__gm__ float *)dev_qMatrix,
        (__gm__ float *)dev_output,
        (__gm__ float *)dev_workspace,
        (__gm__ uint8_t *)dev_tiling
    );

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SUCCESS;
}
