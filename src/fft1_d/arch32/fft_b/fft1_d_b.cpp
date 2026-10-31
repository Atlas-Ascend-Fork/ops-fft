/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../fft1_d.h"
#include "fft_common_core.h"
#include "fft1_d_b_tilingdata.h"
#include "fft1_d_b_kernel.h"

/* ========================================================================== */
/* fft_b: radix-2 batched FFT (n = 256~262144, batch >= 2) implementation     */
/* ========================================================================== */

static bool InitRadixB(uint32_t n, std::vector<uint32_t> &radixVec)
{
    radixVec.clear();
    switch (n) {
        case 256:    radixVec = {16, 16}; break;
        case 512:    radixVec = {16, 32}; break;
        case 1024:   radixVec = {16, 64}; break;
        case 2048:   radixVec = {32, 64}; break;
        case 4096:   radixVec = {64, 64}; break;
        case 8192:   radixVec = {64, 64}; break;
        case 16384:  radixVec = {64, 64}; break;
        case 32768:  radixVec = {32, 32, 32}; break;
        case 65536:  radixVec = {32, 32, 64}; break;
        case 131072: radixVec = {32, 64, 64}; break;
        case 262144: radixVec = {64, 64, 64}; break;
        default:
            // 不再 throw：异常会穿透 extern "C" 边界导致 C 调用方进程终止（issue #72）
            return false;
    }
    return true;
}

static std::vector<int32_t> InitIndexB(uint32_t n, const std::vector<uint32_t> &radixVec)
{
    std::vector<int32_t> index(128 * 128, 0);

    if (n > 16384) {
        for (int64_t i = 0; i < 128 * 64; ++i) {
            index[i * 2] = i;
            index[i * 2 + 1] = i;
        }
    } else if (n == 8192 || n == 16384) {
        for (int64_t i = 0; i < 128 * 64; ++i) {
            index[i * 2] = i * 4;
            index[i * 2 + 1] = (64 * 64 + i) * 4;
        }
    } else {
        int64_t nRadix = radixVec[0];
        int64_t M = radixVec[1] * 2;
        if (M == 0) M = 1;
        int64_t col = 128 * 128 / M;
        int64_t fftN = nRadix * M / 2;
        if (radixVec[0] == radixVec[1] && radixVec[1] == 16) {
            col = col / 2;
        }
        for (int64_t batchId = 0; batchId < (col / nRadix); batchId++) {
            for (int64_t i = 0; i < (M / 2 / 2); i++) {
                for (int64_t j = 0; j < nRadix; j++) {
                    index[batchId * fftN + (i * nRadix + j) * 2] =
                        (batchId * nRadix + i * col + j) * 4;
                    index[batchId * fftN + (i * nRadix + j) * 2 + 1] =
                        (M / 4 * col + batchId * nRadix + i * col + j) * 4;
                }
            }
        }
    }
    return index;
}

static std::vector<float> InitWMatrixRadix2(uint32_t radix, bool forward, bool isLastIter)
{
    int32_t size = 2 * radix * 2 * radix;
    std::vector<float> matrix(size, 0.f);
    double K_2PI = 2.0 * 3.14159265358979323846;

    if (isLastIter) {
        for (int64_t k = 0; k < radix * radix; ++k) {
            int64_t i = k / (radix ? radix : 1);
            int64_t j = k % (radix ? radix : 1);
            matrix[i * 2 * radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
            matrix[i * 2 * radix + radix + j] = sin(-1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(i + radix) * 2 * radix + j] = sin(1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(i + radix) * 2 * radix + radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
        }
    } else {
        for (int64_t k = 0; k < radix * radix; ++k) {
            int64_t i = k / (radix ? radix : 1);
            int64_t j = k % (radix ? radix : 1);
            matrix[2 * i * 2 * radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
            matrix[2 * i * 2 * radix + radix + j] = sin(-1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(2 * i + 1) * 2 * radix + j] = sin(1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(2 * i + 1) * 2 * radix + radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
        }
    }
    return matrix;
}

static std::vector<float> InitTMatrix(uint32_t fftN, const std::vector<uint32_t>& radixVec, uint32_t iterIndex, bool forward)
{
    double K_2PI = 2.0 * 3.14159265358979323846;

    uint32_t tempRow = radixVec[iterIndex] * 2;
    uint32_t tempCol = 1;
    for (uint32_t j = iterIndex + 1; j < radixVec.size(); j++) {
        tempCol *= radixVec[j];
    }

    uint32_t size = tempRow * tempCol;
    std::vector<float> matrix(size, 0.f);

    double angle_base = -1.0 * K_2PI / (tempRow / 2 * tempCol);

    for (uint32_t i = 0; i < tempRow / 2; ++i) {
        for (uint32_t j = 0; j < tempCol; ++j) {
            double angle = angle_base * i * j;
            matrix[2 * i * tempCol + j] = static_cast<float>(cos(angle));
            matrix[(2 * i + 1) * tempCol + j] = static_cast<float>(sin(angle));
        }
    }
    return matrix;
}

constexpr int32_t SCRATCH_SIZES_B = sizeof(float) * 655360;

// 入参校验链（!102 codecheck 拆分）：经 CHECK_ACL 调用保持失败早退语义
static aclError ValidateFft1BParams(const float *x, const float *y, int32_t norm, int isForward, uint32_t batches)
{
    // 防护: 本函数经 ACLFFT_API 导出可被外部直接调用，需校验输入输出指针（issue #76）
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftFft1DB: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // 当前版本仅支持 norm=0（BACKWARD，无缩放）；norm!=0 显式报错而非静默按
    // BACKWARD 计算（issue #224，对齐 dft/fft_n 路径口径，消除 norm 行为随档位分裂）
    if (norm != 0) {
        std::cerr << "[ops-fft] aclfftFft1DB: norm=" << norm << " is not supported (only 0=BACKWARD in this version)"
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // isForward 域校验（issue #229，对齐 dft/arch35 口径）：合法取值仅 0(正向)/1(反向)。
    // 本入口无校验时任意非 0 值经 (isForward != 0) 判定静默按正向执行，方向语义与
    // 文档 0/1 两值契约不符
    if (isForward != 0 && isForward != 1) {
        std::cerr << "[ops-fft] aclfftFft1DB: isForward=" << isForward
                  << " invalid (expected 0 or 1)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // batches==0 拒绝（issue #188，与 n==0 同口径）：batches=0 时 inputSize=0
    // 触发 0 字节 aclrtMalloc 与空内核启动，返回"成功"但输出缓冲未写入；
    // 原 needCoreNum==0 兜底掩盖了该非法输入
    if (batches == 0) {
        std::cerr << "[ops-fft] aclfftFft1DB: batches=0 is not supported" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

// host 侧旋转因子矩阵 / T 矩阵 / 索引表构建（!102 codecheck 拆分）
static void BuildFft1BHostTables(uint32_t n, const std::vector<uint32_t> &radixVec, int isForward,
                                 std::vector<float> &wMatrixHost, uint32_t &wMatrixSize,
                                 std::vector<float> &tMatrixHost, uint32_t &tMatrixSize,
                                 std::vector<int32_t> &indexHost, uint32_t &indexSize)
{
    for (size_t i = 0; i < radixVec.size(); i++) {
        uint32_t radix = radixVec[i];
        uint32_t stageSize = 2 * radix * 2 * radix;
        size_t wBias = wMatrixHost.size();
        wMatrixHost.resize(wBias + stageSize);
        bool isLastIter = (i == radixVec.size() - 1);
        std::vector<float> stageMatrix = InitWMatrixRadix2(radix, isForward != 0, isLastIter);
        for (uint32_t j = 0; j < stageSize; j++) {
            wMatrixHost[wBias + j] = stageMatrix[j];
        }
    }
    wMatrixSize = wMatrixHost.size() * sizeof(float);

    std::vector<uint32_t> tRadixVec = radixVec;
    if (n == 8192) {
        tRadixVec = {2, 64, 64};
    } else if (n == 16384) {
        tRadixVec = {4, 64, 64};
    }
    for (size_t i = 0; i < tRadixVec.size() - 1; i++) {
        std::vector<float> stageT = InitTMatrix(n, tRadixVec, static_cast<uint32_t>(i), isForward != 0);
        tMatrixHost.insert(tMatrixHost.end(), stageT.begin(), stageT.end());
    }
    tMatrixSize = tMatrixHost.size() * sizeof(float);

    indexHost = InitIndexB(n, radixVec);
    indexSize = indexHost.size() * sizeof(int32_t);
}

// 设备缓冲集合（!102 codecheck 拆分）：成员 unique_ptr 析构即释放
struct Fft1BDevBuffers {
    std::unique_ptr<void, AclrtFreeDeleter> input, output, wMatrix, tMatrix, index, workspace, tiling;
};

// 设备缓冲分配与 H2D 上行（!102 codecheck 拆分）：守卫与 malloc 交错（issue #120）的
// 失败自动释放语义由成员 unique_ptr 构造顺序等价保持
static aclError AllocFft1BDevBuffers(Fft1BDevBuffers &dev, const float *x, size_t inputSize, size_t outputSize,
                                     const std::vector<float> &wMatrixHost, size_t wMatrixSize,
                                     const std::vector<float> &tMatrixHost, size_t tMatrixSize,
                                     const std::vector<int32_t> &indexHost, size_t indexSize, size_t workspaceSize)
{
    void *ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.input.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.output.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, wMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.wMatrix.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, tMatrixSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.tMatrix.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, indexSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.index.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.workspace.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, sizeof(Fft1DBTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    dev.tiling.reset(ptr);
    CHECK_ACL(aclrtMemcpy(dev.input.get(), inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev.wMatrix.get(), wMatrixSize, wMatrixHost.data(), wMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev.tMatrix.get(), tMatrixSize, tMatrixHost.data(), tMatrixSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev.index.get(), indexSize, indexHost.data(), indexSize, ACL_MEMCPY_HOST_TO_DEVICE));
    return ACL_SUCCESS;
}

// tiling 填充（!102 codecheck 拆分）
static void FillFft1BTiling(Fft1DBTilingData &tilingData, uint32_t n, uint32_t batches, int isForward,
                            const std::vector<uint32_t> &radixVec)
{
    tilingData.nFFT = n;
    tilingData.batchSize = batches;
    tilingData.fftDirection = (isForward != 0) ? -1 : 1;
    tilingData.iterCount = static_cast<uint32_t>(radixVec.size());
    for (uint32_t i = 0; i < radixVec.size() && i < FFT_B_MAX_ITERATION; i++) {
        tilingData.radixVec[i] = radixVec[i];
    }
}

extern "C" aclError aclfftFft1DB(float *x, float *y, uint32_t n, int32_t norm,
                                  uint32_t batches, int isForward, void *stream) {
    CHECK_ACL(ValidateFft1BParams(x, y, norm, isForward, batches));
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = 1;
    }
    uint32_t needCoreNum = batches > coreNum ? coreNum : batches;
    if (needCoreNum == 0) needCoreNum = 1;

    std::vector<uint32_t> radixVec;
    if (!InitRadixB(n, radixVec)) {
        std::cerr << "[ops-fft] aclfftFft1DB: n=" << n
                  << " not supported (must be power of 2 in [2^8, 2^18])" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    size_t inputSize = static_cast<size_t>(n) * batches * sizeof(float) * 2;
    size_t outputSize = inputSize;

    std::vector<float> wMatrixHost;
    std::vector<float> tMatrixHost;
    std::vector<int32_t> indexHost;
    uint32_t wMatrixSize = 0;
    uint32_t tMatrixSize = 0;
    uint32_t indexSize = 0;
    BuildFft1BHostTables(n, radixVec, isForward, wMatrixHost, wMatrixSize, tMatrixHost, tMatrixSize,
                         indexHost, indexSize);

    uint32_t totalWorkspaceSize = SCRATCH_SIZES_B * needCoreNum;

    Fft1BDevBuffers dev;
    CHECK_ACL(AllocFft1BDevBuffers(dev, x, inputSize, outputSize, wMatrixHost, wMatrixSize,
                                   tMatrixHost, tMatrixSize, indexHost, indexSize, totalWorkspaceSize));

    Fft1DBTilingData tilingData;
    FillFft1BTiling(tilingData, n, batches, isForward, radixVec);
    CHECK_ACL(aclrtMemcpy(dev.tiling.get(), sizeof(Fft1DBTilingData), &tilingData, sizeof(Fft1DBTilingData),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *sync = nullptr;
    CHECK_ACL(aclrtGetHardwareSyncAddr((void **)&sync));
    FFT1DBKernel::fft_b<<<needCoreNum, nullptr, stream>>>(
        (__gm__ uint8_t *)sync, (__gm__ float *)dev.input.get(), (__gm__ float *)dev.wMatrix.get(),
        (__gm__ float *)dev.tMatrix.get(), (__gm__ int32_t *)dev.index.get(), (__gm__ float *)dev.output.get(),
        (__gm__ float *)dev.workspace.get(), (__gm__ uint8_t *)dev.tiling.get());

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev.output.get(), outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SUCCESS;
}
