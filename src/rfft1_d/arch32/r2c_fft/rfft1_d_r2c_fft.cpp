/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../rfft1_d.h"
#include "fft_common_core.h"
#include "fft_all_mix_tiling_data.h"
#include "rfft1_d_r2c_fft_kernel.h"

/* ======================== R2C-specific: GenInputOutputIndex + GenABTable ======================== */

static void GenInputIndex(int64_t fftN, int parity, std::vector<uint32_t>& inputIndex)
{
    // R2C input index (from InitInputOutputIndex, isR2c=true branch)
    int64_t n = fftN * 2;
    int64_t indexSize = (n >= BIASC_SIZE) ? BIAS_SIZE : ((n / 2 + 63) / 64) * 64;
    inputIndex.assign(BIASC_SIZE * 4, 0);

    if (indexSize == BIAS_SIZE) {
        if (n > BIASC_SIZE) {
            for (uint32_t i = 0; i < (uint32_t)indexSize; i++)
                inputIndex[i] = (uint32_t)((BIAS_SIZE - 1 - i) * sizeof(float));
            uint32_t remainSize = (uint32_t)((n - 2) % BIASC_SIZE); // R2C uses (n-2), C2R uses (n+2)
            for (uint32_t i = 0; i < remainSize / 2; i++)
                inputIndex[indexSize + i] = (uint32_t)((remainSize / 2 - 1 - i) * sizeof(float));
        } else {
            for (int i = 0; i < (n / 2 - 1); i++)
                inputIndex[i] = (uint32_t)((n / 2 - 2 - i) * sizeof(float));
            for (int i = n / 2 - 1; i < indexSize; i++)
                inputIndex[i] = 0;
        }
    } else {
        for (int i = 0; i < (n / 2 - 1); i++)
            inputIndex[i] = (uint32_t)((n / 2 - 2 - i) * sizeof(float));
        for (int i = n / 2 - 1; i < indexSize; i++)
            inputIndex[i] = 0;
    }
}

static void GenOutputIndex(int64_t fftN, int parity, std::vector<uint32_t>& outputIndex)
{
    int64_t n = fftN * 2;
    int64_t indexSize = (n >= BIASC_SIZE) ? BIAS_SIZE : ((n / 2 + 63) / 64) * 64;
    outputIndex.assign(indexSize * 2, 0);
    for (int i = 0; i < (int)indexSize; i++) {
        outputIndex[i * 2] = (uint32_t)(i * 4);
        outputIndex[i * 2 + 1] = (uint32_t)((BIAS_SIZE + i) * 4);
    }
}

static void GenABTable(int64_t fftN, int parity, bool forward, std::vector<float>& aTable, std::vector<float>& bTable)
{
    bool isR2c = true; // R2C
    double factor = 1.0;
    if ((isR2c && !forward) || (!isR2c && forward)) {
        factor = -1.0;
    }
    // For R2C forward: (true && false) || (false && true) = false → factor = 1.0

    if (parity == 0) {
        int32_t tableSize = (int32_t)fftN;
        aTable.assign(tableSize * 2, 0.0f);
        bTable.assign(tableSize * 2, 0.0f);
        for (int32_t i = 0; i < tableSize; ++i) {
            double cosVal = std::cos(3.14159265358979323846 * i / tableSize);
            double sinVal = std::sin(3.14159265358979323846 * i / tableSize);
            if (isR2c) {
                aTable[i] = (float)((1.0 - factor * sinVal) / 2.0);
                aTable[tableSize + i] = (float)(cosVal / (-2.0));
                bTable[i] = (float)((1.0 + factor * sinVal) / 2.0);
                bTable[tableSize + i] = (float)(cosVal / 2.0);
            } else {
                aTable[i] = (float)((1.0 - factor * sinVal));
                aTable[tableSize + i] = (float)(cosVal);
                bTable[i] = (float)((1.0 + factor * sinVal));
                bTable[tableSize + i] = (float)(-1.0 * cosVal);
            }
        }
    } else {
        aTable.clear();
        bTable.clear();
    }
}

/* ======================== Host entry point ======================== */

extern "C" aclError aclfftRfft1DR2CFft(float* x, float* y, uint32_t n, uint32_t batches, int isForward, void* stream)
{
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftRfft1DR2CFft: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = plat->GetCoreNumAic();
    if (coreNum == 0)
        coreNum = 1;
    uint32_t maxCore = std::min(coreNum, 20u);
    uint32_t needCoreNum = batches > maxCore ? maxCore : batches;
    if (needCoreNum == 0)
        needCoreNum = 1;

    bool forward = (isForward != 0);
    int64_t parity = n % 2;
    int64_t fftN = (parity == 0) ? (int64_t)n / 2 : (int64_t)n;
    int64_t batchSize = batches;

    // 1. Radix decomposition
    std::vector<int64_t> radixVec;
    if (fftN >= N_FFT_C2C_MAX)
        InitMixRadixLong(fftN, radixVec);
    else
        InitMixRadixShort(fftN, radixVec);
    // 分解完整性校验（issue #113/#114，同 fft1_d_mix）
    if (!IsMixRadixValid(radixVec, fftN)) {
        std::cerr << "[ops-fft] aclfftRfft1DR2CFft: unsupported fftN " << fftN << " (factorization incomplete)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    int64_t radixListLen = (int64_t)radixVec.size();

    // 2. DFT matrix (forward for R2C)
    int64_t dftLen = GetTwiddleMatrixLen(fftN, radixVec);
    std::vector<float> dftHost(dftLen, 0.0f);
    GenWMatrixForwardForMultiLen(radixListLen, radixVec.data(), fftN, dftHost.data());

    // 3. Twiddle matrix
    int64_t twLen = GetTwMatrixLen(fftN, radixVec);
    std::vector<float> twHost(twLen, 0.0f);
    if (twLen > 0)
        GenTwMatrix(fftN, radixVec, twHost.data());

    // 4. Radix list
    std::vector<int32_t> radixListHost(radixVec.begin(), radixVec.end());

    // 5. C2R-specific: input/output index + A/B tables
    std::vector<uint32_t> inputIndex, outputIndex;
    GenInputIndex(fftN, parity, inputIndex);
    GenOutputIndex(fftN, parity, outputIndex);
    std::vector<float> aTable, bTable;
    GenABTable(fftN, parity, forward, aTable, bTable);

    // 6. Workspace
    int64_t wsIn, wsOut, wsSync, wsC2c, wsAux;
    InitMixRadixParam(parity, fftN, batchSize, radixVec, wsIn, wsOut, wsSync, wsC2c, wsAux);

    // 值初始化：同 fft1_d_mix（issue #117），未填充字段随 tiling 上设备前清零
    OpsFft::FftAllMixTilingData tilingData{};
    tilingData.batchSize = batchSize;
    tilingData.fftN = fftN;
    tilingData.radixListLen = radixListLen;
    tilingData.isInverse = 0; // R2C is always forward
    tilingData.workspaceOffsets[0] = 0;
    tilingData.workspaceOffsets[1] = wsIn;
    tilingData.workspaceOffsets[2] = wsIn + wsOut;
    tilingData.workspaceOffsets[3] = wsIn + wsOut + wsSync;
    tilingData.workspaceOffsets[4] = wsIn + wsOut + wsSync + wsC2c;
    tilingData.isOddN = (int32_t)parity;
    int64_t totalWs = wsIn + wsOut + wsSync + wsC2c + wsAux;

    // 7. Allocate & copy
    // R2C: input is n real floats, output is (n/2+1) complex = (n/2+1)*2 floats
    size_t inputSize = static_cast<size_t>(n) * batches * sizeof(float);
    size_t outputSize = static_cast<size_t>(n / 2 + 1) * batches * sizeof(float) * 2;
    void *dIn = nullptr, *dOut = nullptr, *dDft = nullptr, *dTw = nullptr, *dRadix = nullptr, *dWs = nullptr,
         *dTil = nullptr;
    void *dInIdx = nullptr, *dA = nullptr, *dB = nullptr, *dOutIdx = nullptr;
    // 守卫与 malloc 交错（issue #120）：任一分配失败时前序已分配内存自动释放
    CHECK_ACL(aclrtMalloc(&dIn, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g1(dIn);
    CHECK_ACL(aclrtMalloc(&dOut, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g2(dOut);
    CHECK_ACL(aclrtMalloc(&dDft, dftLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g3(dDft);
    if (twLen > 0)
        CHECK_ACL(aclrtMalloc(&dTw, twLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g4(dTw);
    CHECK_ACL(aclrtMalloc(&dRadix, radixListHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g5(dRadix);
    CHECK_ACL(aclrtMalloc(&dWs, totalWs, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g6(dWs);
    CHECK_ACL(aclrtMalloc(&dTil, sizeof(OpsFft::FftAllMixTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g7(dTil);
    // 兜底分配按最大可能拷贝长度（sizeof(float)）而非 1 字节：奇数 n 时表为空但仍执行
    // 4 字节零值拷贝，1 字节分配会越界写（issue #88）
    CHECK_ACL(
        aclrtMalloc(&dInIdx, std::max(sizeof(float), inputIndex.size() * sizeof(uint32_t)), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g8(dInIdx);
    CHECK_ACL(aclrtMalloc(&dA, std::max(sizeof(float), aTable.size() * sizeof(float)), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g9(dA);
    CHECK_ACL(aclrtMalloc(&dB, std::max(sizeof(float), bTable.size() * sizeof(float)), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g10(dB);
    CHECK_ACL(aclrtMalloc(
        &dOutIdx, std::max(sizeof(float), outputIndex.size() * sizeof(uint32_t)), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g11(dOutIdx);
    CHECK_ACL(aclrtMemcpy(dIn, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(
        aclrtMemcpy(dDft, dftLen * sizeof(float), dftHost.data(), dftLen * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    if (twLen > 0)
        CHECK_ACL(
            aclrtMemcpy(dTw, twLen * sizeof(float), twHost.data(), twLen * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        dRadix, radixListHost.size() * sizeof(int32_t), radixListHost.data(), radixListHost.size() * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        dTil, sizeof(OpsFft::FftAllMixTilingData), &tilingData, sizeof(OpsFft::FftAllMixTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE));
    if (!inputIndex.empty())
        CHECK_ACL(aclrtMemcpy(
            dInIdx, inputIndex.size() * sizeof(uint32_t), inputIndex.data(), inputIndex.size() * sizeof(uint32_t),
            ACL_MEMCPY_HOST_TO_DEVICE));
    // 无索引时也应保证 dInIdx 有确定内容，此前误写为 dA（issue #85）
    else {
        float z = 0;
        CHECK_ACL(aclrtMemcpy(dInIdx, sizeof(float), &z, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    if (!aTable.empty())
        CHECK_ACL(aclrtMemcpy(
            dA, aTable.size() * sizeof(float), aTable.data(), aTable.size() * sizeof(float),
            ACL_MEMCPY_HOST_TO_DEVICE));
    else {
        float z = 0;
        CHECK_ACL(aclrtMemcpy(dA, sizeof(float), &z, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    if (!bTable.empty())
        CHECK_ACL(aclrtMemcpy(
            dB, bTable.size() * sizeof(float), bTable.data(), bTable.size() * sizeof(float),
            ACL_MEMCPY_HOST_TO_DEVICE));
    else {
        float z = 0;
        CHECK_ACL(aclrtMemcpy(dB, sizeof(float), &z, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    if (!outputIndex.empty())
        CHECK_ACL(aclrtMemcpy(
            dOutIdx, outputIndex.size() * sizeof(uint32_t), outputIndex.data(), outputIndex.size() * sizeof(uint32_t),
            ACL_MEMCPY_HOST_TO_DEVICE));

    // 8. Select kernel variant + launch
    int32_t aivSplitWay = InitAiVSplitWay(fftN, radixVec);
    uint8_t* sync = nullptr;
    CHECK_ACL(aclrtGetHardwareSyncAddr((void**)&sync));

#define LAUNCH_R2C(IDX)                                                                                            \
    FFT1DR2CFftKernel::fft_r2c_##IDX<<<needCoreNum, nullptr, stream>>>(                                            \
        (__gm__ uint8_t*)sync, (__gm__ float*)dIn, (__gm__ uint32_t*)dInIdx, (__gm__ float*)dA, (__gm__ float*)dB, \
        (__gm__ uint32_t*)dOutIdx, (__gm__ float*)dDft, (__gm__ float*)dTw, (__gm__ int32_t*)dRadix,               \
        (__gm__ float*)dOut, (__gm__ float*)dWs, (__gm__ uint8_t*)dTil)
#define LAUNCH_R2C_EVEN(IDX)                                                                                       \
    FFT1DR2CFftKernel::fft_r2c_##IDX##_even<<<needCoreNum, nullptr, stream>>>(                                     \
        (__gm__ uint8_t*)sync, (__gm__ float*)dIn, (__gm__ uint32_t*)dInIdx, (__gm__ float*)dA, (__gm__ float*)dB, \
        (__gm__ uint32_t*)dOutIdx, (__gm__ float*)dDft, (__gm__ float*)dTw, (__gm__ int32_t*)dRadix,               \
        (__gm__ float*)dOut, (__gm__ float*)dWs, (__gm__ uint8_t*)dTil)
#define LAUNCH_R2C_ODD(IDX)                                                                                        \
    FFT1DR2CFftKernel::fft_r2c_##IDX##_odd<<<needCoreNum, nullptr, stream>>>(                                      \
        (__gm__ uint8_t*)sync, (__gm__ float*)dIn, (__gm__ uint32_t*)dInIdx, (__gm__ float*)dA, (__gm__ float*)dB, \
        (__gm__ uint32_t*)dOutIdx, (__gm__ float*)dDft, (__gm__ float*)dTw, (__gm__ int32_t*)dRadix,               \
        (__gm__ float*)dOut, (__gm__ float*)dWs, (__gm__ uint8_t*)dTil)

    // R2C always uses batch variants (matching original behavior)
    if (parity == 1) {
        switch (aivSplitWay) {
            case 1:
                LAUNCH_R2C_ODD(1);
                break;
            case 2:
                LAUNCH_R2C_ODD(2);
                break;
            case 3:
                LAUNCH_R2C_ODD(3);
                break;
            default:
                LAUNCH_R2C_ODD(1);
                break;
        }
    } else {
        switch (aivSplitWay) {
            case 1:
                LAUNCH_R2C_EVEN(1);
                break;
            case 2:
                LAUNCH_R2C_EVEN(2);
                break;
            case 3:
                LAUNCH_R2C_EVEN(3);
                break;
            default:
                LAUNCH_R2C_EVEN(1);
                break;
        }
    }

#undef LAUNCH_R2C
#undef LAUNCH_R2C_EVEN
#undef LAUNCH_R2C_ODD

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dOut, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SUCCESS;
}
