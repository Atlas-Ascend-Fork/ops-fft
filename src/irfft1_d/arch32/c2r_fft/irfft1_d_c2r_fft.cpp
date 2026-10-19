/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../irfft1_d.h"
#include "fft_common_core.h"
#include "fft_all_mix_tiling_data.h"
#include "irfft1_d_c2r_fft_kernel.h"

/* ======================== C2R-specific: InitInputOutputIndex + InitABTable ======================== */

static void GenInputIndex(int64_t fftN, int parity, std::vector<uint32_t>& inputIndex)
{
    int64_t n = fftN * 2;
    inputIndex.assign(BIASC_SIZE * 4, 0);
    if (parity == 1) {
        if ((fftN / 2 * 2 > BIASC_SIZE)) {
            uint32_t idx_size = BIASC_SIZE;
            for (uint32_t i = 0; i < idx_size / 2; i++) {
                inputIndex[i * 2] = (uint32_t)((idx_size - i * 2) * sizeof(float));
                inputIndex[i * 2 + 1] = (uint32_t)((idx_size - i * 2 + 1) * sizeof(float));
            }
            for (uint32_t i = 0; i < idx_size / 2; i++) {
                inputIndex[idx_size + i * 2] = (uint32_t)((idx_size - i * 2 - 2) * sizeof(float));
                inputIndex[idx_size + i * 2 + 1] = (uint32_t)((idx_size - i * 2 - 1) * sizeof(float));
            }
            uint32_t idx_remain = (uint32_t)((fftN / 2 * 2) % BIASC_SIZE);
            for (uint32_t i = 0; i < idx_remain / 2; i++) {
                inputIndex[idx_size * 2 + i * 2] = (uint32_t)((idx_remain - i * 2 - 2) * sizeof(float));
                inputIndex[idx_size * 2 + i * 2 + 1] = (uint32_t)((idx_remain - i * 2 - 1) * sizeof(float));
            }
        } else {
            uint32_t idx_size = (uint32_t)(fftN / 2) * 2;
            uint32_t idx_padding = (idx_size + 63) / 64 * 64;
            for (uint32_t i = 0; i < idx_size / 2; i++) {
                inputIndex[i * 2] = (uint32_t)((idx_size + 2 - i * 2 - 2) * sizeof(float));
                inputIndex[i * 2 + 1] = (uint32_t)((idx_size + 2 - i * 2 - 1) * sizeof(float));
            }
        }
    } else {
        int64_t indexSize = (n >= BIASC_SIZE) ? BIAS_SIZE : ((n / 2 + 63) / 64) * 64;
        if (indexSize == BIAS_SIZE) {
            if (n > BIASC_SIZE) {
                for (uint32_t i = 0; i < (uint32_t)indexSize; i++)
                    inputIndex[i] = (uint32_t)((BIAS_SIZE - 1 - i) * sizeof(float));
                uint32_t remainSize = (uint32_t)((n + 2) % BIASC_SIZE);
                for (uint32_t i = 0; i < remainSize / 2; i++)
                    inputIndex[indexSize + i] = (uint32_t)((remainSize / 2 - 1 - i) * sizeof(float));
            } else {
                if (n == BIASC_SIZE) {
                    for (int i = 0; i < (n / 2); i++)
                        inputIndex[i] = (uint32_t)((n / 2 - i - 1) * sizeof(float));
                } else {
                    for (int i = 0; i < (n / 2); i++)
                        inputIndex[i] = (uint32_t)((n / 2 - i) * sizeof(float));
                }
                int32_t gather_num = (int32_t)(n + 2);
                int32_t index_size_padding = ((gather_num / 2 + 63) / 64) * 64;
                for (int i = (n / 2); i < index_size_padding; i++)
                    inputIndex[i] = 0;
            }
        } else {
            for (int i = 0; i < (n / 2); i++)
                inputIndex[i] = (uint32_t)((n / 2 - i) * sizeof(float));
            int32_t gather_num = (int32_t)(n + 2);
            int32_t index_size_padding = ((gather_num / 2 + 63) / 64) * 64;
            for (int i = (n / 2); i < index_size_padding; i++)
                inputIndex[i] = 0;
        }
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
    bool isC2R = true; // c2r
    double factor = 1.0;
    if ((isC2R && forward))
        factor = -1.0;

    if (parity == 0) {
        int32_t tableSize = (int32_t)fftN;
        aTable.assign(tableSize * 2, 0.0f);
        bTable.assign(tableSize * 2, 0.0f);
        for (int32_t i = 0; i < tableSize; ++i) {
            double cosVal = std::cos(3.14159265358979323846 * i / tableSize);
            double sinVal = std::sin(3.14159265358979323846 * i / tableSize);
            aTable[i] = (float)((1.0 - factor * sinVal));
            aTable[tableSize + i] = (float)(cosVal);
            bTable[i] = (float)((1.0 + factor * sinVal));
            bTable[tableSize + i] = (float)(-1.0 * cosVal);
        }
    } else {
        aTable.clear();
        bTable.clear();
    }
}

/* ======================== Host entry point ======================== */

extern "C" aclError aclfftIrfft1DC2RFft(float* x, float* y, uint32_t n, uint32_t batches, int isForward, void* stream)
{
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftIrfft1DC2RFft: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // n==0 拒绝（issue #166，对齐同族入口口径）：n=0 时 fftN=0，InitMixRadixShort(0)
    // 短路 push {0}，IsMixRadixValid({0},0) 判 0==0 通过；随后 outputSize=0 触发
    // 0 字节 aclrtMalloc 与空内核启动，行为未定义。同族 aclfftIrfft1DDft（n 域
    // 校验）与 arch35 irfft 入口均已覆盖
    if (n == 0) {
        std::cerr << "[ops-fft] aclfftIrfft1DC2RFft: n=0 is not supported" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // n 上界校验（issue #187，对齐 aclfftFft1DN #161 与 FFT_1D.md 文档口径）：
    // 超界 n 经 O(n) 宿主矩阵分配可能 bad_alloc 穿透 extern "C" 边界致 std::terminate
    if (n > (1u << 27)) {
        std::cerr << "[ops-fft] aclfftIrfft1DC2RFft: n=" << n << " out of supported range (0, 2^27]" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // batches==0 拒绝（issue #188，与 n==0 同口径）：batches=0 时 outputSize=0
    // 触发 0 字节 aclrtMalloc 与空内核启动，返回"成功"但输出缓冲未写入
    if (batches == 0) {
        std::cerr << "[ops-fft] aclfftIrfft1DC2RFft: batches=0 is not supported" << std::endl;
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
    // For c2r: fftN in tiling = n/2 (even) or n (odd)
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
        std::cerr << "[ops-fft] aclfftIrfft1DC2RFft: unsupported fftN " << fftN << " (factorization incomplete)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    int64_t radixListLen = (int64_t)radixVec.size();

    // 2. DFT matrix (inverse for c2r)
    int64_t dftLen = GetTwiddleMatrixLen(fftN, radixVec);
    std::vector<float> dftHost(dftLen, 0.0f);
    // GenWMatrixInverseForMultiLen 内部按 radixListLen 循环，单 radix（==1）同样适用，无需分支（issue #84）
    GenWMatrixInverseForMultiLen(radixListLen, radixVec.data(), fftN, dftHost.data());

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

    OpsFft::FftAllMixTilingData tilingData = {};
    tilingData.batchSize = batchSize;
    tilingData.fftN = fftN;
    tilingData.radixListLen = radixListLen;
    tilingData.isInverse = 1; // c2r is always inverse
    tilingData.workspaceOffsets[0] = 0;
    tilingData.workspaceOffsets[1] = wsIn;
    tilingData.workspaceOffsets[2] = wsIn + wsOut;
    tilingData.workspaceOffsets[3] = wsIn + wsOut + wsSync;
    tilingData.workspaceOffsets[4] = wsIn + wsOut + wsSync + wsC2c;
    tilingData.isOddN = (int32_t)parity;
    int64_t totalWs = wsIn + wsOut + wsSync + wsC2c + wsAux;

    // 7. Allocate & copy
    size_t inputSize = static_cast<size_t>(n / 2 + 1) * batches * sizeof(float) * 2;
    size_t outputSize = static_cast<size_t>(n) * batches * sizeof(float);
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
    CHECK_ACL(aclrtMalloc(&dInIdx, inputIndex.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g8(dInIdx);
    // 兜底分配按最大可能拷贝长度（sizeof(float)）而非 1 字节：奇数 n 时表为空但仍执行
    // 4 字节零值拷贝，1 字节分配会越界写（issue #88）
    CHECK_ACL(aclrtMalloc(&dA, std::max(sizeof(float), aTable.size() * sizeof(float)), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g9(dA);
    CHECK_ACL(aclrtMalloc(&dB, std::max(sizeof(float), bTable.size() * sizeof(float)), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g10(dB);
    CHECK_ACL(aclrtMalloc(&dOutIdx, outputIndex.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
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

#define LAUNCH_C2R(IDX)                                                                                            \
    FFT1DC2RFftKernel::fft_c2r_##IDX<<<needCoreNum, nullptr, stream>>>(                                            \
        (__gm__ uint8_t*)sync, (__gm__ float*)dIn, (__gm__ uint32_t*)dInIdx, (__gm__ float*)dA, (__gm__ float*)dB, \
        (__gm__ uint32_t*)dOutIdx, (__gm__ float*)dDft, (__gm__ float*)dTw, (__gm__ int32_t*)dRadix,               \
        (__gm__ float*)dOut, (__gm__ float*)dWs, (__gm__ uint8_t*)dTil)
#define LAUNCH_C2R_EVEN(IDX)                                                                                       \
    FFT1DC2RFftKernel::fft_c2r_##IDX##_even<<<needCoreNum, nullptr, stream>>>(                                     \
        (__gm__ uint8_t*)sync, (__gm__ float*)dIn, (__gm__ uint32_t*)dInIdx, (__gm__ float*)dA, (__gm__ float*)dB, \
        (__gm__ uint32_t*)dOutIdx, (__gm__ float*)dDft, (__gm__ float*)dTw, (__gm__ int32_t*)dRadix,               \
        (__gm__ float*)dOut, (__gm__ float*)dWs, (__gm__ uint8_t*)dTil)
#define LAUNCH_C2R_ODD(IDX)                                                                                        \
    FFT1DC2RFftKernel::fft_c2r_##IDX##_odd<<<needCoreNum, nullptr, stream>>>(                                      \
        (__gm__ uint8_t*)sync, (__gm__ float*)dIn, (__gm__ uint32_t*)dInIdx, (__gm__ float*)dA, (__gm__ float*)dB, \
        (__gm__ uint32_t*)dOutIdx, (__gm__ float*)dDft, (__gm__ float*)dTw, (__gm__ int32_t*)dRadix,               \
        (__gm__ float*)dOut, (__gm__ float*)dWs, (__gm__ uint8_t*)dTil)

    // C2R always uses batch variants (matching original behavior)
    if (parity == 1) {
        // odd: use odd_batch variant
        switch (aivSplitWay) {
            case 1:
                LAUNCH_C2R_ODD(1);
                break;
            case 2:
                LAUNCH_C2R_ODD(2);
                break;
            case 3:
                LAUNCH_C2R_ODD(3);
                break;
            default:
                LAUNCH_C2R_ODD(1);
                break;
        }
    } else {
        // even: always use even_batch variant
        switch (aivSplitWay) {
            case 1:
                LAUNCH_C2R_EVEN(1);
                break;
            case 2:
                LAUNCH_C2R_EVEN(2);
                break;
            case 3:
                LAUNCH_C2R_EVEN(3);
                break;
            default:
                LAUNCH_C2R_EVEN(1);
                break;
        }
    }

#undef LAUNCH_C2R
#undef LAUNCH_C2R_EVEN
#undef LAUNCH_C2R_ODD

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dOut, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SUCCESS;
}
