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
#include "fft_mix_host_common.h"
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
        int32_t tableStride = tableSize + 1; // Include the Nyquist endpoint.
        aTable.assign(tableStride * 2, 0.0f);
        bTable.assign(tableStride * 2, 0.0f);
        for (int32_t i = 0; i < tableSize; ++i) {
            double cosVal = std::cos(3.14159265358979323846 * i / tableSize);
            double sinVal = std::sin(3.14159265358979323846 * i / tableSize);
            aTable[i] = (float)((1.0 - factor * sinVal));
            aTable[tableStride + i] = (float)(cosVal);
            bTable[i] = (float)((1.0 + factor * sinVal));
            bTable[tableStride + i] = (float)(-1.0 * cosVal);
        }
        aTable[tableSize] = 1.0f;
        aTable[tableStride + tableSize] = -1.0f;
        bTable[tableSize] = 1.0f;
        bTable[tableStride + tableSize] = 1.0f;
    } else {
        aTable.clear();
        bTable.clear();
    }
}

/* ======================== Host entry point ======================== */

// 入参校验链（!102 codecheck 拆分）：经 CHECK_ACL 调用保持失败早退语义
static aclError ValidateIrfft1DC2RFftParams(const float* x, const float* y, uint32_t n, int32_t norm, uint32_t batches)
{
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftIrfft1DC2RFft: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // 当前版本仅支持 norm=0（BACKWARD，无缩放）；norm!=0 显式报错而非静默按
    // BACKWARD 计算（issue #224，对齐 C2R DFT 路径口径，消除 norm 行为随档位分裂）
    if (norm != 0) {
        std::cerr << "[ops-fft] aclfftIrfft1DC2RFft: norm=" << norm << " is not supported (only 0=BACKWARD in this version)"
                  << std::endl;
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
    return ACL_SUCCESS;
}

// 内核档位选择与启动（!102 codecheck 拆分）：aivSplitWay 1..3 有专属内核、4..6 回退
// 档位 1（原 switch default 语义）；parity 奇偶分别派发 odd/even batch 变体
static aclError LaunchC2RFftKernel(int64_t fftN, std::vector<int64_t> &radixVec, int64_t parity,
                                   uint32_t needCoreNum, void *stream, const FftMixHostCommon::MixFftDevBuffers &dev)
{
    int32_t aivSplitWay = InitAiVSplitWay(fftN, radixVec);
    uint8_t* sync = nullptr;
    CHECK_ACL(aclrtGetHardwareSyncAddr((void**)&sync));
    const int idx = (aivSplitWay >= 1 && aivSplitWay <= 3) ? aivSplitWay : 1;
    const int variant = (parity == 1) ? idx : idx + 16;
#define LAUNCH_C2R_VARIANT(NAME, IDX)                                                                              \
    FFT1DC2RFftKernel::fft_c2r_##IDX##NAME<<<needCoreNum, nullptr, stream>>>(                                      \
        (__gm__ uint8_t*)sync, (__gm__ float*)dev.in.get(), (__gm__ uint32_t*)dev.inIdx.get(),                     \
        (__gm__ float*)dev.a.get(), (__gm__ float*)dev.b.get(), (__gm__ uint32_t*)dev.outIdx.get(),                \
        (__gm__ float*)dev.dft.get(), (__gm__ float*)dev.tw.get(), (__gm__ int32_t*)dev.radix.get(),               \
        (__gm__ float*)dev.out.get(), (__gm__ float*)dev.ws.get(), (__gm__ uint8_t*)dev.til.get())
    switch (variant) {
        case 1: LAUNCH_C2R_VARIANT(_odd, 1); break;
        case 2: LAUNCH_C2R_VARIANT(_odd, 2); break;
        case 3: LAUNCH_C2R_VARIANT(_odd, 3); break;
        case 17: LAUNCH_C2R_VARIANT(_even, 1); break;
        case 18: LAUNCH_C2R_VARIANT(_even, 2); break;
        case 19: LAUNCH_C2R_VARIANT(_even, 3); break;
        default: break;
    }
#undef LAUNCH_C2R_VARIANT
    return ACL_SUCCESS;
}

extern "C" aclError aclfftIrfft1DC2RFft(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream)
{
    CHECK_ACL(ValidateIrfft1DC2RFftParams(x, y, n, norm, batches));
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = plat->GetCoreNumAic();
    if (coreNum == 0)
        coreNum = 1;
    uint32_t maxCore = std::min(coreNum, 20u);
    uint32_t needCoreNum = batches > maxCore ? maxCore : batches;
    if (needCoreNum == 0)
        needCoreNum = 1;

    int64_t parity = n % 2;
    // For c2r: fftN in tiling = n/2 (even) or n (odd)
    int64_t fftN = (parity == 0) ? (int64_t)n / 2 : (int64_t)n;
    int64_t batchSize = batches;

    std::vector<int64_t> radixVec;
    std::vector<float> dftHost, twHost;
    std::vector<int32_t> radixListHost;
    std::vector<uint32_t> inputIndex, outputIndex;
    std::vector<float> aTable, bTable;
    int64_t wsIn = 0, wsOut = 0, wsSync = 0, wsC2c = 0, wsAux = 0;
    int64_t radixListLen = 0;
    CHECK_ACL(FftMixHostCommon::BuildMixFftHostData(fftN, batchSize, parity, isForward != 0, "aclfftIrfft1DC2RFft",
                                                    radixVec, dftHost, twHost, radixListHost,
                                                    wsIn, wsOut, wsSync, wsC2c, wsAux, radixListLen));
    GenInputIndex(fftN, parity, inputIndex);
    GenOutputIndex(fftN, parity, outputIndex);
    GenABTable(fftN, parity, isForward != 0, aTable, bTable);

    OpsFft::FftAllMixTilingData tilingData = {};
    FftMixHostCommon::FillMixFftTiling(tilingData, fftN, batchSize, radixListLen, parity, 1, wsIn, wsOut, wsSync,
                                       wsC2c);
    int64_t totalWs = wsIn + wsOut + wsSync + wsC2c + wsAux;

    // 7. Allocate & copy
    size_t inputSize = static_cast<size_t>(n / 2 + 1) * batches * sizeof(float) * 2;
    size_t outputSize = static_cast<size_t>(n) * batches * sizeof(float);
    FftMixHostCommon::MixFftDevBuffers dev;
    // 索引/表缓冲分配尺寸按 C2R 原口径：inIdx/outIdx 按表实际大小，A/B 空表带
    // sizeof(float) 兜底（issue #88）
    CHECK_ACL(FftMixHostCommon::AllocMixFftDevBuffers(
        dev, inputSize, outputSize, dftHost.size(), twHost.size(), radixListHost.size(),
        inputIndex.size() * sizeof(uint32_t),
        std::max(sizeof(float), aTable.size() * sizeof(float)),
        std::max(sizeof(float), bTable.size() * sizeof(float)),
        outputIndex.size() * sizeof(uint32_t), totalWs));
    CHECK_ACL(FftMixHostCommon::CopyMixFftHostToDevice(dev, x, inputSize, dftHost, twHost, radixListHost,
                                                       inputIndex, aTable, bTable, outputIndex, tilingData,
                                                       false));

    CHECK_ACL(LaunchC2RFftKernel(fftN, radixVec, parity, needCoreNum, stream, dev));

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev.out.get(), outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SUCCESS;
}
