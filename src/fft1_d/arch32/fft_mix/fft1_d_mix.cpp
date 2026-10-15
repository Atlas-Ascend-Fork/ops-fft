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
#include "fft_all_mix_tiling_data.h"
#include "fft1_d_mix_kernel.h"

/* ======================== Host entry point ======================== */

extern "C" aclError aclfftFft1DMix(float *x, float *y, uint32_t n,
                                    uint32_t batches, int isForward, void *stream) {
    // 防护: 本函数经 ACLFFT_API 导出可被外部直接调用，需校验输入输出指针（issue #76）
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftFft1DMix: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // n==0 拒绝（issue #164，对齐同族入口口径）：n=0 时 InitMixRadixShort(0)
    // 短路 push {0}，IsMixRadixValid({0},0) 按 0==0 判定通过，分解完整性校验
    // 形同虚设；随后 0 字节 aclrtMalloc（ACL 语义要求 size>0，行为未定义）与
    // 空内核启动。同族 aclfftFft1DDft/Fft1DN/Fft1DC2C/Fft1DC2CMix 均已拒绝 n==0
    if (n == 0) {
        std::cerr << "[ops-fft] aclfftFft1DMix: n=0 is not supported" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = plat->GetCoreNumAic(); if (coreNum == 0) coreNum = 1;
    uint32_t maxCore = std::min(coreNum, 20u);
    uint32_t needCoreNum = batches > maxCore ? maxCore : batches;
    if (needCoreNum == 0) needCoreNum = 1;

    bool forward = (isForward != 0);
    int64_t fftN = n, batchSize = batches;

    // 1. Radix decomposition
    std::vector<int64_t> radixVec;
    if (fftN >= N_FFT_C2C_MAX) InitMixRadixLong(fftN, radixVec);
    else InitMixRadixShort(fftN, radixVec);
    // 分解完整性校验（issue #113/#114）：空分解（全质因子 > 128，如 32771）会令
    // InitAiVSplitWay 取 radixVec[-1] 越界读；部分分解（如 65498=2x32749）会按
    // 错误级数启动内核输出垃圾结果。两类均属不支持域，返回参数错误
    if (!IsMixRadixValid(radixVec, fftN)) {
        std::cerr << "[ops-fft] aclfftFft1DMix: unsupported fftN " << fftN << " (factorization incomplete)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    int64_t radixListLen = (int64_t)radixVec.size();

    // 2. DFT matrix
    // GenWMatrix*ForMultiLen 内部按 radixListLen 循环，单 radix（==1）同样适用，无需分支（issue #84）
    int64_t dftLen = GetTwiddleMatrixLen(fftN, radixVec);
    std::vector<float> dftHost(dftLen, 0.0f);
    if (forward) GenWMatrixForwardForMultiLen(radixListLen, radixVec.data(), fftN, dftHost.data());
    else GenWMatrixInverseForMultiLen(radixListLen, radixVec.data(), fftN, dftHost.data());

    // 3. Twiddle matrix
    int64_t twLen = GetTwMatrixLen(fftN, radixVec);
    std::vector<float> twHost(twLen, 0.0f);
    if (twLen > 0) GenTwMatrix(fftN, radixVec, twHost.data());

    // 4. Radix list (int32)
    std::vector<int32_t> radixListHost(radixVec.begin(), radixVec.end());

    // 5. Workspace
    int64_t parity = 0;
    int64_t wsIn, wsOut, wsSync, wsC2c, wsAux;
    InitMixRadixParam(parity, fftN, batchSize, radixVec, wsIn, wsOut, wsSync, wsC2c, wsAux);

    // 值初始化：结构体含大量 arch35 路径才填充的 "Precomputed plan data" 字段，
    // 整体 memcpy 上设备前清零，避免宿主栈上未初始化内存进入设备（issue #117）
    OpsFft::FftAllMixTilingData tilingData{};
    tilingData.batchSize = batchSize;
    tilingData.fftN = fftN;
    tilingData.radixListLen = radixListLen;
    tilingData.isInverse = forward ? 0 : 1;
    tilingData.workspaceOffsets[0] = 0;
    tilingData.workspaceOffsets[1] = wsIn;
    tilingData.workspaceOffsets[2] = wsIn + wsOut;
    tilingData.workspaceOffsets[3] = wsIn + wsOut + wsSync;
    tilingData.workspaceOffsets[4] = wsIn + wsOut + wsSync + wsC2c;
    int64_t totalWs = wsIn + wsOut + wsSync + wsC2c + wsAux;

    // 6. Allocate & copy
    size_t inputSize = static_cast<size_t>(n) * batches * sizeof(float) * 2;
    void *dIn=nullptr,*dOut=nullptr,*dDft=nullptr,*dTw=nullptr,*dRadix=nullptr,*dWs=nullptr,*dTil=nullptr;
    CHECK_ACL(aclrtMalloc(&dIn, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g1(dIn);
    CHECK_ACL(aclrtMalloc(&dOut, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g2(dOut);
    CHECK_ACL(aclrtMalloc(&dDft, dftLen*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g3(dDft);
    if (twLen > 0) CHECK_ACL(aclrtMalloc(&dTw, twLen*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g4(dTw);
    CHECK_ACL(aclrtMalloc(&dRadix, radixListHost.size()*sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g5(dRadix);
    CHECK_ACL(aclrtMalloc(&dWs, totalWs, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g6(dWs);
    CHECK_ACL(aclrtMalloc(&dTil, sizeof(OpsFft::FftAllMixTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> g7(dTil);
    CHECK_ACL(aclrtMemcpy(dIn, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dDft, dftLen*sizeof(float), dftHost.data(), dftLen*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    if (twLen > 0) CHECK_ACL(aclrtMemcpy(dTw, twLen*sizeof(float), twHost.data(), twLen*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dRadix, radixListHost.size()*sizeof(int32_t), radixListHost.data(), radixListHost.size()*sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dTil, sizeof(OpsFft::FftAllMixTilingData), &tilingData, sizeof(OpsFft::FftAllMixTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    // 7. Select kernel variant based on aivSplitWay (1-6)
    int32_t aivSplitWay = InitAiVSplitWay(fftN, radixVec);
    uint8_t *sync = nullptr;
    CHECK_ACL(aclrtGetHardwareSyncAddr((void**)&sync));
#define LAUNCH_MIX(IDX) FFT1DMixKernel::fft_mix_##IDX<<<needCoreNum, nullptr, stream>>>( \
        (__gm__ uint8_t*)sync, (__gm__ float*)dIn, (__gm__ float*)dDft, (__gm__ float*)dTw, \
        (__gm__ int32_t*)dRadix, (__gm__ float*)dOut, (__gm__ float*)dWs, (__gm__ uint8_t*)dTil)
    switch (aivSplitWay) {
        case 1: LAUNCH_MIX(1); break;
        case 2: LAUNCH_MIX(2); break;
        case 3: LAUNCH_MIX(3); break;
        case 4: LAUNCH_MIX(4); break;
        case 5: LAUNCH_MIX(5); break;
        case 6: LAUNCH_MIX(6); break;
        default: LAUNCH_MIX(1); break;
    }
#undef LAUNCH_MIX

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, inputSize, dOut, inputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SUCCESS;
}
