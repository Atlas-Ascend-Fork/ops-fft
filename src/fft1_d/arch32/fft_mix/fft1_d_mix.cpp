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

// 入参校验链（!102 codecheck 拆分）：经 CHECK_ACL 调用保持失败早退语义
static aclError ValidateFft1DMixParams(const float *x, const float *y, uint32_t n, int32_t norm,
                                       uint32_t batches, int isForward)
{
    // 防护: 本函数经 ACLFFT_API 导出可被外部直接调用，需校验输入输出指针（issue #76）
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftFft1DMix: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // 当前版本仅支持 norm=0（BACKWARD，无缩放）；norm!=0 显式报错而非静默按
    // BACKWARD 计算（issue #224，对齐 dft/fft_n 路径口径，消除 norm 行为随档位分裂）
    if (norm != 0) {
        std::cerr << "[ops-fft] aclfftFft1DMix: norm=" << norm << " is not supported (only 0=BACKWARD in this version)"
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // isForward 域校验（issue #229，对齐 dft/arch35 口径）：合法取值仅 0(正向)/1(反向)。
    // 本入口无校验时任意非 0 值经 (isForward != 0) 判定静默按正向执行，方向语义与
    // 文档 0/1 两值契约不符
    if (isForward != 0 && isForward != 1) {
        std::cerr << "[ops-fft] aclfftFft1DMix: isForward=" << isForward
                  << " invalid (expected 0 or 1)" << std::endl;
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
    // n 上界校验（issue #187，对齐 aclfftFft1DN #161 与 FFT_1D.md 文档口径）：
    // 全为小因子的超界 n（如 3x2^28）可通过分解完整性校验，随后 O(n) 宿主矩阵
    // 分配失败 bad_alloc 穿透 extern "C" 边界致 std::terminate
    if (n > (1u << 27)) {
        std::cerr << "[ops-fft] aclfftFft1DMix: n=" << n << " out of supported range (0, 2^27]" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // batches==0 拒绝（issue #188，与 n==0 同口径）：batches=0 时 inputSize=0
    // 触发 0 字节 aclrtMalloc 与空内核启动，返回"成功"但输出缓冲未写入
    if (batches == 0) {
        std::cerr << "[ops-fft] aclfftFft1DMix: batches=0 is not supported" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

// 宿主侧分解/矩阵/工作区参数构建（!102 codecheck 拆分）
static aclError BuildFft1DMixHostData(int64_t fftN, int64_t batchSize, bool forward,
                                      std::vector<int64_t> &radixVec, std::vector<float> &dftHost,
                                      std::vector<float> &twHost, std::vector<int32_t> &radixListHost,
                                      int64_t &wsIn, int64_t &wsOut, int64_t &wsSync, int64_t &wsC2c,
                                      int64_t &wsAux, int64_t &radixListLen)
{
    // 1. Radix decomposition
    if (fftN >= N_FFT_C2C_MAX) InitMixRadixLong(fftN, radixVec);
    else InitMixRadixShort(fftN, radixVec);
    // 分解完整性校验（issue #113/#114）：空分解（全质因子 > 128，如 32771）会令
    // InitAiVSplitWay 取 radixVec[-1] 越界读；部分分解（如 65498=2x32749）会按
    // 错误级数启动内核输出垃圾结果。两类均属不支持域，返回参数错误
    if (!IsMixRadixValid(radixVec, fftN)) {
        std::cerr << "[ops-fft] aclfftFft1DMix: unsupported fftN " << fftN << " (factorization incomplete)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    radixListLen = (int64_t)radixVec.size();

    // 2. DFT matrix
    // GenWMatrix*ForMultiLen 内部按 radixListLen 循环，单 radix（==1）同样适用，无需分支（issue #84）
    int64_t dftLen = GetTwiddleMatrixLen(fftN, radixVec);
    dftHost.assign(dftLen, 0.0f);
    if (forward) GenWMatrixForwardForMultiLen(radixListLen, radixVec.data(), fftN, dftHost.data());
    else GenWMatrixInverseForMultiLen(radixListLen, radixVec.data(), fftN, dftHost.data());

    // 3. Twiddle matrix
    int64_t twLen = GetTwMatrixLen(fftN, radixVec);
    twHost.assign(twLen, 0.0f);
    if (twLen > 0) GenTwMatrix(fftN, radixVec, twHost.data());

    // 4. Radix list (int32)
    radixListHost.assign(radixVec.begin(), radixVec.end());

    // 5. Workspace
    int64_t parity = 0;
    InitMixRadixParam(parity, fftN, batchSize, radixVec, wsIn, wsOut, wsSync, wsC2c, wsAux);
    return ACL_SUCCESS;
}

// 设备缓冲集合（!102 codecheck 拆分）：成员 unique_ptr 析构即释放
struct Fft1DMixDevBuffers {
    std::unique_ptr<void, AclrtFreeDeleter> in, out, dft, tw, radix, ws, til;
};

// 设备缓冲分配与 H2D 上行（!102 codecheck 拆分）：守卫与 malloc 交错（issue #120）的
// 失败自动释放语义由成员 unique_ptr 构造顺序等价保持
static aclError AllocFft1DMixDevBuffers(Fft1DMixDevBuffers &dev, const float *x, size_t inputSize,
                                        const std::vector<float> &dftHost, const std::vector<float> &twHost,
                                        const std::vector<int32_t> &radixListHost,
                                        const OpsFft::FftAllMixTilingData &tilingData, int64_t totalWs)
{
    int64_t dftLen = static_cast<int64_t>(dftHost.size());
    int64_t twLen = static_cast<int64_t>(twHost.size());
    void *ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.in.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.out.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, dftLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    dev.dft.reset(ptr);
    if (twLen > 0) CHECK_ACL(aclrtMalloc(&ptr, twLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    dev.tw.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, radixListHost.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    dev.radix.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, totalWs, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.ws.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, sizeof(OpsFft::FftAllMixTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    dev.til.reset(ptr);
    CHECK_ACL(aclrtMemcpy(dev.in.get(), inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev.dft.get(), dftLen * sizeof(float), dftHost.data(), dftLen * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    if (twLen > 0) {
        CHECK_ACL(aclrtMemcpy(dev.tw.get(), twLen * sizeof(float), twHost.data(), twLen * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }
    CHECK_ACL(aclrtMemcpy(dev.radix.get(), radixListHost.size() * sizeof(int32_t), radixListHost.data(),
                          radixListHost.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev.til.get(), sizeof(OpsFft::FftAllMixTilingData), &tilingData,
                          sizeof(OpsFft::FftAllMixTilingData), ACL_MEMCPY_HOST_TO_DEVICE));
    return ACL_SUCCESS;
}

// 内核档位选择与启动（!102 codecheck 拆分）
static aclError LaunchFft1DMixKernel(int64_t fftN, std::vector<int64_t> &radixVec, uint32_t needCoreNum,
                                     void *stream, const Fft1DMixDevBuffers &dev)
{
    // 7. Select kernel variant based on aivSplitWay (1-6)
    int32_t aivSplitWay = InitAiVSplitWay(fftN, radixVec);
    uint8_t *sync = nullptr;
    CHECK_ACL(aclrtGetHardwareSyncAddr((void **)&sync));
    void *dIn = dev.in.get();
    void *dOut = dev.out.get();
    void *dDft = dev.dft.get();
    void *dTw = dev.tw.get();
    void *dRadix = dev.radix.get();
    void *dWs = dev.ws.get();
    void *dTil = dev.til.get();
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
    return ACL_SUCCESS;
}

extern "C" aclError aclfftFft1DMix(float *x, float *y, uint32_t n, int32_t norm,
                                    uint32_t batches, int isForward, void *stream) {
    CHECK_ACL(ValidateFft1DMixParams(x, y, n, norm, batches, isForward));
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = plat->GetCoreNumAic(); if (coreNum == 0) coreNum = 1;
    uint32_t maxCore = std::min(coreNum, 20u);
    uint32_t needCoreNum = batches > maxCore ? maxCore : batches;
    if (needCoreNum == 0) needCoreNum = 1;

    bool forward = (isForward != 0);
    int64_t fftN = n, batchSize = batches;

    std::vector<int64_t> radixVec;
    std::vector<float> dftHost;
    std::vector<float> twHost;
    std::vector<int32_t> radixListHost;
    int64_t wsIn = 0, wsOut = 0, wsSync = 0, wsC2c = 0, wsAux = 0;
    int64_t radixListLen = 0;
    CHECK_ACL(BuildFft1DMixHostData(fftN, batchSize, forward, radixVec, dftHost, twHost, radixListHost,
                                    wsIn, wsOut, wsSync, wsC2c, wsAux, radixListLen));

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
    Fft1DMixDevBuffers dev;
    CHECK_ACL(AllocFft1DMixDevBuffers(dev, x, inputSize, dftHost, twHost, radixListHost, tilingData, totalWs));

    CHECK_ACL(LaunchFft1DMixKernel(fftN, radixVec, needCoreNum, stream, dev));

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, inputSize, dev.out.get(), inputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SUCCESS;
}
