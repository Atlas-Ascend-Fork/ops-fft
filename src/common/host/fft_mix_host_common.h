/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_MIX_HOST_COMMON_H
#define FFT_MIX_HOST_COMMON_H

#include <memory>

#include "fft_common_core.h"
#include "../tiling/fft_all_mix_tiling_data.h"

// mix 族（C2C-MIX/R2C-MIX/C2R-MIX）host 侧公共构建流程（!102 codecheck 重复代码收敛：
// rfft1_d_r2c_fft.cpp 与 irfft1_d_c2r_fft.cpp 拆分出的同构 helper 收敛为单份参数化实现，
// R2C/C2R 差异经 forward/isInverse/入口名/空表兜底开关四个参数表达）
namespace FftMixHostCommon {

// 设备缓冲集合：成员 unique_ptr 析构即释放
struct MixFftDevBuffers {
    std::unique_ptr<void, AclrtFreeDeleter> in, out, dft, tw, radix, ws, til, inIdx, a, b, outIdx;
};

// 宿主侧分解/矩阵/索引表/工作区参数构建（1/2/3/4/6 步）；第 5 步 input/output index
// 与 A/B 表因 R2C/C2R 索引布局方向不同，由各算子文件内的 static 生成器自行填充
inline aclError BuildMixFftHostData(int64_t fftN, int64_t batchSize, int64_t parity, bool forward,
                                    const char *entryName, std::vector<int64_t> &radixVec,
                                    std::vector<float> &dftHost, std::vector<float> &twHost,
                                    std::vector<int32_t> &radixListHost,
                                    int64_t &wsIn, int64_t &wsOut, int64_t &wsSync, int64_t &wsC2c,
                                    int64_t &wsAux, int64_t &radixListLen)
{
    // 1. Radix decomposition
    if (fftN >= N_FFT_C2C_MAX)
        InitMixRadixLong(fftN, radixVec);
    else
        InitMixRadixShort(fftN, radixVec);
    // 分解完整性校验（issue #113/#114，同 fft1_d_mix）
    if (!IsMixRadixValid(radixVec, fftN)) {
        std::cerr << "[ops-fft] " << entryName << ": unsupported fftN " << fftN << " (factorization incomplete)"
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    radixListLen = (int64_t)radixVec.size();

    // 2. DFT matrix（R2C 取正向符号、C2R 取逆向符号）
    int64_t dftLen = GetTwiddleMatrixLen(fftN, radixVec);
    dftHost.assign(dftLen, 0.0f);
    // GenWMatrix*ForMultiLen 内部按 radixListLen 循环，单 radix（==1）同样适用，无需分支（issue #84）
    if (forward) {
        GenWMatrixForwardForMultiLen(radixListLen, radixVec.data(), fftN, dftHost.data());
    } else {
        GenWMatrixInverseForMultiLen(radixListLen, radixVec.data(), fftN, dftHost.data());
    }

    // 3. Twiddle matrix
    int64_t twLen = GetTwMatrixLen(fftN, radixVec);
    twHost.assign(twLen, 0.0f);
    if (twLen > 0)
        GenTwMatrix(fftN, radixVec, twHost.data());

    // 4. Radix list
    radixListHost.assign(radixVec.begin(), radixVec.end());

    // 6. Workspace
    InitMixRadixParam(parity, fftN, batchSize, radixVec, wsIn, wsOut, wsSync, wsC2c, wsAux);
    return ACL_SUCCESS;
}

// tiling 填充（isInverse：R2C 0 / C2R 1）
inline void FillMixFftTiling(OpsFft::FftAllMixTilingData &tilingData, int64_t fftN, int64_t batchSize,
                             int64_t radixListLen, int64_t parity, int32_t isInverse,
                             int64_t wsIn, int64_t wsOut, int64_t wsSync, int64_t wsC2c)
{
    tilingData.batchSize = batchSize;
    tilingData.fftN = fftN;
    tilingData.radixListLen = radixListLen;
    tilingData.isInverse = isInverse;
    tilingData.workspaceOffsets[0] = 0;
    tilingData.workspaceOffsets[1] = wsIn;
    tilingData.workspaceOffsets[2] = wsIn + wsOut;
    tilingData.workspaceOffsets[3] = wsIn + wsOut + wsSync;
    tilingData.workspaceOffsets[4] = wsIn + wsOut + wsSync + wsC2c;
    tilingData.isOddN = (int32_t)parity;
}

// 设备缓冲分配：守卫与 malloc 交错（issue #120）的失败自动释放语义由成员 unique_ptr
// 构造顺序等价保持；四个索引/表缓冲的分配尺寸由调用方按各自口径计算（R2C 空表带
// sizeof(float) 兜底——issue #88，C2R 按表实际大小，与各自原实现一致）
inline aclError AllocMixFftDevBuffers(MixFftDevBuffers &dev, size_t inputSize, size_t outputSize,
                                      size_t dftLen, size_t twLen, size_t radixListSize,
                                      size_t inIdxAllocSize, size_t aTableAllocSize, size_t bTableAllocSize,
                                      size_t outputIndexAllocSize, int64_t totalWs)
{
    void *ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.in.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.out.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, dftLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    dev.dft.reset(ptr);
    if (twLen > 0)
        CHECK_ACL(aclrtMalloc(&ptr, twLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    dev.tw.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, radixListSize * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    dev.radix.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, totalWs, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.ws.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, sizeof(OpsFft::FftAllMixTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    dev.til.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, inIdxAllocSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.inIdx.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, aTableAllocSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.a.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, bTableAllocSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.b.reset(ptr);
    CHECK_ACL(aclrtMalloc(&ptr, outputIndexAllocSize, ACL_MEM_MALLOC_HUGE_FIRST));
    dev.outIdx.reset(ptr);
    return ACL_SUCCESS;
}

// 空表写 4 字节零值占位（issue #88 语义），非空表按实际大小上行
inline aclError CopyMixTableOrZero(void *dst, const std::vector<float> &table)
{
    if (!table.empty()) {
        return aclrtMemcpy(dst, table.size() * sizeof(float), table.data(), table.size() * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    }
    float z = 0;
    return aclrtMemcpy(dst, sizeof(float), &z, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
}

// H2D 上行：A/B 空表写 4 字节零值占位（issue #88 语义）；inIdxZeroFallback 为 true 时
// 输入索引空表同样写零值占位，保证设备侧读到确定内容（此前误写为 dA，issue #85，
// R2C 路径口径；C2R 原无该兜底，传 false 保持原行为）
inline aclError CopyMixFftHostToDevice(MixFftDevBuffers &dev, const float* x, size_t inputSize,
                                       const std::vector<float> &dftHost, const std::vector<float> &twHost,
                                       const std::vector<int32_t> &radixListHost,
                                       const std::vector<uint32_t> &inputIndex,
                                       const std::vector<float> &aTable, const std::vector<float> &bTable,
                                       const std::vector<uint32_t> &outputIndex,
                                       const OpsFft::FftAllMixTilingData &tilingData, bool inIdxZeroFallback)
{
    int64_t dftLen = static_cast<int64_t>(dftHost.size());
    int64_t twLen = static_cast<int64_t>(twHost.size());
    CHECK_ACL(aclrtMemcpy(dev.in.get(), inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(
        aclrtMemcpy(dev.dft.get(), dftLen * sizeof(float), dftHost.data(), dftLen * sizeof(float),
                    ACL_MEMCPY_HOST_TO_DEVICE));
    if (twLen > 0)
        CHECK_ACL(
            aclrtMemcpy(dev.tw.get(), twLen * sizeof(float), twHost.data(), twLen * sizeof(float),
                        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        dev.radix.get(), radixListHost.size() * sizeof(int32_t), radixListHost.data(),
        radixListHost.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        dev.til.get(), sizeof(OpsFft::FftAllMixTilingData), &tilingData, sizeof(OpsFft::FftAllMixTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE));
    if (!inputIndex.empty())
        CHECK_ACL(aclrtMemcpy(
            dev.inIdx.get(), inputIndex.size() * sizeof(uint32_t), inputIndex.data(),
            inputIndex.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    else if (inIdxZeroFallback) {
        float z = 0;
        CHECK_ACL(aclrtMemcpy(dev.inIdx.get(), sizeof(float), &z, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    CHECK_ACL(CopyMixTableOrZero(dev.a.get(), aTable));
    CHECK_ACL(CopyMixTableOrZero(dev.b.get(), bTable));
    if (!outputIndex.empty())
        CHECK_ACL(aclrtMemcpy(
            dev.outIdx.get(), outputIndex.size() * sizeof(uint32_t), outputIndex.data(),
            outputIndex.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    return ACL_SUCCESS;
}

} // namespace FftMixHostCommon

#endif
