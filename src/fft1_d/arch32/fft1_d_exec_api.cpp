/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft1_d.h"
#include "fft_exec_helper.h"

enum class FftCoreSelect : int
{
    kNone = 0,
    kDft = 1,
    kFftB = 2,
    kFftN = 3
};
struct FftConfigEntry {
    uint32_t batchLower;
    uint32_t batchUpper;
    std::vector<int64_t> dftRadices;
    std::vector<int64_t> fftBRadices;
    std::vector<int64_t> fftNRadices;
};

static std::vector<FftConfigEntry>& GetFftConfigs()
{
    static std::vector<FftConfigEntry> configs = {
        {8,
         96,
         {256, 512, 1024},
         {2048, 4096, 8192, 16384},
         {32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}},
        {96, 1024, {256, 512}, {1024, 2048, 4096, 8192, 16384}, {}},
        {1024, 65537, {256}, {512, 1024, 2048, 4096, 8192, 16384}, {}}};
    return configs;
}

static FftCoreSelect FindConfig(uint32_t batch, int64_t n)
{
    for (const auto& entry : GetFftConfigs()) {
        if (batch >= entry.batchLower && batch < entry.batchUpper) {
            if (std::find(entry.dftRadices.begin(), entry.dftRadices.end(), n) != entry.dftRadices.end()) {
                return FftCoreSelect::kDft;
            }
            if (std::find(entry.fftBRadices.begin(), entry.fftBRadices.end(), n) != entry.fftBRadices.end()) {
                return FftCoreSelect::kFftB;
            }
            if (std::find(entry.fftNRadices.begin(), entry.fftNRadices.end(), n) != entry.fftNRadices.end()) {
                return FftCoreSelect::kFftN;
            }
            return FftCoreSelect::kNone;
        }
    }
    return FftCoreSelect::kNone;
}

extern "C" {
aclfftResult aclfftExecC2C_1D(aclfftHandle plan, aclfftComplex* idata, aclfftComplex* odata, int direction)
{
    aclfftHandle_t* impl = plan;
    // 防护: 本函数为 weak 符号可被外部直接调用，plan/idata/odata 可能为 NULL（issue #50）
    ACLFFT_CHECK_PARAM(impl != nullptr && idata != nullptr && odata != nullptr, ACLFFT_INVALID_VALUE);
    ACLFFT_CHECK_PARAM(impl->rank == 1, ACLFFT_INVALID_VALUE);
    // direction 域校验（issue #134）：weak 符号可被直接调用，lib 层校验可被绕过
    ACLFFT_CHECK_PARAM(direction == ACLFFT_FORWARD || direction == ACLFFT_BACKWARD, ACLFFT_INVALID_VALUE);

    const uint32_t n = impl->lengths[0];
    const uint32_t batch = impl->batch;
    // norm 值域与 fft1_d.h 文档对齐（0=BACKWARD），normMode 直传（issue #90）
    int32_t fft_norm = impl->normMode;
    int isForward = direction == ACLFFT_FORWARD ? 1 : 0;
    aclError err;

    std::vector<int64_t> factors = orderedFactorize(impl->lengths[0]);
    std::vector<int64_t> uniques = deDuplicates(factors);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto socVersion = ascendcPlatform->GetSocVersion();

    if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        if (impl->stride[0] > 1) {
            // stride kernel 仅支持 [2^8, 2^18] 的 2 的幂，不支持时与水平路径一致返回 NOT_IMPLEMENTED（issue #72）
            static constexpr uint32_t K_N_STRIDE_MIN = 256;
            static constexpr uint32_t K_N_STRIDE_MAX = 262144;
            if (n < K_N_STRIDE_MIN || n > K_N_STRIDE_MAX || (n & (n - 1)) != 0) {
                std::cerr << "[ops-fft] C2C arch32 VERTICAL: n=" << n
                          << " not supported (must be power of 2 in [2^8, 2^18])" << std::endl;
                return ACLFFT_NOT_IMPLEMENTED;
            }
            err = aclfftFft1DStride(
                reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, impl->stride[0], batch, isForward,
                impl->stream);
        } else {
            int radix = ChooseRadix(impl->type, uniques);
            FftCoreSelect configCore = FindConfig(batch, static_cast<int64_t>(n));
            if (configCore == FftCoreSelect::kDft) {
                err = aclfftFft1DDft(
                    reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, fft_norm, batch, isForward,
                    impl->stream);
            } else if (configCore == FftCoreSelect::kFftB) {
                err = aclfftFft1DB(
                    reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, batch, isForward,
                    impl->stream);
            } else if (configCore == FftCoreSelect::kFftN) {
                err = aclfftFft1DN(
                    reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, fft_norm, batch, isForward,
                    impl->stream);
            } else if (n <= K_N_FFT_256) {
                err = aclfftFft1DDft(
                    reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, fft_norm, batch, isForward,
                    impl->stream);
            } else if (radix == K_RADIX_2) {
                if (n < K_N_FFT_32768) {
                    err = aclfftFft1DB(
                        reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, batch, isForward,
                        impl->stream);
                } else {
                    err = aclfftFft1DN(
                        reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, fft_norm, batch, isForward,
                        impl->stream);
                }
            } else if (radix == K_RADIX_MIX) {
                err = aclfftFft1DMix(
                    reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, batch, isForward,
                    impl->stream);
            } else {
                std::cerr << "[ops-fft] C2C arch32: n=" << n << " C2C exec path not implemented" << std::endl;
                return ACLFFT_NOT_IMPLEMENTED;
            }
        }
    } else {
        std::cerr << "  [ERROR] Unsupported SoC: " << SocVersionToString(socVersion) << ", expected Ascend910B"
                  << std::endl;
        err = ACLFFT_EXEC_FAILED;
    }

    // 质因子不支持（内层返回 ACL_ERROR_INVALID_PARAM）翻译为文档承诺的 NOT_IMPLEMENTED，
    // 其余运行期失败仍为 EXEC_FAILED（issue #97）
    if (err == ACL_SUCCESS)
        return ACLFFT_SUCCESS;
    if (err == ACL_ERROR_INVALID_PARAM)
        return ACLFFT_NOT_IMPLEMENTED;
    return ACLFFT_EXEC_FAILED;
}
} // extern "C"
