/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rfft1_d.h"
#include "fft_exec_helper.h"

static aclError DispatchPowerOfTwoR2C(aclfftHandle_t *impl, aclfftReal *idata, aclfftComplex *odata,
                                     int64_t n, uint32_t batch, int isForward)
{
    static constexpr int64_t FFT_1 = 128;
    static constexpr int64_t FFT_2 = 16384;
    static constexpr int64_t FFT_3_1 = 65536;
    static constexpr int64_t FFT_3_2 = 2097152;
    static constexpr int64_t FFT_4 = 33554432;
    if (static_cast<uint64_t>(n) * batch > (1ULL << 31)) {
        std::cerr << "[ops-fft] R2C: n*batch=" << static_cast<uint64_t>(n) * batch << " not supported" << std::endl;
        return ACLFFT_NOT_IMPLEMENTED;
    }
    auto *input = reinterpret_cast<float *>(idata);
    auto *output = reinterpret_cast<float *>(odata);
    uint32_t size = static_cast<uint32_t>(n);
    if (n <= FFT_1) {
        std::cout << "[ops-fft] using FFT 1" << std::endl;
        return aclfftRfft1DR2CFft1(input, output, size, batch, isForward, impl->stream);
    }
    if (n <= FFT_2) {
        std::cout << "[ops-fft] using FFT 2" << std::endl;
        return aclfftRfft1DR2CFft2(input, output, size, batch, isForward, impl->stream);
    }
    if (n <= FFT_3_1) {
        std::cout << "[ops-fft] using FFT 3.1" << std::endl;
        return aclfftRfft1DR2CFft31(input, output, size, batch, isForward, impl->stream);
    }
    if (n <= FFT_3_2) {
        std::cout << "[ops-fft] using FFT 3.2" << std::endl;
        return aclfftRfft1DR2CFft32(input, output, size, batch, isForward, impl->stream);
    }
    if (n <= FFT_4) {
        std::cout << "[ops-fft] using FFT 4" << std::endl;
        return aclfftRfft1DR2CFft4(input, output, size, batch, isForward, impl->stream);
    }
    std::cout << "[ops-fft] using FFT 5" << std::endl;
    return aclfftRfft1DR2CFft5(input, output, size, batch, isForward, impl->stream);
}

static aclError DispatchNonPowerOfTwoR2C(aclfftHandle_t *impl, aclfftReal *idata, aclfftComplex *odata,
                                         int64_t n, int32_t norm, uint32_t batch, int isForward)
{
    static constexpr int64_t K_N_FFT_1024 = 1024;
    auto *input = reinterpret_cast<float *>(idata);
    auto *output = reinterpret_cast<float *>(odata);
    uint32_t size = static_cast<uint32_t>(n);
    if (n <= K_N_FFT_1024) {
        return aclfftRfft1DDft(input, output, size, norm, batch, isForward, impl->stream);
    }
    std::vector<int64_t> factors = orderedFactorize(n);
    std::vector<int64_t> uniques = deDuplicates(factors);
    int radix = ChooseRadix(aclfftType::ACLFFT_R2C, uniques);
    if (radix == K_RADIX_MIX) {
        return aclfftRfft1DR2CFft(input, output, size, batch, isForward, impl->stream);
    }
    std::cerr << "[ops-fft] R2C: n=" << n << " not supported" << std::endl;
    return ACLFFT_NOT_IMPLEMENTED;
}

extern "C" {
aclfftResult aclfftExecR2C_1D(aclfftHandle plan, aclfftReal* idata, aclfftComplex* odata)
{
    aclfftHandle_t* impl = plan;
    ACLFFT_EXEC_1D_ENTRY_CHECKS(impl, idata, odata, "R2C arch32");
    ACLFFT_EXEC_1D_EXPECT_TYPE(impl, aclfftType::ACLFFT_R2C, "R2C arch32");

    const int64_t n = static_cast<int64_t>(impl->lengths[0]);
    const uint32_t batch = impl->batch;
    // norm 值域与 rfft1_d.h 文档对齐（0=BACKWARD），normMode 直传（issue #89，同 #82 修复口径）
    int32_t rfft_norm = impl->normMode;
    int isForward = 1;

    aclError err;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto socVersion = ascendcPlatform->GetSocVersion();

    if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        if (n == (n & -n)) {
            err = DispatchPowerOfTwoR2C(impl, idata, odata, n, batch, isForward);
        } else {
            err = DispatchNonPowerOfTwoR2C(impl, idata, odata, n, rfft_norm, batch, isForward);
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
