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
                                     int64_t n, int32_t norm, uint32_t batch, int isForward)
{
    static constexpr int64_t FFT_1 = 128;
    static constexpr int64_t FFT_2 = 16384;
    static constexpr int64_t FFT_3_1 = 65536;
    static constexpr int64_t FFT_3_2 = 2097152;
    static constexpr int64_t FFT_4 = 33554432;
    // 幂次路径 6 个 kernel 的 tiling N/batch 与输出偏移均按 int32 处理（issue #218/#219）：
    // 1) 输出每行占 n+2 个 float，最后一行偏移 (batch-1)*(n+2)，门禁必须按输出容量
    //    口径 batch*(n+2) 校验，仅看 n*batch 时 n=32768/batch=65536 等组合仍会溢出 INT32_MAX；
    // 2) tiling 以 uint32 写入/读取（下方 host 写入端与 kernel 读取端已对齐），n 超过
    //    int32 有符号范围仍会经 blockN = 常量/N 等有符号运算产生负值，按 DFT 路径
    //    "显式报错而非静默错误" 口径拒绝。
    if (norm != 0) {
        std::cerr << "[ops-fft] R2C: norm=" << norm << " is not supported on power-of-two path"
                  << " (only 0=BACKWARD in this version)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    if (n <= 0 || batch == 0) {
        std::cerr << "[ops-fft] R2C: n=" << n << " batch=" << batch << " invalid" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    if (static_cast<uint64_t>(batch) * (static_cast<uint64_t>(n) + 2) > (1ULL << 31)) {
        std::cerr << "[ops-fft] R2C: batch*(n+2)=" << static_cast<uint64_t>(batch) * (static_cast<uint64_t>(n) + 2)
                  << " not supported (int32 output-offset limit)" << std::endl;
        // 与 norm/n<=0 分支一致返回 aclError 域错误码，由外层统一翻译为
        // ACLFFT_NOT_IMPLEMENTED（文档承诺口径）
        return ACL_ERROR_INVALID_PARAM;
    }
    auto *input = reinterpret_cast<float *>(idata);
    auto *output = reinterpret_cast<float *>(odata);
    uint32_t size = static_cast<uint32_t>(n);
    if (n <= FFT_1) {
        return aclfftRfft1DR2CFft1(input, output, size, batch, isForward, impl->stream);
    }
    if (n <= FFT_2) {
        return aclfftRfft1DR2CFft2(input, output, size, batch, isForward, impl->stream);
    }
    if (n <= FFT_3_1) {
        return aclfftRfft1DR2CFft31(input, output, size, batch, isForward, impl->stream);
    }
    if (n <= FFT_3_2) {
        return aclfftRfft1DR2CFft32(input, output, size, batch, isForward, impl->stream);
    }
    if (n <= FFT_4) {
        return aclfftRfft1DR2CFft4(input, output, size, batch, isForward, impl->stream);
    }
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
        // MIX 分支此前丢弃 norm 形参致 norm!=0 静默按 BACKWARD 计算（issue #224），
        // 补传后由入口显式拒绝
        return aclfftRfft1DR2CFft(input, output, size, norm, batch, isForward, impl->stream);
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
        // 幂次路由需 n>1 下界（issue #216）：n=1 满足 n&(n-1)==0 但会被路由到 fft1，
        // 其 vec 路径按 4 点蝶形展开 compM=TILE_LENGTH/n=4096，越界读写 outQueueDst；
        // n=1 回落 DFT 路径（batch-rfft 之前的行为，输出 1 复数与 DFT 输出布局一致）。
        if (n > 1 && (n & (n - 1)) == 0) {
            err = DispatchPowerOfTwoR2C(impl, idata, odata, n, rfft_norm, batch, isForward);
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
