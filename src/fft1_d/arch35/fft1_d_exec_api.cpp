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

extern "C" {
aclfftResult aclfftExecC2C_1D(aclfftHandle plan, aclfftComplex* idata, aclfftComplex* odata, int direction)
{
    aclfftHandle_t* impl = plan;
    // 防护: 本函数为 weak 符号可被外部直接调用，plan/idata/odata 可能为 NULL（issue #73）
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

    if (socVersion == platform_ascendc::SocVersion::ASCEND950) {
        // Ascend950 暂不支持 VERTICAL（按列 FFT）布局，显式拒绝而非按 batch=1 静默计算（issue #75）
        if (impl->stride[0] > 1) {
            std::cerr << "[ops-fft] C2C arch35: VERTICAL layout (stride=" << impl->stride[0]
                      << ") is not supported on Ascend950" << std::endl;
            return ACLFFT_NOT_IMPLEMENTED;
        }
        int radix = ChooseRadix(impl->type, uniques);
        if (n > 1 && radix == K_RADIX_2) {
            err = aclfftFft1DC2C(
                reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, fft_norm, batch, isForward,
                impl->stream);
        } else if (n > 1 && radix == K_RADIX_MIX) {
            err = aclfftFft1DC2CMix(
                reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata), n, fft_norm, batch, isForward,
                impl->stream);
        } else {
            std::cerr << "[ops-fft] C2C arch35: n=" << n << " radix=" << radix << " not implemented" << std::endl;
            return ACLFFT_NOT_IMPLEMENTED;
        }
    } else {
        std::cerr << "  [ERROR] Unsupported SoC: " << SocVersionToString(socVersion) << ", expected Ascend950"
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