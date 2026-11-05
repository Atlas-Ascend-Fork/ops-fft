/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft2_d.h"
#include "fft_exec_helper.h"
// DD 尺寸域谓词收敛（issue #286）：IsDdSize 原为本文件 static 实现，与 lib 创建期
// 校验两处口径易漂移，统一引用 fft_plan_validate.h 的单一事实源
#include "fft_plan_validate.h"

extern "C" {
aclfftResult aclfftExecC2C_2D(aclfftHandle plan,
                              aclfftComplex* idata,
                              aclfftComplex* odata,
                              int direction) {
    aclfftHandle_t* impl = plan;
    ACLFFT_CHECK_PARAM(impl != nullptr && idata != nullptr && odata != nullptr, ACLFFT_INVALID_VALUE);
    ACLFFT_CHECK_PARAM(impl->rank == 2, ACLFFT_INVALID_VALUE);
    // plan 初始化状态与类型校验（issue #191）：weak 符号可被直接调用，lib 层
    // 校验可被绕过；未初始化 plan 的 lengths 为 0 值垃圾语义，R2C/C2R plan 按
    // C2C 语义访问缓冲造成越界读
    ACLFFT_CHECK_PARAM(impl->is_initialized, ACLFFT_INVALID_VALUE);
    ACLFFT_EXEC_EXPECT_TYPE(impl, aclfftType::ACLFFT_C2C, "C2C 2D arch32");
    // direction 域校验（issue #147）：weak 符号可被直接调用，非法值不得静默按逆变换执行
    ACLFFT_CHECK_PARAM(direction == ACLFFT_FORWARD || direction == ACLFFT_BACKWARD, ACLFFT_INVALID_VALUE);

    int64_t fftSizeX = impl->lengths[0];
    int64_t fftSizeY = impl->lengths[1];

    if (!FftPlanIsDdSize(fftSizeX) || !FftPlanIsDdSize(fftSizeY)) {
        std::cerr << "[ops-fft] FFT2D DD: unsupported size (" << fftSizeX << "x" << fftSizeY
                  << "), only {32,64,128}x{32,64,128} is supported" << std::endl;
        return ACLFFT_NOT_SUPPORTED;
    }

    int isForward = direction == ACLFFT_FORWARD ? 1 : 0;
    aclError err = aclfftFft2DDd(reinterpret_cast<float*>(idata),
                                 reinterpret_cast<float*>(odata),
                                 fftSizeX, fftSizeY,
                                 impl->batch, isForward, impl->stream);

    return (err == ACL_SUCCESS) ? ACLFFT_SUCCESS : ACLFFT_EXEC_FAILED;
}
} // extern "C"
