/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_PLAN_VALIDATE_H
#define FFT_PLAN_VALIDATE_H

#include <iostream>

#include "cann_ops_fft.h"

// 1D fftSize 域约束具名常量（issue #190 口径，#270 收敛两份逐字重复的校验实现），
// 与 docs/zh/API_Reference/FFT_1D.md 声明的支持域一致
constexpr int ACLFFT_1D_HORIZONTAL_MAX_N = 1 << 27;
constexpr int ACLFFT_1D_VERTICAL_MIN_N = 256;
constexpr int ACLFFT_1D_VERTICAL_MAX_N = 1 << 18;

/**
 * @brief 创建期按 dimType 校验 1D fftSize 域（aclfftPlan1d 与 aclfftMakePlan1d 共用）
 *
 * 横向 nx ≤ 2^27；纵向 nx ∈ [256, 2^18] 且为 2 的幂。校验失败返回
 * ACLFFT_INVALID_SIZE，日志前缀由调用方入口名构成。
 */
inline aclfftResult FftPlanValidate1DSize(int dimType, int nx, const char* entryName)
{
    if (dimType == ACLFFT_HORIZONTAL) {
        if (nx > ACLFFT_1D_HORIZONTAL_MAX_N) {
            std::cerr << "[ops-fft] " << entryName << ": nx=" << nx << " exceeds 2^27 for horizontal FFT"
                      << std::endl;
            return ACLFFT_INVALID_SIZE;
        }
    } else {
        const bool powerOfTwo = (nx > 0) && ((nx & (nx - 1)) == 0);
        if (nx < ACLFFT_1D_VERTICAL_MIN_N || nx > ACLFFT_1D_VERTICAL_MAX_N || !powerOfTwo) {
            std::cerr << "[ops-fft] " << entryName << ": nx=" << nx
                      << " out of [256, 262144] or not a power of 2 for vertical FFT" << std::endl;
            return ACLFFT_INVALID_SIZE;
        }
    }
    return ACLFFT_SUCCESS;
}

#endif // FFT_PLAN_VALIDATE_H
