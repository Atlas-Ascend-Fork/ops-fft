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

// 2D DD 执行尺寸域谓词（issue #286）：2D 执行层当前仅 {32,64,128}×{32,64,128}
// 有 DD 实现。原执行层自持 static IsDdSize、创建层无校验，两处口径易漂移；
// 收敛为单一事实源，lib 创建期与 src 执行层共用
inline bool FftPlanIsDdSize(int64_t s)
{
    return s == 32 || s == 64 || s == 128;
}

/**
 * @brief 创建期校验 2D 尺寸域（aclfftMakePlan2d 使用）
 *
 * 非 DD 尺寸返回 ACLFFT_NOT_SUPPORTED（尺寸本身合法但无对应执行实现，与
 * #208 类型门禁及执行层 fft2_d_exec_api.cpp 的错误码一致），日志前缀由
 * 调用方入口名构成
 */
inline aclfftResult FftPlanValidate2DSize(int nx, int ny, const char* entryName)
{
    if (!FftPlanIsDdSize(nx) || !FftPlanIsDdSize(ny)) {
        std::cerr << "[ops-fft] " << entryName << ": size=" << nx << "x" << ny
                  << " not supported, only {32,64,128}x{32,64,128} is supported" << std::endl;
        return ACLFFT_NOT_SUPPORTED;
    }
    return ACLFFT_SUCCESS;
}

#endif // FFT_PLAN_VALIDATE_H
