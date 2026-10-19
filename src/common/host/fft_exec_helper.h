/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_EXEC_C2C_HELPER_H
#define FFT_EXEC_C2C_HELPER_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include "cann_ops_fft.h"
#include "fft_handle_impl.h"
#include "fft_error.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int K_RADIX_2 = 2;
constexpr int K_RADIX_MIX = -2;
constexpr int K_RADIX_ANY = -1;
constexpr int K_N_FFT_256 = 256;
constexpr int K_N_FFT_32768 = 32768;

// R2C/C2R 1D exec 入口公共校验（arch32/arch35 共用，消除跨文件重复）：
// 空指针、rank=1 校验，并拒绝 VERTICAL（stride>1）布局——R2C/C2R 无 stride 路径，
// 纵向 plan 显式返回 NOT_IMPLEMENTED 而非按 batch=1 静默计算（issue #75）。
// 初始化状态与类型校验（issue #191）：weak 符号可被直接调用，lib 层校验可被
// 绕过，未初始化 plan 的 lengths/stride 为 0 值垃圾语义、R2C 入口收到 C2R plan
// 会按错误类型语义访问缓冲，故此处统一校验 is_initialized 与 type
// （调用方通过 ACLFFT_EXEC_1D_EXPECT_TYPE 在 rank 校验后指定期望类型）。
// 注意：fft1_d C2C 不适用本宏（Ascend910B 的 Stride kernel 支持纵向布局）。
#define ACLFFT_EXEC_1D_ENTRY_CHECKS(impl, idata, odata, tag) \
    do { \
        ACLFFT_CHECK_PARAM((impl) != nullptr && (idata) != nullptr && (odata) != nullptr, ACLFFT_INVALID_VALUE); \
        ACLFFT_CHECK_PARAM((impl)->rank == 1, ACLFFT_INVALID_VALUE); \
        ACLFFT_CHECK_PARAM((impl)->is_initialized, ACLFFT_INVALID_VALUE); \
        if ((impl)->stride[0] > 1) { \
            std::cerr << "[ops-fft] " << tag << ": VERTICAL layout (stride=" << (impl)->stride[0] \
                      << ") is not supported" << std::endl; \
            return ACLFFT_NOT_IMPLEMENTED; \
        } \
    } while (0)

// 类型校验（issue #191）：与 ACLFFT_EXEC_1D_ENTRY_CHECKS 配套使用，直连入口
// 收到类型不匹配的 plan（如 R2C plan 传给 ExecC2R_1D）时返回错误而非按错误
// 类型语义访问输入缓冲
#define ACLFFT_EXEC_1D_EXPECT_TYPE(impl, expectedType, tag) \
    do { \
        if ((impl)->type != (expectedType)) { \
            std::cerr << "[ops-fft] " << tag << ": plan type=" << static_cast<int>((impl)->type) \
                      << " does not match expected " << static_cast<int>(expectedType) << std::endl; \
            return ACLFFT_INVALID_VALUE; \
        } \
    } while (0)

inline const char* SocVersionToString(platform_ascendc::SocVersion v) {
    using SV = platform_ascendc::SocVersion;
    switch (v) {
        case SV::ASCEND910:    return "Ascend910";
        case SV::ASCEND910B:   return "Ascend910B";
        case SV::ASCEND950:    return "Ascend950";
        case SV::ASCEND310P:   return "Ascend310P";
        case SV::ASCEND310B:   return "Ascend310B";
        case SV::ASCEND910_93: return "Ascend910_93";
        default:               return "Unknown";
    }
}

inline std::vector<int64_t> RADIX_2 = {2};
inline std::vector<int64_t> RADIX_MIX = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};

inline std::vector<int64_t> orderedFactorize(int64_t size)
{
    std::vector<int64_t> factors{};
    if (size <= 0) {
        return factors;
    }

    int64_t bound = static_cast<int64_t>(std::sqrt(size));
    for (int64_t factor = 2; factor <= bound;) {
        if (size % factor == 0) {
            factors.push_back(factor);
            size /= factor;
            bound = static_cast<int64_t>(std::sqrt(size));
        } else {
            factor++;
        }
    }

    if (size != 1) {
        factors.push_back(size);
    }

    return factors;
}

inline std::vector<int64_t> deDuplicates(const std::vector<int64_t> &duplicates)
{
    std::vector<int64_t> uniques;
    for (int64_t item : duplicates) {
        if (uniques.empty() || uniques.back() != item) {
            uniques.push_back(item);
        }
    }

    return uniques;
}

inline bool Support(const std::vector<int64_t> &uniques, const std::vector<int64_t> &radixSet)
{
    for (int64_t factor : uniques) {
        if (std::find(radixSet.begin(), radixSet.end(), factor) == radixSet.end()) {
            return false;
        }
    }
    return true;
}

inline int ChooseRadix(aclfftType fftType, const std::vector<int64_t> &uniques)
{
    if (fftType == aclfftType::ACLFFT_C2C) {
        if (Support(uniques, RADIX_2)) {
            return K_RADIX_2;
        }
    }

    if (Support(uniques, RADIX_MIX)) {
        return K_RADIX_MIX;
    }

    return K_RADIX_ANY;
}

#endif