/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_RADIX2_HOST_COMMON_H
#define FFT_RADIX2_HOST_COMMON_H

#include <cstdint>
#include <vector>

#include "fft_common_core.h"

// radix-2 族（fft_b/fft_n）host 侧 W/T 矩阵生成公共实现（!131 codecheck 重复代码收敛：
// 两入口原各持一份逐字重复实现，仅注释与换行差异；!102 codecheck 跨文件同构块收敛先例）

// W 矩阵（radix-2 点级旋转矩阵）：isLastIter 区分末级（row i / i+radix 布局）与非末级
// （row 2*i / 2*i+1 布局）；forward 翻转 sin 项符号，方向语义由 kernel 按 fftDirection 消费
inline std::vector<float> InitWMatrixRadix2(uint32_t radix, bool forward, bool isLastIter)
{
    int32_t size = 2 * radix * 2 * radix;
    std::vector<float> matrix(size, 0.f);

    if (isLastIter) {
        // Last iteration: layout with row i and row i+radix
        for (int64_t k = 0; k < radix * radix; ++k) {
            int64_t i = k / (radix ? radix : 1);
            int64_t j = k % (radix ? radix : 1);
            matrix[i * 2 * radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
            matrix[i * 2 * radix + radix + j] =
                sin(-1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(i + radix) * 2 * radix + j] =
                sin(1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(i + radix) * 2 * radix + radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
        }
    } else {
        // Non-last iteration: layout with row 2*i and row 2*i+1
        for (int64_t k = 0; k < radix * radix; ++k) {
            int64_t i = k / (radix ? radix : 1);
            int64_t j = k % (radix ? radix : 1);
            matrix[2 * i * 2 * radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
            matrix[2 * i * 2 * radix + radix + j] =
                sin(-1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(2 * i + 1) * 2 * radix + j] =
                sin(1.0 * K_2PI / (radix ? radix : 1) * i * j) * (forward ? (-1.0) : (1.0));
            matrix[(2 * i + 1) * 2 * radix + radix + j] = cos(-1.0 * K_2PI / (radix ? radix : 1) * i * j);
        }
    }
    return matrix;
}

// T 矩阵（级间旋转矩阵，仅迭代 0..size-2，末级无 T 矩阵）：函数体仅使用 radixVec 与
// iterIndex——fftN/forward 形参已删除（issue #279），T 矩阵不区分方向，方向由 kernel
// 按 fftDirection 处理；K_2PI 复用 fft_common_core.h 公共常量（issue #271）
inline std::vector<float> InitTMatrix(const std::vector<uint32_t>& radixVec, uint32_t iterIndex)
{
    uint32_t tempRow = radixVec[iterIndex] * 2;
    uint32_t tempCol = 1;
    for (uint32_t j = iterIndex + 1; j < radixVec.size(); j++) {
        tempCol *= radixVec[j];
    }

    uint32_t size = tempRow * tempCol; // Note: uses tempRow*tempCol not tempRow*tempCol*2
    std::vector<float> matrix(size, 0.f);

    // Formula: angle = -2*pi/(tempRow/2 * tempCol) * i * j
    // Note: tempRow/2 = radixVec[iterIndex]
    double angle_base = -1.0 * K_2PI / (tempRow / 2 * tempCol);

    for (uint32_t i = 0; i < tempRow / 2; ++i) {
        for (uint32_t j = 0; j < tempCol; ++j) {
            // Layout: [2*i*tempCol + j] for real, [(2*i+1)*tempCol + j] for imag
            double angle = angle_base * i * j;
            matrix[2 * i * tempCol + j] = static_cast<float>(cos(angle));
            matrix[(2 * i + 1) * tempCol + j] = static_cast<float>(sin(angle));
        }
    }
    return matrix;
}

#endif // FFT_RADIX2_HOST_COMMON_H
