/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>

#include "cann_ops_fft.h"
#include "fft_handle_impl.h"
#include "fft_error.h"

/**
 * @brief 创建空的 FFT Plan
 *
 * 创建一个未初始化的 Plan，需要后续调用 aclfftMakePlan1d/2d/3d 来初始化
 */
aclfftResult aclfftCreate(aclfftHandle* plan) {
    // 参数验证
    ACLFFT_CHECK_NULL(plan);

    // 创建 Plan 对象
    aclfftHandle_t* new_plan = new (std::nothrow) aclfftHandle_t();
    if (new_plan == nullptr) {
        return ACLFFT_ALLOC_FAILED;
    }

    *plan = new_plan;
    return ACLFFT_SUCCESS;
}

/**
 * @brief 创建并初始化 1D FFT Plan
 *
 * @param plan 输出参数：返回创建的 Plan 句柄
 * @param nx X 维度大小
 * @param type 变换类型
 * @param batch 批量大小
 * @return aclfftResult 错误码
 */
aclfftResult aclfftPlan1d(aclfftHandle* plan, int nx, aclfftType type, int batch, int dimType) {
    // 参数验证
    ACLFFT_CHECK_NULL(plan);
    ACLFFT_CHECK_PARAM(nx > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(batch > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(dimType == ACLFFT_HORIZONTAL || dimType == ACLFFT_VERTICAL, ACLFFT_INVALID_VALUE);
    ACLFFT_CHECK_PARAM(aclfft_is_valid_type(type), ACLFFT_INVALID_TYPE);
    // 创建期拒绝当前无 1D 执行实现的类型（issue #171，对齐 #135 的 2D 处理）：
    // Z2Z/D2Z/Z2D 执行入口为恒 NOT_IMPLEMENTED 空桩，原行为"创建成功、执行期
    // 报错"使错误暴露过晚。1D 执行层当前仅支持 C2C/R2C/C2R，
    // 待双精度实现落地后放开对应分支。
    if (type != ACLFFT_C2C && type != ACLFFT_R2C && type != ACLFFT_C2R) {
        std::cerr << "[ops-fft] aclfftPlan1d: type=" << static_cast<int>(type)
                  << " has no 1D execution implementation yet" << std::endl;
        return ACLFFT_NOT_SUPPORTED;
    }
    // 创建期按 dimType 强制 FFT_1D.md 声明的 fftSize 域约束（issue #190）：
    // 横向 nx ≤ 2^27；纵向 nx ∈ [256, 2^18] 且为 2 的幂。原仅校验 nx > 0，
    // 非法尺寸 plan 创建成功、错误推迟到执行期才暴露
    if (dimType == ACLFFT_HORIZONTAL) {
        if (nx > (1 << 27)) {
            std::cerr << "[ops-fft] aclfftPlan1d: nx=" << nx << " exceeds 2^27 for horizontal FFT" << std::endl;
            return ACLFFT_INVALID_SIZE;
        }
    } else {
        const bool powerOfTwo = (nx > 0) && ((nx & (nx - 1)) == 0);
        if (nx < 256 || nx > (1 << 18) || !powerOfTwo) {
            std::cerr << "[ops-fft] aclfftPlan1d: nx=" << nx
                      << " out of [256, 262144] or not a power of 2 for vertical FFT" << std::endl;
            return ACLFFT_INVALID_SIZE;
        }
    }

    // 创建 Plan
    aclfftResult res = aclfftCreate(plan);
    if (res != ACLFFT_SUCCESS) {
        return res;
    }

    // 初始化 Plan
    size_t workSize;
    res = aclfftMakePlan1d(*plan, nx, type, batch, dimType, &workSize);
    if (res != ACLFFT_SUCCESS) {
        aclfftDestroy(*plan);
        *plan = nullptr;
    }

    return res;
}

/**
 * @brief 创建并初始化 2D FFT Plan
 */
aclfftResult aclfftPlan2d(aclfftHandle* plan, int batch, int nx, int ny, aclfftType type) {
    // 参数验证
    ACLFFT_CHECK_NULL(plan);
    ACLFFT_CHECK_PARAM(nx > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(ny > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(batch > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(aclfft_is_valid_type(type), ACLFFT_INVALID_TYPE);
    // 创建期拒绝当前无执行实现的类型（issue #135）：2D 执行层仅有 C2C 实现
    // （ExecR2C_2D/ExecC2R_2D 为 weak 声明无定义，Z2Z/D2Z/Z2D Exec 为恒
    // NOT_IMPLEMENTED 桩），原行为"创建成功、执行期报错"使错误暴露过晚。
    // 待 2D R2C/C2R 或双精度实现落地后放开对应分支。
    if (type != ACLFFT_C2C) {
        std::cerr << "[ops-fft] aclfftPlan2d: type=" << static_cast<int>(type)
                  << " has no 2D execution implementation yet" << std::endl;
        return ACLFFT_NOT_SUPPORTED;
    }

    // 创建 Plan
    aclfftResult res = aclfftCreate(plan);
    if (res != ACLFFT_SUCCESS) {
        return res;
    }

    // 初始化 Plan
    size_t workSize;
    res = aclfftMakePlan2d(*plan, batch, nx, ny, type, &workSize);
    // 如果初始化失败，清理 Plan
    if (res != ACLFFT_SUCCESS) {
        aclfftDestroy(*plan);
        *plan = nullptr;
    }

    return res;
}

/**
 * @brief 创建并初始化 3D FFT Plan
 */
aclfftResult aclfftPlan3d(aclfftHandle* plan, int nx, int ny, int nz, aclfftType type) {
    // 参数验证
    ACLFFT_CHECK_NULL(plan);
    ACLFFT_CHECK_PARAM(nx > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(ny > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(nz > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(aclfft_is_valid_type(type), ACLFFT_INVALID_TYPE);

    // 创建 Plan
    aclfftResult res = aclfftCreate(plan);
    if (res != ACLFFT_SUCCESS) {
        return res;
    }

    // 初始化 Plan
    size_t workSize;
    res = aclfftMakePlan3d(*plan, nx, ny, nz, type, &workSize);
    // 如果初始化失败，清理 Plan
    if (res != ACLFFT_SUCCESS) {
        aclfftDestroy(*plan);
        *plan = nullptr;
    }

    return res;
}
