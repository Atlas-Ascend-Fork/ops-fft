/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "cann_ops_fft.h"
#include "fft_handle_impl.h"
#include "fft_error.h"

static const uint32_t COMPLEX_PART = 2;

/**
 * @brief 计算数据大小和元素大小
 */
// 计算数据大小（issue #136）：返回 aclfftResult 便于上浮溢出错误；
// 乘积全程 __builtin_mul_overflow 防回绕，溢出返回 ACLFFT_INVALID_SIZE
static aclfftResult calculate_data_sizes(aclfftHandle_t* plan) {
    // 计算元素大小
    switch (plan->type) {
        case ACLFFT_C2C:
        case ACLFFT_R2C:
        case ACLFFT_C2R:
            plan->element_size = sizeof(float);  // 单精度
            break;
        case ACLFFT_Z2Z:
        case ACLFFT_D2Z:
        case ACLFFT_Z2D:
            plan->element_size = sizeof(double);  // 双精度
            break;
        default:
            plan->element_size = sizeof(float);
            break;
    }

    // 防溢出乘法助手：任意一步回绕即失败
    auto mul_checked = [](size_t a, size_t b, size_t& out) -> bool {
        return !__builtin_mul_overflow(a, b, &out);
    };

    // 计算实数侧全量元素数（所有维度乘积 × batch）
    size_t full_elements = 1;
    for (int i = 0; i < plan->rank; ++i) {
        if (!mul_checked(full_elements, static_cast<size_t>(plan->lengths[i]), full_elements)) {
            return ACLFFT_INVALID_SIZE;
        }
    }
    if (!mul_checked(full_elements, static_cast<size_t>(plan->batch), full_elements)) {
        return ACLFFT_INVALID_SIZE;
    }

    // Hermitian packed 元素数：最后一维用 floor(n/2)+1，其余维不变
    size_t packed_elements = 1;
    for (int i = 0; i < plan->rank - 1; ++i) {
        if (!mul_checked(packed_elements, static_cast<size_t>(plan->lengths[i]), packed_elements)) {
            return ACLFFT_INVALID_SIZE;
        }
    }
    if (!mul_checked(packed_elements, static_cast<size_t>(plan->lengths[plan->rank - 1] / 2 + 1), packed_elements) ||
        !mul_checked(packed_elements, static_cast<size_t>(plan->batch), packed_elements)) {
        return ACLFFT_INVALID_SIZE;
    }

    // 根据类型调整大小（含 ×COMPLEX_PART 与 ×element_size 的溢出防护）
    size_t complex_full = 0;
    size_t complex_packed = 0;
    if (!mul_checked(full_elements, static_cast<size_t>(COMPLEX_PART), complex_full) ||
        !mul_checked(packed_elements, static_cast<size_t>(COMPLEX_PART), complex_packed)) {
        return ACLFFT_INVALID_SIZE;
    }

    if (plan->type == ACLFFT_R2C || plan->type == ACLFFT_D2Z) {
        // 实数到复数：输入是实数全量，输出按 Hermitian packed
        if (!mul_checked(full_elements, plan->element_size, plan->input_size) ||
            !mul_checked(complex_packed, plan->element_size, plan->output_size)) {
            return ACLFFT_INVALID_SIZE;
        }
    } else if (plan->type == ACLFFT_C2R || plan->type == ACLFFT_Z2D) {
        // 复数到实数：输入按 Hermitian packed，输出是实数全量
        if (!mul_checked(complex_packed, plan->element_size, plan->input_size) ||
            !mul_checked(full_elements, plan->element_size, plan->output_size)) {
            return ACLFFT_INVALID_SIZE;
        }
    } else {
        // C2C 或 Z2Z：输入输出都是复数全量
        if (!mul_checked(complex_full, plan->element_size, plan->input_size) ||
            !mul_checked(complex_full, plan->element_size, plan->output_size)) {
            return ACLFFT_INVALID_SIZE;
        }
    }
    return ACLFFT_SUCCESS;
}

/**
 * @brief 初始化 1D FFT Plan
 *
 * @param plan Plan 句柄
 * @param nx X 维度大小
 * @param type 变换类型
 * @param batch 批量大小
 * @param dimType 维度方向，必须为 ACLFFT_HORIZONTAL 或 ACLFFT_VERTICAL
 * @param workSize 输出参数：返回所需工作空间大小（bytes），可为 nullptr
 * @return aclfftResult 错误码
 */
aclfftResult aclfftMakePlan1d(aclfftHandle plan, int nx, aclfftType type, int batch,
                              int dimType, size_t* workSize) {
    aclfftHandle_t* impl = plan;

    // 参数验证
    ACLFFT_CHECK_PLAN_NOT_INITIALIZED(impl);
    ACLFFT_CHECK_PARAM(nx > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(batch > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(dimType == ACLFFT_HORIZONTAL || dimType == ACLFFT_VERTICAL, ACLFFT_INVALID_VALUE);
    ACLFFT_CHECK_PARAM(aclfft_is_valid_type(type), ACLFFT_INVALID_TYPE);

    // 设置基本参数
    impl->rank = 1;
    impl->lengths[0] = nx;
    impl->batch = batch;
    impl->type = type;

    // batch/stride 先定型，再计算数据大小（issue #138）：
    // VERTICAL 编码为 batch=1/stride=batch（有意设计），sizes 必须在定型后计算，
    // 保证 plan 元数据自洽（总量 = batch × 单份）
    if (dimType == ACLFFT_VERTICAL) {
        impl->batch = 1;
        impl->stride[0] = batch;
    } else {
        impl->stride[0] = 1;
    }

    // 计算数据大小（issue #136）：乘积全程用 __builtin_mul_overflow 防溢出回绕，
    // 溢出时置 0 并返回 INVALID_SIZE（当前无消费方，接入前不会误用回绕值）
    if (calculate_data_sizes(impl) != ACLFFT_SUCCESS) {
        return ACLFFT_INVALID_SIZE;
    }

    // 返回工作空间大小（暂时设为 0）
    if (workSize != nullptr) {
        *workSize = 0;
    }

    // 标记为已初始化
    impl->is_initialized = true;

    return ACLFFT_SUCCESS;
}

/**
 * @brief 初始化 2D FFT Plan
 *
 * @param plan Plan 句柄
 * @param batch 批量大小
 * @param nx X 维度大小
 * @param ny Y 维度大小
 * @param type 变换类型
 * @param workSize 输出参数：返回所需工作空间大小（bytes），可为 nullptr
 * @return aclfftResult 错误码
 */
aclfftResult aclfftMakePlan2d(aclfftHandle plan, int batch, int nx, int ny, aclfftType type,
                              size_t* workSize) {
    aclfftHandle_t* impl = plan;

    ACLFFT_CHECK_PLAN_NOT_INITIALIZED(impl);
    ACLFFT_CHECK_PARAM(nx > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(ny > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(batch > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(aclfft_is_valid_type(type), ACLFFT_INVALID_TYPE);

    impl->rank = 2;
    impl->lengths[0] = nx;
    impl->lengths[1] = ny;
    impl->batch = batch;
    impl->type = type;
    impl->stride[0] = 1;
    impl->stride[1] = 1;

    // issue #136：溢出防护版本，失败上浮给调用方（aclfftPlan2d 会清理 plan）
    if (calculate_data_sizes(impl) != ACLFFT_SUCCESS) {
        return ACLFFT_INVALID_SIZE;
    }

    if (workSize != nullptr) {
        *workSize = 0;
    }

    impl->is_initialized = true;

    return ACLFFT_SUCCESS;
}

/**
 * @brief 初始化 3D FFT Plan

 * 占位实现，返回 ACLFFT_NOT_IMPLEMENTED
 * @param plan Plan 句柄
 * @param nx X 维度大小
 * @param ny Y 维度大小
 * @param nz Z 维度大小
 * @param type 变换类型
 * @param workSize 输出参数：返回所需工作空间大小（bytes），可为 nullptr
 * @return aclfftResult 错误码
 */
aclfftResult aclfftMakePlan3d(aclfftHandle plan, int nx, int ny, int nz, aclfftType type,
                              size_t* workSize) {
    // 暂不实现
    return ACLFFT_NOT_IMPLEMENTED;
}
