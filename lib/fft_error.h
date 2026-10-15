/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_ERROR_H
#define FFT_ERROR_H

#include "cann_ops_fft.h"

// ============================================================================
// 错误处理宏
// ============================================================================

/**
 * @brief 参数验证宏
 *
 * 如果条件为 false，返回指定的错误码
 */
#define ACLFFT_CHECK_PARAM(condition, error_code) \
    do { \
        if (!(condition)) { \
            return (error_code); \
        } \
    } while(0)

/**
 * @brief 空指针检查宏
 */
#define ACLFFT_CHECK_NULL(ptr) \
    ACLFFT_CHECK_PARAM((ptr) != nullptr, ACLFFT_INVALID_VALUE)

/**
 * @brief Plan 有效性检查宏
 */
#define ACLFFT_CHECK_PLAN(plan) \
    do { \
        ACLFFT_CHECK_NULL(plan); \
    } while(0)
    // 说明（issue #170）：原 "!(plan)->is_destroyed" 检查臂为恒真死检查——全仓
    // 无任何代码将 is_destroyed 置为 true（destroy 流程直接释放并置空句柄，
    // 不经该标记，见 fft_plan_destroy_api.cpp 注释），字段永远保持初值 false，
    // 不构成 use-after-destroy 防护。生命周期由 destroy-置空约定保障，删除
    // 死字段避免误导后续维护者。

/**
 * @brief Plan 初始化状态检查宏
 */
#define ACLFFT_CHECK_PLAN_INITIALIZED(plan) \
    do { \
        ACLFFT_CHECK_PLAN(plan); \
        ACLFFT_CHECK_PARAM((plan)->is_initialized, ACLFFT_INVALID_PLAN); \
    } while(0)

/**
 * @brief Plan 未初始化检查宏
 */
#define ACLFFT_CHECK_PLAN_NOT_INITIALIZED(plan) \
    do { \
        ACLFFT_CHECK_PLAN(plan); \
        ACLFFT_CHECK_PARAM(!(plan)->is_initialized, ACLFFT_INVALID_VALUE); \
    } while(0)

/**
 * @brief 校验 aclfftType 是否为已定义的合法枚举值
 *
 * 枚举值不连续（0x29/0x2a/0x2c/0x69/0x6a/0x6c），
 * 不能用数值区间判断，须按白名单逐一匹配。
 */
static inline bool aclfft_is_valid_type(aclfftType type) {
    switch (type) {
        case ACLFFT_C2C:
        case ACLFFT_R2C:
        case ACLFFT_C2R:
        case ACLFFT_Z2Z:
        case ACLFFT_D2Z:
        case ACLFFT_Z2D:
            return true;
        default:
            return false;
    }
}

#endif // FFT_ERROR_H
