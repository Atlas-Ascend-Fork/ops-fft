/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_HANDLE_IMPL_H
#define FFT_HANDLE_IMPL_H

#include <memory>
#include "cann_ops_fft.h"

/**
 * @brief FFT Plan 内部实现结构
 *
 * 包含 Plan 配置和算子状态
 * 结构体名称 aclfftHandle_t 与公共头文件中的定义保持一致
 */
struct aclfftHandle_t {
    // ========== 基本配置 ==========
    int32_t rank;                       ///< 维度数 (1, 2, 或 3)
    int32_t lengths[3];                 ///< 每个维度的大小
    int32_t stride[3];                  ///< stride大小
    int32_t batch;                      ///< 批量大小

    aclfftType type;                    ///< 变换类型
    int32_t direction;                  ///< 变换方向（ACLFFT_FORWARD 或 ACLFFT_BACKWARD）
    int32_t normMode;                   ///< 归一化模式（内部使用）

    // ========== 数据布局 ==========
    size_t input_size;                  ///< 输入数据大小（bytes）
    size_t output_size;                 ///< 输出数据大小（bytes）
    size_t element_size;                ///< 元素大小（bytes）

    // ========== 执行配置 ==========
    aclrtStream stream;                 ///< 执行流

    // ========== 算子特定状态 ==========
    void* operator_state;               ///< 算子内部状态（由算子层管理）
    bool has_operator_state;            ///< 是否有算子状态

    // ========== 标志 ==========
    bool is_initialized;                ///< 是否已初始化
    // 说明（issue #170）：is_destroyed 死字段已删除——全仓无置 true 的赋值点，
    // destroy 流程直接释放并置空句柄，生命周期由 destroy-置空约定保障

    /**
     * @brief 构造函数
     */
    aclfftHandle_t()
        : rank(0),
          batch(1),
          type(ACLFFT_C2C),
          direction(ACLFFT_FORWARD),
          normMode(0),
          input_size(0),
          output_size(0),
          element_size(0),
          stream(nullptr),
          operator_state(nullptr),
          has_operator_state(false),
          is_initialized(false),
          lengths{0},
          stride{0} {}

    /**
     * @brief 析构函数
     */
    ~aclfftHandle_t() {
        // 注意：operator_state 由算子层管理释放
        // 这里只需置空
        operator_state = nullptr;
    }
};

#endif // FFT_HANDLE_IMPL_H
