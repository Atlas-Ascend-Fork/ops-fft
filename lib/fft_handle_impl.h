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
    // 说明（issue #214，对齐 #170 is_destroyed 先例）：direction 全仓除构造外
    // 无读写、input_size/output_size/element_size 仅写不读、operator_state/
    // has_operator_state 恒空且析构死存储，均为死字段已删除；normMode 保留——
    // exec 路径读取（如 fft1_d_exec_api.cpp:32），当前仅支持 norm=0（构造为 0，
    // 全仓无写入路径；norm 支持落地时由 plan 入口按文档语义写入）
    int32_t normMode;                   ///< 归一化模式（当前恒 0=BACKWARD；写入责任方为 plan 入口，尚未落地）

    // ========== 执行配置 ==========
    aclrtStream stream;                 ///< 执行流

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
          normMode(0),
          stream(nullptr),
          is_initialized(false),
          lengths{0},
          stride{0} {}
};

#endif // FFT_HANDLE_IMPL_H
