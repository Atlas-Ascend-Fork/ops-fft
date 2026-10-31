/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IRFFT1_D_H
#define IRFFT1_D_H

#include <cstdint>
#include "acl/acl.h"

#if defined(_WIN32) || defined(__CYGWIN__)
#define ACLFFT_API __declspec(dllexport)
#else
#define ACLFFT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * norm 归一化模式：当前版本两个带 norm 参数的入口（Irfft1DFft/Irfft1DDft）均仅支持
 * 0=BACKWARD（无缩放），传非 0 值返回 ACL_ERROR_INVALID_PARAM，后续版本再扩展
 * ORTHO/FORWARD 语义（issue #91）。
 */
ACLFFT_API aclError
aclfftIrfft1DFft(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream);

ACLFFT_API aclError
aclfftIrfft1DDft(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream);

ACLFFT_API aclError aclfftIrfft1DC2RFft(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream);

#ifdef __cplusplus
}
#endif

#endif
