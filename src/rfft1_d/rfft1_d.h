/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rfft1_d.h
 * \brief rfft1_d 算子接口声明
 */

#ifndef RFFT1_D_H
#define RFFT1_D_H

#include <cstdint>
#include "acl/acl.h"

/* 导出宏定义 */
#if defined(_WIN32) || defined(__CYGWIN__)
#define ACLFFT_API __declspec(dllexport)
#else
#define ACLFFT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief rfft1_d FastDFT 算子接口 - 实数到复数的一维 FFT (arch35 FastDFT 实现)
 *
 * @param x 输入实数数组
 * @param y 输出复数数组（实部和虚部交错存储）
 * @param n FFT 长度
 * @param norm 归一化模式（0=BACKWARD, 1=ORTHO, 2=FORWARD）。
 *             注意：当前版本仅 aclfftRfft1D（FastDFT）支持完整 norm 语义；
 *             aclfftRfft1DFft/aclfftRfft1DDft 仅支持 0=BACKWARD，传非 0 值返回
 *             ACL_ERROR_INVALID_PARAM（issue #89）。
 * @param batches 批次数
 * @param stream ACL 流
 * @return ACL 错误码
 */
ACLFFT_API aclError aclfftRfft1D(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, void* stream);

ACLFFT_API aclError aclfftRfft1DR2CFft(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream);

ACLFFT_API aclError
aclfftRfft1DFft(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream);

ACLFFT_API aclError
aclfftRfft1DDft(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, int isForward, void* stream);

ACLFFT_API aclError aclfftRfft1DR2CFft1(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream);

ACLFFT_API aclError aclfftRfft1DR2CFft2(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream);

ACLFFT_API aclError aclfftRfft1DR2CFft31(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream);

ACLFFT_API aclError aclfftRfft1DR2CFft32(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream);

ACLFFT_API aclError aclfftRfft1DR2CFft4(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream);

ACLFFT_API aclError aclfftRfft1DR2CFft5(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream);

#ifdef __cplusplus
}
#endif

#endif // RFFT1_D_H
