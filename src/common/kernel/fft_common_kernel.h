/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_COMMON_KERNEL_H
#define FFT_COMMON_KERNEL_H

#include "kernel_operator.h"
#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "mem.h"
#include "utils.h"

constexpr int32_t ASCENDFFT_FORWARD_VAL = -1;
constexpr int32_t ASCENDFFT_INVERSE_VAL = 1;
constexpr int32_t ASCENDFFT_FORWARD = -1;

constexpr uint64_t INTER_CORE_SYNC_FLAG_ID = 0;
constexpr int16_t INTER_CORE_SYNC_MODE = 0;
constexpr uint64_t VC_SYNC_FLAG_ID = 6;
constexpr uint64_t CV_SYNC_FLAG_ID = 7;
constexpr uint64_t CV_SYNC_MODE = 2;

constexpr uint32_t AIC_PING_SET_FLAG = 2;
constexpr uint32_t AIC_PING_WAIT_FLAG = 3;
constexpr uint32_t AIC_PONG_SET_FLAG = 4;
constexpr uint32_t AIC_PONG_WAIT_FLAG = 5;

constexpr uint32_t AIV_PING_SET_FLAG = 3;
constexpr uint32_t AIV_PING_WAIT_FLAG = 2;
constexpr uint32_t AIV_PONG_SET_FLAG = 5;
constexpr uint32_t AIV_PONG_WAIT_FLAG = 4;

constexpr int32_t BLOCK_LEN_32B = 32 / sizeof(float);
constexpr int32_t COMPLEX_DOUBLE = 2;

constexpr int32_t FFT_BLOCK_SIZE = 16;
constexpr int32_t FFT_C0_SIZE = 32 / sizeof(float);
constexpr int32_t FFT_CUBE_MATRIX_SIZE = FFT_BLOCK_SIZE * FFT_C0_SIZE;
constexpr int32_t FFT_L0AB_PINGPONG_LEN = 32 * 1024 / sizeof(float);
constexpr int32_t FFT_L0C_PINGPONG_LEN = 64 * 1024 / sizeof(float);
constexpr int32_t FFT_L1_PINGPONG_LEN = 256 * 1024 / sizeof(float);

template <typename DTYPE = float>
__aicore__ __attribute__((always_inline)) inline void load_matrix_zZ(
    AscendC::LocalTensor<DTYPE> dst,
    AscendC::GlobalTensor<DTYPE> src,
    int32_t R, int32_t C,
    int32_t valid_row, int32_t valid_col,
    int32_t stride
) {
    constexpr int32_t R0 = 16;
    constexpr int32_t C0 = 32 / sizeof(DTYPE);
    constexpr int STRIDE_LIMIT = 65536;

    int64_t srcNdStride = R0 * stride;
    int64_t srcNStride = stride;

    if (srcNdStride < STRIDE_LIMIT) {
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;

        if (ndNum > 0) {
            AscendC::DataCopy(
                dst, src,
                AscendC::Nd2NzParams(
                    ndNum,          // ndNum
                    R0,             // nValue
                    valid_col,      // dValue
                    R0 * stride,    // srcNdMatrixStride
                    srcNStride,     // srcDValue
                    R0,             // dstNzC0Stride
                    1,              // dstNzNStride
                    R0 * C)         // dstNzMatrixStride
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * R0 * C], src[ndNum * R0 * stride],
                AscendC::Nd2NzParams(
                    1,              // ndNum
                    remains,        // nValue
                    valid_col,      // dValue
                    0,              // srcNdMatrixStride
                    srcNStride,     // srcDValue
                    R0,             // dstNzC0Stride
                    1,              // dstNzNStride
                    0)              // dstNzMatrixStride
            );
        }
    } else if (srcNStride < STRIDE_LIMIT) {
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;
        for (int32_t i = 0; i < ndNum; i++) {
            AscendC::DataCopy(
                dst[i * R0 * C], src[i * R0 * stride],
                AscendC::Nd2NzParams(
                    1,              // ndNum
                    R0,             // nValue
                    valid_col,      // dValue
                    0,              // srcNdMatrixStride
                    srcNStride,     // srcDValue
                    R0,             // dstNzC0Stride
                    1,              // dstNzNStride
                    0)              // dstNzMatrixStride
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * R0 * C], src[ndNum * R0 * stride],
                AscendC::Nd2NzParams(
                    1,              // ndNum
                    remains,        // nValue
                    valid_col,      // dValue
                    0,              // srcNdMatrixStride
                    srcNStride,     // srcDValue
                    R0,             // dstNzC0Stride
                    1,              // dstNzNStride
                    0)              // dstNzMatrixStride
            );
        }
    } else {
        for (int32_t i = 0; i < valid_row; i++) {
            int32_t idxR0 = i / R0;
            int32_t idxInR0 = i % R0;

            AscendC::DataCopy(
                dst[idxR0 * R0 * C + idxInR0 * C0], src[i * stride],
                AscendC::Nd2NzParams(
                    1,              // ndNum
                    1,              // nValue
                    valid_col,      // dValue
                    0,              // srcNdMatrixStride
                    0,              // srcDValue
                    R0,             // dstNzC0Stride
                    0,              // dstNzNStride
                    0)              // dstNzMatrixStride
            );
        }
    }
}

#endif