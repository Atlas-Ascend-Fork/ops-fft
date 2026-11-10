/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPSFFT_COMMON_KERNEL_UTILS
#define OPSFFT_COMMON_KERNEL_UTILS

#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "utils.h"
// W 矩阵分块参数单一实现（issue #318）：与 host 侧 getTile 共用
#include "fft_tile_common.h"

constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(float);  // 32KB
constexpr int32_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(float);   // 64KB
constexpr int16_t BLOCK_SIZE = 16;
constexpr int16_t BLOCK_SIZE_LOG2 = 4;
constexpr int16_t C0_SIZE = 32 / sizeof(float);
constexpr int16_t CUBE_MATRIX_SIZE = BLOCK_SIZE * C0_SIZE;              // 16 * 8
constexpr int32_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(float);  // 256KB
constexpr int64_t UINT16_STRIDE_LIMIT = 65536;
constexpr int64_t UINT32_STRIDE_LIMIT = 4294967296;
constexpr int32_t ASCENDFFT_FORWARD = -1;  // Forward FFT
constexpr int32_t ASCENDFFT_INVERSE = 1;   // Inverse FFT
constexpr int32_t COPY_CALC_3 = 3;
constexpr int32_t COPY_CALC_5 = 5;
constexpr int32_t COPY_CALC_7 = 7;
constexpr int32_t POW_MOV_2 = 2;
constexpr int32_t POW_MOV_8 = 8;
// get_tile 专用常量（R0_SIZE/N1_SIZE_45/N1_SIZE_64/N2_SIZE_8/TILE_128/CALC_TWO/
// CALC_THREE）已随实现并入 tiling/fft_tile_common.h 的 FFT_TILE_* 命名
// （issue #318 常量统一）

__aicore__ __inline__ void __attribute__((always_inline)) isSeq357(int64_t N, bool &result)
{
    int64_t input_len = N;
    int64_t copy = input_len;

    bool is3 = true;
    bool is5 = true;
    bool is7 = true;
    while (copy > 1 && copy % COPY_CALC_3 == 0) {
        copy /= COPY_CALC_3;
    }
    is3 = (copy == 1);

    copy = input_len;
    while (copy > 1 && copy % COPY_CALC_5 == 0) {
        copy /= COPY_CALC_5;
    }
    is5 = (copy == 1);

    copy = input_len;
    while (copy > 1 && copy % COPY_CALC_7 == 0) {
        copy /= COPY_CALC_7;
    }
    is7 = (copy == 1);

    result = (is3 || is5 || is7);
}

/**
 * @brief num 向上按照 padding_num 的倍数取整
 * @param [in] int64_t num：需要取整的数
 * @param [in] int64_t padding_num：需要向上取整的最小粒度
 * @return int64_t 向上取整的值
 */
__aicore__ inline int64_t __attribute__((always_inline)) ROUND(int64_t num, int64_t padding_num)
{
    if (padding_num == 0) {
        return num;
    }
    return ((num + padding_num - 1) / padding_num * padding_num);
}

/**
 * @brief 用于构建AIC和AIV同步的config
 * @param [in] int64_t mode：同步模式
 * @param [in] int64_t flagId：区分不同的同步
 * @return int64_t config：ffts_cross_core_sync第二个参数
 */
__aicore__ inline int64_t __attribute__((always_inline)) GET_FFST_MSG(int64_t mode, int64_t flagId)
{
    int64_t modeOffset = 4;
    int64_t flagOffset = 8;
    return 1 | (mode << modeOffset) | (flagId << flagOffset);
}

/**
 * @brief 取最大值
 * @param [in] int64_t x: 输入数据
 * @param [in] int64_t y: 输入数据
 * @return 最大值
 */

__aicore__ inline int64_t __attribute__((always_inline)) MAX(int64_t x, int64_t y)
{
    return x < y ? y : x;
}

/**
 * @brief 取最小值
 * @param [in] int64_t x: 输入数据
 * @param [in] int64_t y: 输入数据
 * @return 最小值
 */
__aicore__ inline int64_t __attribute__((always_inline)) MIN(int64_t x, int64_t y)
{
    return x < y ? x : y;
}

__aicore__ __inline__ void __attribute__((always_inline))
get_tile(int32_t N1, int64_t N2, int32_t step_index, int32_t step_len, int32_t &tile_M0, int32_t &tile_N0,
         int32_t &tile_K0)
{
    // 收敛为单一实现（issue #318）：原设备版与 host 侧 getTile
    // （common/host/fft_common_core.h）逐行同构但已漂移（防护/常量两侧不对称），
    // 统一走 tiling/fft_tile_common.h 的 FftGetTileParams，本函数保留原签名与
    // aicore 属性作为 kernel 侧入口
    FftGetTileParams(N1, N2, step_index, step_len, tile_M0, tile_N0, tile_K0);
}

__aicore__ __inline__ void __attribute__((always_inline))
get_tile_N0(int32_t N1, int64_t N2, int32_t step_index, int32_t step_len, int32_t &tile_N0)
{
    int32_t tile_M0;
    int32_t tile_K0;
    get_tile(N1, N2, step_index, step_len, tile_M0, tile_N0, tile_K0);
}

/**
 * @brief AIV函数：将行优先的nD矩阵GM读取到UB
 * @param [in] LocalTensor<float> dst：UB 目的地址
 * @param [in] GlobalTensor<float> src: GM 源地址
 * @param [in] int32_t m_actual：在GM中矩阵的列数（不需要对齐）
 * @param [in] int32_t n_actual：在GM中矩阵的行数（不需要对齐）
 * @param [in] int64_t srcStride：在GM中矩阵两行之间的距离（不需要对齐）
 * @param [in] int64_t dstStride：在UB中矩阵两行之间的距离（需要32B对齐）
 */
__aicore__ __inline__ void copy_gm2ubuf(AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, int32_t n_actual, int32_t m_actual,
                                        int64_t srcStride, int64_t dstStride)
{
    int32_t m_round = ROUND(m_actual, C0_SIZE);
    if (m_round == srcStride && m_round == dstStride) {
        AscendC::DataCopyPad(dst, src, AscendC::DataCopyExtParams(1, n_actual * m_round * sizeof(float), 0, 0, 0),
            AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
        return;
    }
    int32_t stride_SIZE = C0_SIZE;
    if (srcStride % 2 == 0) {
        stride_SIZE = C0_SIZE / 2;
    }
    if (srcStride % 4 == 0) {
        stride_SIZE = C0_SIZE / 4;
    }
    if (srcStride % C0_SIZE == 0 && srcStride < UINT32_STRIDE_LIMIT / sizeof(float)) {
        AscendC::DataCopyPad(dst, src, AscendC::DataCopyExtParams(n_actual, m_actual * sizeof(float),
            (srcStride - m_actual) * sizeof(float), (dstStride - m_round) / C0_SIZE, 0),
            AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
    } else if (srcStride * stride_SIZE < UINT32_STRIDE_LIMIT / sizeof(float)) {
        int32_t stride_SIZE_loop = n_actual / stride_SIZE;
        int32_t stride_SIZE_remain = n_actual % stride_SIZE;
        int32_t loop = (stride_SIZE_loop > 0 ? stride_SIZE : stride_SIZE_remain);

        for (int32_t i = 0; i < loop; i++) {
            AscendC::DataCopyPad(dst[i * dstStride], src[i * srcStride],
                AscendC::DataCopyExtParams(stride_SIZE_loop + (i < stride_SIZE_remain), m_actual * sizeof(float),
                (srcStride * stride_SIZE - m_actual) * sizeof(float), (dstStride * stride_SIZE - m_round) / C0_SIZE, 0),
                AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
        }
    } else {
        auto dst_inc = dst;
        auto src_inc = src;
        auto m_byte = m_actual * sizeof(float);
        for (int32_t i = 0; i < n_actual; i++) {
            AscendC::DataCopyPad(dst_inc, src_inc,
                AscendC::DataCopyExtParams(1, m_byte, 0, 0, 0),
                AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
            dst_inc = dst_inc[dstStride];
            src_inc = src_inc[srcStride];
        }
    }
}

/**
 * @brief AIV函数：将列优先的nD矩阵从UB读取到GM
 * @param [in] GlobalTensor<float dst：GM 目的地址
 * @param [in] LocalTensor<float> src: UB 源地址
 * @param [in] int64_t m_actual：在UB中矩阵的列数（不需要对齐）
 * @param [in] int64_t n_actual：在UB中矩阵的行数（不需要对齐）
 * @param [in] int64_t srcStride：在UB中矩阵两行之间的距离（需要32B对齐）
 * @param [in] int64_t dstStride：在GM中矩阵两行之间的距离（不需要对齐）
 */
__aicore__ __inline__ void copy_ubuf2gm(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src, int32_t n_actual, int32_t m_actual,
                                        int64_t srcStride, int64_t dstStride)
{
    int32_t m_round = ROUND(m_actual, C0_SIZE);
    if (m_round == srcStride && m_round == dstStride) {
        AscendC::DataCopyPad(dst, src,
                AscendC::DataCopyExtParams(1, n_actual * m_round * sizeof(float), 0, 0, 0));
        return;
    }
    int32_t stride_SIZE = C0_SIZE;
    if (srcStride % 2 == 0) {
        stride_SIZE = C0_SIZE / 2;
    }
    if (srcStride % 4 == 0) {
        stride_SIZE = C0_SIZE / 4;
    }

    if (dstStride % C0_SIZE == 0 && dstStride < UINT32_STRIDE_LIMIT / sizeof(float)) {
        AscendC::DataCopyPad(dst, src,
                AscendC::DataCopyExtParams(n_actual, m_actual * sizeof(float),
                (srcStride - m_round) / C0_SIZE,  (dstStride - m_actual) * sizeof(float), 0));
    } else if (dstStride * stride_SIZE < UINT32_STRIDE_LIMIT / sizeof(float)) {
        int32_t stride_SIZE_loop = n_actual / stride_SIZE;
        int32_t stride_SIZE_remain = n_actual % stride_SIZE;

        for (int32_t i = 0; i < (stride_SIZE_loop > 0 ? stride_SIZE : stride_SIZE_remain); i++) {
            AscendC::DataCopyPad(dst[i * dstStride], src[i * srcStride],
                AscendC::DataCopyExtParams(stride_SIZE_loop + (i < stride_SIZE_remain), m_actual * sizeof(float),
                (srcStride * stride_SIZE - m_round) / C0_SIZE,  (dstStride * stride_SIZE - m_actual) * sizeof(float), 0));
        }
    } else {
        auto dst_inc = dst;
        auto src_inc = src;
        auto m_byte = m_actual * sizeof(float);
        for (int32_t i = 0; i < n_actual; i++) {
            AscendC::DataCopyPad(dst_inc, src_inc,
                AscendC::DataCopyExtParams(1, m_byte, 0, 0, 0));
            dst_inc = dst_inc[dstStride];
            src_inc = src_inc[srcStride];
        }
    }
}

__aicore__ __inline__ void load_matrix_zN(AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src,
                                          int32_t R, int32_t C,
                                          int32_t valid_row, int32_t valid_col, int64_t stride, int32_t batch_size = 1,
                                          int64_t matrix_stride = 1)
{
    if (batch_size == 1) {
        constexpr int32_t C0 = 32 / sizeof(float);
        constexpr int32_t STRIDE_LIMIT = 65536;

        if (stride < STRIDE_LIMIT) {
            AscendC::DataCopy(
                dst, src,
                AscendC::Nd2NzParams(
                    1,            // ndNum
                    valid_row,             // nValue
                    valid_col,      // dValue
                    0,          // srcNdMatrixStride
                    stride,     // srcDValue
                    R,             // dstNzC0Stride
                    1,              // dstNzNStride
                    0)         // dstNzMatrixStride
            );
        } else {
            for (int32_t i = 0; i < valid_row; i++) {
                AscendC::DataCopy(
                    dst[i * C0], src[i * stride],
                    AscendC::DataCopyParams(
                        (valid_col + C0_SIZE - 1) / C0_SIZE,   // nBurst
                        1,     // lenBurst
                        0,                                      // srcGap
                        R - 1                                       // dstGap
                    )
                );
            }
        }
    } else {
        AscendC::DataCopy(
            dst, src,
            AscendC::Nd2NzParams(
                batch_size,            // ndNum
                valid_row,             // nValue
                valid_col,      // dValue
                matrix_stride,          // srcNdMatrixStride
                stride,     // srcDValue
                R,             // dstNzC0Stride
                1,              // dstNzNStride
                R * C)         // dstNzMatrixStride
        );
    }
}

__aicore__ __inline__ void load_matrix_zZ(AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src,
                                          int32_t R, int32_t C,
                                          int32_t valid_row, int32_t valid_col, int64_t stride, int32_t batch_size = 1,
                                          int64_t matrix_stride = 1)
{
    if (batch_size == 1) {
        constexpr int32_t R0 = 16;
        constexpr int32_t C0 = 32 / sizeof(float);
        constexpr int32_t STRIDE_LIMIT = 65536;

        int64_t srcNdStride = R0 * stride;
        int64_t srcNStride = stride;
        if (srcNdStride < STRIDE_LIMIT) {
            int32_t ndNum = valid_row / R0;
            int32_t remains = valid_row % R0;
            if (ndNum > 0) {
                AscendC::DataCopy(
                    dst, src,
                    AscendC::Nd2NzParams(
                        ndNum,            // ndNum
                        R0,             // nValue
                        valid_col,      // dValue
                        srcNdStride,          // srcNdMatrixStride
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
                        1,            // ndNum
                        remains,             // nValue
                        valid_col,      // dValue
                        0,          // srcNdMatrixStride
                        srcNStride,     // srcDValue
                        R0,             // dstNzC0Stride
                        1,              // dstNzNStride
                        0)         // dstNzMatrixStride
                );
            }
        } else if (srcNStride < STRIDE_LIMIT) {
            int32_t ndNum = valid_row / R0;
            int32_t remains = valid_row % R0;
            for (int32_t i = 0; i < ndNum; i++) {
                AscendC::DataCopy(
                    dst[i * R0 * C], src[i * R0 * stride],
                    AscendC::Nd2NzParams(
                        1,            // ndNum
                        R0,             // nValue
                        valid_col,      // dValue
                        0,          // srcNdMatrixStride
                        srcNStride,     // srcDValue
                        R0,             // dstNzC0Stride
                        1,              // dstNzNStride
                        0)         // dstNzMatrixStride
                );
            }
            if (remains > 0) {
                AscendC::DataCopy(
                    dst[ndNum * R0 * C], src[ndNum * R0 * stride],
                    AscendC::Nd2NzParams(
                        1,            // ndNum
                        remains,             // nValue
                        valid_col,      // dValue
                        0,          // srcNdMatrixStride
                        srcNStride,     // srcDValue
                        R0,             // dstNzC0Stride
                        1,              // dstNzNStride
                        0)         // dstNzMatrixStride
                );
            }
        } else {
            for (int32_t i = 0; i < valid_row; i++) {
                int32_t idxR0 = i / R0;
                int32_t idxInR0 = i % R0;
                AscendC::DataCopy(
                    dst[idxR0 * R0 * C + idxInR0 * C0], src[i * stride],
                    AscendC::DataCopyParams(
                        (valid_col + C0_SIZE - 1) / C0_SIZE,   // nBurst
                        1,     // lenBurst
                        0,                                      // srcGap
                        15                                       // dstGap
                    )
                );
            }
        }
    } else {
        constexpr int32_t R0 = 16;
        constexpr int32_t C0 = 32 / sizeof(float);
        constexpr int32_t STRIDE_LIMIT = 65536;

        int64_t srcNdStride = matrix_stride;
        int64_t srcNStride = stride;
        int32_t loop = (valid_row + R0 - 1) / R0;
        int32_t ndNum = batch_size;
        int32_t remains = valid_row % R0;
        for (int32_t i = 0; i < loop; i++) {
            int32_t actual = R0;
            if (i == loop - 1 && remains > 0) {
                actual = remains;
            }
            AscendC::DataCopy(
                dst[i * R0 * C], src[i * R0 * srcNStride],
                AscendC::Nd2NzParams(
                    ndNum,            // ndNum
                    actual,             // nValue
                    valid_col,      // dValue
                    srcNdStride,          // srcNdMatrixStride
                    srcNStride,     // srcDValue
                    R0,             // dstNzC0Stride
                    1,              // dstNzNStride
                    R * C)         // dstNzMatrixStride
            );
        }
    }
}

#endif
