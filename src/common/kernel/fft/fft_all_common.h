/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPSFFT_COMMON_KERNEL_FFT_ALL_COMMON_H
#define OPSFFT_COMMON_KERNEL_FFT_ALL_COMMON_H

#include <assert.h>

#include "kernel_utils.h"
#include "complex_combination_RI.h"
#include "complex_split_RI.h"
#include "fftmatmul.h"
#include "matrix_dot.h"
#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "utils.h"

__aicore__ __inline__ void __attribute__((always_inline))
fft_matmul(__gm__ float *gm_dft_matrix, __gm__ float *gm_input, __gm__ float *gm_output, __gm__ float *workspace,
           __gm__ float *gm_auxil, __gm__ int32_t *gm_radix_list, int64_t batch_size, int64_t N0, int64_t N1,
           int64_t N2, int64_t tile_M0, int64_t tile_N0, int64_t tile_K0, int64_t step_index, int64_t step_len,
           int64_t ldb, int64_t ldc, int64_t is_complex_split_out_zN, int64_t is_sync, bool seq357, int64_t per_N0)
{
    int32_t lda = ROUND(2 * N1, tile_M0);
    // 将矩阵乘法分成两种情况，分别是在n方向和k方向切分
    if (N1 <= 45 || (N1 <= 64 && N2 <= 8)) {
        fft_matmul_split_n_kernel<true, false, false>(is_complex_split_out_zN, gm_dft_matrix, gm_input, gm_output,
                                                      workspace, gm_auxil, gm_radix_list, batch_size * N0, 2 * N1, N2,
                                                      2 * N1, lda, ldb, ldc, tile_M0, tile_N0, tile_K0, step_index,
                                                      step_len, per_N0, seq357);
    } else {
        fft_matmul_split_k_kernel<true, false, false>(is_complex_split_out_zN, gm_dft_matrix, gm_input, gm_output,
                                                      workspace, gm_auxil, gm_radix_list, batch_size * N0, 2 * N1, N2,
                                                      2 * N1, lda, ldb, ldc, tile_M0, tile_N0, tile_K0, step_index,
                                                      step_len, per_N0, seq357);
    }

    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ __attribute__((always_inline)) void fft_matmul_last_step(
    __gm__ float *gm_dft_matrix, __gm__ float *gm_input, __gm__ float *gm_output, __gm__ float *workspace,
    __gm__ float *gm_auxil, __gm__ int32_t *gm_radix_list, int64_t batch_size, int64_t N0, int64_t N1, int32_t tile_M0,
    int32_t tile_N0, int32_t tile_K0, int32_t step_index, int32_t step_len, int64_t is_sync, bool seq357)
{
    int64_t B = batch_size;
    int64_t trans_a = 0;
    int64_t trans_b = 1;
    int64_t M = N1 * 2;
    int64_t K = 2 * ROUND(N1, 16);
    int64_t N = N0;
    int64_t lda = ROUND(K, 16);
    int64_t ldb = ROUND(K, 16);
    int64_t ldc = N;
    // 将矩阵乘法分成两种情况，分别是在n方向和k方向切分
    if (N1 <= 45 || (N1 <= 64 && N0 <= 8)) {
        fft_matmul_split_n_kernel<true, true, false>(false, gm_dft_matrix, gm_input, gm_output, workspace, gm_auxil,
                                                     gm_radix_list, B, M, N, K, lda, ldb, ldc, tile_M0, tile_N0,
                                                     tile_K0, step_index, step_len, 0, seq357);
    } else {
        fft_matmul_split_k_kernel<true, true, false>(false, gm_dft_matrix, gm_input, gm_output, workspace, gm_auxil,
                                                     gm_radix_list, B, M, N, K, lda, ldb, ldc, tile_M0, tile_N0,
                                                     tile_K0, step_index, step_len, 0, seq357);
    }

    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ __attribute__((always_inline)) void matrix_dot_sync(
    __gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_tw_matrix, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ workspace, __gm__ int32_t *__restrict__ gm_radix_list,
    __gm__ float *__restrict__ gm_auxil, int64_t batch_size, int64_t N0, int64_t N1, int64_t N2, int64_t tile_M0,
    int64_t tile_N2, int64_t tile_K0, int64_t step_index, int64_t step_len, int32_t fftDirection, bool seq357)
{
    matrix_dot_kernel(gm_input, gm_tw_matrix, gm_output, workspace, gm_radix_list, gm_auxil, batch_size, N0, N1, N2,
                      tile_M0, tile_N2, tile_K0, step_index, step_len, fftDirection, seq357);
    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ __attribute__((always_inline)) void stock_fft_mix_aic(
    __gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_dft_matrix_array,
    __gm__ float *__restrict__ gm_tw_matrix_array, __gm__ float *__restrict__ gm_workspace_input,
    __gm__ float *__restrict__ gm_workspace_output, __gm__ float *__restrict__ gm_workspace_sync,
    __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_auxil, int64_t batch_size, int64_t N, int32_t radix_list_len,
    int32_t fftDirection, int64_t max_floats = (1 << 21))
{
    // 除0整改
    if (N == 0) {
        N = 1;
    }
    set_ctrl(sbitset0(get_ctrl(), 48));
    int32_t step_len = radix_list_len;

    int32_t tile_M0;
    int32_t tile_N0;
    int32_t tile_K0;

    // 让每次处理的数据小于 2 ^ 21 次方
    int64_t L2_CACHE_MAX_FLOAT = max_floats;
    int64_t batch_part_len = (L2_CACHE_MAX_FLOAT + N - 1) / N;
    if (batch_part_len == 0) {
        batch_part_len = 1;
    }
    int64_t batch_loop = (batch_size + batch_part_len - 1) / batch_part_len;
    int64_t batch_remain = batch_size % batch_part_len;

    if (batch_loop == 1 && batch_remain > 0) {
        batch_part_len = batch_remain;
    }

    if (N > L2_CACHE_MAX_FLOAT * 5) {
        batch_part_len = batch_size;
        batch_loop = 1;
        batch_remain = batch_size;
    }

    bool seq357 = true;
    isSeq357(N, seq357);

    int64_t batch_actual;

    for (int64_t batch_index = 0; batch_index < batch_loop; batch_index++) {
        batch_actual = batch_part_len;
        if (batch_index == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        auto now_dft_matrix_array = gm_dft_matrix_array;
        auto now_tw_matrix_array = gm_tw_matrix_array;
        __gm__ float *gm_tmp_input = gm_workspace_input;
        __gm__ float *gm_tmp_output = gm_workspace_output;

        int64_t N0 = 1;
        int32_t N1;
        int64_t N2;

        int64_t per_N0 = -1;
        for (int32_t step_index = 0; step_index < step_len; step_index++) {
            N1 = gm_radix_list[step_index];
            if (N1 == 0) {
                N1 = 1;
            }
            N2 = N / N1 / N0;

            get_tile(N1, (N2 > 1) ? N2 : N0, step_index, step_len, tile_M0, tile_N0, tile_K0);

            int64_t N2_padding = ROUND(N2, tile_N0);

            if (N2 > 1) {
                int64_t ldb = N2;
                int64_t ldc = N2;

                WaitFlagDev(0);
                fft_matmul(now_dft_matrix_array, gm_tmp_input, nullptr, gm_workspace_sync, gm_auxil, gm_radix_list,
                           batch_actual, N0, N1, N2, tile_M0, tile_N0, tile_K0, step_index, step_len, ldb, ldc,
                           step_index == 0, true, seq357, per_N0);
            } else {
                WaitFlagDev(0);
                fft_matmul_last_step(now_dft_matrix_array, gm_tmp_input, nullptr, gm_workspace_sync, gm_auxil,
                                     gm_radix_list, batch_actual, N0, N1, tile_M0, tile_N0, tile_K0, step_len - 1,
                                     step_len, true, seq357);
            }
            auto tmp = gm_tmp_input;
            gm_tmp_input = gm_tmp_output;
            gm_tmp_output = tmp;
            if (step_index < step_len - 1) {
                now_dft_matrix_array += ROUND(2 * N1, tile_M0) * ROUND(2 * N1, tile_M0);
                per_N0 = N0;
                N0 *= N1;
            }
        }
    }
    PIPE_BARRIER(ALL);
}

// aiv_split_way 用于选择不同的内核，不同的内核在虚实结合中有不同的分支。各分支差距较大，导致性能较差。用aiv_split_way作为模板参数来选择分支。
template <bool is_vec_vtranspose_load, int32_t aiv_split_way>
__aicore__ __inline__ __attribute__((always_inline)) void
stock_fft_mix_aiv(__gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_dft_matrix_array,
                  __gm__ float *__restrict__ gm_tw_matrix_array, __gm__ float *__restrict__ gm_workspace_input,
                  __gm__ float *__restrict__ gm_workspace_output, __gm__ float *__restrict__ gm_workspace_sync,
                  __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
                  __gm__ float *__restrict__ gm_auxil, int64_t batch_size, int64_t N, int32_t radix_list_len,
                  int32_t fftDirection, int32_t odd_combination = 0, int32_t odd_split = 0)
{
    if (N == 0) {
        N = 1;
    }
    set_ctrl(sbitset0(get_ctrl(), 48));
    int32_t step_len = radix_list_len;

    int32_t tile_M0;
    int32_t tile_N0;
    int32_t tile_K0;
    // 让每次处理的数据小于 2 ^ 21 次方
    int64_t L2_CACHE_MAX_FLOAT = ((int64_t)1 << 21);
    int64_t batch_part_len = (L2_CACHE_MAX_FLOAT + N - 1) / N;
    if (batch_part_len == 0) {
        batch_part_len = 1;
    }
    int64_t batch_loop = (batch_size + batch_part_len - 1) / batch_part_len;
    int64_t batch_remain = batch_size % batch_part_len;

    if (batch_loop == 1 && batch_remain > 0) {
        batch_part_len = batch_remain;
    }

    if (N > L2_CACHE_MAX_FLOAT * 5) {
        batch_part_len = batch_size;
        batch_loop = 1;
        batch_remain = batch_size;
    }

    bool seq357 = true;
    isSeq357(N, seq357);

    int64_t batch_actual;

    for (int64_t batch_index = 0; batch_index < batch_loop; batch_index++) {
        batch_actual = batch_part_len;
        if (batch_index == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        __gm__ float *gm_tmp_input = gm_workspace_input;
        __gm__ float *gm_tmp_output = gm_workspace_output;

        int64_t N0 = 1;
        int32_t N1 = gm_radix_list[0];
        if (N1 == 0) {
            N1 = 1;
        }
        int64_t N2 = N / N1 / N0;

        get_tile(N1, (N2 > 1) ? N2 : N0, 0, step_len, tile_M0, tile_N0, tile_K0);

        // 虚实分离的时候将矩阵排布成(N2 / tile_N0, 2 * N1, tile_N0)，方便下次AIC读取
        if (odd_split == 1) {
            complex_split_RI_tile_zN_odd(gm_input + batch_index * batch_part_len * N, gm_tmp_input, gm_auxil,
                                         batch_actual, N1, N2, ROUND(N2, tile_N0), tile_K0, tile_N0, step_len);
        } else {
            if (is_vec_vtranspose_load) {
                complex_split_RI_tile_zN_vtranspose_load(gm_input + batch_index * batch_part_len * 2 * N, gm_tmp_input,
                                                         gm_auxil, batch_actual, N1, N2, ROUND(N2, tile_N0), tile_K0,
                                                         tile_N0, step_len);
            } else {
                complex_split_RI_tile_zN(gm_input + batch_index * batch_part_len * 2 * N, gm_tmp_input, gm_auxil,
                                         batch_actual, N1, N2, ROUND(N2, tile_N0), tile_K0, tile_N0, step_len);
            }
        }

        FftsCrossCoreSync<PIPE_MTE3, 0>(1);
        WaitFlagDev(1);
        FftsCrossCoreSync<PIPE_MTE3, 2>(0);

        auto now_dft_matrix_array = gm_dft_matrix_array;
        auto now_tw_matrix_array = gm_tw_matrix_array;
        for (int32_t step_index = 0; step_index < step_len; step_index++) {
            if (step_index != 0) {
                N1 = gm_radix_list[step_index];
                N2 = N / N1 / N0;

                get_tile(N1, (N2 > 1) ? N2 : N0, step_index, step_len, tile_M0, tile_N0, tile_K0);
            }

            int64_t N2_padding = ROUND(N2, tile_N0);

            if (N2 > 1) {
                matrix_dot_sync(nullptr, now_tw_matrix_array, gm_tmp_output, gm_workspace_sync, gm_radix_list, gm_auxil,
                                batch_actual, N0, N1, N2, tile_M0, tile_N0, tile_K0, step_index, step_len, fftDirection,
                                seq357);

                FftsCrossCoreSync<PIPE_MTE3, 0>(1);
                WaitFlagDev(1);
                FftsCrossCoreSync<PIPE_MTE3, 2>(0);

            } else {
                if (odd_combination == 1) {
                    complex_combination_vtranspose_sync_RI_odd<aiv_split_way>(
                        nullptr, gm_output + (batch_index)*batch_part_len * N, gm_workspace_sync, gm_auxil, 1,
                        batch_actual, N1, N0, N0, tile_M0, tile_N0, tile_K0, step_len);
                } else {
                    complex_combination_vtranspose_sync_RI<aiv_split_way>(
                        nullptr, gm_output + (batch_index)*batch_part_len * 2 * N, gm_workspace_sync, gm_auxil, 1,
                        batch_actual, N1, N0, N0, tile_M0, tile_N0, tile_K0, step_len);
                }
            }
            auto tmp = gm_tmp_input;
            gm_tmp_input = gm_tmp_output;
            gm_tmp_output = tmp;
            if (step_index < step_len - 1) {
                now_tw_matrix_array += 2 * N1 * ROUND(N2, tile_N0);
                N0 *= N1;
            }
        }
        if (batch_loop != 1) {
            FftsCrossCoreSync<PIPE_MTE3, 0>(1);
            WaitFlagDev(1);
        }
    }
}

// aiv_split_way 用于选择不同的内核，不同的内核在虚实结合中有不同的分支。各分支差距较大，导致性能较差。用aiv_split_way作为模板参数来选择分支。
template<bool is_vec_vtranspose_load, int32_t aiv_split_way>
__aicore__ __inline__  __attribute__((always_inline)) void stock_fft_c2r_mix_even_aiv(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ gm_workspace_input,
    __gm__ float * __restrict__ gm_workspace_output,
    __gm__ float * __restrict__ gm_workspace_sync,
    __gm__ int32_t * __restrict__ gm_radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_auxil,
    int64_t batch_size,
    int64_t N,
    int32_t radix_list_len,
    int32_t fftDirection,
    int64_t max_floats = (1 << 21)
) {
    set_ctrl(sbitset0(get_ctrl(), 48));
    int32_t step_len = radix_list_len;

    int32_t odd_combination = 0;
    int32_t odd_split = 0;

    int32_t tile_M0;
    int32_t tile_N0;
    int32_t tile_K0;
    // 让每次处理的数据小于 2 ^ 21 次方
    int64_t L2_CACHE_MAX_FLOAT = max_floats;
    int64_t batch_part_len = (L2_CACHE_MAX_FLOAT + N - 1) / N;
    int64_t batch_loop = (batch_size + batch_part_len - 1) / batch_part_len;
    int64_t batch_remain = batch_size % batch_part_len;

    if (batch_loop == 1 && batch_remain > 0) {
        batch_part_len = batch_remain;
    }

    if (N > L2_CACHE_MAX_FLOAT * 5) {
        batch_part_len = batch_size;
        batch_loop = 1;
        batch_remain = batch_size;
    }

    bool seq357 = true;
    isSeq357(N, seq357);

    int64_t batch_actual;

    for (int64_t batch_index = 0; batch_index < batch_loop; batch_index++) {

        batch_actual = batch_part_len;
        if (batch_index == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        __gm__ float * gm_tmp_input = gm_workspace_input;
        __gm__ float * gm_tmp_output = gm_workspace_output;

        __gm__ float * gm_current_input_real = gm_input + batch_index * batch_part_len * N;
        __gm__ float * gm_current_input_imag = gm_input + batch_size * N + batch_index * batch_part_len * N;

        int64_t N0 = 1;
        int32_t N1 = gm_radix_list[0];
        int64_t N2 = N / N1 / N0;

        get_tile(
            N1,
            (N2 > 1) ? N2 : N0,
            0,
            step_len,
            tile_M0,
            tile_N0,
            tile_K0
        );

        // 虚实分离的时候将矩阵排布成(N2 / tile_N0, 2 * N1, tile_N0)，方便下次AIC读取
        c2r_even_complex_split_RI_tile_zN(gm_current_input_real, gm_current_input_imag, gm_tmp_input, gm_auxil, batch_actual, N1, N2, ROUND(N2, tile_N0), tile_K0, tile_N0, step_len);

        FftsCrossCoreSync<PIPE_MTE3, 0>(1);
        WaitFlagDev(1);
        FftsCrossCoreSync<PIPE_MTE3, 2>(0);

        auto now_dft_matrix_array = gm_dft_matrix_array;
        auto now_tw_matrix_array = gm_tw_matrix_array;
        for (int64_t step_index = 0; step_index < step_len; step_index++) {
            if (step_index != 0) {
                N1 = gm_radix_list[step_index];
                N2 = N / N1 / N0;

                get_tile(
                    N1,
                    (N2 > 1) ? N2 : N0,
                    step_index,
                    step_len,
                    tile_M0,
                    tile_N0,
                    tile_K0
                );
            }

            int64_t N2_padding = ROUND(N2, tile_N0);

            if (N2 > 1) {
                matrix_dot_sync(nullptr, now_tw_matrix_array, gm_tmp_output, gm_workspace_sync, gm_radix_list, gm_auxil, batch_actual, N0, N1, N2, tile_M0, tile_N0, tile_K0, step_index, step_len, fftDirection, seq357);

                FftsCrossCoreSync<PIPE_MTE3, 0>(1);
                WaitFlagDev(1);
                FftsCrossCoreSync<PIPE_MTE3, 2>(0);

            } else {
                if (odd_combination == 1) {
                    complex_combination_vtranspose_sync_RI_odd<aiv_split_way>(nullptr, gm_output + (batch_index) * batch_part_len * N, gm_workspace_sync, gm_auxil, 1, batch_actual, N1, N0, N0, tile_M0, tile_N0, tile_K0, step_len);
                }
                else {
                    complex_combination_vtranspose_sync_RI<aiv_split_way>(nullptr, gm_output + (batch_index) * batch_part_len * 2 * N, gm_workspace_sync, gm_auxil, 1, batch_actual, N1, N0, N0, tile_M0, tile_N0, tile_K0, step_len);
                }

            }
            auto tmp = gm_tmp_input;
            gm_tmp_input = gm_tmp_output;
            gm_tmp_output = tmp;
            if (step_index < step_len - 1) {
                now_tw_matrix_array += 2 * N1 * ROUND(N2, tile_N0);
                N0 *= N1;
            }
        }
        if (batch_loop != 1) {
            FftsCrossCoreSync<PIPE_MTE3, 0>(1);
            WaitFlagDev(1);
        }
    }
}

// aiv_split_way 用于选择不同的内核，不同的内核在虚实结合中有不同的分支。各分支差距较大，导致性能较差。用aiv_split_way作为模板参数来选择分支。
template<bool is_vec_vtranspose_load, int32_t aiv_split_way>
__aicore__ __inline__  __attribute__((always_inline)) void stock_fft_r2c_mix_even_aiv(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ gm_workspace_input,
    __gm__ float * __restrict__ gm_workspace_output,
    __gm__ float * __restrict__ gm_workspace_sync,
    __gm__ int32_t * __restrict__ gm_radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_auxil,
    int64_t batch_size,
    int64_t N,
    int32_t radix_list_len,
    int32_t fftDirection,
    int64_t max_floats = (1 << 21)
) {
    set_ctrl(sbitset0(get_ctrl(), 48));
    int32_t step_len = radix_list_len;

    int32_t odd_combination = 0;
    int32_t odd_split = 0;

    int32_t tile_M0;
    int32_t tile_N0;
    int32_t tile_K0;
    // 让每次处理的数据小于 2 ^ 21 次方
    int64_t L2_CACHE_MAX_FLOAT = max_floats;
    int64_t batch_part_len = (L2_CACHE_MAX_FLOAT + N - 1) / N;
    int64_t batch_loop = (batch_size + batch_part_len - 1) / batch_part_len;
    int64_t batch_remain = batch_size % batch_part_len;

    if (batch_loop == 1 && batch_remain > 0) {
        batch_part_len = batch_remain;
    }

    if (N > L2_CACHE_MAX_FLOAT * 5) {
        batch_part_len = batch_size;
        batch_loop = 1;
        batch_remain = batch_size;
    }

    bool seq357 = true;
    isSeq357(N, seq357);

    int64_t batch_actual;

    for (int64_t batch_index = 0; batch_index < batch_loop; batch_index++) {

        batch_actual = batch_part_len;
        if (batch_index == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        __gm__ float * gm_tmp_input = gm_workspace_input;
        __gm__ float * gm_tmp_output = gm_workspace_output;

        int64_t N0 = 1;
        int32_t N1 = gm_radix_list[0];
        int64_t N2 = N / N1 / N0;

        get_tile(
            N1,
            (N2 > 1) ? N2 : N0,
            0,
            step_len,
            tile_M0,
            tile_N0,
            tile_K0
        );

        // 虚实分离的时候将矩阵排布成(N2 / tile_N0, 2 * N1, tile_N0)，方便下次AIC读取
        if (odd_split == 1) {
            complex_split_RI_tile_zN_odd(gm_input + batch_index * batch_part_len * N, gm_tmp_input, gm_auxil, batch_actual, N1, N2, ROUND(N2, tile_N0), tile_K0, tile_N0, step_len);
        } else {
            if (is_vec_vtranspose_load) {
                complex_split_RI_tile_zN_vtranspose_load(gm_input + batch_index * batch_part_len * 2 * N, gm_tmp_input, gm_auxil, batch_actual, N1, N2, ROUND(N2, tile_N0), tile_K0, tile_N0, step_len);
            } else {
                complex_split_RI_tile_zN(gm_input + batch_index * batch_part_len * 2 * N, gm_tmp_input, gm_auxil, batch_actual, N1, N2, ROUND(N2, tile_N0), tile_K0, tile_N0, step_len);
            }
        }

        FftsCrossCoreSync<PIPE_MTE3, 0>(1);
        WaitFlagDev(1);
        FftsCrossCoreSync<PIPE_MTE3, 2>(0);

        auto now_dft_matrix_array = gm_dft_matrix_array;
        auto now_tw_matrix_array = gm_tw_matrix_array;
        for (int64_t step_index = 0; step_index < step_len; step_index++) {
            if (step_index != 0) {
                N1 = gm_radix_list[step_index];
                N2 = N / N1 / N0;

                get_tile(
                    N1,
                    (N2 > 1) ? N2 : N0,
                    step_index,
                    step_len,
                    tile_M0,
                    tile_N0,
                    tile_K0
                );
            }

            int64_t N2_padding = ROUND(N2, tile_N0);

            if (N2 > 1) {
                matrix_dot_sync(nullptr, now_tw_matrix_array, gm_tmp_output, gm_workspace_sync, gm_radix_list, gm_auxil, batch_actual, N0, N1, N2, tile_M0, tile_N0, tile_K0, step_index, step_len, fftDirection, seq357);

                FftsCrossCoreSync<PIPE_MTE3, 0>(1);
                WaitFlagDev(1);
                FftsCrossCoreSync<PIPE_MTE3, 2>(0);

            } else {
                __gm__ float * gm_current_output_real = gm_output + batch_index * batch_part_len * N;
                __gm__ float * gm_current_output_imag = gm_output + batch_size * N + batch_index * batch_part_len * N;
                r2c_even_complex_combination_vtranspose_sync_RI<aiv_split_way>(nullptr, gm_current_output_real, gm_current_output_imag, gm_workspace_sync, gm_auxil, 1, batch_actual, N1, N0, N0, tile_M0, tile_N0, tile_K0, step_len);
            }
            auto tmp = gm_tmp_input;
            gm_tmp_input = gm_tmp_output;
            gm_tmp_output = tmp;
            if (step_index < step_len - 1) {
                now_tw_matrix_array += 2 * N1 * ROUND(N2, tile_N0);
                N0 *= N1;
            }
        }
        if (batch_loop != 1) {
            FftsCrossCoreSync<PIPE_MTE3, 0>(1);
            WaitFlagDev(1);
        }
    }
}

__aicore__ __inline__ void Init4Cube(__gm__ uint8_t * __restrict__ ffts_addr) {
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
    SetPadding(0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);
}

__aicore__ __inline__ void Init4Vector(__gm__ uint8_t *__restrict__ ffts_addr)
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
    AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);
}

#include "fft_all_mix_tiling_kernel_utils.h"

__aicore__ __inline__ void common_fft_mix_cube(
    __gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
    __gm__ float *__restrict__ gm_dft_matrix_array, __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace, __gm__ uint8_t *__restrict__ gm_tiling_para)
{
    Init4Cube(ffts_addr);

    OpsFft::FftAllMixTilingData tiling_data;
    InitTilingData(gm_tiling_para, &tiling_data);

    __gm__ float *__restrict__ gm_workspace_input =
        (__gm__ float *__restrict__)((__gm__ uint8_t *__restrict__)gm_workspace + tiling_data.workspaceOffsets[0]);
    __gm__ float *__restrict__ gm_workspace_output =
        (__gm__ float *__restrict__)((__gm__ uint8_t *__restrict__)gm_workspace + tiling_data.workspaceOffsets[1]);
    __gm__ float *__restrict__ gm_workspace_sync =
        (__gm__ float *__restrict__)((__gm__ uint8_t *__restrict__)gm_workspace + tiling_data.workspaceOffsets[2]);
    __gm__ float *__restrict__ gm_auxil =
        (__gm__ float *__restrict__)((__gm__ uint8_t *__restrict__)gm_workspace + tiling_data.workspaceOffsets[4]);

    stock_fft_mix_aic(gm_input, gm_dft_matrix_array, gm_tw_matrix_array, gm_workspace_input, gm_workspace_output,
                      gm_workspace_sync, gm_radix_list, gm_output, gm_auxil, tiling_data.batchSize, tiling_data.fftN,
                      tiling_data.radixListLen, tiling_data.isInverse);
}

#endif
