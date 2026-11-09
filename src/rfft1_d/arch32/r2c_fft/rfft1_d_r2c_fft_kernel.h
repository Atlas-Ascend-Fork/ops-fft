/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RFFT1_D_R2C_FFT_KERNEL_H
#define RFFT1_D_R2C_FFT_KERNEL_H

#include "fft_r2c_common.h"

namespace FFT1DR2CFftKernel {

// Even batch variants
extern "C" __global__ __aicore__ void fft_r2c_1_even(
    __gm__ uint8_t *__restrict__ ffts_addr,
    __gm__ float *__restrict__ gm_input,
    __gm__ uint32_t *__restrict__ gm_input_index,
    __gm__ float *__restrict__ gm_a,
    __gm__ float *__restrict__ gm_b,
    __gm__ uint32_t *__restrict__ gm_output_index,
    __gm__ float *__restrict__ gm_dft_matrix_array,
    __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list,
    __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace,
    __gm__ uint8_t *__restrict__ gm_tiling_para) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_r2c_mix_cube_by_batch(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                     gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 0, 1 << 22);
}
#elif __DAV_C220_VEC__
{
    common_fft_r2c_mix_even_vector_by_batch<1>(ffts_addr, gm_input, gm_input_index, gm_a, gm_b,
                                                gm_output_index, gm_dft_matrix_array, gm_tw_matrix_array,
                                                gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 1 << 22);
}
#endif
}

extern "C" __global__ __aicore__ void fft_r2c_2_even(
    __gm__ uint8_t *__restrict__ ffts_addr,
    __gm__ float *__restrict__ gm_input,
    __gm__ uint32_t *__restrict__ gm_input_index,
    __gm__ float *__restrict__ gm_a,
    __gm__ float *__restrict__ gm_b,
    __gm__ uint32_t *__restrict__ gm_output_index,
    __gm__ float *__restrict__ gm_dft_matrix_array,
    __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list,
    __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace,
    __gm__ uint8_t *__restrict__ gm_tiling_para) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_r2c_mix_cube_by_batch(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                     gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 0, 1 << 22);
}
#elif __DAV_C220_VEC__
{
    common_fft_r2c_mix_even_vector_by_batch<2>(ffts_addr, gm_input, gm_input_index, gm_a, gm_b,
                                                gm_output_index, gm_dft_matrix_array, gm_tw_matrix_array,
                                                gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 1 << 22);
}
#endif
}

extern "C" __global__ __aicore__ void fft_r2c_3_even(
    __gm__ uint8_t *__restrict__ ffts_addr,
    __gm__ float *__restrict__ gm_input,
    __gm__ uint32_t *__restrict__ gm_input_index,
    __gm__ float *__restrict__ gm_a,
    __gm__ float *__restrict__ gm_b,
    __gm__ uint32_t *__restrict__ gm_output_index,
    __gm__ float *__restrict__ gm_dft_matrix_array,
    __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list,
    __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace,
    __gm__ uint8_t *__restrict__ gm_tiling_para) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_r2c_mix_cube_by_batch(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                     gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 0, 1 << 22);
}
#elif __DAV_C220_VEC__
{
    common_fft_r2c_mix_even_vector_by_batch<3>(ffts_addr, gm_input, gm_input_index, gm_a, gm_b,
                                                gm_output_index, gm_dft_matrix_array, gm_tw_matrix_array,
                                                gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 1 << 22);
}
#endif
}

// Odd batch variants
extern "C" __global__ __aicore__ void fft_r2c_1_odd(
    __gm__ uint8_t *__restrict__ ffts_addr,
    __gm__ float *__restrict__ gm_input,
    __gm__ uint32_t *__restrict__ gm_input_index,
    __gm__ float *__restrict__ gm_a,
    __gm__ float *__restrict__ gm_b,
    __gm__ uint32_t *__restrict__ gm_output_index,
    __gm__ float *__restrict__ gm_dft_matrix_array,
    __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list,
    __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace,
    __gm__ uint8_t *__restrict__ gm_tiling_para) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_r2c_mix_cube_by_batch(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                     gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 1);
}
#elif __DAV_C220_VEC__
{
    common_fft_r2c_mix_vector_by_batch<1>(ffts_addr, gm_input, gm_input_index, gm_a, gm_b,
                                           gm_output_index, gm_dft_matrix_array, gm_tw_matrix_array,
                                           gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 1);
}
#endif
}

extern "C" __global__ __aicore__ void fft_r2c_2_odd(
    __gm__ uint8_t *__restrict__ ffts_addr,
    __gm__ float *__restrict__ gm_input,
    __gm__ uint32_t *__restrict__ gm_input_index,
    __gm__ float *__restrict__ gm_a,
    __gm__ float *__restrict__ gm_b,
    __gm__ uint32_t *__restrict__ gm_output_index,
    __gm__ float *__restrict__ gm_dft_matrix_array,
    __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list,
    __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace,
    __gm__ uint8_t *__restrict__ gm_tiling_para) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_r2c_mix_cube_by_batch(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                     gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 1);
}
#elif __DAV_C220_VEC__
{
    common_fft_r2c_mix_vector_by_batch<2>(ffts_addr, gm_input, gm_input_index, gm_a, gm_b,
                                           gm_output_index, gm_dft_matrix_array, gm_tw_matrix_array,
                                           gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 1);
}
#endif
}

extern "C" __global__ __aicore__ void fft_r2c_3_odd(
    __gm__ uint8_t *__restrict__ ffts_addr,
    __gm__ float *__restrict__ gm_input,
    __gm__ uint32_t *__restrict__ gm_input_index,
    __gm__ float *__restrict__ gm_a,
    __gm__ float *__restrict__ gm_b,
    __gm__ uint32_t *__restrict__ gm_output_index,
    __gm__ float *__restrict__ gm_dft_matrix_array,
    __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list,
    __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace,
    __gm__ uint8_t *__restrict__ gm_tiling_para) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_r2c_mix_cube_by_batch(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                     gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 1);
}
#elif __DAV_C220_VEC__
{
    common_fft_r2c_mix_vector_by_batch<3>(ffts_addr, gm_input, gm_input_index, gm_a, gm_b,
                                           gm_output_index, gm_dft_matrix_array, gm_tw_matrix_array,
                                           gm_radix_list, gm_output, gm_workspace, gm_tiling_para, 1);
}
#endif
}

} // namespace FFT1DR2CFftKernel

#endif // RFFT1_D_R2C_FFT_KERNEL_H
