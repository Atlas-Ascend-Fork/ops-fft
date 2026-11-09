/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPSFFT_COMMON_KERNEL_FFT_R2C_COMMON_H
#define OPSFFT_COMMON_KERNEL_FFT_R2C_COMMON_H

#include "kernel_operator.h"

#include "fft_all_common.h"

#include "fft_r2c_c2r_common.h"

__aicore__ __inline__ void common_fft_r2c_mix_cube_by_batch(
    __gm__ uint8_t * __restrict__ ffts_addr,
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ int32_t * __restrict__ gm_radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para,
    int32_t odd,
    int64_t max_floats = (1 << 21)
)
{
    Init4Cube(ffts_addr);

    OpsFft::FftAllMixTilingData tiling_data;
    InitTilingData(gm_tiling_para, &tiling_data);

    AscendC::GlobalTensor<float> input_gm_tensor;
    AscendC::GlobalTensor<float> dft_matrix_array_gm_tensor;
    AscendC::GlobalTensor<float> tw_matrix_array_gm_tensor;
    AscendC::GlobalTensor<int32_t> radix_list_gm_tensor;
    AscendC::GlobalTensor<float> output_gm_tensor;

    input_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    dft_matrix_array_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_dft_matrix_array));
    tw_matrix_array_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_tw_matrix_array));
    radix_list_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_radix_list));
    output_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    AscendC::GlobalTensor<float> workspace_input_gm_tensor;
    AscendC::GlobalTensor<float> workspace_output_gm_tensor;
    AscendC::GlobalTensor<float> workspace_sync_gm_tensor;
    AscendC::GlobalTensor<float> auxil_gm_tensor;

    workspace_input_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[0]));
    workspace_output_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[1]));
    workspace_sync_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[2]));
    auxil_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[4]));

    int64_t n_fft_actual = odd == 1 ? tiling_data.fftN : tiling_data.fftN * 2;
    // 让每次处理的数据小于 2 ^ 21 次方
    int64_t L2_CACHE_MAX_ELES = max_floats;
    int64_t n_batch_in_loop = (L2_CACHE_MAX_ELES + tiling_data.fftN - 1) / tiling_data.fftN;
    int64_t batch_loop = (tiling_data.batchSize + n_batch_in_loop - 1) / n_batch_in_loop;
    int64_t batch_remain = tiling_data.batchSize % n_batch_in_loop;

    if (batch_loop == 1 && batch_remain > 0) {
        n_batch_in_loop = batch_remain;
    }

    if (tiling_data.fftN > L2_CACHE_MAX_ELES * 5) {
        n_batch_in_loop = tiling_data.batchSize;
        batch_loop = 1;
        batch_remain = tiling_data.batchSize;
    }

    int64_t input_eles_one_loop = n_batch_in_loop * n_fft_actual;
    int64_t output_eles_one_loop = n_batch_in_loop * (n_fft_actual / 2 + 1) * 2 * sizeof(float) / sizeof(float);

    for (int64_t batch_index = 0; batch_index < batch_loop; batch_index++) {

        int64_t batch_actual = n_batch_in_loop;
        if (batch_index == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        AscendC::GlobalTensor<float> now_input_gm_tensor = input_gm_tensor[batch_index * input_eles_one_loop];
        AscendC::GlobalTensor<float> now_output_gm_tensor = output_gm_tensor[batch_index * output_eles_one_loop];

        stock_fft_mix_aic(
            now_input_gm_tensor.GetPhyAddr(0),
            dft_matrix_array_gm_tensor.GetPhyAddr(0),
            tw_matrix_array_gm_tensor.GetPhyAddr(0),
            workspace_input_gm_tensor.GetPhyAddr(0),
            workspace_output_gm_tensor.GetPhyAddr(0),
            workspace_sync_gm_tensor.GetPhyAddr(0),
            radix_list_gm_tensor.GetPhyAddr(0),
            now_output_gm_tensor.GetPhyAddr(0),
            auxil_gm_tensor.GetPhyAddr(0),
            batch_actual,
            tiling_data.fftN,
            tiling_data.radixListLen,
            tiling_data.isInverse,
            max_floats
        );
    }

}

template<int32_t aiv_split_way>
__aicore__ __inline__ void common_fft_r2c_mix_vector_by_batch(
    __gm__ uint8_t * __restrict__ ffts_addr,
    __gm__ float * __restrict__ gm_input,
    __gm__ uint32_t * __restrict__ gm_input_index,
    __gm__ float * __restrict__ gm_a,
    __gm__ float * __restrict__ gm_b,
    __gm__ uint32_t * __restrict__ gm_output_index,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ int32_t * __restrict__ gm_radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para,
    int32_t odd
)
{
    Init4Vector(ffts_addr);

    OpsFft::FftAllMixTilingData tiling_data;
    InitTilingData(gm_tiling_para, &tiling_data);

    AscendC::GlobalTensor<float> input_gm_tensor;
    AscendC::GlobalTensor<float> dft_matrix_array_gm_tensor;
    AscendC::GlobalTensor<float> tw_matrix_array_gm_tensor;
    AscendC::GlobalTensor<int32_t> radix_list_gm_tensor;
    AscendC::GlobalTensor<float> output_gm_tensor;
    AscendC::GlobalTensor<uint32_t> input_index_gm_tensor;
    AscendC::GlobalTensor<uint32_t> output_index_gm_tensor;
    AscendC::GlobalTensor<float> a_gm_tensor;
    AscendC::GlobalTensor<float> b_gm_tensor;

    input_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    dft_matrix_array_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_dft_matrix_array));
    tw_matrix_array_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_tw_matrix_array));
    radix_list_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_radix_list));
    output_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));
    input_index_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gm_input_index));
    output_index_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gm_output_index));
    a_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
    b_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));

    AscendC::GlobalTensor<float> workspace_input_gm_tensor;
    AscendC::GlobalTensor<float> workspace_output_gm_tensor;
    AscendC::GlobalTensor<float> workspace_sync_gm_tensor;
    AscendC::GlobalTensor<float> c2c_output_gm_tensor;
    AscendC::GlobalTensor<float> auxil_gm_tensor;

    workspace_input_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[0]));
    workspace_output_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[1]));
    workspace_sync_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[2]));
    c2c_output_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[3]));
    auxil_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[4]));

    int64_t n_fft_actual = odd == 1 ? tiling_data.fftN : tiling_data.fftN * 2;
    // 让每次处理的数据小于 2 ^ 21 次方
    int64_t L2_CACHE_MAX_ELES = (1 << 21);
    int64_t n_batch_in_loop = (L2_CACHE_MAX_ELES + tiling_data.fftN - 1) / tiling_data.fftN;
    int64_t batch_loop = (tiling_data.batchSize + n_batch_in_loop - 1) / n_batch_in_loop;
    int64_t batch_remain = tiling_data.batchSize % n_batch_in_loop;

    if (batch_loop == 1 && batch_remain > 0) {
        n_batch_in_loop = batch_remain;
    }

    if (tiling_data.fftN > L2_CACHE_MAX_ELES * 5) {
        n_batch_in_loop = tiling_data.batchSize;
        batch_loop = 1;
        batch_remain = tiling_data.batchSize;
    }

    int64_t input_eles_one_loop = n_batch_in_loop * n_fft_actual;
    int64_t output_eles_one_loop = n_batch_in_loop * (n_fft_actual / 2 + 1) * 2 * sizeof(float) / sizeof(float);

    for (int64_t batch_index = 0; batch_index < batch_loop; batch_index++) {

        int64_t batch_actual = n_batch_in_loop;
        if (batch_index == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        AscendC::GlobalTensor<float> now_input_gm_tensor = input_gm_tensor[batch_index * input_eles_one_loop];
        AscendC::GlobalTensor<float> now_output_gm_tensor = output_gm_tensor[batch_index * output_eles_one_loop];

        stock_fft_mix_aiv<false, aiv_split_way>(
            now_input_gm_tensor.GetPhyAddr(0),
            dft_matrix_array_gm_tensor.GetPhyAddr(0),
            tw_matrix_array_gm_tensor.GetPhyAddr(0),
            workspace_input_gm_tensor.GetPhyAddr(0),
            workspace_output_gm_tensor.GetPhyAddr(0),
            workspace_sync_gm_tensor.GetPhyAddr(0),
            radix_list_gm_tensor.GetPhyAddr(0),
            c2c_output_gm_tensor.GetPhyAddr(0),
            auxil_gm_tensor.GetPhyAddr(0),
            batch_actual,
            tiling_data.fftN,
            tiling_data.radixListLen,
            tiling_data.isInverse,
            0,
            odd
        );

        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(1);
        AscendC::CrossCoreWaitFlag(1);

        // by-batch 奇数档模板仅被 _odd 变体实例化且三处调用均显式传 odd=1，
        // 原 if/else 的 else 调 R2C（基础档实现）不可达（issue #323）：收敛为
        // 仅保留 live 调用，消除死/活双实现诱导修复错配
        AscendC::PipeBarrier<PIPE_ALL>();
        R2C_odd(c2c_output_gm_tensor.GetPhyAddr(0), input_index_gm_tensor.GetPhyAddr(0), a_gm_tensor.GetPhyAddr(0), b_gm_tensor.GetPhyAddr(0), output_index_gm_tensor.GetPhyAddr(0), now_output_gm_tensor.GetPhyAddr(0), tiling_data.fftN, batch_actual, 0, batch_actual);

        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(1);
        AscendC::CrossCoreWaitFlag(1);
    }

}

template<int32_t aiv_split_way>
__aicore__ __inline__ void common_fft_r2c_mix_even_vector_by_batch(
    __gm__ uint8_t * __restrict__ ffts_addr,
    __gm__ float * __restrict__ gm_input,
    __gm__ uint32_t * __restrict__ gm_input_index,
    __gm__ float * __restrict__ gm_a,
    __gm__ float * __restrict__ gm_b,
    __gm__ uint32_t * __restrict__ gm_output_index,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ int32_t * __restrict__ gm_radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para,
    int64_t max_floats = (1 << 21)
)
{
    Init4Vector(ffts_addr);

    OpsFft::FftAllMixTilingData tiling_data;
    InitTilingData(gm_tiling_para, &tiling_data);

    AscendC::GlobalTensor<float> input_gm_tensor;
    AscendC::GlobalTensor<float> dft_matrix_array_gm_tensor;
    AscendC::GlobalTensor<float> tw_matrix_array_gm_tensor;
    AscendC::GlobalTensor<int32_t> radix_list_gm_tensor;
    AscendC::GlobalTensor<float> output_gm_tensor;
    AscendC::GlobalTensor<uint32_t> input_index_gm_tensor;
    AscendC::GlobalTensor<uint32_t> output_index_gm_tensor;
    AscendC::GlobalTensor<float> a_gm_tensor;
    AscendC::GlobalTensor<float> b_gm_tensor;

    input_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    dft_matrix_array_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_dft_matrix_array));
    tw_matrix_array_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_tw_matrix_array));
    radix_list_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_radix_list));
    output_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));
    input_index_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gm_input_index));
    output_index_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gm_output_index));
    a_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
    b_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));

    AscendC::GlobalTensor<float> workspace_input_gm_tensor;
    AscendC::GlobalTensor<float> workspace_output_gm_tensor;
    AscendC::GlobalTensor<float> workspace_sync_gm_tensor;
    AscendC::GlobalTensor<float> c2c_output_gm_tensor;
    AscendC::GlobalTensor<float> auxil_gm_tensor;

    workspace_input_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[0]));
    workspace_output_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[1]));
    workspace_sync_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[2]));
    c2c_output_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[3]));
    auxil_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reinterpret_cast<__gm__ uint8_t *>(gm_workspace) + tiling_data.workspaceOffsets[4]));

    int64_t n_fft_actual = tiling_data.fftN * 2;
    // 让每次处理的数据小于 2 ^ 21 次方
    int64_t L2_CACHE_MAX_ELES = max_floats;
    int64_t n_batch_in_loop = (L2_CACHE_MAX_ELES + tiling_data.fftN - 1) / tiling_data.fftN;
    int64_t batch_loop = (tiling_data.batchSize + n_batch_in_loop - 1) / n_batch_in_loop;
    int64_t batch_remain = tiling_data.batchSize % n_batch_in_loop;

    if (batch_loop == 1 && batch_remain > 0) {
        n_batch_in_loop = batch_remain;
    }

    if (tiling_data.fftN > L2_CACHE_MAX_ELES * 5) {
        n_batch_in_loop = tiling_data.batchSize;
        batch_loop = 1;
        batch_remain = tiling_data.batchSize;
    }

    int64_t input_eles_one_loop = n_batch_in_loop * n_fft_actual;
    int64_t output_eles_one_loop = n_batch_in_loop * (n_fft_actual / 2 + 1) * 2 * sizeof(float) / sizeof(float);

    for (int64_t batch_index = 0; batch_index < batch_loop; batch_index++) {

        int64_t batch_actual = n_batch_in_loop;
        if (batch_index == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        AscendC::GlobalTensor<float> now_input_gm_tensor = input_gm_tensor[batch_index * input_eles_one_loop];
        AscendC::GlobalTensor<float> now_output_gm_tensor = output_gm_tensor[batch_index * output_eles_one_loop];

        stock_fft_r2c_mix_even_aiv<false, aiv_split_way>(
            now_input_gm_tensor.GetPhyAddr(0),
            dft_matrix_array_gm_tensor.GetPhyAddr(0),
            tw_matrix_array_gm_tensor.GetPhyAddr(0),
            workspace_input_gm_tensor.GetPhyAddr(0),
            workspace_output_gm_tensor.GetPhyAddr(0),
            workspace_sync_gm_tensor.GetPhyAddr(0),
            radix_list_gm_tensor.GetPhyAddr(0),
            c2c_output_gm_tensor.GetPhyAddr(0),
            auxil_gm_tensor.GetPhyAddr(0),
            batch_actual,
            tiling_data.fftN,
            tiling_data.radixListLen,
            tiling_data.isInverse,
            max_floats
        );

        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(1);
        AscendC::CrossCoreWaitFlag(1);

        AscendC::PipeBarrier<PIPE_ALL>();
        R2C_even(c2c_output_gm_tensor.GetPhyAddr(0), input_index_gm_tensor.GetPhyAddr(0), a_gm_tensor.GetPhyAddr(0), gm_b, output_index_gm_tensor.GetPhyAddr(0), now_output_gm_tensor.GetPhyAddr(0), tiling_data.fftN * 2, batch_actual, 0, batch_actual);

        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(1);
        AscendC::CrossCoreWaitFlag(1);

        AscendC::PipeBarrier<PIPE_ALL>();
        R2C_even_tail_calc(c2c_output_gm_tensor.GetPhyAddr(0), now_output_gm_tensor.GetPhyAddr(0), batch_actual, tiling_data.fftN);

        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(1);
        AscendC::CrossCoreWaitFlag(1);
    }

}

#endif
