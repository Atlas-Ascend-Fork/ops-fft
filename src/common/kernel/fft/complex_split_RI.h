/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPSFFT_COMMON_KERNEL_FFT_COMPLEX_SPLIT_RI_H
#define OPSFFT_COMMON_KERNEL_FFT_COMPLEX_SPLIT_RI_H

#include "kernel_operator.h"
#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "mem.h"
#include "utils.h"

// 将（16, row_num) 或者(row_num_padding, 16) 转置，配合UB存放的指针数据来使用
__aicore__ __inline__ __attribute__((overloadable, always_inline)) void transpose_matrix(__ubuf__ uint16_t *output_buf,
                                                                                         __ubuf__ uint16_t *input_buf,
                                                                                         uint8_t repeat_num,
                                                                                         int32_t dstStride = 16,
                                                                                         int32_t srcStride = 16)
{
    vld_va_reg(VA0, (__ubuf__ uint64_t *)input_buf, L128);
    vld_va_reg(VA1, (__ubuf__ uint64_t *)input_buf, H128);
    vld_va_reg(VA2, (__ubuf__ uint64_t *)(output_buf), L128);
    vld_va_reg(VA3, (__ubuf__ uint64_t *)(output_buf), H128);
    PIPE_BARRIER(V);
    scatter_vnchwconv_b16(VA2, VA0, repeat_num, dstStride, srcStride);
}
__aicore__ __inline__ __attribute__((overloadable, always_inline)) void complex_split_RI_tile_zN(
    __gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_output, __gm__ float *__restrict__ gm_auxil,
    int64_t batchSize, int64_t N1, int64_t N2, int64_t N2_padding, int32_t tile_K0, int32_t tile_N0, int32_t step_len)
{
    // 除0整改
    if (tile_N0 == 0) {
        tile_N0 = 1;
    }
    const int64_t N1_round = ROUND(N1, C0_SIZE);
    const int64_t N2_round = ROUND(N2, C0_SIZE);

    const int16_t aic_id = get_block_idx();
    const int16_t aiv_id_per_group = AscendC::GetSubBlockIdx();
    const int16_t group_num = AscendC::GetBlockNum();
    const int16_t subgroup_num = get_subblockdim();
    const int16_t aiv_id = aic_id * subgroup_num + aiv_id_per_group;
    const int16_t aiv_num = group_num * subgroup_num;

    const int32_t max_num = 24 * 1024 / sizeof(float);

    AscendC::GlobalTensor<float> gm_input_tensor, gm_output_tensor;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    int32_t now_UB = 0;
    auto buf1_input = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf1_input_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf1_output_real = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf1_output_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    auto buf2_input = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf2_input_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf2_output_real = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf2_output_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    int32_t N2_max_num = max_num;
    int32_t N1_max_num = N2_max_num / ROUND(N2, tile_N0);
    N1_max_num = N1_max_num + (N1_max_num == 0);

    int32_t batch_max_num;

    N1_max_num = MIN(N1_max_num, N1);
    // 除0整改
    if (N1_max_num == 0) {
        N1_max_num = 1;
    }
    N2_max_num = MIN(N2_max_num, ROUND(N2, tile_N0));
    // 除0整改
    if (N2_max_num == 0) {
        N2_max_num = 1;
    }

    batch_max_num = max_num / (N1_max_num * N2_max_num);
    batch_max_num += (batch_max_num == 0);

    int64_t N1_loop = (N1 + N1_max_num - 1) / N1_max_num;
    // 除0整改
    if (N1_loop == 0) {
        N1_loop = 1;
    }
    int64_t N2_loop = (N2 + N2_max_num - 1) / N2_max_num;
    // 除0整改
    if (N2_loop == 0) {
        N2_loop = 1;
    }
    const int64_t batch_loop = (batchSize + batch_max_num - 1) / batch_max_num;
    const int64_t batch_remain = batchSize % batch_max_num;
    const int64_t N2_remain = N2 % N2_max_num;
    const int64_t N1_remain = N1 % N1_max_num;
    const int64_t loop = batch_loop * N1_loop * N2_loop;

    bool flag = 1;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (int64_t i = 0; i < loop; i++) {
        int64_t loop_idx = i;
        if (loop_idx % static_cast<int64_t>(aiv_num) != static_cast<int64_t>(aiv_id)) {
            continue;
        }

        int64_t N2_block_idx = loop_idx % N2_loop;
        int64_t batch_N1_idx = loop_idx / N2_loop;
        int64_t N1_idx = batch_N1_idx % N1_loop;
        int64_t batch_idx = batch_N1_idx / N1_loop;

        int32_t N2_actual = N2_max_num;
        if (N2_block_idx == N2_loop - 1 && N2_remain > 0) {
            N2_actual = N2_remain;
        }

        int32_t N1_actual = N1_max_num;
        if (N1_idx == N1_loop - 1 && N1_remain > 0) {
            N1_actual = N1_remain;
        }

        int32_t batch_actual = batch_max_num;
        if (batch_idx == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        int32_t N2_actual_round = ROUND(N2_actual, tile_N0);

        auto buf_input = flag ? buf1_input : buf2_input;
        auto buf_input_imag = flag ? buf1_input_imag : buf2_input_imag;
        auto buf_output_real = flag ? buf1_output_real : buf2_output_real;
        auto buf_output_imag = flag ? buf1_output_imag : buf2_output_imag;
        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;
        WAIT_FLAG(MTE3, MTE2, event_id);
        // 按照不同shape来合并访存
        if (N1_actual == N1 && N2_actual == N2 && N2 <= tile_N0) {
            const int64_t data_input_index =
                batch_idx * batch_max_num * N1 * N2 + N1_idx * N1_max_num * N2 + N2_block_idx * N2_max_num;
            copy_gm2ubuf(buf_input, gm_input_tensor[data_input_index * 2], batch_actual * N1_actual, ROUND(2 * N2_actual, 8), 2 * N2,
                         2 * N2_actual_round);

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);
            const int64_t data_output_index = batch_idx * batch_max_num * 2 * N1 * N2_padding +
                                              N1_idx * N1_max_num * tile_N0 + N2_block_idx * 2 * N1 * N2_max_num;
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_real.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                      (batch_actual * N1_actual * 2 * N2_actual_round + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 8, 8);
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_imag.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                      (batch_actual * N1_actual * 2 * N2_actual_round + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 2, 8, 8);
            PIPE_BARRIER(V);
            AscendC::DataCopy(buf_input, buf_output_real,
                AscendC::DataCopyParams(
                    batch_actual,
                    N1_actual * N2_actual_round / C0_SIZE,
                    0,
                    (2 * N1_actual * N2_actual_round - N1_actual * N2_actual_round) / C0_SIZE
                )
            );
            AscendC::DataCopy(buf_input[N1_actual * N2_actual_round], buf_output_imag,
                AscendC::DataCopyParams(
                    batch_actual,
                    N1_actual * N2_actual_round / C0_SIZE,
                    0,
                     (2 * N1_actual * N2_actual_round - N1_actual * N2_actual_round) / C0_SIZE
                )
            );

            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);
            copy_ubuf2gm(gm_output_tensor[data_output_index], buf_input, batch_actual, 2 * N1_actual * tile_N0, 2 * N1_actual * tile_N0,
                         2 * N1 * tile_N0);
        } else if (batch_max_num == 1 && N2_actual == N2 && N2 <= tile_N0) {
            const int64_t data_input_index =
                batch_idx * batch_max_num * N1 * N2 + N1_idx * N1_max_num * N2 + N2_block_idx * N2_max_num;
            copy_gm2ubuf(buf_input, gm_input_tensor[data_input_index * 2], batch_actual * N1_actual, ROUND(2 * N2_actual, 8), 2 * N2,
                         2 * N2_actual_round);

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);
            const int64_t data_output_index = batch_idx * batch_max_num * 2 * N1 * N2_padding +
                                              N1_idx * N1_max_num * tile_N0 + N2_block_idx * 2 * N1 * N2_max_num;
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_real.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                      (batch_actual * N1_actual * 2 * N2_actual_round + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 8, 8);
            SET_FLAG(V, MTE3, event_id);
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_imag.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                      (batch_actual * N1_actual * 2 * N2_actual_round + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 2, 8, 8);
            SET_FLAG(V, MTE3, event_id + 2);
            WAIT_FLAG(V, MTE3, event_id);
            copy_ubuf2gm(gm_output_tensor[data_output_index], buf_output_real, N1_actual, tile_N0, tile_N0, tile_N0);
            WAIT_FLAG(V, MTE3, event_id + 2);
            copy_ubuf2gm(gm_output_tensor[data_output_index + N1 * tile_N0], buf_output_imag, N1_actual, tile_N0, tile_N0, tile_N0);
        } else if (batch_max_num == 1 && N1_max_num == 1) {
            const int64_t data_input_index =
                batch_idx * batch_max_num * N1 * N2 + N1_idx * N1_max_num * N2 + N2_block_idx * N2_max_num;

            copy_gm2ubuf(buf_input, gm_input_tensor[data_input_index * 2], batch_actual * N1_actual, ROUND(2 * N2_actual, 8), 2 * N2,
                         2 * N2_actual_round);

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);
            const int64_t data_output_index = batch_idx * batch_max_num * 2 * N1 * N2_padding +
                                              N1_idx * N1_max_num * tile_N0 + N2_block_idx * 2 * N1 * N2_max_num;
            auto gm_output_real_ptr = gm_output_tensor[data_output_index];
            auto gm_output_imag_ptr = gm_output_tensor[data_output_index + N1 * tile_N0];
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_real.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                      (batch_actual * N1_actual * 2 * N2_actual_round + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 8, 8);
            SET_FLAG(V, MTE3, event_id);
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_imag.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                      (batch_actual * N1_actual * 2 * N2_actual_round + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 2, 8, 8);
            SET_FLAG(V, MTE3, event_id + 2);
            WAIT_FLAG(V, MTE3, event_id);
            AscendC::DataCopyPad(gm_output_real_ptr, buf_output_real,
                AscendC::DataCopyExtParams(N2_actual_round / tile_N0, tile_N0 * sizeof(float),
                                           0, (2 * N1 * tile_N0 - tile_N0) * sizeof(float), 0));
            WAIT_FLAG(V, MTE3, event_id + 2);
            AscendC::DataCopyPad(gm_output_imag_ptr, buf_output_imag,
                AscendC::DataCopyExtParams(N2_actual_round / tile_N0, tile_N0 * sizeof(float),
                                           0, (2 * N1 * tile_N0 - tile_N0) * sizeof(float), 0));
        } else {
            const int64_t data_input_index =
                batch_idx * batch_max_num * N1 * N2 + N1_idx * N1_max_num * N2 + N2_block_idx * N2_max_num;

            copy_gm2ubuf(buf_input, gm_input_tensor[data_input_index * 2], batch_actual * N1_actual, ROUND(2 * N2_actual, 8), 2 * N2,
                         2 * N2_actual_round);

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);
            const int64_t data_output_index = batch_idx * batch_max_num * 2 * N1 * N2_padding +
                                              N1_idx * N1_max_num * tile_N0 + N2_block_idx * 2 * N1 * N2_max_num;
            auto gm_output_real_ptr = gm_output_tensor[data_output_index];
            auto gm_output_imag_ptr = gm_output_tensor[data_output_index + N1 * tile_N0];
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_real.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                      (batch_actual * N1_actual * 2 * N2_actual_round + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 8, 8);
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_imag.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                      (batch_actual * N1_actual * 2 * N2_actual_round + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 2, 8, 8);
            PIPE_BARRIER(V);
            for (int32_t j = 0; j < batch_actual; j++) {
                for (int32_t i = 0; i < N1_actual; i++) {
                    AscendC::DataCopy(buf_input[j * 2 * N1_actual * N2_actual_round + i * tile_N0],
                                      buf_output_real[j * N1_actual * N2_actual_round + i * N2_actual_round],
                        AscendC::DataCopyParams(
                            N2_actual_round / tile_N0,
                            tile_N0 / C0_SIZE,
                            0,
                            (N1_actual * tile_N0 - tile_N0) / C0_SIZE
                        )
                    );
                    AscendC::DataCopy(buf_input[j * 2 * N1_actual * N2_actual_round + N1_actual * N2_actual_round + i * tile_N0],
                                      buf_output_imag[j * N1_actual * N2_actual_round + i * N2_actual_round],
                        AscendC::DataCopyParams(
                            N2_actual_round / tile_N0,
                            tile_N0 / C0_SIZE,
                            0,
                            (N1_actual * tile_N0 - tile_N0) / C0_SIZE
                        )
                    );
                }
            }

            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);
            for (int32_t j = 0; j < batch_actual; j++) {
                AscendC::DataCopyPad(gm_output_real_ptr[j * 2 * N1 * N2_padding], buf_input[j * 2 * N1_actual * N2_actual_round],
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, N1_actual * tile_N0 * sizeof(float),
                                               0, (2 * N1 * tile_N0 - N1_actual * tile_N0) * sizeof(float), 0));
                AscendC::DataCopyPad(gm_output_imag_ptr[j * 2 * N1 * N2_padding],
                                     buf_input[j * 2 * N1_actual * N2_actual_round + N1_actual * N2_actual_round],
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, N1_actual * tile_N0 * sizeof(float),
                                            0, (2 * N1 * tile_N0 - N1_actual * tile_N0) * sizeof(float), 0));
            }
        }
        SET_FLAG(MTE3, MTE2, event_id);
        flag = 1 - flag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

__aicore__ __inline__ __attribute__((overloadable, always_inline)) void c2r_even_complex_split_RI_tile_zN(
    __gm__ float * __restrict__ gm_input_real,
    __gm__ float * __restrict__ gm_input_imag,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_auxil,
    int64_t batchSize,
    int32_t N1,
    int64_t N2,
    int64_t N2_padding,
    int32_t tile_K0,
    int32_t tile_N0,
    int32_t step_len
) {

    const int32_t N1_round = ROUND(N1, C0_SIZE);
    const int64_t N2_round = ROUND(N2, C0_SIZE);

    const int16_t aic_id = get_block_idx();
    const int16_t aiv_id_per_group = AscendC::GetSubBlockIdx();
    const int16_t group_num = AscendC::GetBlockNum();
    const int16_t subgroup_num = get_subblockdim();
    const int16_t aiv_id = aic_id * subgroup_num + aiv_id_per_group;
    const int16_t aiv_num = group_num * subgroup_num;

    const int32_t max_num = 24 * 1024 / sizeof(float);

    AscendC::GlobalTensor<float> gm_input_real_tensor, gm_output_tensor, gm_input_imag_tensor;
    gm_input_real_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input_real));
    gm_input_imag_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input_imag));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    int32_t now_UB = 0;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto buf1_input = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf1_input_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf1_output_real = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf1_output_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    auto buf2_input = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf2_input_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf2_output_real = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf2_output_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    int32_t N2_max_num = max_num;
    int32_t N1_max_num = N2_max_num / ROUND(N2, tile_N0);
    N1_max_num = N1_max_num + (N1_max_num == 0);

    int32_t batch_max_num;

    N1_max_num = MIN(N1_max_num, N1);
    N2_max_num = MIN(N2_max_num, ROUND(N2, tile_N0));

    batch_max_num = max_num / (N1_max_num * N2_max_num);
    batch_max_num += (batch_max_num == 0);

    const int32_t N1_loop = (N1 + N1_max_num - 1) / N1_max_num;
    const int64_t N2_loop = (N2 + N2_max_num - 1) / N2_max_num;
    const int64_t batch_loop = (batchSize + batch_max_num - 1) / batch_max_num;
    const int64_t batch_remain = batchSize % batch_max_num;
    const int64_t N2_remain = N2 % N2_max_num;
    const int32_t N1_remain = N1 % N1_max_num;
    const int64_t loop = batch_loop * N1_loop * N2_loop;

    bool flag = 1;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (int64_t i = 0; i < loop; i++) {
        int64_t loop_idx = i;
        if (loop_idx % aiv_num != aiv_id) {
            continue;
        }

        int64_t N2_block_idx = loop_idx % N2_loop;
        int64_t batch_N1_idx = loop_idx / N2_loop;
        int32_t N1_idx = batch_N1_idx % N1_loop;
        int64_t batch_idx = batch_N1_idx / N1_loop;

        int32_t N2_actual = N2_max_num;
        if (N2_block_idx == N2_loop - 1 && N2_remain > 0) {
            N2_actual = N2_remain;
        }

        int32_t N1_actual = N1_max_num;
        if (N1_idx == N1_loop - 1 && N1_remain > 0) {
            N1_actual = N1_remain;
        }

        int32_t batch_actual = batch_max_num;
        if (batch_idx == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        int32_t N2_actual_round = ROUND(N2_actual, tile_N0);

        auto buf_input = flag ? buf1_input : buf2_input;
        auto buf_input_imag = flag ? buf1_input_imag : buf2_input_imag;
        auto buf_output_real = flag ? buf1_output_real : buf2_output_real;
        auto buf_output_imag = flag ? buf1_output_imag : buf2_output_imag;
        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;
        WAIT_FLAG(MTE3, MTE2, event_id);

        const int64_t data_input_index = batch_idx * batch_max_num * N1 * N2 + N1_idx * N1_max_num * N2 + N2_block_idx * N2_max_num;

        copy_gm2ubuf(buf_output_real, gm_input_real_tensor[data_input_index], batch_actual * N1_actual, ROUND(N2_actual, 8), N2, N2_actual_round);
        copy_gm2ubuf(buf_output_imag, gm_input_imag_tensor[data_input_index], batch_actual * N1_actual, ROUND(N2_actual, 8), N2, N2_actual_round);

        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);
        // compute output index
        const int64_t data_output_index = batch_idx * batch_max_num * 2 * N1 * N2_padding + N1_idx * N1_max_num * tile_N0 + N2_block_idx * 2 * N1 * N2_max_num;
        auto gm_current_output_real = gm_output_tensor[data_output_index];
        auto gm_current_output_imag = gm_output_tensor[data_output_index + N1 * tile_N0];

        // 按照不同shape来合并访存
        if (N1_actual == N1 && N2_actual == N2 && N2 <= tile_N0) {
            AscendC::DataCopy(buf_input, buf_output_real,
                AscendC::DataCopyParams(
                    batch_actual,
                    N1_actual * N2_actual_round / C0_SIZE,
                    0,
                    (2 * N1_actual * N2_actual_round - N1_actual * N2_actual_round) / C0_SIZE
                )
            );
            AscendC::DataCopy(buf_input[N1_actual * N2_actual_round], buf_output_imag,
                AscendC::DataCopyParams(
                    batch_actual,
                    N1_actual * N2_actual_round / C0_SIZE,
                    0,
                    (2 * N1_actual * N2_actual_round - N1_actual * N2_actual_round) / C0_SIZE
                )
            );

            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);
            copy_ubuf2gm(gm_current_output_real, buf_input, batch_actual, 2 * N1_actual * tile_N0, 2 * N1_actual * tile_N0, 2 * N1 * tile_N0);
        } else if (batch_max_num == 1 && N2_actual == N2 && N2 <= tile_N0) {

            SET_FLAG(V, MTE3, event_id);

            SET_FLAG(V, MTE3, event_id + 2);
            WAIT_FLAG(V, MTE3, event_id);
            copy_ubuf2gm(gm_current_output_real, buf_output_real, N1_actual, tile_N0, tile_N0, tile_N0);
            WAIT_FLAG(V, MTE3, event_id + 2);
            copy_ubuf2gm(gm_current_output_imag, buf_output_imag, N1_actual, tile_N0, tile_N0, tile_N0);
        } else if (batch_max_num == 1 && N1_max_num == 1) {

            SET_FLAG(V, MTE3, event_id);

            SET_FLAG(V, MTE3, event_id + 2);
            WAIT_FLAG(V, MTE3, event_id);
            AscendC::DataCopyPad(gm_current_output_real, buf_output_real,
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, tile_N0 * sizeof(float),
                                               0, (2 * N1 * tile_N0 - tile_N0) * sizeof(float), 0));
            WAIT_FLAG(V, MTE3, event_id + 2);
            AscendC::DataCopyPad(gm_current_output_imag, buf_output_imag,
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, tile_N0 * sizeof(float),
                                               0, (2 * N1 * tile_N0 - tile_N0) * sizeof(float), 0));
        } else {
            for (int j = 0; j < batch_actual; j++) {
                for (int i = 0; i < N1_actual; i++) {
                    AscendC::DataCopy(buf_input[j * 2 * N1_actual * N2_actual_round + i * tile_N0],
                                      buf_output_real[j * N1_actual * N2_actual_round + i * N2_actual_round],
                        AscendC::DataCopyParams(
                            N2_actual_round / tile_N0,
                            tile_N0 / C0_SIZE,
                            0,
                            (N1_actual * tile_N0 - tile_N0) / C0_SIZE
                        )
                    );
                    AscendC::DataCopy(buf_input[j * 2 * N1_actual * N2_actual_round + N1_actual * N2_actual_round + i * tile_N0],
                                      buf_output_imag[j * N1_actual * N2_actual_round + i * N2_actual_round],
                        AscendC::DataCopyParams(
                            N2_actual_round / tile_N0,
                            tile_N0 / C0_SIZE,
                            0,
                            (N1_actual * tile_N0 - tile_N0) / C0_SIZE
                        )
                    );
                }
            }

            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);
            for (int j = 0; j < batch_actual; j++) {
                AscendC::DataCopyPad(gm_current_output_real[j * 2 * N1 * N2_padding],
                                     buf_input[j * 2 * N1_actual * N2_actual_round],
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, N1_actual * tile_N0 * sizeof(float),
                                               0, (2 * N1 * tile_N0 - N1_actual * tile_N0) * sizeof(float), 0));
                AscendC::DataCopyPad(gm_current_output_imag[j * 2 * N1 * N2_padding],
                                     buf_input[j * 2 * N1_actual * N2_actual_round + N1_actual * N2_actual_round],
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, N1_actual * tile_N0 * sizeof(float),
                                               0, (2 * N1 * tile_N0 - N1_actual * tile_N0) * sizeof(float), 0));
            }
        }
        SET_FLAG(MTE3, MTE2, event_id);
        flag = 1 - flag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

// 对于 N2 较小的情况， 一次读取多个 N2， 在UB中填充成ROUND(N2, tile_N0)。并存入到全局内存
__aicore__ __inline__ __attribute__((overloadable, always_inline)) void complex_split_RI_tile_zN_vtranspose_load(
    __gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_output, __gm__ float *__restrict__ gm_auxil,
    int64_t batchSize, int64_t N1, int64_t N2, int64_t N2_padding, int32_t tile_K0, int32_t tile_N0, int32_t step_len)
{
    // 除0整改
    if (N2 == 0) {
        N2 = 1;
    }

    const int64_t N1_round = ROUND(N1, C0_SIZE);
    const int64_t N2_round = ROUND(N2, C0_SIZE);

    const int16_t aic_id = get_block_idx();
    const int16_t aiv_id_per_group = AscendC::GetSubBlockIdx();
    const int16_t group_num = AscendC::GetBlockNum();
    const int16_t subgroup_num = get_subblockdim();
    const int16_t aiv_id = aic_id * subgroup_num + aiv_id_per_group;
    const int16_t aiv_num = group_num * subgroup_num;

    AscendC::GlobalTensor<float> gm_input_tensor, gm_output_tensor;
    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    // 最大空间
    const int32_t max_num = 23 * 1024 / sizeof(float);

    int32_t now_UB = 0;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    // addr_buf 存放 scatter_vnchwconv_b16 需要的指针数组
    auto buf1_input_transpose1_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>((uintptr_t)now_UB);
    now_UB += 64;
    auto buf1_output_real_transpose1_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>((uintptr_t)now_UB);
    now_UB += 64;
    auto buf1_input_transpose2_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>((uintptr_t)now_UB);
    now_UB += 64;
    auto buf1_output_real_transpose2_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>((uintptr_t)now_UB);
    now_UB += 64;

    auto buf2_input_transpose1_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>((uintptr_t)now_UB);
    now_UB += 64;
    auto buf2_output_real_transpose1_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>((uintptr_t)now_UB);
    now_UB += 64;
    auto buf2_input_transpose2_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>((uintptr_t)now_UB);
    now_UB += 64;
    auto buf2_output_real_transpose2_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>((uintptr_t)now_UB);
    now_UB += 64;

    auto buf1_input = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf1_input_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    now_UB += C0_SIZE * 16 * sizeof(float);

    auto buf1_output_real = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf1_output_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    now_UB += C0_SIZE * 16 * sizeof(float);

    auto buf2_input = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf2_input_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    now_UB += C0_SIZE * 16 * sizeof(float);

    auto buf2_output_real = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf2_output_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    now_UB += C0_SIZE * 16 * sizeof(float);

    int32_t N2_max_num = max_num;
    int32_t N1_max_num = N2_max_num / ROUND(N2, tile_N0);
    N1_max_num = N1_max_num + (N1_max_num == 0);

    int32_t batch_max_num;

    N1_max_num = MIN(N1_max_num, N1);
    // 除0整改
    if (N1_max_num == 0) {
        N1_max_num = 1;
    }

    N2_max_num = MIN(N2_max_num, ROUND(N2, tile_N0));
    // 除0整改
    if (N2_max_num == 0) {
        N2_max_num = 1;
    }

    batch_max_num = max_num / (N1_max_num * N2_max_num);
    batch_max_num += (batch_max_num == 0);

    // 控制实部和虚部空间一起能读入8 * 16 个tile_N0, 这里默认 N2 < tile_N0
    if (N1_max_num == N1 && N2 <= tile_N0 && batch_max_num * N1_max_num >= 8 * C0_SIZE) {
        batch_max_num = batch_max_num * N1_max_num / (8 * C0_SIZE) * (8 * C0_SIZE) / N1_max_num;
    }

    // 如果读取的N2多于8 * 16 个，希望行宽尽量大。行数为16行。
    int32_t row_num = (2 * batch_max_num * N1_max_num + 16 * C0_SIZE - 1) / (16 * C0_SIZE) * C0_SIZE * N2;
    int32_t row_num_padding = (row_num / N2 * tile_N0);

    // 通过数据搬运和vector来减少scalar，下面的代码和上面注释的代码功能相同
    for (int32_t i = 0; i < 16; i++) {
        buf1_output_real_transpose1_addr_buf.SetValue(i, (uint16_t)(i));
    }

    SET_FLAG(S, V, EVENT_ID0);
    WAIT_FLAG(S, V, EVENT_ID0);

    AscendC::SetMaskCount();
    AscendC::SetVectorMask<int16_t>(0, 16);

    muls_v<ArchType::ASCEND_V220, int16_t>(buf1_input_transpose1_addr_buf.ReinterpretCast<int16_t>(),
                                           buf1_output_real_transpose1_addr_buf.ReinterpretCast<int16_t>(),
                                           row_num / C0_SIZE, 1, 1, 1, 8, 8);
    muls_v<ArchType::ASCEND_V220, int16_t>(buf1_output_real_transpose2_addr_buf.ReinterpretCast<int16_t>(),
                                           buf1_output_real_transpose1_addr_buf.ReinterpretCast<int16_t>(),
                                           (row_num_padding + C0_SIZE) / C0_SIZE, 1, 1, 1, 8, 8);
    AscendC::DataCopy(buf1_input_transpose2_addr_buf,
                      buf1_output_real_transpose1_addr_buf,
        AscendC::DataCopyParams(
            1,
            16 / C0_SIZE,
            0,
            0
        )
    );

    PIPE_BARRIER(V);
    adds_v<ArchType::ASCEND_V220, int16_t>(buf1_input_transpose1_addr_buf.ReinterpretCast<int16_t>(),
                                           buf1_input_transpose1_addr_buf.ReinterpretCast<int16_t>(),
                                           (reinterpret_cast<uintptr_t>(buf1_input.GetPhyAddr())) / 4 / 8,
                                           1, 1, 1, 8, 8);
    adds_v<ArchType::ASCEND_V220, int16_t>(buf1_output_real_transpose1_addr_buf.ReinterpretCast<int16_t>(),
                                           buf1_output_real_transpose1_addr_buf.ReinterpretCast<int16_t>(),
                                           (reinterpret_cast<uintptr_t>(buf1_output_real.GetPhyAddr())) / 4 / 8,
                                           1, 1, 1, 8, 8);
    adds_v<ArchType::ASCEND_V220, int16_t>(buf1_input_transpose2_addr_buf.ReinterpretCast<int16_t>(),
                                           buf1_input_transpose2_addr_buf.ReinterpretCast<int16_t>(),
                                           (reinterpret_cast<uintptr_t>(buf1_input.GetPhyAddr())) / 4 / 8,
                                           1, 1, 1, 8, 8);
    adds_v<ArchType::ASCEND_V220, int16_t>(buf1_output_real_transpose2_addr_buf.ReinterpretCast<int16_t>(),
                                           buf1_output_real_transpose2_addr_buf.ReinterpretCast<int16_t>(),
                                           (reinterpret_cast<uintptr_t>(buf1_output_real.GetPhyAddr())) / 4 / 8,
                                           1, 1, 1, 8, 8);

    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int16_t>((uint64_t)-1, (uint64_t)-1);

    PIPE_BARRIER(V);
    adds_v<ArchType::ASCEND_V220, int16_t>(buf2_input_transpose1_addr_buf.ReinterpretCast<int16_t>(),
                                           buf1_input_transpose1_addr_buf.ReinterpretCast<int16_t>(),
                                           (reinterpret_cast<uintptr_t>(buf2_input.GetPhyAddr())
                                               - reinterpret_cast<uintptr_t>(buf1_input.GetPhyAddr())) / 4 / 8,
                                           1, 1, 1, 8, 8);

    PIPE_BARRIER(V);

    const int64_t N1_loop = (N1 + N1_max_num - 1) / N1_max_num;
    const int64_t N2_loop = (N2 + N2_max_num - 1) / N2_max_num;
    const int64_t batch_loop = (batchSize + batch_max_num - 1) / batch_max_num;
    const int64_t batch_remain = batchSize % batch_max_num;
    const int64_t N2_remain = N2 % N2_max_num;
    const int64_t N1_remain = N1 % N1_max_num;
    const int64_t loop = batch_loop * N1_loop * N2_loop;

    bool flag = 1;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (int64_t i = 0; i < loop; i++) {
        int64_t loop_idx = i;
        if (loop_idx % static_cast<int64_t>(aiv_num) != static_cast<int64_t>(aiv_id)) {
            continue;
        }

        int64_t N2_block_idx = loop_idx % N2_loop;
        int64_t batch_N1_idx = loop_idx / N2_loop;
        int64_t N1_idx = batch_N1_idx % N1_loop;
        int64_t batch_idx = batch_N1_idx / N1_loop;

        int32_t N2_actual = N2_max_num;
        if (N2_block_idx == N2_loop - 1 && N2_remain > 0) {
            N2_actual = N2_remain;
        }

        int32_t N1_actual = N1_max_num;
        if (N1_idx == N1_loop - 1 && N1_remain > 0) {
            N1_actual = N1_remain;
        }

        int32_t batch_actual = batch_max_num;
        if (batch_idx == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        int32_t N2_actual_round = ROUND(N2_actual, tile_N0);

        auto buf_input = flag ? buf1_input : buf2_input;
        auto buf_input_imag = flag ? buf1_input_imag : buf2_input_imag;
        auto buf_output_real = flag ? buf1_output_real : buf2_output_real;
        auto buf_output_imag = flag ? buf1_output_imag : buf2_output_imag;
        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;

        auto buf_input_transpose1_addr_buf = flag ? buf1_input_transpose1_addr_buf : buf2_input_transpose1_addr_buf;
        auto buf_output_real_transpose1_addr_buf = flag ? buf1_output_real_transpose1_addr_buf :
                                                          buf2_output_real_transpose1_addr_buf;

        auto buf_input_transpose2_addr_buf = flag ? buf1_input_transpose2_addr_buf : buf2_input_transpose2_addr_buf;
        auto buf_output_real_transpose2_addr_buf = flag ? buf1_output_real_transpose2_addr_buf :
                                                          buf2_output_real_transpose2_addr_buf;

        WAIT_FLAG(MTE3, MTE2, event_id);

        const int64_t data_input_index =
            batch_idx * batch_max_num * N1 * N2 + N1_idx * N1_max_num * N2 + N2_block_idx * N2_max_num;
        // 读取未对齐的数据
        copy_gm2ubuf(buf_input, gm_input_tensor[data_input_index * 2], 1, batch_actual * N1 * 2 * N2, ROUND(batch_actual * N1 * 2 * N2, 8),
                     ROUND(batch_actual * N1 * 2 * N2, 8));

        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);
        const int64_t data_output_index = batch_idx * batch_max_num * 2 * N1 * N2_padding +
                                          N1_idx * N1_max_num * tile_N0 + N2_block_idx * 2 * N1 * N2_max_num;
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_real.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                  (batch_actual * N1 * 2 * N2 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 8, 8);
        // 将虚部放在实部后面，并且要能将实部和虚部取出来。故放的位置需要32B对齐。
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(buf_output_real[ROUND(batch_actual * N1 * N2, 8 * N2)].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(buf_input.GetPhyAddr()), nullptr,
                  (batch_actual * N1 * 2 * N2 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 2, 8, 8);

        PIPE_BARRIER(V);

        AscendC::DataCopy(buf_input,
                      buf_output_real,
            AscendC::DataCopyParams(
                1,
                2 * ROUND(batch_actual * N1 * N2, 8 * N2) / C0_SIZE,
                0,
                0
            )
        );

        PIPE_BARRIER(V);

        // 矩阵转置 (16, row_num) -> (row_num, 16)
        transpose_matrix(reinterpret_cast<__ubuf__ uint16_t *>(buf_output_real_transpose1_addr_buf.GetPhyAddr()),
                         reinterpret_cast<__ubuf__ uint16_t *>(buf_input_transpose1_addr_buf.GetPhyAddr()),
                         (16 * row_num / (8 * 16)), 16, 1);

        PIPE_BARRIER(V);
        AscendC::Duplicate<float>(buf_input, 0.0, 2 * max_num + 16 * C0_SIZE);
        PIPE_BARRIER(V);

        // 数据搬运 （row_num, 16) -> (row_num_padding + C0_SIZE, 16)
        AscendC::DataCopy(buf_input,
                      buf_output_real,
            AscendC::DataCopyParams(
                row_num / N2,
                N2 * 16 / C0_SIZE,
                0,
                (tile_N0 * 16 - N2 * 16) / C0_SIZE
            )
        );

        PIPE_BARRIER(V);

        // 矩阵转置 （row_num_padding + C0_SIZE, 16) -> (16, row_num_padding + C0_SIZE)
        transpose_matrix(reinterpret_cast<__ubuf__ uint16_t *>(buf_output_real_transpose2_addr_buf.GetPhyAddr()),
                         reinterpret_cast<__ubuf__ uint16_t *>(buf_input_transpose2_addr_buf.GetPhyAddr()),
                         16 * (row_num_padding + C0_SIZE) / (8 * 16), 1, 16);

        PIPE_BARRIER(V);

        AscendC::DataCopy(buf_input,
                      buf_output_real,
            AscendC::DataCopyParams(
                16,
                row_num_padding / C0_SIZE,
                (row_num_padding + C0_SIZE - row_num_padding) / C0_SIZE,
                0
            )
        );

        PIPE_BARRIER(V);

        // 合并实部和虚部的访存
        AscendC::DataCopy(buf_output_real,
                      buf_input,
            AscendC::DataCopyParams(
                batch_actual,
                N1 * tile_N0 / C0_SIZE,
                0,
                (2 * N1 * tile_N0 - N1 * tile_N0) / C0_SIZE
            )
        );
        AscendC::DataCopy(buf_output_real[N1 * tile_N0],
                      buf_input[ROUND(batch_actual * N1 * tile_N0, 8 * tile_N0)],
            AscendC::DataCopyParams(
                batch_actual,
                N1 * tile_N0 / C0_SIZE,
                0,
                (2 * N1 * tile_N0 - N1 * tile_N0) / C0_SIZE
            )
        );

        SET_FLAG(V, MTE3, event_id);
        WAIT_FLAG(V, MTE3, event_id);
        copy_ubuf2gm(gm_output_tensor[data_output_index], buf_output_real, batch_actual, 2 * N1 * tile_N0, 2 * N1 * tile_N0,
                     2 * N1 * tile_N0);
        SET_FLAG(MTE3, MTE2, event_id);
        flag = 1 - flag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

__aicore__ __inline__ __attribute__((always_inline)) void complex_split_RI_tile_zN_odd(
    __gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_output, __gm__ float *__restrict__ gm_auxil,
    int64_t batchSize, int64_t N1, int64_t N2, int64_t N2_padding, int32_t tile_K0, int32_t tile_N0, int32_t step_len)
{
    // 除0整改
    if (tile_N0 == 0) {
        tile_N0 = 1;
    }

    const int64_t N1_round = ROUND(N1, C0_SIZE);
    const int64_t N2_round = ROUND(N2, C0_SIZE);

    const int16_t aic_id = get_block_idx();
    const int16_t aiv_id_per_group = AscendC::GetSubBlockIdx();
    const int16_t group_num = AscendC::GetBlockNum();
    const int16_t subgroup_num = get_subblockdim();
    const int16_t aiv_id = aic_id * subgroup_num + aiv_id_per_group;
    const int16_t aiv_num = group_num * subgroup_num;

    AscendC::GlobalTensor<float> gm_input_tensor, gm_output_tensor;

    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    const int32_t max_num = 24 * 1024 / sizeof(float);
    int32_t now_UB = 0;
    auto buf1_input = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += 2 * max_num * sizeof(float);

    auto buf2_output_real = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf2_output_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    auto buf2_input = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += 2 * max_num * sizeof(float);

    auto buf1_output_real = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);
    auto buf1_output_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)now_UB);
    now_UB += max_num * sizeof(float);

    int32_t N2_max_num = max_num;
    int32_t N1_max_num = N2_max_num / ROUND(N2, tile_N0);  // 64
    N1_max_num = N1_max_num + (N1_max_num == 0);

    int32_t batch_max_num;

    N1_max_num = MIN(N1_max_num, N1);                     // 64
    N2_max_num = MIN(N2_max_num, ROUND(N2, tile_N0));     // 96
    batch_max_num = max_num / (N1_max_num * N2_max_num);  // 1
    batch_max_num += (batch_max_num == 0);

    int32_t N1_loop = (N1 + N1_max_num - 1) / N1_max_num;  // 2
    // 除0整改
    if (N1_loop == 0) {
        N1_loop = 1;
    }
    const int32_t N2_loop = (int32_t)((N2 + N2_max_num - 1) / N2_max_num);  // 1
    int64_t batch_loop = (batchSize + batch_max_num - 1) / batch_max_num;   // batchSize
    // 除0整改
    if (batch_loop == 0) {
        batch_loop = 1;
    }
    const int32_t batch_remain = batchSize % batch_max_num;
    const int32_t N2_remain = N2 % N2_max_num;            // 81
    const int32_t N1_remain = N1 % N1_max_num;            // 17
    const int64_t loop = batch_loop * N1_loop * N2_loop;  // batch * 2
    AscendC::Duplicate<float>(buf2_output_imag, 0.0, 24 * 1024 / sizeof(float));
    PIPE_BARRIER(V);

    bool flag = 0;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (int64_t i = 0; i < loop; i++) {
        int64_t loop_idx = i;
        if (loop_idx % static_cast<int64_t>(aiv_num) != static_cast<int64_t>(aiv_id)) {
            continue;
        }
        int64_t N1_idx = loop_idx % N1_loop;
        int64_t batch_N2_idx = loop_idx / N1_loop;
        int64_t batch_idx = batch_N2_idx % batch_loop;
        int64_t N2_block_idx = batch_N2_idx / batch_loop;

        int32_t N2_actual = N2_max_num;
        if (N2_block_idx == N2_loop - 1 && N2_remain > 0) {
            N2_actual = N2_remain;
        }

        int32_t N1_actual = N1_max_num;
        if (N1_idx == N1_loop - 1 && N1_remain > 0) {
            N1_actual = N1_remain;
        }

        int32_t batch_actual = batch_max_num;
        if (batch_idx == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        int32_t N2_actual_round = ROUND(N2_actual, tile_N0);

        auto buf_input = flag ? buf1_input : buf2_input;
        auto buf_output_real = flag ? buf1_output_real : buf2_output_real;
        auto buf_output_imag = buf2_output_imag;
        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;

        WAIT_FLAG(MTE3, MTE2, event_id);
        const int64_t data_input_index =
            batch_idx * batch_max_num * N1 * N2 + N1_idx * N1_max_num * N2 + N2_block_idx * N2_max_num;
        copy_gm2ubuf(buf_output_real, gm_input_tensor[data_input_index], batch_actual * N1_actual, ROUND(N2_actual, 8), N2, N2_actual_round);

        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);
        const int64_t data_output_index = batch_idx * batch_max_num * 2 * N1 * N2_padding +
                                          N1_idx * N1_max_num * tile_N0 + N2_block_idx * 2 * N1 * N2_max_num;
        auto gm_output_real_ptr = gm_output_tensor[data_output_index];
        auto gm_output_imag_ptr = gm_output_tensor[data_output_index + N1 * tile_N0];

        // 按照不同shape来合并访存
        if (N1_actual == N1 && N2_actual == N2 && N2 <= tile_N0) {
            AscendC::DataCopy(buf_input,
                              buf_output_real,
                AscendC::DataCopyParams(
                    batch_actual,
                    N1_actual * N2_actual_round / C0_SIZE,
                    0,
                    (2 * N1_actual * N2_actual_round - N1_actual * N2_actual_round) / C0_SIZE
                )
            );
            AscendC::DataCopy(buf_input[N1_actual * N2_actual_round],
                              buf_output_imag,
                AscendC::DataCopyParams(
                    batch_actual,
                    N1_actual * N2_actual_round / C0_SIZE,
                    0,
                    (2 * N1_actual * N2_actual_round - N1_actual * N2_actual_round) / C0_SIZE
                )
            );

            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);
            copy_ubuf2gm(gm_output_real_ptr, buf_input, batch_actual, 2 * N1_actual * tile_N0, 2 * N1_actual * tile_N0,
                         2 * N1 * tile_N0);
        } else if (batch_max_num == 1 && N2_actual == N2 && N2 <= tile_N0) {
            SET_FLAG(V, MTE3, event_id);
            SET_FLAG(V, MTE3, event_id + 2);
            WAIT_FLAG(V, MTE3, event_id);
            copy_ubuf2gm(gm_output_real_ptr, buf_output_real, batch_actual, N1_actual * tile_N0, N1_actual * tile_N0,
                         2 * N1 * tile_N0);
            WAIT_FLAG(V, MTE3, event_id + 2);
            copy_ubuf2gm(gm_output_imag_ptr, buf_output_imag, batch_actual, N1_actual * tile_N0, N1_actual * tile_N0,
                         2 * N1 * tile_N0);
        } else if (batch_max_num == 1 && N1_max_num == 1) {
            SET_FLAG(V, MTE3, event_id);
            SET_FLAG(V, MTE3, event_id + 2);
            WAIT_FLAG(V, MTE3, event_id);
            AscendC::DataCopyPad(gm_output_real_ptr,
                                 buf_output_real,
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, tile_N0 * sizeof(float),
                                               0, (2 * N1 * tile_N0 - tile_N0) * sizeof(float), 0));
            WAIT_FLAG(V, MTE3, event_id + 2);
            AscendC::DataCopyPad(gm_output_imag_ptr,
                                 buf_output_imag,
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, tile_N0 * sizeof(float),
                                               0, (2 * N1 * tile_N0 - tile_N0) * sizeof(float), 0));
        } else {
            for (int32_t j = 0; j < batch_actual; j++) {
                for (int32_t i = 0; i < N1_actual; i++) {
                    AscendC::DataCopy(buf_input[j * 2 * N1_actual * N2_actual_round + i * tile_N0],
                              buf_output_real[j * N1_actual * N2_actual_round + i * N2_actual_round],
                        AscendC::DataCopyParams(
                            N2_actual_round / tile_N0,
                            tile_N0 / C0_SIZE,
                            0,
                            (N1_actual * tile_N0 - tile_N0) / C0_SIZE
                        )
                    );
                    AscendC::DataCopy(buf_input[j * 2 * N1_actual * N2_actual_round + N1_actual * N2_actual_round + i * tile_N0],
                              buf_output_imag[j * N1_actual * N2_actual_round + i * N2_actual_round],
                        AscendC::DataCopyParams(
                            N2_actual_round / tile_N0,
                            tile_N0 / C0_SIZE,
                            0,
                            (N1_actual * tile_N0 - tile_N0) / C0_SIZE
                        )
                    );
                }
            }

            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);
            for (int32_t j = 0; j < batch_actual; j++) {
                AscendC::DataCopyPad(gm_output_real_ptr[j * 2 * N1 * N2_padding],
                                     buf_input[j * 2 * N1_actual * N2_actual_round],
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, N1_actual * tile_N0 * sizeof(float),
                                               0, (2 * N1 * tile_N0 - N1_actual * tile_N0) * sizeof(float), 0));
                AscendC::DataCopyPad(gm_output_imag_ptr[j * 2 * N1 * N2_padding],
                                     buf_input[j * 2 * N1_actual * N2_actual_round + N1_actual * N2_actual_round],
                    AscendC::DataCopyExtParams(N2_actual_round / tile_N0, N1_actual * tile_N0 * sizeof(float),
                                               0, (2 * N1 * tile_N0 - N1_actual * tile_N0) * sizeof(float), 0));
            }
        }
        SET_FLAG(MTE3, MTE2, event_id);
        flag = 1 - flag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

#endif
