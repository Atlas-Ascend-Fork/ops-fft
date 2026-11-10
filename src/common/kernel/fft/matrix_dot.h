/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MATRIX_DOT_H_
#define MATRIX_DOT_H_

#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "mem.h"
#include "utils.h"

__aicore__ __inline__ __attribute__((always_inline)) void matrix_dot_kernel(
    __gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_tw_matrix, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ workspace, __gm__ int32_t *__restrict__ gm_radix_list,
    __gm__ float *__restrict__ gm_auxil, int64_t batch_size, int64_t N0, int32_t N1, int64_t N2, int32_t tile_M0,
    int32_t tile_N0, int32_t tile_K0, int32_t step_index, int32_t step_len, int32_t fftDirection, bool seq357 = false)
{
    if (N2 == 0) {
        N2 = 1;
    }
    if (tile_K0 == 0) {
        tile_K0 = 1;
    }
    if (tile_N0 == 0) {
        tile_N0 = 1;
    }
    if (tile_M0 == 0) {
        tile_M0 = 1;
    }
    bool is_vec_padding = !(N2 % 3 == 0 && step_len >= 4);

    const int32_t BUF_SIZE = 16 * 1024;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::GlobalTensor<float> workspace_tensor;
    workspace_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    AscendC::GlobalTensor<float> gm_tw_matrix_tensor, gm_output_tensor;
    gm_tw_matrix_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_tw_matrix));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));
    AscendC::LocalTensor<float> buf_input1_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> buf_input1_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(BUF_SIZE);
    AscendC::LocalTensor<float> buf_input2_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(2 * BUF_SIZE);
    AscendC::LocalTensor<float> buf_input2_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(3 * BUF_SIZE);
    AscendC::LocalTensor<float> buf_tmp1_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * BUF_SIZE);
    AscendC::LocalTensor<float> buf_tmp1_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(5 * BUF_SIZE);
    AscendC::LocalTensor<float> buf_tmp2_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(6 * BUF_SIZE);
    AscendC::LocalTensor<float> buf_tmp2_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(7 * BUF_SIZE);
    AscendC::LocalTensor<float> buf1_tw_matrix_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(8 * BUF_SIZE);
    AscendC::LocalTensor<float> buf1_tw_matrix_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(9 * BUF_SIZE);
    AscendC::LocalTensor<float> buf2_tw_matrix_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(10 * BUF_SIZE);
    AscendC::LocalTensor<float> buf2_tw_matrix_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(11 * BUF_SIZE);

    int32_t batch_len;
    batch_len = L0AB_PINGPONG_BUFFER_LEN * 2 / (tile_K0 * tile_N0);
    batch_len += (batch_len == 0);

    int64_t next_N1;
    int64_t next_N2;
    int32_t next_tile_N0;
    next_N1 = gm_radix_list[step_index + 1];
    if (next_N1 == 0) {
        next_N1 = 1;
    }
    next_N2 = N2 / next_N1;
    if (next_N2 == 0) {
        next_N2 = 1;
    }
    get_tile_N0(next_N1, next_N2, step_index + 1, step_len, next_tile_N0);

    int16_t N1_loop = (2 * N1 + tile_M0 - 1) / tile_M0;
    if (N1_loop == 0) {
        N1_loop = 1;
    }
    int64_t N2_loop = (N2 + tile_N0 - 1) / tile_N0;
    if (N2_loop == 0) {
        N2_loop = 1;
    }
    int64_t batch_loop = (batch_size * N0 + batch_len - 1) / batch_len;
    if (batch_loop == 0) {
        batch_loop = 1;
    }
    int64_t loop = batch_loop * N1_loop * N2_loop;
    int64_t batch_remain = (batch_size * N0) % batch_len;
    int32_t N1_remain = (2 * N1) % tile_M0;
    int64_t N2_remain = N2 % tile_N0;
    int64_t aiv_id_per_group = AscendC::GetSubBlockIdx();
    int64_t group_num = AscendC::GetBlockNum();
    if (group_num == 0) {
        group_num = 1;
    }
    int64_t group_id = get_block_idx();

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    SET_FLAG(V, MTE2, EVENT_ID2);
    SET_FLAG(V, MTE2, EVENT_ID3);

    int32_t per1_N1_idx = -1;
    int32_t per1_N2_idx = -1;

    int32_t per2_N1_idx = -1;
    int32_t per2_N2_idx = -1;

    bool flag = 0;

    FftsCrossCoreSync<PIPE_MTE2, 2>(2);
    FftsCrossCoreSync<PIPE_MTE2, 2>(3);

    int64_t loop_per_group;
    int64_t loop_per_group_remain;
    int64_t loop_per_group_actual;

    if (step_len > 3 && step_index == step_len - 3) {
        loop_per_group = loop / group_num;
        loop_per_group_remain = loop % group_num;
        loop_per_group_actual = loop_per_group;
        if (group_id < loop_per_group_remain) {
            loop_per_group_actual++;
        }
    } else {
        loop_per_group = (loop / N1_loop) / group_num;
        loop_per_group_remain = (loop / N1_loop) % group_num;
        loop_per_group_actual = loop_per_group;
        if (group_id < loop_per_group_remain) {
            loop_per_group_actual++;
        }
        loop_per_group_actual *= static_cast<int64_t>(N1_loop);
    }

    for (int64_t i = 0; i < loop_per_group_actual; i++) {
        int64_t loop_idx;
        int64_t batch_len_idx;
        int64_t batch_N2_idx;
        int64_t N1_N2_idx;
        int64_t N2_idx;
        int64_t out_N1_idx;
        if (step_len > 3 && step_index == step_len - 3) {
            loop_idx =
                group_id * loop_per_group +
                (loop_per_group_remain > 0) * (group_id > loop_per_group_remain ? loop_per_group_remain : group_id) + i;
            batch_len_idx = loop_idx % batch_loop;
            N1_N2_idx = loop_idx / batch_loop;
            N2_idx = N1_N2_idx % N2_loop;
            out_N1_idx = N1_N2_idx / N2_loop;
        } else {
            loop_idx = group_id * loop_per_group * N1_loop +
                       (loop_per_group_remain > 0) *
                           (group_id > loop_per_group_remain ? loop_per_group_remain : group_id) * N1_loop +
                       i;
            out_N1_idx = loop_idx % N1_loop;
            batch_N2_idx = loop_idx / N1_loop;
            batch_len_idx = batch_N2_idx % batch_loop;
            N2_idx = batch_N2_idx / batch_loop;
        }

        auto buf_input_real = flag ? buf_input1_real : buf_input2_real;
        auto buf_input_imag = flag ? buf_input1_imag : buf_input2_imag;
        auto buf_tmp_real = flag ? buf_tmp1_real : buf_tmp2_real;
        auto buf_tmp_imag = flag ? buf_tmp1_imag : buf_tmp2_imag;

        auto buf_tw_matrix_real = flag ? buf1_tw_matrix_real : buf2_tw_matrix_real;
        auto buf_tw_matrix_imag = flag ? buf1_tw_matrix_imag : buf2_tw_matrix_imag;

        auto per_N1_idx = flag ? per1_N1_idx : per2_N1_idx;
        auto per_N2_idx = flag ? per1_N2_idx : per2_N2_idx;

        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;
        auto tw_event_id = flag ? EVENT_ID2 : EVENT_ID3;

        int32_t batch_actual = batch_len;
        if (batch_len_idx == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        int32_t N1_actual = tile_M0;
        if (out_N1_idx == N1_loop - 1 && N1_remain > 0) {
            N1_actual = N1_remain;
        }

        int32_t N2_actual = tile_N0;
        if (N2_idx == N2_loop - 1 && N2_remain > 0) {
            N2_actual = N2_remain;
        }

        int32_t tile_N1_idx = (aiv_id_per_group == 0 ? 0 : N1_actual / 4 * 2);
        int32_t N1_idx = out_N1_idx;
        N1_idx *= tile_M0;
        N1_idx += tile_N1_idx;

        int32_t N1_per_aiv = aiv_id_per_group == 0 ? N1_actual / 4 * 2 : N1_actual - N1_actual / 4 * 2;
        int32_t half_M0 = N1_per_aiv / 2;
        int32_t N2_round = tile_N0;

        WAIT_FLAG(MTE3, MTE2, event_id);

        WaitFlagDev(static_cast<int64_t>(flag) + 4);

        int64_t data_in_index = (static_cast<int64_t>(group_id) * 2 + static_cast<int64_t>(flag)) * batch_len * tile_M0 * tile_N0 + tile_N1_idx * tile_N0;
        copy_gm2ubuf(buf_tmp_real, workspace_tensor[data_in_index], batch_actual, N1_per_aiv * tile_N0, tile_M0 * tile_N0,
                     N1_per_aiv * tile_N0);

        FftsCrossCoreSync<PIPE_MTE2, 2>(flag + 2);

        SET_FLAG(MTE2, V, event_id);

        WAIT_FLAG(V, MTE2, tw_event_id);

        if (per_N1_idx != N1_idx || per_N2_idx != N2_idx) {
            // tw 矩阵 实部第一行是1.0f, 虚部第一行是0.0f
            if (N1_idx == 0 && step_len >= 4) {
                int64_t data_tw_matrix_index =
                    out_N1_idx * tile_M0 * tile_N0 + N2_idx * 2 * N1 * tile_N0 + tile_N1_idx * tile_N0;
                copy_gm2ubuf(buf_tw_matrix_real[tile_N0], gm_tw_matrix_tensor[data_tw_matrix_index + 2 * tile_N0],
                             N1_per_aiv / 2 - 1, tile_N0, 2 * tile_N0, tile_N0);
                copy_gm2ubuf(buf_tw_matrix_imag[tile_N0], gm_tw_matrix_tensor[data_tw_matrix_index + tile_N0 + 2 * tile_N0],
                             N1_per_aiv / 2 - 1, tile_N0, 2 * tile_N0, tile_N0);

                SET_FLAG(MTE2, V, tw_event_id);
                WAIT_FLAG(MTE2, V, tw_event_id);
                AscendC::Duplicate<float>(buf_tw_matrix_real, 1.0, tile_N0);
                AscendC::Duplicate<float>(buf_tw_matrix_imag, 0.0, tile_N0);
                PIPE_BARRIER(V);
            } else {
                int64_t data_tw_matrix_index =
                    out_N1_idx * tile_M0 * tile_N0 + N2_idx * 2 * N1 * tile_N0 + tile_N1_idx * tile_N0;
                copy_gm2ubuf(buf_tw_matrix_real, gm_tw_matrix_tensor[data_tw_matrix_index], N1_per_aiv / 2, tile_N0,
                             2 * tile_N0, tile_N0);
                copy_gm2ubuf(buf_tw_matrix_imag, gm_tw_matrix_tensor[data_tw_matrix_index + tile_N0], N1_per_aiv / 2, tile_N0,
                             2 * tile_N0, tile_N0);

                SET_FLAG(MTE2, V, tw_event_id);
                WAIT_FLAG(MTE2, V, tw_event_id);
            }

            for (int32_t i = 1; i < batch_len; i++) {
                AscendC::DataCopy(buf_tw_matrix_real[i * half_M0 * tile_N0], buf_tw_matrix_real,
                    AscendC::DataCopyParams(1, half_M0 * tile_N0 / C0_SIZE, 0, 0));
                AscendC::DataCopy(buf_tw_matrix_imag[i * half_M0 * tile_N0], buf_tw_matrix_imag,
                    AscendC::DataCopyParams(1, half_M0 * tile_N0 / C0_SIZE, 0, 0));
            }
            PIPE_BARRIER(V);
            if (flag) {
                per1_N1_idx = N1_idx;
                per1_N2_idx = N2_idx;
            } else {
                per2_N1_idx = N1_idx;
                per2_N2_idx = N2_idx;
            }
        }

        WAIT_FLAG(MTE2, V, event_id);

        AscendC::DataCopy(buf_input_real, buf_tmp_real,
            AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, (2 * tile_N0 - tile_N0) / C0_SIZE, 0));
        AscendC::DataCopy(buf_input_imag, buf_tmp_real[tile_N0],
            AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, (2 * tile_N0 - tile_N0) / C0_SIZE, 0));

        if (fftDirection == ASCENDFFT_INVERSE) {
            PIPE_BARRIER(V);
            mul_v<ArchType::ASCEND_V220>(buf_tmp_real, buf_input_real, buf_tw_matrix_imag,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            mul_v<ArchType::ASCEND_V220>(buf_tmp_imag, buf_input_imag, buf_tw_matrix_real,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            sub_v<ArchType::ASCEND_V220>(buf_tmp_imag, buf_tmp_imag, buf_tmp_real,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            mul_v<ArchType::ASCEND_V220>(buf_tmp_real, buf_input_real, buf_tw_matrix_real,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            // vmla(reinterpret_cast<__ubuf__ float *>(buf_tmp_real.GetPhyAddr()),
            //      reinterpret_cast<__ubuf__ float *>(buf_input_imag.GetPhyAddr()),
            //      reinterpret_cast<__ubuf__ float *>(buf_tw_matrix_imag.GetPhyAddr()),
            //      (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            mul_v<ArchType::ASCEND_V220>(buf_input_imag, buf_input_imag, buf_tw_matrix_imag,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            add_v<ArchType::ASCEND_V220>(buf_tmp_real, buf_input_imag, buf_tmp_real,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            SET_FLAG(V, MTE2, tw_event_id);
        } else {
            PIPE_BARRIER(V);
            mul_v<ArchType::ASCEND_V220>(buf_tmp_real, buf_input_real, buf_tw_matrix_real,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            mul_v<ArchType::ASCEND_V220>(buf_tmp_imag, buf_input_imag, buf_tw_matrix_imag,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            sub_v<ArchType::ASCEND_V220>(buf_tmp_real, buf_tmp_real, buf_tmp_imag,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            mul_v<ArchType::ASCEND_V220>(buf_tmp_imag, buf_input_real, buf_tw_matrix_imag,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            // vmla(reinterpret_cast<__ubuf__ float *>(buf_tmp_imag.GetPhyAddr()),
            //      reinterpret_cast<__ubuf__ float *>(buf_input_imag.GetPhyAddr()),
            //      reinterpret_cast<__ubuf__ float *>(buf_tw_matrix_real.GetPhyAddr()),
            //      (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            mul_v<ArchType::ASCEND_V220>(buf_input_imag, buf_input_imag, buf_tw_matrix_real,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            add_v<ArchType::ASCEND_V220>(buf_tmp_imag, buf_input_imag, buf_tmp_imag,
                 (batch_actual * half_M0 * tile_N0 + 8 * C0_SIZE - 1) / (8 * C0_SIZE), 1, 1, 1, 8, 8, 8);
            SET_FLAG(V, MTE2, tw_event_id);
        }

        if (seq357 && step_len >= 3 && step_index == step_len - 3 &&
            is_vec_padding) {  // 需要将数据排布成(batch, N0, 2 * N1, ROUND(N2, tile_N0)) 在该条件中 ROUND(N2, tile_N0)=tile_N0。讨论时间：24_3_14
            PIPE_BARRIER(V);
            AscendC::DataCopy(buf_input_real, buf_tmp_real,
                AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, 0, (2 * tile_N0 - tile_N0) / C0_SIZE));
            AscendC::DataCopy(buf_input_real[tile_N0], buf_tmp_imag,
                AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, 0, (2 * tile_N0 - tile_N0) / C0_SIZE));
            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);

            int64_t N2_padding_add_16 = next_N1 * (next_tile_N0);
            int64_t N2_2_padding_add_16 = 2 * next_N1 * (next_tile_N0);

            int64_t N2_part_idx = (N2_idx * tile_N0);
            int64_t N2_part_idx_floor = ((N2_idx * tile_N0) / next_N2 * next_N2);
            int64_t N2_part_idx_padding = (N2_idx * tile_N0) / next_N2 * ROUND(next_N2, next_tile_N0) +
                                          ((N2_idx * tile_N0) % next_N2) / next_tile_N0 * next_tile_N0;
            int64_t pre_remain_data = N2_part_idx - N2_part_idx_floor;

            int64_t first_data = next_N2 - pre_remain_data;

            bool is_sync = (next_tile_N0 - next_N2) >= 8;

            for (int32_t j = 0; j < batch_actual; j++) {
                int64_t data_out_index = (batch_len_idx * batch_len + j) * N1 * N2_2_padding_add_16 +
                                         (out_N1_idx * tile_M0 / 2 + tile_N1_idx / 2) * N2_2_padding_add_16 +
                                         N2_part_idx_padding;
                int32_t loop = (N2_actual - first_data + next_N2 - 1) / next_N2;
                int32_t loop_remain = (N2_actual - first_data) % next_N2;
                for (int32_t loop_idx = loop - 1; loop_idx >= 0; loop_idx--) {
                    int32_t loop_actual = next_N2;
                    if (loop_idx == loop - 1 && loop_remain > 0) {
                        loop_actual = loop_remain;
                    }
                    int32_t now_data_idx = first_data + loop_idx * next_N2;
                    int32_t now_data_idx_floor = now_data_idx / C0_SIZE * C0_SIZE;
                    int32_t now_data_idx_inc = now_data_idx - now_data_idx_floor;
                    copy_ubuf2gm(gm_output_tensor[data_out_index + (loop_idx + 1) * (next_tile_N0)-now_data_idx_inc],
                                 buf_input_real[j * 2 * half_M0 * tile_N0 + now_data_idx_floor], 2 * half_M0,
                                 loop_actual + now_data_idx_inc, N2_round, N2_padding_add_16);
                    if (!is_sync)
                        PIPE_BARRIER(MTE3);
                }

                if (first_data > 0) {
                    copy_ubuf2gm(gm_output_tensor[data_out_index + pre_remain_data],
                                 buf_input_real[j * 2 * half_M0 * tile_N0], 2 * half_M0,
                                 (!is_sync) ? first_data : ROUND(first_data, C0_SIZE), N2_round, N2_padding_add_16);
                }
            }
        } else if (seq357 && step_len >= 4 &&
                   step_index <
                       step_len - 3) {  // 需要将数据排布成(batch, N0, 2 * N1, ROUND(N2, tile_N0)) . 讨论时间：24_3_14
            PIPE_BARRIER(V);
            AscendC::DataCopy(buf_input_real, buf_tmp_real,
                AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, 0, (2 * tile_N0 - tile_N0) / C0_SIZE));
            AscendC::DataCopy(buf_input_real[tile_N0], buf_tmp_imag,
                AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, 0, (2 * tile_N0 - tile_N0) / C0_SIZE));
            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);
            int64_t N2_padding_add_16 = (next_N1 * ROUND(next_N2, next_tile_N0));
            int64_t N2_2_padding_add_16 = 2 * (next_N1 * ROUND(next_N2, next_tile_N0));

            int64_t N2_part_idx = (N2_idx * tile_N0);
            int64_t N2_part_idx_floor = ((N2_idx * tile_N0) / next_N2 * next_N2);
            int64_t N2_part_idx_padding = (N2_idx * tile_N0) / next_N2 * ROUND(next_N2, next_tile_N0) +
                                          ((N2_idx * tile_N0) % next_N2) / next_tile_N0 * next_tile_N0;
            int64_t pre_remain_data = N2_part_idx;
            while (pre_remain_data - next_N2 >= 0)
                pre_remain_data -= next_N2;
            while (pre_remain_data - next_tile_N0 >= 0)
                pre_remain_data -= next_tile_N0;

            bool is_sync = (next_tile_N0 - next_N2) >= 8;

            for (int32_t j = 0; j < batch_actual; j++) {
                int64_t batch_idx = (batch_len_idx * batch_len + j) / N0;
                int64_t N0_idx = (batch_len_idx * batch_len + j) % N0;

                int64_t data_out_index = (batch_len_idx * batch_len + j) * N1 * N2_2_padding_add_16 +
                                         (out_N1_idx * tile_M0 / 2 + tile_N1_idx / 2) * N2_2_padding_add_16;

                // 不同next_N2的边界在这个tiling中间
                if (N2_idx * tile_N0 / next_N2 * next_N2 != (N2_idx * tile_N0 + N2_actual) / next_N2 * next_N2) {
                    // next_N2的序号会在这个区域变化，故 next_N2_actual 一定小于 N2_actual
                    int64_t next_N2_actual = -N2_idx * tile_N0;
                    while (next_N2_actual <= 0)
                        next_N2_actual += next_N2;

                    // 表示上个next_N2剩余的部分
                    int64_t first_data = static_cast<int64_t>(next_tile_N0) - static_cast<int64_t>(pre_remain_data);
                    first_data = MIN(first_data, next_N2_actual);

                    int64_t first_loop = (next_N2_actual - first_data + next_tile_N0 - 1) / next_tile_N0;

                    int64_t now_data_idx = next_N2_actual;
                    int64_t now_data_idx_floor = now_data_idx / C0_SIZE * C0_SIZE;
                    int64_t now_data_idx_inc = now_data_idx - now_data_idx_floor;
                    int64_t next_idx = (N2_part_idx_padding + (first_loop + 1) * (next_tile_N0)) / next_tile_N0;

                    copy_ubuf2gm(gm_output_tensor[data_out_index + next_idx * next_tile_N0 - now_data_idx_inc],
                                 buf_input_real[j * 2 * half_M0 * tile_N0 + now_data_idx_floor], 2 * half_M0,
                                 N2_actual - next_N2_actual + now_data_idx_inc, N2_round, N2_padding_add_16);

                    if (!is_sync)
                        PIPE_BARRIER(MTE3);

                    if (next_N2_actual > 0) {
                        int64_t next_idx = (N2_part_idx_padding) / next_tile_N0;
                        copy_ubuf2gm(gm_output_tensor[data_out_index + next_idx * next_tile_N0 + pre_remain_data],
                                     buf_input_real[j * 2 * half_M0 * tile_N0], 2 * half_M0, next_N2_actual, N2_round,
                                     N2_padding_add_16);
                    }
                } else {
                    int64_t next_idx = (N2_part_idx_padding) / next_tile_N0;
                    copy_ubuf2gm(gm_output_tensor[data_out_index + next_idx * next_tile_N0 + pre_remain_data],
                                 buf_input_real[j * 2 * half_M0 * tile_N0], 2 * half_M0, N2_actual, N2_round,
                                 N2_padding_add_16);
                }
            }
        } else {
            // 不同分支是为了合并访存指令。
            if (step_index == step_len - 2 && step_len == 2) {
                int64_t N2_padding = ROUND(N2, 16);
                int64_t data_out_index = (batch_len_idx * batch_len) * 2 * N1 * N2_padding +
                                         (out_N1_idx * tile_M0 / 2 + tile_N1_idx / 2) * 2 * N2_padding +
                                         N2_idx * tile_N0;
                PIPE_BARRIER(V);
                AscendC::DataCopy(buf_input_real, buf_tmp_real,
                    AscendC::DataCopyParams(
                        batch_actual * half_M0,
                        ROUND(N2_actual, 16) / C0_SIZE,
                        (tile_N0 - ROUND(N2_actual, 16)) / C0_SIZE,
                        (2 * N2_padding - ROUND(N2_actual, 16)) / C0_SIZE
                    )
                );
                AscendC::DataCopy(buf_input_real[N2_padding], buf_tmp_imag,
                    AscendC::DataCopyParams(
                        batch_actual * half_M0,
                        ROUND(N2_actual, 16) / C0_SIZE,
                        (tile_N0 - ROUND(N2_actual, 16)) / C0_SIZE,
                        (2 * N2_padding - ROUND(N2_actual, 16)) / C0_SIZE
                    )
                );

                SET_FLAG(V, MTE3, event_id);
                WAIT_FLAG(V, MTE3, event_id);
                copy_ubuf2gm(gm_output_tensor[data_out_index], buf_input_real, batch_actual, 2 * half_M0 * N2_padding,
                             2 * half_M0 * N2_padding, 2 * N1 * N2_padding);
            } else if (step_index == step_len - 2 && step_len != 2) {
                int64_t N2_2_padding = 2 * ROUND(N2, 16);
                PIPE_BARRIER(V);
                AscendC::DataCopy(buf_input_real, buf_tmp_real,
                    AscendC::DataCopyParams(
                        batch_actual * half_M0,
                        ROUND(N2_actual, 16) / C0_SIZE,
                        (tile_N0 - ROUND(N2_actual, 16)) / C0_SIZE,
                        (2 * tile_N0 - ROUND(N2_actual, 16)) / C0_SIZE
                    )
                );
                AscendC::DataCopy(buf_input_real[N2_2_padding / 2], buf_tmp_imag,
                    AscendC::DataCopyParams(
                        batch_actual * half_M0,
                        tile_N0 / C0_SIZE,
                        0,
                        (2 * tile_N0 - tile_N0) / C0_SIZE
                    )
                );
                SET_FLAG(V, MTE3, event_id);
                WAIT_FLAG(V, MTE3, event_id);
                for (int32_t j = 0; j < batch_actual; j++) {
                    int64_t N0_idx = (batch_len_idx * batch_len + j) % N0;
                    int64_t batch_idx = (batch_len_idx * batch_len + j) / N0;
                    int64_t data_out_index = batch_idx * N1 * N0 * N2_2_padding +
                                             (out_N1_idx * tile_M0 / 2 + tile_N1_idx / 2) * N0 * N2_2_padding +
                                             N0_idx * N2_2_padding + N2_idx * tile_N0;
                    copy_ubuf2gm(gm_output_tensor[data_out_index], buf_input_real[j * half_M0 * 2 * tile_N0], half_M0,
                                 2 * ROUND(N2_actual, 16), 2 * tile_N0, N0 * N2_2_padding);
                }
            } else if (step_index != step_len - 2 && step_index == 0) {
                PIPE_BARRIER(V);
                AscendC::DataCopy(buf_input_real, buf_tmp_real,
                    AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, 0, (2 * tile_N0 - tile_N0) / C0_SIZE));
                AscendC::DataCopy(buf_input_real[tile_N0], buf_tmp_imag,
                    AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, 0, (2 * tile_N0 - tile_N0) / C0_SIZE));
                SET_FLAG(V, MTE3, event_id);
                WAIT_FLAG(V, MTE3, event_id);
                int64_t data_out_index = (batch_len_idx * batch_len) * N1 * 2 * N2 +
                                         (out_N1_idx * tile_M0 / 2 + tile_N1_idx / 2) * 2 * N2 + N2_idx * tile_N0;
                for (int32_t j = 0; j < batch_actual; j++) {
                    copy_ubuf2gm(gm_output_tensor[data_out_index + j * N1 * 2 * N2],
                                 buf_input_real[j * half_M0 * 2 * tile_N0], 2 * half_M0, N2_actual, tile_N0, N2);
                }
            } else {
                SET_FLAG(V, MTE3, event_id);
                WAIT_FLAG(V, MTE3, event_id);
                int64_t N2_2_padding = (2 * N2);
                for (int32_t j = 0; j < batch_actual; j++) {
                    int64_t N0_idx = (batch_len_idx * batch_len + j) % N0;
                    int64_t batch_idx = (batch_len_idx * batch_len + j) / N0;
                    int64_t data_out_index = batch_idx * N1 * N0 * N2_2_padding +
                                             (out_N1_idx * tile_M0 / 2 + tile_N1_idx / 2) * N0 * N2_2_padding +
                                             N0_idx * N2_2_padding + N2_idx * tile_N0;
                    copy_ubuf2gm(gm_output_tensor[data_out_index], buf_tmp_real[j * half_M0 * tile_N0], half_M0, N2_actual,
                                 N2_round, N0 * N2_2_padding);
                    copy_ubuf2gm(gm_output_tensor[data_out_index + N2_2_padding / 2], buf_tmp_imag[j * half_M0 * tile_N0],
                                 half_M0, N2_actual, N2_round, N0 * N2_2_padding);
                }
            }
        }

        SET_FLAG(MTE3, MTE2, event_id);

        flag = 1 - flag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);

    WAIT_FLAG(V, MTE2, EVENT_ID2);
    WAIT_FLAG(V, MTE2, EVENT_ID3);
}

#endif