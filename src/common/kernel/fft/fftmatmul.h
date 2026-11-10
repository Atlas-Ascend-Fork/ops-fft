/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPSFFT_COMMON_KERNEL_FFT_FFTMATMUL_H
#define OPSFFT_COMMON_KERNEL_FFT_FFTMATMUL_H

#include "kernel_operator.h"
#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "mem.h"
#include "utils.h"

template <bool is_sync, bool trans_b, bool trans_c>
__aicore__ __inline__ __attribute__((overloadable, always_inline)) void fft_matmul_split_n_kernel(
    bool is_complex_split_out_zN, __gm__ float *__restrict__ gm_a, __gm__ float *__restrict__ gm_b,
    __gm__ float *__restrict__ gm_c, __gm__ float *__restrict__ workspace, __gm__ float *__restrict__ gm_auxil,
    __gm__ int32_t *gm_radix_list, int64_t batch_size, int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb,
    int64_t ldc, int32_t tile_M0, int32_t tile_N0, int32_t tile_K0, int32_t step_index, int32_t step_len,
    int64_t per_N0 = 0, bool seq357 = false)
{
    // 除0整改
    if (tile_M0 == 0) {
        tile_M0 = 1;
    }
    if (tile_N0 == 0) {
        tile_N0 = 1;
    }
    if (tile_K0 == 0) {
        tile_K0 = 1;
    }
    // 当radix=3取消掉倒数第三次迭代vec的padding
    bool is_vec_padding = !((N * K / 2) % 3 == 0 && step_len >= 4);

    SET_FLAG(FIX, M, EVENT_ID0);
    SET_FLAG(FIX, M, EVENT_ID1);
    SET_FLAG(FIX, M, EVENT_ID2);
    SET_FLAG(FIX, M, EVENT_ID3);

    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);

    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);
    SET_FLAG(M, MTE1, EVENT_ID2);
    SET_FLAG(M, MTE1, EVENT_ID3);
    int64_t M_round = (M + 16 - 1) / 16 * 16;
    int64_t N_round = (N + 16 - 1) / 16 * 16;
    int64_t K_round = (K + 16 - 1) / 16 * 16;

    AscendC::GlobalTensor<float> gm_a_tensor;
    AscendC::GlobalTensor<float> gm_b_tensor;
    AscendC::GlobalTensor<float> gm_c_tensor;
    AscendC::GlobalTensor<float> workspace_tensor;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_c));
    workspace_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

    AscendC::LocalTensor<float> l1_base_a = buf.GetBuffer<BufferType::ASCEND_CB, float>(0);
    AscendC::LocalTensor<float> l1_base_b= buf.GetBuffer<BufferType::ASCEND_CB, float>(256 * 1024);

    AscendC::LocalTensor<float> l0a_base = buf.GetBuffer<BufferType::ASCEND_L0A, float>(0);
    AscendC::LocalTensor<float> l0b_base = buf.GetBuffer<BufferType::ASCEND_L0B, float>(0);
    AscendC::LocalTensor<float> l0c_base = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    int32_t max_L1_B_buffer_num = L1_PINGPONG_BUFFER_LEN / 2;

    auto l1_buf_a = l1_base_a;
    auto l0a_buf = l0a_base;
    auto l0b_buf = l0b_base;

    bool ping_flag = 0;
    bool mte1_mad_ping_flag = 0;
    bool l0c_ping_flag = 0;

    int32_t batch_len = L0AB_PINGPONG_BUFFER_LEN * 2 / (tile_N0 * tile_K0);
    batch_len += (batch_len == 0);
    int32_t batch_remain = batch_size % batch_len;

    int64_t m_loop = (M + tile_M0 - 1) / tile_M0;
    if (m_loop == 0) {
        m_loop = 1;
    }
    int64_t n_loop = (N + tile_N0 - 1) / tile_N0;
    int64_t k_loop = (K + tile_K0 - 1) / tile_K0;
    int64_t batch_loop = (batch_size + batch_len - 1) / batch_len;
    if (batch_loop == 0) {
        batch_loop = 1;
    }
    int64_t loop = batch_loop * m_loop * n_loop;

    int64_t m_actual = M;
    int64_t m_round = (m_actual + 15) / 16 * 16;
    int64_t k_actual = K;
    int64_t k_round = (k_actual + 15) / 16 * 16;

    int32_t m0_round = m_round;
    int32_t m0_actual = m_actual;
    int32_t k0_round = k_round;
    int32_t k0_actual = k_actual;

    // 每次l0a中放的分块最大为128 * 64
    int32_t n_part_loop = 1;
    int32_t n_part_len = L0AB_PINGPONG_BUFFER_LEN / tile_K0 / 16 * 16;
    if (n_part_len > tile_N0 / 2 && n_part_len < tile_N0)
        n_part_len = MIN(n_part_len, ROUND(tile_N0 / 2, 16));
    n_part_len += (n_part_len == 0);
    n_part_len = MIN(n_part_len, N_round);

    int32_t batch_part_len = L0AB_PINGPONG_BUFFER_LEN / (tile_K0 * n_part_len);
    batch_part_len += (batch_part_len == 0);

    int32_t per_N1 = -1;
    if (step_index > 0 && step_len > 2 && gm_radix_list != nullptr) {
        per_N1 = gm_radix_list[step_index - 1];
    }

    // M * K < 256KB
    // *** load matrix A to L1
    load_matrix_zN(l1_buf_a, gm_a_tensor, tile_M0, tile_K0, m_actual, k_round, lda);

    SET_FLAG(MTE2, MTE1, EVENT_ID0);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

    if (trans_c) {
        auto l1_src_a = l1_buf_a;
        for (int32_t i = 0; i < k0_round / C0_SIZE; i++) {
            AscendC::LoadData(
                l0b_buf[i * m0_round * C0_SIZE],
                l1_src_a[i * C0_SIZE * tile_M0],
                AscendC::LoadData2dParams(
                    0,                   // baseIdx
                    m0_round / BLOCK_SIZE,  // repeat
                    0,                    // srcStride
                    0,                   // sid
                    1,                   // dstGap
                    false,               // transpose
                    inc                  // addr_cal_mode_t
                )
            );
        }
    } else {
        auto l1_src_a = l1_buf_a;
        for (int32_t i = 0; i < m0_round / BLOCK_SIZE; i++) {
            AscendC::LoadData(
                l0a_buf[i * k0_round * BLOCK_SIZE],
                l1_src_a[i * CUBE_MATRIX_SIZE],
                AscendC::LoadData2dParams(
                    0,                   // baseIdx
                    k0_round / C0_SIZE,  // repeat
                    tile_M0 / BLOCK_SIZE, // srcStride
                    0,                   // sid
                    0,                   // dstGap
                    false,               // transpose
                    inc                  // addr_cal_mode_t
                )
            );
        }
    }

    SET_FLAG(MTE1, M, EVENT_ID0);
    WAIT_FLAG(MTE1, M, EVENT_ID0);

    int64_t group_num = AscendC::GetBlockNum();
    if (group_num == 0) {
        group_num = 1;
    }
    int64_t group_id = AscendC::GetBlockIdx();

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
        loop_per_group = (loop / m_loop) / group_num;
        loop_per_group_remain = (loop / m_loop) % group_num;
        loop_per_group_actual = loop_per_group;
        if (group_id < loop_per_group_remain) {
            loop_per_group_actual++;
        }
        loop_per_group_actual *= m_loop;
    }

    for (int64_t i = 0; i < loop_per_group_actual; i++) {
        int64_t loop_idx;
        int64_t batch_idx;
        int64_t batch_n_idx;
        int64_t m_n_idx;
        int64_t m_idx;
        int64_t n_idx;

        if (step_len > 3 && step_index == step_len - 3) {
            loop_idx =
                group_id * loop_per_group +
                (loop_per_group_remain > 0) * (group_id > loop_per_group_remain ? loop_per_group_remain : group_id) + i;
            batch_idx = loop_idx % batch_loop;
            m_n_idx = loop_idx / batch_loop;
            m_idx = m_n_idx / n_loop;
            n_idx = m_n_idx % n_loop;
        } else {
            loop_idx = group_id * loop_per_group * m_loop +
                       (loop_per_group_remain > 0) *
                           (group_id > loop_per_group_remain ? loop_per_group_remain : group_id) * m_loop +
                       i;
            m_idx = loop_idx % m_loop;
            batch_n_idx = loop_idx / m_loop;
            batch_idx = batch_n_idx % batch_loop;
            n_idx = batch_n_idx / batch_loop;
        }

        int64_t offset_b;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * tile_N0) : tile_N0;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        n_part_loop = (n_actual + n_part_len - 1) / n_part_len;

        int32_t batch_actual = batch_len;
        if (batch_idx == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }
        int32_t batch_part_loop = (batch_actual + batch_part_len - 1) / batch_part_len;
        int32_t batch_part_remain = batch_actual % batch_part_len;
        int32_t data_actual_ldb = ldb;

        if (trans_b) {
            offset_b = batch_idx * batch_len * ldb * N + n_idx * tile_N0 * ldb;
        } else {
            if (is_complex_split_out_zN) {
                offset_b = batch_idx * batch_len * K * ROUND(ldb, tile_N0) + n_idx * K * tile_N0;
            } else {
                offset_b = batch_idx * batch_len * K * ldb + n_idx * tile_N0;
            }
        }

        auto l1_buf_b = ping_flag ? l1_base_b : l1_base_b[max_L1_B_buffer_num];
        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
        auto gm_b_ptr = gm_b_tensor[offset_b];
        WAIT_FLAG(MTE1, MTE2, event_id + 2);
        if (trans_b) {
            load_matrix_zN(l1_buf_b, gm_b_tensor[offset_b], tile_N0, tile_K0, n_actual, k_round, data_actual_ldb, batch_actual,
                           ldb * N);
        } else {
            if (seq357 && step_len >= 3 && step_index == step_len - 2 &&
                is_vec_padding) {  // 倒数第二次迭代 N2 较小，直接padding成tile_N0读取
                int64_t out_batch_size = batch_size / per_N1 / per_N0;
                for (int32_t j = 0; j < batch_actual; j++) {
                    int64_t out_batch_idx = (batch_idx * batch_len + j) / (per_N1 * per_N0);
                    int64_t per_batch_idx = (batch_idx * batch_len + j) % (per_N1 * per_N0) / per_N0;
                    int64_t per_N0_idx = (batch_idx * batch_len + j) % (per_N1 * per_N0) % per_N0;
                    offset_b = (out_batch_idx * per_N0 * per_N1 + per_N0_idx * per_N1 + per_batch_idx) * K * (tile_N0);
                    load_matrix_zZ(l1_buf_b[j * tile_K0 * tile_N0], gm_b_tensor[offset_b], tile_K0, tile_N0, k_actual, tile_N0,
                                   tile_N0);
                }
            } else if (
                seq357 && step_len >= 4 && step_index < step_len - 2 &&
                step_index !=
                    0) {  // 迭代次数大于4的其他迭代，按照padding的距离读取矩阵，布局为(2 * N1, ROUND(N2, tile_N0))
                int64_t out_batch_size = batch_size / per_N1 / per_N0;
                for (int32_t j = 0; j < batch_actual; j++) {
                    int64_t out_batch_idx = (batch_idx * batch_len + j) / (per_N1 * per_N0);
                    int64_t per_batch_idx = (batch_idx * batch_len + j) % (per_N1 * per_N0) / per_N0;
                    int64_t per_N0_idx = (batch_idx * batch_len + j) % (per_N1 * per_N0) % per_N0;
                    offset_b = (out_batch_idx * per_N0 * per_N1 + per_N0_idx * per_N1 + per_batch_idx) * K *
                                   ROUND(ldb, tile_N0) +
                               n_idx * tile_N0;
                    load_matrix_zZ(l1_buf_b[j * tile_K0 * tile_N0], gm_b_tensor[offset_b + j * K * ROUND(ldb, tile_N0)],
                                   tile_K0, tile_N0, k_actual, tile_N0, ROUND(ldb, tile_N0));
                }
            } else if (is_complex_split_out_zN) {  // 表示虚实分离会排布成(N2 / tile_N0, 2 * N1, tile_N0)的布局方便读取
                if (k_actual == tile_K0) {
                    load_matrix_zZ(l1_buf_b, gm_b_tensor[offset_b], batch_actual * tile_K0, tile_N0, batch_actual * tile_K0, tile_N0,
                                   tile_N0);
                } else {
                    if (K / BLOCK_SIZE > batch_actual) {
                        for (int32_t j = 0; j < batch_actual; j++) {
                            load_matrix_zZ(l1_buf_b[j * tile_K0 * tile_N0], gm_b_tensor[offset_b + j * K * ROUND(ldb, tile_N0)],
                                           tile_K0, tile_N0, k_actual, tile_N0, tile_N0);
                        }
                    } else {
                        load_matrix_zZ(l1_buf_b, gm_b_tensor[offset_b], tile_K0, tile_N0, k_actual, tile_N0, tile_N0, batch_actual,
                                       K * ROUND(ldb, tile_N0));
                    }
                }
            } else {
                if (K / BLOCK_SIZE > batch_actual) {
                    for (int32_t j = 0; j < batch_actual; j++) {
                        load_matrix_zZ(l1_buf_b[j * tile_K0 * tile_N0], gm_b_tensor[offset_b + j * K * ldb], tile_K0, tile_N0,
                                       k_actual, tile_N0, ldb);
                    }
                } else {
                    load_matrix_zZ(l1_buf_b, gm_b_tensor[offset_b], tile_K0, tile_N0, k_actual, tile_N0, ldb, batch_actual, K * ldb);
                }
            }
        }

        SET_FLAG(MTE2, MTE1, event_id + 2);

        for (int32_t batch_part_idx = 0; batch_part_idx < batch_part_loop; batch_part_idx++) {
            int32_t batch_part_actual = batch_part_len;
            if (batch_part_idx == batch_part_loop - 1 && batch_part_remain > 0) {
                batch_part_actual = batch_part_remain;
            }
            for (int32_t n_part_idx = 0; n_part_idx < n_part_loop; n_part_idx++) {
                int32_t n0_round = (n_part_idx < n_part_loop - 1) ? n_part_len : n_round - n_part_idx * n_part_len;
                int32_t n0_actual = (n_part_idx < n_part_loop - 1) ? n_part_len : n_actual - n_part_idx * n_part_len;
                auto l0c_buf = l0c_base[l0c_ping_flag * L0C_PINGPONG_BUFFER_LEN];
                auto l0c_EVENT_ID = (l0c_ping_flag ? EVENT_ID0 : EVENT_ID1);
                WAIT_FLAG(M, MTE1, l0c_EVENT_ID);
                // *** load matrix B from L1 to L0B
                if (n_part_idx == 0 && batch_part_idx == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id + 2);
                }
                if (trans_c) {
                    l0a_buf = l0a_base[l0c_ping_flag * L0AB_PINGPONG_BUFFER_LEN];
                    auto l1_src_b = l1_buf_b[batch_part_idx * batch_part_len * tile_K0 * tile_N0 +
                                    n_part_idx * n_part_len * C0_SIZE];
                    int32_t loop;
                    int32_t l0a_stride;
                    int32_t l1_src_b_stride;
                    int32_t repeat;
                    int32_t src_stride;
                    int32_t dst_stride;

                    if (k0_round / C0_SIZE < n0_round / BLOCK_SIZE) {
                        loop = k0_round / C0_SIZE;
                        l0a_stride = CUBE_MATRIX_SIZE;
                        l1_src_b_stride = tile_N0 * C0_SIZE;
                        repeat = n0_round / BLOCK_SIZE;
                        src_stride = 1;
                        dst_stride = k0_round / C0_SIZE - 1;
                    } else {
                        loop = n0_round / BLOCK_SIZE;
                        l0a_stride = k0_round * BLOCK_SIZE;
                        l1_src_b_stride = CUBE_MATRIX_SIZE;
                        repeat = k0_round / C0_SIZE;
                        src_stride = tile_N0 / BLOCK_SIZE;
                        dst_stride = 0;
                    }
                    for (int32_t j = 0; j < batch_part_actual; j++) {
                        for (int32_t i = 0; i < loop; i++) {
                            AscendC::LoadData(
                                l0a_buf[j * tile_K0 * n_part_len + i * l0a_stride],
                                l1_src_b[j * tile_K0 * tile_N0 + i * l1_src_b_stride],
                                AscendC::LoadData2dParams(
                                    0,                   // baseIdx
                                    repeat,              // repeat
                                    src_stride,          // srcStride
                                    0,                   // sid
                                    dst_stride,          // dstGap
                                    false,               // transpose
                                    inc                  // addr_cal_mode_t
                                )
                            );
                        }
                    }
                } else {
                    l0b_buf = l0b_base[l0c_ping_flag * L0AB_PINGPONG_BUFFER_LEN];
                    if (trans_b) {
                        auto l1_src_b = l1_buf_b[batch_part_idx * batch_part_len * tile_K0 * tile_N0 +
                                        n_part_idx * n_part_len * C0_SIZE];

                        if (n0_round == tile_N0) {
                            AscendC::LoadData(
                                l0b_buf,
                                l1_src_b,
                                AscendC::LoadData2dParams(
                                    0,                   // baseIdx
                                    batch_part_actual * k0_round / C0_SIZE * n0_round / BLOCK_SIZE,  // repeat
                                    1,                   // srcStride
                                    0,                   // sid
                                    0,                   // dstGap
                                    false,               // transpose
                                    inc                  // addr_cal_mode_t
                                )
                            );
                        } else {
                            for (int32_t j = 0; j < batch_part_actual; j++) {
                                for (int32_t i = 0; i < k0_round / C0_SIZE; i++) {
                                    AscendC::LoadData(
                                        l0b_buf[j * tile_K0 * n_part_len + i * n0_round * C0_SIZE],
                                        l1_src_b[j * tile_K0 * tile_N0 + i * C0_SIZE * tile_N0],
                                        AscendC::LoadData2dParams(
                                            0,                   // baseIdx
                                            n0_round / BLOCK_SIZE,  // repeat
                                            1,                   // srcStride
                                            0,                   // sid
                                            0,                   // dstGap
                                            false,               // transpose
                                            inc                  // addr_cal_mode_t
                                        )
                                    );
                                }
                            }
                        }

                    } else {
                        auto l1_src_b = l1_buf_b[batch_part_idx * batch_part_len * tile_K0 * tile_N0 +
                                        n_part_idx * n_part_len * BLOCK_SIZE];
                        if (k0_round / BLOCK_SIZE >= n0_round / BLOCK_SIZE) {
                            for (int32_t j = 0; j < batch_part_actual; j++) {
                                for (int32_t i = 0; i < n0_round / BLOCK_SIZE; i++) {
                                    AscendC::LoadDataWithTranspose(
                                        l0b_buf[j * tile_K0 * n_part_len + i * CUBE_MATRIX_SIZE],
                                        l1_src_b[j * tile_K0 * tile_N0 + i * 2 * CUBE_MATRIX_SIZE],
                                        AscendC::LoadData2dTransposeParams(
                                            0,                             // indexID
                                            k0_round / BLOCK_SIZE,         // repeat
                                            tile_N0 / BLOCK_SIZE,               // srcStride
                                            2 * n0_round / BLOCK_SIZE - 1,  // dstGap
                                            n0_round / BLOCK_SIZE - 1,      // dstFracStride
                                            inc                           // addrmode
                                        )
                                    );
                                }
                            }
                        } else {
                            for (int32_t j = 0; j < batch_part_actual; j++) {
                                for (int32_t i = 0; i < k0_round / BLOCK_SIZE; i++) {
                                    AscendC::LoadDataWithTranspose(
                                        l0b_buf[j * tile_K0 * n_part_len + i * n0_round * BLOCK_SIZE],
                                        l1_src_b[j * tile_K0 * tile_N0 + i * tile_N0 * BLOCK_SIZE],
                                        AscendC::LoadData2dTransposeParams(
                                            0,                             // indexID
                                            n0_round / BLOCK_SIZE,         // repeat
                                            1,                             // srcStride
                                            0,                             // dstGap
                                            n0_round / BLOCK_SIZE - 1,     // dstFracStride
                                            inc                            // addrmode
                                        )
                                    );
                                }
                            }
                        }
                    }
                }

                if (n_part_idx == n_part_loop - 1 && batch_part_idx == batch_part_loop - 1) {
                    SET_FLAG(MTE1, MTE2, event_id + 2);
                }

                SET_FLAG(MTE1, M, l0c_EVENT_ID);
                WAIT_FLAG(MTE1, M, l0c_EVENT_ID);

                bool init_c = 1;
                WAIT_FLAG(FIX, M, l0c_EVENT_ID);
                for (int32_t j = 0; j < batch_part_actual; j++) {
                    if (trans_c) {
                        AscendC::MmadParams mmParam = AscendC::MmadParams(
                            n0_round,  // m
                            m0_round,  // n
                            k0_actual, // k
                            0,
                            false,
                            init_c
                        );
                        mmParam.kDirectionAlign = true;

                        AscendC::Mmad(
                            l0c_buf[j * tile_M0 * n_part_len], l0a_buf[j * tile_K0 * n_part_len], l0b_buf, mmParam
                        );
                    } else {
                        AscendC::MmadParams mmParam = AscendC::MmadParams(
                            m0_round,  // m
                            n0_actual,  // n
                            k0_actual, // k
                            0,
                            false,
                            init_c
                        );
                        mmParam.kDirectionAlign = true;

                        AscendC::Mmad(
                            l0c_buf[j * tile_M0 * n_part_len], l0a_buf, l0b_buf[j * tile_K0 * n_part_len], mmParam
                        );
                    }
                }

                SET_FLAG(M, MTE1, l0c_EVENT_ID);

                SET_FLAG(M, FIX, l0c_EVENT_ID);
                WAIT_FLAG(M, FIX, l0c_EVENT_ID);
                if (is_sync) {
                    if (n_part_idx == 0 && batch_part_idx == 0) {
                        WaitFlagDev(ping_flag + 2);
                    }
                    if (trans_c) {
                    } else {
                        // 如果 batch_len > 1 去掉矩阵的padding，便于虚实结合按照向量来操作
                        if (step_index == step_len - 1 && batch_len > 1) {
                            if (batch_part_actual > 1) {
                                uint64_t ndNum = batch_part_actual;
                                uint64_t srcNdStride = tile_M0 * n_part_len / (16 * 16);
                                uint64_t dstNdStride = tile_M0 * tile_N0;
                                uint64_t config = ndNum | (srcNdStride << 16) | (dstNdStride << 32);
                                AscendC::SetNdParaImpl(config);
                            }
                            int64_t offset_c = (group_id * 2 + ping_flag) * batch_len * tile_M0 * tile_N0 +
                                               (batch_part_idx * batch_part_len) * tile_M0 * tile_N0;

                            // copy from L0C to gm
                            copy_matrix_cc_to_gm(workspace + offset_c + n_part_idx * n_part_len, reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf.GetPhyAddr())),
                                                 0,          // sid
                                                 n0_actual,  // NSize
                                                 m0_actual,  // MSize
                                                 n0_actual,  // dstStride_dst_D
                                                 m0_round,   // srcStride
                                                 0,          // UnitFlagMode
                                                 NoQuant,    // QuantPRE
                                                 0,          // ReLUPRE
                                                 false,      // channelSplit
                                                 true        // NZ2ND_EN
                            );
                        } else {
                            if (batch_part_actual > 1) {
                                uint64_t ndNum = batch_part_actual;
                                uint64_t srcNdStride = tile_M0 * n_part_len / (16 * 16);
                                uint64_t dstNdStride = tile_M0 * tile_N0;
                                uint64_t config = ndNum | (srcNdStride << 16) | (dstNdStride << 32);
                                AscendC::SetNdParaImpl(config);
                            }
                            int64_t offset_c = (group_id * 2 + ping_flag) * batch_len * tile_M0 * tile_N0 +
                                               (batch_part_idx * batch_part_len) * tile_M0 * tile_N0;

                            // copy from L0C to gm
                            copy_matrix_cc_to_gm(workspace + offset_c + n_part_idx * n_part_len, reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf.GetPhyAddr())),
                                                 0,                                               // sid
                                                 (batch_part_actual > 1) ? n0_round : n0_actual,  // NSize
                                                 m0_actual,                                       // MSize
                                                 tile_N0,                                         // dstStride_dst_D
                                                 m0_round,                                        // srcStride
                                                 0,                                               // UnitFlagMode
                                                 NoQuant,                                         // QuantPRE
                                                 0,                                               // ReLUPRE
                                                 false,                                           // channelSplit
                                                 true                                             // NZ2ND_EN
                            );
                        }
                    }
                    AscendC::SetNdParaImpl(0x1);
                    if (n_part_idx == n_part_loop - 1 && batch_part_idx == batch_part_loop - 1) {
                        FftsCrossCoreSync<PIPE_FIX, 2>(ping_flag + 4);
                    }
                } else {
                    if (trans_c) {
                        if (batch_part_actual > 1) {
                            uint64_t ndNum = batch_part_actual;
                            uint64_t srcNdStride = tile_M0 * n_part_len / (16 * 16);
                            uint64_t dstNdStride = N * ldc;
                            uint64_t config = ndNum | (srcNdStride << 16) | (dstNdStride << 32);
                            AscendC::SetNdParaImpl(config);
                        }
                        int64_t offset_c = (batch_idx * batch_len + batch_part_idx * batch_part_len) * N * ldc +
                                           m_idx * tile_M0 + n_idx * tile_N0 * ldc;
                        // copy from L0C to gm
                        copy_matrix_cc_to_gm(gm_c + offset_c + n_part_idx * n_part_len * ldc, reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf.GetPhyAddr())),
                                             0,          // sid
                                             m0_actual,  // NSize
                                             n0_actual,  // MSize
                                             ldc,        // dstStride_dst_D
                                             n0_round,   // srcStride
                                             0,          // UnitFlagMode
                                             NoQuant,    // QuantPRE
                                             0,          // ReLUPRE
                                             false,      // channelSplit
                                             true        // NZ2ND_EN
                        );
                    } else {
                        if (batch_part_actual > 1) {
                            uint64_t ndNum = batch_part_actual;
                            uint64_t srcNdStride = tile_M0 * n_part_len / (16 * 16);
                            uint64_t dstNdStride = M * ldc;
                            uint64_t config = ndNum | (srcNdStride << 16) | (dstNdStride << 32);
                            AscendC::SetNdParaImpl(config);
                        }
                        int64_t offset_c = (batch_idx * batch_len + batch_part_idx * batch_part_len) * M * ldc +
                                           m_idx * tile_M0 * ldc + n_idx * tile_N0;
                        // copy from L0C to gm
                        copy_matrix_cc_to_gm(gm_c + offset_c + n_part_idx * n_part_len, reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf.GetPhyAddr())),
                                             0,          // sid
                                             n0_round,   // NSize
                                             m0_actual,  // MSize
                                             ldc,        // dstStride_dst_D
                                             m0_round,   // srcStride
                                             0,          // UnitFlagMode
                                             NoQuant,    // QuantPRE
                                             0,          // ReLUPRE
                                             false,      // channelSplit
                                             true        // NZ2ND_EN
                        );
                    }
                    AscendC::SetNdParaImpl(0x1);
                }
                SET_FLAG(FIX, M, l0c_EVENT_ID);
                l0c_ping_flag = 1 - l0c_ping_flag;
            }
        }

        ping_flag = 1 - ping_flag;
    }
    if (is_sync) {
        WaitFlagDev(2);
        WaitFlagDev(3);
    }

    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(M, MTE1, EVENT_ID3);

    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);

    WAIT_FLAG(FIX, M, EVENT_ID0);
    WAIT_FLAG(FIX, M, EVENT_ID1);
    WAIT_FLAG(FIX, M, EVENT_ID2);
    WAIT_FLAG(FIX, M, EVENT_ID3);
}

template <bool is_sync, bool trans_b, bool trans_c>
__aicore__ __inline__ __attribute__((overloadable, always_inline)) void fft_matmul_split_k_kernel(
    bool is_complex_split_out_zN, __gm__ float *__restrict__ gm_a, __gm__ float *__restrict__ gm_b,
    __gm__ float *__restrict__ gm_c, __gm__ float *__restrict__ workspace, __gm__ float *__restrict__ gm_auxil,
    __gm__ int32_t *gm_radix_list, int64_t batch_size, int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb,
    int64_t ldc, int32_t tile_M0, int32_t tile_N0, int32_t tile_K0, int32_t step_index, int32_t step_len,
    int64_t per_N0 = 0, bool seq357 = false)
{
    if (tile_M0 == 0) {
        tile_M0 = 1;
    }
    if (tile_N0 == 0) {
        tile_N0 = 1;
    }
    if (tile_K0 == 0) {
        tile_K0 = 1;
    }
    // 当radix=3取消掉倒数第三次迭代vec的padding
    bool is_vec_padding = !((N * K / 2) % 3 == 0 && step_len >= 4);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);

    SET_FLAG(M, MTE1, EVENT_ID2);
    SET_FLAG(M, MTE1, EVENT_ID3);

    SET_FLAG(FIX, M, EVENT_ID0);
    SET_FLAG(FIX, M, EVENT_ID1);

    int64_t M_round = (M + 16 - 1) / 16 * 16;
    int64_t N_round = (N + 16 - 1) / 16 * 16;
    int64_t K_round = (K + 16 - 1) / 16 * 16;
    int64_t K_K0_round = (K + tile_K0 - 1) / tile_K0 * tile_K0;

    AscendC::GlobalTensor<float> gm_a_tensor;
    AscendC::GlobalTensor<float> gm_b_tensor;
    AscendC::GlobalTensor<float> gm_c_tensor;
    AscendC::GlobalTensor<float> workspace_tensor;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_c));
    workspace_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

    AscendC::LocalTensor<float> l0a_base = buf.GetBuffer<BufferType::ASCEND_L0A, float>(0);
    AscendC::LocalTensor<float> l0b_base = buf.GetBuffer<BufferType::ASCEND_L0B, float>(0);
    AscendC::LocalTensor<float> l0c_base = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    AscendC::LocalTensor<float> l1_base_a = buf.GetBuffer<BufferType::ASCEND_CB, float>(0);
    AscendC::LocalTensor<float> l1_base_b= buf.GetBuffer<BufferType::ASCEND_CB, float>(256 * 1024);

    int32_t batch_len = L0AB_PINGPONG_BUFFER_LEN * 2 / (tile_K0 * tile_N0);
    batch_len += (batch_len == 0);

    int32_t per_N1 = -1;
    if (step_index > 0 && gm_radix_list != nullptr) {
        per_N1 = gm_radix_list[step_index - 1];
    }

    int64_t batch_remain = batch_size % batch_len;

    int64_t m_loop = (M + tile_M0 - 1) / tile_M0;
    int64_t n_loop = (N + tile_N0 - 1) / tile_N0;
    int64_t k_loop = (K + tile_K0 - 1) / tile_K0;
    int64_t batch_loop = (batch_size + batch_len - 1) / batch_len;
    if (batch_loop == 0) {
        batch_loop = 1;
    }
    int64_t loop = batch_loop * m_loop * n_loop;

    bool l0c_ping_flag = 0;
    bool ping_flag = 0;
    bool mte1_mad_ping_flag = 0;

    load_matrix_zN(l1_base_a, gm_a_tensor, M_round, K_round, M, K_round, lda);
    SET_FLAG(MTE2, MTE1, EVENT_ID7);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID7);

    int16_t group_num = AscendC::GetBlockNum();
    if (group_num == 0) {
        group_num = 1;
    }
    int16_t group_id = AscendC::GetBlockIdx();

    int64_t loop_per_group;
    int64_t loop_per_group_remain;
    int64_t loop_per_group_actual;
    if (step_len > 3 && step_index == step_len - 3) {
        loop_per_group = loop / group_num;
        loop_per_group_remain = loop % group_num;
        loop_per_group_actual = loop_per_group;
        if (static_cast<int64_t>(group_id) < loop_per_group_remain) {
            loop_per_group_actual++;
        }
    } else {
        loop_per_group = (loop / m_loop) / group_num;
        loop_per_group_remain = (loop / m_loop) % group_num;
        loop_per_group_actual = loop_per_group;
        if (static_cast<int64_t>(group_id) < loop_per_group_remain) {
            loop_per_group_actual++;
        }
        loop_per_group_actual *= m_loop;
    }

    for (int64_t i = 0; i < loop_per_group_actual; i++) {
        int64_t loop_idx;
        int64_t batch_idx;
        int64_t batch_n_idx;
        int64_t m_n_idx;
        int64_t m_idx;
        int64_t n_idx;

        if (step_len > 3 && step_index == step_len - 3) {
            loop_idx =
                group_id * loop_per_group +
                (loop_per_group_remain > 0) * (group_id > loop_per_group_remain ? loop_per_group_remain : group_id) + i;
            batch_idx = loop_idx % batch_loop;
            m_n_idx = loop_idx / batch_loop;
            m_idx = m_n_idx / n_loop;
            n_idx = m_n_idx % n_loop;
        } else {
            loop_idx = group_id * loop_per_group * m_loop +
                       (loop_per_group_remain > 0) *
                           (group_id > loop_per_group_remain ? loop_per_group_remain : group_id) * m_loop +
                       i;
            m_idx = loop_idx % m_loop;
            batch_n_idx = loop_idx / m_loop;
            batch_idx = batch_n_idx % batch_loop;
            n_idx = batch_n_idx / batch_loop;
        }

        int32_t batch_actual = batch_len;
        if (batch_idx == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        auto l0c_buf = l0c_ping_flag ? l0c_base[L0C_PINGPONG_BUFFER_LEN] : l0c_base;
        auto l0c_EVENT_ID = l0c_ping_flag ? EVENT_ID0 : EVENT_ID1;

        int64_t offset_a, offset_b;
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * tile_M0) : tile_M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * tile_N0) : tile_N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > batch_actual * n_round ? m_round : batch_actual * n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        if (k_part_len > tile_K0 / 2 && k_part_len < tile_K0)
            k_part_len = MIN(k_part_len, ROUND(tile_K0 / 2, 16));

        for (int32_t k_idx = 0; k_idx < (int32_t)k_loop; k_idx++) {
            if (trans_b) {
                offset_b = batch_idx * batch_len * ldb * N + n_idx * tile_N0 * ldb + k_idx * tile_K0;
            } else {
                if (is_complex_split_out_zN) {
                    offset_b = batch_idx * batch_len * K * ROUND(ldb, tile_N0) + k_idx * tile_K0 * tile_N0 +
                               n_idx * K * tile_N0;
                } else {
                    offset_b = batch_idx * batch_len * K * ldb + k_idx * tile_K0 * ldb + n_idx * tile_N0;
                }
            }

            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * tile_K0) : tile_K0;
            int32_t k_round = (k_actual + 15) / 16 * 16;
            int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

            auto l1_buf_a = l1_base_a;
            auto l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN / 2];
            if (k_loop > 1) {
                l1_buf_b = l1_base_b[k_idx * batch_len * tile_K0 * tile_N0];
            }
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
            auto gm_b_ptr = gm_b_tensor[offset_b];
            // *** load matrix B to L1
            WAIT_FLAG(MTE1, MTE2, event_id + 2);
            if (m_idx == 0 || (step_len > 3 && step_index == step_len - 3)) {
                if (trans_b) {
                    load_matrix_zN(l1_buf_b, gm_b_tensor[offset_b], tile_N0, tile_K0, n_actual, k_round, ldb, batch_actual, ldb * N);
                } else {
                    if (seq357 && step_len >= 3 && step_index == step_len - 2 &&
                        is_vec_padding) {  // 倒数第二次迭代 N2 较小，直接padding成tile_N0读取
                        int64_t out_batch_size = batch_size / per_N1 / per_N0;
                        for (int32_t j = batch_actual - 1; j >= 0; j--) {
                            int64_t out_batch_idx = (batch_idx * batch_len + j) / (per_N1 * per_N0);
                            int64_t per_batch_idx = (batch_idx * batch_len + j) % (per_N1 * per_N0) / per_N0;
                            int64_t per_N0_idx = (batch_idx * batch_len + j) % (per_N1 * per_N0) % per_N0;
                            offset_b = (out_batch_idx * per_N0 * per_N1 + per_N0_idx * per_N1 + per_batch_idx) * K *
                                           (tile_N0) +
                                       k_idx * tile_K0 * (tile_N0);
                            auto gm_b_ptr = gm_b_tensor[offset_b];
                            load_matrix_zZ(l1_buf_b[j * tile_K0 * tile_N0], gm_b_ptr, tile_K0, tile_N0, k_actual,
                                           tile_N0, tile_N0);
                        }
                    } else if (
                        seq357 && step_len >= 4 && step_index < step_len - 2 &&
                        step_index !=
                            0) {  // 迭代次数大于4的其他迭代，按照padding的距离读取矩阵，布局为(2 * N1, ROUND(N2, tile_N0))
                        int64_t out_batch_size = batch_size / per_N1 / per_N0;
                        for (int32_t j = batch_actual - 1; j >= 0; j--) {
                            int64_t out_batch_idx = (batch_idx * batch_len + j) / (per_N1 * per_N0);
                            int64_t per_batch_idx = (batch_idx * batch_len + j) % (per_N1 * per_N0) / per_N0;
                            int64_t per_N0_idx = (batch_idx * batch_len + j) % (per_N1 * per_N0) % per_N0;
                            offset_b = (out_batch_idx * per_N0 * per_N1 + per_N0_idx * per_N1 + per_batch_idx) * K *
                                           ROUND(ldb, tile_N0) +
                                       k_idx * tile_K0 * ROUND(ldb, tile_N0) + n_idx * tile_N0;
                            auto gm_b_ptr = gm_b_tensor[offset_b];
                            load_matrix_zZ(l1_buf_b[j * tile_K0 * tile_N0], gm_b_ptr[j * K * ROUND(ldb, tile_N0)],
                                           tile_K0, tile_N0, k_actual, tile_N0, ROUND(ldb, tile_N0));
                        }
                    } else if (
                        is_complex_split_out_zN) {  // 表示虚实分离会排布成(N2 / tile_N0, 2 * N1, tile_N0)的布局方便读取
                        if (k_actual == tile_K0) {
                            load_matrix_zZ(l1_buf_b, gm_b_ptr, batch_actual * tile_K0, tile_N0, batch_actual * tile_K0,
                                           tile_N0, tile_N0);
                        } else {
                            for (int32_t j = batch_actual - 1; j >= 0; j--) {
                                load_matrix_zZ(l1_buf_b[j * tile_K0 * tile_N0], gm_b_ptr[j * K * ROUND(ldb, tile_N0)],
                                               tile_K0, tile_N0, k_actual, tile_N0, tile_N0);
                            }
                        }
                    } else {
                        if (K / BLOCK_SIZE > batch_actual) {
                            for (int32_t j = batch_actual - 1; j >= 0; j--) {
                                load_matrix_zZ(l1_buf_b[j * tile_K0 * tile_N0], gm_b_ptr[j * K * ldb], tile_K0,
                                               tile_N0, k_actual, tile_N0, ldb);
                            }
                        } else {
                            load_matrix_zZ(l1_buf_b, gm_b_ptr, tile_K0, tile_N0, k_actual, tile_N0, ldb, batch_actual,
                                           K * ldb);
                        }
                    }
                }
            }

            SET_FLAG(MTE2, MTE1, event_id + 2);

            for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID2 : EVENT_ID3;
                auto l0a_buf = l0a_base[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];
                auto l0b_buf = l0b_base[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];

                WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                if (trans_c) {
                    auto l1_src_a =
                        l1_buf_a[m_idx * tile_M0 * C0_SIZE + (k_idx * tile_K0 + k_part_idx * k_part_len) * M_round];
                    for (int32_t i = 0; i < k0_round / C0_SIZE; i++) {
                        AscendC::LoadData(
                            l0b_buf[i * m_round * C0_SIZE],
                            l1_src_a[i * C0_SIZE * M_round],
                            AscendC::LoadData2dParams(
                                0,                   // baseIdx
                                m_round / BLOCK_SIZE,  // repeat
                                1,                    // srcStride
                                0,                   // sid
                                0,                   // dstGap
                                false,               // transpose
                                inc                  // addr_cal_mode_t
                            )
                        );
                    }
                } else {
                    auto l1_src_a =
                        l1_buf_a[m_idx * tile_M0 * C0_SIZE + (k_idx * tile_K0 + k_part_idx * k_part_len) * M_round];
                    for (int32_t i = 0; i < m_round / BLOCK_SIZE; i++) {
                        AscendC::LoadData(
                            l0a_buf[i * k0_round * BLOCK_SIZE],
                            l1_src_a[i * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(
                                0,                   // baseIdx
                                k0_round / C0_SIZE,  // repeat
                                M_round / BLOCK_SIZE,  // srcStride
                                0,                   // sid
                                0,                   // dstGap
                                false,               // transpose
                                inc                  // addr_cal_mode_t
                            )
                        );
                    }
                }
                // *** load matrix B from L1 to L0B
                if (k_part_idx == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id + 2);
                }
                if (trans_c) {
                    auto l1_src_b = l1_buf_b[k_part_idx * k_part_len * tile_N0];
                    int32_t loop;
                    int32_t l0a_stride;
                    int32_t l1_src_b_stride;
                    int32_t repeat;
                    int32_t src_stride;
                    int32_t dst_stride;

                    if (k0_round / C0_SIZE < n_round / BLOCK_SIZE) {
                        loop = k0_round / C0_SIZE;
                        l0a_stride = CUBE_MATRIX_SIZE;
                        l1_src_b_stride = tile_N0 * C0_SIZE;
                        repeat = n_round / BLOCK_SIZE;
                        src_stride = 1;
                        dst_stride = k0_round / C0_SIZE - 1;
                    } else {
                        loop = n_round / BLOCK_SIZE;
                        l0a_stride = k0_round * BLOCK_SIZE;
                        l1_src_b_stride = CUBE_MATRIX_SIZE;
                        repeat = k0_round / C0_SIZE;
                        src_stride = tile_N0 / BLOCK_SIZE;
                        dst_stride = 0;
                    }
                    for (int32_t j = 0; j < batch_actual; j++) {
                        for (int32_t i = 0; i < loop; i++) {
                            AscendC::LoadData(
                                l0a_buf[j * k0_round * n_round + i * l0a_stride],
                                l1_src_b[j * tile_K0 * tile_N0 + i * l1_src_b_stride],
                                AscendC::LoadData2dParams(
                                    0,                   // baseIdx
                                    repeat,  // repeat
                                    src_stride,  // srcStride
                                    0,                   // sid
                                    dst_stride,                   // dstGap
                                    false,               // transpose
                                    inc                  // addr_cal_mode_t
                                )
                            );
                        }
                    }
                } else {
                    if (trans_b) {
                        auto l1_src_b = l1_buf_b[k_part_idx * k_part_len * tile_N0];
                        if (n_round == tile_N0) {
                            for (int32_t j = 0; j < batch_actual; j++) {
                                AscendC::LoadData(
                                    l0b_buf[j * k0_round * n_round],
                                    l1_src_b[j * tile_K0 * tile_N0],
                                    AscendC::LoadData2dParams(
                                        0,                   // baseIdx
                                        k0_round / C0_SIZE * n_round / BLOCK_SIZE,  // repeat
                                        1,  // srcStride
                                        0,                   // sid
                                        0,                   // dstGap
                                        false,               // transpose
                                        inc                  // addr_cal_mode_t
                                    )
                                );
                            }
                        } else {
                            for (int32_t j = 0; j < batch_actual; j++) {
                                for (int32_t i = 0; i < k0_round / C0_SIZE; i++) {
                                    AscendC::LoadData(
                                        l0b_buf[j * k0_round * n_round + i * n_round * C0_SIZE],
                                        l1_src_b[j * tile_K0 * tile_N0 + i * C0_SIZE * tile_N0],
                                        AscendC::LoadData2dParams(
                                            0,                   // baseIdx
                                            n_round / BLOCK_SIZE,  // repeat
                                            1,  // srcStride
                                            0,                   // sid
                                            0,                   // dstGap
                                            false,               // transpose
                                            inc                  // addr_cal_mode_t
                                        )
                                    );
                                }
                            }
                        }
                    } else {
                        auto l1_src_b = l1_buf_b[k_part_idx * k_part_len * tile_N0];
                        for (int32_t j = 0; j < batch_actual; j++) {
                            for (int32_t i = 0; i < (int32_t)(n_round / BLOCK_SIZE); i++) {
                                AscendC::LoadDataWithTranspose(
                                    l0b_buf[j * k0_round * n_round + i * CUBE_MATRIX_SIZE],
                                    l1_src_b[j * tile_K0 * tile_N0 + i * 2 * CUBE_MATRIX_SIZE],
                                    AscendC::LoadData2dTransposeParams(
                                        0,                             // indexID
                                        k0_round / BLOCK_SIZE,         // repeat
                                        tile_N0 / BLOCK_SIZE,               // srcStride
                                        2 * n_round / BLOCK_SIZE - 1,  // dstGap
                                        n_round / BLOCK_SIZE - 1,      // dstFracStride
                                        inc                           // addrmode
                                    )
                                );
                            }
                        }
                    }
                }
                if (k_part_idx == k_part_loop - 1) {
                    SET_FLAG(MTE1, MTE2, event_id + 2);
                }

                SET_FLAG(MTE1, M, mte1_mad_event_id);
                WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    WAIT_FLAG(FIX, M, l0c_EVENT_ID);
                }
                for (int32_t j = 0; j < batch_actual; j++) {
                    if (trans_c) {
                        AscendC::MmadParams mmParam = AscendC::MmadParams(
                            n_round,  // m
                            m_round,  // n
                            k0_actual, // k
                            0,
                            false,
                            init_c
                        );
                        mmParam.kDirectionAlign = true;

                        AscendC::Mmad(
                            l0c_buf[j * m_round * n_round], l0a_buf[j * k0_round * n_round], l0b_buf, mmParam
                        );
                    } else {
                        AscendC::MmadParams mmParam = AscendC::MmadParams(
                            m_round,  // m
                            n_actual,  // n
                            k0_actual, // k
                            0,
                            false,
                            init_c
                        );
                        mmParam.kDirectionAlign = true;

                        AscendC::Mmad(
                            l0c_buf[j * m_round * n_round], l0a_buf, l0b_buf[j * k0_round * n_round], mmParam
                        );
                    }
                }

                PIPE_BARRIER(M);
                SET_FLAG(M, MTE1, mte1_mad_event_id);
                mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
            }
            ping_flag = 1 - ping_flag;
        }
        SET_FLAG(M, FIX, l0c_EVENT_ID);
        WAIT_FLAG(M, FIX, l0c_EVENT_ID);

        if (is_sync) {
            WaitFlagDev(l0c_ping_flag + 2);
            if (trans_c) {
            } else {
                // 如果 L0AB_PINGPONG_BUFFER_LEN / (tile_K0 * tile_N0) >= 1 去掉矩阵的padding，便于虚实结合按照向量来操作
                if (step_index == step_len - 1 && L0AB_PINGPONG_BUFFER_LEN / (tile_K0 * tile_N0) >= 1) {
                    if (batch_actual > 1) {
                        uint64_t ndNum = batch_actual;
                        uint64_t srcNdStride = m_round * n_round / (16 * 16);
                        uint64_t dstNdStride = tile_M0 * tile_N0;
                        uint64_t config = ndNum | (srcNdStride << 16) | (dstNdStride << 32);
                        AscendC::SetNdParaImpl(config);
                    }
                    int64_t offset_c = (group_id * 2 + l0c_ping_flag) * batch_len * tile_M0 * tile_N0;

                    // copy from L0C to gm
                    copy_matrix_cc_to_gm(workspace + offset_c, reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf.GetPhyAddr())),
                                         0,         // sid
                                         n_actual,  // NSize
                                         m_actual,  // MSize
                                         n_actual,  // dstStride_dst_D
                                         m_round,   // srcStride
                                         0,         // UnitFlagMode
                                         NoQuant,   // QuantPRE
                                         0,         // ReLUPRE
                                         false,     // channelSplit
                                         true       // NZ2ND_EN
                    );
                } else {
                    if (batch_actual > 1) {
                        uint64_t ndNum = batch_actual;
                        uint64_t srcNdStride = m_round * n_round / (16 * 16);
                        uint64_t dstNdStride = tile_M0 * tile_N0;
                        uint64_t config = ndNum | (srcNdStride << 16) | (dstNdStride << 32);
                        AscendC::SetNdParaImpl(config);
                    }
                    int64_t offset_c = (group_id * 2 + l0c_ping_flag) * batch_len * tile_M0 * tile_N0;

                    // copy from L0C to gm
                    copy_matrix_cc_to_gm(workspace + offset_c, reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf.GetPhyAddr())),
                                         0,         // sid
                                         tile_N0,   // NSize
                                         m_actual,  // MSize
                                         tile_N0,   // dstStride_dst_D
                                         m_round,   // srcStride
                                         0,         // UnitFlagMode
                                         NoQuant,   // QuantPRE
                                         0,         // ReLUPRE
                                         false,     // channelSplit
                                         true       // NZ2ND_EN
                    );
                }
            }
            AscendC::SetNdParaImpl(0x1);
            FftsCrossCoreSync<PIPE_FIX, 2>(l0c_ping_flag + 4);
        } else {
            if (trans_c) {
                if (batch_actual > 1) {
                    uint64_t ndNum = batch_actual;
                    uint64_t srcNdStride = m_round * n_round / (16 * 16);
                    uint64_t dstNdStride = N * ldc;
                    uint64_t config = ndNum | (srcNdStride << 16) | (dstNdStride << 32);
                    AscendC::SetNdParaImpl(config);
                }
                int64_t offset_c = batch_idx * batch_len * N * ldc + m_idx * tile_M0 + n_idx * tile_N0 * ldc;
                // copy from L0C to gm
                copy_matrix_cc_to_gm(gm_c + offset_c, reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf.GetPhyAddr())),
                                     0,         // sid
                                     m_actual,  // NSize
                                     n_actual,  // MSize
                                     ldc,       // dstStride_dst_D
                                     n_round,   // srcStride
                                     0,         // UnitFlagMode
                                     NoQuant,   // QuantPRE
                                     0,         // ReLUPRE
                                     false,     // channelSplit
                                     true       // NZ2ND_EN
                );
            } else {
                if (batch_actual > 1) {
                    uint64_t ndNum = batch_actual;
                    uint64_t srcNdStride = m_round * n_round / (16 * 16);
                    uint64_t dstNdStride = M * ldc;
                    uint64_t config = ndNum | (srcNdStride << 16) | (dstNdStride << 32);
                    AscendC::SetNdParaImpl(config);
                }
                int64_t offset_c = batch_idx * batch_len * M * ldc + m_idx * tile_M0 * ldc + n_idx * tile_N0;
                // copy from L0C to gm
                copy_matrix_cc_to_gm(gm_c + offset_c, reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf.GetPhyAddr())),
                                     0,         // sid
                                     n_actual,  // NSize
                                     m_actual,  // MSize
                                     ldc,       // dstStride_dst_D
                                     m_round,   // srcStride
                                     0,         // UnitFlagMode
                                     NoQuant,   // QuantPRE
                                     0,         // ReLUPRE
                                     false,     // channelSplit
                                     true       // NZ2ND_EN
                );
            }
        }
        AscendC::SetNdParaImpl(0x1);
        l0c_ping_flag = 1 - l0c_ping_flag;

        SET_FLAG(FIX, M, l0c_EVENT_ID);
    }
    if (is_sync) {
        WaitFlagDev(2);
        WaitFlagDev(3);
    }

    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);

    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(M, MTE1, EVENT_ID3);

    WAIT_FLAG(FIX, M, EVENT_ID0);
    WAIT_FLAG(FIX, M, EVENT_ID1);
}

#endif
