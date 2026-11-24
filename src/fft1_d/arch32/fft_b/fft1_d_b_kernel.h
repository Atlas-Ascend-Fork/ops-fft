/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT1_D_B_KERNEL_H
#define FFT1_D_B_KERNEL_H

#include "kernel/fft_common_kernel.h"
// ASCENDFFT_FORWARD 定义来源（issue #445）：fft_common_kernel.h 的全局重复定义
// 已删除，统一引用 kernel/fft/kernel_utils.h 的单一定义（对齐 #303 整改方向）
#include "kernel/fft/kernel_utils.h"

namespace FFT1DBKernel {

constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t C0_SIZE = 32 / sizeof(float);
constexpr int32_t CUBE_MATRIX_SIZE = BLOCK_SIZE * C0_SIZE;
constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(float);
constexpr int32_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(float);
constexpr uint8_t INNER_GROUP_CV_SYNC_MODE = 2;
// fft_batch 合并数基准（issue #382）：n=8192 时合并 64 个 fft_batch，fft_n 更大时
// 按 fft_n / FFT_B_MERGE_N_BASE 反比递减
constexpr int32_t FFT_B_BATCH_GROUP_BASE = 64;
// bulk 分桶预算（issue #382）：batch_size_bulk = 该预算 / (fft_n / 256)，
// 使单个 bulk 覆盖的元素量恒为预算 * 256 点
constexpr int32_t FFT_B_BULK_SUBBLOCK_BUDGET = 12288;
// fft_batch 合并数阈值（issue #423）：fft_n 达到该值时 fft_batch_actual 固定为 1，
// 更小时按 FFT_B_BATCH_GROUP_BASE / (fft_n / 阈值) 反比递减；与选核、repeat 次数、
// 预取开关共享同一编号空间，调参必须同步对账
constexpr int32_t FFT_B_MERGE_N_BASE = 8192;

__aicore__ __inline__ void loop_tiling(int32_t batch_size, int32_t &loop_per_block, int32_t &loop_idx)
{
    loop_per_block = batch_size / block_num;
    int32_t loop_remain_block = batch_size % block_num;
    if (block_idx < loop_remain_block) {
        loop_per_block++;
        loop_idx = block_idx * loop_per_block;
    } else {
        loop_idx = block_idx * loop_per_block + loop_remain_block;
    }
}

__aicore__ __inline__ void step_ppmatmul_0_1(__gm__ float *__restrict__ gm_a, __gm__ float *__restrict__ gm_b,
                                             __gm__ float *__restrict__ gm_c, int32_t batch_size, int32_t iter,
                                             int32_t iter_count, int32_t fft_n, int32_t fft_batch_size, int32_t M,
                                             int32_t K, int32_t N, int32_t batch_id_begin, int32_t batch_id_end)
{
    if (fft_n == 0) {
        fft_n = 128;
    }
    if (N == 0) {
        N = 128;
    }

    SetPadding(0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    AscendC::GlobalTensor<float> gm_a_tensor;
    AscendC::GlobalTensor<float> gm_b_tensor;
    AscendC::GlobalTensor<float> gm_c_tensor;

    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_c));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> l1_base_a_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>(0);
    AscendC::LocalTensor<float> l1_base_b_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>(128 * 1024);

    AscendC::LocalTensor<float> l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, float>(0);
    AscendC::LocalTensor<float> l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, float>(0);
    AscendC::LocalTensor<float> l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    int32_t trans_a = 0;
    int32_t trans_b = 0;

    int32_t M0 = 128;
    int32_t N0 = 128;
    int32_t K0 = 128;
    if (M <= 64 && fft_n < FFT_B_MERGE_N_BASE) {
        M0 = 64;
        N0 = 256;
        K0 = 64;
    } else {
        M0 = 128;
        N0 = 128;
        K0 = 128;
    }

    int32_t m_loop = (M + M0 - 1) / M0;
    int32_t n_loop = (N + N0 - 1) / N0;
    int32_t k_loop = (K + K0 - 1) / K0;

    uint16_t flag_id = 0;
    uint64_t mode = 2;  // inner-group aic/aiv sync
    uint64_t ffts_config = 1 | (mode << 4) | (flag_id << 8);

    int32_t ping_flag = 1;
    int32_t aic_id = AscendC::GetBlockIdx();

    int32_t batch_id = batch_id_begin;
    int32_t batch_end = batch_id_end;
    int32_t N_init = N;
    int32_t batch_num = fft_n >= FFT_B_MERGE_N_BASE ? 1 : min(192 * 64 / fft_n, min(N0 / N_init, batch_size));

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);

    int32_t initG = 0;

    for (; batch_id < batch_end; batch_id += batch_num) {  // 一个核处理batch_num_sum个batch，一次处理batch_num个batch
        int32_t batch_actual = ((batch_id + batch_num) >= batch_end) ? (batch_end - batch_id) : batch_num;
        int32_t fft_multibatch = 1;
        if (fft_n == FFT_B_MERGE_N_BASE || fft_n == 16384) {
            fft_multibatch = 2;
        }
        N = N_init * batch_actual * fft_multibatch;
        int64_t offset_a, offset_b, offset_c;
        for (int32_t fft_batch_id = 0; fft_batch_id < fft_batch_size; ++fft_batch_id) {
            for (int32_t m_idx = 0; m_idx < m_loop; ++m_idx) {
                for (int32_t n_idx = 0; n_idx < n_loop; ++n_idx) {

                    int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;  // M
                    int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;  // N
                    int32_t m_round = (m_actual + 15) / 16 * 16;                         // 256
                    int32_t n_round = (n_actual + 15) / 16 * 16;

                    int32_t mn_max = m_round > n_round ? m_round : n_round;
                    int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;
                    for (int32_t k_idx = 0; k_idx < k_loop; k_idx++) {
                        if (trans_a) {
                            offset_a = static_cast<int64_t>(k_idx) * K0 * M + static_cast<int64_t>(m_idx) * M0;  // 注意fft的矩阵乘，w矩阵没有batch偏移
                        } else {
                            offset_a = static_cast<int64_t>(m_idx) * M0 * K + static_cast<int64_t>(k_idx) * K0;  // 注意fft的矩阵乘，w矩阵没有batch偏移
                        }

                        if (trans_b) {
                            offset_b = static_cast<int64_t>(batch_id) * fft_n * 2 +
                                       static_cast<int64_t>(fft_batch_id) * 128 * 128 +
                                       static_cast<int64_t>(n_idx) * N0 * K +
                                       static_cast<int64_t>(k_idx) * K0;
                        } else {
                            offset_b = static_cast<int64_t>(batch_id) * fft_n * 2 +
                                       static_cast<int64_t>(fft_batch_id) * 128 * 128 +
                                       static_cast<int64_t>(k_idx) * K0 * N +
                                       static_cast<int64_t>(n_idx) * N0;
                        }

                        if (fft_n > 16384) {
                            offset_c = static_cast<int64_t>(batch_id) * fft_n * 2 +
                                       static_cast<int64_t>(fft_batch_id) * M * N +
                                       static_cast<int64_t>(m_idx) * M0 * N +
                                       static_cast<int64_t>(n_idx) * N0;  // batch_id * fft_n * 2
                        } else {
                            offset_c = ping_flag ? static_cast<int64_t>(aic_id) * 128 * 256 : static_cast<int64_t>(aic_id) * 128 * 256 + 128 * 128;
                        }

                        int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;  // K 16
                        int32_t k_round = (k_actual + 15) / 16 * 16;
                        int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

                        auto l1_buf_a_tensor = l1_base_a_tensor;
                        auto l1_buf_b_tensor = ping_flag ? l1_base_b_tensor : l1_base_b_tensor[L1_PINGPONG_BUFFER_LEN];
                        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

                        // *** load matrix A to L1
                        WAIT_FLAG(MTE1, MTE2, event_id);
                        if (initG == 0) {
                            load_matrix_zZ<float>(l1_buf_a_tensor, gm_a_tensor[offset_a], K0, M0, k_actual, m_actual, M);
                            initG = 1;
                        }

                        SET_FLAG(MTE2, MTE1, event_id);

                        // *** load matrix B to L1
                        WAIT_FLAG(MTE1, MTE2, event_id + 2);
                        if (trans_b) {
                            load_matrix_zZ<float>(l1_buf_b_tensor, gm_b_tensor[offset_b], N0, K0, n_actual, k_actual, K);
                        } else {
                            load_matrix_zZ<float>(l1_buf_b_tensor, gm_b_tensor[offset_b], K0, N0, k_actual, n_actual, N);
                        }

                        SET_FLAG(MTE2, MTE1, event_id + 2);

                        SET_FLAG(M, MTE1, EVENT_ID0);
                        SET_FLAG(M, MTE1, EVENT_ID1);
                        for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                            int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len :
                                                                                k_round - k_part_idx * k_part_len;
                            int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len :
                                                                                 k_actual - k_part_idx * k_part_len;

                            auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
                            auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                            auto l0a_buf_tensor = l0a_base_tensor[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];
                            auto l0b_buf_tensor = l0b_base_tensor[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];

                            // *** load matrix A from L1 to L0A
                            if (k_part_idx == 0) {
                                WAIT_FLAG(MTE2, MTE1, event_id);
                            }
                            WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                            if (M == 1 || m_actual == 1 && !trans_a) {
                                AscendC::LoadData(
                                    l0a_buf_tensor,
                                    l1_buf_a_tensor[k_part_idx * k_part_len],
                                    AscendC::LoadData2dParams(
                                        0,                                                      // baseIdx
                                        (k0_round + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,   // repeat
                                        1,                                                      // srcStride
                                        0,                                                      // sid
                                        0,                                                      // dstStride
                                        false,                                                  // transpose
                                        inc                                                     // addr_cal_mode_t
                                    )
                                );
                            } else {
                                if (trans_a) {
                                    auto l1_src_a_tensor = l1_buf_a_tensor[k_part_idx * k_part_len * M0]; // new
                                    for (int32_t i = 0; i < m_round / BLOCK_SIZE; i++) {
                                        AscendC::LoadDataWithTranspose(
                                            l0a_buf_tensor[i * k0_round * BLOCK_SIZE],
                                            l1_src_a_tensor[i * 2 * CUBE_MATRIX_SIZE],
                                            AscendC::LoadData2dTransposeParams(
                                                0,                             // indexID
                                                k0_round / BLOCK_SIZE,         // repeat
                                                M0 / BLOCK_SIZE,               // srcStride
                                                1,                             // dstGap
                                                0,                             // dstFracStride
                                                inc                           // addrmode
                                            )
                                        );
                                    }

                                } else {
                                    auto l1_src_a_tensor = l1_buf_a_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                                    for (int32_t i = 0; i < m_round / BLOCK_SIZE; i++) {
                                        AscendC::LoadData(
                                            l0a_buf_tensor[i * k0_round * BLOCK_SIZE],
                                            l1_src_a_tensor[i * BLOCK_SIZE * K0],
                                            AscendC::LoadData2dParams(
                                                0,                                                      // baseIdx
                                                k0_round / C0_SIZE,   // repeat
                                                1,                                                      // srcStride
                                                0,                                                      // sid
                                                0,                                                      // dstStride
                                                false,                                                  // transpose
                                                inc                                                     // addr_cal_mode_t
                                            )
                                        );
                                    }
                                }
                            }
                            if (k_part_idx == k_part_loop - 1) {
                                SET_FLAG(MTE1, MTE2, event_id);
                            }

                            // *** load matrix B from L1 to L0B
                            if (k_part_idx == 0) {
                                WAIT_FLAG(MTE2, MTE1, event_id + 2);
                            }
                            if (trans_b) {
                                auto l1_src_b_tensor = l1_buf_b_tensor[k_part_idx * k_part_len * BLOCK_SIZE];

                                for (int32_t i = 0; i < k0_round / C0_SIZE; i++) {
                                    AscendC::LoadData(
                                        l0b_buf_tensor[i * n_round * C0_SIZE],
                                        l1_src_b_tensor[i * CUBE_MATRIX_SIZE],
                                        AscendC::LoadData2dParams(
                                            0,                                                      // baseIdx
                                            n_round / BLOCK_SIZE,   // repeat
                                            K0 / C0_SIZE,                                           // srcStride
                                            0,                                                      // sid
                                            0,                                                      // dstStride
                                            false,                                                  // transpose
                                            inc                                                     // addr_cal_mode_t
                                        )
                                    );
                                }
                            } else {
                                auto l1_src_b_tensor = l1_buf_b_tensor[k_part_idx * k_part_len * N0];
                                for (int32_t i = 0; i < n_round / BLOCK_SIZE; i++) {
                                    AscendC::LoadDataWithTranspose(
                                        l0b_buf_tensor[i * CUBE_MATRIX_SIZE],
                                        l1_src_b_tensor[i * 2 * CUBE_MATRIX_SIZE],
                                        AscendC::LoadData2dTransposeParams(
                                            0,                             // indexID
                                            k0_round / BLOCK_SIZE,         // repeat
                                            N0 / BLOCK_SIZE,               // srcStride
                                            2 * n_round / BLOCK_SIZE - 1,  // dstGap
                                            n_round / BLOCK_SIZE - 1,      // dstFracStride
                                            inc                           // addrmode
                                        )
                                    );
                                }
                            }
                            if (k_part_idx == k_part_loop - 1) {
                                SET_FLAG(MTE1, MTE2, event_id + 2);
                            }

                            SET_FLAG(MTE1, M, mte1_mad_event_id);
                            WAIT_FLAG(MTE1, M, mte1_mad_event_id);
                            bool init_c = (k_idx == 0 && k_part_idx == 0);
                            uint8_t unit_flag = 0b10;
                            if (k_idx == k_loop - 1 && k_part_idx == k_part_loop - 1) {
                                unit_flag = 0b11;
                            }

                            if (M != 1 && m_actual == 1 && trans_a) {
                                AscendC::MmadParams mmParam = AscendC::MmadParams(
                                    16, n_actual, k0_actual,
                                    0,
                                    false,
                                    init_c
                                );
                                mmParam.kDirectionAlign = true;
                                mmParam.unitFlag = unit_flag;
                                AscendC::Mmad(
                                    l0c_buf_tensor, l0a_buf_tensor, l0b_buf_tensor, mmParam
                                );
                            } else {
                                AscendC::MmadParams mmParam = AscendC::MmadParams(
                                    m_actual, n_actual, k0_actual,
                                    0,
                                    false,
                                    init_c
                                );
                                mmParam.kDirectionAlign = true;
                                mmParam.unitFlag = unit_flag;
                                AscendC::Mmad(
                                    l0c_buf_tensor, l0a_buf_tensor, l0b_buf_tensor, mmParam
                                );
                            }
                            PIPE_BARRIER(M);
                            SET_FLAG(M, MTE1, mte1_mad_event_id);
                        }

                        WAIT_FLAG(M, MTE1, EVENT_ID0);
                        WAIT_FLAG(M, MTE1, EVENT_ID1);
                    }

                    flag_id = ping_flag ? 5 : 7;
                    WaitFlagDev(flag_id);

                    auto intriParams = AscendC::FixpipeParamsV220(n_actual, // nSize
                                                      m_actual, // mSize
                                                      m_round,   // srcStride
                                                      N,   // dstStride
                                                      false);      // enRelu

                    intriParams.quantPre = QuantMode_t::NoQuant;
                    intriParams.unitFlag = 0b11;
                    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gm_c_tensor[offset_c], l0c_buf_tensor, intriParams);

                    flag_id = ping_flag ? 3 : 6;
                    FftsCrossCoreSync<PIPE_FIX, INNER_GROUP_CV_SYNC_MODE>(flag_id);

                    ping_flag = 1 - ping_flag;  // ought to be in 2 layer loop
                }
            }
        }
    }
    flag_id = 5;
    WaitFlagDev(flag_id);

    flag_id = 7;
    WaitFlagDev(flag_id);

    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
}

__aicore__ __inline__ void step_ppmatmul_2itercount_last_iter(__gm__ float *__restrict__ gm_a,
                                                              __gm__ float *__restrict__ gm_b,
                                                              __gm__ float *__restrict__ gm_c, int32_t batch_size,
                                                              int32_t iter, int32_t iter_count, int32_t fft_n,
                                                              int32_t fft_batch_size, int32_t M, int32_t K, int32_t N,
                                                              int32_t batch_id_begin, int32_t batch_id_end)
{
    if (fft_n == 0) {
        fft_n = 128;
    }
    if (N == 0) {
        N = 128;
    }

    SetPadding(0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    AscendC::GlobalTensor<float> gm_a_tensor;
    AscendC::GlobalTensor<float> gm_b_tensor;
    AscendC::GlobalTensor<float> gm_c_tensor;

    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_c));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> l1_base_a_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>(0);
    AscendC::LocalTensor<float> l1_base_b_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>(128 * 1024);

    AscendC::LocalTensor<float> l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, float>(0);
    AscendC::LocalTensor<float> l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, float>(0);
    AscendC::LocalTensor<float> l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    auto l1_base_a = reinterpret_cast<__cbuf__ float *>((uintptr_t)(l1_base_a_tensor.GetPhyAddr()));
    auto l1_base_b = reinterpret_cast<__cbuf__ float *>((uintptr_t)(l1_base_b_tensor.GetPhyAddr()));

    auto l0a_base = reinterpret_cast<__ca__ float *>((uintptr_t)(l0a_base_tensor.GetPhyAddr()));
    auto l0b_base = reinterpret_cast<__cb__ float *>((uintptr_t)(l0b_base_tensor.GetPhyAddr()));
    auto l0c_buf = reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf_tensor.GetPhyAddr()));

    int32_t trans_a = 0;
    int32_t trans_b = 1;
    int32_t M0 = 64;
    int32_t N0 = 256;
    int32_t K0 = 64;
    if (M <= 64 && fft_n < FFT_B_MERGE_N_BASE) {
        M0 = 64;
        N0 = 256;
        K0 = 64;
    } else {
        M0 = 128;
        N0 = 128;
        K0 = 128;
    }

    int32_t m_loop = (M + M0 - 1) / M0;
    int32_t n_loop = (N + N0 - 1) / N0;
    int32_t k_loop = (K + K0 - 1) / K0;

    uint16_t flag_id = 0;
    uint64_t mode = 2;  // inner-group aic/aiv sync
    uint64_t ffts_config = 1 | (mode << 4) | (flag_id << 8);

    int32_t ping_flag = 1;
    int32_t aic_id = AscendC::GetBlockIdx();

    int32_t batch_id = batch_id_begin;
    int32_t batch_end = min(batch_id_end, batch_id + batch_size);
    int32_t N_init = N;
    int32_t batch_num = fft_n >= FFT_B_MERGE_N_BASE ? 1 : min(192 * 64 / fft_n, N0 / N_init);
    int32_t fft_batch_actual = fft_n >= FFT_B_MERGE_N_BASE ? FFT_B_BATCH_GROUP_BASE / (fft_n / FFT_B_MERGE_N_BASE) : 1;

    // M:128 K:128 N:2
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);

    for (; batch_id < batch_end; batch_id += batch_num) {  // 一个核处理batch_num_sum个batch，一次处理batch_num个batch
        int32_t batch_actual = ((batch_id + batch_num) >= batch_end) ? (batch_end - batch_id) : batch_num;
        N = N_init * batch_num * fft_batch_actual;
        int64_t offset_a, offset_b, offset_c;  // variables for matmul

        for (int32_t fft_batch_id = 0; fft_batch_id < fft_batch_size; ++fft_batch_id) {
            for (int32_t m_idx = 0; m_idx < m_loop; ++m_idx) {
                for (int32_t n_idx = 0; n_idx < n_loop; ++n_idx) {
                    int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
                    int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
                    int32_t m_round = (m_actual + 15) / 16 * 16;
                    int32_t n_round = (n_actual + 15) / 16 * 16;

                    int32_t mn_max = m_round > n_round ? m_round : n_round;
                    int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;
                    for (int32_t k_idx = 0; k_idx < k_loop; k_idx++) {
                        if (trans_a) {
                            offset_a = static_cast<int64_t>(k_idx) * K0 * M + static_cast<int64_t>(m_idx) * M0;  // 注意fft的矩阵乘，w矩阵没有batch偏移
                        } else {
                            offset_a = static_cast<int64_t>(m_idx) * M0 * K + static_cast<int64_t>(k_idx) * K0;  // 注意fft的矩阵乘，w矩阵没有batch偏移
                        }

                        if (trans_b) {
                            offset_b = static_cast<int64_t>(batch_id) * fft_n * 2 + static_cast<int64_t>(fft_batch_id) * K * N +
                                       static_cast<int64_t>(n_idx) * N0 * K + static_cast<int64_t>(k_idx) * K0;
                        } else {
                            offset_b = static_cast<int64_t>(batch_id) * fft_n * 2 + static_cast<int64_t>(fft_batch_id) * K * N +
                                       static_cast<int64_t>(k_idx) * K0 * N + static_cast<int64_t>(n_idx) * N0;
                        }

                        offset_c = ping_flag ? static_cast<int64_t>(aic_id) * 128 * 256 : static_cast<int64_t>(aic_id) * 128 * 256 + 128 * 128;

                        int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;
                        int32_t k_round = (k_actual + 15) / 16 * 16;
                        int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

                        auto l1_buf_a_tensor = ping_flag ? l1_base_a_tensor : l1_base_a_tensor[L1_PINGPONG_BUFFER_LEN];
                        auto l1_buf_b_tensor = ping_flag ? l1_base_b_tensor : l1_base_b_tensor[L1_PINGPONG_BUFFER_LEN];
                        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

                        // *** load matrix A to L1
                        WAIT_FLAG(MTE1, MTE2, event_id);
                        if (trans_a) {
                            if (M == 1) {
                                AscendC::DataCopy(
                                    l1_buf_a_tensor, gm_a_tensor[offset_a],
                                    AscendC::DataCopyParams(
                                        1,                                      // nBurst
                                        (k_actual + C0_SIZE - 1) / C0_SIZE,     // lenBurst
                                        0,                                      // srcGap
                                        0                                       // dstGap
                                    )
                                );

                            } else {
                                load_matrix_zZ<float>(l1_buf_a_tensor, gm_a_tensor[offset_a], K0, M0, k_actual, m_actual, M);
                            }
                        } else {
                            if (m_actual == 1) {
                                AscendC::DataCopy(
                                    l1_buf_a_tensor, gm_a_tensor[offset_a],
                                    AscendC::DataCopyParams(
                                        1,                                      // nBurst
                                        (k_actual + C0_SIZE - 1) / C0_SIZE,     // lenBurst
                                        0,                                      // srcGap
                                        0                                       // dstGap
                                    )
                                );
                            } else {
                                load_matrix_zZ<float>(l1_buf_a_tensor, gm_a_tensor[offset_a], M0, K0, m_actual, k_actual, K);
                            }
                        }
                        SET_FLAG(MTE2, MTE1, event_id);

                        // *** load matrix B to L1
                        WAIT_FLAG(MTE1, MTE2, event_id + 2);
                        if (trans_b) {
                            load_matrix_zZ<float>(l1_buf_b_tensor, gm_b_tensor[offset_b], N0, K0, n_actual, k_actual, K);
                        } else {
                            load_matrix_zZ<float>(l1_buf_b_tensor, gm_b_tensor[offset_b], K0, N0, k_actual, n_actual, N);
                        }

                        SET_FLAG(MTE2, MTE1, event_id + 2);

                        SET_FLAG(M, MTE1, EVENT_ID0);
                        SET_FLAG(M, MTE1, EVENT_ID1);
                        for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                            int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len :
                                                                                k_round - k_part_idx * k_part_len;
                            int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len :
                                                                                 k_actual - k_part_idx * k_part_len;

                            auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
                            auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                            auto l0a_buf = l0a_base + (k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN;
                            auto l0a_buf_tensor = l0a_base_tensor[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];
                            auto l0b_buf = l0b_base + (k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN;
                            auto l0b_buf_tensor = l0b_base_tensor[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];

                            // *** load matrix A from L1 to L0A
                            if (k_part_idx == 0) {
                                WAIT_FLAG(MTE2, MTE1, event_id);
                            }
                            WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                            if (M == 1 || m_actual == 1 && !trans_a) {
                                AscendC::LoadData(
                                    l0a_buf_tensor,
                                    l1_buf_a_tensor[k_part_idx * k_part_len],
                                    AscendC::LoadData2dParams(
                                        0,                                                      // baseIdx
                                        (k0_round + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,   // repeat
                                        1,                                                      // srcStride
                                        0,                                                      // sid
                                        0,                                                      // dstStride
                                        false,                                                  // transpose
                                        inc                                                     // addr_cal_mode_t
                                    )
                                );

                            } else {
                                if (trans_a) {
                                    auto l1_src_a_tensor = l1_buf_a_tensor[k_part_idx * k_part_len * M0];
                                    for (int32_t i = 0; i < m_round / BLOCK_SIZE; i++) {
                                        AscendC::LoadDataWithTranspose(
                                            l0a_buf_tensor[i * k0_round * BLOCK_SIZE],
                                            l1_src_a_tensor[i * 2 * CUBE_MATRIX_SIZE],
                                            AscendC::LoadData2dTransposeParams(
                                                0,                             // indexID
                                                k0_round / BLOCK_SIZE,         // repeat
                                                M0 / BLOCK_SIZE,               // srcStride
                                                1,                             // dstGap
                                                0,                             // dstFracStride
                                                inc                           // addrmode
                                            )
                                        );
                                    }
                                } else {
                                    auto l1_src_a_tensor = l1_buf_a_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                                    for (int32_t i = 0; i < m_round / BLOCK_SIZE; i++) {
                                        AscendC::LoadData(
                                            l0a_buf_tensor[i * k0_round * BLOCK_SIZE],
                                            l1_src_a_tensor[i * BLOCK_SIZE * K0],
                                            AscendC::LoadData2dParams(
                                                0,                                                      // baseIdx
                                                k0_round / C0_SIZE,   // repeat
                                                1,                                                      // srcStride
                                                0,                                                      // sid
                                                0,                                                      // dstStride
                                                false,                                                  // transpose
                                                inc                                                     // addr_cal_mode_t
                                            )
                                        );
                                    }
                                }
                            }
                            if (k_part_idx == k_part_loop - 1) {
                                SET_FLAG(MTE1, MTE2, event_id);
                            }

                            // *** load matrix B from L1 to L0B
                            if (k_part_idx == 0) {
                                WAIT_FLAG(MTE2, MTE1, event_id + 2);
                            }
                            if (trans_b) {
                                auto l1_src_b_tensor = l1_buf_b_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                                for (int32_t i = 0; i < k0_round / C0_SIZE; i++) {
                                    AscendC::LoadData(
                                        l0b_buf_tensor[i * n_round * C0_SIZE],
                                        l1_src_b_tensor[i * CUBE_MATRIX_SIZE],
                                        AscendC::LoadData2dParams(
                                            0,                                                      // baseIdx
                                            n_round / BLOCK_SIZE,   // repeat
                                            K0 / C0_SIZE,                                           // srcStride
                                            0,                                                      // sid
                                            0,                                                      // dstStride
                                            false,                                                  // transpose
                                            inc                                                     // addr_cal_mode_t
                                        )
                                    );
                                }
                            } else {
                                auto l1_src_b_tensor = l1_buf_b_tensor[k_part_idx * k_part_len * N0];
                                for (int32_t i = 0; i < n_round / BLOCK_SIZE; i++) {
                                    AscendC::LoadDataWithTranspose(
                                        l0b_buf_tensor[i * CUBE_MATRIX_SIZE],
                                        l1_src_b_tensor[i * 2 * CUBE_MATRIX_SIZE],
                                        AscendC::LoadData2dTransposeParams(
                                            0,                             // indexID
                                            k0_round / BLOCK_SIZE,         // repeat
                                            N0 / BLOCK_SIZE,               // srcStride
                                            2 * n_round / BLOCK_SIZE - 1,  // dstGap
                                            n_round / BLOCK_SIZE - 1,      // dstFracStride
                                            inc                           // addrmode
                                        )
                                    );
                                }
                            }
                            if (k_part_idx == k_part_loop - 1) {
                                SET_FLAG(MTE1, MTE2, event_id + 2);
                            }

                            SET_FLAG(MTE1, M, mte1_mad_event_id);
                            WAIT_FLAG(MTE1, M, mte1_mad_event_id);
                            bool init_c = (k_idx == 0 && k_part_idx == 0);
                            uint8_t unit_flag = 0b10;
                            if (k_idx == k_loop - 1 && k_part_idx == k_part_loop - 1) {
                                unit_flag = 0b11;
                            }

                            if (M != 1 && m_actual == 1 && trans_a) {
                                AscendC::MmadParams mmParam = AscendC::MmadParams(
                                    16, n_actual, k0_actual,
                                    0,
                                    false,
                                    init_c
                                );
                                mmParam.kDirectionAlign = true;
                                mmParam.unitFlag = unit_flag;
                                AscendC::Mmad(
                                    l0c_buf_tensor, l0a_buf_tensor, l0b_buf_tensor, mmParam
                                );
                            } else {
                                AscendC::MmadParams mmParam = AscendC::MmadParams(
                                    m_actual, n_actual, k0_actual,
                                    0,
                                    false,
                                    init_c
                                );
                                mmParam.kDirectionAlign = true;
                                mmParam.unitFlag = unit_flag;
                                AscendC::Mmad(
                                    l0c_buf_tensor, l0a_buf_tensor, l0b_buf_tensor, mmParam
                                );
                            }
                            PIPE_BARRIER(M);
                            SET_FLAG(M, MTE1, mte1_mad_event_id);
                        }
                        WAIT_FLAG(M, MTE1, EVENT_ID0);
                        WAIT_FLAG(M, MTE1, EVENT_ID1);
                    }

                    flag_id = ping_flag ? 5 : 7;
                    WaitFlagDev(flag_id);

                    auto intriParams = AscendC::FixpipeParamsV220(n_actual, // nSize
                                                      m_actual, // mSize
                                                      m_round,   // srcStride
                                                      N,   // dstStride
                                                      false);      // enRelu

                    intriParams.quantPre = QuantMode_t::NoQuant;
                    intriParams.unitFlag = 0b11;
                    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gm_c_tensor[offset_c], l0c_buf_tensor, intriParams);

                    flag_id = ping_flag ? 3 : 6;
                    FftsCrossCoreSync<PIPE_FIX, INNER_GROUP_CV_SYNC_MODE>(flag_id);

                    ping_flag = 1 - ping_flag;
                }
            }
        }
    }

    flag_id = 5;
    WaitFlagDev(flag_id);

    flag_id = 7;
    WaitFlagDev(flag_id);

    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
}

__aicore__ __inline__ void step_reduce(__gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_output,
                                       int32_t batch_size, int32_t N, int32_t N_radix, int32_t batch_id_begin,
                                       int32_t batch_id_end)
{
    if (N == 0) {
        N = 1;
    }
    if (N_radix == 0) {
        N_radix = 1;
    }

    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    AscendC::GlobalTensor<float> gm_input_tensor;
    AscendC::GlobalTensor<float> gm_output_tensor;

    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    constexpr int32_t element_num_per_block = 8;
    int32_t max_repeat_count = 192;  // 48 * 1024 / 4 / 64 48KB的数据划分最大容纳的 序列长度/64

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> complex_buf1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> complex_buf2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(48 * 1024));
    AscendC::LocalTensor<float> real_buf1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(96 * 1024));
    AscendC::LocalTensor<float> real_buf2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)((96 + 24) * 1024));
    AscendC::LocalTensor<float> imag_buf1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)((96 + 48) * 1024));
    AscendC::LocalTensor<float> imag_buf2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)((96 + 72) * 1024));

    auto complex_buf1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(complex_buf1_tensor.GetPhyAddr()));
    auto complex_buf2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(complex_buf1_tensor.GetPhyAddr()));
    auto real_buf1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(real_buf1_tensor.GetPhyAddr()));
    auto real_buf2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(real_buf2_tensor.GetPhyAddr()));
    auto imag_buf1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(imag_buf1_tensor.GetPhyAddr()));
    auto imag_buf2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(imag_buf2_tensor.GetPhyAddr()));

    int32_t aiv_id = AscendC::GetSubBlockIdx();
    int32_t per_aiv_core_element_num = N;

    int32_t N0 = 128;
    if (N <= 2048) {
        N0 = 256;  // 多Batch合并要求, 数据排布不同
    }

    // 计算每个核要处理的batch数量
    int32_t batch_id = batch_id_begin;
    int32_t batch_end = N >= FFT_B_MERGE_N_BASE ? min(batch_id_end, batch_id + batch_size) : batch_id_end;
    int32_t batch_num =
        N >= FFT_B_MERGE_N_BASE ? 1 : min(max_repeat_count * 64 / N, min(N0 / N_radix, batch_size));  // 一次迭代处理的batch数量

    max_repeat_count = N >= FFT_B_MERGE_N_BASE ? 192 : batch_num * N / 64;

    int32_t ping_flag = 1;

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (; batch_id < batch_end; batch_id += batch_num) {  // 一个核处理batch_num_sum个batch，一次处理batch_num个batch
        int32_t batch_actual = ((batch_id + batch_num) >= batch_end) ? (batch_end - batch_id) : batch_num;
        int32_t all_repeats = batch_actual * N / 64;
        int32_t repeat = N >= FFT_B_MERGE_N_BASE ? 192 : batch_num * N / 64;
        int32_t loop = (all_repeats + max_repeat_count - 1) / max_repeat_count;

        for (int32_t i = 0; i < loop; ++i) {
            if (i == loop - 1) {
                repeat = all_repeats - (loop - 1) * max_repeat_count;
            }
            auto complex_buf = ping_flag ? complex_buf1 : complex_buf2;
            auto complex_buf_tensor = ping_flag ? complex_buf1_tensor : complex_buf2_tensor;
            auto real_buf = ping_flag ? real_buf1 : real_buf2;
            auto real_buf_tensor = ping_flag ? real_buf1_tensor : real_buf2_tensor;
            auto imag_buf = ping_flag ? imag_buf1 : imag_buf2;
            auto imag_buf_tensor = ping_flag ? imag_buf1_tensor : imag_buf2_tensor;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            WAIT_FLAG(MTE3, MTE2, event_id);

            if (N < FFT_B_MERGE_N_BASE) {
                for (int32_t j = 0; j < batch_actual; ++j) {
                    AscendC::DataCopy(
                        complex_buf_tensor[j * N_radix * 2],
                        gm_input_tensor[batch_id * N * 2 + aiv_id * N + j * N * 2 + i * max_repeat_count * 64 * 2],
                        AscendC::DataCopyParams(
                            N / N_radix / 2, // blockCount
                            N_radix / 4, // blockLen
                            0, // srcStride
                            (batch_actual - 1) * N_radix * 2 / 8 // dstStride
                        )
                    );
                }
            } else {
                AscendC::DataCopy(
                    complex_buf_tensor,
                    gm_input_tensor[batch_id * N * 2 + aiv_id * N + i * max_repeat_count * 64],
                    AscendC::DataCopyParams(
                        1, // blockCount
                        repeat * 8, // blockLen
                        0, // srcStride
                        0 // dstStride
                    )
                );
            }
            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);

            uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数

            AscendC::GatherMask(
                real_buf_tensor,
                complex_buf_tensor,
                (uint8_t)1, // src1Pattern
                false, // reduceMode
                0, // mask
                AscendC::GatherMaskParams(1, static_cast<uint8_t>(repeat), 8, 8),
                rsvdCnt
            );

            AscendC::GatherMask(
                imag_buf_tensor,
                complex_buf_tensor,
                (uint8_t)2, // src1Pattern
                false, // reduceMode
                0, // mask
                AscendC::GatherMaskParams(1, static_cast<uint8_t>(repeat), 8, 8),
                rsvdCnt
            );

            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);

            if (N < FFT_B_MERGE_N_BASE) {
                AscendC::DataCopy(
                    gm_output_tensor[batch_id * N * 2 + aiv_id * N / 2 * batch_actual + i * max_repeat_count * 64 * 2],
                    real_buf_tensor,
                    AscendC::DataCopyParams(
                        1, // blockCount
                        repeat * 8 / 2, // blockLen
                        0, // srcStride
                        0 // dstStride
                    )
                );
                AscendC::DataCopy(
                    gm_output_tensor[batch_id * N * 2 + N * batch_actual + aiv_id * N / 2 * batch_actual +
                                    i * max_repeat_count * 64 * 2],
                    imag_buf_tensor,
                    AscendC::DataCopyParams(
                        1, // blockCount
                        repeat * 8 / 2, // blockLen
                        0, // srcStride
                        0 // dstStride
                    )
                );
            } else {
                AscendC::DataCopy(
                    gm_output_tensor[batch_id * N * 2 + aiv_id * N / 2 + i * max_repeat_count * 64 / 2],
                    real_buf_tensor,
                    AscendC::DataCopyParams(
                        1, // blockCount
                        repeat * 8 / 2, // blockLen
                        0, // srcStride
                        0 // dstStride
                    )
                );

                AscendC::DataCopy(
                    gm_output_tensor[batch_id * N * 2 + N + aiv_id * N / 2 + i * max_repeat_count * 64 / 2],
                    imag_buf_tensor,
                    AscendC::DataCopyParams(
                        1, // blockCount
                        repeat * 8 / 2, // blockLen
                        0, // srcStride
                        0 // dstStride
                    )
                );
            }

            SET_FLAG(MTE3, MTE2, event_id);
            ping_flag = 1 - ping_flag;
        }
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

__aicore__ __inline__ void SeparateRIwithRadix2(__gm__ float *__restrict__ gm_input,
                                                __gm__ float *__restrict__ gm_t_matrix,
                                                __gm__ float *__restrict__ gm_output, int32_t fftDirection,
                                                int32_t batch_size, int32_t inputN, int32_t M, int32_t N,
                                                int32_t N_radix,  // 下一次迭代的batch合并维度
                                                int32_t batch_id_begin, int32_t batch_id_end)
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    int32_t batch_num = 1;  // 下一次迭代的batch合并数量
    // restrict M to 4 (radix 2)
    int32_t N0 = 2048;  // 2 * (2048 * 2) * 4B = 32KB

    AscendC::GlobalTensor<float> gm_input_tensor;
    AscendC::GlobalTensor<float> gm_t_matrix_tensor;
    AscendC::GlobalTensor<float> gm_output_tensor;

    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_t_matrix_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_t_matrix));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> input1_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> input1_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(32 * 1024));
    AscendC::LocalTensor<float> input2_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(64 * 1024));
    AscendC::LocalTensor<float> input2_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(96 * 1024));
    AscendC::LocalTensor<float> output_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(128 * 1024));
    AscendC::LocalTensor<float> output_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(160 * 1024));

    auto input1_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_ping_tensor.GetPhyAddr()));
    auto input1_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_pong_tensor.GetPhyAddr()));
    auto input2_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_ping_tensor.GetPhyAddr()));
    auto input2_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_pong_tensor.GetPhyAddr()));
    auto output_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_ping_tensor.GetPhyAddr()));
    auto output_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_pong_tensor.GetPhyAddr()));

    int32_t ping_flag = 1;

    int32_t aiv_id = AscendC::GetSubBlockIdx();

    // 计算每个核要处理的batch数量
    int32_t batch_id = batch_id_begin;
    int32_t batch_end = batch_id_end;

    // load t_matrix
    AscendC::DataCopy(
        input2_ping_tensor,
        gm_t_matrix_tensor[inputN + aiv_id * N / 2],
        AscendC::DataCopyParams(
            2, // blockCount
            N0 / 8, // blockLen
            (N - N0) / 8, // srcStride
            0 // dstStride
        )
    );

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);
    for (; batch_id < batch_end; batch_id += batch_num) {  // 一个核处理batch_num_sum个batch，一次处理batch_num个batch
        int32_t batch_actual = 1;
        for (int32_t n_idx = 0; n_idx < 1; n_idx++) {  // N / 2 means 2 AIV
            auto input1_ub = ping_flag ? input1_ping : input1_pong;
            auto input1_ub_tensor = ping_flag ? input1_ping_tensor : input1_pong_tensor;
            auto input2_ub = input2_ping;
            auto input2_ub_tensor = input2_ping_tensor;
            auto output_ub = ping_flag ? output_ping : output_pong;
            auto output_ub_tensor = ping_flag ? output_ping_tensor : output_pong_tensor;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            int32_t fft_batch_actual = 2;

            int32_t offset_input1 =
                batch_id * 2 * inputN + n_idx * N0 * 2 + aiv_id * N / 2 * 2;  // AIV 0 负责前一半， AIV 1负责后一半
            int32_t offset_input2 = inputN + aiv_id * N / 2 + n_idx * N0;  // inputN表示只读入后一半
            int32_t offset_output_R = batch_id * 2 * inputN + n_idx * N0 * fft_batch_actual +
                                      aiv_id * fft_batch_actual * N / 2;  // batch_actual为1时 2048
            int32_t offset_output_I = batch_id * 2 * inputN + n_idx * N0 * fft_batch_actual +
                                      aiv_id * fft_batch_actual * N / 2 + fft_batch_actual * inputN / 2;

            WAIT_FLAG(MTE3, MTE2, event_id);

            // load input data, radix-2 so 8192 -> 4096
            AscendC::DataCopy(
                input1_ub_tensor,
                gm_input_tensor[offset_input1],
                AscendC::DataCopyParams(
                    2 * N0 / 128, // blockCount
                    128 / 8, // blockLen
                    0, // srcStride
                    128 / 8 // dstStride
                )
            );

            AscendC::DataCopy(
                input1_ub_tensor[128],
                gm_input_tensor[offset_input1 + 2 * N0 * 2],
                AscendC::DataCopyParams(
                    2 * N0 / 128, // blockCount
                    128 / 8, // blockLen
                    0, // srcStride
                    128 / 8 // dstStride
                )
            );

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);

            // separate RI
            int32_t total_len = M / 2 * N0 * 2;  // M / 2 means 2 fft batch, 2 means RI
            int32_t repeat = total_len / 64;     // M / 2 means 2 fft batch, 2 means RI

            uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数

            AscendC::GatherMask(
                output_ub_tensor,
                input1_ub_tensor,
                (uint8_t)1, // src1Pattern
                false, // reduceMode
                0, // mask
                AscendC::GatherMaskParams(1, static_cast<uint8_t>(repeat), 8, 8),
                rsvdCnt
            );

            AscendC::GatherMask(
                output_ub_tensor[batch_actual * M / 2 * N0],
                input1_ub_tensor,
                (uint8_t)2, // src1Pattern
                false, // reduceMode
                0, // mask
                AscendC::GatherMaskParams(1, static_cast<uint8_t>(repeat), 8, 8),
                rsvdCnt
            );

            PIPE_BARRIER(V);

            // radix-2 matmul
            int32_t repeats = batch_actual * N0 / 64;
            uint64_t vecMask[2] = { UINT64_MAX, 0 };

            AscendC::Add(input1_ub_tensor, output_ub_tensor, output_ub_tensor[64],
                vecMask, repeats, {1, 1, 1, 16, 16, 16}
            ); // R1 + R2

            AscendC::Add(input1_ub_tensor[2 * N0], output_ub_tensor[2 * N0], output_ub_tensor[2 * N0 + 64],
                vecMask, repeats, {1, 1, 1, 16, 16, 16}
            ); // I1 + I2

            AscendC::Sub(input2_ub_tensor[batch_actual * 2 * N0], output_ub_tensor, output_ub_tensor[64],
                vecMask, repeats, {1, 1, 1, 8, 16, 16}
            ); // R1 - R2

            AscendC::Sub(input2_ub_tensor[batch_actual * 3 * N0], output_ub_tensor[2 * N0], output_ub_tensor[2 * N0 + 64],
                vecMask, repeats, {1, 1, 1, 8, 16, 16} // I1 - I2
            );

            PIPE_BARRIER(V);

            // radix-2 complex mul
            if (fftDirection == ASCENDFFT_FORWARD) {
                // R1 * R2 and I1 * I2
                AscendC::Mul(output_ub_tensor, input2_ub_tensor[2 * batch_actual * N0], input2_ub_tensor,
                    vecMask, 2 * repeats, {1, 1, 1, 8, 8, 8}
                );

                //  R1 * I2
                AscendC::Mul(
                    output_ub_tensor[2 * batch_actual * N0],
                    input2_ub_tensor[2 * batch_actual * N0],
                    input2_ub_tensor[batch_actual * N0],
                    vecMask, repeats, {1, 1, 1, 8, 8, 8}
                );

                // I1 * R2
                AscendC::Mul(
                    output_ub_tensor[3 * batch_actual * N0],
                    input2_ub_tensor[3 * batch_actual * N0],
                    input2_ub_tensor,
                    vecMask, repeats, {1, 1, 1, 8, 8, 8}
                );
                PIPE_BARRIER(V);
                // real
                AscendC::Sub(
                    input1_ub_tensor[64],
                    output_ub_tensor,
                    output_ub_tensor[batch_actual * N0],
                    vecMask, repeats, {1, 1, 1, 16, 8, 8}
                );
                // img
                AscendC::Add(
                    input1_ub_tensor[2 * N0 + 64],
                    output_ub_tensor[2 * batch_actual * N0],
                    output_ub_tensor[3 * batch_actual * N0],
                    vecMask, repeats, {1, 1, 1, 16, 8, 8}
                );
            } else {
                // R1 * R2 and I1 * I2
                AscendC::Mul(
                    output_ub_tensor,
                    input2_ub_tensor[2 * batch_actual * N0],
                    input2_ub_tensor,
                    vecMask, 2 * repeats, {1, 1, 1, 8, 8, 8}
                );
                //  R1 * I2
                AscendC::Mul(
                    output_ub_tensor[2 * batch_actual * N0],
                    input2_ub_tensor[2 * batch_actual * N0],
                    input2_ub_tensor[batch_actual * N0],
                    vecMask, repeats, {1, 1, 1, 8, 8, 8}
                );
                // I1 * R2
                AscendC::Mul(
                    output_ub_tensor[3 * batch_actual * N0],
                    input2_ub_tensor[3 * batch_actual * N0],
                    input2_ub_tensor,
                    vecMask, repeats, {1, 1, 1, 8, 8, 8}
                );
                PIPE_BARRIER(V);
                // real
                AscendC::Add(
                    input1_ub_tensor[64],
                    output_ub_tensor,
                    output_ub_tensor[batch_actual * N0],
                    vecMask, repeats, {1, 1, 1, 16, 8, 8}
                );
                // img
                AscendC::Sub(
                    input1_ub_tensor[2 * N0 + 64],
                    output_ub_tensor[3 * batch_actual * N0],
                    output_ub_tensor[2 * batch_actual * N0],
                    vecMask, repeats, {1, 1, 1, 16, 8, 8}
                );
            }
            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);

            // load to output
            AscendC::DataCopy(
                gm_output_tensor[offset_output_R],
                input1_ub_tensor,
                AscendC::DataCopyParams(
                    2, // blockCount
                    2 * N0 / 8, // blockLen
                    0, // srcStride
                    2 * N0 / 8 // dstStride
                )
            );

            SET_FLAG(MTE3, MTE2, event_id);
            ping_flag = 1 - ping_flag;
        }
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

__aicore__ __inline__ void SeparateRIwithRadix4(__gm__ float *__restrict__ gm_input,
                                                __gm__ float *__restrict__ gm_t_matrix,
                                                __gm__ float *__restrict__ gm_output, int32_t fftDirection,
                                                int32_t batch_size, int32_t inputN, int32_t M, int32_t N,
                                                int32_t N_radix,  // 下一次迭代的batch合并维度
                                                int32_t batch_id_begin, int32_t batch_id_end)
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    int32_t batch_num = 1;  // 下一次迭代的batch合并数量
    // restrict M to 4 (radix 2)
    int32_t N0 = 32 * 1024 / (4 * M);  // 1024

    AscendC::GlobalTensor<float> gm_input_tensor;
    AscendC::GlobalTensor<float> gm_t_matrix_tensor;
    AscendC::GlobalTensor<float> gm_output_tensor;

    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_t_matrix_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_t_matrix));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> input1_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> input1_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(32 * 1024));
    AscendC::LocalTensor<float> input2_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(64 * 1024));
    AscendC::LocalTensor<float> input2_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(96 * 1024));
    AscendC::LocalTensor<float> output_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(128 * 1024));
    AscendC::LocalTensor<float> output_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(160 * 1024));

    auto input1_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_ping_tensor.GetPhyAddr()));
    auto input1_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_pong_tensor.GetPhyAddr()));
    auto input2_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_ping_tensor.GetPhyAddr()));
    auto input2_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_pong_tensor.GetPhyAddr()));
    auto output_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_ping_tensor.GetPhyAddr()));
    auto output_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_pong_tensor.GetPhyAddr()));

    int32_t ping_flag = 1;

    int32_t aiv_id = AscendC::GetSubBlockIdx();

    // 计算每个核要处理的batch数量
    int32_t batch_id = batch_id_begin;
    int32_t batch_end = batch_id_end;

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);
    for (; batch_id < batch_end; batch_id += batch_num) {  // 一个核处理batch_num_sum个batch，一次处理batch_num个batch
        int32_t batch_actual = 1;
        for (int32_t n_idx = 0; n_idx < (N / 2 / N0); n_idx++) {  // N / 2 means 2 AIV
            auto input1_ub = ping_flag ? input1_ping : input1_pong;
            auto input1_ub_tensor = ping_flag ? input1_ping_tensor : input1_pong_tensor;
            auto input2_ub = ping_flag ? input2_ping : input2_pong;
            auto input2_ub_tensor = ping_flag ? input2_ping_tensor : input2_pong_tensor;
            auto output_ub = ping_flag ? output_ping : output_pong;
            auto output_ub_tensor = ping_flag ? output_ping_tensor : output_pong_tensor;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            int32_t fft_batch_actual = 1;

            int32_t offset_input1 =
                batch_id * 2 * inputN + n_idx * N0 * 2 + aiv_id * N / 2 * 2;  // AIV 0 负责前一半， AIV 1负责后一半
            int32_t offset_input2 = inputN + aiv_id * N / 2 + n_idx * N0;  // inputN表示只读入后一半
            int32_t offset_output_R = batch_id * 2 * inputN + n_idx * N0 + aiv_id * N / 2;  // batch_actual为1时 2048
            int32_t offset_output_I = batch_id * 2 * inputN + n_idx * N0 * fft_batch_actual +
                                      aiv_id * fft_batch_actual * N / 2 + fft_batch_actual * inputN / 2;

            WAIT_FLAG(MTE3, MTE2, event_id);

            // load t_matrix
            for (int32_t t_idx = 0; t_idx < 4; t_idx++) {
                AscendC::DataCopy(
                    input2_ub_tensor[(t_idx / 2) * 4 * N0 + (t_idx % 2) * 64],
                    gm_t_matrix_tensor[aiv_id * N / 2 + n_idx * N0 + t_idx * 2 * N],
                    AscendC::DataCopyParams(
                        N0 / 64, // blockCount
                        64 / 8, // blockLen
                        0, // srcStride
                        64 / 8 // dstStride
                    )
                );

                AscendC::DataCopy(
                    input2_ub_tensor[(t_idx / 2) * 4 * N0 + (t_idx % 2) * 64 + 2 * N0],
                    gm_t_matrix_tensor[aiv_id * N / 2 + n_idx * N0 + t_idx * 2 * N + N],
                    AscendC::DataCopyParams(
                        N0 / 64, // blockCount
                        64 / 8, // blockLen
                        0, // srcStride
                        64 / 8 // dstStride
                    )
                );
            }

            // load input data, radix-2 so 8192 -> 4096
            AscendC::DataCopy(
                input1_ub_tensor,
                gm_input_tensor[offset_input1],
                AscendC::DataCopyParams(
                    4, // blockCount
                    2 * N0 / 8, // blockLen
                    (N * 2 - N0 * 2) / 8, // srcStride
                    0 // dstStride
                )
            );

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);

            // separate RI
            int32_t total_len = M / 2 * N0 * 2;  // M / 2 means 2 fft batch, 2 means RI
            int32_t repeat = total_len / 64;     // M / 2 means 2 fft batch, 2 means RI
            uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数
            uint32_t mask = 0;

            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(output_ub_tensor.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(input1_ub_tensor.GetPhyAddr()), nullptr,
                      static_cast<uint16_t>(repeat),  // repeat
                      1,       // src0BlockStride
                      (uint8_t)1,       // patternMode, 101010…10
                      8,       // src0RepeatStride
                      8        // src1RepeatStride
            );

            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(output_ub_tensor[batch_actual * M / 2 * N0].GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(input1_ub_tensor.GetPhyAddr()), nullptr,
                      static_cast<uint16_t>(repeat),  // repeat
                      1,       // src0BlockStride
                      (uint8_t)2,       // patternMode, 010101…01
                      8,       // src0RepeatStride
                      8        // src1RepeatStride
            );
            PIPE_BARRIER(V);

            // radix-4 matmul
            int32_t repeats = N0 / 64;
            if (fftDirection == ASCENDFFT_FORWARD) {
                // R1
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor, output_ub_tensor, output_ub_tensor[N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor, input1_ub_tensor, output_ub_tensor[2 * N0], repeats, 1, 1, 1, 16, 16, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor, input1_ub_tensor, output_ub_tensor[3 * N0], repeats, 1, 1, 1, 16, 16, 8);

                // I1
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0], output_ub_tensor[4 * N0], output_ub_tensor[5 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0], input1_ub_tensor[2 * N0], output_ub_tensor[6 * N0], repeats, 1, 1, 1, 16, 16, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0], input1_ub_tensor[2 * N0], output_ub_tensor[7 * N0], repeats, 1, 1, 1, 16, 16, 8);

                // R4097
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[64], output_ub_tensor, output_ub_tensor[2 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[64], input1_ub_tensor[64], output_ub_tensor[5 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[64], input1_ub_tensor[64], output_ub_tensor[7 * N0], repeats, 1, 1, 1, 16, 16, 8);

                // I4097
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0 + 64], output_ub_tensor[3 * N0], output_ub_tensor[1 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0 + 64], input1_ub_tensor[2 * N0 + 64], output_ub_tensor[4 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0 + 64], input1_ub_tensor[2 * N0 + 64], output_ub_tensor[6 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 0 -1 0 1 1 0 -1 0

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0], output_ub_tensor, output_ub_tensor[1 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0], input1_ub_tensor[4 * N0], output_ub_tensor[2 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0], input1_ub_tensor[4 * N0], output_ub_tensor[3 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 1 -1 1 -1 0 0 0 0

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0], output_ub_tensor[4 * N0], output_ub_tensor[5 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0], input1_ub_tensor[6 * N0], output_ub_tensor[6 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0], input1_ub_tensor[6 * N0], output_ub_tensor[7 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 0 0 0 0 1 -1 1 -1

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0 + 64], output_ub_tensor, output_ub_tensor[2 * N0], repeats, 1, 1, 1, 16, 8, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0 + 64], input1_ub_tensor[4 * N0 + 64], output_ub_tensor[5 * N0], repeats, 1, 1, 1, 16, 16, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0 + 64], input1_ub_tensor[4 * N0 + 64], output_ub_tensor[7 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 1 0 -1 0 0 -1 0 1

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0 + 64], output_ub_tensor[1 * N0], output_ub_tensor[3 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0 + 64], input1_ub_tensor[6 * N0 + 64], output_ub_tensor[4 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0 + 64], input1_ub_tensor[6 * N0 + 64], output_ub_tensor[6 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 0 1 0 -1 1 0 -1 0

                PIPE_BARRIER(V);
            } else {
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor, output_ub_tensor, output_ub_tensor[N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor, input1_ub_tensor, output_ub_tensor[2 * N0], repeats, 1, 1, 1, 16, 16, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor, input1_ub_tensor, output_ub_tensor[3 * N0], repeats, 1, 1, 1, 16, 16, 8);

                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0], output_ub_tensor[4 * N0], output_ub_tensor[5 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0], input1_ub_tensor[2 * N0], output_ub_tensor[6 * N0], repeats, 1, 1, 1, 16, 16, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0], input1_ub_tensor[2 * N0], output_ub_tensor[7 * N0], repeats, 1, 1, 1, 16, 16, 8);

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[64], output_ub_tensor, output_ub_tensor[2 * N0], repeats, 1, 1, 1, 16, 8, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[64], input1_ub_tensor[64], output_ub_tensor[5 * N0], repeats, 1, 1, 1, 16, 16, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[64], input1_ub_tensor[64], output_ub_tensor[7 * N0], repeats, 1, 1, 1, 16, 16, 8);

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0 + 64], output_ub_tensor[1 * N0], output_ub_tensor[3 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0 + 64], input1_ub_tensor[2 * N0 + 64], output_ub_tensor[4 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[2 * N0 + 64], input1_ub_tensor[2 * N0 + 64], output_ub_tensor[6 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 0 -1 0 1 1 0 -1 0

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0], output_ub_tensor, output_ub_tensor[1 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0], input1_ub_tensor[4 * N0], output_ub_tensor[2 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0], input1_ub_tensor[4 * N0], output_ub_tensor[3 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 1 -1 1 -1 0 0 0 0

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0], output_ub_tensor[4 * N0], output_ub_tensor[5 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0], input1_ub_tensor[6 * N0], output_ub_tensor[6 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0], input1_ub_tensor[6 * N0], output_ub_tensor[7 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 0 0 0 0 1 -1 1 -1

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0 + 64], output_ub_tensor, output_ub_tensor[2 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0 + 64], input1_ub_tensor[4 * N0 + 64], output_ub_tensor[5 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[4 * N0 + 64], input1_ub_tensor[4 * N0 + 64], output_ub_tensor[7 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 1 0 -1 0 0 -1 0 1

                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0 + 64], output_ub_tensor[3 * N0], output_ub_tensor[1 * N0], repeats, 1, 1, 1, 16, 8, 8);
                add_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0 + 64], input1_ub_tensor[6 * N0 + 64], output_ub_tensor[4 * N0], repeats, 1, 1, 1, 16, 16, 8);
                sub_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[6 * N0 + 64], input1_ub_tensor[6 * N0 + 64], output_ub_tensor[6 * N0], repeats, 1, 1, 1, 16, 16, 8);  // 0 1 0 -1 1 0 -1 0

                PIPE_BARRIER(V);
            }

            repeats = 2 * N0 / 64;
            int32_t loop_time = 2;
            // radix-4 complex mul
            if (fftDirection == ASCENDFFT_FORWARD) {
                // R1 * R2 and I1 * I2
                mul_v<ArchType::ASCEND_V220, float>(output_ub_tensor, input1_ub_tensor, input2_ub_tensor, 4 * repeats, 1, 1, 1, 8, 8, 8);
                // real
                PIPE_BARRIER(V);
                for (int32_t loop_idx = 0; loop_idx < loop_time; loop_idx++) {
                    sub_v<ArchType::ASCEND_V220, float>(output_ub_tensor[loop_idx * 4 * N0], output_ub_tensor[loop_idx * 4 * N0],
                         output_ub_tensor[loop_idx * 4 * N0 + 2 * N0], repeats, 1, 1, 1, 8, 8, 8);
                }
                //  R1 * I2
                for (int32_t loop_idx = 0; loop_idx < loop_time; loop_idx++) {
                    mul_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[loop_idx * 4 * N0], input1_ub_tensor[loop_idx * 4 * N0], input2_ub_tensor[loop_idx * 4 * N0 + 2 * N0], repeats, 1, 1, 1, 8, 8, 8);
                }
                // I1 * R2
                for (int32_t loop_idx = 0; loop_idx < loop_time; loop_idx++) {
                    mul_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[loop_idx * 4 * N0 + 2 * N0], input1_ub_tensor[loop_idx * 4 * N0 + 2 * N0], input2_ub_tensor[loop_idx * 4 * N0], repeats, 1, 1, 1, 8, 8, 8);
                }
                PIPE_BARRIER(V);
                // img
                for (int32_t loop_idx = 0; loop_idx < loop_time; loop_idx++) {
                    add_v<ArchType::ASCEND_V220, float>(output_ub_tensor[loop_idx * 4 * N0 + 2 * N0], input1_ub_tensor[loop_idx * 4 * N0],
                         input1_ub_tensor[loop_idx * 4 * N0 + 2 * N0], repeats, 1, 1, 1, 8, 8, 8);
                }
            } else {
                // R1 * R2 and I1 * I2
                mul_v<ArchType::ASCEND_V220, float>(output_ub_tensor, input1_ub_tensor, input2_ub_tensor, 4 * repeats, 1, 1, 1, 8, 8, 8);
                // real
                PIPE_BARRIER(V);
                for (int32_t loop_idx = 0; loop_idx < loop_time; loop_idx++) {
                    add_v<ArchType::ASCEND_V220, float>(output_ub_tensor[loop_idx * 4 * N0], output_ub_tensor[loop_idx * 4 * N0],
                         output_ub_tensor[loop_idx * 4 * N0 + 2 * N0], repeats, 1, 1, 1, 8, 8, 8);
                }
                //  R1 * I2
                for (int32_t loop_idx = 0; loop_idx < loop_time; loop_idx++) {
                    mul_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[loop_idx * 4 * N0], input1_ub_tensor[loop_idx * 4 * N0],
                        input2_ub_tensor[loop_idx * 4 * N0 + 2 * N0], repeats, 1, 1, 1, 8, 8, 8);
                }
                // I1 * R2
                for (int32_t loop_idx = 0; loop_idx < loop_time; loop_idx++) {
                    mul_v<ArchType::ASCEND_V220, float>(input1_ub_tensor[loop_idx * 4 * N0 + 2 * N0], input1_ub_tensor[loop_idx * 4 * N0 + 2 * N0],
                        input2_ub_tensor[loop_idx * 4 * N0], repeats, 1, 1, 1, 8, 8, 8);
                }
                PIPE_BARRIER(V);
                // img
                for (int32_t loop_idx = 0; loop_idx < loop_time; loop_idx++) {
                    sub_v<ArchType::ASCEND_V220, float>(output_ub_tensor[loop_idx * 4 * N0 + 2 * N0], input1_ub_tensor[loop_idx * 4 * N0 + 2 * N0],
                         input1_ub_tensor[loop_idx * 4 * N0], repeats, 1, 1, 1, 8, 8, 8);
                }
            }
            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);

            AscendC::DataCopy(
                gm_output_tensor[batch_id * 2 * inputN + n_idx * 2 * N0 + aiv_id * 4 * N0],
                output_ub_tensor,
                AscendC::DataCopyParams(
                    4, // blockCount
                    2 * N0 / 8, // blockLen
                    0, // srcStride
                    (8192 - 2 * N0) / 8 // dstStride
                )
            );

            SET_FLAG(MTE3, MTE2, event_id);
            ping_flag = 1 - ping_flag;
        }
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

__aicore__ __inline__ void ContinuousMulDev(
    AscendC::LocalTensor<float>& dst,
    AscendC::LocalTensor<float>& src0,
    AscendC::LocalTensor<float>& src1,
    int32_t row, int32_t col)
{
    int32_t len = row * col;

    int32_t ele_num_per_block = 32 / sizeof(float);
    int32_t num_of_8blocks = len / ele_num_per_block / 8;
    int32_t ele_num_in_8blocks = 8 * ele_num_per_block;
    int32_t max_repeat_num = 255;
    int32_t loop = num_of_8blocks / max_repeat_num;  // 1
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)0, (uint64_t)-1);

    for (int32_t i = 0; i < loop; i++) {
        mul_v<ArchType::ASCEND_V220, float>(
            dst[max_repeat_num * ele_num_in_8blocks * i],
            src0[max_repeat_num * ele_num_in_8blocks * i],
            src1[max_repeat_num * ele_num_in_8blocks * i],
            max_repeat_num, 1, 1, 1, 8, 8, 8);
    }

    int32_t num_of_remanent_ele = len % (max_repeat_num * ele_num_in_8blocks);
    int32_t num_of_remanent_blocks = num_of_remanent_ele / ele_num_per_block;
    int32_t repeat = num_of_remanent_blocks / 8;  // 1
    int32_t remain = num_of_remanent_ele % ele_num_in_8blocks;
    // Handle remanent blocks
    if (repeat) {
        mul_v<ArchType::ASCEND_V220, float>(
            dst[max_repeat_num * ele_num_in_8blocks * loop],
            src0[max_repeat_num * ele_num_in_8blocks * loop],
            src1[max_repeat_num * ele_num_in_8blocks * loop],
            repeat, 1, 1, 1, 8, 8, 8);
    }

    if (remain) {
        AscendC::SetMaskCount();
        SetVectorMask<float>((uint64_t)0, (uint64_t)remain);
        mul_v<ArchType::ASCEND_V220, float>(
            dst[max_repeat_num * ele_num_in_8blocks * loop + repeat * ele_num_in_8blocks],
            src0[max_repeat_num * ele_num_in_8blocks * loop + repeat * ele_num_in_8blocks],
            src1[max_repeat_num * ele_num_in_8blocks * loop + repeat * ele_num_in_8blocks], 1, 1, 1, 1, 8, 8, 8);
        AscendC::SetMaskNorm();
        SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
    }
}

__aicore__ __inline__ void PartitionSubDev(
    AscendC::LocalTensor<float>& dst,
    AscendC::LocalTensor<float>& src0,
    AscendC::LocalTensor<float>& src1,
    int32_t row, int32_t col)
{
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)0, (uint64_t)-1);

    if (col == 128) {
        sub_v<ArchType::ASCEND_V220, float>(dst, src0, src1, row / 2, 1, 1, 1, 32, 32, 32);
        sub_v<ArchType::ASCEND_V220, float>(dst[64], src0[64], src1[64], row / 2, 1, 1, 1, 32, 32, 32);
    } else if (col == 64) {
        sub_v<ArchType::ASCEND_V220, float>(dst, src0, src1, row / 2, 1, 1, 1, 16, 16, 16);
    } else if (col == 32) {
        AscendC::SetMaskCount();
        SetVectorMask<float>((uint64_t)0, (uint64_t)32);
        for (int32_t row_idx = 0; row_idx < row / 2; row_idx++) {
            sub_v<ArchType::ASCEND_V220, float>(
                dst[2 * col * row_idx],
                src0[2 * col * row_idx],
                src1[2 * col * row_idx],
                1, 1, 1, 1, 0, 0, 0);  // repeat = 32 / 8
        }
        AscendC::SetMaskNorm();
        SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
    } else if (col == 16) {
        // row-wise repeat
        int32_t repeat = row / 2 / 8;
        // row-wise remain
        int32_t remain = (row / 2) % 8;
        if (repeat) {
            sub_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 4, 4, 4, 32, 32, 32);
            sub_v<ArchType::ASCEND_V220, float>(dst[8], src0[8], src1[8], repeat, 4, 4, 4, 32, 32, 32);
        } else {
            AscendC::SetMaskCount();
            SetVectorMask<float>((uint64_t)0, (uint64_t)(remain * 8));

            sub_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 4, 4, 4, 32, 32, 32);
            sub_v<ArchType::ASCEND_V220, float>(dst[8], src0[8], src1[8], repeat, 4, 4, 4, 32, 32, 32);

            AscendC::SetMaskNorm();
            SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
        }
    } else {
        int32_t repeat = row / 2 / 8;
        int32_t remain = (row / 2) % 8;
        if (repeat) {
            sub_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 2, 2, 2, 16, 16, 16);
        } else {
            AscendC::SetMaskCount();
            SetVectorMask<float>((uint64_t)0, (uint64_t)(remain * 8));

            sub_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 2, 2, 2, 16, 16, 16);

            AscendC::SetMaskNorm();
            SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
        }
    }
}

__aicore__ __inline__ void PartitionAddDev(
    AscendC::LocalTensor<float>& dst,
    AscendC::LocalTensor<float>& src0,
    AscendC::LocalTensor<float>& src1,
    int32_t row, int32_t col)
{
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)0, (uint64_t)-1);

    if (col == 128) {
        add_v<ArchType::ASCEND_V220, float>(dst, src0, src1, row / 2, 1, 1, 1, 32, 32, 32);
        add_v<ArchType::ASCEND_V220, float>(dst[64], src0[64], src1[64], row / 2, 1, 1, 1, 32, 32, 32);
    } else if (col == 64) {
        add_v<ArchType::ASCEND_V220, float>(dst, src0, src1, row / 2, 1, 1, 1, 16, 16, 16);
    } else if (col == 32) {
        AscendC::SetMaskCount();
        SetVectorMask<float>((uint64_t)0, (uint64_t)32);
        for (int32_t row_idx = 0; row_idx < row / 2; row_idx++) {
            add_v<ArchType::ASCEND_V220, float>(dst[2 * col * row_idx], src0[2 * col * row_idx], src1[2 * col * row_idx], 1, 1, 1, 1, 0, 0, 0);
        }
        AscendC::SetMaskNorm();
        SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
    } else if (col == 16) {
        int32_t repeat = row / 2 / 8;    // 列方向几次repeat
        int32_t remain = (row / 2) % 8;  // 列方向不满足8block
        if (repeat) {
            add_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 4, 4, 4, 32, 32, 32);
            add_v<ArchType::ASCEND_V220, float>(dst[8], src0[8], src1[8], repeat, 4, 4, 4, 32, 32, 32);
        } else {
            AscendC::SetMaskCount();
            SetVectorMask<float>((uint64_t)0, (uint64_t)(remain * 8));

            add_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 4, 4, 4, 32, 32, 32);
            add_v<ArchType::ASCEND_V220, float>(dst[8], src0[8], src1[8], repeat, 4, 4, 4, 32, 32, 32);

            AscendC::SetMaskNorm();
            SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
        }
    } else {
        int32_t repeat = row / 2 / 8;    // 列方向几次repeat
        uint32_t remain = (row / 2) % 8;  // 列方向不满足8block
        if (repeat) {
            add_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 2, 2, 2, 16, 16, 16);
        } else {
            AscendC::SetMaskCount();
            SetVectorMask<float>((uint64_t)0, (uint64_t)(remain * 8));

            add_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 2, 2, 2, 16, 16, 16);

            AscendC::SetMaskNorm();
            SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
        }
    }
}

__aicore__ __inline__ void PartitionMulDev(
    AscendC::LocalTensor<float>& dst,
    AscendC::LocalTensor<float>& src0,
    AscendC::LocalTensor<float>& src1,
    int32_t row, int32_t col)
{
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)0, (uint64_t)-1);

    if (col == 128) {
        mul_v<ArchType::ASCEND_V220, float>(dst, src0, src1, row / 2, 1, 1, 1, 32, 32, 32);
        mul_v<ArchType::ASCEND_V220, float>(dst[64], src0[64], src1[64], row / 2, 1, 1, 1, 32, 32, 32);
    } else if (col == 64) {
        mul_v<ArchType::ASCEND_V220, float>(dst, src0, src1, row / 2, 1, 1, 1, 16, 16, 16);
    } else if (col == 32) {
        AscendC::SetMaskCount();
        SetVectorMask<float>((uint64_t)0, (uint64_t)32);
        for (int32_t row_idx = 0; row_idx < row / 2; row_idx++) {
            mul_v<ArchType::ASCEND_V220, float>(dst[2 * col * row_idx], src0[2 * col * row_idx], src1[2 * col * row_idx], 1, 1, 1, 1, 0, 0, 0);
        }
        AscendC::SetMaskNorm();
        SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
    } else if (col == 16) {
        int32_t repeat = row / 2 / 8;
        int32_t remain = (row / 2) % 8;
        if (repeat) {
            mul_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 4, 4, 4, 32, 32, 32);
            mul_v<ArchType::ASCEND_V220, float>(dst[8], src0[8], src1[8], repeat, 4, 4, 4, 32, 32, 32);
        } else {
            AscendC::SetMaskCount();
            SetVectorMask<float>((uint64_t)0, (uint64_t)(remain * 8));

            mul_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 4, 4, 4, 32, 32, 32);
            mul_v<ArchType::ASCEND_V220, float>(dst[8], src0[8], src1[8], repeat, 4, 4, 4, 32, 32, 32);

            AscendC::SetMaskNorm();
            SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
        }
    } else {
        int32_t repeat = row / 2 / 8;
        int32_t remain = (row / 2) % 8;
        if (repeat) {
            mul_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 2, 2, 2, 16, 16, 16);
        } else {
            AscendC::SetMaskCount();
            SetVectorMask<float>((uint64_t)0, (uint64_t)(remain * 8));

            mul_v<ArchType::ASCEND_V220, float>(dst, src0, src1, repeat, 2, 2, 2, 16, 16, 16);

            AscendC::SetMaskNorm();
            SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
        }
    }
}

__aicore__ __inline__ void step_complex_mul_0_1(__gm__ float *__restrict__ input1, __gm__ float *__restrict__ t_matrix,
                                                __gm__ float *__restrict__ output, int32_t fftDirection,
                                                int32_t batch_size, int32_t iter, int32_t iter_count, int32_t fftN,
                                                int32_t fft_batch_size, int32_t row, int32_t col,
                                                int32_t batch_id_begin, int32_t batch_id_end)
{
    if (fftN == 0) {
        fftN = 128;
    }
    if (col == 0) {
        col = 128;
    }

    int32_t complex_rows = row / 2;
    int32_t rows_per_core = complex_rows;

    AscendC::GlobalTensor<float> input1_tensor;
    AscendC::GlobalTensor<float> t_matrix_tensor;
    AscendC::GlobalTensor<float> output_tensor;

    input1_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input1));
    t_matrix_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(t_matrix));
    output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> output_ub1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> output_ub2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(32 * 1024));
    AscendC::LocalTensor<float> input1_ub1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(64 * 1024));
    AscendC::LocalTensor<float> input1_ub2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(96 * 1024));
    AscendC::LocalTensor<float> input2_ub1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(128 * 1024));
    AscendC::LocalTensor<float> input2_ub2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(160 * 1024));

    auto output_ub1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_ub1_tensor.GetPhyAddr()));
    auto output_ub2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_ub2_tensor.GetPhyAddr()));
    auto input1_ub1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_ub1_tensor.GetPhyAddr()));
    auto input1_ub2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_ub2_tensor.GetPhyAddr()));
    auto input2_ub1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_ub1_tensor.GetPhyAddr()));
    auto input2_ub2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_ub2_tensor.GetPhyAddr()));

    int32_t aic_id = (AscendC::GetBlockIdx() - AscendC::GetSubBlockIdx()) / AscendC::GetTaskRation();
    int32_t aiv_id = AscendC::GetSubBlockIdx();
    int32_t trans_a = 0;
    int32_t trans_b = 0;
    int32_t M0 = 128;
    int32_t N0 = 128;
    if (row <= 64 && fftN < FFT_B_MERGE_N_BASE) {  // 多Batch合并,小radix时扩大矩阵
        M0 = 64;
        N0 = 256;
    } else {
        M0 = 128;
        N0 = 128;
    }

    int32_t ping_flag = 1;

    uint16_t flag_id = 5;  // input的反向同步，避免出现counter溢出的情况
    FftsCrossCoreSync<PIPE_MTE2, INNER_GROUP_CV_SYNC_MODE>(flag_id);

    flag_id = 7;  // input的反向同步，避免出现counter溢出的情况
    FftsCrossCoreSync<PIPE_MTE2, INNER_GROUP_CV_SYNC_MODE>(flag_id);

    int32_t m_loop = (row + M0 - 1) / M0;
    int32_t n_loop = (col + N0 - 1) / N0;

    int32_t N_radix = col;

    int32_t batch_id = batch_id_begin;
    int32_t batch_end = min(batch_id_end, batch_id + batch_size);

    int32_t batch_num = fftN >= FFT_B_MERGE_N_BASE ? 1 : min(192 * 64 / fftN, min(N0 / N_radix, batch_size));
    int32_t fft_batch_actual = fftN >= FFT_B_MERGE_N_BASE ? 2 : 1;

    // t_matrix矩阵的提前读入, 读多份
    if (fftN <= 16384) {  // t_matrix矩阵的提前读入, 读多份
        for (int32_t z = 0; z < batch_num * fft_batch_actual; z++) {
            AscendC::DataCopy(
                input2_ub1_tensor[z * row / 2 * col],
                t_matrix_tensor[(aiv_id * row / 2) * col],
                AscendC::DataCopyParams(
                    1, // blockCount
                    row / 2 * col / 8, // blockLen
                    0, // srcStride
                    0 // dstStride
                )
            );
        }
    }

    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);

    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);

    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);

    for (; batch_id < batch_end; batch_id += batch_num) {
        int32_t batch_actual = ((batch_id + batch_num) >= batch_end) ? (batch_end - batch_id) : batch_num;
        batch_actual = batch_actual * fft_batch_actual;
        for (int32_t fft_batch_id = 0; fft_batch_id < fft_batch_size; ++fft_batch_id) {
            for (int32_t m_idx = 0; m_idx < m_loop; ++m_idx) {
                for (int32_t n_idx = 0; n_idx < n_loop; ++n_idx) {  // 与step_ppmatmul_0_1对应,但实际上可以删去

                    int32_t m_actual = (m_idx == (m_loop - 1)) ? (row - m_idx * M0) : M0;
                    int32_t n_actual = (n_idx == (n_loop - 1)) ? (col - n_idx * N0) : N0;

                    auto output_ub = ping_flag ? output_ub1 : output_ub2;
                    auto input1_ub = ping_flag ? input1_ub1 : input1_ub2;
                    auto input2_ub = ping_flag ? input2_ub1 : input2_ub2;

                    auto output_ub_tensor = ping_flag ? output_ub1_tensor : output_ub2_tensor;
                    auto input1_ub_tensor = ping_flag ? input1_ub1_tensor : input1_ub2_tensor;
                    auto input2_ub_tensor = ping_flag ? input2_ub1_tensor : input2_ub2_tensor;
                    if (fftN <= 16384) {
                        input2_ub = input2_ub1;  // 8192 及以上的序列不能预取t_matrix
                        input2_ub_tensor = input2_ub1_tensor;  // newly added
                    }
                    auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
                    auto offset_c = ping_flag ? aic_id * 128 * 256 : aic_id * 128 * 256 + 128 * 128;

                    flag_id = ping_flag ? 3 : 6;
                    WaitFlagDev(flag_id);

                    WAIT_FLAG(V, MTE2, event_id);

                    if (fftN > 16384) {
                        AscendC::DataCopy(
                            input2_ub_tensor,
                            t_matrix_tensor[(m_idx * M0 + aiv_id * m_actual / 2) * col + n_idx * N0],
                            AscendC::DataCopyParams(
                                m_actual / 2, // blockCount
                                n_actual / 8, // blockLen
                                (col - n_actual) / 8, // srcStride
                                0 // dstStride
                            )
                        );
                    } else if (fftN <= FFT_B_MERGE_N_BASE) {
                        for (int32_t i = 0; i < batch_actual; i++) {
                            AscendC::DataCopy(
                                input1_ub_tensor[i * fftN / fft_batch_actual],
                                input1_tensor[offset_c + i * N_radix + aiv_id * fftN / fft_batch_actual * batch_actual],
                                AscendC::DataCopyParams(
                                    m_actual / 2, // blockCount
                                    n_actual / 8, // blockLen
                                    (batch_actual - 1) * N_radix / 8, // srcStride
                                    0 // dstStride
                                )
                            );
                        }
                    } else if (fftN == 16384) {
                        for (int32_t i = 0; i < batch_actual; i++) {
                            AscendC::DataCopy(
                                input1_ub_tensor[i * 64 * 64],
                                input1_tensor[offset_c + i * N_radix + aiv_id * 128 * 64],
                                AscendC::DataCopyParams(
                                    m_actual / 2, // blockCount
                                    n_actual / 8, // blockLen
                                    (batch_actual - 1) * N_radix / 8, // srcStride
                                    0 // dstStride
                                )
                            );
                        }
                    } else {
                        AscendC::DataCopy(
                            input1_ub_tensor,
                            input1_tensor[batch_id * fftN * 2 + fft_batch_id * row * col +
                                            (m_idx * M0 + aiv_id * m_actual / 2) * col + n_idx * N0],
                            AscendC::DataCopyParams(
                                m_actual / 2, // blockCount
                                n_actual / 8, // blockLen
                                (col - n_actual) / 8, // srcStride
                                0 // dstStride
                            )
                        );
                    }

                    flag_id = ping_flag ? 5 : 7;  // input的反向同步，避免出现counter溢出的情况
                    FftsCrossCoreSync<PIPE_MTE2, INNER_GROUP_CV_SYNC_MODE>(flag_id);

                    SET_FLAG(MTE2, V, event_id);
                    WAIT_FLAG(MTE2, V, event_id);
                    WAIT_FLAG(MTE3, V, event_id);

                    if (fftDirection == ASCENDFFT_FORWARD) {  // 8192以上的batch_actual固定为1
                        ContinuousMulDev(output_ub_tensor, input1_ub_tensor, input2_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);

                        auto actual_output_ub_tensor = output_ub_tensor[n_actual];
                        PartitionSubDev(output_ub_tensor, output_ub_tensor, actual_output_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);

                        auto actual_input2_ub_tensor = input2_ub_tensor[n_actual];
                        PartitionMulDev(input1_ub_tensor, input1_ub_tensor, actual_input2_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);

                        auto actual_input1_ub_tensor = input1_ub_tensor[n_actual];
                        PartitionMulDev(actual_input1_ub_tensor, actual_input1_ub_tensor, input2_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);

                        PartitionAddDev(actual_output_ub_tensor, input1_ub_tensor, actual_input1_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);
                    } else {
                        // R * R and I * I
                        ContinuousMulDev(output_ub_tensor, input1_ub_tensor, input2_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);

                        auto actual_output_ub_tensor = output_ub_tensor[n_actual];
                        PartitionAddDev(output_ub_tensor, output_ub_tensor, actual_output_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);

                        // calculate R1 * I2
                        auto actual_input2_ub_tensor = input2_ub_tensor[n_actual];
                        PartitionMulDev(input1_ub_tensor, input1_ub_tensor, actual_input2_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);

                        // calculate I1 * R2
                        auto actual_input1_ub_tensor = input1_ub_tensor[n_actual];
                        PartitionMulDev(actual_input1_ub_tensor, actual_input1_ub_tensor, input2_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);

                        PartitionSubDev(actual_output_ub_tensor, actual_input1_ub_tensor, input1_ub_tensor, batch_actual * m_actual / 2, n_actual);
                        PIPE_BARRIER(V);
                    }

                    SET_FLAG(V, MTE3, event_id);
                    WAIT_FLAG(V, MTE3, event_id);
                    SET_FLAG(V, MTE2, event_id);

                    if (fftN < FFT_B_MERGE_N_BASE) {
                        AscendC::DataCopy(
                            output_tensor[batch_id * fftN * 2 + aiv_id * fftN],
                            output_ub_tensor,
                            AscendC::DataCopyParams(
                                batch_actual, // blockCount
                                fftN / 8, // blockLen
                                0, // srcStride
                                fftN / 8 // dstStride
                            )
                        );
                    } else if (fftN == FFT_B_MERGE_N_BASE) {
                        for (int32_t zz = 0; zz < fft_batch_actual; zz++) {  // 8192 由于有fft_batch级的合并 搬运特殊
                            AscendC::DataCopy(
                                output_tensor[batch_id * fftN * 2 + aiv_id * 64 * 64 * 2 + zz * 64 * 2],
                                output_ub_tensor[zz * 64 * 64],
                                AscendC::DataCopyParams(
                                    32, // blockCount
                                    128 / 8, // blockLen
                                    0, // srcStride
                                    128 / 8 // dstStride
                                )
                            );
                        }
                    } else if (fftN == 16384) {
                        for (int32_t zz = 0; zz < fft_batch_actual; zz++) {  // 8192 由于有fft_batch级的合并 搬运特殊
                            AscendC::DataCopy(
                                output_tensor[batch_id * fftN * 2 + aiv_id * 64 * 64 * 4 + zz * 64 * 2 +
                                                fft_batch_id * 128 * 2],
                                output_ub_tensor[zz * 64 * 64],
                                AscendC::DataCopyParams(
                                    32, // blockCount
                                    128 / 8, // blockLen
                                    0, // srcStride
                                    (4 * 128 - 128) / 8 // dstStride
                                )
                            );
                        }
                    } else {
                        AscendC::DataCopy(
                            output_tensor[batch_id * fftN * 2 + fft_batch_id * row * col +
                                            (m_idx * M0 + aiv_id * m_actual / 2) * col + n_idx * N0],
                            output_ub_tensor,
                            AscendC::DataCopyParams(
                                m_actual / 2, // blockCount
                                n_actual / 8, // blockLen
                                0, // srcStride
                                (col - n_actual) / 8 // dstStride
                            )
                        );
                    }

                    SET_FLAG(MTE3, V, event_id);
                    ping_flag = 1 - ping_flag;
                }
            }
        }
    }
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);

    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);
}

__aicore__ __inline__ void step_2iter_count_gather_real_imag(__gm__ float *__restrict__ gm_output,
                                                             __gm__ float *__restrict__ gm_input,
                                                             __gm__ int32_t *__restrict__ gm_index, int32_t batch_size,
                                                             int32_t fft_batch_size, int32_t row, int32_t col,
                                                             int32_t fftN, int32_t batch_id_begin, int32_t batch_id_end)
{
    if (fftN == 0) {
        fftN = 128;
    }
    if (col == 0) {
        col = 128;
    }

    AscendC::GlobalTensor<float> gm_output_tensor;
    AscendC::GlobalTensor<float> gm_input_tensor;
    AscendC::GlobalTensor<float> gm_index_tensor;

    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));
    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_index_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_index));

    uint16_t flag_id = 5;
    FftsCrossCoreSync<PIPE_MTE2, INNER_GROUP_CV_SYNC_MODE>(flag_id);

    flag_id = 7;
    FftsCrossCoreSync<PIPE_MTE2, INNER_GROUP_CV_SYNC_MODE>(flag_id);

    int32_t M0 = 128;
    int32_t N0 = 128;
    if (row <= 64) {
        M0 = 64;
        N0 = 256;
    } else {
        M0 = 128;
        N0 = 128;
    }

    int32_t m_loop = (row + M0 - 1) / M0;
    int32_t n_loop = (col + N0 - 1) / N0;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> input_ub1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> input_ub2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(32 * 1024));
    AscendC::LocalTensor<float> index_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(64 * 1024));
    AscendC::LocalTensor<float> temp_ub1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(128 * 1024));
    AscendC::LocalTensor<float> temp_ub2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(160 * 1024));
    AscendC::LocalTensor<float> output_ub1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(128 * 1024));
    AscendC::LocalTensor<float> output_ub2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(160 * 1024));

    auto input_ub1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input_ub1_tensor.GetPhyAddr()));
    auto input_ub2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input_ub2_tensor.GetPhyAddr()));
    auto index_ub = reinterpret_cast<__ubuf__ float *>((uintptr_t)(index_ub_tensor.GetPhyAddr()));
    auto temp_ub1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(temp_ub1_tensor.GetPhyAddr()));
    auto temp_ub2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(temp_ub2_tensor.GetPhyAddr()));
    auto output_ub1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_ub1_tensor.GetPhyAddr()));
    auto output_ub2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_ub2_tensor.GetPhyAddr()));

    int32_t input_addr = 1024 * 32;
    int32_t aic_id = (AscendC::GetBlockIdx() - AscendC::GetSubBlockIdx()) / AscendC::GetTaskRation();
    int32_t aiv_id = AscendC::GetSubBlockIdx();
    float fft_n_float = fftN;

    int32_t ping_flag = 1;

    AscendC::DataCopy(
        index_ub_tensor,
        gm_index_tensor,
        AscendC::DataCopyParams(
            1, // blockCount
            FFT_B_MERGE_N_BASE / 8, // blockLen
            0, // srcStride
            0 // dstStride
        )
    );

    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);

    int32_t N_radix = col;

    int32_t batch_id = batch_id_begin;
    int32_t batch_end = min(batch_id_end, batch_id + batch_size);
    int32_t batch_num = fftN >= FFT_B_MERGE_N_BASE ? 1 : min(192 * 64 / fftN, N0 / N_radix);
    int32_t fft_batch_actual = fftN >= FFT_B_MERGE_N_BASE ? FFT_B_BATCH_GROUP_BASE / (fftN / FFT_B_MERGE_N_BASE) : 1;

    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);

    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);

    for (; batch_id < batch_end; batch_id += batch_num) {
        int32_t batch_actual = ((batch_id + batch_num) >= batch_end) ? (batch_end - batch_id) : batch_num;
        batch_actual = batch_actual * fft_batch_actual;
        for (int64_t fft_batch_id = 0; fft_batch_id < fft_batch_size; ++fft_batch_id) {
            for (int64_t m_idx = 0; m_idx < m_loop; ++m_idx) {
                for (int64_t n_idx = 0; n_idx < n_loop; ++n_idx) {
                    int32_t m_actual = (m_idx == (m_loop - 1)) ? (row - m_idx * M0) : M0;
                    int32_t n_actual = (n_idx == (n_loop - 1)) ? (col - n_idx * N0) : N0;

                    auto input_ub = ping_flag ? input_ub1 : input_ub2;
                    auto input_ub_tensor = ping_flag ? input_ub1_tensor : input_ub2_tensor;
                    auto input_addr = ping_flag ? 0 : 1024 * 32;

                    auto output_ub = ping_flag ? output_ub1 : output_ub2;
                    auto output_ub_tensor = ping_flag ? output_ub1_tensor : output_ub2_tensor;
                    auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
                    auto offset_c = ping_flag ? aic_id * 128 * 256 : aic_id * 128 * 256 + 128 * 128;

                    flag_id = ping_flag ? 3 : 6;
                    WaitFlagDev(flag_id);

                    WAIT_FLAG(V, MTE2, event_id);

                    if (fftN >= FFT_B_MERGE_N_BASE) {
                        AscendC::DataCopy(
                            input_ub_tensor,
                            gm_input_tensor[offset_c + aiv_id * 64 * 64],
                            AscendC::DataCopyParams(
                                2, // blockCount
                                64 * 64 / 8, // blockLen
                                64 * 64 / 8, // srcStride
                                0 // dstStride
                            )
                        );
                    } else {
                        AscendC::DataCopy(
                            input_ub_tensor,
                            gm_input_tensor[offset_c + aiv_id * m_actual / 4 * col * batch_num],
                            AscendC::DataCopyParams(
                                2, // blockCount
                                m_actual / 4 * col * batch_num / 8, // blockLen
                                m_actual / 4 * col * batch_num / 8, // srcStride
                                0 // dstStride
                            )
                        );
                    }

                    flag_id = ping_flag ? 5 : 7;
                    FftsCrossCoreSync<PIPE_MTE2, INNER_GROUP_CV_SYNC_MODE>(flag_id);

                    SET_FLAG(MTE2, V, event_id);
                    WAIT_FLAG(MTE2, V, event_id);
                    WAIT_FLAG(MTE3, V, event_id);

                    // 一次处理两行数据
                    vgather((__ubuf__ uint32_t *)(uintptr_t)(output_ub), (__ubuf__ uint32_t *)(uintptr_t)(index_ub),
                            input_addr, 8, m_actual / 2 * col * batch_num * fft_batch_actual / 64);

                    PIPE_BARRIER(V);

                    SET_FLAG(V, MTE3, event_id);
                    WAIT_FLAG(V, MTE3, event_id);
                    SET_FLAG(V, MTE2, event_id);

                    if (fftN == FFT_B_MERGE_N_BASE) {
                        AscendC::DataCopy(
                            gm_output_tensor[batch_id * fftN * 2 + fft_batch_id * 128 * 128 + aiv_id * 128 * 64],
                            output_ub_tensor,
                            AscendC::DataCopyParams(
                                1, // blockCount
                                128 * 64 / 8, // blockLen
                                0, // srcStride
                                0 // dstStride
                            )
                        );
                    } else if (fftN == 16384) {
                        AscendC::DataCopy(
                            gm_output_tensor[batch_id * fftN * 2 + fft_batch_id * 256 + aiv_id * 128 * 64 * 2],
                            output_ub_tensor,
                            AscendC::DataCopyParams(
                                128 * 64 / 256, // blockCount
                                256 / 8, // blockLen
                                0, // srcStride
                                256 / 8 // dstStride
                            )
                        );
                    } else {
                        AscendC::DataCopy(
                            gm_output_tensor[batch_id * fftN * 2 + fft_batch_id * col * 2 +
                                            (m_idx * M0 + aiv_id * m_actual / 2) * col * fft_batch_size +
                                            n_idx * n_actual * 2],
                            output_ub_tensor,
                            AscendC::DataCopyParams(
                                batch_actual, // blockCount
                                fftN / 8, // blockLen
                                0, // srcStride
                                fftN / 8 // dstStride
                            )
                        );
                    }

                    SET_FLAG(MTE3, V, event_id);
                    ping_flag = 1 - ping_flag;
                }
            }
        }
    }
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);

    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);
}

extern "C" __global__ __aicore__ void fft_b(__gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
                                            __gm__ float *__restrict__ gm_w_matrix,
                                            __gm__ float *__restrict__ gm_t_matrix,
                                            __gm__ int32_t *__restrict__ gm_index, __gm__ float *__restrict__ gm_output,
                                            __gm__ float *__restrict__ gm_workspace,
                                            __gm__ uint8_t *__restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
    SetPadding(0);
    AscendC::SetAtomicNone();

    auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    int32_t fft_n = tiling_para[0];
    if (fft_n == 0) {
        fft_n = 128;
    }
    int32_t batch_size = tiling_para[1];
    int32_t fftDirection = tiling_para[2];
    int32_t iter_count = tiling_para[3];
    // 迭代只有一次，表示迭代只需要计算dft，只需要进行矩阵乘计算，不需要vectori算。
    if (iter_count <= 1) {
        return;
    }

    /******分batch*/
    int32_t batch_size_bulk = FFT_B_BULK_SUBBLOCK_BUDGET / (fft_n / 256);
    if (iter_count > 2) {
        batch_size_bulk = batch_size;
    }
    int32_t last_batch_size_bulk = batch_size_bulk;

    int32_t B_num = batch_size / batch_size_bulk;
    int32_t batchRemain = batch_size % batch_size_bulk;
    if (batchRemain > 0) {
        if (B_num != 0) {
            last_batch_size_bulk = batch_size_bulk + batchRemain;
        } else {
            last_batch_size_bulk = batchRemain;
            B_num++;
        }
    }
    int32_t batchPerBlock, batch_id;
    int32_t lastBatchPerBlock, last_batch_id;

    loop_tiling(batch_size_bulk, batchPerBlock, batch_id);
    loop_tiling(last_batch_size_bulk, lastBatchPerBlock, last_batch_id);

    for (int32_t z = 0; z < B_num; z++) {
        int32_t batch_start = z * batch_size_bulk + batch_id;
        int32_t batch_end = batch_start + batchPerBlock;
        int32_t inner_batch = batchPerBlock;

        if (z == B_num - 1) {
            batch_start = z * batch_size_bulk + last_batch_id;
            batch_end = batch_start + lastBatchPerBlock;
            inner_batch = lastBatchPerBlock;
        }

        // 矩阵乘的M K N
        int32_t M, K;
        int32_t N = 1;

        // 第三次迭代开始，需要进行跨stride拷贝，重新分组生成的数据，第三次迭代数据有N2份，每组数据的row是N1.第四次迭代，
        // 数据有N3份，每组数据的row是N1*N2。
        int32_t fft_iter_row;
        // 第三次迭代开始，需要进行跨stride拷贝，所以需要输入拷贝的值。第三次迭代，是N2,第四次迭代是N1*N2.
        int32_t fft_iter_n_stride = 1;

        int32_t fft_ppmatmul_batch = 1;
        int32_t w_matrix_step = 0;
        uint16_t flag_id = 0;

        __gm__ float *iter_input = nullptr;
        __gm__ float *iter_output = nullptr;
        if (iter_count == 3 || iter_count == 5) {
            iter_input = gm_input;
            iter_output = gm_output;
        } else {
            iter_input = gm_output;
            iter_output = gm_input;
        }

        for (int32_t it = 0; it < iter_count; ++it) {
            M = tiling_para[4 + it] * 2;
            K = M;
            N = 1;
            for (int32_t i = it + 1; i < iter_count; ++i) {
                N *= tiling_para[4 + i];
            }

            flag_id = 0;
            WaitFlagDev(flag_id);  // 等待虚实分离和复数乘整个计算结束
            // 仅保留 iter_count == 2, 不存在其它场景
            if (it == 0) {
                M = tiling_para[4 + it] * 2;
                K = M;
                N = 1;
                for (int32_t i = it + 1; i < iter_count; ++i) {
                    N *= tiling_para[4 + i];
                }
                fft_ppmatmul_batch = fft_n * 2 / 16384;
                if (fft_ppmatmul_batch == 0) {
                    fft_ppmatmul_batch = 1;
                }

                step_ppmatmul_0_1(gm_w_matrix + w_matrix_step, gm_output, gm_workspace, inner_batch, it, iter_count,
                                  fft_n, fft_ppmatmul_batch, M, K, N, batch_start,
                                  batch_end);  // 8192及以下case， 固定设置fft_batch为1
            } else if ((it == 1) && (iter_count == 2)) {
                // 两次迭代的第二次计算过程
                M = tiling_para[4 + it] * 2;
                K = M;
                N = tiling_para[4];
                if (fft_n == FFT_B_MERGE_N_BASE) {
                    N = 2;
                }
                if (fft_n == 16384) {
                    N = 4;
                }
                w_matrix_step = tiling_para[4] * tiling_para[4] * 4;
                fft_ppmatmul_batch = fft_n * 2 / 16384;
                if (fft_ppmatmul_batch == 0) {
                    fft_ppmatmul_batch = 1;
                }

                step_ppmatmul_2itercount_last_iter(gm_w_matrix + w_matrix_step, gm_output, gm_workspace, inner_batch,
                                                   it, iter_count, fft_n, fft_ppmatmul_batch, M, K, N, batch_start,
                                                   batch_end);  // 8192及以下case， 固定设置fft_batch为1
            }
        }
    }
}
#elif __DAV_C220_VEC__
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();

    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);

    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    int32_t fft_n = tiling_para[0];
    if (fft_n == 0) {
        fft_n = 128;
    }
    int32_t batch_size = tiling_para[1];
    if (batch_size == 0) {
        batch_size = 1;
    }
    int32_t fftDirection = tiling_para[2];
    int32_t iter_count = tiling_para[3];
    int32_t tilingPara4 = tiling_para[4];
    if (tilingPara4 == 0) {
        tilingPara4 = 1;
    }
    int32_t tilingPara5 = tiling_para[5];
    if (tilingPara5 == 0) {
        tilingPara5 = 1;
    }
    // 迭代只有一次，表示迭代只需要计算dft，只需要进行矩阵乘计算，不需要vectori算。
    if (iter_count <= 1) {
        return;
    }

    /******分batch*/
    int32_t batch_size_bulk = FFT_B_BULK_SUBBLOCK_BUDGET / (fft_n / 256);
    if (iter_count > 2) {
        batch_size_bulk = batch_size;
    }
    int32_t last_batch_size_bulk = batch_size_bulk;

    /******计算循环次数*/
    int32_t B_num = batch_size / batch_size_bulk;
    int32_t batchRemain = batch_size % batch_size_bulk;
    if (batchRemain > 0) {
        if (B_num != 0) {
            last_batch_size_bulk = batch_size_bulk + batchRemain;
        } else {
            last_batch_size_bulk = batchRemain;
            B_num++;
        }
    }

    /******计算核内循环次数*/
    int32_t batchPerBlock, batch_id;
    int32_t lastBatchPerBlock, last_batch_id;
    loop_tiling(batch_size_bulk, batchPerBlock, batch_id);  // 计算一个核分到哪些batch
    loop_tiling(last_batch_size_bulk, lastBatchPerBlock, last_batch_id);

    for (int32_t z = 0; z < B_num; z++) {
        int32_t batch_start = z * batch_size_bulk + batch_id;
        int32_t batch_end = batch_start + batchPerBlock;
        int32_t inner_batch = batchPerBlock;

        if (z == B_num - 1) {
            batch_start = z * batch_size_bulk + last_batch_id;
            batch_end = batch_start + lastBatchPerBlock;
            inner_batch = lastBatchPerBlock;
        }

        int32_t row = 1;  // T矩阵的行数，每次复数乘的T矩阵的行数 第一次迭代N1*2 第二次迭代N2*2
        int32_t col = 1;  // T矩阵的列数，每次复数乘的T矩阵的列数 第一次迭代N2*N3*...*Nk 第二次迭代N3*N4*...*Nk
        uint64_t t_matrix_step = 0;
        int32_t fft_batch_size = 1;

        // SeparateRIwithRadix2过程把虚实分离的结果放入gm_output中
        if (fft_n != FFT_B_MERGE_N_BASE && fft_n != 16384) {
            step_reduce(gm_input, gm_output, inner_batch, fft_n, fft_n / tilingPara4, batch_start, batch_end);
        } else if (fft_n == FFT_B_MERGE_N_BASE) {
            SeparateRIwithRadix2(gm_input, gm_t_matrix, gm_output, fftDirection, inner_batch, fft_n, 4, fft_n / 2, 64,
                                 batch_start, batch_end);
            t_matrix_step += fft_n * 2;
        } else {
            SeparateRIwithRadix4(gm_input, gm_t_matrix, gm_output, fftDirection, inner_batch, fft_n, 8, fft_n / 4, 64,
                                 batch_start, batch_end);
            t_matrix_step += fft_n * 2;
        }
        uint16_t flag_id = 0;
        FftsCrossCoreSync<PIPE_MTE3, INNER_GROUP_CV_SYNC_MODE>(flag_id);

        __gm__ float *iter_input = nullptr;
        __gm__ float *iter_output = nullptr;

        if (iter_count == 3 || iter_count == 5) {
            iter_input = gm_output;
            iter_output = gm_input;
        } else {
            iter_input = gm_input;
            iter_output = gm_output;
        }
        for (int32_t it = 0; it < iter_count; ++it) {
            row = tiling_para[4 + it] * 2;
            col = 1;
            for (int32_t i = it + 1; i < iter_count; ++i) {
                col *= tiling_para[4 + i];
            }
            // 仅保留 iter_count == 2, 不存在其它场景
            if (it == 0) {
                // 两次fft迭代的第一次计算过程
                row = tiling_para[4 + it] * 2;
                col = 1;
                for (int32_t i = it + 1; i < iter_count; ++i) {
                    col *= tiling_para[4 + i];
                }
                fft_batch_size = fft_n * 2 / 16384;
                if (fft_batch_size == 0) {
                    fft_batch_size = 1;
                }
                step_complex_mul_0_1(gm_workspace, gm_t_matrix + t_matrix_step, gm_output, fftDirection, inner_batch,
                                     it, iter_count, fft_n, fft_batch_size, row, col, batch_start,
                                     batch_end);  // 8192及以下case， 固定设置fft_batch为1
                flag_id = 0;
                FftsCrossCoreSync<PIPE_MTE3, INNER_GROUP_CV_SYNC_MODE>(flag_id);
            } else if ((it == 1) && (iter_count == 2)) {
                // 两次fft迭代的最后一次的就算过程
                row = tiling_para[4 + it] * 2;
                col = tiling_para[4];
                if (fft_n == FFT_B_MERGE_N_BASE) {
                    col = 2;
                }
                if (fft_n == 16384) {
                    col = 4;
                }
                fft_batch_size = fft_n * 2 / 16384;
                if (fft_batch_size == 0) {
                    fft_batch_size = 1;
                }
                step_2iter_count_gather_real_imag(gm_output, gm_workspace, gm_index, inner_batch, fft_batch_size, row,
                                                  col, fft_n, batch_start, batch_end);
                // 8192及以下case， 固定设置fft_batch为1
            }
        }
    }
}
#endif
}

} // namespace FFT1DBKernel

#endif
