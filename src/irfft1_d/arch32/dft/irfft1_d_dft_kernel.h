/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IRFFT1_D_DFT_KERNEL_H
#define IRFFT1_D_DFT_KERNEL_H

#include "kernel/fft_common_kernel.h"
#include <stdint.h>

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

namespace Irfft1DDftKernel {

constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(float);
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t C0_SIZE = 32 / sizeof(float);
constexpr int32_t CUBE_MATRIX_SIZE = BLOCK_SIZE * C0_SIZE;
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(float);

__aicore__ __inline__ void load_matrix_zZ(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    int32_t R, int32_t C, int32_t valid_row, int32_t valid_col, int32_t stride
) {
    constexpr int32_t R0 = 16;
    constexpr int32_t C0 = 32 / sizeof(float);
    constexpr int STRIDE_LIMIT = 65536;

    int64_t srcNdStride = R0 * stride;
    int64_t srcNStride = stride;
    if (srcNdStride < STRIDE_LIMIT) {
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;
        if (ndNum > 0) {
            AscendC::DataCopy(
                dst, src,
                AscendC::Nd2NzParams(ndNum, R0, valid_col, R0 * stride, srcNStride, R0, 1, R0 * C)
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * R0 * C], src[ndNum * R0 * stride],
                AscendC::Nd2NzParams(1, remains, valid_col, 0, srcNStride, R0, 1, 0)
            );
        }
    } else if (srcNStride < STRIDE_LIMIT) {
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;
        for (int32_t i = 0; i < ndNum; i++) {
            AscendC::DataCopy(
                dst[i * R0 * C], src[i * R0 * stride],
                AscendC::Nd2NzParams(1, R0, valid_col, 0, srcNStride, R0, 1, 0)
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * R0 * C], src[ndNum * R0 * stride],
                AscendC::Nd2NzParams(1, remains, valid_col, 0, srcNStride, R0, 1, 0)
            );
        }
    } else {
        for (int32_t i = 0; i < valid_row; i++) {
            int32_t idxR0 = i / R0;
            int32_t idxInR0 = i % R0;
            AscendC::DataCopy(
                dst[idxR0 * R0 * C + idxInR0 * C0], src[i * stride],
                AscendC::Nd2NzParams(1, 1, valid_col, 0, 0, R0, 0, 0)
            );
        }
    }
}

extern "C" __global__ __aicore__ void dft_c2r(
    __gm__ float * __restrict__ gm_a,
    __gm__ float * __restrict__ gm_b,
    __gm__ float * __restrict__ gm_c,
    __gm__ uint8_t * __restrict__ workspace,
    __gm__ uint8_t * __restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::SetLoadDataPaddingValue<float>(0.0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    AscendC::GlobalTensor<float> a_gm_tensor;
    AscendC::GlobalTensor<float> b_gm_tensor;
    AscendC::GlobalTensor<float> c_gm_tensor;

    a_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
    b_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));
    c_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_c));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto a_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)0);
    auto b_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)(128 * 1024));
    auto l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, float>((uintptr_t)0);
    auto l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, float>((uintptr_t)0);
    auto l0c_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>((uintptr_t)0);

    auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    int32_t batchSize = tiling_para[0];
    int32_t M = tiling_para[1];
    int32_t N = tiling_para[2];
    int32_t K = tiling_para[3];
    int32_t trans_a = tiling_para[4];
    int32_t trans_b = tiling_para[5];
    int32_t M0 = tiling_para[6];
    int32_t N0 = tiling_para[7];
    int32_t K0 = tiling_para[8];

    int32_t m_loop = (M + M0 - 1) / M0;
    int32_t n_loop = (N + N0 - 1) / N0;
    int32_t k_loop = (K + K0 - 1) / K0;
    int32_t loop = batchSize * m_loop * n_loop;

    int32_t ping_flag = 1;

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);

    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {
        if (loop_idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
            continue;
        }

        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);
        int64_t m_idx;
        int64_t n_idx;

        constexpr int N_COL = 16;
        int tile_block_loop = (n_loop + N_COL - 1) / N_COL;
        int tile_block_idx = in_batch_idx / (N_COL * m_loop);
        int in_tile_block_idx = in_batch_idx % (N_COL * m_loop);
        int n_col = N_COL;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - N_COL * tile_block_idx;
        }
        m_idx = in_tile_block_idx / n_col;
        n_idx = tile_block_idx * N_COL + in_tile_block_idx % n_col;

        int64_t offset_a, offset_b;
        int64_t offset_c = batch_idx * M * N + m_idx * M0 * N + n_idx * N0;
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        for (int32_t k_idx = 0; k_idx < k_loop; k_idx++) {
            if (trans_a) {
                offset_a = batch_idx * M * K + k_idx * K0 * M + m_idx * M0;
            } else {
                offset_a = batch_idx * M * K + m_idx * M0 * K + k_idx * K0;
            }

            if (trans_b) {
                offset_b = batch_idx * K * N + n_idx * N0 * K + k_idx * K0;
            } else {
                offset_b = batch_idx * K * N + k_idx * K0 * N + n_idx * N0;
            }

            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;
            int32_t k_round = (k_actual + 15) / 16 * 16;
            int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

            auto a_l1_tensor = ping_flag ? a_base_l1_tensor : a_base_l1_tensor[L1_PINGPONG_BUFFER_LEN];
            auto b_l1_tensor = ping_flag ? b_base_l1_tensor : b_base_l1_tensor[L1_PINGPONG_BUFFER_LEN];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
            if (trans_a) {
                if (M == 1) {
                    AscendC::DataCopy(
                        a_l1_tensor, a_gm_tensor[offset_a],
                        AscendC::DataCopyParams(1, (k_actual + C0_SIZE - 1) / C0_SIZE, 0, 0)
                    );
                } else {
                    load_matrix_zZ(a_l1_tensor, a_gm_tensor[offset_a], K0, M0, k_actual, m_actual, M);
                }
            } else {
                if (m_actual == 1) {
                    AscendC::DataCopy(
                        a_l1_tensor, a_gm_tensor[offset_a],
                        AscendC::DataCopyParams(1, (k_actual + C0_SIZE - 1) / C0_SIZE, 0, 0)
                    );
                } else {
                    load_matrix_zZ(a_l1_tensor, a_gm_tensor[offset_a], M0, K0, m_actual, k_actual, K);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
            if (trans_b) {
                load_matrix_zZ(b_l1_tensor, b_gm_tensor[offset_b], N0, K0, n_actual, k_actual, K);
            } else {
                load_matrix_zZ(b_l1_tensor, b_gm_tensor[offset_b], K0, N0, k_actual, n_actual, N);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            for (int k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto l0a_tensor = l0a_base_tensor[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];
                auto l0b_tensor = l0b_base_tensor[(k_part_idx % 2) * L0AB_PINGPONG_BUFFER_LEN];

                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                if (M == 1 || m_actual == 1 && !trans_a) {
                    AscendC::LoadData(
                        l0a_tensor, a_l1_tensor[k_part_idx * k_part_len],
                        AscendC::LoadData2dParams(0, (k0_round + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, 1, 0, 0, false, inc)
                    );
                } else {
                    if (trans_a) {
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * M0];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadDataWithTranspose(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
                                AscendC::LoadData2dTransposeParams(0, k0_round / BLOCK_SIZE, M0 / BLOCK_SIZE, 1, 0, inc)
                            );
                        }
                    } else {
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadData(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * BLOCK_SIZE * K0],
                                AscendC::LoadData2dParams(0, k0_round / C0_SIZE, 1, 0, 0, false, inc)
                            );
                        }
                    }
                }
                if (k_part_idx == k_part_loop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
                }

                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);
                }
                if (trans_b) {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                    for (int i = 0; i < k0_round / C0_SIZE; i++) {
                        AscendC::LoadData(
                            l0b_tensor[i * n_round * C0_SIZE],
                            src_b_l1_tensor[i * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(0, n_round / BLOCK_SIZE, K0 / C0_SIZE, 0, 0, false, inc)
                        );
                    }
                } else {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * N0];
                    for (int i = 0; i < n_round / BLOCK_SIZE; i++) {
                        AscendC::LoadDataWithTranspose(
                            l0b_tensor[i * CUBE_MATRIX_SIZE],
                            src_b_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dTransposeParams(0, k0_round / BLOCK_SIZE, N0 / BLOCK_SIZE,
                                2 * n_round / BLOCK_SIZE - 1, n_round / BLOCK_SIZE - 1, inc)
                        );
                    }
                }
                if (k_part_idx == k_part_loop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                }

                if (M != 1 && m_actual == 1 && trans_a) {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(16, n_actual, k0_actual, 0, false, init_c);
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(l0c_tensor, l0a_tensor, l0b_tensor, mmParam);
                } else {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(m_actual, n_actual, k0_actual, 0, false, init_c);
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(l0c_tensor, l0a_tensor, l0b_tensor, mmParam);
                }
                AscendC::PipeBarrier<PIPE_M>();
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
            }

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            ping_flag = 1 - ping_flag;
        }

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        auto intriParams = AscendC::FixpipeParamsV220(n_actual, m_actual, m_round, N, false);
        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = 0;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(c_gm_tensor[offset_c], l0c_tensor, intriParams);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

    AscendC::PipeBarrier<PIPE_ALL>();
}

}

#endif
