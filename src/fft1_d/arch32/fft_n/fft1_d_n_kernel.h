/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT1_D_N_KERNEL_H
#define FFT1_D_N_KERNEL_H

#include "kernel/fft_common_kernel.h"

namespace FFT1DNKernel {

constexpr int32_t ASCENDFFT_FORWARD = -1;  // Forward FFT

using T_INPUT = float;
using T_OUTPUT = float;

constexpr uint64_t INTER_CORE_SYNC_FLAG_ID = 0;
constexpr uint64_t INTER_CORE_SYNC_MODE = 0;
constexpr uint64_t VC_SYNC_FLAG_ID = 6;
constexpr uint64_t CV_SYNC_FLAG_ID = 7;
constexpr uint64_t CV_SYNC_MODE = 2;

constexpr int32_t ND_INPUT = 0;
constexpr int32_t TILING_NZ_INPUT = 1;
constexpr int32_t NZ_INPUT = 2;

constexpr int32_t WORKSPACE_LEN = 128 * 128 * 20;
constexpr int32_t BLOCK_LEN_32B = 32 / sizeof(T_INPUT);
constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(T_INPUT);  // 32 KB
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t C0_SIZE = 32 / sizeof(T_INPUT);
constexpr int32_t CUBE_MATRIX_SIZE = BLOCK_SIZE * C0_SIZE;                // 16 * 8
constexpr int32_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(T_INPUT);  // 256 KB

constexpr uint32_t AIC_PING_SET_FLAG = 2;
constexpr uint32_t AIC_PING_WAIT_FLAG = 3;
constexpr uint32_t AIC_PONG_SET_FLAG = 4;
constexpr uint32_t AIC_PONG_WAIT_FLAG = 5;

constexpr uint32_t AIV_PING_SET_FLAG = 3;
constexpr uint32_t AIV_PING_WAIT_FLAG = 2;
constexpr uint32_t AIV_PONG_SET_FLAG = 5;
constexpr uint32_t AIV_PONG_WAIT_FLAG = 4;

#define AIC_MTE2
#define AIC_MTE1
#define MAD_FIFXPIPE

#define AIV_MTE2
#define AIV_MTE3
#define PIPEV

inline __aicore__ void ConvertLayoutRowMajorTozZ(__gm__ T_INPUT *__restrict__ gm_nd, __gm__ T_INPUT *__restrict__ gm_nz,
                                                 int32_t B, int32_t R, int32_t C)
{
    AscendC::GlobalTensor<T_INPUT> gm_nd_tensor;
    AscendC::GlobalTensor<T_INPUT> gm_nz_tensor;
    gm_nd_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_nd));
    gm_nz_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_nz));

    constexpr int32_t R0 = 96;
    constexpr int32_t C0 = 128;

    int32_t r_loop = (R + R0 - 1) / R0;
    int32_t c_loop = (C + C0 - 1) / C0;
    int32_t loop = B * r_loop * c_loop;

    constexpr int32_t BUF_SIZE = 48 * 1024;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<T_INPUT> buf0Ping = buf.GetBuffer<BufferType::ASCEND_UB, T_INPUT>(0);
    AscendC::LocalTensor<T_INPUT> buf0Pong = buf.GetBuffer<BufferType::ASCEND_UB, T_INPUT>(BUF_SIZE);
    AscendC::LocalTensor<T_INPUT> buf1Ping = buf.GetBuffer<BufferType::ASCEND_UB, T_INPUT>(2 * BUF_SIZE);
    AscendC::LocalTensor<T_INPUT> buf1Pong = buf.GetBuffer<BufferType::ASCEND_UB, T_INPUT>(3 * BUF_SIZE);

    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);
    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);

    int32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int32_t aivIdx = AscendC::GetBlockIdx();

    bool isPing = true;

    for (int32_t loop_idx = aivIdx; loop_idx < loop; loop_idx += aivNum) {
        int64_t batch_idx = loop_idx / (r_loop * c_loop);
        int64_t in_batch_idx = loop_idx % (r_loop * c_loop);
        int64_t r_idx;
        int64_t c_idx;

        r_idx = in_batch_idx / c_loop;
        c_idx = in_batch_idx % c_loop;

        int32_t r_actual = (r_idx < (r_loop - 1)) ? R0 : (R - r_idx * R0);
        int32_t c_actual = (c_idx < (c_loop - 1)) ? C0 : (C - c_idx * C0);
        int32_t c_round = (c_actual + 15) / 16 * 16;

        auto buf0 = isPing ? buf0Ping : buf0Pong;
        auto buf1 = isPing ? buf1Ping : buf1Pong;
        auto buf0EventId = isPing ? EVENT_ID0 : EVENT_ID1;
        auto buf1EventId = isPing ? EVENT_ID0 : EVENT_ID1;

        WAIT_FLAG(V, MTE2, buf0EventId);
        int32_t srcStride = (C - c_round) / BLOCK_LEN_32B;
        int64_t ndOffset = batch_idx * R * C + r_idx * R0 * C + c_idx * C0;
#ifdef AIV_MTE2
        if (srcStride > 65535) {
            for (int64_t r = 0; r < r_actual; r++) {
                AscendC::DataCopy(
                    buf0[r * C0],
                    gm_nd_tensor[ndOffset + r * C],
                    AscendC::DataCopyParams(
                        1,                          // blockCount
                        c_round / BLOCK_LEN_32B,    // blockLen
                        0,                          // srcStride
                        0                           // dstStride
                    )
                );
            }
        } else {
            AscendC::DataCopy(
                buf0,
                gm_nd_tensor[ndOffset],
                AscendC::DataCopyParams(
                    r_actual,                       // blockCount
                    c_round / BLOCK_LEN_32B,        // blockLen
                    srcStride,                      // srcStride
                    0                               // dstStride
                )
            );
        }
#endif

        SET_FLAG(MTE2, V, buf0EventId);
        WAIT_FLAG(MTE2, V, buf0EventId);
        WAIT_FLAG(MTE3, V, buf1EventId);

        int32_t r_round = (r_actual + 15) / 16 * 16;
        int32_t addRepeats = (c_round + 63) / 64;
        // nd2nz
#ifdef PIPEV
        for (int32_t r = 0; r < r_round; r++) {
            adds_v<ArchType::ASCEND_V220, T_INPUT>(
                  buf1[(r / BLOCK_SIZE) * BLOCK_SIZE * c_round + (r % BLOCK_SIZE) * BLOCK_LEN_32B], buf0[r * C0], 0.f,
                  addRepeats,
                  BLOCK_SIZE,
                  1,
                  8 * BLOCK_SIZE,
                  8
            );
        }
#endif

        SET_FLAG(V, MTE2, buf0EventId);

        SET_FLAG(V, MTE3, buf1EventId);
        WAIT_FLAG(V, MTE3, buf1EventId);

        int64_t RC = R * C;
        int32_t R0C = R0 * C;
        int32_t nBursts = c_round * BLOCK_SIZE / BLOCK_LEN_32B;
#ifdef AIV_MTE3
        for (int64_t r = 0; r < r_round / BLOCK_SIZE; r++) {
            AscendC::DataCopy(
                gm_nz_tensor[batch_idx * RC + r_idx * R0C + c_idx * C0 * BLOCK_SIZE + r * C * BLOCK_SIZE],
                buf1[r * c_round * BLOCK_SIZE],
                AscendC::DataCopyParams(
                    1,          // blockCount
                    nBursts,    // blockLen
                    0,          // srcStride
                    0           // dstStride
                )
            );
        }
#endif
        SET_FLAG(MTE3, V, buf1EventId);

        isPing = !isPing;
    }
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);
}

__aicore__ __inline__ void vec_vreduce(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, int32_t len,
                                       uint8_t patternMode)
{
    int32_t ele_num_per_block = 32 / sizeof(float);
    int32_t num_of_8blocks = len / ele_num_per_block / 8;  // 256
    int32_t ele_num_in_8blocks = 8 * ele_num_per_block;    // 64
    int32_t output_repeat_ele_num = ele_num_in_8blocks / 2;
    int32_t max_repeat_num = 255;
    int32_t loop = num_of_8blocks / max_repeat_num;  // 1
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)0, (uint64_t)-1);

    for (int32_t i = 0; i < loop; i++) {
        vreducev2(dst + max_repeat_num * output_repeat_ele_num * i, src + max_repeat_num * ele_num_in_8blocks * i,
                  nullptr, max_repeat_num, 1, patternMode, 8, 8);
    }

    int32_t num_of_remanent_ele = len % (max_repeat_num * ele_num_in_8blocks);  // 64
    int32_t num_of_remanent_blocks = num_of_remanent_ele / ele_num_per_block;   // 8
    int32_t repeat = num_of_remanent_blocks / 8;                                // 1
    int32_t remain = num_of_remanent_ele % ele_num_in_8blocks;
    // Handle remanent blocks
    if (repeat) {
        vreducev2(dst + max_repeat_num * output_repeat_ele_num * loop, src + max_repeat_num * ele_num_in_8blocks * loop,
                  nullptr, repeat, 1, patternMode, 8, 8);
    }

    if (remain) {
        AscendC::SetMaskCount();
        SetVectorMask<float>((uint64_t)0, (uint64_t)remain);
        vreducev2(dst + max_repeat_num * output_repeat_ele_num * loop + repeat * output_repeat_ele_num,
                  src + max_repeat_num * ele_num_in_8blocks * loop + repeat * ele_num_in_8blocks, nullptr, 1, 1,
                  patternMode, 8, 8);
        AscendC::SetMaskNorm();
        SetVectorMask<float>((uint64_t)0, (uint64_t)-1);
    }
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

    uint64_t mask[1] = { UINT64_MAX };

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

__aicore__ __inline__ void PartitionAddDev(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src0,
    AscendC::LocalTensor<float> src1,
    int32_t row, int32_t col)
{
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)0, (uint64_t)-1);

    if (col >= 128) {
        int32_t r_stride = col / 8 * 2;  // repeat stride
        for (int32_t i = 0; i < col / 64; i++) {
            add_v<ArchType::ASCEND_V220, float>(dst[i * 64], src0[i * 64], src1[i * 64], row / 2, 1, 1, 1, r_stride, r_stride, r_stride);
        }
    } else if (col == 64) {
        add_v<ArchType::ASCEND_V220, float>(dst, src0, src1, row / 2, 1, 1, 1, 16, 16, 16);
    } else if (col == 32) {
        AscendC::SetMaskCount();
        SetVectorMask<float>((uint64_t)0, (uint64_t)32);
        for (int32_t row_idx = 0; row_idx < row / 2; row_idx++) {
            add_v<ArchType::ASCEND_V220, float>(dst[2 * col * row_idx], src0[2 * col * row_idx], src1[2 * col * row_idx], 1, 1, 1, 1, 0, 0, 0);  // repeat = 32 / 8
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
        int32_t remain = (row / 2) % 8;  // 列方向不满足8block
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

__aicore__ __inline__ void PartitionSubDev(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src0,
    AscendC::LocalTensor<float> src1,
    int32_t row, int32_t col)
{
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)0, (uint64_t)-1);

    if (col >= 128) {
        int32_t r_stride = col / 8 * 2;  // repeat stride
        for (int32_t i = 0; i < col / 64; i++) {
            sub_v<ArchType::ASCEND_V220, float>(dst[i * 64], src0[i * 64], src1[i * 64], row / 2, 1, 1, 1, r_stride, r_stride, r_stride);
        }
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

__aicore__ __inline__ void PartitionMulDev(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src0,
    AscendC::LocalTensor<float> src1,
    int32_t row, int32_t col)
{
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)0, (uint64_t)-1);

    if (col >= 128) {
        int32_t r_stride = col / 8 * 2;  // repeat stride
        for (int32_t i = 0; i < col / 64; i++) {
            mul_v<ArchType::ASCEND_V220, float>(dst[i * 64], src0[i * 64], src1[i * 64], row / 2, 1, 1, 1, r_stride, r_stride, r_stride);
        }
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

__aicore__ __inline__ void SeparateRIzZLayOut(__gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_output,
                                              int32_t inputN, int32_t M, int32_t N, int32_t batchSize)
{
    int32_t N0 = M < 128 ? 256 : 128;
    int32_t n_loop = N / N0;
    int32_t loop = batchSize * n_loop;

    AscendC::GlobalTensor<float> gm_input_tensor;
    AscendC::GlobalTensor<float> gm_output_tensor;
 
    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> input_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> output_vreduce_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(65536);
    AscendC::LocalTensor<float> output_nz_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(131072);

    auto input_ub = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input_ub_tensor.GetPhyAddr()));
    auto output_vreduce_ub = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_vreduce_ub_tensor.GetPhyAddr()));
    auto output_nz_ub = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_nz_ub_tensor.GetPhyAddr()));

    int32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int32_t aivIdx = AscendC::GetBlockIdx();

    for (int32_t loop_idx = aivIdx; loop_idx < loop; loop_idx += aivNum) {
        int32_t batch_idx = loop_idx / n_loop;
        int32_t n_idx = loop_idx % n_loop;

        int32_t offset_input = batch_idx * 2 * inputN + n_idx * N0 * 2;
        int32_t offset_output = batch_idx * 2 * inputN + M * N0 * n_idx;
#ifdef AIV_MTE2
        if ((N - N0) / 8 < 32768) {                   // (2 * N  - 2 * N0) / 8 < 65536
            AscendC::DataCopy(
                input_ub_tensor,
                gm_input_tensor[offset_input],
                AscendC::DataCopyParams(
                    M / 2, // blockCount
                    2 * N0 / 8, // blockLen
                    2 * N / 8 - 2 * N0 / 8, // srcStride
                    0 // dstStride
                )
            );
        } else {
            for (int32_t i = 0; i < M / 2; ++i) {
                AscendC::DataCopy(
                    input_ub_tensor[i * 2 * N0],
                    gm_input_tensor[offset_input + i * 2 * N],
                    AscendC::DataCopyParams(
                        1, // blockCount
                        2 * N0 / 8, // blockLen
                        0, // srcStride
                        0 // dstStride
                    )
                );
            }
        }
#endif
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
#ifdef PIPEV
        vec_vreduce(reinterpret_cast<__ubuf__ uint32_t *>(output_vreduce_ub),
                    reinterpret_cast<__ubuf__ uint32_t *>(input_ub), M * N0, 1);

        vec_vreduce(reinterpret_cast<__ubuf__ uint32_t *>(output_vreduce_ub) + M / 2 * N0,
                    reinterpret_cast<__ubuf__ uint32_t *>(input_ub), M * N0, 2);

        for (int32_t r = 0; r < M; r++) {
            adds_v<ArchType::ASCEND_V220, float>(
                  output_nz_ub_tensor[(r / BLOCK_SIZE) * BLOCK_SIZE * N0 + (r % BLOCK_SIZE) * BLOCK_LEN_32B],
                  output_vreduce_ub_tensor[r * N0],
                  0.f,
                  N0 / 64,
                  BLOCK_SIZE,
                  1,
                  8 * BLOCK_SIZE,
                  8
            );
        }
#endif
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);

        // ub2out
#ifdef AIV_MTE3
        AscendC::DataCopy(
            gm_output_tensor[offset_output],
            output_nz_ub_tensor,
            AscendC::DataCopyParams(
                1,          // blockCount
                M * N0 / 8, // blockLen
                0,          // srcStride
                0           // dstStride
            )
        );
#endif
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    }
}

__aicore__ __inline__ void SeparateRIwithCalculation(__gm__ float *__restrict__ gm_input,
                                                     __gm__ float *__restrict__ gm_t_matrix,
                                                     __gm__ float *__restrict__ gm_output, int32_t fftDirection,
                                                     int32_t inputN, int32_t M, int32_t N, int32_t batchSize)
{
    AscendC::GlobalTensor<float> gm_input_tensor;
    AscendC::GlobalTensor<float> gm_t_matrix_tensor;
    AscendC::GlobalTensor<float> gm_output_tensor;
    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_t_matrix_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_t_matrix));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));
    // restrict M to 4 (radix 2)
    int32_t N0 = 2048;  // 2 * (2048 * 2) * 4B = 32KB

    // 需要注意N < 2048的场景，会报错
    int32_t n_loop = (N + N0 - 1) / N0;
    int32_t loop = batchSize * n_loop;
    N0 = N < N0 ? N : N0;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> input1_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> input2_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(65536);
    AscendC::LocalTensor<float> output_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(131072);
    AscendC::LocalTensor<float> input1_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(32768);
    AscendC::LocalTensor<float> input2_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(98304);
    AscendC::LocalTensor<float> output_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(163840);

    int32_t ping_flag = 1;
    auto input1_ub = ping_flag ? input1_ping : input1_pong;
    auto input2_ub = ping_flag ? input2_ping : input2_pong;
    auto output_ub = ping_flag ? output_ping : output_pong;
    auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

    int32_t ping_pre_nidx = -1;
    int32_t pong_pre_nidx = -1;
    int32_t pre_nidx = -1;

    int32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    if (aivNum == 0) {
        aivNum = 1;
    }
    int32_t aivIdx = AscendC::GetBlockIdx();
    int32_t gm_loop_idx = 0;

    int32_t total_loop_per_core = loop / aivNum;
    int32_t tail_num = loop % aivNum;

    if (aivIdx < tail_num) {
        gm_loop_idx = aivIdx + aivIdx * total_loop_per_core;
    } else {
        gm_loop_idx = tail_num + aivIdx * total_loop_per_core;
    }

    if (aivIdx < tail_num) {
        total_loop_per_core += 1;
    }

    int32_t gm_loop_idx_end = gm_loop_idx + total_loop_per_core;

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (int32_t loop_idx = gm_loop_idx; loop_idx < gm_loop_idx_end; loop_idx += 1) {
        int32_t batch_idx, n_idx;

        input1_ub = ping_flag ? input1_ping : input1_pong;
        input2_ub = ping_flag ? input2_ping : input2_pong;
        output_ub = ping_flag ? output_ping : output_pong;
        event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
        pre_nidx = ping_flag ? ping_pre_nidx : pong_pre_nidx;

        batch_idx = loop_idx % batchSize;
        n_idx = loop_idx / batchSize;

        int32_t offset_input1 = batch_idx * 2 * inputN + n_idx * N0 * 2;
        int32_t offset_input2 = n_idx * N0;
        int32_t offset_output = batch_idx * 2 * inputN + N0 * n_idx;

        WAIT_FLAG(MTE3, MTE2, event_id);

        // load input data
        if ((N - N0) < 262144) {                       // (2 * N  - 2 * N0) / 8 < 65536
            AscendC::DataCopy(
                input1_ub,
                gm_input_tensor[offset_input1],
                AscendC::DataCopyParams(
                    M / 2,                  // blockCount
                    2 * N0 / 8,             // blockLen
                    2 * N / 8 - 2 * N0 / 8, // srcStride
                    0                       // dstStride
                )
            );
        } else {
            for (int32_t i = 0; i < M / 2; ++i) {
                AscendC::DataCopy(
                    input1_ub[i * 2 * N0],
                    gm_input_tensor[offset_input1 + i * 2 * N],
                    AscendC::DataCopyParams(
                        1,                      // blockCount
                        2 * N0 / 8,             // blockLen
                        0,                      // srcStride
                        0                       // dstStride
                    )
                );
            }
        }

        if (n_idx != pre_nidx) {
            // Load T matrix, radix 2 only load half of the twiddle factor matrix
            if ((N - N0) < 524288) {                          // (N  - N0) / 8 < 65536
                AscendC::DataCopy(
                    input2_ub,
                    gm_t_matrix_tensor[offset_input2],
                    AscendC::DataCopyParams(
                        2,                      // blockCount
                        N0 / 8,                 // blockLen
                        N / 8 - N0 / 8,         // srcStride
                        0                       // dstStride
                    )
                );
            } else {
                for (int32_t i = 0; i < 2; ++i) {
                    AscendC::DataCopy(
                        input2_ub[i * N0],
                        gm_t_matrix_tensor[offset_input2 + i * N],
                        AscendC::DataCopyParams(
                            1,                      // blockCount
                            N0 / 8,                 // blockLen
                            0,                      // srcStride
                            0                       // dstStride
                        )
                    );
                }
            }
            ping_flag ? ping_pre_nidx = n_idx : pong_pre_nidx = n_idx;
        }

        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);

        // separate RI
        vec_vreduce(reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)(output_ub.GetPhyAddr())),
                    reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)(input1_ub.GetPhyAddr())), M * N0, 1);
        vec_vreduce(reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)(output_ub.GetPhyAddr())) + M / 2 * N0,
                    reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)(input1_ub.GetPhyAddr())), M * N0, 2);
        PIPE_BARRIER(V);

        // matmul
        int32_t repeats = N0 / 64;

        add_v<ArchType::ASCEND_V220>(input1_ub, output_ub, output_ub[N0], repeats, 1, 1, 1, 8, 8, 8); // R1 + R2
        add_v<ArchType::ASCEND_V220>(input1_ub[N0], output_ub[2 * N0], output_ub[3 * N0], repeats, 1, 1, 1, 8, 8, 8); // I1 + I2
        sub_v<ArchType::ASCEND_V220>(input2_ub[2 * N0], output_ub, output_ub[N0], repeats, 1, 1, 1, 8, 8, 8); // R1 - R2
        sub_v<ArchType::ASCEND_V220>(input2_ub[3 * N0], output_ub[2 * N0], output_ub[3 * N0], repeats, 1, 1, 1, 8, 8, 8); // I1 - I2
        PIPE_BARRIER(V);

        // complex mul, skip the computation of the first half of the data
        if (fftDirection == ASCENDFFT_FORWARD) {
            // R1 * R2 and I1 * I2
            mul_v<ArchType::ASCEND_V220>(output_ub, input2_ub[2 * N0], input2_ub, 2 * repeats, 1, 1, 1, 8, 8, 8);
            //  R1 * I2
            mul_v<ArchType::ASCEND_V220>(output_ub[2 * N0], input2_ub[2 * N0], input2_ub[N0], repeats, 1, 1, 1, 8, 8, 8);
            // I1 * R2
            mul_v<ArchType::ASCEND_V220>(output_ub[3 * N0], input2_ub[3 * N0], input2_ub, repeats, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            // real
            sub_v<ArchType::ASCEND_V220>(input1_ub[2 * N0], output_ub, output_ub[N0], repeats, 1, 1, 1, 8, 8, 8);
            // img
            add_v<ArchType::ASCEND_V220>(input1_ub[3 * N0], output_ub[2 * N0], output_ub[3 * N0], repeats, 1, 1, 1, 8, 8, 8);
        } else {
            // R1 * R2 and I1 * I2
            mul_v<ArchType::ASCEND_V220>(output_ub, input2_ub[2 * N0], input2_ub, 2 * repeats, 1, 1, 1, 8, 8, 8);
            //  R1 * I2
            mul_v<ArchType::ASCEND_V220>(output_ub[2 * N0], input2_ub[2 * N0], input2_ub[N0], repeats, 1, 1, 1, 8, 8, 8);
            // I1 * R2
            mul_v<ArchType::ASCEND_V220>(output_ub[3 * N0], input2_ub[3 * N0], input2_ub, repeats, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            // real
            add_v<ArchType::ASCEND_V220>(input1_ub[2 * N0], output_ub, output_ub[N0], repeats, 1, 1, 1, 8, 8, 8);
            // img
            sub_v<ArchType::ASCEND_V220>(input1_ub[3 * N0], output_ub[3 * N0], output_ub[2 * N0], repeats, 1, 1, 1, 8, 8, 8);
        }

        SET_FLAG(V, MTE3, event_id);
        WAIT_FLAG(V, MTE3, event_id);

        // ub2out

        if ((N - N0) / 8 < 65536) {
            AscendC::DataCopy(
                gm_output_tensor[offset_output],
                input1_ub,
                AscendC::DataCopyParams(
                    M,                      // blockCount
                    N0 / 8,                 // blockLen
                    0,                      // srcStride
                    N / 8 - N0 / 8          // dstStride
                )
            );
        } else {
            for (int32_t i = 0; i < M; ++i) {
                AscendC::DataCopy(
                    gm_output_tensor[offset_output + i * N],
                    input1_ub[i * N0],
                    AscendC::DataCopyParams(
                        1,                      // blockCount
                        N0 / 8,                 // blockLen
                        0,                      // srcStride
                        0                       // dstStride
                    )
                );
            }
        }

        SET_FLAG(MTE3, MTE2, event_id);
        ping_flag = 1 - ping_flag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

__aicore__ __inline__ void SeparateRI(__gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_output,
                                      int32_t inputN, int32_t batchSize)
{
    AscendC::GlobalTensor<float> gm_input_tensor;
    AscendC::GlobalTensor<float> gm_output_tensor;
    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));

    int32_t N0 = 4096;
    int32_t realN = 2 * inputN;
    int32_t n_loop = (inputN + N0 - 1) / N0;
    if (n_loop == 0) {
        n_loop = 1;
    }

    int32_t loop = batchSize * n_loop;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> input_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> output_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(65536);
    AscendC::LocalTensor<float> input_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(32768);
    AscendC::LocalTensor<float> output_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(131072);

    int32_t ping_flag = 0;
    auto vreduce_input_ub = ping_flag ? input_ping : input_pong;
    auto vreduce_output_ub = ping_flag ? output_ping : output_pong;
    auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);

    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);

    int32_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int32_t aivIdx = AscendC::GetBlockIdx();

    for (int32_t loop_idx = aivIdx; loop_idx < loop; loop_idx += aivNum) {
        vreduce_input_ub = ping_flag ? input_ping : input_pong;
        vreduce_output_ub = ping_flag ? output_ping : output_pong;
        event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

        int32_t batch_idx = loop_idx / n_loop;
        int32_t n_idx = loop_idx % n_loop;

        int32_t n_actual = (n_idx == (n_loop - 1)) ? (inputN - n_idx * N0) : N0;
        int32_t offset_a = batch_idx * realN + n_idx * N0 * 2;
        int32_t offset_br = batch_idx * realN + n_idx * N0;
        int32_t offset_bi = batch_idx * realN + n_idx * N0 + inputN;

        WAIT_FLAG(V, MTE2, event_id);
        AscendC::DataCopy(
            vreduce_input_ub,
            gm_input_tensor[offset_a],
            AscendC::DataCopyParams(
                1,          // blockCount
                2 * n_actual / 8, // blockLen
                0,          // srcStride
                0           // dstStride
            )
        );

        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE3, V, event_id);

        vec_vreduce(reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)(vreduce_output_ub.GetPhyAddr())),
                    reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)(vreduce_input_ub.GetPhyAddr())), 2 * n_actual, 1);
        PIPE_BARRIER(V);

        vec_vreduce(reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)(vreduce_output_ub.GetPhyAddr())) + n_actual,
                    reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)(vreduce_input_ub.GetPhyAddr())), 2 * n_actual, 2);

        SET_FLAG(V, MTE3, event_id);
        WAIT_FLAG(V, MTE3, event_id);

        SET_FLAG(V, MTE2, event_id);

        // ub2out
        AscendC::DataCopy(
            gm_output_tensor[offset_br],
            vreduce_output_ub,
            AscendC::DataCopyParams(
                1,          // blockCount
                n_actual / 8, // blockLen
                0,          // srcStride
                0           // dstStride
            )
        );

        AscendC::DataCopy(
            gm_output_tensor[offset_bi],
            vreduce_output_ub[n_actual],
            AscendC::DataCopyParams(
                1,          // blockCount
                n_actual / 8, // blockLen
                0,          // srcStride
                0           // dstStride
            )
        );

        SET_FLAG(MTE3, V, event_id);
        ping_flag = 1 - ping_flag;
    }
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);

    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);

    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ void ppmatmul(__gm__ T_INPUT *__restrict__ gm_a, __gm__ T_INPUT *__restrict__ gm_b,
                                    __gm__ T_OUTPUT *__restrict__ gm_c, int32_t batchSize, int32_t M, int32_t K,
                                    int32_t N, int32_t layout, int32_t lessTCopy, int32_t syncTilingNum)
{
    if (batchSize <= 0) {
        batchSize = 1;
    }
    uint64_t flag_id;
    uint64_t mode;
    uint64_t config;

    SetPadding(0);
    AscendC::SetAtomicNone();
    uint64_t nd_config = 0x1;
    AscendC::SetNdParaImpl(nd_config);

    AscendC::GlobalTensor<T_INPUT> gm_a_tensor;
    AscendC::GlobalTensor<T_INPUT> gm_b_tensor;
    AscendC::GlobalTensor<T_OUTPUT> gm_c_tensor;

    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_OUTPUT *>(gm_c));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<T_INPUT> l1_base_a_tensor = buf.GetBuffer<BufferType::ASCEND_CB, T_INPUT>(0);
    AscendC::LocalTensor<T_INPUT> l1_base_b_tensor = buf.GetBuffer<BufferType::ASCEND_CB, T_INPUT>(128 * 1024);

    AscendC::LocalTensor<T_INPUT> l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, T_INPUT>(0);
    AscendC::LocalTensor<T_INPUT> l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, T_INPUT>(0);
    AscendC::LocalTensor<float> l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    auto l1_base_a = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(l1_base_a_tensor.GetPhyAddr()));
    auto l1_base_b = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(l1_base_b_tensor.GetPhyAddr()));

    auto l0a_base = reinterpret_cast<__ca__ T_INPUT *>((uintptr_t)(l0a_base_tensor.GetPhyAddr()));
    auto l0b_base = reinterpret_cast<__cb__ T_INPUT *>((uintptr_t)(l0b_base_tensor.GetPhyAddr()));
    auto l0c_buf = reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf_tensor.GetPhyAddr()));

    int32_t trans_a = 0;
    int32_t trans_b = 0;

    int32_t M0 = 128;
    int32_t N0 = M < 128 ? 256 : 128;
    int32_t K0 = 128;

    int32_t resident_a = 1;
    bool first_batch_idx = true;

    int32_t n_loop = (N + N0 - 1) / N0;
    if (n_loop == 0) {
        n_loop = 1;
    }
    int32_t loop = batchSize * n_loop;

    int32_t ping_flag = 1;
    int32_t mte1_mad_ping_flag = 1;
    int32_t sync_ping = 1;

    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);

    int32_t syn_counter = 0;
    int32_t sync_block_num = syncTilingNum;
    int32_t workspace_block_num = 2 * sync_block_num;
    int32_t sync_write_idx = sync_block_num - 1;
    int32_t insyn_index = 0;

    int32_t total_loop_per_core = loop / get_block_num();
    int32_t tail_num = loop % get_block_num();

    int32_t gm_loop_idx = 0;

    if (get_block_idx() < tail_num) {
        gm_loop_idx = get_block_idx() + get_block_idx() * total_loop_per_core;
    } else {
        gm_loop_idx = tail_num + get_block_idx() * total_loop_per_core;
    }

    if (get_block_idx() < tail_num) {
        total_loop_per_core += 1;
    }

    int32_t gm_loop_idx_end = gm_loop_idx + total_loop_per_core;

    for (int32_t loop_idx = gm_loop_idx; loop_idx < gm_loop_idx_end; loop_idx += 1) {
        int64_t batch_idx, n_idx;

        if (lessTCopy) {
            batch_idx = loop_idx % batchSize;
            n_idx = loop_idx / batchSize;
        } else {
            batch_idx = loop_idx / n_loop;
            n_idx = loop_idx % n_loop;
        }

        int64_t offset_b = batch_idx * K * N + n_idx * N0;
        int64_t offset_c = batch_idx * M * N + n_idx * N0;
        int32_t m_actual = M;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        int32_t k_actual = K;
        int32_t k_round = (k_actual + 15) / 16 * 16;
        int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

        __cbuf__ T_INPUT *l1_buf_a = nullptr;
        AscendC::LocalTensor<T_INPUT> l1_buf_a_tensor;

        l1_buf_a = l1_base_a;
        l1_buf_a_tensor = l1_base_a_tensor;
        auto l1_buf_b = ping_flag ? l1_base_b : l1_base_b + L1_PINGPONG_BUFFER_LEN;
        auto l1_buf_b_tensor = ping_flag ? l1_base_b_tensor : l1_base_b_tensor[L1_PINGPONG_BUFFER_LEN];
        event_t l1_event_id_a = EVENT_ID0;
        auto l1_event_id_b = ping_flag ? EVENT_ID2 : EVENT_ID3;

        // *** load matrix A to L1
        if (first_batch_idx) {
            load_matrix_zZ(l1_buf_a_tensor, gm_a_tensor, M0, K0, m_actual, k_actual, K);
            SET_FLAG(MTE2, MTE1, l1_event_id_a);
        }

        // *** load matrix B to L1
        WAIT_FLAG(MTE1, MTE2, l1_event_id_b);
        if (layout == ND_INPUT) {
            load_matrix_zZ(l1_buf_b_tensor, gm_b_tensor[offset_b], K0, N0, k_actual, n_actual, N);

        } else if (layout == TILING_NZ_INPUT) {
            AscendC::DataCopy(
                l1_buf_b_tensor, gm_b_tensor[batch_idx * K * N + n_idx * N0 * M],
                AscendC::DataCopyParams(
                    1,                                      // nBurst
                    N0 * M / 8,     // lenBurst
                    0,                                      // srcGap
                    0                                       // dstGap
                )
            );

        } else if (layout == NZ_INPUT) {
            if (N - N0 < 32768) {
                AscendC::DataCopy(
                    l1_buf_b_tensor, gm_b_tensor[batch_idx * K * N + n_idx * N0 * BLOCK_SIZE],
                    AscendC::DataCopyParams(
                        K0 / BLOCK_SIZE,                        // nBurst
                        N0 * BLOCK_SIZE / C0_SIZE,              // lenBurst
                        (N - N0) * BLOCK_SIZE / C0_SIZE,        // srcGap
                        0                                       // dstGap
                    )
                );
            } else {
                for (int64_t r = 0; r < K0 / BLOCK_SIZE; r++) {
                    AscendC::DataCopy(
                        l1_buf_b_tensor[r * N0 * BLOCK_SIZE], gm_b_tensor[batch_idx * K * N + n_idx * N0 * BLOCK_SIZE + r * N * BLOCK_SIZE],
                        AscendC::DataCopyParams(
                            1,                        // nBurst
                            N0 * BLOCK_SIZE / C0_SIZE,              // lenBurst
                            0,        // srcGap
                            0                                       // dstGap
                        )
                    );
                }
            }
        }
        SET_FLAG(MTE2, MTE1, l1_event_id_b);

        for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
            int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
            int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

            auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
            auto l0a_buf = l0a_base + (1 - mte1_mad_ping_flag) * L0AB_PINGPONG_BUFFER_LEN;
            auto l0b_buf = l0b_base + (1 - mte1_mad_ping_flag) * L0AB_PINGPONG_BUFFER_LEN;
            auto l0a_buf_tensor = l0a_base_tensor[(1 - mte1_mad_ping_flag) * L0AB_PINGPONG_BUFFER_LEN];
            auto l0b_buf_tensor = l0b_base_tensor[(1 - mte1_mad_ping_flag) * L0AB_PINGPONG_BUFFER_LEN];

            // *** load matrix A from L1 to L0A
            if (first_batch_idx) {
                first_batch_idx = false;
                WAIT_FLAG(MTE2, MTE1, l1_event_id_a);
            }

            WAIT_FLAG(M, MTE1, mte1_mad_event_id);

            auto l1_src_a = l1_buf_a + k_part_idx * k_part_len * BLOCK_SIZE;
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

            // *** load matrix B from L1 to L0B
            if (k_part_idx == 0) {
                WAIT_FLAG(MTE2, MTE1, l1_event_id_b);
            }

            auto l1_src_b = l1_buf_b + k_part_idx * k_part_len * N0;
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

            if (k_part_idx == k_part_loop - 1) {
                SET_FLAG(MTE1, MTE2, l1_event_id_b);
            }

            SET_FLAG(MTE1, M, mte1_mad_event_id);
            WAIT_FLAG(MTE1, M, mte1_mad_event_id);

            bool init_c = k_part_idx == 0;
            uint8_t unit_flag = 0b10;
            if (k_part_idx == k_part_loop - 1) {
                unit_flag = 0b11;
            }

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

            PIPE_BARRIER(M);
            SET_FLAG(M, MTE1, mte1_mad_event_id);

            mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
        }

        ping_flag = 1 - ping_flag;

        // load L0C to gm
        if (syn_counter % sync_block_num == 0) {
            flag_id = sync_ping ? AIC_PING_WAIT_FLAG : AIC_PONG_WAIT_FLAG;
            WaitFlagDev(flag_id);
        }

        int32_t N_tiling = N < N0 ? N : N0;

        insyn_index = syn_counter % workspace_block_num;
        auto intriParams = AscendC::FixpipeParamsV220(n_actual, // nSize
                                            m_actual, // mSize
                                            m_round,   // srcStride
                                            N_tiling,   // dstStride
                                            false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = 0b11;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gm_c_tensor[get_block_idx() * WORKSPACE_LEN + insyn_index * 16384], l0c_buf_tensor, intriParams);

        if ((syn_counter % sync_block_num == sync_write_idx) || (syn_counter == total_loop_per_core - 1)) {
            flag_id = sync_ping ? AIC_PING_SET_FLAG : AIC_PONG_SET_FLAG;
            FftsCrossCoreSync<PIPE_FIX, CV_SYNC_MODE>(flag_id);
            sync_ping = 1 - sync_ping;
        }
        syn_counter += 1;
    }

    if (sync_ping) {
        WaitFlagDev(AIC_PING_WAIT_FLAG);
        WaitFlagDev(AIC_PONG_WAIT_FLAG);
    } else {
        WaitFlagDev(AIC_PONG_WAIT_FLAG);
        WaitFlagDev(AIC_PING_WAIT_FLAG);
    }

    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ void ppmatmul_stride(__gm__ T_INPUT *__restrict__ gm_a, __gm__ T_INPUT *__restrict__ gm_b,
                                           __gm__ T_OUTPUT *__restrict__ gm_c, int32_t inputN, int32_t batchSize,
                                           int32_t matmulBatchsize, int32_t matmulLoop, int32_t M, int32_t K,
                                           int32_t N, int32_t lessTCopy, int32_t syncTilingNum)
{
    if (matmulLoop == 0) {
        matmulLoop = 1;
    }
    if (matmulBatchsize == 0) {
        matmulBatchsize = 1;
    }
    if (batchSize == 0) {
        batchSize = 1;
    }
    uint64_t flag_id;
    uint64_t mode;
    uint64_t config;

    SetPadding(0);
    AscendC::SetAtomicNone();
    uint64_t nd_config = 0x1;
    AscendC::SetNdParaImpl(nd_config);

    AscendC::GlobalTensor<T_INPUT> gm_a_tensor;
    AscendC::GlobalTensor<T_INPUT> gm_b_tensor;
    AscendC::GlobalTensor<T_OUTPUT> gm_c_tensor;

    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_OUTPUT *>(gm_c));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<T_INPUT> l1_base_a_tensor = buf.GetBuffer<BufferType::ASCEND_CB, T_INPUT>(0);
    AscendC::LocalTensor<T_INPUT> l1_base_b_tensor = buf.GetBuffer<BufferType::ASCEND_CB, T_INPUT>(128 * 1024);

    AscendC::LocalTensor<T_INPUT> l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, T_INPUT>(0);
    AscendC::LocalTensor<T_INPUT> l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, T_INPUT>(0);
    AscendC::LocalTensor<float> l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    auto l1_base_a = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(l1_base_a_tensor.GetPhyAddr()));
    auto l1_base_b = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(l1_base_b_tensor.GetPhyAddr()));

    auto l0a_base = reinterpret_cast<__ca__ T_INPUT *>((uintptr_t)(l0a_base_tensor.GetPhyAddr()));
    auto l0b_base = reinterpret_cast<__cb__ T_INPUT *>((uintptr_t)(l0b_base_tensor.GetPhyAddr()));
    auto l0c_buf = reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf_tensor.GetPhyAddr()));

    int32_t trans_a = 0;
    int32_t trans_b = 0;

    int32_t M0 = 128;
    int32_t N0 = M < 128 ? 256 : 128;
    int32_t K0 = 128;

    int32_t resident_a = 1;
    bool first_batch_idx = true;

    int32_t m_loop = (M + M0 - 1) / M0;
    int32_t n_loop = (N + N0 - 1) / N0;
    int32_t k_loop = (K + K0 - 1) / K0;
    int32_t loop = batchSize * matmulBatchsize * matmulLoop * m_loop * n_loop;

    int32_t ping_flag = 1;
    int32_t mte1_mad_ping_flag = 1;
    int32_t sync_ping = 1;

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);

    int32_t syn_counter = 0;
    int32_t sync_block_num = syncTilingNum;
    int32_t workspace_block_num = 2 * sync_block_num;
    int32_t sync_write_idx = sync_block_num - 1;
    int32_t insyn_index = 0;

    int32_t total_loop_per_core = loop / get_block_num();
    int32_t tail_num = loop % get_block_num();

    int32_t gm_loop_idx = 0;
    if (get_block_idx() < tail_num) {
        gm_loop_idx = get_block_idx() + get_block_idx() * total_loop_per_core;
    } else {
        gm_loop_idx = tail_num + get_block_idx() * total_loop_per_core;
    }

    if (get_block_idx() < tail_num) {
        total_loop_per_core += 1;
    }

    int32_t gm_loop_idx_end = gm_loop_idx + total_loop_per_core;

    for (int32_t loop_idx = gm_loop_idx; loop_idx < gm_loop_idx_end; loop_idx += 1) {
        int64_t batch_idx, n_idx, mloop_idx, input_batch_idx;

        batch_idx = (loop_idx / matmulLoop) % matmulBatchsize;
        mloop_idx = loop_idx % matmulLoop;
        input_batch_idx = loop_idx / (matmulLoop * matmulBatchsize) % batchSize;
        n_idx = loop_idx / (matmulBatchsize * matmulLoop * batchSize);

        int64_t offset_b =
            input_batch_idx * inputN * 2 + mloop_idx * K * N + batch_idx * K * N * matmulLoop + n_idx * N0;
        int64_t offset_c =
            input_batch_idx * inputN * 2 + mloop_idx * matmulBatchsize * M * N + batch_idx * M * N + n_idx * N0;
        int32_t m_actual = M;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        int32_t k_actual = K;
        int32_t k_round = (k_actual + 15) / 16 * 16;
        int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

        AscendC::LocalTensor<T_INPUT> l1_buf_a_tensor;

        l1_buf_a_tensor = l1_base_a_tensor;
        auto l1_buf_b_tensor = ping_flag ? l1_base_b_tensor : l1_base_b_tensor[L1_PINGPONG_BUFFER_LEN];
        event_t l1_event_id_a = EVENT_ID0;

        auto l1_event_id_b = ping_flag ? EVENT_ID2 : EVENT_ID3;

        // *** load matrix A to L1
        WAIT_FLAG(MTE1, MTE2, l1_event_id_a);

        if (first_batch_idx) {
            load_matrix_zZ(l1_buf_a_tensor, gm_a_tensor, M0, K0, m_actual, k_actual, K);
        }
        SET_FLAG(MTE2, MTE1, l1_event_id_a);

        // *** load matrix B to L1
        WAIT_FLAG(MTE1, MTE2, l1_event_id_b);

        load_matrix_zZ(l1_buf_b_tensor, gm_b_tensor[offset_b], K0, N0, k_actual, n_actual, N);

        SET_FLAG(MTE2, MTE1, l1_event_id_b);

        for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
            int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
            int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

            auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
            auto l0a_buf_tensor = l0a_base_tensor[(1 - mte1_mad_ping_flag) * L0AB_PINGPONG_BUFFER_LEN];
            auto l0b_buf_tensor = l0b_base_tensor[(1 - mte1_mad_ping_flag) * L0AB_PINGPONG_BUFFER_LEN];

            // *** load matrix A from L1 to L0A
            if (k_part_idx == 0) {
                WAIT_FLAG(MTE2, MTE1, l1_event_id_a);
            }
            WAIT_FLAG(M, MTE1, mte1_mad_event_id);

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

            if (k_part_idx == k_part_loop - 1) {
                SET_FLAG(MTE1, MTE2, l1_event_id_a);
            }

            // *** load matrix B from L1 to L0B
            if (k_part_idx == 0) {
                WAIT_FLAG(MTE2, MTE1, l1_event_id_b);
            }

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

            if (k_part_idx == k_part_loop - 1) {
                SET_FLAG(MTE1, MTE2, l1_event_id_b);
            }

            SET_FLAG(MTE1, M, mte1_mad_event_id);
            WAIT_FLAG(MTE1, M, mte1_mad_event_id);

            bool init_c = k_part_idx == 0;
            uint8_t unit_flag = 0b10;
            if (k_part_idx == k_part_loop - 1) {
                unit_flag = 0b11;
            }

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

            PIPE_BARRIER(M);
            SET_FLAG(M, MTE1, mte1_mad_event_id);

            mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
        }

        ping_flag = 1 - ping_flag;
        first_batch_idx = false;

        if (syn_counter % sync_block_num == 0) {
            flag_id = sync_ping ? AIC_PING_WAIT_FLAG : AIC_PONG_WAIT_FLAG;
            WaitFlagDev(flag_id);
        }

        int32_t N_tiling = N < N0 ? N : N0;

        insyn_index = syn_counter % workspace_block_num;

        auto intriParams = AscendC::FixpipeParamsV220(n_actual, // nSize
                                            m_actual, // mSize
                                            m_round,   // srcStride
                                            N_tiling,   // dstStride
                                            false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = 0b11;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gm_c_tensor[get_block_idx() * WORKSPACE_LEN + insyn_index * 16384], l0c_buf_tensor, intriParams);

        if ((syn_counter % sync_block_num == sync_write_idx) || (syn_counter == total_loop_per_core - 1)) {
            flag_id = sync_ping ? AIC_PING_SET_FLAG : AIC_PONG_SET_FLAG;
            FftsCrossCoreSync<PIPE_FIX, CV_SYNC_MODE>(flag_id);
            sync_ping = 1 - sync_ping;
        }
        syn_counter += 1;
    }

    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);

    if (sync_ping) {
        WaitFlagDev(AIC_PING_WAIT_FLAG);
        WaitFlagDev(AIC_PONG_WAIT_FLAG);
    } else {
        WaitFlagDev(AIC_PONG_WAIT_FLAG);
        WaitFlagDev(AIC_PING_WAIT_FLAG);
    }

    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ void ppmatmul_transpose(__gm__ T_INPUT *__restrict__ gm_a, __gm__ T_INPUT *__restrict__ gm_b,
                                              __gm__ T_OUTPUT *__restrict__ gm_c, int32_t inputN, int32_t batchSize,
                                              int32_t matmulBatchsize, int32_t M, int32_t K, int32_t N,
                                              int32_t syncTilingNum)
{
    if (matmulBatchsize == 0) {
        matmulBatchsize = 1;
    }
    if (batchSize == 0) {
        batchSize = 1;
    }
    uint64_t flag_id;
    uint64_t mode;
    uint64_t config;

    SetPadding(0);
    AscendC::SetAtomicNone();
    config = 0x1;
    AscendC::SetNdParaImpl(config);

    AscendC::GlobalTensor<T_INPUT> gm_a_tensor;
    AscendC::GlobalTensor<T_INPUT> gm_b_tensor;
    AscendC::GlobalTensor<T_OUTPUT> gm_c_tensor;

    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_OUTPUT *>(gm_c));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<T_INPUT> l1_base_a_tensor = buf.GetBuffer<BufferType::ASCEND_CB, T_INPUT>(0);
    AscendC::LocalTensor<T_INPUT> l1_base_b_tensor = buf.GetBuffer<BufferType::ASCEND_CB, T_INPUT>(128 * 1024);

    AscendC::LocalTensor<T_INPUT> l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, T_INPUT>(0);
    AscendC::LocalTensor<T_INPUT> l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, T_INPUT>(0);
    AscendC::LocalTensor<float> l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    auto l1_base_a = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(l1_base_a_tensor.GetPhyAddr()));
    auto l1_base_b = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(l1_base_b_tensor.GetPhyAddr()));

    auto l0a_base = reinterpret_cast<__ca__ T_INPUT *>((uintptr_t)(l0a_base_tensor.GetPhyAddr()));
    auto l0b_base = reinterpret_cast<__cb__ T_INPUT *>((uintptr_t)(l0b_base_tensor.GetPhyAddr()));
    auto l0c_buf = reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf_tensor.GetPhyAddr()));

    int32_t trans_a = 0;
    int32_t trans_b = 1;

    int32_t M0 = 128;
    int32_t N0 = M < 128 ? 256 : 128;
    int32_t K0 = 128;

    int32_t resident_a = 1;
    bool first_batch_idx = true;

    int32_t m_loop = (M + M0 - 1) / M0;
    if (m_loop == 0) {
        m_loop = 1;
    }
    int32_t n_loop = (N + N0 - 1) / N0;
    if (n_loop == 0) {
        n_loop = 1;
    }
    int32_t k_loop = (K + K0 - 1) / K0;
    int32_t loop = batchSize * matmulBatchsize * m_loop * n_loop;

    int32_t ping_flag = 1;
    int32_t sync_ping = 1;
    int32_t mte1_mad_ping_flag = 1;

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);

    int32_t syn_counter = 0;
    int32_t sync_block_num = syncTilingNum;
    int32_t workspace_block_num = 2 * sync_block_num;
    int32_t sync_write_flag = sync_block_num - 1;
    int32_t insyn_index = 0;

    int32_t total_loop_per_core = loop / AscendC::GetBlockNum();
    int32_t tail_num = loop % AscendC::GetBlockNum();

    int32_t gm_loop_idx = 0;
    if (AscendC::GetBlockIdx() < tail_num) {
        gm_loop_idx = AscendC::GetBlockIdx() + AscendC::GetBlockIdx() * total_loop_per_core;
    } else {
        gm_loop_idx = tail_num + AscendC::GetBlockIdx() * total_loop_per_core;
    }

    if (AscendC::GetBlockIdx() < tail_num) {
        total_loop_per_core += 1;
    }

    int32_t gm_loop_idx_end = gm_loop_idx + total_loop_per_core;

    for (int32_t loop_idx = gm_loop_idx; loop_idx < gm_loop_idx_end; loop_idx += 1) {
        int64_t batch_idx = loop_idx % matmulBatchsize;
        int64_t m_idx = 0;
        int64_t n_idx = (loop_idx / matmulBatchsize) % n_loop;
        int64_t input_batch_idx = loop_idx / (n_loop * m_loop * matmulBatchsize);

        int64_t offset_b;
        int64_t offset_c = input_batch_idx * inputN * 2 + batch_idx * N + m_idx * M0 * matmulBatchsize * N + n_idx * N0;
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        if (trans_b) {
            offset_b = input_batch_idx * inputN * 2 + batch_idx * K + n_idx * N0 * matmulBatchsize * K;
        } else {
            offset_b = input_batch_idx * inputN * 2 + batch_idx * N + n_idx * N0;
        }

        int32_t k_actual = K;
        int32_t k_round = (k_actual + 15) / 16 * 16;
        int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

        AscendC::LocalTensor<T_INPUT> l1_buf_a_tensor;
        if (!resident_a) {
            l1_buf_a_tensor = ping_flag ? l1_base_a_tensor : l1_base_a_tensor[L1_PINGPONG_BUFFER_LEN];
        } else {
            l1_buf_a_tensor = l1_base_a_tensor;
        }
        auto l1_buf_b_tensor = ping_flag ? l1_base_b_tensor : l1_base_b_tensor[L1_PINGPONG_BUFFER_LEN];
        event_t l1_event_id_a;
        if (!resident_a) {
            l1_event_id_a = ping_flag ? EVENT_ID0 : EVENT_ID1;
        } else {
            l1_event_id_a = EVENT_ID0;
        }
        auto l1_event_id_b = ping_flag ? EVENT_ID2 : EVENT_ID3;

        // *** load matrix A to L1
        WAIT_FLAG(MTE1, MTE2, l1_event_id_a);

        if (first_batch_idx) {
            load_matrix_zZ(l1_buf_a_tensor, gm_a_tensor, M0, K0, m_actual, k_actual, K);
        }

        SET_FLAG(MTE2, MTE1, l1_event_id_a);

        // *** load matrix B to L1
        WAIT_FLAG(MTE1, MTE2, l1_event_id_b);

        if (trans_b) {
            load_matrix_zZ(l1_buf_b_tensor, gm_b_tensor[offset_b], N0, K0, n_actual, k_actual, matmulBatchsize * K);
        } else {
            load_matrix_zZ(l1_buf_b_tensor, gm_b_tensor[offset_b], K0, N0, k_actual, n_actual, matmulBatchsize * N);
        }

        SET_FLAG(MTE2, MTE1, l1_event_id_b);

        for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
            int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
            int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

            auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
            auto l0a_buf_tensor = l0a_base_tensor[(1 - mte1_mad_ping_flag) * L0AB_PINGPONG_BUFFER_LEN];
            auto l0b_buf_tensor = l0b_base_tensor[(1 - mte1_mad_ping_flag) * L0AB_PINGPONG_BUFFER_LEN];

            // *** load matrix A from L1 to L0A
            if (k_part_idx == 0) {
                WAIT_FLAG(MTE2, MTE1, l1_event_id_a);
            }
            WAIT_FLAG(M, MTE1, mte1_mad_event_id);

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

            if (k_part_idx == k_part_loop - 1) {
                SET_FLAG(MTE1, MTE2, l1_event_id_a);
            }

            // *** load matrix B from L1 to L0B
            if (k_part_idx == 0) {
                WAIT_FLAG(MTE2, MTE1, l1_event_id_b);
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
                SET_FLAG(MTE1, MTE2, l1_event_id_b);
            }

            SET_FLAG(MTE1, M, mte1_mad_event_id);
            WAIT_FLAG(MTE1, M, mte1_mad_event_id);

            bool init_c = (k_part_idx == 0);
            uint8_t unit_flag = 0b10;
            if (k_part_idx == k_part_loop - 1) {
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

            mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
        }
        ping_flag = 1 - ping_flag;
        first_batch_idx = false;

        if (syn_counter % sync_block_num == 0) {
            flag_id = sync_ping ? AIC_PING_WAIT_FLAG : AIC_PONG_WAIT_FLAG;
            WaitFlagDev(flag_id);
        }

        int32_t N_tiling = N < N0 ? N : N0;
        insyn_index = syn_counter % workspace_block_num;
        auto intriParams = AscendC::FixpipeParamsV220(n_actual, // nSize
                                            m_actual, // mSize
                                            m_round,   // srcStride
                                            N_tiling,   // dstStride
                                            false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = 0b11;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gm_c_tensor[get_block_idx() * WORKSPACE_LEN + insyn_index * 16384], l0c_buf_tensor, intriParams);

        if ((syn_counter % sync_block_num == sync_write_flag) || (syn_counter == total_loop_per_core - 1)) {
            flag_id = sync_ping ? AIC_PING_SET_FLAG : AIC_PONG_SET_FLAG;
            FftsCrossCoreSync<PIPE_FIX, 2>(flag_id);
            sync_ping = 1 - sync_ping;
        }
        syn_counter += 1;
    }

    if (sync_ping) {
        WaitFlagDev(AIC_PING_WAIT_FLAG);
        WaitFlagDev(AIC_PONG_WAIT_FLAG);
    } else {
        WaitFlagDev(AIC_PONG_WAIT_FLAG);
        WaitFlagDev(AIC_PING_WAIT_FLAG);
    }

    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);

    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ void complex_mul(__gm__ float *__restrict__ gm_a, __gm__ float *__restrict__ gm_b,
                                       __gm__ float *__restrict__ gm_c, int32_t fftDirection, int32_t batchSize,
                                       int32_t M, int32_t N, int32_t lessTCopy, int32_t syncTilingNum)
{
    if (batchSize == 0) {
        batchSize = 1;
    }

    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    int32_t rows = M / 2;
    int32_t N0 = M < 128 ? 256 : 128;

    int32_t n_loop = (N + N0 - 1) / N0;
    if (n_loop == 0) {
        n_loop = 1;
    }
    int32_t loop = batchSize * n_loop;
    N0 = N < N0 ? N : N0;

    uint64_t flag_id;
    uint64_t mode;
    uint64_t config;

    AscendC::GlobalTensor<float> gm_a_tensor;
    AscendC::GlobalTensor<float> gm_b_tensor;
    AscendC::GlobalTensor<float> gm_c_tensor;

    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_c));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> input1_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> input2_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(65536);
    AscendC::LocalTensor<float> output_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(131072);
    AscendC::LocalTensor<float> input1_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(32768);
    AscendC::LocalTensor<float> input2_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(98304);
    AscendC::LocalTensor<float> output_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(163840);

    auto input1_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_ping_tensor.GetPhyAddr()));
    auto input2_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_ping_tensor.GetPhyAddr()));
    auto output_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_ping_tensor.GetPhyAddr()));
    auto input1_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_pong_tensor.GetPhyAddr()));
    auto input2_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_pong_tensor.GetPhyAddr()));
    auto output_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_pong_tensor.GetPhyAddr()));

    int32_t ping_flag = 0;
    auto input1_ub = ping_flag ? input1_ping : input1_pong;
    auto input2_ub = ping_flag ? input2_ping : input2_pong;
    auto output_ub = ping_flag ? output_ping : output_pong;

    auto input1_ub_tensor = ping_flag ? input1_ping_tensor : input1_pong_tensor;
    auto input2_ub_tensor = ping_flag ? input2_ping_tensor : input2_pong_tensor;
    auto output_ub_tensor = ping_flag ? output_ping_tensor : output_pong_tensor;
    auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

    int32_t ping_pre_nidx = -1;
    int32_t pong_pre_nidx = -1;
    int32_t pre_nidx = -1;

    int32_t sync_ping = 1;
    int32_t syn_counter = 0;
    int32_t sync_block_num = syncTilingNum;
    int32_t workspace_block_num = 2 * sync_block_num;
    int32_t sync_write_flag = sync_block_num - 1;
    int32_t insyn_index = 0;

    FftsCrossCoreSync<PIPE_MTE2, CV_SYNC_MODE>(AIV_PING_SET_FLAG);
    FftsCrossCoreSync<PIPE_MTE2, CV_SYNC_MODE>(AIV_PONG_SET_FLAG);

    int32_t total_loop_per_core = loop / AscendC::GetBlockNum();
    int32_t tail_num = loop % AscendC::GetBlockNum();

    int32_t gm_loop_idx = 0;
    auto curr_block_idx = (AscendC::GetBlockIdx() - AscendC::GetSubBlockIdx()) / AscendC::GetTaskRation();
    if (curr_block_idx < tail_num) {
        gm_loop_idx = curr_block_idx + curr_block_idx * total_loop_per_core;
    } else {
        gm_loop_idx = tail_num + curr_block_idx * total_loop_per_core;
    }

    if (curr_block_idx < tail_num) {
        total_loop_per_core += 1;
    }

    int32_t gm_loop_idx_end = gm_loop_idx + total_loop_per_core;

    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);

    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);

    for (int32_t loop_idx = gm_loop_idx; loop_idx < gm_loop_idx_end; loop_idx += 1) {
        int64_t batch_idx, n_idx;

        if (lessTCopy) {
            batch_idx = loop_idx % batchSize;
            n_idx = loop_idx / batchSize;
        } else {
            batch_idx = loop_idx / n_loop;
            n_idx = loop_idx % n_loop;
        }

        input1_ub = ping_flag ? input1_ping : input1_pong;
        input2_ub = ping_flag ? input2_ping : input2_pong;
        output_ub = ping_flag ? output_ping : output_pong;

        input1_ub_tensor = ping_flag ? input1_ping_tensor : input1_pong_tensor;
        input2_ub_tensor = ping_flag ? input2_ping_tensor : input2_pong_tensor;
        output_ub_tensor = ping_flag ? output_ping_tensor : output_pong_tensor;

        event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
        pre_nidx = ping_flag ? ping_pre_nidx : pong_pre_nidx;

        insyn_index = syn_counter % workspace_block_num;
        int64_t offset_a = curr_block_idx * WORKSPACE_LEN + insyn_index * 16384;
        int64_t offset_b = n_idx * N0;
        int64_t offset_c = batch_idx * M * N + n_idx * N0;

        if (AscendC::GetSubBlockIdx() == 1) {
            offset_a = offset_a + N0 * M / 2;
            offset_b = offset_b + N * M / 2;
            offset_c = offset_c + N * M / 2;
        }

        if (syn_counter % sync_block_num == 0) {
            flag_id = sync_ping ? AIV_PING_WAIT_FLAG : AIV_PONG_WAIT_FLAG;
            WaitFlagDev(flag_id);
        }

        WAIT_FLAG(V, MTE2, event_id);

#ifdef AIV_MTE2
        if (n_idx != pre_nidx) {
            if ((N - N0) / 8 < 65536) {
                AscendC::DataCopy(
                    input2_ub_tensor,
                    gm_b_tensor[offset_b],
                    AscendC::DataCopyParams(
                        M / 2, // blockCount
                        N0 / 8, // blockLen
                        N / 8 - N0 / 8, // srcStride
                        0 // dstStride
                    )
                );
            } else {
                for (int32_t i = 0; i < M / 2; ++i) {
                    AscendC::DataCopy(
                        input2_ub_tensor[i * N0],
                        gm_b_tensor[offset_b + i * N],
                        AscendC::DataCopyParams(
                            1, // blockCount
                            N0 / 8, // blockLen
                            0, // srcStride
                            0 // dstStride
                        )
                    );
                }
            }
            ping_flag ? ping_pre_nidx = n_idx : pong_pre_nidx = n_idx;
        }

        // workspace
        AscendC::DataCopy(
            input1_ub_tensor,
            gm_a_tensor[offset_a],
            AscendC::DataCopyParams(
                1, // blockCount
                (M / 2 * N0) / 8, // blockLen
                0, // srcStride
                0 // dstStride
            )
        );
#endif

        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE3, V, event_id);

        if (syn_counter % sync_block_num == sync_write_flag || (syn_counter == total_loop_per_core - 1)) {
            flag_id = sync_ping ? AIV_PING_SET_FLAG : AIV_PONG_SET_FLAG;
            FftsCrossCoreSync<PIPE_MTE2, CV_SYNC_MODE>(flag_id);
            sync_ping = 1 - sync_ping;
        }
#ifdef PIPEV
        // calculation
        if (fftDirection == ASCENDFFT_FORWARD) {
            ContinuousMulDev(output_ub_tensor, input1_ub_tensor, input2_ub_tensor, rows, N0);
            PIPE_BARRIER(V);

            // calculate real part
            PartitionSubDev(output_ub_tensor, output_ub_tensor, output_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);

            // calculate R1 * I2
            PartitionMulDev(input1_ub_tensor, input1_ub_tensor, input2_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);

            // calculate I1 * R2
            PartitionMulDev(input1_ub_tensor[N0], input1_ub_tensor[N0], input2_ub_tensor, rows, N0);
            PIPE_BARRIER(V);

            PartitionAddDev(output_ub_tensor[N0], input1_ub_tensor, input1_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);
        } else {
            ContinuousMulDev(output_ub_tensor, input1_ub_tensor, input2_ub_tensor, rows, N0);
            PIPE_BARRIER(V);

            // calculate real part
            PartitionAddDev(output_ub_tensor, output_ub_tensor, output_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);

            // calculate R1 * I2
            PartitionMulDev(input1_ub_tensor, input1_ub_tensor, input2_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);

            // calculate I1 * R2
            PartitionMulDev(input1_ub_tensor[N0], input1_ub_tensor[N0], input2_ub_tensor, rows, N0);
            PIPE_BARRIER(V);

            PartitionSubDev(output_ub_tensor[N0], input1_ub_tensor[N0], input1_ub_tensor, rows, N0);
            PIPE_BARRIER(V);
        }
#endif
        SET_FLAG(V, MTE3, event_id);
        WAIT_FLAG(V, MTE3, event_id);

        SET_FLAG(V, MTE2, event_id);
#ifdef AIV_MTE3
        // ub2out
        if ((N - N0) / 8 < 65536) {
            AscendC::DataCopy(
                gm_c_tensor[offset_c],
                output_ub_tensor,
                AscendC::DataCopyParams(
                    M / 2, // blockCount
                    N0 / 8, // blockLen
                    0, // srcStride
                    N / 8 - N0 / 8 // dstStride
                )
            );
        } else {
            for (int32_t i = 0; i < M / 2; ++i) {
                AscendC::DataCopy(
                    gm_c_tensor[offset_c + i * N],
                    output_ub_tensor[i * N0],
                    AscendC::DataCopyParams(
                        1, // blockCount
                        N0 / 8, // blockLen
                        0, // srcStride
                        0 // dstStride
                    )
                );
            }
        }
#endif
        SET_FLAG(MTE3, V, event_id);
        ping_flag = 1 - ping_flag;

        syn_counter += 1;
    }
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);

    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);

    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ void complex_mul_stride(__gm__ float *__restrict__ gm_a, __gm__ float *__restrict__ gm_b,
                                              __gm__ float *__restrict__ gm_c, int32_t fftDirection, int32_t inputN,
                                              int32_t batchSize, int32_t matmulBatchsize, int32_t matmulLoop,
                                              int32_t M, int32_t N, int32_t lessTCopy, int32_t syncTilingNum)
{
    if (matmulLoop == 0) {
        matmulLoop = 1;
    }
    if (matmulBatchsize == 0) {
        matmulBatchsize = 1;
    }
    if (batchSize == 0) {
        batchSize = 1;
    }
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    int32_t rows = M / 2;
    int32_t N0 = M < 128 ? 256 : 128;

    int32_t n_loop = (N + N0 - 1) / N0;
    int32_t loop = batchSize * matmulBatchsize * matmulLoop * n_loop;
    N0 = N < N0 ? N : N0;

    uint64_t flag_id;
    uint64_t mode;
    uint64_t config;

    AscendC::GlobalTensor<float> gm_a_tensor;
    AscendC::GlobalTensor<float> gm_b_tensor;
    AscendC::GlobalTensor<float> gm_c_tensor;
 
    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_c));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> input1_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> input2_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(65536);
    AscendC::LocalTensor<float> output_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(131072);
    AscendC::LocalTensor<float> input1_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(32768);
    AscendC::LocalTensor<float> input2_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(98304);
    AscendC::LocalTensor<float> output_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(163840);

    auto input1_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_ping_tensor.GetPhyAddr()));
    auto input2_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_ping_tensor.GetPhyAddr()));
    auto output_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_ping_tensor.GetPhyAddr()));
    auto input1_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input1_pong_tensor.GetPhyAddr()));
    auto input2_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input2_pong_tensor.GetPhyAddr()));
    auto output_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_pong_tensor.GetPhyAddr()));

    int32_t ping_flag = 0;
    auto input1_ub = ping_flag ? input1_ping : input1_pong;
    auto input2_ub = ping_flag ? input2_ping : input2_pong;
    auto output_ub = ping_flag ? output_ping : output_pong;

    auto input1_ub_tensor = ping_flag ? input1_ping_tensor : input1_pong_tensor;
    auto input2_ub_tensor = ping_flag ? input2_ping_tensor : input2_pong_tensor;
    auto output_ub_tensor = ping_flag ? output_ping_tensor : output_pong_tensor;
    auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

    int32_t ping_pre_nidx = -1;
    int32_t pong_pre_nidx = -1;
    int32_t pre_nidx = -1;

    int32_t sync_ping = 1;
    int32_t syn_counter = 0;
    int32_t sync_block_num = syncTilingNum;
    int32_t workspace_block_num = 2 * sync_block_num;
    int32_t sync_write_flag = sync_block_num - 1;
    int32_t insyn_index = 0;

    FftsCrossCoreSync<PIPE_MTE2, CV_SYNC_MODE>(AIV_PING_SET_FLAG);
    FftsCrossCoreSync<PIPE_MTE2, CV_SYNC_MODE>(AIV_PONG_SET_FLAG);

    int32_t total_loop_per_core = loop / get_block_num();
    int32_t tail_num = loop % get_block_num();

    int32_t gm_loop_idx = 0;
    if (get_block_idx() < tail_num) {
        gm_loop_idx = get_block_idx() + get_block_idx() * total_loop_per_core;
    } else {
        gm_loop_idx = tail_num + get_block_idx() * total_loop_per_core;
    }

    if (get_block_idx() < tail_num) {
        total_loop_per_core += 1;
    }

    int32_t gm_loop_idx_end = gm_loop_idx + total_loop_per_core;

    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);

    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);

    for (int32_t loop_idx = gm_loop_idx; loop_idx < gm_loop_idx_end; loop_idx += 1) {
        input1_ub = ping_flag ? input1_ping : input1_pong;
        input2_ub = ping_flag ? input2_ping : input2_pong;
        output_ub = ping_flag ? output_ping : output_pong;
        input1_ub_tensor = ping_flag ? input1_ping_tensor : input1_pong_tensor;
        input2_ub_tensor = ping_flag ? input2_ping_tensor : input2_pong_tensor;
        output_ub_tensor = ping_flag ? output_ping_tensor : output_pong_tensor;
        event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
        pre_nidx = ping_flag ? ping_pre_nidx : pong_pre_nidx;

        int64_t batch_idx, in_batch_idx, n_idx, mloop_idx, input_batch_idx;

        batch_idx = (loop_idx / matmulLoop) % matmulBatchsize;
        mloop_idx = loop_idx % matmulLoop;
        input_batch_idx = loop_idx / (matmulLoop * matmulBatchsize) % batchSize;
        n_idx = loop_idx / (matmulBatchsize * matmulLoop * batchSize);

        // workspace
        insyn_index = syn_counter % workspace_block_num;
        int64_t offset_a = get_block_idx() * WORKSPACE_LEN + insyn_index * 16384;
        int64_t offset_b = n_idx * N0;
        int64_t offset_c =
            input_batch_idx * inputN * 2 + mloop_idx * matmulBatchsize * M * N + batch_idx * M * N + n_idx * N0;

        if (get_subblockid() == 1) {
            offset_a = offset_a + N0 * M / 2;  // workspace中的矩阵乘输出 上下切分
            offset_b = offset_b + N * M / 2;
            offset_c = offset_c + N * M / 2;
        }

        if (syn_counter % sync_block_num == 0) {
            flag_id = sync_ping ? AIV_PING_WAIT_FLAG : AIV_PONG_WAIT_FLAG;
            WaitFlagDev(flag_id);
        }

        WAIT_FLAG(V, MTE2, event_id);
#ifdef AIV_MTE2
        if (n_idx != pre_nidx) {
            if ((N - N0) / 8 < 65536) {
                AscendC::DataCopy(
                    input2_ub_tensor,
                    gm_b_tensor[offset_b],
                    AscendC::DataCopyParams(
                        M / 2, // blockCount
                        N0 / 8, // blockLen
                        N / 8 - N0 / 8, // srcStride
                        0 // dstStride
                    )
                );
            } else {
                for (int32_t i = 0; i < M / 2; ++i) {
                    AscendC::DataCopy(
                        input2_ub_tensor[i * N0],
                        gm_b_tensor[offset_b + i * N],
                        AscendC::DataCopyParams(
                            1, // blockCount
                            N0 / 8, // blockLen
                            0, // srcStride
                            0 // dstStride
                        )
                    );
                }
            }
            ping_flag ? ping_pre_nidx = n_idx : pong_pre_nidx = n_idx;
        }

        // workspace
        AscendC::DataCopy(
            input1_ub_tensor,
            gm_a_tensor[offset_a],
            AscendC::DataCopyParams(
                1, // blockCount
                (M / 2 * N0) / 8, // blockLen
                0, // srcStride
                0 // dstStride
            )
        );
#endif
        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE3, V, event_id);

        if (syn_counter % sync_block_num == sync_write_flag || (syn_counter == total_loop_per_core - 1)) {
            flag_id = sync_ping ? AIV_PING_SET_FLAG : AIV_PONG_SET_FLAG;
            FftsCrossCoreSync<PIPE_MTE2, CV_SYNC_MODE>(flag_id);
            sync_ping = 1 - sync_ping;
        }
#ifdef PIPEV
        if (fftDirection == ASCENDFFT_FORWARD) {
            ContinuousMulDev(output_ub_tensor, input1_ub_tensor, input2_ub_tensor, rows, N0);
            PIPE_BARRIER(V);

            // calculate real part
            PartitionSubDev(output_ub_tensor, output_ub_tensor, output_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);

            // calculate R1 * I2
            PartitionMulDev(input1_ub_tensor, input1_ub_tensor, input2_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);

            // calculate I1 * R2
            PartitionMulDev(input1_ub_tensor[N0], input1_ub_tensor[N0], input2_ub_tensor, rows, N0);
            PIPE_BARRIER(V);

            PartitionAddDev(output_ub_tensor[N0], input1_ub_tensor, input1_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);
        } else {
            ContinuousMulDev(output_ub_tensor, input1_ub_tensor, input2_ub_tensor, rows, N0);
            PIPE_BARRIER(V);

            // calculate real part
            PartitionAddDev(output_ub_tensor, output_ub_tensor, output_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);

            // calculate R1 * I2
            PartitionMulDev(input1_ub_tensor, input1_ub_tensor, input2_ub_tensor[N0], rows, N0);
            PIPE_BARRIER(V);

            // calculate I1 * R2
            PartitionMulDev(input1_ub_tensor[N0], input1_ub_tensor[N0], input2_ub_tensor, rows, N0);
            PIPE_BARRIER(V);

            PartitionSubDev(output_ub_tensor[N0], input1_ub_tensor[N0], input1_ub_tensor, rows, N0);
            PIPE_BARRIER(V);
        }
#endif
        SET_FLAG(V, MTE3, event_id);
        WAIT_FLAG(V, MTE3, event_id);

        SET_FLAG(V, MTE2, event_id);
#ifdef AIV_MTE3
        // ub2out
        if ((N - N0) / 8 < 65536) {
            AscendC::DataCopy(
                gm_c_tensor[offset_c],
                output_ub_tensor,
                AscendC::DataCopyParams(
                    M / 2, // blockCount
                    N0 / 8, // blockLen
                    0, // srcStride
                    N / 8 - N0 / 8 // dstStride
                )
            );
        } else {
            for (int32_t i = 0; i < M / 2; ++i) {
                AscendC::DataCopy(
                    gm_c_tensor[offset_c + i * N],
                    output_ub_tensor[i * N0],
                    AscendC::DataCopyParams(
                        1, // blockCount
                        N0 / 8, // blockLen
                        0, // srcStride
                        0 // dstStride
                    )
                );
            }
        }
#endif
        SET_FLAG(MTE3, V, event_id);
        ping_flag = 1 - ping_flag;

        syn_counter += 1;
    }

    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);

    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);

    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ void combine_RI(__gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_output,
                                      __gm__ int32_t *__restrict__ gm_index, int32_t inputN, int32_t batchSize,
                                      int32_t matmulBatchsize, int32_t M, int32_t N, int32_t syncTilingNum)
{
    if (matmulBatchsize == 0) {
        matmulBatchsize = 1;
    }

    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    int32_t rows = M / 2;
    int32_t N0 = M < 128 ? 256 : 128;

    int32_t n_loop = (N + N0 - 1) / N0;
    if (n_loop == 0) {
        n_loop = 1;
    }
    int32_t loop = batchSize * matmulBatchsize * n_loop;
    N0 = N < N0 ? N : N0;

    uint64_t flag_id;
    uint64_t mode;
    uint64_t config;

    AscendC::GlobalTensor<float> gm_input_tensor;
    AscendC::GlobalTensor<float> gm_output_tensor;
    AscendC::GlobalTensor<uint32_t> gm_index_tensor;
 
    gm_input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));
    gm_index_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gm_index));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<uint32_t> index_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(131072);

    AscendC::LocalTensor<float> input_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(0));
    AscendC::LocalTensor<float> output_ping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(65536));
    AscendC::LocalTensor<float> input_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(32768));
    AscendC::LocalTensor<float> output_pong_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(98304));

    auto index_ub = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)(index_ub_tensor.GetPhyAddr()));

    auto input_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input_ping_tensor.GetPhyAddr()));
    auto output_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_ping_tensor.GetPhyAddr()));
    auto input_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(input_pong_tensor.GetPhyAddr()));
    auto output_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)(output_pong_tensor.GetPhyAddr()));

    int32_t bias_addr_ping = 0;
    int32_t bias_addr_pong = 32768;

    int32_t ping_flag = 0;
    auto input_ub = ping_flag ? input_ping : input_pong;
    auto input_ub_tensor = ping_flag ? input_ping_tensor : input_pong_tensor;
    auto output_ub = ping_flag ? output_ping : output_pong;
    auto output_ub_tensor = ping_flag ? output_ping_tensor : output_pong_tensor;
    auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
    auto vgather_bias = ping_flag ? bias_addr_ping : bias_addr_pong;

    int32_t sync_ping = 1;
    int32_t syn_counter = 0;
    int32_t sync_block_num = syncTilingNum;
    int32_t workspace_block_num = 2 * sync_block_num;
    int32_t sync_write_flag = sync_block_num - 1;
    int32_t insyn_index = 0;

    FftsCrossCoreSync<PIPE_MTE2, CV_SYNC_MODE>(AIV_PING_SET_FLAG);
    FftsCrossCoreSync<PIPE_MTE2, CV_SYNC_MODE>(AIV_PONG_SET_FLAG);

    int32_t total_loop_per_core = loop / AscendC::GetBlockNum();
    int32_t tail_num = loop % AscendC::GetBlockNum();

    int32_t gm_loop_idx = 0;
    if (get_block_idx() < tail_num) {
        gm_loop_idx = get_block_idx() + get_block_idx() * total_loop_per_core;
    } else {
        gm_loop_idx = tail_num + get_block_idx() * total_loop_per_core;
    }

    if (get_block_idx() < tail_num) {
        total_loop_per_core += 1;
    }

    int32_t gm_loop_idx_end = gm_loop_idx + total_loop_per_core;

    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);

    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);

    // copy index
    AscendC::DataCopy(
        index_ub_tensor,
        gm_index_tensor,
        AscendC::DataCopyParams(
            1, // blockCount
            N0 * (M / 2) / 8, // blockLen
            0, // srcStride
            0 // dstStride
        )
    );

    for (int32_t loop_idx = gm_loop_idx; loop_idx < gm_loop_idx_end; loop_idx += 1) {
        input_ub = ping_flag ? input_ping : input_pong;
        input_ub_tensor = ping_flag ? input_ping_tensor : input_pong_tensor;
        output_ub = ping_flag ? output_ping : output_pong;
        output_ub_tensor = ping_flag ? output_ping_tensor : output_pong_tensor;
        event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
        vgather_bias = ping_flag ? bias_addr_ping : bias_addr_pong;

        int64_t batch_idx = loop_idx % matmulBatchsize;
        int64_t n_idx = (loop_idx / matmulBatchsize) % n_loop;
        int64_t input_batch_idx = loop_idx / (n_loop * matmulBatchsize);

        // AIC AIV 同步
        insyn_index = syn_counter % workspace_block_num;
        int64_t offset_R_input = get_block_idx() * WORKSPACE_LEN + insyn_index * 16384;
        int64_t offset_I_input = offset_R_input + N0 * (M / 2);
        int64_t offset_output = input_batch_idx * inputN * 2 + batch_idx * N * 2 + n_idx * N0 * 2;

        if (get_subblockid() == 1) {
            offset_R_input = offset_R_input + N0 * (M / 4);  // 下半部分R的偏移
            offset_I_input = offset_I_input + N0 * (M / 4);  // 下半部分I的偏移
            offset_output = offset_output + N * matmulBatchsize * 2 * (M / 4);
        }

        if (syn_counter % sync_block_num == 0) {
            flag_id = sync_ping ? AIV_PING_WAIT_FLAG : AIV_PONG_WAIT_FLAG;
            WaitFlagDev(flag_id);
        }

        WAIT_FLAG(V, MTE2, event_id);
#ifdef AIV_MTE2
        AscendC::DataCopy(
            input_ub_tensor,
            gm_input_tensor[offset_R_input],
            AscendC::DataCopyParams(
                2, // blockCount
                (M / 4) * N0 / 8, // blockLen
                (M / 4) * N0 / 8, // srcStride
                0 // dstStride
            )
        );

#endif
        if (syn_counter % sync_block_num == sync_write_flag || (syn_counter == total_loop_per_core - 1)) {
            flag_id = sync_ping ? AIV_PING_SET_FLAG : AIV_PONG_SET_FLAG;
            config = 1 | (CV_SYNC_MODE << 4) | (flag_id << 8);
            FftsCrossCoreSync<PIPE_MTE2, CV_SYNC_MODE>(flag_id);
            sync_ping = 1 - sync_ping;
        }

        // calculation
        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE3, V, event_id);
#ifdef PIPEV
        vgather((__ubuf__ uint32_t *)(uintptr_t)output_ub, index_ub, vgather_bias, 8,
                (M / 2) * N0 / 64);  // input bias is 0
#endif
        SET_FLAG(V, MTE3, event_id);
        WAIT_FLAG(V, MTE3, event_id);

        SET_FLAG(V, MTE2, event_id);

#ifdef AIV_MTE3
        // ub2out
        if (2 * (N * matmulBatchsize - N0) / 8 < 65536) {
            AscendC::DataCopy(
                gm_output_tensor[offset_output],
                output_ub_tensor,
                AscendC::DataCopyParams(
                    M / 4, // blockCount
                    2 * N0 / 8, // blockLen
                    0, // srcStride
                    2 * N * matmulBatchsize / 8 - 2 * N0 / 8 // dstStride
                )
            );
        } else {
            for (int32_t i = 0; i < M / 4; ++i) {
                AscendC::DataCopy(
                    gm_output_tensor[offset_output + i * (2 * N * matmulBatchsize)],
                    output_ub_tensor[i * (2 * N0)],
                    AscendC::DataCopyParams(
                        1, // blockCount
                        2 * N0 / 8, // blockLen
                        0, // srcStride
                        1 // dstStride
                    )
                );
            }
        }
#endif
        SET_FLAG(MTE3, V, event_id);
        ping_flag = 1 - ping_flag;

        syn_counter += 1;
    }

    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);

    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);

    PIPE_BARRIER(ALL);
}

constexpr int32_t test_step = 5;
constexpr int32_t step_test = false;

extern "C" __global__ __aicore__ void fft_n(__gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
                                            __gm__ float *__restrict__ gm_w_matrix,
                                            __gm__ float *__restrict__ gm_t_matrix,
                                            __gm__ int32_t *__restrict__ gm_index, __gm__ float *__restrict__ gm_output,
                                            __gm__ float *__restrict__ gm_workspace,
                                            __gm__ uint8_t *__restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    SetPadding(0);
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
    AscendC::SetAtomicNone();

    uint64_t flag_id;
    uint64_t mode;
    uint64_t config;

    auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    int32_t inputN = tiling_para[0];
    int32_t batchSize = tiling_para[1];
    int32_t repeatBatchSize = tiling_para[2];
    if (repeatBatchSize == 0) {
        repeatBatchSize = 1;
    }
    int32_t iterCount = tiling_para[3];

    int32_t N1 = tiling_para[4];
    if (N1 == 0) {
        N1 = 0;
    }
    int32_t N2 = tiling_para[5];
    int32_t N3 = tiling_para[6];
    int32_t N4 = tiling_para[7];
    int32_t N5 = tiling_para[8];
    int32_t fftDirection = tiling_para[29];

    int32_t isOutputHbm[5];
    int32_t lessTCopy[5];
    int32_t syncTilingNum[5];

    for (int32_t i = 0; i < iterCount; i++) {
        isOutputHbm[i] = tiling_para[i + 9];
        lessTCopy[i] = tiling_para[i + 19];
        syncTilingNum[i] = tiling_para[i + 24];
    }
    int32_t M = 0;
    int32_t K = 0;
    int32_t N = 0;

    __gm__ float *__restrict__ gmStepIn;
    __gm__ float *__restrict__ gmStepOut;

    int64_t nRepeats = (static_cast<int64_t>(batchSize) + repeatBatchSize - 1) / repeatBatchSize;

    for (int64_t iRepeat = 0; iRepeat < nRepeats; iRepeat++) {
        int64_t addrBias = iRepeat * static_cast<int64_t>(repeatBatchSize) * inputN * 2;
        int32_t iBatch = iRepeat == (nRepeats - 1)
            ? static_cast<int32_t>(batchSize - repeatBatchSize * iRepeat) : repeatBatchSize;

        int32_t iBatchSize = 1;
        int32_t w_bias = 0;
        int32_t matmulBatchsize = 1;
        int32_t matmulLoop = 0;
        int32_t layout;

        for (int32_t iter_i = 0; iter_i < iterCount; iter_i++) {
            gmStepIn = isOutputHbm[iter_i] ? gm_output : gm_input;

            if (iter_i == test_step + 1) {
                return;
            }

            if (iter_i == iterCount - 1) {
                int32_t pre_radix = tiling_para[4 + iter_i - 1];
                if (pre_radix == 0) {
                    pre_radix = 1;
                }
                int32_t radix = tiling_para[4 + iter_i];
                if (radix == 0) {
                    radix = 1;
                }
                matmulBatchsize = pre_radix;  // N1 * .. Ni-2
                M = radix * 2;
                K = M;
                N = inputN / radix / pre_radix;
                w_bias += 4 * pre_radix * pre_radix;

                WaitFlagDev(VC_SYNC_FLAG_ID);
                ppmatmul_transpose(gm_w_matrix + w_bias, gmStepIn + addrBias, gm_workspace, inputN, iBatch,
                                   matmulBatchsize, M, K, N, syncTilingNum[iter_i]);

            } else if (iter_i < 2) {
                if (iter_i == 0) {
                    iBatchSize = iBatch;
                    M = N1 * 2;
                    K = M;
                    N = inputN / N1;
                    w_bias = 0;
                    layout = N >= 65536 ? TILING_NZ_INPUT : ND_INPUT;

                    if (N1 != 2) {
                        // pass
                    } else {
                        continue;
                    }
                }
                if (iter_i == 1) {
                    iBatchSize = iBatch * N1;
                    M = N2 * 2;
                    K = M;
                    N = inputN / (N1 * N2);
                    w_bias += 4 * N1 * N1;
                    layout = N >= 65536 ? NZ_INPUT : ND_INPUT;
                }

                WaitFlagDev(VC_SYNC_FLAG_ID);
                ppmatmul(gm_w_matrix + w_bias, gmStepIn + addrBias, gm_workspace, iBatchSize, M, K, N, layout,
                         lessTCopy[iter_i], syncTilingNum[iter_i]);
            } else {
                matmulBatchsize *= tiling_para[4 + iter_i - 2];  // N1 * .. Ni-2
                matmulLoop = tiling_para[4 + iter_i - 1];        // Ni-1
                int32_t radix = tiling_para[4 + iter_i];
                M = radix * 2;
                K = M;
                N = inputN / (matmulBatchsize * matmulLoop * radix);  // inputN / (N1 * N2 * N3)
                w_bias += 4 * matmulLoop * matmulLoop;                // radix -1

                WaitFlagDev(VC_SYNC_FLAG_ID);

                ppmatmul_stride(gm_w_matrix + w_bias, gmStepIn + addrBias, gm_workspace, inputN, iBatch,
                                matmulBatchsize, matmulLoop, M, K, N, lessTCopy[iter_i], syncTilingNum[iter_i]);
            }
        }
    }
}

#elif __DAV_C220_VEC__
{
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);
    uint64_t config;

    auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    int32_t inputN = tiling_para[0];
    int32_t batchSize = tiling_para[1];
    int32_t repeatBatchSize = tiling_para[2];
    if (repeatBatchSize == 0) {
        repeatBatchSize = 1;
    }
    int32_t iterCount = tiling_para[3];

    __gm__ float *__restrict__ gmStepIn;
    __gm__ float *__restrict__ gmStepOut;

    int32_t N1 = tiling_para[4];
    if (N1 == 0) {
        N1 = 1;
    }
    int32_t N2 = tiling_para[5];
    if (N2 == 0) {
        N2 = 1;
    }
    int32_t N3 = tiling_para[6];
    int32_t N4 = tiling_para[7];
    int32_t N5 = tiling_para[8];
    int32_t fftDirection = tiling_para[29];

    int32_t isOutputHbm[5];
    int32_t lessTCopy[5];
    int32_t syncTilingNum[5];

    for (int32_t i = 0; i < iterCount; i++) {
        isOutputHbm[i] = tiling_para[i + 14];
        lessTCopy[i] = tiling_para[i + 19];
        syncTilingNum[i] = tiling_para[i + 24];
    }
    int32_t M = 0;
    int32_t N = 0;

    int64_t nRepeats = (static_cast<int64_t>(batchSize) + repeatBatchSize - 1) / repeatBatchSize;

    for (int64_t iRepeat = 0; iRepeat < nRepeats; iRepeat++) {
        int64_t addrBias = iRepeat * static_cast<int64_t>(repeatBatchSize) * inputN * 2;
        int32_t iBatch = iRepeat == (nRepeats - 1)
            ? static_cast<int32_t>(batchSize - repeatBatchSize * iRepeat) : repeatBatchSize;

        int32_t iBatchSize = 0;
        int64_t t_bias = 0;
        int32_t matmulBatchsize = 1;
        int32_t matmulLoop = 0;

        for (int32_t iter_i = 0; iter_i < iterCount; iter_i++) {
            gmStepOut = isOutputHbm[iter_i] ? gm_output : gm_input;

            if (iter_i == test_step + 1) {
                return;
            }

            if (iter_i == iterCount - 1) {
                int32_t pre_radix = tiling_para[4 + iter_i - 1];
                if (pre_radix == 0) {
                    pre_radix = 1;
                }
                int32_t radix = tiling_para[4 + iter_i];
                if (radix == 0) {
                    radix = 1;
                }

                matmulBatchsize = pre_radix;  // N1 * .. Ni-2

                M = radix * 2;
                N = inputN / radix / pre_radix;

                FftsCrossCoreSync<PIPE_MTE3, CV_SYNC_MODE>(VC_SYNC_FLAG_ID);

                combine_RI(gm_workspace, gmStepOut + addrBias, gm_index, inputN, iBatch, matmulBatchsize, M, N,
                           syncTilingNum[iter_i]);
            } else if (iter_i < 2) {
                if (iter_i == 0) {
                    iBatchSize = iBatch;
                    M = N1 * 2;
                    N = inputN / N1;
                    t_bias = 0;

                    if (N1 != 2) {
                        if (N < 65536) {
                            SeparateRI(gm_input + addrBias, gm_output + addrBias, inputN, iBatch);
                        } else {
                            SeparateRIzZLayOut(gm_input + addrBias, gm_output + addrBias, inputN, M, N, iBatch);
                        }
                        FftsCrossCoreSync<PIPE_MTE3, INTER_CORE_SYNC_MODE>(INTER_CORE_SYNC_FLAG_ID);
                        WaitFlagDev(INTER_CORE_SYNC_FLAG_ID);
                    } else {
                        SeparateRIwithCalculation(gm_input + addrBias, gm_t_matrix + inputN, gm_output + addrBias,
                                                  fftDirection, inputN, 4, inputN / 2, iBatch);

                        config = 1 | (INTER_CORE_SYNC_MODE << 4) | (INTER_CORE_SYNC_FLAG_ID << 8);
                        FftsCrossCoreSync<PIPE_MTE3, INTER_CORE_SYNC_MODE>(INTER_CORE_SYNC_FLAG_ID);
                        WaitFlagDev(INTER_CORE_SYNC_FLAG_ID);

                        // finish iter 1
                        continue;
                    }
                }
                if (iter_i == 1) {
                    iBatchSize = iBatch * N1;
                    M = N2 * 2;
                    N = inputN / (N1 * N2);
                    t_bias += (int64_t)inputN * 2;

                    if (N >= 65536) {
                        ConvertLayoutRowMajorTozZ(gm_input + addrBias, gm_output + addrBias, iBatchSize, M, N);
                        FftsCrossCoreSync<PIPE_MTE3, INTER_CORE_SYNC_MODE>(INTER_CORE_SYNC_FLAG_ID);
                        WaitFlagDev(INTER_CORE_SYNC_FLAG_ID);
                    }
                }

                FftsCrossCoreSync<PIPE_MTE3, CV_SYNC_MODE>(VC_SYNC_FLAG_ID);

                complex_mul(gm_workspace, gm_t_matrix + t_bias, gmStepOut + addrBias, fftDirection, iBatchSize, M, N,
                            lessTCopy[iter_i], syncTilingNum[iter_i]);

                FftsCrossCoreSync<PIPE_MTE3, INTER_CORE_SYNC_MODE>(INTER_CORE_SYNC_FLAG_ID);
                WaitFlagDev(INTER_CORE_SYNC_FLAG_ID);
            } else {
                FftsCrossCoreSync<PIPE_MTE3, CV_SYNC_MODE>(VC_SYNC_FLAG_ID);

                matmulBatchsize *= tiling_para[4 + iter_i - 2];
                if (matmulBatchsize == 0) {
                    matmulBatchsize = 1;
                }  // N1 * .. Ni-2
                matmulLoop = tiling_para[4 + iter_i - 1];
                if (matmulLoop == 0) {
                    matmulLoop = 1;
                }  // Ni-1
                int32_t radix = tiling_para[4 + iter_i];
                if (radix == 0) {
                    radix = 1;
                }
                M = radix * 2;
                N = inputN / (matmulBatchsize * matmulLoop * radix);  // inputN / (N1 * N2 * N3)
                t_bias += (int64_t)inputN * 2 / matmulBatchsize;     // t_bias / N1 / N2

                complex_mul_stride(gm_workspace, gm_t_matrix + t_bias, gmStepOut + addrBias, fftDirection, inputN,
                                   iBatch, matmulBatchsize, matmulLoop, M, N, lessTCopy[iter_i], syncTilingNum[iter_i]);

                FftsCrossCoreSync<PIPE_MTE3, INTER_CORE_SYNC_MODE>(INTER_CORE_SYNC_FLAG_ID);
                WaitFlagDev(INTER_CORE_SYNC_FLAG_ID);
            }
        }
    }
}
#endif
}

} // namespace

#endif
