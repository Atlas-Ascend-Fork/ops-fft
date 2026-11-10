/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COMPLEX_COMBINATION_RI_H_
#define COMPLEX_COMBINATION_RI_H_

#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "mem.h"
#include "utils.h"

__aicore__ __inline__ __attribute__((overloadable, always_inline)) void init_ub(AscendC::LocalTensor<uint16_t>& buf1_input_addr_buf,
                                                                                AscendC::LocalTensor<uint16_t>& buf1_output_addr_buf,
                                                                                AscendC::LocalTensor<uint16_t>& buf2_input_addr_buf,
                                                                                AscendC::LocalTensor<uint16_t>& buf2_output_addr_buf,
                                                                                AscendC::LocalTensor<float>& buf1_input_real,
                                                                                AscendC::LocalTensor<float>& buf1_input_imag,
                                                                                AscendC::LocalTensor<float>& buf2_input_real,
                                                                                AscendC::LocalTensor<float>& buf2_input_imag,
                                                                                AscendC::LocalTensor<float>& buf1_output,
                                                                                AscendC::LocalTensor<float>& buf2_output)
{
    const int32_t BUF_SIZE = 23 * 1024;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    buf1_input_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>(0);
    buf1_output_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>(64);
    buf2_input_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>(2 * 64);
    buf2_output_addr_buf = buf.GetBuffer<BufferType::ASCEND_UB, uint16_t>(3 * 64);
    buf1_input_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * 64);
    buf1_input_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * 64 + BUF_SIZE);
    buf2_input_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * 64 + BUF_SIZE * 2);
    buf2_input_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * 64 + BUF_SIZE * 3);
    buf1_output = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * 64 + BUF_SIZE * 4);
    buf2_output = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * 64 + BUF_SIZE * 6);

    for (int32_t i = 0; i < 16; ++i) {
        buf1_input_addr_buf.SetValue(i, (uint16_t)(reinterpret_cast<uintptr_t>(buf1_input_real.GetPhyAddr()) / 4 / 8 + i));
    }

    SET_FLAG(S, V, EVENT_ID0);
    WAIT_FLAG(S, V, EVENT_ID0);

    AscendC::DataCopy(buf1_input_addr_buf[16], buf1_input_addr_buf, AscendC::DataCopyParams(1, 16 / C0_SIZE, 0, 0));

    AscendC::SetMaskCount();
    AscendC::SetVectorMask<float>(0, 32);
    PIPE_BARRIER(V);

    adds_v<ArchType::ASCEND_V220, int16_t>(
        buf1_output_addr_buf.template ReinterpretCast<int16_t>(),
        buf1_input_addr_buf.template ReinterpretCast<int16_t>(),
        (reinterpret_cast<uintptr_t>(buf1_output.GetPhyAddr()) - reinterpret_cast<uintptr_t>(buf1_input_real.GetPhyAddr())) / 4 / 8,
        1, 1, 1, 8, 8);

    AscendC::SetMaskCount();
    AscendC::SetVectorMask<float>(0, 32);

    PIPE_BARRIER(V);

    adds_v<ArchType::ASCEND_V220, int16_t>(
        buf2_input_addr_buf.template ReinterpretCast<int16_t>(),
        buf1_input_addr_buf.template ReinterpretCast<int16_t>(),
        (reinterpret_cast<uintptr_t>(buf2_input_real.GetPhyAddr()) - reinterpret_cast<uintptr_t>(buf1_input_real.GetPhyAddr())) / 4 / 8,
        1, 1, 1, 8, 8);
    adds_v<ArchType::ASCEND_V220, int16_t>(
        buf2_output_addr_buf.template ReinterpretCast<int16_t>(),
        buf1_output_addr_buf.template ReinterpretCast<int16_t>(),
        (reinterpret_cast<uintptr_t>(buf2_output.GetPhyAddr()) - reinterpret_cast<uintptr_t>(buf1_output.GetPhyAddr())) / 4 / 8,
        1, 1, 1, 8, 8);

    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    SET_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID0);
}

__aicore__ __inline__ __attribute__((overloadable, always_inline)) void init_loop(
    int64_t loop, int64_t N1_loop, int64_t group_num, int64_t group_id,
    int64_t &loop_per_group, int64_t &loop_per_group_remain, int64_t &loop_per_group_actual)
{
    if (N1_loop == 0 || group_num == 0) {
        loop_per_group = 0;
        loop_per_group_remain = 0;
        loop_per_group_actual = 0;
        return;
    }
    loop_per_group = (loop / N1_loop) / group_num;
    loop_per_group_remain = (loop / N1_loop) % group_num;
    loop_per_group_actual = loop_per_group;
    if (group_id < loop_per_group_remain) {
        loop_per_group_actual++;
    }
    loop_per_group_actual *= N1_loop;
}

/*
*  @brief 原地转置函数
*/
__aicore__ __inline__ __attribute__((overloadable, always_inline)) void transpose(__ubuf__ uint16_t *output_buf,
                                                                                  __ubuf__ uint16_t *input_buf,
                                                                                  uint8_t repeat_num)
{
    vld_va_reg(VA0, (__ubuf__ uint64_t *)input_buf, L128);
    vld_va_reg(VA1, (__ubuf__ uint64_t *)input_buf, H128);
    vld_va_reg(VA2, (__ubuf__ uint64_t *)(output_buf + 16), L128);
    vld_va_reg(VA3, (__ubuf__ uint64_t *)(output_buf + 16), H128);
    PIPE_BARRIER(V);
    scatter_vnchwconv_b16(VA2, VA0, repeat_num, 16, 16);
}

/*
*  @brief 虚实结合函数 仅仅支持单精度
*  @param data指向数据起始地址 虚部和实部连续存放
*  @param data_start为数据起始块号 将UB空间划分为32B的数据块 从零开始编号
*  @param wksp指向辅助空间起始地址
*  @param len为实部（或者虚部）数据个数 必须为128的倍数
*  @param addr_buf指向一片大小为64B的空间 用作scatter_vnchwconv_b16接口的辅助空间
*  @return none
*/
__aicore__ __inline__ __attribute__((overloadable, always_inline)) void gather(AscendC::LocalTensor<float>& input,
                                                                               AscendC::LocalTensor<float>& output,
                                                                               int32_t len,
                                                                               __ubuf__ uint16_t *addr_input_buf,
                                                                               __ubuf__ uint16_t *addr_output_buf)
{
    // 以16x16的块进行划分
    uint8_t repeat_num = (uint8_t)(len * 2 / 256);

    transpose(addr_output_buf, addr_input_buf, repeat_num * 2);
    PIPE_BARRIER(V);

    AscendC::DataCopy(input, output,
        AscendC::DataCopyParams(
            len * 4 / 32 / 2,
            2,
            0,
            2
        )
    );
    AscendC::DataCopy(input[16], output[len],
        AscendC::DataCopyParams(
            len * 4 / 32 / 2,
            2,
            0,
            2
        )
    );
    PIPE_BARRIER(V);

    transpose(addr_output_buf, addr_input_buf, repeat_num * 2);
    PIPE_BARRIER(V);

    adds_v<ArchType::ASCEND_V220, float>(input,       // dst,
                                         output,      // src0,
                                         0.0f,        // src1,
                                         repeat_num,  // repeat,
                                         2,           // dstBlockStride,
                                         1,           // srcBlockStride,
                                         16 * 2,      // dstRepeatStride,
                                         16 * 2);     // srcRepeatStride

    adds_v<ArchType::ASCEND_V220, float>(input[64 * 2],   // dst,
                                         output[64],      // src0,
                                         0.0f,            // src1,
                                         repeat_num,      // repeat,
                                         2,               // dstBlockStride,
                                         1,               // srcBlockStride,
                                         16 * 2,          // dstRepeatStride,
                                         16 * 2);         //  uint8_t srcRepeatStride

    adds_v<ArchType::ASCEND_V220, float>(input[8],      // dst,
                                         output[128],   // src0,
                                         0.0f,          // src1,
                                         repeat_num,    // repeat,
                                         2,             // dstBlockStride,
                                         1,             // srcBlockStride,
                                         16 * 2,        // dstRepeatStride,
                                         16 * 2);       //  uint8_t srcRepeatStride

    adds_v<ArchType::ASCEND_V220, float>(input[8 + 64 * 2],   // dst,
                                         output[128 + 64],    // src0,
                                         0.0f,                // src1,
                                         repeat_num,          // repeat,
                                         2,                   // dstBlockStride,
                                         1,                   // srcBlockStride,
                                         16 * 2,              // dstRepeatStride,
                                         16 * 2);             //  uint8_t srcRepeatStride
}

template <int32_t aiv_split_way>
__aicore__ __inline__ __attribute__((overloadable, always_inline)) void common_combination_RI(
    __gm__ float *__restrict__ gm_input,
    __gm__ float *__restrict__ gm_output,
    __gm__ float * __restrict__ gm_output_real,
    __gm__ float * __restrict__ gm_output_imag,
    __gm__ float *__restrict__ workspace,
    __gm__ float *__restrict__ gm_auxil,
    int64_t batch_size,
    int64_t N0,
    int32_t N1,
    int64_t N2_padding,
    int64_t N2,
    int32_t tile_M0,
    int32_t tile_N0,
    int32_t tile_K0,
    int32_t step_len,
    int32_t type
) {
    // 除0整改
    if (tile_N0 == 0) {
        tile_N0 = 1;
    }
    if (tile_K0 == 0) {
        tile_K0 = 1;
    }
    if (tile_M0 == 0) {
        tile_M0 = 1;
    }

    int32_t step_index = step_len - 1;  // 虚实结合默认位于最后一次迭代

    AscendC::LocalTensor<uint16_t> buf1_input_addr_buf, buf1_output_addr_buf, buf2_input_addr_buf, buf2_output_addr_buf;
    AscendC::LocalTensor<float> buf1_input_real, buf1_input_imag, buf2_input_real, buf2_input_imag;
    AscendC::LocalTensor<float> buf1_output, buf2_output;
    AscendC::GlobalTensor<float> workspace_tensor, gm_output_tensor, gm_output_real_tensor, gm_output_imag_tensor;
    workspace_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    gm_output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));
    gm_output_real_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output_real));
    gm_output_imag_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output_imag));
    init_ub(buf1_input_addr_buf, buf1_output_addr_buf, buf2_input_addr_buf, buf2_output_addr_buf,
            buf1_input_real, buf1_input_imag, buf2_input_real, buf2_input_imag, buf1_output, buf2_output);

    int32_t batch_len;
    batch_len = L0AB_PINGPONG_BUFFER_LEN * 2 / (tile_N0 * tile_K0);
    batch_len += (batch_len == 0);

    int64_t N1_loop = (2 * N1 + static_cast<int64_t>(tile_M0) - 1) / static_cast<int64_t>(tile_M0);
    if (N1_loop == 0) {
        N1_loop = 1;
    }
    int64_t N2_loop = (N2 + static_cast<int64_t>(tile_N0) - 1) / static_cast<int64_t>(tile_N0);
    int64_t batch_loop = (batch_size * N0 + static_cast<int64_t>(batch_len) - 1) / static_cast<int64_t>(batch_len);
    if (batch_loop == 0) {
        batch_loop = 1;
    }
    int64_t loop = batch_loop * N1_loop * N2_loop;
    int64_t batch_remain = (batch_size * N0) % static_cast<int64_t>(batch_len);
    int64_t N1_remain = (2 * N1) % static_cast<int64_t>(tile_M0);
    int64_t N2_remain = N2 % static_cast<int64_t>(tile_N0);
    int64_t aiv_id_per_group = AscendC::GetSubBlockIdx();
    int64_t group_num = AscendC::GetBlockNum();
    if (group_num == 0) {
        group_num = 1;
    }
    int64_t group_id = get_block_idx();

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    bool flag = 0;

    FftsCrossCoreSync<PIPE_MTE2, 2>(2);
    FftsCrossCoreSync<PIPE_MTE2, 2>(3);

    int64_t loop_per_group, loop_per_group_remain, loop_per_group_actual;
    init_loop(loop, N1_loop, group_num, group_id, loop_per_group, loop_per_group_remain, loop_per_group_actual);

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
        } else {
            loop_idx = group_id * loop_per_group * N1_loop +
                       (loop_per_group_remain > 0) *
                           (group_id > loop_per_group_remain ? loop_per_group_remain : group_id) * N1_loop +
                       i;
        }
        if (step_len > 3 && step_index == step_len - 3) {
            batch_len_idx = loop_idx % batch_loop;
            N1_N2_idx = loop_idx / batch_loop;
            N2_idx = N1_N2_idx % N2_loop;
            out_N1_idx = N1_N2_idx / N2_loop;
        } else {
            out_N1_idx = loop_idx % N1_loop;
            batch_N2_idx = loop_idx / N1_loop;
            batch_len_idx = batch_N2_idx % batch_loop;
            N2_idx = batch_N2_idx / batch_loop;
        }

        int32_t N1_actual = tile_M0;
        if (out_N1_idx == N1_loop - 1 && N1_remain > 0) {
            N1_actual = N1_remain;
        }

        int32_t N2_actual = tile_N0;
        if (N2_idx == N2_loop - 1 && N2_remain > 0) {
            N2_actual = N2_remain;
        }

        int32_t batch_actual = batch_len;
        if (batch_len_idx == batch_loop - 1 && batch_remain > 0) {
            batch_actual = batch_remain;
        }

        auto buf_input_real = flag ? buf1_input_real : buf2_input_real;
        auto buf_input_imag = flag ? buf1_input_imag : buf2_input_imag;
        auto buf_output = flag ? buf1_output : buf2_output;
        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;
        auto buf_input_addr_buf = flag ? buf1_input_addr_buf : buf2_input_addr_buf;
        auto buf_output_addr_buf = flag ? buf1_output_addr_buf : buf2_output_addr_buf;

        int32_t N2_round = tile_N0;
        // 每个选项的选择可以查看main.cpp
        // 主要根据矩阵的大小，batch数量来采取不同策略
        // aiv_split_way == 1
        // 在batch维度划分，输入矩阵的padding已经被去掉，可以看作向量来处理。
        // aiv_split_way == 2
        // 矩阵较大，输入矩阵的padding已经被去掉，可以看作向量来处理。将向量分成两半由不同AIV处理。
        // 矩阵较大，需要按照矩阵来划分成两半。
        if (aiv_split_way == 1) {
            int32_t tile_N1_idx = 0;
            int32_t N1_idx = out_N1_idx;
            N1_idx *= tile_M0;
            N1_idx += tile_N1_idx;

            int32_t N1_per_aiv = N1_actual;
            int32_t half_M0 = N1_per_aiv / 2;

            int32_t now_batch_idx = (aiv_id_per_group == 0 ? 0 : (batch_actual / 2));
            int32_t now_batch_actual =
                (aiv_id_per_group == 0 ? (batch_actual / 2) : (batch_actual - (batch_actual / 2)));

            if (type == 2) {
                PIPE_BARRIER(ALL);
            }

            WAIT_FLAG(MTE3, MTE2, event_id);
            WaitFlagDev(flag + 4);

            int64_t data_in_index =
                (group_id * 2 + static_cast<int64_t>(flag)) * static_cast<int64_t>(batch_len) * static_cast<int64_t>(tile_M0) * static_cast<int64_t>(tile_N0) + static_cast<int64_t>(now_batch_idx) * static_cast<int64_t>(tile_M0) * static_cast<int64_t>(tile_N0);
            int64_t N1_N2_padding = ROUND(half_M0 * N2_actual, 8);
            if (now_batch_actual > 0) {
                copy_gm2ubuf(buf_input_real, workspace_tensor[data_in_index], now_batch_actual, N1_N2_padding,
                             tile_M0 * tile_N0, N1_N2_padding);
                if (type != 2) {
                    copy_gm2ubuf(buf_input_real[ROUND(now_batch_actual * N1_N2_padding, 256)],
                                workspace_tensor[data_in_index + half_M0 * N2_actual], now_batch_actual, N1_N2_padding,
                                tile_M0 * tile_N0, N1_N2_padding);
                }
            }
            FftsCrossCoreSync<PIPE_MTE2, 2>(flag + 2);

            if (now_batch_actual > 0) {
                SET_FLAG(MTE2, V, event_id);
                WAIT_FLAG(MTE2, V, event_id);
                if (type == 0) {
                    gather(buf_input_real, buf_output, ROUND(now_batch_actual * N1_N2_padding, 256),
                            reinterpret_cast<__ubuf__ uint16_t *>(buf_input_addr_buf.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint16_t *>(buf_output_addr_buf.GetPhyAddr()));
                }
                SET_FLAG(V, MTE3, event_id);
                WAIT_FLAG(V, MTE3, event_id);

                if (type == 0) {
                    const int64_t data_output_index = (batch_len_idx * static_cast<int64_t>(batch_len) + static_cast<int64_t>(now_batch_idx)) * N1 * N2 +
                                                    (static_cast<int64_t>(out_N1_idx) * static_cast<int64_t>(tile_M0) / 2 + static_cast<int64_t>(tile_N1_idx)) * N2 + N2_idx * static_cast<int64_t>(tile_N0);
                    copy_ubuf2gm(gm_output_tensor[data_output_index * 2], buf_input_real, now_batch_actual, 2 * half_M0 * N2_actual,
                                2 * N1_N2_padding, 2 * N1 * N2_actual);
                } else if (type == 1) {
                    const int64_t data_output_index = (batch_len_idx * batch_len + now_batch_idx) * N1 * N2 + (out_N1_idx * tile_M0 / 2 + tile_N1_idx) * N2 + N2_idx * tile_N0;

                    copy_ubuf2gm(gm_output_real_tensor[data_output_index], buf_input_real, now_batch_actual, half_M0 * N2_actual, N1_N2_padding, N1 * N2_actual);
                    copy_ubuf2gm(gm_output_imag_tensor[data_output_index], buf_input_real[ROUND(now_batch_actual * N1_N2_padding, 256)], now_batch_actual, half_M0 * N2_actual, N1_N2_padding, N1 * N2_actual);
                } else {
                    const int64_t data_output_index = (batch_len_idx * static_cast<int64_t>(batch_len) + static_cast<int64_t>(now_batch_idx)) * N1 * N2 +
                                    (out_N1_idx * static_cast<int64_t>(tile_M0)) * N2 + N2_idx * static_cast<int64_t>(tile_N0);
                    copy_ubuf2gm(gm_output_tensor[data_output_index], buf_input_real, now_batch_actual, half_M0 * N2_actual, N1_N2_padding,
                                half_M0 * N2_actual);
                }
            }
            SET_FLAG(MTE3, MTE2, event_id);
            flag = 1 - flag;
        } else if (aiv_split_way == 2) {
            int32_t tile_N1_idx = (aiv_id_per_group == 0 ? 0 : N1_actual / 4);
            int32_t N1_idx = out_N1_idx;
            N1_idx *= tile_M0;
            N1_idx += tile_N1_idx;

            int32_t half_M0 = aiv_id_per_group == 0 ? N1_actual / 4 : N1_actual / 2 - N1_actual / 4;
            int32_t N1_per_aiv = half_M0 * 2;

            int32_t N2_round = tile_N0;

            WAIT_FLAG(MTE3, MTE2, event_id);

            WaitFlagDev(flag + 4);

            int64_t data_in_index = (group_id * 2 + static_cast<int64_t>(flag)) * static_cast<int64_t>(batch_len) * static_cast<int64_t>(tile_M0) * static_cast<int64_t>(tile_N0) + static_cast<int64_t>(tile_N1_idx) * static_cast<int64_t>(N2_actual);
            int64_t N1_N2_padding = ROUND(half_M0 * N2_actual, 8);

            copy_gm2ubuf(buf_input_real, workspace_tensor[data_in_index], batch_actual, half_M0 * N2_actual,
                         tile_M0 * tile_N0, N1_N2_padding);
            if (type != 2) {
                copy_gm2ubuf(buf_input_real[ROUND(batch_actual * N1_N2_padding, 256)],
                            workspace_tensor[data_in_index + (N1_actual / 2) * N2_actual],
                            batch_actual, half_M0 * N2_actual,
                            tile_M0 * tile_N0, N1_N2_padding);
            }

            FftsCrossCoreSync<PIPE_MTE2, 2>(flag + 2);

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);
            if (type == 0) {
                gather(buf_input_real, buf_output, ROUND(batch_actual * N1_N2_padding, 256),
                    reinterpret_cast<__ubuf__ uint16_t *>(buf_input_addr_buf.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t *>(buf_output_addr_buf.GetPhyAddr()));
            }
            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);
            if (type == 0) {
                const int64_t data_output_index =
                    (batch_len_idx * static_cast<int64_t>(batch_len)) * N1 * N2 + (out_N1_idx * static_cast<int64_t>(tile_M0) / 2 + tile_N1_idx) * N2;
                copy_ubuf2gm(gm_output_tensor[data_output_index * 2], buf_input_real, batch_actual, 2 * half_M0 * N2_actual, 2 * N1_N2_padding,
                            2 * N1 * N2);
            } else if (type == 1) {
                const int64_t data_output_index = (batch_len_idx * batch_len) * N1 * N2 + (out_N1_idx * tile_M0 / 2 + tile_N1_idx) * N2;

                copy_ubuf2gm(gm_output_real_tensor[data_output_index], buf_input_real, batch_actual, half_M0 * N2_actual, N1_N2_padding, N1 * N2);
                copy_ubuf2gm(gm_output_imag_tensor[data_output_index], buf_input_real[ROUND(batch_actual * N1_N2_padding, 256)], batch_actual, half_M0 * N2_actual, N1_N2_padding, N1 * N2);
            } else {
                const int64_t data_output_index =
                    (batch_len_idx * static_cast<int64_t>(batch_len)) * N1 * N2 + (out_N1_idx * static_cast<int64_t>(tile_M0) / 2 + static_cast<int64_t>(tile_N1_idx)) * N2;
                copy_ubuf2gm(gm_output_tensor[data_output_index], buf_input_real, batch_actual, half_M0 * N2_actual, N1_N2_padding, N1 * N2);
            }

            SET_FLAG(MTE3, MTE2, event_id);
            flag = 1 - flag;
        } else {
            int32_t tile_N1_idx = (aiv_id_per_group == 0 ? 0 : N1_actual / 4 * 2);
            int32_t N1_idx = out_N1_idx;
            N1_idx *= tile_M0;
            N1_idx += tile_N1_idx;

            int32_t N1_per_aiv = aiv_id_per_group == 0 ? N1_actual / 4 * 2 : N1_actual - N1_actual / 4 * 2;
            int32_t half_M0 = N1_per_aiv / 2;

            int32_t N2_round = tile_N0;

            WAIT_FLAG(MTE3, MTE2, event_id);

            int64_t data_in_index = (group_id * 2 + static_cast<int64_t>(flag)) * static_cast<int64_t>(batch_len) * static_cast<int64_t>(tile_M0) * static_cast<int64_t>(tile_N0) + tile_N1_idx * static_cast<int64_t>(tile_N0);

            WaitFlagDev(flag + 4);

            copy_gm2ubuf(buf_output, workspace_tensor[data_in_index], batch_actual,
                         N1_per_aiv * tile_N0, tile_M0 * tile_N0, N1_per_aiv * tile_N0);

            FftsCrossCoreSync<PIPE_MTE2, 2>(flag + 2);

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);

            AscendC::DataCopy(buf_input_real, buf_output,
                AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, (2 * tile_N0 - tile_N0) / C0_SIZE, 0));
            if (type != 2) {
                AscendC::DataCopy(buf_input_real[ROUND(batch_actual * half_M0 * tile_N0, 256)], buf_output[tile_N0],
                    AscendC::DataCopyParams(batch_actual * half_M0, tile_N0 / C0_SIZE, (2 * tile_N0 - tile_N0) / C0_SIZE, 0));
            }
            PIPE_BARRIER(V);

            if (type == 0) {
                gather(buf_input_real, buf_output, ROUND(batch_actual * half_M0 * tile_N0, 256),
                    reinterpret_cast<__ubuf__ uint16_t *>(buf_input_addr_buf.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t *>(buf_output_addr_buf.GetPhyAddr()));
            }
            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);

            if (type == 0) {
                for (int64_t j = 0; j < static_cast<int64_t>(batch_actual); j++) {
                    const int64_t data_output_index = (batch_len_idx * static_cast<int64_t>(batch_len) + j) * N1 * N2 +
                                                    (out_N1_idx * static_cast<int64_t>(tile_M0) / 2 + static_cast<int64_t>(tile_N1_idx) / 2) * N2 + N2_idx * static_cast<int64_t>(tile_N0);
                    copy_ubuf2gm(gm_output_tensor[data_output_index * 2], buf_input_real[j * 2 * half_M0 * tile_N0], N1_per_aiv / 2, 2 * N2_actual,
                                2 * tile_N0, 2 * N2);
                }
            } else if (type == 1) {
                for (int j = 0; j < batch_actual; j++) {
                    const int64_t data_output_index = (batch_len_idx * batch_len + j) * N1 * N2 + (out_N1_idx * tile_M0 / 2 + tile_N1_idx / 2) * N2 + N2_idx * tile_N0;

                    copy_ubuf2gm(gm_output_real_tensor[data_output_index], buf_input_real[j * half_M0 * tile_N0], N1_per_aiv / 2, N2_actual, tile_N0, N2);
                    copy_ubuf2gm(gm_output_imag_tensor[data_output_index], buf_input_real[ROUND(batch_actual * half_M0 * tile_N0, 256) + j * half_M0 * tile_N0], N1_per_aiv / 2, N2_actual, tile_N0, N2);
                }
            } else {
                for (int64_t j = 0; j < static_cast<int64_t>(batch_actual); j++) {
                    const int64_t data_output_index = (batch_len_idx * batch_len + j) * N1 * N2 +
                                                    (out_N1_idx * static_cast<int64_t>(tile_M0) / 2 + static_cast<int64_t>(tile_N1_idx) / 2) * N2 + N2_idx * static_cast<int64_t>(tile_N0);
                    copy_ubuf2gm(gm_output_tensor[data_output_index], buf_input_real[j * half_M0 * tile_N0], N1_per_aiv / 2, N2_actual, tile_N0,
                                N2);
                }
            }
            SET_FLAG(MTE3, MTE2, event_id);
            flag = 1 - flag;
        }
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

template <int32_t aiv_split_way>
__aicore__ __inline__ __attribute__((overloadable, always_inline)) void complex_combination_vtranspose_sync_RI(
    __gm__ float *__restrict__ gm_input, __gm__ float *__restrict__ gm_output, __gm__ float *__restrict__ workspace,
    __gm__ float *__restrict__ gm_auxil, int64_t batch_size, int64_t N0, int32_t N1, int64_t N2_padding, int64_t N2,
    int32_t tile_M0, int32_t tile_N0, int32_t tile_K0, int32_t step_len)
{
    common_combination_RI<aiv_split_way>(
        gm_input, gm_output, nullptr, nullptr, workspace, gm_auxil, batch_size, N0, N1, N2_padding, N2,
        tile_M0, tile_N0, tile_K0, step_len, 0);
}

template<int32_t aiv_split_way>
__aicore__ __inline__ __attribute__((overloadable, always_inline))  void r2c_even_complex_combination_vtranspose_sync_RI(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_output_real,
    __gm__ float * __restrict__ gm_output_imag,
    __gm__ float * __restrict__ workspace,
    __gm__ float * __restrict__ gm_auxil,
    int64_t batch_size,
    int64_t N0,
    int32_t N1,
    int64_t N2_padding,
    int64_t N2,
    int32_t tile_M0,
    int32_t tile_N0,
    int32_t tile_K0,
    int32_t step_len
) {
    common_combination_RI<aiv_split_way>(
        gm_input, nullptr, gm_output_real, gm_output_imag, workspace, gm_auxil, batch_size, N0, N1, N2_padding, N2,
        tile_M0, tile_N0, tile_K0, step_len, 1);
}

template<int32_t aiv_split_way>
__aicore__ __inline__ __attribute__((overloadable, always_inline))  void complex_combination_vtranspose_sync_RI_odd(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ workspace,
    __gm__ float * __restrict__ gm_auxil,
    int64_t batch_size,
    int64_t N0,
    int32_t N1,
    int64_t N2_padding,
    int64_t N2,
    int32_t tile_M0,
    int32_t tile_N0,
    int32_t tile_K0,
    int32_t step_len
) {
    common_combination_RI<aiv_split_way>(
        gm_input, gm_output, nullptr, nullptr, workspace, gm_auxil, batch_size, N0, N1, N2_padding, N2,
        tile_M0, tile_N0, tile_K0, step_len, 2);
}

#endif
