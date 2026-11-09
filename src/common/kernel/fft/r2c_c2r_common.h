/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPSFFT_COMMON_KERNEL_FFT_R2C_C2R_COMMON_H
#define OPSFFT_COMMON_KERNEL_FFT_R2C_C2R_COMMON_H

#include "kernel_operator.h"

#include "mem.h"

__aicore__ __inline__ void C2R_odd(
    __gm__ float * __restrict__ gm_input,
    __gm__ uint32_t * __restrict__ gm_input_index,
    __gm__ float * __restrict__ gm_a,
    __gm__ float * __restrict__ gm_b,
    __gm__ uint32_t * __restrict__ gm_output_index,
    __gm__ float * __restrict__ gm_output,
    uint64_t N,
    uint64_t batch_size,
    uint64_t batch_id_begin,
    uint64_t batch_id_end)
    {
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

        AscendC::GlobalTensor<float> input_gm_tensor;
        AscendC::GlobalTensor<float> output_gm_tensor;
        AscendC::GlobalTensor<uint32_t> input_index_gm_tensor;
        AscendC::GlobalTensor<uint32_t> output_index_gm_tensor;
        AscendC::GlobalTensor<float> a_gm_tensor;
        AscendC::GlobalTensor<float> b_gm_tensor;

        input_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_input));
        output_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_output));
        input_index_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gm_input_index));
        output_index_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gm_output_index));
        a_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_a));
        b_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_b));

        uint64_t vecMask[2] = { UINT64_MAX, 0 };

        uint32_t aiv_id = AscendC::GetBlockIdx();
        uint32_t aiv_num = AscendC::GetBlockNum() * 2;

        // 每次取N个数, 执行batch_size次
        int32_t real_len = (N / 2 + 1) * 2;     // 结果为N + 1，实际输入长度
        int32_t copy_len = (N / 2) * 2;         // 结果为N - 1, 输入长度为((N / 2 + 1) * 2)，但第一个复数不需要共轭倒序

        // 每次把实数和虚部一起取出来
        uint64_t loopN_times = batch_size;
        // 每次取得数的个数
        uint64_t repeat_times = (copy_len + BIASC - 1) / BIASC;

        uint64_t ping = 1;

        // 开多核
        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        AscendC::LocalTensor<float> UB_INPUT1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R1_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * 8192);

        AscendC::LocalTensor<float> UB_INPUT2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(8 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R2_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(12 * 8192);

        AscendC::LocalTensor<uint32_t> reverse_index1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(16 * 8192);
        AscendC::LocalTensor<uint32_t> reverse_index2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(20 * 8192);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        AscendC::DataCopy(reverse_index1_ub_tensor, input_index_gm_tensor, AscendC::DataCopyParams(1, 4096 / 8, 0, 0));    // 搬入512长度
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::DataCopy(reverse_index2_ub_tensor, input_index_gm_tensor, AscendC::DataCopyParams(1, 4096 / 8, 0, 0));    // 搬入512长度
        AscendC::PipeBarrier<PIPE_MTE2>();

#ifndef MIX_CORE
#else
        int32_t remain_size = copy_len % BIASC;
#endif

        for(uint64_t loop = 0; loop < repeat_times; loop++) {       // 序列方向的循环

            int64_t offset_loopc = loop * BIASC;      // 已经处理过的长度
            int64_t actual_n = BIASC;

            int32_t ord_len = actual_n;                 // 读入的长度
            int32_t rev_len = actual_n;                 // 倒序后的长度

            if (repeat_times == 1) {
                ord_len = copy_len + 2;
                rev_len = copy_len;
            }
            else {
                if (loop == 0){
                    ord_len = BIASC + 2;
                    rev_len = BIASC;
                }
                else if (loop == repeat_times - 1) {
                    ord_len = copy_len - loop * BIASC;
                    rev_len = copy_len - loop * BIASC;
                }
                else {
                    ord_len = BIASC;
                    rev_len = BIASC;
                }
            }
            int32_t idx_size = (ord_len + 63) / 64 * 64;    // 输入搬入长度
            int32_t cal_size = (rev_len + 63) / 64 * 64;    // 共轭对应长度

            for(uint64_t loopN = batch_id_begin; loopN < batch_id_end; loopN+=1) {  // Batch方向的循环

                int32_t B_id = loopN - batch_id_begin;
                int32_t N_id = loop;
                int32_t loop_id = N_id * batch_size + B_id;
                if (loop_id % aiv_num != aiv_id) {
                    continue;
                }

                auto event_id = ping ? EVENT_ID0 : EVENT_ID1;
                auto buf0_ub_tensor = ping ? UB_INPUT1_ub_tensor : UB_INPUT2_ub_tensor;   // 复数输入
                auto buf4_ub_tensor = ping ? UB_INPUT_R1_REVERSE_ub_tensor : UB_INPUT_R2_REVERSE_ub_tensor;   // 复数输入
                auto reverse_index_ub_tensor = ping ? reverse_index1_ub_tensor : reverse_index2_ub_tensor;   // 复数输入
                uint64_t index1 = ping ? 0 : 8;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

                int64_t offset = loopN * real_len;
                int64_t offset_out = loopN * 2 * N;

                //  **********搬入***************
                if (repeat_times == 1) {
                    // (actual_n + 2) 代表再加上第一复数
                    AscendC::DataCopyPad(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc], AscendC::DataCopyExtParams(1, ord_len * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE2>();
                }
                else {
                    if (loop == 0) {
                        // (actual_n + 2) 代表再加上第一复数
                        AscendC::DataCopyPad(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc], AscendC::DataCopyExtParams(1, ord_len * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE2>();
                    }
                    else if (repeat_times != 1 && loop == repeat_times - 1) {
                        // + 2 means 第一个复数
                        AscendC::DataCopyPad(buf0_ub_tensor, input_gm_tensor[2 + offset + offset_loopc], AscendC::DataCopyExtParams(1, ord_len * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE2>();
                    }
                    else {
                        // + 2 means 第一个复数
                        AscendC::DataCopyPad(buf0_ub_tensor, input_gm_tensor[2 + offset + offset_loopc], AscendC::DataCopyExtParams(1, ord_len * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE2>();
                    }
                }
                //  **********搬入倒序index数组***************
                if (repeat_times == 1) {
                    AscendC::DataCopy(reverse_index_ub_tensor, input_index_gm_tensor, AscendC::DataCopyParams(1, cal_size / 8, 0, 0));       // 搬入512长度
                    AscendC::PipeBarrier<PIPE_MTE2>();
                }
                else {
                    if (loop == 0) {
                        AscendC::DataCopy(reverse_index_ub_tensor, input_index_gm_tensor, AscendC::DataCopyParams(1, cal_size / 8, 0, 0));       // 搬入512长度
                        AscendC::PipeBarrier<PIPE_MTE2>();
                    }

#ifndef MIX_CORE
                    else if (loop == repeat_times - 1) {
#else
                    else if (loop == repeat_times - 1 && remain_size != 0) {
#endif
                        AscendC::DataCopy(reverse_index_ub_tensor, input_index_gm_tensor[2 * BIASC], AscendC::DataCopyParams(1, cal_size / 8, 0, 0));       // 搬入512长度
                        AscendC::PipeBarrier<PIPE_MTE2>();
                    }
                    else {
                        AscendC::DataCopy(reverse_index_ub_tensor, input_index_gm_tensor[BIASC], AscendC::DataCopyParams(1, cal_size / 8, 0, 0));       // 搬入512长度
                        AscendC::PipeBarrier<PIPE_MTE2>();
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                //  **********[2:N]长度的倒序+共轭***************
                // FI倒序
                // todo GatherMask
                vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf4_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ unsigned int *>(reverse_index_ub_tensor.GetPhyAddr(0)), index1 * UB_VEC_SIZE, 8, cal_size / 64);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<float>(-1, 0xaaaaaaaaaaaaaaaa);
                AscendC::Muls<float, false>(buf4_ub_tensor, buf4_ub_tensor, -1.0f, AscendC::MASK_PLACEHOLDER, cal_size / 64, {1, 1, 8, 8});
                AscendC::SetVectorMask<float>(-1, -1);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
                //  **********搬出***************
                if (repeat_times == 1) {
                    // (actual_n + 2) 代表再加上第一复数
                    int start_idx = 0;
                    int revert_idx = N * 2 - rev_len;
                    AscendC::DataCopyPad(output_gm_tensor[offset_out + start_idx], buf0_ub_tensor, AscendC::DataCopyExtParams(1,  ord_len * sizeof(float), 0, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE3>();
                    AscendC::DataCopyPad(output_gm_tensor[offset_out + revert_idx], buf4_ub_tensor, AscendC::DataCopyExtParams(1,  rev_len * sizeof(float), 0, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE3>();
                }
                else {
                    if (loop == 0) {
                        // (actual_n + 2) 代表再加上第一复数
                        int start_idx = 0;
                        int revert_idx = N * 2 - rev_len;
                        AscendC::DataCopyPad(output_gm_tensor[offset_out + start_idx], buf0_ub_tensor, AscendC::DataCopyExtParams(1,  ord_len * sizeof(float), 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE3>();
                        AscendC::DataCopyPad(output_gm_tensor[offset_out + revert_idx], buf4_ub_tensor, AscendC::DataCopyExtParams(1,  rev_len * sizeof(float), 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE3>();
                    }
                    else if (repeat_times != 1 && loop == repeat_times - 1) {
                        // + 2 means 第一个复数
                        int start_idx = 2 + offset_loopc;
                        int revert_idx = N * 2 - offset_loopc - rev_len;
                        AscendC::DataCopyPad(output_gm_tensor[offset_out + start_idx], buf0_ub_tensor, AscendC::DataCopyExtParams(1,  ord_len * sizeof(float), 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE3>();
                        AscendC::DataCopyPad(output_gm_tensor[offset_out + revert_idx], buf4_ub_tensor, AscendC::DataCopyExtParams(1,  rev_len * sizeof(float), 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE3>();
                    }
                    else {
                        // + 2 means 第一个复数
                        int start_idx = 2 + offset_loopc;
                        int revert_idx = N * 2 - offset_loopc - rev_len;
                        AscendC::DataCopyPad(output_gm_tensor[offset_out + start_idx], buf0_ub_tensor, AscendC::DataCopyExtParams(1,  ord_len * sizeof(float), 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE3>();
                        AscendC::DataCopyPad(output_gm_tensor[offset_out + revert_idx], buf4_ub_tensor, AscendC::DataCopyExtParams(1,  rev_len * sizeof(float), 0, 0, 0));
                        AscendC::PipeBarrier<PIPE_MTE3>();
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                ping = 1 - ping;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

#endif
