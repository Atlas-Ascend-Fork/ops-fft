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


__aicore__ __inline__ void R2C_tail_calc(
    __gm__ float * __restrict__ gm_c2c_output,
    __gm__ float * __restrict__ gm_output,
    uint64_t batch_size,
    uint64_t fft_n)
{
    AscendC::GlobalTensor<float> c2c_output_gm_tensor;
    AscendC::GlobalTensor<float> output_gm_tensor;
    c2c_output_gm_tensor.SetGlobalBuffer(gm_c2c_output);
    output_gm_tensor.SetGlobalBuffer(gm_output);

    uint32_t N = fft_n * 2;
    uint32_t aiv_id = AscendC::GetBlockIdx();
    uint32_t aiv_num = AscendC::GetBlockNum() * 2;

    if (fft_n <= 128) {
        for(int i = 0; i < batch_size; i++) {
            if (aiv_id != 0) {
                continue;
            }
            int64_t offset = i * N;
            int64_t offset_out = i * (N + 2);
            float temp_R0 = c2c_output_gm_tensor.GetValue(offset);                         // 910B的 LD ST 特性
            float temp_I0 = c2c_output_gm_tensor.GetValue(offset + 1);
            output_gm_tensor.SetValue(offset_out, temp_R0 + temp_I0);
            output_gm_tensor.SetValue(offset_out + 1, 0.0);
            output_gm_tensor.SetValue(offset_out + N, temp_R0 - temp_I0);
            output_gm_tensor.SetValue(offset_out + N + 1, 0.0);
        }
        return;
    }

    if (aiv_id == 0) {
        float temp_R0 = c2c_output_gm_tensor.GetValue(0); 
        float temp_I0 = c2c_output_gm_tensor.GetValue(1); 
        output_gm_tensor.SetValue(0, temp_R0 + temp_I0);
        output_gm_tensor.SetValue(0 + 1, 0.0);
        output_gm_tensor.SetValue(0 + N, temp_R0 - temp_I0);
        output_gm_tensor.SetValue(0 + N + 1, 0.0);
    }
    for(int i = 0; i < batch_size - 1; i++) {
        if (i % aiv_num != aiv_id) {
            continue;
        }
        int64_t offset = i * N;
        int64_t offset_out = i * (N + 2);

        float temp_R0 = c2c_output_gm_tensor.GetValue(offset);                         // 910B的 LD ST 特性
        float temp_I0 = c2c_output_gm_tensor.GetValue(offset + 1);
        float temp_R1 = c2c_output_gm_tensor.GetValue(offset + N);                         // 910B的 LD ST 特性
        float temp_I1 = c2c_output_gm_tensor.GetValue(offset + N + 1);

        output_gm_tensor.SetValue(offset_out + N, temp_R0 - temp_I0);
        output_gm_tensor.SetValue(offset_out + N + 1, 0.0);
        output_gm_tensor.SetValue(offset_out + N + 2, temp_R1 + temp_I1);     // 下个Batch的首尾 连续的两个虚数
        output_gm_tensor.SetValue(offset_out + N + 3, 0.0);

        if (i == batch_size - 2) {
            output_gm_tensor.SetValue(offset_out + N + 2 + N, temp_R1 - temp_I1);
            output_gm_tensor.SetValue(offset_out + N + 2 + N + 1, 0.0);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}


__aicore__ __inline__ void R2C(
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
        uint64_t num = N - 2 > BIASC ? BIASC : N - 2;
        int index_size = N >= BIASC ? BIAS : ((N / 2 + 63) / 64) * 64;
        uint64_t lenBurstC = (num + 7) / 8;
        uint64_t lenBurstIndex = (index_size + 7) / 8;
        uint64_t lenBurstOutIndex = (BIASC + 7) / 8;
        // 每次vector乘加需要的次数
        uint8_t repeat = (num / 2 + 63) / 64; //  此处会导致越界UB操作
        // 每次vector reduce需要的次数
        uint8_t repeat_reduce = (num + 63) / 64;
        // 每次vgather output需要的次数
        uint8_t repeat_out = (num + 63) / 64;
        // 每次把实数和虚部一起取出来
        uint64_t loopN_times = batch_size;
        // 每次取得数的个数
        uint64_t repeat_times = (N - 2 + BIASC - 1) / BIASC;

        uint64_t ping = 1;

        // 开多核
        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        AscendC::LocalTensor<float> UB_INPUT1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(2 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_I1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(3 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R1_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_I1_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(5 * 8192);
        AscendC::LocalTensor<float> tvub1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(6 * 8192);
        AscendC::LocalTensor<float> tvub2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(7 * 8192);

        AscendC::LocalTensor<float> UB_INPUT2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(8 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(10 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_I2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(11 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R2_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(12 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_I2_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(13 * 8192);
        AscendC::LocalTensor<float> tvub3_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(14 * 8192);
        AscendC::LocalTensor<float> tvub4_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(15 * 8192);

        AscendC::LocalTensor<float> MatrixA_Real_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(16 * 8192);
        AscendC::LocalTensor<float> MatrixA_Imag_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(17 * 8192);
        AscendC::LocalTensor<float> MatrixB_Real_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(18 * 8192);
        AscendC::LocalTensor<float> MatrixB_Imag_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(19 * 8192);

        AscendC::LocalTensor<uint32_t> reverse_index_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(20 * 8192);
        AscendC::LocalTensor<uint32_t> vgather_index_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(21 * 8192);


#ifndef MIX_CORE
        AscendC::DataCopy(reverse_index_ub_tensor, input_index_gm_tensor, AscendC::DataCopyParams(1, lenBurstIndex, 0, 0));  // 搬入512长度
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::DataCopy(vgather_index_ub_tensor, output_index_gm_tensor, AscendC::DataCopyParams(1, lenBurstOutIndex, 0, 0));  // 搬入4096长度，污染后一个
        AscendC::PipeBarrier<PIPE_MTE2>();
#else
        AscendC::DataCopy(vgather_index_ub_tensor, output_index_gm_tensor, AscendC::DataCopyParams(1, lenBurstOutIndex, 0, 0)); // 搬入4096个数
        AscendC::PipeBarrier<PIPE_MTE2>();
#endif

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

        for(uint64_t loop = 0; loop < repeat_times; loop++) {       // 序列方向的循环

            int64_t offset_loopc = loop * BIASC;      // 已经处理过的长度
            int actual_n = N - 2;

#ifndef MIX_CORE
            actual_n = actual_n < BIASC ? actual_n : ((loop + 1) * BIASC <= actual_n ? BIASC : (actual_n - loop * BIASC));
            bool aligned = (actual_n % BLOCK_SIZE_8 == 0);

            bool init_A = 0;
            bool init_B = 0;
#else
            int reverse_offset = 0;
            if (repeat_times != 1) {                        // 若N-2长度大于4096则循环
                actual_n = (loop + 1) * BIASC <= (N - 2) ? BIASC : ((N - 2) - loop * BIASC);
            }
            
            if (repeat_times != 1 && actual_n != BIASC) {   // 需要读到对应的reverse数组
                reverse_offset = index_size;
            }

            AscendC::DataCopy(reverse_index_ub_tensor, input_index_gm_tensor[reverse_offset], AscendC::DataCopyParams(1, lenBurstIndex, 0, 0));   // 搬入2048个数
            AscendC::PipeBarrier<PIPE_MTE2>();

            bool init_A = 0;
            bool init_B = 0;

            AscendC::DataCopyPad(MatrixA_Real_ub_tensor, a_gm_tensor[offset_loopc / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));      // 搬对应的N-2个数进来
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::DataCopyPad(MatrixA_Imag_ub_tensor, a_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));       // 搬对应的N-2个数进来
            AscendC::PipeBarrier<PIPE_MTE2>();

            AscendC::DataCopyPad(MatrixB_Real_ub_tensor, b_gm_tensor[offset_loopc / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::DataCopyPad(MatrixB_Imag_ub_tensor, b_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
            AscendC::PipeBarrier<PIPE_MTE2>();
#endif

            for(uint64_t loopN = batch_id_begin; loopN < batch_id_end; loopN+=1) {  // Batch方向的循环

#ifndef MIX_CORE
                int32_t B_id = loopN - batch_id_begin;
                int32_t N_id = loop;
                int32_t loop_id = N_id * batch_size + B_id;
#else
                int32_t B_id = loopN - batch_id_begin;
                int32_t N_id = loop;
                int32_t loop_id = B_id * repeat_times + N_id;
#endif

                if (loop_id % aiv_num != aiv_id) {
                    continue;
                }

#ifndef MIX_CORE
                if(aligned && init_A == 0) {
                    AscendC::DataCopy(MatrixA_Real_ub_tensor, a_gm_tensor[offset_loopc / 2 + 1], AscendC::DataCopyParams(1, (lenBurstC + 1) / 2, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopy(MatrixA_Imag_ub_tensor, a_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyParams(1, (lenBurstC + 1) / 2, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    init_A = 1;
                } else if (init_A == 0) {
                    AscendC::DataCopyPad(MatrixA_Real_ub_tensor, a_gm_tensor[offset_loopc / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));    // 搬对应的N-2个数进来
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopyPad(MatrixA_Imag_ub_tensor, a_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));    // 搬对应的N-2个数进来
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    init_A = 1;
                }

                if(aligned && init_B == 0) {
                    AscendC::DataCopy(MatrixB_Real_ub_tensor, b_gm_tensor[offset_loopc / 2 + 1], AscendC::DataCopyParams(1, (lenBurstC + 1) / 2, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopy(MatrixB_Imag_ub_tensor, b_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyParams(1, (lenBurstC + 1) / 2, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    init_B = 1;
                } else if (init_B == 0) {
                    AscendC::DataCopyPad(MatrixB_Real_ub_tensor, b_gm_tensor[offset_loopc / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopyPad(MatrixB_Imag_ub_tensor, b_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    init_B = 1;
                }
#endif

                auto event_id = ping ? EVENT_ID0 : EVENT_ID1;
                auto buf0_ub_tensor = ping ? UB_INPUT1_ub_tensor : UB_INPUT2_ub_tensor;   // 输入、大于512倒序、A矩阵、B矩阵、搬出
                auto buf1_ub_tensor = ping ? UB_INPUT_R1_ub_tensor: UB_INPUT_R2_ub_tensor;    // 输入的虚部
                auto buf2_ub_tensor = ping ? UB_INPUT_I1_ub_tensor: UB_INPUT_I2_ub_tensor;    // 输入的实部
                auto buf4_ub_tensor = ping ? UB_INPUT_R1_REVERSE_ub_tensor: UB_INPUT_R2_REVERSE_ub_tensor;    // 倒序后的实部
                auto buf5_ub_tensor = ping ? UB_INPUT_I1_REVERSE_ub_tensor: UB_INPUT_I2_REVERSE_ub_tensor;    // 倒序后的虚部
                auto temp1_ub_tensor = ping ? tvub1_ub_tensor : tvub3_ub_tensor;
                auto temp2_ub_tensor = ping ? tvub2_ub_tensor : tvub4_ub_tensor;
                uint64_t index1 = ping ? 2: 10;
                uint64_t index2 = ping ? 3: 11;
                uint64_t index3 = ping ? 6: 14;
                uint64_t index4 = ping ? 7: 15;

                int64_t offset = 2 + loopN * N;
                int64_t offset_out = 2 + loopN * (N + 2);

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
                //  **********搬入输入***************

#ifndef MIX_CORE
                if(aligned) {
                    AscendC::DataCopy(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc], AscendC::DataCopyParams(1, lenBurstC / 2, 0, 0));
                } else {
                    AscendC::DataCopyPad(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc], AscendC::DataCopyExtParams(1, actual_n * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                }
#else
                AscendC::DataCopyPad(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc], AscendC::DataCopyExtParams(1, actual_n * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
#endif

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id);
                //  **********输入虚实分离***************
                //输入实数分离
                // todo GatherMask
                vreducev2(
                    reinterpret_cast<__ubuf__ uint32_t *>(buf2_ub_tensor.GetPhyAddr(0)),
                    reinterpret_cast<__ubuf__ uint32_t *>(buf0_ub_tensor.GetPhyAddr(0)),
                    nullptr,
                    repeat_reduce, // repeat
                    1,        // src0BlockStride
                    1,        // patternMode, 101010…10
                    8,        // src0RepeatStride
                    8         // src1RepeatStride
                );
                // 输入虚数分离
                // todo GatherMask
                vreducev2(
                    reinterpret_cast<__ubuf__ uint32_t *>(buf1_ub_tensor.GetPhyAddr(0)),
                    reinterpret_cast<__ubuf__ uint32_t *>(buf0_ub_tensor.GetPhyAddr(0)),
                    nullptr,
                    repeat_reduce, // repeat
                    1,        // src0BlockStride
                    2,        // patternMode, 010101…01
                    8,        // src0RepeatStride
                    8         // src1RepeatStride
                );
                AscendC::PipeBarrier<PIPE_V>();
                //  **********倒序***************
                // 倒序分两种情况
                if(repeat_times == 1) {
                    // 一个UB块就能完成
                    // FI倒序
                    // todo GatherMask
                    vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf4_ub_tensor.GetPhyAddr(0)), (__ubuf__ unsigned int *)reverse_index_ub_tensor.GetPhyAddr(0), index1 * UB_VEC_SIZE, 8, repeat);
                    // FR倒序
                    // todo GatherMask
                    vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf5_ub_tensor.GetPhyAddr(0)), (__ubuf__ unsigned int *)reverse_index_ub_tensor.GetPhyAddr(0), index2 * UB_VEC_SIZE, 8, repeat);
                    AscendC::PipeBarrier<PIPE_V>();
                    // FI共轭
                    AscendC::Muls(buf4_ub_tensor, buf4_ub_tensor, -1.0f, vecMask, repeat, {1, 1, 8, 8});
                } else {
                    // 数据量大于512要用多个UB块操作
                    AscendC::PipeBarrier<PIPE_ALL>();

#ifndef MIX_CORE
                    int64_t offset_loopc_rev = (repeat_times - 1 - loop) * BIASC;
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
                    AscendC::DataCopy(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc_rev - 2], AscendC::DataCopyParams(1, lenBurstC, 0, 0));
#else
                    // int64_t offset_loopc_rev = (repeat_times - 1 - loop) * BIASC;
                    int64_t offset_block = ((loop + 1) * BIASC >= (N - 2)) ? (N - 2) : (loop * BIASC + BIASC);
                    int64_t offset_loopc_rev = (N - 2) - offset_block;
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
                    AscendC::DataCopyPad(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc_rev], AscendC::DataCopyExtParams(1, actual_n * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
#endif

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                    // 大于 UB_VEC_SIZE, 需要倒着读一块， 然后把这块倒着读回来参与计算
                    // 实数分离
                    vreducev2(
                        reinterpret_cast<__ubuf__ uint32_t *>(temp1_ub_tensor.GetPhyAddr(0)),
                        reinterpret_cast<__ubuf__ uint32_t *>(buf0_ub_tensor.GetPhyAddr(0)),
                        nullptr,
                        repeat_reduce, // repeat
                        1,        // src0BlockStride
                        1,        // patternMode, 101010…10
                        8,        // src0RepeatStride
                        8         // src1RepeatStride
                    );
                    // 虚数分离
                    vreducev2(
                        reinterpret_cast<__ubuf__ uint32_t *>(temp2_ub_tensor.GetPhyAddr(0)),
                        reinterpret_cast<__ubuf__ uint32_t *>(buf0_ub_tensor.GetPhyAddr(0)),
                        nullptr,
                        repeat_reduce, // repeat
                        1,        // src0BlockStride
                        2,        // patternMode, 010101…01
                        8,        // src0RepeatStride
                        8         // src1RepeatStride
                    );
                    AscendC::PipeBarrier<PIPE_V>();
                    // FI 倒序
                    // todo GatherMask
                    vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf4_ub_tensor.GetPhyAddr(0)), (__ubuf__ unsigned int *)reverse_index_ub_tensor.GetPhyAddr(0), index4 * UB_VEC_SIZE, 8, repeat);
                    // FR 倒序
                    // todo GatherMask
                    vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf5_ub_tensor.GetPhyAddr(0)), (__ubuf__ unsigned int *)reverse_index_ub_tensor.GetPhyAddr(0), index3 * UB_VEC_SIZE, 8, repeat);
                    AscendC::PipeBarrier<PIPE_V>();
                    // FI共轭
                    AscendC::Muls(buf4_ub_tensor, buf4_ub_tensor, -1.0f, vecMask, repeat, {1, 1, 8, 8});
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::PipeBarrier<PIPE_V>();
                //A的计算
                // 系数A实/虚分离
                //  **********A矩阵相关的计算***************
                // UB_BUF4：FI_REV;   UB_BUF5: FR_REV;   UB_BUF2:FR;    UB_BUF1:FI
                // UB_BUF6: AI;   UB_BUF7: AR
                // FR的计算
                AscendC::Mul(temp1_ub_tensor, buf1_ub_tensor, MatrixA_Real_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});
                AscendC::Mul(temp2_ub_tensor, buf2_ub_tensor, MatrixA_Real_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});
                AscendC::Mul(buf1_ub_tensor, buf1_ub_tensor, MatrixA_Imag_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});
                AscendC::Mul(buf2_ub_tensor, buf2_ub_tensor, MatrixA_Imag_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Sub(buf1_ub_tensor, temp2_ub_tensor, buf1_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FR * AR - FI * AI
                AscendC::Add(buf2_ub_tensor, temp1_ub_tensor, buf2_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FR * AI + FI * AR
                AscendC::PipeBarrier<PIPE_V>();
 
                // B的计算
                // 系数B实/虚分离
                //  **********B矩阵相关的计算***************
                // UB_BUF4：FI_REV;   UB_BUF5: FR_REV;   UB_BUF2:FR;    UB_BUF1:FI
                // UB_BUF6: BI;   UB_BUF7: BR
                // FrevR的计算

                AscendC::Mul(temp1_ub_tensor, buf5_ub_tensor, MatrixB_Real_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FervR * BR
                AscendC::Mul(temp2_ub_tensor, buf4_ub_tensor, MatrixB_Real_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FervI * BR
                AscendC::Mul(buf5_ub_tensor, buf5_ub_tensor, MatrixB_Imag_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});  // FrevR * BI
                AscendC::Mul(buf4_ub_tensor, buf4_ub_tensor, MatrixB_Imag_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});  // FrevI * BI
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Sub(buf4_ub_tensor, temp1_ub_tensor, buf4_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FervR * BR - FrevI * BI
                AscendC::Add(buf5_ub_tensor, temp2_ub_tensor, buf5_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FrevR * BI + FervI * BR
                AscendC::PipeBarrier<PIPE_V>();
 
                //  **********虚实结合***************
                // A和B结果组合
                AscendC::Add(buf1_ub_tensor, buf1_ub_tensor, buf4_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FR * AR - FI * AI
                AscendC::Add(buf2_ub_tensor, buf2_ub_tensor, buf5_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FR * AI + FI * AR
                AscendC::PipeBarrier<PIPE_V>();
                // UB_BUF8实数， UB_BUF9虚数，结果组合
                // 结果在UB_BUF0里面
                // todo GatherMask
                vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf4_ub_tensor.GetPhyAddr(0)), (__ubuf__ unsigned int *)vgather_index_ub_tensor.GetPhyAddr(0), index1 * UB_VEC_SIZE, 8, repeat_out);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
                //  **********搬出***************

#ifndef MIX_CORE
                if(aligned) {
                    AscendC::DataCopy(output_gm_tensor[offset_out + offset_loopc], buf4_ub_tensor, AscendC::DataCopyParams(1, lenBurstC, 0, 0));
                } else {
                    AscendC::DataCopyPad(output_gm_tensor[offset_out + offset_loopc], buf4_ub_tensor, AscendC::DataCopyExtParams(1, actual_n * sizeof(float), 0, 0, 0));
                }
#else
                AscendC::DataCopyPad(output_gm_tensor[offset_out + offset_loopc], buf4_ub_tensor, AscendC::DataCopyExtParams(1, actual_n * sizeof(float), 0, 0, 0));
#endif

                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id);
                ping = 1 - ping;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
}


__aicore__ __inline__ void C2R(
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
        uint64_t num = N + 2 > BIASC ? BIASC : N + 2;
        uint64_t out_num = N > BIASC ? BIASC : N; // 为规避Vgather问题引入的暂时变量 // 
        int index_size = (N + 2)>= BIASC ? BIAS : (((N + 2) / 2 + 63) / 64) * 64;
        uint64_t lenBurstC = (num + 7) / 8;
        uint64_t lenBurstOut = ((N > BIASC ? BIASC : N) + 7) / 8;
        uint64_t lenBurstIndex = (index_size + 7) / 8;
        uint64_t lenBurstOutIndex = (BIASC + 7) / 8;
        // 每次vector乘加需要的次数
        uint8_t repeat = (num / 2 + 63) / 64; //  此处会导致越界UB操作  // 512
        // 每次vector reduce需要的次数
        uint8_t repeat_reduce = (num + 63) / 64;
        // 每次vgather output需要的次数
        uint8_t repeat_out = (out_num + 63) / 64;
        // 每次把实数和虚部一起取出来
        uint64_t loopN_times = batch_size;
        // 每次取得数的个数
        // Keep the number of output tiles based on N. The last input tile may
        // contain the extra Nyquist pair (N+2 floats in total).
        uint64_t repeat_times = (N + BIASC - 1) / BIASC;

        uint64_t ping = 1;
        // 开多核

        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        AscendC::LocalTensor<float> UB_INPUT1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(2 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_I1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(3 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R1_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(4 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_I1_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(5 * 8192);
        AscendC::LocalTensor<float> tvub1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(6 * 8192);
        AscendC::LocalTensor<float> tvub2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(7 * 8192);

        AscendC::LocalTensor<float> UB_INPUT2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(8 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(10 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_I2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(11 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_R2_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(12 * 8192);
        AscendC::LocalTensor<float> UB_INPUT_I2_REVERSE_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(13 * 8192);
        AscendC::LocalTensor<float> tvub3_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(14 * 8192);
        AscendC::LocalTensor<float> tvub4_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(15 * 8192);

        AscendC::LocalTensor<float> MatrixA_Real_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(16 * 8192);
        AscendC::LocalTensor<float> MatrixA_Imag_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(17 * 8192);
        AscendC::LocalTensor<float> MatrixB_Real_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(18 * 8192);
        AscendC::LocalTensor<float> MatrixB_Imag_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(19 * 8192);

        AscendC::LocalTensor<uint32_t> reverse_index_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(20 * 8192);
        AscendC::LocalTensor<uint32_t> vgather_index_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(21 * 8192);


#ifndef MIX_CORE
        AscendC::DataCopy(reverse_index_ub_tensor, input_index_gm_tensor, AscendC::DataCopyParams(1, lenBurstIndex, 0, 0));  // 搬入512长度
        AscendC::PipeBarrier<PIPE_MTE2>();
#endif

        AscendC::DataCopy(vgather_index_ub_tensor, output_index_gm_tensor, AscendC::DataCopyParams(1, lenBurstOutIndex, 0, 0));  // 搬入4096长度，污染后一个
        AscendC::PipeBarrier<PIPE_MTE2>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

        for(uint64_t loop = 0; loop < repeat_times; loop++) {       // 序列方向的循环
            
            int64_t offset_loopc = loop * BIASC;      // 已经处理过的长度
            int actual_n = N + 2;

#ifndef MIX_CORE
            actual_n = actual_n < BIASC ? actual_n : ((loop + 1) * BIASC <= actual_n ? BIASC : (actual_n - loop * BIASC));
            bool aligned = (actual_n % BLOCK_SIZE_8 == 0);

            bool init_A = 0;
            bool init_B = 0;
#else
            int reverse_offset = 0;

            if (repeat_times != 1) {                        // 若N-2长度大于4096则循环
                actual_n = loop + 1 < repeat_times ? BIASC : ((N + 2) - loop * BIASC);
            }
            
            if (repeat_times != 1 && actual_n != BIASC) {   // 需要读到对应的reverse数组
                reverse_offset = index_size;
            }

            if (N == 4096) {
                actual_n = N;           // 4096长度需要特殊处理 
            }
            AscendC::DataCopy(reverse_index_ub_tensor, input_index_gm_tensor[reverse_offset], AscendC::DataCopyParams(1, lenBurstIndex, 0, 0));   // 搬入2048个数
            AscendC::PipeBarrier<PIPE_MTE2>();
            
            AscendC::DataCopyPad(MatrixA_Real_ub_tensor, a_gm_tensor[offset_loopc / 2], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));      // 搬对应的N-2个数进来
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::DataCopyPad(MatrixA_Imag_ub_tensor, a_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));       // 搬对应的N-2个数进来
            AscendC::PipeBarrier<PIPE_MTE2>();

            AscendC::DataCopyPad(MatrixB_Real_ub_tensor, b_gm_tensor[offset_loopc / 2], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
            AscendC::PipeBarrier<PIPE_MTE2>();      // 搬对应的N-2个数进来
            AscendC::DataCopyPad(MatrixB_Imag_ub_tensor, b_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));      // 搬对应的N-2个数进来
            AscendC::PipeBarrier<PIPE_MTE2>();

#endif

            for(uint64_t loopN = batch_id_begin; loopN < batch_id_end; loopN+=1) {  // Batch方向的循环

#ifndef MIX_CORE
                int32_t B_id = loopN - batch_id_begin;
                int32_t N_id = loop;
                int32_t loop_id = N_id * batch_size + B_id;
                if (loop_id % aiv_num != aiv_id) {
                    continue;
                }
                
                if(aligned && init_A == 0) {
                    // LOAD A Matrix
                    AscendC::DataCopy(MatrixA_Real_ub_tensor, a_gm_tensor[offset_loopc / 2], AscendC::DataCopyParams(1, (lenBurstC + 1) / 2, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopy(MatrixA_Imag_ub_tensor, a_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyParams(1, (lenBurstC + 1) / 2, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    init_A = 1;
                } else if (init_A == 0) {
                    AscendC::DataCopyPad(MatrixA_Real_ub_tensor, a_gm_tensor[offset_loopc / 2], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));    // 搬对应的N-2个数进来
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopyPad(MatrixA_Imag_ub_tensor, a_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));    // 搬对应的N-2个数进来
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    init_A = 1;
                }
                

                if(aligned && init_B == 0) {
                    // LOAD B Matrix
                    AscendC::DataCopy(MatrixB_Real_ub_tensor, b_gm_tensor[offset_loopc / 2], AscendC::DataCopyParams(1, (lenBurstC + 1) / 2, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopy(MatrixB_Imag_ub_tensor, b_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyParams(1, (lenBurstC + 1) / 2, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    init_B = 1;
                } else if (init_B == 0) {
                    AscendC::DataCopyPad(MatrixB_Real_ub_tensor, b_gm_tensor[offset_loopc / 2], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    AscendC::DataCopyPad(MatrixB_Imag_ub_tensor, b_gm_tensor[offset_loopc / 2 + N / 2 + 1], AscendC::DataCopyExtParams(1, (actual_n / 2) * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    init_B = 1;
                }
#else
                int32_t B_id = loopN - batch_id_begin;
                int32_t N_id = loop;
                int32_t loop_id = N_id * batch_size + B_id;
                if (loop_id % aiv_num != aiv_id) {
                    continue;
                }
#endif

                auto event_id = ping ? EVENT_ID0 : EVENT_ID1;
                auto buf0_ub_tensor = ping ? UB_INPUT1_ub_tensor : UB_INPUT2_ub_tensor;   // 输入、大于512倒序、A矩阵、B矩阵、搬出
                auto buf1_ub_tensor = ping ? UB_INPUT_R1_ub_tensor: UB_INPUT_R2_ub_tensor;    // 输入的虚部
                auto buf2_ub_tensor = ping ? UB_INPUT_I1_ub_tensor: UB_INPUT_I2_ub_tensor;    // 输入的实部
                auto buf4_ub_tensor = ping ? UB_INPUT_R1_REVERSE_ub_tensor: UB_INPUT_R2_REVERSE_ub_tensor;    // 倒序后的实部
                auto buf5_ub_tensor = ping ? UB_INPUT_I1_REVERSE_ub_tensor: UB_INPUT_I2_REVERSE_ub_tensor;    // 倒序后的虚部
                auto temp1_ub_tensor = ping ? tvub1_ub_tensor : tvub3_ub_tensor;
                auto temp2_ub_tensor = ping ? tvub2_ub_tensor : tvub4_ub_tensor;
                uint64_t index1 = ping ? 2: 10;
                uint64_t index2 = ping ? 3: 11;
                uint64_t index3 = ping ? 6: 14;
                uint64_t index4 = ping ? 7: 15;

                int64_t offset = loopN * (N + 2);
                int64_t offset_out = loopN * N;

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
                //  **********搬入***************
                int actual_n = N + 2;
                actual_n = actual_n < BIASC ? actual_n : ((loop + 1) * BIASC <= actual_n ? BIASC : (actual_n - loop * BIASC));
#ifdef MIX_CORE
                if (repeat_times != 1 && loop + 1 == repeat_times) {
                    actual_n = N + 2 - loop * BIASC;
                }
                repeat = (actual_n / 2 + 63) / 64;
                repeat_reduce = (actual_n + 63) / 64;
#endif

#ifndef MIX_CORE
                bool aligned = (actual_n % BLOCK_SIZE_8 == 0);
#endif

                AscendC::DataCopyPad(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc], AscendC::DataCopyExtParams(1, actual_n * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id);
                //  **********C2R预先置0*****************
                // 使用带mask的vsub将首位虚部以及末位虚部置为0
                if (loop == 0) {
                    AscendC::SetVectorMask<float>(-1, 0x0000000000000002);
                    AscendC::Sub<float, false>(buf0_ub_tensor, buf0_ub_tensor, buf0_ub_tensor, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
                    AscendC::SetVectorMask<float>(-1, -1);
                }
                AscendC::PipeBarrier<PIPE_V>();

#ifndef MIX_CORE
                if (loop == repeat_times - 1) {
#else
                if (loop == repeat_times - 1 && ((N + 2) % BIASC != 0)) {
#endif

                    int setzero_offset = actual_n / 64;
                    int remain_len = actual_n - (setzero_offset) * 64;
                    int mask_offset = remain_len - 1;
                    if (actual_n % 64 == 0) {
                        AscendC::SetVectorMask<float>(0x0, 0x8000000000000000);
                        AscendC::Sub<float, false>(buf0_ub_tensor[(setzero_offset - 1) * 64], buf0_ub_tensor[(setzero_offset - 1) * 64], buf0_ub_tensor[(setzero_offset - 1) * 64], AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
                    }
                    else {
                        AscendC::SetVectorMask<float>(0x0, (uint64_t) 1 << mask_offset);
                        AscendC::Sub<float, false>(buf0_ub_tensor[setzero_offset * 64], buf0_ub_tensor[setzero_offset * 64], buf0_ub_tensor[setzero_offset * 64], AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
                    }
                    AscendC::SetVectorMask<float>(-1, -1);
                }
                AscendC::PipeBarrier<PIPE_V>();
                //  **********输入虚实分离***************
                //输入实数分离
                vreducev2(
                    reinterpret_cast<__ubuf__ uint32_t *>(buf2_ub_tensor.GetPhyAddr(0)),
                    reinterpret_cast<__ubuf__ uint32_t *>(buf0_ub_tensor.GetPhyAddr(0)),
                    nullptr,
                    repeat_reduce, // repeat
                    1,        // src0BlockStride
                    1,        // patternMode, 101010…10
                    8,        // src0RepeatStride
                    8         // src1RepeatStride
                );
                // 输入虚数分离
                vreducev2(
                    reinterpret_cast<__ubuf__ uint32_t *>(buf1_ub_tensor.GetPhyAddr(0)),
                    reinterpret_cast<__ubuf__ uint32_t *>(buf0_ub_tensor.GetPhyAddr(0)),
                    nullptr,
                    repeat_reduce, // repeat
                    1,        // src0BlockStride
                    2,        // patternMode, 010101…01
                    8,        // src0RepeatStride
                    8         // src1RepeatStride
                );
                AscendC::PipeBarrier<PIPE_V>();
                //  **********倒序***************
                // 倒序分两种情况
                if(N < BIASC && repeat_times == 1) {
                    // 一个UB块就能完成
                    // FI倒序
                    // todo GatherMask
                    vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf4_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ uint32_t *>(reverse_index_ub_tensor.GetPhyAddr(0)), index1 * UB_VEC_SIZE, 8, repeat);
                    // FR倒序
                    // todo GatherMask
                    vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf5_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ uint32_t *>(reverse_index_ub_tensor.GetPhyAddr(0)), index2 * UB_VEC_SIZE, 8, repeat);
                    AscendC::PipeBarrier<PIPE_V>();
                    // FI共轭
                    AscendC::Muls(buf4_ub_tensor, buf4_ub_tensor, -1.0f, vecMask, repeat, {1, 1, 8, 8});
                } else {
                    // 数据量大于512要用多个UB块操作
                    AscendC::PipeBarrier<PIPE_ALL>();

#ifndef MIX_CORE
                    int64_t offset_loopc_rev = (repeat_times - 1 - loop) * BIASC;
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
                    AscendC::DataCopy(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc_rev + 2], AscendC::DataCopyParams(1, lenBurstC, 0, 0));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                    // 大于 UB_VEC_SIZE, 需要倒着读一块， 然后把这块倒着读回来参与计算
 
#else
                    int64_t offset_block = loop * BIASC + actual_n;
                    int64_t offset_loopc_rev = (N + 2) - offset_block;
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
                    AscendC::DataCopyPad(buf0_ub_tensor, input_gm_tensor[offset + offset_loopc_rev], AscendC::DataCopyExtParams(1, actual_n * sizeof(float), 0, 0, 0), AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                    // 大于 UB_VEC_SIZE, 需要倒着读一块， 然后把这块倒着读回来参与计算
                    if (loop == 0) {
                        AscendC::SetVectorMask<float>(0x0, 0x8000000000000000);
                        AscendC::Sub<float, false>(buf0_ub_tensor[4032], buf0_ub_tensor[4032], buf0_ub_tensor[4032], AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
                        AscendC::SetVectorMask<float>(-1, -1);
                    }
                    AscendC::PipeBarrier<PIPE_V>();

#endif

                    // 实数分离
                    vreducev2(
                        reinterpret_cast<__ubuf__ uint32_t *>(temp1_ub_tensor.GetPhyAddr(0)),
                        reinterpret_cast<__ubuf__ uint32_t *>(buf0_ub_tensor.GetPhyAddr(0)),
                        nullptr,
                        repeat_reduce, // repeat
                        1,        // src0BlockStride
                        1,        // patternMode, 101010…10
                        8,        // src0RepeatStride
                        8         // src1RepeatStride
                    );
                    // 虚数分离
                    vreducev2(
                        reinterpret_cast<__ubuf__ uint32_t *>(temp2_ub_tensor.GetPhyAddr(0)),
                        reinterpret_cast<__ubuf__ uint32_t *>(buf0_ub_tensor.GetPhyAddr(0)),
                        nullptr,
                        repeat_reduce, // repeat
                        1,        // src0BlockStride
                        2,        // patternMode, 010101…01
                        8,        // src0RepeatStride
                        8         // src1RepeatStride
                    );
                    AscendC::PipeBarrier<PIPE_V>();
                    // FI 倒序
                    // todo GatherMask
                    vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf4_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ uint32_t *>(reverse_index_ub_tensor.GetPhyAddr(0)), index4 * UB_VEC_SIZE, 8, repeat);
                    // FR 倒序
                    // todo GatherMask
                    vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf5_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ uint32_t *>(reverse_index_ub_tensor.GetPhyAddr(0)), index3 * UB_VEC_SIZE, 8, repeat);
                    AscendC::PipeBarrier<PIPE_V>();
                    // FI共轭
                    AscendC::Muls(buf4_ub_tensor, buf4_ub_tensor, -1.0f, vecMask, repeat, {1, 1, 8, 8});
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::PipeBarrier<PIPE_V>();
                //A的计算
                // 系数A实/虚分离
                //  **********A矩阵相关的计算***************
                // UB_BUF4：FI_REV;   UB_BUF5: FR_REV;   UB_BUF2:FR;    UB_BUF1:FI
                // UB_BUF6: AI;   UB_BUF7: AR
                // FR的计算
                AscendC::Mul(temp1_ub_tensor, buf1_ub_tensor, MatrixA_Real_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});
                AscendC::Mul(temp2_ub_tensor, buf2_ub_tensor, MatrixA_Real_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});
                AscendC::Mul(buf1_ub_tensor, buf1_ub_tensor, MatrixA_Imag_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});
                AscendC::Mul(buf2_ub_tensor, buf2_ub_tensor, MatrixA_Imag_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Sub(buf1_ub_tensor, temp2_ub_tensor, buf1_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FR * AR - FI * AI
                AscendC::Add(buf2_ub_tensor, temp1_ub_tensor, buf2_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FR * AI + FI * AR
                AscendC::PipeBarrier<PIPE_V>();
 
                // B的计算
                // 系数B实/虚分离
                //  **********B矩阵相关的计算***************
                // UB_BUF4：FI_REV;   UB_BUF5: FR_REV;   UB_BUF2:FR;    UB_BUF1:FI
                // UB_BUF6: BI;   UB_BUF7: BR
                // FrevR的计算

                AscendC::Mul(temp1_ub_tensor, buf5_ub_tensor, MatrixB_Real_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FervR * BR
                AscendC::Mul(temp2_ub_tensor, buf4_ub_tensor, MatrixB_Real_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FervI * BR
                AscendC::Mul(buf5_ub_tensor, buf5_ub_tensor, MatrixB_Imag_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});  // FrevR * BI
                AscendC::Mul(buf4_ub_tensor, buf4_ub_tensor, MatrixB_Imag_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8});  // FrevI * BI
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(buf4_ub_tensor, temp1_ub_tensor, buf4_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FervR * BR - FrevI * BI
                AscendC::Add(buf5_ub_tensor, temp2_ub_tensor, buf5_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FrevR * BI + FervI * BR
                AscendC::PipeBarrier<PIPE_V>();
 
                //  **********虚实结合***************
                // A和B结果组合
                AscendC::Add(buf1_ub_tensor, buf1_ub_tensor, buf4_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FR * AR - FI * AI
                AscendC::Add(buf2_ub_tensor, buf2_ub_tensor, buf5_ub_tensor, vecMask, repeat, {1, 1, 1, 8, 8, 8}); // FR * AI + FI * AR
                AscendC::PipeBarrier<PIPE_V>();
                // UB_BUF8实数， UB_BUF9虚数，结果组合
                // 结果在UB_BUF0里面
                // todo GatherMask
                vgather(reinterpret_cast<__ubuf__ uint32_t *>(buf4_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ uint32_t *>(vgather_index_ub_tensor.GetPhyAddr(0)), index1 * UB_VEC_SIZE, 8, repeat_out);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
                //  **********搬出***************
                actual_n = N;
                actual_n = actual_n < BIASC ? actual_n : ((loop + 1) * BIASC <= actual_n ? BIASC : (actual_n - loop * BIASC));

#ifndef MIX_CORE
                aligned = (actual_n % BLOCK_SIZE_8 == 0);
                if(aligned) {
                    AscendC::DataCopy(output_gm_tensor[offset_out + offset_loopc], buf4_ub_tensor, AscendC::DataCopyParams(1, lenBurstOut, 0, 0));
                } else {
                    AscendC::DataCopyPad(output_gm_tensor[offset_out + offset_loopc], buf4_ub_tensor, AscendC::DataCopyExtParams(1, actual_n * sizeof(float), 0, 0, 0));
                }
#else
                AscendC::DataCopyPad(output_gm_tensor[offset_out + offset_loopc], buf4_ub_tensor, AscendC::DataCopyExtParams(1, actual_n * sizeof(float), 0, 0, 0));
#endif

                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id);
                ping = 1 - ping;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
}


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
