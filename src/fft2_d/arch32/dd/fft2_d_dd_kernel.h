/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT2_D_DD_KERNEL_H
#define FFT2_D_DD_KERNEL_H

#include "kernel/fft_common_kernel.h"

#include <stdint.h>

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

namespace FFT2DKernel {

const int16_t SYNC_ALL_CORE_MODE_CODE = 0;
const int16_t SYNC_SINGLE_CV_MODE_CODE = 2;

const int16_t SYNC_ALL_CORE_FLAG_ID = 0;
const int16_t SYNC_SINGLE_VC_FLAG_ID = 1;
const int16_t SYNC_SINGLE_CV_FLAG_ID = 2;

/**
 * @brief Ceil(a/b)
 * @param [in] int64_t a：需要取整的数
 * @param [in] int64_t b：需要向上取整的最小粒度
 * @return Ceil(a/b)
 */
__aicore__ inline int64_t __attribute__((overloadable, always_inline)) CeilADivB(int64_t a, int64_t b)
{
    return ((a + b - 1) / b);
}

#ifdef __DAV_C220_VEC__

const uint64_t BLOCK_SIZE = 32 / sizeof(float);
const uint64_t BLOCK_PER_REPEAT = 8;
const uint64_t COMPLEX_DOUBLE = 2;

const uint16_t PATTERN_MODE_REPEAT_TWO_GET_FIRST = 1;
const uint16_t PATTERN_MODE_REPEAT_TWO_GET_SECOND = 2;

/*********************************
 * @brief 从gm指定位置拷入数据至连续ub（使用所有VectorCore）
 * @param [out] ub_dst ub数据块的首地址（float32）
 * @param [in] gm_src 输入数据块的首地址（float32）
 * @param [in] repeat 重复搬运次数
 * @param [in] copyCount 单个数据块长度（float32、假定按block对齐）
 * @param [in] srcDValue 输入每行数据的间隔（假定按block对齐）
*********************************/
__aicore__ __inline__ void LoadDataFromGmToUb(AscendC::LocalTensor<float> dst_ub_tensor, AscendC::GlobalTensor<float> src_gm_tensor, uint32_t copyCount) {

    int32_t copy_block_count = CeilADivB(copyCount, BLOCK_SIZE);
    AscendC::DataCopy(
        dst_ub_tensor,
        src_gm_tensor,
        AscendC::DataCopyParams(
            1, // blockCount
            copy_block_count, // blockLen
            0, // srcStride
            0 // dstStride
        )
    );
}

/*********************************
 * @brief 从ub指定位置搬出连续数据至gm（使用所有VectorCore）
 * @param [out] gm_dst gm输出数据块的首地址（float32）
 * @param [in] ub_src ub源数据块的首地址（float32）
 * @param [in] repeat 重复搬运次数
 * @param [in] copyCount 单个数据块长度（float32、假定按block对齐）
 * @param [in] dstDValue 输入每行数据的间隔（假定按block对齐）
*********************************/
__aicore__ __inline__ void SaveDataFromUbToGm(AscendC::GlobalTensor<float> dst_gm_tensor, AscendC::LocalTensor<float> src_ub_tensor, uint32_t copyCount) {

    int32_t copy_block_count = CeilADivB(copyCount, BLOCK_SIZE);
    AscendC::DataCopy(
        dst_gm_tensor,
        src_ub_tensor,
        AscendC::DataCopyParams(
            1, // blockCount
            copy_block_count, // blockLen
            0, // srcStride
            0 // dstStride
        )
    );
}

__aicore__ __inline__ void SaveDataFromUbToGmWithJump(AscendC::GlobalTensor<float> dst_gm_tensor, AscendC::LocalTensor<float> src_ub_tensor, uint32_t repeat, uint32_t copyCount, uint32_t offset) {
    int32_t copy_block_count = CeilADivB(copyCount, BLOCK_SIZE);
    int32_t gm_dst_stride = offset / BLOCK_SIZE - copy_block_count;

    AscendC::DataCopy(
        dst_gm_tensor,
        src_ub_tensor,
        AscendC::DataCopyParams(
            repeat, // blockCount
            copy_block_count, // blockLen
            0, // srcStride
            gm_dst_stride // dstStride
        )
    );
}

__aicore__ __inline__ void SeparateRIMultiBatch(
    AscendC::GlobalTensor<float> dst_gm_tensor, AscendC::GlobalTensor<float> src_gm_tensor, uint32_t x_len, uint32_t y_len, uint32_t current_batch_nums, bool flag) {
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> complex_buf1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> real_buf1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(48*1024);
    AscendC::LocalTensor<float> imag_buf1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(72*1024);
    AscendC::LocalTensor<float> complex_buf2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(96*1024);
    AscendC::LocalTensor<float> real_buf2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(144*1024);
    AscendC::LocalTensor<float> imag_buf2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(168*1024);

    const int32_t max_datacount_perloop = 48 * 1024 / (COMPLEX_DOUBLE * sizeof(float));
    const int32_t row_count_per_loop = max_datacount_perloop / y_len;

    const int32_t loop_count = CeilADivB(x_len, row_count_per_loop);
    const int32_t loop_remain_row_count = x_len % row_count_per_loop;

    // divide data
    for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {

        int32_t row_count_current = ( (loop_idx == loop_count - 1) && (loop_remain_row_count > 0) ) ? loop_remain_row_count : row_count_per_loop;

        // double buffer prepare
        auto complex_buf_ub_tensor = flag ? complex_buf1_ub_tensor : complex_buf2_ub_tensor;
        auto real_buf_ub_tensor = flag ? real_buf1_ub_tensor : real_buf2_ub_tensor;
        auto imag_buf_ub_tensor = flag ? imag_buf1_ub_tensor : imag_buf2_ub_tensor;
        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        
        //load data from gm to ub
        LoadDataFromGmToUb(complex_buf_ub_tensor, src_gm_tensor[loop_idx * row_count_per_loop * y_len * COMPLEX_DOUBLE], row_count_current * y_len  * COMPLEX_DOUBLE);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
        // separate
        int32_t repeat_count = (row_count_current * y_len * COMPLEX_DOUBLE) / (BLOCK_PER_REPEAT * BLOCK_SIZE);
        int32_t repeat_tail = (row_count_current * y_len * COMPLEX_DOUBLE) % (BLOCK_PER_REPEAT * BLOCK_SIZE);
        int32_t all_repeat_times = repeat_count + (repeat_tail > 0 ? 1 : 0);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(real_buf_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ uint32_t *>(complex_buf_ub_tensor.GetPhyAddr(0)), nullptr, all_repeat_times, 1, PATTERN_MODE_REPEAT_TWO_GET_FIRST, BLOCK_PER_REPEAT, BLOCK_PER_REPEAT);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(imag_buf_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ uint32_t *>(complex_buf_ub_tensor.GetPhyAddr(0)), nullptr, all_repeat_times, 1, PATTERN_MODE_REPEAT_TWO_GET_SECOND, BLOCK_PER_REPEAT, BLOCK_PER_REPEAT);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
        //save data from ub to gm
        // 存储基址与加载侧（loop_idx * row_count_per_loop * y_len）统一口径
        // （issue #126）：本循环加载/存储针对同一批行，按 row_count_current 累加
        // 会在尾循环（x_len % row_count_per_loop != 0 时）错位——先覆盖已写入行，
        // 尾部行永远未写入
        SaveDataFromUbToGmWithJump(dst_gm_tensor[loop_idx * row_count_per_loop * y_len], real_buf_ub_tensor, row_count_current, y_len, y_len * current_batch_nums);
        SaveDataFromUbToGmWithJump(dst_gm_tensor[x_len* y_len * current_batch_nums + loop_idx * row_count_per_loop * y_len], imag_buf_ub_tensor, row_count_current, y_len, y_len * current_batch_nums);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }

}
__aicore__ __inline__ void SeparateRI(
    AscendC::GlobalTensor<float> dst_gm_tensor, AscendC::GlobalTensor<float> src_gm_tensor, uint32_t separate_count, uint32_t real_offset, uint32_t imag_offset) {
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> complex_buf1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> real_buf1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(48*1024);
    AscendC::LocalTensor<float> imag_buf1_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(72*1024);
    AscendC::LocalTensor<float> complex_buf2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(96*1024);
    AscendC::LocalTensor<float> real_buf2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(144*1024);
    AscendC::LocalTensor<float> imag_buf2_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(168*1024);

    const int32_t max_datacount_perloop = 48 * 1024 / (COMPLEX_DOUBLE * sizeof(float));

    const int32_t loop_count = CeilADivB(separate_count, max_datacount_perloop);
    const int32_t loop_remain_count = separate_count % max_datacount_perloop;

    // divide data
    bool flag = 0;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {

        int32_t count_current = ( (loop_idx == loop_count - 1) && (loop_remain_count > 0) ) ? loop_remain_count : max_datacount_perloop;

        // double buffer prepare
        auto complex_buf_ub_tensor = flag ? complex_buf1_ub_tensor : complex_buf2_ub_tensor;
        auto real_buf_ub_tensor = flag ? real_buf1_ub_tensor : real_buf2_ub_tensor;
        auto imag_buf_ub_tensor = flag ? imag_buf1_ub_tensor : imag_buf2_ub_tensor;
        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        //load data from gm to ub
        LoadDataFromGmToUb(complex_buf_ub_tensor, src_gm_tensor[loop_idx * max_datacount_perloop * COMPLEX_DOUBLE],  count_current * COMPLEX_DOUBLE);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
        // separate
        int32_t repeat_count = (count_current * COMPLEX_DOUBLE) / (BLOCK_PER_REPEAT * BLOCK_SIZE);
        int32_t repeat_tail = (count_current * COMPLEX_DOUBLE) % (BLOCK_PER_REPEAT * BLOCK_SIZE);
        int32_t all_repeat_times = repeat_count + (repeat_tail > 0 ? 1 : 0);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(real_buf_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ uint32_t *>(complex_buf_ub_tensor.GetPhyAddr(0)), nullptr, all_repeat_times, 1, PATTERN_MODE_REPEAT_TWO_GET_FIRST, BLOCK_PER_REPEAT, BLOCK_PER_REPEAT);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(imag_buf_ub_tensor.GetPhyAddr(0)), reinterpret_cast<__ubuf__ uint32_t *>(complex_buf_ub_tensor.GetPhyAddr(0)), nullptr, all_repeat_times, 1, PATTERN_MODE_REPEAT_TWO_GET_SECOND, BLOCK_PER_REPEAT, BLOCK_PER_REPEAT);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
        //save data from ub to gm
        SaveDataFromUbToGm(dst_gm_tensor[loop_idx * max_datacount_perloop + real_offset], real_buf_ub_tensor, count_current);
        SaveDataFromUbToGm(dst_gm_tensor[loop_idx * max_datacount_perloop + imag_offset], imag_buf_ub_tensor, count_current);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        flag = 1 - flag;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ __inline__ void SeparateRITWO(
    AscendC::GlobalTensor<float> dst_gm_tensor, AscendC::GlobalTensor<float> src_gm_tensor, uint32_t x_len, uint32_t y_len) {

    uint32_t total_count = x_len * y_len;
    uint32_t first_vec_count = total_count / 2;
    uint32_t second_vec_count = total_count  - first_vec_count;
    uint32_t real_offset = 0;
    uint32_t imag_offset = total_count;
    if (get_subblockid() == 0) {
        SeparateRI(dst_gm_tensor, src_gm_tensor, first_vec_count, real_offset, imag_offset);
    }else {
        SeparateRI(dst_gm_tensor[first_vec_count], src_gm_tensor[first_vec_count * 2], second_vec_count, real_offset, imag_offset);
    }    
    
}
__aicore__ __inline__ void SeparateRITWOMultiBatch(
    AscendC::GlobalTensor<float> dst_gm_tensor, AscendC::GlobalTensor<float> src_gm_tensor, uint32_t x_len, uint32_t y_len, uint32_t current_batch_nums) {

    AscendC::GlobalTensor<float> dst_current_gm_tensor;
    AscendC::GlobalTensor<float> src_current_gm_tensor;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    bool flag = 0;
    for (int i = 0; i < current_batch_nums; i++){
    if (i % get_subblockdim() != get_subblockid()) {
        continue;
        }
        dst_current_gm_tensor = dst_gm_tensor[i * y_len];
        src_current_gm_tensor = src_gm_tensor[i * x_len * y_len * COMPLEX_DOUBLE];
        SeparateRIMultiBatch(dst_current_gm_tensor, src_current_gm_tensor, x_len, y_len, current_batch_nums, flag);
        flag = 1 - flag; 
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}
#endif

#ifdef __DAV_C220_CUBE__

using T_INPUT = float;
using T_OUTPUT = float;

const uint64_t COMPLEX_DOUBLE = 2;

constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(T_INPUT);   // 32 KB
constexpr int32_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(T_OUTPUT);   // 64 KB
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t C0_SIZE = 32 / sizeof(T_INPUT);
constexpr int32_t CUBE_MATRIX_SIZE = BLOCK_SIZE * C0_SIZE;                  // 16 * 8
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(T_INPUT);    // 256 KB

__aicore__ __attribute__((always_inline)) inline void load_matrix_zZ(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    int32_t R, int32_t C,
    int32_t valid_row, int32_t valid_col,
    int32_t stride
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
                AscendC::Nd2NzParams(
                    ndNum,          // ndNum
                    R0,             // nValue
                    valid_col,      // dValue
                    R0 * stride,    // srcNdMatrixStride
                    srcNStride,     // srcDValue
                    R0,             // dstNzC0Stride
                    1,              // dstNzNStride
                    R0 * C)         // dstNzMatrixStride
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * R0 * C], src[ndNum * R0 * stride],
                AscendC::Nd2NzParams(
                    1,              // ndNum
                    remains,        // nValue
                    valid_col,      // dValue
                    0,              // srcNdMatrixStride
                    srcNStride,     // srcDValue
                    R0,             // dstNzC0Stride
                    1,              // dstNzNStride
                    0)              // dstNzMatrixStride
            );
        }
    } else if (srcNStride < STRIDE_LIMIT) {
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;
        for (int32_t i = 0; i < ndNum; i++) {
            AscendC::DataCopy(
                dst[i * R0 * C], src[i * R0 * stride],
                AscendC::Nd2NzParams(
                    1,              // ndNum
                    R0,             // nValue
                    valid_col,      // dValue
                    0,              // srcNdMatrixStride
                    srcNStride,     // srcDValue
                    R0,             // dstNzC0Stride
                    1,              // dstNzNStride
                    0)              // dstNzMatrixStride
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * R0 * C], src[ndNum * R0 * stride],
                AscendC::Nd2NzParams(
                    1,              // ndNum
                    remains,        // nValue
                    valid_col,      // dValue
                    0,              // srcNdMatrixStride
                    srcNStride,     // srcDValue
                    R0,             // dstNzC0Stride
                    1,              // dstNzNStride
                    0)              // dstNzMatrixStride
            );
        }
    } else {
        for (int32_t i = 0; i < valid_row; i++) {
            int32_t idxR0 = i / R0;
            int32_t idxInR0 = i % R0;

            AscendC::DataCopy(
                dst[idxR0 * R0 * C + idxInR0 * C0], src[i * stride],
                AscendC::Nd2NzParams(
                    1,              // ndNum
                    1,              // nValue
                    valid_col,      // dValue
                    0,              // srcNdMatrixStride
                    0,              // srcDValue
                    R0,             // dstNzC0Stride
                    0,              // dstNzNStride
                    0)              // dstNzMatrixStride
            );
        }
    }
}

__aicore__ __inline__ void LoadPMatrix(
    AscendC::GlobalTensor<float> p_gm_tensor,
    int32_t x_len, int32_t y_len)
{   
    int32_t M = x_len * COMPLEX_DOUBLE;
    int32_t K = x_len * COMPLEX_DOUBLE;

    int32_t batchSize = 1;
    int32_t M0 = 128;
    int32_t K0 = 128;

    if (x_len == 32) 
    {
         M0 = 64;
         K0 = 64;
    }

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> p_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)0);

    int32_t m_loop = (M + M0 - 1) / M0; //2
    int32_t k_loop = (K + K0 - 1) / K0; //2
    int32_t loop = batchSize * m_loop; //2

    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {
        int64_t batch_idx = loop_idx / (m_loop);
        int64_t in_batch_idx = loop_idx % (m_loop);

        int64_t m_idx = in_batch_idx;

        int64_t offset_a, offset_l1a;
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
        int32_t m_round = (m_actual + 15) / 16 * 16;

        for (int32_t k_idx = 0; k_idx < k_loop; k_idx++) {

            offset_a = batch_idx * M * K + m_idx * M0 * K + k_idx * K0;            
            offset_l1a = (m_idx * k_loop + k_idx) * M0 * K0;
            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;

            // *** load matrix A to L1              
            load_matrix_zZ(p_l1_tensor[offset_l1a], p_gm_tensor[offset_a], M0, K0, m_actual, k_actual, K);
        }
    }
}

__aicore__ __inline__ void LoadQMatrix(
    AscendC::GlobalTensor<float> q_gm_tensor,
    int32_t x_len, int32_t y_len)
{   
    int32_t K = y_len * COMPLEX_DOUBLE;
    int32_t N = y_len * COMPLEX_DOUBLE;

    int32_t K0 = 128;
    int32_t N0 = 128;

    if (y_len == 32) {
        
         K0 = 64;
         N0 = 64;
    }

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    int32_t P_Matrix_size = x_len * COMPLEX_DOUBLE * x_len * COMPLEX_DOUBLE * sizeof(T_INPUT);
    AscendC::LocalTensor<float> q_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)(0 + P_Matrix_size));

    int32_t k_loop = (K + K0 - 1) / K0; //2
    int32_t n_loop = (N + N0 - 1) / N0; //2
    
    for (int32_t n_idx = 0; n_idx <  n_loop; n_idx++) {

        int64_t offset_b, offset_l1b;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        for (int32_t k_idx = 0; k_idx < k_loop; k_idx++) {

            offset_b = k_idx * K0 * N + n_idx * N0;

            offset_l1b = (n_idx * k_loop + k_idx) * K0 * N0;
            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;

            // *** load matrix A to L1              
            load_matrix_zZ(q_l1_tensor[offset_l1b], q_gm_tensor[offset_b], K0, N0, k_actual, n_actual, N);
        }
    }
}

__aicore__ __inline__ void Matmul(
    AscendC::GlobalTensor<float> c_gm_tensor,
    AscendC::GlobalTensor<float> a_gm_tensor,
    AscendC::GlobalTensor<float> b_gm_tensor,
    int32_t M, int32_t K, int32_t N, int32_t active_SpRI_flag)
{   
    AscendC::SetLoadDataPaddingValue<float>(0.0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto a_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)0);
    auto b_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)(128 * 1024));
    auto l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, float>((uintptr_t)0);
    auto l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, float>((uintptr_t)0);
    auto l0c_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>((uintptr_t)0);

    int32_t batchSize = 1;
    bool trans_a = false;
    bool trans_b = false;
    int32_t M0 = 128;
    int32_t N0 = 128;
    int32_t K0 = 128;

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
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    
    int32_t ping_flag_fix = 1;

    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {

        auto l0c_tensor = ping_flag_fix ? l0c_base_tensor : l0c_base_tensor[L0C_PINGPONG_BUFFER_LEN];
        auto event_id_fix = ping_flag_fix ? EVENT_ID0 : EVENT_ID1;

        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);

        int64_t m_idx = in_batch_idx / n_loop;
        int64_t n_idx = in_batch_idx % n_loop;

        int64_t offset_a, offset_b;
        int64_t offset_c = batch_idx * M * N + m_idx * M0 * N + n_idx * N0;  // // 修改代码，将矩阵乘结果的dst stride修改为K，之前是N。
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

        auto mte1_mad_ping_flag = 1;

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

            // *** load matrix A to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
            if (trans_a) {
                if (M == 1) {
                    AscendC::DataCopy(
                        a_l1_tensor, a_gm_tensor[offset_a],
                        AscendC::DataCopyParams(
                            1,                                      // nBurst
                            (k_actual + C0_SIZE - 1) / C0_SIZE,     // lenBurst
                            0,                                      // srcGap
                            0                                       // dstGap
                        )
                    );
                } else {
                    load_matrix_zZ(a_l1_tensor, a_gm_tensor[offset_a], K0, M0, k_actual, m_actual, M);
                }
            } else {
                if (m_actual == 1) {
                    AscendC::DataCopy(
                        a_l1_tensor, a_gm_tensor[offset_a],
                        AscendC::DataCopyParams(
                            1,                                      // nBurst
                            (k_actual + C0_SIZE - 1) / C0_SIZE,     // lenBurst
                            0,                                      // srcGap
                            0                                       // dstGap
                        )
                    );
                } else {
                    load_matrix_zZ(a_l1_tensor, a_gm_tensor[offset_a], M0, K0, m_actual, k_actual, K);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);

            // *** load matrix B to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
            if (trans_b) {
                load_matrix_zZ(b_l1_tensor, b_gm_tensor[offset_b], N0, K0, n_actual, k_actual, K);
            } else {
                load_matrix_zZ(b_l1_tensor, b_gm_tensor[offset_b], K0, N0, k_actual, n_actual, N);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);

            // 发信号启动下一轮虚实分离
            if (loop_idx == (loop - 1) && k_idx == (k_loop - 1) && active_SpRI_flag == 1) {
                uint64_t flag_id, config;
                flag_id = SYNC_SINGLE_CV_FLAG_ID;
                AscendC::CrossCoreSetFlag<SYNC_SINGLE_CV_MODE_CODE, PIPE_FIX>(flag_id);
            }

            for (int k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;

                auto l0a_tensor = l0a_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];
                auto l0b_tensor = l0b_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];

                // *** load matrix A from L1 to L0A
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                if (M == 1 || m_actual == 1 && !trans_a) {
                    AscendC::LoadData(
                        l0a_tensor,
                        a_l1_tensor[k_part_idx * k_part_len],
                        AscendC::LoadData2dParams(
                            0,                                                      // baseIdx
                            (k0_round + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,   // repeat
                            1,                                                      // srcStride
                            0,                                                      // sid
                            0,                                                      // dstStride
                            false,                                                  // transpose
                            inc                                                       // addr_cal_mode_t
                        )
                    );
                } else {
                    if (trans_a) {
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * M0];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadDataWithTranspose(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
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
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadData(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * BLOCK_SIZE * K0],
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
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
                }

                // *** load matrix B from L1 to L0B
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);
                }
                if (trans_b) {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                    for (int i = 0; i < k0_round / C0_SIZE; i++) {
                        AscendC::LoadData(
                            l0b_tensor[i * n_round * C0_SIZE],
                            src_b_l1_tensor[i * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(
                                0,                                                      // baseIdx
                                n_round / BLOCK_SIZE,                                   // repeat
                                K0 / C0_SIZE,                                           // srcStride
                                0,                                                      // sid
                                0,                                                      // dstStride
                                false,                                                  // transpose
                                inc                                                     // addr_cal_mode_t
                            )
                        );
                    }
                } else {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * N0];
                    for (int i = 0; i < n_round / BLOCK_SIZE; i++) {
                        AscendC::LoadDataWithTranspose(
                            l0b_tensor[i * CUBE_MATRIX_SIZE],
                            src_b_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
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
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(event_id_fix);
                }

                if (M != 1 && m_actual == 1 && trans_a) {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        16, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(
                        l0c_tensor, l0a_tensor, l0b_tensor, mmParam
                    );
                } else {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        m_actual, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(
                        l0c_tensor, l0a_tensor, l0b_tensor, mmParam
                    );
                }

                AscendC::PipeBarrier<PIPE_M>();
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
            }

            ping_flag = 1 - ping_flag;
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        // copy from L0C to gm 
        // 修改代码，将矩阵乘结果的dst stride修改为K，之前是N。修改为K的原因是因为右矩阵是一个方阵，
        // 完整的DFT的结果就是BatchSize * K(这个K表示的是矩阵乘的K参数，这个K参数等于输入参数n。).
        auto intriParams = AscendC::FixpipeParamsV220(n_actual, // nSize
                                            m_actual, // mSize
                                            m_round,   // srcStride
                                            N,   // dstStride
                                            false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = 0;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(c_gm_tensor[offset_c], l0c_tensor, intriParams);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(event_id_fix);
        ping_flag_fix = 1 - ping_flag_fix;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

}

__aicore__ __inline__ void MatmulFirst(
    AscendC::GlobalTensor<float> c_gm_tensor,
    AscendC::GlobalTensor<float> b_gm_tensor,
    int32_t M, int32_t K, int32_t N,
    int32_t x_len, int32_t y_len)
{   
    AscendC::SetLoadDataPaddingValue<float>(0.0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    int32_t P_Matrix_size = x_len *  COMPLEX_DOUBLE  * x_len *  COMPLEX_DOUBLE  * sizeof(T_INPUT);
    int32_t Q_Matrix_size = y_len *  COMPLEX_DOUBLE  * y_len *  COMPLEX_DOUBLE  * sizeof(T_INPUT);

    auto l1_base_a = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)0); //P矩阵地址
    auto l1_base_b = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(0 + P_Matrix_size + Q_Matrix_size));//第一轮Matmul 右矩阵地址
    auto l0a_base = reinterpret_cast<__ca__ T_INPUT *>((uintptr_t)0);
    auto l0b_base = reinterpret_cast<__cb__ T_INPUT *>((uintptr_t)0);
    auto l0c_base = reinterpret_cast<__cc__ float *>((uintptr_t)0);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto a_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)0);
    auto b_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)(0 + P_Matrix_size + Q_Matrix_size));
    auto l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, float>((uintptr_t)0);
    auto l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, float>((uintptr_t)0);
    auto l0c_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>((uintptr_t)0);

    int32_t batchSize = 1;
    bool trans_a = false;
    bool trans_b = false;
    int32_t M0 = 128;
    int32_t N0 = 128;
    int32_t K0 = 128;

    if (x_len == 32) 
    {
         M0 = 64;
         K0 = 64;
         N0 = 256;
    }

    int32_t m_loop = (M + M0 - 1) / M0;  // 1
    int32_t n_loop = (N + N0 - 1) / N0;   // 1
    int32_t k_loop = (K + K0 - 1) / K0;   // 1
    int32_t loop = batchSize * m_loop * n_loop; // 1

    int32_t ping_flag = 1;

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);

    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    
    int32_t ping_flag_fix = 1;

    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {

        auto l0c_tensor = ping_flag_fix ? l0c_base_tensor : l0c_base_tensor[L0C_PINGPONG_BUFFER_LEN];
        auto event_id_fix = ping_flag_fix ? EVENT_ID0 : EVENT_ID1;

        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);

        int64_t m_idx = in_batch_idx / n_loop;
        int64_t n_idx = in_batch_idx % n_loop;

        int64_t offset_a, offset_b;
        int64_t offset_c = batch_idx * M * N + m_idx * M0 * N + n_idx * N0;  // // 修改代码，将矩阵乘结果的dst stride修改为K，之前是N。
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0; // 128
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0; // 128
        int32_t m_round = (m_actual + 15) / 16 * 16; // 128
        int32_t n_round = (n_actual + 15) / 16 * 16; // 128

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16; // 64

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

        auto mte1_mad_ping_flag = 1;

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

            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0; //128
            int32_t k_round = (k_actual + 15) / 16 * 16; //128
            int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len; //2

            int64_t offset_l1a = (m_idx * k_loop + k_idx)* M0 * K0;
            auto a_l1_tensor = a_base_l1_tensor[offset_l1a];

            int64_t l1_pingpong_buffer_len_real = ((512 * 1024)-(P_Matrix_size + Q_Matrix_size)) / 2 / sizeof(T_INPUT);
            auto b_l1_tensor = ping_flag ? b_base_l1_tensor : b_base_l1_tensor[l1_pingpong_buffer_len_real];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            // *** load matrix A to L1(omitted)
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);

            // *** load matrix B to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
            if (trans_b) {
                load_matrix_zZ(b_l1_tensor, b_gm_tensor[offset_b], N0, K0, n_actual, k_actual, K);
            } else {
                load_matrix_zZ(b_l1_tensor, b_gm_tensor[offset_b], K0, N0, k_actual, n_actual, N);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);
            
            if (loop_idx == (loop - 1) && k_idx == (k_loop - 1))  {
                 // 发信号启动下一轮虚实分离
                uint64_t flag_id, config;
                flag_id = SYNC_SINGLE_CV_FLAG_ID;
                AscendC::CrossCoreSetFlag<SYNC_SINGLE_CV_MODE_CODE, PIPE_FIX>(flag_id);
            }

            for (int k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto l0a_tensor = l0a_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];
                auto l0b_tensor = l0b_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];

                // *** load matrix A from L1 to L0A
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                if (M == 1 || m_actual == 1 && !trans_a) {
                    AscendC::LoadData(
                        l0a_tensor,
                        a_l1_tensor[k_part_idx * k_part_len],
                        AscendC::LoadData2dParams(
                            0,                                                      // baseIdx
                            (k0_round + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,   // repeat
                            1,                                                      // srcStride
                            0,                                                      // sid
                            0,                                                      // dstStride
                            false,                                                  // transpose
                            inc                                                       // addr_cal_mode_t
                        )
                    );
                } else {
                    if (trans_a) {
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * M0];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadDataWithTranspose(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
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
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadData(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * BLOCK_SIZE * K0],
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
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
                }

                // *** load matrix B from L1 to L0B
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);
                }
                if (trans_b) {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                    for (int i = 0; i < k0_round / C0_SIZE; i++) {
                        AscendC::LoadData(
                            l0b_tensor[i * n_round * C0_SIZE],
                            src_b_l1_tensor[i * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(
                                0,                                                      // baseIdx
                                n_round / BLOCK_SIZE,                                   // repeat
                                K0 / C0_SIZE,                                           // srcStride
                                0,                                                      // sid
                                0,                                                      // dstStride
                                false,                                                  // transpose
                                inc                                                     // addr_cal_mode_t
                            )
                        );
                    }
                } else {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * N0];
                    for (int i = 0; i < n_round / BLOCK_SIZE; i++) {
                        AscendC::LoadDataWithTranspose(
                            l0b_tensor[i * CUBE_MATRIX_SIZE],
                            src_b_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
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
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(event_id_fix);
                }

                if (M != 1 && m_actual == 1 && trans_a) {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        16, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(
                        l0c_tensor, l0a_tensor, l0b_tensor, mmParam
                    );
                } else {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        m_actual, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(
                        l0c_tensor, l0a_tensor, l0b_tensor, mmParam
                    );
                }

                AscendC::PipeBarrier<PIPE_M>();
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
            }

            ping_flag = 1 - ping_flag;
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        // copy from L0C to gm 
        // 修改代码，将矩阵乘结果的dst stride修改为K，之前是N。修改为K的原因是因为右矩阵是一个方阵，
        // 完整的DFT的结果就是BatchSize * K(这个K表示的是矩阵乘的K参数，这个K参数等于输入参数n。).
        auto intriParams = AscendC::FixpipeParamsV220(n_actual, // nSize
                                            m_actual, // mSize
                                            m_round,   // srcStride
                                            N,   // dstStride
                                            false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = 0;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(c_gm_tensor[offset_c], l0c_tensor, intriParams);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(event_id_fix);
        ping_flag_fix = 1 - ping_flag_fix;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

}

__aicore__ __inline__ void MatmulSecond(
    AscendC::GlobalTensor<float> c_gm_tensor,
    AscendC::GlobalTensor<float> a_gm_tensor,
    // __gm__ float * __restrict__ gm_b,
    int32_t M, int32_t K, int32_t N,
    int32_t x_len, int32_t y_len)
{   
    AscendC::SetLoadDataPaddingValue<float>(0.0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    int32_t P_Matrix_size = x_len * COMPLEX_DOUBLE * x_len * COMPLEX_DOUBLE * sizeof(T_INPUT);
    int32_t Q_Matrix_size = y_len * COMPLEX_DOUBLE * y_len * COMPLEX_DOUBLE * sizeof(T_INPUT);

    auto l1_base_a = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(0 + P_Matrix_size + Q_Matrix_size));
    auto l1_base_b = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(0 + P_Matrix_size));   
    auto l0a_base = reinterpret_cast<__ca__ T_INPUT *>((uintptr_t)0);
    auto l0b_base = reinterpret_cast<__cb__ T_INPUT *>((uintptr_t)0);
    auto l0c_base = reinterpret_cast<__cc__ float *>((uintptr_t)0);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto a_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)(0 + P_Matrix_size + Q_Matrix_size));
    auto b_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)(0 + P_Matrix_size));
    auto l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, float>((uintptr_t)0);
    auto l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, float>((uintptr_t)0);
    auto l0c_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>((uintptr_t)0);

    int32_t batchSize = 1;
    bool trans_a = false;
    bool trans_b = false;
    int32_t M0 = 128;
    int32_t N0 = 128;
    int32_t K0 = 128;

    if (y_len == 32) {

        M0 = 256;
        N0 = 64;
        K0 = 64;
        
    }

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
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    
    int32_t ping_flag_fix = 1;

    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {
        auto l0c_tensor = ping_flag_fix ? l0c_base_tensor : l0c_base_tensor[L0C_PINGPONG_BUFFER_LEN];
        auto event_id_fix = ping_flag_fix ? EVENT_ID0 : EVENT_ID1;

        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);

        int64_t m_idx = in_batch_idx / n_loop;
        int64_t n_idx = in_batch_idx % n_loop;

        int64_t offset_a, offset_b;
        int64_t offset_c = batch_idx * M * N + m_idx * M0 * N + n_idx * N0;  // // 修改代码，将矩阵乘结果的dst stride修改为K，之前是N。
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

        auto mte1_mad_ping_flag = 1;

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

            int64_t offset_l1b = (n_idx * k_loop + k_idx)* K0 * N0;
            auto b_l1_tensor = b_base_l1_tensor[offset_l1b];

            int64_t l1_pingpong_buffer_len_real = ((512 * 1024)-(P_Matrix_size + Q_Matrix_size)) / 2 / sizeof(T_INPUT);
            auto a_l1_tensor = ping_flag ? a_base_l1_tensor : a_base_l1_tensor[l1_pingpong_buffer_len_real];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
            

            // *** load matrix A to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
            if (trans_a) {
                if (M == 1) {
                    AscendC::DataCopy(
                        a_l1_tensor, a_gm_tensor[offset_a],
                        AscendC::DataCopyParams(
                            1,                                      // nBurst
                            (k_actual + C0_SIZE - 1) / C0_SIZE,     // lenBurst
                            0,                                      // srcGap
                            0                                       // dstGap
                        )
                    );
                } else {
                    load_matrix_zZ(a_l1_tensor, a_gm_tensor[offset_a], K0, M0, k_actual, m_actual, M);
                }
            } else {
                if (m_actual == 1) {
                    AscendC::DataCopy(
                        a_l1_tensor, a_gm_tensor[offset_a],
                        AscendC::DataCopyParams(
                            1,                                      // nBurst
                            (k_actual + C0_SIZE - 1) / C0_SIZE,     // lenBurst
                            0,                                      // srcGap
                            0                                       // dstGap
                        )
                    );
                } else {
                    load_matrix_zZ(a_l1_tensor, a_gm_tensor[offset_a], M0, K0, m_actual, k_actual, K);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);

            // *** load matrix B to L1 (omitted)
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);

            for (int k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto l0a_tensor = l0a_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];
                auto l0b_tensor = l0b_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];

                // *** load matrix A from L1 to L0A
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                if (M == 1 || m_actual == 1 && !trans_a) {
                    AscendC::LoadData(
                        l0a_tensor,
                        a_l1_tensor[k_part_idx * k_part_len],
                        AscendC::LoadData2dParams(
                            0,                                                      // baseIdx
                            (k0_round + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,   // repeat
                            1,                                                      // srcStride
                            0,                                                      // sid
                            0,                                                      // dstStride
                            false,                                                  // transpose
                            inc                                                       // addr_cal_mode_t
                        )
                    );
                } else {
                    if (trans_a) {
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * M0];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadDataWithTranspose(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
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
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadData(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * BLOCK_SIZE * K0],
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
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
                }

                // *** load matrix B from L1 to L0B
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);
                }
                if (trans_b) {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                    for (int i = 0; i < k0_round / C0_SIZE; i++) {
                        AscendC::LoadData(
                            l0b_tensor[i * n_round * C0_SIZE],
                            src_b_l1_tensor[i * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(
                                0,                                                      // baseIdx
                                n_round / BLOCK_SIZE,                                   // repeat
                                K0 / C0_SIZE,                                           // srcStride
                                0,                                                      // sid
                                0,                                                      // dstStride
                                false,                                                  // transpose
                                inc                                                     // addr_cal_mode_t
                            )
                        );
                    }
                } else {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * N0];
                    for (int i = 0; i < n_round / BLOCK_SIZE; i++) {
                        AscendC::LoadDataWithTranspose(
                            l0b_tensor[i * CUBE_MATRIX_SIZE],
                            src_b_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
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
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(event_id_fix);
                }

                if (M != 1 && m_actual == 1 && trans_a) {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        16, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(
                        l0c_tensor, l0a_tensor, l0b_tensor, mmParam
                    );
                } else {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        m_actual, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(
                        l0c_tensor, l0a_tensor, l0b_tensor, mmParam
                    );
                }

                AscendC::PipeBarrier<PIPE_M>();
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
            }

            ping_flag = 1 - ping_flag;
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        // copy from L0C to gm 
        // 修改代码，将矩阵乘结果的dst stride修改为K，之前是N。修改为K的原因是因为右矩阵是一个方阵，
        // 完整的DFT的结果就是BatchSize * K(这个K表示的是矩阵乘的K参数，这个K参数等于输入参数n。).
        auto intriParams = AscendC::FixpipeParamsV220(n_actual, // nSize
                                            m_actual, // mSize
                                            m_round,   // srcStride
                                            N,   // dstStride
                                            false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = 0;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(c_gm_tensor[offset_c], l0c_tensor, intriParams);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(event_id_fix);
        ping_flag_fix = 1 - ping_flag_fix;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
}

__aicore__ __inline__ void MatmulWithSpecialTransfer(
    AscendC::GlobalTensor<float> c_gm_tensor,
    AscendC::GlobalTensor<float> b_gm_tensor,
     int32_t M, int32_t K, int32_t N,
     int32_t x_len, int32_t y_len, int32_t current_batch_nums)
{   
    AscendC::SetLoadDataPaddingValue<float>(0.0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    int32_t P_Matrix_size = x_len * COMPLEX_DOUBLE * x_len * COMPLEX_DOUBLE * sizeof(T_INPUT);
    int32_t Q_Matrix_size = y_len * COMPLEX_DOUBLE* y_len * COMPLEX_DOUBLE * sizeof(T_INPUT);

    auto l1_base_a = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)0); 
    auto l1_base_b = reinterpret_cast<__cbuf__ T_INPUT *>((uintptr_t)(0 + P_Matrix_size + Q_Matrix_size));  // 128 KB
    auto l0a_base = reinterpret_cast<__ca__ T_INPUT *>((uintptr_t)0);
    auto l0b_base = reinterpret_cast<__cb__ T_INPUT *>((uintptr_t)0);
    auto l0c_base = reinterpret_cast<__cc__ float *>((uintptr_t)0);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto a_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)0);
    auto b_base_l1_tensor = buf.GetBuffer<BufferType::ASCEND_CB, float>((uintptr_t)(0 + P_Matrix_size + Q_Matrix_size));
    auto l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, float>((uintptr_t)0);
    auto l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, float>((uintptr_t)0);
    auto l0c_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>((uintptr_t)0);

    int32_t batchSize = 1;
    bool trans_a = false;
    bool trans_b = false;
    int32_t M0 = 128;
    int32_t N0 = 128;
    int32_t K0 = 128;

    if (x_len == 32) 
    {
         M0 = 64;
         K0 = 64;
         N0 = 256;
    }

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
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    
    int32_t ping_flag_fix = 1;

    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {

        auto l0c_tensor = ping_flag_fix ? l0c_base_tensor : l0c_base_tensor[L0C_PINGPONG_BUFFER_LEN];
        auto event_id_fix = ping_flag_fix ? EVENT_ID0 : EVENT_ID1;

        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int64_t in_batch_idx = loop_idx % (m_loop * n_loop);

        int64_t m_idx = in_batch_idx / n_loop;
        int64_t n_idx = in_batch_idx % n_loop;

        int64_t offset_a, offset_b;
        int64_t offset_c = batch_idx * M * N + m_idx * M0 * N + n_idx * N0;  // // 修改代码，将矩阵乘结果的dst stride修改为K，之前是N。
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

        auto mte1_mad_ping_flag = 1;

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

            int64_t offset_l1a = (m_idx * k_loop + k_idx)* M0 * K0;
            auto a_l1_tensor = a_base_l1_tensor[offset_l1a];

            int64_t l1_pingpong_buffer_len_real = ((512 * 1024)-(P_Matrix_size + Q_Matrix_size)) / 2 / sizeof(T_INPUT);
            auto b_l1_tensor = ping_flag ? b_base_l1_tensor : b_base_l1_tensor[l1_pingpong_buffer_len_real];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            // *** load matrix A to L1 (omitted)
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);

            // *** load matrix B to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
            if (trans_b) {
                load_matrix_zZ(b_l1_tensor, b_gm_tensor[offset_b], N0, K0, n_actual, k_actual, K);
            } else {
                load_matrix_zZ(b_l1_tensor, b_gm_tensor[offset_b], K0, N0, k_actual, n_actual, N);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);

            if (loop_idx == (loop - 1) && k_idx == (k_loop - 1))  {
                 // 发信号启动下一轮虚实分离
                uint64_t flag_id, config;
                flag_id = SYNC_SINGLE_CV_FLAG_ID;
                AscendC::CrossCoreSetFlag<SYNC_SINGLE_CV_MODE_CODE, PIPE_FIX>(flag_id);
            }

            for (int k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto l0a_tensor = l0a_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];
                auto l0b_tensor = l0b_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];

                // *** load matrix A from L1 to L0A
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id);
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                if (M == 1 || m_actual == 1 && !trans_a) {
                    AscendC::LoadData(
                        l0a_tensor,
                        a_l1_tensor[k_part_idx * k_part_len],
                        AscendC::LoadData2dParams(
                            0,                                                      // baseIdx
                            (k0_round + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,   // repeat
                            1,                                                      // srcStride
                            0,                                                      // sid
                            0,                                                      // dstStride
                            false,                                                  // transpose
                            inc                                                       // addr_cal_mode_t
                        )
                    );
                } else {
                    if (trans_a) {
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * M0];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadDataWithTranspose(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
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
                        auto src_a_l1_tensor = a_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                        for (int i = 0; i < m_round / BLOCK_SIZE; i++) {
                            AscendC::LoadData(
                                l0a_tensor[i * k0_round * BLOCK_SIZE],
                                src_a_l1_tensor[i * BLOCK_SIZE * K0],
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
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id);
                }

                // *** load matrix B from L1 to L0B
                if (k_part_idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_id + 2);
                }
                if (trans_b) {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * BLOCK_SIZE];
                    for (int i = 0; i < k0_round / C0_SIZE; i++) {
                        AscendC::LoadData(
                            l0b_tensor[i * n_round * C0_SIZE],
                            src_b_l1_tensor[i * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(
                                0,                                                      // baseIdx
                                n_round / BLOCK_SIZE,                                   // repeat
                                K0 / C0_SIZE,                                           // srcStride
                                0,                                                      // sid
                                0,                                                      // dstStride
                                false,                                                  // transpose
                                inc                                                     // addr_cal_mode_t
                            )
                        );
                    }
                } else {
                    auto src_b_l1_tensor = b_l1_tensor[k_part_idx * k_part_len * N0];
                    for (int i = 0; i < n_round / BLOCK_SIZE; i++) {
                        AscendC::LoadDataWithTranspose(
                            l0b_tensor[i * CUBE_MATRIX_SIZE],
                            src_b_l1_tensor[i * 2 * CUBE_MATRIX_SIZE],
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
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event_id + 2);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(event_id_fix);
                }

                if (M != 1 && m_actual == 1 && trans_a) {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        16, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(
                        l0c_tensor, l0a_tensor, l0b_tensor, mmParam
                    );
                } else {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        m_actual, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;
                    mmParam.unitFlag = 0;
                    AscendC::Mmad(
                        l0c_tensor, l0a_tensor, l0b_tensor, mmParam
                    );
                }

                AscendC::PipeBarrier<PIPE_M>();
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1_mad_event_id);
                mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
            }

            ping_flag = 1 - ping_flag;
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        // copy from L0C to gm 
        //实现特殊搬运，多batch搬运完可直接乘矩阵Q
        if (x_len == 128 && y_len == 32){
            int64_t offset_c_new = m_idx  * x_len * y_len;
            for (int i = 0; i < current_batch_nums; i++){
                auto intriParams = AscendC::FixpipeParamsV220(y_len, // nSize
                                                    m_actual, // mSize
                                                    m_round,   // srcStride
                                                    y_len,   // dstStride
                                                    false);      // enRelu

                intriParams.quantPre = QuantMode_t::NoQuant;
                intriParams.unitFlag = 0;
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(c_gm_tensor[offset_c_new + i * x_len * 2 * y_len], l0c_tensor[i * x_len * y_len], intriParams);
            }
        }else {
            for (int i = 0; i < current_batch_nums; i++){
                auto intriParams = AscendC::FixpipeParamsV220(y_len, // nSize
                                                    x_len * 2, // mSize
                                                    m_round,   // srcStride
                                                    y_len,   // dstStride
                                                    false);      // enRelu

                intriParams.quantPre = QuantMode_t::NoQuant;
                intriParams.unitFlag = 0;
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(c_gm_tensor[offset_c + i * x_len * 2 * y_len], l0c_tensor[i * x_len * 2 * y_len], intriParams);
            }
        }
        
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(event_id_fix);
        ping_flag_fix = 1 - ping_flag_fix;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
}
#endif

extern "C" __global__ __aicore__ void dd(
    __gm__ uint8_t *__restrict__ ffts_addr,
    __gm__ float *__restrict__ gm_input,
    __gm__ float *__restrict__ gm_P_matrix,
    __gm__ float *__restrict__ gm_Q_matrix,
    __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace,
    __gm__ uint8_t *__restrict__ tiling_para_gm) 
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_VEC__
{
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    const int32_t batch_size = tiling_para[0];
    const int32_t dft_xlength = tiling_para[1];
    const int32_t dft_ylength = tiling_para[2];
    const int32_t batch_nums_per_loop = tiling_para[3];

    const int32_t loop_count = CeilADivB(batch_size, batch_nums_per_loop);
    const int32_t batch_remain = batch_size % batch_nums_per_loop;
    const int32_t l2_cache_loop_num = 500;
    const int32_t big_loop_count = CeilADivB(loop_count, l2_cache_loop_num);
    const int32_t big_loop_remain = loop_count % l2_cache_loop_num;

    uint64_t flag_id = SYNC_ALL_CORE_FLAG_ID;

    __gm__ float *__restrict__ gm_input_current;
    __gm__ float *__restrict__ gm_workspace_current;
    __gm__ float *__restrict__ gm_output_current;
    int32_t current_batch_nums;
    int32_t current_loop_nums;

    AscendC::GlobalTensor<float> input_current_gm_tensor;
    AscendC::GlobalTensor<float> workspace_current_gm_tensor;
    AscendC::GlobalTensor<float> output_current_gm_tensor;

    if (loop_count <= l2_cache_loop_num) {
        for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx  ++){
            if ((loop_idx % AscendC::GetBlockNum()) != (AscendC::GetBlockIdx() / 2)) {
                continue;
            }
            if ((loop_idx == loop_count - 1) && (batch_remain > 0)){
                current_batch_nums = batch_remain;
            }else {
                current_batch_nums = batch_nums_per_loop;
            }

            input_current_gm_tensor.SetGlobalBuffer(gm_input + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            workspace_current_gm_tensor.SetGlobalBuffer(gm_workspace + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            output_current_gm_tensor.SetGlobalBuffer(gm_output + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));

            if (current_batch_nums == 1) {
                SeparateRITWO(output_current_gm_tensor, input_current_gm_tensor, dft_xlength, dft_ylength);
            }else {
                SeparateRITWOMultiBatch(output_current_gm_tensor, input_current_gm_tensor, dft_xlength, dft_ylength, current_batch_nums);
            }

            // 发信号启动矩阵乘迭代
            flag_id = SYNC_SINGLE_VC_FLAG_ID;
            AscendC::CrossCoreSetFlag<SYNC_SINGLE_CV_MODE_CODE, PIPE_MTE3>(flag_id);

            flag_id = SYNC_SINGLE_CV_FLAG_ID;
            AscendC::CrossCoreWaitFlag(flag_id); // 等待上一轮计算结束
        }
    }else {
        //先执行后几轮大循环
        for (int big_loop_idx = 1; big_loop_idx < big_loop_count; big_loop_idx++) {

            if ((big_loop_idx == big_loop_count - 1) && (big_loop_remain > 0)){
                current_loop_nums = big_loop_remain;
            }else {
                current_loop_nums = l2_cache_loop_num;
            }

            for (int32_t loop_idx = 0; loop_idx < current_loop_nums; loop_idx  ++){
            if ((loop_idx % AscendC::GetBlockNum()) != (AscendC::GetBlockIdx() / 2)) {
                continue;
            }
            if ((big_loop_idx == big_loop_count - 1) && (loop_idx == current_loop_nums - 1) && (batch_remain > 0)){
                current_batch_nums = batch_remain;
            }else {
                current_batch_nums = batch_nums_per_loop;
            }

            input_current_gm_tensor.SetGlobalBuffer(gm_input + (big_loop_idx * l2_cache_loop_num + loop_idx) * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            workspace_current_gm_tensor.SetGlobalBuffer(gm_workspace + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            output_current_gm_tensor.SetGlobalBuffer(gm_output + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));

            if (current_batch_nums == 1) {
                SeparateRITWO(output_current_gm_tensor, input_current_gm_tensor, dft_xlength, dft_ylength);
            }else {
                SeparateRITWOMultiBatch(output_current_gm_tensor, input_current_gm_tensor, dft_xlength, dft_ylength, current_batch_nums);
            }

                // 发信号启动矩阵乘迭代
            flag_id = SYNC_SINGLE_VC_FLAG_ID;
            AscendC::CrossCoreSetFlag<SYNC_SINGLE_CV_MODE_CODE, PIPE_MTE3>(flag_id); 

            flag_id = SYNC_SINGLE_CV_FLAG_ID;
            AscendC::CrossCoreWaitFlag(flag_id); // 等待上一轮计算结束
        }
        }
        //再执行第一轮大循环
        for (int32_t loop_idx = 0; loop_idx < l2_cache_loop_num; loop_idx  ++){
            if ((loop_idx % AscendC::GetBlockNum()) != (AscendC::GetBlockIdx() / 2)) {
                continue;
            }
            if ((loop_idx == loop_count - 1) && (batch_remain > 0)){
                current_batch_nums = batch_remain;
            }else {
                current_batch_nums = batch_nums_per_loop;
            }

            input_current_gm_tensor.SetGlobalBuffer(gm_input + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            workspace_current_gm_tensor.SetGlobalBuffer(gm_workspace + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            output_current_gm_tensor.SetGlobalBuffer(gm_output + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));

            if (current_batch_nums == 1) {
                SeparateRITWO(output_current_gm_tensor, input_current_gm_tensor, dft_xlength, dft_ylength);
            }else {
                SeparateRITWOMultiBatch(output_current_gm_tensor, input_current_gm_tensor, dft_xlength, dft_ylength, current_batch_nums);
            }

                // 发信号启动矩阵乘迭代
            flag_id = SYNC_SINGLE_VC_FLAG_ID;
            AscendC::CrossCoreSetFlag<SYNC_SINGLE_CV_MODE_CODE, PIPE_MTE3>(flag_id);

            flag_id = SYNC_SINGLE_CV_FLAG_ID;
            AscendC::CrossCoreWaitFlag(flag_id); // 等待上一轮计算结束
        }

    }
}
#elif __DAV_C220_CUBE__
{
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
    const int32_t batch_size = tiling_para[0];
    const int32_t dft_xlength = tiling_para[1];
    const int32_t dft_ylength = tiling_para[2];
    const int32_t batch_nums_per_loop = tiling_para[3];

    const int32_t loop_count = CeilADivB(batch_size, batch_nums_per_loop);
    const int32_t batch_remain = batch_size % batch_nums_per_loop;
    const int32_t l2_cache_loop_num = 500;
    const int32_t big_loop_count = CeilADivB(loop_count, l2_cache_loop_num);
    const int32_t big_loop_remain = loop_count % l2_cache_loop_num;

    uint64_t flag_id, config;

    __gm__ float *__restrict__ gm_input_current;
    __gm__ float *__restrict__ gm_workspace_current;
    __gm__ float *__restrict__ gm_output_current;

    AscendC::GlobalTensor<float> input_current_gm_tensor;
    AscendC::GlobalTensor<float> workspace_current_gm_tensor;
    AscendC::GlobalTensor<float> output_current_gm_tensor;

    AscendC::GlobalTensor<float> P_matrix_gm_tensor;
    P_matrix_gm_tensor.SetGlobalBuffer(gm_P_matrix);
    AscendC::GlobalTensor<float> Q_matrix_gm_tensor;
    Q_matrix_gm_tensor.SetGlobalBuffer(gm_Q_matrix);

    int32_t current_batch_nums;
    int32_t active_SpRI_flag;
    int32_t current_loop_nums;
    if (loop_count <= l2_cache_loop_num) {
        for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx  ++){
            if (loop_idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                continue;
            }
            if ((loop_idx == loop_count - 1) && (batch_remain > 0)){
                current_batch_nums = batch_remain;
            }else {
                current_batch_nums = batch_nums_per_loop;
            }

            input_current_gm_tensor.SetGlobalBuffer(gm_input + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            workspace_current_gm_tensor.SetGlobalBuffer(gm_workspace + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            output_current_gm_tensor.SetGlobalBuffer(gm_output + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));

            if (dft_xlength == 128 && dft_ylength == 128) { // shape(128,128)无法做优化
                flag_id = SYNC_SINGLE_VC_FLAG_ID;
                AscendC::CrossCoreWaitFlag(flag_id); // 等待虚实分离计算结束

                active_SpRI_flag = 1;
                Matmul(workspace_current_gm_tensor, P_matrix_gm_tensor, output_current_gm_tensor, dft_xlength * COMPLEX_DOUBLE, dft_xlength * COMPLEX_DOUBLE, dft_ylength, active_SpRI_flag); 
            
                
            } else {
                LoadPMatrix(P_matrix_gm_tensor, dft_xlength, dft_ylength);
                LoadQMatrix(Q_matrix_gm_tensor, dft_xlength, dft_ylength);

                flag_id = SYNC_SINGLE_VC_FLAG_ID;
                AscendC::CrossCoreWaitFlag(flag_id); // 等待虚实分离计算结束

                if (current_batch_nums == 1) {
                    MatmulFirst(workspace_current_gm_tensor, output_current_gm_tensor, dft_xlength * COMPLEX_DOUBLE, dft_xlength * COMPLEX_DOUBLE, dft_ylength, dft_xlength, dft_ylength);
                } else {
                    MatmulWithSpecialTransfer(workspace_current_gm_tensor, output_current_gm_tensor, 
                    dft_xlength * COMPLEX_DOUBLE, dft_xlength * COMPLEX_DOUBLE, dft_ylength * current_batch_nums, dft_xlength, dft_ylength, current_batch_nums);
                }
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);

        for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx  ++){
            if (loop_idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                continue;
            }
            if ((loop_idx == loop_count - 1) && (batch_remain > 0)){
                current_batch_nums = batch_remain;
            }else {
                current_batch_nums = batch_nums_per_loop;
            }

            input_current_gm_tensor.SetGlobalBuffer(gm_input + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            workspace_current_gm_tensor.SetGlobalBuffer(gm_workspace + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            output_current_gm_tensor.SetGlobalBuffer(gm_output + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));

            if (dft_xlength == 128 && dft_ylength == 128) { // shape(128,128)无法做优化
                active_SpRI_flag = 0;
                Matmul(output_current_gm_tensor, workspace_current_gm_tensor, Q_matrix_gm_tensor, dft_xlength * current_batch_nums, dft_ylength * COMPLEX_DOUBLE, dft_ylength * COMPLEX_DOUBLE, active_SpRI_flag);
            }else {
                MatmulSecond(output_current_gm_tensor, workspace_current_gm_tensor, 
            dft_xlength * current_batch_nums, dft_ylength * COMPLEX_DOUBLE, dft_ylength * COMPLEX_DOUBLE, dft_xlength, dft_ylength);
            }
            
        }
    }else {
         //先执行后几轮大循环
        for (int big_loop_idx = 1; big_loop_idx < big_loop_count; big_loop_idx++) {

            if ((big_loop_idx == big_loop_count - 1) && (big_loop_remain > 0)){
                current_loop_nums = big_loop_remain;
            }else {
                current_loop_nums = l2_cache_loop_num;
            }

            for (int32_t loop_idx = 0; loop_idx < current_loop_nums; loop_idx  ++){
            if (loop_idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                continue;
            }
            if ((big_loop_idx == big_loop_count - 1) && (loop_idx == current_loop_nums - 1) && (batch_remain > 0)){
                current_batch_nums = batch_remain;
            }else {
                current_batch_nums = batch_nums_per_loop;
            }

            workspace_current_gm_tensor.SetGlobalBuffer(gm_workspace + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            output_current_gm_tensor.SetGlobalBuffer(gm_output + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));

            if (dft_xlength == 128 && dft_ylength == 128) { // shape(128,128)无法做优化
                flag_id = SYNC_SINGLE_VC_FLAG_ID;
                AscendC::CrossCoreWaitFlag(flag_id); // 等待虚实分离计算结束

                active_SpRI_flag = 1;
                Matmul(workspace_current_gm_tensor, P_matrix_gm_tensor, output_current_gm_tensor, dft_xlength * COMPLEX_DOUBLE, dft_xlength * COMPLEX_DOUBLE, dft_ylength, active_SpRI_flag); 
            
                
            } else {
                LoadPMatrix(P_matrix_gm_tensor, dft_xlength, dft_ylength);
                LoadQMatrix(Q_matrix_gm_tensor, dft_xlength, dft_ylength);

                flag_id = SYNC_SINGLE_VC_FLAG_ID;
                AscendC::CrossCoreWaitFlag(flag_id); // 等待虚实分离计算结束

                if (current_batch_nums == 1) {
                    MatmulFirst(workspace_current_gm_tensor, output_current_gm_tensor, dft_xlength * COMPLEX_DOUBLE, dft_xlength * COMPLEX_DOUBLE, dft_ylength, dft_xlength, dft_ylength);
                } else {
                    MatmulWithSpecialTransfer(workspace_current_gm_tensor, output_current_gm_tensor, 
                    dft_xlength * COMPLEX_DOUBLE, dft_xlength * COMPLEX_DOUBLE, dft_ylength * current_batch_nums, dft_xlength, dft_ylength, current_batch_nums);
                }
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);

        for (int32_t loop_idx = 0; loop_idx < current_loop_nums; loop_idx  ++){
            if (loop_idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                continue;
            }
            if ((big_loop_idx == big_loop_count - 1) && (loop_idx == current_loop_nums - 1) && (batch_remain > 0)){
                current_batch_nums = batch_remain;
            }else {
                current_batch_nums = batch_nums_per_loop;
            }

            workspace_current_gm_tensor.SetGlobalBuffer(gm_workspace + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            output_current_gm_tensor.SetGlobalBuffer(gm_output + (big_loop_idx * l2_cache_loop_num + loop_idx) * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));

            if (dft_xlength == 128 && dft_ylength == 128) { // shape(128,128)无法做优化
                active_SpRI_flag = 0;
                Matmul(output_current_gm_tensor, workspace_current_gm_tensor, Q_matrix_gm_tensor, dft_xlength * current_batch_nums, dft_ylength * COMPLEX_DOUBLE, dft_ylength * COMPLEX_DOUBLE, active_SpRI_flag);
            }else {
                MatmulSecond(output_current_gm_tensor, workspace_current_gm_tensor, 
            dft_xlength * current_batch_nums, dft_ylength * COMPLEX_DOUBLE, dft_ylength * COMPLEX_DOUBLE, dft_xlength, dft_ylength);
            }
            
        }
        } 
        //再执行第一轮大循环
         for (int32_t loop_idx = 0; loop_idx < l2_cache_loop_num; loop_idx  ++){
            if (loop_idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                continue;
            }
            if ((loop_idx == loop_count - 1) && (batch_remain > 0)){
                current_batch_nums = batch_remain;
            }else {
                current_batch_nums = batch_nums_per_loop;
            }

            input_current_gm_tensor.SetGlobalBuffer(gm_input + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            workspace_current_gm_tensor.SetGlobalBuffer(gm_workspace + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            output_current_gm_tensor.SetGlobalBuffer(gm_output + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));

            if (dft_xlength == 128 && dft_ylength == 128) { // shape(128,128)无法做优化
                flag_id = SYNC_SINGLE_VC_FLAG_ID;
                AscendC::CrossCoreWaitFlag(flag_id); // 等待虚实分离计算结束

                active_SpRI_flag = 1;
                Matmul(workspace_current_gm_tensor, P_matrix_gm_tensor, output_current_gm_tensor, dft_xlength * COMPLEX_DOUBLE, dft_xlength * COMPLEX_DOUBLE, dft_ylength, active_SpRI_flag); 
            
                
            } else {
                LoadPMatrix(P_matrix_gm_tensor, dft_xlength, dft_ylength);
                LoadQMatrix(Q_matrix_gm_tensor, dft_xlength, dft_ylength);

                flag_id = SYNC_SINGLE_VC_FLAG_ID;
                AscendC::CrossCoreWaitFlag(flag_id); // 等待虚实分离计算结束

                if (current_batch_nums == 1) {
                    MatmulFirst(workspace_current_gm_tensor, output_current_gm_tensor, dft_xlength * COMPLEX_DOUBLE, dft_xlength * COMPLEX_DOUBLE, dft_ylength, dft_xlength, dft_ylength);
                } else {
                    MatmulWithSpecialTransfer(workspace_current_gm_tensor, output_current_gm_tensor, 
                    dft_xlength * COMPLEX_DOUBLE, dft_xlength * COMPLEX_DOUBLE, dft_ylength * current_batch_nums, dft_xlength, dft_ylength, current_batch_nums);
                }
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);

        for (int32_t loop_idx = 0; loop_idx < l2_cache_loop_num; loop_idx  ++){
            if (loop_idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                continue;
            }
            if ((loop_idx == loop_count - 1) && (batch_remain > 0)){
                current_batch_nums = batch_remain;
            }else {
                current_batch_nums = batch_nums_per_loop;
            }

            input_current_gm_tensor.SetGlobalBuffer(gm_input + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            workspace_current_gm_tensor.SetGlobalBuffer(gm_workspace + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));
            output_current_gm_tensor.SetGlobalBuffer(gm_output + loop_idx * batch_nums_per_loop * (dft_xlength * dft_ylength * COMPLEX_DOUBLE));

            if (dft_xlength == 128 && dft_ylength == 128) { // shape(128,128)无法做优化
                active_SpRI_flag = 0;
                Matmul(output_current_gm_tensor, workspace_current_gm_tensor, Q_matrix_gm_tensor, dft_xlength * current_batch_nums, dft_ylength * COMPLEX_DOUBLE, dft_ylength * COMPLEX_DOUBLE, active_SpRI_flag);
            }else {
                MatmulSecond(output_current_gm_tensor, workspace_current_gm_tensor, 
            dft_xlength * current_batch_nums, dft_ylength * COMPLEX_DOUBLE, dft_ylength * COMPLEX_DOUBLE, dft_xlength, dft_ylength);
            }
            
        }  
    }
    
}
#endif
}

} // namespace FFT2DKernel

#endif
