/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT1_D_STRIDE_KERNEL_H
#define FFT1_D_STRIDE_KERNEL_H

#include "kernel/fft_common_kernel.h"

namespace FFT1DStrideKernel {

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#define ASCENDFFT_FORWARD -1  // Forward FFT
#define ASCENDFFT_INVERSE 1   // Inverse FFT

const int64_t BLOCK_SIZE = 32 / sizeof(float);
const int64_t BLOCK_PER_REPEAT = 8;
const int64_t COMPLEX_DOUBLE = 2;
const uint16_t PATTERN_MODE_REPEAT_TWO_GET_FIRST = 1;
const uint16_t PATTERN_MODE_REPEAT_TWO_GET_SECOND = 2;

constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(float);  // 32 KB
constexpr int32_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(float);   // 64 KB
constexpr int32_t R0 = 16;
constexpr int32_t C0_SIZE = 32 / sizeof(float);
constexpr int32_t CUBE_MATRIX_SIZE = R0 * C0_SIZE;                      // 16 * 8
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(float);  // 256 KB

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

/*********************************
 * @brief 从gm指定位置拷入数据至连续ub（使用所有VectorCore）
 * @param [out] ub_dst ub数据块的首地址（float32）
 * @param [in] gm_src 输入数据块的首地址（float32）
 * @param [in] repeat 重复搬运次数
 * @param [in] copyCount 单个数据块长度（float32、假定按block对齐）
 * @param [in] srcDValue 输入每行数据的间隔（假定按block对齐）
*********************************/
__aicore__ __inline__ void LoadDataFromGmToUbDev(
    AscendC::LocalTensor<float>& ub_dst_tensor,
    AscendC::GlobalTensor<float>& gm_src_tensor,
    int32_t repeat, int32_t copyCount, int32_t srcDValue)
{
    auto ub_dst = reinterpret_cast<__ubuf__ float *>((uintptr_t)(ub_dst_tensor.GetPhyAddr()));
    auto gm_src = reinterpret_cast<__gm__ float *>((uintptr_t)(gm_src_tensor.GetPhyAddr()));

    int32_t copy_block_count = copyCount / BLOCK_SIZE;
    int32_t src_gap_block_count = srcDValue / BLOCK_SIZE - copy_block_count;
    if (src_gap_block_count > UINT16_MAX) {
        auto ub_current = ub_dst;
        auto ub_current_tensor = ub_dst_tensor;
        auto gm_current = gm_src;
        auto gm_current_tensor = gm_src_tensor;
        for (int32_t i = 0; i < repeat; i++) {
            AscendC::DataCopy(
                ub_current_tensor,
                gm_current_tensor,
                AscendC::DataCopyParams(
                    1, // blockCount
                    copy_block_count, // blockLen
                    0, // srcStride
                    0 // dstStride
                )
            );
            ub_current += copyCount;
            gm_current += srcDValue;
        }
    } else {
        AscendC::DataCopy(
            ub_dst_tensor,
            gm_src_tensor,
            AscendC::DataCopyParams(
                repeat, // blockCount
                copy_block_count, // blockLen
                src_gap_block_count, // srcStride
                0 // dstStride
            )
        );
    }
}

/*********************************
 * @brief 从ub指定位置搬出连续数据至gm（使用所有VectorCore）
 * @param [out] gm_dst gm输出数据块的首地址（float32）
 * @param [in] ub_src ub源数据块的首地址（float32）
 * @param [in] repeat 重复搬运次数
 * @param [in] copyCount 单个数据块长度（float32、假定按block对齐）
 * @param [in] dstDValue 输入每行数据的间隔（假定按block对齐）
*********************************/
__aicore__ __inline__ void SaveDataFromUbToGmDev(
    AscendC::GlobalTensor<float>& gm_dst_tensor,
    AscendC::LocalTensor<float>& ub_src_tensor,
    int32_t repeat, int32_t copyCount, int32_t dstDValue)
{
    auto gm_dst = reinterpret_cast<__gm__ float *>((uintptr_t)(gm_dst_tensor.GetPhyAddr()));
    auto ub_src = reinterpret_cast<__ubuf__ float *>((uintptr_t)(ub_src_tensor.GetPhyAddr()));

    int32_t copy_block_count = copyCount / BLOCK_SIZE;
    int32_t dst_gap_block_count = dstDValue / BLOCK_SIZE - copy_block_count;

    if (dst_gap_block_count > UINT16_MAX) {
        auto ub_current = ub_src;
        auto ub_current_tensor = ub_src_tensor;
        auto gm_current = gm_dst;
        auto gm_current_tensor = gm_dst_tensor;
        for (int32_t i = 0; i < repeat; i++) {
            AscendC::DataCopy(
                gm_current_tensor,
                ub_current_tensor,
                AscendC::DataCopyParams(
                    1, // blockCount
                    copy_block_count, // blockLen
                    0, // srcStride
                    0 // dstStride
                )
            );
            ub_current += copyCount;
            gm_current += dstDValue;
        }
    } else {
        AscendC::DataCopy(
            gm_dst_tensor,
            ub_src_tensor,
            AscendC::DataCopyParams(
                repeat, // blockCount
                copy_block_count, // blockLen
                0, // srcStride
                dst_gap_block_count // dstStride
            )
        );
    }
}

// (n,d,s_0,2)->(2,n,s_0)
__aicore__ __inline__ void SeparateRI(__gm__ float *__restrict__ gm_dst, __gm__ float *__restrict__ gm_src,
                                      int32_t nValue, int32_t dValue, int32_t srcDValue)
{
    // 除0整改
    if (dValue == 0) {
        dValue = 1;
    }

    AscendC::GlobalTensor<float> gm_src_tensor;
    AscendC::GlobalTensor<float> gm_dst_tensor;

    gm_src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_src));
    gm_dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_dst));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> complex_buf1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> real_buf1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(48 * 1024));
    AscendC::LocalTensor<float> imag_buf1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(72 * 1024));
    AscendC::LocalTensor<float> complex_buf2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(96 * 1024));
    AscendC::LocalTensor<float> real_buf2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(144 * 1024));
    AscendC::LocalTensor<float> imag_buf2_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(168 * 1024));

    auto complex_buf1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(complex_buf1_tensor.GetPhyAddr()));
    auto real_buf1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(real_buf1_tensor.GetPhyAddr()));
    auto imag_buf1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(imag_buf1_tensor.GetPhyAddr()));
    auto complex_buf2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(complex_buf2_tensor.GetPhyAddr()));
    auto real_buf2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(real_buf2_tensor.GetPhyAddr()));
    auto imag_buf2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(imag_buf2_tensor.GetPhyAddr()));

    const int16_t aiv_id = AscendC::GetBlockIdx();
    const int16_t aiv_num = AscendC::GetBlockNum();

    const int32_t max_datacount_perloop = 48 * 1024 / (COMPLEX_DOUBLE * sizeof(float));

    const int32_t n_repeat_count = max_datacount_perloop / dValue;

    const int32_t n_loop_count = CeilADivB(nValue, n_repeat_count);
    const int32_t n_loop_remain_count = nValue % n_repeat_count;

    // divide data
    bool flag = 0;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (int32_t n_loop_idx = aiv_id; n_loop_idx < n_loop_count; n_loop_idx += aiv_num) {
        int32_t n_repeat_current =
            ((n_loop_idx == n_loop_count - 1) && (n_loop_remain_count > 0)) ? n_loop_remain_count : n_repeat_count;

        // double buffer prepare
        auto complex_buf = flag ? complex_buf1 : complex_buf2;
        auto complex_buf_tensor = flag ? complex_buf1_tensor : complex_buf2_tensor;
        auto real_buf = flag ? real_buf1 : real_buf2;
        auto real_buf_tensor = flag ? real_buf1_tensor : real_buf2_tensor;
        auto imag_buf = flag ? imag_buf1 : imag_buf2;
        auto imag_buf_tensor = flag ? imag_buf1_tensor : imag_buf2_tensor;
        auto event_id = flag ? EVENT_ID0 : EVENT_ID1;

        WAIT_FLAG(MTE3, MTE2, event_id);
        //load data from gm to ub
        // GM 元素偏移按 int64 计算（issue #234）：n*stride 可达 2^30 量级，
        // int32 乘法在 ×COMPLEX_DOUBLE 后回绕致越界读写
        auto gm_src_tensor_shifted =
            gm_src_tensor[static_cast<int64_t>(n_loop_idx) * n_repeat_count * srcDValue * COMPLEX_DOUBLE];
        LoadDataFromGmToUbDev(complex_buf_tensor, gm_src_tensor_shifted,
                           n_repeat_current, dValue * COMPLEX_DOUBLE, srcDValue * COMPLEX_DOUBLE);
        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);
        // separate
        int32_t repeat_count = n_repeat_current * dValue * COMPLEX_DOUBLE / (BLOCK_PER_REPEAT * BLOCK_SIZE);
        uint64_t rsvdCnt = 0;
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(real_buf), reinterpret_cast<__ubuf__ uint32_t *>(complex_buf),
                  nullptr, repeat_count, 1, PATTERN_MODE_REPEAT_TWO_GET_FIRST, BLOCK_PER_REPEAT, BLOCK_PER_REPEAT);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(imag_buf), reinterpret_cast<__ubuf__ uint32_t *>(complex_buf),
                  nullptr, repeat_count, 1, PATTERN_MODE_REPEAT_TWO_GET_SECOND, BLOCK_PER_REPEAT, BLOCK_PER_REPEAT);

        SET_FLAG(V, MTE3, event_id);
        WAIT_FLAG(V, MTE3, event_id);
        //save data from ub to gm
        auto gm_dst_tensor_shifted1 = gm_dst_tensor[static_cast<int64_t>(n_loop_idx) * n_repeat_count * dValue];
        SaveDataFromUbToGmDev(gm_dst_tensor_shifted1, real_buf_tensor, n_repeat_current, dValue, dValue);

        auto gm_dst_tensor_shifted2 = gm_dst_tensor[static_cast<int64_t>(nValue) * dValue +
                                                    static_cast<int64_t>(n_loop_idx) * n_repeat_count * dValue];
        SaveDataFromUbToGmDev(gm_dst_tensor_shifted2, imag_buf_tensor, n_repeat_current, dValue, dValue);
        SET_FLAG(MTE3, MTE2, event_id);
        flag = 1 - flag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

// (P,Q,2,S)->(Q,P,S,2)
__aicore__ __inline__ void TransposeWithStrideAndCombineRI(__gm__ float *__restrict__ gm_dst,
                                                           __gm__ float *__restrict__ gm_src, int32_t P, int32_t Q,
                                                           int32_t dValue, int32_t dstDValue)
{
    const int16_t aiv_id = AscendC::GetBlockIdx();
    const int16_t aiv_num = AscendC::GetBlockNum();

    AscendC::GlobalTensor<float> gm_src_tensor;
    AscendC::GlobalTensor<float> gm_dst_tensor;

    gm_src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_src));
    gm_dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_dst));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> buf10_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> buf11_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(48 * 1024));
    AscendC::LocalTensor<float> buf20_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(96 * 1024));
    AscendC::LocalTensor<float> buf21_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((uintptr_t)(144 * 1024));

    auto buf10 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(buf10_tensor.GetPhyAddr()));
    auto buf11 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(buf11_tensor.GetPhyAddr()));
    auto buf20 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(buf20_tensor.GetPhyAddr()));
    auto buf21 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(buf21_tensor.GetPhyAddr()));

    int32_t singleUnitElementCount = dValue * COMPLEX_DOUBLE;
    // 除0整改
    if (singleUnitElementCount == 0) {
        singleUnitElementCount = 1;
    }
    // assume Q = 64
    int32_t singleBufRepeatCnt = 48 * 1024 / (singleUnitElementCount * sizeof(float));

    int32_t p_loop_count = CeilADivB(P, singleBufRepeatCnt);
    int32_t p_loop_remain = P % singleBufRepeatCnt;

    // divide data
    bool flag = 0;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);
    for (int32_t q_idx = 0; q_idx < Q; q_idx++) {
        for (int32_t p_loop_idx = 0; p_loop_idx < p_loop_count; p_loop_idx++) {
            if ((p_loop_idx + q_idx * p_loop_count) % aiv_num != aiv_id) {
                continue;
            }
            int32_t current_repeat_cnt = ((p_loop_idx == p_loop_count - 1) && (p_loop_remain > 0)) ? p_loop_remain :
                                                                                                     singleBufRepeatCnt;
            auto buf0 = flag ? buf10 : buf20;
            auto buf0_tensor = flag ? buf10_tensor : buf20_tensor;
            auto buf1 = flag ? buf11 : buf21;
            auto buf1_tensor = flag ? buf11_tensor : buf21_tensor;
            auto event_id = flag ? EVENT_ID0 : EVENT_ID1;
            WAIT_FLAG(MTE3, MTE2, event_id);
            auto gm_src_tensor_shifted1 =
                gm_src_tensor[static_cast<int64_t>(p_loop_idx) * singleBufRepeatCnt * Q * singleUnitElementCount +
                              q_idx * singleUnitElementCount];
            LoadDataFromGmToUbDev(buf0_tensor, gm_src_tensor_shifted1,
                               current_repeat_cnt, dValue, Q * singleUnitElementCount);
            auto buf0_tensor_shifted = buf0_tensor[current_repeat_cnt * dValue];
            auto gm_src_tensor_shifted2 =
                gm_src_tensor[static_cast<int64_t>(p_loop_idx) * singleBufRepeatCnt * Q * singleUnitElementCount +
                              q_idx * singleUnitElementCount + dValue];
            LoadDataFromGmToUbDev(buf0_tensor_shifted, gm_src_tensor_shifted2,
                               current_repeat_cnt, dValue, Q * singleUnitElementCount);

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);

            // vtranspose
            for (int32_t i = 0; i < current_repeat_cnt * dValue / 64; i++) {
                transpose_v<ArchType::ASCEND_V220, uint16_t>(
                    buf1_tensor[i * 128].template ReinterpretCast<uint16_t>(),
                    buf0_tensor[i * 128].template ReinterpretCast<uint16_t>()
                );
            }
            PIPE_BARRIER(V);

            AscendC::DataCopy(
                buf0_tensor,
                buf1_tensor,
                AscendC::DataCopyParams(
                    current_repeat_cnt * dValue / 16, // blockCount
                    2, // blockLen
                    0, // srcStride
                    2 // dstStride
                )
            );

            AscendC::DataCopy(
                buf0_tensor[16],
                buf1_tensor[current_repeat_cnt * dValue],
                AscendC::DataCopyParams(
                    current_repeat_cnt * dValue / 16, // blockCount
                    2, // blockLen
                    0, // srcStride
                    2 // dstStride
                )
            );
            PIPE_BARRIER(V);

            // vtranspose
            for (int32_t i = 0; i < current_repeat_cnt * dValue / 64; i++) {
                transpose_v<ArchType::ASCEND_V220, uint16_t>(
                    buf1_tensor[i * 128].template ReinterpretCast<uint16_t>(),
                    buf0_tensor[i * 128].template ReinterpretCast<uint16_t>()
                );
            }
            PIPE_BARRIER(V);

            adds_v<ArchType::ASCEND_V220, float>(buf0_tensor, buf1_tensor, 0.0f,
                  current_repeat_cnt * dValue / 128,  // repeat
                  2,                                  // dstBlockStride
                  1,                                  // srcBlockStride
                  16 * 2,                             // dstRepeatStride
                  16 * 2                              // srcRepeatStride
            );

            adds_v<ArchType::ASCEND_V220, float>(buf0_tensor[8], buf1_tensor[128], 0.0f,
                  current_repeat_cnt * dValue / 128,  // repeat
                  2,                                  // dstBlockStride
                  1,                                  // srcBlockStride
                  16 * 2,                             // dstRepeatStride
                  16 * 2                              // srcRepeatStride
            );

            adds_v<ArchType::ASCEND_V220, float>(buf0_tensor[64 * 2], buf1_tensor[64], 0.0f,
                  current_repeat_cnt * dValue / 128,  // repeat
                  2,                                  // dstBlockStride
                  1,                                  // srcBlockStride
                  16 * 2,                             // dstRepeatStride
                  16 * 2                              // srcRepeatStride
            );

            adds_v<ArchType::ASCEND_V220, float>(buf0_tensor[64 * 2 + 8], buf1_tensor[64 + 128], 0.0f,
                  current_repeat_cnt * dValue / 128,  // repeat
                  2,                                  // dstBlockStride
                  1,                                  // srcBlockStride
                  16 * 2,                             // dstRepeatStride
                  16 * 2                              // srcRepeatStride
            );

            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);
            auto gm_dst_final_shifted =
                gm_dst_tensor[static_cast<int64_t>(q_idx) * P * dstDValue * COMPLEX_DOUBLE +
                              static_cast<int64_t>(p_loop_idx) * singleBufRepeatCnt * dstDValue * COMPLEX_DOUBLE];
            SaveDataFromUbToGmDev(gm_dst_final_shifted, buf0_tensor, current_repeat_cnt, singleUnitElementCount, dstDValue * COMPLEX_DOUBLE);
            SET_FLAG(MTE3, MTE2, event_id);
            flag = 1 - flag;
        }
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

// inner transpose from (P,Q,K,N) to (Q,P,K,N), then matmul (M,K) (K,N) with iterbatch Q*P.
__aicore__ __inline__ void TransposeAndMatmul(__gm__ float *__restrict__ gm_c, __gm__ float *__restrict__ gm_a,
                                              __gm__ float *__restrict__ gm_b, int32_t P, int32_t Q, int32_t M,
                                              int32_t K, int32_t N)
{
    // 除0整改
    if (Q == 0) {
        Q = 1;
    }
    set_padding(0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    set_nd_para(config);

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
    auto l0c_base = reinterpret_cast<__cc__ float *>((uintptr_t)(l0c_buf_tensor.GetPhyAddr()));

    int32_t batchSize = P * Q;
    bool trans_a = false;
    bool trans_b = false;
    int32_t M0 = 128;
    int32_t N0 = 128;
    int32_t K0 = 128;
    if (M <= 64) {
        M0 = 64;
        N0 = 256;
        K0 = 64;
    }

    // 除0整改
    int32_t m_loop = (M + M0 - 1) / M0;
    if (m_loop == 0) {
        m_loop = 1;
    }
    // 除0整改
    int32_t n_loop = (N + N0 - 1) / N0;
    if (n_loop == 0) {
        n_loop = 1;
    }
    int32_t k_loop = (K + K0 - 1) / K0;
    int32_t loop = batchSize * m_loop * n_loop;

    int32_t ping_flag = 1;

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);

    SET_FLAG(FIX, M, EVENT_ID0);
    SET_FLAG(FIX, M, EVENT_ID1);

    int32_t ping_flag_fix = 1;

    int32_t blockNum = AscendC::GetBlockNum();
    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {
        if (loop_idx % blockNum != AscendC::GetBlockIdx()) {
            continue;
        }

        auto l0c_buf = ping_flag_fix ? l0c_base : l0c_base + L0C_PINGPONG_BUFFER_LEN;
        auto event_id_fix = ping_flag_fix ? EVENT_ID0 : EVENT_ID1;

        int64_t batch_idx = static_cast<int64_t>(loop_idx) / (static_cast<int64_t>(m_loop) * static_cast<int64_t>(n_loop));
        int64_t real_batch_idx = (batch_idx) / static_cast<int64_t>(Q) + (batch_idx) % static_cast<int64_t>(Q) * static_cast<int64_t>(P);

        int64_t in_batch_idx = static_cast<int64_t>(loop_idx) % (static_cast<int64_t>(m_loop) * static_cast<int64_t>(n_loop));
        int64_t m_idx = in_batch_idx / static_cast<int64_t>(n_loop);
        int64_t n_idx = in_batch_idx % static_cast<int64_t>(n_loop);

        int64_t offset_a, offset_b;
        int64_t offset_c = real_batch_idx * static_cast<int64_t>(M) * static_cast<int64_t>(N) + static_cast<int64_t>(m_idx) * static_cast<int64_t>(M0) * static_cast<int64_t>(N) + static_cast<int64_t>(n_idx) * static_cast<int64_t>(N0);
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);

        auto mte1_mad_ping_flag = 1;

        for (int32_t k_idx = 0; k_idx < k_loop; k_idx++) {
            if (trans_a) {
                offset_a = real_batch_idx * M * K + k_idx * K0 * M + m_idx * M0;
            } else {
                offset_a = real_batch_idx * M * K + m_idx * M0 * K + k_idx * K0;
            }

            if (trans_b) {
                offset_b = batch_idx * K * N + n_idx * N0 * K + k_idx * K0;
            } else {
                offset_b = batch_idx * K * N + k_idx * K0 * N + n_idx * N0;
            }

            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;
            int32_t k_round = (k_actual + 15) / 16 * 16;
            int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

            auto l1_buf_a = ping_flag ? l1_base_a : l1_base_a + L1_PINGPONG_BUFFER_LEN;
            auto l1_buf_a_tensor = ping_flag ? l1_base_a_tensor : l1_base_a_tensor[L1_PINGPONG_BUFFER_LEN];
            auto l1_buf_b = ping_flag ? l1_base_b : l1_base_b + L1_PINGPONG_BUFFER_LEN;
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

            for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto l0a_buf = l0a_base + mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN;
                auto l0b_buf = l0b_base + mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN;
                auto l0a_buf_tensor = l0a_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];
                auto l0b_buf_tensor = l0b_base_tensor[mte1_mad_ping_flag * L0AB_PINGPONG_BUFFER_LEN];

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
                        for (int32_t i = 0; i < m_round / R0; i++) {
                            AscendC::LoadDataWithTranspose(
                                l0a_buf_tensor[i * k0_round * R0],
                                l1_src_a_tensor[i * 2 * CUBE_MATRIX_SIZE],
                                AscendC::LoadData2dTransposeParams(
                                    0,                             // indexID
                                    k0_round / R0,         // repeat
                                    M0 / R0,               // srcStride
                                    1,                             // dstGap
                                    0,                             // dstFracStride
                                    inc                           // addrmode
                                )
                            );
                        }
                    } else {
                        auto l1_src_a_tensor = l1_buf_a_tensor[k_part_idx * k_part_len * R0];
                        for (int32_t i = 0; i < m_round / R0; i++) {
                            AscendC::LoadData(
                                l0a_buf_tensor[i * k0_round * R0],
                                l1_src_a_tensor[i * R0 * K0],
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
                    auto l1_src_b_tensor = l1_buf_b_tensor[k_part_idx * k_part_len * R0];
                    for (int32_t i = 0; i < k0_round / C0_SIZE; i++) {
                        AscendC::LoadData(
                            l0b_buf_tensor[i * n_round * C0_SIZE],
                            l1_src_b_tensor[i * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(
                                0,                                                      // baseIdx
                                n_round / R0,   // repeat
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
                    for (int32_t i = 0; i < n_round / R0; i++) {
                        AscendC::LoadDataWithTranspose(
                            l0b_buf_tensor[i * CUBE_MATRIX_SIZE],
                            l1_src_b_tensor[i * 2 * CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dTransposeParams(
                                0,                             // indexID
                                k0_round / R0,         // repeat
                                N0 / R0,               // srcStride
                                2 * n_round / R0 - 1,  // dstGap
                                n_round / R0 - 1,      // dstFracStride
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
                if (init_c) {
                    WAIT_FLAG(FIX, M, event_id_fix);
                }

                if (M != 1 && m_actual == 1 && trans_a) {
                    mad(l0c_buf, l0a_buf, l0b_buf,
                        16,         // m
                        k0_actual,  // k
                        n_actual,   // n
                        0,          // unitFlag
                        1,          // kDirectionAlign
                        0,          // cmatrixSource
                        init_c      // cmatrixInitVal
                    );
                } else {
                    mad(l0c_buf, l0a_buf, l0b_buf,
                        m_actual,   // m
                        k0_actual,  // k
                        n_actual,   // n
                        0,          // unitFlag
                        1,          // kDirectionAlign
                        0,          // cmatrixSource
                        init_c      // cmatrixInitVal
                    );
                }

                PIPE_BARRIER(M);
                SET_FLAG(M, MTE1, mte1_mad_event_id);
                mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
            }

            ping_flag = 1 - ping_flag;
        }

        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);

        SET_FLAG(M, FIX, EVENT_ID0);
        WAIT_FLAG(M, FIX, EVENT_ID0);

        // copy from L0C to gm
        copy_matrix_cc_to_gm(gm_c + offset_c, l0c_buf,
                             0,         // sid
                             n_actual,  // NSize
                             m_actual,  // MSize
                             N,         // dstStride_dst_D
                             m_round,   // srcStride
                             0,         // UnitFlagMode
                             NoQuant,   // QuantPRE
                             0,         // ReLUPRE
                             false,     // channelSplit
                             true       // NZ2ND_EN
        );

        SET_FLAG(FIX, M, event_id_fix);
        ping_flag_fix = 1 - ping_flag_fix;
    }

    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);

    WAIT_FLAG(FIX, M, EVENT_ID0);
    WAIT_FLAG(FIX, M, EVENT_ID1);

    PIPE_BARRIER(ALL);
}

extern "C" __global__ __aicore__ void fft_stride(__gm__ uint8_t *__restrict__ ffts_addr,
                                                 __gm__ float *__restrict__ gm_input,
                                                 __gm__ float *__restrict__ gm_s_matrix,
                                                 __gm__ float *__restrict__ gm_output,
                                                 __gm__ float *__restrict__ gm_workspace,
                                                 __gm__ uint8_t *__restrict__ tiling_para_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#ifdef __DAV_C220_CUBE__
    {
        // Init
        set_padding(0);
        AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
        AscendC::SetAtomicNone();

        uint64_t flag_id = SYNC_ALL_CORE_FLAG_ID;
        uint64_t mode = 0;  // all core sync
        uint64_t config = 1 | (mode << 4) | (flag_id << 8);

        auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
        const int32_t batch_size = tiling_para[0];
        const int32_t fft_length = tiling_para[1];
        const int32_t fft_stride = tiling_para[2];
        const int32_t fftDirection = tiling_para[3];
        const int32_t iter_count = tiling_para[4];
        const int32_t S0 = tiling_para[10];

        const int32_t big_loop_count = CeilADivB(fft_stride, S0);
        const int32_t big_loop_remain_count = fft_stride % S0;

        __gm__ float *__restrict__ gm_wksp2 = gm_workspace + fft_length * S0 * COMPLEX_DOUBLE;

        // matmul iter
        for (int32_t big_loop_idx = 0; big_loop_idx < big_loop_count; big_loop_idx++) {
            int32_t S0_current =
                ((big_loop_idx == big_loop_count - 1) && (big_loop_remain_count > 0)) ? big_loop_remain_count : S0;
            auto gm_iter_s_matrix = gm_s_matrix;
            int32_t tempN;
            int32_t tempBatch = 1;
            int32_t tempRepeat = 1;
            int32_t tempRemain = fft_length;

            uint64_t flag_id = SYNC_SINGLE_VC_FLAG_ID;
            WaitFlagDev(flag_id);  // 等待虚实分离计算结束

            // fftiter
            for (int32_t iter_idx = 0; iter_idx < iter_count; iter_idx++) {
                tempN = tiling_para[5 + iter_idx];
                auto gm_iter_in = (iter_idx % 2 == 0) ? gm_workspace : gm_wksp2;
                auto gm_iter_out = (iter_idx % 2 == 0) ? gm_wksp2 : gm_workspace;
                tempRemain /= tempN;
                TransposeAndMatmul(gm_iter_out, gm_iter_s_matrix, gm_iter_in, tempBatch, tempRepeat / tempBatch, tempN * 2,
                                tempN * 2, tempRemain * S0_current);
                gm_iter_s_matrix += tempRepeat * 2 * tempN * 2 * tempN;

                flag_id = SYNC_ALL_CORE_FLAG_ID;
                FftsCrossCoreSync<PIPE_FIX, SYNC_ALL_CORE_MODE_CODE>(flag_id);
                flag_id = SYNC_ALL_CORE_FLAG_ID;
                WaitFlagDev(flag_id);  // 全cube同步

                tempBatch *= tempRepeat;
                tempRepeat *= tempN;
            }
            // 发信号启动最终迭代（转置+layout）
            flag_id = SYNC_SINGLE_CV_FLAG_ID;
            FftsCrossCoreSync<PIPE_FIX, SYNC_SINGLE_CV_MODE_CODE>(flag_id);
        }
    }

#elif __DAV_C220_VEC__
    {
        // Init
        AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

        auto tiling_para = reinterpret_cast<__gm__ int32_t *>(tiling_para_gm);
        const int32_t batch_size = tiling_para[0];
        const int32_t fft_length = tiling_para[1];
        const int32_t fft_stride = tiling_para[2];
        const int32_t fftDirection = tiling_para[3];
        const int32_t iter_count = tiling_para[4];
        const int32_t S0 = tiling_para[10];

        const int32_t big_loop_count = CeilADivB(fft_stride, S0);
        const int32_t big_loop_remain_count = fft_stride % S0;

        __gm__ float *__restrict__ gm_wksp2 = gm_workspace + fft_length * S0 * COMPLEX_DOUBLE;

        uint64_t flag_id = SYNC_ALL_CORE_FLAG_ID;
        uint64_t mode = 0;  // all core sync
        uint64_t config = 1 | (mode << 4) | (flag_id << 8);

        __gm__ float *__restrict__ gm_input_current = gm_input;
        __gm__ float *__restrict__ gm_output_current = gm_output;

        for (int32_t big_loop_idx = 0; big_loop_idx < big_loop_count; big_loop_idx++) {
            int32_t S0_current =
                ((big_loop_idx == big_loop_count - 1) && (big_loop_remain_count > 0)) ? big_loop_remain_count : S0;
            // 虚实分离 -- check ok
            SeparateRI(gm_workspace, gm_input_current, fft_length, S0_current, fft_stride);

            flag_id = SYNC_ALL_CORE_FLAG_ID;
            FftsCrossCoreSync<PIPE_MTE3, SYNC_ALL_CORE_FLAG_ID>(flag_id);

            flag_id = SYNC_ALL_CORE_FLAG_ID;
            WaitFlagDev(flag_id);  // 等待虚实分离计算结束

            // 发信号启动矩阵乘迭代
            flag_id = SYNC_SINGLE_VC_FLAG_ID;
            FftsCrossCoreSync<PIPE_MTE3, SYNC_SINGLE_CV_MODE_CODE>(flag_id);

            uint64_t flag_id = SYNC_SINGLE_CV_FLAG_ID;
            WaitFlagDev(flag_id);  // 等待cube计算结束
            TransposeWithStrideAndCombineRI(gm_output_current, (iter_count % 2 == 0) ? gm_workspace : gm_wksp2,
                                            fft_length / tiling_para[5 + iter_count - 1], tiling_para[5 + iter_count - 1],
                                            S0_current, fft_stride);

            flag_id = SYNC_ALL_CORE_FLAG_ID;
            FftsCrossCoreSync<PIPE_MTE3, SYNC_ALL_CORE_FLAG_ID>(flag_id);

            flag_id = SYNC_ALL_CORE_FLAG_ID;
            WaitFlagDev(flag_id);  // 等待本轮计算结束

            gm_input_current += S0 * COMPLEX_DOUBLE;
            gm_output_current += S0 * COMPLEX_DOUBLE;
        }

        // transpose and complex combination
    }
 #endif
}

} // namespace

#endif
