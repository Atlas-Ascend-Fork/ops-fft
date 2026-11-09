/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDOPS_FFT_ALL_MIX_TILING_KERNEL_UTILS
#define ASDOPS_FFT_ALL_MIX_TILING_KERNEL_UTILS


#include "fft_all_mix_tiling_data.h"

constexpr int64_t batch_size_offset = offsetof(OpsFft::FftAllMixTilingData, batchSize);
constexpr int64_t fftN_offset = offsetof(OpsFft::FftAllMixTilingData, fftN);
constexpr int64_t radix_list_len_offset = offsetof(OpsFft::FftAllMixTilingData, radixListLen);
constexpr int64_t is_inverse_offset = offsetof(OpsFft::FftAllMixTilingData, isInverse);
constexpr int64_t workspace_offsets_offset = offsetof(OpsFft::FftAllMixTilingData, workspaceOffsets);

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, OpsFft::FftAllMixTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->batchSize = (*(const __gm__ int64_t *)(p_tilingdata + batch_size_offset));
    tilingdata->fftN = (*(const __gm__ int64_t *)(p_tilingdata + fftN_offset));
    tilingdata->radixListLen = (*(const __gm__ int32_t *)(p_tilingdata + radix_list_len_offset));
    tilingdata->isInverse = (*(const __gm__ int32_t *)(p_tilingdata + is_inverse_offset));
    tilingdata->workspaceOffsets[0] = (*(const __gm__ int64_t *)(p_tilingdata + workspace_offsets_offset));
    tilingdata->workspaceOffsets[1] =
        (*(const __gm__ int64_t *)(p_tilingdata + workspace_offsets_offset + sizeof(int64_t)));
    tilingdata->workspaceOffsets[2] =
        (*(const __gm__ int64_t *)(p_tilingdata + workspace_offsets_offset + sizeof(int64_t) * 2));
    tilingdata->workspaceOffsets[3] =
        (*(const __gm__ int64_t *)(p_tilingdata + workspace_offsets_offset + sizeof(int64_t) * 3));
    tilingdata->workspaceOffsets[4] =
        (*(const __gm__ int64_t *)(p_tilingdata + workspace_offsets_offset + sizeof(int64_t) * 4));
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    // 搬运块数按 32B 块粒度向上取整（issue #299）：原 sizeof/32+1 在结构体大小
    // 恰为 32 倍数时固定多搬一块、否则越界读；与 host 侧 tiling 缓冲按
    // ROUND_UP(sizeof, 32) 分配成对修改（fft_mix_host_common.h /
    // fft1_d_mix.cpp），块数与分配始终覆盖同一字节范围
    int32_t tilingBlockNum = (sizeof(OpsFft::FftAllMixTilingData) + 31) / 32;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, tilingBlockNum, 0, 0);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->batchSize = (*(const __gm__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + batch_size_offset));
    tilingdata->fftN = (*(const __gm__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + fftN_offset));
    tilingdata->radixListLen =
        (*(const __gm__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + radix_list_len_offset));
    tilingdata->isInverse = (*(const __gm__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + is_inverse_offset));
    tilingdata->workspaceOffsets[0] =
        (*(const __gm__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + workspace_offsets_offset));
    tilingdata->workspaceOffsets[1] =
        (*(const __gm__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + workspace_offsets_offset + sizeof(int64_t)));
    tilingdata->workspaceOffsets[2] = (*(const __gm__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub +
                                                                 workspace_offsets_offset + sizeof(int64_t) * 2));
    tilingdata->workspaceOffsets[3] = (*(const __gm__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub +
                                                                 workspace_offsets_offset + sizeof(int64_t) * 3));
    tilingdata->workspaceOffsets[4] = (*(const __gm__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub +
                                                                 workspace_offsets_offset + sizeof(int64_t) * 4));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#endif
