/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_ALL_MIX_TILING_DATA_H
#define FFT_ALL_MIX_TILING_DATA_H

#include <cstdint>

namespace OpsFft {
constexpr int32_t MAX_FFT_STAGES_TILING = 32;

struct FftAllMixTilingData {
    int64_t batchSize;
    int64_t fftN;
    int32_t radixListLen;
    // for now, c2r is only inverse and inverse = 1, r2c is only forward and inverse = 0
    int32_t isInverse;
    int64_t workspaceOffsets[5];

    // Precomputed plan data (moved from device to host to avoid VLOOP nesting)
    int32_t radix_arr[MAX_FFT_STAGES_TILING];
    int64_t M_arr[MAX_FFT_STAGES_TILING];
    int64_t dft_offset_arr[MAX_FFT_STAGES_TILING];
    int64_t tw_offset_arr[MAX_FFT_STAGES_TILING];
    int64_t currentBatch_arr[MAX_FFT_STAGES_TILING];

    int32_t isOddN;  // 新增：1 if N is odd, 0 if N is even
    int32_t outer;
    int32_t scaleOut;
    int32_t transpose;
};
}

// arch35 c2c_mix 等以全局名引用该类型（issue #254 收敛后保持原引用形态）
using OpsFft::FftAllMixTilingData;

// 设备侧按 C 布局 offset 解析 tiling，布局漂移无编译期信号，以字节数断言兜底
// （int64*2 + int32*2 + int64[5] + int32[32] + int64[32]*4 + int32*4 = 1232 字节）
static_assert(sizeof(OpsFft::FftAllMixTilingData) == 1232,
              "FftAllMixTilingData layout changed; sync all device-side parsing sites");

#endif
