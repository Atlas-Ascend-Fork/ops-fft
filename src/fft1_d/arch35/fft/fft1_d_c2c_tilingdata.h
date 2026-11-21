/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT1_D_C2C_ARCH35_TILINGDATA_H
#define FFT1_D_C2C_ARCH35_TILINGDATA_H

#include <cstdint>

constexpr int32_t MAX_FFT_STAGES_TILING = 32;

struct Fft1DC2CTilingData {
    int64_t batchSize;
    int64_t fftN;
    int32_t radixListLen;
    int32_t isInverse;
    int64_t workspaceOffsets[5];

    int32_t radix_arr[MAX_FFT_STAGES_TILING];
    int64_t M_arr[MAX_FFT_STAGES_TILING];
    int64_t dft_offset_arr[MAX_FFT_STAGES_TILING];
    int64_t tw_offset_arr[MAX_FFT_STAGES_TILING];
};

#endif