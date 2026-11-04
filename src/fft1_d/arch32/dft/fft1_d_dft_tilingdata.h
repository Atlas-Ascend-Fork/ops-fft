/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fft1_d_dft_tilingdata.h
 * \brief
 */

#ifndef FFT1_D_DFT_TILINGDATA_H
#define FFT1_D_DFT_TILINGDATA_H

#include <cstdint>

struct Fft1DDFTTilingData {
    int32_t batchSize{0};
    int32_t m{0};
    int32_t n{0};
    int32_t k{0};
    int32_t transA{0};
    int32_t transB{0};
    int32_t m0{0};
    int32_t n0{0};
    int32_t k0{0};
    // loopTime/batchNumPerLoop/batchNumPerCore/batchTailNum/batchTailNumPerCore/
    // batchTailCoreNum 六个死字段删除（issue #276）：host 从不写入、kernel 从不读取，
    // 且注释携带的 "×40" 魔法数陈旧说明与现存布局不符
};

#endif // FFT1_D_DFT_TILINGDATA_H
