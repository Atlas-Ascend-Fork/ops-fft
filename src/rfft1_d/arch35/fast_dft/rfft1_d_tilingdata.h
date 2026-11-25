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
 * \file rfft1_d_tilingdata.h
 * \brief
 */

#ifndef RFFT1_D_FAST_DFT_TILINGDATA_H
#define RFFT1_D_FAST_DFT_TILINGDATA_H

// 内核仅消费 length/batchesPerCore/leftOverBatches/dftRealOverallSize 四个字段
// （rfft1_d_fast_dft.cpp 内核入口）；其余 19 个字段（factors/prevRadices/nextRadices/
// dftImagOverallSize/twiddleOverallSize/fftMatrOverallSize/三个 offsets 数组/
// tmpLenPerBatch/tmpSizePerBatch/matmulTmpsLen/matmulTmpsSize 等）全仓零消费，
// 属死存储，已随 CalcDftSizes 死推导一并删除（issue #407）
struct Rfft1DTilingData {
    int32_t length;
    uint32_t batchesPerCore;
    uint32_t leftOverBatches;
    uint32_t dftRealOverallSize;
};

#endif // RFFT1_D_FAST_DFT_TILINGDATA_H
