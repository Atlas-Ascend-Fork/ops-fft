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

struct Rfft1DTilingData {
    int32_t length;
    uint8_t isBluestein;
    int32_t lengthPad;
    uint32_t factors[3];
    uint32_t prevRadices[3];
    uint32_t nextRadices[3];
    uint8_t prevRadicesAlign[3];
    int32_t outLength;
    uint32_t tailSize;
    uint32_t batchesPerCore;
    uint32_t leftOverBatches;
    uint32_t tmpLenPerBatch;
    uint32_t tmpSizePerBatch;
    uint32_t matmulTmpsLen;
    uint32_t matmulTmpsSize;
    int32_t normal;
    uint32_t dftRealOverallSize;
    uint32_t dftImagOverallSize;
    uint32_t twiddleOverallSize;
    uint32_t fftMatrOverallSize;
    uint32_t dftRealOffsets[3];
    uint32_t dftImagOffsets[3];
    uint32_t twiddleOffsets[3];
};

#endif // RFFT1_D_FAST_DFT_TILINGDATA_H
