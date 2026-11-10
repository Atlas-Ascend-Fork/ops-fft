/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPSFFT_COMMON_KERNEL_BASE_SIMD_H
#define OPSFFT_COMMON_KERNEL_BASE_SIMD_H

#include "hardware.h"
#include "kernel_operator.h"

/////////////////////////////////////////////////////
// vadd
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void add_v(AscendC::LocalTensor<DType> dst,
                             AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    AscendC::Add<DType, false>(
        dst,
        src0,
        src1,
        (uint64_t)0,
        repeat,
        AscendC::BinaryRepeatParams(
            dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

/////////////////////////////////////////////////////
// vadds
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void adds_v(AscendC::LocalTensor<DType> dst,
                              AscendC::LocalTensor<DType> src,
                              DType scalarValue,
                              uint8_t repeat,
                              uint8_t dstBlockStride,
                              uint8_t srcBlockStride,
                              uint8_t dstRepeatStride,
                              uint8_t srcRepeatStride)
{
    AscendC::Adds<DType, false>(
        dst,
        src,
        scalarValue,
        (uint64_t)0,
        repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}
/////////////////////////////////////////////////////
// vmul
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void mul_v(AscendC::LocalTensor<DType> dst,
                             AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    AscendC::Mul<DType, false>(
        dst,
        src0,
        src1,
        (uint64_t)0,
        repeat,
        AscendC::BinaryRepeatParams(
            dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

/////////////////////////////////////////////////////
// vmuls
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void muls_v(AscendC::LocalTensor<DType> dst,
                              AscendC::LocalTensor<DType> src0,
                              DType src1,
                              uint8_t repeat,
                              uint16_t dstBlockStride,
                              uint16_t srcBlockStride,
                              uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
    AscendC::Muls<DType, false>(
        dst,
        src0,
        src1,
        (uint64_t)0,
        repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

/////////////////////////////////////////////////////
// vsub
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void sub_v(AscendC::LocalTensor<DType> dst,
                             AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    AscendC::Sub<DType, false>(
        dst,
        src0,
        src1,
        (uint64_t)0,
        repeat,
        AscendC::BinaryRepeatParams(
            dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride));
}
/////////////////////////////////////////////////////
// vtranspose
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void transpose_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src)
{
    AscendC::Transpose(dst, src);
}
#endif