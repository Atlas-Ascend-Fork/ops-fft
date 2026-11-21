/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT1_D_C2C_ARCH35_KERNEL_H
#define FFT1_D_C2C_ARCH35_KERNEL_H

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

/**
 * @file fft1_d_c2c_kernel.h
 * @brief Ascend950 C2C Stockham mixed-radix FFT kernel (SIMT/VF)
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr uint32_t VF_MAX_THREAD_NUM = 256;
constexpr int32_t MAX_FFT_STAGES = 32;

__simt_callee__ inline void ComplexMul(
    float ar, float ai, float br, float bi, float &cr, float &ci)
{
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

__simt_callee__ inline int32_t FindRadix(int64_t n)
{
    if (n % 2 == 0) return 2;
    if (n % 3 == 0) return 3;
    if (n % 5 == 0) return 5;
    if (n % 7 == 0) return 7;
    if (n % 11 == 0) return 11;
    if (n % 13 == 0) return 13;
    if (n % 17 == 0) return 17;
    if (n % 19 == 0) return 19;
    return 0;
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftC2CCopyIn(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_workspace,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t totalComplex = batchSize * fftN;
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t wsFloatOffset = wsOffset / sizeof(float);

    for (int64_t idx = tid; idx < totalComplex; idx += stride) {
        int64_t inFloatIdx = idx * 2;
        int64_t outFloatIdx = wsFloatOffset + idx * 2;
        gm_workspace[outFloatIdx] = gm_input[inFloatIdx];
        gm_workspace[outFloatIdx + 1] = gm_input[inFloatIdx + 1];
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftC2CStockhamForwardStage(
    __gm__ float * __restrict__ inputBuf,
    __gm__ float * __restrict__ outputBuf,
    __gm__ float * __restrict__ dftMat,
    __gm__ float * __restrict__ twMat,
    int32_t radix,
    int64_t M,
    int64_t currentBatch,
    int32_t step,
    int32_t radixListLen)
{
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t totalOutputComplex = currentBatch * radix * M;

    for (int64_t outIdx = tid; outIdx < totalOutputComplex; outIdx += stride) {
        int64_t batchIdx = outIdx / (radix * M);
        int64_t rem = outIdx - batchIdx * (radix * M);
        int64_t u = rem / M;
        int64_t n2 = rem - u * M;

        float resultRe = 0.0f;
        float resultIm = 0.0f;

        for (int32_t v = 0; v < radix; v++) {
            float wrRe = dftMat[(2 * u) * radix + v];
            float wrIm = dftMat[(2 * u + 1) * radix + v];
            int64_t xFloatIdx = (batchIdx * radix * M + v * M + n2) * 2;
            float xRe = inputBuf[xFloatIdx];
            float xIm = inputBuf[xFloatIdx + 1];
            float mulRe, mulIm;
            ComplexMul(wrRe, wrIm, xRe, xIm, mulRe, mulIm);
            resultRe += mulRe;
            resultIm += mulIm;
        }

        if (step < radixListLen - 1) {
            float tRe = twMat[(2 * u) * M + n2];
            float tIm = twMat[(2 * u + 1) * M + n2];

            float tmpRe, tmpIm;
            ComplexMul(resultRe, resultIm, tRe, tIm, tmpRe, tmpIm);
            resultRe = tmpRe;
            resultIm = tmpIm;
        }

        int64_t outComplexIdx = (batchIdx * radix + u) * M + n2;
        int64_t outFloatIdx = outComplexIdx * 2;
        outputBuf[outFloatIdx] = resultRe;
        outputBuf[outFloatIdx + 1] = resultIm;
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftC2CTransposeStage(
    __gm__ float * __restrict__ inputBuf,
    __gm__ float * __restrict__ outputBuf,
    int32_t radix,
    int64_t M,
    int64_t currentBatch)
{
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t totalComplex = currentBatch * radix * M;

    for (int64_t outIdx = tid; outIdx < totalComplex; outIdx += stride) {
        int64_t batchIdx = outIdx / (M * radix);
        int64_t rem = outIdx - batchIdx * (M * radix);
        int64_t n2 = rem / radix;
        int64_t k1 = rem - n2 * radix;

        int64_t inComplexIdx = batchIdx * radix * M + k1 * M + n2;
        int64_t inFloatIdx = inComplexIdx * 2;

        float re = inputBuf[inFloatIdx];
        float im = inputBuf[inFloatIdx + 1];

        int64_t outFloatIdx = outIdx * 2;
        outputBuf[outFloatIdx] = re;
        outputBuf[outFloatIdx + 1] = im;
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftC2CCopyOut(
    __gm__ float * __restrict__ gm_workspace,
    __gm__ float * __restrict__ gm_output,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t totalComplex = batchSize * fftN;
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t wsFloatOffset = wsOffset / sizeof(float);
    // 归一化由矩阵内烘焙符号承担、无 1/N 缩放：正/逆变换 CopyOut 均不做方向相关缩放（issue #399）
    float norm = 1.0f;

    for (int64_t idx = tid; idx < totalComplex; idx += stride) {
        int64_t wsIdx = wsFloatOffset + idx * 2;
        gm_output[idx * 2] = gm_workspace[wsIdx] * norm;
        gm_output[idx * 2 + 1] = gm_workspace[wsIdx + 1] * norm;
    }
}

class FftC2CKernelMultiCore {
public:
    __aicore__ inline FftC2CKernelMultiCore() {}

    __aicore__ inline void Init(
        __gm__ float * __restrict__ gm_input,
        __gm__ float * __restrict__ gm_dft_matrix_array,
        __gm__ float * __restrict__ gm_tw_matrix_array,
        __gm__ float * __restrict__ radix_list,
        __gm__ float * __restrict__ gm_output,
        __gm__ float * __restrict__ gm_workspace,
        __gm__ uint8_t * __restrict__ gm_tiling_para);

    __aicore__ inline void Process();

private:
    __gm__ float * __restrict__ gm_input_core_;
    __gm__ float * __restrict__ gm_dft_matrix_array_;
    __gm__ float * __restrict__ gm_tw_matrix_array_;
    __gm__ float * __restrict__ gm_output_core_;
    __gm__ float * __restrict__ gm_workspace_core_;

    int64_t batchSize_;
    int64_t fftN_;
    int32_t radixListLen_;
    int64_t workspaceOffsets_[5];
    int64_t localWorkspaceOffsets_[2];

    __gm__ int32_t * __restrict__ gm_radix_arr_;
    __gm__ int64_t * __restrict__ gm_M_arr_;
    __gm__ int64_t * __restrict__ gm_dft_offset_arr_;
    __gm__ int64_t * __restrict__ gm_tw_offset_arr_;

    int64_t blockIdx_;
    int64_t blockNum_;
    int64_t batchStart_;
    int64_t localBatchSize_;
};

__aicore__ inline void FftC2CKernelMultiCore::Init(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para)
{
    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(gm_tiling_para);

    batchSize_ = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf));
    fftN_ = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 8));
    radixListLen_ = (*(__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + 16));
    // tiling 偏移 20 为 isInverse（host 侧写入、tiling 布局保持不变）；矩阵符号
    // 已按方向烘焙、CopyOut 无方向相关缩放，kernel 不再读取该字段（issue #399）

    for (int32_t i = 0; i < 5; i++) {
        workspaceOffsets_[i] = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 24 + 8 * i));
    }

    int64_t baseOffset = 24 + 5 * 8;
    gm_radix_arr_ = (__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + baseOffset);
    gm_M_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4);
    gm_dft_offset_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4 + MAX_FFT_STAGES * 8);
    gm_tw_offset_arr_ = (__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + baseOffset + MAX_FFT_STAGES * 4 + MAX_FFT_STAGES * 8 * 2);

    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    blockNum_ = static_cast<int64_t>(GetBlockNum());

    int64_t batchesPerCore = batchSize_ / blockNum_;
    int64_t extraBatches = batchSize_ % blockNum_;

    if (blockIdx_ < extraBatches) {
        batchStart_ = blockIdx_ * (batchesPerCore + 1);
        localBatchSize_ = batchesPerCore + 1;
    } else {
        batchStart_ = extraBatches * (batchesPerCore + 1) + (blockIdx_ - extraBatches) * batchesPerCore;
        localBatchSize_ = batchesPerCore;
    }

    int64_t perBatchComplexBytes = fftN_ * sizeof(float) * 2;
    int64_t inputByteOffset = batchStart_ * fftN_ * sizeof(float) * 2;
    int64_t outputByteOffset = batchStart_ * fftN_ * sizeof(float) * 2;

    gm_input_core_ = (__gm__ float *)((__gm__ uint8_t *)gm_input + inputByteOffset);
    gm_output_core_ = (__gm__ float *)((__gm__ uint8_t *)gm_output + outputByteOffset);
    gm_workspace_core_ = gm_workspace;
    gm_dft_matrix_array_ = gm_dft_matrix_array;
    gm_tw_matrix_array_ = gm_tw_matrix_array;

    localWorkspaceOffsets_[0] = workspaceOffsets_[0] + batchStart_ * perBatchComplexBytes;
    localWorkspaceOffsets_[1] = workspaceOffsets_[1] + batchStart_ * perBatchComplexBytes;
}

__aicore__ inline void FftC2CKernelMultiCore::Process()
{
    if (blockIdx_ > blockNum_ - 1) return;
    if (localBatchSize_ <= 0) return;

    Simt::VF_CALL<FftC2CCopyIn>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
        gm_input_core_,
        gm_workspace_core_,
        localBatchSize_,
        fftN_,
        localWorkspaceOffsets_[0]);

    int64_t tempBatch = localBatchSize_;
    for (int32_t step = 0; step < radixListLen_; step++) {
        __gm__ float * __restrict__ inputBuf;
        __gm__ float * __restrict__ outputBuf;
        if (step % 2 == 0) {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_core_ + localWorkspaceOffsets_[0]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_core_ + localWorkspaceOffsets_[1]);
        } else {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_core_ + localWorkspaceOffsets_[1]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_core_ + localWorkspaceOffsets_[0]);
        }

        __gm__ float * __restrict__ dftMat = gm_dft_matrix_array_ + gm_dft_offset_arr_[step];
        __gm__ float * __restrict__ twMat  = gm_tw_matrix_array_  + gm_tw_offset_arr_[step];

        Simt::VF_CALL<FftC2CStockhamForwardStage>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            inputBuf,
            outputBuf,
            dftMat,
            twMat,
            gm_radix_arr_[step],
            gm_M_arr_[step],
            tempBatch,
            step,
            radixListLen_);

        tempBatch *= gm_radix_arr_[step];
    }

    bool useWs1 = (radixListLen_ % 2 == 1);
    tempBatch = localBatchSize_;

    for (int32_t s = 0; s < radixListLen_; s++) {
        tempBatch *= gm_radix_arr_[s];
    }

    for (int32_t step = radixListLen_ - 1; step >= 0; step--) {
        __gm__ float * __restrict__ inputBuf;
        __gm__ float * __restrict__ outputBuf;
        if (useWs1) {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_core_ + localWorkspaceOffsets_[1]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_core_ + localWorkspaceOffsets_[0]);
        } else {
            inputBuf  = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_core_ + localWorkspaceOffsets_[0]);
            outputBuf = (__gm__ float *)((__gm__ uint8_t *)gm_workspace_core_ + localWorkspaceOffsets_[1]);
        }

        Simt::VF_CALL<FftC2CTransposeStage>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            inputBuf,
            outputBuf,
            gm_radix_arr_[step],
            gm_M_arr_[step],
            tempBatch / gm_radix_arr_[step]);

        tempBatch /= gm_radix_arr_[step];
        useWs1 = !useWs1;
    }

    int64_t finalWsOffset = useWs1 ? localWorkspaceOffsets_[1] : localWorkspaceOffsets_[0];

    Simt::VF_CALL<FftC2CCopyOut>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
        gm_workspace_core_,
        gm_output_core_,
        localBatchSize_,
        fftN_,
        finalWsOffset);

    return;
}

#endif

extern "C" __global__ __aicore__ void fft_c2c_arch35_multi_core(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::GlobalTensor<uint64_t> global;
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(global);
    FftC2CKernelMultiCore kernel;
    kernel.Init(gm_input, gm_dft_matrix_array, gm_tw_matrix_array, radix_list,
                gm_output, gm_workspace, gm_tiling_para);
    kernel.Process();
#endif
}

#endif
