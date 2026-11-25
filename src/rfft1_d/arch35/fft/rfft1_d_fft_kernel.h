/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RFFT1_D_FFT_KERNEL_H
#define RFFT1_D_FFT_KERNEL_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

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
    return 0;
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2CPackRealToComplex(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_workspace,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t N_half = fftN / 2;
    int64_t totalComplex = batchSize * N_half;

    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());

    int64_t wsFloatOffset = wsOffset / sizeof(float);

    for (int64_t idx = tid; idx < totalComplex; idx += stride) {
        int64_t b = idx / N_half;
        int64_t k = idx % N_half;

        int64_t inIdxReal = b * fftN + 2 * k;
        int64_t inIdxImag = b * fftN + 2 * k + 1;

        int64_t outFloatIdx = wsFloatOffset + (b * N_half + k) * 2;
        gm_workspace[outFloatIdx]     = gm_input[inIdxReal];
        gm_workspace[outFloatIdx + 1] = gm_input[inIdxImag];
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2COddPackRealToComplex(
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
        int64_t b = idx / fftN;
        int64_t k = idx % fftN;
        int64_t outFloatIdx = wsFloatOffset + (b * fftN + k) * 2;

        gm_workspace[outFloatIdx]     = gm_input[b * fftN + k];  // Real
        gm_workspace[outFloatIdx + 1] = 0.0f;                     // Imag = 0
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2CStockhamForwardStage(
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

        for (int64_t v = 0; v < radix; v++) {
            float wrRe = dftMat[(2 * u) * radix + v];
            float wrIm = dftMat[(2 * u + 1) * radix + v];
            int64_t xFloatIdx = (batchIdx * radix * M + v * M + n2) * 2;
            float xRe = inputBuf[xFloatIdx];
            float xIm = inputBuf[xFloatIdx + 1];
            resultRe += wrRe * xRe - wrIm * xIm;
            resultIm += wrRe * xIm + wrIm * xRe;
        }

        if (step < radixListLen - 1) {
            float tRe = twMat[(2 * u) * M + n2];
            float tIm = twMat[(2 * u + 1) * M + n2];
            float tmpRe = resultRe * tRe - resultIm * tIm;
            float tmpIm = resultRe * tIm + resultIm * tRe;
            resultRe = tmpRe;
            resultIm = tmpIm;
        }

        int64_t outComplexIdx = (batchIdx * radix + u) * M + n2;
        int64_t outFloatIdx = outComplexIdx * 2;
        outputBuf[outFloatIdx]     = resultRe;
        outputBuf[outFloatIdx + 1] = resultIm;
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2CTransposeStage(
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
        outputBuf[outFloatIdx]     = re;
        outputBuf[outFloatIdx + 1] = im;
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2CPostProcess(
    __gm__ float * __restrict__ gm_workspace,
    __gm__ float * __restrict__ gm_tw_post,
    __gm__ float * __restrict__ gm_output,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t N_half = fftN / 2;
    int64_t D = N_half + 1;

    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());

    int64_t wsFloatOffset = wsOffset / sizeof(float);

    for (int64_t idx = tid; idx < batchSize * D; idx += stride) {
        int64_t b = idx / D;
        int64_t k = idx % D;

        int64_t outFloatIdx = (b * D + k) * 2;

        if (k == 0) {
            int64_t z0FloatIdx = wsFloatOffset + (b * N_half) * 2;
            float z0Re = gm_workspace[z0FloatIdx];
            float z0Im = gm_workspace[z0FloatIdx + 1];
            gm_output[outFloatIdx]     = z0Re + z0Im;
            gm_output[outFloatIdx + 1] = 0.0f;
        } else if (k == N_half) {
            int64_t z0FloatIdx = wsFloatOffset + (b * N_half) * 2;
            float z0Re = gm_workspace[z0FloatIdx];
            float z0Im = gm_workspace[z0FloatIdx + 1];
            gm_output[outFloatIdx]     = z0Re - z0Im;
            gm_output[outFloatIdx + 1] = 0.0f;
        } else {
            int64_t zkFloatIdx = wsFloatOffset + (b * N_half + k) * 2;
            float zkRe = gm_workspace[zkFloatIdx];
            float zkIm = gm_workspace[zkFloatIdx + 1];

            int64_t znkFloatIdx = wsFloatOffset + (b * N_half + (N_half - k)) * 2;
            float znkRe =  gm_workspace[znkFloatIdx];
            float znkIm = -gm_workspace[znkFloatIdx + 1];

            float wkRe = gm_tw_post[k * 2];
            float wkIm = gm_tw_post[k * 2 + 1];

            float X_e_Re = (zkRe + znkRe) / 2.0f;
            float X_e_Im = (zkIm + znkIm) / 2.0f;

            float X_o_Re = (zkIm - znkIm) / 2.0f;
            float X_o_Im = (znkRe - zkRe) / 2.0f;

            float mulRe, mulIm;
            ComplexMul(wkRe, wkIm, X_o_Re, X_o_Im, mulRe, mulIm);

            gm_output[outFloatIdx]     = X_e_Re + mulRe;
            gm_output[outFloatIdx + 1] = X_e_Im + mulIm;
        }
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void FftR2COddTruncateOutput(
    __gm__ float * __restrict__ gm_workspace,
    __gm__ float * __restrict__ gm_output,
    int64_t batchSize,
    int64_t fftN,
    int64_t wsOffset)
{
    int64_t D = fftN / 2 + 1;  // 输出大小
    int64_t totalOutput = batchSize * D;
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());
    int64_t wsFloatOffset = wsOffset / sizeof(float);

    for (int64_t idx = tid; idx < totalOutput; idx += stride) {
        int64_t b = idx / D;
        int64_t k = idx % D;
        int64_t inFloatIdx = wsFloatOffset + (b * fftN + k) * 2;
        int64_t outFloatIdx = (b * D + k) * 2;

        gm_output[outFloatIdx]     = gm_workspace[inFloatIdx];
        gm_output[outFloatIdx + 1] = gm_workspace[inFloatIdx + 1];
    }
}

class FftR2CKernelMultiCore {
public:
    __aicore__ inline FftR2CKernelMultiCore() {}

    __aicore__ inline void Init(
        __gm__ float * __restrict__ gm_input,
        __gm__ float * __restrict__ gm_dft_matrix_array,
        __gm__ float * __restrict__ gm_tw_matrix_array,
        __gm__ float * __restrict__ gm_tw_post_process,
        __gm__ float * __restrict__ radix_list,
        __gm__ float * __restrict__ gm_output,
        __gm__ float * __restrict__ gm_workspace,
        __gm__ uint8_t * __restrict__ gm_tiling_para);

    __aicore__ inline void Process();

private:
    __gm__ float * __restrict__ gm_input_core_;
    __gm__ float * __restrict__ gm_dft_matrix_array_;
    __gm__ float * __restrict__ gm_tw_matrix_array_;
    __gm__ float * __restrict__ gm_tw_post_process_;
    __gm__ float * __restrict__ gm_output_core_;
    __gm__ float * __restrict__ gm_workspace_core_;

    int64_t batchSize_;
    int64_t fftN_;
    int32_t radixListLen_;
    int32_t isOddN_;  // 新增：标记是否为奇数 N
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

__aicore__ inline void FftR2CKernelMultiCore::Init(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ gm_tw_post_process,
    __gm__ float * __restrict__ radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para)
{
    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(gm_tiling_para);

    batchSize_ = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf));
    fftN_ = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 8));
    radixListLen_ = (*(__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + 16));
    isOddN_ = (*(__gm__ int32_t *)((__gm__ uint8_t *)tiling_buf + 1216));

    for (int32_t i = 0; i < 5; i++) {
        workspaceOffsets_[i] = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 24 + 8 * i));
    }

    int64_t baseOffset = 24 + 5 * 8;  // 64
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

    int64_t fftPointCount = isOddN_ ? fftN_ : (fftN_ / 2);
    int64_t perBatchComplexBytes = fftPointCount * sizeof(float) * 2;

    int64_t inputByteOffset = batchStart_ * fftN_ * sizeof(float);
    int64_t outputByteOffset = batchStart_ * (fftN_ / 2 + 1) * sizeof(float) * 2;  // 输出始终是 N//2+1

    gm_input_core_ = (__gm__ float *)((__gm__ uint8_t *)gm_input + inputByteOffset);
    gm_output_core_ = (__gm__ float *)((__gm__ uint8_t *)gm_output + outputByteOffset);
    gm_workspace_core_ = gm_workspace;
    gm_dft_matrix_array_ = gm_dft_matrix_array;
    gm_tw_matrix_array_ = gm_tw_matrix_array;
    gm_tw_post_process_ = gm_tw_post_process;

    localWorkspaceOffsets_[0] = workspaceOffsets_[0] + batchStart_ * perBatchComplexBytes;
    localWorkspaceOffsets_[1] = workspaceOffsets_[1] + batchStart_ * perBatchComplexBytes;
}

__aicore__ inline void FftR2CKernelMultiCore::Process()
{
    if (blockIdx_ > blockNum_ - 1) return;

    if (localBatchSize_ <= 0) {
        return;
    }

    if (isOddN_) {
        Simt::VF_CALL<FftR2COddPackRealToComplex>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            gm_input_core_,
            gm_workspace_core_,
            localBatchSize_,
            fftN_,
            localWorkspaceOffsets_[0]);
    } else {
        Simt::VF_CALL<FftR2CPackRealToComplex>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            gm_input_core_,
            gm_workspace_core_,
            localBatchSize_,
            fftN_,
            localWorkspaceOffsets_[0]);
    }

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

        Simt::VF_CALL<FftR2CStockhamForwardStage>(
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

    bool dataInWs1 = (radixListLen_ % 2 == 1);

    bool useWs1 = dataInWs1;
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

        Simt::VF_CALL<FftR2CTransposeStage>(
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

    if (isOddN_) {
        Simt::VF_CALL<FftR2COddTruncateOutput>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            gm_workspace_core_,
            gm_output_core_,
            localBatchSize_,
            fftN_,
            finalWsOffset);
    } else {
        Simt::VF_CALL<FftR2CPostProcess>(
            Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1},
            gm_workspace_core_,
            gm_tw_post_process_,
            gm_output_core_,
            localBatchSize_,
            fftN_,
            finalWsOffset);
    }

    return;
}

#endif

extern "C" __global__ __aicore__ void fft_r2c_multi_core(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_dft_matrix_array,
    __gm__ float * __restrict__ gm_tw_matrix_array,
    __gm__ float * __restrict__ gm_tw_post_process,
    __gm__ float * __restrict__ radix_list,
    __gm__ float * __restrict__ gm_output,
    __gm__ float * __restrict__ gm_workspace,
    __gm__ uint8_t * __restrict__ gm_tiling_para)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::GlobalTensor<uint64_t> global;
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(global);
    FftR2CKernelMultiCore kernel;
    kernel.Init(gm_input, gm_dft_matrix_array, gm_tw_matrix_array, gm_tw_post_process, radix_list,
                gm_output, gm_workspace, gm_tiling_para);
    kernel.Process();
#endif
}

#endif
