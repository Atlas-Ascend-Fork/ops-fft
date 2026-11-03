#ifndef FFT1_D_C2C_MIX_KERNEL_H
#define FFT1_D_C2C_MIX_KERNEL_H
// FftAllMixTilingData 统一取自公共定义（issue #254，原 arch35 本地副本已删除）
#include "fft_all_mix_tiling_data.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)


/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr int THREAD_NUM_RAD2 = 1024;
constexpr int THREAD_NUM_RAD3 = 512;
constexpr int THREAD_NUM = 256;
constexpr int32_t MAX_FFT_STAGES_TILING = 32;


template <bool SCALE_OUT, bool TRANSPOSE>
__simt_vf__ __launch_bounds__(THREAD_NUM_RAD2) inline void stockhamForwardRadix2(
    __gm__ float* in,
    __gm__ float* twMat,
    __gm__ float* out,
    int batch,
    int M,
    int N,
    int batchOffset,
    int len,
    int radix, // unused, should be 2
    int prev,
    int part
) {
    int blocks = N / len;

    int perBatch = blocks * prev;
    int total = batch * perBatch;

    float scale = 1.0f;
    if constexpr (SCALE_OUT) {
        scale = 1.0f / static_cast<float>(N);
    }

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int bid = i / perBatch;
        int rem = i - bid * perBatch;

        int block = rem / prev;
        int j = rem - block * prev;

        int batchBase = bid * N;

        int base = batchBase + block * prev + j;

        int in0 = base;
        int in1 = base + part;

        int inOff0 = in0 << 1;
        int inOff1 = in1 << 1;

        float uRe = in[inOff0];
        float uIm = in[inOff0 + 1];

        float vRe = in[inOff1];
        float vIm = in[inOff1 + 1];

        int twOff = (prev + j) << 1;

        float wRe = twMat[twOff];
        float wIm = twMat[twOff + 1];

        float tRe = vRe * wRe - vIm * wIm;
        float tIm = vRe * wIm + vIm * wRe;

        int y0 = block * len + j;
        int y1 = y0 + prev;

        int out_idx0;
        int out_idx1;
        if constexpr (TRANSPOSE) {
            int realBatch = bid + batchOffset;
            int realBid = realBatch / M;
            int x = realBatch - realBid * M;
            out_idx0 = (realBid * N + y0) * M + x;
            out_idx1 = (realBid * N + y1) * M + x;
        } else {
            out_idx0 = batchBase + y0;
            out_idx1 = batchBase + y1;
        }

        int outOff0 = out_idx0 << 1;
        int outOff1 = out_idx1 << 1;

        float y0Re = uRe + tRe;
        float y0Im = uIm + tIm;

        float y1Re = uRe - tRe;
        float y1Im = uIm - tIm;

        if constexpr (SCALE_OUT) {
            y0Re *= scale;
            y0Im *= scale;
            y1Re *= scale;
            y1Im *= scale;
        }

        out[outOff0]     = y0Re;
        out[outOff0 + 1] = y0Im;
        out[outOff1]     = y1Re;
        out[outOff1 + 1] = y1Im;
    }
}

template <bool SCALE_OUT, bool TRANSPOSE>
__simt_vf__ __launch_bounds__(THREAD_NUM_RAD3) inline void stockhamForwardRadix3(
    __gm__ float* in,
    __gm__ float* twMat,
    __gm__ float* dftMat,
    __gm__ float* out,
    int batch,
    int M,
    int N,
    int batchOffset,
    int len,
    int radix, // unused, should be 3
    int prev,
    int part
) {
    __ubuf__ float dft[2][3][3];

    for (int i = threadIdx.x; i < 3 * 3 * 2; i += blockDim.x) {
        int c = i & 1;
        int idx = i >> 1;

        int q = idx / 3;
        int p = idx - q * 3;

        dft[c][q][p] = dftMat[i];
    }

    asc_syncthreads();

    float scale = 1.0f;
    if constexpr (SCALE_OUT) {
        scale = 1.0f / static_cast<float>(N);
    }

    int blocks = N / len;
    int perBatch = blocks * prev;
    int total = batch * perBatch;

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int bid = i / perBatch;
        int rem = i - bid * perBatch;

        int block = rem / prev;
        int j = rem - block * prev;

        int batchBase = bid * N;
        int base = batchBase + block * prev + j;

        int in0 = base;
        int in1 = base + part;
        int in2 = base + part + part;

        int inOff0 = in0 << 1;
        int inOff1 = in1 << 1;
        int inOff2 = in2 << 1;

        float x0Re = in[inOff0];
        float x0Im = in[inOff0 + 1];

        float x1Re = in[inOff1];
        float x1Im = in[inOff1 + 1];

        float x2Re = in[inOff2];
        float x2Im = in[inOff2 + 1];

        float b0Re = x0Re;
        float b0Im = x0Im;

        int twOff1 = (prev + j) << 1;
        int twOff2 = ((prev << 1) + j) << 1;

        float w1Re = twMat[twOff1];
        float w1Im = twMat[twOff1 + 1];

        float w2Re = twMat[twOff2];
        float w2Im = twMat[twOff2 + 1];

        float b1Re = x1Re * w1Re - x1Im * w1Im;
        float b1Im = x1Re * w1Im + x1Im * w1Re;

        float b2Re = x2Re * w2Re - x2Im * w2Im;
        float b2Im = x2Re * w2Im + x2Im * w2Re;

        int outBase = block * len + j;

        int realBid = 0;
        int x = 0;
        if constexpr (TRANSPOSE) {
            int realBatch = bid + batchOffset;
            realBid = realBatch / M;
            x = realBatch - realBid * M;
        }

        // q = 0
        {
            float yRe = b0Re + b1Re + b2Re;
            float yIm = b0Im + b1Im + b2Im;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 1
        {
            float yRe = b0Re;
            float yIm = b0Im;

            float d11Re = dft[0][1][1];
            float d11Im = dft[1][1][1];

            float d12Re = dft[0][1][2];
            float d12Im = dft[1][1][2];

            yRe += b1Re * d11Re - b1Im * d11Im;
            yIm += b1Re * d11Im + b1Im * d11Re;

            yRe += b2Re * d12Re - b2Im * d12Im;
            yIm += b2Re * d12Im + b2Im * d12Re;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + prev;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 2
        {
            float yRe = b0Re;
            float yIm = b0Im;

            float d21Re = dft[0][2][1];
            float d21Im = dft[1][2][1];

            float d22Re = dft[0][2][2];
            float d22Im = dft[1][2][2];

            yRe += b1Re * d21Re - b1Im * d21Im;
            yIm += b1Re * d21Im + b1Im * d21Re;

            yRe += b2Re * d22Re - b2Im * d22Im;
            yIm += b2Re * d22Im + b2Im * d22Re;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + (prev << 1);
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }
    }

    asc_syncthreads();
}

template <bool SCALE_OUT, bool TRANSPOSE>
__simt_vf__ __launch_bounds__(THREAD_NUM) inline void stockhamForwardRadix5(
    __gm__ float* in,
    __gm__ float* twMat,
    __gm__ float* dftMat,
    __gm__ float* out,
    int batch,
    int M,
    int N,
    int batchOffset,
    int len,
    int radix, // unused, should be 5
    int prev,
    int part
) {
    __ubuf__ float dft[2][5][5];

    for (int i = threadIdx.x; i < 5 * 5 * 2; i += blockDim.x) {
        int c = i & 1;
        int idx = i >> 1;

        int q = idx / 5;
        int p = idx - q * 5;

        dft[c][q][p] = dftMat[i];
    }

    asc_syncthreads();

    float scale = 1.0f;
    if constexpr (SCALE_OUT) {
        scale = 1.0f / static_cast<float>(N);
    }

    int blocks = N / len;
    int perBatch = blocks * prev;
    int total = batch * perBatch;

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int bid = i / perBatch;
        int rem = i - bid * perBatch;

        int block = rem / prev;
        int j = rem - block * prev;

        int batchBase = bid * N;
        int base = batchBase + block * prev + j;

        int part2 = part + part;
        int part3 = part2 + part;
        int part4 = part3 + part;

        int inOff0 = base << 1;
        int inOff1 = (base + part) << 1;
        int inOff2 = (base + part2) << 1;
        int inOff3 = (base + part3) << 1;
        int inOff4 = (base + part4) << 1;

        float x0Re = in[inOff0];
        float x0Im = in[inOff0 + 1];

        float x1Re = in[inOff1];
        float x1Im = in[inOff1 + 1];

        float x2Re = in[inOff2];
        float x2Im = in[inOff2 + 1];

        float x3Re = in[inOff3];
        float x3Im = in[inOff3 + 1];

        float x4Re = in[inOff4];
        float x4Im = in[inOff4 + 1];

        float b0Re = x0Re;
        float b0Im = x0Im;

        int twOff1 = (prev + j) << 1;
        int twOff2 = ((prev << 1) + j) << 1;
        int twOff3 = (prev * 3 + j) << 1;
        int twOff4 = ((prev << 2) + j) << 1;

        float w1Re = twMat[twOff1];
        float w1Im = twMat[twOff1 + 1];

        float w2Re = twMat[twOff2];
        float w2Im = twMat[twOff2 + 1];

        float w3Re = twMat[twOff3];
        float w3Im = twMat[twOff3 + 1];

        float w4Re = twMat[twOff4];
        float w4Im = twMat[twOff4 + 1];

        float b1Re = x1Re * w1Re - x1Im * w1Im;
        float b1Im = x1Re * w1Im + x1Im * w1Re;

        float b2Re = x2Re * w2Re - x2Im * w2Im;
        float b2Im = x2Re * w2Im + x2Im * w2Re;

        float b3Re = x3Re * w3Re - x3Im * w3Im;
        float b3Im = x3Re * w3Im + x3Im * w3Re;

        float b4Re = x4Re * w4Re - x4Im * w4Im;
        float b4Im = x4Re * w4Im + x4Im * w4Re;

        int outBase = block * len + j;

        int realBid = 0;
        int x = 0;
        if constexpr (TRANSPOSE) {
            int realBatch = bid + batchOffset;
            realBid = realBatch / M;
            x = realBatch - realBid * M;
        }

        // q = 0
        {
            float yRe = b0Re + b1Re + b2Re + b3Re + b4Re;
            float yIm = b0Im + b1Im + b2Im + b3Im + b4Im;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 1
        {
            float yRe = b0Re;
            float yIm = b0Im;

            float dRe, dIm;

            dRe = dft[0][1][1]; dIm = dft[1][1][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][1][2]; dIm = dft[1][1][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][1][3]; dIm = dft[1][1][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][1][4]; dIm = dft[1][1][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + prev;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 2
        {
            float yRe = b0Re;
            float yIm = b0Im;

            float dRe, dIm;

            dRe = dft[0][2][1]; dIm = dft[1][2][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][2][2]; dIm = dft[1][2][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][2][3]; dIm = dft[1][2][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][2][4]; dIm = dft[1][2][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + (prev << 1);
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 3
        {
            float yRe = b0Re;
            float yIm = b0Im;

            float dRe, dIm;

            dRe = dft[0][3][1]; dIm = dft[1][3][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][3][2]; dIm = dft[1][3][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][3][3]; dIm = dft[1][3][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][3][4]; dIm = dft[1][3][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + prev * 3;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 4
        {
            float yRe = b0Re;
            float yIm = b0Im;

            float dRe, dIm;

            dRe = dft[0][4][1]; dIm = dft[1][4][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][4][2]; dIm = dft[1][4][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][4][3]; dIm = dft[1][4][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][4][4]; dIm = dft[1][4][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + (prev << 2);
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }
    }

    asc_syncthreads();
}

template <bool SCALE_OUT, bool TRANSPOSE>
__simt_vf__ __launch_bounds__(THREAD_NUM) inline void stockhamForwardRadix7(
    __gm__ float* in,
    __gm__ float* twMat,
    __gm__ float* dftMat,
    __gm__ float* out,
    int batch,
    int M,
    int N,
    int batchOffset,
    int len,
    int radix, // unused, should be 7
    int prev,
    int part
) {
    __ubuf__ float dft[2][7][7];

    // Keep full 7x7 load. Simple and tiny enough.
    for (int i = threadIdx.x; i < 7 * 7 * 2; i += blockDim.x) {
        int c = i & 1;
        int idx = i >> 1;

        int q = idx / 7;
        int p = idx - q * 7;

        dft[c][q][p] = dftMat[i];
    }

    asc_syncthreads();

    float scale = 1.0f;
    if constexpr (SCALE_OUT) {
        scale = 1.0f / static_cast<float>(N);
    }

    int blocks = N / len;
    int perBatch = blocks * prev;
    int total = batch * perBatch;

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int bid = i / perBatch;
        int rem = i - bid * perBatch;

        int block = rem / prev;
        int j = rem - block * prev;

        int batchBase = bid * N;
        int base = batchBase + block * prev + j;

        int p0 = base;
        int p1 = p0 + part;
        int p2 = p1 + part;
        int p3 = p2 + part;
        int p4 = p3 + part;
        int p5 = p4 + part;
        int p6 = p5 + part;

        int off0 = p0 << 1;
        int off1 = p1 << 1;
        int off2 = p2 << 1;
        int off3 = p3 << 1;
        int off4 = p4 << 1;
        int off5 = p5 << 1;
        int off6 = p6 << 1;

        float x0Re = in[off0], x0Im = in[off0 + 1];
        float x1Re = in[off1], x1Im = in[off1 + 1];
        float x2Re = in[off2], x2Im = in[off2 + 1];
        float x3Re = in[off3], x3Im = in[off3 + 1];
        float x4Re = in[off4], x4Im = in[off4 + 1];
        float x5Re = in[off5], x5Im = in[off5 + 1];
        float x6Re = in[off6], x6Im = in[off6 + 1];

        // twiddle p = 0 is always 1 + 0i
        float b0Re = x0Re;
        float b0Im = x0Im;

        int tw1 = (prev + j) << 1;
        int tw2 = ((prev << 1) + j) << 1;
        int tw3 = (prev * 3 + j) << 1;
        int tw4 = ((prev << 2) + j) << 1;
        int tw5 = (prev * 5 + j) << 1;
        int tw6 = (prev * 6 + j) << 1;

        float w1Re = twMat[tw1], w1Im = twMat[tw1 + 1];
        float w2Re = twMat[tw2], w2Im = twMat[tw2 + 1];
        float w3Re = twMat[tw3], w3Im = twMat[tw3 + 1];
        float w4Re = twMat[tw4], w4Im = twMat[tw4 + 1];
        float w5Re = twMat[tw5], w5Im = twMat[tw5 + 1];
        float w6Re = twMat[tw6], w6Im = twMat[tw6 + 1];

        float b1Re = x1Re * w1Re - x1Im * w1Im;
        float b1Im = x1Re * w1Im + x1Im * w1Re;

        float b2Re = x2Re * w2Re - x2Im * w2Im;
        float b2Im = x2Re * w2Im + x2Im * w2Re;

        float b3Re = x3Re * w3Re - x3Im * w3Im;
        float b3Im = x3Re * w3Im + x3Im * w3Re;

        float b4Re = x4Re * w4Re - x4Im * w4Im;
        float b4Im = x4Re * w4Im + x4Im * w4Re;

        float b5Re = x5Re * w5Re - x5Im * w5Im;
        float b5Im = x5Re * w5Im + x5Im * w5Re;

        float b6Re = x6Re * w6Re - x6Im * w6Im;
        float b6Im = x6Re * w6Im + x6Im * w6Re;

        int outBase = block * len + j;

        int realBid = 0;
        int x = 0;
        if constexpr (TRANSPOSE) {
            int realBatch = bid + batchOffset;
            realBid = realBatch / M;
            x = realBatch - realBid * M;
        }

        // q = 0:
        {
            float yRe = b0Re + b1Re + b2Re + b3Re + b4Re + b5Re + b6Re;
            float yIm = b0Im + b1Im + b2Im + b3Im + b4Im + b5Im + b6Im;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 1:
        {
            float yRe = b0Re;
            float yIm = b0Im;
            float dRe, dIm;

            dRe = dft[0][1][1]; dIm = dft[1][1][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][1][2]; dIm = dft[1][1][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][1][3]; dIm = dft[1][1][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][1][4]; dIm = dft[1][1][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            dRe = dft[0][1][5]; dIm = dft[1][1][5];
            yRe += b5Re * dRe - b5Im * dIm;
            yIm += b5Re * dIm + b5Im * dRe;

            dRe = dft[0][1][6]; dIm = dft[1][1][6];
            yRe += b6Re * dRe - b6Im * dIm;
            yIm += b6Re * dIm + b6Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + prev;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 2:
        {
            float yRe = b0Re;
            float yIm = b0Im;
            float dRe, dIm;

            dRe = dft[0][2][1]; dIm = dft[1][2][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][2][2]; dIm = dft[1][2][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][2][3]; dIm = dft[1][2][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][2][4]; dIm = dft[1][2][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            dRe = dft[0][2][5]; dIm = dft[1][2][5];
            yRe += b5Re * dRe - b5Im * dIm;
            yIm += b5Re * dIm + b5Im * dRe;

            dRe = dft[0][2][6]; dIm = dft[1][2][6];
            yRe += b6Re * dRe - b6Im * dIm;
            yIm += b6Re * dIm + b6Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + (prev << 1);
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 3:
        {
            float yRe = b0Re;
            float yIm = b0Im;
            float dRe, dIm;

            dRe = dft[0][3][1]; dIm = dft[1][3][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][3][2]; dIm = dft[1][3][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][3][3]; dIm = dft[1][3][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][3][4]; dIm = dft[1][3][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            dRe = dft[0][3][5]; dIm = dft[1][3][5];
            yRe += b5Re * dRe - b5Im * dIm;
            yIm += b5Re * dIm + b5Im * dRe;

            dRe = dft[0][3][6]; dIm = dft[1][3][6];
            yRe += b6Re * dRe - b6Im * dIm;
            yIm += b6Re * dIm + b6Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + prev * 3;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 4:
        {
            float yRe = b0Re;
            float yIm = b0Im;
            float dRe, dIm;

            dRe = dft[0][4][1]; dIm = dft[1][4][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][4][2]; dIm = dft[1][4][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][4][3]; dIm = dft[1][4][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][4][4]; dIm = dft[1][4][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            dRe = dft[0][4][5]; dIm = dft[1][4][5];
            yRe += b5Re * dRe - b5Im * dIm;
            yIm += b5Re * dIm + b5Im * dRe;

            dRe = dft[0][4][6]; dIm = dft[1][4][6];
            yRe += b6Re * dRe - b6Im * dIm;
            yIm += b6Re * dIm + b6Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + (prev << 2);
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 5:
        {
            float yRe = b0Re;
            float yIm = b0Im;
            float dRe, dIm;

            dRe = dft[0][5][1]; dIm = dft[1][5][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][5][2]; dIm = dft[1][5][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][5][3]; dIm = dft[1][5][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][5][4]; dIm = dft[1][5][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            dRe = dft[0][5][5]; dIm = dft[1][5][5];
            yRe += b5Re * dRe - b5Im * dIm;
            yIm += b5Re * dIm + b5Im * dRe;

            dRe = dft[0][5][6]; dIm = dft[1][5][6];
            yRe += b6Re * dRe - b6Im * dIm;
            yIm += b6Re * dIm + b6Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + prev * 5;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }

        // q = 6:
        {
            float yRe = b0Re;
            float yIm = b0Im;
            float dRe, dIm;

            dRe = dft[0][6][1]; dIm = dft[1][6][1];
            yRe += b1Re * dRe - b1Im * dIm;
            yIm += b1Re * dIm + b1Im * dRe;

            dRe = dft[0][6][2]; dIm = dft[1][6][2];
            yRe += b2Re * dRe - b2Im * dIm;
            yIm += b2Re * dIm + b2Im * dRe;

            dRe = dft[0][6][3]; dIm = dft[1][6][3];
            yRe += b3Re * dRe - b3Im * dIm;
            yIm += b3Re * dIm + b3Im * dRe;

            dRe = dft[0][6][4]; dIm = dft[1][6][4];
            yRe += b4Re * dRe - b4Im * dIm;
            yIm += b4Re * dIm + b4Im * dRe;

            dRe = dft[0][6][5]; dIm = dft[1][6][5];
            yRe += b5Re * dRe - b5Im * dIm;
            yIm += b5Re * dIm + b5Im * dRe;

            dRe = dft[0][6][6]; dIm = dft[1][6][6];
            yRe += b6Re * dRe - b6Im * dIm;
            yIm += b6Re * dIm + b6Im * dRe;

            if constexpr (SCALE_OUT) {
                yRe *= scale;
                yIm *= scale;
            }

            int y = outBase + prev * 6;
            int out_idx;
            if constexpr (TRANSPOSE) {
                out_idx = (realBid * N + y) * M + x;
            } else {
                out_idx = batchBase + y;
            }

            int outOff = out_idx << 1;
            out[outOff] = yRe;
            out[outOff + 1] = yIm;
        }
    }

    asc_syncthreads();
}

const int jTILE = 64;

template <bool SCALE_OUT, bool TRANSPOSE>
__simt_vf__ __launch_bounds__(THREAD_NUM) inline void stockhamForwardGeneral(
    __gm__ float* in,
    __gm__ float* twMat,
    __gm__ float* dftMat,
    __gm__ float* out,
    int batch,
    int M,
    int N,
    int batchOffset,
    int len,
    int radix,
    int prev,
    int part
) {
    __ubuf__ float dft[2][20][20];
    __ubuf__ float tw[2][20][jTILE];

    for (int i = threadIdx.x; i < radix * radix * 2; i += blockDim.x) {
        int c = i & 1;
        int idx = i >> 1;

        int q = idx / radix;
        int p = idx - q * radix;

        dft[c][q][p] = dftMat[i];
    }

    float scale = 1.0f;
    if constexpr (SCALE_OUT) {
        scale = 1.0f / static_cast<float>(N);
    }

    int blocks = N / len;

    for (int jBase = 0; jBase < prev; jBase += jTILE) {
        int tail = (prev - jBase < jTILE) ? (prev - jBase) : jTILE;

        int twCount = radix * tail * 2;

        for (int i = threadIdx.x; i < twCount; i += blockDim.x) {
            int c = i & 1;
            int id = i >> 1;

            int p = id / tail;
            int j = id - p * tail;

            int globalJ = jBase + j;

            tw[c][p][j] = twMat[(p * prev + globalJ) * 2 + c];
        }

        asc_syncthreads();

        int radixTail = radix * tail;
        int perBatch = blocks * radixTail;
        int total = batch * perBatch;

        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            int bid = i / perBatch;
            int rem = i - bid * perBatch;

            int block = rem / radixTail;
            int r = rem - block * radixTail;

            int q = r / tail;
            int j = r - q * tail;

            int globalJ = jBase + j;

            int batchBase = bid * N;
            int blockBaseIn = batchBase + block * prev + globalJ;

            int y = block * len + q * prev + globalJ;
            int out_idx;
            if constexpr (TRANSPOSE) {
                int realBatch = bid + batchOffset;
                int realBid = realBatch / M;
                int x = realBatch - realBid * M;  // realBatch % M
                // out[realBid][y][x]
                out_idx = (realBid * N + y) * M + x;
            } else {
                // out[bid][y]
                out_idx = batchBase + y;
            }

            float resRe = 0.0f;
            float resIm = 0.0f;

            int in_idx = blockBaseIn;

            { // p = 0 case, dft and tw are 0
                int inOff = in_idx << 1;
                float xRe = in[inOff];
                float xIm = in[inOff + 1];
                resRe += xRe;
                resIm += xIm;
                in_idx += part;
            }
            for (int p = 1; p < radix; ++p) {
                int inOff = in_idx << 1;

                float xRe = in[inOff];
                float xIm = in[inOff + 1];

                float twRe = tw[0][p][j];
                float twIm = tw[1][p][j];

                float dftRe = dft[0][q][p];
                float dftIm = dft[1][q][p];

                float bRe = xRe * twRe - xIm * twIm;
                float bIm = xRe * twIm + xIm * twRe;

                resRe += bRe * dftRe - bIm * dftIm;
                resIm += bRe * dftIm + bIm * dftRe;

                in_idx += part;
            }

            if constexpr (SCALE_OUT) {
                resRe *= scale;
                resIm *= scale;
            }

            int outOff = out_idx << 1;

            out[outOff]     = resRe;
            out[outOff + 1] = resIm;
        }

        asc_syncthreads();
    }
}

#endif

extern "C" __global__ __vector__ void fft_c2c_arch35_mix_multi_core(
    __gm__ float *input,
    __gm__ float *dftMatrix,
    __gm__ float *twMat,
    __gm__ int32_t *radixList,
    __gm__ float *output,
    __gm__ float *workspace,
    __gm__ uint8_t *tiling)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

    AscendC::GlobalTensor<uint64_t> global;
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(global);
    auto tilingData = reinterpret_cast<__gm__ FftAllMixTilingData *>(tiling);

    int32_t batchSize = static_cast<int32_t>(tilingData->batchSize);
    int32_t fftN = static_cast<int32_t>(tilingData->fftN);
    int32_t radixListLen = tilingData->radixListLen;
    int32_t outer = tilingData->outer;
    int32_t scaleOut = tilingData->scaleOut;
    int32_t transpose = tilingData->transpose;

    int32_t blockIdx = AscendC::GetBlockIdx();
    int32_t blockNum = AscendC::GetBlockNum();

    int32_t localBatch = (batchSize + blockNum - 1) / blockNum;
    int32_t batchStart = localBatch * blockIdx;

    if (batchStart >= batchSize) {
        return;
    }

    localBatch = (localBatch < batchSize - batchStart)
        ? localBatch
        : (batchSize - batchStart);

    int64_t localOffset = static_cast<int64_t>(batchStart) * fftN * 2;
    int64_t fullBufferFloats = static_cast<int64_t>(batchSize) * fftN * 2;

    __gm__ float *inputLocal = input + localOffset;
    __gm__ float *workspaceLocal0 = workspace + localOffset;
    __gm__ float *workspaceLocal1 = workspace + fullBufferFloats + localOffset;
    __gm__ float *outputLocal = output + localOffset;

    int32_t len = 1;
    int32_t twOffset = 0;
    int32_t dftOffset = 0;

    for (int32_t stage = 0; stage < radixListLen; ++stage) {
        int32_t radix = radixList[stage];

        int32_t prev = len;
        len *= radix;

        int32_t part = fftN / radix;
        bool isLast = (stage == radixListLen - 1);

        __gm__ float *stageInput = inputLocal;
        if (stage > 0) {
            stageInput = ((stage - 1) & 1) ? workspaceLocal1 : workspaceLocal0;
        }

        __gm__ float *stageOutput = outputLocal;

        if (isLast) {
            if (transpose) {
                stageOutput = output;
            } else {
                stageOutput = outputLocal;
            }
        } else {
            stageOutput = (stage & 1) ? workspaceLocal1 : workspaceLocal0;
        }

        __gm__ float *twStage = twMat + static_cast<int64_t>(twOffset) * 2;
        __gm__ float *dftStage = dftMatrix + static_cast<int64_t>(dftOffset) * 2;

        AscendC::PipeBarrier<PIPE_V>();

        if (isLast) {
            if (scaleOut) {
                if (transpose) {
                    if (radix == 2) {
                        asc_vf_call<stockhamForwardRadix2<true, true>>(
                            dim3(THREAD_NUM_RAD2),
                            stageInput,
                            twStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 3) {
                        asc_vf_call<stockhamForwardRadix3<true, true>>(
                            dim3(THREAD_NUM_RAD3),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 5) {
                        asc_vf_call<stockhamForwardRadix5<true, true>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 7) {
                        asc_vf_call<stockhamForwardRadix7<true, true>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else {
                        asc_vf_call<stockhamForwardGeneral<true, true>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    }
                } else {
                    if (radix == 2) {
                        asc_vf_call<stockhamForwardRadix2<true, false>>(
                            dim3(THREAD_NUM_RAD2),
                            stageInput,
                            twStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 3) {
                        asc_vf_call<stockhamForwardRadix3<true, false>>(
                            dim3(THREAD_NUM_RAD3),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 5) {
                        asc_vf_call<stockhamForwardRadix5<true, false>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 7) {
                        asc_vf_call<stockhamForwardRadix7<true, false>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else {
                        asc_vf_call<stockhamForwardGeneral<true, false>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    }
                }
            } else {
                if (transpose) {
                    if (radix == 2) {
                        asc_vf_call<stockhamForwardRadix2<false, true>>(
                            dim3(THREAD_NUM_RAD2),
                            stageInput,
                            twStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 3) {
                        asc_vf_call<stockhamForwardRadix3<false, true>>(
                            dim3(THREAD_NUM_RAD3),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 5) {
                        asc_vf_call<stockhamForwardRadix5<false, true>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 7) {
                        asc_vf_call<stockhamForwardRadix7<false, true>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else {
                        asc_vf_call<stockhamForwardGeneral<false, true>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    }
                } else {
                    if (radix == 2) {
                        asc_vf_call<stockhamForwardRadix2<false, false>>(
                            dim3(THREAD_NUM_RAD2),
                            stageInput,
                            twStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 3) {
                        asc_vf_call<stockhamForwardRadix3<false, false>>(
                            dim3(THREAD_NUM_RAD3),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 5) {
                        asc_vf_call<stockhamForwardRadix5<false, false>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else if (radix == 7) {
                        asc_vf_call<stockhamForwardRadix7<false, false>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    } else {
                        asc_vf_call<stockhamForwardGeneral<false, false>>(
                            dim3(THREAD_NUM),
                            stageInput,
                            twStage,
                            dftStage,
                            stageOutput,
                            localBatch,
                            outer,
                            fftN,
                            batchStart,
                            len,
                            radix,
                            prev,
                            part);
                    }
                }
            }
        } else {
            if (radix == 2) {
                asc_vf_call<stockhamForwardRadix2<false, false>>(
                    dim3(THREAD_NUM_RAD2),
                    stageInput,
                    twStage,
                    stageOutput,
                    localBatch,
                    outer,
                    fftN,
                    batchStart,
                    len,
                    radix,
                    prev,
                    part);
            } else if (radix == 3) {
                asc_vf_call<stockhamForwardRadix3<false, false>>(
                    dim3(THREAD_NUM_RAD3),
                    stageInput,
                    twStage,
                    dftStage,
                    stageOutput,
                    localBatch,
                    outer,
                    fftN,
                    batchStart,
                    len,
                    radix,
                    prev,
                    part);
            } else if (radix == 5) {
                asc_vf_call<stockhamForwardRadix5<false, false>>(
                    dim3(THREAD_NUM),
                    stageInput,
                    twStage,
                    dftStage,
                    stageOutput,
                    localBatch,
                    outer,
                    fftN,
                    batchStart,
                    len,
                    radix,
                    prev,
                    part);
            } else if (radix == 7) {
                asc_vf_call<stockhamForwardRadix7<false, false>>(
                    dim3(THREAD_NUM),
                    stageInput,
                    twStage,
                    dftStage,
                    stageOutput,
                    localBatch,
                    outer,
                    fftN,
                    batchStart,
                    len,
                    radix,
                    prev,
                    part);
            } else {
                asc_vf_call<stockhamForwardGeneral<false, false>>(
                    dim3(THREAD_NUM),
                    stageInput,
                    twStage,
                    dftStage,
                    stageOutput,
                    localBatch,
                    outer,
                    fftN,
                    batchStart,
                    len,
                    radix,
                    prev,
                    part);
            }
        }

        twOffset += radix * prev;
        dftOffset += radix * radix;
    }

#endif
}

#endif // FFT1_D_C2C_MIX_KERNEL_H
