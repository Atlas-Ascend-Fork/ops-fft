/*
 * PACO FFT — An I/O-Efficient Parallel FFT for Heterogeneous Architectures via a Single Global Exchange and Implementation
 * 
 * Copyright (c) 2026 Shina Guo <fs20091003@163.com>, Yuan Tang <yuantang@fudan.edu.cn>
 * All rights reserved.
 *
 * This software is dual-licensed:
 *   1) Community license: use, modify, and redistribute freely under the
 *      GNU General Public License v3 or later (GPL-3.0-or-later);
 *   2) Commercial license: for closed-source or commercial use / redistribution,
 *      contact the copyright holder for a separate license.
 * You may choose either. Default terms are GPL-3.0-or-later.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you may redistribute it and/or modify it under
 * the terms of the GPL. It is distributed "as is", without warranty.
 * See COPYING (full GPL) and COMMERCIAL-LICENSE (commercial terms) at the repo root.
 * Commercial licensing contact: yuantang@fudan.edu.cn
 *
 * -------------------------------------------------------------------------
 * 中文版 (Chinese):
 *
 * PACO FFT — An I/O-Efficient Parallel FFT for Heterogeneous Architectures via a Single Global Exchange and Implementation
 *
 * 本软件采用双重许可（dual licensing）：
 *   1) 社区许可：依据 GNU 通用公共许可证第 3 版或（你选择的）更新版本
 *      （GPL-3.0-or-later）自由使用、修改与再分发；
 *   2) 商业许可：闭源商用 / 再分发请与版权人联系获取单独授权。
 * 二者任选其一。默认条款以 GPL-3.0-or-later 为准。
 *
 * 本程序是自由软件，你可依 GPL 条款重新发布或修改；基于“原样”提供，无担保。
 * 详情见仓库根目录 COPYING（GPL 全文）与 COMMERCIAL-LICENSE（商业条款）。
 * 商业授权联系：yuantang@fudan.edu.cn
 */

#include "../rfft1_d.h"
#include "fft_common_core.h"
#include "rfft1_d_r2c_fft5_tilingdata.h"
#include "rfft1_d_r2c_fft5_kernel.h"
#include "../rfft1_d_r2c_host_common.h"
#include <complex>
#include <cmath>

namespace RfftHost = Rfft1DR2CCommonHost;
constexpr int TILE_LENGTH = RfftHost::TILE_LENGTH;

static bool GetFft5Factors(uint32_t n, int &halfN, int &n1, int &n2, int &n3, int &n4, int &n5)
{
    if (n < 4) return false;
    halfN = static_cast<int>(n / 2);
    n1 = RfftHost::RootCeil(halfN, 5);
    if (n1 == 0 || n1 <= 0) return false;
    n2 = RfftHost::RootCeil(halfN / n1, 4);
    if (n2 == 0 || n2 <= 0) return false;
    n3 = RfftHost::RootCeil(halfN / n1 / n2, 3);
    if (n3 == 0 || n3 <= 0) return false;
    n4 = RfftHost::RootCeil(halfN / n1 / n2 / n3, 2);
    if (n4 == 0 || n4 <= 0) return false;
    n5 = halfN / n1 / n2 / n3 / n4;
    return n5 > 0;
}

static void GenerateVectorTwiddle(__uint32_t n, __uint32_t n1, __uint32_t n2, __uint32_t n3,
                                  __uint32_t n4, __uint32_t n5, int blockNum, uint8_t *twiddleBuf)
{
    int row = n1 * n2;
    int col = n3 * n4 * n5;
    int stepN = 128;
    int l2Step = 1;
    int l2Blocks = row / stepN / l2Step;
    RfftHost::GenerateBlockedTwiddle(n, row, col, blockNum, stepN, l2Step, twiddleBuf);

    static constexpr uint32_t vectorTileLength = 4096;
    int calcN = std::max<uint32_t>(1, std::min(n / 2, vectorTileLength));
    int vecNum = blockNum * 2;
    for (int blockIdx = 0; blockIdx < vecNum; ++blockIdx) {
        auto twiddleReal = reinterpret_cast<float *>(twiddleBuf + (blockNum * l2Blocks + 1) * col * sizeof(float) * 2 + blockIdx * vectorTileLength * sizeof(float));
        auto twiddleImag = reinterpret_cast<float *>(twiddleBuf + (blockNum * l2Blocks + 1) * col * sizeof(float) * 2 + blockIdx * vectorTileLength * sizeof(float) + vecNum * vectorTileLength * sizeof(float));
        int totalNum = n / 2 / calcN;
        int averageNum = totalNum / vecNum;
        int remainderNum = totalNum % vecNum;
        int startIdx = (averageNum * blockIdx + std::min(blockIdx, remainderNum)) * calcN;
        int calcNum = (averageNum + (blockIdx < remainderNum)) * calcN;
        (void)calcNum;
        for (int i = 0; i < calcN; ++i) {
            int k = startIdx + i;
            double angle = -2 * acos(-1) * k / (n * 2);
            twiddleReal[i] = static_cast<float>(cos(angle) / 2);
            twiddleImag[i] = static_cast<float>(sin(angle) / 2);
        }
    }
}

static void GenerateDftTables(uint8_t *twiddleBuf, __uint32_t n, __uint32_t n1, __uint32_t n2,
                              __uint32_t n3, __uint32_t n4, __uint32_t n5, int blockNum)
{
    int row = n1 * n2;
    int col = n3 * n4 * n5;
    int l2Blocks = row / 128;
    static constexpr uint32_t vectorTileLength = 4096;
    auto buf = reinterpret_cast<float *>(twiddleBuf + (blockNum * l2Blocks + 1) * col * sizeof(float) * 2 + (blockNum * 2) * vectorTileLength * sizeof(float) * 2);
    RfftHost::GenerateDft<>(buf, n1);
    buf += n1 * n1 * 3;
    RfftHost::GenerateDftWithTwiddle(buf, n2, n1);
    buf += n2 * n2 * n1 * 3;
    RfftHost::GenerateDft<>(buf, n3);
    buf += n3 * n3 * 3;
    RfftHost::GenerateDftWithTwiddle(buf, n4, n3);

    auto genDftWithTwiddle2 = [](float *matrixBuf, int dftN, int m1, int m2, int ifTranspose) {
        if (dftN <= 0 || m1 <= 0 || m2 <= 0) return;
        auto curBuf = matrixBuf;
        int64_t m = static_cast<int64_t>(m1) * m2;
        if (ifTranspose) std::swap(m1, m2);
        for (int k1 = 0; k1 < m1; ++k1) {
            for (int k2 = 0; k2 < m2; ++k2) {
                int k = ifTranspose ? k2 * m1 + k1 : k1 * m2 + k2;
                RfftHost::GenerateDftTwiddleMatrix(curBuf, dftN, m, k);
                curBuf += dftN * dftN * 3;
            }
        }
    };
    buf += n4 * n4 * n3 * 3;
    genDftWithTwiddle2(buf, n5, n4, n3, 1);
}

static void GenerateTileRotations(uint8_t *twiddleBuf, __uint32_t n, __uint32_t n1, __uint32_t n2,
                                  __uint32_t n3, __uint32_t n4, __uint32_t n5, int blockNum)
{
    typedef std::complex<float> C;
    static constexpr uint32_t vectorTileLength = 4096;
    int row = n1 * n2;
    int col = n3 * n4 * n5;
    int l2Blocks = row / 128;
    int calcN = std::max<uint32_t>(1, std::min(n / 2, vectorTileLength));
    int totalNum = n / 2 / calcN;
    uint64_t matFloats = (static_cast<uint64_t>(n1) * n1 + static_cast<uint64_t>(n2) * n2 * n1 +
                          static_cast<uint64_t>(n3) * n3 + static_cast<uint64_t>(n4) * n4 * n3 +
                          static_cast<uint64_t>(n5) * n5 * n4 * n3) * 3ULL;
    auto sbuf = reinterpret_cast<float *>(twiddleBuf + static_cast<uint64_t>(col) * (blockNum * l2Blocks + 1) * sizeof(float) * 2 +
                 static_cast<uint64_t>(blockNum * 2) * vectorTileLength * sizeof(float) * 2 + matFloats * sizeof(float));
    static const double pi = acos(-1);
    for (int j = 0; j < totalNum; ++j) {
        C wk(cos(pi * j * calcN * 2 / (n * 2)), -sin(pi * j * calcN * 2 / (n * 2)));
        sbuf[j] = wk.real();
        sbuf[totalNum + j] = wk.imag();
    }
}

void GenerateTiling(__uint32_t n, __uint32_t n1, __uint32_t n2, __uint32_t n3, __uint32_t n4, __uint32_t n5, int blockNum, uint8_t *twiddleBuf, uint8_t *gatherBuf) {
    if (n == 0 || n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0 || n5 == 0 || blockNum <= 0) {
        return;
    }
    GenerateVectorTwiddle(n, n1, n2, n3, n4, n5, blockNum, twiddleBuf);
    GenerateDftTables(twiddleBuf, n, n1, n2, n3, n4, n5, blockNum);
    GenerateTileRotations(twiddleBuf, n, n1, n2, n3, n4, n5, blockNum);
    RfftHost::GenerateGather(gatherBuf, TILE_LENGTH);
}

extern "C" aclError aclfftRfft1DR2CFft5(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream)
{
    int halfN = 0, n1 = 0, n2 = 0, n3 = 0, n4 = 0, n5 = 0;
    if (!GetFft5Factors(n, halfN, n1, n2, n3, n4, n5)) return ACL_ERROR_INVALID_PARAM;
    uint32_t coreNum = RfftHost::GetCoreNum();

    int64_t fftN = static_cast<int64_t>(n);

    int row = n1 * n2;
    int col = n3 * n4 * n5;

    int stepN = 128;
    int l2Step = 1;
    int l2Blocks = row / stepN / l2Step;

    size_t inputSize = batches * fftN * sizeof(float);
    size_t outputSize = batches * (halfN + 1) * sizeof(float) * 2;
    size_t twiddleSize = col * (1 + coreNum * l2Blocks) * sizeof(float) * 2 + (coreNum * 2) * TILE_LENGTH * sizeof(float) * 2 + (n1 * n1 + n2 * n2 * n1 + n3 * n3 + n4 * n4 * n3 + n5 * n5 * n4 * n3) * sizeof(float) * 3 + static_cast<size_t>(halfN / 2 / std::min<int64_t>(halfN / 2, TILE_LENGTH)) * 2 * sizeof(float);
    size_t workSize = static_cast<size_t>(halfN) * 4 * sizeof(float);
    size_t gatherSize = TILE_LENGTH * 2 * sizeof(uint32_t);
    size_t eyeSize = 32 * 32 * sizeof(float);
    uint32_t tilingSize = sizeof(Rfft1DR2CFftTilingData);

    Rfft1DR2CFftTilingData tilingData;
    RfftHost::InitializeTiling<5>(tilingData, fftN, batches, n1, n2, n3, n4, n5);

    uint8_t* twiddleHost;
    uint8_t* gatherHost;

    CHECK_ACL(aclrtMallocHost((void**)(&twiddleHost), twiddleSize));
    CHECK_ACL(aclrtMallocHost((void**)(&gatherHost), gatherSize));
    GenerateTiling(halfN, n1, n2, n3, n4, n5, coreNum, twiddleHost, gatherHost);

    RfftHost::DeviceBuffers dev;
    RfftHost::BufferSizes sizes{inputSize, outputSize, workSize, twiddleSize, gatherSize, tilingSize, 0};
    CHECK_ACL(dev.Prepare(sizes, x, twiddleHost, gatherHost, &tilingData));

    Rfft1DR2CFft5Kernel::fft5_r2c<<<coreNum, nullptr, stream>>>(
        (__gm__ uint8_t *)dev.input.get(),
        (__gm__ uint8_t *)dev.output.get(),
        (__gm__ uint8_t *)dev.workspace.get(),
        (__gm__ uint8_t *)dev.twiddle.get(),
        (__gm__ uint8_t *)dev.gather.get(),
        (__gm__ uint8_t *)dev.tiling.get()
    );

    return RfftHost::CompleteFft(dev, y, outputSize, stream, twiddleHost, gatherHost);
}
