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
#include "rfft1_d_r2c_fft4_tilingdata.h"
#include "rfft1_d_r2c_fft4_kernel.h"
#include "../rfft1_d_r2c_host_common.h"
#include <complex>
#include <cmath>
#include <limits>

namespace RfftHost = Rfft1DR2CCommonHost;
constexpr int TILE_LENGTH = RfftHost::TILE_LENGTH;

static bool GetFft4Factors(uint32_t n, int &n1, int &n2, int &n3, int &n4)
{
    if (n == 0) return false;
    int64_t fftN = static_cast<int64_t>(n);
    n1 = RfftHost::RootCeil(fftN, 4);
    if (n1 == 0 || n1 <= 0) return false;
    n2 = RfftHost::RootCeil(fftN / n1, 3);
    if (n2 == 0 || n2 <= 0) return false;
    n3 = RfftHost::RootCeil(fftN / n1 / n2, 2);
    if (n3 == 0 || n3 <= 0) return false;
    n4 = fftN / n1 / n2 / n3;
    return n4 > 0;
}

void GenerateTiling(int n, int n1, int n2, int n3, int n4, int blockNum, uint8_t *twiddleBuf, uint8_t *gatherBuf, uint8_t *eyeBuf) {
    if (n <= 0 || n1 <= 0 || n2 <= 0 || n3 <= 0 || n4 <= 0 || blockNum <= 0) {
        return;
    }
    static const double pi = acos(-1);
    typedef std::complex<float> C;
    int row = n1 * n2;
    int64_t colProduct = static_cast<int64_t>(n3) * n4;
    if (colProduct > std::numeric_limits<int>::max()) {
        return;
    }
    int col = static_cast<int>(colProduct);
    if (col <= 0) {
        return;
    }
    {
        int stepN = 128;
        int l2Step = std::min(row / stepN, 4194304 / 2 / col / stepN);
        if (l2Step <= 0) {
            l2Step = 1;
        }
        if (l2Step == 0) {
            return;
        }
        int l2Blocks = row / stepN / l2Step;
        RfftHost::GenerateBlockedTwiddle(n, row, col, blockNum, stepN, l2Step, twiddleBuf);
        {

            auto buf = reinterpret_cast<float *>(twiddleBuf + (blockNum * l2Blocks + 1) * col * sizeof(float) * 2);
            RfftHost::GenerateDft<>(buf, n1);

            buf += n1 * n1 * 3;
            RfftHost::GenerateDftWithTwiddle(buf, n2, n1);

            buf += n2 * n2 * n1 * 3;
            RfftHost::GenerateDft<>(buf, n3);

            buf += n3 * n3 * 3;
            RfftHost::GenerateDftWithTwiddle(buf, n4, n3);
        }
        {
            auto buf = reinterpret_cast<float *>(eyeBuf);
            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 32; ++j) {
                    if (i == j) buf[i * 32 + j] = 1;
                    else buf[i * 32 + j] = 0;
                }
            }
        }
    }
    RfftHost::GenerateGather(gatherBuf, TILE_LENGTH);
}

static aclError ExecuteFft4(RfftHost::DeviceBuffers &dev, uint32_t coreNum, void *stream,
                            float *y, size_t outputSize)
{
    Rfft1DR2CFft4Kernel::fft4_r2c<<<coreNum, nullptr, stream>>>(
        (__gm__ uint8_t *)dev.input.get(),
        (__gm__ uint8_t *)dev.output.get(),
        (__gm__ uint8_t *)dev.workspace.get(),
        (__gm__ uint8_t *)dev.twiddle.get(),
        (__gm__ uint8_t *)dev.gather.get(),
        (__gm__ uint8_t *)dev.eye.get(),
        (__gm__ uint8_t *)dev.tiling.get()
    );
    return dev.CopyOutput(y, outputSize, stream);
}

extern "C" aclError aclfftRfft1DR2CFft4(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream)
{
    int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    if (!GetFft4Factors(n, n1, n2, n3, n4)) return ACL_ERROR_INVALID_PARAM;
    uint32_t coreNum = RfftHost::GetCoreNum();

    int64_t fftN = static_cast<int64_t>(n);

    int row = n1 * n2;
    int col = n3 * n4;

    int stepN = 128;
    int l2Step = std::min(row / stepN, 4194304 / 2 / col / stepN);
    if (l2Step <= 0) {
        l2Step = 1;
    }
    if (l2Step == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    int l2Blocks = row / stepN / l2Step;

    int64_t halfN = fftN / 2;
    size_t inputSize = batches * fftN * sizeof(float);
    size_t outputSize = batches * (halfN + 1) * sizeof(float) * 2;
    size_t twiddleSize = col * (1 + coreNum * l2Blocks) * sizeof(float) * 2 + (n1 * n1 + n3 * n3) * sizeof(float) * 3 + (n2 * n2 * n1 + n4 * n4 * n3) * sizeof(float) * 3;
    size_t workSize = fftN * (batches + 2) * sizeof(float);
    size_t gatherSize = TILE_LENGTH * 2 * sizeof(uint32_t);
    size_t eyeSize = 32 * 32 * sizeof(float);
    uint32_t tilingSize = sizeof(Rfft1DR2CFftTilingData);

    Rfft1DR2CFftTilingData tilingData;
    RfftHost::InitializeTiling<4>(tilingData, fftN, batches, n1, n2, n3, n4);

    uint8_t* twiddleHost;
    uint8_t* gatherHost;
    uint8_t* eyeHost;

    CHECK_ACL(aclrtMallocHost((void**)(&twiddleHost), twiddleSize));
    CHECK_ACL(aclrtMallocHost((void**)(&gatherHost), gatherSize));
    CHECK_ACL(aclrtMallocHost((void**)(&eyeHost), eyeSize));
    GenerateTiling(fftN, n1, n2, n3, n4, coreNum, twiddleHost, gatherHost, eyeHost);

    RfftHost::DeviceBuffers dev;
    RfftHost::BufferSizes sizes{inputSize, outputSize, workSize, twiddleSize, gatherSize, tilingSize, eyeSize};
    CHECK_ACL(dev.Prepare(sizes, x, twiddleHost, gatherHost, &tilingData, eyeHost));

    CHECK_ACL(ExecuteFft4(dev, coreNum, stream, y, outputSize));

    return RfftHost::FreeHostBuffers(twiddleHost, gatherHost, eyeHost);
}
