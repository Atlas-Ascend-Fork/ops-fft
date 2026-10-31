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
 * See COPYING-PACO (full GPL) and LICENSE-PACO (commercial terms) at the repo root.
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
 * 详情见仓库根目录 COPYING-PACO（GPL 全文）与 LICENSE-PACO（商业条款）。
 * 商业授权联系：yuantang@fudan.edu.cn
 */

#include "../rfft1_d.h"
#include "fft_common_core.h"
#include "rfft1_d_r2c_fft1_tilingdata.h"
#include "rfft1_d_r2c_fft1_kernel.h"
#include "../rfft1_d_r2c_host_common.h"
#include <complex>
#include <cmath>

namespace RfftHost = Rfft1DR2CCommonHost;
constexpr int TILE_LENGTH = RfftHost::TILE_LENGTH;

void GenerateTiling(int n, uint8_t *twiddleBuf, uint8_t *gatherBuf) {
    if (n == 0) {
        return;
    }
    if (n <= 0) {
        return;
    }
    static const double pi = acos(-1);
    typedef std::complex<float> C;
    {

        auto buf = reinterpret_cast<float *>(twiddleBuf);
        RfftHost::GenerateDft<true>(buf, n);
    }
    {
        static constexpr int TILE_LENGTH = 4096;
        auto buf = reinterpret_cast<uint32_t *>(gatherBuf);
        int m = TILE_LENGTH / n;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                buf[j * m + i] = (i * n + j) * sizeof(float);
            }
        }
        buf += TILE_LENGTH * 2;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n + 2; ++j) {
                if (j % 2 == 0) {
                    buf[i * (n + 2) + j] = ((j / 2) * m + i) * sizeof(float);
                } else {
                    buf[i * (n + 2) + j] = ((j / 2 + n / 2 + 1) * m + i) * sizeof(float);
                }
            }
        }
    }
}

extern "C" aclError aclfftRfft1DR2CFft1(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream)
{
    if (n == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint32_t coreNum = RfftHost::GetCoreNum();

    int64_t fftN = static_cast<int64_t>(n);

    int64_t halfN = fftN / 2;
    size_t inputSize = batches * fftN * sizeof(float);
    size_t outputSize = batches * (halfN + 1) * sizeof(float) * 2;
    size_t twiddleSize = (fftN * fftN) * 2 * sizeof(float);
    size_t gatherSize = TILE_LENGTH * 4 * sizeof(uint32_t);
    uint32_t tilingSize = sizeof(Rfft1DR2CFftTilingData);

    Rfft1DR2CFftTilingData tilingData;
    tilingData.N = fftN;
    tilingData.batch = batches;

    // host pinned 缓冲纳入 RAII（issue #223）：任一 CHECK_ACL 早退路径自动释放，
    // 不再依赖函数末尾的显式 aclrtFreeHost（原早退路径会跳过释放造成泄漏）
    RfftHost::HostBuffers host;
    CHECK_ACL(host.Allocate(twiddleSize, gatherSize));
    GenerateTiling(fftN, static_cast<uint8_t *>(host.twiddle.get()), static_cast<uint8_t *>(host.gather.get()));

    RfftHost::DeviceBuffers dev;
    RfftHost::BufferSizes sizes{inputSize, outputSize, 0, twiddleSize, gatherSize, tilingSize, 0};
    CHECK_ACL(dev.Prepare(sizes, x, host.twiddle.get(), host.gather.get(), &tilingData));
    // cube 路径 Fixpipe 每行只写前 N+1 个 float，输出行宽 N+2 中下标 N+1 的 Nyquist
    // 虚部槽位（Im(X[N/2])，实输入恒为 0）无人写入，来自未初始化内存（issue #215，
    // 旧 DFT 路径曾显式写 0）。启动前将输出缓冲清零，保证该槽位确定性为 0。
    CHECK_ACL(aclrtMemset(dev.output.get(), outputSize, 0, outputSize));

    Rfft1DR2CFft1Kernel::fft1_r2c<<<coreNum, nullptr, stream>>>(
        (__gm__ uint8_t *)dev.input.get(),
        (__gm__ uint8_t *)dev.output.get(),
        (__gm__ uint8_t *)dev.twiddle.get(),
        (__gm__ uint8_t *)dev.gather.get(),
        (__gm__ uint8_t *)dev.tiling.get()
    );

    CHECK_ACL(dev.CopyOutput(y, outputSize, stream));

    return ACL_SUCCESS;
}
