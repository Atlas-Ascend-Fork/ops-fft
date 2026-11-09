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
#include "rfft1_d_r2c_fft2_tilingdata.h"
#include "rfft1_d_r2c_fft2_kernel.h"
#include "../rfft1_d_r2c_host_common.h"
#include <complex>
#include <cmath>

namespace RfftHost = Rfft1DR2CCommonHost;
constexpr int TILE_LENGTH = RfftHost::TILE_LENGTH;

void GenerateTiling(int n, int n1, int n2, int blockNum, uint8_t *twiddleBuf, uint8_t *gatherBuf) {
    if (n <= 0 || n1 <= 0 || n2 <= 0 || blockNum <= 0) {
        return;
    }
    static const double pi = acos(-1);
    typedef std::complex<float> C;
    RfftHost::GenerateTwiddle(reinterpret_cast<float *>(twiddleBuf), n, n1, n2);
    {

        auto buf = reinterpret_cast<float *>(twiddleBuf + n * sizeof(float) * 2);
        RfftHost::GenerateDft<>(buf, n1);

        buf += n1 * n1 * 3;
        RfftHost::GenerateDft<>(buf, n2);

    }
    int calcN = std::min(n / 2, TILE_LENGTH);
    RfftHost::GenerateGather(gatherBuf, calcN);
}

// 因子分解与乘积回等校验（超大函数拆分，对齐 r2c_fft4 GetFft4Factors 先例）：
// RootCeil 初因子、n2 推导与因子乘积回等（issue #292：非 2 幂 n 的分解乘积
// 不等于 n，如 n=6 得 4*1=4 != 6，按失真结构静默算错，显式拒绝）
static bool GetFft2Factors(uint32_t n, int &n1, int &n2)
{
    int64_t fftN = static_cast<int64_t>(n);
    n1 = RfftHost::RootCeil(fftN, 2);
    if (n1 <= 0) {
        return false;
    }
    n2 = fftN / n1;
    if (n2 <= 0) {
        return false;
    }
    if (static_cast<int64_t>(n1) * n2 != fftN) {
        std::cerr << "[ops-fft] aclfftRfft1DR2CFft2: n=" << n << " factors " << n1 << "*" << n2
                  << " do not reproduce n" << std::endl;
        return false;
    }
    return true;
}

extern "C" aclError aclfftRfft1DR2CFft2(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream)
{
    // n==0/batches==0 与 batch*(n+2) 上界门禁（issue #239，防直连绕过 exec 层）
    CHECK_ACL(RfftHost::ValidateR2CPowerOfTwoEntry("aclfftRfft1DR2CFft2", n, batches, isForward));
    // fft2 档位域校验（issue #294）：与 exec 分发层一致（DispatchPowerOfTwoR2C 将
    // (128, 16384] 路由到 fft2，n<=128 走 fft1）。n 超上界时内核 blockN = 2621440/N
    // 截断为 0，i/blockN 除零且 vec/cube 跨核互等死锁；n<=128 直连实测内核挂死
    // （master 即存在，blockN 大于 batch/2 的 warmup 组合）。入口按分发域整体
    // 门禁提前拒绝，内核侧另有 blockN==0 防护双保险
    if (n <= 128 || n > 16384) {
        std::cerr << "[ops-fft] aclfftRfft1DR2CFft2: n=" << n
                  << " is not supported (fft2 path supports 128 < n <= 16384)" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    uint32_t coreNum = RfftHost::GetCoreNum();

    int64_t fftN = static_cast<int64_t>(n);
    int n1 = 0;
    int n2 = 0;
    if (!GetFft2Factors(n, n1, n2)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    Rfft1DR2CFftTilingData tilingData;
    RfftHost::InitializeTiling<2>(tilingData, fftN, batches, n1, n2);

    int64_t halfN = fftN / 2;
    size_t inputSize = batches * fftN * sizeof(float);
    size_t outputSize = batches * (halfN + 1) * sizeof(float) * 2;
    size_t twiddleSize = fftN * 2 * sizeof(float) + n1 * n1 * 3 * sizeof(float) + n2 * n2 * 3 * sizeof(float);
    size_t workSize = fftN * sizeof(float);
    size_t gatherSize = TILE_LENGTH * 2 * sizeof(uint32_t);
    uint32_t tilingSize = sizeof(Rfft1DR2CFftTilingData);

    // host pinned 缓冲纳入 RAII（issue #223）：任一 CHECK_ACL 早退路径自动释放
    RfftHost::HostBuffers host;
    CHECK_ACL(host.Allocate(twiddleSize, gatherSize));
    GenerateTiling(fftN, n1, n2, coreNum, static_cast<uint8_t *>(host.twiddle.get()),
                   static_cast<uint8_t *>(host.gather.get()));

    RfftHost::DeviceBuffers dev;
    RfftHost::BufferSizes sizes{inputSize, outputSize, workSize, twiddleSize, gatherSize, tilingSize, 0};
    CHECK_ACL(dev.Prepare(sizes, x, host.twiddle.get(), host.gather.get(), &tilingData));

    // 输出缓冲清零（issue #293）：UnPack2 每行只覆盖前 N 个 float，Nyquist 实部由
    // SetValueBatch 补写 +N 槽位，+N+1 槽位（Nyquist 虚部，实输入恒 0）无人写入，
    // 输出含未初始化内存；对齐 fft1 路径 issue #215 的清零口径
    CHECK_ACL(aclrtMemset(dev.output.get(), outputSize, 0, outputSize));

    Rfft1DR2CFft2Kernel::fft2_r2c<<<coreNum, nullptr, stream>>>(
        (__gm__ uint8_t *)dev.input.get(),
        (__gm__ uint8_t *)dev.output.get(),
        (__gm__ uint8_t *)dev.workspace.get(),
        (__gm__ uint8_t *)dev.twiddle.get(),
        (__gm__ uint8_t *)dev.gather.get(),
        (__gm__ uint8_t *)dev.tiling.get()
    );

    return RfftHost::CompleteFft(dev, y, outputSize, stream);
}
