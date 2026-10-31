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
#include "rfft1_d_r2c_fft3_1_tilingdata.h"
#include "rfft1_d_r2c_fft3_1_kernel.h"
#include "../rfft1_d_r2c_host_common.h"
#include <complex>
#include <cmath>

namespace RfftHost = Rfft1DR2CCommonHost;
constexpr int TILE_LENGTH = RfftHost::TILE_LENGTH;

extern "C" aclError aclfftRfft1DR2CFft31(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream)
{
    return RfftHost::RunFft3<Rfft1DR2CFftTilingData>(x, y, n, batches, isForward, stream,
        [](uint32_t coreNum, void *stream, RfftHost::DeviceBuffers &dev) {
            Rfft1DR2CFft31Kernel::fft3_1_r2c<<<coreNum, nullptr, stream>>>(
                (__gm__ uint8_t *)dev.input.get(), (__gm__ uint8_t *)dev.output.get(),
                (__gm__ uint8_t *)dev.workspace.get(), (__gm__ uint8_t *)dev.twiddle.get(),
                (__gm__ uint8_t *)dev.gather.get(), (__gm__ uint8_t *)dev.tiling.get());
        });
}
