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

#ifndef RFFT1_D_R2C_HOST_COMMON_H
#define RFFT1_D_R2C_HOST_COMMON_H

#include "fft_common_core.h"
#include <complex>
#include <memory>

namespace Rfft1DR2CCommonHost {

constexpr int TILE_LENGTH = 4096;

inline uint32_t GetCoreNum()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = platform->GetCoreNumAic();
    return coreNum == 0 ? 1 : coreNum;
}

inline int RootCeil(int x, int k)
{
    if (k == 0) {
        return 1;
    }
    if (x <= 0 || k <= 0) {
        return 1;
    }
    int root = static_cast<int>(pow(x, 1.0 / k));
    int rt = 1 << (31 - __builtin_clz(root));
    if (static_cast<int>(pow(rt, k)) != x) rt <<= 1;
    return rt;
}

// Keep the interleaved FFT1 matrix separate from the three-plane Cube layout.
template<bool Interleaved = false>
inline void GenerateDft(float *buf, int dftN)
{
    if (dftN == 0) return;
    if (dftN <= 0) return;
    static const double pi = acos(-1);
    for (int i = 0; i < dftN; ++i) {
        for (int j = 0; j < dftN; ++j) {
            std::complex<float> wk(cos(pi * i * j * 2 / dftN), -sin(pi * i * j * 2 / dftN));
            int offset = i * dftN + j;
            if (Interleaved) {
                buf[offset * 2] = wk.real();
                buf[offset * 2 + 1] = wk.imag();
            } else {
                buf[offset] = wk.real();
                buf[offset + dftN * dftN] = wk.imag();
                buf[offset + dftN * dftN * 2] = -wk.imag();
            }
        }
    }
}

inline void GenerateDftTwiddleMatrix(float *buf, int dftN, int64_t m, int k)
{
    if (dftN == 0) return;
    if (dftN <= 0 || m <= 0) return;
    static const double pi = acos(-1);
    for (int i = 0; i < dftN; ++i) {
        for (int j = 0; j < dftN; ++j) {
            std::complex<float> wk(cos(pi * i * j * 2 / dftN + pi * k * j * 2 / (static_cast<double>(dftN) * m)),
                                   -sin(pi * i * j * 2 / dftN + pi * k * j * 2 / (static_cast<double>(dftN) * m)));
            buf[i * dftN + j] = wk.real();
            buf[i * dftN + j + dftN * dftN] = wk.imag();
            buf[i * dftN + j + dftN * dftN * 2] = -wk.imag();
        }
    }
}

inline void GenerateDftWithTwiddle(float *buf, int dftN, int m)
{
    if (dftN <= 0 || m <= 0) return;
    for (int k = 0; k < m; ++k) {
        GenerateDftTwiddleMatrix(buf, dftN, m, k);
        buf += dftN * dftN * 3;
    }
}

inline void GenerateTwiddle(float *buf, int n, int row, int col)
{
    if (n == 0) return;
    if (n <= 0) return;
    static const double pi = acos(-1);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            std::complex<float> wk(cos(pi * i * j * 2 / n), -sin(pi * i * j * 2 / n));
            buf[i * col + j] = wk.real();
            buf[i * col + j + n] = wk.imag();
        }
    }
}

inline void GenerateGather(uint8_t *gatherBuf, int calcN)
{
    auto buf = reinterpret_cast<uint32_t *>(gatherBuf);
    for (int i = 0; i < calcN; ++i) buf[i] = (calcN - i - 1) * sizeof(float);
    buf += TILE_LENGTH;
    for (int i = 0; i < TILE_LENGTH / 2; ++i) {
        buf[i * 2] = i * sizeof(float);
        buf[i * 2 + 1] = (i + TILE_LENGTH) * sizeof(float);
    }
}

struct BufferSizes {
    size_t input, output, workspace, twiddle, gather, tiling, eye;
};

struct DeviceBuffers {
    using Buffer = std::unique_ptr<void, AclrtFreeDeleter>;
    Buffer input, output, workspace, twiddle, gather, eye, tiling;

    aclError Prepare(const BufferSizes &sizes, const void *x, const void *twiddleHost,
                     const void *gatherHost, const void *tilingHost, const void *eyeHost = nullptr)
    {
        CHECK_ACL(Allocate(input, sizes.input));
        CHECK_ACL(Allocate(output, sizes.output));
        if (sizes.workspace != 0) CHECK_ACL(Allocate(workspace, sizes.workspace));
        CHECK_ACL(Allocate(twiddle, sizes.twiddle));
        CHECK_ACL(Allocate(gather, sizes.gather));
        if (sizes.eye != 0) CHECK_ACL(Allocate(eye, sizes.eye));
        CHECK_ACL(Allocate(tiling, sizes.tiling));
        CHECK_ACL(Copy(input, sizes.input, x));
        CHECK_ACL(Copy(twiddle, sizes.twiddle, twiddleHost));
        CHECK_ACL(Copy(gather, sizes.gather, gatherHost));
        if (sizes.eye != 0) CHECK_ACL(Copy(eye, sizes.eye, eyeHost));
        return Copy(tiling, sizes.tiling, tilingHost);
    }

    aclError CopyOutput(void *y, size_t size, void *stream)
    {
        CHECK_ACL(aclrtSynchronizeStream(stream));
        return aclrtMemcpy(y, size, output.get(), size, ACL_MEMCPY_DEVICE_TO_HOST);
    }

private:
    static aclError Allocate(Buffer &buffer, size_t size)
    {
        void *ptr = nullptr;
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        buffer.reset(ptr);
        return ACL_SUCCESS;
    }

    static aclError Copy(const Buffer &buffer, size_t size, const void *source)
    {
        return aclrtMemcpy(buffer.get(), size, source, size, ACL_MEMCPY_HOST_TO_DEVICE);
    }
};

inline void GenerateBlockedTwiddle(int n, int row, int col, int blockNum,
                                   int stepN, int l2Step, uint8_t *twiddleBuf)
{
    if (n == 0) return;
    if (stepN == 0) return;
    if (l2Step == 0) return;
    if (blockNum == 0) return;
    if (n <= 0 || col <= 0 || blockNum <= 0 || stepN <= 0 || l2Step <= 0) return;
    static const double pi = acos(-1);
    using C = std::complex<float>;
    int l2Blocks = row / stepN / l2Step;
    for (int i = 0, l2Idx = 0; l2Idx < row / stepN; ++i, l2Idx += l2Step) {
        for (int blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
            int averageM = l2Step * stepN / blockNum;
            int remainderM = l2Step * stepN % blockNum;
            int startIdx = blockIdx * averageM + std::min(blockIdx, remainderM);
            int calcM = averageM + (blockIdx < remainderM);
            auto buf = reinterpret_cast<float *>(twiddleBuf + col * (blockIdx + i * blockNum) * sizeof(float));

            for (int j = 0; j < col; ++j) {
                C wk(cos(pi * (startIdx + l2Idx * stepN) * j * 2 / n), -sin(pi * (startIdx + l2Idx * stepN) * j * 2 / n));
                buf[j] = wk.real();
                buf[j + col * (blockNum * l2Blocks + 1)] = wk.imag();
            }
        }
    }
    auto buf = reinterpret_cast<float *>(twiddleBuf + col * blockNum * l2Blocks * sizeof(float));
    for (int j = 0; j < col; ++j) {
        C wk(cos(pi * j * 2 / n), -sin(pi * j * 2 / n));
        buf[j] = wk.real();
        buf[j + col * (blockNum * l2Blocks + 1)] = wk.imag();
    }
}

inline void GenerateTiling3(int n, int n1, int n2, int n3, int blockNum, uint8_t *twiddleBuf, uint8_t *gatherBuf) {
    if (n <= 0 || n1 <= 0 || n2 <= 0 || n3 <= 0 || blockNum <= 0) {
        return;
    }
    static const double pi = acos(-1);
    typedef std::complex<float> C;
    GenerateTwiddle(reinterpret_cast<float *>(twiddleBuf), n, n1 * n2, n3);
    {

        auto buf = reinterpret_cast<float *>(twiddleBuf + n * sizeof(float) * 2);
        GenerateDft<>(buf, n1);

        buf += n1 * n1 * 3;
        GenerateDftWithTwiddle(buf, n2, n1);

        buf += n1 * n2 * n2 * 3;
        GenerateDft<>(buf, n3);
    }
    int calcN = std::min(n / 2, TILE_LENGTH);
    GenerateGather(gatherBuf, calcN);
}

inline bool GetFft3Factors(uint32_t n, int &n1, int &n2, int &n3)
{
    if (n == 0) return false;
    int64_t fftN = static_cast<int64_t>(n);
    n1 = RootCeil(fftN, 3);
    if (n1 == 0 || n1 <= 0) return false;
    n2 = RootCeil(fftN / n1, 2);
    if (n2 == 0 || n2 <= 0) return false;
    n3 = fftN / n1 / n2;
    return n3 > 0;
}

template<int Radices, typename TilingData>
inline void InitializeTiling(TilingData &tilingData, int64_t n, uint32_t batches,
                             int n1, int n2, int n3 = 0, int n4 = 0, int n5 = 0)
{
    tilingData.N = n;
    tilingData.batch = batches;
    tilingData.n1 = n1;
    tilingData.n2 = n2;
    if constexpr (Radices >= 3) tilingData.n3 = n3;
    if constexpr (Radices >= 4) tilingData.n4 = n4;
    if constexpr (Radices >= 5) tilingData.n5 = n5;
}

inline aclError AllocateFft3HostBuffers(uint8_t *&twiddleHost, uint8_t *&gatherHost,
                                        size_t twiddleSize, size_t gatherSize)
{
    CHECK_ACL(aclrtMallocHost((void**)(&twiddleHost), twiddleSize));
    return aclrtMallocHost((void**)(&gatherHost), gatherSize);
}

template<typename Launch>
inline aclError ExecuteFft3(DeviceBuffers &dev, float *y, size_t outputSize, void *stream,
                            uint32_t coreNum, Launch launch)
{
    launch(coreNum, stream, dev);
    return dev.CopyOutput(y, outputSize, stream);
}

inline aclError FreeHostBuffers(uint8_t *twiddleHost, uint8_t *gatherHost, uint8_t *eyeHost = nullptr)
{
    CHECK_ACL(aclrtFreeHost(twiddleHost));
    CHECK_ACL(aclrtFreeHost(gatherHost));
    if (eyeHost != nullptr) CHECK_ACL(aclrtFreeHost(eyeHost));
    return ACL_SUCCESS;
}

inline aclError CompleteFft(DeviceBuffers &dev, float *y, size_t outputSize, void *stream,
                            uint8_t *twiddleHost, uint8_t *gatherHost)
{
    CHECK_ACL(dev.CopyOutput(y, outputSize, stream));
    return FreeHostBuffers(twiddleHost, gatherHost);
}

template<typename TilingData, typename Launch>
inline aclError RunFft3(float *x, float *y, uint32_t n,
                                     uint32_t batches, int isForward, void *stream, Launch launch)
{
    int n1 = 0, n2 = 0, n3 = 0;
    if (!GetFft3Factors(n, n1, n2, n3)) return ACL_ERROR_INVALID_PARAM;
    uint32_t coreNum = GetCoreNum();
    int64_t fftN = static_cast<int64_t>(n);
    int64_t halfN = fftN / 2;
    size_t inputSize = batches * fftN * sizeof(float);
    size_t outputSize = batches * (halfN + 1) * sizeof(float) * 2;
    size_t twiddleSize = fftN * sizeof(float) * 2 + (n1 * n1 + n1 * n2 * n2 + n3 * n3) * sizeof(float) * 3;
    size_t workSize = fftN * (batches + 2) * sizeof(float);
    size_t gatherSize = TILE_LENGTH * 2 * sizeof(uint32_t);
    uint32_t tilingSize = sizeof(TilingData);

    TilingData tilingData;
    InitializeTiling<3>(tilingData, n, batches, n1, n2, n3);

    uint8_t* twiddleHost;
    uint8_t* gatherHost;
    CHECK_ACL(AllocateFft3HostBuffers(twiddleHost, gatherHost, twiddleSize, gatherSize));
    GenerateTiling3(fftN, n1, n2, n3, coreNum, twiddleHost, gatherHost);

    DeviceBuffers dev;
    BufferSizes sizes{inputSize, outputSize, workSize, twiddleSize, gatherSize, tilingSize, 0};
    CHECK_ACL(dev.Prepare(sizes, x, twiddleHost, gatherHost, &tilingData));

    CHECK_ACL(ExecuteFft3(dev, y, outputSize, stream, coreNum, launch));
    CHECK_ACL(FreeHostBuffers(twiddleHost, gatherHost));

    return ACL_SUCCESS;
}

} // namespace Rfft1DR2CCommonHost

#endif
