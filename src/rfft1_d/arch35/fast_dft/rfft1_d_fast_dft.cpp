/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <vector>

#include "rfft1_d_tilingdata.h"
#include "rfft1_d_fast.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "acl/acl.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"
#include "../../rfft1_d.h"
#include "../../rfft1_d_size_utils.h"

static const uint32_t COMPLEX_PART = 2;
static const double INIT_VALUE = 0.;
static const uint32_t DFT_BORDER_VALUE = 4096;
static const uint32_t RFFT_SYMMETRY_DIVISOR = 2;
// 归一化值域与 rfft1_d.h 文档对齐：0=BACKWARD（不缩放）、1=ORTHO（1/sqrt(n)）、2=FORWARD（1/n）（issue #82）
static const uint32_t BACKWARD = 0;
static const uint32_t FORWARD = 2;

// 恒补齐口径（issue #407，生成/消费两侧共用）：无论 n 是否已对齐一律追加一个完整
// pad 段（n + pad - n % pad），与 Rfft1DDftGen 生成侧历史行为一致；pad 档位取
// NZ_BORDER(=8) 与 NZ_BLOCK(=16) 两档（常量定义见 rfft1_d_fast.h），以模板形参
// 固化为编译期常量（运行期无除法参数面）
template <uint32_t Pad>
static inline size_t DftMatrPadLen(size_t n)
{
    return n + (Pad - n % Pad);
}

// ACL 错误检查宏：失败时返回真实 aclError 而非魔数 1（issue #93，与 fft_common_core.h 同款）
#define CHECK_ACL(call)                                                                              \
    do {                                                                                             \
        aclError err = (call);                                                                       \
        if (err != ACL_SUCCESS) {                                                                    \
            std::cerr << "ACL error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return err;                                                                              \
        }                                                                                            \
    } while (0)

struct AclrtFreeDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

// CalcCooleyTukeyFactors / CalcBluesteinFactors 已删除（issue #99/#272/#407）：
// 前者仅在不可达分支写入 factors（issue #99）；后者产物 factors/prevRadices 等
// 仅被零消费的 tiling 字段链式引用，随 19 个死字段一并清理（issue #407）

// dftRealOverallSize 与 Rfft1DDftGen 实际上传矩阵口径对齐（issue #407）：
// NZ_BORDER(=8) 补齐维 × NZ_BLOCK(=16) 补齐维。原"列条件补齐"在 n%16==0 时不生效，
// 低报实际上传尺寸（如 n=32：40×32=1280 < 实际 40×48=1920）；n > DFT_BORDER_VALUE
// 的 Bluestein 累积分支随死字段删除（入口已拒绝 n > 4096，不可达）
static uint32_t CalcDftRealOverallSize(uint32_t n)
{
    return static_cast<uint32_t>(DftMatrPadLen<NZ_BORDER>(n) * DftMatrPadLen<NZ_BLOCK>(n));
}

static int SetTilingData(Rfft1DTilingData& tiling, uint32_t n, uint32_t coreNum, uint32_t batches)
{
    // factors/Bluestein 补长/tmpLenPerBatch 等推导已删除（issue #407）：仅被零消费的
    // tiling 字段链式引用；归一化在 host 生成 DFT 矩阵时烤入（issue #121），
    // norm 入参随之移除
    tiling.length = n;
    tiling.batchesPerCore = batches / (coreNum ? coreNum : 1);
    tiling.leftOverBatches = batches % (coreNum ? coreNum : 1);
    tiling.dftRealOverallSize = CalcDftRealOverallSize(n);

    return ACL_SUCCESS;
}

static std::vector<float> Rfft1DDftGen(int64_t fftLength, int64_t norm)
{
    std::vector<float> dftNz;
    double normParam = 1.;

    if (norm != BACKWARD) {
        normParam = norm == FORWARD ? normParam / double(fftLength) : normParam / sqrt(double(fftLength));
    }

    size_t fftLenPad = DftMatrPadLen<NZ_BORDER>(fftLength);
    size_t fftLenPadRow = DftMatrPadLen<NZ_BLOCK>(fftLength);

    size_t fftOut = fftLength / RFFT_SYMMETRY_DIVISOR + 1;
    // reserve 维度订正（issue #407）：实际矩阵为 fftLenPad × fftLenPadRow，
    // 原 fftLenPadRow * fftLenPadRow 两维混用恒超额预留
    dftNz.reserve(fftLenPad * fftLenPadRow);

    std::vector<double> dftNd(fftLenPad * fftLenPadRow, INIT_VALUE);
    size_t k = 0;

    if ((fftLength % NZ_BLOCK != 0) && (fftLength % NZ_BLOCK <= NZ_BORDER)) {
        for (int i = 0; i < fftLength; ++i) {
            for (size_t j = 0; j < fftOut; ++j) {
                double param = -2. * M_PI * i * j / double(fftLength);
                dftNd[k] = normParam * cos(param);
                ++k;
                dftNd[k] = normParam * sin(param);
                ++k;
            }
        }
        dftNz.assign(dftNd.begin(), dftNd.end());
        return dftNz;
    } else {
        for (int i = 0; i < fftLength; ++i) {
            for (size_t j = 0; j < fftOut; ++j) {
                double param = -2. * M_PI * i * j / double(fftLength);
                dftNd[k] = normParam * cos(param);
                ++k;
                dftNd[k] = normParam * sin(param);
                ++k;
            }
            for (size_t j = 0; j + COMPLEX_PART * fftOut < fftLenPad; ++j) {
                dftNd[k] = INIT_VALUE;
                ++k;
            }
        }
        for (size_t n = 0; n < fftLenPad / NZ_BORDER; ++n) {
            for (size_t i = 0; i < fftLenPadRow; ++i) {
                for (size_t blockIdx = 0; blockIdx < NZ_BORDER; ++blockIdx) {
                    dftNz.emplace_back(static_cast<float>(dftNd[i * fftLenPad + NZ_BORDER * n + blockIdx]));
                }
            }
        }
        return dftNz;
    }
}

extern "C" __global__ __aicore__ void rfft1_d_fast_dft(
    GM_ADDR x, GM_ADDR dft, GM_ADDR y, GM_ADDR workspace, Rfft1DTilingData tilingData)
{
    if (tilingData.length <= DFT_BORDER_VALUE) {
        // norm 死传参已删除：归一化在 host 生成 DFT 矩阵时烤入（issue #121）
        KernelRfftFastDFT op(
            tilingData.length, tilingData.batchesPerCore, tilingData.leftOverBatches, tilingData.dftRealOverallSize);

        uint32_t modeLength = COMPLEX_PART * (tilingData.length / RFFT_SYMMETRY_DIVISOR + 1);
        auto t1 = PrepareTiling((op.batches + op.advancedBatches) / GetBlockNum(), op.modeLength, tilingData.length);
        REGIST_MATMUL_OBJ(&op.pipe, workspace, op.matmulObj, (void*)&t1, op.matmulObjNZ, (void*)&t1);

        op.Init(x, dft, y);
        op.Process();
    }
    return;
}

extern "C" aclError aclfftRfft1D(float* x, float* y, uint32_t n, int32_t norm, uint32_t batches, void* stream)
{
    if (x == nullptr || y == nullptr) {
        std::cerr << "[ops-fft] aclfftRfft1D: input/output pointer is null" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    if (n == 0) {
        std::cerr << "[ops-fft] aclfftRfft1D: n must be positive, got 0" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    if (n > DFT_BORDER_VALUE) {
        std::cerr << "[ops-fft] aclfftRfft1D: n=" << n << " exceeds FastDFT limit (" << DFT_BORDER_VALUE
                  << "), not implemented" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // norm 合法值校验（issue #118）：Rfft1DDftGen 对 norm != BACKWARD 落入
    // else 分支按 1/sqrt(n)（ORTHO 口径）静默缩放，非法取值（如 5、-1）会
    // 产出错误缩放结果；与 #89/#90 的 norm 校验口径对齐
    if (norm < 0 || norm > 2) {
        std::cerr << "[ops-fft] aclfftRfft1D: norm=" << norm << " is invalid (expected 0=BACKWARD, 1=ORTHO, 2=FORWARD)"
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    // batches==0 拒绝（issue #226，补齐 arch35 对 arch32 #188 同款守卫的遗漏）：
    // batches=0 时输入尺寸为 0，0 字节拷贝与空内核启动返回"成功"但输出缓冲未写入
    if (batches == 0) {
        std::cerr << "[ops-fft] aclfftRfft1D: batches=0 is not supported" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAic();
    // 平台信息查询失败时 GetCoreNumAic 可能返回 0（同仓其余 13 处入口均已兜底），
    // 以 0 block 启动内核行为未定义，D2H 会把未写入的缓冲当结果返回（issue #112）
    if (coreNum == 0)
    {
        coreNum = 1;
    }

    auto dft = Rfft1DDftGen(n, norm);

    uint32_t sysWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();

    const size_t inputSize = rfft1_d::InputSize(n, batches);
    const size_t dftSize = dft.size() * sizeof(float);
    const size_t outputSize = rfft1_d::OutputSize(n, batches);

    void* dev_x = nullptr;
    void* dev_dft = nullptr;
    void* dev_y = nullptr;
    void* workspace_ptr = nullptr;

    CHECK_ACL(aclrtMalloc(&dev_x, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_x_guard(dev_x);
    CHECK_ACL(aclrtMalloc(&dev_dft, dftSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_dft_guard(dev_dft);
    CHECK_ACL(aclrtMalloc(&dev_y, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> d_y_guard(dev_y);
    CHECK_ACL(aclrtMalloc(&workspace_ptr, sysWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    std::unique_ptr<void, AclrtFreeDeleter> workspace_ptr_guard(workspace_ptr);

    CHECK_ACL(aclrtMemcpy(dev_x, inputSize, x, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_dft, dftSize, dft.data(), dftSize, ACL_MEMCPY_HOST_TO_DEVICE));

    Rfft1DTilingData tilingData = {};
    aclError ret = SetTilingData(tilingData, n, coreNum, batches);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    rfft1_d_fast_dft<<<coreNum, nullptr, stream>>>(
        (__gm__ uint8_t*)dev_x, (__gm__ uint8_t*)dev_dft, (__gm__ uint8_t*)dev_y, (__gm__ uint8_t*)workspace_ptr,
        tilingData);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y, outputSize, dev_y, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    return ACL_SUCCESS;
}
