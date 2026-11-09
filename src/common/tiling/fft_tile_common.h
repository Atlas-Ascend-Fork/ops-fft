/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file only in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPSFFT_COMMON_TILING_FFT_TILE_COMMON_H
#define OPSFFT_COMMON_TILING_FFT_TILE_COMMON_H

// W 矩阵分块参数单一实现（issue #318）：host 侧 getTile（common/host/fft_common_core.h，
// W 矩阵布局由此计算）与设备侧 get_tile（common/kernel/fft/kernel_utils.h，kernel 消费
// 同一布局）原为两份逐行同构实现且已漂移——host 独有 N1<1 clamp（issue #122）、device
// 独有 tile_K0==0 兜底、常量命名两套（N1_MAX45/N1_SIZE_45、ROUND_16/R0_SIZE）。收敛为
// 本文件单一实现供 host/device 公用：防护取两侧并集、常量统一命名。纯算术实现，无
// std/aicore 依赖，host 与 kernel 编译单元均可包含（经 OPS_FFT 目标公共 include 路径
// src/common/tiling 可达）。
// 注意：本头文件与 fft_all_mix_tiling_data.h 的 static_assert 字节数断言无关联，
// 分块参数变化不影响 tiling 结构布局。

// 本头文件包含时假定 CANN 基础头（kernel_operator.h/common.h 等）已先序包含：
// 宿主侧（fft_common_core.h）与 kernel 侧（kernel_utils.h）包含点均满足。
// host/device 双态标记：bisheng（__CCE__）下为 __host_aicore__（[host, aicore]，
// 宿主与设备均可调用），宿主编译下为空（kernel_macros.h 的 ASCENDC_HOST_AICORE
// 语义）；头文件链未定义该宏时兜底为空
#ifndef ASCENDC_HOST_AICORE
#define ASCENDC_HOST_AICORE
#endif

constexpr int32_t FFT_TILE_R0_SIZE = 16;

constexpr int32_t FFT_TILE_N1_MAX45 = 45;
constexpr int32_t FFT_TILE_N1_MAX64 = 64;
constexpr int32_t FFT_TILE_N2_MAX8 = 8;
constexpr int32_t FFT_TILE_128 = 128;
constexpr int32_t FFT_TILE_CALC_TWO = 2;
constexpr int32_t FFT_TILE_STEP_LEN_THREE = 3;
// L0A/L0B ping-pong 缓冲可容纳的 float 数（原 host 版 L0AB_BUF 与 device 版
// L0AB_PINGPONG_BUFFER_LEN = 32*1024/sizeof(float) 数值一致，均为 8192）
constexpr int32_t FFT_TILE_L0AB_PINGPONG = 128 * 64;

ASCENDC_HOST_AICORE inline int64_t FftTileRoundUp(int64_t num, int64_t pad)
{
    return (num + pad - 1) / pad * pad;
}

ASCENDC_HOST_AICORE inline int64_t FftTileMin(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

ASCENDC_HOST_AICORE inline void FftGetTileParams(int64_t N1, int64_t N2, int64_t stepIndex, int64_t stepLen,
                             int32_t &tileM0, int32_t &tileN0, int32_t &tileK0)
{
    if (N1 <= FFT_TILE_N1_MAX45 || (N1 <= FFT_TILE_N1_MAX64 && N2 <= FFT_TILE_N2_MAX8)) {
        // N1<1 clamp（issue #122，原 host 版独有）：上游 FindTwoRadix/分解校验已保证
        // N1>=1，本分支当前不可达；clamp 而非 throw，防异常穿越 extern "C" 边界
        if (N1 < 1)
            N1 = 1;
        tileM0 = FftTileRoundUp(FFT_TILE_CALC_TWO * N1, FFT_TILE_R0_SIZE);
        tileK0 = tileM0;
        if (stepIndex == stepLen - 1)
            tileK0 = FFT_TILE_CALC_TWO * FftTileRoundUp(N1, FFT_TILE_R0_SIZE);
        // tile_K0==0 兜底（原 device 版独有，防护并集）：防异常输入下后续除零
        if (tileK0 == 0)
            tileK0 = FFT_TILE_R0_SIZE;
        tileN0 = FFT_TILE_L0AB_PINGPONG * FFT_TILE_CALC_TWO / tileK0 / FFT_TILE_R0_SIZE * FFT_TILE_R0_SIZE;
        tileN0 = FftTileMin(tileN0, FftTileRoundUp(N2, FFT_TILE_R0_SIZE));
    } else {
        tileM0 = FFT_TILE_128;
        tileN0 = FFT_TILE_128;
        tileK0 = FFT_TILE_128;
        if (stepIndex == stepLen - 1) {
            if (tileK0 > FFT_TILE_CALC_TWO * FftTileRoundUp(N1, FFT_TILE_R0_SIZE) / FFT_TILE_CALC_TWO &&
                tileK0 < FFT_TILE_CALC_TWO * FftTileRoundUp(N1, FFT_TILE_R0_SIZE))
                tileK0 = FftTileMin(tileK0, FftTileRoundUp(
                    FFT_TILE_CALC_TWO * FftTileRoundUp(N1, FFT_TILE_R0_SIZE) / FFT_TILE_CALC_TWO, FFT_TILE_R0_SIZE));
            tileK0 = FftTileMin(tileK0, FFT_TILE_CALC_TWO * FftTileRoundUp(N1, FFT_TILE_R0_SIZE));
            if (tileK0 > FFT_TILE_N1_MAX64)
                tileK0 = FftTileRoundUp(tileK0, FFT_TILE_128);
        } else {
            if (tileK0 > FFT_TILE_CALC_TWO * N1 / FFT_TILE_CALC_TWO && tileK0 < FFT_TILE_CALC_TWO * N1)
                tileK0 = FftTileMin(tileK0, FftTileRoundUp(
                    FFT_TILE_CALC_TWO * N1 / FFT_TILE_CALC_TWO, FFT_TILE_R0_SIZE));
            tileK0 = FftTileMin(tileK0, FftTileRoundUp(FFT_TILE_CALC_TWO * N1, FFT_TILE_R0_SIZE));
        }
        if (tileM0 > FFT_TILE_CALC_TWO * N1 / FFT_TILE_CALC_TWO && tileM0 < FFT_TILE_CALC_TWO * N1)
            tileM0 = FftTileMin(tileM0, FftTileRoundUp(
                FFT_TILE_CALC_TWO * N1 / FFT_TILE_CALC_TWO, FFT_TILE_R0_SIZE));
        tileM0 = FftTileMin(tileM0, FftTileRoundUp(FFT_TILE_CALC_TWO * N1, FFT_TILE_R0_SIZE));
        tileN0 = FftTileMin(tileN0, FftTileRoundUp(N2, FFT_TILE_R0_SIZE));
    }
    if (((stepLen <= FFT_TILE_STEP_LEN_THREE || stepIndex != stepLen - FFT_TILE_CALC_TWO)) && tileN0 > FFT_TILE_N1_MAX64) {
        if (tileK0 * FftTileRoundUp(tileN0, FFT_TILE_128) <= FFT_TILE_L0AB_PINGPONG * FFT_TILE_CALC_TWO)
            tileN0 = FftTileRoundUp(tileN0, FFT_TILE_128);
        else
            tileN0 = tileN0 / FFT_TILE_128 * FFT_TILE_128;
    }
}

#endif // OPSFFT_COMMON_TILING_FFT_TILE_COMMON_H
