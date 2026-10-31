/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_COMMON_CORE_H
#define FFT_COMMON_CORE_H

#include <cmath>
#include <numeric>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "acl/acl.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"

constexpr uint32_t DFT_SIZE_MULTIPLIER = 2;
static const double INIT_VALUE = 0.;
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;

// ACL 错误检查宏：失败时返回真实 aclError 而非魔数 1，使导出入口的运行期错误
// 可按 acl 错误码体系判读（issue #93）
#define CHECK_ACL(call)                                                                              \
    do {                                                                                             \
        aclError err = (call);                                                                       \
        if (err != ACL_SUCCESS) {                                                                    \
            std::cerr << "ACL error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return err;                                                                              \
        }                                                                                            \
    } while (0)

// 自定义删除器，安全处理空指针
struct AclrtFreeDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

// 自定义删除器（host 固定内存），安全处理空指针（issue #223，与 AclrtFreeDeleter 同口径）
struct AclrtFreeHostDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFreeHost(ptr);
        }
    }
};

/* ======================== Mix-radix FFT shared constants ======================== */

constexpr int64_t TWO_MUL = 2;
constexpr int64_t RADIX_ADDR_START = 128;
constexpr int64_t RADIX_ADDR_END = 2;
constexpr int32_t RADIX_LEN_EVEN_NUM = 2;
constexpr int32_t RADIX_LEN_THREE_NUM = 3;
constexpr int32_t RADIX_LEN_FIVE_NUM = 5;
constexpr int32_t RADIX_LEN_FOUR_NUM = 4;
constexpr int32_t ROUND_16 = 16;
constexpr int32_t ROUND_32 = 32;
constexpr int32_t AUXILSIZE = 64;
constexpr int32_t VECVTRANS_LEN = 88;
constexpr int32_t RADIX_LIST_LEN = 45;
constexpr int32_t AIVSPLITWAY_TWO = 2;
constexpr int32_t AIVSPLITWAY_THREE = 3;
constexpr int64_t L2_CACHE_MAX_FLOAT_21 = (1 << 21);
constexpr int64_t L2_CACHE_MAX_FLOAT_22 = (1 << 22);
constexpr int64_t EVEN_THRESHOLD = 524288 / 2;
constexpr int64_t L0AB_BUF = 128 * 64;
constexpr int64_t N_FFT_C2C_MAX = 2 * 128 * 128;
constexpr int32_t RADIX_LEN_19683 = 19683;
constexpr int32_t RADIX_LEN_243 = 243;
constexpr int32_t RADIX_LEN_729 = 729;
constexpr int32_t RADIX_LEN_59049 = 59049;
constexpr int32_t RADIX_LEN_177147 = 177147;
constexpr int32_t RADIX_LEN_1594323 = 1594323;
constexpr int32_t RADIX_LEN_129140163 = 129140163;
constexpr int32_t RADIX_LEN_387420489 = 387420489;
constexpr int32_t RADIX_LEN_625 = 625;
constexpr int32_t RADIX_LEN_3125 = 3125;
constexpr int32_t RADIX_LEN_15625 = 15625;
constexpr int32_t RADIX_LEN_78125 = 78125;
constexpr int32_t RADIX_LEN_390625 = 390625;
constexpr int32_t RADIX_LEN_2401 = 2401;
constexpr int32_t RADIX_LEN_16807 = 16807;
constexpr int32_t RADIX_LEN_117649 = 117649;
constexpr int32_t RADIX_LEN_76800 = 76800;
constexpr int64_t BIAS_SIZE = 2048;
constexpr int64_t BIASC_SIZE = 4096;
constexpr int64_t RADIX_8K = 8192;

/* ======================== Mix-radix FFT shared functions ======================== */

inline int64_t ROUND_UP(int64_t num, int64_t pad)
{
    if (pad == 0)
        return 1;
    return (num + pad - 1) / pad * pad;
}
inline int64_t MIN_(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

inline void getTile(
    int64_t N1, int64_t N2, int64_t stepIndex, int64_t stepLen, int32_t& tileM0, int32_t& tileN0, int32_t& tileK0)
{
    constexpr int32_t N1_MAX45 = 45, N1_MAX64 = 64, N2_MAX8 = 8;
    constexpr int32_t TILE_CONST = 128, CALC_TWO = 2, STEP_LEN_THREE = 3;
    if (N1 <= N1_MAX45 || (N1 <= N1_MAX64 && N2 <= N2_MAX8)) {
        // N1=0 防护（issue #122）：上游 FindTwoRadix/分解校验已保证 N1>=1，
        // 此分支当前不可达；改 throw 为 clamp 防御——异常穿越 extern "C" 会
        // std::terminate 进程，与全仓错误码约定不符
        if (N1 < 1)
            N1 = 1;
        tileM0 = ROUND_UP(CALC_TWO * N1, ROUND_16);
        tileK0 = tileM0;
        if (stepIndex == stepLen - 1)
            tileK0 = CALC_TWO * ROUND_UP(N1, ROUND_16);
        tileN0 = L0AB_BUF * CALC_TWO / tileK0 / ROUND_16 * ROUND_16;
        tileN0 = MIN_(tileN0, ROUND_UP(N2, ROUND_16));
    } else {
        tileM0 = TILE_CONST;
        tileN0 = TILE_CONST;
        tileK0 = TILE_CONST;
        if (stepIndex == stepLen - 1) {
            if (tileK0 > CALC_TWO * ROUND_UP(N1, ROUND_16) / CALC_TWO && tileK0 < CALC_TWO * ROUND_UP(N1, ROUND_16))
                tileK0 = MIN_(tileK0, ROUND_UP(CALC_TWO * ROUND_UP(N1, ROUND_16) / CALC_TWO, ROUND_16));
            tileK0 = MIN_(tileK0, CALC_TWO * ROUND_UP(N1, ROUND_16));
            if (tileK0 > N1_MAX64)
                tileK0 = ROUND_UP(tileK0, TILE_CONST);
        } else {
            if (tileK0 > CALC_TWO * N1 / CALC_TWO && tileK0 < CALC_TWO * N1)
                tileK0 = MIN_(tileK0, ROUND_UP(CALC_TWO * N1 / CALC_TWO, ROUND_16));
            tileK0 = MIN_(tileK0, ROUND_UP(CALC_TWO * N1, ROUND_16));
        }
        if (tileM0 > CALC_TWO * N1 / CALC_TWO && tileM0 < CALC_TWO * N1)
            tileM0 = MIN_(tileM0, ROUND_UP(CALC_TWO * N1 / CALC_TWO, ROUND_16));
        tileM0 = MIN_(tileM0, ROUND_UP(CALC_TWO * N1, ROUND_16));
        tileN0 = MIN_(tileN0, ROUND_UP(N2, ROUND_16));
    }
    if (((stepLen <= STEP_LEN_THREE || stepIndex != stepLen - CALC_TWO)) && tileN0 > N1_MAX64) {
        if (tileK0 * ROUND_UP(tileN0, TILE_CONST) <= L0AB_BUF * CALC_TWO)
            tileN0 = ROUND_UP(tileN0, TILE_CONST);
        else
            tileN0 = tileN0 / TILE_CONST * TILE_CONST;
    }
}

inline std::vector<int64_t> FindTwoRadix(std::vector<int64_t>& factors, int64_t n)
{
    if (n == 0)
        n = 1;
    // 空因子向量防护：InitMixRadixShort 精化循环 erase 后可能传入空向量，
    // 直接取 factors[0] 为越界 UB（fftN 为质数场景，issue #100）
    if (factors.empty()) {
        std::vector<int64_t> rl = {1, n};
        std::sort(rl.begin(), rl.end());
        return rl;
    }
    int64_t n1 = factors[0];
    if (n1 == 0)
        n1 = 1;
    float minRatio = std::max((float)(n / (double)(n1 * n1)), (float)((double)(n1 * n1) / n));
    int64_t minN1 = n1;
    int64_t fLen = (int64_t)factors.size();
    for (int64_t i = 0; i < (1 << fLen); i++) {
        n1 = 1;
        for (int64_t j = 0; j < fLen; j++)
            if ((uint64_t)i & (1LL << j))
                n1 *= factors[j];
        float ratio = std::max((float)(n / (double)(n1 * n1)), (float)((double)(n1 * n1) / n));
        if (ratio < minRatio) {
            minN1 = n1;
            minRatio = ratio;
        }
    }
    std::vector<int64_t> rl = {minN1, n / minN1};
    std::sort(rl.begin(), rl.end());
    return rl;
}

// 混合基分解完整性校验（issue #113/#114）：radixVec 非空且各因子乘积等于 n。
// 不满足时说明 n 含 >128（Long）/无法合并（Short 素数回退）的质因子，内核不支持，
// 调用方应返回 ACL_ERROR_INVALID_PARAM——避免 InitAiVSplitWay 对空向量取
// radixVec[-1] 的越界读，以及按错误分解级数启动内核输出垃圾结果
inline bool IsMixRadixValid(const std::vector<int64_t>& radixVec, int64_t fftN)
{
    if (radixVec.empty())
        return false;
    int64_t product = 1;
    for (int64_t r : radixVec)
        product *= r;
    return product == fftN;
}

inline void InitMixRadixLong(int64_t fftN, std::vector<int64_t>& radixVec)
{
    std::vector<int64_t> radixArr;
    for (int64_t i = RADIX_ADDR_START; i >= RADIX_ADDR_END; i--)
        radixArr.push_back(i);
    std::vector<int64_t> radixList;
    int64_t inputLen = fftN;
    if (inputLen == RADIX_LEN_19683) {
        radixVec = {27, 9, 81};
        return;
    }
    if (inputLen == RADIX_LEN_243) {
        // 243=3^5=27*9；原 {27,9,81} 系自 RADIX_LEN_19683 分支复制的笔误，乘积 19683!=243（issue #228）
        radixVec = {27, 9};
        return;
    }
    if (inputLen == RADIX_LEN_729) {
        radixVec = {27, 27};
        return;
    }
    if (inputLen == RADIX_LEN_59049) {
        radixVec = {27, 27, 81};
        return;
    }
    if (inputLen == RADIX_LEN_177147) {
        radixVec = {27, 81, 81};
        return;
    }
    if (inputLen == RADIX_LEN_1594323) {
        radixVec = {27, 27, 27, 81};
        return;
    }
    if (inputLen == RADIX_LEN_129140163) {
        radixVec = {27, 27, 27, 81, 81};
        return;
    }
    if (inputLen == RADIX_LEN_387420489) {
        radixVec = {27, 27, 81, 81, 81};
        return;
    }
    if (inputLen == RADIX_LEN_625) {
        radixVec = {25, 25};
        return;
    }
    if (inputLen == RADIX_LEN_3125) {
        radixVec = {25, 125};
        return;
    }
    if (inputLen == RADIX_LEN_15625) {
        radixVec = {125, 125};
        return;
    }
    if (inputLen == RADIX_LEN_78125) {
        radixVec = {25, 25, 125};
        return;
    }
    if (inputLen == RADIX_LEN_390625) {
        radixVec = {25, 125, 125};
        return;
    }
    if (inputLen == RADIX_LEN_2401) {
        radixVec = {49, 49};
        return;
    }
    if (inputLen == RADIX_LEN_16807) {
        radixVec = {7, 49, 49};
        return;
    }
    if (inputLen == RADIX_LEN_117649) {
        radixVec = {49, 49, 49};
        return;
    }
    if (inputLen == RADIX_LEN_76800) {
        radixVec = {8, 80, 120};
        return;
    }
    for (int64_t r : radixArr) {
        while (inputLen % r == 0) {
            radixList.push_back(r);
            inputLen /= r;
        }
    }
    std::sort(radixList.begin(), radixList.end());
    radixVec = radixList;
}

inline void InitMixRadixShort(int64_t fftN, std::vector<int64_t>& radixVec)
{
    int64_t n = fftN;
    std::vector<int64_t> radixList;
    if (fftN <= RADIX_ADDR_START) {
        radixList.push_back(fftN);
        radixVec = radixList;
        return;
    }
    std::vector<int64_t> primes;
    std::vector<bool> minp(n + 1, true);
    for (int64_t i = 2; i < n + 1; i++) {
        if (minp[i])
            primes.push_back(i);
        for (int64_t j = i * i; j < n + 1; j += i)
            minp[j] = false;
    }
    std::vector<int64_t> factors;
    for (int64_t p : primes) {
        while (n % p == 0) {
            factors.push_back(p);
            n /= p;
        }
        if (n == 1)
            break;
    }
    radixList = FindTwoRadix(factors, fftN);
    if (radixList[1] > RADIX_ADDR_START) {
        std::vector<int64_t> newRadixList;
        int64_t len = (int64_t)factors.size();
        for (int64_t i = 0; i < len; i++) {
            int64_t t = factors[i];
            if (t == 0)
                continue;
            factors.erase(factors.begin() + i);
            newRadixList = FindTwoRadix(factors, fftN / t);
            factors.insert(factors.begin() + i, t);
            if (newRadixList[1] <= RADIX_ADDR_START) {
                newRadixList.insert(newRadixList.begin(), t);
                radixList = newRadixList;
                break;
            }
        }
    }
    radixVec = radixList;
}

inline int64_t GetTwiddleMatrixLen(int64_t fftN, const std::vector<int64_t>& radixVec)
{
    int64_t n = fftN, dftMatrixLen = 0, n0 = 1;
    int64_t stepLen = (int64_t)radixVec.size();
    for (int64_t s = 0; s < stepLen; s++) {
        int64_t n1 = radixVec[s];
        if (n1 == 0)
            continue;
        int64_t n2 = n / n1 / n0;
        int32_t tM0, tN0, tK0;
        getTile(n1, (n2 > 1) ? n2 : n0, s, stepLen, tM0, tN0, tK0);
        dftMatrixLen += ROUND_UP(TWO_MUL * n1, tM0) * ROUND_UP(TWO_MUL * n1, tK0);
        n0 *= n1;
    }
    return dftMatrixLen;
}

inline void GenWMatrixForwardForMultiLen(int64_t stepLen, int64_t* radixListPtr, int64_t fftN, float* host)
{
    int64_t n0 = 1;
    for (int64_t s = 0; s < stepLen; s++) {
        int64_t n1 = radixListPtr[s];
        if (n1 == 0)
            continue;
        int64_t n2 = fftN / n1 / n0;
        int32_t tM0, tN0, tK0;
        getTile(n1, (n2 > 1) ? n2 : n0, s, stepLen, tM0, tN0, tK0);
        int64_t batchLen = L0AB_BUF * 2 / (tK0 * tN0);
        batchLen += (batchLen == 0);
        if (s == stepLen - 1 && batchLen > 1 && (n1 <= AUXILSIZE && n0 <= (int64_t)8)) {
            int64_t lda = TWO_MUL * ROUND_UP(n1, ROUND_16);
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1, j = k % n1;
                host[i * lda + j] = cos(-K_2PI / n1 * i * j);
                host[i * lda + ROUND_UP(n1, ROUND_16) + j] = -sin(-K_2PI / n1 * i * j);
                host[n1 * lda + i * lda + j] = sin(-K_2PI / n1 * i * j);
                host[n1 * lda + i * lda + ROUND_UP(n1, ROUND_16) + j] = cos(-K_2PI / n1 * i * j);
            }
        } else if (s == stepLen - 1 && L0AB_BUF / (tK0 * tN0) >= 1) {
            int64_t n1Loop = (2 * n1 + tM0 - 1) / tM0;
            int64_t lda = TWO_MUL * ROUND_UP(n1, ROUND_16);
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1, j = k % n1;
                int64_t n1Idx = i / (tM0 / 2), n1In = i % (tM0 / 2);
                int64_t n1Act = (n1Idx == n1Loop - 1) ? (2 * n1 - n1Idx * tM0) : tM0;
                host[n1Idx * tM0 * lda + n1In * lda + j] = cos(-K_2PI / n1 * i * j);
                host[n1Idx * tM0 * lda + n1In * lda + ROUND_UP(n1, ROUND_16) + j] = -sin(-K_2PI / n1 * i * j);
                host[n1Idx * tM0 * lda + n1In * lda + (n1Act / 2) * lda + j] = sin(-K_2PI / n1 * i * j);
                host[n1Idx * tM0 * lda + n1In * lda + (n1Act / 2) * lda + ROUND_UP(n1, ROUND_16) + j] =
                    cos(-K_2PI / n1 * i * j);
            }
        } else if (s == stepLen - 1) {
            int64_t lda = TWO_MUL * TWO_MUL * ROUND_UP(n1, ROUND_16);
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1, j = k % n1;
                host[i * lda + j] = cos(-K_2PI / n1 * i * j);
                host[i * lda + ROUND_UP(n1, ROUND_16) + j] = -sin(-K_2PI / n1 * i * j);
                host[i * lda + TWO_MUL * ROUND_UP(n1, ROUND_16) + j] = sin(-K_2PI / n1 * i * j);
                host[i * lda + TWO_MUL * ROUND_UP(n1, ROUND_16) + ROUND_UP(n1, ROUND_16) + j] =
                    cos(-K_2PI / n1 * i * j);
            }
        } else {
            int64_t lda = ROUND_UP(TWO_MUL * n1, tM0);
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1, j = k % n1;
                host[i * TWO_MUL * lda + j] = cos(-K_2PI / n1 * i * j);
                host[i * TWO_MUL * lda + n1 + j] = -sin(-K_2PI / n1 * i * j);
                host[i * TWO_MUL * lda + lda + j] = sin(-K_2PI / n1 * i * j);
                host[i * TWO_MUL * lda + lda + n1 + j] = cos(-K_2PI / n1 * i * j);
            }
            host += ROUND_UP(TWO_MUL * n1, tM0) * ROUND_UP(TWO_MUL * n1, tK0);
            n0 *= n1;
        }
    }
}

inline void GenWMatrixInverseForMultiLen(int64_t stepLen, int64_t* radixListPtr, int64_t fftN, float* host)
{
    int64_t n0 = 1;
    for (int64_t s = 0; s < stepLen; s++) {
        int64_t n1 = radixListPtr[s];
        if (n1 == 0)
            continue;
        int64_t n2 = fftN / n1 / n0;
        int32_t tM0, tN0, tK0;
        getTile(n1, (n2 > 1) ? n2 : n0, s, stepLen, tM0, tN0, tK0);
        if (tK0 == 0)
            tK0 = 1;
        if (tN0 == 0)
            tN0 = 1;
        int64_t batchLen = L0AB_BUF * 2 / (tK0 * tN0);
        batchLen += (batchLen == 0);
        if (s == stepLen - 1 && batchLen > 1 && (n1 <= AUXILSIZE && n0 <= (int64_t)8)) {
            int64_t lda = TWO_MUL * ROUND_UP(n1, ROUND_16);
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1, j = k % n1;
                host[i * lda + j] = cos(-K_2PI / n1 * i * j);
                host[i * lda + ROUND_UP(n1, ROUND_16) + j] = sin(-K_2PI / n1 * i * j);
                host[n1 * lda + i * lda + j] = -sin(-K_2PI / n1 * i * j);
                host[n1 * lda + i * lda + ROUND_UP(n1, ROUND_16) + j] = cos(-K_2PI / n1 * i * j);
            }
        } else if (s == stepLen - 1 && L0AB_BUF / (tK0 * tN0) >= 1) {
            int64_t n1Loop = (2 * n1 + tM0 - 1) / tM0;
            int64_t lda = TWO_MUL * ROUND_UP(n1, ROUND_16);
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1, j = k % n1;
                int64_t n1Idx = i / (tM0 / 2), n1In = i % (tM0 / 2);
                int64_t n1Act = (n1Idx == n1Loop - 1) ? (2 * n1 - n1Idx * tM0) : tM0;
                host[n1Idx * tM0 * lda + n1In * lda + j] = cos(-K_2PI / n1 * i * j);
                host[n1Idx * tM0 * lda + n1In * lda + ROUND_UP(n1, ROUND_16) + j] = sin(-K_2PI / n1 * i * j);
                host[n1Idx * tM0 * lda + n1In * lda + (n1Act / 2) * lda + j] = -sin(-K_2PI / n1 * i * j);
                host[n1Idx * tM0 * lda + n1In * lda + (n1Act / 2) * lda + ROUND_UP(n1, ROUND_16) + j] =
                    cos(-K_2PI / n1 * i * j);
            }
        } else if (s == stepLen - 1) {
            int64_t lda = TWO_MUL * TWO_MUL * ROUND_UP(n1, ROUND_16);
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1, j = k % n1;
                host[i * lda + j] = cos(-K_2PI / n1 * i * j);
                host[i * lda + ROUND_UP(n1, ROUND_16) + j] = sin(-K_2PI / n1 * i * j);
                host[i * lda + TWO_MUL * ROUND_UP(n1, ROUND_16) + j] = -sin(-K_2PI / n1 * i * j);
                host[i * lda + TWO_MUL * ROUND_UP(n1, ROUND_16) + ROUND_UP(n1, ROUND_16) + j] =
                    cos(-K_2PI / n1 * i * j);
            }
        } else {
            int64_t lda = ROUND_UP(TWO_MUL * n1, tM0);
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1, j = k % n1;
                host[i * TWO_MUL * lda + j] = cos(-K_2PI / n1 * i * j);
                host[i * TWO_MUL * lda + n1 + j] = sin(-K_2PI / n1 * i * j);
                host[i * TWO_MUL * lda + lda + j] = -sin(-K_2PI / n1 * i * j);
                host[i * TWO_MUL * lda + lda + n1 + j] = cos(-K_2PI / n1 * i * j);
            }
            host += ROUND_UP(TWO_MUL * n1, tM0) * ROUND_UP(TWO_MUL * n1, tK0);
            n0 *= n1;
        }
    }
}

inline int64_t GetTwMatrixLen(int64_t fftN, const std::vector<int64_t>& radixVec)
{
    int64_t n = fftN, twLen = 0;
    int64_t stepLen = (int64_t)radixVec.size();
    if (stepLen == 1)
        return 0;
    int64_t n0 = 1;
    for (int64_t s = 0; s < stepLen; s++) {
        int64_t n1 = radixVec[s];
        if (n1 == 0)
            continue;
        int64_t n2 = n / n1 / n0;
        int32_t tM0, tN0, tK0;
        getTile(n1, (n2 > 1) ? n2 : n0, s, stepLen, tM0, tN0, tK0);
        twLen += TWO_MUL * n1 * ROUND_UP(n2, tN0);
        n0 *= n1;
    }
    return twLen;
}

inline void GenTwMatrix(int64_t fftN, const std::vector<int64_t>& radixVec, float* host)
{
    int64_t stepLen = (int64_t)radixVec.size();
    int64_t n0 = 1;
    for (int64_t s = 0; s < stepLen - 1; s++) {
        int64_t n1 = radixVec[s];
        if (n1 == 0)
            continue;
        int64_t n2 = fftN / n1 / n0;
        int32_t tM0, tN0, tK0;
        getTile(n1, (n2 > 1) ? n2 : n0, s, stepLen, tM0, tN0, tK0);
        for (int64_t k = 0; k < n1 * n2; k++) {
            int64_t i = k / n2, j = k % n2;
            host[i * 2 * tN0 + (j / tN0) * 2 * n1 * tN0 + (j % tN0)] = cos(-K_2PI / (n1 * n2) * i * j);
            host[i * 2 * tN0 + tN0 + (j / tN0) * 2 * n1 * tN0 + (j % tN0)] = sin(-K_2PI / (n1 * n2) * i * j);
        }
        host += 2 * n1 * ROUND_UP(n2, tN0);
        n0 *= n1;
    }
}

inline void InitMixRadixParam(
    int64_t parity, int64_t fftN, int64_t batchSize, std::vector<int64_t>& radixVec, int64_t& wsIn, int64_t& wsOut,
    int64_t& wsSync, int64_t& wsC2c, int64_t& wsAux)
{
    if (fftN == 0)
        fftN = 1;
    int64_t n = fftN, stepLen = (int64_t)radixVec.size();
    int64_t maxFloat = (parity == 0 && fftN <= EVEN_THRESHOLD) ? L2_CACHE_MAX_FLOAT_22 : L2_CACHE_MAX_FLOAT_21;
    int64_t batchPartLen = (maxFloat + fftN - 1) / fftN;
    if (batchPartLen == 0)
        batchPartLen = 1;
    int64_t batchLoop = (batchSize + batchPartLen - 1) / batchPartLen;
    int64_t batchRemain = batchSize % batchPartLen;
    if (batchLoop == 1 && batchRemain > 0)
        batchPartLen = batchRemain;
    if (n > maxFloat * RADIX_LEN_FIVE_NUM)
        batchPartLen = batchSize;
    int64_t tmpWsMax = (stepLen == 1) ? 0 : batchPartLen * 2 * fftN * sizeof(float);
    int64_t n0 = 1, tmpWsSync = 0;
    for (int64_t s = 0; s < stepLen; s++) {
        int64_t n1 = radixVec[s];
        if (n1 == 0)
            continue;
        int64_t n2 = fftN / n1 / n0;
        int32_t tM0, tN0, tK0;
        getTile(n1, (n2 > 1) ? n2 : n0, s, stepLen, tM0, tN0, tK0);
        int64_t N2p = ROUND_UP(n2, tN0);
        int64_t bLen = L0AB_BUF * 2 / (tK0 * tN0);
        bLen += (bLen == 0);
        int64_t tmpNow = 32;
        if (s == 0 || (stepLen >= RADIX_LEN_FOUR_NUM && s < stepLen - RADIX_LEN_EVEN_NUM && s != 0))
            tmpNow = batchPartLen * n0 * TWO_MUL * n1 * N2p * sizeof(float);
        if (stepLen >= RADIX_LEN_THREE_NUM && s == stepLen - RADIX_LEN_EVEN_NUM)
            tmpNow = batchPartLen * n0 * TWO_MUL * n1 * tN0 * sizeof(float);
        if (s == stepLen - 1)
            tmpNow = batchPartLen * TWO_MUL * ROUND_UP(n1, ROUND_16) * ROUND_UP(n0, tN0) * sizeof(float);
        tmpNow = ROUND_UP(tmpNow, RADIX_ADDR_START);
        int64_t tmpNowSync = 20 * 2 * bLen * tM0 * tN0 * sizeof(float);
        tmpWsMax = std::max(tmpNow, tmpWsMax);
        tmpWsSync = std::max(tmpNowSync, tmpWsSync);
        n0 *= n1;
    }
    wsIn = ROUND_UP(TWO_MUL * tmpWsMax, ROUND_32);
    wsOut = ROUND_UP(TWO_MUL * tmpWsMax, ROUND_32);
    wsSync = ROUND_UP(tmpWsSync, ROUND_32);
    wsC2c = ROUND_UP((int64_t)sizeof(float) * 2 * fftN * batchSize, ROUND_32);
    wsAux = AUXILSIZE;
}

inline int32_t InitAiVSplitWay(int64_t fftN, std::vector<int64_t>& radixVec)
{
    int64_t stepLen = (int64_t)radixVec.size();
    int64_t n = fftN;
    bool isVecVtransLoad = false;
    {
        int64_t n0 = 1;
        for (int64_t s = 0; s < stepLen - 1; s++) {
            int64_t n1 = radixVec[s];
            if (n1 == 0)
                continue;
            int64_t n2 = n / n1 / n0;
            int32_t tM0, tN0, tK0;
            getTile(n1, (n2 > 1) ? n2 : n0, s, stepLen, tM0, tN0, tK0);
            if (s == 0 && stepLen >= RADIX_LEN_EVEN_NUM)
                isVecVtransLoad = ROUND_UP(n2, tN0) <= VECVTRANS_LEN;
            n0 *= n1;
        }
    }
    int64_t n1 = radixVec[stepLen - 1];
    if (n1 == 0)
        n1 = 1;
    int64_t n0 = n / n1;
    int32_t tM0, tN0, tK0;
    getTile(n1, n0, stepLen - 1, stepLen, tM0, tN0, tK0);
    if (tN0 == 0)
        tN0 = 1;
    if (tK0 == 0)
        tK0 = 1;
    int64_t bLen = L0AB_BUF * 2 / (tN0 * tK0);
    bLen += (bLen == 0);
    int32_t aivSplitWay = (bLen > 1 && (n1 <= RADIX_LIST_LEN || (n1 <= AUXILSIZE && n0 <= (int64_t)8))) ?
                              1 :
                              ((L0AB_BUF / (tK0 * tN0) >= 1) ? AIVSPLITWAY_TWO : AIVSPLITWAY_THREE);
    return aivSplitWay;
}

#endif
