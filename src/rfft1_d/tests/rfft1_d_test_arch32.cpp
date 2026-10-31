/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rfft1_d_test.h"
#include "rfft1_d/rfft1_d.h"

namespace Rfft1DApiIntegrationTest {
    // issue #227: 对照 fft1_d 的 entry_reject_regression（fft1_d_test_arch32.cpp:121），
    // rfft1_d 此前仅有精度用例、无任何入口拒绝断言——入口校验被重构删除时 UT 仍全绿
    // （fft1_d 的 isForward 校验曾有 #147 先例）。逐一断言既有校验链的错误码
    static void run_entry_reject_regression_test(OpsFftTest::TestStats& stats) {
        TEST_CASE_BEGIN("entry_reject_regression");
        float dummy = 0.0f;
        bool all_ok = true;
        // 1) 空指针拒绝（DFT/MIX/幂次直连入口，issue #76）
        if (aclfftRfft1DDft(nullptr, &dummy, 256, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft(nullptr, &dummy, 2048, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft1(nullptr, &dummy, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 2) norm!=0 拒绝（DFT 既有 #89 口径；MIX 为 #224 补齐）
        if (aclfftRfft1DDft(&dummy, &dummy, 256, 1, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft(&dummy, &dummy, 2048, 1, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 3) n==0 拒绝（DFT n 域 [1,1024] 下界 #131；MIX #165；幂次 6 入口既有守卫）
        if (aclfftRfft1DDft(&dummy, &dummy, 0, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft(&dummy, &dummy, 0, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft2(&dummy, &dummy, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft31(&dummy, &dummy, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft32(&dummy, &dummy, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft4(&dummy, &dummy, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft5(&dummy, &dummy, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 4) n 上界拒绝（DFT [1,1024] 上界 #131；MIX (0, 2^27] 上界 #187）
        if (aclfftRfft1DDft(&dummy, &dummy, 1025, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft(&dummy, &dummy, (1u << 27) + 1, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 5) batches==0 / >INT32_MAX 拒绝（issue #188/#162）
        if (aclfftRfft1DDft(&dummy, &dummy, 256, 0, 0, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DDft(&dummy, &dummy, 256, 0, 0x80000000u, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft(&dummy, &dummy, 2048, 0, 0, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (all_ok) {
            TEST_CASE_PASS_DETAIL(stats, "entry_reject_regression",
                                  "all rfft1_d reject paths return the documented error codes");
        } else {
            TEST_CASE_FAIL_DETAIL(stats, "entry_reject_regression",
                                  "some reject path missing or wrong error code (see build log)");
        }
    }

    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        std::vector<R2CTestCase> cases = {
            // r2c n<=1024 -> arch32/dft/ : dft_r2c
            {"r2c_basic",  8,    1},
            {"r2c_n16",    16,   1},
            {"r2c_n1024",  1024, 1},
            // DFT 路径奇数 n（issue #227）：文档 FFT_1D.md 承诺 DFT 支持任意
            // 正整数 1<=nx<=1024，奇数 n 此前零覆盖
            {"r2c_dft_n7",   7,   1},
            {"r2c_dft_n15",  15,  1},
            {"r2c_dft_n255", 255, 1},
            // r2c n>1024 radix-mix -> arch32/r2c_fft/ : fft_r2c
            {"r2c_fft_n1536",  1536,  4},
            {"r2c_fft_n8192",  8192,  4},
            {"r2c_fft_n2048_b100",  2048, 100},

            // added
            {"r2c_fft1_1",  2,    3},
            {"r2c_fft1_2",  4,    16384},
            {"r2c_fft1_3",  8,    1},
            {"r2c_fft1_4",  128,    16384},

            {"r2c_fft2_1",  256,    3},
            {"r2c_fft2_2",  512,    16384},
            {"r2c_fft2_3",  4096,    1},
            {"r2c_fft2_4",  16384,    16384},

            {"r2c_fft3_1_1",  32768,    3},
            {"r2c_fft3_1_2",  32768,    4096},
            {"r2c_fft3_1_3",  65536,    1},
            {"r2c_fft3_1_4",  65536,    4096},

            {"r2c_fft3_2_1",  131072,    3},
            {"r2c_fft3_2_2",  262144,    2048},
            {"r2c_fft3_2_3",  1048576,    1},
            {"r2c_fft3_2_4",  2097152,    64},

            {"r2c_fft4_1",  4194304,    3},
            {"r2c_fft4_2",  8388608,    16},
            {"r2c_fft4_3",  16777216,    1},
            {"r2c_fft4_4",  33554432,    8},

            {"r2c_fft5_1",  67108864,    4},
            {"r2c_fft5_2",  134217728,    2},
            {"r2c_fft5_3",  268435456,    1},
            {"r2c_fft5_4",  536870912,    1},
        };
        OpsFftTest::RunBatchTests(stream, stats, "rfft1_d", cases, run_r2c_case);
        run_entry_reject_regression_test(stats);
    }
}

REGISTER_OP_TEST(Rfft1DApiIntegration)
