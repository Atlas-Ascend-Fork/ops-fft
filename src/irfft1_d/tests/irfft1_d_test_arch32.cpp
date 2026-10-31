/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "irfft1_d_test.h"
#include "irfft1_d/irfft1_d.h"

namespace Irfft1DApiIntegrationTest {
    // issue #227: 对照 fft1_d 的 entry_reject_regression（fft1_d_test_arch32.cpp:121），
    // irfft1_d 此前仅有精度用例、无任何入口拒绝断言。逐一断言既有校验链的错误码
    static void run_entry_reject_regression_test(OpsFftTest::TestStats& stats) {
        TEST_CASE_BEGIN("entry_reject_regression");
        float dummy = 0.0f;
        bool all_ok = true;
        // 1) 空指针拒绝（DFT/MIX 直连入口，issue #76）
        if (aclfftIrfft1DDft(nullptr, &dummy, 256, 0, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftIrfft1DC2RFft(nullptr, &dummy, 2048, 0, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 2) norm!=0 拒绝（DFT 既有 #91 口径；MIX 为 #224 补齐）
        if (aclfftIrfft1DDft(&dummy, &dummy, 256, 1, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftIrfft1DC2RFft(&dummy, &dummy, 2048, 1, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 3) isForward 拒绝（DFT：C2R 恒为逆变换仅接受 0，issue #169）
        if (aclfftIrfft1DDft(&dummy, &dummy, 256, 0, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 4) n==0 / 上界拒绝（DFT n 域 [1,1024]，issue #131；MIX #166/#187）
        if (aclfftIrfft1DDft(&dummy, &dummy, 0, 0, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftIrfft1DDft(&dummy, &dummy, 1025, 0, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftIrfft1DC2RFft(&dummy, &dummy, 0, 0, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftIrfft1DC2RFft(&dummy, &dummy, (1u << 27) + 1, 0, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 5) batches==0 / >INT32_MAX 拒绝（issue #188/#163）
        if (aclfftIrfft1DDft(&dummy, &dummy, 256, 0, 0, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftIrfft1DDft(&dummy, &dummy, 256, 0, 0x80000000u, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftIrfft1DC2RFft(&dummy, &dummy, 2048, 0, 0, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (all_ok) {
            TEST_CASE_PASS_DETAIL(stats, "entry_reject_regression",
                                  "all irfft1_d reject paths return the documented error codes");
        } else {
            TEST_CASE_FAIL_DETAIL(stats, "entry_reject_regression",
                                  "some reject path missing or wrong error code (see build log)");
        }
    }

    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        std::vector<C2RTestCase> cases = {
            // c2r n<=1024 -> arch32/dft/ : dft_c2r
            {"c2r_basic", 8,  1},
            {"c2r_n16",   16, 1},
            // DFT 路径奇数 n（issue #227）：文档 FFT_1D.md 承诺 DFT 支持任意
            // 正整数 1<=nx<=1024，奇数 n 此前零覆盖（存量 1025 走 radix-mix 路径）
            {"c2r_dft_n7",   7,   1},
            {"c2r_dft_n15",  15,  1},
            {"c2r_dft_n255", 255, 1},
            // c2r n>1024 radix-mix -> arch32/c2r_fft/ : fft_c2r
            {"c2r_fft_n1536",  1536,  4},
            {"c2r_fft_n8192",  8192,  4},
            {"c2r_fft_n8192_nyquist", 8192, 1},
            {"c2r_fft_n2048_b100",  2048, 100},
            {"c2r_fft_odd_n1025",  1025,  4},
        };
        OpsFftTest::RunBatchTests(stream, stats, "irfft1_d", cases, run_c2r_case);
        run_entry_reject_regression_test(stats);
    }
}

REGISTER_OP_TEST(Irfft1DApiIntegration)
