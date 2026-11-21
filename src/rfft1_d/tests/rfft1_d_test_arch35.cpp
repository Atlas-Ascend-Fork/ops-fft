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

namespace Rfft1DApiIntegrationTest {
    // issue #408: 对照 arch32 的 entry_reject_regression（rfft1_d_test_arch32.cpp，
    // #227 引入），aclfftRfft1D（FastDFT 直连入口）在 rfft1_d_fast_dft.cpp 有 5 道
    // 负向守卫（空指针/n==0/n>4096/norm 非法/batches==0），arch35 测试此前仅正向
    // 精度用例、零拒绝断言——入口校验被重构删除时 950 侧 UT 仍全绿（#147/#284
    // 同类先例）。逐一断言既有校验链的错误码
    static void run_entry_reject_regression_test(OpsFftTest::TestStats& stats) {
        TEST_CASE_BEGIN("entry_reject_regression");
        float dummy = 0.0f;
        bool all_ok = true;
        // 1) 空指针拒绝（issue #76 同族口径）
        if (aclfftRfft1D(nullptr, &dummy, 256, 0, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1D(&dummy, nullptr, 256, 0, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 2) n==0 拒绝
        if (aclfftRfft1D(&dummy, &dummy, 0, 0, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 3) n 上界拒绝（FastDFT 支持域 (0, 4096]）
        if (aclfftRfft1D(&dummy, &dummy, 4097, 0, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 4) norm 值域拒绝（0=BACKWARD/1=ORTHO/2=FORWARD，issue #118 口径）
        if (aclfftRfft1D(&dummy, &dummy, 256, 3, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1D(&dummy, &dummy, 256, -1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 5) batches==0 拒绝（issue #226）
        if (aclfftRfft1D(&dummy, &dummy, 256, 0, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (all_ok) {
            TEST_CASE_PASS_DETAIL(stats, "entry_reject_regression",
                                  "all aclfftRfft1D reject paths return ACL_ERROR_INVALID_PARAM");
        } else {
            TEST_CASE_FAIL_DETAIL(stats, "entry_reject_regression",
                                  "some reject path missing or wrong error code (see build log)");
        }
    }

    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        std::vector<R2CTestCase> cases = {
            // r2c n<=4096 -> arch35/fast_dft/ : rfft1_d_fast_dft
            {"r2c_n2048",  2048,  1},
            {"r2c_n4096",  4096,  1},
            {"r2c_n2100",  2100,  1},
            // issue #408: odd-n FastDFT coverage (documented 1<=n<=1024 lower
            // segment of the FastDFT domain; 999 hits the n%16 in 1..8 matrix
            // branch, 1023 the zero-padded branch)
            {"r2c_n999",   999,   1},
            {"r2c_n1023",  1023,  1},
            // r2c n>4096 radix-mix -> arch35/fft/ : fft_r2c_multi_core
            {"r2c_n5040",  5040,  1},
            {"r2c_n25200", 25200, 1},
        };
        OpsFftTest::RunBatchTests(stream, stats, "rfft1_d", cases, run_r2c_case);
        run_entry_reject_regression_test(stats);
    }
}

REGISTER_OP_TEST(Rfft1DApiIntegration)
