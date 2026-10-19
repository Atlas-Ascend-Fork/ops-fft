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

namespace Irfft1DApiIntegrationTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        std::vector<C2RTestCase> cases = {
            // c2r n<=1024 -> arch32/dft/ : dft_c2r
            {"c2r_basic", 8,  1},
            {"c2r_n16",   16, 1},
            // c2r n>1024 radix-mix -> arch32/c2r_fft/ : fft_c2r
            {"c2r_fft_n1536",  1536,  4},
            {"c2r_fft_n8192",  8192,  4},
            {"c2r_fft_n8192_nyquist", 8192, 1},
            {"c2r_fft_n2048_b100",  2048, 100},
            {"c2r_fft_odd_n1025",  1025,  4},
        };
        OpsFftTest::RunBatchTests(stream, stats, "irfft1_d", cases, run_c2r_case);
    }
}

REGISTER_OP_TEST(Irfft1DApiIntegration)
