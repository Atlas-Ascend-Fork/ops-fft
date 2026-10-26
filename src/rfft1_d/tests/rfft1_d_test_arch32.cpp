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
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        std::vector<R2CTestCase> cases = {
            // r2c n<=1024 -> arch32/dft/ : dft_r2c
            {"r2c_basic",  8,    1},
            {"r2c_n16",    16,   1},
            {"r2c_n1024",  1024, 1},
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
    }
}

REGISTER_OP_TEST(Rfft1DApiIntegration)
