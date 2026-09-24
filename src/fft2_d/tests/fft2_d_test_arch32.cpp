/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft2_d_test.h"

namespace Fft2DApiIntegrationTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        std::vector<C2C2DTestCase> cases = {
            // c2c 2D radix-2 fused (32-128) -> arch32/dd/ : dd
            {"dd_32x32_forward",    32,  32,  1, ACLFFT_FORWARD},
            {"dd_32x64_forward",    32,  64,  1, ACLFFT_FORWARD},
            {"dd_64x64_forward",    64,  64,  1, ACLFFT_FORWARD},
            {"dd_64x128_forward",   64,  128, 1, ACLFFT_FORWARD},
            {"dd_128x128_forward",  128, 128, 1, ACLFFT_FORWARD},
            {"dd_32x32_backward",   32,  32,  1, ACLFFT_BACKWARD},
            {"dd_64x64_backward",   64,  64,  1, ACLFFT_BACKWARD},
            {"dd_128x128_backward", 128, 128, 1, ACLFFT_BACKWARD},
            // 多 batch 用例（issue #125）：覆盖 CalcBatchNumsPerLoop 全部 6 种
            // batchNumsPerLoop>1 组合，命中 SeparateRITWOMultiBatch /
            // MatmulWithSpecialTransfer 内核路径
            {"dd_128x32_forward_b2",  128, 32,  2, ACLFFT_FORWARD},
            {"dd_64x64_forward_b2",   64,  64,  2, ACLFFT_FORWARD},
            {"dd_64x32_forward_b4",   64,  32,  4, ACLFFT_FORWARD},
            {"dd_32x128_forward_b2",  32,  128, 2, ACLFFT_FORWARD},
            {"dd_32x64_forward_b4",   32,  64,  4, ACLFFT_FORWARD},
            {"dd_32x32_forward_b8",   32,  32,  8, ACLFFT_FORWARD},
            {"dd_64x64_backward_b2",  64,  64,  2, ACLFFT_BACKWARD},
        };
        OpsFftTest::RunBatchTests(stream, stats, "fft2_d", cases, run_c2c_2d_case);
    }
}

REGISTER_OP_TEST(Fft2DApiIntegration)
