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
    // 前置声明：调用点位于 run_all_tests 内，而定义在其后（static 内部链接，
    // 调用点之前必须有可见声明，否则编译失败）
    static void run_makeplan2d_type_reject_test(OpsFftTest::TestStats& stats);

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
        run_makeplan2d_type_reject_test(stats);
    }

    // issue #208: aclfftMakePlan2d 创建期拒绝当前无 2D 执行实现的类型
    //（Z2Z/D2Z/R2C/C2R），错误码 ACLFFT_NOT_SUPPORTED，plan 不被置为已初始化，
    // 与 aclfftPlan2d/aclfftMakePlan1d 门禁口径一致
    static void run_makeplan2d_type_reject_test(OpsFftTest::TestStats& stats) {
        TEST_CASE_BEGIN("makeplan2d_type_reject");
        bool all_ok = true;
        const aclfftType unsupported_types[] = {ACLFFT_Z2Z, ACLFFT_D2Z, ACLFFT_Z2D, ACLFFT_R2C, ACLFFT_C2R};
        for (aclfftType type : unsupported_types) {
            aclfftHandle plan = nullptr;
            if (aclfftCreate(&plan) != ACLFFT_SUCCESS) {
                all_ok = false;
                continue;
            }
            size_t workSize = 0;
            aclfftResult res = aclfftMakePlan2d(plan, 1, 64, 64, type, &workSize);
            if (res != ACLFFT_NOT_SUPPORTED) {
                all_ok = false;
            }
            aclfftDestroy(plan);
        }
        // C2C 正常路径不受影响
        aclfftHandle plan = nullptr;
        if (aclfftCreate(&plan) == ACLFFT_SUCCESS) {
            size_t workSize = 0;
            if (aclfftMakePlan2d(plan, 1, 64, 64, ACLFFT_C2C, &workSize) != ACLFFT_SUCCESS) {
                all_ok = false;
            }
            aclfftDestroy(plan);
        } else {
            all_ok = false;
        }
        if (all_ok) {
            TEST_CASE_PASS_DETAIL(stats, "makeplan2d_type_reject",
                                   "MakePlan2d rejects Z2Z/D2Z/Z2D/R2C/C2R with ACLFFT_NOT_SUPPORTED; C2C unaffected");
        } else {
            TEST_CASE_FAIL_DETAIL(stats, "makeplan2d_type_reject",
                                   "MakePlan2d type gate mismatch (see build log)");
        }
    }
}

REGISTER_OP_TEST(Fft2DApiIntegration)
