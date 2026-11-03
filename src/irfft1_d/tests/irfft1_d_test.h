/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "test_common.h"
#include "cann_ops_fft.h"

struct C2RTestCase {
    std::string name;
    uint32_t n;
    uint32_t batch;
};

inline bool run_c2r_case(aclrtStream stream, const C2RTestCase& tc) {
    std::string data_dir = OpsFftTest::GetDataRoot("irfft1_d");
    std::string input_file = data_dir + "/" + tc.name + "_input.bin";
    std::string output_file = data_dir + "/" + tc.name + "_output.bin";

    std::vector<float> input_flat;
    if (!OpsFftTest::ReadFile(input_file, input_flat) || input_flat.empty()) return false;

    uint32_t total_input = (tc.n / 2 + 1) * tc.batch * 2;
    if (input_flat.size() != total_input) return false;

    uint32_t output_size = tc.n * tc.batch;
    std::vector<aclfftComplex> input_complex((tc.n / 2 + 1) * tc.batch);
    for (uint32_t i = 0; i < (tc.n / 2 + 1) * tc.batch; i++) {
        input_complex[i].x = input_flat[i * 2];
        input_complex[i].y = input_flat[i * 2 + 1];
    }

    std::vector<float> output(output_size);
    aclfftHandle plan;
    if (aclfftPlan1d(&plan, tc.n, ACLFFT_C2R, tc.batch, ACLFFT_HORIZONTAL) != ACLFFT_SUCCESS) return false;
    auto res = aclfftExecC2R(plan, input_complex.data(), output.data());
    aclfftDestroy(plan);
    if (res != ACLFFT_SUCCESS) return false;

    // 写盘失败按用例失败处理（issue #249）：环境问题不再伪装成比对失败/假通过
    if (!OpsFftTest::WriteFile(output_file, output)) {
        return false;
    }
    return true;
}

namespace Irfft1DApiIntegrationTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats);
}
