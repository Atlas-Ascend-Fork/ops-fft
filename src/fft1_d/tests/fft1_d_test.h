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

struct C2CTestCase {
    std::string name;
    uint32_t n;
    uint32_t batch;
    int direction;
    int layout = ACLFFT_HORIZONTAL;
};

// issue #400 codecheck 收敛：诊断输出补充后 run_c2c_case 超出超大函数阈值，
// 将执行阶段（roundtrip/单向 plan→exec→destroy）拆为独立 helper
inline bool exec_c2c_plan_exec(const C2CTestCase& tc, std::vector<aclfftComplex>& input,
                               std::vector<aclfftComplex>& output, uint32_t total_complex) {
    if (tc.direction == 0) {
        aclfftHandle plan_fwd, plan_bwd;
        // roundtrip 布局与正向分支一致取用例 layout 字段（issue #387）：原先硬编码
        // ACLFFT_HORIZONTAL，未来新增 VERTICAL roundtrip 用例时会被静默按水平执行
        if (aclfftPlan1d(&plan_fwd, tc.n, ACLFFT_C2C, tc.batch, tc.layout) != ACLFFT_SUCCESS) {
            std::cerr << "  [ERROR] " << tc.name << ": aclfftPlan1d forward failed" << std::endl;
            return false;
        }
        if (aclfftPlan1d(&plan_bwd, tc.n, ACLFFT_C2C, tc.batch, tc.layout) != ACLFFT_SUCCESS) {
            std::cerr << "  [ERROR] " << tc.name << ": aclfftPlan1d backward failed" << std::endl;
            aclfftDestroy(plan_fwd); return false;
        }
        std::vector<aclfftComplex> mid(total_complex);
        auto res = aclfftExecC2C(plan_fwd, input.data(), mid.data(), ACLFFT_FORWARD);
        if (res != ACLFFT_SUCCESS) {
            std::cerr << "  [ERROR] " << tc.name << ": aclfftExecC2C forward failed, ret=" << res << std::endl;
            aclfftDestroy(plan_fwd); aclfftDestroy(plan_bwd); return false;
        }
        res = aclfftExecC2C(plan_bwd, mid.data(), output.data(), ACLFFT_BACKWARD);
        aclfftDestroy(plan_fwd); aclfftDestroy(plan_bwd);
        if (res != ACLFFT_SUCCESS) {
            std::cerr << "  [ERROR] " << tc.name << ": aclfftExecC2C backward failed, ret=" << res << std::endl;
            return false;
        }
    } else {
        aclfftHandle plan;
        if (aclfftPlan1d(&plan, tc.n, ACLFFT_C2C, tc.batch, tc.layout) != ACLFFT_SUCCESS) {
            std::cerr << "  [ERROR] " << tc.name << ": aclfftPlan1d failed" << std::endl;
            return false;
        }
        auto res = aclfftExecC2C(plan, input.data(), output.data(), tc.direction);
        aclfftDestroy(plan);
        if (res != ACLFFT_SUCCESS) {
            std::cerr << "  [ERROR] " << tc.name << ": aclfftExecC2C failed, ret=" << res << std::endl;
            return false;
        }
    }
    return true;
}

inline bool run_c2c_case(aclrtStream stream, const C2CTestCase& tc) {
    std::string data_dir = OpsFftTest::GetDataRoot("fft1_d");
    std::string input_file = data_dir + "/" + tc.name + "_input.bin";
    std::string output_file = data_dir + "/" + tc.name + "_output.bin";

    std::vector<float> input_flat;
    // 失败分支诊断输出（issue #400）：区分文件缺失/尺寸不符/入口调用失败，不再静默 return false
    if (!OpsFftTest::ReadFile(input_file, input_flat) || input_flat.empty()) {
        std::cerr << "  [ERROR] " << tc.name << ": input file missing or empty: " << input_file << std::endl;
        return false;
    }

    uint32_t total_complex = tc.n * tc.batch;
    if (input_flat.size() != total_complex * 2) {
        std::cerr << "  [ERROR] " << tc.name << ": input size mismatch: got " << input_flat.size()
                  << ", expected " << total_complex * 2 << std::endl;
        return false;
    }

    std::vector<aclfftComplex> input(total_complex);
    std::vector<aclfftComplex> output(total_complex);
    for (uint32_t i = 0; i < total_complex; i++) {
        input[i].x = input_flat[i * 2];
        input[i].y = input_flat[i * 2 + 1];
    }

    if (!exec_c2c_plan_exec(tc, input, output, total_complex)) {
        return false;
    }

    std::vector<float> output_flat(total_complex * 2);
    for (uint32_t i = 0; i < total_complex; i++) {
        output_flat[i * 2] = output[i].x;
        output_flat[i * 2 + 1] = output[i].y;
    }
    // 写盘失败按用例失败处理（issue #249）：环境问题不再伪装成比对失败/假通过
    if (!OpsFftTest::WriteFile(output_file, output_flat)) {
        std::cerr << "  [ERROR] " << tc.name << ": output file write failed: " << output_file << std::endl;
        return false;
    }
    return true;
}

namespace Fft1DApiIntegrationTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats);
}
