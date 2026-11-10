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
// 直连入口声明（issue #295/#298 直连覆盖用例；issue #330 门控后仅 arch32 TU
// 的直连分支需要）：rfft1_d.h 经 src/ 根 include 路径可见，仅含声明
#include "rfft1_d/rfft1_d.h"

struct R2CTestCase {
    std::string name;
    uint32_t n;
    uint32_t batch;
    // 0 = plan+exec 分发（默认）；1 = 直连 aclfftRfft1DR2CFft32（issue #295 覆盖
    // calcNum>=2 的配对行 Nyquist 实部，档位取 exec 分发域之外的 2 幂 n）。
    // 直连分支仅在定义 OPS_FFT_TEST_ARCH32_DIRECT_ENTRY 的 TU 编入（issue #330）
    int direct_entry = 0;
};

inline bool run_r2c_case(aclrtStream stream, const R2CTestCase& tc) {
    std::string data_dir = OpsFftTest::GetDataRoot("rfft1_d");
    std::string input_file = data_dir + "/" + tc.name + "_input.bin";
    std::string output_file = data_dir + "/" + tc.name + "_output.bin";

    std::vector<float> input_flat;
    if (!OpsFftTest::ReadFile(input_file, input_flat) || input_flat.empty()) return false;

    uint32_t total_input = tc.n * tc.batch;
    if (input_flat.size() != total_input) return false;

    uint32_t output_size = (tc.n / 2 + 1) * tc.batch;
    std::vector<aclfftComplex> output(output_size);

    // 直连分支（issue #330）：aclfftRfft1DR2CFft32 仅在 arch32 源中定义，而本头被
    // arch32/arch35 测试 TU 共同编入（tests/CMakeLists.txt 仅剔除 _arch32 后缀
    // 翻译单元），无条件引用会使 950 开启 BUILD_TESTING 时链接 all_ops_test 报
    // undefined reference。仅在定义 OPS_FFT_TEST_ARCH32_DIRECT_ENTRY 的 TU
    // （arch32 测试）编入直连分支，arch35 保持 plan+exec 路径
#if defined(OPS_FFT_TEST_ARCH32_DIRECT_ENTRY)
    if (tc.direct_entry == 1) {
        if (aclfftRfft1DR2CFft32(input_flat.data(), reinterpret_cast<float*>(output.data()), tc.n, tc.batch, 1,
                                 stream) != ACL_SUCCESS)
            return false;
    } else
#endif
    {
        aclfftHandle plan;
        if (aclfftPlan1d(&plan, tc.n, ACLFFT_R2C, tc.batch, ACLFFT_HORIZONTAL) != ACLFFT_SUCCESS) return false;
        auto res = aclfftExecR2C(plan, input_flat.data(), output.data());
        aclfftDestroy(plan);
        if (res != ACLFFT_SUCCESS) return false;
    }

    std::vector<float> output_flat(output_size * 2);
    for (uint32_t i = 0; i < output_size; i++) {
        output_flat[i * 2] = output[i].x;
        output_flat[i * 2 + 1] = output[i].y;
    }
    // 写盘失败按用例失败处理（issue #249）：环境问题不再伪装成比对失败/假通过
    if (!OpsFftTest::WriteFile(output_file, output_flat)) {
        return false;
    }
    return true;
}

namespace Rfft1DApiIntegrationTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats);
}
