/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 直连覆盖分支按 arch 门控（issue #330）：本 TU 定义该宏后，共享头中的
// aclfftRfft1DR2CFft32 直连分支仅在此编译单元编入（arch35 侧不引用该符号）
#define OPS_FFT_TEST_ARCH32_DIRECT_ENTRY
#include "rfft1_d_test.h"
#include "rfft1_d/rfft1_d.h"

namespace Rfft1DApiIntegrationTest {
    // issue #235: 直连执行回归——twLen==0（单级 radix 分解）时 mix 族设备缓冲曾
    // 别名共享（dev.tw 与 dev.dft 指向同一地址），析构双重释放；修复后直连执行
    // 应返回成功且进程存活（回归信号为崩溃/ACL 内存报错）
    static void run_twlen_zero_direct_exec_test(aclrtStream stream, OpsFftTest::TestStats& stats) {
        TEST_CASE_BEGIN("twlen_zero_direct_exec");
        bool all_ok = true;
        {
            // aclfftRfft1DR2CFft n=4：fftN=2 单因子分解，GetTwMatrixLen==0
            std::vector<float> in(4, 1.0f);
            std::vector<aclfftComplex> out(3);
            if (aclfftRfft1DR2CFft(in.data(), reinterpret_cast<float*>(out.data()), 4, 0, 1, 1, stream) !=
                ACL_SUCCESS) all_ok = false;
        }
        // fft1_d 域的 aclfftFft1DMix n=64 直连执行用例见 fft1_d_test_arch32.cpp
        if (all_ok) {
            TEST_CASE_PASS_DETAIL(stats, "twlen_zero_direct_exec",
                                  "R2C direct entry with twLen==0 executes without double free");
        } else {
            TEST_CASE_FAIL_DETAIL(stats, "twlen_zero_direct_exec",
                                  "twLen==0 direct execution failed (double free or ACL error)");
        }
    }

    // issue #293: fft2/fft3_1/fft3_2/fft4 路径输出行宽 N+2 个 float，第 +N+1 槽位为
    // Nyquist 虚部（实输入下恒 0），原先无写入者、host 缓冲也未清零，输出含未初始
    // 化内存。修复后启动前 memset 清零；本用例在各路径 2 幂档位内直连执行并断言
    // 该槽位确定性为 0（若清零被移除，设备脏内存场景下将出现非零值）
    static void run_nyquist_imag_zero_test(aclrtStream stream, OpsFftTest::TestStats& stats) {
        TEST_CASE_BEGIN("r2c_nyquist_imag_zero");
        struct PathCase {
            const char* name;
            uint32_t n;
            aclError (*entry)(float*, float*, uint32_t, uint32_t, int, void*);
        };
        const PathCase pcs[] = {
            {"fft2", 256, aclfftRfft1DR2CFft2},
            {"fft3_1", 32768, aclfftRfft1DR2CFft31},
            {"fft3_2", 131072, aclfftRfft1DR2CFft32},
            {"fft4", 4194304, aclfftRfft1DR2CFft4},
        };
        bool all_ok = true;
        for (const auto& pc : pcs) {
            std::vector<float> in(pc.n, 1.0f);
            // 每行 N+2 个 float（(N/2+1) 个复数），batch=1
            std::vector<float> out(pc.n + 2, 7.0f);
            aclError err = pc.entry(in.data(), out.data(), pc.n, 1, 1, stream);
            if (err != ACL_SUCCESS) {
                std::cout << "[RUN] nyquist " << pc.name << ": entry returned " << err << std::endl;
                all_ok = false;
                continue;
            }
            if (out[pc.n + 1] != 0.0f) {
                std::cout << "[RUN] nyquist " << pc.name << ": y[N+1]=" << out[pc.n + 1] << " (expect 0)" << std::endl;
                all_ok = false;
            }
        }
        if (all_ok) {
            TEST_CASE_PASS_DETAIL(stats, "r2c_nyquist_imag_zero",
                                  "Nyquist imag slot y[N+1] is zero on fft2/fft3_1/fft3_2/fft4 paths");
        } else {
            TEST_CASE_FAIL_DETAIL(stats, "r2c_nyquist_imag_zero",
                                  "Nyquist imag slot not zero (see build log)");
        }
    }

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
        // 6) isForward 方向契约（issue #240）：R2C 恒为正向仅接受 1，语义非法方向显式拒绝
        if (aclfftRfft1DR2CFft(&dummy, &dummy, 2048, 0, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DDft(&dummy, &dummy, 256, 0, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        //    幂次六直连入口 isForward!=1 拒绝（issue #284，#240 契约统一遗留面：
        //    原先形参被静默丢弃按正向计算返回成功）
        if (aclfftRfft1DR2CFft1(&dummy, &dummy, 8, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft2(&dummy, &dummy, 16, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft31(&dummy, &dummy, 256, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft32(&dummy, &dummy, 4096, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft4(&dummy, &dummy, 65536, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft5(&dummy, &dummy, 1048576, 1, 0, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 7) 幂次直连入口门禁（issue #237/#239）：n=1 越界写防护、batches==0、
        //    batch*(n+2) 溢出——原先只在 exec 分发层设防，直连可绕过
        if (aclfftRfft1DR2CFft1(&dummy, &dummy, 1, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft1(&dummy, &dummy, 2, 0, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft31(&dummy, &dummy, 65536, 40000, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft5(&dummy, &dummy, 67108864, 0, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 8) 非 2 幂 n 拒绝（issue #292）：幂次六直连入口为 2 幂档位实现，非 2 幂 n
        //    因子分解失真曾静默算错（覆盖 3、5、6、7、10、12 取值）
        if (aclfftRfft1DR2CFft1(&dummy, &dummy, 3, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft2(&dummy, &dummy, 5, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft31(&dummy, &dummy, 6, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft32(&dummy, &dummy, 7, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft4(&dummy, &dummy, 10, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft5(&dummy, &dummy, 12, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        // 9) fft2 直连档位域拒绝（issue #294）：域为 (128, 16384]——n>16384 时内核
        //    blockN=2621440/N 截断为 0、除零与跨核互等死锁；n<=128 时 master 内核
        //    直连实测挂死（存量缺陷），入口与 exec 分发域同口径整体拒绝
        if (aclfftRfft1DR2CFft2(&dummy, &dummy, 4194304, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft2(&dummy, &dummy, 16, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
        if (aclfftRfft1DR2CFft2(&dummy, &dummy, 128, 1, 1, nullptr) != ACL_ERROR_INVALID_PARAM) all_ok = false;
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
            {"r2c_fft5_2",  134217728,   2},

            // issue #295: fft3_2 直连覆盖——exec 分发域外 2 幂 n（16384 走 fft2 档）
            // 单核 calcNum>=2 的组合，第二段 SetValueBatch 原先写死 1 行，
            // 配对行 Nyquist 实部缺失；直连+golden 对拍验证
            {"r2c_fft3_2_direct_b200",  16384,  200, 1},
        };
        OpsFftTest::RunBatchTests(stream, stats, "rfft1_d", cases, run_r2c_case);
        run_entry_reject_regression_test(stats);
        run_twlen_zero_direct_exec_test(stream, stats);
        run_nyquist_imag_zero_test(stream, stats);
    }
}

REGISTER_OP_TEST(Rfft1DApiIntegration)
