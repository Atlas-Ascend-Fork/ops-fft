/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_common.h
 * @brief 公共测试框架 - 提供测试统计、断言宏和 ACL 初始化功能
 */

#ifndef CANN_OPS_FFT_TEST_COMMON_H
#define CANN_OPS_FFT_TEST_COMMON_H

#include <iostream>
#include <string>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <array>
#include <map>
#include <sstream>
#include "acl/acl.h"

namespace OpsFftTest {

// 测试结果统计
struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;

    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << name << ": ";
        }
        std::cout << "总测试数=" << total << ", 通过=" << passed << ", 失败=" << failed << std::endl;
    }
};

// 全局测试统计（所有算子累计）
extern TestStats g_global_stats;
extern int g_compare_failures;
extern std::vector<std::string> g_failed_cases;

// 测试用例开始标记
#define TEST_CASE_BEGIN(test_name) \
    do { \
        std::cout << "[RUN] " << (test_name) << "..." << std::endl; \
    } while(0)

// 测试用例通过标记（自动更新统计）
#define TEST_CASE_PASS(local_stats, test_name) \
    do { \
        (local_stats).total++; \
        (local_stats).passed++; \
        OpsFftTest::g_global_stats.total++; \
        OpsFftTest::g_global_stats.passed++; \
        std::cout << "[PASS] " << (test_name) << std::endl; \
    } while(0)

// 测试用例失败标记（自动更新统计）
#define TEST_CASE_FAIL(local_stats, test_name) \
    do { \
        (local_stats).total++; \
        (local_stats).failed++; \
        OpsFftTest::g_global_stats.total++; \
        OpsFftTest::g_global_stats.failed++; \
        std::cout << "[FAIL] " << (test_name) << std::endl; \
    } while(0)

// 测试用例通过标记（带详情）
#define TEST_CASE_PASS_DETAIL(local_stats, test_name, detail) \
    do { \
        (local_stats).total++; \
        (local_stats).passed++; \
        OpsFftTest::g_global_stats.total++; \
        OpsFftTest::g_global_stats.passed++; \
        std::cout << "[PASS] " << (detail) << std::endl; \
        std::cout << "----------------------------------------" << std::endl; \
    } while(0)

// 测试用例失败标记（带详情）
#define TEST_CASE_FAIL_DETAIL(local_stats, test_name, detail) \
    do { \
        (local_stats).total++; \
        (local_stats).failed++; \
        OpsFftTest::g_global_stats.total++; \
        OpsFftTest::g_global_stats.failed++; \
        OpsFftTest::g_failed_cases.push_back(test_name); \
        std::cout << "[FAIL] " << (detail) << std::endl; \
        std::cout << "----------------------------------------" << std::endl; \
    } while(0)

// 断言宏：失败时打印错误信息并退出，成功时不打印
#define TEST_ASSERT(local_stats, condition, error_msg) \
    do { \
        if (!(condition)) { \
            (local_stats).total++; \
            (local_stats).failed++; \
            OpsFftTest::g_global_stats.total++; \
            OpsFftTest::g_global_stats.failed++; \
            std::cerr << "  [ERROR] " << (error_msg) << std::endl; \
            std::exit(1); \
        } \
    } while(0)

// 精确数组比较宏（无容差，高效 - 使用 std::equal）
#define TEST_ASSERT_ARRAY_EQ(local_stats, actual, expected, length, error_msg) \
    do { \
        bool all_match = std::equal((expected), (expected) + (length), (actual)); \
        if (!all_match) { \
            auto iter = std::mismatch((expected), (expected) + (length), (actual)); \
            size_t mismatch_idx = std::distance((expected), iter.first); \
            std::cerr << "  [ERROR] " << (error_msg) << " at index " << (mismatch_idx) << std::endl; \
        } \
        TEST_ASSERT((local_stats), all_match, error_msg); \
    } while(0)

// 容差数组比较宏（浮点数，带容差）
#define TEST_ASSERT_ARRAY_NEAR(local_stats, actual, expected, length, tol, error_msg) \
    do { \
        bool all_match = true; \
        size_t first_mismatch = 0; \
        if ((actual).size() != (expected).size()) { \
            all_match = false; \
            std::cerr << "  [ERROR] " << (error_msg) << ": actual.size() (" << (actual).size() \
                      << ") != expected.size() (" << (expected).size() << ")" << std::endl; \
        } else { \
            for (size_t _i = 0; _i < static_cast<size_t>(length); ++_i) { \
                if (std::abs((actual)[_i] - (expected)[_i]) > (tol)) { \
                    all_match = false; \
                    first_mismatch = _i; \
                    break; \
                } \
            } \
            if (!all_match) { \
                std::cerr << "  [ERROR] " << (error_msg) << " at index " << first_mismatch \
                          << ": actual=" << (actual)[first_mismatch] \
                          << ", expected=" << (expected)[first_mismatch] << std::endl; \
            } \
        } \
        TEST_ASSERT((local_stats), all_match, error_msg); \
    } while(0)

// 测试抬头宏
#define TEST_PRINT_HEADER(op_name) \
    do { \
        std::cout << "========================================" << std::endl; \
        std::cout << "    " << (op_name) << "算子单元测试" << std::endl; \
        std::cout << "========================================" << std::endl; \
        std::cout << std::endl; \
    } while(0)

// 测试结果宏（包含抬头和结尾）
#define TEST_PRINT_RESULT(op_name, local_stats) \
    do { \
        std::cout << std::endl; \
        std::cout << "========================================" << std::endl; \
        std::cout << "       " << (op_name) << "算子测试结果" << std::endl; \
        std::cout << "========================================" << std::endl; \
        (local_stats).print(#op_name); \
        std::cout << "========================================" << std::endl; \
    } while(0)

// 打印单个测试结果宏（用于循环中，name 是字符串变量）
#define TEST_PRINT_RESULT_NAME(name, local_stats) \
    do { \
        std::cout << std::endl; \
        (local_stats).print(name); \
    } while(0)

// 打印全局测试结果宏
#define TEST_PRINT_GLOBAL_RESULT() \
    do { \
        std::cout << std::endl; \
        std::cout << "========================================" << std::endl; \
        std::cout << "       全局测试结果" << std::endl; \
        std::cout << "========================================" << std::endl; \
        OpsFftTest::g_global_stats.print("全部算子"); \
        std::cout << "========================================" << std::endl; \
    } while(0)

// 打印失败用例列表宏
#define TEST_PRINT_FAILED_CASES() \
    do { \
        if (!OpsFftTest::g_failed_cases.empty()) { \
            std::cout << std::endl; \
            std::cout << "========================================" << std::endl; \
            std::cout << "       失败用例列表" << std::endl; \
            std::cout << "========================================" << std::endl; \
            for (const auto& case_name : OpsFftTest::g_failed_cases) { \
                std::cout << "  - " << case_name << std::endl; \
            } \
            std::cout << "========================================" << std::endl; \
        } \
    } while(0)

// ACL 管理
class ACLManager {
public:
    static int init(aclrtStream& stream);
    static void finalize(aclrtStream& stream);
};

// 测试函数类型
using TestFunc = void(*)(aclrtStream, TestStats&);

// 测试注册表
struct TestRegistry {
    struct TestEntry {
        const char* name;
        TestFunc func;
    };

    static std::vector<TestEntry>& get_tests() {
        static std::vector<TestEntry> tests;
        return tests;
    }

    static void register_test(const char* name, TestFunc func) {
        get_tests().push_back({name, func});
    }
};

// 自动注册宏
#define REGISTER_OP_TEST(op_name) \
    namespace op_name##Test { \
        void run_all_tests(aclrtStream, OpsFftTest::TestStats&); \
        namespace { \
            struct Registrar { \
                Registrar() { \
                    OpsFftTest::TestRegistry::register_test(#op_name, run_all_tests); \
                } \
            }; \
            static Registrar registrar; \
        } \
    }

// Read .bin file into a vector<float>
inline bool ReadFile(const std::string& filepath, std::vector<float>& buf) {
    FILE* fp = fopen(filepath.c_str(), "rb");
    if (!fp) {
        std::cerr << "  [ERROR] Cannot open " << filepath << " for reading" << std::endl;
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    size_t num_floats = fsize / sizeof(float);
    buf.resize(num_floats);
    size_t read_count = fread(buf.data(), sizeof(float), num_floats, fp);
    fclose(fp);
    if (read_count != num_floats) {
        std::cerr << "  [ERROR] Read " << read_count << " floats, expected " << num_floats << " from " << filepath << std::endl;
        return false;
    }
    return true;
}

// Write vector<float> to .bin file
inline bool WriteFile(const std::string& filepath, const std::vector<float>& buf) {
    FILE* fp = fopen(filepath.c_str(), "wb");
    if (!fp) {
        std::cerr << "  [ERROR] Cannot open " << filepath << " for writing" << std::endl;
        return false;
    }
    size_t write_count = fwrite(buf.data(), sizeof(float), buf.size(), fp);
    fclose(fp);
    if (write_count != buf.size()) {
        std::cerr << "  [ERROR] Write " << write_count << " floats, expected " << buf.size() << " to " << filepath << std::endl;
        return false;
    }
    return true;
}

// Get data directory path (relative to test executable)
inline std::string GetDataDir() {
    const char* env = std::getenv("OPS_FFT_DATA_DIR");
    if (env && std::strlen(env) > 0) {
        return std::string(env);
    }
    return "fft1_d_data/";
}

// Get operator-specific data root directory
inline std::string GetDataRoot(const std::string& op_name) {
    const char* env = std::getenv("OPS_FFT_TEST_DATA_ROOT");
    if (env && std::strlen(env) > 0) {
        return std::string(env) + "/" + op_name + "/tests/" + op_name + "_data";
    }
    char exe_path[4096] = {};
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        std::string path(exe_path);
        for (int i = 0; i < 3; i++) {
            std::string::size_type pos = path.rfind('/');
            if (pos != std::string::npos) path = path.substr(0, pos);
        }
        return path + "/src/" + op_name + "/tests/" + op_name + "_data";
    }
    return op_name + "_data";
}

// Generate test case data via python script
inline void GenCaseData(const std::string& data_dir, const std::string& name) {
    std::string cmd = "cd " + data_dir + " && PYTHONPATH=" + data_dir + "/../../common/host TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 gen_data.py --case " + name + " 2>/dev/null";
    system(cmd.c_str());
}

// Cleanup test case .bin files
inline void CleanupCaseData(const std::string& data_dir, const std::string& name) {
    std::string cmd = "rm -f " + data_dir + "/" + name + "_*.bin";
    system(cmd.c_str());
}

// Compare output vs golden and cleanup, return true if passed
inline bool CompareAndCleanup(const std::string& data_dir, const std::string& name, std::string& detail) {
    std::string cmd = "cd " + data_dir + " && PYTHONPATH=" + data_dir + "/../../common/host TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 compare_data.py --case " + name + " 2>/dev/null";
    std::array<char, 512> buffer;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        g_compare_failures++;
        detail = "compare_data.py failed to start";
        return false;
    }
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        detail += buffer.data();
    }
    int ret = pclose(pipe);
    while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r')) {
        detail.pop_back();
    }
    if (ret != 0) {
        g_compare_failures++;
        return false;
    }
    return true;
}

// Report error, mark case as failed, cleanup files, print separator
// Usage: return FailAndCleanup(stats, name, data_dir, "error message");
inline int FailAndCleanup(TestStats& stats, const std::string& name,
                          const std::string& data_dir, const std::string& error_msg) {
    std::cerr << "  [ERROR] " << error_msg << std::endl;
    TEST_CASE_FAIL(stats, name);
    std::cout << "----------------------------------------" << std::endl;
    CleanupCaseData(data_dir, name);
    return 0;
}

// Batch generate all test data (one Python call)
// 退出码检查（issue #212）：gen_data.py 失败（缺 torch/numpy/脚本异常）原先
// 无感知，后续比对会因 golden 缺失而整体失败，此处提前打印醒目告警定位根因
inline void BatchGenAllData(const std::string& data_dir) {
    std::string cmd = "cd " + data_dir + " && PYTHONPATH=" + data_dir + "/../../common/host TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -u gen_data.py";
    int genRes = system(cmd.c_str());
    if (genRes != 0) {
        std::cerr << "[ERROR] gen_data.py failed (exit=" << genRes << ") in " << data_dir
                  << "; subsequent comparisons will fail due to missing golden files." << std::endl;
    }
}

// Cleanup all .bin files in data dir
inline void CleanupAllData(const std::string& data_dir) {
    std::string cmd = "rm -f " + data_dir + "/*.bin";
    system(cmd.c_str());
}

// Batch compare all cases (one Python call), returns map of name -> (passed, detail)
struct CompareResult {
    bool passed;
    std::string detail;
};

inline std::map<std::string, CompareResult> BatchCompareAllData(const std::string& data_dir) {
    std::map<std::string, CompareResult> results;
    std::string cmd = "cd " + data_dir + " && PYTHONPATH=" + data_dir + "/../../common/host TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -u compare_data.py";
    std::string output;
    std::array<char, 4096> buffer;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return results;
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        output += buffer.data();
    }
    pclose(pipe);

    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        bool passed = false;
        std::string name, detail;
        if (line.find("[PASS]") == 0) {
            passed = true;
            size_t name_start = 7;
            size_t colon = line.find(':', name_start);
            if (colon != std::string::npos) {
                name = line.substr(name_start, colon - name_start);
                detail = line.substr(colon + 2);
            }
        } else if (line.find("[FAIL]") == 0) {
            passed = false;
            size_t name_start = 7;
            size_t colon = line.find(':', name_start);
            if (colon != std::string::npos) {
                name = line.substr(name_start, colon - name_start);
                detail = line.substr(colon + 2);
            }
        }
        if (!name.empty()) {
            while (!name.empty() && name.back() == ' ') name.pop_back();
            while (!name.empty() && name.front() == ' ') name.erase(name.begin());
            results[name] = {passed, detail};
        }
    }
    return results;
}

// Generic batch test runner: gen all → run all → compare all → report → cleanup
// CaseT: test case struct type (must have .name field)
// RunFunc: bool(aclrtStream, const CaseT&) — returns true on NPU success
template <typename CaseT, typename RunFunc>
inline void RunBatchTests(aclrtStream stream, TestStats& stats,
                           const std::string& op_name,
                           const std::vector<CaseT>& cases,
                           RunFunc run_func) {
    std::string data_dir = GetDataRoot(op_name);

    BatchGenAllData(data_dir);

    std::map<std::string, bool> npu_success;
    for (const auto& tc : cases) {
        npu_success[tc.name] = run_func(stream, tc);
    }

    auto compare_results = BatchCompareAllData(data_dir);

    for (const auto& tc : cases) {
        std::cout << "[RUN] " << tc.name << std::endl;
        if (!npu_success[tc.name]) {
            TEST_CASE_FAIL_DETAIL(stats, tc.name, "NPU execution failed");
            continue;
        }
        auto it = compare_results.find(tc.name);
        if (it == compare_results.end()) {
            TEST_CASE_FAIL_DETAIL(stats, tc.name, "No compare result found");
            continue;
        }
        if (it->second.passed) {
            TEST_CASE_PASS_DETAIL(stats, tc.name, it->second.detail);
        } else {
            TEST_CASE_FAIL_DETAIL(stats, tc.name, it->second.detail);
        }
    }

    CleanupAllData(data_dir);
}

} // namespace OpsFftTest

#endif // CANN_OPS_FFT_TEST_COMMON_H
