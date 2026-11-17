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
 * @file test_common.cpp
 * @brief 公共测试框架实现 - ACL 初始化和全局测试统计
 */

#include "test_common.h"

namespace OpsFftTest {

// 全局测试统计定义
TestStats g_global_stats;
std::vector<std::string> g_failed_cases;

// 测试设备号可经环境变量 OPS_FFT_TEST_DEVICE 覆盖（issue #362）：
// 多卡 CI 环境 0 号卡被占用/不可用时可在指定卡上运行；
// 未设置时取 0，行为与原实现一致
int GetTestDeviceId()
{
    const char *env = std::getenv("OPS_FFT_TEST_DEVICE");
    if (env == nullptr || *env == '\0') {
        return 0;
    }
    return std::atoi(env);
}

// 异常退出兜底清理（issue #384）：TEST_ASSERT 失败原先直接 std::exit(1)，ACL 资源
// 与运行期 .bin 产物均不释放，失败清单与全局汇总也无处可打。init 成功后注册
// atexit 兜底：正常收尾路径 ACLManager::finalize 先行置位跳过；断言退出路径在此
// 完成全局汇总/失败清单补打、设备/流释放与残留 bin 清理，与 main 正常收尾口径一致
namespace {

aclrtStream g_exit_cleanup_stream = nullptr;
bool g_exit_cleanup_registered = false;
bool g_acl_finalized_normally = false;

void ExitCleanupHandler()
{
    if (g_acl_finalized_normally) {
        return;
    }
    std::cout << std::endl;
    std::cout << "[WARN] abnormal exit: ACL finalize and residual data cleanup performed by atexit fallback"
              << std::endl;
    TEST_PRINT_GLOBAL_RESULT();
    TEST_PRINT_FAILED_CASES();
    if (g_exit_cleanup_stream != nullptr) {
        aclrtDestroyStream(g_exit_cleanup_stream);
        g_exit_cleanup_stream = nullptr;
    }
    aclrtResetDevice(GetTestDeviceId());
    aclFinalize();
    const char *data_root = std::getenv("OPS_FFT_TEST_DATA_ROOT");
    if (data_root != nullptr && *data_root != '\0') {
        // 清理范围与 main 收尾一致：限定各算子 *_data 测试数据目录（issue #363）
        std::string cleanup_cmd = std::string("find ") + ShellQuote(data_root) +
                                  " -type f -path '*/tests/*_data/*' -name '*.bin' -delete 2>/dev/null";
        system(cleanup_cmd.c_str());
    }
}

} // namespace

// ACL 初始化
int ACLManager::init(aclrtStream& stream) {
    // 初始化 ACL
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclInit failed: " << ret << std::endl;
        return -1;
    }

    // 设置设备
    const int device_id = GetTestDeviceId();
    ret = aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice(" << device_id << ") failed: " << ret << std::endl;
        aclFinalize();
        return -1;
    }

    // 创建流
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed: " << ret << std::endl;
        aclrtResetDevice(device_id);
        aclFinalize();
        return -1;
    }

    // atexit 兜底注册（issue #384）：TEST_ASSERT 断言失败 std::exit 退出时仍能
    // 释放 ACL 资源并清理运行期 .bin 产物；正常路径 finalize 后由置位标记跳过
    if (!g_exit_cleanup_registered) {
        g_exit_cleanup_stream = stream;
        g_exit_cleanup_registered = true;
        std::atexit(ExitCleanupHandler);
    }

    return 0;
}

// ACL 清理
void ACLManager::finalize(aclrtStream& stream) {
    // 正常收尾置位（issue #384）：atexit 兜底不再重复执行
    g_acl_finalized_normally = true;
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
        stream = nullptr;
    }
    aclrtResetDevice(GetTestDeviceId());
    aclFinalize();
}

} // namespace OpsFftTest
