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

    return 0;
}

// ACL 清理
void ACLManager::finalize(aclrtStream& stream) {
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
        stream = nullptr;
    }
    aclrtResetDevice(GetTestDeviceId());
    aclFinalize();
}

} // namespace OpsFftTest
