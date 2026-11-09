# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ops-fft 安装路径定义

# 包名称
set(PKG_NAME "ops_fft")

# 架构检测
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    set(ARCH aarch64)
else()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH x86_64)
endif()

# Built-in 算子包安装路径 (简化版: lib64 和 include)
set(PATH_NAME "ops_fft")

# 头文件安装到 ops_fft/include/
set(OPS_FFT_INC_INSTALL_DIR           ops_fft/include)
# 库文件安装到 ops_fft/lib64/
set(OPS_FFT_LIB_INSTALL_DIR           ops_fft/lib64)
# 版本信息
set(VERSION_INFO_INSTALL_DIR        ops_fft)

# 输出路径
# CMAKE_INSTALL_PREFIX 尊重用户传参（issue #315）：原无条件覆盖使
# -DCMAKE_INSTALL_PREFIX=... 被静默丢弃、产物恒落源码树 build_out。CMake 对
# 该变量恒有预定义（未传参时初始化为 /usr/local，NOT DEFINED 恒假），故按
# “等于内置默认值即视为未传参”判定；显式传 /usr/local 的场景按默认口径处理
if(NOT DEFINED CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build_out)
endif()

# 打包相关路径
set(SCRIPT_PACKAGE_DIR ${CMAKE_SOURCE_DIR}/scripts/package)
set(COMMON_SCRIPT_DIR ${SCRIPT_PACKAGE_DIR}/common)
set(OPS_FFT_SCRIPT_DIR ${SCRIPT_PACKAGE_DIR}/ops_fft/scripts)

message(STATUS "Package: ${PKG_NAME}")
message(STATUS "Architecture: ${ARCH}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
