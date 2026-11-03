#-----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#-----------------------------------------------------------------------------

##############################################################################
# lib/CMakeLists.txt
#
# 功能:
#   将 API 层源文件添加到最终动态库
#
# 编译语言:
#   使用 ASC 编译器编译（与算子层保持一致）
##############################################################################

##############################################################################
# 编译 API 层（添加到最终动态库）
##############################################################################

message(STATUS "==========================================")
message(STATUS "Configuring API Layer")
message(STATUS "==========================================")

# 自动收集 API 层所有 .cpp 源文件
file(GLOB API_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# 检查是否找到源文件
if(NOT API_SOURCES)
    message(FATAL_ERROR "API layer: No .cpp source files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# 直接将 API 源文件添加到最终动态库
target_sources(${OPS_FFT} PRIVATE ${API_SOURCES})

message(STATUS "API layer sources found:")
foreach(API_SRC ${API_SOURCES})
    message(STATUS "  ${API_SRC}")
endforeach()

message(STATUS "NPU architecture: ${ASCEND_NPU_ARCH}")
message(STATUS "==========================================")
message(STATUS "")

