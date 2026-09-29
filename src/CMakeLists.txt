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
# src/CMakeLists.txt
#
# 功能:
#   注册算子源文件，添加到最终动态库
##############################################################################

# 引入算子注册函数
include(${CMAKE_SOURCE_DIR}/cmake/func.cmake)

##############################################################################
# 注册算子并添加到最终动态库
##############################################################################

message(STATUS "==========================================")
message(STATUS "Registering operators...")
message(STATUS "==========================================")

# 首先把所有算子的 include path 加入（包括未启用的算子）
# 这样 fft_exec_api.cpp 可以无条件 include 所有算子头文件
file(GLOB ALL_OP_DIRS RELATIVE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/*)
foreach(OP_DIR ${ALL_OP_DIRS})
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/src/${OP_DIR})
        target_include_directories(${OPS_FFT} PRIVATE ${CMAKE_SOURCE_DIR}/src/${OP_DIR})
    endif()
endforeach()

# Add common subdirectories to include path
target_include_directories(${OPS_FFT} PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/host
    ${CMAKE_SOURCE_DIR}/src/common/kernel/base
    ${CMAKE_SOURCE_DIR}/src/common/kernel/fft
    ${CMAKE_SOURCE_DIR}/src/common/tiling
)


# 编译启用的算子
foreach(OP ${BUILD_OPERATORS})
    if(EXISTS ${CMAKE_SOURCE_DIR}/src/${OP}/CMakeLists.txt)
        add_subdirectory(${OP})
        message(STATUS "Added operator directory: ${OP}")
    else()
        message(WARNING "Operator ${OP} missing CMakeLists.txt, skipping")
    endif()
endforeach()

message(STATUS "==========================================")
message(STATUS "Operator registration completed")
message(STATUS "==========================================")
message(STATUS "")

# Ascend C's lower-level API headers are not covered by the public CANN include
# directories above. The shared FFT kernel helpers include kernel_event.h and
# kernel_tensor.h directly, so expose the installed implementation headers.
set(ASCENDC_BASIC_API_INCLUDE_DIR "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api")
if(IS_DIRECTORY "${ASCENDC_BASIC_API_INCLUDE_DIR}")
    target_include_directories(${OPS_FFT} PRIVATE "${ASCENDC_BASIC_API_INCLUDE_DIR}")
endif()
