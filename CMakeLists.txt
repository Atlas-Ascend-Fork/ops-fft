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
# ops-fft CMake 配置
##############################################################################

cmake_minimum_required(VERSION 3.16)

##############################################################################
# 一、基础配置
##############################################################################

# 项目声明
project(ops_fft)

# 包含变量和版本配置
include(cmake/func.cmake)
include(cmake/variables.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/version.cmake)

# 设置项目变量
set(OPS_FFT ops_fft)
set(PROJECT_VERSION "9.0.0")

# 查找 ASC 编译器
find_package(ASC REQUIRED)
project(${OPS_FFT} VERSION ${PROJECT_VERSION} LANGUAGES ASC CXX)

##############################################################################
# 二、环境初始化（封装）
##############################################################################

include(cmake/init_env.cmake)
init_env()  # 发现算子

##############################################################################
# 三、核心构建流程
##############################################################################

# 3.1 编译测试（可选）
option(BUILD_TESTING "Build tests" OFF)

##############################################################################
# 3.2 创建最终动态库（统一管理 API 层和算子层）
##############################################################################

# 创建占位源文件
file(WRITE ${CMAKE_BINARY_DIR}/ops_fft_stub.cpp "// Stub file for ops_fft library")

# 创建最终动态库目标
add_library(${OPS_FFT} SHARED ${CMAKE_BINARY_DIR}/ops_fft_stub.cpp)

# 设置库属性
set_target_properties(${OPS_FFT} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "cann_ops_fft"  # 生成 libcann_ops_fft.so
)

# 设置公共包含目录
target_include_directories(${OPS_FFT} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
    ${ASCEND_HOME_PATH}/pkg_inc/op_common/
    ${ASCEND_HOME_PATH}/pkg_inc/base/
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/op_common/
)

# 设置库链接目录（目标属性，不影响其他目标）
target_link_directories(${OPS_FFT} PUBLIC ${ASCEND_HOME_PATH}/lib64)

# 链接 CANN 库
target_link_libraries(${OPS_FFT} PUBLIC
    tiling_api
)

# 设置编译选项
target_compile_options(${OPS_FFT} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASCEND_NPU_ARCH}>
)

##############################################################################
# 3.3 编译 API 层（添加到最终动态库）
##############################################################################

add_subdirectory(lib)

##############################################################################
# 3.4 编译算子层（添加到最终动态库）
##############################################################################

add_subdirectory(src)

# 全部算子因架构不支持被跳过时提前失败（issue #211）：ops_fft_stub.cpp 仍会以
# --npu-arch dav-2101 交给 ASC 编译器并在深处报 "Unsupported NPU architecture"，
# 错误信息无法指向 SoC 选型错误；此处给出明确根因提示
get_property(_FFT_ALL_SKIPPED GLOBAL PROPERTY OPS_FFT_ALL_OPERATORS_SKIPPED)
if(_FFT_ALL_SKIPPED)
    message(FATAL_ERROR
        "All operators were skipped: ASCEND_NPU_ARCH '${ASCEND_NPU_ARCH}' is not supported. "
        "Supported SoC: Ascend950 (dav-3510/arch35), Ascend910B (dav-2201/arch32). "
        "Check your --soc option.")
endif()

##############################################################################
# 3.5 统一设置所有源文件语言属性
##############################################################################

# 获取所有源文件
get_target_property(ALL_SOURCES ${OPS_FFT} SOURCES)

# 设置所有源文件语言为 ASC
if(ALL_SOURCES)
    set_source_files_properties(${ALL_SOURCES} PROPERTIES LANGUAGE ASC)
    message(STATUS "Set ASC language for all sources")
endif()

##############################################################################
# 3.6 安装规则
##############################################################################

# 安装最终动态库
install(TARGETS ${OPS_FFT}
    LIBRARY DESTINATION ${OPS_FFT_LIB_INSTALL_DIR}
    RUNTIME DESTINATION bin
)

# 安装公共头文件（安装到 math_libs 子目录）
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cann_ops_fft.h
    DESTINATION ${OPS_FFT_INC_INSTALL_DIR}
)

##############################################################################
# 3.7 编译测试
##############################################################################
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

##############################################################################
# 3.8 打包配置
##############################################################################
option(ENABLE_PACKAGE "Enable package generation" OFF)
if(ENABLE_PACKAGE)
    # 设置第三方库路径
    if(NOT CANN_3RD_LIB_PATH)
        set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party CACHE STRING "CANN third party lib path")
    endif()
    # 检查构建依赖
    check_pkg_build_deps(ops_fft)
    # 生成版本信息目标
    add_version_info_targets()
    # 打包
    include(cmake/package.cmake)
    pack_built_in()
endif()

##############################################################################
# 四、构建摘要
##############################################################################

message(STATUS "")
message(STATUS "==========================================")
message(STATUS "  Build Summary")
message(STATUS "==========================================")
message(STATUS "  Version   : ${PROJECT_VERSION}")
message(STATUS "  NPU       : ${ASCEND_NPU_ARCH}")
message(STATUS "  Operators : ${BUILD_OPERATORS}")
message(STATUS "  Tests     : ${BUILD_TESTING}")
message(STATUS "==========================================")
