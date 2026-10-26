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
# func.cmake
#
# 公共函数集合
#
# 提供统一的算子注册和测试注册接口
##############################################################################

##############################################################################
# 函数: register_operator
#
# 功能: 注册算子源文件到 ops_fft 动态库
#
# 参数:
#   NAME        - 算子名称 (必需)
#   ARCH_DIR    - 架构特定目录名 (可选，自动根据SoC选择)
#
# 说明:
#   - 自动收集当前目录下所有 .cpp 文件
#   - 根据 ASCEND_NPU_ARCH 自动选择 arch 目录:
#     dav-3510 (Ascend950) -> arch35
#     dav-2201 (Ascend910B) -> arch32
#   - ARCH_DIR 参数仍可手动覆盖自动选择
##############################################################################
function(register_operator)
    # 解析参数
    cmake_parse_arguments(ARG
        ""                              # 选项
        "NAME;ARCH_DIR"                 # 单值参数
        ""                              # 多值参数
        ${ARGN}
    )

    # 必需参数检查
    if(NOT ARG_NAME)
        message(FATAL_ERROR "register_operator: NAME parameter is required")
    endif()

    # 根据 ASCEND_NPU_ARCH 自动选择架构目录
    if(NOT ARG_ARCH_DIR)
        if("${ASCEND_NPU_ARCH}" STREQUAL "dav-3510")
            set(ARG_ARCH_DIR "arch35")
        elseif("${ASCEND_NPU_ARCH}" STREQUAL "dav-2201")
            set(ARG_ARCH_DIR "arch32")
        else()
            # 未知架构显式跳过（issue #211，对齐 build.md "缺少实现目录会被跳过"
            # 口径）：原静默回退 arch35 会以 --npu-arch dav-2101 编译面向 dav-3510
            # 的源码，错误 SoC 选型被掩盖到内核编译期才暴露甚至产出未验证二进制；
            # 全部算子均被跳过时由顶层 CMakeLists 提前 FATAL_ERROR
            message(STATUS "  Operator: ${ARG_NAME} - SKIPPED (unsupported ASCEND_NPU_ARCH '${ASCEND_NPU_ARCH}')")
            set(OP_${ARG_NAME}_SKIPPED TRUE PARENT_SCOPE)
            get_property(_ANY_SUPPORTED GLOBAL PROPERTY OPS_FFT_HAS_SUPPORTED_OPERATOR)
            if(NOT _ANY_SUPPORTED)
                set_property(GLOBAL PROPERTY OPS_FFT_ALL_OPERATORS_SKIPPED TRUE)
            endif()
            target_include_directories(${OPS_FFT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            return()
        endif()
    endif()

    # 记录"至少一个算子以受支持架构注册"（issue #211）：供顶层判定是否需要
    # 提前失败——仅跳过部分算子（如 fft2_d 无 arch32 目录）时正常继续
    set_property(GLOBAL PROPERTY OPS_FFT_HAS_SUPPORTED_OPERATOR TRUE)

    # 检查 arch 目录是否存在
    set(ARCH_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_ARCH_DIR}")
    if(NOT EXISTS "${ARCH_DIR_PATH}")
        message(STATUS "  Operator: ${ARG_NAME} - SKIPPED (no ${ARG_ARCH_DIR} directory for ${ASCEND_NPU_ARCH})")
        set(OP_${ARG_NAME}_SKIPPED TRUE PARENT_SCOPE)
        # 即使跳过，也加入算子主目录的 include path（让 fft_exec_api.cpp 能找到头文件）
        target_include_directories(${OPS_FFT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        return()
    endif()
    set(OP_${ARG_NAME}_SKIPPED FALSE PARENT_SCOPE)

    message(STATUS "  Operator: ${ARG_NAME}, ARCH_DIR: ${ARG_ARCH_DIR} (NPU: ${ASCEND_NPU_ARCH})")

    # 算子名称转大写
    string(TOUPPER ${ARG_NAME} OP_UPPER)

    # 算子源文件路径
    set(OP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    # 收集当前目录下的所有 .cpp 文件（非递归）
    file(GLOB CPP_SOURCES "${OP_SOURCE_DIR}/*.cpp")

    # 收集架构目录下的所有 .cpp 文件（非递归）
    file(GLOB_RECURSE ARCH_SOURCES "${OP_SOURCE_DIR}/${ARG_ARCH_DIR}/*.cpp")

    # 合并所有源文件
    set(ALL_SOURCES ${CPP_SOURCES} ${ARCH_SOURCES})

    message(STATUS "  Registered operator: ${ARG_NAME}")
    if(CPP_SOURCES)
        message(STATUS "    C++ Sources: ${CPP_SOURCES}")
    endif()
    if(ARCH_SOURCES)
        message(STATUS "    ASC Sources: ${ARCH_SOURCES}")
    endif()

    # 一次性将所有源文件添加到 ops_fft 库
    if(ALL_SOURCES)
        target_sources(${OPS_FFT} PRIVATE ${ALL_SOURCES})
    endif()

    # 添加算子特定的包含目录（主目录 + arch目录及其子目录）
    # 递归收集arch目录下的所有子目录作为include path
    file(GLOB_RECURSE ARCH_INC_DIRS "${OP_SOURCE_DIR}/${ARG_ARCH_DIR}/*.h")
    foreach(inc_file ${ARCH_INC_DIRS})
        get_filename_component(inc_dir ${inc_file} DIRECTORY)
        list(APPEND ARCH_INCLUDE_DIRS ${inc_dir})
    endforeach()
    list(REMOVE_DUPLICATES ARCH_INCLUDE_DIRS)

    target_include_directories(${OPS_FFT} PRIVATE
        ${OP_SOURCE_DIR}
        ${OP_SOURCE_DIR}/${ARG_ARCH_DIR}
        ${ARCH_INCLUDE_DIRS}
    )


endfunction()

# 说明（issue #198）：原 register_operator_test(OP_NAME) 函数已删除——全仓无
# 调用点的死代码，且其假设的 tests/<op>_test.cpp 布局与仓库实际的
# tests/all_tests.cpp.in 公共测试框架不符，一旦被调用将因源文件不存在导致
# CMake 配置失败。测试注册由 tests/CMakeLists.txt 的 all_tests 模板机制承担。
##############################################################################
# 版本和依赖管理函数
##############################################################################

##############################################################################
# 宏: replace_cur_major_minor_ver
#
# 功能: 替换依赖版本变量中的 CUR_MAJOR_MINOR_VER 占位符
##############################################################################
macro(replace_cur_major_minor_ver)
    string(REPLACE CUR_MAJOR_MINOR_VER "${CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_VERSION_MAJOR_MINOR}" depend "${depend}")
endmacro()

##############################################################################
# 函数: set_package
#
# 功能: 设置包名和版本号
#
# 参数:
#   name        - 包名称 (必需)
#   VERSION     - 版本号 (必需)
##############################################################################
function(set_package name)
    cmake_parse_arguments(VERSION "" "VERSION" "" ${ARGN})
    set(VERSION "${VERSION_VERSION}")
    if(NOT name)
        message(FATAL_ERROR "The name parameter is not set in set_package.")
    endif()
    if(NOT VERSION)
        message(FATAL_ERROR "The VERSION parameter is not set in set_package(${name}).")
    endif()
    string(REGEX MATCH "^([0-9]+\\.[0-9]+)" VERSION_MAJOR_MINOR "${VERSION}")
    list(APPEND CANN_VERSION_PACKAGES "${name}")
    set(CANN_VERSION_PACKAGES "${CANN_VERSION_PACKAGES}" PARENT_SCOPE)
    set(CANN_VERSION_CURRENT_PACKAGE "${name}" PARENT_SCOPE)
    set(CANN_VERSION_${name}_VERSION "${VERSION}" PARENT_SCOPE)
    set(CANN_VERSION_${name}_VERSION_MAJOR_MINOR "${VERSION_MAJOR_MINOR}" PARENT_SCOPE)
    set(CANN_VERSION_${name}_BUILD_DEPS PARENT_SCOPE)
    set(CANN_VERSION_${name}_RUN_DEPS PARENT_SCOPE)
endfunction()

##############################################################################
# 函数: set_build_dependencies
#
# 功能: 设置构建依赖
#
# 参数:
#   pkg_name    - 依赖包名称 (必需)
#   depend      - 依赖版本信息 (必需)
##############################################################################
function(set_build_dependencies pkg_name depend)
    if(NOT CANN_VERSION_CURRENT_PACKAGE)
        message(FATAL_ERROR "The set_package must be invoked first.")
    endif()
    if(NOT pkg_name)
        message(FATAL_ERROR "The pkg_name parameter is not set in set_build_dependencies.")
    endif()
    if(NOT depend)
        message(FATAL_ERROR "The depend parameter is not set in set_build_dependencies.")
    endif()
    replace_cur_major_minor_ver()
    list(APPEND CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_BUILD_DEPS "${pkg_name}" "${depend}")
    set(CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_BUILD_DEPS "${CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_BUILD_DEPS}" PARENT_SCOPE)
endfunction()

##############################################################################
# 函数: set_run_dependencies
#
# 功能: 设置运行时依赖
#
# 参数:
#   pkg_name    - 依赖包名称 (必需)
#   depend      - 依赖版本信息 (必需)
##############################################################################
function(set_run_dependencies pkg_name depend)
    if(NOT CANN_VERSION_CURRENT_PACKAGE)
        message(FATAL_ERROR "The set_package must be invoked first.")
    endif()
    if(NOT pkg_name)
        message(FATAL_ERROR "The pkg_name parameter is not set in set_run_dependencies.")
    endif()
    if(NOT depend)
        message(FATAL_ERROR "The depend parameter is not set in set_run_dependencies.")
    endif()
    replace_cur_major_minor_ver()
    list(APPEND CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_RUN_DEPS "${pkg_name}" "${depend}")
    set(CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_RUN_DEPS "${CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_RUN_DEPS}" PARENT_SCOPE)
endfunction()

##############################################################################
# 函数: check_pkg_build_deps
#
# 功能: 检查构建依赖是否满足
#
# 参数:
#   pkg_name    - 包名称 (必需)
##############################################################################
function(check_pkg_build_deps pkg_name)
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_build_dependencies.py "$ENV{ASCEND_HOME_PATH}" ${CANN_VERSION_${pkg_name}_BUILD_DEPS}
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "Check ${pkg_name} build dependencies failed!")
    endif()
endfunction()

##############################################################################
# 函数: add_version_info_targets
#
# 功能: 添加版本信息生成目标
#
# 说明:
#   - 为每个已注册的包生成 version.{pkg_name}.info 文件
#   - 目标名格式为：version_{pkg_name}_info
##############################################################################
function(add_version_info_targets)
    foreach(pkg_name ${CANN_VERSION_PACKAGES})
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/version.${pkg_name}.info
            COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_version_info.py --output ${CMAKE_BINARY_DIR}/version.${pkg_name}.info
                    "${CANN_VERSION_${pkg_name}_VERSION}" ${CANN_VERSION_${pkg_name}_RUN_DEPS}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_version_info.py
            VERBATIM
        )
        add_custom_target(version_${pkg_name}_info ALL DEPENDS ${CMAKE_BINARY_DIR}/version.${pkg_name}.info)
    endforeach()
endfunction()