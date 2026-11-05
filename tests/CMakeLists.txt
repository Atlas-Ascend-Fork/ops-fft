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
# tests/CMakeLists.txt
#
# 自动扫描所有算子测试并生成 all_tests.cpp
##############################################################################

# 只在 BUILD_TESTING 开启时执行
if(BUILD_TESTING)

    # 扫描所有测试头文件和源文件（使用通配符）
    set(TEST_HEADERS "")
    set(TEST_SOURCES "")

    foreach(OP ${BUILD_OPERATORS})
        # 跳过没有对应 arch 目录的算子的测试（全局属性由 register_operator 写入，issue #241）
        get_property(_OP_SKIPPED GLOBAL PROPERTY OP_${OP}_SKIPPED)
        if(_OP_SKIPPED)
            message(STATUS "  Skipping tests for ${OP} (no arch directory)")
            continue()
        endif()

        set(OP_TEST_DIR "${CMAKE_SOURCE_DIR}/src/${OP}/tests")

        # 检查测试目录是否存在
        if(EXISTS ${OP_TEST_DIR})
            # 扫描单元测试头文件 (*_test.h)
            # CONFIGURE_DEPENDS（issue #290，#258 已修口径）：glob 结果还决定
            # all_tests.cpp 自动生成（configure_file 模板）的 include 列表，
            # 测试文件增删后不重新 configure 即静默漏编/漏跑，须依赖重配置
            file(GLOB_RECURSE OP_TEST_HEADERS CONFIGURE_DEPENDS
                "${OP_TEST_DIR}/*test*.h"
            )
            list(APPEND TEST_HEADERS ${OP_TEST_HEADERS})

            # 扫描单元测试源文件 (*_test.cpp)，CONFIGURE_DEPENDS 同上（issue #290）
            file(GLOB_RECURSE OP_TEST_SOURCES CONFIGURE_DEPENDS
                "${OP_TEST_DIR}/*test*.cpp"
            )
            list(APPEND TEST_SOURCES ${OP_TEST_SOURCES})
        endif()
    endforeach()

    # 根据 ASCEND_NPU_ARCH 确定当前架构标签（与 func.cmake 一致）
    if("${ASCEND_NPU_ARCH}" STREQUAL "dav-3510")
        set(CURRENT_ARCH_TAG "arch35")
        set(EXCLUDE_ARCH_TAG "_arch32")
    elseif("${ASCEND_NPU_ARCH}" STREQUAL "dav-2201")
        set(CURRENT_ARCH_TAG "arch32")
        set(EXCLUDE_ARCH_TAG "_arch35")
    else()
        set(CURRENT_ARCH_TAG "arch35")
        set(EXCLUDE_ARCH_TAG "_arch32")
    endif()

    # 去重并排序
    list(REMOVE_DUPLICATES TEST_HEADERS)
    list(REMOVE_DUPLICATES TEST_SOURCES)
    list(SORT TEST_HEADERS)
    list(SORT TEST_SOURCES)

    # 过滤不属于当前架构的测试文件
    # 命名规范：*_test_arch32.cpp / *_test_arch32.h = arch32 独有
    #           *_test_arch35.cpp / *_test_arch35.h = arch35 独有
    #           *_test.cpp / *_test.h（无 arch 后缀）= 通用测试
    set(FILTERED_HEADERS "")
    set(FILTERED_SOURCES "")

    foreach(hdr ${TEST_HEADERS})
        string(FIND "${hdr}" "${EXCLUDE_ARCH_TAG}" _arch_pos)
        if(_arch_pos LESS 0)
            list(APPEND FILTERED_HEADERS ${hdr})
        endif()
    endforeach()

    foreach(src ${TEST_SOURCES})
        string(FIND "${src}" "${EXCLUDE_ARCH_TAG}" _arch_pos)
        if(_arch_pos LESS 0)
            list(APPEND FILTERED_SOURCES ${src})
        endif()
    endforeach()

    set(TEST_HEADERS ${FILTERED_HEADERS})
    set(TEST_SOURCES ${FILTERED_SOURCES})

    message(STATUS "当前架构: ${CURRENT_ARCH_TAG}, 排除后缀: ${EXCLUDE_ARCH_TAG}")
    message(STATUS "找到测试头文件: ${TEST_HEADERS}")
    message(STATUS "找到测试源文件: ${TEST_SOURCES}")

    # 生成 include 语句列表（转换为相对于 src 目录的路径）
    set(TEST_INCLUDES "")
    foreach(test_header ${TEST_HEADERS})
        # 转换为相对路径
        file(RELATIVE_PATH rel_header ${CMAKE_SOURCE_DIR}/src ${test_header})
        set(TEST_INCLUDES "${TEST_INCLUDES}#include \"${rel_header}\"\n")
    endforeach()

    # 配置生成 all_tests.cpp
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/all_tests.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/all_tests.cpp
        @ONLY
    )

    # 创建测试可执行文件
    add_executable(all_ops_test
        ${CMAKE_CURRENT_BINARY_DIR}/all_tests.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp
        ${TEST_SOURCES}
    )

    # 设置包含目录
    target_include_directories(all_ops_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_HOME_PATH}/include/
        ${ASCEND_HOME_PATH}/pkg_inc/op_common/
        ${ASCEND_HOME_PATH}/pkg_inc/base/
        ${ASCEND_HOME_PATH}/pkg_inc/
    )

    # 链接库
    target_link_libraries(all_ops_test PRIVATE
        ${OPS_FFT}  # 链接到 CMake 目标
        ${ASCEND_HOME_PATH}/lib64/libascendcl.so
        ${ASCEND_HOME_PATH}/lib64/libacl_rt.so
    )

    message(STATUS "已配置测试目标: all_ops_test")

endif()
