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
# 环境初始化模块
# 功能：检查环境、设置 NPU 架构、发现算子
##############################################################################

macro(init_env)
    # 1. 检查 ASCEND 环境
    _check_ascend()

    # 2. 设置 NPU 架构
    _setup_npu_arch()

    # 3. 发现算子
    _discover_operators()
endmacro()

##############################################################################
# 内部宏
##############################################################################

# 1. 检查 ASCEND 路径
macro(_check_ascend)
    if(NOT EXISTS "${ASCEND_HOME_PATH}")
        message(FATAL_ERROR
            "✗ ASCEND_HOME_PATH not found: ${ASCEND_HOME_PATH}\n"
            "  Fix: source /usr/local/Ascend/cann/set_env.sh"
        )
    endif()
    message(STATUS "✓ ASCEND: ${ASCEND_HOME_PATH}")
endmacro()

# 2. 设置 NPU 架构（SoC 映射）
macro(_setup_npu_arch)
    # SoC 映射收敛为单一事实源（issue #257）：表定义见 soc_map.cmake，
    # build.sh 侧亦经该文件查询，不再两处重复维护
    include(${CMAKE_SOURCE_DIR}/cmake/soc_map.cmake)

    # 确定架构优先级：显式指定 > SoC映射 > 默认
    if(DEFINED ASCEND_NPU_ARCH AND NOT "${ASCEND_NPU_ARCH}" STREQUAL "")
        message(STATUS "? NPU: ${ASCEND_NPU_ARCH} (user)")
    elseif(DEFINED ASCEND_SOC)
        fft_soc_to_npu_arch("${ASCEND_SOC}" _ARCH)
        if(_ARCH STREQUAL "")
            message(FATAL_ERROR "? Unknown SoC: ${ASCEND_SOC}")
        endif()
        set(ASCEND_NPU_ARCH "${_ARCH}")
        message(STATUS "? NPU: ${ASCEND_NPU_ARCH} (${ASCEND_SOC})")
    else()
        set(ASCEND_NPU_ARCH "dav-3510")
        message(STATUS "? NPU: ${ASCEND_NPU_ARCH} (default)")
    endif()
endmacro()

# 3. 发现算子
macro(_discover_operators)
    # 清空变量（防止多次 configure 时累积）
    set(OPERATOR_DIRS "")
    set(BUILD_OPERATORS "")

    # 扫描 src 目录
    file(GLOB OPS RELATIVE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/*)
    foreach(OP ${OPS})
        # 仅收录含 CMakeLists.txt 的算子目录（issue #334）：src/common 等公共
        # 目录混入候选集会让 --ops=common 通过校验却注册 0 个算子，产生
        # "空测试假 SUCCESS"的空跑；默认全量 configure 也不再触发
        # "Operator common missing CMakeLists.txt" 的 WARNING 噪音
        if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/src/${OP} AND EXISTS ${CMAKE_SOURCE_DIR}/src/${OP}/CMakeLists.txt)
            list(APPEND OPERATOR_DIRS ${OP})
        endif()
    endforeach()
    list(SORT OPERATOR_DIRS)

    # 解析用户指定（不强制清空，检查是否已定义）
    if(NOT DEFINED ENABLED_OPERATORS OR "${ENABLED_OPERATORS}" STREQUAL "")
        set(BUILD_OPERATORS ${OPERATOR_DIRS})
        message(STATUS "✓ Build: all (${OPERATOR_DIRS})")
    else()
        string(REPLACE "," ";" OP_LIST "${ENABLED_OPERATORS}")
        foreach(OP ${OP_LIST})
            list(FIND OPERATOR_DIRS ${OP} IDX)
            if(IDX EQUAL -1)
                message(FATAL_ERROR "✗ Unknown operator: ${OP}")
            endif()
            list(APPEND BUILD_OPERATORS ${OP})
        endforeach()
        message(STATUS "✓ Build: ${BUILD_OPERATORS}")
    endif()
endmacro()
