# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# makeself_built_in.cmake - ops-fft 内置自解压打包脚本

# 设置 makeself 路径
set(MAKESELF_EXE ${CPACK_MAKESELF_PATH}/makeself.sh)
set(MAKESELF_HEADER_EXE ${CPACK_MAKESELF_PATH}/makeself-header.sh)

if(NOT MAKESELF_EXE)
    message(FATAL_ERROR "makeself not found!")
endif()

# 创建临时安装目录
set(STAGING_DIR "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging")

# 先删除再创建
file(REMOVE_RECURSE "${STAGING_DIR}")
file(MAKE_DIRECTORY "${STAGING_DIR}")

# 执行安装到临时目录
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${CPACK_CMAKE_BINARY_DIR}" --prefix "${STAGING_DIR}"
    RESULT_VARIABLE INSTALL_RESULT
)

if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Installation to staging directory failed: ${INSTALL_RESULT}")
endif()

# 生成安装配置文件
set(CSV_OUTPUT "${CPACK_CMAKE_BINARY_DIR}/filelist.csv")
file(REMOVE "${CSV_OUTPUT}")
execute_process(
    COMMAND python3 "${CPACK_CMAKE_SOURCE_DIR}/scripts/package/package.py"
        --pkg_name ops_fft --chip_name "${CPACK_SOC}" --os_arch "linux-${CPACK_ARCH}"
    WORKING_DIRECTORY "${CPACK_CMAKE_BINARY_DIR}"
    OUTPUT_VARIABLE result
    ERROR_VARIABLE error
    RESULT_VARIABLE code
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "package.py exit code: ${code}")
if(NOT "${code}" STREQUAL "0")
    message(FATAL_ERROR "Filelist generation failed (exit code: ${code}). stdout: ${result} stderr: ${error}")
endif()

if(NOT EXISTS "${CSV_OUTPUT}")
    message(FATAL_ERROR "Filelist generation succeeded but output file was not created: ${CSV_OUTPUT}")
endif()
message(STATUS "Filelist generated successfully")

# 复制版本信息
set(SCENE_OUT_PUT ${CPACK_CMAKE_BINARY_DIR}/scene.info)
set(OPS_FFT_VERSION_OUT_PUT ${CPACK_CMAKE_BINARY_DIR}/version.ops_fft.info)
set(OPS_FFT_VERSION_H ${CPACK_CMAKE_BINARY_DIR}/ops_fft_version.h)

if(EXISTS ${SCENE_OUT_PUT})
    configure_file(
        ${SCENE_OUT_PUT}
        ${STAGING_DIR}/share/info/ops_fft/
        COPYONLY
    )
endif()

if(EXISTS ${CSV_OUTPUT})
    configure_file(
        ${CSV_OUTPUT}
        ${STAGING_DIR}/share/info/ops_fft/script/
        COPYONLY
    )
endif()

if(EXISTS ${OPS_FFT_VERSION_OUT_PUT})
    configure_file(
        ${OPS_FFT_VERSION_OUT_PUT}
        ${STAGING_DIR}/share/info/ops_fft/
        COPYONLY
    )
endif()

if(EXISTS ${OPS_FFT_VERSION_H})
    configure_file(
        ${OPS_FFT_VERSION_H}
        ${STAGING_DIR}/share/info/ops_fft/
        COPYONLY
    )
endif()

# makeself打包
file(STRINGS ${CPACK_CMAKE_BINARY_DIR}/makeself.txt script_output)
string(REPLACE " " ";" makeself_param_string "${script_output}")
string(REGEX MATCH "cann.*\\.run" package_name "${makeself_param_string}")

list(LENGTH makeself_param_string LIST_LENGTH)
math(EXPR INSERT_INDEX "${LIST_LENGTH} - 2")
list(INSERT makeself_param_string ${INSERT_INDEX} "${STAGING_DIR}")

message(STATUS "script output: ${script_output}")
message(STATUS "makeself: ${makeself_param_string}")
message(STATUS "package: ${package_name}")

execute_process(
    COMMAND bash ${MAKESELF_EXE}
        --header ${MAKESELF_HEADER_EXE}
        --help-header share/info/ops_fft/script/help.info
        ${makeself_param_string} share/info/ops_fft/script/install.sh
    WORKING_DIRECTORY ${STAGING_DIR}
    RESULT_VARIABLE EXEC_RESULT
    ERROR_VARIABLE EXEC_ERROR
)

if(NOT EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "makeself packaging failed: ${EXEC_ERROR}")
endif()

# 移动最终包到输出目录
execute_process(
    COMMAND mkdir -p ${CPACK_PACKAGE_DIRECTORY}
)
execute_process(
    COMMAND mv ${STAGING_DIR}/${package_name} ${CPACK_PACKAGE_DIRECTORY}/
)
execute_process(
    COMMAND echo "Build package success: ${CPACK_PACKAGE_DIRECTORY}/${package_name}"
)

message(STATUS "Package created successfully: ${CPACK_PACKAGE_DIRECTORY}/${package_name}")
