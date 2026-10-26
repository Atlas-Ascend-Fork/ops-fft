# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# makeself 工具下载

set(MAKESELF_NAME "makeself")
set(MAKESELF_PATH ${CANN_3RD_LIB_PATH}/makeself)
set(MAKESELF_VERSION_PKG makeself-release-2.5.0-patch1.tar.gz)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# 默认配置的makeself还是不存在则下载
if (NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh" OR NOT EXISTS "${MAKESELF_PATH}/makeself.sh")
    # 优先查找本地压缩包
    if(EXISTS "${CANN_3RD_LIB_PATH}/pkg/${MAKESELF_VERSION_PKG}")
        set(REQ_URL "file://${CANN_3RD_LIB_PATH}/pkg/${MAKESELF_VERSION_PKG}")
        message(STATUS "[ThirdPartyLib][makeself] found in ${REQ_URL}.")
    elseif(EXISTS "${CANN_3RD_LIB_PATH}/${MAKESELF_VERSION_PKG}")
        set(REQ_URL "file://${CANN_3RD_LIB_PATH}/${MAKESELF_VERSION_PKG}")
        message(STATUS "[ThirdPartyLib][makeself] found in ${REQ_URL}.")
    else()
        set(REQ_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/${MAKESELF_VERSION_PKG}")
        message(STATUS "[ThirdPartyLib][makeself] ${REQ_URL} not found, need download.")
    endif()

    include(FetchContent)
    FetchContent_Declare(
        ${MAKESELF_NAME}
        URL ${REQ_URL}
        URL_HASH SHA256=bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a
        DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}/pkg
        SOURCE_DIR "${MAKESELF_PATH}"  # 直接解压到此目录
    )
    FetchContent_MakeAvailable(${MAKESELF_NAME})
    # 执行位修复（issue #209）：原块目录错误（指向不存在的 ${CMAKE_BINARY_DIR}/makeself）
    # 且 "-E env / CMAKE_TLS_VERIFY=0" 被追加为 chmod 参数、结果变量无消费，修复从未生效。
    # chmod 目标与 FetchContent 的 SOURCE_DIR(${MAKESELF_PATH}) 对齐，结果显式校验；
    # 如需放宽下载 TLS 校验应在 FetchContent 之前 set(CMAKE_TLS_VERIFY 0)，此处不涉及
    execute_process(
        COMMAND chmod 700 "${MAKESELF_PATH}/makeself.sh"
        COMMAND chmod 700 "${MAKESELF_PATH}/makeself-header.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT CHMOD_RESULT EQUAL 0)
        message(FATAL_ERROR "chmod makeself scripts failed: ${CHMOD_RESULT}")
    endif()
endif()

# 验证文件存在与可执行位（原仅查存在性，chmod 失效被掩盖）
if(NOT EXISTS "${MAKESELF_PATH}/makeself.sh")
    message(FATAL_ERROR "makeself.sh not found at ${MAKESELF_PATH}")
endif()
if(NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh")
    message(FATAL_ERROR "makeself-header.sh not found at ${MAKESELF_PATH}")
endif()
execute_process(
    COMMAND test -x "${MAKESELF_PATH}/makeself.sh"
    RESULT_VARIABLE EXEC_RESULT
)
if(NOT EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "makeself.sh is not executable after chmod")
endif()

message(STATUS "MAKESELF_PATH: ${MAKESELF_PATH}")
