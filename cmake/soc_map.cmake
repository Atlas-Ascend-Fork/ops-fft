# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# SoC -> NPU arch 单一事实源（issue #257）：build.sh 与 cmake 的映射均由本文件提供，
# 新增/调整 SoC 仅需修改 MAPPINGS 一处。

set(SOC_TO_NPU_ARCH_MAPPINGS
    "Ascend950,dav-3510"
    "Ascend910B,dav-2201"
    "Ascend910_93,dav-2201"
    "Ascend910,dav-2101"
    "Ascend310P,dav-2101"
)

# 供 include 使用：查表并设置 ${out_var}；未识别时 out_var 为空
function(fft_soc_to_npu_arch soc_name out_var)
    foreach(_entry ${SOC_TO_NPU_ARCH_MAPPINGS})
        string(REPLACE "," ";" _pair "${_entry}")
        list(GET _pair 0 _name)
        list(GET _pair 1 _arch)
        if(_name STREQUAL "${soc_name}")
            set(${out_var} "${_arch}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} "" PARENT_SCOPE)
endfunction()

# 脚本模式：cmake -DSOC_NAME=<名称> -P soc_map.cmake → stdout 输出 arch；
# 未识别的 SoC 输出 UNSUPPORTED_SOC 并以非零码退出（build.sh 据此报错）
if(DEFINED SOC_NAME)
    fft_soc_to_npu_arch("${SOC_NAME}" _RES)
    if(_RES STREQUAL "")
        execute_process(COMMAND ${CMAKE_COMMAND} -E echo "UNSUPPORTED_SOC")
        message(FATAL_ERROR "Unknown SoC: ${SOC_NAME}")
    else()
        # cmake message() 走 stderr，stdout 输出须经 execute_process（build.sh 捕获用）
        execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${_RES}")
    endif()
endif()
