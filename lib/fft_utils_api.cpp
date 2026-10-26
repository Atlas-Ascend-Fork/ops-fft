/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_ops_fft.h"

/**
 * @brief 获取错误描述字符串
 *
 * @param result 错误码
 * @return 错误描述字符串
 */
const char* aclfftGetErrorString(aclfftResult result) {
    switch (result) {
        case ACLFFT_SUCCESS:
            return "Success";
        case ACLFFT_INVALID_PLAN:
            return "Invalid plan handle";
        case ACLFFT_ALLOC_FAILED:
            return "Memory allocation failed";
        case ACLFFT_INVALID_TYPE:
            return "Invalid data type";
        case ACLFFT_INVALID_VALUE:
            return "Invalid parameter value";
        case ACLFFT_INTERNAL_ERROR:
            return "Internal error";
        case ACLFFT_EXEC_FAILED:
            return "Execution failed";
        case ACLFFT_SETUP_FAILED:
            return "Setup failed";
        case ACLFFT_INVALID_SIZE:
            return "Invalid size parameter";
        case ACLFFT_UNALIGNED_DATA:
            return "Data not aligned";
        case ACLFFT_INCOMPLETE_PARAMETER_LIST:
            return "Incomplete parameter list";
        case ACLFFT_INVALID_DEVICE:
            return "Invalid device";
        case ACLFFT_PARSE_ERROR:
            return "Parse error";
        case ACLFFT_NO_WORKSPACE:
            return "No workspace available";
        case ACLFFT_NOT_IMPLEMENTED:
            return "Feature not implemented";
        case ACLFFT_NOT_SUPPORTED:
            return "Feature not supported";
        default:
            return "Unknown error";
    }
}

// ---------------------------------------------------------------------------
// 未实现接口桩（issue #213）：头文件以 ACLFFT_API 导出但全仓无定义的 9 个接口
// 在此提供返回 ACLFFT_NOT_IMPLEMENTED 的桩，使导出符号在库内均有定义
//（Windows __declspec(dllexport) 分支下无定义符号会链接失败；Linux 下调用方
// 链接期 undefined reference）。口径与 fft_exec_api.cpp 的 aclfftExecZ2Z/
// D2Z/Z2D 桩一致；对应能力落地后以真实实现替换。
// ---------------------------------------------------------------------------

aclfftResult aclfftGetSize1d(aclfftHandle plan, int nx, aclfftType type, int batch, size_t* workSize) {
    (void)plan; (void)nx; (void)type; (void)batch; (void)workSize;
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftGetSize2d(aclfftHandle plan, int nx, int ny, aclfftType type, size_t* workSize) {
    (void)plan; (void)nx; (void)ny; (void)type; (void)workSize;
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftGetSize3d(aclfftHandle plan, int nx, int ny, int nz, aclfftType type, size_t* workSize) {
    (void)plan; (void)nx; (void)ny; (void)nz; (void)type; (void)workSize;
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftGetSize(aclfftHandle plan, size_t* workSize) {
    (void)plan; (void)workSize;
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftSetAutoAllocation(aclfftHandle plan, int autoAllocate) {
    (void)plan; (void)autoAllocate;
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftSetWorkArea(aclfftHandle plan, void* workArea) {
    (void)plan; (void)workArea;
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftSetStride(aclfftHandle plan, int32_t stride) {
    (void)plan; (void)stride;
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftGetVersion(int* version) {
    (void)version;
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftGetProperty(aclfftLibraryPropertyType type, int* value) {
    (void)type; (void)value;
    return ACLFFT_NOT_IMPLEMENTED;
}
