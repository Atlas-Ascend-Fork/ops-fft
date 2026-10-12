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
#include "fft_handle_impl.h"
#include "fft_error.h"

extern "C" {
    __attribute__((weak))
    aclfftResult aclfftExecC2C_1D(aclfftHandle, aclfftComplex*, aclfftComplex*, int);
    __attribute__((weak))
    aclfftResult aclfftExecC2C_2D(aclfftHandle, aclfftComplex*, aclfftComplex*, int);

    __attribute__((weak))
    aclfftResult aclfftExecR2C_1D(aclfftHandle, aclfftReal*, aclfftComplex*);
    __attribute__((weak))
    aclfftResult aclfftExecR2C_2D(aclfftHandle, aclfftReal*, aclfftComplex*);

    __attribute__((weak))
    aclfftResult aclfftExecC2R_1D(aclfftHandle, aclfftComplex*, aclfftReal*);
    __attribute__((weak))
    aclfftResult aclfftExecC2R_2D(aclfftHandle, aclfftComplex*, aclfftReal*);
}

aclfftResult aclfftExecC2C(aclfftHandle plan,
                           aclfftComplex* idata,
                           aclfftComplex* odata,
                           int direction) {
    aclfftHandle_t* impl = plan;
    ACLFFT_CHECK_PLAN_INITIALIZED(impl);
    ACLFFT_CHECK_NULL(idata);
    ACLFFT_CHECK_NULL(odata);
    ACLFFT_CHECK_PARAM(impl->type == ACLFFT_C2C, ACLFFT_INVALID_TYPE);
    // direction 域校验（issue #134）：合法取值仅 ACLFFT_FORWARD/ACLFFT_BACKWARD，
    // 防止非法值在算子层被静默折算为反向变换
    ACLFFT_CHECK_PARAM(direction == ACLFFT_FORWARD || direction == ACLFFT_BACKWARD, ACLFFT_INVALID_VALUE);

    if (impl->rank == 1 && aclfftExecC2C_1D)
        return aclfftExecC2C_1D(plan, idata, odata, direction);
    if (impl->rank == 2 && aclfftExecC2C_2D)
        return aclfftExecC2C_2D(plan, idata, odata, direction);

    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftExecR2C(aclfftHandle plan,
                           aclfftReal* idata,
                           aclfftComplex* odata) {
    aclfftHandle_t* impl = plan;
    ACLFFT_CHECK_PLAN_INITIALIZED(impl);
    ACLFFT_CHECK_NULL(idata);
    ACLFFT_CHECK_NULL(odata);
    ACLFFT_CHECK_PARAM(impl->type == ACLFFT_R2C, ACLFFT_INVALID_TYPE);

    if (impl->rank == 1 && aclfftExecR2C_1D)
        return aclfftExecR2C_1D(plan, idata, odata);
    if (impl->rank == 2 && aclfftExecR2C_2D)
        return aclfftExecR2C_2D(plan, idata, odata);

    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftExecC2R(aclfftHandle plan,
                           aclfftComplex* idata,
                           aclfftReal* odata) {
    aclfftHandle_t* impl = plan;
    ACLFFT_CHECK_PLAN_INITIALIZED(impl);
    ACLFFT_CHECK_NULL(idata);
    ACLFFT_CHECK_NULL(odata);
    ACLFFT_CHECK_PARAM(impl->type == ACLFFT_C2R, ACLFFT_INVALID_TYPE);

    if (impl->rank == 1 && aclfftExecC2R_1D)
        return aclfftExecC2R_1D(plan, idata, odata);
    if (impl->rank == 2 && aclfftExecC2R_2D)
        return aclfftExecC2R_2D(plan, idata, odata);

    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftExecZ2Z(aclfftHandle plan,
                           aclfftDoubleComplex* idata,
                           aclfftDoubleComplex* odata,
                           int direction) {
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftExecD2Z(aclfftHandle plan,
                           aclfftDoubleReal* idata,
                           aclfftDoubleComplex* odata) {
    return ACLFFT_NOT_IMPLEMENTED;
}

aclfftResult aclfftExecZ2D(aclfftHandle plan,
                           aclfftDoubleComplex* idata,
                           aclfftDoubleReal* odata) {
    return ACLFFT_NOT_IMPLEMENTED;
}
