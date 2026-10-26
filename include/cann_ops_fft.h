/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cann_ops_fft.h
 * \brief CANN FFT 公共 API 头文件
 */

#ifndef CANN_OPS_FFT_H
#define CANN_OPS_FFT_H

#include <stdint.h>
#include <stddef.h>
#include "acl/acl.h"

/* 导出宏定义 */
#if defined(_WIN32) || defined(__CYGWIN__)
    #define ACLFFT_API __declspec(dllexport)
#else
    #define ACLFFT_API __attribute__((visibility("default")))
#endif

/* 枚举和类型定义 - 放在 extern "C" 之前，确保 C++ 中可见 */
/*! @brief Result/status/error codes */
typedef enum aclfftResult_t
{
    /*! CANN FFT operation was successful */
    ACLFFT_SUCCESS = 0,
    /*! CANN FFT was passed an invalid plan handle */
    ACLFFT_INVALID_PLAN = 1,
    /*! CANN FFT failed to allocate NPU or CPU memory */
    ACLFFT_ALLOC_FAILED = 2,
    /*! Invalid or unsupported transform type, including a Plan/Exec type mismatch. */
    ACLFFT_INVALID_TYPE = 3,
    /*! User specified an invalid pointer or parameter */
    ACLFFT_INVALID_VALUE = 4,
    /*! Driver or internal CANN FFT library error */
    ACLFFT_INTERNAL_ERROR = 5,
    /*! Failed to execute an FFT on the NPU */
    ACLFFT_EXEC_FAILED = 6,
    /*! CANN FFT failed to initialize */
    ACLFFT_SETUP_FAILED = 7,
    /*! User specified an invalid transform size */
    ACLFFT_INVALID_SIZE = 8,
    /*! No longer used */
    ACLFFT_UNALIGNED_DATA = 9,
    /*! Missing parameters in call */
    ACLFFT_INCOMPLETE_PARAMETER_LIST = 10,
    /*! Execution of a plan was on different NPU than plan creation */
    ACLFFT_INVALID_DEVICE = 11,
    /*! Internal plan database error */
    ACLFFT_PARSE_ERROR = 12,
    /*! No workspace has been provided prior to plan execution */
    ACLFFT_NO_WORKSPACE = 13,
    /*! Function does not implement functionality for parameters given. */
    ACLFFT_NOT_IMPLEMENTED = 14,
    /*! Operation is not supported for parameters given. */
    ACLFFT_NOT_SUPPORTED = 16
} aclfftResult;

/*! @brief Transform type
 *  @details This type is used to declare the Fourier transform type that will be executed.
 *  */
typedef enum aclfftType_t
{
    /*! Real to complex (interleaved) */
    ACLFFT_R2C = 0x2a,
    /*! Complex (interleaved) to real */
    ACLFFT_C2R = 0x2c,
    /*! Complex to complex (interleaved) */
    ACLFFT_C2C = 0x29,
    /*! Double to double-complex (interleaved) */
    ACLFFT_D2Z = 0x6a,
    /*! Double-complex (interleaved) to double */
    ACLFFT_Z2D = 0x6c,
    /*! Double-complex to double-complex (interleaved) */
    ACLFFT_Z2Z = 0x69
} aclfftType;

typedef enum aclfftLibraryPropertyType_t
{
    ACLFFT_MAJOR_VERSION,
    ACLFFT_MINOR_VERSION,
    ACLFFT_PATCH_LEVEL
} aclfftLibraryPropertyType;

/*! @brief Perform a forward FFT.
 * */
#define ACLFFT_FORWARD (-1)
/*! @brief Perform a backward/inverse FFT.
 * */
#define ACLFFT_BACKWARD 1

#define ACLFFT_HORIZONTAL 0
#define ACLFFT_VERTICAL 1

typedef struct aclfftHandle_t* aclfftHandle;

/*! @brief Single-precision floating point complex type
 *  */
typedef struct {
    float x;  ///< 实部
    float y;  ///< 虚部
} aclfftComplex;

/*! @brief Double-precision floating point complex type
 *  */
typedef struct {
    double x;  ///< 实部
    double y;  ///< 虚部
} aclfftDoubleComplex;

/*! @brief Single-precision floating point type
 *  */
typedef float aclfftReal;

/*! @brief Double-precision floating point type
 *  */
typedef double aclfftDoubleReal;

/* 函数声明 - 使用 C 链接 */
#ifdef __cplusplus
extern "C" {
#endif

/*! @brief Create a new one-dimensional FFT plan.
 *
 *  @details Allocate and initialize a new one-dimensional FFT plan.
 *
 *  @param[out] plan Pointer to the FFT plan handle.
 *  @param[in] nx FFT length.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to compute.
 *  @param[in] dimType Dimension direction, must be ACLFFT_HORIZONTAL or ACLFFT_VERTICAL.
 */
ACLFFT_API aclfftResult aclfftPlan1d(aclfftHandle* plan,
                                     int           nx,
                                     aclfftType    type,
                                     int           batch, /* deprecated: batch 参数由 plan 携带，见 aclfftPlan1d 文档 */
                                     int           dimType);

/*! @brief Create a new two-dimensional FFT plan.
 *
 *  @details Allocate and initialize a new two-dimensional FFT plan.
 *  Two-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in y-direction (j index) vary the
 *  fastest.
 *
 *  @param[out] plan Pointer to the FFT plan handle.
 *  @param[in] nx Number of elements in the x-direction (slow index).
 *  @param[in] ny Number of elements in the y-direction (fast index).
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to compute.
 */
ACLFFT_API aclfftResult aclfftPlan2d(aclfftHandle* plan, int batch, int nx, int ny, aclfftType type);

/*! @brief Create a new three-dimensional FFT plan.
 *
 *  @details Allocate and initialize a new three-dimensional FFT plan.
 *  Three-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in z-direction (k index) vary the
 *  fastest.
 *
 *  @param[out] plan Pointer to the FFT plan handle.
 *  @param[in] nx Number of elements in the x-direction (slowest index).
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] nz Number of elements in the z-direction (fastest index).
 *  @param[in] type FFT type.
 */
ACLFFT_API aclfftResult
    aclfftPlan3d(aclfftHandle* plan, int nx, int ny, int nz, aclfftType type);

/*! @brief Allocate a new plan.
 *
 *  @param[out] plan Pointer to the FFT plan handle to be allocated.
 */
ACLFFT_API aclfftResult aclfftCreate(aclfftHandle* plan);

 /*! @brief Initialize a new one-dimensional FFT plan.
  *
  *  @details Assumes that the plan has been created already, and
  *  modifies the plan associated with the plan handle.
  *
  *  @param[in] plan Handle of the FFT plan.
  *  @param[in] nx FFT length.
  *  @param[in] type FFT type.
  *  @param[in] batch Number of batched transforms to compute.
  *  @param[in] dimType Dimension direction, must be ACLFFT_HORIZONTAL or ACLFFT_VERTICAL.
  *  @param[out] workSize Pointer to work area size (returned value).
  */
 ACLFFT_API aclfftResult aclfftMakePlan1d(aclfftHandle plan,
                                          int          nx,
                                          aclfftType   type,
                                          int          batch,
                                          int          dimType,
                                          size_t* workSize);

/*! @brief Initialize a new two-dimensional FFT plan.
 *
 *  @details Assumes that the plan has been created already, and
 *  modifies the plan associated with the plan handle.
 *  Two-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in y-direction (j index) vary the
 *  fastest.
 *
 *  @param[in] plan Handle of the FFT plan.
 *  @param[in] nx Number of elements in the x-direction (slow index).
 *  @param[in] ny Number of elements in the y-direction (fast index).
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to compute.
 *  @param[out] workSize Pointer to work area size (returned value).
 */
ACLFFT_API aclfftResult
    aclfftMakePlan2d(aclfftHandle plan, int batch, int nx, int ny, aclfftType type, size_t* workSize);

/*! @brief Initialize a new three-dimensional FFT plan.
 *
 *  @details Assumes that the plan has been created already, and
 *  modifies the plan associated with the plan handle.
 *  Three-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in z-direction (k index) vary the
 *  fastest.
 *
 *  @param[in] plan Handle of the FFT plan.
 *  @param[in] nx Number of elements in the x-direction (slowest index).
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] nz Number of elements in the z-direction (fastest index).
 *  @param[in] type FFT type.
 *  @param[out] workSize Pointer to work area size (returned value).
 */
ACLFFT_API aclfftResult
    aclfftMakePlan3d(aclfftHandle plan, int nx, int ny, int nz, aclfftType type, size_t* workSize);

/*! @brief Return size of the work area size required for a 1D plan.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] nx Number of elements in the x-direction.
 *  @param[in] type FFT type.
 *  @param[in] batch Number of batched transforms to perform.
 *  @param[out] workSize Pointer to work area size (returned value).
 */
ACLFFT_API aclfftResult aclfftGetSize1d(aclfftHandle plan,
                                        int          nx,
                                        aclfftType   type,
                                        int          batch, /* deprecated: batch 参数由 plan 携带，见 aclfftPlan1d 文档 */
                                        size_t*      workSize);

/*! @brief Return size of the work area size required for a 2D plan.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] nx Number of elements in the x-direction.
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] type FFT type.
 *  @param[out] workSize Pointer to work area size (returned value).
 */
ACLFFT_API aclfftResult
    aclfftGetSize2d(aclfftHandle plan, int nx, int ny, aclfftType type, size_t* workSize);

/*! @brief Return size of the work area size required for a 3D plan.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] nx Number of elements in the x-direction.
 *  @param[in] ny Number of elements in the y-direction.
 *  @param[in] nz Number of elements in the z-direction.
 *  @param[in] type FFT type.
 *  @param[out] workSize Pointer to work area size (returned value).
 */
ACLFFT_API aclfftResult
    aclfftGetSize3d(aclfftHandle plan, int nx, int ny, int nz, aclfftType type, size_t* workSize);

/*! @brief Return size of the work area size required for a rank-dimensional plan.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[out] workSize Pointer to work area size (returned value).
 */
ACLFFT_API aclfftResult aclfftGetSize(aclfftHandle plan, size_t* workSize);

/*! @brief Set the plan's auto-allocation flag.  The plan will allocate its own workarea.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] autoAllocate 0 to disable auto-allocation, non-zero to enable.
 */
ACLFFT_API aclfftResult aclfftSetAutoAllocation(aclfftHandle plan, int autoAllocate);

/*! @brief Set the plan's work area.
 *
 *  @param[in] plan Pointer to the FFT plan.
 *  @param[in] workArea Pointer to the work area.
 */
ACLFFT_API aclfftResult aclfftSetWorkArea(aclfftHandle plan, void* workArea);

/*! @brief Execute a (float) complex-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data.
 *  @param[out] odata Output data.
 *  @param[in] direction Either `ACLFFT_FORWARD` or `ACLFFT_BACKWARD`.
 */
ACLFFT_API aclfftResult aclfftExecC2C(aclfftHandle    plan,
                                      aclfftComplex* idata,
                                      aclfftComplex* odata,
                                      int             direction);

/*! @brief Execute a (float) real-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed. For an in-place transform, allocate enough storage
 *  for the complex output: at least `2 * (nx / 2 + 1) * batch` floats.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data.
 *  @param[out] odata Output data.
 */
ACLFFT_API aclfftResult aclfftExecR2C(aclfftHandle plan,
                                      aclfftReal*    idata,
                                      aclfftComplex* odata);

/*! @brief Execute a (float) complex-to-real FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data.
 *  @param[out] odata Output data.
 */
ACLFFT_API aclfftResult aclfftExecC2R(aclfftHandle    plan,
                                      aclfftComplex* idata,
                                      aclfftReal*    odata);

/*! @brief Execute a (double) complex-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data.
 *  @param[out] odata Output data.
 *  @param[in] direction Either `ACLFFT_FORWARD` or `ACLFFT_BACKWARD`.
 */
ACLFFT_API aclfftResult aclfftExecZ2Z(aclfftHandle            plan,
                                      aclfftDoubleComplex* idata,
                                      aclfftDoubleComplex* odata,
                                      int                     direction);

/*! @brief Execute a (double) real-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data.
 *  @param[out] odata Output data.
 */
ACLFFT_API aclfftResult aclfftExecD2Z(aclfftHandle         plan,
                                      aclfftDoubleReal*    idata,
                                      aclfftDoubleComplex* odata);

/*! @brief Execute a (double) complex-to-real FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] idata Input data.
 *  @param[out] odata Output data.
 */
ACLFFT_API aclfftResult aclfftExecZ2D(aclfftHandle            plan,
                                      aclfftDoubleComplex* idata,
                                      aclfftDoubleReal*    odata);

/*! @brief Set ACL stream to execute plan on.
 *
 *  @details Associates an ACL stream with an CANN FFT plan.  All kernels
 *  launched by this plan are associated with the provided stream.
 *
 *  @param[in] plan The FFT plan.
 *  @param[in] stream The ACL stream.
 */
ACLFFT_API aclfftResult aclfftSetStream(aclfftHandle plan, aclrtStream stream);
ACLFFT_API aclfftResult aclfftSetStride(aclfftHandle plan, int32_t stride);

/*! @brief Destroy and deallocate an existing plan.
 *
 *  @param[in] plan Handle of the FFT plan to be destroyed.
 */
ACLFFT_API aclfftResult aclfftDestroy(aclfftHandle plan);

/*! @brief Get CANN FFT version.
 *
 *  @param[out] version CANN FFT version (returned value).
 */
ACLFFT_API aclfftResult aclfftGetVersion(int* version);

/*! @brief Get library property.
 *
 *  @param[in] type Property type.
 *  @param[out] value Returned value.
 */
ACLFFT_API aclfftResult aclfftGetProperty(aclfftLibraryPropertyType type, int* value);

/*! @brief Get error description string.
 *
 *  @param[in] result Error code.
 *  @return Error description string.
 */
ACLFFT_API const char* aclfftGetErrorString(aclfftResult result);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CANN_OPS_FFT_H
