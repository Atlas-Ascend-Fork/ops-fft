# API_Reference

ops-fft 提供基于 Plan 模型的 `aclfft` 系列 C 接口，用于在 Ascend NPU 上执行快速傅里叶变换。接口声明见 `include/cann_ops_fft.h`。

## 接口列表

| 文档 | 说明 |
| :--- | :--- |
| [FFT公共接口](./FFT_Common_Interface.md) | 公共接口（`aclfftCreate`/`aclfftMakePlan1d`/`aclfftMakePlan2d`/`aclfftSetStream`/`aclfftDestroy`/`aclfftGetErrorString`）、数据类型定义与错误码。 |
| [FFT_1D](./FFT_1D.md) | 一维 FFT 接口（`aclfftPlan1d`/`aclfftExecC2C`/`aclfftExecR2C`/`aclfftExecC2R`），对应算子 fft1_d、rfft1_d、irfft1_d。 |
| [FFT_2D](./FFT_2D.md) | 二维 FFT 接口（`aclfftPlan2d`/`aclfftExecC2C`），对应算子 fft2_d。 |

## 调用流程

ops-fft 采用 Plan 模型，典型调用流程如下：

1. AscendCL 初始化：`aclInit` → `aclrtSetDevice` → `aclrtCreateStream`。
2. 创建并初始化 Plan：`aclfftPlan1d`/`aclfftPlan2d`。
3. 执行变换：`aclfftExecC2C`/`aclfftExecR2C`/`aclfftExecC2R`，输入输出为 Host 侧指针，接口内部完成数据搬运与同步，返回即完成。
4. 销毁 Plan：`aclfftDestroy`。
5. AscendCL 去初始化：`aclrtDestroyStream` → `aclrtResetDevice` → `aclFinalize`。

> **说明**：当前版本 Exec 接口的输入输出为 Host 侧指针，算子内部自行管理 Device 内存与 workspace，无需用户分配 workspace（`aclfftMakePlan1d`/`aclfftMakePlan2d` 的 `workSize` 恒返回 0）。
