# CANN FFT Library - 接口实现状态文档

本文档总结了 CANN FFT 库的所有接口及其当前实现状态。

---

## 概述

CANN FFT 库提供 FFT 接口，用于在华为昇腾 NPU 上执行快速傅里叶变换。库分为两层：

- **API 层**（`lib/`）：提供面向用户的高级接口
- **算子层**（`src/`）：实现底层算子内核，使用 ASC 编译器编译

当前版本：`9.0.0`

---

## 接口分类

接口按功能分为以下类别：

1. **Plan 创建接口**：创建和初始化 FFT Plan
2. **Plan 执行接口**：执行各种类型的 FFT 变换
3. **Plan 管理接口**：设置流、工作区等
4. **工具接口**：获取错误信息、版本等

---

## 实现状态说明

| 状态 | 说明 |
|------|------|
| ✅ 已实现 | 接口已完整实现并可正常使用 |
| ❌ 未实现 | 接口未实现或仅有占位实现 |

---

## 已实现的接口

### Plan 创建与初始化

| 接口 | 功能 | 状态 |
|------|------|------|
| `aclfftCreate` | 创建空的 FFT Plan 句柄 | ✅ |
| `aclfftPlan1d` | 创建并初始化一维 FFT Plan | ✅ |
| `aclfftPlan2d` | 创建并初始化二维 FFT Plan | ✅ |
| `aclfftMakePlan1d` | 初始化一维 FFT Plan | ✅ |
| `aclfftMakePlan2d` | 初始化二维 FFT Plan | ✅ |

### Plan 执行

| 接口 | 功能 | 状态 | 底层算子 |
|------|------|------|---------|
| `aclfftExecC2C` | 执行复数到复数的 FFT | ✅ | `fft1_d`, `fft2_d` |
| `aclfftExecR2C` | 执行实数到复数的一维 FFT | ✅ | `rfft1_d` |
| `aclfftExecC2R` | 执行复数到实数的 IFFT | ✅ | `irfft1_d` |

### Plan 管理

| 接口 | 功能 | 状态 |
|------|------|------|
| `aclfftDestroy` | 销毁 FFT Plan 并释放资源 | ✅ |
| `aclfftSetStream` | 设置 Plan 的执行流 | ✅ |

### 工具接口

| 接口 | 功能 | 状态 |
|------|------|------|
| `aclfftGetErrorString` | 获取错误码的描述字符串 | ✅ |

---

## 未实现的接口

### Plan 创建与初始化

| 接口 | 功能 | 状态 |
|------|------|------|
| `aclfftPlan3d` | 创建并初始化三维 FFT Plan | ❌ |
| `aclfftMakePlan3d` | 初始化三维 FFT Plan | ❌ |

### Plan 执行（双精度）

| 接口 | 功能 | 状态 | 需要算子 |
|------|------|------|---------|
| `aclfftExecZ2Z` | 执行双精度复数到复数的 FFT | ❌ | 双精度 `c2c` |
| `aclfftExecD2Z` | 执行双精度实数到复数的 FFT | ❌ | 双精度 `rfft` |
| `aclfftExecZ2D` | 执行双精度复数到实数的 IFFT | ❌ | 双精度 `irfft` |

### 工作空间管理

| 接口 | 功能 | 状态 |
|------|------|------|
| `aclfftGetSize1d` | 获取 1D Plan 所需工作空间大小 | ❌ |
| `aclfftGetSize2d` | 获取 2D Plan 所需工作空间大小 | ❌ |
| `aclfftGetSize3d` | 获取 3D Plan 所需工作空间大小 | ❌ |
| `aclfftGetSize` | 获取已创建 Plan 的工作空间大小 | ❌ |
| `aclfftSetAutoAllocation` | 设置是否自动分配工作空间 | ❌ |
| `aclfftSetWorkArea` | 设置自定义工作空间 | ❌ |

### 库信息

| 接口 | 功能 | 状态 |
|------|------|------|
| `aclfftGetVersion` | 获取库版本号 | ❌ |
| `aclfftGetProperty` | 获取库属性（主版本号、次版本号、补丁级别） | ❌ |

---

## 算子层实现状态

### 已实现算子

#### fft1_d - 一维复数到复数 FFT

**功能**：一维复数到复数的快速傅里叶变换

**算子位置**：`src/fft1_d/`

#### fft2_d - 二维复数到复数 FFT

**功能**：二维复数到复数的快速傅里叶变换

**算子位置**：`src/fft2_d/`

#### irfft1_d - 一维复数到实数 IFFT

**功能**：一维复数到实数的逆快速傅里叶变换

**算子位置**：`src/irfft1_d/`

#### rfft1_d - 一维实数到复数 FFT

**功能**：一维实数到复数的快速傅里叶变换

**使用示例**：
```cpp
#include "cann_ops_fft.h"

// 创建 Plan
aclfftHandle plan;
aclfftPlan1d(&plan, 1024, ACLFFT_R2C, 1, ACLFFT_HORIZONTAL);

// 准备输入数据（设备指针）
aclfftReal* d_input = ...;  // 1024 个实数
aclfftComplex* d_output = ...;  // 513 个复数

// 执行 FFT
aclfftExecR2C(plan, d_input, d_output);

// 销毁 Plan
aclfftDestroy(plan);
```

**支持的特性**：
- ✅ 批量变换（batch > 1）
- ✅ 归一化模式配置
- ✅ 流式执行
- ✅ In-place/Out-of-place 变换

**算子位置**：`src/rfft1_d/`

---

## 后续实现计划

### Phase 1：完善单精度单维度 FFT

**目标**：完成所有一维单精度 FFT 接口

1. 实现 `c2c_1d` 算子，更新 `aclfftExecC2C` 实现
2. 实现 `irfft1_d` 算子，更新 `aclfftExecC2R` 实现

---

### Phase 2：高级 Plan 创建与内存管理

**目标**：8 个新接口，支持高级数据布局和内存管理

#### 高级 Plan 创建（2 个接口）

| 接口 | 功能 | 优先级 |
|------|------|--------|
| `aclfftPlanMany` | 创建高级批量 FFT Plan（支持自定义数据布局） | P1 |
| `aclfftMakePlanMany` | 初始化批量 Plan（支持嵌入维度、步长、批次距离） | P1 |

**新增特性**：
- 自定义输入/输出嵌入维度
- 自定义数据步长
- 自定义批次距离
- 支持非连续存储的高级数据布局

#### 内存管理（3 个接口）

| 接口 | 功能 | 优先级 |
|------|------|--------|
| `aclfftGetSize` | 获取 Plan 所需工作空间大小 | P1 |
| `aclfftSetWorkArea` | 设置用户管理的工作空间 | P1 |
| `aclfftSetAutoAllocation` | 设置是否自动分配工作空间 | P1 |

**内存管理模式**：
- 自动分配模式：库自动管理工作空间
- 用户管理模式：用户预分配并传入工作空间
- 共享工作空间：多个 Plan 共享同一块工作空间

#### 工具函数（3 个接口）

| 接口 | 功能 | 优先级 |
|------|------|--------|
| `aclfftSetScaleFactor` | 设置输出缩放因子 | P1 |
| `aclfftGetVersion` | 获取库版本号 | P1 |
| `aclfftGetErrorString` | 获取错误描述字符串 | P1 |

---

### Phase 3：多维度 FFT 支持

**目标**：扩展到二维和三维 FFT

1. 实现二维 FFT：实现 `c2c_2d` 算子，完成 `aclfftMakePlan2d` 实现
2. 实现三维 FFT：实现 `c2c_3d` 算子，完成 `aclfftMakePlan3d` 实现

---

### Phase 4：双精度支持

**目标**：实现双精度 FFT 接口

1. 实现双精度算子：`z2z_1d`, `d2z_1d`, `z2d_1d` 及对应的二维和三维版本
2. 更新 API 实现：`aclfftExecZ2Z`, `aclfftExecD2Z`, `aclfftExecZ2D`

---

### Phase 5：工作空间估算与库信息

**目标**：4 个新接口，支持工作空间估算和库信息查询

| 接口 | 功能 | 优先级 |
|------|------|--------|
| `aclfftGetSize1d` | 获取 1D Plan 工作空间大小 | P2 |
| `aclfftGetSize2d` | 获取 2D Plan 工作空间大小 | P2 |
| `aclfftGetSize3d` | 获取 3D Plan 工作空间大小 | P2 |
| `aclfftGetProperty` | 获取库属性（版本、设备数等） | P2 |

---

### Phase 6：高级特性

**目标**：17 个新接口，支持回调系统、多设备和扩展数据类型

#### 工作空间估算（4 个接口）

| 接口 | 功能 | 优先级 |
|------|------|--------|
| `aclfftEstimateSize1d` | 估算 1D FFT 工作空间（无需创建 Plan） | P2 |
| `aclfftEstimateSize2d` | 估算 2D FFT 工作空间 | P2 |
| `aclfftEstimateSize3d` | 估算 3D FFT 工作空间 | P2 |
| `aclfftEstimateSizeMany` | 估算批量 FFT 工作空间 | P2 |

#### 扩展工具函数（2 个接口）

| 接口 | 功能 | 优先级 |
|------|------|--------|
| `aclfftGetStream` | 获取当前执行流 | P2 |
| `aclfftGetProperty` | 获取库属性（设备数、最大 Plan 数等） | P2 |

#### 回调系统（3 个接口）

| 接口 | 功能 | 优先级 |
|------|------|--------|
| `aclfftSetCallback` | 设置回调函数（数据预处理/后处理） | P2 |
| `aclfftClearCallback` | 清除回调函数 | P2 |
| `aclfftSetCallbackSharedSize` | 设置回调共享内存大小 | P2 |

**回调类型**：
- Load 回调：FFT 计算前数据预处理
- Store 回调：FFT 计算后数据后处理

#### 多设备支持（5 个接口）

| 接口 | 功能 | 优先级 |
|------|------|--------|
| `aclfftSetDevices` | 设置 Plan 使用的设备列表 | P2 |
| `aclfftMultiMalloc` | 分配多设备内存 | P2 |
| `aclfftMultiFree` | 释放多设备内存 | P2 |
| `aclfftMultiMemcpy` | 多设备内存拷贝 | P2 |
| `aclfftExecMulti` | 使用多设备描述符执行 | P2 |

**多设备数据分布策略**：
- Batch 分割：按 batch 维度分配到不同设备
- 维度分割：按空间维度分割（2D/3D）

#### 扩展数据类型（3 个接口）

| 接口 | 功能 | 优先级 |
|------|------|--------|
| `aclfftMakePlanManyExt` | 使用扩展数据类型创建 Plan | P2 |
| `aclfftGetSizeManyExt` | 获取扩展 Plan 的工作空间 | P2 |
| `aclfftExecExt` | 通用执行函数（类型无关） | P2 |

**支持的扩展类型**：
- 半精度浮点（FP16）
- BFloat16
- 半精度/BFloat16 复数

---

## 实现优先级总结

| Phase | 接口数量 | 优先级 | 预计工期 | 主要功能 |
|-------|---------|--------|----------|---------|
| **Phase 1** | 2 个算子 | P0 | 1 周 | 完善单精度 1D FFT |
| **Phase 2** | 8 个 API | P1 | 2 周 | 高级 Plan 创建与内存管理 |
| **Phase 3** | 2 个算子 + 2 个 API | P1 | 1 周 | 多维度 FFT 支持 |
| **Phase 4** | 3 个算子 + 3 个 API | P1 | 1 周 | 双精度支持 |
| **Phase 5** | 4 个 API | P2 | 1 周 | 工作空间估算 |
| **Phase 6** | 17 个 API | P2 | 3-4 周 | 高级特性（回调、多设备、扩展类型） |

**总计**：
- 算子：7 个（Phase 1: 2, Phase 3: 2, Phase 4: 3）
- API 接口：36 个（当前已实现: 11，后续规划: 25）

---

## 统计摘要

### 当前实现状态

| 类别 | 总数 | 已实现 | 未实现 |
|------|------|--------|--------|
| Plan 创建 | 4 | 3 | 1 |
| Plan 初始化 | 3 | 2 | 1 |
| Plan 执行（单精度） | 3 | 3 | 0 |
| Plan 执行（双精度） | 3 | 0 | 3 |
| Plan 管理 | 4 | 2 | 2 |
| 工作空间查询 | 4 | 0 | 4 |
| 工具接口 | 3 | 1 | 2 |
| **当前总计** | **24** | **11** | **13** |

**当前完成度**：
- 已实现：45.8% (11/24)
- 未实现：54.2% (13/24)

---

### 完整规划（包含后续 Phase）

| Phase | 接口分类 | 接口数量 | 实现状态 |
|-------|---------|---------|---------|
| **当前版本** | 基础接口 | 24 | 11 已实现 / 13 未实现 |
| **Phase 2** | 高级 Plan 创建 | 2 | 规划中 |
| | 内存管理 | 3 | 规划中 |
| | 工具函数 | 3 | 规划中 |
| **Phase 3** | 多维度支持 | 2 个算子 | 规划中 |
| **Phase 4** | 双精度支持 | 3 个算子 | 规划中 |
| **Phase 5** | 工作空间估算 | 4 | 规划中 |
| **Phase 6** | 回调系统 | 3 | 规划中 |
| | 多设备支持 | 5 | 规划中 |
| | 扩展数据类型 | 3 | 规划中 |
| | 扩展工具 | 2 | 规划中 |
| **总计** | **所有接口** | **51+** | **11 已实现 / 40+ 规划中** |

**完整规划统计**：
- 当前已实现：11 个接口（21.6%）
- Phase 2 规划：8 个接口（高级功能）
- Phase 3 规划：2 个算子
- Phase 4 规划：3 个算子
- Phase 5 规划：4 个接口
- Phase 6 规划：17 个接口（高级特性）
- 总计：51+ 个接口和算子

---

## 附录：数据类型定义

### 主要数据类型

| 类型 | 说明 |
|------------|--------------|
| `aclfftHandle` | Plan 句柄 |
| `aclfftResult` | 错误码 |
| `aclfftType` | FFT 类型 |
| `aclfftComplex` | 单精度复数 |
| `aclfftDoubleComplex` | 双精度复数 |
| `aclfftReal` | 单精度实数 |
| `aclfftDoubleReal` | 双精度实数 |
| `aclrtStream` | 流句柄 |

### 错误码定义

| 错误码 | 说明 |
|--------|------|
| `ACLFFT_SUCCESS` | 成功 |
| `ACLFFT_INVALID_PLAN` | 无效 Plan |
| `ACLFFT_ALLOC_FAILED` | 分配失败 |
| `ACLFFT_INVALID_TYPE` | 无效类型 |
| `ACLFFT_INVALID_VALUE` | 无效参数 |
| `ACLFFT_INTERNAL_ERROR` | 内部错误 |
| `ACLFFT_EXEC_FAILED` | 执行失败 |
| `ACLFFT_SETUP_FAILED` | 设置失败 |
| `ACLFFT_INVALID_SIZE` | 无效大小 |
| `ACLFFT_UNALIGNED_DATA` | 数据未对齐 |
| `ACLFFT_INCOMPLETE_PARAMETER_LIST` | 参数不完整 |
| `ACLFFT_INVALID_DEVICE` | 无效设备 |
| `ACLFFT_PARSE_ERROR` | 解析错误 |
| `ACLFFT_NO_WORKSPACE` | 无工作空间 |
| `ACLFFT_NOT_IMPLEMENTED` | 未实现 |
| `ACLFFT_NOT_SUPPORTED` | 不支持 |

---

**文档版本**：9.0.0
**最后更新**：2026-03-16
**维护者**：CANN FFT Team
