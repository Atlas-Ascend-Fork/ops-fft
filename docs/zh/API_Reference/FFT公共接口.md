# FFT公共接口

ops-fft 基于 Plan（计划）模型组织 FFT 计算：先创建/初始化一个 Plan，再执行变换，最后销毁 Plan。本文档描述 Plan 生命周期与配置类公共接口（`aclfftCreate`/`aclfftMakePlan1d`/`aclfftMakePlan2d`/`aclfftSetStream`/`aclfftDestroy`/`aclfftGetErrorString`）、数据类型定义以及错误码；对外调用入口 `aclfftPlan1d`/`aclfftPlan2d` 与执行接口（`aclfftExecC2C`/`aclfftExecR2C`/`aclfftExecC2R`）请分别参考 [FFT_1D](./FFT_1D.md)、[FFT_2D](./FFT_2D.md)。

> **说明**：ops-fft 的接口风格借鉴 cuFFT，接口前缀为 `aclfft`。`aclfftExecC2C`/`aclfftExecR2C`/`aclfftExecC2R` 的输入输出参数为 **Host 侧指针**（`aclfftComplex*`/`aclfftReal*`），算子内部完成 Host→Device 拷贝、核函数执行、Device→Host 拷贝与流同步，**接口返回即表示计算完成**，无需用户手动管理 Device 内存与 workspace。

## 数据类型定义

| 类型 | 定义 | 说明 |
| :--- | :--- | :--- |
| `aclfftHandle` | `typedef struct aclfftHandle_t* aclfftHandle` | FFT Plan 句柄，由 `aclfftCreate`/`aclfftPlan1d`/`aclfftPlan2d` 创建。 |
| `aclfftComplex` | `struct { float x; float y; }` | 单精度复数，`x` 为实部，`y` 为虚部，实虚交错存储。 |
| `aclfftDoubleComplex` | `struct { double x; double y; }` | 双精度复数（当前版本未启用）。 |
| `aclfftReal` | `typedef float aclfftReal` | 单精度实数。 |
| `aclfftDoubleReal` | `typedef double aclfftDoubleReal` | 双精度实数（当前版本未启用）。 |

## 枚举与宏定义

### aclfftType（变换类型）

```Cpp
typedef enum aclfftType_t {
    ACLFFT_R2C = 0x2a,  // 实数到复数
    ACLFFT_C2R = 0x2c,  // 复数到实数
    ACLFFT_C2C = 0x29,  // 复数到复数
    ACLFFT_D2Z = 0x6a,  // 双精度实数到复数（当前版本未实现）
    ACLFFT_Z2D = 0x6c,  // 双精度复数到实数（当前版本未实现）
    ACLFFT_Z2Z = 0x69   // 双精度复数到复数（当前版本未实现）
} aclfftType;
```

### 变换方向

| 宏 | 值 | 说明 |
| :--- | :---: | :--- |
| `ACLFFT_FORWARD` | -1 | 正向变换（FFT）。对应标准 DFT，不做缩放。 |
| `ACLFFT_BACKWARD` | 1 | 逆向变换。对应未归一化的逆 DFT，结果乘以 N（即 `N × IFFT`）。 |

> **归一化说明**：当前版本归一化模式固定为 BACKWARD 语义，不可通过公共 API 配置。
> - `ACLFFT_FORWARD`：$y[k]=\sum_{n=0}^{N-1}x[n]\cdot e^{-j2\pi kn/N}$，无缩放。
> - `ACLFFT_BACKWARD`：$x[n]=\sum_{k=0}^{N-1}y[k]\cdot e^{j2\pi kn/N}$，结果乘以 N（等价于 `torch.fft.ifft(...) * N`）。若需得到归一化的逆变换，需对结果额外除以 N。

### 维度方向（dimType）

| 宏 | 值 | 说明 |
| :--- | :---: | :--- |
| `ACLFFT_HORIZONTAL` | 0 | 横向 FFT，对每一行做一维变换，数据按行连续排布。 |
| `ACLFFT_VERTICAL` | 1 | 纵向 FFT，对每一列做一维变换，列间以 `batch` 为步长排布。 |

## aclfftCreate

- **功能描述**：分配并创建一个新的 FFT Plan 句柄。创建后 Plan 尚未初始化，需进一步调用 `aclfftMakePlan1d`/`aclfftMakePlan2d` 配置变换参数后才能执行。

- **函数原型**：

```Cpp
aclfftResult aclfftCreate(aclfftHandle* plan);
```

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>plan（aclfftHandle*）</td>
      <td>输出</td>
      <td>指向 Plan 句柄的指针，用于接收创建的句柄。不能为空；<code>*plan</code> 应为 nullptr（空句柄）后调用，若需重复创建，须先通过 <code>aclfftDestroy</code> 释放旧句柄，否则旧句柄将无法释放（内存泄漏）。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：返回 `aclfftResult` 状态码，具体参见[错误码](#错误码aclfftresult)。

## aclfftMakePlan1d

- **功能描述**：对一个已通过 `aclfftCreate` 创建的 Plan 进行一维 FFT 配置初始化，设置信号长度、变换类型、批次数与维度方向。对外入口 `aclfftPlan1d` 即由 `aclfftCreate` 与本接口组合而成。

- **函数原型**：

```Cpp
aclfftResult aclfftMakePlan1d(aclfftHandle plan,
                              int          nx,
                              aclfftType   type,
                              int          batch,
                              int          dimType,
                              size_t*      workSize);
```

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>plan（aclfftHandle）</td>
      <td>输入</td>
      <td>已创建但尚未初始化的 Plan 句柄。</td>
    </tr>
    <tr>
      <td>nx（int）</td>
      <td>输入</td>
      <td>一维 FFT 信号长度 N。需大于 0。各芯片支持的取值范围参见 [FFT_1D](./FFT_1D.md#约束说明)。</td>
    </tr>
    <tr>
      <td>type（aclfftType）</td>
      <td>输入</td>
      <td>变换类型：<ul><li><code>ACLFFT_C2C</code>：复数到复数。</li><li><code>ACLFFT_R2C</code>：实数到复数。</li><li><code>ACLFFT_C2R</code>：复数到实数。</li></ul></td>
    </tr>
    <tr>
      <td>batch（int）</td>
      <td>输入</td>
      <td>批处理变换数量。需大于 0。</td>
    </tr>
    <tr>
      <td>dimType（int）</td>
      <td>输入</td>
      <td>变换维度方向：<ul><li><code>ACLFFT_HORIZONTAL</code>：横向 FFT，内部 stride 置 1。</li><li><code>ACLFFT_VERTICAL</code>：纵向 FFT，内部将 batch 转为列间 stride、batch 置 1。</li></ul></td>
    </tr>
    <tr>
      <td>workSize（size_t*）</td>
      <td>输出</td>
      <td>用于返回所需 workspace 大小。当前版本始终返回 0（算子内部自行管理 device 内存），可不分配 workspace。传空指针则不返回。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：返回 `aclfftResult` 状态码，具体参见[错误码](#错误码aclfftresult)。

## aclfftMakePlan2d

- **功能描述**：对一个已通过 `aclfftCreate` 创建的 Plan 进行二维 FFT 配置初始化。对外入口 `aclfftPlan2d` 即由 `aclfftCreate` 与本接口组合而成。

- **函数原型**：

```Cpp
aclfftResult aclfftMakePlan2d(aclfftHandle plan,
                              int          batch,
                              int          nx,
                              int          ny,
                              aclfftType   type,
                              size_t*      workSize);
```

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>plan（aclfftHandle）</td>
      <td>输入</td>
      <td>已创建但尚未初始化的 Plan 句柄。</td>
    </tr>
    <tr>
      <td>batch（int）</td>
      <td>输入</td>
      <td>批处理变换数量。需大于 0。</td>
    </tr>
    <tr>
      <td>nx（int）</td>
      <td>输入</td>
      <td>x 方向（慢索引）元素个数。需大于 0。支持的取值范围参见 [FFT_2D](./FFT_2D.md#约束说明)。</td>
    </tr>
    <tr>
      <td>ny（int）</td>
      <td>输入</td>
      <td>y 方向（快索引）元素个数。需大于 0。支持的取值范围参见 [FFT_2D](./FFT_2D.md#约束说明)。</td>
    </tr>
    <tr>
      <td>type（aclfftType）</td>
      <td>输入</td>
      <td>变换类型，二维当前仅支持 <code>ACLFFT_C2C</code>。</td>
    </tr>
    <tr>
      <td>workSize（size_t*）</td>
      <td>输出</td>
      <td>用于返回所需 workspace 大小。当前版本始终返回 0，可不分配 workspace。传空指针则不返回。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：返回 `aclfftResult` 状态码，具体参见[错误码](#错误码aclfftresult)。

## aclfftSetStream

- **功能描述**：为 Plan 绑定 ACL 流。Plan 执行时所有核函数启动均关联到该流。若不调用本接口，Plan 使用默认流（`nullptr`）。

- **函数原型**：

```Cpp
aclfftResult aclfftSetStream(aclfftHandle plan, aclrtStream stream);
```

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>plan（aclfftHandle）</td>
      <td>输入</td>
      <td>FFT Plan 句柄，需为有效且未销毁的句柄。</td>
    </tr>
    <tr>
      <td>stream（aclrtStream）</td>
      <td>输入</td>
      <td>ACL 流对象。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：返回 `aclfftResult` 状态码，具体参见[错误码](#错误码aclfftresult)。

> **说明**：由于 Exec 接口内部已完成流同步，一般情况下无需显式调用 `aclrtSynchronizeStream`。若用户自行管理流并在 Exec 之后复用该流，建议显式同步。

## aclfftDestroy

- **功能描述**：销毁 Plan 句柄并释放其占用的资源。销毁后句柄不可再使用。

- **函数原型**：

```Cpp
aclfftResult aclfftDestroy(aclfftHandle plan);
```

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>plan（aclfftHandle）</td>
      <td>输入</td>
      <td>待销毁的 FFT Plan 句柄。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：返回 `aclfftResult` 状态码，具体参见[错误码](#错误码aclfftresult)。

## aclfftGetErrorString

- **功能描述**：根据错误码返回对应的英文描述字符串。

- **函数原型**：

```Cpp
const char* aclfftGetErrorString(aclfftResult result);
```

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>result（aclfftResult）</td>
      <td>输入</td>
      <td>错误码。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：指向错误描述字符串的指针。

## 错误码（aclfftResult）

`aclfftResult` 定义于 `include/cann_ops_fft.h`，各取值含义如下：

| 枚举值 | 数值 | 说明 |
| :--- | :---: | :--- |
| `ACLFFT_SUCCESS` | 0 | 操作成功。 |
| `ACLFFT_INVALID_PLAN` | 1 | 无效的 Plan 句柄（如已销毁或未创建）。 |
| `ACLFFT_ALLOC_FAILED` | 2 | NPU 或 CPU 内存分配失败。 |
| `ACLFFT_INVALID_TYPE` | 3 | 变换类型不匹配（如 Plan 类型与 Exec 接口不一致）。 |
| `ACLFFT_INVALID_VALUE` | 4 | 无效参数（如空指针、`dimType` 非法）。 |
| `ACLFFT_INTERNAL_ERROR` | 5 | 驱动或内部库错误。 |
| `ACLFFT_EXEC_FAILED` | 6 | NPU 执行失败（含当前 SoC 不支持该算子路径）。 |
| `ACLFFT_SETUP_FAILED` | 7 | 初始化失败。 |
| `ACLFFT_INVALID_SIZE` | 8 | 无效变换尺寸（如 `nx`/`batch` ≤ 0）。 |
| `ACLFFT_UNALIGNED_DATA` | 9 | 保留，当前未使用。 |
| `ACLFFT_INCOMPLETE_PARAMETER_LIST` | 10 | 参数缺失。 |
| `ACLFFT_INVALID_DEVICE` | 11 | Plan 创建与执行不在同一 NPU。 |
| `ACLFFT_PARSE_ERROR` | 12 | 内部 Plan 解析错误。 |
| `ACLFFT_NO_WORKSPACE` | 13 | 执行前未提供 workspace。 |
| `ACLFFT_NOT_IMPLEMENTED` | 14 | 功能未实现（如超出支持的 `fftSize` 范围、3D 变换；3D plan 创建期传入双精度类型也返回本码）。 |
| `ACLFFT_NOT_SUPPORTED` | 16 | 参数组合不支持（如 `fft2_d` 尺寸不在支持集合内、1D/2D plan 创建期传入当前无执行实现的双精度类型 `Z2Z`/`D2Z`/`Z2D`）。 |

> **说明**：数值 15 在枚举中未使用。算子层内部若返回 ACL 错误，Exec 分发层按错误类型转换：`ACL_ERROR_INVALID_PARAM`（如质因子不支持、尺寸超限）转换为 `ACLFFT_NOT_IMPLEMENTED`，其余运行期失败统一转换为 `ACLFFT_EXEC_FAILED`。

## 约束说明

- 当前版本仅支持单精度（FP32）。`aclfftExecZ2Z`/`aclfftExecD2Z`/`aclfftExecZ2D`（双精度）接口已声明但返回 `ACLFFT_NOT_IMPLEMENTED`。
- `aclfftPlan3d`/`aclfftMakePlan3d`（三维变换）已声明但返回 `ACLFFT_NOT_IMPLEMENTED`。
- `aclfftGetVersion`/`aclfftGetProperty`/`aclfftSetAutoAllocation`/`aclfftSetWorkArea`/`aclfftSetStride`/`aclfftGetSize`/`aclfftGetSize1d`/`aclfftGetSize2d`/`aclfftGetSize3d` 等接口当前版本暂未实现，调用前请勿依赖。
- Exec 接口的输入输出为 Host 侧指针，调用方需保证缓冲区大小满足对应变换的 shape 要求（参见 [FFT_1D](./FFT_1D.md)、[FFT_2D](./FFT_2D.md) 的约束说明）。
- 调用任何 `aclfft*` 接口前需先完成 AscendCL 初始化（`aclInit`、`aclrtSetDevice`）。
