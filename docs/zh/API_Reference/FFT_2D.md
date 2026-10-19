# FFT_2D

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 910B</term> | √ |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> **说明**：`fft2_d`（二维 C2C）当前仅在 Ascend 910B 上实现，Ascend 950 系列暂不支持。

## 功能说明

- **接口功能**：
  - `aclfftPlan2d`：一步完成 Plan 的创建与二维 FFT 配置初始化。
  - `aclfftExecC2C`：执行复数到复数的二维 FFT（对应算子 `fft2_d`）。

- **计算公式**：\
  二维离散傅里叶变换（2D DFT）将二维信号在两个维度上分别做傅里叶变换。对于尺寸为 N1×N2 的二维信号，其 2D DFT 表达式如下：

  ![公式](./figures/FFT_2D_1.png)

  二维数据按 C 顺序（行主序）存储，即 y 方向（j 索引）变化最快。直接使用矩阵乘完成 2D DFT 时间复杂度太高，因此采用快速傅里叶变换，利用三角函数在复数域的旋转对称性，将序列拆分成子序列，通过蝶形运算以降低计算的复杂度：

  ![公式](./figures/fft_1d_2.png)

## 函数原型

```Cpp
aclfftResult aclfftPlan2d(aclfftHandle* plan,
                          int           batch,
                          int           nx,
                          int           ny,
                          aclfftType    type);
```

```Cpp
aclfftResult aclfftExecC2C(aclfftHandle     plan,
                           aclfftComplex*   idata,
                           aclfftComplex*   odata,
                           int              direction);
```

## aclfftPlan2d

- **功能描述**：分配并创建一个新的 Plan，并按二维 FFT 参数完成初始化。等价于依次调用 `aclfftCreate` 与 `aclfftMakePlan2d`（详见[FFT公共接口](./FFT公共接口.md)）；若初始化失败会自动销毁句柄并将 `*plan` 置空。

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
      <td>指向 Plan 句柄的指针，用于接收创建的句柄。不能为空。</td>
    </tr>
    <tr>
      <td>batch（int）</td>
      <td>输入</td>
      <td>批处理中二维变换的数量。需大于 0。</td>
    </tr>
    <tr>
      <td>nx（int）</td>
      <td>输入</td>
      <td>x 方向（慢索引）的元素个数，即行数。需大于 0。</td>
    </tr>
    <tr>
      <td>ny（int）</td>
      <td>输入</td>
      <td>y 方向（快索引）的元素个数，即列数。需大于 0。</td>
    </tr>
    <tr>
      <td>type（aclfftType）</td>
      <td>输入</td>
      <td>变换类型，二维当前仅支持 <code>ACLFFT_C2C</code>。</td>
    </tr>
  </tbody>
  </table>

> **说明**：注意 `aclfftPlan2d` 的参数顺序为 `(plan, batch, nx, ny, type)`，`batch` 在 `nx`/`ny` 之前。

- **返回值**：返回 `aclfftResult` 状态码，具体参见[FFT公共接口](./FFT公共接口.md#错误码aclfftresult)。

## aclfftExecC2C

- **功能描述**：执行复数到复数的二维 FFT。输入输出均为复数（实虚交错），数据按行主序排布（y 方向索引变化最快）。

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
      <td>已初始化且类型为 <code>ACLFFT_C2C</code> 的二维 Plan 句柄。</td>
    </tr>
    <tr>
      <td>idata（aclfftComplex*）</td>
      <td>输入</td>
      <td>Host 侧输入数据指针，shape 为 (batch, nx, ny)，共 batch*nx*ny 个复数，行主序。</td>
    </tr>
    <tr>
      <td>odata（aclfftComplex*）</td>
      <td>输出</td>
      <td>Host 侧输出数据指针，shape 与 idata 相同。</td>
    </tr>
    <tr>
      <td>direction（int）</td>
      <td>输入</td>
      <td>变换方向：<code>ACLFFT_FORWARD</code>（正向）或 <code>ACLFFT_BACKWARD</code>（逆向）。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：返回 `aclfftResult` 状态码，具体参见[FFT公共接口](./FFT公共接口.md#错误码aclfftresult)。

### 变换缩放规则

- 正向变换不缩放：`Y[kx, ky] = Σ(x=0..nx-1, y=0..ny-1) x[x, y] · exp(-2πi(kx·x/nx + ky·y/ny))`，其中 `0≤kx<nx`、`0≤ky<ny`。
- 反向变换也不缩放：`y[x, y] = Σ(kx=0..nx-1, ky=0..ny-1) Y[kx, ky] · exp(+2πi(kx·x/nx + ky·y/ny))`。
- 因此正向后接反向的结果为原输入乘以 `nx*ny`。与默认归一化的 `torch.fft.ifft2` 对比时，需将参考结果乘以 `nx*ny`。

## 约束说明

> 本节按芯片类型拆分，约束值均依据 ops-fft 源码实际实现。表格仅列该芯片已实现路径。

### 通用约束

- 仅支持 FP32。复数以 `aclfftComplex{float x, y}` 实虚交错存储。
- aclfftPlan2d
  - nx、ny 仅支持 {32, 64, 128} 共 9 种组合（DD 核为固定矩阵乘实现，与质因数分解无关，
    其余尺寸在 aclfftExecC2C_2D 执行时返回 ACLFFT_NOT_SUPPORTED，issue #95）。
  - batch在存储允许范围内应无额外约束。
  - 输入的元素不支持inf、-inf和nan，如果输入中包含这些值， 那么结果为未定义。

### Ascend 910B（arch32）

| 算子 | Kernel（接口） | 触发条件 | fftSize(nx) | fftSize(ny) | batch | 横向/纵向 | 前向/后向 | 输入类型 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| C2C | DD（`aclfftFft2DDd`） | 固定 DD 核，仅看尺寸组合（与质因数分解无关，issue #95） | nx ∈ {32,64,128} | ny ∈ {32,64,128} | 任意 | 2D | 前向+后向 | 复数（aclfftComplex） |

- DD 核支持 9 种 (nx, ny) 组合：{32,64,128}×{32,64,128}。

### Ascend 950（arch35）

不支持二维 FFT。


## 调用示例

示例代码旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

```Cpp
#include <iostream>
#include <vector>
#include "cann_ops_fft.h"
#include "acl/acl.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define ACLFFT_CHECK(err)                                                   \
    do {                                                                    \
        aclfftResult _e = (err);                                            \
        if (_e != ACLFFT_SUCCESS) {                                         \
            std::cout << "aclfft failed: " << aclfftGetErrorString(_e)      \
                      << std::endl;                                         \
            exit(-1);                                                       \
        }                                                                   \
    } while (0)

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，AscendCL 初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 创造 Host 侧数据：batch 个 nx*ny 的二维复数矩阵，行主序
    int batch = 1, fftX = 64, fftY = 64;
    // int batch = 1, fftX = 128, fftY = 32;  // 其它 {32,64,128} 组合
    const int64_t tensorSize = static_cast<int64_t>(batch) * fftX * fftY;
    std::vector<aclfftComplex> inputHostData(tensorSize, {0.0f, 0.0f});
    std::vector<aclfftComplex> outputHostData(tensorSize, {0.0f, 0.0f});
    for (int64_t i = 0; i < tensorSize; i++) {
        inputHostData[i] = {static_cast<float>(i), static_cast<float>(i + 1)};
    }

    // 创建并初始化二维 C2C Plan（参数顺序：plan, batch, nx, ny, type）
    aclfftHandle plan;
    ACLFFT_CHECK(aclfftPlan2d(&plan, batch, fftX, fftY, ACLFFT_C2C));
    ACLFFT_CHECK(aclfftSetStream(plan, stream));

    // 执行正向二维 FFT
    ACLFFT_CHECK(aclfftExecC2C(plan, inputHostData.data(), outputHostData.data(), ACLFFT_FORWARD));

    ACLFFT_CHECK(aclfftDestroy(plan));

    // 打印输出前 8 个复数
    for (int64_t i = 0; i < 8; i++) {
        std::cout << "(" << outputHostData[i].x << ", " << outputHostData[i].y << ")\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
