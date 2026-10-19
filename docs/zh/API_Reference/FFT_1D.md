# FFT_1D

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Ascend 910B</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> **说明**：不同算子（C2C/R2C/C2R）在各芯片上的 `fftSize` 支持范围不同，详见[约束说明](#约束说明)。

## 功能说明

- **接口功能**：
  - `aclfftPlan1d`：一步完成 Plan 的创建与一维 FFT 配置初始化。
  - `aclfftExecC2C`：执行复数到复数的一维 FFT（对应算子 `fft1_d`）。
  - `aclfftExecR2C`：执行实数到复数的一维 FFT（对应算子 `rfft1_d`）。
  - `aclfftExecC2R`：执行复数到实数的一维逆 FFT（对应算子 `irfft1_d`）。

- **计算公式**：\
  傅里叶变换（Fourier transform）是一种线性积分变换，用于信号在时域和频域之间的变换。对于给定长度为 N 的信号，其离散形式 DFT（Discrete Fourier Transform）表达式如下：

  ![公式](./figures/fft_1d_1.png)

  将系数矩阵（N*N）和时域信号（N*1）看作两个 Tensor，在 NPU 上直接使用矩阵乘可完成 DFT，但时间复杂度太高，因此需要快速傅里叶变换。其基本原理是利用三角函数在复数域的旋转对称性，将序列拆分成子序列，通过蝶形运算以降低计算的复杂度：

  ![公式](./figures/fft_1d_2.png)

  - 正向变换（`ACLFFT_FORWARD`）：$y[k]=\sum_{n=0}^{N-1}x[n]\cdot e^{-j2\pi kn/N}$，无缩放。
  - 逆向变换（`ACLFFT_BACKWARD`）：$x[n]=\sum_{k=0}^{N-1}y[k]\cdot e^{j2\pi kn/N}$，结果乘以 N（未归一化）。

## 函数原型

```Cpp
aclfftResult aclfftPlan1d(aclfftHandle* plan,
                          int           nx,
                          aclfftType    type,
                          int           batch,
                          int           dimType);
```

```Cpp
aclfftResult aclfftExecC2C(aclfftHandle     plan,
                           aclfftComplex*   idata,
                           aclfftComplex*   odata,
                           int              direction);
```

```Cpp
aclfftResult aclfftExecR2C(aclfftHandle   plan,
                           aclfftReal*    idata,
                           aclfftComplex* odata);
```

```Cpp
aclfftResult aclfftExecC2R(aclfftHandle     plan,
                           aclfftComplex*   idata,
                           aclfftReal*      odata);
```

## aclfftPlan1d

- **功能描述**：分配并创建一个新的 Plan，并按一维 FFT 参数完成初始化。等价于依次调用 `aclfftCreate` 与 `aclfftMakePlan1d`（详见[FFT公共接口](./FFT公共接口.md)）；若初始化失败会自动销毁句柄并将 `*plan` 置空。

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
      <td>nx（int）</td>
      <td>输入</td>
      <td>一维 FFT 信号长度，对应公式中的 N。需大于 0。</td>
    </tr>
    <tr>
      <td>type（aclfftType）</td>
      <td>输入</td>
      <td>变换类型：<ul><li><code>ACLFFT_C2C</code>：复数到复数。</li><li><code>ACLFFT_R2C</code>：实数到复数。</li><li><code>ACLFFT_C2R</code>：复数到实数。</li></ul></td>
    </tr>
    <tr>
      <td>batch（int）</td>
      <td>输入</td>
      <td>批处理中一维变换的数量。需大于 0。</td>
    </tr>
    <tr>
      <td>dimType（int）</td>
      <td>输入</td>
      <td>变换维度方向：<ul><li><code>ACLFFT_HORIZONTAL</code>：横向 FFT，对每一行做变换，数据按行连续排布。</li><li><code>ACLFFT_VERTICAL</code>：纵向 FFT，对每一列做变换，列间以 batch 为步长排布。</li></ul></td>
    </tr>
  </tbody>
  </table>

- **返回值**：返回 `aclfftResult` 状态码，具体参见[FFT公共接口](./FFT公共接口.md#错误码aclfftresult)。

## aclfftExecC2C

- **功能描述**：执行复数到复数的一维 FFT。输入输出均为复数（实虚交错）。

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
      <td>已初始化且类型为 <code>ACLFFT_C2C</code> 的 Plan 句柄。</td>
    </tr>
    <tr>
      <td>idata（aclfftComplex*）</td>
      <td>输入</td>
      <td>Host 侧输入数据指针。<ul><li>横向：shape 为 (batch, nx)，共 batch*nx 个复数。</li><li>纵向：shape 为 (nx, batch)，按列主序排布，共 nx*batch 个复数。</li></ul></td>
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

## aclfftExecR2C

- **功能描述**：执行实数到复数的一维 FFT。利用实序列频谱的共轭对称性，仅输出前 `nx/2+1` 个频点。

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
      <td>已初始化且类型为 <code>ACLFFT_R2C</code> 的 Plan 句柄。</td>
    </tr>
    <tr>
      <td>idata（aclfftReal*）</td>
      <td>输入</td>
      <td>Host 侧实数输入指针，shape 为 (batch, nx)，共 batch*nx 个实数。</td>
    </tr>
    <tr>
      <td>odata（aclfftComplex*）</td>
      <td>输出</td>
      <td>Host 侧复数输出指针，shape 为 (batch, floor(nx/2)+1)，共 batch*(floor(nx/2)+1) 个复数；独立输出缓冲区至少容纳上述复数数量。</td>
    </tr>
  </tbody>
  </table>

- **缓冲区容量**：当 `idata` 与 `odata` 指向同一块内存时，需为压缩复数输出预留容量，至少为 `batch * 2 * (floor(nx/2)+1)` 个实数（float）。因此偶数 `nx` 时需 `batch*(nx+2)` 个 float，奇数 `nx` 时需 `batch*(nx+1)` 个 float；仅按 `batch*nx` 个 float 分配不足。分离缓冲区时，输入至少为 `batch*nx` 个 float，输出至少为 `batch*(floor(nx/2)+1)` 个复数。

- **返回值**：返回 `aclfftResult` 状态码，具体参见[FFT公共接口](./FFT公共接口.md#错误码aclfftresult)。

## aclfftExecC2R

- **功能描述**：执行复数到实数的一维逆 FFT，是 `aclfftExecR2C` 的逆过程。输入为 `nx/2+1` 个共轭对称频点，输出为 `nx` 个实数。

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
      <td>已初始化且类型为 <code>ACLFFT_C2R</code> 的 Plan 句柄。</td>
    </tr>
    <tr>
      <td>idata（aclfftComplex*）</td>
      <td>输入</td>
      <td>Host 侧复数输入指针，shape 为 (batch, nx/2+1)，共 batch*(nx/2+1) 个复数。</td>
    </tr>
    <tr>
      <td>odata（aclfftReal*）</td>
      <td>输出</td>
      <td>Host 侧实数输出指针，shape 为 (batch, nx)，共 batch*nx 个实数。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：返回 `aclfftResult` 状态码，具体参见[FFT公共接口](./FFT公共接口.md#错误码aclfftresult)。

## 约束说明

> 本节按芯片类型拆分，约束值均依据 ops-fft 源码实际实现。表格仅列该芯片已实现路径，每行标注触发条件、fftSize/batch 约束、横向·纵向、前向·后向、输入类型。

### 通用约束

- 仅支持 FP32。复数以 `aclfftComplex{float x, y}` 实虚交错存储，实数以 `aclfftReal`（float）存储。
- aclfftPlan1d
  - 对横向FFT：
      - fftSize需保证不超过$2^{27}$且分解质因数后不包含超过47的质因子（与分芯片表格及 RADIX_MIX 实现一致）。
      - batchSize在存储允许范围内应无额外约束。
      - 输入的元素个数理论支持[1，$2^{30}$]。
      - 当前功能实现所限，横向FFT输入长度（fftSize）大于等于32768且为2的幂的时候，会修改输入数据，需提前做好备份。
  - 对纵向FFT：
    - 纵向FFT当前仅在 Ascend 910B 上支持 C2C（Stride kernel）；其余变换（R2C/C2R）及 Ascend 950 上的纵向 plan 在执行时返回 ACLFFT_NOT_IMPLEMENTED。
    - fftSize需保证是2的幂且大于等于256、小于等于262144（2^18，与下方芯片表格及 Stride 实现一致，issue #92）。
    - batchSize需保证大于0；在存储允许范围内无其他额外约束（与实现校验一致：plan 创建层仅校验 batch>0，Stride 内核无 128 整数倍限制，issue #172）。
    - 输入的元素个数理论支持[1，$2^{30}$]。
    - 输入的元素不支持inf、-inf和nan，如果输入中包含这些值, 那么结果为未定义。


### Ascend 910B（arch32）

| 算子 | Kernel（接口） | 触发条件 | fftSize(nx) | batch | 横向/纵向 | 前向/后向 | 输入类型 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| C2C | FFTB（`aclfftFft1DB`） | radix==2 | 2 的幂，256 < nx < 32768 | 任意 | 横向 | 前向+后向 | 复数（aclfftComplex） |
| C2C | FFTN（`aclfftFft1DN`） | radix==2 | 2 的幂，32768 ≤ nx ≤ $2^{27}$ | 任意 | 横向 | 前向+后向 | 复数（aclfftComplex） |
| C2C | Stride（`aclfftFft1DStride`） | stride>1 | 2 的幂，256 ≤ nx ≤ 262144 | 任意 | **纵向** | 前向+后向 | 复数（aclfftComplex） |
| C2C | DFT（`aclfftFft1DDft`） | 表内/默认/兜底 | 任意正整数，n≤256（受 batch 影响，见选核说明） | 任意 | 横向 | 前向+后向 | 复数（aclfftComplex） |
| C2C | Mix（`aclfftFft1DMix`） | radix==mix | 256 < nx ≤ $2^{27}$，质因子 ⊆ {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47} | 任意 | 横向 | 前向+后向 | 复数（aclfftComplex） |
| R2C | DFT（`aclfftRfft1DDft`） | n ≤ 1024 | 任意正整数，1 ≤ nx ≤ 1024 | 任意 | 横向 | **仅前向** | 实数（aclfftReal） |
| R2C | R2C FFT（`aclfftRfft1DR2CFft`） | radix==mix | 1024 < nx ≤ $2^{27}$，质因子 ⊆ {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47} | 任意 | 横向 | **仅前向** | 实数（aclfftReal） |
| C2R | DFT（`aclfftIrfft1DDft`） | n ≤ 1024 | 任意正整数，1 ≤ nx ≤ 1024 | 任意 | 横向 | **仅后向** | 复数（aclfftComplex） |
| C2R | C2R FFT（`aclfftIrfft1DC2RFft`） | radix==mix | 1024 < nx ≤ $2^{27}$，质因子 ⊆ {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47} | 任意 | 横向 | **仅后向** | 复数（aclfftComplex） |

### Ascend 950（arch35）

> 注：Ascend 950 暂不支持纵向（VERTICAL）1D FFT，纵向 plan 执行时返回 `ACLFFT_NOT_IMPLEMENTED`。

| 算子 | Kernel（接口） | 触发条件 | fftSize(nx) | batch | 横向/纵向 | 前向/后向 | 输入类型 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| C2C | Stockham radix-2（`aclfftFft1DC2C`，`src/fft1_d/arch35/fft/`） | radix==2（n 为 2 的幂） | 1 < nx ≤ $2^{27}$ | 任意 | 横向 | 前向+后向 | 复数（aclfftComplex） |
| C2C | Stockham mixed-radix（`aclfftFft1DC2CMix`，`src/fft1_d/arch35/mix/`） | radix==mix（n 含 3/5/7/11/13/17/19 等非 2 质因子） | 1 < nx ≤ $2^{27}$，质因子 ⊆ {2,3,5,7,11,13,17,19} | 任意 | 横向 | 前向+后向 | 复数（aclfftComplex） |
| R2C | FFT（`aclfftRfft1DFft`） | n > 1024 且 radix==mix（**优先级最高**） | 1024 < nx ≤ $2^{27}$，质因子 ⊆ {2,3,5,7} | 任意 | 横向 | **仅前向** | 实数（aclfftReal） |
| R2C | FastDFT（`aclfftRfft1D`） | 兜底（**优先级最低**） | 任意正整数，1 ≤ nx ≤ 4096 | 任意 | 横向 | **仅前向** | 实数（aclfftReal） |
| C2R | C2R FFT（`aclfftIrfft1DFft`） | radix==mix | 1024 < nx ≤ $2^{27}$，质因子 ⊆ {2,3,5,7} | 任意 | 横向 | **仅后向** | 复数（aclfftComplex） |

### 补充说明

**910B C2C 横向 kernel 选择**

910B 横向 C2C 实际使用的 kernel 受 batch 和 nx 共同影响：

| batch 范围 | DFT 支持的 nx | FFTB 支持的 nx | FFTN 支持的 nx |
| :--- | :--- | :--- | :--- |
| [8, 96) | {256, 512, 1024} | {2048, 4096, 8192, 16384} | {32768 ~ $2^{22}$} |
| [96, 1024) | {256, 512} | {1024 ~ 16384} | — |
| [1024, 65537) | {256} | {512 ~ 16384} | — |

batch 不在上述范围或 nx 未命中时按以下规则兜底：`nx ≤ 256 → DFT`；`2 的幂且 < 32768 → FFTB`；`2 的幂且 ≥ 32768 → FFTN`；`含非 2 质因子（radix==mix）且 nx > 256 → Mix（aclfftFft1DMix）`；其余不支持。因此 DFT 在小 batch 时可处理更大的 nx（如 batch∈[8,96)

**910B R2C / C2R 选核**

`n ≤ 1024 → DFT`；`n > 1024 且 radix==mix → R2C FFT（aclfftRfft1DR2CFft）/ C2R FFT（aclfftIrfft1DC2RFft）`；其余不支持。质因子需 ⊆ {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47}。

**950 C2C 选核（radix-2 vs mixed-radix）**

由 `ChooseRadix` 按 nx 质因子选择：`2 的幂 → aclfftFft1DC2C`；`含 {3,5,7,11,13,17,19} 等非 2 质因子 → aclfftFft1DC2CMix`。**质因子上限为 19**——含 23 及以上更大质因子时不支持（返回 `ACLFFT_NOT_IMPLEMENTED`）。**nx 下界为 2**——选核与执行均要求 nx > 1，nx=1 的 plan 可创建成功但执行时落入兜底分支返回 `ACLFFT_NOT_IMPLEMENTED`（issue #194）。

**950 R2C 选核（FFT 优先，FastDFT 兜底）**

约束规则：`n > 1024 且 radix==mix → FFT（aclfftRfft1DFft，质因子 ⊆ {2,3,5,7}）`，**优先级最高**——重叠区间（1024 < n ≤ 4096 且 radix==mix）也走 FFT。`n ≤ 4096 且不满足上述 FFT 条件 → FastDFT（aclfftRfft1D，任意正整数）`，**优先级最低（兜底）**；其余不支持。

> **注**：950 R2C 的 FastDFT 兜底范围（≤4096）大于 910B R2C 的 DFT（≤1024）。重叠区间优先走 FFT，仅当 radix!=mix 或 n ≤ 1024 时才回退到 FastDFT。

**950 C2R 选核**

`n > 1024 且 radix==mix → C2R FFT（aclfftIrfft1DFft，质因子 ⊆ {2,3,5,7}）`。**`n ≤ 1024` 不支持**——950 C2R 无 DFT 兜底（与 910B 不同，910B 的 `n≤1024` 走 DFT）。

## 调用示例

示例代码旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

### C2C_1D

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

    // 创造 Host 侧数据：batch 路、每路 Nfft 点复数
    int batch = 32, Nfft = 128;
    // int batch = 32, Nfft = 32768;  // 大 N（2 的幂）走 FFTN 路径
    const int64_t tensorSize = static_cast<int64_t>(batch) * Nfft;
    std::vector<aclfftComplex> inputHostData(tensorSize, {0.0f, 0.0f});
    std::vector<aclfftComplex> outputHostData(tensorSize, {0.0f, 0.0f});
    for (int64_t i = 0; i < tensorSize; i++) {
        inputHostData[i] = {static_cast<float>(i), static_cast<float>(i + 1)};
    }

    // 创建并初始化一维 C2C Plan
    aclfftHandle plan;
    ACLFFT_CHECK(aclfftPlan1d(&plan, Nfft, ACLFFT_C2C, batch, ACLFFT_HORIZONTAL));
    ACLFFT_CHECK(aclfftSetStream(plan, stream));

    // 执行正向 FFT（输入输出为 Host 指针，接口内部完成 H2D/启动/D2H/同步）
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

### R2C_1D

以下示例复用 C2C_1D 中的 `Init`、`ACLFFT_CHECK`、`CHECK_RET` 等公共代码，仅给出 `main` 部分。

```Cpp
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int batch = 32, Nfft = 256;
    // int batch = 32, Nfft = 4096;  // 950 FFT 路径（n>1024 且 radix==mix，优先级最高）
    const int64_t inSize = static_cast<int64_t>(batch) * Nfft;
    const int64_t outSize = static_cast<int64_t>(batch) * (Nfft / 2 + 1);

    std::vector<aclfftReal> inputHostData(inSize, 0.0f);          // 实数输入
    std::vector<aclfftComplex> outputHostData(outSize, {0.0f, 0.0f});  // 复数输出
    for (int64_t i = 0; i < inSize; i++) {
        inputHostData[i] = static_cast<float>(i);
    }

    aclfftHandle plan;
    ACLFFT_CHECK(aclfftPlan1d(&plan, Nfft, ACLFFT_R2C, batch, ACLFFT_HORIZONTAL));
    ACLFFT_CHECK(aclfftSetStream(plan, stream));

    ACLFFT_CHECK(aclfftExecR2C(plan, inputHostData.data(), outputHostData.data()));

    ACLFFT_CHECK(aclfftDestroy(plan));

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

### C2R_1D（逆 RFFT）

以下示例复用 C2C_1D 中的 `Init`、`ACLFFT_CHECK`、`CHECK_RET` 等公共代码，仅给出 `main` 部分。

```Cpp
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int batch = 32, Nfft = 256;
    const int64_t inSize = static_cast<int64_t>(batch) * (Nfft / 2 + 1);
    const int64_t outSize = static_cast<int64_t>(batch) * Nfft;

    std::vector<aclfftComplex> inputHostData(inSize, {0.0f, 0.0f});  // 复数输入（共轭对称频谱）
    std::vector<aclfftReal> outputHostData(outSize, 0.0f);            // 实数输出
    for (int64_t i = 0; i < inSize; i++) {
        inputHostData[i] = {static_cast<float>(i), static_cast<float>(i + 1)};
    }

    aclfftHandle plan;
    ACLFFT_CHECK(aclfftPlan1d(&plan, Nfft, ACLFFT_C2R, batch, ACLFFT_HORIZONTAL));
    ACLFFT_CHECK(aclfftSetStream(plan, stream));

    ACLFFT_CHECK(aclfftExecC2R(plan, inputHostData.data(), outputHostData.data()));

    ACLFFT_CHECK(aclfftDestroy(plan));

    for (int64_t i = 0; i < 8; i++) {
        std::cout << outputHostData[i] << "\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

> **提示**：若需正向 R2C 后再逆向 C2R 还原信号，注意逆向结果为未归一化逆变换（乘以 N），需对结果除以 N 才能得到原信号。
