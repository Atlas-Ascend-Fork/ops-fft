# 算子开发指南

## 目录结构

开发一个算子需要以下文件（以 `rfft1_d` 为参照，实际文件名可自定义）：

```
${op_name}/                              # 算子名的小写下划线形式
├── CMakeLists.txt                       # 算子编译配置（register_operator 注册）
├── ${op_name}.h                         # 算子头文件
├── arch32/                              # Ascend 910B 架构实现
│   ├── ${op_name}_exec_api.cpp          # Host 侧分发入口（Exec 接口实现）
│   └── ${impl}/                         # 具体实现目录（如 dft、r2c_fft 等）
│       ├── ${op_name}_${impl}.cpp       # Host 侧实现（Tiling 计算、内存管理、核函数调用）
│       ├── ${op_name}_${impl}_kernel.h  # Kernel 实现（核函数逻辑）
│       └── ${op_name}_${impl}_tilingdata.h  # Tiling 数据结构（可选，也可内联）
├── arch35/                              # Ascend 950 架构实现
│   ├── ${op_name}_exec_api.cpp          # Host 侧分发入口
│   └── ${impl}/                         # 具体实现目录（如 fast_dft、fft 等）
│       ├── ${op_name}_${impl}.cpp
│       ├── ${op_name}_${impl}_kernel.h
│       └── ${op_name}_${impl}_tilingdata.h
└── tests/                               # 测试用例目录（按架构拆分）
    ├── ${op_name}_test.h                # 测试头文件
    ├── ${op_name}_test_arch32.cpp       # arch32 测试用例
    ├── ${op_name}_test_arch35.cpp       # arch35 测试用例
    └── ${op_name}_data/                 # 测试数据目录（gen_data.py / compare_data.py）
```

**说明**：
- Host 与 Kernel 通常分文件：`*_kernel.h` 放核函数，`*.cpp` 放 Host 侧（Tiling、内存、核函数调用）
- TilingData 可独立为 `_tilingdata.h`，也可定义在 `.cpp` 中
- `arch32/`（910B）与 `arch35/`（950）按 SOC 架构分目录，由 `register_operator` 自动发现
- 测试按架构拆分为 `_test_arch32.cpp` / `_test_arch35.cpp`

---

## 文件说明

### Host + Kernel 实现

**文件**：`<op_name>.cpp`

**作用**：
- **Host 部分**：实现对外接口、Tiling 计算、内存管理、核函数调用
- **Kernel 部分**：实现实际的核函数逻辑

**基本结构**：

```cpp
#include "acl/acl.h"
#include "kernel_operator.h"

#define GM_ADDR uint8_t*

// ========== Tiling 数据结构 ==========
namespace <OpName>Op {
    struct <OpName>TilingData {
        int64_t totalLength;
        int64_t usedCoreNum;
        // ... 其他 Tiling 参数
    };
}

// ========== Host 部分：对外接口 ==========

extern "C" aclError aclfft<OpName>(float* input, float* output,
                                         int64_t size, void* stream)
{
    // 1. 参数检查
    if (input == nullptr || output == nullptr || size <= 0) {
        return ACL_ERROR_INVALID_PARAM;
    }

    // 2. 计算 Tiling 数据（可用函数封装）
    <OpName>Op::<OpName>TilingData tilingData;
    tilingData.totalLength = size;
    tilingData.usedCoreNum = CalculateCoreNum(size);  // 自定义函数
    // ... 其他 Tiling 参数

    // 3. 分配设备内存
    uint8_t *inputDevice, *outputDevice, *tilingDevice;
    aclrtMalloc((void**)&inputDevice, size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outputDevice, size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDevice, sizeof(tilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    // 4. 拷贝数据到设备（先转换为 uint8_t*）
    uint8_t* inputHost = reinterpret_cast<uint8_t*>(input);
    uint8_t* outputHost = reinterpret_cast<uint8_t*>(output);
    aclrtMemcpy(inputDevice, size * sizeof(float), inputHost, size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(tilingData), &tilingData, sizeof(tilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 调用核函数 <<<Block, workspace, stream>>>
    <op_name>_kernel_do(inputDevice, outputDevice, tilingDevice,
                        nullptr, tilingData.usedCoreNum, stream);

    // 6. 同步并拷贝结果回主机
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outputHost, size * sizeof(float), outputDevice, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // 7. 释放设备内存
    aclrtFree(inputDevice);
    aclrtFree(outputDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}

// ========== Kernel 部分：核函数实现 ==========

using namespace AscendC;

extern "C" __global__ __aicore__ void <op_name>(GM_ADDR input, GM_ADDR output,
                                                GM_ADDR tiling)
{
    // Kernel 类型声明
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 初始化
    TPipe pipe;
    // ... 初始化 LocalTensor、GlobalTensor、TQue 等

    // 解析 Tiling 数据
    auto tilingData = (<OpName>Op::<OpName>TilingData*)tiling;

    // 核心计算逻辑
    for (int i = 0; i < tilingData->blockLoopCnt; ++i) {
        // 1. DataCopy: GM -> LocalTensor
        // 2. 计算
        // 3. DataCopy: LocalTensor -> GM
    }
}

// 核函数封装
void <op_name>_kernel_do(GM_ADDR input, GM_ADDR output, GM_ADDR tiling,
                         GM_ADDR workspace, uint32_t numBlocks, void *stream)
{
    <op_name><<<numBlocks, workspace, stream>>>(input, output, tiling);
}
```

**Host 部分关键点**：
1. 定义 TilingData 结构体（或 include 独立的 `_struct.h`）
2. 计算 Tiling 参数
3. 实现对外接口（如 `aclfftRfft1D`、`aclfftIrfft1DFft` 等，前缀统一为 `aclfft`，由 `lib/` 的 `aclfftExec*` 分发层调用）
4. 使用 `<<<numBlocks, workspace, stream>>>` 调用核函数

**Kernel 部分关键点**：
1. 使用 `__global__ __aicore__` 标记核函数
2. 实现 GM ↔ LocalTensor 的数据搬运
3. 实现核心计算逻辑

---

### Tiling 数据结构（可选）

**文件**：`<op_name>_struct.h`

**作用**：定义 Host 传递给 Kernel 的 Tiling 参数

**基本结构**：

```cpp
#ifndef <OP_NAME>_STRUCT_H
#define <OP_NAME>_STRUCT_H

#include <cstdint>

namespace <OpName>Op {

struct <OpName>TilingData {
    int64_t totalLength;
    int64_t usedCoreNum;
    int64_t blockFormer;
    int64_t blockLoopCnt;
    int64_t blockTail;
    // ... 其他 Tiling 参数
};

} // namespace <OpName>Op

#endif
```

**说明**：
- 这是一个简单的 C 结构体
- 只包含基本数据类型（int64_t 等）
- Host 计算参数，Kernel 读取参数
- 也可以直接定义在 `.cpp` 文件中

---

### 编译配置

**文件**：`CMakeLists.txt`

```cmake
register_operator(
    NAME <op_name>     # 自动发现 arch32/arch35 下的源文件
)
```

---

### 测试文件（强烈推荐）

参见 [测试编写指南](test_writing_guide.md)。

**说明**：虽然测试文件不是必需的，但强烈建议为每个算子编写单元测试，以确保算子实现的正确性。

---

## 开发流程

### 步骤 1：创建目录和文件

```bash
mkdir -p src/<op_name>/tests
touch src/<op_name>/<op_name>.cpp
touch src/<op_name>/CMakeLists.txt
```

（可选）独立的 struct 文件：
```bash
touch src/<op_name>/<op_name>_struct.h
```

（可选）测试文件：
```bash
touch src/<op_name>/tests/<op_name>_test.h
touch src/<op_name>/tests/<op_name>_test.cpp
```

### 步骤 2：编写 Host + Kernel 实现

在 `<op_name>.cpp` 中：
1. 定义 TilingData 结构体（或 include 独立的 `_struct.h`）
2. 实现对外接口（如 `aclfftRfft1D`、`aclfftIrfft1DFft` 等，前缀统一为 `aclfft`，由 `lib/` 的 `aclfftExec*` 分发层调用）
3. 计算 Tiling 参数
4. 实现核函数（使用 `__global__ __aicore__`）

### 步骤 3：配置编译

在 `CMakeLists.txt` 中注册算子。

### 步骤 4：编写测试（推荐）

参考 [测试编写指南](test_writing_guide.md)。

### 步骤 5：编译验证

```bash
./build.sh --ops=<op_name> --run
```

---

## 关键概念

### Host vs Kernel

| 层面 | 运行位置 | 职责 |
|------|---------|------|
| **Host** | CPU | 对外接口、Tiling 计算、内存管理、启动 Kernel |
| **Kernel** | NPU AI Core | 实际计算逻辑 |

### Tiling

**目的**：将大任务切分成适合 NPU 执行的小块

**关键参数**：
- `usedCoreNum` - 使用多少个 AI Core
- `blockFormer` - 每次迭代处理多少数据
- `blockLoopCnt` - 每个核迭代多少次

**计算原则**：
- 充分利用 AI Core 并行能力
- 数据不超过 Unified Buffer 容量
- 对齐到 32 字节边界

### <<<>>> 核函数调用

```cpp
<kernel_func><<<numBlocks, workspace, stream>>>(args...);
```

**参数说明**：
- `numBlocks` - 使用多少个 AI Core（Block）
- `workspace` - 共享内存指针，通常设置为 `nullptr`
- `stream` - ACL 执行流

---

## 完整示例

参见 `src/rfft1_d/` 目录（实际文件）：
- `rfft1_d.h` - 算子头文件
- `CMakeLists.txt` - 编译配置（`register_operator(NAME rfft1_d)`）
- `arch32/rfft1_d_exec_api.cpp` - 910B Exec 分发入口
- `arch32/dft/`、`arch32/r2c_fft/` - 910B 实现：`rfft1_d_dft.*` / `rfft1_d_r2c_fft.*`（`.cpp` + `_kernel.h` + `_tilingdata.h`）
- `arch35/rfft1_d_exec_api.cpp` - 950 Exec 分发入口
- `arch35/fast_dft/`、`arch35/fft/` - 950 实现：`rfft1_d_fast_dft.cpp`/`rfft1_d_fast.h`/`rfft1_d_tilingdata.h`、`rfft1_d_fft.*`
- `tests/` - `rfft1_d_test.h`、`rfft1_d_test_arch32.cpp`、`rfft1_d_test_arch35.cpp`、`rfft1_d_data/`（`gen_data.py`、`compare_data.py`）

---

## 相关文档

- [测试编写指南](test_writing_guide.md)
- [build 参数说明](../context/build.md)
