# 算子测试编写指南

## 文件结构

写一个算子的测试需按架构拆分（参照 `src/rfft1_d/tests/`）：

```
src/<op_name>/tests/
├── <op_name>_test.h            # 头文件（必需，声明 run_all_tests）
├── <op_name>_test_arch32.cpp   # Ascend 910B 测试用例（必需）
├── <op_name>_test_arch35.cpp   # Ascend 950 测试用例（必需）
└── <op_name>_data/             # 测试数据目录（gen_data.py / compare_data.py）
```

## 头文件（必需）

**文件**：`<op_name>_test.h`

**内容**：
```cpp
#pragma once
#include "test_common.h"

namespace <OpName>Test {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats);
}
```

**说明**：
- 命名空间：`<OpName>Test`（首字母大写，用于隔离不同算子的测试）
- 函数名：`run_all_tests`（固定名称，测试框架会调用此函数）
- 参数说明：
  - `aclrtStream stream` - ACL 运行流，用于执行算子
  - `TestStats& stats` - 测试统计对象，用于记录测试结果

---

## 实现文件（必需）

**文件**：`<op_name>_test.cpp`

**必须包含的 3 个部分**：

### 测试用例函数（必需，至少1个）

**函数签名**：
```cpp
void test_xxx(aclrtStream stream, OpsFftTest::TestStats& stats);
```

**参数说明**：
- `stream` - 传递给算子调用，用于在 NPU 上执行
- `stats` - 传递给所有测试宏（TEST_ASSERT、TEST_CASE_PASS 等），用于自动统计

**示例**（以一维 C2C FFT 为例，ops-fft 统一用 Plan 模型的 `aclfft*` 接口，输入输出为 Host 指针）：

```cpp
void test_xxx(aclrtStream stream, OpsFftTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_xxx");

    // 准备数据（复数，实虚交错）：N=2，input = {1+2j, 3+4j}
    aclfftComplex input[]    = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    aclfftComplex expected[] = {{4.0f, 6.0f}, {-2.0f, -2.0f}}; // 正向 DFT 结果
    aclfftComplex output[2];

    // 调用算子（Plan 模型：创建 Plan → Exec → 销毁）
    aclfftHandle plan;
    aclfftPlan1d(&plan, 2, ACLFFT_C2C, 1, ACLFFT_HORIZONTAL);
    aclfftResult result = aclfftExecC2C(plan, input, output, ACLFFT_FORWARD);
    aclfftDestroy(plan);

    // 验证结果
    TEST_ASSERT(stats, result == ACLFFT_SUCCESS, "failed");
    TEST_ASSERT_ARRAY_NEAR(stats, (float*)output, (float*)expected, 4, 1e-6f, "mismatch");

    TEST_CASE_PASS(stats, "test_xxx");
}
```

### run_all_tests 实现（必需）

```cpp
namespace <OpName>Test {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        test_xxx(stream, stats);
        // 可以添加更多测试函数
    }
}
```

### 自动注册（必需）

```cpp
REGISTER_OP_TEST(<OpName>)
```

---

## 完整示例

**my_op_test.h**
```cpp
#pragma once
#include "test_common.h"

namespace MyOpTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats);
}
```

**my_op_test.cpp**
```cpp
#include "cann_ops_fft.h"
#include "my_op_test.h"

void test_basic(aclrtStream stream, OpsFftTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_basic");

    // 复数输入：N=2，input = {1+2j, 3+4j}
    aclfftComplex input[]    = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    aclfftComplex expected[] = {{4.0f, 6.0f}, {-2.0f, -2.0f}}; // 正向 DFT 结果
    aclfftComplex y[2];

    aclfftHandle plan;
    aclfftPlan1d(&plan, 2, ACLFFT_C2C, 1, ACLFFT_HORIZONTAL);
    aclfftResult result = aclfftExecC2C(plan, input, y, ACLFFT_FORWARD);
    aclfftDestroy(plan);

    TEST_ASSERT(stats, result == ACLFFT_SUCCESS, "failed");
    TEST_ASSERT_ARRAY_NEAR(stats, (float*)y, (float*)expected, 4, 1e-6f, "mismatch");

    TEST_CASE_PASS(stats, "test_basic");
}

namespace MyOpTest {
    void run_all_tests(aclrtStream stream, OpsFftTest::TestStats& stats) {
        test_basic(stream, stats);
    }
}

REGISTER_OP_TEST(MyOp)
```

---

## 可用的宏

### 用例控制
```cpp
TEST_CASE_BEGIN("test_name")    // 标记开始
TEST_CASE_PASS(stats, "test_name")  // 标记通过
```

### 断言
```cpp
TEST_ASSERT(stats, condition, "error")              // 条件断言
TEST_ASSERT_ARRAY_EQ(stats, actual, expected, size, "error")   // 精确比较
TEST_ASSERT_ARRAY_NEAR(stats, actual, expected, size, tol, "error")  // 容差比较
```

---

## 编译运行

```bash
./build.sh --ops=<op_name> --run
```

---

## 参考示例

完整示例（数据驱动 + Plan 模型 `aclfftExec*` 接口，按架构拆分）：

- `src/rfft1_d/tests/rfft1_d_test.h` - 测试头文件（`Rfft1DApiIntegrationTest::run_all_tests` 声明、用例结构、`aclfftPlan1d`/`aclfftExecR2C`/`aclfftDestroy` 调用）
- `src/rfft1_d/tests/rfft1_d_test_arch32.cpp` - 910B 测试用例
- `src/rfft1_d/tests/rfft1_d_test_arch35.cpp` - 950 测试用例
- `src/rfft1_d/tests/rfft1_d_data/` - 测试数据（`gen_data.py` 生成、`compare_data.py` 比对）

> **说明**：ops-fft 的测试统一通过 `aclfftPlan1d`/`aclfftExecC2C|R2C|C2R`/`aclfftDestroy` 的 Plan 模型调用算子（输入输出为 Host 指针），而非直接调用 kernel 入口；下文示例为简化说明用的最小写法。
