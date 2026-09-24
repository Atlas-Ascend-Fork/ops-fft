# 项目目录

> 本章罗列的部分目录是可选的，请以实际交付件为准。尤其**单算子目录**，不同场景下交付件有差异。

项目全量目录层级介绍如下：

```
├── cmake                                               # 项目工程编译目录
│   ├── func.cmake                                      # 公共函数
│   ├── init_env.cmake                                  # 环境初始化
│   ├── makeself_built_in.cmake                         # makeself内置打包配置
│   ├── package.cmake                                   # 打包配置
│   ├── variables.cmake                                 # CMake变量定义
│   └── third_party                                     # 第三方依赖配置
│       └── makeself-fetch.cmake                        # makeself依赖拉取配置
├── docs                                                # 项目相关文档目录
│   ├── README.md                                       # 文档目录索引
│   ├── implementation.md                               # 实现说明文档
│   └── zh                                              # 中文文档目录
│       ├── op_list.md                                  # 算子列表
│       ├── context                                     # 公共文档目录
│       │   ├── build.md                                # 构建说明
│       │   ├── dir_structure.md                        # 目录结构说明
│       │   └── quick_install.md                        # 快速安装指南
│       ├── invocation                                  # 算子调用文档目录
│       │   └── quick_op_invocation.md                  # 算子快速调用指南
│       ├── develop                                     # 算子开发文档目录
│       │   ├── operator_development_guide.md           # 算子开发指南
│       │   └── test_writing_guide.md                   # 测试用例编写指南
│       ├── API_Reference                              # FFT 算子接口参考文档目录
│       │   ├── FFT_1D.md                               # 一维 FFT 算子文档
│       │   ├── FFT_2D.md                               # 二维 FFT 算子文档
│       │   ├── FFT公共接口.md                         # FFT 公共接口文档
│       │   ├── README.md                               # 接口参考索引
│       │   └── figures                                 # 接口文档配图
│       │       ├── FFT_2D_1.png
│       │       ├── fft_1d_1.png
│       │       └── fft_1d_2.png
│       └── debug                                       # 调试调优文档目录
│           └── op_debug_prof.md                        # 算子调试调优指南
├── include                                             # 头文件目录
│   └── cann_ops_fft.h                                  # API头文件
├── lib                                                 # FFT库实现目录
│   ├── CMakeLists.txt                                  # 库编译配置
│   ├── fft_error.h                                     # 错误码定义
│   ├── fft_exec_api.cpp                                # FFT执行接口
│   ├── fft_handle_impl.h                               # Handle实现
│   ├── fft_plan_api.cpp                                # Plan创建接口
│   ├── fft_plan_destroy_api.cpp                        # Plan销毁接口
│   ├── fft_plan_init_api.cpp                           # Plan初始化接口
│   ├── fft_stream_api.cpp                              # 流管理接口
│   └── fft_utils_api.cpp                               # 工具函数接口
├── scripts                                             # 脚本目录，包含自定义算子、Kernel构建相关配置文件
│   ├── check_build_dependencies.py                     # 构建依赖检查脚本
│   ├── generate_version_info.py                        # 版本信息生成脚本
│   └── package                                         # 打包相关脚本
│       ├── package.py                                  # 打包主脚本
│       ├── common                                      # 公共打包工具
│       │   ├── cfg                                     # 打包配置
│       │   ├── py                                      # 打包 Python 工具
│       │   └── sh                                      # 打包 Shell 脚本
│       ├── latest_manager                              # 版本管理脚本
│       │   └── scripts
│       ├── module                                      # 打包模块配置
│       │   └── ascend                                  # Ascend打包模块
│       │       ├── EngineeringCommon.xml               # 通用工程配置
│       │       ├── EngineeringFiles.xml                # 工程文件配置
│       │       ├── OpsFft.xml                          # FFT算子打包配置
│       │       └── OpsFftInc.xml                       # FFT算子头文件打包配置
│       └── ops_fft                                     # ops_fft打包配置
│           ├── ops_fft.xml                             # 打包配置文件
│           └── scripts                                 # 安装/卸载脚本
│               ├── install.sh                          # 安装脚本
│               ├── uninstall.sh                        # 卸载脚本
│               └── ...                                 # 其他脚本
├── src                                                 # 源码目录
│   ├── CMakeLists.txt                                  # 算子编译入口
│   ├── common                                          # 算子公共代码
│   │   ├── host                                        # Host 侧公共代码
│   │   │   ├── fft_common_core.h                       # Host 侧公共头文件
│   │   │   ├── fft_exec_helper.h                       # Host 侧执行辅助
│   │   │   └── test_helper.py                          # 测试辅助脚本
│   │   ├── kernel                                      # Kernel 侧公共实现
│   │   │   ├── base                                    # 基础设施（common/func/iterators 等）
│   │   │   │   └── iterators                           # 各级存储迭代器
│   │   │   └── fft                                     # FFT Kernel 公共实现
│   │   └── tiling                                      # Tiling 数据公共定义
│   ├── fft1_d                                          # fft1_d 算子目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── fft1_d.h                                    # 算子头文件
│   │   ├── arch32                                      # Ascend910B 架构实现
│   │   │   ├── dft                                     # DFT 实现
│   │   │   ├── fft_b                                   # FFT-B 实现
│   │   │   ├── fft_mix                                 # Mix FFT 实现
│   │   │   ├── fft_n                                   # FFT-N 实现
│   │   │   └── fft_stride                              # Stride FFT 实现
│   │   ├── arch35                                      # Ascend950 架构实现
│   │   │   ├── fft                                     # C2C 实现
│   │   │   └── mix                                     # C2C Mix 实现
│   │   └── tests                                       # 算子测试用例目录
│   │       ├── fft1_d_test.h
│   │       ├── fft1_d_test_arch32.cpp
│   │       ├── fft1_d_test_arch35.cpp
│   │       └── fft1_d_data                             # 测试数据目录
│   ├── fft2_d                                          # fft2_d 算子目录
│   │   ├── CMakeLists.txt
│   │   ├── fft2_d.h
│   │   ├── arch32                                      # Ascend910B 架构实现
│   │   │   └── dd                                      # DD 实现
│   │   └── tests
│   │       ├── fft2_d_test.h
│   │       ├── fft2_d_test_arch32.cpp
│   │       └── fft2_d_data
│   ├── irfft1_d                                        # irfft1_d 算子目录
│   │   ├── CMakeLists.txt
│   │   ├── irfft1_d.h
│   │   ├── arch32                                      # Ascend910B 架构实现
│   │   │   ├── c2r_fft                                 # C2R FFT 实现
│   │   │   └── dft                                     # DFT 实现
│   │   ├── arch35                                      # Ascend950 架构实现
│   │   │   └── fft                                     # FFT 实现
│   │   └── tests
│   │       ├── irfft1_d_test.h
│   │       ├── irfft1_d_test_arch32.cpp
│   │       ├── irfft1_d_test_arch35.cpp
│   │       └── irfft1_d_data
│   └── rfft1_d                                         # rfft1_d 算子目录
│       ├── CMakeLists.txt
│       ├── rfft1_d.h
│       ├── arch32                                      # Ascend910B 架构实现
│       │   ├── dft                                     # DFT 实现
│       │   └── r2c_fft                                 # R2C FFT 实现
│       ├── arch35                                      # Ascend950 架构实现
│       │   ├── fast_dft                               # Fast DFT 实现
│       │   └── fft                                     # FFT 实现
│       └── tests
│           ├── rfft1_d_test.h
│           ├── rfft1_d_test_arch32.cpp
│           ├── rfft1_d_test_arch35.cpp
│           └── rfft1_d_data
├── tests                                               # 项目级测试目录
│   ├── CMakeLists.txt                                  # 测试编译配置
│   ├── all_tests.cpp.in                                # 测试入口模板
│   ├── test_common.cpp                                 # 公共测试代码
│   └── test_common.h                                   # 公共测试头文件
├── CMakeLists.txt                                      # 项目工程cmakelist入口
├── CONTRIBUTING.md                                     # 项目贡献指南文件
├── README.md                                           # 项目工程总介绍文档
├── QUICKSTART.md                                       # 快速入门指南
├── SECURITY.md                                         # 项目安全声明文件
├── OAT.xml                                             # 开源审计工具配置
├── build.sh                                            # 项目工程编译脚本
├── version.cmake                                       # 版本号CMake配置
└── version.info                                        # 项目版本信息
```

## 目录说明

### 核心目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `src/` | 算子源码目录，包含所有算子的实现代码 |
| `src/fft1_d/` | fft1_d 算子目录，实现一维复数 FFT 运算 |
| `src/fft2_d/` | fft2_d 算子目录，实现二维复数 FFT 运算 |
| `src/irfft1_d/` | irfft1_d 算子目录，实现一维复数到实数 IFFT 运算 |
| `src/rfft1_d/` | rfft1_d 算子目录，实现一维实数 FFT 运算 |
| `src/common/` | 算子公共代码，包含通用工具函数、迭代器、内存管理等 |
| `include/` | API头文件目录 |
| `lib/` | FFT库实现目录，提供Plan/Exec/Stream等接口 |
| `cmake/` | CMake编译配置文件 |

### 文档目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `docs/` | 项目文档目录 |
| `docs/zh/` | 中文文档目录 |
| `docs/zh/context/` | 公共文档，如环境部署、目录介绍、快速安装等 |
| `docs/zh/invocation/` | 算子调用相关文档 |
| `docs/zh/develop/` | 算子开发相关文档 |
| `docs/zh/API_Reference/` | FFT 算子设计文档 |
| `docs/zh/debug/` | 调试调优相关文档 |

### 构建相关

| 文件 | 说明 |
| :--- | :--- |
| `build.sh` | 项目编译脚本，支持多种编译选项 |
| `CMakeLists.txt` | CMake配置文件 |
| `version.cmake` | 版本号CMake配置 |
| `version.info` | 版本信息文件 |
| `OAT.xml` | 开源审计工具配置 |

## 算子目录结构

每个算子目录（如`src/rfft1_d/`）的典型结构如下：

```
${op_name}/                              # 算子名的小写下划线形式
├── CMakeLists.txt                       # 算子编译配置文件
├── ${op_name}.h                         # 算子接口头文件
├── arch32/                              # Ascend910B 架构实现
│   └── ${impl}/                         # 具体实现目录（如 dft, fft_b 等）
│       ├── ${op_name}_${impl}.cpp       # Host 侧实现
│       └── ${op_name}_${impl}_kernel.h  # Kernel 实现
├── arch35/                              # Ascend950 架构实现
│   └── ${impl}/                         # 具体实现目录
│       ├── ${op_name}_${impl}.cpp       # Host 侧实现
│       └── ${op_name}_${impl}_kernel.h  # Kernel 实现
└── tests/                               # 测试用例目录
    ├── ${op_name}_test.h                # 测试头文件
    ├── ${op_name}_test_arch32.cpp       # arch32 测试用例
    ├── ${op_name}_test_arch35.cpp       # arch35 测试用例
    └── ${op_name}_data/                 # 测试数据目录
        ├── gen_data.py                  # 测试数据生成脚本
        └── compare_data.py              # 测试数据比对脚本
```

> **说明**：不同算子的交付件可能有差异，请以实际目录为准。
