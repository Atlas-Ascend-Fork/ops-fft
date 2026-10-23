# ops-fft

## 🔥 最新动态
- [2026/03] 实现完整的 Plan 管理机制，支持 Plan 创建、执行、销毁全生命周期
- [2026/03] 新增 `rfft1_d` 算子，支持实数到复数的一维 FFT
- [2026/03] 建立完整的测试框架，支持单元测试、自动化测试统计
- [2026/03] 实现标准化的打包流程，生成 .run 安装包，支持 install/uninstall/upgrade 完整生命周期管理

## 🚀 概述

ops-fft 是 [CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供 FFT 类计算的基础算子库，采用模块化设计，支持灵活的算子开发和管理。

### 主要特性

- ✅ **双层架构** - API 层提供易用接口，算子层实现高性能计算
- ✅ **模块化设计** - 支持动态添加算子模块，每个算子独立开发和测试
- ✅ **标准 CMake 构建** - 跨平台编译支持，统一的构建流程
- ✅ **完整测试体系** - 基于自定义测试框架，支持自动化测试
- ✅ **便捷打包** - 一键生成 .run 安装包，支持完整生命周期管理

## 📝 版本配套

当前仓库已验证通过的 CANN Toolkit 如下：

| CANN 版本 | 发布时间 | 分支 |
| --- | --- | --- |
| [CANN 9.0.0-beta.2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) | `2026/03/30` | master |

请根据实际 CPU 架构，从上述链接目录中自行选择对应的 `.run` 安装包。

toolkit 安装包文件名格式如下：

- `Ascend-cann-toolkit_${cann_version}_linux-aarch64.run`
- `Ascend-cann-toolkit_${cann_version}_linux-x86_64.run`

1. **安装社区版 CANN Toolkit**

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - `${cann_version}`：表示 toolkit 安装包版本号，需满足上文的最低版本要求。
    - `${arch}`：表示 CPU 架构，如 `aarch64`、`x86_64`。
    - `${install_path}`：表示指定安装路径，默认安装在 `/usr/local/Ascend` 目录。

2. **配置环境变量**

   安装完成后，请先执行：

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

   请将 `${install_path}` 替换为 toolkit 的实际安装目录，例如 `/usr/local/Ascend` 或 `${HOME}/Ascend`。

## ⚡️ 快速入门

### 编译与测试

详细的 build.sh 参数说明请参考 [build 参数说明](docs/zh/context/build.md)。

```bash
# 编译所有算子（默认 8 线程）
./build.sh

# 编译指定算子
./build.sh --ops=rfft1_d

# 编译并运行测试
./build.sh --run

# 编译并打包成 .run 文件
./build.sh --pkg

# 查看完整帮助信息
./build.sh --help
```

### 安装

```bash
# 标准安装（需要 root 权限）
sudo ./cann-950-ops-fft_9.0.0_linux-*.run --full

# 查看安装包信息
./cann-950-ops-fft_9.0.0_linux-*.run --help

# 安装到自定义路径
sudo ./cann-950-ops-fft_9.0.0_linux-*.run --full --install-path=/opt/ascend

# 卸载
sudo ./cann-950-ops-fft_9.0.0_linux-*.run --uninstall

# 升级
sudo ./cann-950-ops-fft_9.0.0_linux-*.run --upgrade
```

## 📖 项目说明

### API 接口实现

完整的接口实现状态请参考 [接口实现文档](docs/implementation.md)。各接口的详细说明（参数、约束、调用示例）请参考 [接口参考文档](docs/zh/API_Reference/README.md)。

**已实现的接口（11 个）：**

| 类别 | 接口 | 功能 |
|------|------|------|
| Plan 创建 | `aclfftCreate` | 创建空的 FFT Plan 句柄 |
| Plan 创建 | `aclfftPlan1d` | 创建并初始化一维 FFT Plan |
| Plan 创建 | `aclfftPlan2d` | 创建并初始化二维 FFT Plan |
| Plan 初始化 | `aclfftMakePlan1d` | 初始化一维 FFT Plan |
| Plan 初始化 | `aclfftMakePlan2d` | 初始化二维 FFT Plan |
| 执行接口 | `aclfftExecC2C` | 执行复数到复数的 FFT |
| 执行接口 | `aclfftExecR2C` | 执行实数到复数的一维 FFT |
| 执行接口 | `aclfftExecC2R` | 执行复数到实数的 IFFT |
| Plan 管理 | `aclfftDestroy` | 销毁 FFT Plan 并释放资源 |
| Plan 管理 | `aclfftSetStream` | 设置 Plan 的执行流 |
| 工具接口 | `aclfftGetErrorString` | 获取错误码的描述字符串 |

### 支持的算子

当前支持的算子列表：

| 算子名称 | 描述 | 状态 |
|---------|------|------|
| [fft1_d](src/fft1_d/) | 一维复数FFT运算 | ✅ 已实现 |
| [fft2_d](src/fft2_d/) | 二维复数FFT运算 | ✅ 已实现 |
| [irfft1_d](src/irfft1_d/) | 一维复数到实数IFFT运算 | ✅ 已实现 |
| [rfft1_d](src/rfft1_d/) | 一维实数FFT运算 | ✅ 已实现 |

更多算子正在持续开发中...

### SoC 支持矩阵

| SoC 型号 | SOC_VERSION | 支持状态 |
|---------|-------------|---------|
| Ascend950 | ascend950dt_9595 | ✅ 默认支持 |
| Ascend910B | ascend910b | ✅ 已支持 |
| Ascend910_93 | ascend910_93 | ❌ 暂不支持 |
| Ascend910 | ascend910 | ❌ 暂不支持 |
| Ascend310P | ascend310p | ❌ 暂不支持 |
| Ascend310B | ascend310b | ❌ 暂不支持 |

## 🔍 目录结构

```
ops-fft/
├── cmake/                      # CMake 配置文件
│   ├── func.cmake             # 公共函数（算子注册等）
│   ├── init_env.cmake         # 环境初始化
│   ├── package.cmake          # 打包配置
│   ├── makeself_built_in.cmake # 打包脚本
│   ├── variables.cmake        # 变量定义
│   └── third_party/           # 第三方 CMake 模块
│       └── makeself-fetch.cmake
├── docs/                       # 文档目录
│   ├── README.md              # 文档导航
│   ├── implementation.md      # 接口实现状态文档
│   └── zh/                    # 中文文档
│       ├── op_list.md         # 算子列表
│       ├── context/           # 背景与环境
│       │   ├── build.md       # 编译参数说明
│       │   ├── dir_structure.md # 目录结构说明
│       │   └── quick_install.md # 快速安装指南
│       ├── debug/             # 调试相关
│       │   └── op_debug_prof.md # 算子调试与 Profiling
│       ├── develop/             # 开发指南
│       │   ├── operator_development_guide.md # 算子开发指南
│       │   └── test_writing_guide.md         # 测试编写指南
│       ├── API_Reference/      # API 参考文档
│       │   ├── README.md      # API 文档导航
│       │   ├── FFT_1D.md      # 1D FFT 接口说明
│       │   ├── FFT_2D.md      # 2D FFT 接口说明
│       │   └── figures/       # 文档图片
│       └── invocation/        # 调用示例
│           └── quick_op_invocation.md
├── include/                    # 公共头文件
│   └── cann_ops_fft.h         # API 头文件
├── lib/                        # API 层实现
│   ├── fft_plan_api.cpp       # Plan 创建接口
│   ├── fft_plan_init_api.cpp  # Plan 初始化接口
│   ├── fft_exec_api.cpp       # FFT 执行接口
│   ├── fft_plan_destroy_api.cpp # Plan 销毁接口
│   ├── fft_stream_api.cpp     # 流管理接口
│   ├── fft_utils_api.cpp      # 工具接口
│   ├── fft_error.h            # 错误处理
│   ├── fft_handle_impl.h      # Plan 内部实现
│   └── CMakeLists.txt
├── scripts/                    # 脚本目录
│   ├── check_build_dependencies.py # 构建依赖检查
│   ├── generate_version_info.py    # 版本信息生成
│   └── package/               # 打包相关脚本
│       ├── common/            # 公共打包工具
│       ├── latest_manager/    # 版本管理脚本
│       ├── module/            # 模块配置（XML）
│       ├── ops_fft/           # ops-fft 安装脚本
│       └── package.py         # 打包入口
├── src/                        # 算子层实现
│   ├── common/                 # 公共代码
│   │   ├── host/               # Host 侧公共代码
│   │   ├── kernel/             # Kernel 侧公共实现
│   │   └── tiling/             # Tiling 数据公共定义
│   ├── fft1_d/                 # 一维复数 FFT 算子
│   │   ├── fft1_d.h            # 算子接口
│   │   ├── arch32/             # Ascend910B 架构实现
│   │   │   ├── dft/            # DFT 实现
│   │   │   ├── fft_b/          # FFT-B 实现
│   │   │   ├── fft_mix/        # Mix FFT 实现
│   │   │   ├── fft_n/          # FFT-N 实现
│   │   │   └── fft_stride/     # Stride FFT 实现
│   │   ├── arch35/             # Ascend950 架构实现
│   │   │   ├── fft/            # C2C 实现
│   │   │   └── mix/            # C2C Mix 实现
│   │   └── tests/              # 算子测试
│   ├── fft2_d/                 # 二维复数 FFT 算子
│   │   ├── fft2_d.h            # 算子接口
│   │   ├── arch32/             # Ascend910B 架构实现
│   │   │   └── dd/             # DD 实现
│   │   └── tests/              # 算子测试
│   ├── irfft1_d/               # 一维复数到实数 IFFT 算子
│   │   ├── irfft1_d.h          # 算子接口
│   │   ├── arch32/             # Ascend910B 架构实现
│   │   │   ├── c2r_fft/        # C2R FFT 实现
│   │   │   └── dft/            # DFT 实现
│   │   ├── arch35/             # Ascend950 架构实现
│   │   │   └── fft/            # FFT 实现
│   │   └── tests/              # 算子测试
│   ├── rfft1_d/                # 一维实数 FFT 算子
│   │   ├── rfft1_d.h           # 算子接口
│   │   ├── arch32/             # Ascend910B 架构实现
│   │   │   ├── dft/            # DFT 实现
│   │   │   └── r2c_fft/        # R2C FFT 实现
│   │   ├── arch35/             # Ascend950 架构实现
│   │   │   ├── fast_dft/       # 快速 DFT 实现
│   │   │   └── fft/            # FFT 实现
│   │   └── tests/              # 算子测试
│   └── CMakeLists.txt
├── tests/                      # 测试框架
│   ├── test_common.h          # 测试框架头文件
│   ├── test_common.cpp        # 测试框架实现
│   ├── all_tests.cpp.in       # 测试入口模板
│   └── CMakeLists.txt
├── build.sh                    # 编译脚本
├── CHANGELOG.md                # 版本变更记录
├── classify_rule.yaml          # 文件分类规则
├── CMakeLists.txt              # 主 CMake 配置
├── CONTRIBUTING.md             # 贡献指南
├── install_deps.sh             # 依赖安装脚本
├── LICENSE                     # 许可证
├── OAT.xml                     # 许可证扫描配置
├── QUICKSTART.md               # 快速入门文档
├── requirements.txt            # Python 依赖列表
├── SECURITY.md                 # 安全声明
├── Third_Party_Open_Source_Software_List.yaml  # 三方软件清单
├── Third_Party_Open_Source_Software_Notice     # 三方软件声明
├── version.cmake               # 版本信息（CMake）
└── README.md                   # 本文件
```

## 🛠️ 开发指南

### 添加新算子

详细的算子开发指南请参考 [算子开发指南](docs/zh/develop/operator_development_guide.md)，包括：

- 完整的目录结构说明
- Host + Kernel 实现模板
- Tiling 数据结构定义
- 完整开发流程

**快速开始**：

1. **创建目录**
```bash
mkdir -p src/my_op/arch35 # Huawei Ascend 950 架构实现目录
mkdir -p src/my_op/tests
```

2. **编写算子实现**
创建 `src/my_op/my_op.cpp`，包含：
- Host 部分：对外接口、Tiling 计算、内存管理
- Kernel 部分：核函数实现

3. **创建 CMakeLists.txt**
```cmake
register_operator(NAME my_op ARCH_DIR arch35)
```

4. **编写测试**（推荐）
参考 [测试编写指南](docs/zh/develop/test_writing_guide.md)

5. **编译验证**
```bash
./build.sh --ops=my_op --run
```

完整示例参考 `src/rfft1_d/` 目录。

### 添加新接口

如果要添加新的 API 接口：

1. **在 `include/cann_ops_fft.h` 中声明接口**
2. **在 `lib/` 中创建对应的实现文件**（如 `fft_my_api.cpp`）
3. **在 `lib/CMakeLists.txt` 中添加源文件**
4. **编写测试验证功能**

### 编写测试

ops-fft 提供了轻量级、自动化的测试框架。详细的测试编写指南请参考 [测试编写指南](docs/zh/develop/test_writing_guide.md)，包括：

- 测试框架特性
- 测试文件结构
- 核心宏和函数说明
- 完整编写步骤和示例
- 最佳实践和常见问题

## 💬 相关信息

- **许可证**: [CANN Open Software License Agreement Version 2.0](LICENSE)
- **安全声明**: [SECURITY.md](SECURITY.md)
- **贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **所属 SIG**: [CANN Community](https://gitcode.com/cann/community)

## 🤝 联系我们

本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**: 通过 [Issues](https://gitcode.com/cann/ops-fft/issues) 提交问题
- **社区互动**: 通过 [Discussions](https://gitcode.com/cann/ops-fft/discussions) 参与交流
- **技术专栏**: 通过 [Wiki](https://gitcode.com/cann/ops-fft/wiki) 获取技术文章
