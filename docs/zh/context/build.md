# build.sh 参数说明

## 简介

build.sh 是 ops-fft 项目的构建脚本，位于项目根目录下。该脚本通过配置不同参数实现多种功能，包括编译算子、运行测试、生成安装包等。

## 使用方法

### 配置环境变量

在使用 build.sh 之前，需要先配置 CANN 环境变量：

```bash
# 默认路径安装
source /usr/local/Ascend/cann/set_env.sh

# 验证环境变量
echo $ASCEND_HOME_PATH
```

### 构建命令格式

```bash
./build.sh [OPTIONS]
```

## 参数说明

build.sh 支持多种功能，可通过 `--help` 参数查看所有选项：

```bash
./build.sh --help
```

| 参数 | 必选/可选 | 说明 |
|------|----------|------|
| `--ops=OP_LIST` | 可选 | 指定要编译的算子列表，多个算子用逗号分隔（如：`--ops=rfft1_d,irfft1_d`）。不指定时编译所有算子。 |
| `--run` | 可选 | 编译后执行测试。需要配合 `BUILD_TESTING=ON` 使用。 |
| `--pkg` | 可选 | 编译并打包成 .run 安装包。 |
| `--soc=SOC` | 可选 | 指定目标 SoC 型号，支持大小写不敏感输入（如：`--soc=ascend950` 或 `--soc=Ascend950`）。默认为 `Ascend950`。 |
| `--cann_3rd_lib_path=PATH` | 可选 | 指定 CANN 三方依赖目录。默认为项目根目录下的 `third_party`，指定的目录必须已存在。 |
| `-j[N]` | 可选 | 指定编译线程数，默认为 8（如：`-j16`）。若线程数超过 CPU 核心数，会自动调整为 CPU 核心数。 |
| `--test-timeout=N` | 可选 | 指定测试超时时间（单位：秒），默认为 300。仅在 `--run` 模式下有效。 |
| `-h, --help` | 可选 | 显示帮助信息。 |

## 支持的 SoC 型号

| SoC 型号 | SOC_VERSION（CANN 编译器） | NPU 架构 | 说明 |
|---------|---------------------------|-----------|------|
| Ascend950 | ascend950 | dav-3510（arch35） | 默认支持，对应 `src/<op>/arch35/` 实现 |
| Ascend910B | ascend910b | dav-2201（arch32） | 已支持，对应 `src/<op>/arch32/` 实现 |

> **说明**：`build.sh` 的 `--soc` 还可识别 `ascend910_93`/`ascend910`/`ascend310p` 等名称，但当前仅有 Ascend950 与 Ascend910B 提供算子实现目录（`arch35`/`arch32`）；其他型号的 NPU 架构（如 dav-2101）不受支持，各算子注册时会被显式跳过（日志输出 `SKIPPED (unsupported ASCEND_NPU_ARCH ...)`）。


## 使用示例

### 基本编译

```bash
# 编译所有算子（默认 8 线程）
./build.sh

# 编译指定算子
./build.sh --ops=rfft1_d

# 编译多个算子
./build.sh --ops=rfft1_d,irfft1_d

# 使用 16 线程编译
./build.sh -j16

# 指定 CANN 三方依赖目录
./build.sh --cann_3rd_lib_path=/path/to/third_party
```

### 编译并测试

```bash
# 编译所有算子并运行测试
./build.sh --run

# 编译指定算子并运行测试
./build.sh --ops=rfft1_d --run

# 指定测试超时时间（600 秒）
./build.sh --run --test-timeout=600
```

### 打包

```bash
# 编译所有算子并打包（默认 SoC: Ascend950）
./build.sh --pkg

# 编译指定算子并打包
./build.sh --ops=rfft1_d --pkg

# 为指定 SoC 打包
./build.sh --soc=Ascend950 --pkg

# 大小写不敏感
./build.sh --soc=ascend950 --pkg
```

### 组合使用

```bash
# 编译 rfft1_d 算子、运行测试、使用 16 线程
./build.sh --ops=rfft1_d --run -j16

# 编译所有算子、打包、指定 SoC
./build.sh --pkg --soc=Ascend950 -j16
```

## 行为说明

| 命令 | 行为 |
|------|------|
| 无参数 | 编译所有算子，不执行测试 |
| `--ops=rfft1_d` | 只编译 rfft1_d 算子，不执行测试 |
| `--ops=rfft1_d,irfft1_d` | 编译 rfft1_d 和 irfft1_d 算子，不执行测试 |
| `--run` | 编译所有算子，并执行所有算子的测试 |
| `--ops=rfft1_d --run` | 编译 rfft1_d 算子，并执行 rfft1_d 算子的测试 |
| `--ops=rfft1_d,irfft1_d --run` | 编译 rfft1_d、irfft1_d 算子，并执行这些算子的测试 |
| `--pkg` | 编译所有算子并打包成 .run 文件（默认 SoC: Ascend950） |
| `--ops=rfft1_d --pkg` | 编译 rfft1_d 算子并打包成 .run 文件 |
| `--soc=ascend950 --pkg` | 为 Ascend950 芯片打包（支持小写） |

## 输出说明

### 编译输出

编译成功后，生成的文件位于 `build/` 目录：

```
build/
├── libcann_ops_fft.so    # 动态库（CMakeLists 中 OUTPUT_NAME=cann_ops_fft）
├── tests/
│   └── all_ops_test         # 测试可执行文件
└── ...
```

### 打包输出

打包成功后，会在 `build/` 目录生成 .run 文件：

```
build_out/cann-{soc}-ops-fft_{version}_linux-{arch}.run
```

例如：
- `cann-950-ops-fft_9.0.0_linux-x86_64.run`
- `cann-950-ops-fft_9.0.0_linux-aarch64.run`

## 注意事项

1. **环境变量要求**：必须设置 `ASCEND_HOME_PATH` 或 `ASCEND_INSTALL_PATH` 环境变量（二选一），否则脚本会报错退出。`ASCEND_INSTALL_PATH` 优先级更高。

2. **线程数限制**：如果指定的线程数超过 CPU 核心数，脚本会自动调整为 CPU 核心数。

3. **测试超时**：默认测试超时时间为 300 秒，可根据实际情况调整。

4. **SoC 大小写**：`--soc` 参数支持大小写不敏感输入，脚本会自动标准化为首字母大写、其余小写的格式。

5. **算子验证**：如果使用 `--ops` 指定了不存在的算子，脚本会列出所有可用的算子并报错退出。

## 相关文档

- [环境部署](quick_install.md)
- [算子调用](../invocation/quick_op_invocation.md)
- [算子开发](../develop/operator_development_guide.md)
