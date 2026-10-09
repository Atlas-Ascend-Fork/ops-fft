# 算子调用

## 前提条件

- 环境部署：调用项目算子之前，请先参考[环境部署](../context/quick_install.md)完成基础环境搭建。
- 调用算子列表：项目可调用的算子参见[算子列表](../op_list.md)。

## 编译执行

基于社区版CANN包对算子源码修改时，可采用如下方式进行源码编译：

- [ops-fft包](#ops-fft包)：选择整个项目编译生成的包称为ops-fft包，可**完整替换**CANN包对应部分。

### ops-fft包

1. **编译ops-fft包**

    进入项目根目录，执行如下编译命令：

    ```bash
    # 编译所有算子并生成安装包
    bash build.sh --pkg
    ```

    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-950-ops-fft_9.0.0_linux-*.run" successfully created.
    Build package success: build_out/cann-950-ops-fft_9.0.0_linux-*.run
    ```

    编译成功后，run包存放于项目根目录的build目录下。

2. **安装ops-fft包**

    ```bash
    # 安装命令
    ./build_out/cann-*-ops-fft-*linux*.run --full
    ```

    ops-fft安装在`${ASCEND_HOME_PATH}/cann`路径中，`${ASCEND_HOME_PATH}`表示CANN软件安装目录。

3. **配置环境变量**

    ```bash
    source ${ASCEND_HOME_PATH}/cann/set_env.bash
    ```

4. **（可选）卸载ops-fft包**

    ```bash
    # 卸载命令
    ./build_out/cann-*-ops-fft-*linux*.run --uninstall
    ```

## 本地验证

通过项目根目录build.sh脚本，可快速调用算子和UT用例，验证项目功能是否正常，build参数介绍参见[build参数说明](../context/build.md)。

### 运行测试

```bash
# 编译并运行测试
bash build.sh --run

# 编译指定算子并运行测试
bash build.sh --ops=rfft1_d --run
```

执行测试后会打印执行结果，以rfft1_d算子为例，结果如下：

```
all_ops_test .......... Passed * sec
```

### 编译选项说明

| 选项 | 说明 | 示例 |
| :--- | :--- | :--- |
| `--ops=NAME` | 编译指定算子 | `--ops=rfft1_d` |
| `--build-type=TYPE` | 构建类型（Release/Debug） | `--build-type=Debug` |
| `--run` | 编译后运行测试 | `--run` |
| `--pkg` | 生成安装包 | `--pkg` |
| `-j[N]` | 并行编译线程数 | `-j8` |
| `-v` | 详细输出 | `-v` |

### 常用命令示例

```bash
# 基本编译
bash build.sh

# 编译并运行测试
bash build.sh --run

# 编译指定算子并运行测试
bash build.sh --ops=rfft1_d --run

# 编译生成安装包
bash build.sh --pkg

# 多线程编译
bash build.sh -j16

# 调试模式编译
bash build.sh --build-type=Debug --run

# 详细输出
bash build.sh -v --run
```

## 更多帮助

- [CANN 开发文档](https://www.hiascend.com/document)
- [Ascend C API 参考](https://hiascend.com/document/redirect/CannCommunityAscendCApi)
