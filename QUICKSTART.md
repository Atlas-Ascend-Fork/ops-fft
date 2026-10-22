# 算子开发快速入门：基于ops-fft仓

本指南旨在帮助你快速上手基于CANN和`ops-fft`算子仓的使用，最简化地完成环境安装、编译部署及算子运行。

## 一、环境安装

### 有环境场景：Docker安装

Docker安装环境以Atlas A2产品（910B）为例。

**前提条件**：
* **Docker环境**：宿主机已安装Docker引擎（版本1.11.2及以上）。
* **驱动与固件**：宿主机已安装昇腾NPU的[驱动与固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha)Ascend HDK 24.1.0版本以上。安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)》。
    > **注意**：使用`npu-smi info`查看对应的驱动与固件版本。

#### 下载镜像

拉取已预集成CANN软件包及`ops-fft`所需依赖的镜像。

* **操作步骤**：
    1. 以root用户登录宿主机。
    2. 执行拉取命令（请根据你的宿主机架构选择）：
        * ARM架构：
        ```bash
        docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```
        * X86架构：
        ```bash
        docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```
        > **注意**：正常网速下，镜像下载时间约为5-10分钟。

#### Docker运行

请根据以下命令运行docker：

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```

#### 检查环境

进入容器后，验证环境和驱动是否正常。

* **检查NPU设备**：
    ```bash
    npu-smi info
    ```
    正常判据：能列出NPU设备且各设备行`Health`字段为`OK`。若命令不存在或列不出设备，优先检查容器是否按上文命令挂载了`/dev/davinci*`、`/usr/local/dcmi`、`/usr/local/bin/npu-smi`、`/usr/local/Ascend/driver/lib64/`与`/usr/local/Ascend/driver/version.info`等路径。
* **检查CANN安装**：
    ```bash
    cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    ```
    正常判据：输出非空且形如`Version=Bxxx`的版本串。若文件不存在，说明容器内CANN Toolkit安装路径异常，可执行`ls -l /usr/local/Ascend/ascend-toolkit/latest`确认软链是否有效，必要时重新进入镜像。

#### 版本配套与目标SoC

ops-fft算子包版本号（`9.0.0`）与CANN Toolkit版本号相互独立，包名为`cann-{soc}-ops-fft_{算子包版本}_linux-{arch}.run`。常用组合如下：

| 目标SoC | CANN Toolkit | 算子包（`--soc`） | 说明 |
| --- | --- | --- | --- |
| Atlas 950（默认） | CANN 9.0.0-beta.2（见[README版本配套](README.md)） | `cann-950-ops-fft_9.0.0_linux-*.run`（`--soc=Ascend950`，默认） | master分支已验证组合 |
| Atlas A2（910B） | 本文镜像内置8.5.0-910b | `cann-910b-ops-fft_9.0.0_linux-*.run`（`--soc=Ascend910B`） | 需显式指定`--soc` |

> **注意**：`build.sh`默认目标SoC为`Ascend950`。在910B环境的容器中直接执行`bash build.sh --pkg`得到的是950目标包，无法在910B上运行；请按上表为目标SoC显式传入`--soc`。驱动与固件前提见上文"HDK 24.1.0版本以上"。

### 无环境场景：WebIDE开发（建设中）

对于无环境的用户，提供WebIDE开发方式，**该方式当前不可用，尚在建设中**；在此期间请参考上文"Docker安装"方式搭建环境，建设进展请关注仓库公告。

## 二、编译部署

### 拉取ops-fft仓库代码

```bash
git clone https://gitcode.com/cann/ops-fft.git
cd ops-fft
```

### 编译ops-fft算子包

```bash
bash build.sh --pkg
```

若提示如下信息，说明编译成功。

```bash
Self-extractable archive "cann-950-ops-fft_9.0.0_linux-*.run" successfully created.
Build package success: build_out/cann-950-ops-fft_9.0.0_linux-*.run
```

编译成功后，run包存放于项目根目录的build_out目录下。

> **注意**：编译前必须设置`ASCEND_HOME_PATH`（或`ASCEND_INSTALL_PATH`，优先级更高）环境变量。若未设置，脚本会以`[ERROR] ASCEND_HOME_PATH environment variable is not set!`退出；若指向的目录不存在，会以`[ERROR] ASCEND_HOME_PATH directory does not exist`退出。可执行`source <CANN安装目录>/set_env.sh`（如`/usr/local/Ascend/ascend-toolkit/latest/set_env.sh`）后再重试；若source后仍不生效，检查是否在`su`降权后丢失了环境变量，改用`sudo -E`或切换到安装CANN的用户执行。

### 安装ops-fft算子包

```bash
./build_out/cann-950-ops-fft_9.0.0_linux-*.run --full
```

ops-fft安装在`${ASCEND_HOME_PATH}/cann`路径中，`${ASCEND_HOME_PATH}`表示CANN软件安装目录。

### 配置环境变量

```bash
source ${ASCEND_HOME_PATH}/cann/set_env.bash
```

### 运行测试

```bash
bash build.sh --run
```

测试在NPU上执行（需按上文完成环境变量配置并保证设备可用），编译后自动运行`tests/all_ops_test`测试集并输出gtest结果。最小成功判据：末尾出现`[SUCCESS] All tests passed`且进程退出码为0。

常用选项：

* `--ops=rfft1_d`：只编译并运行指定算子（逗号分隔多个算子）的测试，适合单算子smoke验证；
* `--test-timeout=600`：调整测试超时时间（默认300秒）。

失败排查：测试失败时脚本以`[ERROR] NPU test failed (exit code: N)`退出，gtest失败用例的详细信息会直接打印在终端输出中；若提示测试可执行文件不存在，先确认上一步编译成功；若因超时退出（`[ERROR] Test timeout`），可用`--test-timeout`放宽限制或用`--ops`缩小范围后重试。

## 三、更多帮助

- [CANN 开发文档](https://www.hiascend.com/document)
- [Ascend C API 参考](https://hiascend.com/document/redirect/CannCommunityAscendCApi)
