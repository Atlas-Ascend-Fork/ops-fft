#!/bin/bash

#-----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#-----------------------------------------------------------------------------

##############################################################################
# ops-fft Build Script
#
# 使用方法:
#   ./build.sh [OPTIONS]
#
# 选项:
#   --ops=OP_LIST       指定要编译的算子列表 (逗号分隔)
#   --run               编译后执行测试
#   --pkg               编译并打包成 .run 文件
#   --soc=SOC           指定目标 SoC 型号 (支持: Ascend950、Ascend910B、
#                       Ascend910_93、Ascend910、Ascend310P，支持小写输入)
#   -j[N]               编译线程数，默认为 8，例如: -j16
#   --test-timeout=N    测试超时时间（秒），默认为 300
#   --cann_3rd_lib_path=PATH
#                       指定 CANN 三方依赖目录 (默认: ./third_party)
#   -h, --help          显示帮助信息
#
# 行为说明:
#   无参数                          编译所有算子，不执行测试
#   --ops=rfft1_d                  只编译 rfft1_d 算子，不执行测试
#   --ops=rfft1_d,irfft1_d         编译 rfft1_d 和 irfft1_d 算子，不执行测试
#   --run                          编译所有算子，并执行所有算子的测试
#   --ops=rfft1_d --run            编译 rfft1_d 算子，并执行 rfft1_d 算子的测试
#   --ops=rfft1_d,irfft1_d --run   编译 rfft1_d,irfft1_d 算子，并执行这些算子的测试
#   --pkg                          编译所有算子并打包成 .run 文件 (默认 SoC: Ascend950)
#   --ops=rfft1_d --pkg            编译 rfft1_d 算子并打包成 .run 文件
#   --soc=ascend950 --pkg          为 Ascend950 芯片打包 (支持小写)
#   -j16                           使用 16 个线程编译
#
# 示例:
#   ./build.sh                            # 编译所有算子 (默认 8 线程)
#   ./build.sh --ops=rfft1_d              # 只编译 rfft1_d 算子
#   ./build.sh -j16                       # 使用 16 线程编译所有算子
#   ./build.sh --run                      # 编译所有算子并执行测试
#   ./build.sh --ops=rfft1_d --run        # 编译 rfft1_d 算子并执行测试
#   ./build.sh --pkg                      # 编译所有算子并打包 (默认: Ascend950)
#   ./build.sh --ops=rfft1_d --pkg        # 编译 rfft1_d 算子并打包
#   ./build.sh --soc=ascend950 --pkg      # 为 Ascend950 打包 (支持小写输入)
#   ./build.sh --test-timeout=600 --run   # 运行测试 with 600s timeout
#
# 支持的 SoC 型号:
#   Ascend950   (dav-3510, 默认)
#   Ascend910B  (dav-2201)
#   Ascend910_93(dav-2201)
#   Ascend910   (dav-2101)
#   Ascend310P  (dav-2101)
# 注: Ascend310B 可识别但当前版本暂不支持构建
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
BUILD_OPERATORS="all"
RUN_TESTS=false
ENABLE_PACKAGE=false
SOC_NAME="Ascend950"  # 默认 SoC 型号
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 尽早校验仓库完整性，避免在错误目录下构建无关工程（issue #83）
if [ ! -f "${SCRIPT_DIR}/CMakeLists.txt" ]; then
    echo -e "${RED}错误: ${SCRIPT_DIR} 下未找到 CMakeLists.txt，请确认脚本位于 ops-fft 仓库根目录${NC}" >&2
    exit 1
fi
# 构建目录基于脚本绝对路径拼接，支持从任意工作目录调用本脚本（issue #83）
BUILD_DIR="${SCRIPT_DIR}/build"
THREAD_NUM=8  # 默认编译线程数
# CPU 核数可靠探测（issue #195）：原管道 `cat /proc/cpuinfo | grep | wc -l` 的
# 退出码取决于最后的 wc（几乎恒成功），|| 兜底不可达；无 /proc/cpuinfo 环境
# 会得到 CORE_NUMS=0 并以 -j0 构建。改为 nproc → getconf → /proc/cpuinfo → 4。
# 命令替换内部补 || true（PR #93 检视修正 + issue #199 补齐 2/3 级）：本脚本顶部
# set -e，命令替换非零退出会连同其所在赋值语句一起使脚本中止（如极端环境下
# nproc 不存在/失败、getconf 不存在、/proc/cpuinfo 缺失或无 "processor" 行时
# grep -c 以退出码 1 结束），兜底链根本不会执行，CORE_NUMS=4 终极兜底不可达
CORE_NUMS=$(nproc 2>/dev/null || true)
if [ -z "$CORE_NUMS" ] || [ "$CORE_NUMS" -lt 1 ] 2>/dev/null; then
    CORE_NUMS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
fi
if [ -z "$CORE_NUMS" ] || [ "$CORE_NUMS" -lt 1 ] 2>/dev/null; then
    CORE_NUMS=$(cat /proc/cpuinfo 2>/dev/null | grep -c "processor" || true)
fi
if [ -z "$CORE_NUMS" ] || [ "$CORE_NUMS" -lt 1 ] 2>/dev/null; then
    CORE_NUMS=4
fi
TEST_TIMEOUT=300  # 默认测试超时时间（秒）
BUILD_OUT_DIR="${SCRIPT_DIR}/build_out"
VERBOSE=""
CANN_3RD_LIB_PATH="${SCRIPT_DIR}/third_party"
TEST_FILTER=""  # 测试过滤器（逗号分隔的算子名称，如 "fft1_d"）

# 设置 _ASCEND_INSTALL_PATH（优先级：ASCEND_INSTALL_PATH > ASCEND_HOME_PATH > 默认值）
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
fi

# 统一导出 ASCEND_HOME_PATH，使后续 check_ascend_env 与 CMake 均可使用
export ASCEND_HOME_PATH="${_ASCEND_INSTALL_PATH}"

# SoC 名称映射到完整的 SOC_VERSION（CANN ASC 编译器要求的格式）
get_soc_version() {
    local soc_name="$1"
    local soc_lower=$(echo "$soc_name" | tr '[:upper:]' '[:lower:]')

    # 映射表：友好名称 -> CANN 期望的完整版本标识
    case "$soc_lower" in
        "ascend950")
            echo "Ascend950"
            ;;
        "ascend910b"|"ascend910_b")
            echo "Ascend910B"
            ;;
        "ascend910_93")
            echo "Ascend910_93"
            ;;
        "ascend910")
            echo "Ascend910"
            ;;
        "ascend310p"|"ascend310_p")
            echo "Ascend310P"
            ;;
        "ascend310b"|"ascend310_b")
            echo "Ascend310B"
            ;;
        *)
            log_error "Unknown SoC: '${soc_name}'. Recognized SoCs: Ascend950, Ascend910B, Ascend910_93, Ascend910, Ascend310P, Ascend310B"
            exit 1
            ;;
    esac
}

# SoC 名称映射到 NPU 架构标识（用于 CMake arch 目录选择）
soc_to_npu_arch() {
    local soc_name="$1"
    local soc_lower=$(echo "$soc_name" | tr '[:upper:]' '[:lower:]')

    case "$soc_lower" in
        "ascend950")
            echo "dav-3510"
            ;;
        "ascend910b"|"ascend910_b")
            echo "dav-2201"
            ;;
        "ascend910_93")
            echo "dav-2201"
            ;;
        "ascend910")
            echo "dav-2101"
            ;;
        "ascend310p"|"ascend310_p")
            echo "dav-2101"
            ;;
        "ascend310b"|"ascend310_b")
            log_error "SoC '${soc_name}' is recognized but not supported in the current version."
            exit 1
            ;;
        *)
            log_error "Unknown SoC architecture: '${soc_name}'. Supported SoCs: Ascend950, Ascend910B, Ascend910_93, Ascend910, Ascend310P"
            exit 1
            ;;
    esac
}

# 检查 ASCEND 环境变量
check_ascend_env() {
    log_info "Checking ASCEND environment..."

    if [ -z "${ASCEND_HOME_PATH}" ]; then
        log_error "ASCEND_HOME_PATH environment variable is not set!"
        echo ""
        echo "Please source the CANN environment script:"
        echo "  source /usr/local/Ascend/cann/set_env.sh"
        echo "or set ASCEND_INSTALL_PATH / ASCEND_HOME_PATH manually."
        echo ""
        echo "Then run build.sh again."
        exit 1
    fi

    if [ ! -d "${ASCEND_HOME_PATH}" ]; then
        log_error "ASCEND_HOME_PATH directory does not exist: ${ASCEND_HOME_PATH}"
        exit 1
    fi

    log_success "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
}

# 日志函数（统一接口）
log() {
    local level=$1
    local msg=$2
    local color=""
    local prefix=""

    case $level in
        info)
            color="${BLUE}"
            prefix="[INFO]"
            ;;
        success)
            color="${GREEN}"
            prefix="[SUCCESS]"
            ;;
        warning)
            color="${YELLOW}"
            prefix="[WARNING]"
            ;;
        error)
            color="${RED}"
            prefix="[ERROR]"
            ;;
        verbose)
            if [ -z "$VERBOSE" ]; then
                return
            fi
            color="${BLUE}"
            prefix="[VERBOSE]"
            ;;
        *)
            color="${NC}"
            prefix="[LOG]"
            ;;
    esac

    echo -e "${color}${prefix}${NC} ${msg}"
}

# 兼容性别名
log_info() { log info "$1"; }
log_success() { log success "$1"; }
log_warning() { log warning "$1"; }
log_error() { log error "$1"; }
log_verbose() { log verbose "$1"; }

# 显示帮助信息
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --ops=OP_LIST       Specify operators to build (comma-separated)
  --run               Run tests after build
  --test=TEST_FILTER  Run only specified test(s) (comma-separated, e.g., fft1_d)
  --pkg               Build package (.run file)
  --soc=SOC           Target SoC model (default: Ascend950, case-insensitive)
  -j[N]               Number of compile threads (default: 8), e.g., -j16
  --test-timeout=N    Test timeout in seconds (default: 300)
  --cann_3rd_lib_path=PATH
                      CANN third-party dependency path (default: ./third_party)
  -h, --help          Show this help message
  -v, --verbose       Verbose output
  --make_clean        Clean build artifacts

Supported SoC models:
  Ascend950     (dav-3510, default)
  Ascend910B    (dav-2201)
  Ascend910_93  (dav-2201)
  Ascend910     (dav-2101)
  Ascend310P    (dav-2101)

Note: Ascend310B is recognized but not supported in the current version.
      Other SoC models are not supported.

Examples:
  $(basename "$0")                              # Build all operators (default 8 threads)
  $(basename "$0") --ops=rfft1_d                # Build only 'rfft1_d' operator
  $(basename "$0") --ops=rfft1_d,irfft1_d       # Build 'rfft1_d' and 'irfft1_d' operators
  $(basename "$0") -j16                         # Build with 16 threads
  $(basename "$0") --run                        # Build all and run tests
  $(basename "$0") --ops=rfft1_d --run          # Build 'rfft1_d' and run tests
  $(basename "$0") --pkg                        # Build package (default SoC: Ascend950)
  $(basename "$0") --ops=rfft1_d --pkg          # Build 'rfft1_d' and package
  $(basename "$0") --soc=Ascend950 --pkg        # Build package for Ascend950
  $(basename "$0") --soc=Ascend910B --pkg       # Build package for Ascend910B
  $(basename "$0") --soc=ascend950 --pkg        # Build package (lowercase also works)
  $(basename "$0") --cann_3rd_lib_path=/path    # Specify CANN third-party dependencies
  $(basename "$0") --test-timeout=600 --run     # Run tests with 600s timeout
  $(basename "$0") --make_clean               # Clean build artifacts

EOF
}

# 解析逗号分隔的算子列表
parse_operators() {
    local ops_str=$1
    if [ "$ops_str" = "all" ]; then
        echo "all"
    else
        IFS=',' read -ra OPS <<< "$ops_str"
        printf '%s\n' "${OPS[@]}"
    fi
}

# 获取所有可用的算子
get_available_operators() {
    find "$SCRIPT_DIR/src" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort
}

# 验证指定的算子是否存在
validate_operators() {
    local ops=($1)
    local available_ops=($(get_available_operators))
    local invalid_ops=()

    for op in "${ops[@]}"; do
        local found=false
        for avail_op in "${available_ops[@]}"; do
            if [ "$op" = "$avail_op" ]; then
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            invalid_ops+=("$op")
        fi
    done

    if [ ${#invalid_ops[@]} -gt 0 ]; then
        log_error "Invalid operators: ${invalid_ops[*]}"
        log_info "Available operators: ${available_ops[*]}"
        exit 1
    fi
}

# 清理构建目录
clean_build() {
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        log_success "Build directory cleaned"
    else
        log_warning "Build directory does not exist, nothing to clean"
    fi
}

clean_build_out() {
    if [ -d "${BUILD_OUT_DIR}" ]; then
        rm -rf "${BUILD_OUT_DIR}"
        log_success "build_out directory cleaned"
    else
        log_info "build_out directory does not exist, nothing to clean"
    fi
}

# 构建项目
build_project() {
    log_info "Starting build..."

    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # 配置CMake
    log_info "Configuring CMake..."
    local cmake_args=()

    if [ "$BUILD_OPERATORS" != "all" ]; then
        cmake_args+=(-DENABLED_OPERATORS="${BUILD_OPERATORS}")
        log_info "Enabled operators: ${BUILD_OPERATORS}"
    else
        cmake_args+=(-DENABLED_OPERATORS="")
        log_info "All operators enabled"
    fi

    # 如果需要运行测试，启用测试编译
    if [ "$RUN_TESTS" = true ]; then
        cmake_args+=(-DBUILD_TESTING=ON)
        log_info "Test compilation: ENABLED"
    else
        cmake_args+=(-DBUILD_TESTING=OFF)
        log_info "Test compilation: DISABLED"
    fi

    # 如果需要打包，启用打包（显式传递 OFF 防止缓存）
    if [ "$ENABLE_PACKAGE" = true ]; then
        cmake_args+=(-DENABLE_PACKAGE=ON)
        log_info "Package generation: ENABLED"
    else
        cmake_args+=(-DENABLE_PACKAGE=OFF)
    fi

    # 传递 SoC 型号和 ASCEND 路径
    # SOC_VERSION: ASC 编译器期望简化格式（如 ascend950，不是 ascend950dt_9595）
    # ASCEND_SOC: init_env.cmake 用于映射 NPU 架构
    SOC_NAME_LOWER=$(echo "${SOC_NAME}" | tr '[:upper:]' '[:lower:]')
    cmake_args+=(-DSOC_VERSION="${SOC_NAME_LOWER}")
    NPU_ARCH=$(soc_to_npu_arch "$SOC_NAME")
    cmake_args+=(-DASCEND_NPU_ARCH="${NPU_ARCH}")
    cmake_args+=(-DASCEND_SOC="${SOC_NAME}")
    cmake_args+=(-DASCEND_HOME_PATH="${ASCEND_HOME_PATH}")
    log_info "Target SoC: ${SOC_NAME}"
    log_info "  -> SOC_VERSION: ${SOC_NAME_LOWER} (for ASC compiler)"
    log_info "  -> ASCEND_NPU_ARCH: ${NPU_ARCH} (for arch dir selection)"
    log_info "  -> ASCEND_SOC: ${SOC_NAME} (for NPU arch mapping)"
    log_info "ASCEND path: ${ASCEND_HOME_PATH}"

    cmake_args+=(-DCANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}")

    cmake "${cmake_args[@]}" ..

    # 编译
    log_info "Compiling with ${THREAD_NUM} threads..."
    # 临时禁用 set -e，手动处理错误（同 run_tests 的既有模式），
    # 使下面的失败提示真正可达（issue #109）
    set +e
    cmake --build . -j${THREAD_NUM} ${VERBOSE}
    build_result=$?
    set -e

    if [ $build_result -eq 0 ]; then
        log_success "Build succeeded"
    else
        log_error "Build failed"
        exit 1
    fi

    cd "$SCRIPT_DIR"
}

# 打包
build_package() {
    log_info "Building package..."

    cd "$BUILD_DIR"

    # 运行 ctest package target
    log_info "Running CPack..."
    # 临时禁用 set -e，手动处理错误（同 run_tests 的既有模式），
    # 使下面的失败提示真正可达（issue #109）
    set +e
    cpack
    cpack_result=$?
    set -e

    if [ $cpack_result -eq 0 ]; then
        log_success "Package created successfully"
        # 查找生成的 .run 文件
        local run_file=$(ls *.run 2>/dev/null | head -n 1)
        if [ -n "$run_file" ]; then
            log_success "Package file: $BUILD_DIR/$run_file"
        fi
    else
        log_error "Package creation failed"
        exit 1
    fi

    cd "$SCRIPT_DIR"
}

# 运行测试
run_tests() {
    log_info "Running tests (timeout: ${TEST_TIMEOUT}s)..."

    cd "$BUILD_DIR" || {
        log_error "Build directory not found: $BUILD_DIR"
        exit 1
    }

    # 检查测试可执行文件
    if [ ! -f "./tests/all_ops_test" ]; then
        log_error "Test executable not found: ./tests/all_ops_test"
        log_error "Please build the project first with: ./build.sh --run"
        exit 1
    fi

    # 临时禁用 set -e，手动处理错误
    set +e
    if [ -n "$TEST_FILTER" ]; then
        log_info "Test filter: ${TEST_FILTER}"
        export OPS_FFT_TEST_FILTER="${TEST_FILTER}"
    fi

    export OPS_FFT_TEST_DATA_ROOT="$SCRIPT_DIR/src"
    timeout -k 1s ${TEST_TIMEOUT}s ./tests/all_ops_test 2>&1
    test_result=$?
    set -e

    cd "$SCRIPT_DIR"

    if [ $test_result -ge 124 ]; then
        log_error "Test timeout (${TEST_TIMEOUT}s exceeded)"
        exit 1
    elif [ $test_result -ne 0 ]; then
        log_error "NPU test failed (exit code: $test_result)"
        exit 1
    fi
    log_success "All tests passed"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ops=*)
                # 提取等号后的值
                BUILD_OPERATORS="${1#*=}"
                shift
                ;;
            --run)
                RUN_TESTS=true
                shift
                ;;
            --test=*)
                TEST_FILTER="${1#*=}"
                RUN_TESTS=true
                shift
                ;;
            --pkg)
                ENABLE_PACKAGE=true
                shift
                ;;
            --soc=*)
                # 提取 SoC 型号并标准化（支持小写输入）
                SOC_INPUT="${1#*=}"
                SOC_NAME=$(get_soc_version "${SOC_INPUT}")
                shift
                ;;
            -j*)
                # 提取线程数
                if [[ "$1" == "-j" ]]; then
                    # -j N 的形式
                    if [[ -z "$2" ]]; then
                        log_error "Missing thread number after -j"
                        exit 1
                    fi
                    if [[ "$2" == -* ]]; then
                        local suggested_num="${2#-}"
                        log_error "Invalid thread number: $2 (did you mean -j${suggested_num}?)"
                        exit 1
                    fi
                    THREAD_NUM="$2"
                    shift 2
                else
                    # -jN 的形式
                    THREAD_NUM="${1#-j}"
                    shift
                fi
                if [[ ! "$THREAD_NUM" =~ ^[0-9]+$ ]]; then
                    log_error "non-integer argument:$THREAD_NUM"
                    exit 1
                fi
                if [ "$THREAD_NUM" -lt 1 ]; then
                    log_error "Thread number must be >= 1, got: $THREAD_NUM"
                    exit 1
                fi
                ;;
            --test-timeout=*)
                # 提取测试超时时间
                TEST_TIMEOUT="${1#*=}"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            --make_clean)
                clean_build
                clean_build_out
                exit 0
                ;;
            -v|--verbose)
                VERBOSE="-v"
                shift
                ;;
            --cann_3rd_lib_path=*)
                CANN_3RD_LIB_PATH="$(realpath "${1#*=}")"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo ""
                show_help
                exit 1
                ;;
        esac
    done
}

# 检查依赖项
check_dependencies() {
    log_info "Checking dependencies..."

    local missing_deps=0

    # 检查cmake
    if ! command -v cmake &> /dev/null; then
        log_error "cmake is not installed"
        missing_deps=$((missing_deps + 1))
    else
        local cmake_version=$(cmake --version | head -n1 | awk '{print $3}')
        log_success "cmake is installed (version: $cmake_version)"
    fi

    # 检查make
    if ! command -v make &> /dev/null; then
        log_warning "make is not installed, will use ninja or other generator"
    else
        local make_version=$(make --version | head -n1)
        log_success "make is installed ($make_version)"
    fi

    # 检查编译器 (gcc/g++)
    if ! command -v g++ &> /dev/null; then
        log_error "g++ is not installed"
        missing_deps=$((missing_deps + 1))
    else
        local gcc_version=$(g++ --version | head -n1)
        log_success "g++ is installed ($gcc_version)"
    fi

    # 检查ASC编译器
    if ! command -v bisheng &> /dev/null; then
        log_error "bisheng compiler (bisheng) is not installed"
        missing_deps=$((missing_deps + 1))
    else
        local asc_version=$(bisheng --version | head -n1)
        log_success "bisheng compiler is installed ($asc_version)"
    fi

    # 检查Python (用于测试脚本)
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        log_warning "Python is not installed, some test scripts may not run"
    else
        local python_cmd="python3"
        if ! command -v python3 &> /dev/null; then
            python_cmd="python"
        fi
        local python_version=$($python_cmd --version 2>&1)
        log_success "Python is installed ($python_version)"
    fi

    if [ $missing_deps -gt 0 ]; then
        log_error "Missing $missing_deps required dependencies, please install and retry"
        exit 1
    fi

    # ASCEND 环境检查（强制要求）
    check_ascend_env

    log_success "All dependencies checked"
}

# 主函数
main() {
    echo "=========================================="
    echo "    ops-fft Build Script"
    echo "=========================================="
    echo ""

    parse_arguments "$@"

    # 依赖检查
    check_dependencies

    # 检查并调整线程数
    if [ "$THREAD_NUM" -gt "$CORE_NUMS" ]; then
        log_warning "Thread num $THREAD_NUM over core num $CORE_NUMS, adjust to $CORE_NUMS"
        THREAD_NUM=$CORE_NUMS
    fi
    # 钳位后下界保护（issue #195）：CORE_NUMS 异常为 0 时 THREAD_NUM 会被钳为 0，
    # cmake -j0 使 GNU make 不限并行度，与参数解析处对 -j0 的显式拒绝口径矛盾
    if [ "$THREAD_NUM" -lt 1 ]; then
        log_warning "Thread num adjusted to non-positive value, reset to default 8"
        THREAD_NUM=8
    fi

    # 验证算子
    if [ "$BUILD_OPERATORS" != "all" ]; then
        IFS=',' read -ra OPS <<< "$BUILD_OPERATORS"
        validate_operators "${OPS[*]}"
    fi

    # 显示执行计划
    echo ""
    log_info "========== Build Plan =========="
    if [ "$BUILD_OPERATORS" = "all" ]; then
        log_info "Build operators: all"
    else
        log_info "Build operators: ${BUILD_OPERATORS}"
    fi

    if [ "$RUN_TESTS" = true ]; then
        log_success "Run tests: YES"
        log_info "Test timeout: ${TEST_TIMEOUT}s"
        if [ "$BUILD_OPERATORS" != "all" ]; then
            log_info "Test operators: ${BUILD_OPERATORS}"
        else
            log_info "Test operators: all built operators"
        fi
    else
        log_info "Run tests: NO"
    fi

    if [ "$ENABLE_PACKAGE" = true ]; then
        log_success "Build package: YES"
        log_info "Target SoC: ${SOC_NAME}"
    else
        log_info "Build package: NO"
    fi
    echo "=================================="
    echo ""

    # 构建项目
    build_project

    # 运行测试（如果指定）
    if [ "$RUN_TESTS" = true ]; then
        run_tests
    fi

    # 打包（如果指定）
    if [ "$ENABLE_PACKAGE" = true ]; then
        build_package
    fi

    log_success "All operations completed!"
}

# 运行主函数
main "$@"
