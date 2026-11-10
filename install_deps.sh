#!/bin/bash

# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -euo pipefail

run_command() {
    local cmd="$*"
    local exit_code=0
    echo "Executing command: $cmd" >&2

    if "$@"; then
        return 0
    else
        exit_code=$?
    fi
    echo "Command execution failed: $cmd (exit code: $exit_code)" >&2
    return "$exit_code"
}

version_ge() {
    # Version comparison, format: xx.xx.xx
    IFS='.' read -r -a curr_arr <<< "$1"
    IFS='.' read -r -a req_arr <<< "$2"

    for ((i=0; i<${#req_arr[@]}; i++)); do
        curr=${curr_arr[i]:-0}
        req=${req_arr[i]}
        if (( curr > req )); then
            return 0
        elif (( curr < req )); then
            return 1
        fi
    done
    return 0
}

version_lt() {
    ! version_ge "$1" "$2"
}

# GCC/g++ 版本解析（issue #312）：原 `--version | awk '/^gcc/ {print $4}'` 依赖
# 品牌串字段位——Ubuntu 格式“gcc (品牌) 11.4.0 ...”第 4 段为版本，RHEL 格式
# “gcc (GCC) 8.3.1 20190507 ...”第 4 段为日期、无日期后缀时为空串，取错值后
# version_ge 误判触发安装分支。改为对首行提取首个 x.y.z 版本串，无 x.y.z 时
# 回退 -dumpfullversion（GCC 7+ 恒输出完整版本），仍无则回 0.0.0 走安装分支
parse_gcc_version() {
    local compiler="$1"
    local ver
    ver=$("$compiler" --version 2>/dev/null | head -n 1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
    if [ -z "$ver" ]; then
        ver=$("$compiler" -dumpfullversion 2>/dev/null | head -n 1)
    fi
    if [ -z "$ver" ]; then
        ver="0.0.0"
    fi
    echo "$ver"
}

detect_os() {
    # OS detection, supports debian (uses apt), rhel (uses dnf or yum), macos
    if [[ "$(uname -s)" == "Linux" ]]; then
        if [[ -f /etc/debian_version ]]; then
            OS="debian"
            PKG_MANAGER="apt"
        elif [[ -f /etc/redhat-release ]]; then
            OS="rhel"
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
        else
            echo "自动安装脚本不支持该Linux发行版本，请手动安装依赖"
            exit 1
        fi
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        OS="macos"
        if ! command -v brew &> /dev/null; then
            echo "Please install Homebrew first"
            exit 1
        fi
        PKG_MANAGER="brew"
    else
        echo "Unsupported OS type"
        exit 1
    fi
}

install_gawk() {
    echo -e "\n==== Checking Gawk ===="

    if command -v gawk &> /dev/null; then
        echo "Gawk has been installed"
        return
    fi

    echo "Installing Gawk..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER update
            run_command sudo $PKG_MANAGER install -y gawk
            ;;
        rhel)
            run_command sudo $PKG_MANAGER install -y gawk
            ;;
        macos)
            run_command brew install gawk
            ;;
    esac

    if command -v gawk &> /dev/null; then
        echo "Gawk installed successfully"
    else
        echo "Gawk installation failed"
        exit 1
    fi
}

install_python() {
    # Python version >= 3.7.0, < 3.14 (issue #244: align with torch 2.8.0 support matrix;
    # the former <3.10 hard cap exited on mainstream distros like Ubuntu 22.04/24.04)
    # Note: versions beyond the cap warn but no longer abort the whole setup
    echo -e "\n==== Checking Python ===="
    local req_ver="3.7.0"
    local max_ver="3.14.0"
    local curr_ver=""

    if command -v python3 &> /dev/null; then
        curr_ver=$(python3 --version 2>&1 | awk '{print $2}')
        echo "Current Python version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver" && version_lt "$curr_ver" "$max_ver"; then
            echo "Python version meets requirements"
            return
        elif version_ge "$curr_ver" "$max_ver"; then
            echo "WARNING: Python version $curr_ver is newer than the tested range (>= 3.7.0, < 3.14); continuing, report issues if build tools misbehave"
        fi
    fi
    echo "Installing Python..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER update
            run_command sudo $PKG_MANAGER install -y python3 python3-pip python3-dev
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command sudo $PKG_MANAGER install -y centos-release-scl
                run_command sudo $PKG_MANAGER install -y rh-python38 rh-python38-python-devel
                run_command source /opt/rh/rh-python38/enable
                echo "Need to execute 'source /opt/rh/rh-python38/enable' to activate python3.8"
            else
                run_command sudo $PKG_MANAGER install -y python3 python3-pip python3-devel
            fi
            ;;
        macos)
            run_command brew install python@3.9
            # 幂等追加（issue #331）：重复运行不再向 ~/.zshrc 叠加重复行；
            # 不在安装器进程内 source 用户 rc——bash 下执行 ~/.zshrc 的 zsh
            # 语法（compdef/setopt/oh-my-zsh 等）退出码常非零，set -e 会误判
            # 安装失败中断流程，改为提示用户重开终端生效
            if ! grep -qF 'export PATH="/usr/local/opt/python@3.9/bin:$PATH"' ~/.zshrc 2>/dev/null; then
                echo 'export PATH="/usr/local/opt/python@3.9/bin:$PATH"' >> ~/.zshrc
            fi
            echo "python@3.9 installed; open a new terminal (or run 'source ~/.zshrc') to take effect"
            ;;
    esac

    if command -v python3 &> /dev/null; then
        curr_ver=$(python3 --version 2>&1 | awk '{print $2}')
        if version_ge "$curr_ver" "$req_ver" && version_lt "$curr_ver" "$max_ver"; then
            echo "Python installed successfully ($curr_ver)"
        else
            # 报错文案与 max_ver 统一（issue #311）：实际校验上限已按 #244 提升至
            # < 3.14，文案停留 "< 3.10" 会误导 3.10~3.13 用户降级到 3.9
            echo "Python version $curr_ver does not meet requirements (required: >= 3.7.0, < 3.14), please install manually"
            exit 1
        fi
    else
        echo "Python installation failed"
        exit 1
    fi
}

install_gcc() {
    # GCC version >= 7.3.0
    echo -e "\n==== Checking GCC ===="
    local req_ver="7.3.0"
    local curr_ver=""

    if command -v gcc &> /dev/null; then
        curr_ver=$(parse_gcc_version gcc)
    elif command -v g++ &> /dev/null; then
        curr_ver=$(parse_gcc_version g++)
    else
        curr_ver="0.0.0"
    fi
    echo "Current GCC version: $curr_ver"
    if version_ge "$curr_ver" "$req_ver"; then
        echo "GCC version meets requirements ($curr_ver)"
        return
    fi

    echo "Installing GCC..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER update
            run_command sudo $PKG_MANAGER install -y gcc-9 g++-9
            run_command sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
                --slave /usr/bin/g++ g++ /usr/bin/g++-9
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command sudo $PKG_MANAGER install -y centos-release-scl
                run_command sudo $PKG_MANAGER install -y devtoolset-9-gcc devtoolset-9-gcc-c++
                run_command source /opt/rh/devtoolset-9/enable
                echo "Need to execute 'source /opt/rh/devtoolset-9/enable' to activate GCC9"
            else
                run_command sudo $PKG_MANAGER install -y gcc gcc-c++
            fi
            ;;
        macos)
            if ! xcode-select -p &> /dev/null; then
                xcode-select --install
            fi
            run_command brew install gcc@11
            # 幂等追加（issue #331）：同 python 分支，不 source 用户 rc、不重复追加
            if ! grep -qF 'export CC=/usr/local/bin/gcc-11' ~/.zshrc 2>/dev/null; then
                echo 'export CC=/usr/local/bin/gcc-11' >> ~/.zshrc
            fi
            if ! grep -qF 'export CXX=/usr/local/bin/g++-11' ~/.zshrc 2>/dev/null; then
                echo 'export CXX=/usr/local/bin/g++-11' >> ~/.zshrc
            fi
            echo "gcc@11 installed; open a new terminal (or run 'source ~/.zshrc') to take effect"
            ;;
    esac

    if command -v gcc &> /dev/null; then
        curr_ver=$(parse_gcc_version gcc)
        if version_ge "$curr_ver" "$req_ver"; then
            echo "GCC installed successfully ($curr_ver)"
        else
            echo "GCC version still doesn't meet requirements, please install manually."
            exit 1
        fi
    else
        echo "GCC installation failed"
        exit 1
    fi
}

install_cmake() {
    # CMake version >= 3.16.0
    echo -e "\n==== Checking CMake ===="
    local req_ver="3.16.0"
    local curr_ver=""

    if command -v cmake &> /dev/null; then
        curr_ver=$(cmake --version | awk '/^cmake/ {print $3}')
        echo "Current CMake version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "CMake meets requirements"
            return
        fi
    fi

    echo "Installing CMake..."
    case "$OS" in
        debian)
            if grep -q "Ubuntu 18.04" /etc/os-release; then
                run_command wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
                run_command echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
                run_command sudo apt update
                run_command sudo apt install -y cmake make
            else
                run_command sudo $PKG_MANAGER update
                run_command sudo $PKG_MANAGER install -y cmake make
            fi
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command sudo $PKG_MANAGER install -y epel-release
                run_command sudo $PKG_MANAGER install -y cmake3 make
                run_command sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
            else
                run_command sudo $PKG_MANAGER install -y cmake make
            fi
            ;;
        macos)
            run_command brew install cmake
            ;;
    esac

    if command -v cmake &> /dev/null; then
        curr_ver=$(cmake --version | awk '/^cmake/ {print $3}')
        if version_ge "$curr_ver" "$req_ver"; then
            echo "CMake installed successfully ($curr_ver)"
        else
            echo "CMake version still doesn't meet requirements, please install manually"
            exit 1
        fi
    else
        echo "CMake installation failed"
        exit 1
    fi
}

install_pigz() {
    # pigz version >= 2.4
    echo -e "\n==== Checking pigz ===="
    local req_ver="2.4"
    local curr_ver=""

    if command -v pigz &> /dev/null; then
        curr_ver=$(pigz --version 2>&1 | awk '{print $2}')
        echo "Current pigz version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "pigz meets requirements"
            return
        fi
    fi

    read -p "Install pigz? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping pigz installation"
        return
    fi

    echo "Installing pigz..."
    case "$OS" in
        debian|rhel)
            run_command sudo $PKG_MANAGER install -y pigz
            ;;
        macos)
            run_command brew install pigz
            ;;
    esac

    if command -v pigz &> /dev/null; then
        curr_ver=$(pigz --version 2>&1 | awk '{print $2}')
        echo "pigz installed successfully ($curr_ver)"
    else
        echo "pigz installation failed, can be ignored"
    fi
}

main() {
    echo "===================================================="
    echo "Starting project dependency installation"
    echo "===================================================="

    detect_os
    install_gawk
    install_python
    install_gcc
    install_cmake
    install_pigz
    # dos2unix removed: no scripts in the repository call dos2unix

    echo -e "===================================================="
    echo "All dependencies installed successfully!"
    echo "===================================================="
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
