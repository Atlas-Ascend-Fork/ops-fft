#!/bin/sh
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e

# 本脚本按约定在"待清理目录"内执行（makeself 安装完成/中断时清理解包目录，
# 或卸载时清理安装目录）。rm -rf 具有极强破坏性，删除前必须校验目标合法性，
# 防止被复制到错误目录（如家目录、/）后误删当前目录全部内容（issue #159）：
#   1) 拒绝空路径与根目录 "/";
#   2) 本脚本自身必须位于待删目录之内，否则视为误用；
#   3) 打印待删除路径后再执行，路径加引号避免空格被 word-split。
target_dir="$(pwd -P)"
script_dir="$(cd "$(dirname "$0")" && pwd -P)"
if [ -z "$target_dir" ] || [ "$target_dir" = "/" ]; then
    echo "[cleanup] refuse to delete unsafe target: '$target_dir'" >&2
    exit 1
fi
case "$script_dir" in
    "$target_dir"|"$target_dir"/*)
        ;;
    *)
        echo "[cleanup] refuse: this script ($script_dir) is not inside the target directory ($target_dir)" >&2
        exit 1
        ;;
esac
echo "[cleanup] removing directory: $target_dir"
rm -rf -- "$target_dir"
