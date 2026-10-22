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

CURPATH=$(dirname "$(readlink -f "$0")")
VAR_PATH="$CURPATH"
common_func_path="$CURPATH/manager/common_func.inc"
version_compatiable_path="$CURPATH/manager/version_compatiable.inc"
common_func_v2_path="$CURPATH/manager/common_func_v2.inc"
version_cfg_path="$CURPATH/manager/version_cfg.inc"
script_operator_path="$CURPATH/manager/script_operator.inc"
manager_func_path="$CURPATH/manager/manager_func.sh"

. "$common_func_path"
. "$version_compatiable_path"
. "$common_func_v2_path"
. "$version_cfg_path"
. "$script_operator_path"
. "$manager_func_path"

set_comm_log "Latest_manager" "$COMM_LOGFILE"

VERSION=""
VERSION_DIR=""
PACKAGE=""
DOCKER_ROOT=""
INTPUT_INSTALL_FOR_ALL="n"
IS_RECREATE_SOFTLINK="n"
INCREMENT="n"
USE_SHARE_INFO="n"

# A value-taking option must be followed by its value; otherwise shift 2 fails
# without changing the positional parameters and the parse loop spins forever.
check_option_value()
{
    if [ $# -lt 2 ]; then
        comm_log "ERROR" "option $1 requires a value"
        exit 1
    fi
}

while true
do
    case "$1" in
    --version)
        check_option_value "$@"
        VERSION="$2"
        shift 2
        ;;
    --version-dir)
        check_option_value "$@"
        VERSION_DIR="$2"
        shift 2
        ;;
    --package)
        check_option_value "$@"
        PACKAGE="$2"
        shift 2
        ;;
    --package-dir)
        check_option_value "$@"
        shift 2
        ;;
    --serial)
        check_option_value "$@"
        shift 2
        ;;
    --install-type)
        check_option_value "$@"
        shift 2
        ;;
    --feature)
        check_option_value "$@"
        shift 2
        ;;
    --chip)
        check_option_value "$@"
        shift 2
        ;;
    --filelist)
        check_option_value "$@"
        shift 2
        ;;
    --docker-root)
        check_option_value "$@"
        DOCKER_ROOT="$2"
        shift 2
        ;;
    --recreate-softlink)
        IS_RECREATE_SOFTLINK="y"
        shift 1
        ;;
    --install-for-all)
        INTPUT_INSTALL_FOR_ALL="y"
        shift 1
        ;;
    --increment)
        INCREMENT="y"
        shift 1
        ;;
    --use-share-info)
        USE_SHARE_INFO="y"
        shift 1
        ;;
    -*)
        comm_log "ERROR" "Unsupported parameters : $1"
        exit 1
        ;;
    *)
        break
        ;;
    esac
done

if [ "$PACKAGE" != "" ]; then
    get_titled_package_name "TITLED_PACKAGE" "$PACKAGE"
    set_comm_log "$TITLED_PACKAGE" "$COMM_LOGFILE"
fi

OPERATION="$1"

if [ "$OPERATION" = "package_installed" ]; then
    package_installed "$CURPATH" "$VERSION" "$VERSION_DIR" "$PACKAGE" \
        "$INTPUT_INSTALL_FOR_ALL" "$DOCKER_ROOT"
elif [ "$OPERATION" = "package_pre_uninstall" ]; then
    package_pre_uninstall "$CURPATH" "$VERSION" "$VERSION_DIR" "$PACKAGE" \
        "$INTPUT_INSTALL_FOR_ALL" "$DOCKER_ROOT"
elif [ "$OPERATION" = "package_uninstalled" ]; then
    package_uninstalled "$CURPATH" "$VERSION" "$VERSION_DIR" "$PACKAGE" "$IS_RECREATE_SOFTLINK" \
        "$DOCKER_ROOT"
elif [ "$OPERATION" = "package_create_softlink" ]; then
    package_create_softlink "$CURPATH" "$VERSION" "$VERSION_DIR" "$PACKAGE" "$DOCKER_ROOT"
elif [ "$OPERATION" = "package_remove_softlink" ]; then
    package_remove_softlink "$CURPATH" "$VERSION" "$VERSION_DIR" "$PACKAGE" "$DOCKER_ROOT"
elif [ "$OPERATION" = "create_version_softlink" ]; then
    create_version_softlink "$CURPATH" "$VERSION_DIR"
elif [ "$OPERATION" = "remove_latest_softlink" ]; then
    remove_latest_softlink "$CURPATH"
elif [ "$OPERATION" = "migrate_latest_data" ]; then
    migrate_latest_data "$CURPATH"
else
    comm_log "ERROR" "Unsupported operation : $OPERATION"
    exit 1
fi
