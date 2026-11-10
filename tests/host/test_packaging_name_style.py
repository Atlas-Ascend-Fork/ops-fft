#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


"""Check actual CLI options and current XML package naming on the Host."""

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/package"))
SPEC = importlib.util.spec_from_file_location("actual_package_tool", ROOT / "scripts/package/package.py")
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)
XML = ROOT / "scripts/package/ops_fft/ops_fft.xml"
CURRENT_INFO = {item.tag: item.text for item in ET.parse(XML).getroot().find("package_info")}

CASES = [
    ([], "common", "cann-950-ops-fft_9.0.0_linux-aarch64.run"),
    ([], "underline", "cann_950_ops-fft_9.0.0_linux-aarch64.run"),
    (["--chip_name=ascend910_93", "--os_arch=linux-x86_64"], "common",
     "cann-A3-ops-fft_9.0.0_linux-x86_64.run"),
    (["--chip_name=ascend910_93", "--os_arch=linux-x86_64"], "underline",
     "cann_A3_ops-fft_9.0.0_linux-x86_64.run"),
    (["--ext_name=lab"], "common", "cann-950-ops-fft_9.0.0_linux-aarch64-lab.run"),
    (["--ext_name=lab"], "underline", "cann_950_ops-fft_9.0.0_linux-aarch64_lab.run"),
    (["--not_in_name=chip_name,version"], "common", "cann-ops-fft_linux-aarch64.run"),
    (["--not_in_name=chip_name,version"], "underline", "cann_ops-fft_linux-aarch64.run"),
    ([], "other-style-retains-current-fallback", "cann-950-ops-fft_9.0.0_linux-aarch64.run"),
]


@pytest.mark.parametrize("additional,style,expected", CASES)
def test_cli_package_style_and_existing_controls(additional, style, expected, monkeypatch):
    assert CURRENT_INFO["product_name"] == "cann"
    assert CURRENT_INFO["func_name"] == "ops-fft"
    argv = ["package.py", "--os_arch=linux-aarch64", "--pkg-name-style=" + style] + additional
    monkeypatch.setattr(sys, "argv", argv)
    args = PACKAGE.args_parse()
    assert args.pkg_name_style == style
    actual = PACKAGE.PackageName(CURRENT_INFO, args, "9.0.0").getvalue()
    assert actual == expected
