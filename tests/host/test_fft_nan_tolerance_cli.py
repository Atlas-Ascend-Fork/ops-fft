# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run all four actual comparison CLIs on explicit finite Host data fixtures."""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS = ("fft1_d", "fft2_d", "rfft1_d", "irfft1_d")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("arguments", [
    ["--atol=nan"], ["--rtol=nan"], ["--atol=nan", "--rtol=nan"],
])
def test_nan_cli_tolerance_is_rejected_before_comparison(operation, arguments, tmp_path):
    run, golden, actual = run_actual_comparison(operation, arguments, False, tmp_path)
    assert run.returncode == 2, run.stdout + run.stderr
    assert "comparison tolerance must not be NaN" in run.stderr
    assert golden.exists() and actual.exists(), "invalid CLI must not consume comparison fixtures"


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(("arguments", "matching", "expected_exit"), [
    ([], False, 1),
    ([], True, 0),
    (["--atol=inf", "--rtol=inf"], False, 0),
    (["--atol=-inf", "--rtol=inf"], True, 1),
    (["--atol=inf", "--rtol=-inf"], True, 1),
    (["--atol=0", "--rtol=0"], True, 0),
])
def test_original_ordered_tolerance_and_cleanup_controls(operation, arguments, matching, expected_exit, tmp_path):
    run, golden, actual = run_actual_comparison(operation, arguments, matching, tmp_path)
    assert run.returncode == expected_exit, run.stdout + run.stderr
    assert not golden.exists() and not actual.exists()


def first_config_name(script):
    for node in ast.parse(script.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for item in node.targets:
            if isinstance(item, ast.Name) and item.id == "configs":
                return next(iter(ast.literal_eval(node.value)))
    raise AssertionError("actual CLI has no configs assignment")


def run_actual_comparison(operation, arguments, matching, tmp_path):
    # Mirror only the unchanged real CLI and the actual shared helper so its
    # relative import/file I/O resolve within this test's temporary tree.
    script_path = Path("src") / operation / "tests" / (operation + "_data") / "compare_data.py"
    script = tmp_path / script_path
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / script_path, script)
    helper_path = Path("src/common/host/test_helper.py")
    (tmp_path / helper_path).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / helper_path, tmp_path / helper_path)
    case = first_config_name(script)
    golden_path = script.parent / (case + "_golden.bin")
    actual_path = script.parent / (case + "_output.bin")
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data.tofile(golden_path)
    (data if matching else data * 11).tofile(actual_path)
    command = [sys.executable, str(script), "--case=" + case, *arguments]
    run = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False,
                         env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"))
    return run, golden_path, actual_path
