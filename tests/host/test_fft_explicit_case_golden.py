# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'src/common/host/test_helper.py'
SPEC = importlib.util.spec_from_file_location('fft_helper_under_test', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_explicit_case_requires_its_golden_file(tmp_path, monkeypatch):
    np.array([42.0], dtype=np.float32).tofile(tmp_path / 'selected_output.bin')
    monkeypatch.setattr(sys, 'argv', ['compare', '--case', 'selected'])
    assert MODULE.compare_main(str(tmp_path), {'selected': {'atol': 0.01, 'rtol': 0.001}}) == 1


def test_explicit_case_with_valid_golden_passes(tmp_path, monkeypatch):
    for side in ('golden', 'output'):
        np.array([42.0], dtype=np.float32).tofile(tmp_path / f'selected_{side}.bin')
    monkeypatch.setattr(sys, 'argv', ['compare', '--case', 'selected'])
    assert MODULE.compare_main(str(tmp_path), {'selected': {'atol': 0.01, 'rtol': 0.001}}) == 0
