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


def test_generator_cleanup_preserves_unrelated_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['generate'])
    unrelated = tmp_path / 'measurements.bin'
    unrelated.write_bytes(b'keep')
    (tmp_path / 'fft_old_output.bin').write_bytes(b'old')

    def generate(config, seed):
        np.array([seed], dtype=np.float32).tofile(tmp_path / f"{config['name']}_input.bin")

    assert MODULE.gen_main([{'name': 'fft_old'}], generate) == 0
    assert unrelated.read_bytes() == b'keep'
    assert not (tmp_path / 'fft_old_output.bin').exists()
    assert (tmp_path / 'fft_old_input.bin').exists()
