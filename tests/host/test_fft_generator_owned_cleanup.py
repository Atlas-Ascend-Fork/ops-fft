# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import importlib.util
import sys
from pathlib import Path

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


@pytest.mark.parametrize("seed", (12345, 42))
def test_selected_generation_invalidates_only_its_previous_output(tmp_path, monkeypatch, seed):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate", "--case", "selected", "--seed", str(seed)])
    selected_output = tmp_path / "selected_output.bin"
    np.array([seed], dtype=np.float32).tofile(selected_output)
    neighbors = {"other_output.bin": b"other", "measurements.bin": b"keep"}
    for name, content in neighbors.items():
        (tmp_path / name).write_bytes(content)

    def generate(config, current_seed):
        assert config["name"] == "selected"
        np.array([current_seed], dtype=np.float32).tofile("selected_input.bin")
        np.array([current_seed], dtype=np.float32).tofile("selected_golden.bin")

    assert MODULE.gen_main([{"name": "selected"}, {"name": "other"}], generate) == 0
    assert not selected_output.exists()
    assert not MODULE.compare_one(str(tmp_path), "selected", 0, 0)
    for name, content in neighbors.items():
        assert (tmp_path / name).read_bytes() == content
    # A genuinely new output can still pass. This is Host fixture data, not
    # a device kernel result.
    np.array([seed], dtype=np.float32).tofile(selected_output)
    assert MODULE.compare_one(str(tmp_path), "selected", 0, 0)


def test_unknown_selection_preserves_existing_assets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate", "--case", "unknown"])
    output = tmp_path / "selected_output.bin"
    output.write_bytes(b"keep")

    def generate(config, seed):
        pytest.fail("unknown selection must not generate data")

    assert MODULE.gen_main([{"name": "selected"}], generate) == 1
    assert output.read_bytes() == b"keep"


def test_failed_selected_generation_cannot_reuse_old_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate", "--case", "selected"])
    output = tmp_path / "selected_output.bin"
    output.write_bytes(b"stale")

    def generate(config, seed):
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        MODULE.gen_main([{"name": "selected"}], generate)
    assert not output.exists()


@pytest.mark.parametrize("selected,neighbor", (
    ("dd_32x32_forward", "dd_32x32_forward_b8"),
    ("dd_32x64_forward", "dd_32x64_forward_b4"),
    ("dd_64x64_forward", "dd_64x64_forward_b2"),
    ("dd_64x64_backward", "dd_64x64_backward_b2"),
))
def test_selected_real_fft2_generator_preserves_prefix_neighbor(
        tmp_path, monkeypatch, selected, neighbor):
    generator_source = SOURCE.parents[2] / "fft2_d/tests/fft2_d_data/gen_data.py"
    spec = importlib.util.spec_from_file_location("fft2_prefix_generator", generator_source)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert {selected, neighbor} <= {config["name"] for config in generator.configs}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate", "--case", selected])
    suffixes = ("input", "golden", "golden_forward", "output")
    for suffix in suffixes:
        (tmp_path / f"{selected}_{suffix}.bin").write_bytes(b"stale")
        (tmp_path / f"{neighbor}_{suffix}.bin").write_bytes(b"neighbor")
    unrelated = tmp_path / f"{selected}_measurements.bin"
    unrelated.write_bytes(b"keep")
    assert MODULE.gen_main(generator.configs, generator.gen_one) == 0
    assert not (tmp_path / f"{selected}_output.bin").exists()
    assert not (tmp_path / f"{selected}_golden_forward.bin").exists()
    for suffix in suffixes:
        assert (tmp_path / f"{neighbor}_{suffix}.bin").read_bytes() == b"neighbor"
    assert unrelated.read_bytes() == b"keep"
