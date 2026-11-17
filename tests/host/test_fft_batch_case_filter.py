# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Pin the --cases batch contract and failure-signal semantics of test_helper.py (issue #366).

Covers the four behaviors called out in the issue: subset filtering exits 0,
unknown names keep the failure signal (compare counts them, gen returns 1),
an empty --cases value falls back to the full set, and the no-filter default
compares everything. Also pins the summary-line accounting invariant from
issue #369: passed + failed must always equal the reported total.
"""
import importlib.util
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'src/common/host/test_helper.py'
SPEC = importlib.util.spec_from_file_location('fft_helper_under_test', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CONFIGS = {
    'batch_a': {'atol': 1e-6, 'rtol': 1e-6},
    'batch_b': {'atol': 1e-6, 'rtol': 1e-6},
    'batch_c': {'atol': 1e-6, 'rtol': 1e-6},
}


def capture_helper_logs():
    stream = io.StringIO()
    MODULE.logger.handlers = [logging.StreamHandler(stream)]
    MODULE.logger.setLevel(logging.INFO)
    return stream


def write_matching_bins(curr_dir, name):
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values.tofile(curr_dir / f"{name}_golden.bin")
    values.tofile(curr_dir / f"{name}_output.bin")


def parse_summary(stream_text):
    summary = None
    for line in stream_text.splitlines():
        if line.startswith('总测试数='):
            fields = dict(item.split("=") for item in line.split(", "))
            summary = {key: int(value) for key, value in fields.items()}
            break
    if summary is None:
        pytest.fail("batch summary line not found")
    return summary


def test_parse_case_filter_variants():
    assert MODULE.parse_case_filter(None) is None
    assert MODULE.parse_case_filter('') is None
    assert MODULE.parse_case_filter('   ') is None
    assert MODULE.parse_case_filter('batch_a,batch_b') == ['batch_a', 'batch_b']
    assert MODULE.parse_case_filter(' batch_a , batch_b , ') == ['batch_a', 'batch_b']
    assert MODULE.parse_case_filter(',,') is None


def test_compare_subset_only_runs_selected_cases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in CONFIGS:
        write_matching_bins(tmp_path, name)
    monkeypatch.setattr(sys, 'argv', ['compare', '--cases', 'batch_a,batch_c'])
    stream = capture_helper_logs()
    assert MODULE.compare_main(str(tmp_path), CONFIGS) == 0
    out = stream.getvalue()
    assert '[PASS] batch_a' in out
    assert '[PASS] batch_c' in out
    assert '[PASS] batch_b' not in out
    assert '[FAIL]' not in out
    assert parse_summary(out) == {'总测试数': 2, '通过': 2, '失败': 0}


def test_compare_empty_cases_value_runs_full_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in CONFIGS:
        write_matching_bins(tmp_path, name)
    monkeypatch.setattr(sys, 'argv', ['compare', '--cases', ''])
    stream = capture_helper_logs()
    assert MODULE.compare_main(str(tmp_path), CONFIGS) == 0
    out = stream.getvalue()
    for name in CONFIGS:
        assert f'[PASS] {name}' in out
    assert parse_summary(out) == {'总测试数': 3, '通过': 3, '失败': 0}


def test_compare_without_filter_runs_full_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in CONFIGS:
        write_matching_bins(tmp_path, name)
    monkeypatch.setattr(sys, 'argv', ['compare'])
    stream = capture_helper_logs()
    assert MODULE.compare_main(str(tmp_path), CONFIGS) == 0
    assert parse_summary(stream.getvalue()) == {'总测试数': 3, '通过': 3, '失败': 0}


def test_compare_unknown_cases_fail_with_consistent_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_matching_bins(tmp_path, 'batch_a')
    monkeypatch.setattr(sys, 'argv', ['compare', '--cases', 'batch_a,ghost'])
    stream = capture_helper_logs()
    assert MODULE.compare_main(str(tmp_path), CONFIGS) == 1
    out = stream.getvalue()
    assert '[FAIL] ghost: no config in compare_data.py' in out
    assert '[PASS] batch_a' in out
    summary = parse_summary(out)
    assert summary == {'总测试数': 2, '通过': 1, '失败': 1}
    assert summary['通过'] + summary['失败'] == summary['总测试数']


def test_compare_summary_stays_consistent_when_cases_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ('batch_a', 'batch_b'):
        write_matching_bins(tmp_path, name)
    np.array([9.0, 9.0, 9.0], dtype=np.float32).tofile(tmp_path / 'batch_c_output.bin')
    np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(tmp_path / 'batch_c_golden.bin')
    monkeypatch.setattr(sys, 'argv', ['compare'])
    stream = capture_helper_logs()
    assert MODULE.compare_main(str(tmp_path), CONFIGS) == 1
    summary = parse_summary(stream.getvalue())
    assert summary == {'总测试数': 3, '通过': 2, '失败': 1}
    assert summary['通过'] + summary['失败'] == summary['总测试数']


def test_gen_subset_only_generates_selected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generated = []

    def generate(config, seed):
        generated.append(config['name'])

    monkeypatch.setattr(sys, 'argv', ['generate', '--cases', 'batch_a,batch_b'])
    assert MODULE.gen_main([{'name': name} for name in CONFIGS], generate) == 0
    assert sorted(generated) == ['batch_a', 'batch_b']


def test_gen_unknown_case_fails_without_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def generate(config, seed):
        pytest.fail("unknown selection must not generate data")

    monkeypatch.setattr(sys, 'argv', ['generate', '--cases', 'batch_a,ghost'])
    assert MODULE.gen_main([{'name': 'batch_a'}], generate) == 1
