#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression checks for invalid FFT comparison artifacts using NumPy and Torch."""

import importlib.util
import itertools
import logging
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "src/common/host/test_helper.py"
CLI_CASES = {
    "fft1_d": "dft_forward",
    "fft2_d": "dd_32x32_forward",
    "rfft1_d": "r2c_basic",
    "irfft1_d": "c2r_basic",
}


class TestComparisonArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("fft_test_helper", HELPER_PATH)
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)
        cls.scenarios = 0

    @classmethod
    def tearDownClass(cls):
        logging.getLogger(__name__).info("comparison scenarios: %s", cls.scenarios)

    def test_finite_matching_arrays_pass(self):
        for values, detail in itertools.product(
            ([1.0, -2.0, 0.0, 4.0], [0.0]), (False, True)
        ):
            with self.subTest(values=values, detail=detail):
                self._check_comparison((values, values), detail, True)

    def test_partial_values_in_either_file_fail(self):
        for role, tail, detail in itertools.product((0, 1), (1, 2, 3), (False, True)):
            with self.subTest(role=role, tail=tail, detail=detail):
                self._check_comparison(([1.0], [1.0]), detail, False, (role, tail))

    def test_nonfinite_values_in_either_file_fail(self):
        for value, role, detail in itertools.product(
            (np.nan, np.inf, -np.inf), (0, 1, 2), (False, True)
        ):
            arrays = [[1.0, -2.0], [1.0, -2.0]]
            if role in (0, 2):
                arrays[0][0] = value
            if role in (1, 2):
                arrays[1][0] = value
            with self.subTest(value=value, role=role, detail=detail):
                self._check_comparison(arrays, detail, False)

    def test_shape_and_accuracy_checks_remain(self):
        pairs = (([1.0], [1.0, 1.0]), ([1.0], [2.0]), ([0.01], [0.0101]))
        for arrays, detail in itertools.product(pairs, (False, True)):
            with self.subTest(arrays=arrays, detail=detail):
                self._check_comparison(arrays, detail, False)

    def test_missing_output_and_golden_fail(self):
        for role, detail in itertools.product((0, 1), (False, True)):
            with self.subTest(missing=role, detail=detail):
                self._check_comparison(
                    ([1.0], [1.0]), detail, False, (role, "missing")
                )

    def test_four_real_clis_preserve_valid_comparison(self):
        for operator, case in CLI_CASES.items():
            with self.subTest(operator=operator):
                self._check_cli(operator, case, "valid")

    def test_four_real_clis_reject_invalid_artifacts(self):
        for operator, corruption in itertools.product(CLI_CASES, ("nan", "tail")):
            with self.subTest(operator=operator, corruption=corruption):
                self._check_cli(operator, CLI_CASES[operator], corruption)

    def _check_comparison(self, arrays, detail, expected, corruption=None):
        type(self).scenarios += 1
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"case_{role}.bin" for role in ("golden", "output")]
            for path, values in zip(paths, arrays):
                np.asarray(values, dtype=np.float32).tofile(path)
            if corruption is not None:
                role, mutation = corruption
                if mutation == "missing":
                    paths[role].unlink()
                else:
                    paths[role].write_bytes(paths[role].read_bytes() + bytes(mutation))
            self.assertEqual(
                self.helper.compare_one(temporary, "case", 1e-2, 1e-3, detail), expected
            )

    def _check_cli(self, operator, case, corruption):
        type(self).scenarios += 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper = root / "src/common/host/test_helper.py"
            helper.parent.mkdir(parents=True)
            shutil.copy2(HELPER_PATH, helper)
            relative = (
                Path("src")
                / operator
                / "tests"
                / (operator + "_data")
                / "compare_data.py"
            )
            script = root / relative
            script.parent.mkdir(parents=True)
            shutil.copy2(ROOT / relative, script)
            for role in ("golden", "output"):
                np.array([1.0, -2.0, 0.0, 4.0], dtype=np.float32).tofile(
                    script.parent / f"{case}_{role}.bin"
                )
            output = script.parent / f"{case}_output.bin"
            if corruption == "nan":
                np.array([np.nan, -2.0, 0.0, 4.0], dtype=np.float32).tofile(output)
            elif corruption == "tail":
                output.write_bytes(output.read_bytes() + b"x")
            result = subprocess.run(
                [sys.executable, str(script), "--case", case],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=45,
            )
            if corruption == "valid":
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(list(script.parent.glob("*.bin")))


if __name__ == "__main__":
    unittest.main()
