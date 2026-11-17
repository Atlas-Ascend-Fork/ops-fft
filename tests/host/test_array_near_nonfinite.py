# Copyright (c) 2026 Xztxzt
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile the real C++ assertion and inspect its process verdict on Host values."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def assertion_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for this C++ assertion test")
    directory = tmp_path_factory.mktemp("array_near")
    include = directory / "include/acl"
    include.mkdir(parents=True)
    # Only an opaque declaration is needed; no ACL functions are called.
    (include / "acl.h").write_text("#pragma once\nusing aclrtStream = void*;\n")
    source = directory / "check.cpp"
    source.write_text(r'''#include <cmath>
#include "test_common.h"
namespace OpsFftTest {
TestStats g_global_stats;
std::vector<std::string> g_failed_cases;
}
int main(int argc, char** argv)
{
    if (argc != 4) return 2;
    OpsFftTest::TestStats stats;
    std::vector<float> actual{1.0f, 2.0f, std::stof(argv[1]), 4.0f};
    std::vector<float> expected{1.0f, 2.0f, std::stof(argv[2]), 4.0f};
    TEST_ASSERT_ARRAY_NEAR(stats, actual, expected, 4, std::stof(argv[3]), "array near verdict");
    TEST_CASE_PASS(stats, "array_near");
}
''')
    binary = directory / "check"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(include.parent), "-I" + str(ROOT / "tests"),
                    str(source), "-o", str(binary)], check=True)
    return binary


@pytest.mark.parametrize("actual,expected,tolerance,passes", [
    ("3", "3", "0", True),
    ("3.125", "3", "0.25", True),
    ("3.25", "3", "0.25", True),
    ("3.5", "3", "0.25", False),
    ("-0", "0", "0", True),
    ("nan", "3", "0.25", False),
    ("3", "nan", "0.25", False),
    ("nan", "nan", "0.25", False),
    ("inf", "3", "0.25", False),
    ("-inf", "3", "0.25", False),
    ("3", "inf", "0.25", False),
    ("inf", "inf", "0.25", False),
    ("-inf", "-inf", "0.25", False),
])
def test_array_near_requires_finite_values(assertion_binary, actual, expected, tolerance, passes):
    run = subprocess.run([str(assertion_binary), actual, expected, tolerance],
                         capture_output=True, text=True, check=False)
    assert run.returncode == (0 if passes else 1), run.stdout + run.stderr
    if passes:
        assert "[PASS] array_near" in run.stdout
    else:
        assert "[PASS]" not in run.stdout
        assert "at index 2" in run.stderr
